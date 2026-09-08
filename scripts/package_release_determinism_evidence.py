from __future__ import annotations

"""Package three production runs as formal determinism release evidence.

The packager never executes inference and never changes ``models.lock``.  It
validates three already-completed, current-lock, image-only production runs,
copies only their official JSON bytes into flat ``run1``/``run2``/``run3``
directories, writes the environment manifest, and publishes the whole tree in
one rename.  Existing destinations are always rejected.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


REQUIRED_RUNS = 3
REQUIRED_CASES = 50
EVIDENCE_SCOPE = "current_lock_production_image_only_inference"
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CANONICAL_CONTROLS = {
    "models_lock": Path("models/models.lock.json"),
    "pipeline_config": Path("configs/pipeline.yaml"),
    "profile": Path("configs/profiles/competition.yaml"),
    "official_profile": Path("src/pcbparse/resources/official_schema.json"),
}


class DeterminismEvidenceError(ValueError):
    """The source runs cannot safely form formal determinism evidence."""


@dataclass(frozen=True, slots=True)
class ReleaseCase:
    case_id: str
    duplicate_group: str
    image_sha256: str


@dataclass(frozen=True, slots=True)
class OfficialOutput:
    case_id: str
    source_path: Path
    target_name: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ValidatedRun:
    summary_path: Path
    summary_sha256: str
    prediction_manifest_path: Path
    prediction_manifest_sha256: str
    inference_run_id: str
    completed_at_utc: str
    model_lock_sha256: str
    config_sha256: str
    split_manifest_sha256: str
    prediction_set_sha256: str
    resolved_perception_config_sha256: str
    normalized_controls: tuple[tuple[str, str], ...]
    outputs: tuple[OfficialOutput, ...]


def package_release_determinism_evidence(
    *,
    workspace_root: str | Path,
    source_summaries: Sequence[str | Path],
    release_manifest: str | Path,
    split: str,
    output_root: str | Path,
    models_lock: str | Path = "models/models.lock.json",
    pipeline_config: str | Path = "configs/pipeline.yaml",
    production_profile: str | Path = "configs/profiles/competition.yaml",
    official_profile: str | Path = "src/pcbparse/resources/official_schema.json",
) -> dict[str, Any]:
    """Validate and atomically publish formal three-run determinism evidence."""

    root = Path(workspace_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise DeterminismEvidenceError(f"workspace root is not a directory: {root}")
    if len(source_summaries) != REQUIRED_RUNS:
        raise DeterminismEvidenceError(
            f"exactly {REQUIRED_RUNS} source summaries are required"
        )
    split_name = str(split).strip()
    if not split_name:
        raise DeterminismEvidenceError("split must be non-empty")

    control_paths = {
        "models_lock": _canonical_control(
            root, models_lock, _CANONICAL_CONTROLS["models_lock"]
        ),
        "pipeline_config": _canonical_control(
            root, pipeline_config, _CANONICAL_CONTROLS["pipeline_config"]
        ),
        "profile": _canonical_control(
            root, production_profile, _CANONICAL_CONTROLS["profile"]
        ),
        "official_profile": _canonical_control(
            root, official_profile, _CANONICAL_CONTROLS["official_profile"]
        ),
    }
    manifest_path = _workspace_file(root, release_manifest, "release manifest")
    destination = _workspace_path(root, output_root, "output root")
    determinism_base = (root / "artifacts" / "determinism").resolve(strict=False)
    if not _contains(determinism_base, destination) or destination == determinism_base:
        raise DeterminismEvidenceError(
            "output root must be a new child of artifacts/determinism"
        )
    if destination.exists():
        raise DeterminismEvidenceError(
            f"refusing to reuse or overwrite existing output root: {destination}"
        )

    source_paths = tuple(
        _workspace_file(root, value, f"source summary {index}")
        for index, value in enumerate(source_summaries, start=1)
    )
    if len({str(path).casefold() for path in source_paths}) != REQUIRED_RUNS:
        raise DeterminismEvidenceError("source summaries must be three distinct files")
    if len({str(path.parent).casefold() for path in source_paths}) != REQUIRED_RUNS:
        raise DeterminismEvidenceError(
            "source summaries must come from three distinct inference run directories"
        )
    for source_path in source_paths:
        if _contains(source_path.parent, destination) or _contains(
            destination, source_path.parent
        ):
            raise DeterminismEvidenceError(
                "output root must not contain or be contained by a source run"
            )

    contract = _load_hard_gate_contract(root)
    config_binding = contract._production_config_binding(  # type: ignore[attr-defined]
        root, control_paths["pipeline_config"], control_paths["profile"]
    )
    if config_binding.get("status") != "valid":
        raise DeterminismEvidenceError(
            "production config/profile binding is invalid: "
            + "; ".join(str(item) for item in config_binding.get("problems", []))
        )
    config_sha256 = config_binding.get("effective_config_sha256")
    if not _is_sha256(config_sha256):
        raise DeterminismEvidenceError("production semantic config SHA-256 is unavailable")
    model_lock_sha256 = _sha256_file(control_paths["models_lock"])
    manifest_sha256 = _sha256_file(manifest_path)
    release_cases = _load_release_cases(manifest_path, split=split_name)

    runs = tuple(
        _validate_source_run(
            root=root,
            contract=contract,
            summary_path=summary_path,
            release_cases=release_cases,
            release_manifest=manifest_path,
            release_manifest_sha256=manifest_sha256,
            split=split_name,
            model_lock_sha256=model_lock_sha256,
            config_sha256=config_sha256,
            control_paths=control_paths,
        )
        for summary_path in source_paths
    )
    _validate_cross_run_identity(runs)

    environment = _environment_snapshot(
        root=root,
        control_paths=control_paths,
        release_manifest=manifest_path,
        release_manifest_sha256=manifest_sha256,
        model_lock_sha256=model_lock_sha256,
        config_sha256=config_sha256,
        split=split_name,
    )
    environment_fingerprint = _json_sha256(environment)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-", dir=str(destination.parent)
        )
    ).resolve(strict=True)
    published = False
    try:
        declared_runs: list[dict[str, Any]] = []
        for number, source_run in enumerate(runs, start=1):
            run_name = f"run{number}"
            run_dir = staging / run_name
            run_dir.mkdir()
            for output in source_run.outputs:
                payload = output.source_path.read_bytes()
                if hashlib.sha256(payload).hexdigest() != output.sha256:
                    raise DeterminismEvidenceError(
                        f"source official output changed during packaging: {output.source_path}"
                    )
                _atomic_write_bytes(run_dir / output.target_name, payload)
            files, tree_sha256 = _tree_manifest(run_dir)
            expected_files = [
                {"path": output.target_name, "sha256": output.sha256}
                for output in sorted(
                    source_run.outputs, key=lambda item: item.target_name
                )
            ]
            if files != expected_files:
                raise DeterminismEvidenceError(
                    f"{run_name} staged file manifest differs from source summary"
                )
            declared_runs.append(
                {
                    "run": run_name,
                    "environment_fingerprint_sha256": environment_fingerprint,
                    "config_sha256": config_sha256,
                    "model_lock_sha256": model_lock_sha256,
                    "output_tree_sha256": tree_sha256,
                    "files": files,
                    "inference_run_id": source_run.inference_run_id,
                    "completed_at_utc": source_run.completed_at_utc,
                    "source_summary_path": _relative(root, source_run.summary_path),
                    "source_summary_sha256": source_run.summary_sha256,
                    "source_prediction_manifest_path": _relative(
                        root, source_run.prediction_manifest_path
                    ),
                    "source_prediction_manifest_sha256": (
                        source_run.prediction_manifest_sha256
                    ),
                    "source_prediction_set_sha256": source_run.prediction_set_sha256,
                }
            )

        if len({item["output_tree_sha256"] for item in declared_runs}) != 1:
            raise DeterminismEvidenceError(
                "three staged official-output trees are not byte-identical"
            )
        generated_at_utc = _utc_now()
        generated_at = _parse_utc(generated_at_utc, "generated_at_utc")
        for number, source_run in enumerate(runs, start=1):
            completed = _parse_utc(
                source_run.completed_at_utc, f"run{number} completed_at_utc"
            )
            if completed > generated_at:
                raise DeterminismEvidenceError(
                    f"run{number} completion is later than evidence generation time"
                )

        environment_manifest: dict[str, Any] = {
            "schema_version": 3,
            "kind": "same-environment-determinism",
            "status": "ok",
            "formal_release_evidence": True,
            "generated_at_utc": generated_at_utc,
            "evidence_scope": EVIDENCE_SCOPE,
            "current_lock_inference_executed": True,
            "models_loaded": True,
            "model_lock_sha256": model_lock_sha256,
            "config_sha256": config_sha256,
            "release_manifest_path": _relative(root, manifest_path),
            "release_manifest_sha256": manifest_sha256,
            "split": split_name,
            "cases": REQUIRED_CASES,
            "environment_fingerprint_sha256": environment_fingerprint,
            "environment": environment,
            "runs": declared_runs,
        }
        environment_path = staging / "environment.json"
        _atomic_write_json(environment_path, environment_manifest)

        gate = contract._audit_determinism(  # type: ignore[attr-defined]
            root,
            staging,
            environment_path,
            model_lock_sha256,
            config_sha256,
            require_formal_binding=True,
        )
        if gate.get("status") != "pass":
            reasons = gate.get("environment_problems", [])
            raise DeterminismEvidenceError(
                "staged evidence is rejected by the release hard-gate contract: "
                + "; ".join(str(item) for item in reasons)
            )

        if destination.exists():
            raise DeterminismEvidenceError(
                f"output root appeared during packaging; refusing overwrite: {destination}"
            )
        os.rename(staging, destination)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)

    final_environment = destination / "environment.json"
    return {
        "ok": True,
        "status": "ok",
        "formal_release_evidence": True,
        "output_root": str(destination),
        "environment_manifest": str(final_environment),
        "environment_manifest_sha256": _sha256_file(final_environment),
        "model_lock_sha256": model_lock_sha256,
        "config_sha256": config_sha256,
        "release_manifest_sha256": manifest_sha256,
        "environment_fingerprint_sha256": environment_fingerprint,
        "runs": [
            {
                "run": f"run{number}",
                "inference_run_id": run.inference_run_id,
                "source_summary_sha256": run.summary_sha256,
                "prediction_set_sha256": run.prediction_set_sha256,
            }
            for number, run in enumerate(runs, start=1)
        ],
    }


def _validate_source_run(
    *,
    root: Path,
    contract: ModuleType,
    summary_path: Path,
    release_cases: tuple[ReleaseCase, ...],
    release_manifest: Path,
    release_manifest_sha256: str,
    split: str,
    model_lock_sha256: str,
    config_sha256: str,
    control_paths: Mapping[str, Path],
) -> ValidatedRun:
    audited = contract._audit_production_summary_binding(  # type: ignore[attr-defined]
        root,
        summary_path,
        model_lock_sha256,
        config_sha256,
        require_success=True,
        require_embedded_time=True,
    )
    if audited.get("valid") is not True:
        raise DeterminismEvidenceError(
            f"source summary is not formal production evidence: {summary_path}: "
            + "; ".join(str(item) for item in audited.get("problems", []))
        )
    summary = _read_json_object(summary_path, "source summary")
    required_summary_checks = {
        "schema_version": summary.get("schema_version") == 2,
        "split": summary.get("split") == split,
        "selected_cases": summary.get("selected_cases") == REQUIRED_CASES,
        "succeeded": summary.get("succeeded") == REQUIRED_CASES,
        "failed": summary.get("failed") == 0,
        "not_run": summary.get("not_run") == 0,
        "all_successful": summary.get("all_successful") is True,
        "split_manifest_sha256": summary.get("split_manifest_sha256")
        == release_manifest_sha256,
        "model_lock_sha256": summary.get("model_lock_sha256")
        == model_lock_sha256,
        "config_hash": summary.get("config_hash") == config_sha256,
    }
    failed_checks = [name for name, passed in required_summary_checks.items() if not passed]
    if failed_checks:
        raise DeterminismEvidenceError(
            f"source summary fails release50 checks {failed_checks}: {summary_path}"
        )

    run_id = summary.get("inference_run_id")
    completed_at = summary.get("completed_at_utc")
    prediction_set_sha256 = summary.get("prediction_set_sha256")
    resolved_perception_sha256 = summary.get("resolved_perception_config_sha256")
    for name, value in (
        ("inference_run_id", run_id),
        ("prediction_set_sha256", prediction_set_sha256),
        ("resolved_perception_config_sha256", resolved_perception_sha256),
    ):
        if not _is_sha256(value):
            raise DeterminismEvidenceError(
                f"source summary lacks lowercase SHA-256 {name}: {summary_path}"
            )
    if not isinstance(completed_at, str):
        raise DeterminismEvidenceError(
            f"source summary lacks completed_at_utc: {summary_path}"
        )
    _parse_utc(completed_at, "completed_at_utc")

    controls = summary.get("control_inputs")
    if not isinstance(controls, Mapping):
        raise DeterminismEvidenceError(
            f"source summary lacks control_inputs: {summary_path}"
        )
    normalized_controls: list[tuple[str, str]] = []
    for name, expected in control_paths.items():
        actual = _workspace_file(
            root, controls.get(name), f"source summary control_inputs.{name}"
        )
        if actual != expected:
            raise DeterminismEvidenceError(
                f"source summary control_inputs.{name} is not canonical: {summary_path}"
            )
        normalized_controls.append((name, str(actual)))
    declared_manifest = _workspace_file(
        root,
        controls.get("split_manifest"),
        "source summary control_inputs.split_manifest",
    )
    if declared_manifest != release_manifest:
        raise DeterminismEvidenceError(
            f"source summary does not use the requested frozen release manifest: {summary_path}"
        )
    dataset_root = _workspace_directory(
        root, controls.get("dataset_root"), "source summary dataset_root"
    )
    normalized_controls.extend(
        (("dataset_root", str(dataset_root)), ("split_manifest", str(declared_manifest)))
    )

    raw_manifest = summary.get("prediction_manifest")
    if not isinstance(raw_manifest, str) or not raw_manifest.strip():
        raise DeterminismEvidenceError(
            f"source summary lacks prediction_manifest: {summary_path}"
        )
    prediction_manifest = _child_file(
        summary_path.parent,
        raw_manifest,
        "source prediction manifest",
    )
    rows = _read_jsonl(prediction_manifest)
    if len(rows) != REQUIRED_CASES:
        raise DeterminismEvidenceError(
            f"source prediction manifest must contain {REQUIRED_CASES} rows: "
            f"{prediction_manifest}"
        )
    expected_ids = [case.case_id for case in release_cases]
    actual_ids = [row.get("case_id") for row in rows]
    if actual_ids != expected_ids:
        raise DeterminismEvidenceError(
            f"source prediction cases/order differ from frozen release manifest: {prediction_manifest}"
        )

    outputs: list[OfficialOutput] = []
    seen_targets: set[str] = set()
    for index, (case, row) in enumerate(zip(release_cases, rows, strict=True)):
        row_checks = {
            "duplicate_group": row.get("duplicate_group") == case.duplicate_group,
            "image_sha256": row.get("image_sha256") == case.image_sha256,
            "status": row.get("status") == "ok",
            "error": row.get("error") is None,
            "provider_kind": row.get("provider_kind") == "production",
            "sidecar_access": row.get("sidecar_access") is False,
            "model_lock_sha256": row.get("model_lock_sha256")
            == model_lock_sha256,
            "config_hash": row.get("config_hash") == config_sha256,
            "inference_run_id": row.get("inference_run_id") == run_id,
        }
        failed_row_checks = [
            name for name, passed in row_checks.items() if not passed
        ]
        if failed_row_checks:
            raise DeterminismEvidenceError(
                f"prediction row {index} fails checks {failed_row_checks}: {prediction_manifest}"
            )
        expected_relative = f"official/{case.case_id}.json"
        if str(row.get("official_path", "")).replace("\\", "/") != expected_relative:
            raise DeterminismEvidenceError(
                f"case {case.case_id} official_path is not canonical: {prediction_manifest}"
            )
        official_path = _child_file(
            summary_path.parent,
            expected_relative,
            f"case {case.case_id} official output",
        )
        official_sha256 = _sha256_file(official_path)
        if (
            row.get("official_sha256") != official_sha256
            or row.get("official_output_sha256") != official_sha256
        ):
            raise DeterminismEvidenceError(
                f"case {case.case_id} official output hash differs from manifest"
            )
        _read_json_object(official_path, f"case {case.case_id} official output")
        target_name = f"{case.case_id}.json"
        normalized_target = target_name.casefold()
        if normalized_target in seen_targets:
            raise DeterminismEvidenceError(
                f"case-insensitive official output collision: {target_name}"
            )
        seen_targets.add(normalized_target)
        outputs.append(
            OfficialOutput(
                case_id=case.case_id,
                source_path=official_path,
                target_name=target_name,
                sha256=official_sha256,
            )
        )

    return ValidatedRun(
        summary_path=summary_path,
        summary_sha256=_sha256_file(summary_path),
        prediction_manifest_path=prediction_manifest,
        prediction_manifest_sha256=_sha256_file(prediction_manifest),
        inference_run_id=str(run_id),
        completed_at_utc=completed_at,
        model_lock_sha256=model_lock_sha256,
        config_sha256=config_sha256,
        split_manifest_sha256=release_manifest_sha256,
        prediction_set_sha256=str(prediction_set_sha256),
        resolved_perception_config_sha256=str(resolved_perception_sha256),
        normalized_controls=tuple(sorted(normalized_controls)),
        outputs=tuple(sorted(outputs, key=lambda item: item.target_name)),
    )


def _validate_cross_run_identity(runs: Sequence[ValidatedRun]) -> None:
    if len({run.inference_run_id for run in runs}) != REQUIRED_RUNS:
        raise DeterminismEvidenceError(
            "three source summaries must have distinct inference_run_id values"
        )
    shared_fields = {
        "model_lock_sha256": {run.model_lock_sha256 for run in runs},
        "config_sha256": {run.config_sha256 for run in runs},
        "split_manifest_sha256": {run.split_manifest_sha256 for run in runs},
        "prediction_set_sha256": {run.prediction_set_sha256 for run in runs},
        "resolved_perception_config_sha256": {
            run.resolved_perception_config_sha256 for run in runs
        },
        "control_inputs": {run.normalized_controls for run in runs},
    }
    mismatched = [name for name, values in shared_fields.items() if len(values) != 1]
    if mismatched:
        raise DeterminismEvidenceError(
            f"source runs do not share one release identity: {mismatched}"
        )
    expected = [
        (output.target_name, output.sha256) for output in runs[0].outputs
    ]
    for number, run in enumerate(runs[1:], start=2):
        actual = [(output.target_name, output.sha256) for output in run.outputs]
        if actual != expected:
            raise DeterminismEvidenceError(
                f"run{number} official JSON paths or byte hashes differ from run1"
            )


def _load_release_cases(path: Path, *, split: str) -> tuple[ReleaseCase, ...]:
    rows = _read_jsonl(path)
    if not rows:
        raise DeterminismEvidenceError("release manifest is empty")
    seen_ids: set[str] = set()
    group_rows: dict[str, list[dict[str, Any]]] = {}
    group_splits: dict[str, set[str]] = {}
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        case_id = row.get("sample_id")
        group = row.get("duplicate_group")
        row_split = row.get("split")
        size = row.get("duplicate_group_size")
        files = row.get("files")
        image = files.get("image") if isinstance(files, Mapping) else None
        image_sha256 = image.get("sha256") if isinstance(image, Mapping) else None
        if (
            not isinstance(case_id, str)
            or not _SAFE_CASE_ID.fullmatch(case_id)
            or case_id in {".", ".."}
        ):
            raise DeterminismEvidenceError(
                f"release manifest row {index} has unsafe sample_id"
            )
        if case_id in seen_ids:
            raise DeterminismEvidenceError(
                f"release manifest has duplicate sample_id {case_id!r}"
            )
        seen_ids.add(case_id)
        if not isinstance(group, str) or not group:
            raise DeterminismEvidenceError(
                f"release manifest case {case_id} has no duplicate_group"
            )
        if not isinstance(row_split, str) or not row_split:
            raise DeterminismEvidenceError(
                f"release manifest case {case_id} has no split"
            )
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise DeterminismEvidenceError(
                f"release manifest case {case_id} has invalid duplicate_group_size"
            )
        if not _is_sha256(image_sha256):
            raise DeterminismEvidenceError(
                f"release manifest case {case_id} lacks image SHA-256"
            )
        normalized_row = {
            "sample_id": case_id,
            "duplicate_group": group,
            "duplicate_group_size": size,
            "split": row_split,
            "image_sha256": image_sha256,
        }
        normalized.append(normalized_row)
        group_rows.setdefault(group, []).append(normalized_row)
        group_splits.setdefault(group, set()).add(row_split)
    leaking = sorted(group for group, values in group_splits.items() if len(values) != 1)
    if leaking:
        raise DeterminismEvidenceError(
            f"release manifest duplicate groups leak across splits: {leaking[:10]}"
        )
    for group, members in group_rows.items():
        declared = {int(member["duplicate_group_size"]) for member in members}
        if declared != {len(members)}:
            raise DeterminismEvidenceError(
                f"release manifest duplicate group {group!r} is incomplete or inconsistent"
            )
    selected = [row for row in normalized if row["split"] == split]
    if len(selected) != REQUIRED_CASES:
        raise DeterminismEvidenceError(
            f"release split {split!r} must contain exactly {REQUIRED_CASES} cases, "
            f"found {len(selected)}"
        )
    selected.sort(key=lambda item: _natural_key(item["sample_id"]))
    return tuple(
        ReleaseCase(
            case_id=str(row["sample_id"]),
            duplicate_group=str(row["duplicate_group"]),
            image_sha256=str(row["image_sha256"]),
        )
        for row in selected
    )


def _environment_snapshot(
    *,
    root: Path,
    control_paths: Mapping[str, Path],
    release_manifest: Path,
    release_manifest_sha256: str,
    model_lock_sha256: str,
    config_sha256: str,
    split: str,
) -> dict[str, Any]:
    packages = {}
    for distribution in (
        "numpy",
        "onnxruntime-gpu",
        "Pillow",
        "pydantic",
        "torch",
        "torchvision",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    relevant_environment = {
        name: os.environ.get(name)
        for name in (
            "CUBLAS_WORKSPACE_CONFIG",
            "CUDA_DEVICE_ORDER",
            "CUDA_VISIBLE_DEVICES",
            "MKL_NUM_THREADS",
            "OMP_NUM_THREADS",
            "PYTHONHASHSEED",
        )
    }
    source_files: dict[str, str | None] = {}
    for relative in (
        "scripts/package_release_determinism_evidence.py",
        "scripts/audit_release_hard_gates.py",
        "src/pcbparse/evaluation/infer_manifest.py",
        "src/pcbparse/output/official_serializer.py",
        "requirements.runtime.lock",
    ):
        source = (root / relative).resolve(strict=False)
        source_files[relative] = _sha256_file(source) if source.is_file() else None
    return {
        "schema_version": 1,
        "kind": "production-inference-environment-fingerprint",
        "capture_scope": (
            "packaging-host runtime plus byte-bound production controls and source code"
        ),
        "workspace_root": str(root),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve(strict=False)),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "environment_variables": relevant_environment,
        "controls": {
            "model_lock_sha256": model_lock_sha256,
            "config_sha256": config_sha256,
            "pipeline_config_sha256": _sha256_file(control_paths["pipeline_config"]),
            "production_profile_sha256": _sha256_file(control_paths["profile"]),
            "official_profile_sha256": _sha256_file(control_paths["official_profile"]),
            "release_manifest_path": _relative(root, release_manifest),
            "release_manifest_sha256": release_manifest_sha256,
            "split": split,
            "cases": REQUIRED_CASES,
        },
        "source_files": source_files,
    }


def _load_hard_gate_contract(root: Path) -> ModuleType:
    script = (root / "scripts" / "audit_release_hard_gates.py").resolve(strict=False)
    if not script.is_file():
        raise DeterminismEvidenceError(
            f"release hard-gate contract does not exist: {script}"
        )
    module_name = "_pcbparse_release_hard_gate_contract"
    specification = importlib.util.spec_from_file_location(module_name, script)
    if specification is None or specification.loader is None:
        raise DeterminismEvidenceError(
            f"cannot load release hard-gate contract: {script}"
        )
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def _canonical_control(root: Path, value: str | Path, relative: Path) -> Path:
    actual = _workspace_file(root, value, f"canonical control {relative.as_posix()}")
    expected = (root / relative).resolve(strict=False)
    if actual != expected:
        raise DeterminismEvidenceError(
            f"formal evidence requires canonical control {expected}, got {actual}"
        )
    return actual


def _workspace_path(root: Path, value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    resolved = (path if path.is_absolute() else root / path).resolve(strict=False)
    if not _contains(root, resolved):
        raise DeterminismEvidenceError(f"{label} escapes workspace: {resolved}")
    return resolved


def _workspace_file(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise DeterminismEvidenceError(f"{label} is missing")
    resolved = _workspace_path(root, value, label)
    if not resolved.is_file():
        raise DeterminismEvidenceError(f"{label} is not a file: {resolved}")
    return resolved


def _workspace_directory(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise DeterminismEvidenceError(f"{label} is missing")
    resolved = _workspace_path(root, value, label)
    if not resolved.is_dir():
        raise DeterminismEvidenceError(f"{label} is not a directory: {resolved}")
    return resolved


def _child_file(parent: Path, value: str | Path, label: str) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        raise DeterminismEvidenceError(f"{label} path must be relative")
    resolved = (parent / raw).resolve(strict=False)
    if not _contains(parent.resolve(), resolved):
        raise DeterminismEvidenceError(f"{label} escapes its run directory")
    if not resolved.is_file():
        raise DeterminismEvidenceError(f"{label} is not a file: {resolved}")
    return resolved


def _contains(parent: Path, candidate: Path) -> bool:
    try:
        candidate.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise DeterminismEvidenceError(f"{label} root is not an object: {path}")
    return value


def _read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, DeterminismEvidenceError) as exc:
        raise DeterminismEvidenceError(f"invalid JSON at {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DeterminismEvidenceError(f"cannot read JSONL at {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (json.JSONDecodeError, DeterminismEvidenceError) as exc:
            raise DeterminismEvidenceError(
                f"invalid JSONL at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise DeterminismEvidenceError(
                f"JSONL row at {path}:{line_number} is not an object"
            )
        rows.append(value)
    return rows


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise DeterminismEvidenceError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise DeterminismEvidenceError(f"non-standard JSON constant {value!r}")


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _tree_manifest(directory: Path) -> tuple[list[dict[str, str]], str]:
    files = sorted(
        (path for path in directory.rglob("*.json") if path.is_file()),
        key=lambda item: item.relative_to(directory).as_posix(),
    )
    manifest = [
        {
            "path": path.relative_to(directory).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in files
    ]
    return manifest, _json_sha256(manifest)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, _pretty_json_bytes(value))


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_HEX_SHA256.fullmatch(value))


def _parse_utc(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DeterminismEvidenceError(f"{label} is not ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DeterminismEvidenceError(f"{label} has no timezone")
    return parsed.astimezone(timezone.utc)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError as exc:  # guarded by workspace path validation
        raise DeterminismEvidenceError(f"path escapes workspace: {path}") from exc


def _natural_key(value: object) -> tuple[tuple[int, object], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold(), part)
        for part in re.split(r"(\d+)", str(value))
        if part
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate three current-lock production release50 runs and atomically "
            "package flat official-JSON determinism evidence."
        )
    )
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument(
        "--source-summary",
        action="append",
        required=True,
        help="Production inference summary.json (repeat exactly three times, in run order).",
    )
    parser.add_argument("--release-manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--models-lock", default="models/models.lock.json")
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    parser.add_argument(
        "--production-profile", default="configs/profiles/competition.yaml"
    )
    parser.add_argument(
        "--official-profile",
        default="src/pcbparse/resources/official_schema.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        result = package_release_determinism_evidence(
            workspace_root=arguments.workspace_root,
            source_summaries=arguments.source_summary,
            release_manifest=arguments.release_manifest,
            split=arguments.split,
            output_root=arguments.output_root,
            models_lock=arguments.models_lock,
            pipeline_config=arguments.pipeline_config,
            production_profile=arguments.production_profile,
            official_profile=arguments.official_profile,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DeterminismEvidenceError",
    "build_parser",
    "main",
    "package_release_determinism_evidence",
]
