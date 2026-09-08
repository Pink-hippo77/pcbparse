from __future__ import annotations

"""Create an unblessed Component candidate from calibrated test evidence."""

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pcbparse.training.artifacts import (  # noqa: E402
    CandidateBundleError,
    ModelCard,
    build_candidate_lock_entry,
    finalize_candidate_bundle,
    load_candidate_bundle,
)


CONFIRMATION_KIND = "pcbparse_component_calibrated_frozen_test_confirmation"
CONFIRMATION_STATUS = "calibrated_test_confirmation_passed"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repackage unchanged Component weights with a hash-bound calibrated "
            "runtime threshold and independent frozen-test confirmation."
        )
    )
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument(
        "--source-bundle",
        type=Path,
        default=Path("work/train-component-v5-fullmix/bundle.manifest.json"),
    )
    parser.add_argument(
        "--parent-status",
        type=Path,
        default=Path("work/train-component-v5-fullmix/torch_run_status.json"),
    )
    parser.add_argument(
        "--confirmation-status",
        type=Path,
        default=Path(
            "work/confirm-component-v5-calibrated-test/calibrated_test_status.json"
        ),
    )
    parser.add_argument(
        "--calibration-audit",
        type=Path,
        default=Path(
            "artifacts/audits/component_v4_v5_stitched_threshold_sweep.json"
        ),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("work/candidates-runtime-bound/component-v5-calibrated-test"),
    )
    return parser


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON object key {key!r}")
        output[key] = value
    return output


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_file(workspace: Path, raw: str | Path, label: str) -> Path:
    path = Path(raw).expanduser()
    resolved = (path if path.is_absolute() else workspace / path).resolve(strict=False)
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside workspace") from exc
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"{label} is missing or empty: {resolved}")
    return resolved


def _relative(path: Path, workspace: Path) -> str:
    return path.relative_to(workspace).as_posix()


def _verify_payload(document: Mapping[str, Any], field: str, label: str) -> str:
    declared = document.get(field)
    if not isinstance(declared, str) or len(declared) != 64:
        raise ValueError(f"{label} lacks {field}")
    payload = copy.deepcopy(dict(document))
    payload.pop(field, None)
    if _canonical_sha256(payload) != declared:
        raise ValueError(f"{label} canonical payload SHA-256 mismatch")
    return declared


def _verify_declared_source(
    workspace: Path,
    raw: Any,
    expected_path: Path,
    label: str,
) -> None:
    entry = _mapping(raw, label)
    declared_path = _resolve_file(workspace, str(entry.get("path", "")), label)
    if declared_path != expected_path:
        raise ValueError(f"{label} path differs from the supplied source")
    if entry.get("sha256") != _sha256(expected_path):
        raise ValueError(f"{label} SHA-256 mismatch")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n"
    )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def package_candidate(
    *,
    workspace: Path,
    source_bundle_path: Path,
    parent_status_path: Path,
    confirmation_status_path: Path,
    calibration_audit_path: Path,
    destination: Path,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"destination already exists; refusing overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_bundle = load_candidate_bundle(source_bundle_path)
    if source_bundle.task != "component" or source_bundle.inference is None:
        raise CandidateBundleError("source bundle is not a runtime-bound Component candidate")
    build_candidate_lock_entry(source_bundle_path, lock_directory=source_bundle_path.parent)
    source_root = source_bundle_path.parent.resolve(strict=True)
    source_artifact = _resolve_file(
        workspace, source_root / source_bundle.artifact_path, "source Component artifact"
    )
    if _sha256(source_artifact) != source_bundle.artifact_sha256:
        raise CandidateBundleError("source Component artifact SHA-256 mismatch")
    source_card_path = _resolve_file(
        workspace, source_root / source_bundle.model_card_path, "source model card"
    )
    source_card = ModelCard.model_validate(_read_object(source_card_path, "source model card"))

    parent = _read_object(parent_status_path, "formal parent status")
    if (
        parent.get("task") != "component"
        or parent.get("status") != "candidate_evidence_ready"
        or parent.get("formal_evidence") is not True
        or parent.get("registerable") is not True
    ):
        raise ValueError("formal Component parent is not candidate-evidence-ready")
    artifact = _mapping(parent.get("artifact"), "parent artifact")
    parent_artifact_path = _resolve_file(workspace, str(artifact.get("path", "")), "parent artifact")
    if parent_artifact_path != source_artifact or artifact.get("sha256") != source_bundle.artifact_sha256:
        raise ValueError("source bundle artifact differs from the formal parent")
    lineage = _mapping(parent.get("lineage"), "parent lineage")
    if (
        lineage.get("training_config_sha256") != source_bundle.training_config_sha256
        or lineage.get("dataset_config_hash") != source_bundle.dataset_config_hash
        or lineage.get("source_manifest_sha256") != source_bundle.source_manifest_sha256
        or lineage.get("split_manifest_sha256") != source_bundle.split_manifest_sha256
    ):
        raise ValueError("source bundle lineage differs from the formal parent")

    confirmation = _read_object(confirmation_status_path, "test confirmation status")
    confirmation_payload = _verify_payload(
        confirmation, "confirmation_payload_sha256", "test confirmation"
    )
    if (
        confirmation.get("kind") != CONFIRMATION_KIND
        or confirmation.get("status") != CONFIRMATION_STATUS
        or confirmation.get("formal_evidence") is not True
        or confirmation.get("registerable_repackage_evidence") is not True
        or confirmation.get("promotion_eligible") is not True
        or confirmation.get("blessing_authorized") is not False
        or confirmation.get("gates_passed") is not True
        or confirmation.get("protected_sources_unchanged") is not True
    ):
        raise ValueError("test confirmation is not final passing repackage evidence")
    confirmation_lineage = _mapping(confirmation.get("lineage"), "confirmation lineage")
    _verify_declared_source(
        workspace,
        confirmation_lineage.get("parent_status"),
        parent_status_path,
        "confirmation parent status",
    )
    _verify_declared_source(
        workspace,
        _mapping(confirmation.get("calibration"), "confirmation calibration").get("audit"),
        calibration_audit_path,
        "confirmation calibration audit",
    )
    confirmation_artifacts = _mapping(confirmation.get("artifacts"), "confirmation artifacts")
    predictions_entry = _mapping(confirmation_artifacts.get("predictions"), "confirmation predictions")
    predictions_path = _resolve_file(
        workspace, str(predictions_entry.get("path", "")), "confirmation predictions"
    )
    if predictions_entry.get("sha256") != _sha256(predictions_path):
        raise ValueError("confirmation prediction SHA-256 mismatch")
    capture_entry = _mapping(
        confirmation_artifacts.get("captured_population"), "captured population"
    )
    capture_path = _resolve_file(
        workspace, str(capture_entry.get("path", "")), "captured population"
    )
    if capture_entry.get("sha256") != _sha256(capture_path):
        raise ValueError("captured population SHA-256 mismatch")
    selection = _mapping(confirmation.get("selection_boundary"), "confirmation selection")
    if (
        selection.get("split") != "test"
        or selection.get("available_test_samples") != 30
        or selection.get("selected_test_samples") != 30
        or selection.get("train_samples_loaded") != 0
        or selection.get("val_samples_loaded") != 0
    ):
        raise ValueError("confirmation does not cover the exact independent 30-case test split")
    protocol = _mapping(confirmation.get("protocol"), "confirmation protocol")
    selected_threshold = float(protocol.get("selected_score_threshold"))
    if not 0.0 <= selected_threshold <= 1.0:
        raise ValueError("confirmed score threshold is invalid")
    calibration = _mapping(confirmation.get("calibration"), "confirmation calibration")
    if float(calibration.get("selected_score_threshold")) != selected_threshold:
        raise ValueError("selection and confirmation score thresholds differ")

    inference = dict(source_bundle.inference)
    inference["score_threshold"] = selected_threshold
    metrics_raw = _mapping(confirmation.get("metrics"), "confirmation metrics")
    metrics = {
        name: float(metrics_raw[name])
        for name in (
            "bbox_map50",
            "precision50",
            "recall50",
            "f1_50",
            "pred_to_gt_ratio",
        )
    }
    gates = [dict(_mapping(item, "confirmation gate")) for item in confirmation.get("gates", [])]
    if not gates or any(item.get("passed") is not True for item in gates):
        raise ValueError("confirmation configured gates did not all pass")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.package-", dir=destination.parent)
    ).resolve(strict=True)
    moved = False
    try:
        copied_artifact = staging / source_artifact.name
        copied_predictions = staging / "predictions.jsonl"
        evidence_root = staging / "evidence"
        evidence_root.mkdir(parents=True)
        copied_calibration = evidence_root / "validation_calibration.json"
        copied_confirmation = evidence_root / "frozen_test_confirmation.json"
        copied_capture = evidence_root / "test_capture_population_scorefloor020.jsonl"
        copied_parent = evidence_root / "formal_parent_status.json"
        copied_source_bundle = evidence_root / "source_bundle.manifest.json"
        for source, target in (
            (source_artifact, copied_artifact),
            (predictions_path, copied_predictions),
            (calibration_audit_path, copied_calibration),
            (confirmation_status_path, copied_confirmation),
            (capture_path, copied_capture),
            (parent_status_path, copied_parent),
            (source_bundle_path, copied_source_bundle),
        ):
            shutil.copyfile(source, target)
        attachments = [
            {
                "kind": kind,
                "path": target.relative_to(staging).as_posix(),
                "sha256": _sha256(target),
            }
            for kind, target in (
                ("validation_score_calibration", copied_calibration),
                ("independent_frozen_test_confirmation", copied_confirmation),
                ("test_capture_population_scorefloor020", copied_capture),
                ("formal_parent_status", copied_parent),
                ("source_candidate_bundle", copied_source_bundle),
            )
        ]
        version = (
            f"component-v5-calibrated-{source_bundle.artifact_sha256[:12]}-"
            f"{confirmation_payload[:12]}"
        )
        manifest_path = finalize_candidate_bundle(
            staging,
            artifact_path=copied_artifact,
            evaluation_predictions_path=copied_predictions,
            task="component",
            version=version,
            license_name=source_bundle.license,
            precision=source_bundle.precision,
            adapter_entrypoint=source_bundle.adapter_smoke.entrypoint,
            inference=inference,
            training_config_sha256=source_bundle.training_config_sha256,
            dataset_config_hash=source_bundle.dataset_config_hash,
            source_manifest_sha256=source_bundle.source_manifest_sha256,
            split_manifest_sha256=source_bundle.split_manifest_sha256,
            gate_policy=source_bundle.gate_policy,
            export_consistency=source_bundle.export_consistency,
            evaluation={
                "evaluator": "pcbparse.component.tiled_frozen_test_confirmation",
                "evaluator_version": "1",
                "split": "test",
                "dataset_config_hash": source_bundle.dataset_config_hash,
                "split_manifest_sha256": source_bundle.split_manifest_sha256,
                "predictions_sha256": _sha256(copied_predictions),
                "metrics": metrics,
                "gates": gates,
            },
            intended_use=(
                "Dataset300-style PCB schematic Component detection using fixed "
                "512/64 tiling, class-aware NMS 0.55, and the hash-bound score threshold"
            ),
            limitations=[
                *source_card.limitations,
                "The score threshold was selected on frozen validation and independently confirmed on the 30-image internal test split.",
                "The internal test split is not the organizer's hidden leaderboard set.",
                "This candidate remains unblessed until four-task system validation passes.",
            ],
            evidence_attachments=attachments,
            out_of_scope_use=[
                "use with a score threshold other than the hash-bound inference contract",
                "official leaderboard claims without organizer evaluation",
                "production deployment before four-task deterministic-Junction system validation",
            ],
        )
        lock_entry = build_candidate_lock_entry(manifest_path, lock_directory=staging)
        package_audit: dict[str, Any] = {
            "schema_version": 1,
            "kind": "pcbparse_component_calibrated_candidate_package",
            "status": "candidate_evidence_ready",
            "created_at_utc": _utc_now(),
            "task": "component",
            "version": version,
            "source": {
                "bundle_manifest": {
                    "path": _relative(source_bundle_path, workspace),
                    "sha256": _sha256(source_bundle_path),
                },
                "artifact_sha256": source_bundle.artifact_sha256,
                "parent_status_sha256": _sha256(parent_status_path),
                "calibration_audit_sha256": _sha256(calibration_audit_path),
                "test_confirmation_sha256": _sha256(confirmation_status_path),
                "test_confirmation_payload_sha256": confirmation_payload,
            },
            "candidate": {
                "bundle_manifest_sha256": _sha256(manifest_path),
                "artifact_sha256": _sha256(copied_artifact),
                "predictions_sha256": _sha256(copied_predictions),
                "score_threshold": selected_threshold,
                "evaluation_split": "test",
                "metrics": metrics,
                "evidence_attachment_count": len(attachments),
                "lock_entry_payload_sha256": _canonical_sha256(lock_entry),
            },
            "checks": {
                "model_weights_unchanged": _sha256(copied_artifact) == source_bundle.artifact_sha256,
                "validation_selection_hash_bound": True,
                "independent_test_confirmation_hash_bound": True,
                "runtime_threshold_hash_bound": True,
                "configured_test_gate_passed": True,
                "candidate_lock_entry_buildable": True,
            },
            "boundaries": {
                "model_retrained": False,
                "test_thresholds_searched": False,
                "models_lock_modified": False,
                "blessed": False,
            },
        }
        package_audit["package_payload_sha256"] = _canonical_sha256(package_audit)
        _atomic_write_json(staging / "package.audit.json", package_audit)
        os.replace(staging, destination)
        moved = True
    finally:
        if not moved and staging.exists():
            shutil.rmtree(staging)

    final_manifest = destination / "bundle.manifest.json"
    final_audit = destination / "package.audit.json"
    return {
        "ok": True,
        "status": "candidate_evidence_ready",
        "task": "component",
        "bundle_manifest": str(final_manifest),
        "bundle_manifest_sha256": _sha256(final_manifest),
        "package_audit": str(final_audit),
        "package_audit_sha256": _sha256(final_audit),
        "score_threshold": selected_threshold,
        "artifact_sha256": source_bundle.artifact_sha256,
        "blessed": False,
        "models_lock_modified": False,
    }


def main() -> int:
    arguments = _parser().parse_args()
    workspace = arguments.workspace_root.expanduser().resolve(strict=True)
    source_bundle = _resolve_file(workspace, arguments.source_bundle, "source bundle")
    parent_status = _resolve_file(workspace, arguments.parent_status, "parent status")
    confirmation = _resolve_file(
        workspace, arguments.confirmation_status, "confirmation status"
    )
    calibration = _resolve_file(workspace, arguments.calibration_audit, "calibration audit")
    destination_raw = Path(arguments.destination).expanduser()
    destination = (
        destination_raw if destination_raw.is_absolute() else workspace / destination_raw
    ).resolve(strict=False)
    try:
        destination.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("destination must stay inside workspace") from exc
    result = package_candidate(
        workspace=workspace,
        source_bundle_path=source_bundle,
        parent_status_path=parent_status,
        confirmation_status_path=confirmation,
        calibration_audit_path=calibration,
        destination=destination,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
