from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from string import Formatter
from typing import Any, Iterable, Mapping, Sequence


REQUIRED_MODEL_TASKS = ("component", "junction", "ocr", "pin", "wire")
DETERMINISTIC_JUNCTION_MODEL_TASKS = ("component", "ocr", "pin", "wire")
COORDINATE_ROUNDTRIP_LIMIT_PX = 0.5
DETERMINISM_RUNS = 3
PERFORMANCE_CASES = 50
CANONICAL_RELEASE_MANIFEST = "data/splits/dataset300_release50_v1.jsonl"
CANONICAL_RELEASE_SPLIT = "release50"
RELEASE_SUITE_CONTRACT = "canonical_dataset300_release50_run_binding_v1"
PERFORMANCE_DEADLINE_SECONDS = 2 * 60 * 60
PERFORMANCE_RESERVE_FRACTION = 0.10
PERFORMANCE_PASS_SECONDS = PERFORMANCE_DEADLINE_SECONDS * (
    1.0 - PERFORMANCE_RESERVE_FRACTION
)
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
OFFICIAL_GUIDE_SHA256 = (
    "45eedbc195ad71bbb03044c4606ea85ccf3eccebbb3d9b3304d6e325e1acc5e8"
)
PRODUCTION_RUNTIME_SOURCE_DIRECTORIES = (
    "src/pcbparse/association",
    "src/pcbparse/graph",
    "src/pcbparse/io",
    "src/pcbparse/ir",
    "src/pcbparse/nets",
    "src/pcbparse/output",
    "src/pcbparse/perception",
    "src/pcbparse/pipeline",
    "src/pcbparse/reasoning",
    "src/pcbparse/training",
    "src/pcbparse/validation",
)
PRODUCTION_RUNTIME_SOURCE_FILES = (
    "src/pcbparse/config.py",
    "src/pcbparse/evaluation/infer_manifest.py",
    "src/pcbparse/evaluation/raw_cli.py",
    "src/pcbparse/resources/official_schema.json",
    "src/pcbparse/service/readiness.py",
)


class AuditInputError(ValueError):
    """Raised for malformed evidence; a release audit must fail closed."""


def _reject_constant(value: str) -> None:
    raise AuditInputError(f"non-standard JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditInputError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, AuditInputError) as exc:
        raise AuditInputError(f"invalid JSON at {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AuditInputError(f"cannot read JSONL at {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (json.JSONDecodeError, AuditInputError) as exc:
            raise AuditInputError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise AuditInputError(f"JSONL row at {path}:{line_number} is not an object")
        rows.append(row)
    return rows


def _canonical_bytes(value: Any) -> bytes:
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


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _compact_json_sha256(value: Any) -> str:
    """Match ``pcbparse.config.config_hash`` without importing the package."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _independent_production_runtime_source_manifest(root: Path) -> dict[str, Any]:
    """Recompute the inference runtime source population without importing it.

    This intentionally duplicates the release contract owned by
    ``evaluation.infer_manifest``.  The hard-gate auditor must not trust the
    code that produced the evidence it is checking, otherwise a historical
    summary could be wrapped with hashes from newer source bytes.
    """

    repository_root = root.resolve(strict=True)
    paths: set[Path] = set()
    for relative in PRODUCTION_RUNTIME_SOURCE_FILES:
        path = (repository_root / relative).resolve(strict=False)
        try:
            path.relative_to(repository_root)
        except ValueError as exc:  # pragma: no cover - repository constants.
            raise AuditInputError(
                f"runtime source path escapes workspace: {relative}"
            ) from exc
        if not path.is_file() or path.stat().st_size <= 0:
            raise AuditInputError(
                f"runtime source file is missing or empty: {relative}"
            )
        paths.add(path)
    for relative in PRODUCTION_RUNTIME_SOURCE_DIRECTORIES:
        directory = (repository_root / relative).resolve(strict=False)
        try:
            directory.relative_to(repository_root)
        except ValueError as exc:  # pragma: no cover - repository constants.
            raise AuditInputError(
                f"runtime source directory escapes workspace: {relative}"
            ) from exc
        if not directory.is_dir():
            raise AuditInputError(f"runtime source directory is missing: {relative}")
        discovered = [
            path.resolve()
            for path in directory.rglob("*.py")
            if path.is_file()
        ]
        if not discovered:
            raise AuditInputError(
                f"runtime source directory contains no Python files: {relative}"
            )
        paths.update(discovered)
    files = [
        {
            "path": path.relative_to(repository_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(
            paths, key=lambda item: item.relative_to(repository_root).as_posix()
        )
    ]
    return {
        "schema_version": 1,
        "contract": "production_runtime_source_manifest_v1",
        "file_count": len(files),
        "files": files,
        "source_set_sha256": _compact_json_sha256(files),
    }


def _relative(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)


def _path_evidence(root: Path, path: Path | None) -> dict[str, Any]:
    present = path is not None and path.is_file()
    return {
        "path": _relative(root, path),
        "present": present,
        "sha256": _sha256_file(path) if present and path is not None else None,
    }


def _resolve_optional(root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    resolved = (path if path.is_absolute() else root / path).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise AuditInputError(f"evidence path escapes workspace: {resolved}") from exc
    return resolved


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path.resolve()
    return None


def _parse_utc_timestamp(value: Any, *, field: str) -> tuple[datetime | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, f"{field} is missing"
    rendered = value.strip()
    try:
        parsed = datetime.fromisoformat(rendered[:-1] + "+00:00" if rendered.endswith("Z") else rendered)
    except ValueError:
        return None, f"{field} is not an ISO-8601 timestamp"
    if parsed.tzinfo is None:
        return None, f"{field} must include a timezone"
    parsed = parsed.astimezone(timezone.utc)
    if parsed > datetime.now(timezone.utc) + timedelta(minutes=5):
        # Allow ordinary host/container clock skew, but never let a materially
        # future-dated file win automatic selection.
        return None, f"{field} is future-dated"
    return parsed, None


def _resolve_workspace_evidence_path(
    root: Path,
    raw: Any,
    *,
    field: str,
) -> tuple[Path | None, str | None]:
    if not isinstance(raw, str) or not raw.strip():
        return None, f"{field} is missing"
    candidate = Path(raw.strip()).expanduser()
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError:
        return None, f"{field} escapes the workspace"
    return resolved, None


def _production_config_binding(
    root: Path,
    pipeline_path: Path | None,
    profile_path: Path | None,
) -> dict[str, Any]:
    """Resolve the exact semantic config hash used by formal production runs.

    The production harness hashes the pipeline after applying the selected
    profile.  Reproducing that small, stable transformation here keeps this
    independent audit executable even when ``PYTHONPATH`` is not configured.
    """

    pipeline = _path_evidence(root, pipeline_path)
    profile = _path_evidence(root, profile_path)
    problems: list[str] = []
    effective: dict[str, Any] | None = None
    if not pipeline["present"]:
        problems.append("production pipeline config is missing")
    if not profile["present"]:
        problems.append("production profile is missing")
    if not problems and pipeline_path is not None and profile_path is not None:
        try:
            raw_pipeline = _read_json(pipeline_path)
            raw_profile = _read_json(profile_path)
        except AuditInputError as exc:
            problems.append(str(exc))
        else:
            if not isinstance(raw_pipeline, Mapping):
                problems.append("production pipeline config root is not an object")
            if not isinstance(raw_profile, Mapping):
                problems.append("production profile root is not an object")
            required = {
                "name",
                "provider",
                "qwen_enabled",
                "artifact_policy",
                "strict_validation",
            }
            if isinstance(raw_profile, Mapping):
                missing = sorted(required - set(raw_profile))
                if missing:
                    problems.append(f"production profile misses fields {missing!r}")
                if str(raw_profile.get("name", "")).strip().lower() not in {
                    "baseline",
                    "competition",
                }:
                    problems.append("production profile name is not baseline or competition")
                if str(raw_profile.get("provider", "")).strip().lower() != "production":
                    problems.append("production profile provider is not production")
            if not problems and isinstance(raw_pipeline, Mapping) and isinstance(raw_profile, Mapping):
                effective = copy.deepcopy(dict(raw_pipeline))
                run = effective.setdefault("run", {})
                qwen = effective.setdefault("qwen", {})
                validation = effective.setdefault("validation", {})
                if not all(isinstance(value, dict) for value in (run, qwen, validation)):
                    problems.append("pipeline run/qwen/validation sections must be objects")
                    effective = None
                else:
                    run["profile"] = str(raw_profile["name"]).strip().lower()
                    run["artifact_policy"] = str(raw_profile["artifact_policy"])
                    qwen["enabled"] = bool(raw_profile["qwen_enabled"])
                    validation["require_zero_errors"] = bool(raw_profile["strict_validation"])

    graph = effective.get("graph", {}) if isinstance(effective, Mapping) else {}
    learned_junctions_enabled = (
        graph.get("learned_junctions_enabled")
        if isinstance(graph, Mapping)
        else None
    )
    if learned_junctions_enabled is not None and type(learned_junctions_enabled) is not bool:
        problems.append("pipeline graph.learned_junctions_enabled must be boolean")
    return {
        "status": "valid" if not problems and effective is not None else "invalid",
        "pipeline": pipeline,
        "profile": profile,
        "effective_config_sha256": (
            _compact_json_sha256(effective) if effective is not None and not problems else None
        ),
        "profile_name": (
            effective.get("run", {}).get("profile") if isinstance(effective, Mapping) else None
        ),
        "learned_junctions_enabled": learned_junctions_enabled,
        "effective_config": effective,
        "problems": problems,
    }


def _stable_prediction_set_sha256(rows: Sequence[Mapping[str, Any]]) -> str | None:
    fields = (
        "case_id",
        "duplicate_group",
        "image_sha256",
        "ir_sha256",
        "official_sha256",
        "official_output_sha256",
        "provider_kind",
        "model_lock_sha256",
        "config_hash",
        "sidecar_access",
        "status",
        "error",
    )
    if any(any(field not in row for field in fields) for row in rows):
        return None
    return _compact_json_sha256([{field: row[field] for field in fields} for row in rows])


def _audit_production_summary_binding(
    root: Path,
    summary_path: Path,
    model_lock_sha256: str | None,
    production_config_sha256: str | None,
    *,
    require_success: bool,
    require_embedded_time: bool,
) -> dict[str, Any]:
    """Verify a summary, its JSONL bytes, rows, and materialized outputs."""

    evidence = _path_evidence(root, summary_path)
    problems: list[str] = []
    payload: Mapping[str, Any] = {}
    rows: list[dict[str, Any]] = []
    generated: datetime | None = None
    manifest_path: Path | None = None
    expected_shared_inputs: set[str] | None = None
    dataset_root: Path | None = None
    expected_model_task_plan: dict[str, Any] | None = None
    runtime_source_manifest: Mapping[str, Any] | None = None
    if not isinstance(model_lock_sha256, str) or not HEX_SHA256.fullmatch(model_lock_sha256):
        problems.append("current models.lock SHA-256 is unavailable")
    if not isinstance(production_config_sha256, str) or not HEX_SHA256.fullmatch(
        production_config_sha256
    ):
        problems.append("current production config SHA-256 is unavailable")
    if not evidence["present"]:
        problems.append("production summary is missing")
    else:
        try:
            loaded = _read_json(summary_path)
        except AuditInputError as exc:
            problems.append(str(exc))
        else:
            if not isinstance(loaded, Mapping):
                problems.append("production summary root is not an object")
            else:
                payload = loaded
                current_binding = _production_config_binding(
                    root,
                    root / "configs" / "pipeline.yaml",
                    root / "configs" / "profiles" / "competition.yaml",
                )
                learned_junctions_enabled = current_binding.get(
                    "learned_junctions_enabled"
                )
                expected_tasks = sorted(
                    REQUIRED_MODEL_TASKS
                    if learned_junctions_enabled is True
                    else DETERMINISTIC_JUNCTION_MODEL_TASKS
                )
                expected_model_task_plan = {
                    "contract": "production_model_task_plan_v1",
                    "junction_mode": (
                        "learned_model_plus_deterministic_vectorizer"
                        if learned_junctions_enabled is True
                        else "deterministic_vectorizer"
                    ),
                    "required_model_tasks": expected_tasks,
                    "loaded_model_tasks": expected_tasks,
                    "learned_junction_model_required": learned_junctions_enabled is True,
                    "learned_junction_model_loaded": learned_junctions_enabled is True,
                    "learned_junction_model_invoked": learned_junctions_enabled is True,
                }
                resolved_perception = loaded.get("resolved_perception_config")
                readiness_payload = loaded.get("readiness")
                checks = {
                    "kind": loaded.get("kind") == "production-image-only-inference",
                    "accuracy_mode": loaded.get("accuracy_mode") == "image-only",
                    "execution_mode": loaded.get("execution_mode") == "production",
                    "provider_kind": loaded.get("provider_kind") == "production",
                    "sidecar_access": loaded.get("sidecar_access") is False,
                    "model_lock_sha256": loaded.get("model_lock_sha256") == model_lock_sha256,
                    "config_hash": loaded.get("config_hash") == production_config_sha256,
                    "readiness_status": isinstance(loaded.get("readiness"), Mapping)
                    and loaded.get("readiness", {}).get("status") == "ready",
                    "readiness_ready": isinstance(loaded.get("readiness"), Mapping)
                    and loaded.get("readiness", {}).get("ready") is True,
                    "current_config_binding": current_binding.get("status") == "valid"
                    and current_binding.get("effective_config_sha256")
                    == production_config_sha256,
                    "resolved_perception_config": isinstance(
                        resolved_perception, Mapping
                    )
                    and resolved_perception.get("learned_junctions_enabled")
                    is learned_junctions_enabled,
                    "resolved_perception_config_sha256": isinstance(
                        resolved_perception, Mapping
                    )
                    and loaded.get("resolved_perception_config_sha256")
                    == _compact_json_sha256(resolved_perception),
                    "model_task_plan": loaded.get("model_task_plan")
                    == expected_model_task_plan,
                    "readiness_required_tasks": isinstance(
                        readiness_payload, Mapping
                    )
                    and readiness_payload.get("required_learned_tasks")
                    == expected_tasks,
                    "readiness_registered_tasks": isinstance(
                        readiness_payload, Mapping
                    )
                    and readiness_payload.get("registered_learned_tasks")
                    == expected_tasks,
                    "readiness_verified_tasks": isinstance(
                        readiness_payload, Mapping
                    )
                    and readiness_payload.get("verified_learned_tasks")
                    == expected_tasks,
                    "exact_case_coverage": isinstance(loaded.get("coverage"), Mapping)
                    and loaded.get("coverage", {}).get("exact_case_coverage") is True,
                    "exact_duplicate_group_coverage": isinstance(loaded.get("coverage"), Mapping)
                    and loaded.get("coverage", {}).get("exact_duplicate_group_coverage") is True,
                }
                problems.extend(name for name, passed in checks.items() if not passed)
                if require_success:
                    success_checks = {
                        "status": loaded.get("status") == "ok",
                        "all_successful": loaded.get("all_successful") is True,
                        "failed": loaded.get("failed") == 0,
                        "not_run": loaded.get("not_run") == 0,
                        "succeeded_equals_selected": type(loaded.get("succeeded")) is int
                        and loaded.get("succeeded") == loaded.get("selected_cases"),
                    }
                    problems.extend(name for name, passed in success_checks.items() if not passed)
                raw_time = loaded.get("generated_at_utc", loaded.get("completed_at_utc"))
                generated, time_problem = _parse_utc_timestamp(
                    raw_time,
                    field="production summary generated_at_utc",
                )
                if require_embedded_time and time_problem:
                    problems.append(time_problem)
                inference_run_id = loaded.get("inference_run_id")
                if require_embedded_time and (
                    not isinstance(inference_run_id, str)
                    or not HEX_SHA256.fullmatch(inference_run_id)
                ):
                    problems.append("production summary lacks a SHA-256 inference_run_id")

                if require_embedded_time:
                    raw_runtime_sources = loaded.get("runtime_source_manifest")
                    if not isinstance(raw_runtime_sources, Mapping):
                        problems.append(
                            "production summary runtime_source_manifest is missing"
                        )
                    else:
                        runtime_source_manifest = raw_runtime_sources
                        try:
                            current_runtime_sources = (
                                _independent_production_runtime_source_manifest(root)
                            )
                        except AuditInputError as exc:
                            problems.append(str(exc))
                        else:
                            raw_runtime_files = raw_runtime_sources.get("files")
                            runtime_checks = {
                                "runtime_source_schema": raw_runtime_sources.get(
                                    "schema_version"
                                )
                                == 1,
                                "runtime_source_contract": raw_runtime_sources.get(
                                    "contract"
                                )
                                == "production_runtime_source_manifest_v1",
                                "runtime_source_file_count": isinstance(
                                    raw_runtime_files, list
                                )
                                and type(raw_runtime_sources.get("file_count")) is int
                                and raw_runtime_sources.get("file_count")
                                == len(raw_runtime_files),
                                "runtime_source_current_bytes": dict(
                                    raw_runtime_sources
                                )
                                == current_runtime_sources,
                            }
                            problems.extend(
                                name
                                for name, passed in runtime_checks.items()
                                if not passed
                            )

                    timing = loaded.get("timing")
                    if not isinstance(timing, Mapping):
                        problems.append("production summary timing contract is missing")
                    else:
                        timing_checks = {
                            "timing_contract": timing.get("contract")
                            == "production_wall_clock_scope_v1",
                            "timing_scope": timing.get("scope")
                            == "function_entry_through_prediction_manifest_fsync",
                            "timing_cold_start": timing.get("cold_start_included") is True,
                            "timing_config_load": timing.get(
                                "config_and_profile_load_included"
                            )
                            is True,
                            "timing_readiness": timing.get("readiness_included") is True,
                            "timing_provider_load": timing.get(
                                "provider_and_model_load_included"
                            )
                            is True,
                            "timing_case_loop": timing.get("case_loop_included") is True,
                            "timing_prediction_manifest": timing.get(
                                "prediction_manifest_write_included"
                            )
                            is True,
                            "timing_summary_boundary": timing.get(
                                "summary_write_and_cli_teardown_included"
                            )
                            is False,
                        }
                        problems.extend(
                            name for name, passed in timing_checks.items() if not passed
                        )
                        timing_values: dict[str, float] = {}
                        for field in (
                            "initialization_seconds",
                            "case_loop_seconds",
                            "evidence_finalize_seconds",
                            "total_elapsed_seconds",
                        ):
                            raw_value = timing.get(field)
                            if (
                                not isinstance(raw_value, (int, float))
                                or isinstance(raw_value, bool)
                                or not math.isfinite(float(raw_value))
                                or float(raw_value) < 0.0
                            ):
                                problems.append(
                                    f"production summary timing.{field} is invalid"
                                )
                            else:
                                timing_values[field] = float(raw_value)
                        raw_elapsed = loaded.get("elapsed_seconds")
                        if (
                            not isinstance(raw_elapsed, (int, float))
                            or isinstance(raw_elapsed, bool)
                            or not math.isfinite(float(raw_elapsed))
                            or float(raw_elapsed) < 0.0
                        ):
                            problems.append("production summary elapsed_seconds is invalid")
                        elif len(timing_values) == 4:
                            total = timing_values["total_elapsed_seconds"]
                            partitioned = (
                                timing_values["initialization_seconds"]
                                + timing_values["case_loop_seconds"]
                                + timing_values["evidence_finalize_seconds"]
                            )
                            if not math.isclose(
                                float(raw_elapsed), total, rel_tol=0.0, abs_tol=1e-9
                            ):
                                problems.append(
                                    "production summary elapsed_seconds differs from timing total"
                                )
                            if not math.isclose(
                                partitioned, total, rel_tol=0.0, abs_tol=1e-9
                            ):
                                problems.append(
                                    "production summary timing partitions do not reproduce total"
                                )

                if require_embedded_time:
                    control_inputs = loaded.get("control_inputs")
                    if not isinstance(control_inputs, Mapping):
                        problems.append("production summary control_inputs is missing")
                    else:
                        expected_controls = {
                            "pipeline_config": root / "configs" / "pipeline.yaml",
                            "profile": root / "configs" / "profiles" / "competition.yaml",
                            "models_lock": root / "models" / "models.lock.json",
                            "official_profile": (
                                root
                                / "src"
                                / "pcbparse"
                                / "resources"
                                / "official_schema.json"
                            ),
                        }
                        resolved_controls: dict[str, Path] = {}
                        for field, expected_path in expected_controls.items():
                            resolved, path_problem = _resolve_workspace_evidence_path(
                                root,
                                control_inputs.get(field),
                                field=f"production summary control_inputs.{field}",
                            )
                            if path_problem:
                                problems.append(path_problem)
                            elif resolved != expected_path.resolve(strict=False):
                                problems.append(
                                    f"production summary control_inputs.{field} is not the canonical release input"
                                )
                            elif resolved is not None and not resolved.is_file():
                                problems.append(
                                    f"production summary control_inputs.{field} is missing"
                                )
                            elif resolved is not None:
                                resolved_controls[field] = resolved

                        dataset_root, dataset_problem = _resolve_workspace_evidence_path(
                            root,
                            control_inputs.get("dataset_root"),
                            field="production summary control_inputs.dataset_root",
                        )
                        if dataset_problem:
                            problems.append(dataset_problem)
                        elif dataset_root is None or not dataset_root.is_dir():
                            problems.append("production summary dataset_root is missing")

                        split_path, split_problem = _resolve_workspace_evidence_path(
                            root,
                            control_inputs.get("split_manifest"),
                            field="production summary control_inputs.split_manifest",
                        )
                        if split_problem:
                            problems.append(split_problem)
                        elif split_path is None or not split_path.is_file():
                            problems.append("production summary split_manifest is missing")
                        else:
                            if loaded.get("split_manifest_sha256") != _sha256_file(split_path):
                                problems.append(
                                    "production summary split_manifest_sha256 does not match bytes"
                                )
                            if loaded.get("selection_control_paths_read") != [str(split_path)]:
                                problems.append(
                                    "production summary selection_control_paths_read is not exactly the split manifest"
                                )

                        lock_path = resolved_controls.get("models_lock")
                        model_paths: list[Path] = []
                        lock_model_tasks: list[str] = []
                        if lock_path is not None:
                            try:
                                lock_payload = _read_json(lock_path)
                            except AuditInputError as exc:
                                problems.append(str(exc))
                            else:
                                raw_models = (
                                    lock_payload.get("models")
                                    if isinstance(lock_payload, Mapping)
                                    else None
                                )
                                if not isinstance(raw_models, list):
                                    problems.append("models.lock has no models list")
                                else:
                                    for model_index, raw_model in enumerate(raw_models):
                                        raw_task = (
                                            raw_model.get("task")
                                            if isinstance(raw_model, Mapping)
                                            else None
                                        )
                                        if not isinstance(raw_task, str) or not raw_task:
                                            problems.append(
                                                f"models.lock entry {model_index} has no task"
                                            )
                                        else:
                                            lock_model_tasks.append(raw_task)
                                        raw_model_path = (
                                            raw_model.get("path")
                                            if isinstance(raw_model, Mapping)
                                            else None
                                        )
                                        if not isinstance(raw_model_path, str) or not raw_model_path:
                                            problems.append(
                                                f"models.lock entry {model_index} has no artifact path"
                                            )
                                            continue
                                        model_path = (
                                            lock_path.parent / raw_model_path
                                        ).resolve(strict=False)
                                        try:
                                            model_path.relative_to(lock_path.parent.resolve())
                                        except ValueError:
                                            problems.append(
                                                f"models.lock entry {model_index} artifact escapes models root"
                                            )
                                            continue
                                        if not model_path.is_file():
                                            problems.append(
                                                f"models.lock entry {model_index} artifact is missing"
                                            )
                                            continue
                                        if raw_model.get("sha256") != _sha256_file(model_path):
                                            problems.append(
                                                f"models.lock entry {model_index} artifact hash does not match bytes"
                                            )
                                        model_paths.append(model_path)
                                    if (
                                        expected_model_task_plan is not None
                                        and sorted(lock_model_tasks)
                                        != expected_model_task_plan["required_model_tasks"]
                                    ):
                                        problems.append(
                                            "models.lock task set differs from resolved model task plan"
                                        )
                        if len(resolved_controls) == len(expected_controls):
                            expected_shared_inputs = {
                                str(path.resolve(strict=False)).casefold()
                                for path in [*resolved_controls.values(), *model_paths]
                            }

                raw_manifest = loaded.get("prediction_manifest")
                if not isinstance(raw_manifest, str) or not raw_manifest.strip():
                    problems.append("prediction_manifest is missing")
                else:
                    candidate = (summary_path.parent / raw_manifest.strip()).resolve(strict=False)
                    try:
                        candidate.relative_to(summary_path.parent.resolve())
                    except ValueError:
                        problems.append("prediction_manifest escapes the summary directory")
                    else:
                        manifest_path = candidate
                        if not manifest_path.is_file():
                            problems.append("prediction_manifest file is missing")
                        elif loaded.get("prediction_manifest_sha256") != _sha256_file(manifest_path):
                            problems.append("prediction_manifest_sha256 does not match bytes")
                        else:
                            try:
                                rows = _read_jsonl(manifest_path)
                            except AuditInputError as exc:
                                problems.append(str(exc))

    if rows:
        if payload.get("selected_cases") != len(rows):
            problems.append("prediction manifest row count differs from selected_cases")
        case_ids = [row.get("case_id") for row in rows]
        if any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
            problems.append("prediction manifest has invalid case IDs")
        if len(set(case_ids)) != len(case_ids):
            problems.append("prediction manifest has duplicate case IDs")
        coverage = payload.get("coverage")
        if require_embedded_time and isinstance(coverage, Mapping):
            actual_case_hash = _compact_json_sha256(case_ids)
            if coverage.get("manifest_case_ids_sha256") != actual_case_hash:
                problems.append("coverage manifest_case_ids_sha256 does not match rows")
            if coverage.get("expected_case_ids_sha256") != actual_case_hash:
                problems.append("coverage expected_case_ids_sha256 does not match rows")
            membership: dict[str, list[str]] = {}
            for row in rows:
                group = str(row.get("duplicate_group", ""))
                membership.setdefault(group, []).append(str(row.get("case_id", "")))
            membership = dict(sorted(membership.items()))
            if coverage.get("expected_duplicate_group_membership_sha256") != _compact_json_sha256(
                membership
            ):
                problems.append(
                    "coverage duplicate-group membership hash does not match rows"
                )
        elif require_embedded_time:
            problems.append("production summary coverage is missing")
        official_paths: set[str] = set()
        ir_paths: set[str] = set()
        image_paths: set[str] = set()
        for index, row in enumerate(rows):
            if row.get("model_lock_sha256") != model_lock_sha256:
                problems.append(f"prediction row {index} is not bound to current models.lock")
            if row.get("config_hash") != production_config_sha256:
                problems.append(f"prediction row {index} is not bound to production config")
            if require_embedded_time and row.get("inference_run_id") != payload.get(
                "inference_run_id"
            ):
                problems.append(
                    f"prediction row {index} inference_run_id does not match summary"
                )
            if row.get("provider_kind") != "production" or row.get("sidecar_access") is not False:
                problems.append(f"prediction row {index} is not formal image-only production")
            if require_success and (row.get("status") != "ok" or row.get("error") is not None):
                problems.append(f"prediction row {index} is not successful")
            if require_embedded_time:
                input_paths = row.get("input_paths_read")
                if not isinstance(input_paths, list) or not input_paths:
                    problems.append(f"prediction row {index} lacks input_paths_read")
                else:
                    resolved_inputs: list[Path] = []
                    for input_index, raw_input in enumerate(input_paths):
                        resolved_input, input_problem = _resolve_workspace_evidence_path(
                            root,
                            raw_input,
                            field=f"prediction row {index} input_paths_read[{input_index}]",
                        )
                        if input_problem:
                            problems.append(input_problem)
                        elif resolved_input is not None:
                            resolved_inputs.append(resolved_input)
                    normalized_inputs = [
                        str(path.resolve(strict=False)).casefold() for path in resolved_inputs
                    ]
                    if len(set(normalized_inputs)) != len(normalized_inputs):
                        problems.append(
                            f"prediction row {index} input_paths_read contains duplicate paths"
                        )
                    if resolved_inputs:
                        image_path = resolved_inputs[0]
                        normalized_image = normalized_inputs[0]
                        if normalized_image in image_paths:
                            problems.append(
                                f"prediction row {index} reuses an image input"
                            )
                        image_paths.add(normalized_image)
                        if dataset_root is None:
                            problems.append(
                                f"prediction row {index} image input is not bound to dataset_root"
                            )
                        else:
                            try:
                                image_path.relative_to(dataset_root.resolve())
                            except ValueError:
                                problems.append(
                                    f"prediction row {index} image input escapes dataset_root"
                                )
                        if not image_path.is_file():
                            problems.append(f"prediction row {index} image input is missing")
                        elif row.get("image_sha256") != _sha256_file(image_path):
                            problems.append(
                                f"prediction row {index} image_sha256 does not match bytes"
                            )
                        if expected_shared_inputs is None or set(normalized_inputs[1:]) != expected_shared_inputs:
                            problems.append(
                                f"prediction row {index} input_paths_read is not the exact release allowlist"
                            )
            for path_field, hash_field in (
                ("official_path", "official_sha256"),
                ("ir_path", "ir_sha256"),
            ):
                raw_path = row.get(path_field)
                if raw_path is None and not require_success:
                    continue
                if not isinstance(raw_path, str) or not raw_path.strip():
                    problems.append(f"prediction row {index} lacks {path_field}")
                    continue
                seen_paths = official_paths if path_field == "official_path" else ir_paths
                candidate = (summary_path.parent / raw_path.strip()).resolve(strict=False)
                try:
                    candidate.relative_to(summary_path.parent.resolve())
                except ValueError:
                    problems.append(f"prediction row {index} {path_field} escapes run directory")
                    continue
                normalized_output_path = str(candidate).casefold()
                if normalized_output_path in seen_paths:
                    problems.append(f"prediction row {index} reuses {path_field}")
                seen_paths.add(normalized_output_path)
                if not candidate.is_file():
                    problems.append(f"prediction row {index} {path_field} is missing")
                else:
                    actual_hash = _sha256_file(candidate)
                    if row.get(hash_field) != actual_hash:
                        problems.append(
                            f"prediction row {index} {hash_field} does not match bytes"
                        )
                    if (
                        path_field == "official_path"
                        and row.get("official_output_sha256") != actual_hash
                    ):
                        problems.append(
                            f"prediction row {index} official_output_sha256 does not match bytes"
                        )
                    if path_field == "ir_path" and expected_model_task_plan is not None:
                        try:
                            ir_payload = _read_json(candidate)
                        except AuditInputError as exc:
                            problems.append(str(exc))
                        else:
                            metadata = (
                                ir_payload.get("metadata")
                                if isinstance(ir_payload, Mapping)
                                else None
                            )
                            if not isinstance(metadata, Mapping):
                                problems.append(
                                    f"prediction row {index} Circuit-IR metadata is missing"
                                )
                            else:
                                if metadata.get("model_task_plan") != expected_model_task_plan:
                                    problems.append(
                                        f"prediction row {index} Circuit-IR model task plan differs"
                                    )
                                model_versions = metadata.get("model_versions")
                                if (
                                    not isinstance(model_versions, Mapping)
                                    or sorted(model_versions) != expected_model_task_plan[
                                        "loaded_model_tasks"
                                    ]
                                ):
                                    problems.append(
                                        f"prediction row {index} Circuit-IR model_versions task set differs"
                                    )
                                if (
                                    expected_model_task_plan["junction_mode"]
                                    == "deterministic_vectorizer"
                                    and metadata.get("quarantined_model_versions", {})
                                    not in ({}, None)
                                ):
                                    problems.append(
                                        f"prediction row {index} retains a learned Junction model in deterministic mode"
                                    )
        stable_hash = _stable_prediction_set_sha256(rows)
        if stable_hash is None:
            problems.append("prediction rows lack stable prediction-set fields")
        elif payload.get("prediction_set_sha256") != stable_hash:
            problems.append("prediction_set_sha256 does not match manifest rows")
    elif require_success and evidence["present"]:
        problems.append("prediction manifest is empty")

    return {
        "path": evidence["path"],
        "sha256": evidence["sha256"],
        "present": evidence["present"],
        "valid": not problems,
        "formal": not problems,
        "generated_at_utc": (
            generated.isoformat().replace("+00:00", "Z") if generated is not None else None
        ),
        "completed_at_utc": payload.get(
            "completed_at_utc", payload.get("generated_at_utc")
        ),
        "generated_timestamp": generated.timestamp() if generated is not None else None,
        "inference_run_id": payload.get("inference_run_id"),
        "model_lock_sha256": payload.get("model_lock_sha256"),
        "config_hash": payload.get("config_hash"),
        "status": payload.get("status"),
        "all_successful": payload.get("all_successful"),
        "selected_cases": payload.get("selected_cases"),
        "succeeded": payload.get("succeeded"),
        "failed": payload.get("failed"),
        "not_run": payload.get("not_run"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "timing": payload.get("timing"),
        "model_task_plan": payload.get("model_task_plan"),
        "runtime_source_manifest": (
            dict(runtime_source_manifest)
            if isinstance(runtime_source_manifest, Mapping)
            else None
        ),
        "runtime_source_manifest_sha256": (
            _compact_json_sha256(runtime_source_manifest)
            if isinstance(runtime_source_manifest, Mapping)
            else None
        ),
        "junction_runtime_non_load_non_invocation": bool(
            expected_model_task_plan is not None
            and expected_model_task_plan["junction_mode"]
            == "deterministic_vectorizer"
            and payload.get("model_task_plan") == expected_model_task_plan
            and not problems
        ),
        "prediction_manifest": _relative(root, manifest_path),
        "prediction_manifest_sha256": (
            _sha256_file(manifest_path) if manifest_path is not None and manifest_path.is_file() else None
        ),
        "prediction_set_sha256": payload.get("prediction_set_sha256"),
        "official_output_hashes": sorted(
            str(row.get("official_sha256"))
            for row in rows
            if isinstance(row.get("official_sha256"), str)
        ),
        "official_output_manifest": sorted(
            (
                {
                    "path": (
                        str(row["official_path"]).replace("\\", "/")[len("official/") :]
                        if str(row["official_path"]).replace("\\", "/").startswith(
                            "official/"
                        )
                        else str(row["official_path"]).replace("\\", "/")
                    ),
                    "sha256": str(row["official_sha256"]),
                }
                for row in rows
                if isinstance(row.get("official_path"), str)
                and isinstance(row.get("official_sha256"), str)
            ),
            key=lambda item: item["path"],
        ),
        "problems": problems,
    }


def _template_pattern(template: str, fields: set[str]) -> re.Pattern[str]:
    pattern = "^"
    seen: set[str] = set()
    try:
        for literal, field_name, _format_spec, _conversion in Formatter().parse(template):
            pattern += re.escape(literal)
            if field_name is not None:
                if field_name not in fields or field_name in seen:
                    raise AuditInputError(
                        f"unsupported or repeated template field {field_name!r}"
                    )
                pattern += f"(?P<{field_name}>.+?)"
                seen.add(field_name)
    except ValueError as exc:
        raise AuditInputError(f"invalid template {template!r}: {exc}") from exc
    if seen != fields:
        raise AuditInputError(
            f"template {template!r} must contain fields {sorted(fields)!r}"
        )
    return re.compile(pattern + "$")


def analyze_guide_output(output: Any, schema: Mapping[str, Any]) -> dict[str, Any]:
    """Count guide-v1 exact-shape and dangling-pin failures.

    This deliberately describes *internal guide evidence*.  It never upgrades
    the local profile to an official competition contract.
    """

    extra_fields = 0
    dangling_references = 0
    schema_errors: list[str] = []
    examples: list[str] = []

    def problem(message: str) -> None:
        schema_errors.append(message)
        if len(examples) < 20:
            examples.append(message)

    def exact_mapping(value: Any, expected: Sequence[str], location: str) -> Mapping[str, Any] | None:
        nonlocal extra_fields
        if not isinstance(value, Mapping):
            problem(f"{location} must be an object")
            return None
        actual = set(value)
        wanted = set(expected)
        extras = sorted(str(item) for item in actual - wanted)
        missing = sorted(str(item) for item in wanted - actual)
        extra_fields += len(extras)
        if extras:
            problem(f"{location} has extra keys {extras!r}")
        if missing:
            problem(f"{location} misses keys {missing!r}")
        return value

    top_level = schema.get("top_level_keys")
    if not isinstance(top_level, list) or not all(isinstance(item, str) for item in top_level):
        raise AuditInputError("local guide schema has invalid top_level_keys")
    root = exact_mapping(output, top_level, "output")
    if root is None:
        return {
            "extra_fields": extra_fields,
            "dangling_references": dangling_references,
            "schema_error_count": len(schema_errors),
            "examples": examples,
            "clean": False,
        }

    components = root.get("components")
    pins = root.get("pins")
    nets = root.get("nets")
    if not isinstance(components, Mapping):
        problem("output.components must be an object")
        components = {}
    if not isinstance(pins, Mapping):
        problem("output.pins must be an object")
        pins = {}
    if not isinstance(nets, Mapping):
        problem("output.nets must be an object")
        nets = {}

    component_fields = schema.get("component", {}).get("fields")
    pin_fields = schema.get("pin", {}).get("fields")
    net_fields = schema.get("net", {}).get("fields")
    if not all(
        isinstance(value, list) and all(isinstance(item, str) for item in value)
        for value in (component_fields, pin_fields, net_fields)
    ):
        raise AuditInputError("local guide schema has invalid section fields")

    for key, component in components.items():
        exact_mapping(component, component_fields, f"components.{key}")

    pin_template = schema.get("pin", {}).get("key_template")
    reference_template = schema.get("net", {}).get("hypergraph_pin_template")
    if not isinstance(pin_template, str) or not isinstance(reference_template, str):
        raise AuditInputError("local guide schema lacks pin/reference templates")
    pin_pattern = _template_pattern(pin_template, {"number"})
    available_refs: set[str] = set()
    for component_key, group in pins.items():
        if component_key not in components:
            dangling_references += 1
            problem(f"pins.{component_key} has no matching component")
        if not isinstance(group, Mapping):
            problem(f"pins.{component_key} must be an object")
            continue
        for pin_key, pin in group.items():
            match = pin_pattern.fullmatch(str(pin_key))
            if match is None:
                problem(f"pins.{component_key}.{pin_key} violates pin key template")
            else:
                available_refs.add(
                    reference_template.format(
                        component_key=component_key,
                        pin_number=match.group("number"),
                    )
                )
            pin_value = exact_mapping(pin, pin_fields, f"pins.{component_key}.{pin_key}")
            if pin_value is not None:
                exact_mapping(
                    pin_value.get("point"),
                    ["x", "y"],
                    f"pins.{component_key}.{pin_key}.point",
                )

    separator = schema.get("net", {}).get("hypergraph_separator", ",")
    if not isinstance(separator, str) or not separator:
        raise AuditInputError("local guide schema has invalid hypergraph separator")
    for net_key, net in nets.items():
        net_value = exact_mapping(net, net_fields, f"nets.{net_key}")
        if net_value is None:
            continue
        hypergraph = net_value.get("hyperGraph")
        if not isinstance(hypergraph, str) or not hypergraph.startswith("(") or not hypergraph.endswith(")"):
            problem(f"nets.{net_key}.hyperGraph must be a parenthesized string")
        else:
            encoded = hypergraph[1:-1]
            references = encoded.split(separator) if encoded else []
            for reference in references:
                if reference not in available_refs:
                    dangling_references += 1
                    problem(f"nets.{net_key} references unknown pin {reference!r}")
        edges = net_value.get("edges")
        if not isinstance(edges, Mapping):
            problem(f"nets.{net_key}.edges must be an object")
            continue
        for edge_key, edge in edges.items():
            if not isinstance(edge, list) or len(edge) != 2:
                problem(f"nets.{net_key}.edges.{edge_key} must contain two points")
                continue
            exact_mapping(edge[0], ["x", "y"], f"nets.{net_key}.edges.{edge_key}[0]")
            exact_mapping(edge[1], ["x", "y"], f"nets.{net_key}.edges.{edge_key}[1]")

    return {
        "extra_fields": extra_fields,
        "dangling_references": dangling_references,
        "schema_error_count": len(schema_errors),
        "examples": examples,
        "clean": not schema_errors and extra_fields == 0 and dangling_references == 0,
    }


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


def _audit_determinism(
    root: Path,
    determinism_root: Path,
    environment_path: Path | None,
    model_lock_sha256: str | None,
    production_config_sha256: str | None = None,
    *,
    require_formal_binding: bool = False,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    manifests: list[list[dict[str, str]]] = []
    for number in range(1, DETERMINISM_RUNS + 1):
        run_dir = determinism_root / f"run{number}"
        if run_dir.is_dir():
            manifest, tree_sha256 = _tree_manifest(run_dir)
        else:
            manifest, tree_sha256 = [], _json_sha256([])
        manifests.append(manifest)
        runs.append(
            {
                "run": f"run{number}",
                "path": _relative(root, run_dir),
                "present": run_dir.is_dir(),
                "json_files": len(manifest),
                "tree_sha256": tree_sha256,
                "files": manifest,
            }
        )

    nonempty = all(item["json_files"] > 0 for item in runs)
    identical_paths = nonempty and all(
        [entry["path"] for entry in manifest]
        == [entry["path"] for entry in manifests[0]]
        for manifest in manifests[1:]
    )
    identical_hashes = identical_paths and all(
        [entry["sha256"] for entry in manifest]
        == [entry["sha256"] for entry in manifests[0]]
        for manifest in manifests[1:]
    )

    environment = _path_evidence(root, environment_path)
    environment_valid = False
    environment_problems: list[str] = []
    environment_payload: Mapping[str, Any] = {}
    formal_summaries: list[dict[str, Any]] = []
    formal_summary_paths: set[str] = set()
    inference_run_ids: set[str] = set()
    environment_generated: datetime | None = None
    if require_formal_binding:
        if not isinstance(model_lock_sha256, str) or not HEX_SHA256.fullmatch(
            model_lock_sha256
        ):
            environment_problems.append("current models.lock SHA-256 is unavailable")
        if not isinstance(production_config_sha256, str) or not HEX_SHA256.fullmatch(
            production_config_sha256
        ):
            environment_problems.append("current production config SHA-256 is unavailable")
    if environment["present"] and environment_path is not None:
        try:
            payload = _read_json(environment_path)
        except AuditInputError as exc:
            environment_problems.append(str(exc))
        else:
            if not isinstance(payload, Mapping):
                environment_problems.append("environment evidence root is not an object")
            elif payload.get("kind") != "same-environment-determinism":
                environment_problems.append("environment evidence kind is not same-environment-determinism")
            else:
                environment_payload = payload
                if require_formal_binding:
                    formal_checks = {
                        "status": payload.get("status") == "ok",
                        "formal_release_evidence": payload.get("formal_release_evidence") is True,
                        "model_lock_sha256": payload.get("model_lock_sha256")
                        == model_lock_sha256,
                        "config_sha256": payload.get("config_sha256")
                        == production_config_sha256,
                        "evidence_scope": payload.get("evidence_scope")
                        == "current_lock_production_image_only_inference",
                    }
                    environment_problems.extend(
                        name for name, passed in formal_checks.items() if not passed
                    )
                    environment_generated, time_problem = _parse_utc_timestamp(
                        payload.get("generated_at_utc"),
                        field="determinism environment generated_at_utc",
                    )
                    if time_problem:
                        environment_problems.append(time_problem)
                declared_runs = payload.get("runs")
                if not isinstance(declared_runs, list) or len(declared_runs) != DETERMINISM_RUNS:
                    environment_problems.append("environment evidence must contain exactly three runs")
                else:
                    by_name = {
                        str(item.get("run")): item
                        for item in declared_runs
                        if isinstance(item, Mapping)
                    }
                    if set(by_name) != {"run1", "run2", "run3"}:
                        environment_problems.append("environment run IDs must be run1, run2, run3")
                    fingerprints: set[str] = set()
                    configs: set[str] = set()
                    for item in runs:
                        declared = by_name.get(item["run"], {})
                        fingerprint = declared.get("environment_fingerprint_sha256")
                        config = declared.get("config_sha256")
                        if not isinstance(fingerprint, str) or not HEX_SHA256.fullmatch(fingerprint):
                            environment_problems.append(
                                f"{item['run']} lacks a SHA-256 environment fingerprint"
                            )
                        else:
                            fingerprints.add(fingerprint)
                        if not isinstance(config, str) or not HEX_SHA256.fullmatch(config):
                            environment_problems.append(f"{item['run']} lacks a SHA-256 config hash")
                        else:
                            configs.add(config)
                            if (
                                production_config_sha256 is not None
                                and config != production_config_sha256
                            ):
                                environment_problems.append(
                                    f"{item['run']} is not bound to the production config"
                                )
                        if declared.get("output_tree_sha256") != item["tree_sha256"]:
                            environment_problems.append(
                                f"{item['run']} declared output tree hash does not match bytes"
                            )
                        if declared.get("model_lock_sha256") != model_lock_sha256:
                            environment_problems.append(
                                f"{item['run']} is not bound to the current models.lock"
                            )
                        if require_formal_binding:
                            run_id = declared.get("inference_run_id")
                            if not isinstance(run_id, str) or not HEX_SHA256.fullmatch(run_id):
                                environment_problems.append(
                                    f"{item['run']} lacks a SHA-256 inference_run_id"
                                )
                            else:
                                inference_run_ids.add(run_id)
                            completed, completed_problem = _parse_utc_timestamp(
                                declared.get("completed_at_utc"),
                                field=f"{item['run']} completed_at_utc",
                            )
                            if completed_problem:
                                environment_problems.append(completed_problem)
                            elif (
                                environment_generated is not None
                                and completed is not None
                                and completed > environment_generated
                            ):
                                environment_problems.append(
                                    f"{item['run']} completion is newer than determinism environment"
                                )
                            declared_files = declared.get("files")
                            if declared_files != item["files"]:
                                environment_problems.append(
                                    f"{item['run']} declared file manifest does not match bytes"
                                )
                            summary_path, summary_problem = _resolve_workspace_evidence_path(
                                root,
                                declared.get("source_summary_path"),
                                field=f"{item['run']} source_summary_path",
                            )
                            if summary_problem:
                                environment_problems.append(summary_problem)
                            elif summary_path is not None:
                                formal_summary_paths.add(str(summary_path).casefold())
                                summary = _audit_production_summary_binding(
                                    root,
                                    summary_path,
                                    model_lock_sha256,
                                    production_config_sha256,
                                    require_success=True,
                                    require_embedded_time=True,
                                )
                                formal_summaries.append(summary)
                                if not summary["valid"]:
                                    environment_problems.extend(
                                        f"{item['run']} summary: {problem}"
                                        for problem in summary["problems"]
                                    )
                                if declared.get("source_summary_sha256") != summary.get("sha256"):
                                    environment_problems.append(
                                        f"{item['run']} source summary hash does not match bytes"
                                    )
                                if run_id != summary.get("inference_run_id"):
                                    environment_problems.append(
                                        f"{item['run']} inference_run_id does not match its summary"
                                    )
                                if declared.get("completed_at_utc") != summary.get(
                                    "completed_at_utc"
                                ):
                                    environment_problems.append(
                                        f"{item['run']} completed_at_utc does not match its summary"
                                    )
                                if item["files"] != summary.get("official_output_manifest"):
                                    environment_problems.append(
                                        f"{item['run']} output manifest does not match bound production summary"
                                    )
                                summary_timestamp = summary.get("generated_timestamp")
                                if (
                                    completed is not None
                                    and isinstance(summary_timestamp, (int, float))
                                    and summary_timestamp > completed.timestamp()
                                ):
                                    environment_problems.append(
                                        f"{item['run']} summary is newer than its declared completion"
                                    )
                    if len(fingerprints) != 1:
                        environment_problems.append("runs do not share one environment fingerprint")
                    if len(configs) != 1:
                        environment_problems.append("runs do not share one config hash")
                    if require_formal_binding and len(inference_run_ids) != DETERMINISM_RUNS:
                        environment_problems.append(
                            "formal evidence does not identify three distinct inference runs"
                        )
                    if require_formal_binding and len(formal_summary_paths) != DETERMINISM_RUNS:
                        environment_problems.append(
                            "formal evidence does not bind three distinct production summaries"
                        )
                    environment_valid = not environment_problems

    evidence_scope = environment_payload.get("evidence_scope")
    current_lock_inference_proven = bool(
        environment_valid
        and evidence_scope == "current_lock_production_image_only_inference"
        and environment_payload.get("current_lock_inference_executed") is True
        and environment_payload.get("models_loaded") is True
    )
    serializer_replay_proven = bool(
        environment_valid
        and evidence_scope == "frozen_ir_current_serializer_replay"
        and environment_payload.get("current_lock_inference_executed") is False
        and environment_payload.get("models_loaded") is False
    )

    if require_formal_binding and not environment["present"]:
        environment_problems.append("formal determinism environment manifest is missing")

    if not nonempty and not environment["present"]:
        status = "unverified"
    elif not nonempty or not identical_paths or not identical_hashes:
        status = "fail"
    elif environment["present"] and not environment_valid:
        status = "fail"
    elif not environment["present"]:
        status = "unverified"
    elif current_lock_inference_proven:
        status = "pass"
    else:
        # A frozen-IR serializer replay is useful deterministic serializer
        # evidence, but it did not execute the artifacts named by the current
        # lock.  Keep the release-level inference determinism gate open.
        status = "unverified"
    return {
        "status": status,
        "required_runs": DETERMINISM_RUNS,
        "runs": runs,
        "nonempty_runs": nonempty,
        "identical_relative_paths": identical_paths,
        "identical_json_hashes": identical_hashes,
        "direct_hash_subcheck": "pass" if identical_hashes else "fail",
        "same_environment_proven": environment_valid,
        "evidence_scope": evidence_scope,
        "current_lock_inference_proven": current_lock_inference_proven,
        "serializer_replay_subcheck": "pass" if serializer_replay_proven else "unverified",
        "environment_evidence": environment,
        "environment_problems": environment_problems,
        "formal_summary_bindings": formal_summaries,
        "formal_binding_required": require_formal_binding,
        "production_config_sha256": production_config_sha256,
        "finding": (
            "Three output trees are byte-identical but same-environment provenance is absent."
            if identical_hashes and not environment["present"]
            else (
                "Three current-serializer frozen-IR replay trees are byte-identical in one "
                "environment, but current-lock production inference was not executed."
                if serializer_replay_proven
                else None
            )
        ),
    }


def _select_determinism_evidence(
    root: Path,
    base: Path,
    model_lock_sha256: str | None,
    production_config_sha256: str | None,
) -> tuple[Path | None, Path | None, dict[str, Any]]:
    candidates: list[Path] = []
    if base.is_dir() and any((base / f"run{number}").is_dir() for number in range(1, 4)):
        candidates.append(base.resolve())
    if base.is_dir():
        for child in sorted(base.iterdir(), key=lambda item: item.name.casefold()):
            if child.is_dir() and (
                (child / "environment.json").is_file()
                or (child / "environment_manifest.json").is_file()
                or any((child / f"run{number}").is_dir() for number in range(1, 4))
            ):
                candidates.append(child.resolve())

    inventory: list[dict[str, Any]] = []
    current_lineage: list[tuple[float, Path, Path, dict[str, Any], bool]] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).casefold()
        if key in seen:
            continue
        seen.add(key)
        environment_path = _first_existing(
            [candidate / "environment.json", candidate / "environment_manifest.json"]
        )
        audited = _audit_determinism(
            root,
            candidate,
            environment_path,
            model_lock_sha256,
            production_config_sha256,
            require_formal_binding=True,
        )
        payload: Mapping[str, Any] = {}
        generated_timestamp: float | None = None
        if environment_path is not None:
            try:
                raw = _read_json(environment_path)
            except AuditInputError:
                raw = {}
            if isinstance(raw, Mapping):
                payload = raw
                generated, _ = _parse_utc_timestamp(
                    raw.get("generated_at_utc"), field="generated_at_utc"
                )
                run_completion_times: list[datetime] = []
                declared_runs = raw.get("runs")
                if isinstance(declared_runs, list):
                    for index, declared_run in enumerate(declared_runs):
                        if not isinstance(declared_run, Mapping):
                            continue
                        completed, _ = _parse_utc_timestamp(
                            declared_run.get("completed_at_utc"),
                            field=f"runs[{index}].completed_at_utc",
                        )
                        if completed is not None:
                            run_completion_times.append(completed)
                generated_timestamp = (
                    max(item.timestamp() for item in run_completion_times)
                    if len(run_completion_times) == DETERMINISM_RUNS
                    else None
                )
        reasons = list(audited.get("environment_problems", []))
        if audited.get("status") != "pass" and not reasons:
            reasons.append(f"determinism gate status is {audited.get('status')}")
        record = {
            "path": _relative(root, candidate),
            "environment": _relative(root, environment_path),
            "environment_sha256": (
                _sha256_file(environment_path) if environment_path is not None else None
            ),
            "generated_at_utc": payload.get("generated_at_utc"),
            "latest_run_completed_timestamp": generated_timestamp,
            "evidence_scope": payload.get("evidence_scope"),
            "declared_model_lock_sha256": payload.get("model_lock_sha256"),
            "declared_config_sha256": payload.get("config_sha256"),
            "eligible": audited.get("status") == "pass",
            "rejection_reasons": reasons,
        }
        inventory.append(record)
        declares_current = bool(
            environment_path is not None
            and generated_timestamp is not None
            and payload.get("model_lock_sha256") == model_lock_sha256
            and payload.get("config_sha256") == production_config_sha256
            and payload.get("evidence_scope")
            == "current_lock_production_image_only_inference"
        )
        record["declares_current_lineage"] = declares_current
        if declares_current and environment_path is not None:
            current_lineage.append(
                (
                    generated_timestamp or 0.0,
                    candidate,
                    environment_path,
                    record,
                    audited.get("status") == "pass",
                )
            )

    current_lineage.sort(key=lambda item: (item[0], str(item[1]).casefold()), reverse=True)
    selected_root: Path | None = None
    selected_environment: Path | None = None
    ambiguity: list[str] = []
    if current_lineage:
        newest_time = current_lineage[0][0]
        newest = [item for item in current_lineage if item[0] == newest_time]
        if len(newest) == 1:
            _, candidate_root, candidate_environment, selected_record, passed = newest[0]
            selected_record["newest_current_candidate"] = True
            if passed:
                selected_root = candidate_root
                selected_environment = candidate_environment
                selected_record["selected"] = True
            else:
                selected_record["rejection_reasons"].append(
                    "newest current-lineage candidate failed formal audit; older evidence is not reused"
                )
        else:
            ambiguity = [str(item[1]) for item in newest]
            for item in newest:
                item[3]["rejection_reasons"].append(
                    "automatic selection is ambiguous at the newest timestamp"
                )
                item[3]["eligible"] = False

    return selected_root, selected_environment, {
        "mode": "automatic",
        "selected_path": _relative(root, selected_root),
        "selected_environment": _relative(root, selected_environment),
        "expected_model_lock_sha256": model_lock_sha256,
        "expected_production_config_sha256": production_config_sha256,
        "requirements": [
            "current models.lock SHA-256",
            "current production semantic config SHA-256",
            "formal successful production status",
            "embedded UTC generation time",
            "byte-bound prediction and output manifests",
            "three distinct inference_run_id and production-summary paths",
            "current-lock image-only inference scope",
        ],
        "ambiguous_newest_candidates": ambiguity,
        "candidates": inventory,
        "finding": (
            "No automatically eligible current-lock formal determinism evidence exists; "
            "use explicit paths only for diagnostic inspection."
            if selected_root is None
            else None
        ),
    }


def _audit_internal_guide_outputs(
    root: Path,
    determinism: Mapping[str, Any],
    local_schema_path: Path,
) -> dict[str, Any]:
    schema_evidence = _path_evidence(root, local_schema_path)
    if not schema_evidence["present"]:
        return {
            "status": "unverified",
            "profile": schema_evidence,
            "profile_authority": "local_only_not_official",
            "outputs_checked": 0,
            "extra_fields": None,
            "dangling_references": None,
            "schema_errors": None,
            "problems": ["local guide schema is missing"],
        }
    try:
        raw_schema = _read_json(local_schema_path)
        schema = raw_schema.get("official_schema", raw_schema)
        if not isinstance(schema, Mapping):
            raise AuditInputError("local guide schema root is not an object")
    except AuditInputError as exc:
        return {
            "status": "fail",
            "profile": schema_evidence,
            "profile_authority": "local_only_not_official",
            "outputs_checked": 0,
            "extra_fields": None,
            "dangling_references": None,
            "schema_errors": None,
            "problems": [str(exc)],
        }

    totals = Counter()
    outputs: list[dict[str, Any]] = []
    for run in determinism.get("runs", []):
        run_dir = root / str(run.get("path")) if run.get("path") else None
        if run_dir is None or not run_dir.is_dir():
            continue
        for item in run.get("files", []):
            relative = item.get("path")
            if not isinstance(relative, str):
                continue
            path = run_dir / relative
            try:
                result = analyze_guide_output(_read_json(path), schema)
            except AuditInputError as exc:
                result = {
                    "extra_fields": 0,
                    "dangling_references": 0,
                    "schema_error_count": 1,
                    "examples": [str(exc)],
                    "clean": False,
                }
            totals.update(
                {
                    "extra_fields": int(result["extra_fields"]),
                    "dangling_references": int(result["dangling_references"]),
                    "schema_errors": int(result["schema_error_count"]),
                }
            )
            outputs.append(
                {
                    "path": _relative(root, path),
                    "sha256": item.get("sha256"),
                    **result,
                }
            )
    clean = bool(outputs) and not any(totals.values())
    return {
        "status": "pass" if clean else ("fail" if outputs else "unverified"),
        "profile": schema_evidence,
        "profile_version": schema.get("version"),
        "profile_component_fields": schema.get("component", {}).get("fields"),
        "profile_authority": "local_only_not_official",
        "outputs_checked": len(outputs),
        "extra_fields": totals["extra_fields"],
        "dangling_references": totals["dangling_references"],
        "schema_errors": totals["schema_errors"],
        "outputs": outputs,
        "problems": [],
    }


def _audit_official_contract(
    root: Path,
    guide_path: Path | None,
    qa_path: Path | None,
    sample_path: Path | None,
    evaluator_path: Path | None,
    evaluator_report_path: Path | None,
    run1_tree_sha256: str | None,
    run1_json_files: int,
    internal_guide: Mapping[str, Any],
) -> dict[str, Any]:
    guide = _path_evidence(root, guide_path)
    guide["expected_sha256"] = OFFICIAL_GUIDE_SHA256
    guide["hash_verified"] = guide.get("sha256") == OFFICIAL_GUIDE_SHA256
    qa = _path_evidence(root, qa_path)
    sample = _path_evidence(root, sample_path)
    evaluator = _path_evidence(root, evaluator_path)
    report = _path_evidence(root, evaluator_report_path)
    report_valid = False
    report_problems: list[str] = []
    report_payload: Mapping[str, Any] = {}
    if report["present"] and evaluator_report_path is not None:
        try:
            loaded = _read_json(evaluator_report_path)
        except AuditInputError as exc:
            report_problems.append(str(exc))
        else:
            if not isinstance(loaded, Mapping):
                report_problems.append("official evaluator report root is not an object")
            else:
                report_payload = loaded
                checks = {
                    "kind": loaded.get("kind") == "official-evaluator-release-report",
                    "official_sample_sha256": loaded.get("official_sample_sha256") == sample.get("sha256"),
                    "official_evaluator_sha256": loaded.get("official_evaluator_sha256") == evaluator.get("sha256"),
                    "evaluated_output_tree_sha256": loaded.get("evaluated_output_tree_sha256") == run1_tree_sha256,
                    "exit_code": loaded.get("exit_code") == 0,
                    "cases_evaluated": type(loaded.get("cases_evaluated")) is int
                    and loaded.get("cases_evaluated") == run1_json_files
                    and run1_json_files > 0,
                    "extra_fields": loaded.get("extra_fields") == 0,
                    "dangling_references": loaded.get("dangling_references") == 0,
                }
                report_problems.extend(name for name, passed in checks.items() if not passed)
                report_valid = not report_problems and sample["present"] and evaluator["present"]

    guide_verified = guide["present"] and guide["hash_verified"]
    guide_name_case_conflict = bool(
        guide_verified
        and internal_guide.get("profile_component_fields")
        == ["type", "name", "value", "bbox"]
    )
    internal_shape_failed = bool(
        internal_guide.get("status") == "fail"
        and (
            (internal_guide.get("extra_fields") or 0) > 0
            or (internal_guide.get("schema_errors") or 0) > 0
        )
    )
    internal_reference_failed = bool(
        (internal_guide.get("dangling_references") or 0) > 0
    )
    official_shape_negative = report["present"] and (
        report_payload.get("extra_fields", 0) != 0
        or report_payload.get("exit_code", 0) != 0
    )
    official_reference_negative = report["present"] and (
        report_payload.get("dangling_references", 0) != 0
        or report_payload.get("exit_code", 0) != 0
    )
    if internal_shape_failed or official_shape_negative or guide_name_case_conflict:
        shape_status = "fail"
    elif report_valid:
        shape_status = "pass"
    else:
        shape_status = "unverified"
    if internal_reference_failed or official_reference_negative:
        reference_status = "fail"
    elif report_valid:
        reference_status = "pass"
    else:
        reference_status = "unverified"
    return {
        "official_guide": guide,
        "official_qa": qa,
        "official_sample": sample,
        "official_evaluator": evaluator,
        "official_evaluator_report": report,
        "official_evaluator_report_valid": report_valid,
        "report_problems": report_problems,
        "official_guide_authority_available": guide_verified,
        "official_machine_contract_available": sample["present"] and evaluator["present"],
        "guide_contract": {
            "top_level_keys": ["components", "pins", "nets"],
            "component_json_example_fields": ["type", "Name", "value", "bbox"],
            "component_field_table_fields": ["type", "name", "value", "bbox"],
            "pin_fields": ["pinname", "point"],
            "net_fields": ["hyperGraph", "edges"],
            "edge_point_count": 2,
            "coordinate_origin": "bottom_left",
            "x_axis": "right",
            "y_axis": "up",
            "submission_layout": "submission/case_001/case_001.json",
            "case_count_range": [30, 50],
            "deadline_seconds_lt": PERFORMANCE_DEADLINE_SECONDS,
            "document_conflicts": [
                "Pages 13, 15, and 16 JSON examples use `Name`, while pages 14-16 field tables use `name`.",
            ],
            "resolved_local_compatibility": [
                "The local strict validator accepts finite integer or floating-point coordinates.",
                "The local strict validator accepts page-20 identical endpoint pairs.",
                "The current default profile explicitly selects the JSON-example `Name` spelling.",
            ],
            "resolution_authority_missing": not qa["present"] and not evaluator["present"],
        },
        "official_json_zero_extra_fields": {
            "status": shape_status,
            "observed_official_extra_fields": report_payload.get("extra_fields"),
            "internal_guide_subcheck": internal_guide.get("status"),
            "guide_name_case_conflict": guide_name_case_conflict,
            "boundary": (
                "The local profile selects table spelling `name`, which conflicts with guide JSON "
                "examples using `Name`; without Q&A/evaluator, zero extra fields cannot be proven."
                if guide_name_case_conflict
                else (
                    "The current profile selects guide-example `Name` and the strict local replay has "
                    "zero extra fields, but the guide's `Name`/`name` conflict remains unresolved without "
                    "Q&A or the official evaluator."
                )
            ),
        },
        "official_json_zero_dangling_references": {
            "status": reference_status,
            "observed_official_dangling_references": report_payload.get("dangling_references"),
            "internal_guide_subcheck": internal_guide.get("status"),
            "boundary": (
                "The official guide defines hyperGraph structure, but local guide-v1 validation cannot "
                "substitute for the promised official evaluator."
            ),
        },
    }


def _audit_model_lock(
    root: Path,
    lock_path: Path,
    production_config_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    evidence = _path_evidence(root, lock_path)
    problems: list[str] = []
    entries: list[dict[str, Any]] = []
    learned_junctions_enabled = (
        production_config_binding.get("learned_junctions_enabled")
        if isinstance(production_config_binding, Mapping)
        else None
    )
    production_loader = root / "src" / "pcbparse" / "perception" / "production_provider.py"
    composite_provider = root / "src" / "pcbparse" / "perception" / "composite_provider.py"
    readiness_source = root / "src" / "pcbparse" / "service" / "readiness.py"
    loader_text = production_loader.read_text(encoding="utf-8") if production_loader.is_file() else ""
    composite_text = (
        composite_provider.read_text(encoding="utf-8") if composite_provider.is_file() else ""
    )
    readiness_text = readiness_source.read_text(encoding="utf-8") if readiness_source.is_file() else ""
    junction_loaded_unconditionally = 'junction_provider=loaded["junction"]' in loader_text
    config_aware_loader_present = all(
        token in loader_text
        for token in (
            "required_tasks = required_model_tasks_for_config(resolved_config)",
            "required_tasks=required_tasks",
            'junction_provider=loaded.get("junction")',
            "if task in required_tasks",
        )
    ) and not junction_loaded_unconditionally
    junction_call_guard_present = all(
        token in composite_text
        for token in (
            "self.config.learned_junctions_enabled",
            "self.junction_provider.classify",
            '"quarantined_model_versions"',
        )
    )
    readiness_default_requires_junction = all(
        token in readiness_text
        for token in (
            'REQUIRED_MODEL_TASKS = frozenset({"component", "junction", "ocr", "pin", "wire"})',
            "REQUIRED_LEARNED_TASKS = REQUIRED_MODEL_TASKS",
        )
    )
    readiness_supports_configured_task_plan = all(
        token in readiness_text
        for token in (
            "required_tasks: Iterable[str] = REQUIRED_MODEL_TASKS",
            "required = frozenset",
            '"unexpected_task"',
        )
    )
    runtime_task_plan_metadata_present = all(
        token in composite_text
        for token in (
            '"contract": "production_model_task_plan_v1"',
            '"learned_junction_model_loaded"',
            '"learned_junction_model_invoked"',
            '"model_task_plan": self.model_task_plan',
        )
    )
    inactive_junction_static_eligible = bool(
        learned_junctions_enabled is False
        and config_aware_loader_present
        and readiness_supports_configured_task_plan
        and runtime_task_plan_metadata_present
    )
    inactive_junction_exemption_proven = False
    exemption_applied = inactive_junction_static_eligible
    expected_tasks = set(
        DETERMINISTIC_JUNCTION_MODEL_TASKS
        if exemption_applied
        else REQUIRED_MODEL_TASKS
    )
    runtime_evidence_problem = (
        "deterministic Junction exemption lacks current formal runtime non-load/non-invocation evidence"
    )
    junction_policy = {
        "learned_junctions_enabled": learned_junctions_enabled,
        "junction_output_path_disabled_by_config": learned_junctions_enabled is False,
        "composite_call_guard_and_quarantine_metadata_present": junction_call_guard_present,
        "production_loader_loads_junction_unconditionally": junction_loaded_unconditionally,
        "production_loader_uses_configured_task_plan": config_aware_loader_present,
        "readiness_requires_junction_by_default": readiness_default_requires_junction,
        "readiness_supports_exact_configured_task_plan": readiness_supports_configured_task_plan,
        "runtime_task_plan_metadata_present": runtime_task_plan_metadata_present,
        "inactive_dependency_exemption_static_eligible": inactive_junction_static_eligible,
        "hash_bound_runtime_non_load_non_invocation_evidence": False,
        "inactive_dependency_exemption_proven": inactive_junction_exemption_proven,
        "exemption_applied": exemption_applied,
        "decision": (
            "Deterministic Junction mode uses an exact four-model lock; release remains blocked "
            "until a current formal production run proves Junction was neither loaded nor invoked."
            if exemption_applied
            else (
                "Junction remains required because the disabled-mode loader/readiness/runtime "
                "non-use contract is not statically proven."
                if learned_junctions_enabled is False
                else "Junction is an active required learned task."
            )
        ),
    }
    required_tasks_list = sorted(expected_tasks)
    inactive_exemptions = ["junction"] if exemption_applied else []
    if not evidence["present"]:
        return {
            "status": "fail",
            "lock": evidence,
            "required_tasks": required_tasks_list,
            "inactive_task_exemptions": inactive_exemptions,
            "junction_dependency_policy": junction_policy,
            "entries": entries,
            "problems": ["models.lock is missing"],
        }
    try:
        payload = _read_json(lock_path)
    except AuditInputError as exc:
        return {
            "status": "fail",
            "lock": evidence,
            "required_tasks": required_tasks_list,
            "inactive_task_exemptions": inactive_exemptions,
            "junction_dependency_policy": junction_policy,
            "entries": entries,
            "problems": [str(exc)],
        }
    if not isinstance(payload, Mapping) or not isinstance(payload.get("models"), list):
        return {
            "status": "fail",
            "lock": evidence,
            "required_tasks": required_tasks_list,
            "inactive_task_exemptions": inactive_exemptions,
            "junction_dependency_policy": junction_policy,
            "entries": entries,
            "problems": ["models.lock must contain a models list"],
        }

    task_counts = Counter()
    for index, raw in enumerate(payload["models"]):
        if not isinstance(raw, Mapping):
            problems.append(f"entry {index} is not an object")
            continue
        task = str(raw.get("task", "")).strip().lower()
        task_counts[task] += 1
        relative = raw.get("path")
        model_path = None
        if isinstance(relative, str) and relative.strip():
            candidate = Path(relative.strip()).expanduser()
            model_path = (
                candidate if candidate.is_absolute() else lock_path.parent / candidate
            ).resolve(strict=False)
        within_workspace = False
        within_models_root = False
        if model_path is not None:
            try:
                model_path.relative_to(root)
                within_workspace = True
            except ValueError:
                problems.append(f"task {task!r} artifact path escapes workspace")
            try:
                model_path.relative_to(lock_path.parent.resolve())
                within_models_root = True
            except ValueError:
                problems.append(f"task {task!r} artifact path escapes models root")
        present = (
            model_path is not None
            and within_workspace
            and within_models_root
            and model_path.is_file()
        )
        expected = raw.get("sha256")
        actual = _sha256_file(model_path) if present and model_path is not None else None
        hash_matches = (
            isinstance(expected, str)
            and bool(HEX_SHA256.fullmatch(expected.lower()))
            and actual == expected.lower()
        )
        blessed = raw.get("status") == "blessed" and raw.get("blessed") is True
        if not task:
            problems.append(f"entry {index} lacks a task")
        if not present:
            problems.append(f"task {task!r} artifact is missing")
        if not hash_matches:
            problems.append(f"task {task!r} artifact hash is not locked to its bytes")
        if not blessed:
            problems.append(f"task {task!r} is not blessed")
        entries.append(
            {
                "task": task,
                "path": _relative(root, model_path),
                "present": present,
                "within_workspace": within_workspace,
                "within_models_root": within_models_root,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "hash_matches": hash_matches,
                "status": raw.get("status"),
                "blessed": raw.get("blessed"),
            }
        )
    actual_tasks = set(task_counts)
    if actual_tasks != expected_tasks:
        problems.append(
            f"model task set differs; missing={sorted(expected_tasks - actual_tasks)!r}, "
            f"extra={sorted(actual_tasks - expected_tasks)!r}"
        )
    duplicates = sorted(task for task, count in task_counts.items() if count != 1)
    if duplicates:
        problems.append(f"model tasks do not occur exactly once: {duplicates!r}")
    if exemption_applied:
        problems.append(runtime_evidence_problem)
    return {
        "status": "pass" if not problems else "fail",
        "lock": evidence,
        "required_tasks": required_tasks_list,
        "inactive_task_exemptions": inactive_exemptions,
        "junction_dependency_policy": junction_policy,
        "artifact_hashes_all_match": bool(entries) and all(item["hash_matches"] for item in entries),
        "all_entries_blessed": bool(entries)
        and all(item["status"] == "blessed" and item["blessed"] is True for item in entries),
        "entries": entries,
        "problems": problems,
    }


def _discover_production_summaries(root: Path, explicit: Sequence[Path]) -> list[Path]:
    candidates = list(explicit)
    work = root / "work"
    if not explicit and work.is_dir():
        candidates.extend(work.glob("production-e2e-*/summary.json"))
    unique: dict[str, Path] = {}
    for path in candidates:
        if explicit or path.is_file():
            unique[str(path.resolve()).casefold()] = path.resolve()
    return sorted(unique.values(), key=lambda item: _relative(root, item) or str(item))


def _audit_performance(
    root: Path,
    summaries: Sequence[Path],
    model_lock_sha256: str | None,
    production_config_sha256: str | None = None,
    *,
    require_formal_binding: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for path in summaries:
        if require_formal_binding:
            record = _audit_production_summary_binding(
                root,
                path,
                model_lock_sha256,
                production_config_sha256,
                require_success=True,
                require_embedded_time=True,
            )
            records.append(record)
            continue
        try:
            payload = _read_json(path)
        except AuditInputError as exc:
            records.append({"path": _relative(root, path), "valid": False, "problem": str(exc)})
            continue
        if not isinstance(payload, Mapping) or payload.get("kind") != "production-image-only-inference":
            continue
        records.append(
            {
                "path": _relative(root, path),
                "sha256": _sha256_file(path),
                "valid": True,
                "status": payload.get("status"),
                "all_successful": payload.get("all_successful"),
                "selected_cases": payload.get("selected_cases"),
                "succeeded": payload.get("succeeded"),
                "failed": payload.get("failed"),
                "elapsed_seconds": payload.get("elapsed_seconds"),
                "model_lock_sha256": payload.get("model_lock_sha256"),
                "prediction_set_sha256": payload.get("prediction_set_sha256"),
            }
        )
    declared_current_50 = [
        item
        for item in records
        if item.get("selected_cases") == PERFORMANCE_CASES
        and item.get("model_lock_sha256") == model_lock_sha256
        and (
            not require_formal_binding
            or item.get("config_hash") == production_config_sha256
        )
    ]
    newest_candidates = declared_current_50
    newest_ambiguity: list[str] = []
    missing_embedded_times = 0
    if require_formal_binding and declared_current_50:
        missing_embedded_times = sum(
            not isinstance(item.get("generated_timestamp"), (int, float))
            for item in declared_current_50
        )
        if not missing_embedded_times:
            newest_time = max(float(item["generated_timestamp"]) for item in declared_current_50)
            newest_candidates = [
                item
                for item in declared_current_50
                if float(item["generated_timestamp"]) == newest_time
            ]
            if len(newest_candidates) != 1:
                newest_ambiguity = [str(item.get("path")) for item in newest_candidates]
        else:
            newest_candidates = []
    exact_50 = [item for item in newest_candidates if item.get("valid")]
    passing = [
        item
        for item in exact_50
        if item.get("status") == "ok"
        and item.get("all_successful") is True
        and item.get("succeeded") == PERFORMANCE_CASES
        and isinstance(item.get("elapsed_seconds"), (int, float))
        and not isinstance(item.get("elapsed_seconds"), bool)
        and math.isfinite(float(item["elapsed_seconds"]))
        and float(item["elapsed_seconds"]) >= 0.0
        and float(item["elapsed_seconds"]) <= PERFORMANCE_PASS_SECONDS
    ]
    if require_formal_binding and (missing_embedded_times or newest_ambiguity):
        status = "fail"
    elif passing:
        status = "pass"
    elif declared_current_50:
        status = "fail"
    else:
        status = "unverified"
    return (
        {
            "status": status,
            "required_cases": PERFORMANCE_CASES,
            "deadline_seconds": PERFORMANCE_DEADLINE_SECONDS,
            "required_reserve_fraction": PERFORMANCE_RESERVE_FRACTION,
            "pass_elapsed_seconds_lte": PERFORMANCE_PASS_SECONDS,
            "exact_current_lock_runs": len(exact_50),
            "declared_current_lock_config_50_case_runs": len(declared_current_50),
            "passing_runs": len(passing),
            "selected_newest_run": newest_candidates[0]
            if len(newest_candidates) == 1
            else None,
            "best_passing_run": passing[0] if passing else None,
            "newest_timestamp_ambiguity": newest_ambiguity,
            "current_runs_missing_embedded_time": missing_embedded_times,
            "finding": (
                "No exact 50-case current-lock run exists; one-case timing is not extrapolated."
                if not declared_current_50
                else None
            ),
            "formal_binding_required": require_formal_binding,
            "production_config_sha256": production_config_sha256,
        },
        records,
    )


def _bind_inactive_junction_runtime_evidence(
    model_gate: dict[str, Any],
    performance_gate: Mapping[str, Any],
) -> None:
    """Close the deterministic-Junction exemption only with a formal run."""

    policy = model_gate.get("junction_dependency_policy")
    if not isinstance(policy, dict) or policy.get("exemption_applied") is not True:
        return
    selected = performance_gate.get("selected_newest_run")
    runtime_proven = bool(
        performance_gate.get("status") == "pass"
        and isinstance(selected, Mapping)
        and selected.get("formal") is True
        and selected.get("junction_runtime_non_load_non_invocation") is True
        and selected.get("model_task_plan")
        == {
            "contract": "production_model_task_plan_v1",
            "junction_mode": "deterministic_vectorizer",
            "required_model_tasks": list(DETERMINISTIC_JUNCTION_MODEL_TASKS),
            "loaded_model_tasks": list(DETERMINISTIC_JUNCTION_MODEL_TASKS),
            "learned_junction_model_required": False,
            "learned_junction_model_loaded": False,
            "learned_junction_model_invoked": False,
        }
    )
    policy["hash_bound_runtime_non_load_non_invocation_evidence"] = runtime_proven
    policy["inactive_dependency_exemption_proven"] = runtime_proven
    policy["runtime_evidence_summary"] = (
        {
            "path": selected.get("path"),
            "sha256": selected.get("sha256"),
            "inference_run_id": selected.get("inference_run_id"),
        }
        if runtime_proven and isinstance(selected, Mapping)
        else None
    )
    marker = (
        "deterministic Junction exemption lacks current formal runtime non-load/non-invocation evidence"
    )
    problems = model_gate.get("problems")
    if isinstance(problems, list) and runtime_proven:
        model_gate["problems"] = [problem for problem in problems if problem != marker]
    policy["decision"] = (
        "Current formal production evidence proves the exact four-model deterministic Junction "
        "plan; no learned Junction artifact was loaded or invoked."
        if runtime_proven
        else "The exact four-model plan is statically configured, but current formal runtime "
        "non-load/non-invocation evidence is missing or invalid."
    )
    model_gate["status"] = "pass" if not model_gate.get("problems") else "fail"


def _audit_coordinate_report(
    root: Path,
    report_path: Path | None,
    model_lock_sha256: str | None,
    *,
    official_guide_verified: bool,
    production_config_sha256: str | None = None,
    require_formal_binding: bool = False,
) -> dict[str, Any]:
    evidence = _path_evidence(root, report_path)
    serializer_path = root / "src" / "pcbparse" / "output" / "official_serializer.py"
    derive_path = root / "src" / "pcbparse" / "data" / "derive.py"
    serializer_evidence = _path_evidence(root, serializer_path)
    derive_evidence = _path_evidence(root, derive_path)
    coordinate_auditor_path = (
        root / "scripts" / "generate_current_coordinate_evidence.py"
    )
    coordinate_auditor_evidence = _path_evidence(root, coordinate_auditor_path)
    serializer_source = (
        serializer_path.read_text(encoding="utf-8") if serializer_path.is_file() else ""
    )
    derive_source = derive_path.read_text(encoding="utf-8") if derive_path.is_file() else ""
    internal_y_axis_down = '"image_y_axis": "down"' in derive_source
    serializer_emits_y_directly = '"y": round_half_away_from_zero(point.y)' in serializer_source
    serializer_converts_to_bottom_left = "image_height - point.y" in serializer_source
    static_origin_mismatch = bool(
        official_guide_verified
        and internal_y_axis_down
        and not serializer_converts_to_bottom_left
    )
    problems: list[str] = []
    release_binding_problems: list[str] = []
    source_summary: dict[str, Any] | None = None
    payload: Mapping[str, Any] = {}
    if static_origin_mismatch:
        problems.append(
            "internal image coordinates use y-down and the serializer emits y directly, "
            "but the official guide requires bottom-left/y-up coordinates"
        )
    if require_formal_binding:
        if not isinstance(model_lock_sha256, str) or not HEX_SHA256.fullmatch(
            model_lock_sha256
        ):
            release_binding_problems.append("current models.lock SHA-256 is unavailable")
        if not isinstance(production_config_sha256, str) or not HEX_SHA256.fullmatch(
            production_config_sha256
        ):
            release_binding_problems.append("current production config SHA-256 is unavailable")
    if evidence["present"] and report_path is not None:
        try:
            loaded = _read_json(report_path)
        except AuditInputError as exc:
            problems.append(str(exc))
        else:
            if not isinstance(loaded, Mapping):
                problems.append("coordinate report root is not an object")
            else:
                payload = loaded
                if loaded.get("kind") != "coordinate-roundtrip-release-report":
                    problems.append("coordinate report kind is invalid")
                if loaded.get("model_lock_sha256") != model_lock_sha256:
                    problems.append("coordinate report is not bound to current models.lock")
                if loaded.get("strict_image_only") is not True:
                    problems.append("coordinate report is not a strict image-only run")
                if loaded.get("official_coordinate_origin") != "bottom_left":
                    problems.append("coordinate report does not certify bottom-left origin")
                if loaded.get("official_y_axis") != "up":
                    problems.append("coordinate report does not certify upward y-axis")
                if loaded.get("origin_conversion_verified") is not True:
                    problems.append("coordinate report does not verify official origin conversion")
                for field in ("config_sha256", "source_prediction_set_sha256"):
                    value = loaded.get(field)
                    if not isinstance(value, str) or not HEX_SHA256.fullmatch(value):
                        problems.append(f"coordinate report lacks {field}")
                for field in ("cases_checked", "points_checked"):
                    value = loaded.get(field)
                    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                        problems.append(f"coordinate report {field} must be positive")
                maximum = loaded.get("max_roundtrip_error_px")
                if (
                    not isinstance(maximum, (int, float))
                    or isinstance(maximum, bool)
                    or not math.isfinite(float(maximum))
                ):
                    problems.append("coordinate report max error must be finite")
                elif float(maximum) < 0.0:
                    problems.append("coordinate report max error cannot be negative")
                elif float(maximum) > COORDINATE_ROUNDTRIP_LIMIT_PX:
                    problems.append("coordinate roundtrip exceeds 0.5 px")
                forward_maximum = loaded.get("max_source_to_serialized_error_px")
                if (
                    not isinstance(forward_maximum, (int, float))
                    or isinstance(forward_maximum, bool)
                    or not math.isfinite(float(forward_maximum))
                ):
                    problems.append(
                        "coordinate report forward serialization error must be finite"
                    )
                elif float(forward_maximum) < 0.0:
                    problems.append(
                        "coordinate report forward serialization error cannot be negative"
                    )
                elif float(forward_maximum) > COORDINATE_ROUNDTRIP_LIMIT_PX:
                    problems.append(
                        "coordinate forward serialization error exceeds 0.5 px"
                    )
                evidence_scope = loaded.get("evidence_scope")
                if evidence_scope not in {
                    "current_lock_production_image_only_inference",
                    "frozen_ir_current_serializer_replay",
                }:
                    problems.append("coordinate report evidence_scope is invalid")
                elif evidence_scope == "current_lock_production_image_only_inference":
                    if loaded.get("current_lock_inference_executed") is not True:
                        problems.append(
                            "coordinate report did not execute current-lock inference"
                        )
                    if loaded.get("models_loaded") is not True:
                        problems.append("coordinate report did not load current-lock models")
                    if require_formal_binding:
                        if (
                            not serializer_evidence.get("present")
                            or not isinstance(serializer_evidence.get("sha256"), str)
                            or not HEX_SHA256.fullmatch(
                                str(serializer_evidence.get("sha256"))
                            )
                        ):
                            release_binding_problems.append(
                                "current serializer source is missing or unhashable"
                            )
                        elif loaded.get("serializer_sha256") != serializer_evidence.get("sha256"):
                            release_binding_problems.append(
                                "coordinate report is not bound to current serializer bytes"
                            )
                        if (
                            not derive_evidence.get("present")
                            or not isinstance(derive_evidence.get("sha256"), str)
                            or not HEX_SHA256.fullmatch(str(derive_evidence.get("sha256")))
                        ):
                            release_binding_problems.append(
                                "current coordinate-lineage source is missing or unhashable"
                            )
                        elif loaded.get("coordinate_lineage_sha256") != derive_evidence.get("sha256"):
                            release_binding_problems.append(
                                "coordinate report is not bound to current coordinate-lineage bytes"
                            )
                        if (
                            not coordinate_auditor_evidence.get("present")
                            or loaded.get("coordinate_auditor_sha256")
                            != coordinate_auditor_evidence.get("sha256")
                        ):
                            release_binding_problems.append(
                                "coordinate report is not bound to the current coordinate auditor"
                            )
                        if loaded.get("config_sha256") != production_config_sha256:
                            release_binding_problems.append(
                                "coordinate report is not bound to the production config"
                            )
                        if loaded.get("status") != "ok":
                            release_binding_problems.append(
                                "coordinate report formal status is not ok"
                            )
                        if loaded.get("formal_release_evidence") is not True:
                            release_binding_problems.append(
                                "coordinate report is not marked formal release evidence"
                            )
                        release_suite_checks = {
                            "release_suite_contract": loaded.get(
                                "release_suite_contract"
                            )
                            == RELEASE_SUITE_CONTRACT,
                            "release_suite_split": loaded.get(
                                "release_suite_split"
                            )
                            == CANONICAL_RELEASE_SPLIT,
                            "release_suite_cases": loaded.get(
                                "release_suite_cases"
                            )
                            == PERFORMANCE_CASES,
                            "release_suite_case_coverage": loaded.get(
                                "cases_checked"
                            )
                            == PERFORMANCE_CASES,
                            "canonical_release_manifest_path": loaded.get(
                                "release_manifest_path"
                            )
                            == CANONICAL_RELEASE_MANIFEST,
                        }
                        release_binding_problems.extend(
                            name
                            for name, passed in release_suite_checks.items()
                            if not passed
                        )
                        module_counts = loaded.get(
                            "local_to_global_module_objects_checked"
                        )
                        expected_modules = {
                            "component",
                            "text",
                            "pin",
                            "segment",
                            "junction",
                        }
                        if (
                            not isinstance(module_counts, Mapping)
                            or set(module_counts) != expected_modules
                            or any(
                                type(module_counts.get(module)) is not int
                                or module_counts.get(module, 0) <= 0
                                for module in expected_modules
                            )
                        ):
                            release_binding_problems.append(
                                "coordinate report lacks positive local-to-global coverage for every module"
                            )
                        source_run_binding = loaded.get("source_run_binding")
                        if not isinstance(source_run_binding, Mapping):
                            release_binding_problems.append(
                                "coordinate report source_run_binding is missing"
                            )
                        else:
                            source_run_checks = {
                                "source_run_binding_contract": source_run_binding.get(
                                    "contract"
                                )
                                == RELEASE_SUITE_CONTRACT,
                                "source_run_binding_manifest": source_run_binding.get(
                                    "release_manifest_path"
                                )
                                == CANONICAL_RELEASE_MANIFEST,
                                "source_run_binding_split": source_run_binding.get(
                                    "split"
                                )
                                == CANONICAL_RELEASE_SPLIT,
                                "source_run_binding_cases": source_run_binding.get(
                                    "cases"
                                )
                                == PERFORMANCE_CASES,
                                "source_run_binding_hash": loaded.get(
                                    "source_run_binding_sha256"
                                )
                                == _compact_json_sha256(source_run_binding),
                                "source_run_binding_summary": source_run_binding.get(
                                    "source_summary_sha256"
                                )
                                == loaded.get("source_summary_sha256"),
                                "source_run_binding_run_id": source_run_binding.get(
                                    "source_inference_run_id"
                                )
                                == loaded.get("source_inference_run_id"),
                                "source_run_binding_prediction_set": source_run_binding.get(
                                    "source_prediction_set_sha256"
                                )
                                == loaded.get("source_prediction_set_sha256"),
                            }
                            release_binding_problems.extend(
                                name
                                for name, passed in source_run_checks.items()
                                if not passed
                            )
                        report_generated, time_problem = _parse_utc_timestamp(
                            loaded.get("generated_at_utc"),
                            field="coordinate report generated_at_utc",
                        )
                        if time_problem:
                            release_binding_problems.append(time_problem)
                        summary_path, summary_problem = _resolve_workspace_evidence_path(
                            root,
                            loaded.get("source_summary_path"),
                            field="coordinate report source_summary_path",
                        )
                        if summary_problem:
                            release_binding_problems.append(summary_problem)
                        elif summary_path is not None:
                            source_summary = _audit_production_summary_binding(
                                root,
                                summary_path,
                                model_lock_sha256,
                                production_config_sha256,
                                require_success=True,
                                require_embedded_time=True,
                            )
                            if not source_summary["valid"]:
                                release_binding_problems.extend(
                                    f"source summary: {problem}"
                                    for problem in source_summary["problems"]
                                )
                            if loaded.get("source_summary_sha256") != source_summary.get(
                                "sha256"
                            ):
                                release_binding_problems.append(
                                    "coordinate report source summary hash does not match bytes"
                                )
                            if loaded.get("source_prediction_manifest_sha256") != source_summary.get(
                                "prediction_manifest_sha256"
                            ):
                                release_binding_problems.append(
                                    "coordinate report prediction manifest hash does not match source"
                                )
                            if loaded.get("source_prediction_set_sha256") != source_summary.get(
                                "prediction_set_sha256"
                            ):
                                release_binding_problems.append(
                                    "coordinate report prediction-set hash does not match source"
                                )
                            if loaded.get("source_inference_run_id") != source_summary.get(
                                "inference_run_id"
                            ):
                                release_binding_problems.append(
                                    "coordinate report inference_run_id does not match source"
                                )
                            if loaded.get("source_completed_at_utc") != source_summary.get(
                                "completed_at_utc"
                            ):
                                release_binding_problems.append(
                                    "coordinate report completion time does not match source"
                                )
                            if loaded.get(
                                "source_inference_runtime_manifest"
                            ) != source_summary.get("runtime_source_manifest"):
                                release_binding_problems.append(
                                    "coordinate report inference runtime manifest does not match source"
                                )
                            if loaded.get(
                                "source_inference_runtime_manifest_sha256"
                            ) != source_summary.get("runtime_source_manifest_sha256"):
                                release_binding_problems.append(
                                    "coordinate report inference runtime manifest hash does not match source"
                                )
                            summary_timestamp = source_summary.get("generated_timestamp")
                            if (
                                report_generated is not None
                                and isinstance(summary_timestamp, (int, float))
                                and summary_timestamp > report_generated.timestamp()
                            ):
                                release_binding_problems.append(
                                    "coordinate report predates its source summary"
                                )
                            if loaded.get("cases_checked") != source_summary.get(
                                "selected_cases"
                            ):
                                release_binding_problems.append(
                                    "coordinate report does not cover every source case"
                                )

                            required_recompute_paths = {
                                "release_manifest": loaded.get(
                                    "release_manifest_path"
                                ),
                                "pipeline_config": loaded.get(
                                    "pipeline_config_path"
                                ),
                                "profile": loaded.get("profile_path"),
                            }
                            resolved_recompute_paths: dict[str, Path] = {}
                            for field, raw_path in required_recompute_paths.items():
                                resolved_path, resolved_problem = (
                                    _resolve_workspace_evidence_path(
                                        root,
                                        raw_path,
                                        field=f"coordinate report {field}",
                                    )
                                )
                                if resolved_problem:
                                    release_binding_problems.append(
                                        resolved_problem
                                    )
                                elif resolved_path is not None:
                                    resolved_recompute_paths[field] = resolved_path
                            if len(resolved_recompute_paths) == len(
                                required_recompute_paths
                            ):
                                try:
                                    spec = importlib.util.spec_from_file_location(
                                        "_current_coordinate_evidence_recompute",
                                        coordinate_auditor_path,
                                    )
                                    if spec is None or spec.loader is None:
                                        raise RuntimeError(
                                            "cannot load coordinate evidence generator"
                                        )
                                    coordinate_module = (
                                        importlib.util.module_from_spec(spec)
                                    )
                                    spec.loader.exec_module(coordinate_module)
                                    recomputed = coordinate_module.recompute_coordinate_evidence(
                                        workspace_root=root,
                                        source_summary=summary_path,
                                        release_manifest=resolved_recompute_paths[
                                            "release_manifest"
                                        ],
                                        current_models_lock=(
                                            root / "models" / "models.lock.json"
                                        ),
                                        pipeline_config=resolved_recompute_paths[
                                            "pipeline_config"
                                        ],
                                        profile=resolved_recompute_paths["profile"],
                                    )
                                except Exception as exc:
                                    release_binding_problems.append(
                                        "coordinate evidence recomputation failed: "
                                        f"{type(exc).__name__}: {exc}"
                                    )
                                else:
                                    exact_fields = (
                                        "model_lock_sha256",
                                        "config_sha256",
                                        "source_inference_run_id",
                                        "source_started_at_utc",
                                        "source_completed_at_utc",
                                        "source_summary_path",
                                        "source_summary_sha256",
                                        "source_prediction_manifest_path",
                                        "source_prediction_manifest_sha256",
                                        "source_prediction_set_sha256",
                                        "source_run_binding",
                                        "source_run_binding_sha256",
                                        "source_inference_runtime_manifest",
                                        "source_inference_runtime_manifest_sha256",
                                        "release_manifest_path",
                                        "release_manifest_sha256",
                                        "release_suite_contract",
                                        "release_suite_split",
                                        "release_suite_cases",
                                        "release_suite_duplicate_groups",
                                        "release_suite_duplicate_group_membership_sha256",
                                        "pipeline_config_path",
                                        "profile_path",
                                        "official_profile_path",
                                        "official_profile_sha256",
                                        "cases_checked",
                                        "case_ids_sha256",
                                        "points_checked",
                                        "component_bbox_points_checked",
                                        "pin_points_checked",
                                        "net_edge_points_checked",
                                        "zero_length_edges_checked",
                                        "ir_geometry_points_range_checked",
                                        "local_to_global_transform_points_checked",
                                        "local_to_global_coordinate_evidence_checked",
                                        "local_to_global_module_objects_checked",
                                        "persisted_transform_points_checked",
                                        "max_source_to_serialized_error_px",
                                        "max_roundtrip_error_px",
                                        "coordinate_case_manifest_semantic_sha256",
                                        "source_input_manifest",
                                        "source_input_manifest_sha256",
                                        "source_output_manifest",
                                        "source_output_manifest_sha256",
                                        "runtime_coordinate_sources",
                                        "runtime_coordinate_sources_sha256",
                                        "serializer_sha256",
                                        "coordinate_lineage_sha256",
                                        "coordinate_auditor_sha256",
                                    )
                                    for field in exact_fields:
                                        if loaded.get(field) != recomputed.get(field):
                                            release_binding_problems.append(
                                                f"coordinate report {field} differs from recomputation"
                                            )
                                    case_manifest_path, case_manifest_problem = (
                                        _resolve_workspace_evidence_path(
                                            root,
                                            loaded.get(
                                                "coordinate_case_manifest_path"
                                            ),
                                            field=(
                                                "coordinate report "
                                                "coordinate_case_manifest_path"
                                            ),
                                        )
                                    )
                                    if case_manifest_problem:
                                        release_binding_problems.append(
                                            case_manifest_problem
                                        )
                                    elif (
                                        case_manifest_path is None
                                        or not case_manifest_path.is_file()
                                    ):
                                        release_binding_problems.append(
                                            "coordinate case manifest is missing"
                                        )
                                    else:
                                        if loaded.get(
                                            "coordinate_case_manifest_sha256"
                                        ) != _sha256_file(case_manifest_path):
                                            release_binding_problems.append(
                                                "coordinate case manifest hash does not match bytes"
                                            )
                                        try:
                                            case_rows = _read_jsonl(
                                                case_manifest_path
                                            )
                                        except AuditInputError as exc:
                                            release_binding_problems.append(str(exc))
                                        else:
                                            if case_rows != recomputed.get(
                                                "coordinate_case_manifest"
                                            ):
                                                release_binding_problems.append(
                                                    "coordinate case manifest differs from recomputation"
                                                )
                else:
                    if loaded.get("current_lock_inference_executed") is not False:
                        problems.append(
                            "frozen-IR replay must explicitly deny current-lock inference"
                        )
                    if loaded.get("models_loaded") is not False:
                        problems.append(
                            "frozen-IR replay must explicitly state that models were not loaded"
                        )
                    source_lock = loaded.get("source_inference_model_lock_sha256")
                    if not isinstance(source_lock, str) or not HEX_SHA256.fullmatch(
                        source_lock
                    ):
                        problems.append(
                            "frozen-IR replay lacks source inference model-lock SHA-256"
                        )
                    if require_formal_binding:
                        release_binding_problems.append(
                            "frozen-IR replay cannot prove current-lock production inference"
                        )
    if payload.get("evidence_scope") == "current_lock_production_image_only_inference":
        problems.extend(release_binding_problems)
    report_valid = bool(evidence["present"] and not problems)
    evidence_scope = payload.get("evidence_scope")
    current_lock_inference_proven = bool(
        report_valid
        and evidence_scope == "current_lock_production_image_only_inference"
        and payload.get("current_lock_inference_executed") is True
        and payload.get("models_loaded") is True
    )
    serializer_replay_proven = bool(
        report_valid
        and evidence_scope == "frozen_ir_current_serializer_replay"
        and payload.get("current_lock_inference_executed") is False
        and payload.get("models_loaded") is False
    )
    if not evidence["present"] and static_origin_mismatch:
        status = "fail"
    elif not evidence["present"]:
        status = "unverified"
    elif problems:
        status = "fail"
    elif current_lock_inference_proven:
        status = "pass"
    else:
        # Coordinate math and strict serialization are proven by the replay,
        # but the release-level gate remains tied to a real current-lock
        # image-only inference path.
        status = "unverified"
    return {
        "status": status,
        "limit_px": COORDINATE_ROUNDTRIP_LIMIT_PX,
        "report": evidence,
        "official_coordinate_origin": "bottom_left" if official_guide_verified else None,
        "official_y_axis": "up" if official_guide_verified else None,
        "internal_y_axis_down": internal_y_axis_down,
        "serializer_emits_y_directly": serializer_emits_y_directly,
        "serializer_converts_to_bottom_left": serializer_converts_to_bottom_left,
        "static_origin_mismatch": static_origin_mismatch,
        "serializer_source": serializer_evidence,
        "coordinate_lineage_source": derive_evidence,
        "coordinate_auditor_source": coordinate_auditor_evidence,
        "evidence_scope": evidence_scope,
        "current_lock_inference_proven": current_lock_inference_proven,
        "serializer_replay_subcheck": "pass" if serializer_replay_proven else "unverified",
        "cases_checked": payload.get("cases_checked"),
        "points_checked": payload.get("points_checked"),
        "max_roundtrip_error_px": payload.get("max_roundtrip_error_px"),
        "problems": problems,
        "release_binding_problems": release_binding_problems,
        "source_summary_binding": source_summary,
        "formal_binding_required": require_formal_binding,
        "production_config_sha256": production_config_sha256,
        "boundary": (
            "The existing subpixel unit test is synthetic and does not test the official bottom-left/y-up conversion."
            if not evidence["present"]
            else (
                "Current serializer H-y round-trip and strict-schema replay passed on a frozen "
                "real image-only IR, but current-lock production inference was not executed."
                if serializer_replay_proven
                else None
            )
        ),
    }


def _bind_coordinate_to_release_runs(
    coordinate_gate: dict[str, Any],
    performance_gate: Mapping[str, Any],
    determinism_gate: Mapping[str, Any],
) -> None:
    """Require coordinate, performance, and determinism to share one 50-case run."""

    if coordinate_gate.get("status") != "pass":
        coordinate_gate["release_run_cross_binding"] = {
            "status": "unverified",
            "problems": ["coordinate evidence is not independently valid"],
        }
        return

    fields = (
        "path",
        "sha256",
        "inference_run_id",
        "model_lock_sha256",
        "config_hash",
        "selected_cases",
        "prediction_manifest_sha256",
        "prediction_set_sha256",
    )

    def identity(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        return {field: value.get(field) for field in fields}

    coordinate_source = coordinate_gate.get("source_summary_binding")
    performance_source = performance_gate.get("selected_newest_run")
    coordinate_identity = identity(coordinate_source)
    performance_identity = identity(performance_source)
    determinism_sources = [
        value
        for value in determinism_gate.get("formal_summary_bindings", [])
        if isinstance(value, Mapping)
    ]
    determinism_identities = [identity(value) for value in determinism_sources]
    problems: list[str] = []
    if performance_gate.get("status") != "pass" or performance_identity is None:
        problems.append("no passing selected canonical 50-case performance run")
    elif coordinate_identity != performance_identity:
        problems.append(
            "coordinate evidence is not bound to the selected 50-case performance summary"
        )
    if determinism_gate.get("status") != "pass":
        problems.append("three-run determinism evidence is not passing")
    elif coordinate_identity not in determinism_identities:
        problems.append(
            "coordinate/performance summary is not one of the three determinism runs"
        )
    if (
        coordinate_identity is None
        or coordinate_identity.get("selected_cases") != PERFORMANCE_CASES
    ):
        problems.append("coordinate source is not an exact 50-case summary")

    cross_binding = {
        "status": "pass" if not problems else "fail",
        "contract": "coordinate_performance_determinism_shared_run_v1",
        "coordinate_source": coordinate_identity,
        "performance_source": performance_identity,
        "determinism_source_count": len(determinism_identities),
        "coordinate_source_in_determinism": coordinate_identity
        in determinism_identities,
        "problems": problems,
    }
    coordinate_gate["release_run_cross_binding"] = cross_binding
    if problems:
        coordinate_gate["status"] = "fail"
        coordinate_gate["current_lock_inference_proven"] = False
        coordinate_gate.setdefault("problems", []).extend(
            problem
            for problem in problems
            if problem not in coordinate_gate.get("problems", [])
        )
        coordinate_gate.setdefault("release_binding_problems", []).extend(
            problem
            for problem in problems
            if problem not in coordinate_gate.get("release_binding_problems", [])
        )


def _container_build_context_binding(root: Path) -> dict[str, Any]:
    files: set[Path] = set()
    for relative in (
        ".dockerignore",
        "docker/Dockerfile",
        "docker/entrypoint.sh",
        "pyproject.toml",
        "README.md",
        "requirements.lock",
        "requirements.runtime.lock",
        "models/models.lock.json",
    ):
        path = root / relative
        if path.is_file():
            files.add(path.resolve())
    for directory in (
        root / "src",
        root / "configs",
        root / "data_tools",
        root / "models",
    ):
        if directory.is_dir():
            for path in directory.rglob("*"):
                if not path.is_file():
                    continue
                relative = path.relative_to(root).as_posix()
                if "__pycache__" in path.parts or path.suffix.casefold() == ".pyc":
                    continue
                if relative == "configs/qwen_local.json":
                    continue
                if relative.startswith("models/weights/") or relative.endswith(".bak"):
                    continue
                files.add(path.resolve())
    manifest = [
        {"path": path.relative_to(root).as_posix(), "sha256": _sha256_file(path)}
        for path in sorted(files, key=lambda item: item.relative_to(root).as_posix())
    ]
    return {
        "files": len(manifest),
        "manifest_sha256": _json_sha256(manifest) if manifest else None,
        "manifest": manifest,
    }


def _audit_offline_container(
    root: Path,
    report_path: Path | None,
    model_lock_sha256: str | None,
    production_config_sha256: str | None = None,
    *,
    require_formal_binding: bool = False,
) -> dict[str, Any]:
    evidence = _path_evidence(root, report_path)
    dockerfile = root / "docker" / "Dockerfile"
    docker_evidence = _path_evidence(root, dockerfile)
    docker_text = dockerfile.read_text(encoding="utf-8") if dockerfile.is_file() else ""
    build_context = _container_build_context_binding(root)
    runtime_requirements = root / "requirements.runtime.lock"
    static_lock_reference = (
        "COPY models" in docker_text
        and "--models-lock" in docker_text
        and "/app/models/models.lock.json" in docker_text
    )
    payload: Mapping[str, Any] = {}
    problems: list[str] = []
    source_summary: dict[str, Any] | None = None
    if require_formal_binding:
        if not isinstance(model_lock_sha256, str) or not HEX_SHA256.fullmatch(
            model_lock_sha256
        ):
            problems.append("current models.lock SHA-256 is unavailable")
        if not isinstance(production_config_sha256, str) or not HEX_SHA256.fullmatch(
            production_config_sha256
        ):
            problems.append("current production config SHA-256 is unavailable")
    if evidence["present"] and report_path is not None:
        try:
            loaded = _read_json(report_path)
        except AuditInputError as exc:
            problems.append(str(exc))
        else:
            if not isinstance(loaded, Mapping):
                problems.append("offline report root is not an object")
            else:
                payload = loaded
                checks = {
                    "kind": loaded.get("kind") == "offline-container-release-report",
                    "model_lock_sha256": loaded.get("model_lock_sha256") == model_lock_sha256,
                    "container_image_digest": isinstance(loaded.get("container_image_digest"), str)
                    and bool(IMAGE_DIGEST.fullmatch(loaded.get("container_image_digest", ""))),
                    "network_mode": loaded.get("network_mode") == "none",
                    "egress_probe_blocked": loaded.get("egress_probe_blocked") is True,
                    "healthcheck_passed": loaded.get("healthcheck_passed") is True,
                    "inference_passed": loaded.get("inference_passed") is True,
                    "official_outputs_validated": loaded.get("official_outputs_validated") is True,
                    "dockerfile_present": docker_evidence["present"],
                    "dockerfile_references_embedded_models_lock": static_lock_reference,
                    "dockerfile_sha256": loaded.get("dockerfile_sha256")
                    == docker_evidence.get("sha256"),
                    "runtime_requirements_sha256": loaded.get("runtime_requirements_sha256")
                    == (
                        _sha256_file(runtime_requirements)
                        if runtime_requirements.is_file()
                        else None
                    ),
                    "embedded_models_lock_sha256": loaded.get(
                        "embedded_models_lock_sha256"
                    )
                    == model_lock_sha256,
                    "build_context_manifest_sha256": loaded.get(
                        "build_context_manifest_sha256"
                    )
                    == build_context.get("manifest_sha256"),
                }
                problems.extend(name for name, passed in checks.items() if not passed)
                if require_formal_binding:
                    formal_checks = {
                        "status": loaded.get("status") == "ok",
                        "formal_release_evidence": loaded.get("formal_release_evidence") is True,
                        "config_sha256": loaded.get("config_sha256")
                        == production_config_sha256,
                    }
                    problems.extend(name for name, passed in formal_checks.items() if not passed)
                    report_generated, time_problem = _parse_utc_timestamp(
                        loaded.get("generated_at_utc"),
                        field="offline report generated_at_utc",
                    )
                    if time_problem:
                        problems.append(time_problem)
                    summary_path, summary_problem = _resolve_workspace_evidence_path(
                        root,
                        loaded.get("source_summary_path"),
                        field="offline report source_summary_path",
                    )
                    if summary_problem:
                        problems.append(summary_problem)
                    elif summary_path is not None:
                        source_summary = _audit_production_summary_binding(
                            root,
                            summary_path,
                            model_lock_sha256,
                            production_config_sha256,
                            require_success=True,
                            require_embedded_time=True,
                        )
                        if not source_summary["valid"]:
                            problems.extend(
                                f"source summary: {problem}"
                                for problem in source_summary["problems"]
                            )
                        if loaded.get("source_summary_sha256") != source_summary.get("sha256"):
                            problems.append("offline report source summary hash does not match bytes")
                        if loaded.get("source_prediction_manifest_sha256") != source_summary.get(
                            "prediction_manifest_sha256"
                        ):
                            problems.append(
                                "offline report prediction manifest hash does not match source"
                            )
                        if loaded.get("source_prediction_set_sha256") != source_summary.get(
                            "prediction_set_sha256"
                        ):
                            problems.append(
                                "offline report prediction-set hash does not match source"
                            )
                        summary_timestamp = source_summary.get("generated_timestamp")
                        if (
                            report_generated is not None
                            and isinstance(summary_timestamp, (int, float))
                            and summary_timestamp > report_generated.timestamp()
                        ):
                            problems.append("offline report predates its source summary")
    if not evidence["present"]:
        status = "unverified"
    elif problems:
        status = "fail"
    else:
        status = "pass"
    return {
        "status": status,
        "report": evidence,
        "dockerfile": docker_evidence,
        "dockerfile_references_embedded_models_lock": static_lock_reference,
        "build_context": build_context,
        "network_mode": payload.get("network_mode"),
        "container_image_digest": payload.get("container_image_digest"),
        "problems": problems,
        "source_summary_binding": source_summary,
        "formal_binding_required": require_formal_binding,
        "production_config_sha256": production_config_sha256,
        "boundary": (
            "Dockerfile presence is not evidence of a runtime launched with network=none."
            if not evidence["present"]
            else None
        ),
    }


def _select_release_report(
    root: Path,
    candidates: Sequence[Path],
    *,
    kind: str,
    model_lock_sha256: str | None,
    production_config_sha256: str | None,
    official_guide_verified: bool,
) -> tuple[Path | None, dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    current_lineage: list[tuple[float, Path, dict[str, Any], bool]] = []
    unique: dict[str, Path] = {
        str(path.resolve(strict=False)).casefold(): path.resolve(strict=False)
        for path in candidates
        if path.is_file()
    }
    for path in sorted(unique.values(), key=lambda item: str(item).casefold()):
        if kind == "coordinate":
            audited = _audit_coordinate_report(
                root,
                path,
                model_lock_sha256,
                official_guide_verified=official_guide_verified,
                production_config_sha256=production_config_sha256,
                require_formal_binding=True,
            )
            reasons = list(audited.get("problems", [])) + list(
                audited.get("release_binding_problems", [])
            )
        elif kind == "offline":
            audited = _audit_offline_container(
                root,
                path,
                model_lock_sha256,
                production_config_sha256,
                require_formal_binding=True,
            )
            reasons = list(audited.get("problems", []))
        else:  # pragma: no cover - internal call contract
            raise AuditInputError(f"unknown release report kind {kind!r}")
        try:
            loaded = _read_json(path)
        except AuditInputError:
            loaded = {}
        if not isinstance(loaded, Mapping):
            loaded = {}
        generated, time_problem = _parse_utc_timestamp(
            loaded.get("generated_at_utc"), field=f"{kind} report generated_at_utc"
        )
        if time_problem and time_problem not in reasons:
            reasons.append(time_problem)
        if audited.get("status") != "pass" and not reasons:
            reasons.append(f"{kind} gate status is {audited.get('status')}")
        record = {
            "path": _relative(root, path),
            "sha256": _sha256_file(path),
            "generated_at_utc": loaded.get("generated_at_utc"),
            "evidence_scope": loaded.get("evidence_scope"),
            "declared_model_lock_sha256": loaded.get("model_lock_sha256"),
            "declared_config_sha256": loaded.get("config_sha256"),
            "eligible": audited.get("status") == "pass",
            "rejection_reasons": list(dict.fromkeys(reasons)),
        }
        inventory.append(record)
        declares_current = bool(
            generated is not None
            and loaded.get("model_lock_sha256") == model_lock_sha256
            and loaded.get("config_sha256") == production_config_sha256
            and (
                kind != "coordinate"
                or loaded.get("evidence_scope")
                == "current_lock_production_image_only_inference"
            )
        )
        record["declares_current_lineage"] = declares_current
        if declares_current and generated is not None:
            current_lineage.append(
                (generated.timestamp(), path, record, audited.get("status") == "pass")
            )

    current_lineage.sort(key=lambda item: (item[0], str(item[1]).casefold()), reverse=True)
    selected: Path | None = None
    ambiguity: list[str] = []
    if current_lineage:
        newest_time = current_lineage[0][0]
        newest = [item for item in current_lineage if item[0] == newest_time]
        if len(newest) == 1:
            _, candidate, record, passed = newest[0]
            record["newest_current_candidate"] = True
            if passed:
                selected = candidate
                record["selected"] = True
            else:
                record["rejection_reasons"].append(
                    "newest current-lineage candidate failed formal audit; older evidence is not reused"
                )
        else:
            ambiguity = [str(item[1]) for item in newest]
            for item in newest:
                item[2]["eligible"] = False
                item[2]["rejection_reasons"].append(
                    "automatic selection is ambiguous at the newest timestamp"
                )

    return selected, {
        "mode": "automatic",
        "kind": kind,
        "selected_path": _relative(root, selected),
        "expected_model_lock_sha256": model_lock_sha256,
        "expected_production_config_sha256": production_config_sha256,
        "ambiguous_newest_candidates": ambiguity,
        "candidates": inventory,
        "finding": (
            f"No automatically eligible current-lock formal {kind} report exists; "
            "use an explicit path only for diagnostic inspection."
            if selected is None
            else None
        ),
    }


def _is_traceable_error(row: Mapping[str, Any]) -> tuple[bool, list[str]]:
    problems: list[str] = []
    for field in ("case_id", "image_sha256", "config_hash", "model_lock_sha256"):
        value = row.get(field)
        if not isinstance(value, str) or not value:
            problems.append(field)
    for field in ("image_sha256", "config_hash", "model_lock_sha256"):
        value = row.get(field)
        if isinstance(value, str) and not HEX_SHA256.fullmatch(value):
            problems.append(f"{field}_format")
    error = row.get("error")
    if not isinstance(error, Mapping):
        problems.append("error")
        return False, problems
    for field in ("type", "message", "stage"):
        if not isinstance(error.get(field), str) or not error.get(field):
            problems.append(f"error.{field}")
    error_id = error.get("error_id")
    if not isinstance(error_id, str) or not HEX_SHA256.fullmatch(error_id):
        problems.append("error.error_id")
    diagnostic_count = error.get("diagnostic_count")
    diagnostics = error.get("diagnostics")
    diagnostic_codes = error.get("diagnostic_codes")
    if not isinstance(diagnostic_count, int) or isinstance(diagnostic_count, bool):
        problems.append("error.diagnostic_count")
    if not isinstance(diagnostic_codes, Mapping):
        problems.append("error.diagnostic_codes")
    if not isinstance(diagnostics, list):
        problems.append("error.diagnostics")
    else:
        if isinstance(diagnostic_count, int) and not isinstance(
            diagnostic_count, bool
        ) and diagnostic_count != len(diagnostics):
            problems.append("error.diagnostic_count_mismatch")
        actual_codes = dict(
            sorted(
                Counter(
                    str(item.get("code", "UNKNOWN"))
                    for item in diagnostics
                    if isinstance(item, Mapping)
                ).items()
            )
        )
        if isinstance(diagnostic_codes, Mapping) and dict(diagnostic_codes) != actual_codes:
            problems.append("error.diagnostic_codes_mismatch")
    if isinstance(error_id, str) and HEX_SHA256.fullmatch(error_id):
        error_without_id = dict(error)
        error_without_id.pop("error_id", None)
        expected_error_id = _compact_json_sha256(
            {
                "case_id": row.get("case_id"),
                "image_sha256": row.get("image_sha256"),
                "config_hash": row.get("config_hash"),
                "model_lock_sha256": row.get("model_lock_sha256"),
                "error": error_without_id,
            }
        )
        if error_id != expected_error_id:
            problems.append("error.error_id_mismatch")
    return not problems, problems


def _audit_traceable_errors(
    root: Path,
    model_lock_sha256: str | None,
    harness_path: Path,
    production_config_sha256: str | None = None,
    *,
    require_formal_binding: bool = False,
) -> dict[str, Any]:
    harness_evidence = _path_evidence(root, harness_path)
    source = harness_path.read_text(encoding="utf-8") if harness_path.is_file() else ""
    harness_coverage = all(
        token in source
        for token in (
            "def _traceable_error(",
            '"error_id"',
            '"diagnostic_codes"',
            '"diagnostics"',
        )
    )
    manifests = sorted((root / "work").glob("production-e2e-*/predictions.jsonl")) if (root / "work").is_dir() else []
    errors: list[dict[str, Any]] = []
    for path in manifests:
        summary_binding: dict[str, Any] | None = None
        summary_manifest_bound = not require_formal_binding
        if require_formal_binding:
            summary_binding = _audit_production_summary_binding(
                root,
                path.parent / "summary.json",
                model_lock_sha256,
                production_config_sha256,
                require_success=False,
                require_embedded_time=True,
            )
            declared_manifest = summary_binding.get("prediction_manifest")
            summary_manifest_bound = bool(
                summary_binding.get("valid") is True
                and isinstance(declared_manifest, str)
                and declared_manifest.casefold()
                == (_relative(root, path) or "").casefold()
                and summary_binding.get("prediction_manifest_sha256") == _sha256_file(path)
            )
        try:
            rows = _read_jsonl(path)
        except AuditInputError as exc:
            errors.append(
                {
                    "path": _relative(root, path),
                    "current_lock": False,
                    "traceable": False,
                    "problems": [str(exc)],
                }
            )
            continue
        for row in rows:
            if row.get("status") != "error":
                continue
            traceable, problems = _is_traceable_error(row)
            errors.append(
                {
                    "path": _relative(root, path),
                    "case_id": row.get("case_id"),
                    "current_lock": row.get("model_lock_sha256") == model_lock_sha256,
                    "current_production_binding": bool(
                        row.get("model_lock_sha256") == model_lock_sha256
                        and (
                            not require_formal_binding
                            or (
                                row.get("config_hash") == production_config_sha256
                                and summary_manifest_bound
                            )
                        )
                    ),
                    "summary_manifest_bound": (
                        summary_manifest_bound if summary_binding is not None else None
                    ),
                    "traceable": traceable,
                    "error_type": row.get("error", {}).get("type")
                    if isinstance(row.get("error"), Mapping)
                    else None,
                    "problems": problems,
                }
            )
    current = [item for item in errors if item.get("current_production_binding")]
    if not harness_coverage:
        status = "fail"
    elif current and all(item["traceable"] for item in current):
        status = "pass"
    elif current:
        status = "fail"
    else:
        status = "unverified"
    return {
        "status": status,
        "harness_coverage": "pass" if harness_coverage else "fail",
        "harness": harness_evidence,
        "historic_error_rows": len(errors),
        "historic_traceable_rows": sum(bool(item["traceable"]) for item in errors),
        "current_lock_error_rows": len(current),
        "current_lock_traceable_rows": sum(bool(item["traceable"]) for item in current),
        "rows": errors,
        "formal_binding_required": require_formal_binding,
        "production_config_sha256": production_config_sha256,
        "boundary": (
            "The harness now emits hash-bound diagnostic errors, but no current-lock release probe has regenerated evidence."
            if harness_coverage and not current
            else None
        ),
    }


def _harness_coverage(root: Path) -> dict[str, Any]:
    paths = {
        "production_e2e": root / "tests" / "integration" / "test_production_eval_harness.py",
        "serializer": root / "tests" / "unit" / "test_validation_serializer.py",
        "coordinate": root / "tests" / "unit" / "test_ir_graph.py",
        "readiness": root / "tests" / "integration" / "test_http_readiness.py",
        "audit": root / "tests" / "unit" / "test_release_hard_gate_audit.py",
    }
    evidence = {name: _path_evidence(root, path) for name, path in paths.items()}
    contents = {
        name: path.read_text(encoding="utf-8") if path.is_file() else ""
        for name, path in paths.items()
    }
    return {
        "files": evidence,
        "official_exact_shape_internal": all(
            token in contents["serializer"]
            for token in ("test_extra_field_guard_rejects_root_and_nested_extras", "validate_official_output")
        ),
        "dangling_reference_internal": "test_unknown_references_fail_before_output" in contents["serializer"],
        "same_process_three_run_hash": all(
            token in contents["production_e2e"]
            for token in ("third = self._run", 'third.summary["prediction_set_sha256"]')
        ),
        "coordinate_unit_only": all(
            token in contents["coordinate"]
            for token in ("max_roundtrip_error", "0.5")
        ),
        "model_readiness_fail_closed": "test_client_cannot_bypass_server_side_readiness_gate" in contents["readiness"],
        "hard_gate_audit_tests": evidence["audit"]["present"],
    }


def build_audit(
    *,
    workspace_root: str | Path,
    models_lock: str | Path = "models/models.lock.json",
    production_config: str | Path = "configs/pipeline.yaml",
    production_profile: str | Path = "configs/profiles/competition.yaml",
    determinism_root: str | Path | None = None,
    determinism_environment: str | Path | None = None,
    official_guide: str | Path | None = None,
    official_qa: str | Path | None = None,
    official_sample: str | Path | None = None,
    official_evaluator: str | Path | None = None,
    official_evaluator_report: str | Path | None = None,
    coordinate_report: str | Path | None = None,
    offline_report: str | Path | None = None,
    performance_summaries: Sequence[str | Path] = (),
) -> dict[str, Any]:
    root = Path(workspace_root).expanduser().resolve()
    lock_path = _resolve_optional(root, models_lock)
    if lock_path is None:
        raise AuditInputError("models_lock cannot be null")
    production_config_path = _resolve_optional(root, production_config)
    production_profile_path = _resolve_optional(root, production_profile)
    production_binding = _production_config_binding(
        root,
        production_config_path,
        production_profile_path,
    )
    production_config_sha256 = production_binding.get("effective_config_sha256")
    guide_path = _resolve_optional(root, official_guide) or _first_existing(
        [
            root
            / "artifacts"
            / "reference"
            / "2026_EDA_Challenge_Task6_AI_PCB_Schematic_Official_Guide.pdf"
        ]
    )
    qa_path = _resolve_optional(root, official_qa) or _first_existing(
        [
            root / "artifacts" / "reference" / "official_qa.pdf",
            root / "artifacts" / "reference" / "official_qa.docx",
            root / "artifacts" / "reference" / "official_qa.txt",
        ]
    )
    sample_path = _resolve_optional(root, official_sample) or _first_existing(
        [
            root / "sample_submission.json",
            root / "official" / "sample_submission.json",
            root / "data" / "official" / "sample_submission.json",
            root / "Dataset300" / "sample_submission.json",
        ]
    )
    evaluator_path = _resolve_optional(root, official_evaluator) or _first_existing(
        [
            root / "official_evaluator.py",
            root / "official" / "evaluator.py",
            root / "official" / "official_evaluator.py",
            root / "evaluator" / "official_evaluator.py",
        ]
    )
    # Machine-evaluator reports are never guessed by filename: without a
    # current-lock/config/time manifest contract an old report can validate the
    # wrong output tree.  Callers may still pass a path for diagnostic review.
    evaluator_report_path = _resolve_optional(root, official_evaluator_report)

    model_gate = _audit_model_lock(root, lock_path, production_binding)
    lock_sha256 = model_gate["lock"].get("sha256")

    if determinism_root is not None or determinism_environment is not None:
        environment_path = _resolve_optional(root, determinism_environment)
        if determinism_root is not None:
            determinism_path = _resolve_optional(root, determinism_root)
        elif environment_path is not None:
            determinism_path = environment_path.parent
        else:  # pragma: no cover - guarded by branch condition
            determinism_path = None
        if determinism_path is None:
            raise AuditInputError("explicit determinism evidence cannot resolve to a path")
        if environment_path is None:
            environment_path = _first_existing(
                [
                    determinism_path / "environment.json",
                    determinism_path / "environment_manifest.json",
                ]
            )
        determinism_selection = {
            "mode": "explicit",
            "selected_path": _relative(root, determinism_path),
            "selected_environment": _relative(root, environment_path),
            "expected_model_lock_sha256": lock_sha256,
            "expected_production_config_sha256": production_config_sha256,
            "finding": (
                "Explicit evidence is audited but never treated as current merely because its path was supplied."
            ),
            "candidates": [],
        }
    else:
        determinism_path, environment_path, determinism_selection = _select_determinism_evidence(
            root,
            root / "artifacts" / "determinism",
            lock_sha256,
            production_config_sha256,
        )
        if determinism_path is None:
            determinism_path = root / "artifacts" / "determinism" / "__no_current_formal_evidence__"

    determinism_gate = _audit_determinism(
        root,
        determinism_path,
        environment_path,
        lock_sha256,
        production_config_sha256,
        require_formal_binding=True,
    )
    local_schema_path = root / "src" / "pcbparse" / "resources" / "official_schema.json"
    internal_guide = _audit_internal_guide_outputs(
        root,
        determinism_gate,
        local_schema_path,
    )
    run1_tree_sha256 = (
        determinism_gate["runs"][0]["tree_sha256"]
        if determinism_gate.get("runs")
        else None
    )
    run1_json_files = (
        determinism_gate["runs"][0]["json_files"]
        if determinism_gate.get("runs")
        else 0
    )
    official_contract = _audit_official_contract(
        root,
        guide_path,
        qa_path,
        sample_path,
        evaluator_path,
        evaluator_report_path,
        run1_tree_sha256,
        run1_json_files,
        internal_guide,
    )

    release_root = root / "artifacts" / "release"
    if coordinate_report is not None:
        coordinate_report_path = _resolve_optional(root, coordinate_report)
        coordinate_selection = {
            "mode": "explicit",
            "kind": "coordinate",
            "selected_path": _relative(root, coordinate_report_path),
            "expected_model_lock_sha256": lock_sha256,
            "expected_production_config_sha256": production_config_sha256,
            "finding": (
                "Explicit evidence is audited but never treated as current merely because its path was supplied."
            ),
            "candidates": [],
        }
    else:
        coordinate_report_path, coordinate_selection = _select_release_report(
            root,
            list(release_root.glob("*coordinate*.json")) if release_root.is_dir() else [],
            kind="coordinate",
            model_lock_sha256=lock_sha256,
            production_config_sha256=production_config_sha256,
            official_guide_verified=official_contract["official_guide_authority_available"],
        )
    if offline_report is not None:
        offline_report_path = _resolve_optional(root, offline_report)
        offline_selection = {
            "mode": "explicit",
            "kind": "offline",
            "selected_path": _relative(root, offline_report_path),
            "expected_model_lock_sha256": lock_sha256,
            "expected_production_config_sha256": production_config_sha256,
            "finding": (
                "Explicit evidence is audited but never treated as current merely because its path was supplied."
            ),
            "candidates": [],
        }
    else:
        offline_report_path, offline_selection = _select_release_report(
            root,
            list(release_root.glob("*offline*.json")) if release_root.is_dir() else [],
            kind="offline",
            model_lock_sha256=lock_sha256,
            production_config_sha256=production_config_sha256,
            official_guide_verified=official_contract["official_guide_authority_available"],
        )

    explicit_summaries = [
        path
        for item in performance_summaries
        if (path := _resolve_optional(root, item)) is not None
    ]
    summary_paths = _discover_production_summaries(root, explicit_summaries)
    performance_gate, summaries = _audit_performance(
        root,
        summary_paths,
        lock_sha256,
        production_config_sha256,
        require_formal_binding=True,
    )
    _bind_inactive_junction_runtime_evidence(model_gate, performance_gate)
    coordinate_gate = _audit_coordinate_report(
        root,
        coordinate_report_path,
        lock_sha256,
        official_guide_verified=official_contract["official_guide_authority_available"],
        production_config_sha256=production_config_sha256,
        require_formal_binding=True,
    )
    _bind_coordinate_to_release_runs(
        coordinate_gate,
        performance_gate,
        determinism_gate,
    )
    offline_gate = _audit_offline_container(
        root,
        offline_report_path,
        lock_sha256,
        production_config_sha256,
        require_formal_binding=True,
    )
    trace_gate = _audit_traceable_errors(
        root,
        lock_sha256,
        root / "src" / "pcbparse" / "evaluation" / "infer_manifest.py",
        production_config_sha256,
        require_formal_binding=True,
    )
    coverage = _harness_coverage(root)

    gates = {
        "official_json_zero_extra_fields": official_contract[
            "official_json_zero_extra_fields"
        ],
        "official_json_zero_dangling_references": official_contract[
            "official_json_zero_dangling_references"
        ],
        "coordinate_roundtrip_lte_0_5px": coordinate_gate,
        "same_environment_three_run_json_hash": determinism_gate,
        "fifty_cases_under_two_hours_with_ten_percent_reserve": performance_gate,
        "offline_container": offline_gate,
        "locked_blessed_models": model_gate,
        "traceable_errors": trace_gate,
    }
    blocking = [name for name, value in gates.items() if value.get("status") != "pass"]
    report: dict[str, Any] = {
        "schema_version": 2,
        "kind": "independent-release-hard-gate-audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "audit_role": "diagnostic_fail_closed_no_bless_no_promotion",
        "execution": {
            "device": "cpu_only",
            "gpu_used": False,
            "models_loaded": False,
            "inference_executed": False,
            "training_configuration_modified": False,
            "models_lock_modified": False,
            "test_data_used_for_tuning": False,
        },
        "policy": {
            "coordinate_roundtrip_limit_px": COORDINATE_ROUNDTRIP_LIMIT_PX,
            "determinism_runs": DETERMINISM_RUNS,
            "performance_cases": PERFORMANCE_CASES,
            "performance_deadline_seconds": PERFORMANCE_DEADLINE_SECONDS,
            "performance_minimum_reserve_fraction": PERFORMANCE_RESERVE_FRACTION,
            "performance_pass_seconds_lte": PERFORMANCE_PASS_SECONDS,
            "missing_official_authority_policy": "unverified_and_blocking",
            "missing_evidence_policy": "unverified_and_blocking",
            "automatic_evidence_selection_policy": (
                "current lock + current production config + formal status + embedded UTC time + byte manifests"
            ),
        },
        "production_configuration": {
            key: value for key, value in production_binding.items() if key != "effective_config"
        },
        "evidence_selection": {
            "determinism": determinism_selection,
            "coordinate": coordinate_selection,
            "offline_container": offline_selection,
            "official_evaluator_report": {
                "mode": "explicit" if official_evaluator_report is not None else "none",
                "selected_path": _relative(root, evaluator_report_path),
                "finding": (
                    None
                    if official_evaluator_report is not None
                    else "Automatic evaluator-report selection is disabled until a current-lock/config/time manifest is available."
                ),
            },
            "performance_summaries": {
                "mode": "explicit" if performance_summaries else "automatic_inventory",
                "selected_paths": [_relative(root, path) for path in summary_paths],
                "expected_model_lock_sha256": lock_sha256,
                "expected_production_config_sha256": production_config_sha256,
            },
        },
        "authority": {
            "official_guide": official_contract["official_guide"],
            "official_qa": official_contract["official_qa"],
            "official_sample": official_contract["official_sample"],
            "official_evaluator": official_contract["official_evaluator"],
            "official_evaluator_report": official_contract["official_evaluator_report"],
            "official_guide_authority_available": official_contract[
                "official_guide_authority_available"
            ],
            "official_machine_contract_available": official_contract[
                "official_machine_contract_available"
            ],
            "official_evaluator_report_valid": official_contract[
                "official_evaluator_report_valid"
            ],
            "report_problems": official_contract["report_problems"],
            "guide_contract": official_contract["guide_contract"],
        },
        "internal_guide_v1_subcheck": internal_guide,
        "production_summary_inventory": summaries,
        "harness_coverage": coverage,
        "gates": gates,
        "decision": {
            "release_ready": not blocking,
            "promotion_authorized": False,
            "bless_authorized": False,
            "blocking_gates": blocking,
            "passed_gates": [
                name for name, value in gates.items() if value.get("status") == "pass"
            ],
            "failed_gates": [
                name for name, value in gates.items() if value.get("status") == "fail"
            ],
            "unverified_gates": [
                name
                for name, value in gates.items()
                if value.get("status") == "unverified"
            ],
        },
    }
    report["audit_payload_sha256"] = _json_sha256(report)
    return report


def write_audit(report: Mapping[str, Any], output: str | Path) -> Path:
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        report,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit release hard gates from existing CPU-readable evidence. "
            "Missing official or runtime evidence remains blocking."
        )
    )
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--models-lock", default="models/models.lock.json")
    parser.add_argument("--production-config", default="configs/pipeline.yaml")
    parser.add_argument(
        "--production-profile",
        default="configs/profiles/competition.yaml",
    )
    parser.add_argument(
        "--determinism-root",
        help=(
            "Explicit diagnostic evidence root. When omitted, only a formally bound "
            "current-lock/config candidate may be selected automatically."
        ),
    )
    parser.add_argument("--determinism-environment")
    parser.add_argument("--official-guide")
    parser.add_argument("--official-qa")
    parser.add_argument("--official-sample")
    parser.add_argument("--official-evaluator")
    parser.add_argument("--official-evaluator-report")
    parser.add_argument("--coordinate-report")
    parser.add_argument("--offline-report")
    parser.add_argument("--performance-summary", action="append", default=[])
    parser.add_argument(
        "--output",
        default="artifacts/audits/release_hard_gate_audit.json",
    )
    parser.add_argument(
        "--require-release-ready",
        action="store_true",
        help="return exit code 2 unless every hard gate passes",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.workspace_root).expanduser().resolve()
    report = build_audit(
        workspace_root=root,
        models_lock=args.models_lock,
        production_config=args.production_config,
        production_profile=args.production_profile,
        determinism_root=args.determinism_root,
        determinism_environment=args.determinism_environment,
        official_guide=args.official_guide,
        official_qa=args.official_qa,
        official_sample=args.official_sample,
        official_evaluator=args.official_evaluator,
        official_evaluator_report=args.official_evaluator_report,
        coordinate_report=args.coordinate_report,
        offline_report=args.offline_report,
        performance_summaries=args.performance_summary,
    )
    output = _resolve_optional(root, args.output)
    if output is None:
        raise AuditInputError("output cannot be null")
    write_audit(report, output)
    print(
        json.dumps(
            {
                "ok": True,
                "release_ready": report["decision"]["release_ready"],
                "output": str(output),
                "audit_payload_sha256": report["audit_payload_sha256"],
                "blocking_gates": report["decision"]["blocking_gates"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if args.require_release_ready and not report["decision"]["release_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
