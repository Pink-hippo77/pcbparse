from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import secrets
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pcbparse.config import config_hash, load_config
from pcbparse.evaluation.raw_cli import (
    AccuracyManifestError,
    PredictionRecord,
    assert_accuracy_artifact,
)
from pcbparse.pipeline.orchestrator import IRProvider, infer_one
from pcbparse.service.readiness import ReadinessGateError, readiness_report


_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PRODUCTION_PROFILES = frozenset({"baseline", "competition"})
_FORBIDDEN_PROVIDER_MARKERS = (
    "dataset_debug",
    "dataset_golden",
    "datasetgolden",
    "emptyimageprovider",
    "empty_disaster_recovery",
    "heuristic",
    "debug",
)
_PRODUCTION_RUNTIME_SOURCE_DIRECTORIES = (
    "src/pcbparse/association",
    "src/pcbparse/graph",
    "src/pcbparse/io",
    "src/pcbparse/ir",
    "src/pcbparse/knowledge",
    "src/pcbparse/nets",
    "src/pcbparse/output",
    "src/pcbparse/perception",
    "src/pcbparse/pipeline",
    "src/pcbparse/reasoning",
    "src/pcbparse/training",
    "src/pcbparse/validation",
)
_PRODUCTION_RUNTIME_SOURCE_FILES = (
    "src/pcbparse/config.py",
    "src/pcbparse/evaluation/infer_manifest.py",
    "src/pcbparse/evaluation/raw_cli.py",
    "src/pcbparse/resources/official_schema.json",
    "src/pcbparse/service/readiness.py",
)


class ProductionEvaluationError(ValueError):
    """Raised when an accuracy run cannot prove production image-only input."""


@dataclass(frozen=True, slots=True)
class SelectedCase:
    case_id: str
    duplicate_group: str
    image_path: Path
    expected_image_sha256: str


@dataclass(frozen=True, slots=True)
class ProductionEvaluationResult:
    output_dir: Path
    prediction_manifest: Path
    summary_path: Path
    summary: dict[str, Any]


ReadinessChecker = Callable[..., dict[str, Any]]
ProviderLoader = Callable[[Path], IRProvider]


def production_runtime_source_manifest(
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Hash the exact repository source population allowed to affect inference.

    The manifest is captured both before model/config loading and after the
    prediction manifest has been fsynced.  A source edit during a formal run
    therefore fails closed, and a historical summary cannot later be wrapped
    in evidence produced by different runtime bytes.
    """

    root = (
        Path(repository_root).expanduser().resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    paths: set[Path] = set()
    for relative in _PRODUCTION_RUNTIME_SOURCE_FILES:
        path = (root / relative).resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError as exc:  # pragma: no cover - constants are repository-owned.
            raise ProductionEvaluationError(
                f"runtime source path escapes repository: {relative}"
            ) from exc
        if not path.is_file() or path.stat().st_size <= 0:
            raise ProductionEvaluationError(
                f"runtime source file is missing or empty: {relative}"
            )
        paths.add(path)
    for relative in _PRODUCTION_RUNTIME_SOURCE_DIRECTORIES:
        directory = (root / relative).resolve(strict=False)
        try:
            directory.relative_to(root)
        except ValueError as exc:  # pragma: no cover - constants are repository-owned.
            raise ProductionEvaluationError(
                f"runtime source directory escapes repository: {relative}"
            ) from exc
        if not directory.is_dir():
            raise ProductionEvaluationError(
                f"runtime source directory is missing: {relative}"
            )
        discovered = [path.resolve() for path in directory.rglob("*.py") if path.is_file()]
        if not discovered:
            raise ProductionEvaluationError(
                f"runtime source directory contains no Python files: {relative}"
            )
        paths.update(discovered)
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix())
    ]
    return {
        "schema_version": 1,
        "contract": "production_runtime_source_manifest_v1",
        "file_count": len(files),
        "files": files,
        "source_set_sha256": _json_hash(files),
    }


def run_production_evaluation(
    *,
    dataset_root: str | Path,
    split_manifest: str | Path,
    split: str,
    models_lock: str | Path,
    pipeline_config: str | Path,
    profile: str | Path,
    output_dir: str | Path,
    official_profile: str | Path | Mapping[str, Any] | None = None,
    fail_fast: bool = False,
    readiness_checker: ReadinessChecker | None = None,
    provider_loader: ProviderLoader | None = None,
) -> ProductionEvaluationResult:
    """Run a fail-closed, image-only production inference split.

    The split manifest is used only to select and verify image files.  No JSON,
    EDS, NET, or other annotation sidecar path is passed to the provider or to
    :func:`infer_one`.  ``readiness_checker`` and ``provider_loader`` are test
    seams; the CLI always uses the verified production implementations.
    """

    # The competition deadline applies to the whole submitted program, so the
    # canonical elapsed time starts before configuration, readiness, and model
    # loading.  Per-case timings remain separate diagnostics.
    run_started = time.monotonic()
    started_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    repository_root = Path(__file__).resolve().parents[3]
    runtime_source_manifest_before = production_runtime_source_manifest(repository_root)

    dataset = Path(dataset_root).expanduser().resolve()
    split_path = Path(split_manifest).expanduser().resolve()
    lock_path = Path(models_lock).expanduser().resolve()
    config_path = Path(pipeline_config).expanduser().resolve()
    profile_path = _resolve_profile_path(profile)
    destination = Path(output_dir).expanduser().resolve()

    if not dataset.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset}")
    if not split_path.is_file():
        raise FileNotFoundError(f"split manifest does not exist: {split_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"pipeline config does not exist: {config_path}")
    base_config = load_config(config_path)
    profile_payload = load_config(profile_path)
    effective_config = _apply_accuracy_profile(base_config, profile_payload)
    max_total_seconds = _resolve_total_time_limit(effective_config)
    effective_config_hash = config_hash(effective_config)
    perception_config = _perception_config_from_pipeline(effective_config)
    resolved_perception_config = perception_config.to_dict()
    required_model_tasks = sorted(
        _required_model_tasks_for_perception_config(perception_config)
    )
    from pcbparse.association.ocr_common import OCRAssociationConfig
    from pcbparse.knowledge.model_catalog import load_model_catalog
    association_config = OCRAssociationConfig.coerce(effective_config)
    static_catalog_inputs: list[Path] = []
    if association_config.component_model_catalog_path is not None:
        catalog_path = Path(association_config.component_model_catalog_path).resolve()
        load_model_catalog(str(catalog_path), association_config.component_model_catalog_sha256)
        static_catalog_inputs.append(catalog_path)

    checker = readiness_checker or readiness_report
    readiness = checker(
        provider_name="production",
        models_lock=lock_path,
        required_tasks=required_model_tasks,
    )
    if not lock_path.is_file():
        # Use the canonical gate so callers receive the same structured report
        # as the CLI/service instead of silently accepting an injected provider.
        raise ReadinessGateError(dict(readiness))
    if readiness.get("ready") is not True or readiness.get("status") != "ready":
        raise ReadinessGateError(dict(readiness))
    if str(readiness.get("provider", "")).strip().lower() != "production":
        raise ProductionEvaluationError(
            "accuracy mode requires readiness.provider='production'"
        )

    lock_sha256 = _sha256_file(lock_path)
    model_paths = _model_paths_from_lock(lock_path)
    selected_cases, selection = _select_cases(
        dataset,
        split_path,
        split=str(split).strip(),
    )

    provider = (
        provider_loader(lock_path)
        if provider_loader is not None
        else _load_production_provider(lock_path, perception_config)
    )
    _assert_production_provider(provider)
    provider_model_task_plan = _assert_provider_model_task_plan(
        provider,
        required_tasks=required_model_tasks,
        learned_junctions_enabled=perception_config.learned_junctions_enabled,
    )

    resolved_official_profile, official_profile_input = _resolve_official_profile(
        official_profile
    )
    shared_inputs = _unique_paths(
        [
            config_path,
            profile_path,
            lock_path,
            *model_paths,
            *static_catalog_inputs,
            *([official_profile_input] if official_profile_input is not None else []),
        ]
    )

    destination.mkdir(parents=True, exist_ok=True)
    official_dir = destination / "official"
    ir_dir = destination / "circuit_ir"
    official_dir.mkdir(parents=True, exist_ok=True)
    ir_dir.mkdir(parents=True, exist_ok=True)

    inference_run_id = secrets.token_hex(32)
    case_loop_started = time.monotonic()
    rows: list[dict[str, Any]] = []
    stop_after_error = False
    stop_after_time_limit = False
    for case in selected_cases:
        input_paths = _path_strings([case.image_path, *shared_inputs])
        stage_evidence: dict[str, dict[str, Any]] = {}

        def save_stage(stage: str, ir: Any) -> None:
            # Serialize immediately: later association/pruning mutates the IR.
            # Evidence is kept outside official outputs and never uses GT.
            if stage not in {"perception", "association", "pre_prune", "final"}:
                raise ProductionEvaluationError(f"unexpected evidence stage: {stage}")
            payload = _canonical_json_bytes(ir.model_dump(mode="json"))
            path = destination / "stage_evidence" / case.case_id / f"{stage}.circuit-ir.json"
            _atomic_write(path, payload)
            stage_evidence[stage] = {
                "path": path.relative_to(destination).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "state": str(ir.state.value),
                "counts": {
                    name: len(getattr(ir, name))
                    for name in ("components", "pins", "texts", "segments", "physical_nets", "electrical_nets")
                },
            }
        if stop_after_time_limit:
            rows.append(
                _base_record(
                    case,
                    input_paths=input_paths,
                    config_sha256=effective_config_hash,
                    lock_sha256=lock_sha256,
                    inference_run_id=inference_run_id,
                    status="not_run",
                    elapsed_seconds=0.0,
                    error={
                        "type": "CompetitionTimeLimitExceeded",
                        "message": (
                            "not run because the configured whole-program competition "
                            "time limit was already exceeded"
                        ),
                    },
                )
            )
            continue
        if stop_after_error:
            rows.append(
                _base_record(
                    case,
                    input_paths=input_paths,
                    config_sha256=effective_config_hash,
                    lock_sha256=lock_sha256,
                    inference_run_id=inference_run_id,
                    status="not_run",
                    elapsed_seconds=0.0,
                    error={
                        "type": "FailFastSkipped",
                        "message": "not run because an earlier case failed with fail_fast=true",
                    },
                )
            )
            continue

        if time.monotonic() - run_started >= max_total_seconds:
            stop_after_time_limit = True
            rows.append(
                _base_record(
                    case,
                    input_paths=input_paths,
                    config_sha256=effective_config_hash,
                    lock_sha256=lock_sha256,
                    inference_run_id=inference_run_id,
                    status="not_run",
                    elapsed_seconds=0.0,
                    error={
                        "type": "CompetitionTimeLimitExceeded",
                        "message": (
                            "not run because the configured whole-program competition "
                            f"time limit of {max_total_seconds:.3f}s was reached"
                        ),
                    },
                )
            )
            continue

        case_started = time.monotonic()
        official_path = official_dir / f"{case.case_id}.json"
        ir_path = ir_dir / f"{case.case_id}.circuit-ir.json"
        try:
            actual_image_sha256 = _sha256_file(case.image_path)
            if actual_image_sha256 != case.expected_image_sha256:
                raise ProductionEvaluationError(
                    f"case {case.case_id}: image SHA-256 differs from split manifest"
                )
            result = infer_one(
                case.image_path,
                official_path,
                provider=provider,
                config=effective_config,
                official_profile=resolved_official_profile,
                evidence_sink=save_stage,
                deadline_at=run_started + max_total_seconds,
            )
            if result.context.config_hash != effective_config_hash:
                raise ProductionEvaluationError(
                    f"case {case.case_id}: pipeline returned an unexpected config hash"
                )
            if result.ir.image.sha256 != actual_image_sha256:
                raise ProductionEvaluationError(
                    f"case {case.case_id}: provider image SHA-256 does not match the selected image"
                )
            if Path(result.ir.image.path).expanduser().resolve() != case.image_path:
                raise ProductionEvaluationError(
                    f"case {case.case_id}: provider reported a different input image path"
                )
            assert_accuracy_artifact(
                PredictionRecord(
                    case_id=case.case_id,
                    ir_path=ir_path,
                    provider_kind="production",
                    sidecar_access=False,
                    image_sha256=actual_image_sha256,
                ),
                result.ir,
            )
            _assert_production_ir_metadata(
                result.ir.metadata,
                case_id=case.case_id,
                expected_model_task_plan=provider_model_task_plan,
            )
            context_manifest = result.context.manifest()
            ir_payload = _canonical_json_bytes(result.ir.model_dump(mode="json"))
            _atomic_write(ir_path, ir_payload)
            elapsed = time.monotonic() - case_started
            rows.append(
                {
                    **_base_record(
                        case,
                        input_paths=input_paths,
                        config_sha256=effective_config_hash,
                        lock_sha256=lock_sha256,
                        inference_run_id=inference_run_id,
                        status="ok",
                        elapsed_seconds=elapsed,
                        error=None,
                    ),
                    "image_sha256": actual_image_sha256,
                    "ir_path": ir_path.relative_to(destination).as_posix(),
                    "ir_sha256": hashlib.sha256(ir_payload).hexdigest(),
                    "official_path": official_path.relative_to(destination).as_posix(),
                    "official_sha256": _sha256_file(official_path),
                    "official_output_sha256": result.output_hash,
                    "runtime_evidence": _runtime_evidence(context_manifest),
                    "quality": dict(result.quality),
                    "stage_evidence": stage_evidence,
                }
            )
        except Exception as exc:
            elapsed = time.monotonic() - case_started
            context_manifest = getattr(exc, "context_manifest", None)
            # Save failure IR for replay/debugging when available.
            failure_ir_path = None
            failure_ir_sha256 = None
            partial_ir = getattr(exc, "partial_ir", None)
            if partial_ir is not None:
                try:
                    failure_ir_payload = _canonical_json_bytes(partial_ir.model_dump(mode="json"))
                    failure_ir_path = ir_path
                    _atomic_write(ir_path, failure_ir_payload)
                    failure_ir_sha256 = hashlib.sha256(failure_ir_payload).hexdigest()
                except Exception:
                    pass  # Best effort; don't mask the original error
            row = _base_record(
                case,
                input_paths=input_paths,
                config_sha256=effective_config_hash,
                lock_sha256=lock_sha256,
                inference_run_id=inference_run_id,
                status="error",
                elapsed_seconds=elapsed,
                error=_traceable_error(
                    exc,
                    case_id=case.case_id,
                    image_sha256=case.expected_image_sha256,
                    config_sha256=effective_config_hash,
                    lock_sha256=lock_sha256,
                ),
                runtime_evidence=(
                    _runtime_evidence(context_manifest)
                    if isinstance(context_manifest, Mapping)
                    else None
                ),
            )
            if failure_ir_path is not None:
                row["ir_path"] = failure_ir_path.relative_to(destination).as_posix()
                row["ir_sha256"] = failure_ir_sha256
            row["stage_evidence"] = stage_evidence
            rows.append(row)
            if fail_fast:
                stop_after_error = True

    case_loop_completed = time.monotonic()
    _assert_exact_output_coverage(selected_cases, rows, selection)
    manifest_path = destination / "predictions.jsonl"
    manifest_payload = _canonical_jsonl_bytes(rows)
    _atomic_write(manifest_path, manifest_payload)

    runtime_source_manifest_after = production_runtime_source_manifest(repository_root)
    if runtime_source_manifest_after != runtime_source_manifest_before:
        raise ProductionEvaluationError(
            "production runtime source bytes changed during inference"
        )

    status_counts = Counter(str(row["status"]) for row in rows)
    succeeded = status_counts["ok"]
    failed = status_counts["error"]
    not_run = status_counts["not_run"]
    overall_status = "ok" if succeeded == len(rows) else ("partial" if succeeded else "failed")
    stable_prediction_sha256 = _stable_prediction_set_hash(rows)
    evidence_completed = time.monotonic()
    initialization_seconds = case_loop_started - run_started
    case_loop_seconds = case_loop_completed - case_loop_started
    evidence_finalize_seconds = evidence_completed - case_loop_completed
    total_elapsed_seconds = evidence_completed - run_started
    within_competition_time_limit = total_elapsed_seconds <= max_total_seconds
    if not within_competition_time_limit:
        stop_after_time_limit = True
    completed_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    latency_summary = _latency_summary(rows)
    token_summary = _token_usage_summary(rows)
    summary = {
        "schema_version": 2,
        "kind": "production-image-only-inference",
        "inference_run_id": inference_run_id,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "generated_at_utc": completed_at_utc,
        "accuracy_mode": "image-only",
        "execution_mode": "production",
        "provider_kind": "production",
        "sidecar_access": False,
        "status": overall_status,
        "all_successful": overall_status == "ok",
        "split": str(split).strip(),
        "selected_cases": len(selected_cases),
        "selected_duplicate_groups": selection["selected_duplicate_groups"],
        "attempted": succeeded + failed,
        "succeeded": succeeded,
        "failed": failed,
        "not_run": not_run,
        "quality_status_counts": dict(sorted(Counter(
            str(row.get("quality", {}).get("status", row["status"])) for row in rows
        ).items())),
        "fail_fast": bool(fail_fast),
        "elapsed_seconds": total_elapsed_seconds,
        "timing": {
            "contract": "production_wall_clock_scope_v1",
            "scope": "function_entry_through_prediction_manifest_fsync",
            "cold_start_included": True,
            "config_and_profile_load_included": True,
            "readiness_included": True,
            "provider_and_model_load_included": True,
            "case_loop_included": True,
            "prediction_manifest_write_included": True,
            "summary_write_and_cli_teardown_included": False,
            "initialization_seconds": initialization_seconds,
            "case_loop_seconds": case_loop_seconds,
            "evidence_finalize_seconds": evidence_finalize_seconds,
            "total_elapsed_seconds": total_elapsed_seconds,
            "competition_time_limit_seconds": max_total_seconds,
            "competition_time_limit_enforced": True,
            "within_competition_time_limit": within_competition_time_limit,
            "case_latency_seconds": latency_summary,
        },
        "token_usage": token_summary,
        "config_hash": effective_config_hash,
        "resolved_perception_config": resolved_perception_config,
        "resolved_perception_config_sha256": _json_hash(resolved_perception_config),
        "model_task_plan": provider_model_task_plan,
        "model_lock_sha256": lock_sha256,
        "runtime_source_manifest": runtime_source_manifest_before,
        "prediction_manifest": manifest_path.name,
        "prediction_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "prediction_set_sha256": stable_prediction_sha256,
        "split_manifest_sha256": _sha256_file(split_path),
        "coverage": {
            **selection,
            "manifest_case_ids_sha256": _string_list_hash(
                [str(row["case_id"]) for row in rows]
            ),
            "exact_case_coverage": True,
            "exact_duplicate_group_coverage": True,
            "duplicate_case_ids": [],
        },
        "control_inputs": {
            "dataset_root": str(dataset),
            "split_manifest": str(split_path),
            "pipeline_config": str(config_path),
            "profile": str(profile_path),
            "official_profile": (
                str(official_profile_input) if official_profile_input is not None else "mapping"
            ),
            "models_lock": str(lock_path),
            "competition_time_limit_seconds": max_total_seconds,
        },
        "selection_control_paths_read": [str(split_path)],
        "inference_input_policy": (
            "Per-case input_paths_read contains only the selected raster image, "
            "pipeline/profile/serializer configuration, models.lock, and model artifacts."
        ),
        "readiness": {
            "status": readiness.get("status"),
            "ready": readiness.get("ready"),
            "required_learned_tasks": readiness.get("required_learned_tasks", []),
            "registered_learned_tasks": readiness.get("registered_learned_tasks", []),
            "verified_learned_tasks": readiness.get("verified_learned_tasks", []),
        },
    }
    summary_path = destination / "summary.json"
    _atomic_write(summary_path, _pretty_json_bytes(summary))
    return ProductionEvaluationResult(
        output_dir=destination,
        prediction_manifest=manifest_path,
        summary_path=summary_path,
        summary=summary,
    )


def _base_record(
    case: SelectedCase,
    *,
    input_paths: list[str],
    config_sha256: str,
    lock_sha256: str,
    inference_run_id: str,
    status: str,
    elapsed_seconds: float,
    error: dict[str, Any] | None,
    runtime_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "duplicate_group": case.duplicate_group,
        "image_sha256": case.expected_image_sha256,
        "ir_path": None,
        "ir_sha256": None,
        "official_path": None,
        "official_sha256": None,
        "official_output_sha256": None,
        "provider_kind": "production",
        "model_lock_sha256": lock_sha256,
        "config_hash": config_sha256,
        "inference_run_id": inference_run_id,
        "sidecar_access": False,
        "input_paths_read": input_paths,
        "elapsed_seconds": elapsed_seconds,
        "runtime_evidence": runtime_evidence,
        "status": status,
        "error": error,
    }


def _resolve_total_time_limit(config: Mapping[str, Any]) -> float:
    run_config = config.get("run", {})
    raw = run_config.get("max_total_seconds", 7200.0) if isinstance(run_config, Mapping) else 7200.0
    if (
        isinstance(raw, bool)
        or not isinstance(raw, (int, float))
        or not math.isfinite(float(raw))
        or float(raw) <= 0.0
    ):
        raise ProductionEvaluationError(
            "run.max_total_seconds must be a finite positive number"
        )
    return float(raw)


def _runtime_evidence(context_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contract": "run_context_runtime_evidence_v1",
        "time": context_manifest.get("time", {}),
        "tokens": context_manifest.get("tokens", {}),
        "trace": context_manifest.get("trace", []),
    }


def _latency_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    values = sorted(
        float(row["elapsed_seconds"])
        for row in rows
        if row.get("status") in {"ok", "error"}
        and isinstance(row.get("elapsed_seconds"), (int, float))
        and not isinstance(row.get("elapsed_seconds"), bool)
        and math.isfinite(float(row["elapsed_seconds"]))
        and float(row["elapsed_seconds"]) >= 0.0
    )
    if not values:
        return {
            "contract": "case_latency_summary_v1",
            "count": 0,
            "min_seconds": None,
            "mean_seconds": None,
            "p50_seconds": None,
            "p95_seconds": None,
            "max_seconds": None,
        }
    return {
        "contract": "case_latency_summary_v1",
        "count": len(values),
        "min_seconds": values[0],
        "mean_seconds": sum(values) / len(values),
        "p50_seconds": _percentile(values, 0.50),
        "p95_seconds": _percentile(values, 0.95),
        "max_seconds": values[-1],
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile for an empty list")
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] + weight * (values[upper] - values[lower])


def _token_usage_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    usage_rows: list[Mapping[str, Any]] = []
    for row in rows:
        evidence = row.get("runtime_evidence")
        tokens = evidence.get("tokens") if isinstance(evidence, Mapping) else None
        if isinstance(tokens, Mapping):
            usage_rows.append(tokens)
    fields = ("calls", "input_tokens", "output_tokens")
    totals = {
        field: sum(
            int(item[field])
            for item in usage_rows
            if isinstance(item.get(field), int) and not isinstance(item.get(field), bool)
        )
        for field in fields
    }
    per_case_calls = [
        int(item["calls"])
        for item in usage_rows
        if isinstance(item.get("calls"), int) and not isinstance(item.get("calls"), bool)
    ]
    return {
        "contract": "run_context_token_usage_summary_v1",
        "cases_with_evidence": len(usage_rows),
        "cases_without_evidence": len(rows) - len(usage_rows),
        **{f"total_{field}": value for field, value in totals.items()},
        "max_calls_per_case": max(per_case_calls, default=0),
    }


def _traceable_error(
    exc: Exception,
    *,
    case_id: str,
    image_sha256: str,
    config_sha256: str,
    lock_sha256: str,
) -> dict[str, Any]:
    """Return stable failure details without serializing a Python traceback.

    PipelineError exposes structured validator diagnostics.  Other exceptions
    still receive a stable error ID bound to the case, image, configuration,
    and model lock so an E2E failure can be correlated without relying on wall
    clock time or an unstable traceback.
    """

    raw_diagnostics = getattr(exc, "diagnostics", None)
    diagnostics: list[dict[str, Any]] = []
    if isinstance(raw_diagnostics, list):
        for item in raw_diagnostics:
            if hasattr(item, "model_dump"):
                payload = item.model_dump(mode="json")
            elif isinstance(item, Mapping):
                payload = dict(item)
            else:
                continue
            diagnostics.append(
                {
                    "severity": payload.get("severity"),
                    "code": payload.get("code"),
                    "message": payload.get("message"),
                    "object_ids": payload.get("object_ids", []),
                    "stage": payload.get("stage"),
                }
            )

    diagnostics.sort(
        key=lambda item: (
            str(item.get("severity", "")),
            str(item.get("code", "")),
            str(item.get("stage", "")),
            str(item.get("message", "")),
            tuple(str(value) for value in item.get("object_ids", [])),
        )
    )
    diagnostic_codes = dict(
        sorted(Counter(str(item.get("code", "UNKNOWN")) for item in diagnostics).items())
    )
    stages = sorted(
        {
            str(item["stage"])
            for item in diagnostics
            if isinstance(item.get("stage"), str) and item["stage"]
        }
    )
    stage = stages[0] if len(stages) == 1 else ("multiple" if stages else "inference")
    error: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "stage": stage,
        "diagnostic_count": len(diagnostics),
        "diagnostic_codes": diagnostic_codes,
        "diagnostics": diagnostics,
    }
    identity = {
        "case_id": case_id,
        "image_sha256": image_sha256,
        "config_hash": config_sha256,
        "model_lock_sha256": lock_sha256,
        "error": error,
    }
    error["error_id"] = _json_hash(identity)
    return error


def _select_cases(
    dataset_root: Path,
    split_manifest: Path,
    *,
    split: str,
) -> tuple[list[SelectedCase], dict[str, Any]]:
    if not split:
        raise ProductionEvaluationError("split must be a non-empty string")
    records = _read_split_manifest(split_manifest)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        group = str(record["duplicate_group"])
        groups[group].append(record)
        group_splits[group].add(str(record["split"]))
    leaking = sorted(group for group, values in group_splits.items() if len(values) != 1)
    if leaking:
        raise ProductionEvaluationError(
            f"duplicate groups leak across splits: {leaking[:10]}"
        )
    for group, members in groups.items():
        declared_sizes = {
            int(item["duplicate_group_size"])
            for item in members
            if isinstance(item.get("duplicate_group_size"), int)
        }
        if declared_sizes and declared_sizes != {len(members)}:
            raise ProductionEvaluationError(
                f"duplicate group {group!r} declares sizes {sorted(declared_sizes)}, "
                f"but manifest contains {len(members)} member(s)"
            )

    selected_rows = [record for record in records if record["split"] == split]
    if not selected_rows:
        raise ProductionEvaluationError(f"selected split has no cases: {split!r}")
    selected_rows.sort(key=lambda item: _natural_key(item["sample_id"]))

    selected: list[SelectedCase] = []
    seen_images: dict[Path, str] = {}
    for record in selected_rows:
        files = record["files"]
        image = files["image"]
        relative = Path(image["path"])
        if relative.is_absolute():
            raise ProductionEvaluationError(
                f"case {record['sample_id']}: image path must be relative to dataset root"
            )
        resolved = (dataset_root / relative).resolve()
        try:
            resolved.relative_to(dataset_root)
        except ValueError as exc:
            raise ProductionEvaluationError(
                f"case {record['sample_id']}: image path escapes dataset root"
            ) from exc
        if resolved in seen_images:
            raise ProductionEvaluationError(
                f"cases {seen_images[resolved]!r} and {record['sample_id']!r} select the same image path"
            )
        seen_images[resolved] = record["sample_id"]
        selected.append(
            SelectedCase(
                case_id=record["sample_id"],
                duplicate_group=record["duplicate_group"],
                image_path=resolved,
                expected_image_sha256=image["sha256"],
            )
        )

    selected_groups = sorted(
        {case.duplicate_group for case in selected}, key=_natural_key
    )
    membership = {
        group: [case.case_id for case in selected if case.duplicate_group == group]
        for group in selected_groups
    }
    expected_case_ids = [case.case_id for case in selected]
    return selected, {
        "expected_case_ids_sha256": _string_list_hash(expected_case_ids),
        "expected_duplicate_group_membership_sha256": _json_hash(membership),
        "selected_duplicate_groups": len(selected_groups),
    }


def _read_split_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProductionEvaluationError(
                f"invalid split manifest line {line_number}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise ProductionEvaluationError(
                f"split manifest line {line_number} must be an object"
            )
        sample_id = raw.get("sample_id")
        duplicate_group = raw.get("duplicate_group")
        split = raw.get("split")
        files = raw.get("files")
        image = files.get("image") if isinstance(files, dict) else None
        image_path = image.get("path") if isinstance(image, dict) else None
        image_sha = image.get("sha256") if isinstance(image, dict) else None
        if not isinstance(sample_id, str) or not sample_id:
            raise ProductionEvaluationError(
                f"split manifest line {line_number} has no sample_id"
            )
        if not _SAFE_CASE_ID.fullmatch(sample_id) or sample_id in {".", ".."}:
            raise ProductionEvaluationError(
                f"case {sample_id!r}: sample_id is not safe for deterministic output filenames"
            )
        if sample_id in seen:
            raise ProductionEvaluationError(
                f"duplicate sample_id in split manifest: {sample_id!r}"
            )
        seen.add(sample_id)
        if not isinstance(duplicate_group, str) or not duplicate_group:
            raise ProductionEvaluationError(
                f"case {sample_id}: duplicate_group must be non-empty"
            )
        if not isinstance(split, str) or not split:
            raise ProductionEvaluationError(f"case {sample_id}: split must be non-empty")
        if not isinstance(image_path, str) or not image_path:
            raise ProductionEvaluationError(f"case {sample_id}: files.image.path is missing")
        if not isinstance(image_sha, str) or not _HEX_SHA256.fullmatch(image_sha.lower()):
            raise ProductionEvaluationError(
                f"case {sample_id}: files.image.sha256 is not a SHA-256 digest"
            )
        normalized = dict(raw)
        normalized["sample_id"] = sample_id
        normalized["duplicate_group"] = duplicate_group
        normalized["split"] = split
        normalized["files"] = {**files, "image": {**image, "sha256": image_sha.lower()}}
        rows.append(normalized)
    if not rows:
        raise ProductionEvaluationError("split manifest is empty")
    return rows


def _assert_exact_output_coverage(
    selected: list[SelectedCase],
    rows: list[dict[str, Any]],
    selection: dict[str, Any],
) -> None:
    expected_ids = [case.case_id for case in selected]
    actual_ids = [str(row.get("case_id", "")) for row in rows]
    duplicate_ids = sorted(
        case_id for case_id, count in Counter(actual_ids).items() if count > 1
    )
    if duplicate_ids or actual_ids != expected_ids:
        raise ProductionEvaluationError(
            "prediction rows do not exactly cover selected cases; "
            f"duplicates={duplicate_ids}, expected={expected_ids[:10]}, actual={actual_ids[:10]}"
        )
    membership: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        membership[str(row["duplicate_group"])].append(str(row["case_id"]))
    if _json_hash(dict(sorted(membership.items()))) != selection[
        "expected_duplicate_group_membership_sha256"
    ]:
        raise ProductionEvaluationError(
            "prediction rows do not exactly cover selected duplicate groups"
        )


def _apply_accuracy_profile(
    config: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    required = {"name", "provider", "qwen_enabled", "artifact_policy", "strict_validation"}
    missing = sorted(required - set(profile))
    if missing:
        raise ProductionEvaluationError(f"profile is missing required fields: {missing}")
    name = str(profile["name"]).strip().lower()
    provider = str(profile["provider"]).strip().lower()
    if name not in _PRODUCTION_PROFILES:
        raise ProductionEvaluationError(
            "accuracy mode accepts only baseline or competition production profiles"
        )
    if provider != "production":
        raise ProductionEvaluationError(
            "accuracy mode requires profile.provider='production'"
        )
    effective = copy.deepcopy(config)
    run = effective.setdefault("run", {})
    qwen = effective.setdefault("qwen", {})
    validation = effective.setdefault("validation", {})
    if not all(isinstance(value, dict) for value in (run, qwen, validation)):
        raise ProductionEvaluationError(
            "pipeline run/qwen/validation sections must be objects"
        )
    run["profile"] = name
    run["artifact_policy"] = str(profile["artifact_policy"])
    qwen["enabled"] = bool(profile["qwen_enabled"])
    validation["require_zero_errors"] = bool(profile["strict_validation"])
    return effective


def _assert_production_provider(provider: IRProvider) -> None:
    if not callable(getattr(provider, "build_ir", None)):
        raise ProductionEvaluationError("production provider does not implement build_ir()")
    declared = getattr(provider, "provider_kind", None)
    if declared is not None and str(declared).strip().lower() != "production":
        raise ProductionEvaluationError(
            "accuracy mode rejects a provider that does not declare provider_kind='production'"
        )
    descriptor = " ".join(
        (
            type(provider).__module__,
            type(provider).__name__,
            str(getattr(provider, "version", "")),
            str(getattr(provider, "execution_mode", "")),
        )
    ).replace("-", "_").casefold()
    if any(marker in descriptor for marker in _FORBIDDEN_PROVIDER_MARKERS):
        raise ProductionEvaluationError(
            "accuracy mode rejects annotation-backed, debug, empty, or heuristic providers"
        )
    if not isinstance(getattr(provider, "version", None), str) or not provider.version:
        raise ProductionEvaluationError("production provider has no non-empty version")


def _assert_production_ir_metadata(
    metadata: Mapping[str, Any],
    *,
    case_id: str,
    expected_model_task_plan: Mapping[str, Any],
) -> None:
    declared = metadata.get("provider")
    rendered = json.dumps(
        declared,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ).replace("-", "_").casefold()
    if not rendered or rendered == "null":
        raise ProductionEvaluationError(
            f"case {case_id}: Circuit-IR has no provider provenance"
        )
    if any(marker in rendered for marker in _FORBIDDEN_PROVIDER_MARKERS):
        raise ProductionEvaluationError(
            f"case {case_id}: Circuit-IR provider provenance is not production-safe"
        )
    explicit_kind = (
        str(declared.get("provider_kind", "")).strip().lower()
        if isinstance(declared, Mapping)
        else ""
    )
    if "production" not in rendered and explicit_kind != "production":
        raise ProductionEvaluationError(
            f"case {case_id}: Circuit-IR provider provenance does not declare production"
        )
    if metadata.get("model_task_plan") != expected_model_task_plan:
        raise ProductionEvaluationError(
            f"case {case_id}: Circuit-IR model task plan differs from the loaded provider"
        )
    model_versions = metadata.get("model_versions")
    if not isinstance(model_versions, Mapping):
        raise ProductionEvaluationError(
            f"case {case_id}: Circuit-IR model_versions must be an object"
        )
    expected_loaded = set(expected_model_task_plan["loaded_model_tasks"])
    if set(model_versions) != expected_loaded:
        raise ProductionEvaluationError(
            f"case {case_id}: Circuit-IR model_versions do not exactly match loaded model tasks"
        )
    if expected_model_task_plan["junction_mode"] == "deterministic_vectorizer":
        quarantined = metadata.get("quarantined_model_versions", {})
        if quarantined not in ({}, None):
            raise ProductionEvaluationError(
                f"case {case_id}: deterministic Junction mode cannot retain a quarantined learned model"
            )


def _assert_provider_model_task_plan(
    provider: IRProvider,
    *,
    required_tasks: list[str],
    learned_junctions_enabled: bool,
) -> dict[str, Any]:
    raw = getattr(provider, "model_task_plan", None)
    if callable(raw):
        raw = raw()
    if not isinstance(raw, Mapping):
        raise ProductionEvaluationError(
            "production provider must expose a model_task_plan object"
        )
    try:
        plan = json.loads(
            json.dumps(raw, ensure_ascii=False, allow_nan=False, sort_keys=True)
        )
    except (TypeError, ValueError) as exc:
        raise ProductionEvaluationError(
            f"production provider model_task_plan is not JSON-safe: {exc}"
        ) from exc
    expected_required = sorted(required_tasks)
    expected_junction_mode = (
        "learned_model_plus_deterministic_vectorizer"
        if learned_junctions_enabled
        else "deterministic_vectorizer"
    )
    if plan.get("contract") != "production_model_task_plan_v1":
        raise ProductionEvaluationError("production provider model task plan contract is invalid")
    if plan.get("junction_mode") != expected_junction_mode:
        raise ProductionEvaluationError(
            "production provider Junction mode differs from resolved pipeline configuration"
        )
    for field in ("required_model_tasks", "loaded_model_tasks"):
        values = plan.get(field)
        if (
            not isinstance(values, list)
            or any(not isinstance(item, str) or not item for item in values)
            or values != sorted(set(values))
        ):
            raise ProductionEvaluationError(
                f"production provider model task plan {field} must be a sorted unique string list"
            )
    if plan["required_model_tasks"] != expected_required:
        raise ProductionEvaluationError(
            "production provider required model tasks differ from readiness policy"
        )
    if plan["loaded_model_tasks"] != expected_required:
        raise ProductionEvaluationError(
            "production provider did not load exactly the readiness-verified model tasks"
        )
    expected_junction = learned_junctions_enabled
    for field, expected in (
        ("learned_junction_model_required", expected_junction),
        ("learned_junction_model_loaded", expected_junction),
        ("learned_junction_model_invoked", expected_junction),
    ):
        if type(plan.get(field)) is not bool or plan[field] is not expected:
            raise ProductionEvaluationError(
                f"production provider model task plan {field} is inconsistent"
            )
    return plan


def _model_paths_from_lock(lock_path: Path) -> list[Path]:
    try:
        payload = json.loads(
            lock_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ProductionEvaluationError(f"models.lock is invalid: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ProductionEvaluationError("models.lock must contain a models list")
    paths: list[Path] = []
    for index, entry in enumerate(payload["models"]):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ProductionEvaluationError(
                f"models.lock entry {index} has no model path"
            )
        path = Path(entry["path"]).expanduser()
        paths.append((path if path.is_absolute() else lock_path.parent / path).resolve())
    return _unique_paths(paths)


def _resolve_profile_path(profile: str | Path) -> Path:
    candidate = Path(profile).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    workspace = Path("configs/profiles") / f"{profile}.yaml"
    if workspace.is_file():
        return workspace.resolve()
    packaged = Path(__file__).resolve().parents[1] / "resources" / "profiles" / f"{profile}.json"
    if packaged.is_file():
        return packaged.resolve()
    raise FileNotFoundError(f"production profile does not exist: {profile}")


def _resolve_official_profile(
    profile: str | Path | Mapping[str, Any] | None,
) -> tuple[str | Path | dict[str, Any], Path | None]:
    if isinstance(profile, Mapping):
        return dict(profile), None
    if profile is None:
        path = Path(__file__).resolve().parents[1] / "resources" / "official_schema.json"
    else:
        path = Path(profile).expanduser()
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"official serializer profile does not exist: {resolved}")
    return resolved, resolved


def _perception_config_from_pipeline(pipeline_config: Mapping[str, Any]) -> Any:
    from pcbparse.perception.production_provider import (
        composite_provider_config_from_pipeline,
    )

    return composite_provider_config_from_pipeline(pipeline_config)


def _required_model_tasks_for_perception_config(
    perception_config: Any,
) -> frozenset[str]:
    from pcbparse.perception.production_provider import required_model_tasks_for_config

    return required_model_tasks_for_config(perception_config)


def _load_production_provider(lock_path: Path, perception_config: Any) -> IRProvider:
    from pcbparse.perception.production_provider import load_production_provider

    return load_production_provider(lock_path, config=perception_config)


def _stable_prediction_set_hash(rows: list[dict[str, Any]]) -> str:
    stable = [
        {
            "case_id": row["case_id"],
            "duplicate_group": row["duplicate_group"],
            "image_sha256": row["image_sha256"],
            "ir_sha256": row["ir_sha256"],
            "official_sha256": row["official_sha256"],
            "official_output_sha256": row["official_output_sha256"],
            "provider_kind": row["provider_kind"],
            "model_lock_sha256": row["model_lock_sha256"],
            "config_hash": row["config_hash"],
            "sidecar_access": row["sidecar_access"],
            "status": row["status"],
            "error": row["error"],
        }
        for row in rows
    ]
    return _json_hash(stable)


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


def _canonical_jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) for row in rows)


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _string_list_hash(values: list[str]) -> str:
    return _json_hash(values)


def _path_strings(paths: list[Path]) -> list[str]:
    return [str(path) for path in _unique_paths(paths)]


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    output: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if path not in seen:
            seen.add(path)
            output.append(path)
    return output


def _natural_key(value: object) -> tuple[tuple[int, object], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold(), part)
        for part in re.split(r"(\d+)", str(value))
        if part
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


__all__ = [
    "ProductionEvaluationError",
    "ProductionEvaluationResult",
    "SelectedCase",
    "run_production_evaluation",
]
