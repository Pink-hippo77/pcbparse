#!/usr/bin/env python3
"""Build and verify a controlled current-lock production trace probe.

The ``build`` command publishes a new, immutable probe directory containing
only one deliberately undecodable ``.png`` in its dataset root, a one-case
split manifest, and a source report bound to the current canonical release
controls.  The ``verify`` command never runs inference; it validates a future
production failure run and atomically writes a byte-bound verification report.

Both commands are fail closed.  Existing outputs are never reused or
overwritten, and a directory or report is published only after all staging
checks have completed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence


PROBE_CASE_ID = "release-trace-invalid-png-v1"
PROBE_DUPLICATE_GROUP = "release-trace-invalid-png-v1"
PROBE_SPLIT = "trace"
PROBE_IMAGE_NAME = f"{PROBE_CASE_ID}.png"
# A PNG signature without the mandatory IHDR chunk cannot be decoded by a
# conforming PNG decoder.  Keeping the payload this small also makes accidental
# replacement with a valid image immediately detectable by exact-byte checks.
PROBE_IMAGE_BYTES = b"\x89PNG\r\n\x1a\n"
PROBE_IMAGE_SHA256 = "4c4b6a3be1314ab86138bef4314dde022e600960d8689a2c8f8631802d20dab6"

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_CONTROLS = {
    "models_lock": Path("models/models.lock.json"),
    "pipeline_config": Path("configs/pipeline.yaml"),
    "production_profile": Path("configs/profiles/competition.yaml"),
    "official_profile": Path("src/pcbparse/resources/official_schema.json"),
}
_PROBE_BASE = Path("artifacts/release/trace-probes")
_VERIFICATION_BASE = Path("artifacts/release/trace-verifications")
_SOURCE_REPORT_KEYS = {
    "schema_version",
    "kind",
    "status",
    "created_at_utc",
    "purpose",
    "probe",
    "expected_inference_outcome",
    "controls",
    "outputs",
    "source_report_payload_sha256",
}
_STABLE_PREDICTION_FIELDS = (
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


class TraceProbeError(ValueError):
    """Raised when trace probe evidence is ambiguous or invalid."""


@dataclass(frozen=True, slots=True)
class ControlBindings:
    models_lock: Path
    pipeline_config: Path
    production_profile: Path
    official_profile: Path
    model_lock_sha256: str
    effective_config_sha256: str
    file_sha256: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ValidatedProbe:
    root: Path
    dataset_root: Path
    image_path: Path
    split_manifest: Path
    source_report: Path
    image_sha256: str
    split_manifest_sha256: str
    source_report_sha256: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or verify a fixed current-lock negative trace probe without "
            "executing production inference."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build",
        help="Atomically build a new image-only, one-case trace probe source.",
    )
    _add_workspace_and_control_arguments(build)
    build.add_argument("--output-root", type=Path, required=True)

    verify = subparsers.add_parser(
        "verify",
        help="Validate a future production failure and publish a bound report.",
    )
    _add_workspace_and_control_arguments(verify)
    verify.add_argument("--probe-root", type=Path, required=True)
    verify.add_argument("--inference-summary", type=Path, required=True)
    verify.add_argument("--verification-report", type=Path, required=True)
    return parser


def _add_workspace_and_control_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument(
        "--models-lock", type=Path, default=_CANONICAL_CONTROLS["models_lock"]
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=_CANONICAL_CONTROLS["pipeline_config"],
    )
    parser.add_argument(
        "--production-profile",
        type=Path,
        default=_CANONICAL_CONTROLS["production_profile"],
    )
    parser.add_argument(
        "--official-profile",
        type=Path,
        default=_CANONICAL_CONTROLS["official_profile"],
    )


def build_trace_probe(
    *,
    workspace_root: str | Path,
    output_root: str | Path,
    models_lock: str | Path = _CANONICAL_CONTROLS["models_lock"],
    pipeline_config: str | Path = _CANONICAL_CONTROLS["pipeline_config"],
    production_profile: str | Path = _CANONICAL_CONTROLS["production_profile"],
    official_profile: str | Path = _CANONICAL_CONTROLS["official_profile"],
) -> dict[str, Any]:
    """Atomically publish a new fixed trace-probe source directory."""

    root = _workspace_root(workspace_root)
    contract = _load_hard_gate_contract(root)
    controls = _current_control_bindings(
        root=root,
        contract=contract,
        models_lock=models_lock,
        pipeline_config=pipeline_config,
        production_profile=production_profile,
        official_profile=official_profile,
    )
    destination = _new_child_output(
        root,
        output_root,
        base_relative=_PROBE_BASE,
        label="probe output root",
        require_json=False,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-", dir=str(destination.parent)
        )
    ).resolve(strict=True)
    published = False
    try:
        dataset = staging / "dataset"
        dataset.mkdir()
        image_path = dataset / PROBE_IMAGE_NAME
        _atomic_write_bytes(image_path, PROBE_IMAGE_BYTES)

        manifest_payload = _canonical_jsonl_bytes([_probe_manifest_row()])
        manifest_path = staging / "split.jsonl"
        _atomic_write_bytes(manifest_path, manifest_payload)

        report = _source_report_payload(
            root=root,
            controls=controls,
            manifest_sha256=_sha256_bytes(manifest_payload),
        )
        report["source_report_payload_sha256"] = _json_sha256(report)
        _atomic_write_bytes(staging / "source_report.json", _pretty_json_bytes(report))

        _validate_probe_source(staging, root=root, controls=controls)
        refreshed_controls = _current_control_bindings(
            root=root,
            contract=contract,
            models_lock=controls.models_lock,
            pipeline_config=controls.pipeline_config,
            production_profile=controls.production_profile,
            official_profile=controls.official_profile,
        )
        if refreshed_controls != controls:
            raise TraceProbeError("canonical controls changed while building the probe")
        if destination.exists():
            raise TraceProbeError(
                f"probe output appeared during staging; refusing overwrite: {destination}"
            )
        os.rename(staging, destination)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)

    validated = _validate_probe_source(destination, root=root, controls=controls)
    return {
        "ok": True,
        "status": "ready",
        "probe_root": str(destination),
        "dataset_root": str(validated.dataset_root),
        "image_path": str(validated.image_path),
        "image_sha256": validated.image_sha256,
        "split_manifest": str(validated.split_manifest),
        "split_manifest_sha256": validated.split_manifest_sha256,
        "source_report": str(validated.source_report),
        "source_report_sha256": validated.source_report_sha256,
        "model_lock_sha256": controls.model_lock_sha256,
        "config_sha256": controls.effective_config_sha256,
    }


def verify_trace_probe_result(
    *,
    workspace_root: str | Path,
    probe_root: str | Path,
    inference_summary: str | Path,
    verification_report: str | Path,
    models_lock: str | Path = _CANONICAL_CONTROLS["models_lock"],
    pipeline_config: str | Path = _CANONICAL_CONTROLS["pipeline_config"],
    production_profile: str | Path = _CANONICAL_CONTROLS["production_profile"],
    official_profile: str | Path = _CANONICAL_CONTROLS["official_profile"],
) -> dict[str, Any]:
    """Validate one formal controlled failure and publish its byte bindings."""

    root = _workspace_root(workspace_root)
    contract = _load_hard_gate_contract(root)
    controls = _current_control_bindings(
        root=root,
        contract=contract,
        models_lock=models_lock,
        pipeline_config=pipeline_config,
        production_profile=production_profile,
        official_profile=official_profile,
    )
    probe_path = _workspace_directory(root, probe_root, "probe root")
    required_probe_base = (root / _PROBE_BASE).resolve(strict=False)
    if probe_path == required_probe_base or not _contains(required_probe_base, probe_path):
        raise TraceProbeError(
            f"probe root must be a child of {_PROBE_BASE.as_posix()}: {probe_path}"
        )
    probe = _validate_probe_source(probe_path, root=root, controls=controls)

    summary_path = _workspace_file(root, inference_summary, "inference summary")
    _validate_summary_location(root, summary_path)
    summary_bytes = summary_path.read_bytes()
    summary = _decode_json_object(
        summary_bytes,
        source=str(summary_path),
        label="inference summary",
    )
    predictions_path = summary_path.parent / "predictions.jsonl"
    if not predictions_path.is_file() or predictions_path.is_symlink():
        raise TraceProbeError(
            "trace inference predictions.jsonl is missing or is a symlink"
        )
    prediction_bytes = predictions_path.read_bytes()

    audited = contract._audit_production_summary_binding(  # type: ignore[attr-defined]
        root,
        summary_path,
        controls.model_lock_sha256,
        controls.effective_config_sha256,
        require_success=False,
        require_embedded_time=True,
    )
    if audited.get("valid") is not True:
        raise TraceProbeError(
            "inference summary fails formal production binding: "
            + "; ".join(str(item) for item in audited.get("problems", []))
        )
    if summary_path.read_bytes() != summary_bytes:
        raise TraceProbeError("inference summary changed during formal validation")
    if predictions_path.read_bytes() != prediction_bytes:
        raise TraceProbeError("prediction manifest changed during formal validation")
    _validate_failure_summary(
        summary,
        root=root,
        summary_path=summary_path,
        probe=probe,
        controls=controls,
    )

    rows = _decode_jsonl(
        prediction_bytes,
        source=str(predictions_path),
    )
    if len(rows) != 1:
        raise TraceProbeError(
            f"trace prediction manifest must contain exactly one row, got {len(rows)}"
        )
    if prediction_bytes != _canonical_jsonl_bytes(rows):
        raise TraceProbeError(
            "trace prediction manifest is not canonical producer JSONL bytes"
        )
    row = rows[0]
    error_id = _validate_failure_row(
        row,
        root=root,
        summary=summary,
        probe=probe,
        controls=controls,
        contract=contract,
    )
    _validate_run_tree(summary_path.parent)

    summary_sha256 = _sha256_bytes(summary_bytes)
    predictions_sha256 = _sha256_bytes(prediction_bytes)
    if summary.get("prediction_manifest_sha256") != predictions_sha256:
        raise TraceProbeError(
            "summary prediction_manifest_sha256 does not match prediction bytes"
        )

    refreshed_controls = _current_control_bindings(
        root=root,
        contract=contract,
        models_lock=controls.models_lock,
        pipeline_config=controls.pipeline_config,
        production_profile=controls.production_profile,
        official_profile=controls.official_profile,
    )
    if refreshed_controls != controls:
        raise TraceProbeError("canonical controls changed during trace validation")
    refreshed_probe = _validate_probe_source(probe_path, root=root, controls=controls)
    if refreshed_probe != probe:
        raise TraceProbeError("probe source changed during trace validation")
    if summary_path.read_bytes() != summary_bytes:
        raise TraceProbeError("inference summary changed before report publication")
    if predictions_path.read_bytes() != prediction_bytes:
        raise TraceProbeError("prediction manifest changed before report publication")

    report_path = _new_child_output(
        root,
        verification_report,
        base_relative=_VERIFICATION_BASE,
        label="verification report",
        require_json=True,
    )
    report_payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pcbparse_controlled_current_lock_trace_probe_verification_v1",
        "status": "pass",
        "formal_production_binding": True,
        "current_lock_inference_executed": True,
        "generated_at_utc": _utc_now(),
        "probe_source": {
            "root": _relative(root, probe.root),
            "dataset_root": _relative(root, probe.dataset_root),
            "image_path": _relative(root, probe.image_path),
            "image_sha256": probe.image_sha256,
            "split_manifest_path": _relative(root, probe.split_manifest),
            "split_manifest_sha256": probe.split_manifest_sha256,
            "source_report_path": _relative(root, probe.source_report),
            "source_report_sha256": probe.source_report_sha256,
        },
        "inference_evidence": {
            "summary_path": _relative(root, summary_path),
            "summary_sha256": summary_sha256,
            "predictions_path": _relative(root, predictions_path),
            "predictions_sha256": predictions_sha256,
            "prediction_set_sha256": summary["prediction_set_sha256"],
            "inference_run_id": summary["inference_run_id"],
            "selected_cases": 1,
            "succeeded": 0,
            "failed": 1,
            "not_run": 0,
            "error_id": error_id,
            "error_type": row["error"]["type"],
            "error_stage": row["error"]["stage"],
            "timing": summary["timing"],
        },
        "controls": _control_report(root, controls),
        "bindings": {
            "fixed_probe_bytes": True,
            "image_sha_matches_split_and_prediction": True,
            "source_report_matches_current_controls": True,
            "summary_and_predictions_bytes_bound": True,
            "stable_error_id_recomputed": True,
            "formal_timing_contract_validated": True,
        },
    }
    report_payload["verification_report_payload_sha256"] = _json_sha256(
        report_payload
    )
    _atomic_publish_new_file(report_path, _pretty_json_bytes(report_payload))
    return {
        "ok": True,
        "status": "pass",
        "verification_report": str(report_path),
        "verification_report_sha256": _sha256_file(report_path),
        "summary_sha256": summary_sha256,
        "predictions_sha256": predictions_sha256,
        "image_sha256": probe.image_sha256,
        "split_manifest_sha256": probe.split_manifest_sha256,
        "source_report_sha256": probe.source_report_sha256,
        "model_lock_sha256": controls.model_lock_sha256,
        "config_sha256": controls.effective_config_sha256,
        "error_id": error_id,
    }


def _probe_manifest_row() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sample_id": PROBE_CASE_ID,
        "duplicate_group": PROBE_DUPLICATE_GROUP,
        "duplicate_group_size": 1,
        "split": PROBE_SPLIT,
        "files": {
            "image": {
                "path": PROBE_IMAGE_NAME,
                "sha256": PROBE_IMAGE_SHA256,
            }
        },
    }


def _source_report_payload(
    *,
    root: Path,
    controls: ControlBindings,
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "pcbparse_controlled_current_lock_trace_probe_source_v1",
        "status": "ready",
        "created_at_utc": _utc_now(),
        "purpose": "release_traceability_negative_probe",
        "probe": {
            "case_id": PROBE_CASE_ID,
            "duplicate_group": PROBE_DUPLICATE_GROUP,
            "split": PROBE_SPLIT,
            "image_path": f"dataset/{PROBE_IMAGE_NAME}",
            "image_sha256": PROBE_IMAGE_SHA256,
            "byte_length": len(PROBE_IMAGE_BYTES),
            "decoder_contract": "png_signature_only_without_mandatory_ihdr_v1",
            "expected_decode_result": "error",
        },
        "expected_inference_outcome": {
            "selected_cases": 1,
            "succeeded": 0,
            "failed": 1,
            "not_run": 0,
            "prediction_status": "error",
            "expected_process_exit_code": 1,
        },
        "controls": _control_report(root, controls),
        "outputs": {
            "dataset_root": "dataset",
            "only_image_files_materialized": True,
            "image_count": 1,
            "split_manifest": {
                "path": "split.jsonl",
                "sha256": manifest_sha256,
                "row_count": 1,
            },
            "source_report": "source_report.json",
        },
    }


def _validate_probe_source(
    probe_root: Path,
    *,
    root: Path,
    controls: ControlBindings,
) -> ValidatedProbe:
    probe_root = probe_root.resolve(strict=True)
    if not probe_root.is_dir() or probe_root.is_symlink():
        raise TraceProbeError(f"probe root is not a regular directory: {probe_root}")
    expected_root_entries = {"dataset", "split.jsonl", "source_report.json"}
    actual_root_entries = {path.name for path in probe_root.iterdir()}
    if actual_root_entries != expected_root_entries:
        raise TraceProbeError(
            "probe root contains unexpected or missing entries: "
            f"expected={sorted(expected_root_entries)}, actual={sorted(actual_root_entries)}"
        )

    dataset = probe_root / "dataset"
    image = dataset / PROBE_IMAGE_NAME
    if not dataset.is_dir() or dataset.is_symlink():
        raise TraceProbeError("probe dataset root is missing or is a symlink")
    dataset_entries = list(dataset.iterdir())
    if dataset_entries != [image] or not image.is_file() or image.is_symlink():
        raise TraceProbeError(
            "probe dataset root must contain exactly the fixed regular .png file"
        )
    image_bytes = image.read_bytes()
    if image_bytes != PROBE_IMAGE_BYTES:
        raise TraceProbeError("probe .png bytes differ from the fixed undecodable payload")
    if _sha256_bytes(image_bytes) != PROBE_IMAGE_SHA256:
        raise TraceProbeError("internal fixed probe SHA-256 contract is inconsistent")

    manifest = probe_root / "split.jsonl"
    if not manifest.is_file() or manifest.is_symlink():
        raise TraceProbeError("probe split manifest is missing or is a symlink")
    expected_manifest_bytes = _canonical_jsonl_bytes([_probe_manifest_row()])
    manifest_bytes = manifest.read_bytes()
    if manifest_bytes != expected_manifest_bytes:
        raise TraceProbeError("probe split manifest differs from its fixed canonical bytes")
    manifest_sha256 = _sha256_bytes(manifest_bytes)

    report_path = probe_root / "source_report.json"
    if not report_path.is_file() or report_path.is_symlink():
        raise TraceProbeError("probe source report is missing or is a symlink")
    report = _read_json_object(report_path, "probe source report")
    if set(report) != _SOURCE_REPORT_KEYS:
        raise TraceProbeError(
            "probe source report has unexpected or missing top-level fields"
        )
    declared_payload_sha = report.get("source_report_payload_sha256")
    report_body = dict(report)
    report_body.pop("source_report_payload_sha256", None)
    if declared_payload_sha != _json_sha256(report_body):
        raise TraceProbeError("probe source report payload SHA-256 is invalid")
    _parse_utc(report.get("created_at_utc"), "probe source report created_at_utc")

    expected = _source_report_payload(
        root=root,
        controls=controls,
        manifest_sha256=manifest_sha256,
    )
    expected.pop("created_at_utc")
    actual = dict(report_body)
    actual.pop("created_at_utc")
    if actual != expected:
        raise TraceProbeError(
            "probe source report is not bound to the fixed source and current controls"
        )
    return ValidatedProbe(
        root=probe_root,
        dataset_root=dataset,
        image_path=image,
        split_manifest=manifest,
        source_report=report_path,
        image_sha256=PROBE_IMAGE_SHA256,
        split_manifest_sha256=manifest_sha256,
        source_report_sha256=_sha256_file(report_path),
    )


def _validate_failure_summary(
    summary: Mapping[str, Any],
    *,
    root: Path,
    summary_path: Path,
    probe: ValidatedProbe,
    controls: ControlBindings,
) -> None:
    exact_checks = {
        "schema_version": summary.get("schema_version") == 2,
        "kind": summary.get("kind") == "production-image-only-inference",
        "accuracy_mode": summary.get("accuracy_mode") == "image-only",
        "execution_mode": summary.get("execution_mode") == "production",
        "provider_kind": summary.get("provider_kind") == "production",
        "sidecar_access": summary.get("sidecar_access") is False,
        "status": summary.get("status") == "failed",
        "all_successful": summary.get("all_successful") is False,
        "split": summary.get("split") == PROBE_SPLIT,
        "selected_cases": type(summary.get("selected_cases")) is int
        and summary.get("selected_cases") == 1,
        "selected_duplicate_groups": type(summary.get("selected_duplicate_groups"))
        is int
        and summary.get("selected_duplicate_groups") == 1,
        "attempted": type(summary.get("attempted")) is int
        and summary.get("attempted") == 1,
        "succeeded": type(summary.get("succeeded")) is int
        and summary.get("succeeded") == 0,
        "failed": type(summary.get("failed")) is int
        and summary.get("failed") == 1,
        "not_run": type(summary.get("not_run")) is int
        and summary.get("not_run") == 0,
        "model_lock_sha256": summary.get("model_lock_sha256")
        == controls.model_lock_sha256,
        "config_hash": summary.get("config_hash")
        == controls.effective_config_sha256,
        "split_manifest_sha256": summary.get("split_manifest_sha256")
        == probe.split_manifest_sha256,
        "prediction_manifest": summary.get("prediction_manifest")
        == "predictions.jsonl",
    }
    failed = [name for name, passed in exact_checks.items() if not passed]
    if failed:
        raise TraceProbeError(f"trace inference summary fails exact checks: {failed}")

    controls_payload = summary.get("control_inputs")
    if not isinstance(controls_payload, Mapping):
        raise TraceProbeError("trace inference summary lacks control_inputs")
    expected_source_controls = {
        "dataset_root": probe.dataset_root,
        "split_manifest": probe.split_manifest,
    }
    for name, expected in expected_source_controls.items():
        actual = _workspace_existing_path(
            root=root,
            value=controls_payload.get(name),
            label=f"trace inference control_inputs.{name}",
        )
        if actual != expected:
            raise TraceProbeError(
                f"trace inference control_inputs.{name} is not the fixed probe source"
            )

    coverage = summary.get("coverage")
    if not isinstance(coverage, Mapping):
        raise TraceProbeError("trace inference summary lacks coverage")
    case_ids = [PROBE_CASE_ID]
    membership = {PROBE_DUPLICATE_GROUP: [PROBE_CASE_ID]}
    coverage_checks = {
        "expected_case_ids_sha256": coverage.get("expected_case_ids_sha256")
        == _json_sha256(case_ids),
        "manifest_case_ids_sha256": coverage.get("manifest_case_ids_sha256")
        == _json_sha256(case_ids),
        "expected_duplicate_group_membership_sha256": coverage.get(
            "expected_duplicate_group_membership_sha256"
        )
        == _json_sha256(membership),
        "exact_case_coverage": coverage.get("exact_case_coverage") is True,
        "exact_duplicate_group_coverage": coverage.get(
            "exact_duplicate_group_coverage"
        )
        is True,
        "duplicate_case_ids": coverage.get("duplicate_case_ids") == [],
    }
    failed_coverage = [
        name for name, passed in coverage_checks.items() if not passed
    ]
    if failed_coverage:
        raise TraceProbeError(
            f"trace inference summary fails coverage checks: {failed_coverage}"
        )
    if not _is_sha256(summary.get("inference_run_id")):
        raise TraceProbeError("trace inference summary has no valid inference_run_id")
    if not _is_sha256(summary.get("prediction_set_sha256")):
        raise TraceProbeError("trace inference summary has no prediction_set_sha256")
    if summary_path.parent / "predictions.jsonl" != (
        summary_path.parent / str(summary["prediction_manifest"])
    ).resolve(strict=False):
        raise TraceProbeError("trace prediction manifest is not canonical")


def _validate_failure_row(
    row: Mapping[str, Any],
    *,
    root: Path,
    summary: Mapping[str, Any],
    probe: ValidatedProbe,
    controls: ControlBindings,
    contract: ModuleType,
) -> str:
    checks = {
        "case_id": row.get("case_id") == PROBE_CASE_ID,
        "duplicate_group": row.get("duplicate_group") == PROBE_DUPLICATE_GROUP,
        "image_sha256": row.get("image_sha256") == probe.image_sha256,
        "ir_path": row.get("ir_path") is None,
        "ir_sha256": row.get("ir_sha256") is None,
        "official_path": row.get("official_path") is None,
        "official_sha256": row.get("official_sha256") is None,
        "official_output_sha256": row.get("official_output_sha256") is None,
        "provider_kind": row.get("provider_kind") == "production",
        "model_lock_sha256": row.get("model_lock_sha256")
        == controls.model_lock_sha256,
        "config_hash": row.get("config_hash")
        == controls.effective_config_sha256,
        "inference_run_id": row.get("inference_run_id")
        == summary.get("inference_run_id"),
        "sidecar_access": row.get("sidecar_access") is False,
        "status": row.get("status") == "error",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise TraceProbeError(f"trace prediction row fails exact checks: {failed}")

    inputs = row.get("input_paths_read")
    if not isinstance(inputs, list) or not inputs:
        raise TraceProbeError("trace prediction row lacks input_paths_read")
    image_input = _workspace_file(
        root, inputs[0], "trace prediction image input"
    )
    if image_input != probe.image_path:
        raise TraceProbeError("trace prediction row does not bind the fixed probe image")
    elapsed = row.get("elapsed_seconds")
    if (
        not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise TraceProbeError("trace prediction elapsed_seconds is invalid")

    error = row.get("error")
    if not isinstance(error, Mapping):
        raise TraceProbeError("trace prediction row lacks a structured error")
    for field in ("type", "message", "stage"):
        value = error.get(field)
        if not isinstance(value, str) or not value.strip():
            raise TraceProbeError(f"trace prediction error.{field} must be non-empty")
    _validate_diagnostics(error)
    traceable, problems = contract._is_traceable_error(row)  # type: ignore[attr-defined]
    if traceable is not True:
        raise TraceProbeError(
            "trace prediction error is not accepted by the hard-gate contract: "
            + "; ".join(str(item) for item in problems)
        )
    error_without_id = dict(error)
    error_id = error_without_id.pop("error_id", None)
    expected_error_id = _json_sha256(
        {
            "case_id": row.get("case_id"),
            "image_sha256": row.get("image_sha256"),
            "config_hash": row.get("config_hash"),
            "model_lock_sha256": row.get("model_lock_sha256"),
            "error": error_without_id,
        }
    )
    if error_id != expected_error_id:
        raise TraceProbeError("trace prediction error_id cannot be stably recomputed")
    return expected_error_id


def _validate_diagnostics(error: Mapping[str, Any]) -> None:
    count = error.get("diagnostic_count")
    codes = error.get("diagnostic_codes")
    diagnostics = error.get("diagnostics")
    if type(count) is not int or count < 0:
        raise TraceProbeError("trace prediction diagnostic_count is invalid")
    if not isinstance(codes, Mapping):
        raise TraceProbeError("trace prediction diagnostic_codes is invalid")
    if not isinstance(diagnostics, list) or len(diagnostics) != count:
        raise TraceProbeError("trace prediction diagnostics do not match diagnostic_count")
    for index, item in enumerate(diagnostics):
        if not isinstance(item, Mapping):
            raise TraceProbeError(f"trace prediction diagnostic {index} is not an object")
        if not isinstance(item.get("object_ids"), list):
            raise TraceProbeError(
                f"trace prediction diagnostic {index} object_ids is not a list"
            )
    actual_codes = dict(
        sorted(Counter(str(item.get("code", "UNKNOWN")) for item in diagnostics).items())
    )
    if dict(codes) != actual_codes:
        raise TraceProbeError("trace prediction diagnostic_codes cannot be recomputed")
    if sum(
        value
        for value in codes.values()
        if type(value) is int and value >= 0
    ) != count or any(type(value) is not int or value <= 0 for value in codes.values()):
        if count != 0 or dict(codes):
            raise TraceProbeError("trace prediction diagnostic_codes counts are invalid")


def _validate_run_tree(run_root: Path) -> None:
    expected = {"summary.json", "predictions.jsonl", "official", "circuit_ir"}
    actual = {path.name for path in run_root.iterdir()}
    if actual != expected:
        raise TraceProbeError(
            "trace inference run contains unexpected or missing entries: "
            f"expected={sorted(expected)}, actual={sorted(actual)}"
        )
    for directory_name in ("official", "circuit_ir"):
        directory = run_root / directory_name
        if not directory.is_dir() or directory.is_symlink() or any(directory.iterdir()):
            raise TraceProbeError(
                f"trace inference {directory_name} directory must exist and be empty"
            )
    for name in ("summary.json", "predictions.jsonl"):
        path = run_root / name
        if not path.is_file() or path.is_symlink():
            raise TraceProbeError(f"trace inference {name} is missing or is a symlink")


def _validate_summary_location(root: Path, summary: Path) -> None:
    work = (root / "work").resolve(strict=False)
    if summary.name != "summary.json":
        raise TraceProbeError("inference summary must be named summary.json")
    if summary.parent.parent.resolve(strict=False) != work:
        raise TraceProbeError(
            "trace inference summary must be directly below workspace work/"
        )
    if not summary.parent.name.startswith("production-e2e-"):
        raise TraceProbeError(
            "trace inference directory must match work/production-e2e-*"
        )


def _current_control_bindings(
    *,
    root: Path,
    contract: ModuleType,
    models_lock: str | Path,
    pipeline_config: str | Path,
    production_profile: str | Path,
    official_profile: str | Path,
) -> ControlBindings:
    paths = {
        "models_lock": _canonical_control(
            root, models_lock, _CANONICAL_CONTROLS["models_lock"]
        ),
        "pipeline_config": _canonical_control(
            root, pipeline_config, _CANONICAL_CONTROLS["pipeline_config"]
        ),
        "production_profile": _canonical_control(
            root, production_profile, _CANONICAL_CONTROLS["production_profile"]
        ),
        "official_profile": _canonical_control(
            root, official_profile, _CANONICAL_CONTROLS["official_profile"]
        ),
    }
    _validate_model_lock(paths["models_lock"])
    binding = contract._production_config_binding(  # type: ignore[attr-defined]
        root, paths["pipeline_config"], paths["production_profile"]
    )
    if binding.get("status") != "valid":
        raise TraceProbeError(
            "current production config/profile binding is invalid: "
            + "; ".join(str(item) for item in binding.get("problems", []))
        )
    config_sha = binding.get("effective_config_sha256")
    if not _is_sha256(config_sha):
        raise TraceProbeError("current effective production config SHA-256 is unavailable")
    file_sha256 = {name: _sha256_file(path) for name, path in paths.items()}
    return ControlBindings(
        models_lock=paths["models_lock"],
        pipeline_config=paths["pipeline_config"],
        production_profile=paths["production_profile"],
        official_profile=paths["official_profile"],
        model_lock_sha256=file_sha256["models_lock"],
        effective_config_sha256=config_sha,
        file_sha256=file_sha256,
    )


def _validate_model_lock(lock_path: Path) -> None:
    payload = _read_json_object(lock_path, "models.lock")
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise TraceProbeError("models.lock must contain at least one model")
    seen_tasks: set[str] = set()
    seen_paths: set[str] = set()
    model_root = lock_path.parent.resolve()
    for index, item in enumerate(models):
        if not isinstance(item, Mapping):
            raise TraceProbeError(f"models.lock entry {index} is not an object")
        task = item.get("task")
        raw_path = item.get("path")
        declared_sha = item.get("sha256")
        if not isinstance(task, str) or not task.strip() or task in seen_tasks:
            raise TraceProbeError(f"models.lock entry {index} has invalid/duplicate task")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise TraceProbeError(f"models.lock entry {index} has no artifact path")
        if not _is_sha256(declared_sha):
            raise TraceProbeError(f"models.lock entry {index} has invalid SHA-256")
        artifact = (model_root / raw_path).resolve(strict=False)
        if not _contains(model_root, artifact) or artifact == model_root:
            raise TraceProbeError(f"models.lock entry {index} artifact escapes models root")
        normalized = str(artifact).casefold()
        if normalized in seen_paths:
            raise TraceProbeError(f"models.lock entry {index} reuses an artifact path")
        if not artifact.is_file() or artifact.is_symlink():
            raise TraceProbeError(f"models.lock entry {index} artifact is missing or a symlink")
        if _sha256_file(artifact) != declared_sha:
            raise TraceProbeError(f"models.lock entry {index} artifact SHA-256 mismatch")
        seen_tasks.add(task)
        seen_paths.add(normalized)


def _control_report(root: Path, controls: ControlBindings) -> dict[str, Any]:
    return {
        "models_lock": {
            "path": _relative(root, controls.models_lock),
            "sha256": controls.model_lock_sha256,
        },
        "pipeline_config": {
            "path": _relative(root, controls.pipeline_config),
            "sha256": controls.file_sha256["pipeline_config"],
        },
        "production_profile": {
            "path": _relative(root, controls.production_profile),
            "sha256": controls.file_sha256["production_profile"],
        },
        "official_profile": {
            "path": _relative(root, controls.official_profile),
            "sha256": controls.file_sha256["official_profile"],
        },
        "effective_config_sha256": controls.effective_config_sha256,
    }


def _load_hard_gate_contract(root: Path) -> ModuleType:
    script = root / "scripts" / "audit_release_hard_gates.py"
    if not script.is_file():
        raise TraceProbeError(f"release hard-gate contract is missing: {script}")
    module_name = "_pcbparse_release_trace_probe_hard_gate_contract"
    specification = importlib.util.spec_from_file_location(module_name, script)
    if specification is None or specification.loader is None:
        raise TraceProbeError(f"cannot load release hard-gate contract: {script}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def _workspace_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise TraceProbeError(f"workspace root is not a directory: {root}")
    return root


def _canonical_control(root: Path, value: str | Path, relative: Path) -> Path:
    actual = _workspace_file(root, value, f"canonical control {relative.as_posix()}")
    expected = (root / relative).resolve(strict=False)
    if actual != expected:
        raise TraceProbeError(
            f"formal trace evidence requires canonical control {expected}, got {actual}"
        )
    return actual


def _new_child_output(
    root: Path,
    value: str | Path,
    *,
    base_relative: Path,
    label: str,
    require_json: bool,
) -> Path:
    output = _workspace_path(root, value, label)
    base = (root / base_relative).resolve(strict=False)
    if output.parent != base:
        raise TraceProbeError(
            f"{label} must be a new direct child of {base_relative.as_posix()}"
        )
    if require_json and output.suffix.casefold() != ".json":
        raise TraceProbeError(f"{label} must have a .json suffix")
    if output.exists():
        raise TraceProbeError(f"refusing to reuse or overwrite existing {label}: {output}")
    return output


def _workspace_path(root: Path, value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    resolved = (path if path.is_absolute() else root / path).resolve(strict=False)
    if not _contains(root, resolved):
        raise TraceProbeError(f"{label} escapes workspace: {resolved}")
    return resolved


def _workspace_existing_path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise TraceProbeError(f"{label} is missing")
    resolved = _workspace_path(root, value, label)
    if not resolved.exists():
        raise TraceProbeError(f"{label} does not exist: {resolved}")
    return resolved


def _workspace_file(root: Path, value: object, label: str) -> Path:
    resolved = _workspace_existing_path(root, value, label)
    if not resolved.is_file() or resolved.is_symlink():
        raise TraceProbeError(f"{label} is not a regular file: {resolved}")
    return resolved


def _workspace_directory(root: Path, value: object, label: str) -> Path:
    resolved = _workspace_existing_path(root, value, label)
    if not resolved.is_dir() or resolved.is_symlink():
        raise TraceProbeError(f"{label} is not a regular directory: {resolved}")
    return resolved


def _contains(parent: Path, candidate: Path) -> bool:
    try:
        candidate.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def _relative(root: Path, path: Path) -> str:
    return path.resolve(strict=False).relative_to(root.resolve()).as_posix()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise TraceProbeError(f"cannot read {label} at {path}: {exc}") from exc
    return _decode_json_object(payload, source=str(path), label=label)


def _read_json(path: Path) -> Any:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise TraceProbeError(f"cannot read JSON at {path}: {exc}") from exc
    try:
        return _decode_json(payload, source=str(path))
    except (UnicodeError, json.JSONDecodeError, TraceProbeError) as exc:
        raise TraceProbeError(f"invalid JSON at {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise TraceProbeError(f"cannot read JSONL at {path}: {exc}") from exc
    return _decode_jsonl(payload, source=str(path))


def _decode_json_object(
    payload: bytes,
    *,
    source: str,
    label: str,
) -> dict[str, Any]:
    try:
        value = _decode_json(payload, source=source)
    except (UnicodeError, json.JSONDecodeError, TraceProbeError) as exc:
        raise TraceProbeError(f"invalid {label} at {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise TraceProbeError(f"{label} root is not an object: {source}")
    return value


def _decode_json(payload: bytes, *, source: str) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise TraceProbeError(f"JSON is not UTF-8 at {source}: {exc}") from exc
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def _decode_jsonl(payload: bytes, *, source: str) -> list[dict[str, Any]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise TraceProbeError(f"JSONL is not UTF-8 at {source}: {exc}") from exc
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
        except (json.JSONDecodeError, TraceProbeError) as exc:
            raise TraceProbeError(
                f"invalid JSONL at {source}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise TraceProbeError(
                f"JSONL row at {source}:{line_number} is not an object"
            )
        rows.append(value)
    if not rows:
        raise TraceProbeError(f"JSONL is empty: {source}")
    return rows


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise TraceProbeError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise TraceProbeError(f"non-standard JSON constant {value!r}")


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


def _canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) for row in rows)


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
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _HEX_SHA256.fullmatch(value) is not None


def _parse_utc(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise TraceProbeError(f"{field} must be a non-empty timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TraceProbeError(f"{field} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TraceProbeError(f"{field} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        raise TraceProbeError(f"refusing to overwrite staged path: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise TraceProbeError(f"output appeared during staging: {path}")
        os.rename(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_publish_new_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise TraceProbeError(f"refusing to overwrite verification report: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise TraceProbeError(
                f"verification report appeared during staging: {path}"
            )
        os.rename(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    common = {
        "workspace_root": args.workspace_root,
        "models_lock": args.models_lock,
        "pipeline_config": args.pipeline_config,
        "production_profile": args.production_profile,
        "official_profile": args.official_profile,
    }
    if args.command == "build":
        result = build_trace_probe(output_root=args.output_root, **common)
        print(result["probe_root"])
        print(f"image_sha256={result['image_sha256']}")
        print(f"split_manifest_sha256={result['split_manifest_sha256']}")
        print(f"source_report_sha256={result['source_report_sha256']}")
        return 0
    result = verify_trace_probe_result(
        probe_root=args.probe_root,
        inference_summary=args.inference_summary,
        verification_report=args.verification_report,
        **common,
    )
    print(result["verification_report"])
    print(f"verification_report_sha256={result['verification_report_sha256']}")
    print(f"error_id={result['error_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
