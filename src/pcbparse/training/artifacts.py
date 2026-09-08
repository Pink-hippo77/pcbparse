from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import MODEL_TASKS, MetricGate, ModelTask


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ENTRYPOINT = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$"
)
_SUPPORTED_PRECISIONS = frozenset({"fp16", "fp32", "bf16", "int8", "uint8", "mixed"})


class CandidateBundleError(RuntimeError):
    """A candidate artifact lacks independently verifiable evidence."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MetricGateResult(_StrictModel):
    metric: str = Field(min_length=1)
    operator: Literal["gte", "lte"]
    threshold: float
    value: float
    passed: bool

    @model_validator(mode="after")
    def validate_comparison(self) -> "MetricGateResult":
        if not math.isfinite(self.threshold) or not math.isfinite(self.value):
            raise ValueError("metric gate threshold and value must be finite")
        calculated = (
            self.value >= self.threshold
            if self.operator == "gte"
            else self.value <= self.threshold
        )
        if self.passed is not calculated:
            raise ValueError(
                f"gate {self.metric!r} passed={self.passed} contradicts its numeric comparison"
            )
        if not calculated:
            raise ValueError(f"metric gate {self.metric!r} did not pass")
        return self


class ExportConsistency(_StrictModel):
    reference_backend: str = Field(min_length=1)
    export_backend: str = Field(min_length=1)
    sample_count: int = Field(gt=0)
    minimum_samples: int = Field(gt=0)
    max_abs_error: float = Field(ge=0)
    tolerance: float = Field(ge=0)
    input_manifest_sha256: str
    passed: Literal[True]

    @model_validator(mode="after")
    def validate_consistency(self) -> "ExportConsistency":
        if not _SHA256.fullmatch(self.input_manifest_sha256):
            raise ValueError("input_manifest_sha256 must be a lowercase SHA-256 digest")
        if not math.isfinite(self.max_abs_error) or not math.isfinite(self.tolerance):
            raise ValueError("export error and tolerance must be finite")
        if self.sample_count < self.minimum_samples:
            raise ValueError("export consistency sample_count is below minimum_samples")
        if self.max_abs_error > self.tolerance:
            raise ValueError("export consistency error exceeds tolerance")
        return self


class EvaluationEvidence(_StrictModel):
    evaluator: str = Field(min_length=1)
    evaluator_version: str = Field(min_length=1)
    split: Literal["val", "test"]
    dataset_config_hash: str
    split_manifest_sha256: str
    predictions_sha256: str
    metrics: dict[str, float] = Field(min_length=1)
    gates: tuple[MetricGateResult, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_metrics(self) -> "EvaluationEvidence":
        for field_name in (
            "dataset_config_hash",
            "split_manifest_sha256",
            "predictions_sha256",
        ):
            if not _SHA256.fullmatch(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in self.metrics.values()):
            raise ValueError("evaluation metrics must be finite numbers")
        gate_keys = [(gate.metric, gate.operator) for gate in self.gates]
        if len(gate_keys) != len(set(gate_keys)):
            raise ValueError(
                "evaluation metric/operator gate pairs must be unique"
            )
        for gate in self.gates:
            if gate.metric not in self.metrics:
                raise ValueError(f"gate metric {gate.metric!r} is missing from metrics")
            if float(self.metrics[gate.metric]) != gate.value:
                raise ValueError(f"gate metric {gate.metric!r} value differs from metrics")
        return self


class AdapterImportSmoke(_StrictModel):
    entrypoint: str
    module: str = Field(min_length=1)
    attribute: str = Field(min_length=1)
    import_passed: Literal[True]
    callable_passed: Literal[True]
    tested_at_utc: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_entrypoint(self) -> "AdapterImportSmoke":
        if not _ENTRYPOINT.fullmatch(self.entrypoint):
            raise ValueError("adapter entrypoint must be package.module:callable")
        if self.entrypoint != f"{self.module}:{self.attribute}":
            raise ValueError("adapter module/attribute do not reproduce entrypoint")
        return self


class EvidenceAttachment(_StrictModel):
    """One immutable evidence file carried by a candidate bundle."""

    kind: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str

    @model_validator(mode="after")
    def validate_attachment(self) -> "EvidenceAttachment":
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("evidence attachment sha256 must be a lowercase SHA-256 digest")
        path = Path(self.path)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("evidence attachment path must be bundle-relative")
        return self


class GatePolicy(_StrictModel):
    """Thresholds copied from the hashed training configuration."""

    task_manifest_sha256: str
    export_min_samples: int = Field(gt=0)
    export_max_abs_error: float = Field(ge=0)
    evaluation_gates: tuple[MetricGate, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_policy(self) -> "GatePolicy":
        if not _SHA256.fullmatch(self.task_manifest_sha256):
            raise ValueError("task_manifest_sha256 must be a lowercase SHA-256 digest")
        if not math.isfinite(self.export_max_abs_error):
            raise ValueError("export_max_abs_error must be finite")
        gate_keys = [
            (gate.metric, gate.operator) for gate in self.evaluation_gates
        ]
        if len(gate_keys) != len(set(gate_keys)):
            raise ValueError(
                "configured evaluation metric/operator gate pairs must be unique"
            )
        return self


class ModelCard(_StrictModel):
    schema_version: Literal[1]
    task: ModelTask
    version: str = Field(min_length=1)
    license: str = Field(min_length=1)
    created_at_utc: str = Field(min_length=1)
    intended_use: str = Field(min_length=1)
    out_of_scope_use: tuple[str, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)
    training_config_sha256: str
    dataset_config_hash: str
    source_manifest_sha256: str
    split_manifest_sha256: str
    artifact_sha256: str
    artifact_bytes: int = Field(gt=0)
    evaluation_predictions_path: str | None = None
    inference: dict[str, Any] | None = None
    evidence_attachments: tuple[EvidenceAttachment, ...] = ()
    gate_policy: GatePolicy
    evaluation: EvaluationEvidence
    export_consistency: ExportConsistency

    @model_validator(mode="after")
    def validate_hashes(self) -> "ModelCard":
        for field_name in (
            "training_config_sha256",
            "dataset_config_hash",
            "source_manifest_sha256",
            "split_manifest_sha256",
            "artifact_sha256",
        ):
            if not _SHA256.fullmatch(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        _validate_attachment_uniqueness(self.evidence_attachments)
        return self


class CandidateBundleManifest(_StrictModel):
    schema_version: Literal[1]
    status: Literal["candidate"]
    blessed: Literal[False]
    task: ModelTask
    version: str = Field(min_length=1)
    license: str = Field(min_length=1)
    precision: str = Field(min_length=1)
    created_at_utc: str = Field(min_length=1)
    artifact_path: str = Field(min_length=1)
    artifact_sha256: str
    artifact_bytes: int = Field(gt=0)
    model_card_path: str = Field(min_length=1)
    model_card_sha256: str
    training_config_sha256: str
    dataset_config_hash: str
    source_manifest_sha256: str
    split_manifest_sha256: str
    evaluation_predictions_path: str | None = None
    inference: dict[str, Any] | None = None
    evidence_attachments: tuple[EvidenceAttachment, ...] = ()
    gate_policy: GatePolicy
    export_consistency: ExportConsistency
    evaluation: EvaluationEvidence
    adapter_smoke: AdapterImportSmoke

    @model_validator(mode="after")
    def validate_lineage(self) -> "CandidateBundleManifest":
        for field_name in (
            "artifact_sha256",
            "model_card_sha256",
            "training_config_sha256",
            "dataset_config_hash",
            "source_manifest_sha256",
            "split_manifest_sha256",
        ):
            if not _SHA256.fullmatch(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        if self.precision not in _SUPPORTED_PRECISIONS:
            raise ValueError(f"unsupported candidate precision {self.precision!r}")
        if self.evaluation.dataset_config_hash != self.dataset_config_hash:
            raise ValueError("evaluation dataset_config_hash differs from bundle lineage")
        if self.evaluation.split_manifest_sha256 != self.split_manifest_sha256:
            raise ValueError("evaluation split hash differs from bundle lineage")
        _validate_gate_evidence(
            self.gate_policy, self.export_consistency, self.evaluation
        )
        _validate_attachment_uniqueness(self.evidence_attachments)
        return self


def finalize_candidate_bundle(
    bundle_directory: str | Path,
    *,
    artifact_path: str | Path,
    evaluation_predictions_path: str | Path,
    task: ModelTask,
    version: str,
    license_name: str,
    precision: str,
    adapter_entrypoint: str,
    inference: dict[str, Any],
    training_config_sha256: str,
    dataset_config_hash: str,
    source_manifest_sha256: str,
    split_manifest_sha256: str,
    gate_policy: GatePolicy | dict[str, Any],
    export_consistency: ExportConsistency | dict[str, Any],
    evaluation: EvaluationEvidence | dict[str, Any],
    intended_use: str,
    limitations: tuple[str, ...] | list[str],
    evidence_attachments: tuple[EvidenceAttachment | dict[str, Any], ...]
    | list[EvidenceAttachment | dict[str, Any]] = (),
    out_of_scope_use: tuple[str, ...] | list[str] = (
        "production deployment before five-task system validation",
    ),
) -> Path:
    """Write a cryptographically bound *candidate* bundle, never a blessed lock.

    The model artifact must already exist inside ``bundle_directory``.  Metric
    gates and export consistency are validated numerically, and the adapter is
    imported in the current environment before any candidate metadata is made.
    """

    if task not in MODEL_TASKS:
        raise CandidateBundleError(f"unsupported task {task!r}")
    bundle_root = Path(bundle_directory).expanduser().resolve(strict=False)
    bundle_root.mkdir(parents=True, exist_ok=True)
    artifact = Path(artifact_path).expanduser()
    if not artifact.is_absolute():
        artifact = bundle_root / artifact
    artifact = artifact.resolve(strict=False)
    artifact_relative = _relative_file(artifact, bundle_root, "artifact")
    reserved_metadata = {
        (bundle_root / "model_card.json").resolve(strict=False),
        (bundle_root / "bundle.manifest.json").resolve(strict=False),
    }
    if artifact in reserved_metadata:
        raise CandidateBundleError(
            "model artifact path collides with candidate metadata"
        )
    if artifact.stat().st_size <= 0:
        raise CandidateBundleError("model artifact is empty")
    artifact_digest = _sha256_file(artifact)
    policy = GatePolicy.model_validate(gate_policy)
    export = ExportConsistency.model_validate(export_consistency)
    evidence = EvaluationEvidence.model_validate(evaluation)
    _validate_gate_evidence(policy, export, evidence)
    predictions = Path(evaluation_predictions_path).expanduser()
    if not predictions.is_absolute():
        predictions = bundle_root / predictions
    predictions = predictions.resolve(strict=False)
    predictions_relative = _relative_file(
        predictions, bundle_root, "evaluation predictions"
    )
    if predictions == artifact:
        raise CandidateBundleError(
            "model artifact and evaluation predictions must be different files"
        )
    if predictions in reserved_metadata:
        raise CandidateBundleError(
            "evaluation predictions path collides with candidate metadata"
        )
    if predictions.stat().st_size <= 0:
        raise CandidateBundleError("evaluation predictions are empty")
    if _sha256_file(predictions) != evidence.predictions_sha256:
        raise CandidateBundleError(
            "evaluation predictions SHA-256 differs from evaluation evidence"
        )
    smoke = _adapter_import_smoke(adapter_entrypoint)
    runtime_inference = validate_inference_contract(task, inference)
    attachments = tuple(
        EvidenceAttachment.model_validate(item) for item in evidence_attachments
    )
    _validate_attachment_uniqueness(attachments)
    for attachment in attachments:
        attachment_path = _resolve_bundle_file(
            bundle_root, attachment.path, f"evidence attachment {attachment.kind!r}"
        )
        if attachment_path in reserved_metadata or attachment_path in {
            artifact,
            predictions,
        }:
            raise CandidateBundleError(
                f"evidence attachment {attachment.kind!r} collides with another bundle role"
            )
        _verify_bound_file(
            attachment_path,
            attachment.sha256,
            None,
            f"evidence attachment {attachment.kind!r}",
        )
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    card = ModelCard(
        schema_version=1,
        task=task,
        version=version,
        license=license_name,
        created_at_utc=now,
        intended_use=intended_use,
        out_of_scope_use=tuple(out_of_scope_use),
        limitations=tuple(limitations),
        training_config_sha256=training_config_sha256,
        dataset_config_hash=dataset_config_hash,
        source_manifest_sha256=source_manifest_sha256,
        split_manifest_sha256=split_manifest_sha256,
        artifact_sha256=artifact_digest,
        artifact_bytes=artifact.stat().st_size,
        evaluation_predictions_path=predictions_relative.as_posix(),
        inference=runtime_inference,
        evidence_attachments=attachments,
        gate_policy=policy,
        evaluation=evidence,
        export_consistency=export,
    )
    card_path = bundle_root / "model_card.json"
    _atomic_write_model(card_path, card)
    card_digest = _sha256_file(card_path)
    manifest = CandidateBundleManifest(
        schema_version=1,
        status="candidate",
        blessed=False,
        task=task,
        version=version,
        license=license_name,
        precision=precision,
        created_at_utc=now,
        artifact_path=artifact_relative.as_posix(),
        artifact_sha256=artifact_digest,
        artifact_bytes=artifact.stat().st_size,
        model_card_path=card_path.relative_to(bundle_root).as_posix(),
        model_card_sha256=card_digest,
        training_config_sha256=training_config_sha256,
        dataset_config_hash=dataset_config_hash,
        source_manifest_sha256=source_manifest_sha256,
        split_manifest_sha256=split_manifest_sha256,
        evaluation_predictions_path=predictions_relative.as_posix(),
        inference=runtime_inference,
        evidence_attachments=attachments,
        gate_policy=policy,
        export_consistency=export,
        evaluation=evidence,
        adapter_smoke=smoke,
    )
    manifest_path = bundle_root / "bundle.manifest.json"
    _atomic_write_model(manifest_path, manifest)
    return manifest_path


def load_candidate_bundle(path: str | Path) -> CandidateBundleManifest:
    manifest_path = Path(path).expanduser().resolve(strict=False)
    try:
        raw = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        return CandidateBundleManifest.model_validate(raw)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise CandidateBundleError(
            f"candidate bundle is invalid: {type(exc).__name__}: {exc}"
        ) from exc


def build_candidate_lock_entry(
    bundle_manifest: str | Path,
    *,
    lock_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Return an unblessed lock-entry candidate after re-verifying all evidence.

    This function never writes or edits ``models.lock.json``.  A separate human
    approval/review step must change status to ``blessed`` and ``blessed`` to
    true before the production readiness gate can accept an entry.
    """

    manifest_path = Path(bundle_manifest).expanduser().resolve(strict=False)
    bundle = load_candidate_bundle(manifest_path)
    bundle_root = manifest_path.parent
    artifact = _resolve_bundle_file(bundle_root, bundle.artifact_path, "artifact")
    card_path = _resolve_bundle_file(bundle_root, bundle.model_card_path, "model card")
    _verify_bound_file(
        artifact, bundle.artifact_sha256, bundle.artifact_bytes, "artifact"
    )
    _verify_bound_file(card_path, bundle.model_card_sha256, None, "model card")
    if bundle.evaluation_predictions_path is None:
        raise CandidateBundleError(
            "candidate bundle lacks hash-bound evaluation prediction bytes"
        )
    predictions_path = _resolve_bundle_file(
        bundle_root,
        bundle.evaluation_predictions_path,
        "evaluation predictions",
    )
    _verify_bound_file(
        predictions_path,
        bundle.evaluation.predictions_sha256,
        None,
        "evaluation predictions",
    )
    for attachment in bundle.evidence_attachments:
        attachment_path = _resolve_bundle_file(
            bundle_root,
            attachment.path,
            f"evidence attachment {attachment.kind!r}",
        )
        _verify_bound_file(
            attachment_path,
            attachment.sha256,
            None,
            f"evidence attachment {attachment.kind!r}",
        )
    try:
        card = ModelCard.model_validate(
            json.loads(
                card_path.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise CandidateBundleError(f"model card is invalid: {exc}") from exc
    if (
        card.task != bundle.task
        or card.version != bundle.version
        or card.license != bundle.license
        or card.created_at_utc != bundle.created_at_utc
        or card.artifact_sha256 != bundle.artifact_sha256
        or card.artifact_bytes != bundle.artifact_bytes
        or card.training_config_sha256 != bundle.training_config_sha256
        or card.dataset_config_hash != bundle.dataset_config_hash
        or card.source_manifest_sha256 != bundle.source_manifest_sha256
        or card.split_manifest_sha256 != bundle.split_manifest_sha256
        or card.evaluation_predictions_path
        != bundle.evaluation_predictions_path
        or card.inference != bundle.inference
        or card.evidence_attachments != bundle.evidence_attachments
        or card.gate_policy != bundle.gate_policy
        or card.evaluation != bundle.evaluation
        or card.export_consistency != bundle.export_consistency
    ):
        raise CandidateBundleError("model card lineage differs from bundle manifest")
    smoke = _adapter_import_smoke(bundle.adapter_smoke.entrypoint)
    if smoke.module != bundle.adapter_smoke.module or smoke.attribute != bundle.adapter_smoke.attribute:
        raise CandidateBundleError("adapter smoke metadata differs from live import")
    if bundle.inference is None:
        raise CandidateBundleError(
            "candidate bundle lacks a hash-bound runtime inference contract"
        )
    runtime_inference = validate_inference_contract(bundle.task, bundle.inference)

    if lock_directory is None:
        lock_path_value = str(artifact)
    else:
        lock_root = Path(lock_directory).expanduser().resolve(strict=False)
        try:
            lock_path_value = Path(os.path.relpath(artifact, lock_root)).as_posix()
        except ValueError as exc:
            raise CandidateBundleError("artifact and lock directory are on different volumes") from exc
    return {
        "task": bundle.task,
        "path": lock_path_value,
        "sha256": bundle.artifact_sha256,
        "status": "candidate",
        "blessed": False,
        "version": bundle.version,
        "license": bundle.license,
        "precision": bundle.precision,
        "adapter": bundle.adapter_smoke.entrypoint,
        "inference": runtime_inference,
        "bundle_manifest_sha256": _sha256_file(manifest_path),
        "training_config_sha256": bundle.training_config_sha256,
        "dataset_config_hash": bundle.dataset_config_hash,
        "gate_policy": bundle.gate_policy.model_dump(mode="json"),
        "evaluation": bundle.evaluation.model_dump(mode="json"),
    }


_INFERENCE_KEYS: dict[ModelTask, frozenset[str]] = {
    "component": frozenset({"device", "score_threshold", "max_detections"}),
    "pin": frozenset(
        {
            "device",
            "input_size",
            "output_stride",
            "score_threshold",
            "nms_kernel",
            "nms_radius_heatmap",
            "max_detections_per_component",
            "base_length_pixels",
            "box_margin_px",
            "box_prior_mode",
        }
    ),
    "ocr": frozenset(
        {
            "device",
            "detector_threshold",
            "detector_min_area",
            "recognition_threshold",
            "detector_batch_size",
            "recognition_batch_size",
            "max_detections_per_view",
            "max_recognition_width",
        }
    ),
    "wire": frozenset(
        {"device", "threshold", "batch_size", "output_kind", "mean", "std"}
    ),
    "junction": frozenset(
        {
            "device",
            "input_size",
            "patch_size_original_px",
            "score_threshold",
            "candidate_merge_tolerance_px",
            "segment_touch_tolerance_px",
            "maximum_candidates",
            "batch_size",
        }
    ),
}


def build_inference_contract(task: ModelTask, task_config: Any) -> dict[str, Any]:
    """Return the deterministic adapter options used by a trained candidate.

    Model bytes alone do not reproduce validation: detector thresholds, NMS,
    input geometry, and output caps materially affect predictions.  Every new
    bundle therefore binds the complete outcome-affecting adapter contract.
    """

    if task == "component":
        configured_max = getattr(
            task_config, "component_box_detections_per_img", None
        )
        raw = {
            "device": "auto",
            "score_threshold": float(task_config.component_score_threshold),
            # The production adapter's explicit output cap is 300.  When the
            # training config leaves TorchVision's internal detector cap at its
            # architecture default, retaining 300 here prevents an unrelated
            # adapter default from becoming an unbound part of deployment.
            "max_detections": int(
                configured_max if configured_max is not None else 300
            ),
        }
    elif task == "pin":
        input_height, input_width = (
            int(value) for value in task_config.input_size
        )
        raw = {
            "device": "auto",
            # Training configuration stores image shapes as (height, width),
            # while PinAdapterConfig and ``_view_tensor`` consume
            # (width, height).  Square production plans hid this distinction;
            # serialize the adapter's coordinate order explicitly.
            "input_size": [input_width, input_height],
            "output_stride": 4,
            "score_threshold": float(task_config.pin_score_threshold),
            "nms_kernel": 3,
            "nms_radius_heatmap": float(task_config.pin_nms_radius),
            "max_detections_per_component": int(task_config.pin_max_detections),
            "base_length_pixels": 8.0,
            "box_margin_px": float(task_config.pin_box_margin_px),
            "box_prior_mode": str(task_config.pin_box_prior_mode),
        }
    elif task == "ocr":
        raw = {
            "device": "auto",
            "detector_threshold": float(task_config.ocr_detect_threshold),
            "detector_min_area": int(task_config.ocr_min_area),
            "recognition_threshold": 0.0,
            "detector_batch_size": 4,
            "recognition_batch_size": 32,
            "max_detections_per_view": 512,
            "max_recognition_width": 4096,
        }
    elif task == "wire":
        raw = {
            "device": "auto",
            "threshold": 0.5,
            "batch_size": 4,
            "output_kind": "logits",
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        }
    elif task == "junction":
        width, height = (int(value) for value in task_config.input_size)
        if width != height:
            raise CandidateBundleError(
                "junction inference requires a square configured input_size"
            )
        raw = {
            "device": "auto",
            "input_size": width,
            "patch_size_original_px": 64.0,
            "score_threshold": 0.55,
            "candidate_merge_tolerance_px": 2.0,
            "segment_touch_tolerance_px": 2.5,
            "maximum_candidates": 16384,
            "batch_size": 128,
        }
    else:  # pragma: no cover - ModelTask typing plus caller validation owns this.
        raise CandidateBundleError(f"unsupported inference-contract task {task!r}")
    return validate_inference_contract(task, raw)


def validate_inference_contract(
    task: ModelTask, raw: dict[str, Any]
) -> dict[str, Any]:
    """Validate and canonicalize one task's hash-bound runtime options."""

    if not isinstance(raw, dict) or not raw:
        raise CandidateBundleError("runtime inference contract must be a non-empty object")
    expected = _INFERENCE_KEYS[task]
    actual = set(raw)
    if actual != expected:
        raise CandidateBundleError(
            "runtime inference contract keys differ for "
            f"{task}: missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )
    try:
        encoded = json.dumps(
            raw,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        canonical = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CandidateBundleError(
            f"runtime inference contract is not finite JSON: {exc}"
        ) from exc
    if not isinstance(canonical.get("device"), str) or not canonical["device"].strip():
        raise CandidateBundleError("runtime inference device must be non-empty text")

    probability_fields = {
        "component": ("score_threshold",),
        "pin": ("score_threshold",),
        "ocr": ("detector_threshold", "recognition_threshold"),
        "wire": ("threshold",),
        "junction": ("score_threshold",),
    }[task]
    for name in probability_fields:
        value = canonical[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise CandidateBundleError(f"runtime inference {name} must be in [0,1]")

    positive_integer_fields = {
        "component": ("max_detections",),
        "pin": (
            "output_stride",
            "nms_kernel",
            "max_detections_per_component",
        ),
        "ocr": (
            "detector_min_area",
            "detector_batch_size",
            "recognition_batch_size",
            "max_detections_per_view",
            "max_recognition_width",
        ),
        "wire": ("batch_size",),
        "junction": ("input_size", "maximum_candidates", "batch_size"),
    }[task]
    for name in positive_integer_fields:
        value = canonical[name]
        if type(value) is not int or value <= 0:
            raise CandidateBundleError(
                f"runtime inference {name} must be a positive integer"
            )

    if task == "pin":
        size = canonical["input_size"]
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(type(value) is not int or value <= 0 for value in size)
        ):
            raise CandidateBundleError(
                "runtime inference input_size must contain two positive integers"
            )
        if canonical["output_stride"] != 4:
            raise CandidateBundleError("runtime inference Pin output_stride must be 4")
        if canonical["nms_kernel"] % 2 != 1:
            raise CandidateBundleError("runtime inference Pin nms_kernel must be odd")
        for name in ("nms_radius_heatmap", "box_margin_px"):
            value = canonical[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or float(value) < 0.0
            ):
                raise CandidateBundleError(
                    f"runtime inference Pin {name} must be non-negative"
                )
        base_length = canonical["base_length_pixels"]
        if (
            isinstance(base_length, bool)
            or not isinstance(base_length, (int, float))
            or float(base_length) <= 0.0
        ):
            raise CandidateBundleError(
                "runtime inference Pin base_length_pixels must be positive"
            )
        if canonical["box_prior_mode"] not in {"filled_rect_v1", "boundary_band_v2"}:
            raise CandidateBundleError("runtime inference Pin box_prior_mode is invalid")
    if task == "wire":
        if canonical["output_kind"] not in {"logits", "probabilities"}:
            raise CandidateBundleError("runtime inference Wire output_kind is invalid")
        for name in ("mean", "std"):
            values = canonical[name]
            if (
                not isinstance(values, list)
                or len(values) != 3
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in values
                )
            ):
                raise CandidateBundleError(
                    f"runtime inference Wire {name} must contain three numbers"
                )
        if any(float(value) <= 0 for value in canonical["std"]):
            raise CandidateBundleError("runtime inference Wire std must be positive")
    if task == "junction":
        for name in (
            "patch_size_original_px",
            "candidate_merge_tolerance_px",
            "segment_touch_tolerance_px",
        ):
            value = canonical[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or float(value) <= 0.0
            ):
                raise CandidateBundleError(
                    f"runtime inference Junction {name} must be positive"
                )
    return canonical


def _adapter_import_smoke(entrypoint: str) -> AdapterImportSmoke:
    if not isinstance(entrypoint, str) or not _ENTRYPOINT.fullmatch(entrypoint):
        raise CandidateBundleError("adapter entrypoint must be package.module:callable")
    module_name, attribute_name = entrypoint.rsplit(":", 1)
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute_name)
    except (ImportError, AttributeError) as exc:
        raise CandidateBundleError(
            f"adapter import smoke failed for {entrypoint!r}: {type(exc).__name__}: {exc}"
        ) from exc
    if not callable(factory):
        raise CandidateBundleError(f"adapter attribute {entrypoint!r} is not callable")
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return AdapterImportSmoke(
        entrypoint=entrypoint,
        module=module_name,
        attribute=attribute_name,
        import_passed=True,
        callable_passed=True,
        tested_at_utc=now,
    )


def _validate_gate_evidence(
    policy: GatePolicy,
    export: ExportConsistency,
    evaluation: EvaluationEvidence,
) -> None:
    if export.input_manifest_sha256 != policy.task_manifest_sha256:
        raise ValueError("export consistency input is not the bound task manifest")
    if export.minimum_samples != policy.export_min_samples:
        raise ValueError("export consistency minimum_samples differs from gate policy")
    if export.tolerance != policy.export_max_abs_error:
        raise ValueError("export consistency tolerance differs from gate policy")
    expected = {
        (gate.metric, gate.operator): gate.threshold
        for gate in policy.evaluation_gates
    }
    actual = {
        (gate.metric, gate.operator): gate.threshold
        for gate in evaluation.gates
    }
    if actual != expected:
        raise ValueError("evaluation gates differ from the hashed training gate policy")


def _validate_attachment_uniqueness(
    attachments: tuple[EvidenceAttachment, ...],
) -> None:
    kinds = [attachment.kind for attachment in attachments]
    paths = [attachment.path for attachment in attachments]
    if len(kinds) != len(set(kinds)):
        raise ValueError("evidence attachment kinds must be unique")
    if len(paths) != len(set(paths)):
        raise ValueError("evidence attachment paths must be unique")


def _verify_bound_file(
    path: Path, expected_sha256: str, expected_bytes: int | None, label: str
) -> None:
    if not path.is_file():
        raise CandidateBundleError(f"bound {label} file is missing: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise CandidateBundleError(f"bound {label} byte size changed")
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise CandidateBundleError(f"bound {label} SHA-256 changed")


def _resolve_bundle_file(root: Path, relative: str, label: str) -> Path:
    path = (root / Path(relative)).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CandidateBundleError(f"{label} path escapes bundle directory") from exc
    return path


def _relative_file(path: Path, root: Path, label: str) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise CandidateBundleError(f"{label} must be inside bundle_directory") from exc
    if not path.is_file():
        raise CandidateBundleError(f"{label} file is missing: {path}")
    return relative


def _atomic_write_model(path: Path, model: BaseModel) -> None:
    payload = json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON object key {key!r}")
        output[key] = value
    return output


__all__ = [
    "AdapterImportSmoke",
    "CandidateBundleError",
    "CandidateBundleManifest",
    "EvaluationEvidence",
    "EvidenceAttachment",
    "ExportConsistency",
    "GatePolicy",
    "MetricGateResult",
    "ModelCard",
    "build_candidate_lock_entry",
    "build_inference_contract",
    "finalize_candidate_bundle",
    "load_candidate_bundle",
    "validate_inference_contract",
]
