from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ModelTask = Literal["component", "pin", "ocr", "wire", "junction"]
DatasetTask = Literal[
    "component_detection",
    "pin_keypoints",
    "ocr_text",
    "wire_paths",
    "junction_patches",
]
MODEL_TASKS = ("component", "pin", "ocr", "wire", "junction")
DATASET_TASK_BY_MODEL_TASK: dict[str, str] = {
    "component": "component_detection",
    "pin": "pin_keypoints",
    "ocr": "ocr_text",
    "wire": "wire_paths",
    "junction": "junction_patches",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ENTRYPOINT = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$"
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetBinding(_StrictModel):
    derived_root: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    source_root: str = Field(min_length=1)
    source_manifest: str = Field(min_length=1)
    split_manifest: str = Field(min_length=1)
    derivation_version: str = Field(min_length=1)
    expected_sample_count: int = Field(gt=0)
    config_hash: str
    summary_sha256: str
    source_manifest_sha256: str
    split_manifest_sha256: str
    task_manifest_sha256: dict[DatasetTask, str]

    @model_validator(mode="after")
    def validate_hashes_and_tasks(self) -> "DatasetBinding":
        hash_values = {
            "config_hash": self.config_hash,
            "summary_sha256": self.summary_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            **{
                f"task_manifest_sha256.{task}": value
                for task, value in self.task_manifest_sha256.items()
            },
        }
        for name, value in hash_values.items():
            if not _SHA256.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        expected = set(DATASET_TASK_BY_MODEL_TASK.values())
        if set(self.task_manifest_sha256) != expected:
            raise ValueError(
                "task_manifest_sha256 must contain exactly the five derived tasks"
            )
        return self


class MetricGate(_StrictModel):
    metric: str = Field(min_length=1)
    operator: Literal["gte", "lte"] = "gte"
    threshold: float

    @model_validator(mode="after")
    def finite_threshold(self) -> "MetricGate":
        if not math.isfinite(self.threshold):
            raise ValueError("metric gate threshold must be finite")
        return self


class PretrainedArtifact(_StrictModel):
    """Immutable, locally cached initialization weights.

    Training never downloads a model implicitly.  A task may opt into a
    pretrained initialization only by naming the workspace-relative artifact
    together with the digest, provenance, and license that were reviewed when
    it was cached.
    """

    path: str = Field(min_length=1)
    sha256: str
    source: str = Field(min_length=1)
    license: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_artifact(self) -> "PretrainedArtifact":
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("pretrained_artifact.sha256 must be a lowercase SHA-256 digest")
        artifact = Path(self.path)
        if artifact.is_absolute() or ".." in artifact.parts:
            raise ValueError(
                "pretrained_artifact.path must be a workspace-relative path without '..'"
            )
        return self


class TaskTrainingConfig(_StrictModel):
    dataset_task: DatasetTask
    architecture: str = Field(min_length=1)
    runner_entrypoint: str | None = None
    pretrained_artifact: PretrainedArtifact | None = None
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    gradient_accumulation: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    input_size: tuple[int, int]
    precision: Literal["fp16", "fp32", "bf16"] = "fp16"
    export_format: Literal["onnx", "torchscript"] = "onnx"
    export_opset: int | None = Field(default=17, ge=11)
    export_max_abs_error: float = Field(ge=0)
    export_min_samples: int = Field(ge=1)
    evaluation_gates: tuple[MetricGate, ...] = Field(min_length=1)
    # Per-task inference/tuning knobs. They are part of the training policy so
    # a gate result is reproducible with exactly the same decoding settings.
    pin_score_threshold: float = 0.15
    pin_nms_radius: float = 2.0
    pin_box_margin_px: float = 0.0
    pin_box_prior_mode: Literal["filled_rect_v1", "boundary_band_v2"] = (
        "filled_rect_v1"
    )
    pin_max_detections: int = 128
    pin_grad_clip: float = 1.0
    pin_count_loss_weight: float = 0.0
    pin_count_target_multiplier: float = 4.0
    pin_base_channels: int = 16
    pin_heatmap_loss: Literal["focal", "bce", "weighted_bce"] = "weighted_bce"
    component_score_threshold: float = 0.0
    component_box_detections_per_img: int | None = None
    # Set power > 0 to sample train images by their rarest usable Component
    # label.  Validation remains untouched.  A cap prevents one annotation
    # from dominating an epoch on the strongly long-tailed Dataset300 split.
    component_balanced_sampling_power: float = 0.0
    component_balanced_sampling_max_weight: float = 8.0
    # Optional train-only view-mixture target.  Component validation remains
    # sequential full-view evidence; this knob prevents a much larger tile
    # population from almost eliminating full schematics during an epoch.
    component_full_view_sampling_fraction: float | None = None
    ocr_detect_threshold: float = 0.5
    ocr_min_area: int = 4
    ocr_iou_threshold: float = 0.5
    ocr_detector_size: int = 512
    # Recognition-only train sampling for rare alphanumeric prefixes such as
    # schematic reference designators.  The weights are derived exclusively
    # from frozen train rows; validation/test remain sequential and unweighted.
    ocr_recognition_balanced_sampling_power: float = 0.0
    ocr_recognition_balanced_sampling_max_weight: float = 8.0
    # Filled text boxes that touch or overlap collapse into one binary-mask
    # component.  Shrinking only the positive training target preserves the
    # full ground-truth boxes used by the detector metric while teaching the
    # segmentation head to leave an instance-separating border.
    ocr_target_shrink_ratio: float = 0.0
    # Train-only synthetic junction fill per missing/under-represented class.
    # 0 disables synthetic rows entirely; validation always stays real.
    junction_synthetic_per_class: int = 0

    @model_validator(mode="after")
    def validate_runtime_fields(self) -> "TaskTrainingConfig":
        if len(self.input_size) != 2 or any(value <= 0 for value in self.input_size):
            raise ValueError("input_size must contain two positive integers")
        if self.runner_entrypoint is not None and not _ENTRYPOINT.fullmatch(
            self.runner_entrypoint
        ):
            raise ValueError("runner_entrypoint must be package.module:callable")
        gate_keys = [(gate.metric, gate.operator) for gate in self.evaluation_gates]
        if len(gate_keys) != len(set(gate_keys)):
            raise ValueError("evaluation gate metric/operator pairs must be unique")
        if self.export_format == "torchscript" and self.export_opset is not None:
            raise ValueError("export_opset applies only to ONNX exports")
        if not 0.0 <= self.pin_score_threshold <= 1.0:
            raise ValueError("pin_score_threshold must be in [0, 1]")
        if self.pin_nms_radius < 0:
            raise ValueError("pin_nms_radius must be non-negative")
        if not math.isfinite(self.pin_box_margin_px) or self.pin_box_margin_px < 0:
            raise ValueError("pin_box_margin_px must be finite and non-negative")
        if self.pin_max_detections <= 0:
            raise ValueError("pin_max_detections must be positive")
        if self.pin_grad_clip < 0:
            raise ValueError("pin_grad_clip must be non-negative")
        if not 0.0 <= self.pin_count_loss_weight:
            raise ValueError("pin_count_loss_weight must be non-negative")
        if self.pin_count_target_multiplier <= 0:
            raise ValueError("pin_count_target_multiplier must be positive")
        if self.pin_base_channels < 8 or self.pin_base_channels % 2 != 0:
            raise ValueError("pin_base_channels must be an even integer >= 8")
        if not 0.0 <= self.component_score_threshold <= 1.0:
            raise ValueError("component_score_threshold must be in [0, 1]")
        if self.component_box_detections_per_img is not None and self.component_box_detections_per_img <= 0:
            raise ValueError("component_box_detections_per_img must be positive")
        if (
            not math.isfinite(self.component_balanced_sampling_power)
            or not 0.0 <= self.component_balanced_sampling_power <= 1.0
        ):
            raise ValueError("component_balanced_sampling_power must be finite and in [0, 1]")
        if (
            not math.isfinite(self.component_balanced_sampling_max_weight)
            or self.component_balanced_sampling_max_weight < 1.0
        ):
            raise ValueError(
                "component_balanced_sampling_max_weight must be finite and >= 1"
            )
        if self.component_full_view_sampling_fraction is not None and (
            not math.isfinite(self.component_full_view_sampling_fraction)
            or not 0.0 < self.component_full_view_sampling_fraction < 1.0
        ):
            raise ValueError(
                "component_full_view_sampling_fraction must be finite and in (0, 1)"
            )
        if not 0.0 <= self.ocr_detect_threshold <= 1.0:
            raise ValueError("ocr_detect_threshold must be in [0, 1]")
        if self.ocr_min_area < 0:
            raise ValueError("ocr_min_area must be non-negative")
        if self.ocr_detector_size <= 0:
            raise ValueError("ocr_detector_size must be positive")
        if (
            not math.isfinite(self.ocr_recognition_balanced_sampling_power)
            or not 0.0 <= self.ocr_recognition_balanced_sampling_power <= 1.0
        ):
            raise ValueError(
                "ocr_recognition_balanced_sampling_power must be finite and in [0, 1]"
            )
        if (
            not math.isfinite(self.ocr_recognition_balanced_sampling_max_weight)
            or self.ocr_recognition_balanced_sampling_max_weight < 1.0
        ):
            raise ValueError(
                "ocr_recognition_balanced_sampling_max_weight must be finite and >= 1"
            )
        if not 0.0 < self.ocr_iou_threshold <= 1.0:
            raise ValueError("ocr_iou_threshold must be in (0, 1]")
        if (
            not math.isfinite(self.ocr_target_shrink_ratio)
            or not 0.0 <= self.ocr_target_shrink_ratio < 0.5
        ):
            raise ValueError("ocr_target_shrink_ratio must be finite and in [0, 0.5)")
        if self.junction_synthetic_per_class < 0:
            raise ValueError("junction_synthetic_per_class must be non-negative")
        return self


class RuntimeBudget(_StrictModel):
    seed: int = Field(ge=0)
    deterministic_algorithms: bool = True
    max_vram_mb: int = Field(gt=0, le=4096)
    dataloader_workers: int = Field(ge=0, le=2)
    pin_memory: bool = True
    checkpoint_every_epochs: int = Field(gt=0)
    early_stopping_patience: int = Field(gt=0)


class TrainingConfig(_StrictModel):
    schema_version: Literal[1]
    name: str = Field(min_length=1)
    dataset: DatasetBinding
    runtime: RuntimeBudget
    tasks: dict[ModelTask, TaskTrainingConfig]

    @model_validator(mode="after")
    def validate_complete_task_matrix(self) -> "TrainingConfig":
        if set(self.tasks) != set(MODEL_TASKS):
            raise ValueError("tasks must contain exactly component, pin, ocr, wire, junction")
        for model_task, expected_dataset_task in DATASET_TASK_BY_MODEL_TASK.items():
            actual = self.tasks[model_task].dataset_task
            if actual != expected_dataset_task:
                raise ValueError(
                    f"task {model_task!r} must use dataset_task {expected_dataset_task!r}"
                )
        # The shipped defaults target a 4 GiB GPU.  Larger effective batches are
        # achieved by accumulation rather than unsafe per-step allocations.
        if self.runtime.max_vram_mb > 4096:
            raise ValueError("max_vram_mb exceeds the supported 4 GiB profile")
        return self


def load_training_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path).expanduser().resolve(strict=False)
    raw = _load_strict_json(config_path)
    if not isinstance(raw, dict):
        raise ValueError("training config root must be an object")
    return TrainingConfig.model_validate(raw)


def training_config_sha256(config: TrainingConfig) -> str:
    return _training_config_document_sha256(config.model_dump(mode="json"))


def training_config_resume_sha256s(config: TrainingConfig) -> tuple[str, ...]:
    """Return current and narrowly compatible historical config digests.

    A formal resume remains byte-bound to its training policy. Additive fields
    may be omitted from historical digests only when every task still has the
    exact no-op default, so a real sampling, target, or decoder change can
    never pass as an equivalent resume.
    """

    document = config.model_dump(mode="json")
    current = _training_config_document_sha256(document)
    additive_defaults: dict[str, Any] = {
        "component_balanced_sampling_power": 0.0,
        "component_balanced_sampling_max_weight": 8.0,
        "component_full_view_sampling_fraction": None,
        "ocr_target_shrink_ratio": 0.0,
        "pin_box_prior_mode": "filled_rect_v1",
        "ocr_recognition_balanced_sampling_power": 0.0,
        "ocr_recognition_balanced_sampling_max_weight": 8.0,
    }
    tasks = document.get("tasks")
    if not isinstance(tasks, dict) or not all(isinstance(task, dict) for task in tasks.values()):
        return (current,)
    # Schema history is ordered newest-to-oldest. Each generation removes all
    # fields introduced since that historical document shape.
    removable_generations = (
        (
            "component_full_view_sampling_fraction",
        ),
        (
            "component_full_view_sampling_fraction",
            "ocr_recognition_balanced_sampling_power",
            "ocr_recognition_balanced_sampling_max_weight",
        ),
        (
            "component_full_view_sampling_fraction",
            "ocr_recognition_balanced_sampling_power",
            "ocr_recognition_balanced_sampling_max_weight",
            "pin_box_prior_mode",
        ),
        tuple(additive_defaults),
    )
    compatible = [current]
    for fields in removable_generations:
        if any(
            task.get(field) != additive_defaults[field]
            for task in tasks.values()
            for field in fields
        ):
            continue
        legacy_document = {
            **document,
            "tasks": {
                name: {
                    key: value
                    for key, value in task.items()
                    if key not in fields
                }
                for name, task in tasks.items()
            },
        }
        legacy = _training_config_document_sha256(legacy_document)
        if legacy not in compatible:
            compatible.append(legacy)
    return tuple(compatible)


def _training_config_document_sha256(document: dict[str, Any]) -> str:
    payload = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read strict training config {path}: {type(exc).__name__}: {exc}"
        ) from exc


__all__ = [
    "DATASET_TASK_BY_MODEL_TASK",
    "DatasetBinding",
    "DatasetTask",
    "MODEL_TASKS",
    "MetricGate",
    "ModelTask",
    "PretrainedArtifact",
    "RuntimeBudget",
    "TaskTrainingConfig",
    "TrainingConfig",
    "load_training_config",
    "training_config_resume_sha256s",
    "training_config_sha256",
]
