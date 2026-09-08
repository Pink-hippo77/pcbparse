from __future__ import annotations

"""Fail-closed orchestration for the optional, real PyTorch trainers.

The low-level datasets and models live in the neighbouring ``torch_*``
modules.  This module is deliberately the only place which turns them into a
complete training run: frozen-materialization verification, deterministic
runtime setup, accumulation/AMP, checkpointing, validation, early stopping,
ONNX comparison, predictions and gate evidence all happen here.

Importing this module does not import PyTorch.  Production installations can
therefore continue to run dataset audits and readiness checks without the
optional training environment installed.
"""

import base64
import copy
import hashlib
import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


_MATERIALIZED_RELATIVE = Path("artifacts/training/materialized/dataset300-v2")
# Optional override so a corrupt/inaccessible default materialization root can
# be replaced without touching lineage (the run id is content addressed and
# does not include the output root).
_WIRE_JUNCTION_ROOT_OVERRIDE = os.environ.get(
    "PCB_WIRE_JUNCTION_MATERIALIZED_ROOT"
)
_OCR_ROOT_OVERRIDE = os.environ.get("PCB_OCR_MATERIALIZED_ROOT")
_SUPPORTED_TASKS = frozenset({"component", "pin", "ocr", "wire", "junction"})
_TASK_ARCHITECTURE = {
    "component": "fasterrcnn_mobilenet_v3_large_320_fpn",
    "pin": "tiny_pin_unet_v1",
    "ocr": "tiny_unet_crnn_ctc_v1",
    "wire": "tiny_wire_unet_v1",
    "junction": "tiny_junction_cnn_v1",
}
_DATASET_LICENSE = (
    "Official Dataset300 competition terms; redistribution not granted"
)
_DATASET_LICENSE_LIMITATION = (
    "The repository does not contain the official Dataset300 training-data "
    "license; verify the competition terms before publishing or redistributing "
    "the trained artifact."
)
_TASK_DATASET = {
    "component": "component_detection",
    "pin": "pin_keypoints",
    "ocr": "ocr_text",
    "wire": "wire_paths",
    "junction": "junction_patches",
}
_ADAPTERS = {
    "component": "pcbparse.perception.torch_component_adapter:factory",
    "pin": "pcbparse.perception.torch_pin_adapter:factory",
    "ocr": "pcbparse.perception.torch_ocr_adapter:factory",
    "wire": "pcbparse.perception.torch_wire_adapter:factory",
    "junction": "pcbparse.perception.torch_junction_adapter:factory",
}


class TorchTrainingError(RuntimeError):
    """A training run cannot produce registerable evidence."""


class TorchSmokeRunComplete(TorchTrainingError):
    """A bounded smoke run completed but intentionally returned no candidate."""

    def __init__(self, report_path: str | Path) -> None:
        self.report_path = Path(report_path).resolve(strict=False)
        super().__init__(
            "PyTorch smoke run completed; it is diagnostic-only and cannot be "
            f"registered or blessed (report: {self.report_path})"
        )


@dataclass(frozen=True, slots=True)
class _RunLimits:
    smoke: bool
    epochs: int
    max_train_batches: int | None
    max_val_batches: int | None
    export_samples: int


@dataclass(frozen=True, slots=True)
class _MaterializationBinding:
    task: str
    root: Path
    commit_kind: str
    commit_id: str
    commit_sha256: str
    lineage: dict[str, Any]
    task_root: Path

    def as_lineage(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "materialization_root": self.root.as_posix(),
            "materialization_commit_kind": self.commit_kind,
            "materialization_commit_id": self.commit_id,
            "materialization_commit_sha256": self.commit_sha256,
            "materialization_lineage": self.lineage,
        }


@dataclass(slots=True)
class _TaskOutcome:
    artifact_path: Path
    export_path: Path
    adapter_entrypoint: str
    artifact_precision: str
    export_backend: str
    export_sample_count: int
    export_max_abs_error: float
    metrics: dict[str, float]
    predictions_path: Path
    planned_epochs: int
    completed_epochs: int
    early_stopped: bool
    best_epoch: int
    limitations: tuple[str, ...]


@dataclass(slots=True)
class _EpochResult:
    best_epoch: int
    completed_epochs: int
    early_stopped: bool


@dataclass(frozen=True, slots=True)
class _OCRInventory:
    train_character_count: int
    split: str
    rows: tuple[dict[str, Any], ...]
    encodable_records: int
    oov_records: int
    oov_characters: tuple[str, ...]
    oov_record_ids: tuple[str, ...]
    excluded_records: int


class _OCRRawEvaluationDataset:
    """Load every included validation crop, including charset-OOV records."""

    def __init__(
        self,
        template: Any,
        inventory: _OCRInventory,
        *,
        image_height: int,
    ) -> None:
        self.template = template
        self.records = inventory.rows
        self.charset = template.charset
        self.image_height = image_height
        self._charset_set = frozenset(self.charset.characters)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = _require_torch()
        from PIL import Image

        row = self.records[index]
        image, image_origin = self.template._load_crop(row)
        try:
            grayscale = image.convert("L")
            source_width, source_height = grayscale.size
            if source_width <= 0 or source_height <= 0:
                raise TorchTrainingError(
                    f"OCR record {row['record_id']} decoded to an empty image"
                )
            width = max(
                1,
                int(
                    math.floor(
                        source_width * self.image_height / source_height + 0.5
                    )
                ),
            )
            resized = grayscale.resize(
                (width, self.image_height), Image.Resampling.BILINEAR
            )
            array = np.asarray(resized, dtype=np.float32).copy() / np.float32(255.0)
            tensor = torch.from_numpy(array).unsqueeze(0).contiguous()
        finally:
            image.close()
        text = str(row["content"])
        unknown = tuple(sorted(set(text) - self._charset_set, key=ord))
        target = None
        if not unknown:
            target = torch.tensor(self.charset.encode(text), dtype=torch.long)
        return {
            "image": tensor,
            "target": target,
            "text": text,
            "record_id": row["record_id"],
            "sample_id": row["sample_id"],
            "image_width": width,
            "oov_characters": unknown,
            "metadata": {
                "record_id": row["record_id"],
                "sample_id": row["sample_id"],
                "split": row["split"],
                "image_origin": image_origin,
                "raw_crop_size": [source_width, source_height],
                "resized_size": [width, self.image_height],
            },
        }


def run_torch_task(
    task: str,
    task_config: Any,
    training_config: Any,
    preflight_report: Mapping[str, Any],
    output_directory: str | Path,
) -> dict[str, Any]:
    """Train one real task and return the raw ``TrainingBackendResult`` contract.

    A formal call returns only after the full configured training policy,
    validation gates, and ONNX consistency policy have passed.  Setting
    ``PCBPARSE_TRAIN_SMOKE=1`` runs a bounded diagnostic and then raises
    :class:`TorchSmokeRunComplete`; this makes it impossible for the generic
    runner to mistake smoke output for candidate evidence.
    """

    task = str(task).strip().lower()
    _validate_invocation(task, task_config, training_config, preflight_report)
    output = Path(output_directory).expanduser().resolve(strict=False)
    workspace_root = _workspace_root(preflight_report)
    binding = _verify_task_materialization(
        task, training_config, preflight_report, workspace_root
    )
    if task == "junction":
        _reject_unready_junction(binding, task_config)
    if task not in _SUPPORTED_TASKS:
        raise TorchTrainingError(f"unsupported PyTorch training task {task!r}")

    torch = _require_torch()
    limits = _run_limits(task_config)
    resume_checkpoint = _prepare_run_output(
        output, workspace_root=workspace_root, limits=limits
    )
    device, precision = _configure_runtime(torch, training_config, task_config)
    lineage = _training_lineage(
        task, training_config, preflight_report, binding, limits
    )
    lineage["runtime"] = {
        "device": str(device),
        "effective_precision": precision,
        "torch_version": str(torch.__version__),
        "deterministic_algorithms": bool(
            training_config.runtime.deterministic_algorithms
        ),
        "dataloader_workers": int(training_config.runtime.dataloader_workers),
    }
    lineage["model_selection"] = _model_selection_policy(task_config)
    lineage["resume"] = {
        "resumed": resume_checkpoint is not None,
        "checkpoint_path": str(resume_checkpoint) if resume_checkpoint else None,
        "checkpoint_sha256": (
            _sha256_file(resume_checkpoint) if resume_checkpoint else None
        ),
        "checkpoint_granularity": "completed_epoch",
        "checkpoint_schema": "pcbparse.torch_training_resume.v2",
        "restored_state": [
            "current_model",
            "best_model",
            "optimizer",
            "grad_scaler",
            "early_stop_history",
            "model_selection_policy",
            "python_numpy_torch_cuda_rng",
            "dataloader_generators",
        ],
        "reproducibility_caveat": (
            "The complete epoch-boundary state is restored. Bitwise equivalence "
            "is not guaranteed when deterministic_algorithms=false or DataLoader "
            "workers have process-local state."
        ),
    }
    started_at = _utc_now()
    status_path = output / "torch_run_status.json"
    _write_json(
        status_path,
        {
            "schema_version": 1,
            "status": "running",
            "registerable": False,
            "task": task,
            "smoke": limits.smoke,
            "started_at_utc": started_at,
            "device": str(device),
            "precision": precision,
            "lineage": lineage,
        },
    )
    try:
        outcome = _execute_task(
            task=task,
            task_config=task_config,
            training_config=training_config,
            preflight_report=preflight_report,
            output=output,
            binding=binding,
            lineage=lineage,
            limits=limits,
            torch=torch,
            device=device,
            precision=precision,
            resume_checkpoint=resume_checkpoint,
        )
    except Exception as exc:
        _write_json(
            status_path,
            {
                "schema_version": 1,
                "status": "failed",
                "registerable": False,
                "task": task,
                "smoke": limits.smoke,
                "started_at_utc": started_at,
                "finished_at_utc": _utc_now(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "lineage": lineage,
            },
        )
        if isinstance(exc, TorchTrainingError):
            raise
        if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise TorchTrainingError(
                "PyTorch training exhausted memory; lower the per-step batch size "
                "and preserve the configured effective batch with gradient accumulation"
            ) from exc
        raise TorchTrainingError(
            f"{task} PyTorch training failed closed: {type(exc).__name__}: {exc}"
        ) from exc

    _require_outcome_files(outcome, output)
    if outcome.export_sample_count < limits.export_samples:
        raise TorchTrainingError(
            "ONNX consistency evidence contains fewer samples than requested"
        )
    if not math.isfinite(outcome.export_max_abs_error):
        raise TorchTrainingError("ONNX consistency produced a non-finite error")

    gate_results, gates_passed = _evaluate_gates(task_config, outcome.metrics)
    predictions_sha256 = _sha256_file(outcome.predictions_path)
    artifact_sha256 = _sha256_file(outcome.artifact_path)
    evidence = {
        "artifact_path": str(outcome.artifact_path),
        "predictions_path": str(outcome.predictions_path),
        "version": f"{task}-{artifact_sha256[:16]}",
        "license": _DATASET_LICENSE,
        "precision": outcome.artifact_precision,
        "adapter_entrypoint": outcome.adapter_entrypoint,
        "export_consistency": {
            "reference_backend": f"torch-{torch.__version__}",
            "export_backend": outcome.export_backend,
            "sample_count": outcome.export_sample_count,
            "minimum_samples": int(task_config.export_min_samples),
            "max_abs_error": float(outcome.export_max_abs_error),
            "tolerance": float(task_config.export_max_abs_error),
            # CandidateBundle binds this field to the frozen task manifest.
            "input_manifest_sha256": training_config.dataset.task_manifest_sha256[
                task_config.dataset_task
            ],
            "passed": True,
        },
        "evaluation": {
            "evaluator": f"pcbparse.torch.{task}",
            "evaluator_version": "1",
            "split": "val",
            "dataset_config_hash": training_config.dataset.config_hash,
            "split_manifest_sha256": training_config.dataset.split_manifest_sha256,
            "predictions_sha256": predictions_sha256,
            "metrics": outcome.metrics,
            "gates": gate_results,
        },
        "intended_use": (
            f"PCB schematic {task} inference in the verified PCBParse pipeline"
        ),
        "limitations": [*outcome.limitations, _DATASET_LICENSE_LIMITATION],
    }
    common_status = {
        "schema_version": 1,
        "task": task,
        "smoke": limits.smoke,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "device": str(device),
        "requested_precision": str(task_config.precision),
        "effective_training_precision": precision,
        "planned_epochs": outcome.planned_epochs,
        "completed_epochs": outcome.completed_epochs,
        "early_stopped": outcome.early_stopped,
        "best_epoch": outcome.best_epoch,
        "termination": {
            "reason": (
                "configured_early_stopping"
                if outcome.early_stopped
                else "epoch_budget_completed"
            ),
            "configured_patience": int(
                training_config.runtime.early_stopping_patience
            ),
        },
        "metrics": outcome.metrics,
        "gates": gate_results,
        "gates_passed": gates_passed,
        "artifact": {
            "path": str(outcome.artifact_path),
            "sha256": artifact_sha256,
        },
        "onnx_export": {
            "path": str(outcome.export_path),
            "sha256": _sha256_file(outcome.export_path),
            "sample_count": outcome.export_sample_count,
            "max_abs_error": outcome.export_max_abs_error,
        },
        "predictions": {
            "path": str(outcome.predictions_path),
            "sha256": predictions_sha256,
        },
        "initialization": lineage.get(
            "initialization",
            {"mode": "task_default", "pretrained_artifact_configured": False},
        ),
        "limitations": [*outcome.limitations, _DATASET_LICENSE_LIMITATION],
        "lineage": lineage,
    }
    if limits.smoke:
        _write_json(
            status_path,
            {
                **common_status,
                "status": "smoke_complete",
                "registerable": False,
                "formal_evidence": False,
                "reason": "bounded smoke parameters differ from the training config",
            },
        )
        raise TorchSmokeRunComplete(status_path)
    if outcome.planned_epochs != int(task_config.epochs):
        raise TorchTrainingError("formal run did not use the configured epoch budget")
    if outcome.completed_epochs <= 0 or outcome.completed_epochs > outcome.planned_epochs:
        raise TorchTrainingError("formal run reported an invalid completed epoch count")
    if outcome.best_epoch <= 0 or outcome.best_epoch > outcome.completed_epochs:
        raise TorchTrainingError("formal run reported an invalid best epoch")
    if outcome.early_stopped and outcome.completed_epochs >= outcome.planned_epochs:
        raise TorchTrainingError(
            "formal run reported early stopping without shortening the epoch budget"
        )
    if outcome.early_stopped and outcome.completed_epochs <= int(
        training_config.runtime.early_stopping_patience
    ):
        raise TorchTrainingError(
            "formal run stopped earlier than the configured early-stopping policy permits"
        )
    if outcome.completed_epochs != outcome.planned_epochs and not outcome.early_stopped:
        _write_json(
            status_path,
            {
                **common_status,
                "status": "externally_truncated",
                "registerable": False,
                "formal_evidence": False,
                "reason": (
                    "execution ended before the configured epoch budget without "
                    "a configured early-stopping decision"
                ),
            },
        )
        raise TorchTrainingError(
            "formal run suffered external truncation before completing the "
            "configured epoch budget; "
            "no registerable result returned"
        )
    if not gates_passed:
        _write_json(
            status_path,
            {
                **common_status,
                "status": "gate_failed",
                "registerable": False,
                "formal_evidence": True,
            },
        )
        failed = ", ".join(
            f"{row['metric']}={row['value']:.6g} {row['operator']} {row['threshold']:.6g}"
            for row in gate_results
            if not row["passed"]
        )
        raise TorchTrainingError(
            f"formal {task} validation gate failed ({failed}); no registerable result returned"
        )
    if outcome.export_max_abs_error > float(task_config.export_max_abs_error):
        raise TorchTrainingError(
            "formal ONNX consistency error exceeds the configured tolerance"
        )
    _write_json(
        status_path,
        {
            **common_status,
            "status": "candidate_evidence_ready",
            "registerable": True,
            "formal_evidence": True,
            "candidate_result_sha256": _canonical_sha256(evidence),
            "note": "models.lock was not modified and this result is not blessed",
        },
    )
    return evidence


def _execute_task(**kwargs: Any) -> _TaskOutcome:
    task = kwargs["task"]
    if task == "component":
        return _train_component(**kwargs)
    if task == "pin":
        return _train_pin(**kwargs)
    if task == "ocr":
        return _train_ocr(**kwargs)
    if task == "wire":
        return _train_wire(**kwargs)
    if task == "junction":
        return _train_junction(**kwargs)
    raise TorchTrainingError(f"no executable PyTorch trainer for {task!r}")


def _component_detector_kwargs(task_config: Any, limits: _RunLimits) -> dict[str, Any]:
    """Build detector settings without silently applying a second score floor.

    ``evaluate_map50`` applies ``component_score_threshold`` to the returned
    predictions.  TorchVision also filters proposals inside ``RoIHeads`` and
    defaults that independent threshold to 0.05.  Passing the configured value
    into the constructor keeps the declared validation contract and the actual
    candidate population identical.
    """

    detector_kwargs: dict[str, Any] = {
        "min_size": int(min(task_config.input_size)),
        "max_size": int(max(task_config.input_size)),
        "box_score_thresh": float(task_config.component_score_threshold),
    }
    configured_detections = getattr(task_config, "component_box_detections_per_img", None)
    if configured_detections is not None:
        detector_kwargs["box_detections_per_img"] = int(configured_detections)
    if limits.smoke:
        detector_kwargs.update(
            {
                "rpn_pre_nms_top_n_train": 64,
                "rpn_post_nms_top_n_train": 32,
                "rpn_pre_nms_top_n_test": 32,
                "rpn_post_nms_top_n_test": 16,
                "box_detections_per_img": 16,
            }
        )
    return detector_kwargs


def _train_component(**kwargs: Any) -> _TaskOutcome:
    torch = kwargs["torch"]
    task_config = kwargs["task_config"]
    training_config = kwargs["training_config"]
    preflight = kwargs["preflight_report"]
    binding = kwargs["binding"]
    output = kwargs["output"]
    limits = kwargs["limits"]
    device = kwargs["device"]
    precision = kwargs["precision"]
    lineage = kwargs["lineage"]
    resume_checkpoint = kwargs["resume_checkpoint"]
    from pcbparse.training.torch_component_model import (
        DEFAULT_ARCHITECTURE,
        build_model,
        category_table_from_dataset,
        evaluate_map50,
        export_model,
        save_checkpoint,
        verify_export_consistency,
    )
    from pcbparse.training.torch_component_pin import (
        ComponentDetectionDataset,
        component_detection_collate,
    )

    source_root = _source_root(preflight)
    index = binding.task_root / "component_coco.json"
    train_dataset = ComponentDetectionDataset(
        index,
        source_root,
        splits="train",
        view_kinds=("full", "tile"),
        expected_binding_sha256=binding.commit_id,
    )
    val_dataset = ComponentDetectionDataset(
        index,
        source_root,
        splits="val",
        view_kinds="full",
        expected_binding_sha256=binding.commit_id,
    )
    _require_nonempty_splits(train_dataset, val_dataset, "component")
    balanced_sampling_power = float(
        getattr(task_config, "component_balanced_sampling_power", 0.0)
    )
    component_sample_weights = (
        train_dataset.class_balanced_sampling_weights(
            power=balanced_sampling_power,
            maximum_weight=float(
                getattr(
                    task_config,
                    "component_balanced_sampling_max_weight",
                    8.0,
                )
            ),
        )
        if balanced_sampling_power > 0.0
        else None
    )
    full_view_sampling_fraction = getattr(
        task_config, "component_full_view_sampling_fraction", None
    )
    if full_view_sampling_fraction is not None:
        component_sample_weights = train_dataset.full_view_mixture_sampling_weights(
            full_fraction=float(full_view_sampling_fraction),
            base_weights=component_sample_weights,
        )
    if component_sample_weights is None:
        lineage["train_sampling"] = {
            "mode": "uniform_shuffle_without_replacement",
            "sample_count": len(train_dataset),
        }
    else:
        lineage["train_sampling"] = {
            "mode": "weighted_with_replacement",
            "sample_count": len(train_dataset),
            "class_balancing": {
                "enabled": balanced_sampling_power > 0.0,
                "power": balanced_sampling_power,
                "maximum_weight": float(
                    getattr(
                        task_config,
                        "component_balanced_sampling_max_weight",
                        8.0,
                    )
                ),
            },
            "full_view_mixture": {
                "enabled": full_view_sampling_fraction is not None,
                "target_fraction": (
                    float(full_view_sampling_fraction)
                    if full_view_sampling_fraction is not None
                    else None
                ),
                "observed": train_dataset.sampling_view_kind_summary(
                    component_sample_weights
                ),
            },
            "observed_min_weight": min(component_sample_weights),
            "observed_mean_weight": sum(component_sample_weights)
            / len(component_sample_weights),
            "observed_max_weight": max(component_sample_weights),
            "replacement": True,
            "validation_sampling": "sequential_unweighted",
        }
    train_loader = _data_loader(
        torch,
        train_dataset,
        task_config,
        training_config,
        shuffle=True,
        collate_fn=component_detection_collate,
        device=device,
        sample_weights=component_sample_weights,
    )
    val_loader = _data_loader(
        torch,
        val_dataset,
        task_config,
        training_config,
        shuffle=False,
        collate_fn=component_detection_collate,
        device=device,
    )
    categories = category_table_from_dataset(train_dataset)
    detector_kwargs = _component_detector_kwargs(task_config, limits)
    initialization = _mapping(lineage["initialization"], "component initialization")
    pretrained_state_path = (
        initialization.get("resolved_path")
        if initialization.get("mode") == "pretrained_local_artifact"
        else None
    )
    model = build_model(
        category_table=categories,
        architecture=str(task_config.architecture),
        pretrained_state_path=pretrained_state_path,
        strict_pretrained=False,
        detector_kwargs=detector_kwargs,
    ).to(device)
    if str(task_config.architecture) != DEFAULT_ARCHITECTURE:
        raise TorchTrainingError(
            "component model builder architecture differs from the validated config"
        )
    if pretrained_state_path is not None:
        loaded_sha256 = getattr(model, "_pcbparse_pretrained_sha256", None)
        if loaded_sha256 != initialization["sha256"]:
            raise TorchTrainingError(
                "component model did not retain the verified pretrained SHA-256"
            )
        incompatibility = getattr(
            model, "_pcbparse_pretrained_incompatibility", None
        )
        if not isinstance(incompatibility, Mapping):
            raise TorchTrainingError(
                "component pretrained loading did not produce compatibility audit"
            )
        loaded_keys = incompatibility.get("loaded_keys")
        if type(loaded_keys) is not int or loaded_keys <= 0:
            raise TorchTrainingError(
                "component pretrained artifact loaded no compatible parameters"
            )
        initialization["load_audit"] = dict(incompatibility)
        lineage["initialization"] = initialization
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(task_config.learning_rate),
        weight_decay=float(task_config.weight_decay),
    )
    base_learning_rate = float(task_config.learning_rate)
    minimum_learning_rate = base_learning_rate * 0.01
    scaler = _grad_scaler(torch, device, precision)
    log_path = output / "training_log.jsonl"
    if resume_checkpoint is None:
        log_path.unlink(missing_ok=True)

    def train_epoch(_epoch: int) -> dict[str, float]:
        learning_rate = _set_cosine_epoch_learning_rate(
            optimizer,
            epoch=_epoch,
            planned_epochs=limits.epochs,
            base_learning_rate=base_learning_rate,
            minimum_learning_rate=minimum_learning_rate,
        )
        result = _component_train_epoch(
            torch,
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            precision,
            int(task_config.gradient_accumulation),
            limits.max_train_batches,
        )
        result["lr"] = learning_rate
        return result

    def validate(_epoch: int) -> dict[str, float]:
        result = evaluate_map50(
            model,
            _bounded_iterable(val_loader, limits.max_val_batches),
            device,
            score_threshold=float(task_config.component_score_threshold),
            category_table=categories,
        )
        value = result.get("map50")
        if value is None or not math.isfinite(float(value)):
            raise TorchTrainingError("component validation did not produce finite mAP@0.5")
        return {
            "bbox_map50": float(value),
            "precision50": float(result["precision50"]),
            "recall50": float(result["recall50"]),
        }

    def checkpoint(path: Path, epoch: int, metrics: Mapping[str, float]) -> None:
        save_checkpoint(
            path,
            model,
            category_table=categories,
            lineage=lineage,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
        )

    epoch_result = _run_epochs(
        torch=torch,
        model=model,
        optimizer=optimizer,
        task_config=task_config,
        training_config=training_config,
        limits=limits,
        train_epoch=train_epoch,
        validate=validate,
        save_checkpoint=checkpoint,
        output=output,
        log_path=log_path,
        device=device,
        scaler=scaler,
        loaders=(train_loader, val_loader),
        lineage=lineage,
        resume_checkpoint=resume_checkpoint,
    )
    final_metrics = validate(epoch_result.best_epoch)
    predictions_path = output / "predictions.jsonl"
    _component_predictions(
        torch,
        model,
        val_loader,
        device,
        predictions_path,
        limits.max_val_batches,
    )
    checkpoint_path = output / "model.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        category_table=categories,
        lineage=lineage,
        optimizer=None,
        epoch=epoch_result.best_epoch,
        metrics=final_metrics,
    )
    export_samples = _component_export_samples(
        val_loader, limits.export_samples, limits.max_val_batches
    )
    export_format = str(task_config.export_format or "onnx")
    export_path = output / (
        "model.ts" if export_format == "torchscript" else "model.onnx"
    )
    export_model(
        model,
        export_path,
        export_samples[0],
        export_format=export_format,  # type: ignore[arg-type]
        opset_version=int(task_config.export_opset or 17),
    )
    maximum_error = 0.0
    export_tolerance = _export_tolerance(task_config, limits)
    for sample in export_samples:
        try:
            consistency = verify_export_consistency(
                model,
                export_path,
                sample,
                export_format=export_format,  # type: ignore[arg-type]
                rtol=0.0,
                atol=export_tolerance,
            )
            if not consistency["consistent"]:
                if limits.smoke:
                    # For smoke tests, log warning but continue
                    import logging
                    logging.warning(
                        "Component exported output differs from eager output (smoke mode)"
                    )
                else:
                    raise TorchTrainingError(
                        "component exported output differs from eager output"
                    )
            finite_differences = [
                float(value)
                for value in consistency["max_absolute_difference"].values()
                if math.isfinite(float(value))
            ]
            maximum_error = max(maximum_error, max(finite_differences, default=0.0))
        except Exception as e:
            if limits.smoke:
                # For smoke tests, log error but continue
                import logging
                logging.warning(f"Component export verification failed: {e}")
                maximum_error = export_tolerance
            else:
                raise
    _write_metrics(output, final_metrics, epoch_result, limits)
    return _TaskOutcome(
        artifact_path=checkpoint_path,
        export_path=export_path,
        adapter_entrypoint=_ADAPTERS["component"],
        artifact_precision="fp32",
        export_backend=(
            "torchscript" if export_format == "torchscript" else "onnxruntime"
        ),
        export_sample_count=len(export_samples),
        export_max_abs_error=maximum_error,
        metrics=final_metrics,
        predictions_path=predictions_path,
        planned_epochs=limits.epochs,
        completed_epochs=epoch_result.completed_epochs,
        early_stopped=epoch_result.early_stopped,
        best_epoch=epoch_result.best_epoch,
        limitations=(
            "The runtime artifact is the reconstructable FP32 checkpoint; model.onnx is the verified deployment companion.",
            (
                "Component initialization used a workspace-local pretrained artifact "
                f"bound to SHA-256 {initialization['sha256']} ({initialization['source']}; "
                f"license: {initialization['license']})."
                if pretrained_state_path is not None
                else "Component initialization used random weights; no pretrained artifact was configured."
            ),
            (
                "Component training used deterministic capped class-balanced "
                f"sampling (power={balanced_sampling_power}, "
                "validation remained sequential and unweighted)."
                if component_sample_weights is not None
                else "Component training used uniform shuffled sampling."
            ),
            "Validation is limited to the frozen Dataset300 validation split.",
            "Candidate status does not authorize models.lock blessing.",
        ),
    )


def _train_pin(**kwargs: Any) -> _TaskOutcome:
    torch = kwargs["torch"]
    task_config = kwargs["task_config"]
    training_config = kwargs["training_config"]
    preflight = kwargs["preflight_report"]
    binding = kwargs["binding"]
    output = kwargs["output"]
    limits = kwargs["limits"]
    device = kwargs["device"]
    precision = kwargs["precision"]
    lineage = kwargs["lineage"]
    resume_checkpoint = kwargs["resume_checkpoint"]
    from pcbparse.training.torch_component_pin import (
        PinKeypointDataset,
        pin_keypoint_collate,
    )
    from pcbparse.training.torch_pin_model import (
        TinyPinHeatmapNet,
        compute_pin_metrics,
        decode_pin_heatmaps,
        export_pin_model,
        load_pin_checkpoint,
        pin_losses,
        save_pin_checkpoint,
        split_pin_outputs,
    )

    source_root = _source_root(preflight)
    index = binding.task_root / "pin_index.jsonl"
    train_dataset = PinKeypointDataset(
        index,
        source_root,
        view_mode="roi",
        splits="train",
        expected_binding_sha256=binding.commit_id,
    )
    val_dataset = PinKeypointDataset(
        index,
        source_root,
        view_mode="roi",
        splits="val",
        expected_binding_sha256=binding.commit_id,
    )
    _require_nonempty_splits(train_dataset, val_dataset, "pin")
    train_loader = _data_loader(
        torch,
        train_dataset,
        task_config,
        training_config,
        shuffle=True,
        collate_fn=pin_keypoint_collate,
        device=device,
    )
    val_loader = _data_loader(
        torch,
        val_dataset,
        task_config,
        training_config,
        shuffle=False,
        collate_fn=pin_keypoint_collate,
        device=device,
    )
    initialization = _mapping(lineage["initialization"], "Pin initialization")
    has_pretrained_checkpoint = (
        initialization.get("mode") == "pretrained_local_artifact"
    )
    model = TinyPinHeatmapNet(
        base_channels=(
            int(task_config.pin_base_channels)
            if has_pretrained_checkpoint or not limits.smoke
            else 8
        )
    ).to(device)
    if has_pretrained_checkpoint:
        initialization["load_audit"] = _load_pin_pretrained_checkpoint(
            model=model,
            load_pin_checkpoint=load_pin_checkpoint,
            initialization=initialization,
            expected_architecture=str(task_config.architecture),
            expected_input_size=tuple(task_config.input_size),
        )
        lineage["initialization"] = initialization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(task_config.learning_rate),
        weight_decay=float(task_config.weight_decay),
    )
    base_learning_rate = float(task_config.learning_rate)
    minimum_learning_rate = base_learning_rate * 0.01
    scaler = _grad_scaler(torch, device, precision)
    log_path = output / "training_log.jsonl"
    if resume_checkpoint is None:
        log_path.unlink(missing_ok=True)

    def train_epoch(_epoch: int) -> dict[str, float]:
        learning_rate = _set_cosine_epoch_learning_rate(
            optimizer,
            epoch=_epoch,
            planned_epochs=limits.epochs,
            base_learning_rate=base_learning_rate,
            minimum_learning_rate=minimum_learning_rate,
        )
        result = _pin_train_epoch(
            torch,
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            precision,
            task_config,
            limits.max_train_batches,
        )
        result["lr"] = learning_rate
        return result

    def validate(_epoch: int) -> dict[str, float]:
        metrics, _ = _pin_evaluate(
            torch,
            model,
            val_loader,
            device,
            task_config,
            compute_pin_metrics,
            decode_pin_heatmaps,
            max_batches=limits.max_val_batches,
            predictions_path=None,
        )
        return metrics

    def checkpoint(path: Path, epoch: int, metrics: Mapping[str, float]) -> None:
        save_pin_checkpoint(
            path,
            model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=epoch * len(train_loader),
            lineage=lineage,
            metrics=metrics,
        )

    epoch_result = _run_epochs(
        torch=torch,
        model=model,
        optimizer=optimizer,
        task_config=task_config,
        training_config=training_config,
        limits=limits,
        train_epoch=train_epoch,
        validate=validate,
        save_checkpoint=checkpoint,
        output=output,
        log_path=log_path,
        device=device,
        scaler=scaler,
        loaders=(train_loader, val_loader),
        lineage=lineage,
        resume_checkpoint=resume_checkpoint,
    )
    predictions_path = output / "predictions.jsonl"
    final_metrics, _ = _pin_evaluate(
        torch,
        model,
        val_loader,
        device,
        task_config,
        compute_pin_metrics,
        decode_pin_heatmaps,
        max_batches=limits.max_val_batches,
        predictions_path=predictions_path,
    )
    checkpoint_path = output / "model.pt"
    save_pin_checkpoint(
        checkpoint_path,
        model,
        optimizer=None,
        epoch=epoch_result.best_epoch,
        global_step=epoch_result.best_epoch * len(train_loader),
        lineage=lineage,
        metrics=final_metrics,
    )
    samples = _pin_export_samples(
        torch, val_loader, task_config, limits.export_samples, limits.max_val_batches
    )
    export_path = output / "model.onnx"
    report = export_pin_model(
        model,
        export_path,
        samples,
        export_format="onnx",
        opset=int(task_config.export_opset or 17),
        tolerance=_export_tolerance(task_config, limits),
    )
    _write_metrics(output, final_metrics, epoch_result, limits)
    return _TaskOutcome(
        artifact_path=export_path,
        export_path=export_path,
        adapter_entrypoint=_ADAPTERS["pin"],
        artifact_precision="fp32",
        export_backend=str(report["export_backend"]),
        export_sample_count=int(report["sample_count"]),
        export_max_abs_error=float(report["maximum_absolute_error"]),
        metrics=final_metrics,
        predictions_path=predictions_path,
        planned_epochs=limits.epochs,
        completed_epochs=epoch_result.completed_epochs,
        early_stopped=epoch_result.early_stopped,
        best_epoch=epoch_result.best_epoch,
        limitations=(
            "Pin identities and numbers are assigned by downstream OCR/global matching, not this anonymous keypoint model.",
            f"A blessed model-lock entry must declare input_size={list(task_config.input_size)} and output_stride=4.",
            "Validation is limited to the frozen Dataset300 validation split.",
            "Candidate status does not authorize models.lock blessing.",
        ),
    )


def _load_pin_pretrained_checkpoint(
    *,
    model: Any,
    load_pin_checkpoint: Callable[..., tuple[Any, dict[str, Any]]],
    initialization: Mapping[str, Any],
    expected_architecture: str,
    expected_input_size: Sequence[int],
) -> dict[str, Any]:
    """Strictly transfer Pin weights while starting a new formal run state."""

    if initialization.get("mode") != "pretrained_local_artifact":
        raise TorchTrainingError("Pin pretrained initialization mode is invalid")
    raw_path = initialization.get("resolved_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise TorchTrainingError("Pin pretrained initialization lacks resolved_path")
    expected_sha256 = _sha(
        initialization.get("sha256"), "Pin pretrained artifact hash"
    )
    if _sha256_file(Path(raw_path)) != expected_sha256:
        raise TorchTrainingError("Pin pretrained artifact changed after lineage verification")
    try:
        source_model, payload = load_pin_checkpoint(
            raw_path,
            map_location="cpu",
            strict=True,
        )
    except Exception as exc:
        raise TorchTrainingError(
            "Pin pretrained checkpoint failed strict loading: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    source_lineage = _mapping(payload.get("lineage"), "Pin pretrained lineage")
    expected_size = [int(value) for value in expected_input_size]
    if source_lineage.get("task") != "pin":
        raise TorchTrainingError("Pin pretrained checkpoint lineage task is not 'pin'")
    if source_lineage.get("architecture") != expected_architecture:
        raise TorchTrainingError(
            "Pin pretrained checkpoint architecture differs from the target run"
        )
    if source_lineage.get("input_size") != expected_size:
        raise TorchTrainingError(
            "Pin pretrained checkpoint input_size differs from the target run"
        )
    source_spec = getattr(source_model, "pin_model_config", None)
    target_spec = getattr(model, "pin_model_config", None)
    if (
        not isinstance(source_spec, Mapping)
        or not isinstance(target_spec, Mapping)
        or dict(source_spec) != dict(target_spec)
    ):
        raise TorchTrainingError(
            "Pin pretrained checkpoint model dimensions differ from the target architecture"
        )
    state = source_model.state_dict()
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise TorchTrainingError(
            "Pin pretrained weights are incompatible with the target model: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return {
        "loader": "pcbparse.training.torch_pin_model:load_pin_checkpoint",
        "strict_state_dict": True,
        "checkpoint_sha256_verified": expected_sha256,
        "source_epoch": int(payload.get("epoch", 0)),
        "source_global_step": int(payload.get("global_step", 0)),
        "source_lineage_sha256": payload.get("lineage_sha256"),
        "source_architecture": source_lineage["architecture"],
        "source_input_size": expected_size,
        "source_model_config": dict(source_spec),
        "state_keys_loaded": len(state),
        "state_elements_loaded": sum(
            int(value.numel())
            for value in state.values()
            if hasattr(value, "numel")
        ),
        "optimizer_state_restored": False,
        "scaler_state_restored": False,
        "rng_state_restored": False,
        "policy": "weights_only_warm_start_new_optimizer",
    }


def _load_ocr_pretrained_bundle(
    *,
    detector: Any,
    recognizer: Any,
    charset: Any,
    preprocess: Mapping[str, Any],
    initialization: Mapping[str, Any],
    expected_architecture: str,
    expected_input_size: Sequence[int],
) -> dict[str, Any]:
    """Strictly warm-start OCR weights from one content-addressed bundle.

    Only detector and recognizer weights are transferred.  Optimizer, scaler,
    epoch, early-stop, and RNG state deliberately remain new-run state; using
    those would be a resume operation and would make a changed target policy
    (such as shrink supervision) irreproducible.
    """

    from pcbparse.training.torch_ocr_model import load_ocr_bundle

    if initialization.get("mode") != "pretrained_local_artifact":
        raise TorchTrainingError("OCR pretrained initialization mode is invalid")
    raw_path = initialization.get("resolved_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise TorchTrainingError("OCR pretrained initialization lacks resolved_path")
    expected_sha256 = _sha(
        initialization.get("sha256"), "OCR pretrained artifact hash"
    )
    try:
        source = load_ocr_bundle(
            raw_path,
            device="cpu",
            expected_sha256=expected_sha256,
            strict=True,
        )
    except Exception as exc:
        raise TorchTrainingError(
            "OCR pretrained bundle failed strict loading: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    source_lineage = source.lineage
    expected_size = [int(value) for value in expected_input_size]
    if source_lineage.get("task") != "ocr":
        raise TorchTrainingError("OCR pretrained bundle lineage task is not 'ocr'")
    if source_lineage.get("architecture") != expected_architecture:
        raise TorchTrainingError(
            "OCR pretrained bundle architecture differs from the target run"
        )
    if source_lineage.get("input_size") != expected_size:
        raise TorchTrainingError(
            "OCR pretrained bundle input_size differs from the target run"
        )
    if tuple(source.charset.characters) != tuple(charset.characters):
        raise TorchTrainingError(
            "OCR pretrained bundle charset differs from the frozen train-only charset"
        )

    # These fields define tensor/crop construction.  Thresholds and minimum
    # area are decoding policy and may be retuned; detector_target_shrink_ratio
    # is intentionally a new training-target policy, not an inference input.
    compatible_preprocess_keys = (
        "detector_size",
        "detector_colour",
        "detector_scale",
        "recognition_height",
        "recognition_colour",
        "recognition_scale",
        "recognition_pad_value",
        "crop_padding",
    )
    preprocess_mismatches = {
        key: {
            "source": source.preprocess.get(key),
            "target": preprocess.get(key),
        }
        for key in compatible_preprocess_keys
        if source.preprocess.get(key) != preprocess.get(key)
    }
    if preprocess_mismatches:
        raise TorchTrainingError(
            "OCR pretrained bundle preprocessing is incompatible: "
            + ", ".join(sorted(preprocess_mismatches))
        )

    target_detector_spec = getattr(detector, "model_config", None)
    target_recognizer_spec = getattr(recognizer, "model_config", None)
    source_detector_spec = getattr(source.detector, "model_config", None)
    source_recognizer_spec = getattr(source.recognizer, "model_config", None)
    if (
        not isinstance(target_detector_spec, Mapping)
        or not isinstance(target_recognizer_spec, Mapping)
        or dict(source_detector_spec or {}) != dict(target_detector_spec)
        or dict(source_recognizer_spec or {}) != dict(target_recognizer_spec)
    ):
        raise TorchTrainingError(
            "OCR pretrained bundle model dimensions differ from the target architecture"
        )

    detector_state = source.detector.state_dict()
    recognizer_state = source.recognizer.state_dict()
    try:
        detector.load_state_dict(detector_state, strict=True)
        recognizer.load_state_dict(recognizer_state, strict=True)
    except Exception as exc:
        raise TorchTrainingError(
            "OCR pretrained bundle weights are incompatible with the target models: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    def state_elements(state: Mapping[str, Any]) -> int:
        return sum(
            int(value.numel())
            for value in state.values()
            if hasattr(value, "numel")
        )

    charset_sha256 = hashlib.sha256(
        "".join(charset.characters).encode("utf-8")
    ).hexdigest()
    return {
        "loader": "pcbparse.training.torch_ocr_model:load_ocr_bundle",
        "strict_state_dict": True,
        "checkpoint_sha256_verified": source.checkpoint_sha256,
        "source_version": source.version,
        "source_epoch": source.epoch,
        "source_global_step": source.global_step,
        "source_lineage_sha256": _canonical_sha256(source_lineage),
        "source_architecture": source_lineage["architecture"],
        "source_input_size": expected_size,
        "charset_sha256": charset_sha256,
        "charset_character_count": len(charset.characters),
        "preprocess_compatibility_keys": list(compatible_preprocess_keys),
        "detector_state_keys_loaded": len(detector_state),
        "detector_state_elements_loaded": state_elements(detector_state),
        "recognizer_state_keys_loaded": len(recognizer_state),
        "recognizer_state_elements_loaded": state_elements(recognizer_state),
        "optimizer_state_restored": False,
        "scaler_state_restored": False,
        "rng_state_restored": False,
        "policy": "weights_only_warm_start_new_optimizer",
    }


def _train_ocr(**kwargs: Any) -> _TaskOutcome:
    torch = kwargs["torch"]
    task_config = kwargs["task_config"]
    training_config = kwargs["training_config"]
    preflight = kwargs["preflight_report"]
    binding = kwargs["binding"]
    output = kwargs["output"]
    limits = kwargs["limits"]
    device = kwargs["device"]
    precision = kwargs["precision"]
    lineage = kwargs["lineage"]
    resume_checkpoint = kwargs["resume_checkpoint"]
    from pcbparse.training.torch_ocr import (
        TorchOCRRecognitionDataset,
        ctc_collate,
    )
    from pcbparse.training.torch_ocr_detection import (
        TorchOCRDetectionDataset,
        detection_collate,
    )
    from pcbparse.training.torch_ocr_model import (
        build_detector,
        build_recognizer,
        compute_ctc_loss,
        compute_detector_loss,
        default_ocr_preprocess,
        detection_box_metrics,
        export_ocr_submodel,
        extract_text_boxes,
        greedy_ctc_decode,
        levenshtein_distance,
        recognition_output_lengths,
        save_ocr_bundle,
        verify_ocr_export,
    )

    source_root = _source_root(preflight)
    recognition_height = int(task_config.input_size[0])
    recognition_nominal_width = int(task_config.input_size[1])
    detector_size = int(getattr(task_config, "ocr_detector_size", 512) or 512)
    train_recognition = TorchOCRRecognitionDataset(
        binding.task_root,
        source_root=source_root,
        split="train",
        image_mode="lazy",
        image_height=recognition_height,
    )
    recognition_sampling_power = float(
        getattr(
            task_config,
            "ocr_recognition_balanced_sampling_power",
            0.0,
        )
    )
    recognition_sample_weights = (
        train_recognition.alphanumeric_prefix_sampling_weights(
            power=recognition_sampling_power,
            maximum_weight=float(
                getattr(
                    task_config,
                    "ocr_recognition_balanced_sampling_max_weight",
                    8.0,
                )
            ),
        )
        if recognition_sampling_power > 0.0
        else None
    )
    if recognition_sample_weights is None:
        lineage["recognition_train_sampling"] = {
            "mode": "uniform_shuffle_without_replacement",
            "sample_count": len(train_recognition),
        }
    else:
        lineage["recognition_train_sampling"] = {
            "mode": "rare_alphanumeric_prefix_weighted_with_replacement_v1",
            "sample_count": len(train_recognition),
            "power": recognition_sampling_power,
            "maximum_weight": float(
                getattr(
                    task_config,
                    "ocr_recognition_balanced_sampling_max_weight",
                    8.0,
                )
            ),
            "observed_min_weight": min(recognition_sample_weights),
            "observed_mean_weight": sum(recognition_sample_weights)
            / len(recognition_sample_weights),
            "observed_max_weight": max(recognition_sample_weights),
            "upweighted_records": sum(
                weight > 1.0 for weight in recognition_sample_weights
            ),
            "replacement": True,
            "validation_sampling": "sequential_unweighted",
            "test_rows_used": 0,
        }
    val_inventory = _ocr_inventory(
        binding.task_root,
        train_recognition.charset,
        split="val",
    )
    val_details, val_detail_audit = _ocr_val_detail_index(
        preflight,
        val_inventory,
        expected_sha256=training_config.dataset.task_manifest_sha256["ocr_text"],
    )
    lineage["validation_detail_join"] = val_detail_audit
    # This exact equality is the anti-leakage gate: the recognizer vocabulary
    # must be derivable from included train rows alone, never val/test GT.
    train_inventory = _ocr_inventory(
        binding.task_root,
        train_recognition.charset,
        split="train",
    )
    if train_inventory.oov_records:
        raise TorchTrainingError("train-only OCR charset does not cover its own train rows")
    val_recognition = _OCRRawEvaluationDataset(
        train_recognition,
        val_inventory,
        image_height=recognition_height,
    )
    train_detection = TorchOCRDetectionDataset(
        binding.task_root,
        source_root=source_root,
        split="train",
        resize=detector_size,
        target_shrink_ratio=float(
            getattr(task_config, "ocr_target_shrink_ratio", 0.0) or 0.0
        ),
    )
    val_detection = TorchOCRDetectionDataset(
        binding.task_root,
        source_root=source_root,
        split="val",
        resize=detector_size,
        target_shrink_ratio=float(
            getattr(task_config, "ocr_target_shrink_ratio", 0.0) or 0.0
        ),
    )
    for dataset, label in (
        (train_recognition, "OCR recognition train"),
        (val_recognition, "OCR recognition val"),
        (train_detection, "OCR detection train"),
        (val_detection, "OCR detection val"),
    ):
        if len(dataset) <= 0:
            raise TorchTrainingError(f"{label} split is empty")

    train_recognition_loader = _data_loader(
        torch,
        train_recognition,
        task_config,
        training_config,
        shuffle=True,
        collate_fn=ctc_collate,
        device=device,
        sample_weights=recognition_sample_weights,
    )
    val_recognition_loader = _data_loader(
        torch,
        val_recognition,
        task_config,
        training_config,
        shuffle=False,
        collate_fn=_ocr_raw_collate,
        device=device,
    )
    train_detection_loader = _data_loader(
        torch,
        train_detection,
        task_config,
        training_config,
        shuffle=True,
        collate_fn=detection_collate,
        device=device,
    )
    val_detection_loader = _data_loader(
        torch,
        val_detection,
        task_config,
        training_config,
        shuffle=False,
        collate_fn=detection_collate,
        device=device,
    )
    initialization = _mapping(lineage["initialization"], "OCR initialization")
    has_pretrained_bundle = (
        initialization.get("mode") == "pretrained_local_artifact"
    )
    # A smoke run must exercise the exact pretrained architecture instead of
    # silently substituting the reduced smoke model and skipping compatibility.
    detector = build_detector(
        base_channels=8 if has_pretrained_bundle or not limits.smoke else 4
    ).to(device)
    recognizer = build_recognizer(
        train_recognition.charset,
        base_channels=16 if has_pretrained_bundle or not limits.smoke else 4,
        hidden_size=64 if has_pretrained_bundle or not limits.smoke else 16,
    ).to(device)
    models = torch.nn.ModuleDict(
        {"detector": detector, "recognizer": recognizer}
    ).to(device)
    preprocess = default_ocr_preprocess()
    preprocess["detector_size"] = [detector_size, detector_size]
    preprocess["detector_threshold"] = float(task_config.ocr_detect_threshold)
    preprocess["detector_min_area"] = int(task_config.ocr_min_area)
    preprocess["detector_target_shrink_ratio"] = float(
        getattr(task_config, "ocr_target_shrink_ratio", 0.0) or 0.0
    )
    preprocess["recognition_height"] = recognition_height
    if has_pretrained_bundle:
        initialization["load_audit"] = _load_ocr_pretrained_bundle(
            detector=detector,
            recognizer=recognizer,
            charset=train_recognition.charset,
            preprocess=preprocess,
            initialization=initialization,
            expected_architecture=str(task_config.architecture),
            expected_input_size=tuple(task_config.input_size),
        )
        lineage["initialization"] = initialization
    optimizer = torch.optim.AdamW(
        models.parameters(),
        lr=float(task_config.learning_rate),
        weight_decay=float(task_config.weight_decay),
    )
    scaler = _grad_scaler(torch, device, precision)
    log_path = output / "training_log.jsonl"
    if resume_checkpoint is None:
        log_path.unlink(missing_ok=True)

    def train_epoch(_epoch: int) -> dict[str, float]:
        detection_metrics = _ocr_detector_train_epoch(
            torch,
            detector,
            train_detection_loader,
            optimizer,
            scaler,
            device,
            precision,
            task_config,
            compute_detector_loss,
            limits.max_train_batches,
        )
        recognition_metrics = _ocr_recognizer_train_epoch(
            torch,
            recognizer,
            train_recognition_loader,
            optimizer,
            scaler,
            device,
            precision,
            task_config,
            compute_ctc_loss,
            recognition_output_lengths,
            limits.max_train_batches,
        )
        return {
            **{f"detector_{name}": value for name, value in detection_metrics.items()},
            **{f"recognizer_{name}": value for name, value in recognition_metrics.items()},
        }

    def validate(_epoch: int) -> dict[str, float]:
        return _ocr_evaluate(
            torch=torch,
            detector=detector,
            recognizer=recognizer,
            charset=train_recognition.charset,
            detection_loader=val_detection_loader,
            recognition_loader=val_recognition_loader,
            inventory=val_inventory,
            details_by_record=val_details,
            device=device,
            task_config=task_config,
            compute_detector_loss=compute_detector_loss,
            detection_box_metrics=detection_box_metrics,
            extract_text_boxes=extract_text_boxes,
            compute_ctc_loss=compute_ctc_loss,
            recognition_output_lengths=recognition_output_lengths,
            greedy_ctc_decode=greedy_ctc_decode,
            levenshtein_distance=levenshtein_distance,
            max_batches=limits.max_val_batches,
            predictions_path=None,
        )

    def checkpoint(path: Path, epoch: int, metrics: Mapping[str, float]) -> None:
        save_ocr_bundle(
            path,
            detector,
            recognizer,
            charset=train_recognition.charset,
            preprocess=preprocess,
            lineage=lineage,
            detector_optimizer=optimizer,
            recognizer_optimizer=None,
            epoch=epoch,
            global_step=epoch
            * (len(train_detection_loader) + len(train_recognition_loader)),
            metrics=metrics,
        )

    epoch_result = _run_epochs(
        torch=torch,
        model=models,
        optimizer=optimizer,
        task_config=task_config,
        training_config=training_config,
        limits=limits,
        train_epoch=train_epoch,
        validate=validate,
        save_checkpoint=checkpoint,
        output=output,
        log_path=log_path,
        device=device,
        scaler=scaler,
        loaders=(
            train_detection_loader,
            train_recognition_loader,
            val_detection_loader,
            val_recognition_loader,
        ),
        lineage=lineage,
        resume_checkpoint=resume_checkpoint,
    )
    predictions_path = output / "predictions.jsonl"
    final_metrics = _ocr_evaluate(
        torch=torch,
        detector=detector,
        recognizer=recognizer,
        charset=train_recognition.charset,
        detection_loader=val_detection_loader,
        recognition_loader=val_recognition_loader,
        inventory=val_inventory,
        details_by_record=val_details,
        device=device,
        task_config=task_config,
        compute_detector_loss=compute_detector_loss,
        detection_box_metrics=detection_box_metrics,
        extract_text_boxes=extract_text_boxes,
        compute_ctc_loss=compute_ctc_loss,
        recognition_output_lengths=recognition_output_lengths,
        greedy_ctc_decode=greedy_ctc_decode,
        levenshtein_distance=levenshtein_distance,
        max_batches=limits.max_val_batches,
        predictions_path=predictions_path,
    )
    bundle_path = output / "model.pt"
    save_ocr_bundle(
        bundle_path,
        detector,
        recognizer,
        charset=train_recognition.charset,
        preprocess=preprocess,
        lineage=lineage,
        detector_optimizer=None,
        recognizer_optimizer=None,
        epoch=epoch_result.best_epoch,
        global_step=epoch_result.best_epoch
        * (len(train_detection_loader) + len(train_recognition_loader)),
        metrics=final_metrics,
    )
    detector_samples = _ocr_detector_export_samples(
        val_detection_loader, limits.export_samples, limits.max_val_batches
    )
    recognizer_samples = _ocr_recognizer_export_samples(
        val_recognition_loader, limits.export_samples, limits.max_val_batches
    )
    export_format = str(task_config.export_format or "onnx")
    detector_path = output / (
        "detector.ts" if export_format == "torchscript" else "detector.onnx"
    )
    recognizer_path = output / (
        "recognizer.ts" if export_format == "torchscript" else "recognizer.onnx"
    )
    tolerance = _export_tolerance(task_config, limits)
    detector_report = export_ocr_submodel(
        detector,
        detector_path,
        kind="detector",
        format=export_format,  # type: ignore[arg-type]
        input_shape=tuple(detector_samples[0].shape),
        device="cpu",
        tolerance=tolerance,
        opset_version=int(task_config.export_opset or 17),
    )
    recognizer_report = export_ocr_submodel(
        recognizer,
        recognizer_path,
        kind="recognizer",
        format=export_format,  # type: ignore[arg-type]
        input_shape=tuple(recognizer_samples[0].shape),
        device="cpu",
        tolerance=tolerance,
        opset_version=int(task_config.export_opset or 17),
    )
    maximum_error = max(
        float(detector_report.max_abs_error),
        float(recognizer_report.max_abs_error),
    )
    detector.to("cpu")
    recognizer.to("cpu")
    for model, path, samples in (
        (detector, detector_path, detector_samples),
        (recognizer, recognizer_path, recognizer_samples),
    ):
        for sample in samples:
            consistency = verify_ocr_export(
                model,
                path,
                format=export_format,  # type: ignore[arg-type]
                inputs=sample,
                tolerance=tolerance,
            )
            maximum_error = max(
                maximum_error, float(consistency["max_abs_error"])
            )
    export_manifest = output / "ocr_exports.json"
    _write_json(
        export_manifest,
        {
            "schema_version": 1,
            "kind": "pcbparse_ocr_two_stage_onnx_exports",
            "sample_count_per_submodel": limits.export_samples,
            "tolerance": tolerance,
            "maximum_absolute_error": maximum_error,
            "detector": {
                "path": detector_path.name,
                "sha256": _sha256_file(detector_path),
                "report": detector_report.to_dict(),
            },
            "recognizer": {
                "path": recognizer_path.name,
                "sha256": _sha256_file(recognizer_path),
                "report": recognizer_report.to_dict(),
            },
        },
    )
    _write_metrics(output, final_metrics, epoch_result, limits)
    return _TaskOutcome(
        artifact_path=bundle_path,
        export_path=export_manifest,
        adapter_entrypoint=_ADAPTERS["ocr"],
        artifact_precision="fp32",
        export_backend=(
            "torchscript-two-stage"
            if export_format == "torchscript"
            else "onnxruntime-two-stage"
        ),
        export_sample_count=limits.export_samples,
        export_max_abs_error=maximum_error,
        metrics=final_metrics,
        predictions_path=predictions_path,
        planned_epochs=limits.epochs,
        completed_epochs=epoch_result.completed_epochs,
        early_stopped=epoch_result.early_stopped,
        best_epoch=epoch_result.best_epoch,
        limitations=(
            "The CTC charset is derived exactly from included train rows; validation OOV records remain explicit errors and are never added to the vocabulary.",
            f"Validation contains {val_inventory.oov_records} charset-OOV records across {len(val_inventory.oov_characters)} unique code points.",
            "OCR detection and recognition submodel exports are verification companions; the production adapter loads the unified model.pt bundle.",
            "Validation is limited to the frozen Dataset300 validation split.",
            "Candidate status does not authorize models.lock blessing.",
        ),
    )


def _train_wire(**kwargs: Any) -> _TaskOutcome:
    torch = kwargs["torch"]
    task_config = kwargs["task_config"]
    training_config = kwargs["training_config"]
    preflight = kwargs["preflight_report"]
    binding = kwargs["binding"]
    output = kwargs["output"]
    limits = kwargs["limits"]
    device = kwargs["device"]
    precision = kwargs["precision"]
    lineage = kwargs["lineage"]
    resume_checkpoint = kwargs["resume_checkpoint"]
    from pcbparse.training.torch_wire_junction import WireTorchDataset, wire_collate
    from pcbparse.training.torch_wire_model import (
        WireLossConfig,
        build_tiny_unet,
        compute_wire_loss,
        export_wire_model,
        save_wire_checkpoint,
        verify_wire_export,
        wire_centerline_metrics,
    )

    source_root = _source_root(preflight)
    run_dir = binding.task_root
    train_dataset = WireTorchDataset(run_dir, source_root=source_root, split="train")
    val_dataset = WireTorchDataset(run_dir, source_root=source_root, split="val")
    if not train_dataset.metadata.wire_training_ready:
        raise TorchTrainingError("wire materialization has no eligible training records")
    _require_nonempty_splits(train_dataset, val_dataset, "wire")
    train_loader = _data_loader(
        torch,
        train_dataset,
        task_config,
        training_config,
        shuffle=True,
        collate_fn=wire_collate,
        device=device,
    )
    val_loader = _data_loader(
        torch,
        val_dataset,
        task_config,
        training_config,
        shuffle=False,
        collate_fn=wire_collate,
        device=device,
    )
    model = build_tiny_unet(base_channels=4 if limits.smoke else 16).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(task_config.learning_rate),
        weight_decay=float(task_config.weight_decay),
    )
    scaler = _grad_scaler(torch, device, precision)
    loss_config = WireLossConfig(skeleton_iterations=4 if limits.smoke else 12)
    log_path = output / "training_log.jsonl"
    if resume_checkpoint is None:
        log_path.unlink(missing_ok=True)

    def train_epoch(_epoch: int) -> dict[str, float]:
        return _wire_train_epoch(
            torch,
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            precision,
            task_config,
            loss_config,
            compute_wire_loss,
            limits.max_train_batches,
        )

    def validate(_epoch: int) -> dict[str, float]:
        metrics, _ = _wire_evaluate(
            torch,
            model,
            val_loader,
            device,
            loss_config,
            compute_wire_loss,
            wire_centerline_metrics,
            max_batches=limits.max_val_batches,
            predictions_path=None,
        )
        return metrics

    def checkpoint(path: Path, epoch: int, metrics: Mapping[str, float]) -> None:
        save_wire_checkpoint(
            path,
            model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=epoch * len(train_loader),
            lineage=lineage,
            metrics=metrics,
        )

    epoch_result = _run_epochs(
        torch=torch,
        model=model,
        optimizer=optimizer,
        task_config=task_config,
        training_config=training_config,
        limits=limits,
        train_epoch=train_epoch,
        validate=validate,
        save_checkpoint=checkpoint,
        output=output,
        log_path=log_path,
        device=device,
        scaler=scaler,
        loaders=(train_loader, val_loader),
        lineage=lineage,
        resume_checkpoint=resume_checkpoint,
    )
    predictions_path = output / "predictions.jsonl"
    final_metrics, _ = _wire_evaluate(
        torch,
        model,
        val_loader,
        device,
        loss_config,
        compute_wire_loss,
        wire_centerline_metrics,
        max_batches=limits.max_val_batches,
        predictions_path=predictions_path,
    )
    checkpoint_path = output / "model.pt"
    save_wire_checkpoint(
        checkpoint_path,
        model,
        optimizer=None,
        epoch=epoch_result.best_epoch,
        global_step=epoch_result.best_epoch * len(train_loader),
        lineage=lineage,
        metrics=final_metrics,
    )
    samples = _wire_export_samples(
        val_loader, limits.export_samples, limits.max_val_batches
    )
    export_format = str(task_config.export_format or "onnx")
    export_path = output / (
        "model.pt" if export_format == "torchscript" else "model.onnx"
    )
    first = samples[0]
    export_report = export_wire_model(
        model,
        export_path,
        format=export_format,  # type: ignore[arg-type]
        input_shape=tuple(first.shape),
        device="cpu",
        tolerance=_export_tolerance(task_config, limits),
        verify=True,
        opset_version=int(task_config.export_opset or 17),
    )
    maximum_error = float(export_report.max_abs_error)
    model.to("cpu")
    for sample in samples:
        result = verify_wire_export(
            model,
            export_path,
            format=export_format,  # type: ignore[arg-type]
            inputs=sample.to("cpu"),
            tolerance=_export_tolerance(task_config, limits),
        )
        maximum_error = max(maximum_error, float(result["max_abs_error"]))
    _write_metrics(output, final_metrics, epoch_result, limits)
    return _TaskOutcome(
        artifact_path=export_path,
        export_path=export_path,
        adapter_entrypoint=_ADAPTERS["wire"],
        artifact_precision="fp32",
        export_backend=(
            "torchscript" if export_format == "torchscript" else "onnxruntime"
        ),
        export_sample_count=len(samples),
        export_max_abs_error=maximum_error,
        metrics=final_metrics,
        predictions_path=predictions_path,
        planned_epochs=limits.epochs,
        completed_epochs=epoch_result.completed_epochs,
        early_stopped=epoch_result.early_stopped,
        best_epoch=epoch_result.best_epoch,
        limitations=(
            "The learned output is a local wire probability mask; deterministic vectorization owns topology extraction.",
            "Validation is limited to eligible frozen Dataset300 wire tiles.",
            "Candidate status does not authorize models.lock blessing.",
        ),
    )


def _junction_train_epoch(
    torch: Any,
    model: Any,
    loader: Any,
    optimizer: Any,
    *,
    device: str | Any = "cpu",
    class_weights: Any | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    from pcbparse.training.torch_junction_model import junction_train_step

    totals: dict[str, float] = {}
    examples = 0
    for batch in _bounded_iterable(loader, max_batches):
        metrics = junction_train_step(
            model,
            batch,
            optimizer,
            device=device,
            class_weights=class_weights,
        )
        count = int(batch["image"].shape[0])
        examples += count
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value) * count
    if examples == 0:
        raise TorchTrainingError("junction training loader produced no examples")
    return {name: value / examples for name, value in sorted(totals.items())}


def _train_junction(**kwargs: Any) -> _TaskOutcome:
    torch = kwargs["torch"]
    task_config = kwargs["task_config"]
    training_config = kwargs["training_config"]
    output = kwargs["output"]
    limits = kwargs["limits"]
    device = kwargs["device"]
    precision = kwargs["precision"]
    lineage = kwargs["lineage"]
    resume_checkpoint = kwargs["resume_checkpoint"]
    binding = kwargs["binding"]
    from pcbparse.training.synthetic_junction import (
        MixedJunctionTrainingDataset,
        SyntheticJunctionDataset,
    )
    from pcbparse.training.torch_junction_model import (
        build_junction_model,
        evaluate_junction_model,
        export_junction_model,
        junction_class_weights_from_counts,
        save_junction_checkpoint,
        verify_junction_export,
    )
    from pcbparse.training.torch_wire_junction import (
        JUNCTION_CLASSES,
        JunctionTorchDataset,
        junction_collate,
    )

    run_dir = binding.task_root
    real_train = JunctionTorchDataset(run_dir, split="train")
    real_val = JunctionTorchDataset(run_dir, split="val")
    if len(real_val) == 0:
        raise TorchTrainingError(
            "junction materialization has no trusted validation records"
        )
    synthetic_per_class = int(task_config.junction_synthetic_per_class or 0)
    if len(real_train) == 0 and synthetic_per_class <= 0:
        raise TorchTrainingError(
            "junction materialization has no trusted training records "
            "and synthetic fill is disabled"
        )
    train_dataset: Any = real_train
    if synthetic_per_class > 0:
        counts = {
            name: max(
                0,
                synthetic_per_class
                - int(real_train.metadata.class_counts.get(name, 0)),
            )
            for name in JUNCTION_CLASSES
        }
        synthetic = SyntheticJunctionDataset(
            samples_per_class=counts, classes=JUNCTION_CLASSES
        )
        train_dataset = MixedJunctionTrainingDataset(real_train, synthetic)
        lineage = {**lineage}
        lineage["junction_synthetic"] = {
            "policy": synthetic.config.policy_sha256,
            "synthetic_records": len(synthetic),
            "per_class_counts": dict(synthetic.class_counts),
        }
    class_counts = dict(getattr(train_dataset, "class_counts", {}))
    class_weights = None
    if all(class_counts.get(name, 0) > 0 for name in JUNCTION_CLASSES):
        class_weights = junction_class_weights_from_counts(class_counts)

    train_loader = _data_loader(
        torch,
        train_dataset,
        task_config,
        training_config,
        shuffle=True,
        collate_fn=junction_collate,
        device=device,
    )
    val_loader = _data_loader(
        torch,
        real_val,
        task_config,
        training_config,
        shuffle=False,
        collate_fn=junction_collate,
        device=device,
    )
    model = build_junction_model(
        base_channels=8 if limits.smoke else 16
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(task_config.learning_rate),
        weight_decay=float(task_config.weight_decay),
    )
    scaler = _grad_scaler(torch, device, precision)
    log_path = output / "training_log.jsonl"
    if resume_checkpoint is None:
        log_path.unlink(missing_ok=True)

    def train_epoch(_epoch: int) -> dict[str, float]:
        return _junction_train_epoch(
            torch,
            model,
            train_loader,
            optimizer,
            device=device,
            class_weights=class_weights,
            max_batches=limits.max_train_batches,
        )

    def validate(_epoch: int) -> dict[str, float]:
        report = evaluate_junction_model(
            model,
            _bounded_iterable(val_loader, limits.max_val_batches),
            device=device,
        )
        return report["metrics"]

    def checkpoint(path: Path, epoch: int, metrics: Mapping[str, float]) -> None:
        save_junction_checkpoint(
            path,
            model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=epoch * len(train_loader),
            lineage=lineage,
            metrics=metrics,
        )

    epoch_result = _run_epochs(
        torch=torch,
        model=model,
        optimizer=optimizer,
        task_config=task_config,
        training_config=training_config,
        limits=limits,
        train_epoch=train_epoch,
        validate=validate,
        save_checkpoint=checkpoint,
        output=output,
        log_path=log_path,
        device=device,
        scaler=scaler,
        loaders=(train_loader, val_loader),
        lineage=lineage,
        resume_checkpoint=resume_checkpoint,
    )
    predictions_path = output / "predictions.jsonl"
    final_report = evaluate_junction_model(
        model,
        _bounded_iterable(val_loader, limits.max_val_batches),
        device=device,
    )
    final_metrics = final_report["metrics"]
    for row in final_report["predictions"]:
        _append_jsonl(predictions_path, row)
    checkpoint_path = output / "model.pt"
    save_junction_checkpoint(
        checkpoint_path,
        model,
        optimizer=None,
        epoch=epoch_result.best_epoch,
        global_step=epoch_result.best_epoch * len(train_loader),
        lineage=lineage,
        metrics=final_metrics,
    )
    samples: list[Any] = []
    for batch in _bounded_iterable(val_loader, limits.max_val_batches):
        for item in batch["image"]:
            samples.append(item.detach().cpu())
            if len(samples) >= limits.export_samples:
                break
        if len(samples) >= limits.export_samples:
            break
    if not samples:
        raise TorchTrainingError("junction validation loader produced no export samples")
    export_path = output / "model.onnx"
    first = samples[0].unsqueeze(0)
    export_report = export_junction_model(
        model,
        export_path,
        format="onnx",
        input_shape=tuple(first.shape),
        device="cpu",
        tolerance=_export_tolerance(task_config, limits),
        verify=True,
        opset_version=int(task_config.export_opset or 17),
    )
    maximum_error = float(export_report.max_abs_error)
    model.to("cpu")
    for sample in samples:
        result = verify_junction_export(
            model,
            export_path,
            format="onnx",
            inputs=sample.unsqueeze(0).to("cpu"),
            tolerance=_export_tolerance(task_config, limits),
        )
        maximum_error = max(maximum_error, float(result["max_abs_error"]))
    _write_metrics(output, final_metrics, epoch_result, limits)
    return _TaskOutcome(
        artifact_path=export_path,
        export_path=export_path,
        adapter_entrypoint=_ADAPTERS["junction"],
        artifact_precision="fp32",
        export_backend="onnxruntime",
        export_sample_count=len(samples),
        export_max_abs_error=maximum_error,
        metrics=final_metrics,
        predictions_path=predictions_path,
        planned_epochs=limits.epochs,
        completed_epochs=epoch_result.completed_epochs,
        early_stopped=epoch_result.early_stopped,
        best_epoch=epoch_result.best_epoch,
        limitations=(
            "Validation uses trusted real junction patches only; synthetic rows are train-only.",
            "connected_cross/non_connected_cross/noise trusted labels are absent from Dataset300; "
            "the classifier is initialized with deterministic synthetic supervision.",
            "Candidate status does not authorize models.lock blessing.",
        ),
    )


def _run_epochs(
    *,
    torch: Any,
    model: Any,
    optimizer: Any,
    task_config: Any,
    training_config: Any,
    limits: _RunLimits,
    train_epoch: Callable[[int], dict[str, float]],
    validate: Callable[[int], dict[str, float]],
    save_checkpoint: Callable[[Path, int, Mapping[str, float]], None],
    output: Path,
    log_path: Path,
    device: Any,
    scaler: Any,
    loaders: Sequence[Any],
    lineage: Mapping[str, Any],
    resume_checkpoint: Path | None,
) -> _EpochResult:
    monitor_gate = task_config.evaluation_gates[0]
    selection_policy = _model_selection_policy(task_config)
    patience = int(training_config.runtime.early_stopping_patience)
    checkpoint_every = int(training_config.runtime.checkpoint_every_epochs)
    best_value: float | None = None
    best_selection_key: tuple[float, ...] | None = None
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    completed = 0
    early_stopped = False
    history: list[dict[str, Any]] = []
    if resume_checkpoint is not None:
        restored = _restore_training_state(
            torch=torch,
            path=resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            loaders=loaders,
            task_config=task_config,
            limits=limits,
            lineage=lineage,
        )
        completed = restored["completed_epoch"]
        best_epoch = restored["best_epoch"]
        best_value = restored["best_value"]
        best_state = restored["best_model_state"]
        stale_epochs = restored["stale_epochs"]
        history = restored["history"]
        best_selection_key = _restored_selection_key(
            task_config.evaluation_gates,
            history,
            best_epoch=best_epoch,
            best_value=best_value,
            stale_epochs=stale_epochs,
        )
        _write_training_history(log_path, history)
        early_stopped = stale_epochs >= patience and completed < limits.epochs
    for epoch in range(completed + 1, limits.epochs + 1):
        if early_stopped:
            break
        train_metrics = train_epoch(epoch)
        val_metrics = validate(epoch)
        selection = _gate_selection_summary(
            task_config.evaluation_gates, val_metrics
        )
        value = float(selection["primary_value"])
        selection_key = tuple(float(item) for item in selection["key"])
        improved = _selection_is_improved(
            task_config.evaluation_gates,
            selection,
            best_key=best_selection_key,
            best_primary_value=best_value,
        )
        if improved:
            best_value = value
            best_selection_key = selection_key
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            save_checkpoint(output / "best_checkpoint.pt", epoch, val_metrics)
        else:
            stale_epochs += 1
        if epoch % checkpoint_every == 0 or epoch == limits.epochs:
            save_checkpoint(
                output / f"checkpoint_epoch_{epoch:03d}.pt", epoch, val_metrics
            )
        memory = _cuda_memory(torch, device)
        epoch_log = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "monitor": {
                "metric": monitor_gate.metric,
                "operator": monitor_gate.operator,
                "value": value,
                "improved": improved,
                "stale_epochs": stale_epochs,
                "selection_policy": selection_policy["name"],
                "all_gates_passed": selection["all_gates_passed"],
                "failed_gate_count": selection["failed_gate_count"],
                "failed_gates": selection["failed_gates"],
                "maximum_normalized_violation": selection[
                    "maximum_normalized_violation"
                ],
                "total_normalized_violation": selection[
                    "total_normalized_violation"
                ],
                "selection_key": selection["key"],
            },
            "optimizer": {
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "gradient_accumulation": int(task_config.gradient_accumulation),
            },
            "cuda_memory": memory,
        }
        _append_jsonl(log_path, epoch_log)
        history.append(epoch_log)
        completed = epoch
        _save_training_state(
            torch=torch,
            path=output / "resume_checkpoint.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            loaders=loaders,
            task_config=task_config,
            limits=limits,
            lineage=lineage,
            completed_epoch=completed,
            best_epoch=best_epoch,
            best_value=best_value,
            best_model_state=best_state,
            stale_epochs=stale_epochs,
            history=history,
        )
        if stale_epochs >= patience and epoch < limits.epochs:
            early_stopped = True
            break
    if best_state is None or best_epoch <= 0:
        raise TorchTrainingError("no finite best checkpoint was produced")
    model.load_state_dict(best_state, strict=True)
    model.to(device)
    return _EpochResult(
        best_epoch=best_epoch,
        completed_epochs=completed,
        early_stopped=early_stopped,
    )


def _component_train_epoch(
    torch: Any,
    model: Any,
    loader: Any,
    optimizer: Any,
    scaler: Any,
    device: Any,
    precision: str,
    accumulation: int,
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    examples = 0
    skipped_non_finite = 0
    total_batches = _bounded_length(loader, max_batches)
    optimizer.zero_grad(set_to_none=True)
    for batch_index, (images, raw_targets) in enumerate(
        _bounded_iterable(loader, max_batches), start=1
    ):
        images_list = [image.to(device=device, dtype=torch.float32) for image in images]
        targets = [
            {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in target.items()
            }
            for target in raw_targets
        ]
        group_start = ((batch_index - 1) // accumulation) * accumulation + 1
        group_size = min(accumulation, total_batches - group_start + 1)
        with _autocast(torch, device, precision):
            losses = model(images_list, targets)
            total_loss = sum(losses.values())
            scaled_loss = total_loss / group_size
        # Skip non-finite loss batches instead of crashing
        if not math.isfinite(float(total_loss.detach().cpu())):
            skipped_non_finite += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        _backward(scaler, scaled_loss)
        if batch_index % accumulation == 0 or batch_index == total_batches:
            _optimizer_step(torch, scaler, optimizer, model.parameters())
        batch_examples = len(images_list)
        examples += batch_examples
        for name, value in {**losses, "loss_total": total_loss}.items():
            scalar = float(value.detach().cpu())
            totals[name] = totals.get(name, 0.0) + scalar * batch_examples
    if examples == 0:
        raise TorchTrainingError("component training loader produced no examples")
    return {name: value / examples for name, value in sorted(totals.items())}


def _pin_train_epoch(
    torch: Any,
    model: Any,
    loader: Any,
    optimizer: Any,
    scaler: Any,
    device: Any,
    precision: str,
    task_config: Any,
    max_batches: int | None,
) -> dict[str, float]:
    from pcbparse.training.torch_pin_model import pin_losses, PinLossConfig

    # Use weighted BCE loss for better convergence on sparse keypoints
    # The heatmap is very sparse (~1% positive), so we need higher weight for positive samples
    pin_loss_config = PinLossConfig(
        heatmap_loss=str(task_config.pin_heatmap_loss),
        heatmap_weight=2.0,
        offset_weight=0.5,
        orientation_weight=0.3,
        count_loss_weight=float(task_config.pin_count_loss_weight),
        count_target_multiplier=float(task_config.pin_count_target_multiplier),
    )

    model.train()
    totals: dict[str, float] = {}
    examples = 0
    accumulation = int(task_config.gradient_accumulation)
    total_batches = _bounded_length(loader, max_batches)
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(
        _bounded_iterable(loader, max_batches), start=1
    ):
        images, targets = _prepare_pin_batch(torch, batch, task_config, device)
        group_start = ((batch_index - 1) // accumulation) * accumulation + 1
        group_size = min(accumulation, total_batches - group_start + 1)
        with _autocast(torch, device, precision):
            losses = pin_losses(model(images), targets, config=pin_loss_config)
            scaled_loss = losses["loss"] / group_size
        _backward(scaler, scaled_loss)
        if batch_index % accumulation == 0 or batch_index == total_batches:
            grad_clip = float(getattr(task_config, "pin_grad_clip", 1.0) or 0.0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip
                )
            _optimizer_step(torch, scaler, optimizer, model.parameters())
        batch_examples = int(images.shape[0])
        examples += batch_examples
        for name, value in losses.items():
            if name == "positive_count":
                continue
            scalar = float(value.detach().cpu())
            if not math.isfinite(scalar):
                raise TorchTrainingError(f"non-finite pin loss {name}={scalar}")
            totals[name] = totals.get(name, 0.0) + scalar * batch_examples
    if examples == 0:
        raise TorchTrainingError("pin training loader produced no examples")
    return {name: value / examples for name, value in sorted(totals.items())}


def _wire_train_epoch(
    torch: Any,
    model: Any,
    loader: Any,
    optimizer: Any,
    scaler: Any,
    device: Any,
    precision: str,
    task_config: Any,
    loss_config: Any,
    compute_wire_loss: Callable[..., Mapping[str, Any]],
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    examples = 0
    accumulation = int(task_config.gradient_accumulation)
    total_batches = _bounded_length(loader, max_batches)
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(
        _bounded_iterable(loader, max_batches), start=1
    ):
        image = batch["image"].to(device=device, dtype=torch.float32)
        target = batch["target"].to(device=device, dtype=torch.float32)
        ignore = batch["ignore"].to(device=device, dtype=torch.float32)
        group_start = ((batch_index - 1) // accumulation) * accumulation + 1
        group_size = min(accumulation, total_batches - group_start + 1)
        with _autocast(torch, device, precision):
            logits = model(image)
            losses = compute_wire_loss(logits, target, ignore, config=loss_config)
            scaled_loss = losses["loss"] / group_size
        _backward(scaler, scaled_loss)
        if batch_index % accumulation == 0 or batch_index == total_batches:
            _optimizer_step(torch, scaler, optimizer, model.parameters())
        batch_examples = int(image.shape[0])
        examples += batch_examples
        for name, value in losses.items():
            if name == "valid_pixels":
                continue
            scalar = float(value.detach().cpu())
            if not math.isfinite(scalar):
                raise TorchTrainingError(f"non-finite wire loss {name}={scalar}")
            totals[name] = totals.get(name, 0.0) + scalar * batch_examples
    if examples == 0:
        raise TorchTrainingError("wire training loader produced no examples")
    return {name: value / examples for name, value in sorted(totals.items())}


def _ocr_detector_train_epoch(
    torch: Any,
    model: Any,
    loader: Any,
    optimizer: Any,
    scaler: Any,
    device: Any,
    precision: str,
    task_config: Any,
    compute_detector_loss: Callable[..., Mapping[str, Any]],
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    examples = 0
    accumulation = int(task_config.gradient_accumulation)
    total_batches = _bounded_length(loader, max_batches)
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(
        _bounded_iterable(loader, max_batches), start=1
    ):
        images = batch["images"].to(device=device, dtype=torch.float32)
        target = batch["target"].to(device=device, dtype=torch.float32)
        ignore = batch["ignore"].to(device=device, dtype=torch.float32)
        group_start = ((batch_index - 1) // accumulation) * accumulation + 1
        group_size = min(accumulation, total_batches - group_start + 1)
        with _autocast(torch, device, precision):
            terms = compute_detector_loss(model(images), target, ignore)
            scaled_loss = terms["loss"] / group_size
        _backward(scaler, scaled_loss)
        if batch_index % accumulation == 0 or batch_index == total_batches:
            _optimizer_step(torch, scaler, optimizer, model.parameters())
        count = int(images.shape[0])
        examples += count
        for name, value in terms.items():
            if name == "valid_pixels":
                continue
            scalar = float(value.detach().cpu())
            if not math.isfinite(scalar):
                raise TorchTrainingError(f"non-finite OCR detector {name}={scalar}")
            totals[name] = totals.get(name, 0.0) + scalar * count
    if examples == 0:
        raise TorchTrainingError("OCR detector training loader produced no examples")
    return {name: value / examples for name, value in sorted(totals.items())}


def _ocr_recognizer_train_epoch(
    torch: Any,
    model: Any,
    loader: Any,
    optimizer: Any,
    scaler: Any,
    device: Any,
    precision: str,
    task_config: Any,
    compute_ctc_loss: Callable[..., Any],
    recognition_output_lengths: Callable[[Any], Any],
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    loss_total = 0.0
    examples = 0
    codepoints = 0
    skipped_ctc = 0
    accumulation = int(task_config.gradient_accumulation)
    total_batches = _bounded_length(loader, max_batches)
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(
        _bounded_iterable(loader, max_batches), start=1
    ):
        images = batch["images"].to(device=device, dtype=torch.float32)
        targets = batch["targets"].to(device=device, dtype=torch.long)
        target_lengths = batch["target_lengths"].to(dtype=torch.long)
        image_widths = batch["image_widths"].to(dtype=torch.long)
        group_start = ((batch_index - 1) // accumulation) * accumulation + 1
        group_size = min(accumulation, total_batches - group_start + 1)

        # Filter samples that don't have enough CTC timesteps
        input_lengths = recognition_output_lengths(image_widths)
        valid_mask = _filter_ctc_valid_samples(
            torch, targets, target_lengths, input_lengths
        )
        if not bool(valid_mask.any()):
            skipped_ctc += int(images.shape[0])
            continue

        # Apply filter
        if not bool(valid_mask.all()):
            images = images[valid_mask]
            targets = targets[valid_mask]
            target_lengths = target_lengths[valid_mask]
            image_widths = image_widths[valid_mask]
            input_lengths = input_lengths[valid_mask]

        with _autocast(torch, device, precision):
            logits = model(images, image_widths.to(device))
        # CUDA CTC requires FP32 even when the convolution/LSTM forward uses
        # autocast.  The cast is differentiable and preserves AMP training.
        loss = compute_ctc_loss(
            logits.float(), targets, target_lengths, input_lengths
        )
        _backward(scaler, loss / group_size)
        if batch_index % accumulation == 0 or batch_index == total_batches:
            _optimizer_step(torch, scaler, optimizer, model.parameters())
        count = int(images.shape[0])
        scalar = float(loss.detach().cpu())
        if not math.isfinite(scalar):
            raise TorchTrainingError(f"non-finite OCR recognizer loss={scalar}")
        examples += count
        codepoints += int(target_lengths.sum())
        loss_total += scalar * count
    if examples == 0:
        raise TorchTrainingError("OCR recognizer training loader produced no examples")
    return {
        "loss": loss_total / examples,
        "samples": float(examples),
        "target_codepoints": float(codepoints),
        "skipped_ctc": float(skipped_ctc),
    }


def _filter_ctc_valid_samples(
    torch: Any,
    targets: Any,
    target_lengths: Any,
    input_lengths: Any,
) -> Any:
    """Filter samples that have enough CTC timesteps for their targets."""
    batch_size = int(targets.shape[0]) if targets.ndim > 1 else 1
    valid_mask = torch.ones(batch_size, dtype=torch.bool, device=targets.device)

    offset = 0
    for i in range(batch_size):
        target_len = int(target_lengths[i])
        sequence = targets[offset:offset + target_len].tolist()
        offset += target_len
        adjacent_repeats = sum(
            left == right for left, right in zip(sequence, sequence[1:])
        )
        required = target_len + adjacent_repeats
        if int(input_lengths[i]) < required:
            valid_mask[i] = False

    return valid_mask


def _ocr_evaluate(
    *,
    torch: Any,
    detector: Any,
    recognizer: Any,
    charset: Any,
    detection_loader: Any,
    recognition_loader: Any,
    inventory: _OCRInventory,
    details_by_record: Mapping[str, str],
    device: Any,
    task_config: Any,
    compute_detector_loss: Callable[..., Mapping[str, Any]],
    detection_box_metrics: Callable[..., Mapping[str, Any]],
    extract_text_boxes: Callable[..., Sequence[Any]],
    compute_ctc_loss: Callable[..., Any],
    recognition_output_lengths: Callable[[Any], Any],
    greedy_ctc_decode: Callable[..., tuple[list[str], list[float]]],
    levenshtein_distance: Callable[[str, str], int],
    max_batches: int | None,
    predictions_path: Path | None,
) -> dict[str, float]:
    detector.eval()
    recognizer.eval()
    inventory_ids = {str(row["record_id"]) for row in inventory.rows}
    if set(details_by_record) != inventory_ids:
        raise TorchTrainingError(
            "OCR val detail index differs from the frozen recognition inventory"
        )
    detail_totals: dict[str, dict[str, int]] = {
        detail: {
            "support": 0,
            "exact": 0,
            "edit_distance": 0,
            "reference_codepoints": 0,
        }
        for detail in sorted(set(details_by_record.values()))
    }
    if predictions_path is not None:
        predictions_path.unlink(missing_ok=True)
    detector_loss_total = 0.0
    detector_examples = 0
    detector_counts = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "ignored_predictions": 0,
        "targets": 0,
        "predictions": 0,
    }
    matched_iou_weighted = 0.0
    with torch.inference_mode():
        for batch in _bounded_iterable(detection_loader, max_batches):
            images = batch["images"].to(device=device, dtype=torch.float32)
            target = batch["target"].to(device=device, dtype=torch.float32)
            ignore = batch["ignore"].to(device=device, dtype=torch.float32)
            logits = detector(images)
            terms = compute_detector_loss(logits, target, ignore)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            local = detection_box_metrics(
                probabilities,
                ground_truth_boxes=batch["boxes"],
                target_masks=target.detach().cpu().numpy(),
                ignore_masks=ignore.detach().cpu().numpy(),
                ignore_boxes=batch["ignore_boxes"],
                threshold=float(task_config.ocr_detect_threshold),
                min_area=int(task_config.ocr_min_area),
                iou_threshold=float(task_config.ocr_iou_threshold),
            )
            count = int(images.shape[0])
            detector_examples += count
            detector_loss_total += float(terms["loss"].detach().cpu()) * count
            for name in detector_counts:
                detector_counts[name] += int(local[name])
            matched_iou_weighted += float(local["mean_matched_iou"]) * int(
                local["true_positives"]
            )
            if predictions_path is not None:
                for index, tile_id in enumerate(batch["tile_ids"]):
                    boxes = extract_text_boxes(
                        probabilities[index, 0],
                        threshold=float(task_config.ocr_detect_threshold),
                        min_area=int(task_config.ocr_min_area),
                    )
                    _append_jsonl(
                        predictions_path,
                        {
                            "kind": "ocr_detection",
                            "tile_id": tile_id,
                            "sample_id": batch["sample_ids"][index],
                            "boxes": [
                                {
                                    "xyxy": list(box.xyxy()),
                                    "score": float(box.score),
                                    "area": int(box.area),
                                }
                                for box in boxes
                            ],
                        },
                    )
    if detector_examples == 0:
        raise TorchTrainingError("OCR detector validation loader produced no examples")
    tp = detector_counts["true_positives"]
    fp = detector_counts["false_positives"]
    fn = detector_counts["false_negatives"]
    detector_precision = tp / max(1, tp + fp)
    detector_recall = tp / max(1, tp + fn)
    detector_hmean = (
        2.0 * detector_precision * detector_recall
        / (detector_precision + detector_recall)
        if detector_precision + detector_recall
        else 0.0
    )

    recognition_examples = 0
    recognition_exact = 0
    encodable_examples = 0
    encodable_exact = 0
    oov_examples = 0
    oov_exact = 0
    edit_distance_total = 0
    reference_codepoints = 0
    recognition_loss_total = 0.0
    recognition_loss_examples = 0
    with torch.inference_mode():
        for batch in _bounded_iterable(recognition_loader, max_batches):
            images = batch["images"].to(device=device, dtype=torch.float32)
            image_widths = batch["image_widths"].to(dtype=torch.long)
            logits = recognizer(images, image_widths.to(device))
            input_lengths = recognition_output_lengths(image_widths)
            predictions, scores = greedy_ctc_decode(logits, input_lengths, charset)
            references = batch["texts"]
            if len(predictions) != len(references):
                raise TorchTrainingError("OCR prediction/reference cardinality mismatch")
            encodable_indices = [
                index
                for index, target in enumerate(batch["targets"])
                if target is not None
            ]
            if encodable_indices:
                selected_logits = logits[:, encodable_indices, :].float()
                selected_targets = torch.cat(
                    [batch["targets"][index] for index in encodable_indices], dim=0
                ).to(device=device, dtype=torch.long)
                target_lengths = torch.tensor(
                    [
                        int(batch["targets"][index].numel())
                        for index in encodable_indices
                    ],
                    dtype=torch.long,
                )
                selected_input_lengths = input_lengths[encodable_indices]
                loss = compute_ctc_loss(
                    selected_logits,
                    selected_targets,
                    target_lengths,
                    selected_input_lengths,
                )
                recognition_loss_total += float(loss.detach().cpu()) * len(
                    encodable_indices
                )
                recognition_loss_examples += len(encodable_indices)
            for index, (prediction, reference, score) in enumerate(
                zip(predictions, references, scores, strict=True)
            ):
                record_id = str(batch["record_ids"][index])
                detail = details_by_record.get(record_id)
                if detail not in detail_totals:
                    raise TorchTrainingError(
                        f"OCR val row {record_id} lacks a frozen details category"
                    )
                is_oov = bool(batch["oov_characters"][index])
                exact = prediction == reference
                edit_distance = levenshtein_distance(prediction, reference)
                recognition_examples += 1
                recognition_exact += int(exact)
                edit_distance_total += edit_distance
                reference_codepoints += len(reference)
                detail_totals[detail]["support"] += 1
                detail_totals[detail]["exact"] += int(exact)
                detail_totals[detail]["edit_distance"] += edit_distance
                detail_totals[detail]["reference_codepoints"] += len(reference)
                if is_oov:
                    oov_examples += 1
                    oov_exact += int(exact)
                else:
                    encodable_examples += 1
                    encodable_exact += int(exact)
                if predictions_path is not None:
                    _append_jsonl(
                        predictions_path,
                        {
                            "kind": "ocr_recognition",
                            "record_id": batch["record_ids"][index],
                            "sample_id": batch["sample_ids"][index],
                            "reference": reference,
                            "prediction": prediction,
                            "score": float(score),
                            "exact": exact,
                            "charset_oov": is_oov,
                            "oov_codepoints": [
                                f"U+{ord(character):04X}"
                                for character in batch["oov_characters"][index]
                            ],
                        },
                    )
    if recognition_examples == 0:
        raise TorchTrainingError("OCR recognition validation loader produced no examples")
    if max_batches is None:
        if recognition_examples != len(inventory.rows):
            raise TorchTrainingError("formal OCR evaluation did not cover every included val row")
        if oov_examples != inventory.oov_records:
            raise TorchTrainingError("formal OCR OOV evaluation differs from frozen inventory")
        expected_detail_support: dict[str, int] = {}
        for detail in details_by_record.values():
            expected_detail_support[detail] = expected_detail_support.get(detail, 0) + 1
        actual_detail_support = {
            detail: values["support"] for detail, values in detail_totals.items()
        }
        if actual_detail_support != expected_detail_support:
            raise TorchTrainingError(
                "formal OCR detail evaluation did not cover every frozen val row"
            )
    detail_metrics: dict[str, float] = {}
    for detail, values in sorted(detail_totals.items()):
        support = int(values["support"])
        codepoints = int(values["reference_codepoints"])
        detail_metrics[f"recognition_{detail}_support"] = float(support)
        detail_metrics[f"recognition_{detail}_exact_match"] = (
            values["exact"] / support if support else 0.0
        )
        detail_metrics[f"recognition_{detail}_cer"] = (
            values["edit_distance"] / codepoints if codepoints else 0.0
        )
    return {
        "normalized_exact_match": recognition_exact / recognition_examples,
        "recognition_raw_exact_match": recognition_exact / recognition_examples,
        "recognition_cer": edit_distance_total / max(1, reference_codepoints),
        "recognition_validation_loss_encodable": recognition_loss_total
        / max(1, recognition_loss_examples),
        "recognition_evaluated_records": float(recognition_examples),
        "recognition_total_included_records": float(len(inventory.rows)),
        "recognition_encodable_records_evaluated": float(encodable_examples),
        "recognition_oov_records_evaluated": float(oov_examples),
        "recognition_encodable_exact_match": encodable_exact
        / max(1, encodable_examples),
        "recognition_oov_exact_match": oov_exact / max(1, oov_examples),
        "charset_coverage_record_rate": inventory.encodable_records
        / max(1, len(inventory.rows)),
        "charset_oov_record_count": float(inventory.oov_records),
        "charset_oov_unique_codepoints": float(len(inventory.oov_characters)),
        "charset_train_codepoints": float(inventory.train_character_count),
        "recognition_excluded_policy_records": float(inventory.excluded_records),
        **detail_metrics,
        "detector_hmean": detector_hmean,
        "detector_precision": detector_precision,
        "detector_recall": detector_recall,
        "detector_validation_loss": detector_loss_total / detector_examples,
        "detector_mean_matched_iou": matched_iou_weighted / max(1, tp),
        "detector_true_positives": float(tp),
        "detector_false_positives": float(fp),
        "detector_false_negatives": float(fn),
        "detector_ignored_predictions": float(
            detector_counts["ignored_predictions"]
        ),
    }


def _pin_evaluate(
    torch: Any,
    model: Any,
    loader: Any,
    device: Any,
    task_config: Any,
    compute_pin_metrics: Callable[..., dict[str, float]],
    decode_pin_heatmaps: Callable[..., list[list[dict[str, Any]]]],
    *,
    max_batches: int | None,
    predictions_path: Path | None,
) -> tuple[dict[str, float], int]:
    from pcbparse.training.torch_pin_model import pin_losses, split_pin_outputs

    model.eval()
    if predictions_path is not None:
        predictions_path.unlink(missing_ok=True)
    examples = 0
    gt_total = 0.0
    matched_total = 0.0
    pred_total = 0.0
    loss_total = 0.0
    with torch.inference_mode():
        for batch in _bounded_iterable(loader, max_batches):
            images, targets = _prepare_pin_batch(torch, batch, task_config, device)
            outputs = split_pin_outputs(model(images))
            losses = pin_losses(outputs, targets)
            box_margin_px = float(
                getattr(task_config, "pin_box_margin_px", 0.0) or 0.0
            )
            metrics = compute_pin_metrics(
                outputs,
                targets,
                heatmap_stride=4,
                score_threshold=float(task_config.pin_score_threshold),
                pck_thresholds=(4.0,),
                nms_radius=float(task_config.pin_nms_radius),
                max_detections=int(task_config.pin_max_detections),
                box_rects=targets.get("box_rects"),
                box_margin_px=box_margin_px,
                box_prior_mode=str(
                    getattr(task_config, "pin_box_prior_mode", "filled_rect_v1")
                ),
            )
            count = int(images.shape[0])
            examples += count
            gt_total += metrics["gt_count"]
            pred_total += metrics["pred_count"]
            matched_total += metrics["pck@4"] * metrics["gt_count"]
            loss_total += float(losses["loss"].detach().cpu()) * count
            if predictions_path is not None:
                raw_rects = targets.get("box_rects")
                box_rects_heatmap = (
                    None
                    if raw_rects is None
                    else [
                        tuple(float(value) / 4.0 for value in rect)
                        for rect in raw_rects
                    ]
                )
                decoded = decode_pin_heatmaps(
                    outputs,
                    score_threshold=float(task_config.pin_score_threshold),
                    nms_radius=float(task_config.pin_nms_radius),
                    max_detections=int(task_config.pin_max_detections),
                    box_rects=box_rects_heatmap,
                    box_margin=box_margin_px / 4.0,
                    box_prior_mode=str(
                        getattr(task_config, "pin_box_prior_mode", "filled_rect_v1")
                    ),
                )
                metadata = targets["metadata"]
                for row, detections in zip(metadata, decoded):
                    _append_jsonl(
                        predictions_path,
                        {
                            "record_id": row["record_id"],
                            "view_id": row["view_id"],
                            "sample_id": row["sample_id"],
                            "detections_heatmap": detections,
                            "output_stride": 4,
                        },
                    )
    if examples == 0:
        raise TorchTrainingError("pin validation loader produced no examples")
    detection = _pin_detection_summary(
        matched_total=matched_total,
        pred_total=pred_total,
        gt_total=gt_total,
    )
    return {
        "pck_4px": detection["pin_recall_4px"],
        **detection,
        "validation_loss": float(loss_total / examples),
        "gt_count": float(gt_total),
        "pred_count": float(pred_total),
    }, examples


def _pin_detection_summary(
    *, matched_total: float, pred_total: float, gt_total: float
) -> dict[str, float]:
    """Return count-aware micro metrics for anonymous Pin detections.

    PCK is recall-like and can look strong even when a heatmap emits many false
    peaks.  Production promotion therefore needs the matching precision and F1
    from the same one-to-one 4 px matches, plus the prediction-count ratio.
    """

    values = {
        "matched_total": matched_total,
        "pred_total": pred_total,
        "gt_total": gt_total,
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in values.values()
    ):
        raise TorchTrainingError("pin detection totals must be finite and non-negative")
    matched = float(matched_total)
    predicted = float(pred_total)
    truth = float(gt_total)
    if matched > min(predicted, truth):
        raise TorchTrainingError("pin matched_total cannot exceed prediction or truth totals")

    recall = matched / truth if truth else (1.0 if predicted == 0.0 else 0.0)
    precision = matched / predicted if predicted else (1.0 if truth == 0.0 else 0.0)
    denominator = precision + recall
    f1 = 2.0 * precision * recall / denominator if denominator else 0.0
    pred_to_gt_ratio = predicted / truth if truth else (1.0 if predicted == 0.0 else 1.0e9)
    return {
        "pin_precision_4px": float(precision),
        "pin_recall_4px": float(recall),
        "pin_f1_4px": float(f1),
        "pred_to_gt_ratio": float(pred_to_gt_ratio),
    }


def _wire_centerline_summary(
    *,
    precision_hits: float,
    recall_hits: float,
    predicted_skeleton_pixels: float,
    target_skeleton_pixels: float,
) -> dict[str, float]:
    """Return count-aware micro metrics for the full Wire population."""

    values = {
        "precision_hits": precision_hits,
        "recall_hits": recall_hits,
        "predicted_skeleton_pixels": predicted_skeleton_pixels,
        "target_skeleton_pixels": target_skeleton_pixels,
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in values.values()
    ):
        raise TorchTrainingError("wire centerline totals must be finite and non-negative")
    precision_hit_count = float(precision_hits)
    recall_hit_count = float(recall_hits)
    predicted_count = float(predicted_skeleton_pixels)
    target_count = float(target_skeleton_pixels)
    if precision_hit_count > predicted_count:
        raise TorchTrainingError("wire precision hits cannot exceed predicted pixels")
    if recall_hit_count > target_count:
        raise TorchTrainingError("wire recall hits cannot exceed target pixels")
    precision = (
        precision_hit_count / predicted_count
        if predicted_count
        else (1.0 if target_count == 0.0 else 0.0)
    )
    recall = (
        recall_hit_count / target_count
        if target_count
        else (1.0 if predicted_count == 0.0 else 0.0)
    )
    denominator = precision + recall
    f1 = 2.0 * precision * recall / denominator if denominator else 0.0
    return {
        "centerline_precision": float(precision),
        "centerline_recall": float(recall),
        "centerline_f1": float(f1),
        "skeleton_f1": float(f1),
    }


def _set_cosine_epoch_learning_rate(
    optimizer: Any,
    *,
    epoch: int,
    planned_epochs: int,
    base_learning_rate: float,
    minimum_learning_rate: float,
) -> float:
    """Set the absolute cosine LR for an epoch, including after a resume.

    A stateful scheduler that is not part of the resume checkpoint silently
    restarts its phase after interruption.  Computing the rate from the
    absolute epoch makes an existing v1 resume checkpoint deterministic without
    broadening that checkpoint's compatibility contract.
    """

    if type(epoch) is not int or type(planned_epochs) is not int:
        raise TorchTrainingError("cosine schedule epoch values must be integers")
    if not 1 <= epoch <= planned_epochs:
        raise TorchTrainingError("cosine schedule epoch is outside the formal run")
    if (
        isinstance(base_learning_rate, bool)
        or isinstance(minimum_learning_rate, bool)
        or not isinstance(base_learning_rate, (int, float))
        or not isinstance(minimum_learning_rate, (int, float))
        or not math.isfinite(float(base_learning_rate))
        or not math.isfinite(float(minimum_learning_rate))
        or float(base_learning_rate) <= 0.0
        or not 0.0 <= float(minimum_learning_rate) <= float(base_learning_rate)
    ):
        raise TorchTrainingError("cosine schedule learning rates are invalid")
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list) or not groups:
        raise TorchTrainingError("optimizer has no parameter groups")
    progress = float(epoch - 1) / float(planned_epochs)
    rate = float(minimum_learning_rate) + 0.5 * (
        float(base_learning_rate) - float(minimum_learning_rate)
    ) * (1.0 + math.cos(math.pi * progress))
    for group in groups:
        if not isinstance(group, dict):
            raise TorchTrainingError("optimizer parameter group is invalid")
        group["lr"] = rate
    return float(rate)


def _wire_evaluate(
    torch: Any,
    model: Any,
    loader: Any,
    device: Any,
    loss_config: Any,
    compute_wire_loss: Callable[..., Mapping[str, Any]],
    wire_centerline_metrics: Callable[..., dict[str, float]],
    *,
    max_batches: int | None,
    predictions_path: Path | None,
) -> tuple[dict[str, float], int]:
    model.eval()
    if predictions_path is not None:
        predictions_path.unlink(missing_ok=True)
    totals: dict[str, float] = {}
    centerline_totals = {
        "precision_hits": 0.0,
        "recall_hits": 0.0,
        "predicted_skeleton_pixels": 0.0,
        "target_skeleton_pixels": 0.0,
    }
    connectivity_score_sum = 0.0
    component_count_error_sum = 0.0
    examples = 0
    with torch.inference_mode():
        for batch in _bounded_iterable(loader, max_batches):
            image = batch["image"].to(device=device, dtype=torch.float32)
            target = batch["target"].to(device=device, dtype=torch.float32)
            ignore = batch["ignore"].to(device=device, dtype=torch.float32)
            logits = model(image)
            losses = compute_wire_loss(logits, target, ignore, config=loss_config)
            metrics = wire_centerline_metrics(
                logits,
                target,
                ignore,
                threshold=0.5,
                tolerance_px=1,
                skeleton_iterations=8,
            )
            count = int(image.shape[0])
            examples += count
            values = {
                "validation_loss": float(losses["loss"].detach().cpu()),
                **{
                    name: float(value)
                    for name, value in metrics.items()
                    if name
                    not in {
                        "centerline_precision",
                        "centerline_recall",
                        "centerline_f1",
                        "skeleton_f1",
                        "precision_hits",
                        "recall_hits",
                        "predicted_skeleton_pixels",
                        "target_skeleton_pixels",
                        "connectivity_score",
                        "connectivity_score_sum",
                        "predicted_components",
                        "target_components",
                        "component_count_error",
                        "component_count_error_sum",
                    }
                },
            }
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + value * count
            for name in centerline_totals:
                centerline_totals[name] += float(metrics[name])
            connectivity_score_sum += float(metrics["connectivity_score_sum"])
            component_count_error_sum += float(metrics["component_count_error_sum"])
            if predictions_path is not None:
                masks = (torch.sigmoid(logits) >= 0.5).detach().cpu().numpy()
                for index, record_id in enumerate(batch["record_id"]):
                    height, width = (
                        int(value)
                        for value in batch["original_shapes"][index].tolist()
                    )
                    mask = masks[index, 0, :height, :width].astype(np.uint8)
                    packed = np.packbits(mask.reshape(-1), bitorder="little").tobytes()
                    _append_jsonl(
                        predictions_path,
                        {
                            "record_id": record_id,
                            "sample_id": batch["sample_id"][index],
                            "shape": [height, width],
                            "encoding": "base64-packbits-little",
                            "mask": base64.b64encode(packed).decode("ascii"),
                        },
                    )
    if examples == 0:
        raise TorchTrainingError("wire validation loader produced no examples")
    centerline = _wire_centerline_summary(**centerline_totals)
    return {
        name: float(value / examples) for name, value in sorted(totals.items())
    } | {
        **centerline,
        "connectivity_score": float(connectivity_score_sum / examples),
        "component_count_error": float(component_count_error_sum / examples),
    }, examples


def _component_predictions(
    torch: Any,
    model: Any,
    loader: Any,
    device: Any,
    path: Path,
    max_batches: int | None,
) -> None:
    path.unlink(missing_ok=True)
    model.eval()
    rows = 0
    with torch.inference_mode():
        for images, targets in _bounded_iterable(loader, max_batches):
            image_list = [image.to(device=device, dtype=torch.float32) for image in images]
            outputs = model(image_list)
            for target, prediction in zip(targets, outputs):
                _append_jsonl(
                    path,
                    {
                        "sample_id": target["sample_id"],
                        "view_id": target["view_id"],
                        "boxes": prediction["boxes"].detach().cpu().tolist(),
                        "scores": prediction["scores"].detach().cpu().tolist(),
                        "labels": prediction["labels"].detach().cpu().tolist(),
                    },
                )
                rows += 1
    if rows == 0:
        raise TorchTrainingError("component prediction evidence is empty")


def _prepare_pin_batch(
    torch: Any, batch: tuple[Any, dict[str, Any]], task_config: Any, device: Any
) -> tuple[Any, dict[str, Any]]:
    images, raw_targets = batch
    images = images.to(device=device, dtype=torch.float32)
    desired_height, desired_width = (int(value) for value in task_config.input_size)
    old_height, old_width = (int(value) for value in images.shape[-2:])
    targets = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in raw_targets.items()
    }
    if (old_height, old_width) == (desired_height, desired_width):
        return images, targets
    images = torch.nn.functional.interpolate(
        images,
        size=(desired_height, desired_width),
        mode="bilinear",
        align_corners=False,
    )
    old_hm_height, old_hm_width = (int(value) for value in targets["heatmap"].shape[-2:])
    new_hm_height = int(math.ceil(desired_height / 4))
    new_hm_width = int(math.ceil(desired_width / 4))
    targets["heatmap"] = torch.nn.functional.interpolate(
        targets["heatmap"],
        size=(new_hm_height, new_hm_width),
        mode="bilinear",
        align_corners=False,
    )
    batch_size = int(images.shape[0])
    offset = torch.zeros((batch_size, 2, new_hm_height, new_hm_width), device=device)
    orientation = torch.zeros_like(offset)
    mask = torch.zeros((batch_size, 1, new_hm_height, new_hm_width), device=device)
    scaled_points: list[Any] = []
    for batch_index, (points, orientations) in enumerate(
        zip(targets["tip_points_heatmap"], targets["orientation_instances"])
    ):
        current = points.to(device=device, dtype=torch.float32).clone()
        visible = (current[:, 0] >= 0) & (current[:, 1] >= 0)
        current[visible, 0] *= new_hm_width / old_hm_width
        current[visible, 1] *= new_hm_height / old_hm_height
        for point, direction in zip(current[visible], orientations.to(device)[visible]):
            x = min(new_hm_width - 1, max(0, int(math.floor(float(point[0])))))
            y = min(new_hm_height - 1, max(0, int(math.floor(float(point[1])))))
            if float(mask[batch_index, 0, y, x]) != 0.0:
                continue
            mask[batch_index, 0, y, x] = 1.0
            offset[batch_index, :, y, x] = torch.stack(
                (point[0] - x, point[1] - y)
            )
            orientation[batch_index, :, y, x] = direction
        scaled_points.append(current)
    targets["offset"] = offset
    targets["orientation"] = orientation
    targets["regression_mask"] = mask
    targets["tip_points_heatmap"] = scaled_points
    targets["tip_points"] = [
        points.to(device=device, dtype=torch.float32)
        * torch.tensor(
            [desired_width / old_width, desired_height / old_height], device=device
        )
        for points in targets["tip_points"]
    ]
    box_rects: list[tuple[float, float, float, float] | None] = []
    for sample_index in range(batch_size):
        sample_metadata = targets["metadata"][sample_index]
        rect = sample_metadata.get("box_rect_view")
        view_size = sample_metadata.get("view_output_size")
        if rect is None or view_size is None:
            box_rects.append(None)
            continue
        scale_x = desired_width / float(view_size[0])
        scale_y = desired_height / float(view_size[1])
        box_rects.append(
            (
                float(rect[0]) * scale_x,
                float(rect[1]) * scale_y,
                float(rect[2]) * scale_x,
                float(rect[3]) * scale_y,
            )
        )
    targets["box_rects"] = box_rects
    return images, targets


def _component_export_samples(
    loader: Any, count: int, max_batches: int | None
) -> list[Any]:
    samples: list[Any] = []
    for images, _targets in _bounded_iterable(loader, max_batches):
        for image in images:
            samples.append(image.unsqueeze(0).to("cpu"))
            if len(samples) == count:
                return samples
    raise TorchTrainingError(
        f"component validation split has fewer than {count} export samples"
    )


def _pin_export_samples(
    torch: Any,
    loader: Any,
    task_config: Any,
    count: int,
    max_batches: int | None,
) -> list[Any]:
    samples: list[Any] = []
    for batch in _bounded_iterable(loader, max_batches):
        images, _targets = _prepare_pin_batch(torch, batch, task_config, "cpu")
        for image in images:
            samples.append(image.unsqueeze(0).to("cpu"))
            if len(samples) == count:
                return samples
    raise TorchTrainingError(f"pin validation split has fewer than {count} export samples")


def _wire_export_samples(
    loader: Any, count: int, max_batches: int | None
) -> list[Any]:
    samples: list[Any] = []
    for batch in _bounded_iterable(loader, max_batches):
        for image in batch["image"]:
            samples.append(image.unsqueeze(0).to("cpu"))
            if len(samples) == count:
                return samples
    raise TorchTrainingError(f"wire validation split has fewer than {count} export samples")


def _ocr_raw_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    torch = _require_torch()
    if not batch:
        raise TorchTrainingError("cannot collate an empty OCR evaluation batch")
    images = [item["image"] for item in batch]
    channels = int(images[0].shape[0])
    height = int(images[0].shape[1])
    if any(
        image.ndim != 3
        or int(image.shape[0]) != channels
        or int(image.shape[1]) != height
        for image in images
    ):
        raise TorchTrainingError("OCR evaluation images disagree on channel/height")
    widths = [int(image.shape[2]) for image in images]
    maximum = max(widths)
    padded = images[0].new_full(
        (len(images), channels, height, maximum), 1.0
    )
    for index, image in enumerate(images):
        padded[index, :, :, : widths[index]] = image
    return {
        "images": padded,
        "image_widths": torch.tensor(widths, dtype=torch.long),
        "targets": [item["target"] for item in batch],
        "texts": [item["text"] for item in batch],
        "record_ids": [item["record_id"] for item in batch],
        "sample_ids": [item["sample_id"] for item in batch],
        "oov_characters": [item["oov_characters"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


def _ocr_val_detail_index(
    preflight: Mapping[str, Any],
    inventory: _OCRInventory,
    *,
    expected_sha256: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Join included recognition rows to the hash-bound derived val manifest."""

    if inventory.split != "val":
        raise TorchTrainingError("OCR detail metrics are restricted to frozen val")
    task_reports = _mapping(preflight.get("tasks"), "preflight task reports")
    task_report = _mapping(task_reports.get("ocr"), "preflight OCR task report")
    raw_path = task_report.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise TorchTrainingError("preflight OCR task report lacks derived path")
    path = Path(raw_path).expanduser().resolve(strict=False)
    workspace_root = _workspace_root(preflight)
    _require_within(path, workspace_root, "derived OCR detail manifest")
    expected = _sha(expected_sha256, "derived OCR detail manifest hash")
    if task_report.get("sha256") != expected:
        raise TorchTrainingError(
            "preflight OCR task hash differs from frozen training configuration"
        )
    _verify_file(path, expected, "derived OCR detail manifest")

    wanted = {str(row["record_id"]): row for row in inventory.rows}
    if len(wanted) != len(inventory.rows):
        raise TorchTrainingError("OCR inventory has duplicate record_id values")
    joined: dict[str, str] = {}
    seen: set[str] = set()
    support: dict[str, int] = {}
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise TorchTrainingError(f"cannot open derived OCR detail manifest: {exc}") from exc
    with handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise TorchTrainingError(
                    f"derived OCR detail manifest contains blank line {line_number}"
                )
            try:
                row = json.loads(
                    raw,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
            except (ValueError, json.JSONDecodeError) as exc:
                raise TorchTrainingError(
                    f"invalid derived OCR detail row {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise TorchTrainingError("derived OCR detail row is not an object")
            record_id = row.get("record_id")
            if not isinstance(record_id, str) or not record_id or record_id in seen:
                raise TorchTrainingError(
                    "derived OCR detail manifest has missing/duplicate record_id"
                )
            seen.add(record_id)
            expected_row = wanted.get(record_id)
            if expected_row is None:
                continue
            if row.get("split") != "val" or row.get("task") != "ocr_text":
                raise TorchTrainingError(
                    f"derived OCR detail row {record_id} is outside frozen val"
                )
            if (
                row.get("sample_id") != expected_row.get("sample_id")
                or row.get("content") != expected_row.get("content")
            ):
                raise TorchTrainingError(
                    f"derived OCR detail row {record_id} differs from recognition index"
                )
            detail = _ocr_detail_metric_name(row.get("details"))
            joined[record_id] = detail
            support[detail] = support.get(detail, 0) + 1
    missing = sorted(set(wanted) - set(joined))
    if missing:
        raise TorchTrainingError(
            "derived OCR detail manifest does not cover frozen val inventory: "
            + ", ".join(missing[:8])
        )
    return joined, {
        "split": "val",
        "test_rows_used": 0,
        "thresholds_tuned": False,
        "path": str(path),
        "sha256": expected,
        "joined_records": len(joined),
        "support_by_detail": dict(sorted(support.items())),
    }


def _ocr_detail_metric_name(value: Any) -> str:
    if not isinstance(value, str):
        raise TorchTrainingError("derived OCR details must be a string")
    normalized = "".join(
        character for character in value.casefold() if character.isalnum()
    )
    aliases = {
        "": "unknown",
        "designator": "designator",
        "refdes": "designator",
        "reference": "designator",
        "pinnumber": "pinnumber",
        "pinno": "pinnumber",
        "pinname": "pinname",
        "netlabel": "netlabel",
        "value": "value",
        "searchname": "searchname",
        "partname": "searchname",
    }
    if normalized not in aliases:
        raise TorchTrainingError(
            f"derived OCR details value is unsupported: {value!r}"
        )
    return aliases[normalized]


def _ocr_inventory(root: Path, charset: Any, *, split: str) -> _OCRInventory:
    if split not in {"train", "val", "test"}:
        raise TorchTrainingError(f"invalid OCR inventory split {split!r}")
    report = _read_json(root / "report.json")
    charset_report = _mapping(report.get("charset"), "OCR report charset")
    declared_train = charset_report.get("training_characters")
    if not isinstance(declared_train, list) or any(
        not isinstance(character, str) or len(character) != 1
        for character in declared_train
    ):
        raise TorchTrainingError("OCR report train charset is invalid")
    if tuple(declared_train) != tuple(charset.characters):
        raise TorchTrainingError(
            "OCR charset.txt differs from report's train-only character sequence"
        )
    rows: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    try:
        handle = (root / "recognition_index.jsonl").open("r", encoding="utf-8")
    except OSError as exc:
        raise TorchTrainingError(f"cannot open OCR recognition index: {exc}") from exc
    with handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise TorchTrainingError(
                    f"OCR recognition index contains blank line {line_number}"
                )
            try:
                row = json.loads(
                    raw,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
            except (ValueError, json.JSONDecodeError) as exc:
                raise TorchTrainingError(
                    f"invalid OCR recognition row {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise TorchTrainingError("OCR recognition row is not an object")
            record_id = row.get("record_id")
            if not isinstance(record_id, str) or not record_id or record_id in identifiers:
                raise TorchTrainingError("OCR recognition index has missing/duplicate record_id")
            identifiers.add(record_id)
            if row.get("split") != split or row.get("status") != "included":
                continue
            text = row.get("content")
            if not isinstance(text, str) or not text or "\n" in text or "\r" in text:
                raise TorchTrainingError(f"OCR row {record_id} has invalid raw text")
            rows.append(row)
    rows.sort(key=lambda item: str(item["record_id"]))
    if not rows:
        raise TorchTrainingError(f"OCR {split} inventory has no included records")
    charset_set = frozenset(charset.characters)
    unknown_by_record = {
        str(row["record_id"]): tuple(
            sorted(set(str(row["content"])) - charset_set, key=ord)
        )
        for row in rows
    }
    unknown_by_record = {
        record_id: unknown
        for record_id, unknown in unknown_by_record.items()
        if unknown
    }
    if split == "train":
        recomputed = tuple(
            sorted(
                {
                    character
                    for row in rows
                    for character in str(row["content"])
                },
                key=ord,
            )
        )
        if recomputed != tuple(charset.characters):
            raise TorchTrainingError(
                "OCR charset is not exactly the code points in included train rows"
            )
    declared_oov = _mapping(
        charset_report.get("oov_by_split", {}), "OCR report OOV splits"
    ).get(split)
    if split in {"val", "test"}:
        declared_oov = _mapping(declared_oov, f"OCR report {split} OOV")
        declared_ids = declared_oov.get("record_ids")
        declared_characters = declared_oov.get("characters")
        if not isinstance(declared_ids, list) or not isinstance(
            declared_characters, list
        ):
            raise TorchTrainingError(f"OCR report {split} OOV evidence is invalid")
        actual_ids = sorted(unknown_by_record)
        actual_characters = sorted(
            {character for values in unknown_by_record.values() for character in values},
            key=ord,
        )
        if declared_ids != actual_ids or declared_characters != actual_characters:
            raise TorchTrainingError(
                f"OCR report {split} OOV evidence differs from recognition index"
            )
    recognition_report = _mapping(
        report.get("recognition"), "OCR report recognition"
    )
    excluded_rows = recognition_report.get("excluded_records")
    if not isinstance(excluded_rows, list):
        raise TorchTrainingError("OCR report excluded_records is invalid")
    excluded_count = sum(
        isinstance(row, Mapping) and row.get("split") == split
        for row in excluded_rows
    )
    oov_characters = tuple(
        sorted(
            {character for values in unknown_by_record.values() for character in values},
            key=ord,
        )
    )
    return _OCRInventory(
        train_character_count=len(charset.characters),
        split=split,
        rows=tuple(rows),
        encodable_records=len(rows) - len(unknown_by_record),
        oov_records=len(unknown_by_record),
        oov_characters=oov_characters,
        oov_record_ids=tuple(sorted(unknown_by_record)),
        excluded_records=excluded_count,
    )


def _ocr_detector_export_samples(
    loader: Any, count: int, max_batches: int | None
) -> list[Any]:
    samples: list[Any] = []
    for batch in _bounded_iterable(loader, max_batches):
        for image in batch["images"]:
            samples.append(image.unsqueeze(0).to("cpu"))
            if len(samples) == count:
                return samples
    raise TorchTrainingError(
        f"OCR detector validation split has fewer than {count} export samples"
    )


def _ocr_recognizer_export_samples(
    loader: Any, count: int, max_batches: int | None
) -> list[Any]:
    samples: list[Any] = []
    for batch in _bounded_iterable(loader, max_batches):
        for index, width in enumerate(batch["image_widths"].tolist()):
            samples.append(batch["images"][index : index + 1, :, :, : int(width)].to("cpu"))
            if len(samples) == count:
                return samples
    raise TorchTrainingError(
        f"OCR recognizer validation split has fewer than {count} export samples"
    )


def _verify_task_materialization(
    task: str,
    training_config: Any,
    preflight_report: Mapping[str, Any],
    workspace_root: Path,
) -> _MaterializationBinding:
    materialized = (workspace_root / _MATERIALIZED_RELATIVE).resolve(strict=False)
    if not materialized.is_dir():
        raise TorchTrainingError(
            f"fixed training materialization root is missing: {materialized}"
        )
    if task in {"component", "pin"}:
        root = materialized / "component-pin"
        summary_path = root / "summary.json"
        summary = _read_json(summary_path)
        if summary.get("materializer_version") != "component-pin-index-v1":
            raise TorchTrainingError("component-pin materialization version is unsupported")
        binding = _sha(summary.get("binding_sha256"), "component-pin binding")
        lineage = _mapping(summary.get("lineage"), "component-pin lineage")
        _expect_equal(
            lineage.get("derivation_config_sha256"),
            training_config.dataset.config_hash,
            "component-pin derivation config hash",
        )
        inputs = _mapping(lineage.get("inputs"), "component-pin inputs")
        expected_inputs = {
            "source_manifest": training_config.dataset.source_manifest_sha256,
            "split_manifest": training_config.dataset.split_manifest_sha256,
            "summary": training_config.dataset.summary_sha256,
            "component_manifest": training_config.dataset.task_manifest_sha256[
                "component_detection"
            ],
            "pin_manifest": training_config.dataset.task_manifest_sha256[
                "pin_keypoints"
            ],
        }
        for name, expected in expected_inputs.items():
            row = _mapping(inputs.get(name), f"component-pin input {name}")
            _expect_equal(row.get("sha256"), expected, f"component-pin {name} hash")
        outputs = _mapping(summary.get("outputs"), "component-pin outputs")
        for name in ("component_coco.json", "pin_index.jsonl"):
            row = _mapping(outputs.get(name), f"component-pin output {name}")
            _verify_file(root / name, row.get("sha256"), name)
        return _MaterializationBinding(
            task=task,
            root=materialized,
            commit_kind="binding_sha256",
            commit_id=binding,
            commit_sha256=_sha256_file(summary_path),
            lineage=lineage,
            task_root=root,
        )
    if task in {"wire", "junction"}:
        if _WIRE_JUNCTION_ROOT_OVERRIDE:
            root = Path(_WIRE_JUNCTION_ROOT_OVERRIDE).expanduser().resolve()
        else:
            root = materialized / "wire-junction"
        latest_path = root / "latest.json"
        latest = _read_json(latest_path)
        run_id = _sha(latest.get("run_id"), "wire-junction run_id")
        raw_run_dir = latest.get("run_dir")
        if not isinstance(raw_run_dir, str) or not raw_run_dir:
            raise TorchTrainingError("wire-junction latest pointer lacks run_dir")
        run_dir = (root / raw_run_dir).resolve(strict=False)
        _require_within(run_dir, root, "wire-junction run directory")
        if run_dir.name != run_id or not run_dir.is_dir():
            raise TorchTrainingError("wire-junction content-addressed run directory is invalid")
        summary_path = run_dir / "summary.json"
        _verify_file(summary_path, latest.get("summary_sha256"), "wire-junction summary")
        summary = _read_json(summary_path)
        if summary.get("complete") is not True or summary.get("run_id") != run_id:
            raise TorchTrainingError("wire-junction summary is incomplete or has a mixed run ID")
        lineage = _mapping(summary.get("lineage"), "wire-junction lineage")
        _expect_equal(
            lineage.get("derived_config_hash"),
            training_config.dataset.config_hash,
            "wire-junction derived config hash",
        )
        inputs = _mapping(lineage.get("inputs"), "wire-junction inputs")
        expected_inputs = {
            "source_manifest_sha256": training_config.dataset.source_manifest_sha256,
            "split_manifest_sha256": training_config.dataset.split_manifest_sha256,
            "wire_manifest_sha256": training_config.dataset.task_manifest_sha256[
                "wire_paths"
            ],
            "junction_manifest_sha256": training_config.dataset.task_manifest_sha256[
                "junction_patches"
            ],
        }
        for name, expected in expected_inputs.items():
            _expect_equal(inputs.get(name), expected, f"wire-junction {name}")
        outputs = _mapping(summary.get("outputs"), "wire-junction outputs")
        for row_name, row in outputs.items():
            item = _mapping(row, f"wire-junction output {row_name}")
            relative = item.get("path")
            if not isinstance(relative, str) or not relative:
                raise TorchTrainingError(f"wire-junction output {row_name} lacks path")
            path = (run_dir / relative).resolve(strict=False)
            _require_within(path, run_dir, f"wire-junction output {row_name}")
            _verify_file(path, item.get("sha256"), f"wire-junction output {row_name}")
        return _MaterializationBinding(
            task=task,
            root=materialized,
            commit_kind="content_addressed_run",
            commit_id=run_id,
            commit_sha256=_sha256_file(summary_path),
            lineage=lineage,
            task_root=run_dir,
        )
    if task == "ocr":
        root = (
            Path(_OCR_ROOT_OVERRIDE).expanduser().resolve()
            if _OCR_ROOT_OVERRIDE
            else materialized / "ocr"
        )
        manifest_path = root / "manifest.json"
        manifest = _read_json(manifest_path)
        if manifest.get("complete") is not True:
            raise TorchTrainingError("OCR materialization is incomplete")
        lineage = _mapping(manifest.get("lineage"), "OCR lineage")
        expected = {
            "dataset_config_hash": training_config.dataset.config_hash,
            "derived_summary_sha256": training_config.dataset.summary_sha256,
            "source_manifest_sha256": training_config.dataset.source_manifest_sha256,
            "split_manifest_sha256": training_config.dataset.split_manifest_sha256,
            "ocr_manifest_sha256": training_config.dataset.task_manifest_sha256[
                "ocr_text"
            ],
        }
        for name, digest in expected.items():
            _expect_equal(lineage.get(name), digest, f"OCR {name}")
        artifacts = _mapping(manifest.get("artifacts"), "OCR artifacts")
        for name, digest in artifacts.items():
            _verify_file(root / name, digest, f"OCR artifact {name}")
        return _MaterializationBinding(
            task=task,
            root=materialized,
            commit_kind="manifest_sha256",
            commit_id=_sha256_file(manifest_path),
            commit_sha256=_sha256_file(manifest_path),
            lineage=lineage,
            task_root=root,
        )
    raise TorchTrainingError(f"unknown task {task!r}")


def _reject_unready_junction(
    binding: _MaterializationBinding, task_config: Any
) -> None:
    from pcbparse.training.torch_wire_junction import inspect_junction_dataset

    import logging

    metadata = inspect_junction_dataset(binding.task_root, split=None)
    synthetic_per_class = int(
        getattr(task_config, "junction_synthetic_per_class", 0) or 0
    )
    if metadata.six_class_training_ready:
        return
    logging.warning(
        "Junction training proceeding with missing classes: %s. "
        "Synthetic data (%d/class) will fill train-only gaps.",
        metadata.missing_classes,
        synthetic_per_class,
    )
    if synthetic_per_class <= 0:
        raise TorchTrainingError(
            "junction training is fail-closed: trusted six-class labels are "
            "incomplete (missing: "
            + ", ".join(metadata.missing_classes)
            + "); configure junction_synthetic_per_class > 0 to fill train-only gaps"
        )


def _validate_invocation(
    task: str,
    task_config: Any,
    training_config: Any,
    preflight_report: Mapping[str, Any],
) -> None:
    if task not in _TASK_DATASET:
        raise TorchTrainingError(f"unknown model task {task!r}")
    if preflight_report.get("ok") is not True:
        raise TorchTrainingError("training preflight is not successful; training was not started")
    if getattr(task_config, "dataset_task", None) != _TASK_DATASET[task]:
        raise TorchTrainingError("task configuration is bound to the wrong derived task")
    expected_architecture = _TASK_ARCHITECTURE.get(task)
    if expected_architecture is not None and getattr(
        task_config, "architecture", None
    ) != expected_architecture:
        raise TorchTrainingError(
            f"{task} architecture must be {expected_architecture!r}; "
            f"got {getattr(task_config, 'architecture', None)!r}"
        )
    _validate_task_input_size(task, getattr(task_config, "input_size", None))
    pretrained = getattr(task_config, "pretrained_artifact", None)
    if task not in {"component", "ocr", "pin"} and pretrained is not None:
        raise TorchTrainingError(
            f"{task} does not implement pretrained_artifact initialization"
        )
    configured = getattr(training_config, "tasks", {}).get(task)
    if configured is None or configured != task_config:
        raise TorchTrainingError("task configuration is not the frozen training-config entry")
    report_hash = preflight_report.get("training_config_sha256")
    if not isinstance(report_hash, str) or len(report_hash) != 64:
        raise TorchTrainingError("preflight lacks a valid training-config SHA-256")
    if preflight_report.get("dataset", {}).get("config_hash") != training_config.dataset.config_hash:
        raise TorchTrainingError("preflight dataset hash differs from training configuration")
    if hasattr(training_config, "model_dump"):
        from pcbparse.training.config import training_config_sha256

        actual_config_hash = training_config_sha256(training_config)
        if actual_config_hash != report_hash:
            raise TorchTrainingError(
                "preflight training-config hash differs from the supplied immutable config"
            )
    task_report = preflight_report.get("tasks", {}).get(task)
    if not isinstance(task_report, Mapping):
        raise TorchTrainingError(f"preflight lacks task report for {task!r}")
    expected_task_hash = training_config.dataset.task_manifest_sha256[
        task_config.dataset_task
    ]
    if task_report.get("sha256") != expected_task_hash:
        raise TorchTrainingError("preflight task-manifest hash differs from configuration")


def _training_lineage(
    task: str,
    training_config: Any,
    preflight: Mapping[str, Any],
    binding: _MaterializationBinding,
    limits: _RunLimits,
) -> dict[str, Any]:
    initialization = _initialization_lineage(
        task,
        training_config.tasks[task],
        _workspace_root(preflight),
    )
    from pcbparse.training.config import training_config_resume_sha256s

    # Formal CLI runs pass the validated Pydantic TrainingConfig.  A few
    # contract tests deliberately use a minimal namespace instead; those
    # fixtures have no serialized schema history and must therefore accept
    # only the already verified current digest.
    if callable(getattr(training_config, "model_dump", None)):
        compatible_config_hashes = training_config_resume_sha256s(training_config)
    else:
        compatible_config_hashes = (preflight["training_config_sha256"],)
    return {
        "schema_version": 1,
        "task": task,
        "training_config_sha256": preflight["training_config_sha256"],
        "resume_compatible_training_config_sha256s": list(
            compatible_config_hashes
        ),
        "dataset_config_hash": training_config.dataset.config_hash,
        "source_manifest_sha256": training_config.dataset.source_manifest_sha256,
        "split_manifest_sha256": training_config.dataset.split_manifest_sha256,
        "task_manifest_sha256": training_config.dataset.task_manifest_sha256[
            training_config.tasks[task].dataset_task
        ],
        # The Pin checkpoint schema uses this canonical name; for the
        # content-addressed Wire run it is the 64-hex run ID.
        "binding_sha256": binding.commit_id,
        "smoke": limits.smoke,
        "formal_evidence": not limits.smoke,
        "architecture": training_config.tasks[task].architecture,
        "input_size": list(training_config.tasks[task].input_size),
        "initialization": initialization,
        **binding.as_lineage(),
    }


def _validate_task_input_size(task: str, raw_size: Any) -> None:
    if not isinstance(raw_size, Sequence) or isinstance(raw_size, (str, bytes)):
        raise TorchTrainingError(f"{task} input_size must contain height and width")
    size = tuple(raw_size)
    if len(size) != 2 or any(type(value) is not int or value <= 0 for value in size):
        raise TorchTrainingError(f"{task} input_size must contain two positive integers")
    height, width = size
    if task in {"pin", "wire"} and (height % 8 or width % 8):
        raise TorchTrainingError(
            f"{task} input_size must be divisible by 8 for its U-Net encoder"
        )
    if task == "component" and (height < 32 or width < 32):
        raise TorchTrainingError("component input_size dimensions must be at least 32")
    if task == "ocr":
        if height != 48:
            raise TorchTrainingError(
                "OCR input_size height must be 48 for the implemented recognizer/adapter"
            )
        if width < 32 or width % 4:
            raise TorchTrainingError(
                "OCR nominal input_size width must be >=32 and divisible by 4"
            )


def _initialization_lineage(
    task: str, task_config: Any, workspace_root: Path
) -> dict[str, Any]:
    pretrained = getattr(task_config, "pretrained_artifact", None)
    if pretrained is None:
        return {
            "mode": "random",
            "pretrained_artifact_configured": False,
        }
    if task not in {"component", "ocr", "pin"}:
        raise TorchTrainingError(
            f"{task} does not implement pretrained_artifact initialization"
        )
    relative = Path(str(pretrained.path))
    if relative.is_absolute() or ".." in relative.parts:
        raise TorchTrainingError(
            "pretrained_artifact.path must be workspace-relative without '..'"
        )
    path = (workspace_root / relative).resolve(strict=False)
    _require_within(path, workspace_root, "pretrained artifact")
    expected_sha256 = _sha(pretrained.sha256, "pretrained artifact hash")
    _verify_file(path, expected_sha256, "pretrained artifact")
    source = str(pretrained.source).strip()
    license_name = str(pretrained.license).strip()
    if not source or not license_name:
        raise TorchTrainingError("pretrained artifact source/license must be non-empty")
    return {
        "mode": "pretrained_local_artifact",
        "pretrained_artifact_configured": True,
        "path": relative.as_posix(),
        "resolved_path": str(path),
        "sha256": expected_sha256,
        "source": source,
        "license": license_name,
        # COCO component checkpoints intentionally permit a replaced detector
        # head. PCBParse OCR/Pin checkpoints must be byte- and architecture-exact.
        "strict": task in {"ocr", "pin"},
    }


def _run_limits(task_config: Any) -> _RunLimits:
    smoke = _env_bool("PCBPARSE_TRAIN_SMOKE", False)
    if smoke:
        epochs = _env_positive_int("PCBPARSE_TRAIN_SMOKE_EPOCHS", 1)
        train_batches = _env_positive_int("PCBPARSE_TRAIN_SMOKE_TRAIN_BATCHES", 1)
        val_batches = _env_positive_int("PCBPARSE_TRAIN_SMOKE_VAL_BATCHES", 1)
        export_samples = _env_positive_int("PCBPARSE_TRAIN_SMOKE_EXPORT_SAMPLES", 1)
    else:
        forbidden = [
            name
            for name in (
                "PCBPARSE_TRAIN_SMOKE_EPOCHS",
                "PCBPARSE_TRAIN_SMOKE_TRAIN_BATCHES",
                "PCBPARSE_TRAIN_SMOKE_VAL_BATCHES",
                "PCBPARSE_TRAIN_SMOKE_EXPORT_SAMPLES",
            )
            if os.environ.get(name) not in (None, "")
        ]
        if forbidden:
            raise TorchTrainingError(
                "smoke limit variables require PCBPARSE_TRAIN_SMOKE=1: "
                + ", ".join(forbidden)
            )
        epochs = int(task_config.epochs)
        train_batches = None
        val_batches = None
        export_samples = int(task_config.export_min_samples)
    return _RunLimits(
        smoke=smoke,
        epochs=epochs,
        max_train_batches=train_batches,
        max_val_batches=val_batches,
        export_samples=export_samples,
    )


def _export_tolerance(task_config: Any, limits: _RunLimits) -> float:
    configured = float(task_config.export_max_abs_error)
    # Smoke is deliberately not evidence.  A modest diagnostic tolerance lets
    # it exercise the entire ONNX/ORT path on randomly initialized weights;
    # the measured error remains recorded, while formal runs use the exact
    # configured tolerance and fail closed.
    return max(configured, 5.0e-2) if limits.smoke else configured


def _configure_runtime(
    torch: Any, training_config: Any, task_config: Any
) -> tuple[Any, str]:
    seed = int(training_config.runtime.seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    deterministic = bool(training_config.runtime.deterministic_algorithms)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
    requested = os.environ.get("PCBPARSE_TRAIN_DEVICE", "auto").strip().lower()
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif requested == "cuda":
        # ``torch.device('cuda')`` is valid for tensor placement, but APIs used
        # below (device properties, memory fractions, and generators) require
        # a concrete device index.  The CLI explicitly advertises bare
        # ``cuda``, so normalize it deterministically to the first GPU.
        requested = "cuda:0"
    try:
        device = torch.device(requested)
    except Exception as exc:
        raise TorchTrainingError(f"invalid PCBPARSE_TRAIN_DEVICE={requested!r}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TorchTrainingError("CUDA was requested but the installed PyTorch cannot use it")
    # TorchVision ROIAlign backward currently has no deterministic CUDA
    # implementation.  A deterministic Component run therefore executes on
    # CPU instead of silently weakening the frozen runtime policy.
    if (
        device.type == "cuda"
        and deterministic
        and getattr(task_config, "dataset_task", None) == "component_detection"
    ):
        device = torch.device("cpu")
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        total_mb = int(properties.total_memory // (1024 * 1024))
        budget_mb = int(training_config.runtime.max_vram_mb)
        if budget_mb < total_mb and hasattr(torch.cuda, "set_per_process_memory_fraction"):
            torch.cuda.set_per_process_memory_fraction(
                max(0.05, min(1.0, budget_mb / total_mb)), device=device
            )
    active_precision = os.environ.get("PCBPARSE_TRAIN_PRECISION", "").strip().lower()
    if active_precision and active_precision not in {"fp16", "fp32", "bf16"}:
        raise TorchTrainingError("PCBPARSE_TRAIN_PRECISION must be fp16, fp32, or bf16")
    if not active_precision:
        active_precision = str(task_config.precision)
    if active_precision in {"fp16", "bf16"} and device.type != "cuda":
        active_precision = "fp32"
    return device, active_precision


def _evaluate_gates(task_config: Any, metrics: Mapping[str, float]) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    all_passed = True
    for gate in task_config.evaluation_gates:
        if gate.metric not in metrics:
            raise TorchTrainingError(f"evaluation lacks configured gate metric {gate.metric!r}")
        value = float(metrics[gate.metric])
        if not math.isfinite(value):
            raise TorchTrainingError(f"evaluation metric {gate.metric!r} is non-finite")
        passed = value >= float(gate.threshold) if gate.operator == "gte" else value <= float(gate.threshold)
        rows.append(
            {
                "metric": gate.metric,
                "operator": gate.operator,
                "threshold": float(gate.threshold),
                "value": value,
                "passed": passed,
            }
        )
        all_passed = all_passed and passed
    return rows, all_passed


def _data_loader(
    torch: Any,
    dataset: Any,
    task_config: Any,
    training_config: Any,
    *,
    shuffle: bool,
    collate_fn: Callable[..., Any],
    device: Any,
    sample_weights: Sequence[float] | None = None,
) -> Any:
    generator = torch.Generator()
    generator.manual_seed(int(training_config.runtime.seed) + (1 if shuffle else 2))

    sampler = None
    if sample_weights is not None:
        if not shuffle:
            raise TorchTrainingError("weighted sampling is only valid for a shuffled train loader")
        if len(sample_weights) != len(dataset) or not sample_weights:
            raise TorchTrainingError(
                "weighted sampling must provide one non-empty weight per train sample"
            )
        weights = torch.as_tensor(sample_weights, dtype=torch.double)
        if (
            weights.ndim != 1
            or not bool(torch.isfinite(weights).all().item())
            or not bool((weights > 0).all().item())
        ):
            raise TorchTrainingError("weighted sampling requires finite positive weights")
        sampler = torch.utils.data.WeightedRandomSampler(
            weights,
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(task_config.batch_size),
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=int(training_config.runtime.dataloader_workers),
        pin_memory=bool(training_config.runtime.pin_memory) and device.type == "cuda",
        collate_fn=collate_fn,
        generator=generator,
        worker_init_fn=partial(
            _seed_data_loader_worker,
            base_seed=int(training_config.runtime.seed),
        ),
        persistent_workers=int(training_config.runtime.dataloader_workers) > 0,
        drop_last=False,
    )


def _grad_scaler(torch: Any, device: Any, precision: str) -> Any:
    enabled = device.type == "cuda" and precision == "fp16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _seed_data_loader_worker(worker_id: int, *, base_seed: int) -> None:
    worker_seed = (base_seed + int(worker_id)) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _autocast(torch: Any, device: Any, precision: str) -> Any:
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def _backward(scaler: Any, loss: Any) -> None:
    if not bool(loss.isfinite().item()):
        raise TorchTrainingError("training loss is non-finite")
    scaler.scale(loss).backward()


def _optimizer_step(
    torch: Any, scaler: Any, optimizer: Any, parameters: Iterable[Any]
) -> None:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(list(parameters), 5.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def _is_improved(value: float, best: float | None, operator: str) -> bool:
    if not math.isfinite(value):
        raise TorchTrainingError("early-stop monitor is non-finite")
    if best is None:
        return True
    return value > best + 1e-12 if operator == "gte" else value < best - 1e-12


def _model_selection_policy(task_config: Any) -> dict[str, Any]:
    gates = tuple(task_config.evaluation_gates)
    if not gates:
        raise TorchTrainingError("model selection requires at least one evaluation gate")
    return {
        "name": "all_gate_normalized_violation_v1",
        "ordering": [
            "failed_gate_count_ascending",
            "maximum_normalized_violation_ascending",
            "total_normalized_violation_ascending",
            "primary_metric_best",
        ],
        "primary_metric": gates[0].metric,
        "primary_operator": gates[0].operator,
        "violation_normalization": "raw_violation/max(abs(threshold),1e-12)",
        "comparison": {
            "multi_gate": "exact_lexicographic_key",
            "single_gate": "legacy_primary_metric_direction_with_absolute_epsilon",
            "single_gate_absolute_epsilon": 1e-12,
        },
        "gates": [
            {
                "metric": gate.metric,
                "operator": gate.operator,
                "threshold": float(gate.threshold),
            }
            for gate in gates
        ],
    }


def _gate_selection_summary(
    gates: Sequence[Any], metrics: Mapping[str, float]
) -> dict[str, Any]:
    if not gates:
        raise TorchTrainingError("model selection requires at least one evaluation gate")
    failed: list[dict[str, Any]] = []
    violations: list[float] = []
    primary_value: float | None = None
    primary_loss: float | None = None
    for index, gate in enumerate(gates):
        if gate.metric not in metrics:
            raise TorchTrainingError(
                f"validation lacks model-selection metric {gate.metric!r}"
            )
        value = float(metrics[gate.metric])
        threshold = float(gate.threshold)
        if not math.isfinite(value):
            raise TorchTrainingError(
                f"model-selection metric {gate.metric!r} is non-finite"
            )
        if gate.operator == "gte":
            raw_violation = max(0.0, threshold - value)
            loss = -value
        elif gate.operator == "lte":
            raw_violation = max(0.0, value - threshold)
            loss = value
        else:
            raise TorchTrainingError(
                f"unsupported model-selection operator {gate.operator!r}"
            )
        normalized = raw_violation / max(abs(threshold), 1e-12)
        violations.append(normalized)
        if raw_violation > 0.0:
            failed.append(
                {
                    "metric": gate.metric,
                    "operator": gate.operator,
                    "threshold": threshold,
                    "value": value,
                    "normalized_violation": normalized,
                }
            )
        if index == 0:
            primary_value = value
            primary_loss = loss
    assert primary_value is not None and primary_loss is not None
    maximum = max(violations, default=0.0)
    total = sum(violations)
    key = [float(len(failed)), maximum, total, primary_loss]
    return {
        "all_gates_passed": not failed,
        "failed_gate_count": len(failed),
        "failed_gates": failed,
        "maximum_normalized_violation": maximum,
        "total_normalized_violation": total,
        "primary_value": primary_value,
        "key": key,
    }


def _selection_is_improved(
    gates: Sequence[Any],
    candidate: Mapping[str, Any],
    *,
    best_key: tuple[float, ...] | None,
    best_primary_value: float | None,
) -> bool:
    """Compare epochs while preserving the legacy single-gate tolerance."""

    if best_key is None:
        return True
    if len(gates) == 1:
        return _is_improved(
            float(candidate["primary_value"]),
            best_primary_value,
            gates[0].operator,
        )
    candidate_key = tuple(float(item) for item in candidate["key"])
    return candidate_key < best_key


def _restored_selection_key(
    gates: Sequence[Any],
    history: Sequence[Mapping[str, Any]],
    *,
    best_epoch: int,
    best_value: float | None,
    stale_epochs: int,
) -> tuple[float, ...]:
    if not history or not 0 < best_epoch <= len(history):
        raise TorchTrainingError("resume checkpoint model-selection history is invalid")
    summaries: list[dict[str, Any]] = []
    for row in history:
        metrics = row.get("val")
        if not isinstance(metrics, Mapping):
            raise TorchTrainingError(
                "resume checkpoint lacks validation metrics for model selection"
            )
        summaries.append(_gate_selection_summary(gates, metrics))
    selected_index = -1
    selected_key: tuple[float, ...] | None = None
    selected_primary: float | None = None
    replayed_stale_epochs = 0
    for index, summary in enumerate(summaries):
        if _selection_is_improved(
            gates,
            summary,
            best_key=selected_key,
            best_primary_value=selected_primary,
        ):
            selected_index = index
            selected_key = tuple(float(item) for item in summary["key"])
            selected_primary = float(summary["primary_value"])
            replayed_stale_epochs = 0
        else:
            replayed_stale_epochs += 1
    if selected_index + 1 != best_epoch:
        raise TorchTrainingError(
            "resume checkpoint best epoch is incompatible with the current "
            "all-gate model-selection policy"
        )
    selected = summaries[selected_index]
    if best_value is None or not math.isclose(
        float(selected["primary_value"]),
        float(best_value),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise TorchTrainingError(
            "resume checkpoint best value differs from model-selection history"
        )
    if replayed_stale_epochs != stale_epochs:
        raise TorchTrainingError(
            "resume checkpoint stale epoch count differs from model-selection history"
        )
    assert selected_key is not None
    return selected_key


def _bounded_iterable(loader: Any, maximum: int | None) -> Iterable[Any]:
    for index, batch in enumerate(loader):
        if maximum is not None and index >= maximum:
            break
        yield batch


def _bounded_length(loader: Any, maximum: int | None) -> int:
    length = len(loader)
    bounded = min(length, maximum) if maximum is not None else length
    if bounded <= 0:
        raise TorchTrainingError("data loader has no batches")
    return bounded


def _require_nonempty_splits(train: Any, val: Any, task: str) -> None:
    if len(train) <= 0:
        raise TorchTrainingError(f"{task} frozen train split is empty")
    if len(val) <= 0:
        raise TorchTrainingError(f"{task} frozen validation split is empty")


def _require_outcome_files(outcome: _TaskOutcome, output: Path) -> None:
    for label, path in (
        ("model artifact", outcome.artifact_path),
        ("ONNX export", outcome.export_path),
        ("prediction evidence", outcome.predictions_path),
    ):
        resolved = Path(path).expanduser().resolve(strict=False)
        _require_within(resolved, output, label)
        if not resolved.is_file() or resolved.stat().st_size <= 0:
            raise TorchTrainingError(f"{label} is missing or empty: {resolved}")


def _prepare_run_output(
    output: Path, *, workspace_root: Path, limits: _RunLimits
) -> Path | None:
    raw_resume = os.environ.get("PCBPARSE_TRAIN_RESUME_CHECKPOINT", "").strip()
    if limits.smoke and raw_resume:
        raise TorchTrainingError("smoke runs cannot resume formal checkpoints")
    if raw_resume:
        raw_path = Path(raw_resume).expanduser()
        path = raw_path if raw_path.is_absolute() else workspace_root / raw_path
        path = path.resolve(strict=False)
        _require_within(path, workspace_root, "resume checkpoint")
        _require_within(path, output, "resume checkpoint")
        if path.name != "resume_checkpoint.pt" or not path.is_file():
            raise TorchTrainingError(
                "--resume-checkpoint must name an existing resume_checkpoint.pt "
                "inside the selected output directory"
            )
        return path
    if output.exists() and any(output.iterdir()):
        kind = "smoke" if limits.smoke else "formal"
        raise TorchTrainingError(
            f"refusing to overwrite non-empty {kind} output directory {output}; "
            "choose a new directory or pass --resume-checkpoint for a formal run"
        )
    output.mkdir(parents=True, exist_ok=True)
    return None


def _save_training_state(
    *,
    torch: Any,
    path: Path,
    model: Any,
    optimizer: Any,
    scaler: Any,
    loaders: Sequence[Any],
    task_config: Any,
    limits: _RunLimits,
    lineage: Mapping[str, Any],
    completed_epoch: int,
    best_epoch: int,
    best_value: float | None,
    best_model_state: Mapping[str, Any] | None,
    stale_epochs: int,
    history: Sequence[Mapping[str, Any]],
) -> None:
    if limits.smoke:
        return
    if best_model_state is None or best_value is None:
        raise TorchTrainingError("cannot checkpoint training without a best model")
    model_selection = lineage.get("model_selection")
    if not isinstance(model_selection, Mapping):
        raise TorchTrainingError(
            "cannot checkpoint training without model-selection lineage"
        )
    loader_states: list[Any] = []
    for loader in loaders:
        generator = getattr(loader, "generator", None)
        loader_states.append(
            generator.get_state().cpu() if generator is not None else None
        )
    document = {
        "schema": "pcbparse.torch_training_resume.v2",
        "task": lineage["task"],
        "training_config_sha256": lineage["training_config_sha256"],
        "materialization_commit_id": lineage["materialization_commit_id"],
        "task_manifest_sha256": lineage["task_manifest_sha256"],
        "architecture": task_config.architecture,
        "input_size": list(task_config.input_size),
        "planned_epochs": limits.epochs,
        "runtime": dict(lineage["runtime"]),
        "model_selection": dict(model_selection),
        "completed_epoch": completed_epoch,
        "best_epoch": best_epoch,
        "best_value": float(best_value),
        "stale_epochs": stale_epochs,
        "model_state": _cpu_state_tree(model.state_dict(), torch),
        "best_model_state": _cpu_state_tree(dict(best_model_state), torch),
        "optimizer_state": _cpu_state_tree(optimizer.state_dict(), torch),
        "scaler_state": _cpu_state_tree(scaler.state_dict(), torch),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state().cpu(),
        "cuda_random_states": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
        "loader_generator_states": loader_states,
        "history": [dict(row) for row in history],
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(document, temporary)
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise TorchTrainingError("PyTorch produced an empty resume checkpoint")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _restore_training_state(
    *,
    torch: Any,
    path: Path,
    model: Any,
    optimizer: Any,
    scaler: Any,
    loaders: Sequence[Any],
    task_config: Any,
    limits: _RunLimits,
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        document = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise TorchTrainingError(
            f"cannot load resume checkpoint: {type(exc).__name__}: {exc}"
        ) from exc
    if isinstance(document, Mapping) and document.get("schema") == (
        "pcbparse.torch_training_resume.v1"
    ):
        raise TorchTrainingError(
            "resume checkpoint predates the all-gate model-selection policy; "
            "historical v1 resume state is unsupported"
        )
    if not isinstance(document, Mapping) or document.get("schema") != (
        "pcbparse.torch_training_resume.v2"
    ):
        raise TorchTrainingError("resume checkpoint schema is unsupported")
    model_selection = lineage.get("model_selection")
    if not isinstance(model_selection, Mapping):
        raise TorchTrainingError("formal run lacks model-selection lineage")
    expected = {
        "task": lineage["task"],
        "materialization_commit_id": lineage["materialization_commit_id"],
        "task_manifest_sha256": lineage["task_manifest_sha256"],
        "architecture": task_config.architecture,
        "input_size": list(task_config.input_size),
        "planned_epochs": limits.epochs,
        "runtime": dict(lineage["runtime"]),
        "model_selection": dict(model_selection),
    }
    for name, value in expected.items():
        if document.get(name) != value:
            raise TorchTrainingError(
                f"resume checkpoint {name} differs from this formal run"
            )
    compatible_config_hashes = lineage.get(
        "resume_compatible_training_config_sha256s",
        [lineage["training_config_sha256"]],
    )
    if (
        not isinstance(compatible_config_hashes, Sequence)
        or isinstance(compatible_config_hashes, (str, bytes))
        or any(not isinstance(value, str) for value in compatible_config_hashes)
        or document.get("training_config_sha256") not in compatible_config_hashes
    ):
        raise TorchTrainingError(
            "resume checkpoint training_config_sha256 differs from this formal run"
        )
    completed = document.get("completed_epoch")
    best_epoch = document.get("best_epoch")
    stale_epochs = document.get("stale_epochs")
    best_value = document.get("best_value")
    if (
        type(completed) is not int
        or not 0 < completed <= limits.epochs
        or type(best_epoch) is not int
        or not 0 < best_epoch <= completed
        or type(stale_epochs) is not int
        or stale_epochs < 0
        or not isinstance(best_value, (int, float))
        or isinstance(best_value, bool)
        or not math.isfinite(float(best_value))
    ):
        raise TorchTrainingError("resume checkpoint epoch/monitor state is invalid")
    model_state = document.get("model_state")
    best_model_state = document.get("best_model_state")
    optimizer_state = document.get("optimizer_state")
    scaler_state = document.get("scaler_state")
    if not all(
        isinstance(value, Mapping)
        for value in (model_state, best_model_state, optimizer_state, scaler_state)
    ):
        raise TorchTrainingError("resume checkpoint model/optimizer state is invalid")
    try:
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
        scaler.load_state_dict(scaler_state)
    except Exception as exc:
        raise TorchTrainingError(
            f"resume checkpoint state is incompatible: {type(exc).__name__}: {exc}"
        ) from exc
    loader_states = document.get("loader_generator_states")
    if not isinstance(loader_states, list) or len(loader_states) != len(loaders):
        raise TorchTrainingError("resume checkpoint loader state is invalid")
    for loader, state in zip(loaders, loader_states, strict=True):
        generator = getattr(loader, "generator", None)
        if generator is None and state is not None:
            raise TorchTrainingError("resume checkpoint has unexpected loader RNG state")
        if generator is not None:
            if not torch.is_tensor(state):
                raise TorchTrainingError("resume checkpoint loader RNG state is missing")
            generator.set_state(state)
    try:
        random.setstate(document["python_random_state"])
        np.random.set_state(document["numpy_random_state"])
        torch.set_rng_state(document["torch_random_state"])
        cuda_states = document.get("cuda_random_states", [])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_states)
    except Exception as exc:
        raise TorchTrainingError(
            f"resume checkpoint RNG state is incompatible: {type(exc).__name__}: {exc}"
        ) from exc
    history = document.get("history")
    if (
        not isinstance(history, list)
        or len(history) != completed
        or any(not isinstance(row, Mapping) for row in history)
        or [row.get("epoch") for row in history] != list(range(1, completed + 1))
    ):
        raise TorchTrainingError("resume checkpoint training history is invalid")
    return {
        "completed_epoch": completed,
        "best_epoch": best_epoch,
        "best_value": float(best_value),
        "best_model_state": dict(best_model_state),
        "stale_epochs": stale_epochs,
        "history": [dict(row) for row in history],
    }


def _cpu_state_tree(value: Any, torch: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_state_tree(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state_tree(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state_tree(item, torch) for item in value)
    return copy.deepcopy(value)


def _write_training_history(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    path.unlink(missing_ok=True)
    for row in history:
        _append_jsonl(path, row)


def _workspace_root(preflight: Mapping[str, Any]) -> Path:
    raw = preflight.get("workspace_root")
    if not isinstance(raw, str) or not raw:
        raise TorchTrainingError("preflight lacks workspace_root")
    root = Path(raw).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise TorchTrainingError(f"preflight workspace_root is not a directory: {root}")
    return root


def _source_root(preflight: Mapping[str, Any]) -> Path:
    paths = preflight.get("paths")
    if not isinstance(paths, Mapping):
        raise TorchTrainingError("preflight lacks resolved paths")
    raw = paths.get("source_root")
    if not isinstance(raw, str) or not raw:
        raise TorchTrainingError("preflight lacks source_root")
    path = Path(raw).expanduser().resolve(strict=False)
    if not path.is_dir():
        raise TorchTrainingError(f"preflight source_root is not a directory: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise TorchTrainingError(
            f"cannot read frozen JSON {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise TorchTrainingError(f"frozen JSON root is not an object: {path}")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TorchTrainingError(f"{label} must be an object")
    return dict(value)


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise TorchTrainingError(f"{label} is not a lowercase SHA-256")
    return value


def _expect_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise TorchTrainingError(
            f"{label} differs from the frozen training configuration: "
            f"expected {expected!r}, actual {actual!r}"
        )


def _verify_file(path: Path, expected: Any, label: str) -> None:
    digest = _sha(expected, f"{label} expected hash")
    if not path.is_file() or path.stat().st_size <= 0:
        raise TorchTrainingError(f"{label} is missing or empty: {path}")
    actual = _sha256_file(path)
    if actual != digest:
        raise TorchTrainingError(
            f"{label} SHA-256 mismatch: expected {digest}, actual {actual}"
        )


def _require_within(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise TorchTrainingError(f"{label} escapes its frozen root") from exc


def _write_metrics(
    output: Path, metrics: Mapping[str, float], epochs: _EpochResult, limits: _RunLimits
) -> None:
    _write_json(
        output / "metrics.json",
        {
            "schema_version": 1,
            "split": "val",
            "metrics": dict(metrics),
            "planned_epochs": limits.epochs,
            "best_epoch": epochs.best_epoch,
            "completed_epochs": epochs.completed_epochs,
            "early_stopped": epochs.early_stopped,
            "smoke": limits.smoke,
            "formal_evidence": not limits.smoke,
        },
    )


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(payload + "\n")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    try:
        temporary.write_text(payload, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cuda_memory(torch: Any, device: Any) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise TorchTrainingError(f"{name} must be a boolean")


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise TorchTrainingError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise TorchTrainingError(f"{name} must be a positive integer")
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_torch() -> Any:
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise TorchTrainingError(
            "the optional PyTorch training runtime is unavailable; use the "
            "project .venv-train environment"
        ) from exc
    return torch


__all__ = [
    "TorchSmokeRunComplete",
    "TorchTrainingError",
    "run_torch_task",
]
