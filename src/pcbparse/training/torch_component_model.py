from __future__ import annotations

"""TorchVision Component detector training and artifact utilities.

The module intentionally imports neither :mod:`torch` nor :mod:`torchvision` at
module import time.  PCBParse's inference/runtime package therefore remains
importable in a lightweight environment; a clear dependency error is raised
only when a torch-backed operation is requested.

The supported detector is TorchVision's real
``fasterrcnn_mobilenet_v3_large_320_fpn`` architecture.  ``build_model`` always
passes ``weights=None`` and ``weights_backbone=None`` so tests, offline builds,
and production startup can never initiate an implicit network download.  A
local, explicitly supplied state file is the only pretraining path.
"""

import hashlib
import importlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CHECKPOINT_FORMAT = "pcbparse-torch-component-v2"
EXPORT_FORMAT = "pcbparse-torch-component-export-v2"
_LEGACY_CHECKPOINT_FORMAT = "pcbparse-torch-component-v1"
DEFAULT_ARCHITECTURE = "fasterrcnn_mobilenet_v3_large_320_fpn"
_SUPPORTED_ARCHITECTURES = frozenset({DEFAULT_ARCHITECTURE})


class TorchComponentDependencyError(RuntimeError):
    """Raised when an explicitly requested torch operation is unavailable."""


class TorchComponentArtifactError(RuntimeError):
    """Raised when a checkpoint/export violates the frozen artifact contract."""


@dataclass(slots=True)
class LoadedComponentCheckpoint:
    """A reconstructed detector and its validated, immutable artifact metadata."""

    model: Any
    architecture: str
    category_table: tuple[dict[str, Any], ...]
    preprocessing: dict[str, Any]
    framework: dict[str, str]
    lineage: dict[str, Any]
    epoch: int
    metrics: dict[str, Any]
    optimizer_state: dict[str, Any] | None
    checkpoint_sha256: str
    path: Path

    @property
    def num_classes(self) -> int:
        return len(self.category_table) + 1


def build_model(
    num_classes: int | None = None,
    *,
    category_table: Mapping[int, str] | Sequence[Mapping[str, Any] | str] | None = None,
    architecture: str = DEFAULT_ARCHITECTURE,
    pretrained_state_path: str | Path | None = None,
    strict_pretrained: bool = True,
    detector_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Build the production Component detector without downloading weights.

    ``num_classes`` follows TorchVision and *includes* background class 0.  It
    can be omitted when ``category_table`` is provided.  Official category IDs
    may be sparse and may include 0; the frozen label contract maps them to
    contiguous TorchVision foreground labels ``1..N``.

    ``detector_kwargs`` is recorded on the model and frozen into every
    checkpoint so reconstruction is exact.  This also permits small ``min_size``
    and RPN proposal limits for bounded smoke tests without changing the formal
    Faster R-CNN architecture.
    """

    torch, torchvision = _require_torchvision()
    if architecture not in _SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"unsupported Component architecture {architecture!r}; "
            f"supported={sorted(_SUPPORTED_ARCHITECTURES)}"
        )
    normalized_categories: tuple[dict[str, Any], ...] | None = None
    if category_table is not None:
        normalized_categories = normalize_category_table(category_table)
        inferred = len(normalized_categories) + 1
        if num_classes is not None and num_classes != inferred:
            raise ValueError(
                f"num_classes={num_classes} differs from category-table head size {inferred}"
            )
        num_classes = inferred
    if type(num_classes) is not int or num_classes < 2:
        raise ValueError("num_classes must be an integer >= 2 including background")

    raw_kwargs = dict(detector_kwargs or {})
    forbidden = {"weights", "weights_backbone", "num_classes"}.intersection(raw_kwargs)
    if forbidden:
        raise ValueError(
            "detector_kwargs may not override fail-closed weight/class arguments: "
            + ", ".join(sorted(forbidden))
        )
    frozen_kwargs = _json_mapping(raw_kwargs, "detector_kwargs")
    constructor = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn
    model = constructor(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes,
        **raw_kwargs,
    )
    model._pcbparse_architecture = architecture
    model._pcbparse_num_classes = num_classes
    model._pcbparse_detector_kwargs = frozen_kwargs
    if normalized_categories is not None:
        model._pcbparse_category_table = [dict(item) for item in normalized_categories]

    if pretrained_state_path is not None:
        state_path = Path(pretrained_state_path).expanduser().resolve(strict=True)
        payload = _safe_torch_load(torch, state_path, map_location="cpu")
        state = _extract_model_state(payload, state_path)
        if strict_pretrained:
            model.load_state_dict(state, strict=True)
        else:
            # ``Module.load_state_dict(strict=False)`` tolerates missing and
            # unexpected keys, but it still raises for a same-named tensor
            # whose shape differs.  A COCO detector necessarily has a
            # different classification/regression head from Dataset300, so
            # filter by both name and shape and make every skipped parameter
            # auditable instead of silently abandoning all pretraining.
            target_state = model.state_dict()
            compatible: dict[str, Any] = {}
            unexpected: list[str] = []
            shape_mismatch: list[dict[str, Any]] = []
            remapped_keys: list[dict[str, str]] = []
            for source_key, value in state.items():
                key = _remap_legacy_detection_key(source_key, target_state)
                if key != source_key:
                    remapped_keys.append({"source": source_key, "target": key})
                if key not in target_state:
                    unexpected.append(source_key)
                    continue
                target = target_state[key]
                if tuple(value.shape) != tuple(target.shape):
                    shape_mismatch.append(
                        {
                            "key": key,
                            "source_shape": list(value.shape),
                            "target_shape": list(target.shape),
                        }
                    )
                    continue
                compatible[key] = value
            incompatibility = model.load_state_dict(compatible, strict=False)
            model._pcbparse_pretrained_incompatibility = {
                "loaded_keys": len(compatible),
                "missing_keys": sorted(incompatibility.missing_keys),
                "unexpected_keys": sorted(
                    set(unexpected).union(incompatibility.unexpected_keys)
                ),
                "shape_mismatch": sorted(
                    shape_mismatch, key=lambda item: str(item["key"])
                ),
                "remapped_keys": sorted(
                    remapped_keys, key=lambda item: (item["source"], item["target"])
                ),
            }
        model._pcbparse_pretrained_sha256 = _sha256_file(state_path)
    return model


def _remap_legacy_detection_key(
    key: str, target_state: Mapping[str, Any]
) -> str:
    """Map official legacy TorchVision detection keys to the current tree."""

    if key in target_state:
        return key
    if key.startswith("backbone.fpn.inner_blocks.") or key.startswith(
        "backbone.fpn.layer_blocks."
    ):
        prefix, suffix = key.rsplit(".", 1)
        candidate = f"{prefix}.0.{suffix}"
        return candidate if candidate in target_state else key
    if key.startswith("rpn.head.conv."):
        prefix, suffix = key.rsplit(".", 1)
        candidate = f"{prefix}.0.0.{suffix}"
        return candidate if candidate in target_state else key
    return key


def normalize_category_table(
    categories: Mapping[int, str] | Sequence[Mapping[str, Any] | str],
) -> tuple[dict[str, Any], ...]:
    """Return the frozen official-ID/model-label Component mapping.

    Canonical rows always contain ``official_id``, ``model_label``, ``name``,
    and the COCO-compatible alias ``id == model_label``.  Materialized
    Dataset300 rows (``id`` plus ``source_category_id``) are accepted without
    remapping their already-contiguous model labels.  Plain legacy COCO rows
    are treated as official IDs and deterministically mapped, in official-ID
    order, to labels ``1..N``.  This input migration is deliberately separate
    from checkpoint compatibility: v1 checkpoints are rejected because their
    detector heads cannot prove which interpretation of ``id`` was used.
    """

    rows: list[dict[str, Any]] = []
    if isinstance(categories, Mapping):
        source: Iterable[tuple[Any, Any]] = categories.items()
        for raw_id, raw_name in source:
            rows.append({"official_id": raw_id, "name": raw_name})
    elif isinstance(categories, Sequence) and not isinstance(categories, (str, bytes)):
        for index, raw in enumerate(categories):
            if isinstance(raw, str):
                rows.append({"official_id": index, "name": raw})
            elif isinstance(raw, Mapping):
                rows.append(dict(raw))
            else:
                raise ValueError("category_table rows must be mappings or strings")
    else:
        raise ValueError("category_table must be a mapping or sequence")
    if not rows:
        raise ValueError("category_table cannot be empty")

    parsed: list[tuple[int, int | None, dict[str, Any]]] = []
    seen_official_ids: set[int] = set()
    seen_model_labels: set[int] = set()
    seen_names: set[str] = set()
    for row in rows:
        explicit_model_label = row.get("model_label")
        if explicit_model_label is None and "source_category_id" in row:
            explicit_model_label = row.get("id")
        raw_official_id = row.get(
            "official_id",
            row.get("source_category_id", row.get("id")),
        )
        official_id = _nonnegative_category_id(raw_official_id, "official_id")
        model_label: int | None = None
        if explicit_model_label is not None:
            if type(explicit_model_label) is not int or explicit_model_label <= 0:
                raise ValueError("Component model_label values must be positive integers")
            model_label = explicit_model_label
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Component category {official_id} has an empty name")
        name = name.strip()
        if official_id in seen_official_ids:
            raise ValueError(f"duplicate Component official_id {official_id}")
        if model_label is not None and model_label in seen_model_labels:
            raise ValueError(f"duplicate Component model_label {model_label}")
        if name in seen_names:
            raise ValueError(f"duplicate Component category name {name!r}")
        seen_official_ids.add(official_id)
        if model_label is not None:
            seen_model_labels.add(model_label)
        seen_names.add(name)
        clean = dict(row)
        clean["name"] = name
        parsed.append((official_id, model_label, clean))

    explicit_count = sum(model_label is not None for _, model_label, _ in parsed)
    if explicit_count not in (0, len(parsed)):
        raise ValueError(
            "Component category table mixes explicit and implicit model labels"
        )
    if explicit_count == 0:
        ordered = sorted(parsed, key=lambda item: (item[0], item[2]["name"]))
        parsed = [
            (official_id, index, row)
            for index, (official_id, _model_label, row) in enumerate(ordered, start=1)
        ]
    else:
        expected = set(range(1, len(parsed) + 1))
        if seen_model_labels != expected:
            raise ValueError(
                "Component model_label values must be contiguous 1..N; "
                f"expected={sorted(expected)}, actual={sorted(seen_model_labels)}"
            )

    normalized: list[dict[str, Any]] = []
    for official_id, model_label, row in parsed:
        assert model_label is not None
        clean = {
            "id": model_label,
            "official_id": official_id,
            "model_label": model_label,
            "name": row["name"],
            "source_category_id": str(official_id),
        }
        # Preserve JSON-safe descriptive fields (for example COCO
        # ``supercategory`` and the original source ID) without accepting
        # executable/pickled values.
        for key, value in sorted(row.items(), key=lambda item: str(item[0])):
            if key in {
                "id",
                "official_id",
                "model_label",
                "name",
                "source_category_id",
            }:
                continue
            if not isinstance(key, str) or not key:
                raise ValueError("category metadata keys must be non-empty strings")
            clean[key] = _json_value(value, f"category[{official_id}].{key}")
        normalized.append(clean)
    normalized.sort(key=lambda item: item["model_label"])
    return tuple(normalized)


def _nonnegative_category_id(value: Any, label: str) -> int:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or not stripped.isdecimal():
            raise ValueError(f"Component {label} must be a non-negative integer")
        value = int(stripped)
    if type(value) is not int or value < 0:
        raise ValueError(f"Component {label} must be a non-negative integer")
    return value


def category_table_from_dataset(dataset: Any) -> tuple[dict[str, Any], ...]:
    """Extract and validate ``ComponentDetectionDataset.categories``."""

    categories = getattr(dataset, "categories", None)
    if categories is None:
        raise ValueError("dataset does not expose a categories table")
    return normalize_category_table(categories)


def train_one_epoch(
    model: Any,
    data_loader: Iterable[Any],
    optimizer: Any,
    device: str | Any,
    *,
    grad_clip_norm: float | None = None,
) -> dict[str, Any]:
    """Train one real Faster R-CNN epoch and return sample-weighted losses."""

    torch, _ = _require_torchvision()
    resolved_device = torch.device(device)
    if grad_clip_norm is not None and (
        isinstance(grad_clip_norm, bool)
        or not isinstance(grad_clip_norm, (int, float))
        or not math.isfinite(float(grad_clip_norm))
        or grad_clip_norm <= 0
    ):
        raise ValueError("grad_clip_norm must be a finite positive number")
    model.to(resolved_device)
    model.train()
    totals: dict[str, float] = {}
    total_examples = 0
    batch_count = 0
    for raw_batch in data_loader:
        images, targets = _prepare_detection_batch(raw_batch, torch, resolved_device)
        optimizer.zero_grad(set_to_none=True)
        losses = model(images, targets)
        if not isinstance(losses, Mapping) or not losses:
            raise RuntimeError("Component detector training forward returned no loss mapping")
        total_loss = sum(loss for loss in losses.values())
        if total_loss.ndim != 0 or not bool(torch.isfinite(total_loss).item()):
            details = {
                str(name): float(loss.detach().cpu())
                for name, loss in losses.items()
            }
            raise FloatingPointError(f"non-finite Component loss: {details}")
        total_loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(grad_clip_norm),
            )
        optimizer.step()
        count = len(images)
        total_examples += count
        batch_count += 1
        for name, loss in losses.items():
            value = float(loss.detach().cpu())
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite Component loss {name}={value}")
            totals[str(name)] = totals.get(str(name), 0.0) + value * count
        totals["loss_total"] = totals.get("loss_total", 0.0) + float(
            total_loss.detach().cpu()
        ) * count
    if batch_count == 0 or total_examples == 0:
        raise ValueError("Component training loader produced no batches")
    return {
        "num_batches": batch_count,
        "num_examples": total_examples,
        "losses": {
            name: value / total_examples for name, value in sorted(totals.items())
        },
    }


def evaluate_map50(
    model: Any,
    data_loader: Iterable[Any],
    device: str | Any,
    *,
    score_threshold: float = 0.0,
    iou_threshold: float = 0.5,
    category_table: Mapping[int, str] | Sequence[Mapping[str, Any] | str] | None = None,
) -> dict[str, Any]:
    """Evaluate one epoch with COCO-style 101-point interpolated AP at IoU 0.5.

    Matching is class-aware and one-to-one per image.  Predictions are globally
    sorted by confidence within each class, exactly as AP requires; this is not
    an image-wise mean or a best-IoU proxy.  Classes without ground truth are
    excluded from macro mAP and reported explicitly.
    """

    torch, _ = _require_torchvision()
    score_threshold = _unit_interval(score_threshold, "score_threshold")
    iou_threshold = _unit_interval(iou_threshold, "iou_threshold", inclusive_zero=False)
    resolved_device = torch.device(device)
    model.to(resolved_device)
    was_training = bool(model.training)
    model.eval()
    ground_truth: dict[int, dict[int, list[tuple[float, float, float, float]]]] = {}
    predictions: dict[int, list[tuple[float, int, int, tuple[float, float, float, float]]]] = {}
    image_count = 0
    try:
        with torch.inference_mode():
            for raw_batch in data_loader:
                images, targets = _prepare_detection_batch(raw_batch, torch, resolved_device)
                outputs = model(images)
                if not isinstance(outputs, Sequence) or len(outputs) != len(images):
                    raise RuntimeError("Component detector evaluation output cardinality mismatch")
                for local_index, (target, output) in enumerate(zip(targets, outputs)):
                    image_index = image_count + local_index
                    target_boxes = _cpu_box_rows(target.get("boxes"), "target boxes")
                    target_labels = _cpu_int_rows(target.get("labels"), "target labels")
                    if len(target_boxes) != len(target_labels):
                        raise ValueError("target boxes/labels cardinality mismatch")
                    for box, label in zip(target_boxes, target_labels):
                        if label <= 0:
                            raise ValueError("ground-truth Component labels must be positive")
                        ground_truth.setdefault(label, {}).setdefault(image_index, []).append(box)

                    if not isinstance(output, Mapping):
                        raise RuntimeError("Component detector output must be a mapping")
                    output_boxes = _cpu_box_rows(output.get("boxes"), "prediction boxes")
                    output_labels = _cpu_int_rows(output.get("labels"), "prediction labels")
                    output_scores = _cpu_float_rows(output.get("scores"), "prediction scores")
                    if not (len(output_boxes) == len(output_labels) == len(output_scores)):
                        raise RuntimeError("prediction boxes/labels/scores cardinality mismatch")
                    for prediction_index, (box, label, score) in enumerate(
                        zip(output_boxes, output_labels, output_scores)
                    ):
                        if score < score_threshold or label <= 0:
                            continue
                        predictions.setdefault(label, []).append(
                            (score, image_index, prediction_index, box)
                        )
                image_count += len(images)
    finally:
        if was_training:
            model.train()
    if image_count == 0:
        raise ValueError("Component evaluation loader produced no images")

    normalized_categories = (
        normalize_category_table(category_table) if category_table is not None else ()
    )
    category_rows = {
        item["model_label"]: item for item in normalized_categories
    }
    all_class_ids = sorted(set(ground_truth).union(predictions).union(category_rows))
    per_class: dict[str, dict[str, Any]] = {}
    evaluated_aps: list[float] = []
    total_tp = total_fp = total_fn = 0
    for class_id in all_class_ids:
        gt_by_image = ground_truth.get(class_id, {})
        gt_count = sum(len(rows) for rows in gt_by_image.values())
        matched = {
            image_id: [False] * len(rows) for image_id, rows in gt_by_image.items()
        }
        ranked = sorted(
            predictions.get(class_id, []),
            key=lambda item: (-item[0], item[1], item[2]),
        )
        tp_flags: list[int] = []
        fp_flags: list[int] = []
        for _score, image_id, _prediction_index, box in ranked:
            candidates = gt_by_image.get(image_id, [])
            best_index = -1
            best_iou = -1.0
            for gt_index, gt_box in enumerate(candidates):
                if matched[image_id][gt_index]:
                    continue
                overlap = _box_iou(box, gt_box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = gt_index
            if best_index >= 0 and best_iou >= iou_threshold:
                matched[image_id][best_index] = True
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)
        tp = sum(tp_flags)
        fp = sum(fp_flags)
        fn = gt_count - tp
        ap50 = _interpolated_ap101(tp_flags, fp_flags, gt_count) if gt_count else None
        if ap50 is not None:
            evaluated_aps.append(ap50)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_class[str(class_id)] = {
            "name": category_rows.get(class_id, {}).get("name", str(class_id)),
            "model_label": class_id,
            "official_category_id": category_rows.get(class_id, {}).get("official_id"),
            "ground_truth": gt_count,
            "predictions": len(ranked),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "ap50": ap50,
        }
    map50 = sum(evaluated_aps) / len(evaluated_aps) if evaluated_aps else None
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    return {
        "metric": "coco_style_101_point_ap",
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "num_images": image_count,
        "map50": map50,
        "coco_style_map50": map50,
        "precision50": precision,
        "recall50": recall,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "evaluated_class_ids": [
            class_id for class_id in all_class_ids if class_id in ground_truth
        ],
        "classes_without_ground_truth": [
            class_id for class_id in all_class_ids if class_id not in ground_truth
        ],
        "per_class": per_class,
    }


def evaluate_one_epoch(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility name for :func:`evaluate_map50`."""

    return evaluate_map50(*args, **kwargs)


def default_preprocessing(model: Any) -> dict[str, Any]:
    """Describe the only pixel contract accepted by the runtime adapter."""

    transform = getattr(model, "transform", None)
    min_size = list(getattr(transform, "min_size", (320,)))
    max_size = int(getattr(transform, "max_size", 640))
    image_mean = [float(value) for value in getattr(transform, "image_mean", [0.485, 0.456, 0.406])]
    image_std = [float(value) for value in getattr(transform, "image_std", [0.229, 0.224, 0.225])]
    return {
        "color_space": "RGB",
        "tensor_layout": "CHW",
        "pixel_dtype": "float32",
        "pixel_range": [0.0, 1.0],
        "resize_owner": "torchvision_generalized_rcnn_transform",
        "min_size": min_size,
        "max_size": max_size,
        "image_mean": image_mean,
        "image_std": image_std,
        "coordinate_output": "input_view_xyxy",
    }


def save_checkpoint(
    path: str | Path,
    model: Any,
    *,
    category_table: Mapping[int, str] | Sequence[Mapping[str, Any] | str],
    lineage: Mapping[str, Any],
    preprocessing: Mapping[str, Any] | None = None,
    optimizer: Any | None = None,
    epoch: int = 0,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically save a portable, ``weights_only=True``-loadable checkpoint."""

    torch, torchvision = _require_torchvision()
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    categories = normalize_category_table(category_table)
    model_categories = getattr(model, "_pcbparse_category_table", None)
    if model_categories is not None and normalize_category_table(model_categories) != categories:
        raise TorchComponentArtifactError(
            "checkpoint category table differs from the table used to build the detector"
        )
    architecture = getattr(model, "_pcbparse_architecture", None)
    if architecture not in _SUPPORTED_ARCHITECTURES:
        raise TorchComponentArtifactError(
            "model is not a reconstructable PCBParse Component detector; use build_model()"
        )
    expected_num_classes = len(categories) + 1
    actual_num_classes = _model_num_classes(model)
    if actual_num_classes != expected_num_classes:
        raise TorchComponentArtifactError(
            f"detector head has {actual_num_classes} classes but category table requires "
            f"{expected_num_classes}"
        )
    if type(epoch) is not int or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    clean_lineage = _json_mapping(lineage, "lineage")
    if not clean_lineage:
        raise ValueError("lineage cannot be empty")
    clean_preprocessing = _json_mapping(
        preprocessing if preprocessing is not None else default_preprocessing(model),
        "preprocessing",
    )
    _validate_preprocessing(clean_preprocessing)
    _validate_preprocessing_against_model(clean_preprocessing, model)
    clean_metrics = _json_mapping(metrics or {}, "metrics")
    detector_kwargs = _json_mapping(
        getattr(model, "_pcbparse_detector_kwargs", {}), "detector_kwargs"
    )
    state = {
        str(name): tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "architecture": architecture,
        "num_classes": expected_num_classes,
        "category_table": [dict(item) for item in categories],
        "preprocessing": clean_preprocessing,
        "framework": {
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
        },
        "lineage": clean_lineage,
        "epoch": epoch,
        "metrics": clean_metrics,
        "build": {"detector_kwargs": detector_kwargs},
        "model_state": state,
        "optimizer_state": _cpu_tree(optimizer.state_dict(), torch) if optimizer is not None else None,
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    digest = _sha256_file(destination)
    return {
        "path": str(destination),
        "sha256": digest,
        "bytes": destination.stat().st_size,
        "format": CHECKPOINT_FORMAT,
        "architecture": architecture,
        "num_classes": expected_num_classes,
        "epoch": epoch,
    }


def load_checkpoint(
    path: str | Path,
    *,
    device: str | Any = "cpu",
    expected_sha256: str | None = None,
    strict: bool = True,
) -> LoadedComponentCheckpoint:
    """Validate and reconstruct a Component detector from a safe checkpoint."""

    torch, _ = _require_torchvision()
    checkpoint_path = Path(path).expanduser().resolve(strict=True)
    digest = _sha256_file(checkpoint_path)
    if expected_sha256 is not None:
        if not _is_sha256(expected_sha256):
            raise ValueError("expected_sha256 must be a lowercase/uppercase SHA-256 hex string")
        if digest.lower() != expected_sha256.lower():
            raise TorchComponentArtifactError(
                f"Component checkpoint SHA-256 mismatch: expected {expected_sha256}, actual {digest}"
            )
    payload = _safe_torch_load(torch, checkpoint_path, map_location="cpu")
    if isinstance(payload, Mapping) and payload.get("format") == _LEGACY_CHECKPOINT_FORMAT:
        raise TorchComponentArtifactError(
            "legacy Component checkpoint v1 lacks the explicit official_id/model_label "
            "contract and must be retrained or explicitly migrated"
        )
    if not isinstance(payload, Mapping) or payload.get("format") != CHECKPOINT_FORMAT:
        raise TorchComponentArtifactError(
            f"unsupported Component checkpoint format in {checkpoint_path}"
        )
    architecture = payload.get("architecture")
    if architecture not in _SUPPORTED_ARCHITECTURES:
        raise TorchComponentArtifactError(f"unsupported checkpoint architecture {architecture!r}")
    categories = normalize_category_table(_require_sequence(payload.get("category_table"), "category_table"))
    expected_num_classes = len(categories) + 1
    if payload.get("num_classes") != expected_num_classes:
        raise TorchComponentArtifactError("checkpoint num_classes/category table mismatch")
    preprocessing = _json_mapping(_require_mapping(payload.get("preprocessing"), "preprocessing"), "preprocessing")
    _validate_preprocessing(preprocessing)
    framework_raw = _require_mapping(payload.get("framework"), "framework")
    framework = _json_mapping(framework_raw, "framework")
    if not all(isinstance(framework.get(name), str) and framework[name] for name in ("torch", "torchvision")):
        raise TorchComponentArtifactError("checkpoint framework versions are missing")
    lineage = _json_mapping(_require_mapping(payload.get("lineage"), "lineage"), "lineage")
    if not lineage:
        raise TorchComponentArtifactError("checkpoint lineage is empty")
    epoch = payload.get("epoch")
    if type(epoch) is not int or epoch < 0:
        raise TorchComponentArtifactError("checkpoint epoch is invalid")
    metrics = _json_mapping(_require_mapping(payload.get("metrics", {}), "metrics"), "metrics")
    build = _require_mapping(payload.get("build"), "build")
    detector_kwargs = _json_mapping(
        _require_mapping(build.get("detector_kwargs", {}), "build.detector_kwargs"),
        "build.detector_kwargs",
    )
    state = _extract_model_state(payload, checkpoint_path)
    model = build_model(
        expected_num_classes,
        category_table=categories,
        architecture=str(architecture),
        detector_kwargs=detector_kwargs,
    )
    _validate_preprocessing_against_model(preprocessing, model)
    incompatibility = model.load_state_dict(state, strict=bool(strict))
    if not strict and (incompatibility.missing_keys or incompatibility.unexpected_keys):
        model._pcbparse_checkpoint_incompatibility = {
            "missing_keys": list(incompatibility.missing_keys),
            "unexpected_keys": list(incompatibility.unexpected_keys),
        }
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise TorchComponentDependencyError("CUDA was requested for Component inference but is unavailable")
    model.to(resolved_device)
    model.eval()
    optimizer_state_raw = payload.get("optimizer_state")
    if optimizer_state_raw is not None and not isinstance(optimizer_state_raw, Mapping):
        raise TorchComponentArtifactError("checkpoint optimizer_state is invalid")
    return LoadedComponentCheckpoint(
        model=model,
        architecture=str(architecture),
        category_table=categories,
        preprocessing=preprocessing,
        framework={name: str(value) for name, value in framework.items()},
        lineage=lineage,
        epoch=epoch,
        metrics=metrics,
        optimizer_state=dict(optimizer_state_raw) if optimizer_state_raw is not None else None,
        checkpoint_sha256=digest,
        path=checkpoint_path,
    )


def export_model(
    model: Any,
    path: str | Path,
    sample_image: Any,
    *,
    export_format: str | None = None,
    opset_version: int = 18,
) -> dict[str, Any]:
    """Export a single-image detector wrapper to TorchScript or ONNX.

    The exported callable accepts ``[1,3,H,W]`` float RGB in ``[0,1]`` and
    returns ``(boxes, scores, labels)`` in input-image coordinates.  The model's
    own GeneralizedRCNN transform remains inside the graph.
    """

    torch, _ = _require_torchvision()
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    kind = _resolve_export_format(destination, export_format)
    batch = _normalize_export_input(sample_image, torch)
    device = _model_device(model, torch)
    batch = batch.to(device)

    class _SingleImageDetector(torch.nn.Module):
        def __init__(self, detector: Any) -> None:
            super().__init__()
            self.detector = detector

        def forward(self, image_batch: Any) -> tuple[Any, Any, Any]:
            prediction = self.detector([image_batch[0]])[0]
            return prediction["boxes"], prediction["scores"], prediction["labels"]

    wrapper = _SingleImageDetector(model).eval()
    was_training = bool(model.training)
    model.eval()
    temporary_path = (
        Path(tempfile.gettempdir()) / f".{destination.name}.{os.getpid()}.tmp"
        if kind == "torchscript"
        else destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    )
    temporary_path.unlink(missing_ok=True)
    try:
        if kind == "torchscript":
            with torch.inference_mode():
                traced = torch.jit.trace(wrapper, batch, strict=False, check_trace=False)
            torch.jit.save(traced, str(temporary_path))
        else:
            if type(opset_version) is not int or opset_version < 11:
                raise ValueError("opset_version must be an integer >= 11")
            with torch.inference_mode():
                torch.onnx.export(
                    wrapper,
                    batch,
                    str(temporary_path),
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=["images"],
                    output_names=["boxes", "scores", "labels"],
                    dynamo=False,
                )
        try:
            os.replace(temporary_path, destination)
        except OSError:
            # torch.jit.save cannot write non-ASCII (e.g. Chinese) paths on
            # Windows; stage in the ASCII %TEMP% directory then copy.
            shutil.copy2(temporary_path, destination)
            temporary_path.unlink(missing_ok=True)
    finally:
        temporary_path.unlink(missing_ok=True)
        if was_training:
            model.train()
    sidecar = destination.with_suffix(destination.suffix + ".json")
    metadata = {
        "format": EXPORT_FORMAT,
        "export_type": kind,
        "architecture": getattr(model, "_pcbparse_architecture", "unknown"),
        "input_shape": list(batch.shape),
        "input_contract": (
            "float32 RGB NCHW [0,1], batch=1, fixed exported height/width"
        ),
        "outputs": ["boxes_xyxy", "scores", "labels"],
        "label_contract": "background=0; model_label=1..N; official_id is metadata",
        "category_table": [
            dict(item)
            for item in normalize_category_table(
                getattr(model, "_pcbparse_category_table", ())
            )
        ] if getattr(model, "_pcbparse_category_table", None) else [],
        "artifact_sha256": _sha256_file(destination),
        "artifact_bytes": destination.stat().st_size,
        **({"opset_version": opset_version} if kind == "onnx" else {}),
    }
    _atomic_json(sidecar, metadata)
    return {**metadata, "path": str(destination), "sidecar": str(sidecar)}


def verify_export_consistency(
    model: Any,
    exported_path: str | Path,
    sample_image: Any,
    *,
    export_format: str | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> dict[str, Any]:
    """Compare native and exported boxes/scores/labels on the same tensor."""

    torch, _ = _require_torchvision()
    path = Path(exported_path).expanduser().resolve(strict=True)
    kind = _resolve_export_format(path, export_format)
    if rtol < 0 or atol < 0 or not math.isfinite(rtol) or not math.isfinite(atol):
        raise ValueError("rtol and atol must be finite non-negative numbers")
    batch = _normalize_export_input(sample_image, torch)
    device = _model_device(model, torch)
    native_batch = batch.to(device)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.inference_mode():
            native = model([native_batch[0]])[0]
            native_values = (
                native["boxes"].detach().cpu(),
                native["scores"].detach().cpu(),
                native["labels"].detach().cpu(),
            )
        if kind == "torchscript":
            # torch.jit.load cannot open non-ASCII (e.g. Chinese) paths on
            # Windows; stage a copy in the ASCII %TEMP% directory first.
            descriptor, staged_name = tempfile.mkstemp(
                prefix=".component-verify-", suffix=".pt", dir=tempfile.gettempdir()
            )
            os.close(descriptor)
            staged = Path(staged_name)
            staged.unlink()
            try:
                shutil.copy2(path, staged)
                scripted = torch.jit.load(str(staged), map_location=device).eval()
            finally:
                staged.unlink(missing_ok=True)
            with torch.inference_mode():
                exported = scripted(native_batch)
            exported_values = tuple(value.detach().cpu() for value in exported)
        else:
            try:
                ort = importlib.import_module("onnxruntime")
            except (ImportError, OSError) as exc:
                raise TorchComponentDependencyError(
                    "onnxruntime is required to verify an ONNX Component export"
                ) from exc
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            output = session.run(None, {session.get_inputs()[0].name: batch.cpu().numpy()})
            exported_values = tuple(torch.from_numpy(value) for value in output)
    finally:
        if was_training:
            model.train()
    if len(exported_values) != 3:
        raise TorchComponentArtifactError("exported detector did not return three outputs")
    names = ("boxes", "scores", "labels")
    differences: dict[str, float] = {}
    consistent = True
    for index, name in enumerate(names):
        native_value = native_values[index]
        exported_value = exported_values[index]
        if tuple(native_value.shape) != tuple(exported_value.shape):
            consistent = False
            differences[name] = math.inf
            continue
        if name == "labels":
            equal = bool(torch.equal(native_value.to(torch.int64), exported_value.to(torch.int64)))
            differences[name] = 0.0 if equal else math.inf
        else:
            if native_value.numel():
                differences[name] = float(
                    torch.max(torch.abs(native_value.float() - exported_value.float())).item()
                )
            else:
                differences[name] = 0.0
            equal = bool(
                torch.allclose(
                    native_value.float(), exported_value.float(), rtol=rtol, atol=atol
                )
            )
        consistent = consistent and equal
    return {
        "consistent": consistent,
        "export_type": kind,
        "rtol": rtol,
        "atol": atol,
        "max_absolute_difference": differences,
        "detections": int(native_values[0].shape[0]),
    }


def _prepare_detection_batch(batch: Any, torch: Any, device: Any) -> tuple[list[Any], list[dict[str, Any]]]:
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise ValueError("Component loader batches must be (images, targets)")
    raw_images, raw_targets = batch
    if torch.is_tensor(raw_images):
        if raw_images.ndim == 3:
            raw_images = [raw_images]
        elif raw_images.ndim == 4:
            raw_images = list(raw_images)
        else:
            raise ValueError("Component images tensor must be CHW or NCHW")
    elif isinstance(raw_images, Sequence):
        raw_images = list(raw_images)
    else:
        raise ValueError("Component images must be a tensor or sequence")
    if isinstance(raw_targets, Mapping):
        raw_targets = [raw_targets]
    elif isinstance(raw_targets, Sequence):
        raw_targets = list(raw_targets)
    else:
        raise ValueError("Component targets must be a mapping or sequence")
    if len(raw_images) != len(raw_targets) or not raw_images:
        raise ValueError("Component images/targets cardinality mismatch or empty batch")
    images: list[Any] = []
    targets: list[dict[str, Any]] = []
    for image, target in zip(raw_images, raw_targets):
        if not torch.is_tensor(image) or image.ndim != 3 or int(image.shape[0]) != 3:
            raise ValueError("each Component image must be a [3,H,W] tensor")
        if not image.is_floating_point():
            raise ValueError("Component images must be floating tensors in [0,1]")
        if image.numel() and (
            float(image.min().detach().cpu()) < 0.0
            or float(image.max().detach().cpu()) > 1.0
        ):
            raise ValueError("Component image pixels must be in [0,1]")
        if not isinstance(target, Mapping):
            raise ValueError("each Component target must be a mapping")
        moved = {
            str(key): value.to(device) if torch.is_tensor(value) else value
            for key, value in target.items()
        }
        _validate_target(moved, torch)
        images.append(image.to(device))
        targets.append(moved)
    return images, targets


def _validate_target(target: Mapping[str, Any], torch: Any) -> None:
    boxes = target.get("boxes")
    labels = target.get("labels")
    if not torch.is_tensor(boxes) or boxes.ndim != 2 or tuple(boxes.shape[1:]) != (4,):
        raise ValueError("Component target boxes must have shape [N,4]")
    if not torch.is_tensor(labels) or labels.ndim != 1 or len(labels) != len(boxes):
        raise ValueError("Component target labels must have shape [N]")
    if boxes.numel():
        if not bool(torch.all(torch.isfinite(boxes)).item()):
            raise ValueError("Component target boxes contain non-finite values")
        if not bool(torch.all(boxes[:, 2] > boxes[:, 0]).item()) or not bool(
            torch.all(boxes[:, 3] > boxes[:, 1]).item()
        ):
            raise ValueError("Component target boxes must have positive area")
    if labels.numel() and not bool(torch.all(labels > 0).item()):
        raise ValueError("Component target labels must be positive")


def _cpu_box_rows(value: Any, name: str) -> list[tuple[float, float, float, float]]:
    if value is None or getattr(value, "ndim", None) != 2 or tuple(value.shape[1:]) != (4,):
        raise ValueError(f"{name} must have shape [N,4]")
    rows: list[tuple[float, float, float, float]] = []
    for raw in value.detach().cpu().tolist():
        box = tuple(float(item) for item in raw)
        if any(not math.isfinite(item) for item in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"{name} contain an invalid box {box}")
        rows.append(box)
    return rows


def _cpu_int_rows(value: Any, name: str) -> list[int]:
    if value is None or getattr(value, "ndim", None) != 1:
        raise ValueError(f"{name} must have shape [N]")
    return [int(item) for item in value.detach().cpu().tolist()]


def _cpu_float_rows(value: Any, name: str) -> list[float]:
    if value is None or getattr(value, "ndim", None) != 1:
        raise ValueError(f"{name} must have shape [N]")
    rows = [float(item) for item in value.detach().cpu().tolist()]
    if any(not math.isfinite(item) for item in rows):
        raise ValueError(f"{name} contain non-finite values")
    return rows


def _interpolated_ap101(tp_flags: Sequence[int], fp_flags: Sequence[int], gt_count: int) -> float:
    if gt_count <= 0:
        raise ValueError("gt_count must be positive for AP")
    cumulative_tp = 0
    cumulative_fp = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for tp, fp in zip(tp_flags, fp_flags):
        cumulative_tp += int(tp)
        cumulative_fp += int(fp)
        recalls.append(cumulative_tp / gt_count)
        precisions.append(cumulative_tp / (cumulative_tp + cumulative_fp))
    total = 0.0
    for step in range(101):
        recall_threshold = step / 100.0
        eligible = [
            precision
            for recall, precision in zip(recalls, precisions)
            if recall >= recall_threshold
        ]
        total += max(eligible, default=0.0)
    return total / 101.0


def _box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    denominator = left_area + right_area - intersection
    return intersection / denominator if denominator > 0 else 0.0


def _extract_model_state(payload: Any, path: Path) -> Mapping[str, Any]:
    state: Any
    if isinstance(payload, Mapping) and "model_state" in payload:
        state = payload["model_state"]
    elif isinstance(payload, Mapping) and "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload
    if not isinstance(state, Mapping) or not state:
        raise TorchComponentArtifactError(f"{path} does not contain a model state mapping")
    if any(not isinstance(key, str) for key in state):
        raise TorchComponentArtifactError(f"{path} model state has a non-string key")
    return state


def _model_num_classes(model: Any) -> int:
    try:
        value = int(model.roi_heads.box_predictor.cls_score.out_features)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TorchComponentArtifactError("cannot determine detector head class count") from exc
    return value


def _safe_torch_load(torch: Any, path: Path, *, map_location: str | Any) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        raise TorchComponentDependencyError(
            "this PyTorch version lacks safe weights_only checkpoint loading"
        ) from exc
    except Exception as exc:
        raise TorchComponentArtifactError(
            f"cannot safely load Component state {path}: {type(exc).__name__}: {exc}"
        ) from exc


def _cpu_tree(value: Any, torch: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        output: dict[Any, Any] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise TorchComponentArtifactError(
                    "optimizer state mapping keys must be strings or integers"
                )
            output[key] = _cpu_tree(item, torch)
        return output
    if isinstance(value, list):
        return [_cpu_tree(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item, torch) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TorchComponentArtifactError(
        f"optimizer state contains unsupported value {type(value).__name__}"
    )


def _validate_preprocessing(value: Mapping[str, Any]) -> None:
    expected = {
        "color_space": "RGB",
        "tensor_layout": "CHW",
        "pixel_dtype": "float32",
        "resize_owner": "torchvision_generalized_rcnn_transform",
        "coordinate_output": "input_view_xyxy",
    }
    for key, required in expected.items():
        if value.get(key) != required:
            raise TorchComponentArtifactError(
                f"unsupported preprocessing {key}={value.get(key)!r}; required {required!r}"
            )
    if value.get("pixel_range") != [0.0, 1.0]:
        raise TorchComponentArtifactError("preprocessing pixel_range must be [0.0, 1.0]")


def _validate_preprocessing_against_model(value: Mapping[str, Any], model: Any) -> None:
    actual = default_preprocessing(model)
    for key, actual_value in actual.items():
        if value.get(key) != actual_value:
            raise TorchComponentArtifactError(
                f"checkpoint preprocessing {key}={value.get(key)!r} differs from "
                f"the reconstructed model value {actual_value!r}"
            )


def _require_torchvision() -> tuple[Any, Any]:
    try:
        torch = importlib.import_module("torch")
        torchvision = importlib.import_module("torchvision")
    except (ImportError, OSError) as exc:
        raise TorchComponentDependencyError(
            "PyTorch and TorchVision are required for the Component detector. "
            "Use the project's .venv-train environment; core PCBParse remains torch-optional."
        ) from exc
    return torch, torchvision


def _normalize_export_input(value: Any, torch: Any) -> Any:
    if not torch.is_tensor(value):
        raise ValueError("sample_image must be a torch tensor")
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4 or int(value.shape[0]) != 1 or int(value.shape[1]) != 3:
        raise ValueError("sample_image must have shape [3,H,W] or [1,3,H,W]")
    if not value.is_floating_point():
        raise ValueError("sample_image must be floating RGB in [0,1]")
    if value.numel() and (float(value.min()) < 0.0 or float(value.max()) > 1.0):
        raise ValueError("sample_image pixels must be in [0,1]")
    return value.contiguous()


def _model_device(model: Any, torch: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _resolve_export_format(path: Path, explicit: str | None) -> str:
    if explicit is None:
        suffix = path.suffix.lower()
        if suffix == ".onnx":
            return "onnx"
        if suffix in {".pt", ".pth", ".ts", ".torchscript"}:
            return "torchscript"
        raise ValueError("cannot infer export format; use .onnx/.pt or pass export_format")
    normalized = explicit.strip().lower().replace("_", "")
    aliases = {"onnx": "onnx", "torchscript": "torchscript", "jit": "torchscript"}
    if normalized not in aliases:
        raise ValueError("export_format must be 'onnx' or 'torchscript'")
    return aliases[normalized]


def _json_mapping(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    clean: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field} keys must be non-empty strings")
        clean[key] = _json_value(item, f"{field}.{key}")
    return clean


def _json_value(value: Any, field: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return _json_mapping(value, field)
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{field}[]") for item in value]
    raise ValueError(f"{field} contains non-JSON value {type(value).__name__}")


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TorchComponentArtifactError(f"checkpoint {field} must be a mapping")
    return value


def _require_sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TorchComponentArtifactError(f"checkpoint {field} must be an array")
    return value


def _unit_interval(value: Any, name: str, *, inclusive_zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    minimum_ok = number >= 0.0 if inclusive_zero else number > 0.0
    if not minimum_ok or number > 1.0:
        bracket = "[0,1]" if inclusive_zero else "(0,1]"
        raise ValueError(f"{name} must be in {bracket}")
    return number


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdefABCDEF" for char in value
    )


__all__ = [
    "CHECKPOINT_FORMAT",
    "DEFAULT_ARCHITECTURE",
    "EXPORT_FORMAT",
    "LoadedComponentCheckpoint",
    "TorchComponentArtifactError",
    "TorchComponentDependencyError",
    "build_model",
    "category_table_from_dataset",
    "default_preprocessing",
    "evaluate_map50",
    "evaluate_one_epoch",
    "export_model",
    "load_checkpoint",
    "normalize_category_table",
    "save_checkpoint",
    "train_one_epoch",
    "verify_export_consistency",
]
