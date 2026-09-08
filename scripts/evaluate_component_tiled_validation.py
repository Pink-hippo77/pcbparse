from __future__ import annotations

"""Run a fixed, production-style Component replay on frozen validation images.

This evaluator is intentionally diagnostic-only.  It binds a completed formal
Component checkpoint to the immutable Dataset300-v2 materialization, then runs
the production ``ImageView`` tiler, torch adapter, coordinate restoration, and
class-aware cross-tile NMS.  The protocol is fixed to configuration files that
predate this evaluator: 512 px tiles with 64 px overlap, score threshold 0.20,
and NMS IoU 0.55.  No threshold or geometry search is implemented.

The script never registers, blesses, or edits a model lock.  It accepts only a
formal parent run, scores only the frozen validation split, and records zero
train/test usage.  ``--max-val-samples`` exists solely for deterministic CPU
smoke tests and makes the result explicitly incomplete.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pcbparse.io import load_image, tile_image_view  # noqa: E402
from pcbparse.perception.composite_provider import (  # noqa: E402
    _class_aware_nms,
    _restore_components,
)
from pcbparse.perception.torch_component_adapter import (  # noqa: E402
    TorchComponentAdapter,
)
from pcbparse.training.config import (  # noqa: E402
    load_training_config,
    training_config_resume_sha256s,
    training_config_sha256,
)
from pcbparse.training.torch_component_model import load_checkpoint  # noqa: E402


FINAL_FORMAL_STATUSES = frozenset({"candidate_evidence_ready", "gate_failed"})
EXPECTED_TILE_SIZE = 512
EXPECTED_TILE_OVERLAP = 64
EXPECTED_SCORE_THRESHOLD = 0.20
EXPECTED_NMS_IOU = 0.55
MATCH_IOU = 0.50
MAX_DETECTIONS_PER_TILE = 300


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only, hash-bound, diagnostic Component replay using fixed "
            "production-style 512/64 tiling and class-aware NMS 0.55."
        )
    )
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/train/dataset300_v2_component_v4.json"),
    )
    parser.add_argument(
        "--parent-status",
        type=Path,
        default=Path("work/train-component-v4/torch_run_status.json"),
    )
    parser.add_argument(
        "--component-coco",
        type=Path,
        default=Path(
            "artifacts/training/materialized/dataset300-v2/"
            "component-pin/component_coco.json"
        ),
    )
    parser.add_argument(
        "--materialization-summary",
        type=Path,
        default=Path(
            "artifacts/training/materialized/dataset300-v2/"
            "component-pin/summary.json"
        ),
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=Path("configs/pipeline.component512-experiment.yaml"),
    )
    parser.add_argument(
        "--thresholds-config",
        type=Path,
        default=Path("configs/thresholds.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work/eval-component-v4-tiled-cpu"),
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help=(
            "Lexicographic validation prefix for CPU smoke only; omitted means "
            "all frozen validation samples."
        ),
    )
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {label} JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an array")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _positive_integer(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_within(
    root: Path,
    raw: Path | str,
    label: str,
    *,
    kind: str = "file",
    must_exist: bool = True,
) -> Path:
    path = Path(raw).expanduser()
    resolved = (path if path.is_absolute() else root / path).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside workspace: {resolved}") from exc
    if must_exist:
        if kind == "file" and (not resolved.is_file() or resolved.stat().st_size <= 0):
            raise FileNotFoundError(f"{label} is missing or empty: {resolved}")
        if kind == "directory" and not resolved.is_dir():
            raise FileNotFoundError(f"{label} directory is missing: {resolved}")
    return resolved


def _relative(path: Path, workspace: Path) -> str:
    return path.relative_to(workspace).as_posix()


def _fixed_protocol(
    pipeline: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    materialization_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract and fail closed on the single predeclared evaluation protocol."""

    image = _mapping(pipeline.get("image"), "pipeline.image")
    component = _mapping(thresholds.get("component"), "thresholds.component")
    material_config = _mapping(
        materialization_summary.get("config"), "materialization_summary.config"
    )
    tile_size = _positive_integer(
        image.get("component_tile_size"), "pipeline.image.component_tile_size"
    )
    tile_overlap = image.get("component_tile_overlap")
    if type(tile_overlap) is not int or tile_overlap < 0:
        raise ValueError("pipeline.image.component_tile_overlap must be non-negative")
    score_threshold = _finite(
        component.get("min_score"), "thresholds.component.min_score"
    )
    nms_iou = _finite(component.get("nms_iou"), "thresholds.component.nms_iou")
    material_tile_size = _positive_integer(
        material_config.get("tile_size"), "materialization config.tile_size"
    )
    material_tile_overlap = material_config.get("tile_overlap")
    if type(material_tile_overlap) is not int or material_tile_overlap < 0:
        raise ValueError("materialization config.tile_overlap must be non-negative")
    observed = (
        tile_size,
        tile_overlap,
        score_threshold,
        nms_iou,
        material_tile_size,
        material_tile_overlap,
    )
    expected = (
        EXPECTED_TILE_SIZE,
        EXPECTED_TILE_OVERLAP,
        EXPECTED_SCORE_THRESHOLD,
        EXPECTED_NMS_IOU,
        EXPECTED_TILE_SIZE,
        EXPECTED_TILE_OVERLAP,
    )
    if observed != expected:
        raise ValueError(
            "fixed tiled protocol drifted; expected pipeline/materialization "
            "512/64, score threshold 0.20, and class-aware NMS 0.55; "
            f"observed={observed!r}"
        )
    return {
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "tile_stride": tile_size - tile_overlap,
        "score_threshold": score_threshold,
        "component_nms_iou": nms_iou,
        "match_iou": MATCH_IOU,
        "max_detections_per_tile": MAX_DETECTIONS_PER_TILE,
        "thresholds_searched": False,
        "tiling_configs_searched": False,
        "nms_configs_searched": False,
    }


def _parent_contract(
    *,
    workspace: Path,
    config_path: Path,
    parent_status_path: Path,
    component_coco_path: Path,
    materialization_summary_path: Path,
) -> dict[str, Any]:
    config = load_training_config(config_path)
    task_config = config.tasks["component"]
    current_config_sha256 = training_config_sha256(config)
    compatible_config_sha256s = training_config_resume_sha256s(config)
    parent = _read_object(parent_status_path, "parent Component status")
    if parent.get("task") != "component":
        raise ValueError("parent status task is not component")
    if parent.get("status") not in FINAL_FORMAL_STATUSES:
        raise ValueError("parent Component run is not in a completed formal state")
    if parent.get("formal_evidence") is not True or parent.get("smoke") is not False:
        raise ValueError("parent Component run must be formal and non-smoke")
    if not parent.get("finished_at_utc"):
        raise ValueError("parent Component run lacks a completion timestamp")
    if task_config.component_box_detections_per_img != MAX_DETECTIONS_PER_TILE:
        raise ValueError(
            "training config component_box_detections_per_img differs from fixed "
            f"production contract {MAX_DETECTIONS_PER_TILE}"
        )

    lineage = _mapping(parent.get("lineage"), "parent.lineage")
    parent_config_sha256 = _nonempty_string(
        lineage.get("training_config_sha256"),
        "parent.lineage.training_config_sha256",
    )
    if parent_config_sha256 not in compatible_config_sha256s:
        raise ValueError(
            "parent training config hash is not in the explicit no-op schema history"
        )
    summary = _read_object(materialization_summary_path, "materialization summary")
    binding = _nonempty_string(
        summary.get("binding_sha256"), "materialization binding_sha256"
    )
    if lineage.get("binding_sha256") != binding:
        raise ValueError("parent/materialization binding mismatch")
    summary_sha256 = _sha256(materialization_summary_path)
    if lineage.get("materialization_commit_sha256") != summary_sha256:
        raise ValueError("parent materialization summary SHA-256 mismatch")
    outputs = _mapping(summary.get("outputs"), "materialization outputs")
    coco_output = _mapping(outputs.get("component_coco.json"), "Component COCO output")
    coco_sha256 = _sha256(component_coco_path)
    if coco_output.get("sha256") != coco_sha256:
        raise ValueError("Component COCO SHA-256 differs from materialization commit")

    artifact = _mapping(parent.get("artifact"), "parent.artifact")
    checkpoint_path = _resolve_within(
        workspace,
        _nonempty_string(artifact.get("path"), "parent.artifact.path"),
        "parent checkpoint",
    )
    checkpoint_sha256 = _sha256(checkpoint_path)
    if artifact.get("sha256") != checkpoint_sha256:
        raise ValueError("parent Component checkpoint SHA-256 mismatch")

    predictions = _mapping(parent.get("predictions"), "parent.predictions")
    parent_predictions_path = _resolve_within(
        workspace,
        _nonempty_string(predictions.get("path"), "parent.predictions.path"),
        "parent predictions",
    )
    parent_predictions_sha256 = _sha256(parent_predictions_path)
    if predictions.get("sha256") != parent_predictions_sha256:
        raise ValueError("parent prediction evidence SHA-256 mismatch")
    source_root = _resolve_within(
        workspace,
        config.dataset.source_root,
        "Dataset300 source root",
        kind="directory",
    )
    return {
        "config": config,
        "task_config": task_config,
        "parent": parent,
        "lineage": dict(lineage),
        "binding_sha256": binding,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "source_root": source_root,
        "evidence": {
            "parent_status": {
                "path": _relative(parent_status_path, workspace),
                "sha256": _sha256(parent_status_path),
                "status": parent["status"],
            },
            "checkpoint": {
                "path": _relative(checkpoint_path, workspace),
                "sha256": checkpoint_sha256,
            },
            "parent_predictions": {
                "path": _relative(parent_predictions_path, workspace),
                "sha256": parent_predictions_sha256,
                "used_for_tiled_metrics": False,
            },
            "training_config": {
                "path": _relative(config_path, workspace),
                "sha256": _sha256(config_path),
                "current_semantic_sha256": current_config_sha256,
                "parent_semantic_sha256": parent_config_sha256,
                "compatible_semantic_sha256s": list(compatible_config_sha256s),
                "declared_component_score_threshold": float(
                    task_config.component_score_threshold
                ),
                "declared_max_detections_per_image": int(
                    task_config.component_box_detections_per_img
                ),
            },
            "component_coco": {
                "path": _relative(component_coco_path, workspace),
                "sha256": coco_sha256,
            },
            "materialization_summary": {
                "path": _relative(materialization_summary_path, workspace),
                "sha256": summary_sha256,
                "binding_sha256": binding,
            },
        },
        "materialization_summary": summary,
    }


def _bbox(value: Any, label: str) -> tuple[float, float, float, float]:
    rows = _sequence(value, label)
    if len(rows) != 4:
        raise ValueError(f"{label} must contain four coordinates")
    x1, y1, x2, y2 = (_finite(item, label) for item in rows)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{label} must have positive area")
    return x1, y1, x2, y2


def _frozen_validation_samples(
    *,
    component_coco: Mapping[str, Any],
    source_root: Path,
    binding_sha256: str,
    max_val_samples: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if max_val_samples is not None and max_val_samples <= 0:
        raise ValueError("max-val-samples must be positive when supplied")
    info = _mapping(component_coco.get("info"), "Component COCO info")
    if info.get("binding_sha256") != binding_sha256:
        raise ValueError("Component COCO/materialization binding mismatch")
    raw_categories = _sequence(component_coco.get("categories"), "COCO categories")
    category_by_id: dict[int, dict[str, Any]] = {}
    categories: list[dict[str, Any]] = []
    for raw in raw_categories:
        category = _mapping(raw, "COCO category")
        category_id = _positive_integer(category.get("id"), "COCO category id")
        if category_id in category_by_id:
            raise ValueError(f"duplicate COCO category id {category_id}")
        name = _nonempty_string(category.get("name"), "COCO category name")
        official_id = int(
            _nonempty_string(
                str(category.get("source_category_id", "")),
                "COCO category source_category_id",
            )
        )
        row = {"index_id": category_id, "official_id": official_id, "name": name}
        category_by_id[category_id] = row
        categories.append(row)
    categories.sort(key=lambda item: (item["official_id"], item["name"]))

    raw_images = _sequence(component_coco.get("images"), "COCO images")
    full_val: list[dict[str, Any]] = []
    split_view_counts: dict[str, int] = defaultdict(int)
    for raw in raw_images:
        image = _mapping(raw, "COCO image")
        split = _nonempty_string(image.get("split"), "COCO image split")
        view_kind = _nonempty_string(image.get("view_kind"), "COCO image view_kind")
        split_view_counts[f"{split}:{view_kind}"] += 1
        if split != "val" or view_kind != "full":
            continue
        if image.get("binding_sha256") != binding_sha256:
            raise ValueError("validation image binding differs from materialization")
        full_val.append(dict(image))
    full_val.sort(key=lambda item: str(item.get("sample_id")))
    sample_ids = [
        _nonempty_string(item.get("sample_id"), "validation sample_id")
        for item in full_val
    ]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("validation materialization has duplicate full views")
    if not full_val:
        raise ValueError("frozen validation split has no full views")
    selected = full_val[:max_val_samples] if max_val_samples is not None else full_val
    selected_ids = {int(item["id"]) for item in selected}
    selected_by_image_id = {int(item["id"]): item for item in selected}
    for image in selected:
        source_size = _sequence(image.get("source_size"), "validation source_size")
        if len(source_size) != 2:
            raise ValueError("validation source_size must contain width and height")
        width = _positive_integer(source_size[0], "validation source width")
        height = _positive_integer(source_size[1], "validation source height")
        source_image = _nonempty_string(
            image.get("source_image"), "validation source_image"
        )
        image_path = (source_root / source_image).resolve(strict=False)
        try:
            image_path.relative_to(source_root)
        except ValueError as exc:
            raise ValueError("validation source image escapes Dataset300 root") from exc
        if not image_path.is_file() or image_path.stat().st_size <= 0:
            raise FileNotFoundError(f"validation source image is missing: {image_path}")
        expected_sha256 = _nonempty_string(
            image.get("source_image_sha256"), "validation source image SHA-256"
        )
        if _sha256(image_path) != expected_sha256:
            raise ValueError(f"validation source image SHA-256 mismatch: {source_image}")
        image["_resolved_path"] = image_path
        image["_source_width"] = width
        image["_source_height"] = height
        image["_ground_truth"] = []

    # Only annotations owned by selected validation full views are interpreted.
    # Test/train geometries are never extracted into the metric population.
    annotation_ids: set[int] = set()
    for raw in _sequence(component_coco.get("annotations"), "COCO annotations"):
        annotation = _mapping(raw, "COCO annotation")
        image_id = annotation.get("image_id")
        if image_id not in selected_ids:
            continue
        annotation_id = _positive_integer(annotation.get("id"), "COCO annotation id")
        if annotation_id in annotation_ids:
            raise ValueError(f"duplicate selected annotation id {annotation_id}")
        annotation_ids.add(annotation_id)
        if annotation.get("ignore") not in (0, False, None):
            raise ValueError("full-view validation ground truth unexpectedly ignored")
        category_id = _positive_integer(
            annotation.get("category_id"), "COCO annotation category_id"
        )
        category = category_by_id.get(category_id)
        if category is None:
            raise ValueError("validation annotation references an unknown category")
        box = _bbox(annotation.get("source_bbox"), "validation source_bbox")
        image = selected_by_image_id[int(image_id)]
        width = int(image["_source_width"])
        height = int(image["_source_height"])
        if not (0.0 <= box[0] < box[2] <= width and 0.0 <= box[1] < box[3] <= height):
            raise ValueError("validation source_bbox lies outside source image")
        source_category = annotation.get("source_category")
        if source_category is not None and source_category != category["name"]:
            raise ValueError("validation annotation category name mapping changed")
        image["_ground_truth"].append(
            {
                "annotation_id": annotation_id,
                "category": category["name"],
                "official_id": category["official_id"],
                "box": list(box),
            }
        )
    for image in selected:
        image["_ground_truth"].sort(key=lambda item: item["annotation_id"])
    boundary = {
        "split": "val",
        "available_val_samples": len(full_val),
        "selected_val_samples": len(selected),
        "selection_policy": (
            "all frozen validation samples"
            if max_val_samples is None
            else "lexicographic validation prefix for CPU smoke only"
        ),
        "selected_sample_ids": [str(item["sample_id"]) for item in selected],
        "selected_sample_ids_sha256": _canonical_sha256(
            [str(item["sample_id"]) for item in selected]
        ),
        "selected_validation_annotations": sum(
            len(item["_ground_truth"]) for item in selected
        ),
        "train_samples_loaded": 0,
        "train_images_decoded": 0,
        "train_annotations_scored": 0,
        "test_samples_loaded": 0,
        "test_images_decoded": 0,
        "test_annotations_scored": 0,
        "index_split_view_counts_metadata_only": dict(sorted(split_view_counts.items())),
    }
    return selected, categories, boundary


def _box_iou(
    left: Sequence[float], right: Sequence[float]
) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0.0:
        return 0.0
    left_area = (float(left[2]) - float(left[0])) * (
        float(left[3]) - float(left[1])
    )
    right_area = (float(right[2]) - float(right[0])) * (
        float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _interpolated_ap101(
    tp_flags: Sequence[int], fp_flags: Sequence[int], support: int
) -> float | None:
    if support <= 0:
        return None
    cumulative_tp: list[int] = []
    cumulative_fp: list[int] = []
    running_tp = running_fp = 0
    for tp, fp in zip(tp_flags, fp_flags):
        running_tp += int(tp)
        running_fp += int(fp)
        cumulative_tp.append(running_tp)
        cumulative_fp.append(running_fp)
    precisions = [
        tp / (tp + fp) if tp + fp else 0.0
        for tp, fp in zip(cumulative_tp, cumulative_fp)
    ]
    recalls = [tp / support for tp in cumulative_tp]
    return sum(
        max(
            (precision for precision, recall in zip(precisions, recalls) if recall >= level),
            default=0.0,
        )
        for level in (index / 100.0 for index in range(101))
    ) / 101.0


def _evaluate_predictions(
    prediction_rows: Sequence[Mapping[str, Any]],
    ground_truth_by_sample: Mapping[str, Sequence[Mapping[str, Any]]],
    categories: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = MATCH_IOU,
) -> dict[str, Any]:
    """Use the formal detector's global confidence-ranked IoU@0.5 contract."""

    gt_by_class: dict[str, dict[str, list[Sequence[float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    predictions_by_class: dict[
        str, list[tuple[float, str, int, Sequence[float]]]
    ] = defaultdict(list)
    for sample_id, rows in ground_truth_by_sample.items():
        for row in rows:
            gt_by_class[str(row["category"])][sample_id].append(row["box"])
    for row in prediction_rows:
        sample_id = str(row["sample_id"])
        for prediction_index, prediction in enumerate(
            _sequence(row.get("predictions"), "sample predictions")
        ):
            item = _mapping(prediction, "sample prediction")
            predictions_by_class[str(item["category"])].append(
                (
                    float(item["score"]),
                    sample_id,
                    prediction_index,
                    item["box"],
                )
            )
    category_names = [str(item["name"]) for item in categories]
    unexpected = sorted(set(predictions_by_class).difference(category_names))
    if unexpected:
        raise ValueError(f"predictions contain out-of-contract categories: {unexpected}")
    total_tp = total_fp = total_fn = 0
    evaluated_aps: list[float] = []
    per_class: dict[str, Any] = {}
    for category in categories:
        name = str(category["name"])
        class_gt = gt_by_class.get(name, {})
        support = sum(len(items) for items in class_gt.values())
        matched = {
            sample_id: [False] * len(items) for sample_id, items in class_gt.items()
        }
        ranked = sorted(
            predictions_by_class.get(name, []),
            key=lambda item: (-item[0], item[1], item[2]),
        )
        tp_flags: list[int] = []
        fp_flags: list[int] = []
        for _score, sample_id, _index, predicted_box in ranked:
            candidates = class_gt.get(sample_id, [])
            best_index = -1
            best_iou = -1.0
            for gt_index, gt_box in enumerate(candidates):
                if matched[sample_id][gt_index]:
                    continue
                overlap = _box_iou(predicted_box, gt_box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = gt_index
            if best_index >= 0 and best_iou >= iou_threshold:
                matched[sample_id][best_index] = True
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)
        tp = sum(tp_flags)
        fp = sum(fp_flags)
        fn = support - tp
        ap50 = _interpolated_ap101(tp_flags, fp_flags, support)
        if ap50 is not None:
            evaluated_aps.append(ap50)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[name] = {
            "official_category_id": int(category["official_id"]),
            "ground_truth": support,
            "predictions": len(ranked),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision50": precision,
            "recall50": recall,
            "f1_50": f1,
            "ap50": ap50,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    prediction_count = total_tp + total_fp
    gt_count = total_tp + total_fn
    return {
        "metric_contract": "global-score-ranked class-aware one-to-one IoU@0.5",
        "iou_threshold": iou_threshold,
        "bbox_map50": (
            sum(evaluated_aps) / len(evaluated_aps) if evaluated_aps else None
        ),
        "precision50": precision,
        "recall50": recall,
        "f1_50": f1,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "prediction_count": prediction_count,
        "ground_truth_count": gt_count,
        "pred_to_gt_ratio": prediction_count / gt_count if gt_count else None,
        "evaluated_class_count": len(evaluated_aps),
        "classes_without_ground_truth": sorted(
            name for name, row in per_class.items() if row["ground_truth"] == 0
        ),
        "per_class": dict(sorted(per_class.items())),
    }


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _prediction_line(row: Mapping[str, Any]) -> str:
    return json.dumps(
        row,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def _prepare_output(path: Path, workspace: Path) -> None:
    try:
        path.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("output directory must stay inside workspace") from exc
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _run_inference(
    *,
    samples: Sequence[dict[str, Any]],
    adapter: TorchComponentAdapter,
    protocol: Mapping[str, Any],
    predictions_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    elapsed_by_sample: dict[str, float] = {}
    local_candidates = 0
    final_predictions = 0
    total_tiles = 0
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{predictions_path.name}.", suffix=".tmp", dir=predictions_path.parent
    )
    started_total = time.perf_counter()
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for sample in samples:
                sample_id = str(sample["sample_id"])
                started = time.perf_counter()
                original = load_image(sample["_resolved_path"])
                if (original.width, original.height) != (
                    int(sample["_source_width"]),
                    int(sample["_source_height"]),
                ):
                    raise ValueError(f"decoded validation size changed for {sample_id}")
                views = tile_image_view(
                    original,
                    tile_size=int(protocol["tile_size"]),
                    overlap=int(protocol["tile_overlap"]),
                )
                candidates = []
                for view in views:
                    candidates.extend(_restore_components(adapter.predict(view, None), view))
                merged = _class_aware_nms(
                    candidates, float(protocol["component_nms_iou"])
                )
                predictions = [
                    {
                        "box": [
                            float(item.bbox.x1),
                            float(item.bbox.y1),
                            float(item.bbox.x2),
                            float(item.bbox.y2),
                        ],
                        "category": str(item.category),
                        "score": float(item.score),
                        "source_view_id": str(item.evidence[0].view_id),
                    }
                    for item in merged
                ]
                row = {
                    "sample_id": sample_id,
                    "source_image_sha256": str(sample["source_image_sha256"]),
                    "shape": [original.height, original.width],
                    "tile_count": len(views),
                    "candidate_count_before_cross_tile_nms": len(candidates),
                    "predictions": predictions,
                }
                handle.write(_prediction_line(row))
                rows.append(row)
                total_tiles += len(views)
                local_candidates += len(candidates)
                final_predictions += len(predictions)
                elapsed_by_sample[sample_id] = time.perf_counter() - started
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, predictions_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return rows, {
        "elapsed_seconds": time.perf_counter() - started_total,
        "elapsed_by_sample_seconds": elapsed_by_sample,
        "sample_count": len(samples),
        "tile_count": total_tiles,
        "candidate_count_before_cross_tile_nms": local_candidates,
        "prediction_count_after_cross_tile_nms": final_predictions,
    }


def main(argv: Iterable[str] | None = None) -> int:
    # This standalone evaluator must stay off the GPU even when another
    # training process is using CUDA.  Keep the assignment inside the entry
    # point so importing pure helpers in a shared unittest process has no
    # environment side effect.  All torch-backed calls remain below it.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    arguments = _parser().parse_args(argv)
    workspace = Path(arguments.workspace_root).expanduser().resolve(strict=True)
    config_path = _resolve_within(
        workspace, arguments.training_config, "training config"
    )
    parent_status_path = _resolve_within(
        workspace, arguments.parent_status, "parent Component status"
    )
    component_coco_path = _resolve_within(
        workspace, arguments.component_coco, "Component COCO"
    )
    materialization_summary_path = _resolve_within(
        workspace, arguments.materialization_summary, "materialization summary"
    )
    pipeline_config_path = _resolve_within(
        workspace, arguments.pipeline_config, "pipeline config"
    )
    thresholds_config_path = _resolve_within(
        workspace, arguments.thresholds_config, "thresholds config"
    )
    output_dir = _resolve_within(
        workspace,
        arguments.output_dir,
        "output directory",
        kind="directory",
        must_exist=False,
    )
    cpu_threads = _positive_integer(arguments.cpu_threads, "cpu-threads")
    _prepare_output(output_dir, workspace)

    contract = _parent_contract(
        workspace=workspace,
        config_path=config_path,
        parent_status_path=parent_status_path,
        component_coco_path=component_coco_path,
        materialization_summary_path=materialization_summary_path,
    )
    pipeline = _read_object(pipeline_config_path, "pipeline config")
    thresholds = _read_object(thresholds_config_path, "thresholds config")
    protocol = _fixed_protocol(
        pipeline, thresholds, contract["materialization_summary"]
    )
    component_coco = _read_object(component_coco_path, "Component COCO")
    samples, categories, selection_boundary = _frozen_validation_samples(
        component_coco=component_coco,
        source_root=contract["source_root"],
        binding_sha256=contract["binding_sha256"],
        max_val_samples=arguments.max_val_samples,
    )

    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError("PyTorch is required for tiled Component inference") from exc
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(20260812)
    torch.use_deterministic_algorithms(True)
    loaded = load_checkpoint(
        contract["checkpoint_path"],
        device="cpu",
        expected_sha256=contract["checkpoint_sha256"],
        strict=True,
    )
    if loaded.lineage.get("binding_sha256") != contract["binding_sha256"]:
        raise ValueError("checkpoint/materialization binding mismatch")
    model_categories = sorted(
        (
            int(item["official_id"]),
            str(item["name"]),
        )
        for item in loaded.category_table
    )
    index_categories = sorted(
        (int(item["official_id"]), str(item["name"])) for item in categories
    )
    if model_categories != index_categories:
        raise ValueError("checkpoint and materialized Component category tables differ")
    adapter = TorchComponentAdapter(
        loaded,
        device="cpu",
        score_threshold=float(protocol["score_threshold"]),
        max_detections=int(protocol["max_detections_per_tile"]),
    )
    predictions_path = output_dir / "predictions.jsonl"
    prediction_rows, performance = _run_inference(
        samples=samples,
        adapter=adapter,
        protocol=protocol,
        predictions_path=predictions_path,
    )
    ground_truth_by_sample = {
        str(item["sample_id"]): list(item["_ground_truth"]) for item in samples
    }
    metrics = _evaluate_predictions(
        prediction_rows,
        ground_truth_by_sample,
        categories,
        iou_threshold=float(protocol["match_iou"]),
    )
    complete_validation = len(samples) == selection_boundary["available_val_samples"]
    finished_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    status: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pcbparse_component_tiled_frozen_val_diagnostic",
        "task": "component",
        "status": (
            "diagnostic_evaluation_complete"
            if complete_validation
            else "diagnostic_smoke_complete"
        ),
        "registerable": False,
        "formal_evidence": False,
        "promotion_eligible": False,
        "blessing_authorized": False,
        "smoke": not complete_validation,
        "finished_at_utc": finished_at,
        "device": "cpu",
        "cpu_threads": torch.get_num_threads(),
        "cuda_visible_devices_forced": "",
        "deterministic_algorithms": True,
        "protocol": protocol,
        "selection_boundary": selection_boundary,
        "metrics": metrics,
        "performance": performance,
        "artifacts": {
            "predictions": {
                "path": _relative(predictions_path, workspace),
                "sha256": _sha256(predictions_path),
                "row_count": len(prediction_rows),
            }
        },
        "lineage": {
            **contract["evidence"],
            "pipeline_config": {
                "path": _relative(pipeline_config_path, workspace),
                "sha256": _sha256(pipeline_config_path),
            },
            "thresholds_config": {
                "path": _relative(thresholds_config_path, workspace),
                "sha256": _sha256(thresholds_config_path),
            },
        },
        "interpretation_boundary": {
            "training_gate_reused": False,
            "parent_gate_status_changed": False,
            "models_lock_read": False,
            "models_lock_modified": False,
            "validation_configuration_tuned": False,
            "test_split_used": False,
            "full_validation_conclusion_allowed": complete_validation,
        },
        "limitations": [
            "This replay is diagnostic-only and cannot register, bless, or promote a model.",
            "The fixed score threshold 0.20 and NMS 0.55 are read from a pre-existing thresholds config; no validation search is implemented.",
            "The fixed 512/64 geometry is shared by the pre-existing pipeline experiment and training materialization.",
            "Only frozen validation images and annotations are scored; train and test usage is zero.",
            *(
                []
                if complete_validation
                else [
                    "This CPU smoke covers only a lexicographic validation prefix and cannot estimate full-validation performance."
                ]
            ),
        ],
    }
    status["diagnostic_payload_sha256"] = _canonical_sha256(status)
    status_path = output_dir / "diagnostic_status.json"
    _atomic_write_json(status_path, status)
    print(status_path)
    print(json.dumps(metrics, sort_keys=True, separators=(",", ":")))
    print(f"diagnostic_payload_sha256={status['diagnostic_payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
