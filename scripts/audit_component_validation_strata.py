from __future__ import annotations

"""Build a hash-bound, frozen-validation Component stratification audit.

The audit consumes predictions already emitted by a completed formal run or by
a diagnostic replay whose parent formal status is independently hash-verified.
It does not run inference, search thresholds, or read test records into any
metric population.  Class and source/domain metrics use the same class-aware,
one-to-one, globally score-ranked IoU@0.5 matching as formal training.  Size
metrics use non-overlapping area bins (materialized full-view coordinates for
formal predictions, original source coordinates for stitched predictions) and
suppress matches to out-of-bin truth, analogous to an area-range detector
evaluation.
"""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FINAL_FORMAL_STATUSES = frozenset({"candidate_evidence_ready", "gate_failed"})
DIAGNOSTIC_STATUS = "diagnostic_evaluation_complete"
TILED_DIAGNOSTIC_KIND = "pcbparse_component_tiled_frozen_val_diagnostic"
SIZE_BINS: tuple[tuple[str, float, float | None], ...] = (
    ("small", 0.0, 32.0**2),
    ("medium", 32.0**2, 96.0**2),
    ("large", 96.0**2, None),
)


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Audit completed Component predictions on the frozen validation "
            "split by class, materialized-view size, and source/domain."
        )
    )
    parser.add_argument("--workspace-root", type=Path, default=root)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("work/train-component-v3/predictions.jsonl"),
    )
    parser.add_argument(
        "--run-status",
        type=Path,
        default=Path("work/train-component-v3/torch_run_status.json"),
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
        "--component-manifest",
        type=Path,
        default=Path("data/derived/dataset300-v2/component_detection.jsonl"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/manifests/dataset300.jsonl"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/splits/dataset300_split.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/component_v3_val_stratified.json"),
    )
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser


def _resolve_within(
    root: Path, raw: Path, label: str, *, must_exist: bool
) -> Path:
    path = raw.expanduser()
    resolved = (path if path.is_absolute() else root / path).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside workspace: {resolved}") from exc
    if must_exist and (not resolved.is_file() or resolved.stat().st_size <= 0):
        raise FileNotFoundError(f"{label} is missing or empty: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
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
            raise ValueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {label} JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(
                    raw,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"invalid {label} JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"{label} JSONL row is not an object at {path}:{line_number}"
                )
            output.append(value)
    if not output:
        raise ValueError(f"{label} JSONL is empty: {path}")
    return output


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


def _unit_interval(value: Any, label: str, *, positive: bool = False) -> float:
    result = _finite(value, label)
    lower_ok = result > 0.0 if positive else result >= 0.0
    if not lower_ok or result > 1.0:
        bracket = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{label} must be in {bracket}")
    return result


def _xyxy(value: Any, label: str) -> tuple[float, float, float, float]:
    rows = _sequence(value, label)
    if len(rows) != 4:
        raise ValueError(f"{label} must contain four coordinates")
    x1, y1, x2, y2 = (_finite(item, label) for item in rows)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{label} must have positive width and height")
    return x1, y1, x2, y2


def _xywh_to_xyxy(value: Any, label: str) -> tuple[float, float, float, float]:
    rows = _sequence(value, label)
    if len(rows) != 4:
        raise ValueError(f"{label} must contain four coordinates")
    x, y, width, height = (_finite(item, label) for item in rows)
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"{label} must have positive width and height")
    return x, y, x + width, y + height


def _box_area(box: tuple[float, float, float, float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0.0:
        return 0.0
    union = _box_area(left) + _box_area(right) - intersection
    return intersection / union if union > 0.0 else 0.0


def _in_area_range(
    area: float, area_range: tuple[float, float | None] | None
) -> bool:
    if area_range is None:
        return True
    minimum, maximum = area_range
    return area >= minimum and (maximum is None or area < maximum)


def _interpolated_ap101(
    tp_flags: Sequence[int], fp_flags: Sequence[int], support: int
) -> float | None:
    if support <= 0:
        return None
    if len(tp_flags) != len(fp_flags):
        raise ValueError("TP/FP flag cardinality mismatch")
    cumulative_tp = 0
    cumulative_fp = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for tp, fp in zip(tp_flags, fp_flags):
        cumulative_tp += int(tp)
        cumulative_fp += int(fp)
        recalls.append(cumulative_tp / support)
        denominator = cumulative_tp + cumulative_fp
        precisions.append(cumulative_tp / denominator if denominator else 0.0)
    total = 0.0
    for index in range(101):
        recall_threshold = index / 100.0
        total += max(
            (
                precision
                for recall, precision in zip(recalls, precisions)
                if recall >= recall_threshold
            ),
            default=0.0,
        )
    return total / 101.0


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    denominator = precision + recall
    f1 = 2.0 * precision * recall / denominator if denominator else 0.0
    return precision, recall, f1


def _match_class(
    truths: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    view_order: Mapping[str, int],
    iou_threshold: float,
    area_range: tuple[float, float | None] | None,
) -> dict[str, Any]:
    truth_by_view: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sorted(
        truths,
        key=lambda item: (
            view_order[str(item["view_id"])],
            int(item["annotation_id"]),
        ),
    ):
        truth_by_view[str(row["view_id"])].append(row)
    matched = {
        view_id: [False] * len(rows) for view_id, rows in truth_by_view.items()
    }
    ranked = sorted(
        predictions,
        key=lambda item: (
            -float(item["score"]),
            view_order[str(item["view_id"])],
            int(item["prediction_index"]),
        ),
    )
    support = sum(
        _in_area_range(float(row["area"]), area_range) for row in truths
    )
    tp_flags: list[int] = []
    fp_flags: list[int] = []
    ignored_predictions = 0
    for prediction in ranked:
        view_id = str(prediction["view_id"])
        candidates = truth_by_view.get(view_id, [])
        selected = -1
        # In-range truth has priority.  Matching out-of-range truth suppresses
        # that detection for this area stratum instead of manufacturing an FP.
        for desired_in_range in (True, False):
            best_iou = -1.0
            best_index = -1
            for index, truth in enumerate(candidates):
                if matched[view_id][index]:
                    continue
                if _in_area_range(float(truth["area"]), area_range) is not desired_in_range:
                    continue
                overlap = _box_iou(
                    prediction["box"],  # type: ignore[arg-type]
                    truth["box"],  # type: ignore[arg-type]
                )
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = index
            if best_index >= 0 and best_iou >= iou_threshold:
                selected = best_index
                break
        if selected >= 0:
            matched[view_id][selected] = True
            if _in_area_range(float(candidates[selected]["area"]), area_range):
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                ignored_predictions += 1
            continue
        if _in_area_range(float(prediction["area"]), area_range):
            tp_flags.append(0)
            fp_flags.append(1)
        else:
            ignored_predictions += 1
    tp = sum(tp_flags)
    fp = sum(fp_flags)
    fn = support - tp
    precision, recall, f1 = _prf(tp, fp, fn)
    return {
        "support": support,
        "evaluated_predictions": tp + fp,
        "ignored_predictions": ignored_predictions,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision50": precision,
        "recall50": recall,
        "f1_50": f1,
        "ap50": _interpolated_ap101(tp_flags, fp_flags, support),
    }


def evaluate_stratum(
    truths: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    categories: Sequence[Mapping[str, Any]],
    view_order: Mapping[str, int],
    iou_threshold: float = 0.5,
    view_ids: Iterable[str] | None = None,
    area_range: tuple[float, float | None] | None = None,
) -> dict[str, Any]:
    """Evaluate a deterministic image/area stratum without model inference."""

    threshold = _unit_interval(iou_threshold, "iou_threshold", positive=True)
    selected_views = set(view_order) if view_ids is None else set(view_ids)
    unknown_views = selected_views - set(view_order)
    if unknown_views:
        raise ValueError(f"unknown evaluation views: {sorted(unknown_views)[:4]}")
    selected_truths = [row for row in truths if row["view_id"] in selected_views]
    selected_predictions = [
        row for row in predictions if row["view_id"] in selected_views
    ]
    category_by_label: dict[int, Mapping[str, Any]] = {}
    for category in categories:
        label = category.get("model_label", category.get("id"))
        if type(label) is not int or label <= 0 or label in category_by_label:
            raise ValueError("categories require unique positive model labels")
        category_by_label[label] = category
    observed_labels = {
        int(row["label"]) for row in [*selected_truths, *selected_predictions]
    }
    if observed_labels - set(category_by_label):
        raise ValueError(
            f"records contain unknown labels: {sorted(observed_labels - set(category_by_label))}"
        )
    by_class: dict[str, dict[str, Any]] = {}
    aps: list[float] = []
    total_tp = total_fp = total_fn = total_support = 0
    evaluated_predictions = ignored_predictions = 0
    for label in sorted(category_by_label):
        category = category_by_label[label]
        result = _match_class(
            [row for row in selected_truths if int(row["label"]) == label],
            [row for row in selected_predictions if int(row["label"]) == label],
            view_order=view_order,
            iou_threshold=threshold,
            area_range=area_range,
        )
        result.update(
            {
                "name": str(category.get("name", label)),
                "model_label": label,
                "official_category_id": category.get("official_category_id"),
            }
        )
        by_class[str(label)] = result
        if result["ap50"] is not None:
            aps.append(float(result["ap50"]))
        total_support += int(result["support"])
        evaluated_predictions += int(result["evaluated_predictions"])
        ignored_predictions += int(result["ignored_predictions"])
        total_tp += int(result["true_positives"])
        total_fp += int(result["false_positives"])
        total_fn += int(result["false_negatives"])
    precision, recall, f1 = _prf(total_tp, total_fp, total_fn)
    return {
        "support": total_support,
        "image_support": len(selected_views),
        "evaluated_predictions": evaluated_predictions,
        "ignored_predictions": ignored_predictions,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision50": precision,
        "recall50": recall,
        "f1_50": f1,
        "map50": sum(aps) / len(aps) if aps else None,
        "evaluated_class_count": len(aps),
        "per_class": by_class,
    }


def _expect_sha(actual: str, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} expected SHA-256 is missing or invalid")
    if actual != expected.lower():
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected.lower()}, actual={actual}"
        )


def _unique_index(
    rows: Iterable[Mapping[str, Any]], key: str, label: str
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        identity = _nonempty_string(row.get(key), f"{label}.{key}")
        if identity in output:
            raise ValueError(f"duplicate {label} {key}: {identity}")
        output[identity] = row
    return output


def _relative(path: Path, workspace: Path) -> str:
    return path.relative_to(workspace).as_posix()


def _without_per_class(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "per_class"}


def _close(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _verified_reference(
    *,
    workspace: Path,
    reference: Mapping[str, Any],
    expected_path: Path,
    label: str,
) -> Path:
    raw_path = _nonempty_string(reference.get("path"), f"{label}.path")
    path = _resolve_within(
        workspace, Path(raw_path), label, must_exist=True
    )
    if path != expected_path:
        raise ValueError(
            f"{label} path differs from supplied evidence: "
            f"expected={expected_path}, actual={path}"
        )
    _expect_sha(_sha256(path), reference.get("sha256"), label)
    return path


def _validate_tiled_diagnostic_status(
    *,
    workspace: Path,
    status: Mapping[str, Any],
    predictions_path: Path,
    component_coco_path: Path,
    materialization_summary_path: Path,
    score_threshold: float,
    iou_threshold: float,
) -> tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Verify the stitched prediction artifact and its formal parent chain."""

    if (
        status.get("kind") != TILED_DIAGNOSTIC_KIND
        or status.get("status") != DIAGNOSTIC_STATUS
        or status.get("formal_evidence") is not False
        or status.get("registerable") is not False
        or status.get("promotion_eligible") is not False
        or status.get("blessing_authorized") is not False
        or status.get("smoke") is not False
    ):
        raise ValueError(
            "tiled Component status must be complete, full-validation, and "
            "strictly non-registerable"
        )
    if not status.get("finished_at_utc") or status.get("device") != "cpu":
        raise ValueError("tiled Component status lacks its completed CPU runtime contract")

    artifacts = _mapping(status.get("artifacts"), "tiled status.artifacts")
    prediction_reference = _mapping(
        artifacts.get("predictions"), "tiled status.artifacts.predictions"
    )
    _verified_reference(
        workspace=workspace,
        reference=prediction_reference,
        expected_path=predictions_path,
        label="tiled predictions",
    )
    lineage = _mapping(status.get("lineage"), "tiled status.lineage")
    parent_reference = _mapping(
        lineage.get("parent_status"), "tiled status.lineage.parent_status"
    )
    parent_path = _resolve_within(
        workspace,
        Path(_nonempty_string(parent_reference.get("path"), "parent status path")),
        "tiled diagnostic parent status",
        must_exist=True,
    )
    parent_sha256 = _sha256(parent_path)
    _expect_sha(
        parent_sha256,
        parent_reference.get("sha256"),
        "tiled diagnostic parent status",
    )
    parent = _read_object(parent_path, "tiled diagnostic parent status")
    if (
        parent.get("task") != "component"
        or parent.get("status") not in FINAL_FORMAL_STATUSES
        or parent.get("formal_evidence") is not True
        or parent.get("smoke") is not False
        or not parent.get("finished_at_utc")
    ):
        raise ValueError("tiled diagnostic parent is not a completed formal Component run")
    if parent_reference.get("status") != parent.get("status"):
        raise ValueError("tiled diagnostic parent status label differs from parent bytes")

    _verified_reference(
        workspace=workspace,
        reference=_mapping(
            lineage.get("component_coco"), "tiled status.lineage.component_coco"
        ),
        expected_path=component_coco_path,
        label="tiled Component COCO",
    )
    materialization_reference = _mapping(
        lineage.get("materialization_summary"),
        "tiled status.lineage.materialization_summary",
    )
    _verified_reference(
        workspace=workspace,
        reference=materialization_reference,
        expected_path=materialization_summary_path,
        label="tiled materialization summary",
    )

    checkpoint_reference = _mapping(
        lineage.get("checkpoint"), "tiled status.lineage.checkpoint"
    )
    parent_artifact = _mapping(parent.get("artifact"), "tiled parent.artifact")
    parent_checkpoint_path = _resolve_within(
        workspace,
        Path(_nonempty_string(parent_artifact.get("path"), "parent artifact path")),
        "tiled parent checkpoint",
        must_exist=True,
    )
    _verified_reference(
        workspace=workspace,
        reference=checkpoint_reference,
        expected_path=parent_checkpoint_path,
        label="tiled parent checkpoint",
    )
    _expect_sha(
        _sha256(parent_checkpoint_path),
        parent_artifact.get("sha256"),
        "tiled parent checkpoint/parent status",
    )

    protocol = _mapping(status.get("protocol"), "tiled status.protocol")
    expected_protocol = {
        "tile_size": 512,
        "tile_overlap": 64,
        "tile_stride": 448,
        "max_detections_per_tile": 300,
        "thresholds_searched": False,
        "tiling_configs_searched": False,
        "nms_configs_searched": False,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(
                f"tiled diagnostic protocol mismatch for {key}: "
                f"expected={expected!r}, actual={protocol.get(key)!r}"
            )
    if not _close(protocol.get("score_threshold"), score_threshold):
        raise ValueError("audit score threshold differs from tiled diagnostic protocol")
    if not _close(protocol.get("component_nms_iou"), 0.55):
        raise ValueError("tiled diagnostic NMS IoU is not the fixed 0.55 contract")
    if not _close(protocol.get("match_iou"), iou_threshold):
        raise ValueError("audit IoU threshold differs from tiled diagnostic protocol")

    boundary = _mapping(
        status.get("selection_boundary"), "tiled status.selection_boundary"
    )
    if boundary.get("split") != "val":
        raise ValueError("tiled diagnostic status is not restricted to validation")
    for key in (
        "train_samples_loaded",
        "train_images_decoded",
        "train_annotations_scored",
        "test_samples_loaded",
        "test_images_decoded",
        "test_annotations_scored",
    ):
        if boundary.get(key) != 0:
            raise ValueError(f"tiled diagnostic boundary requires {key}=0")
    interpretation = _mapping(
        status.get("interpretation_boundary"),
        "tiled status.interpretation_boundary",
    )
    expected_interpretation = {
        "training_gate_reused": False,
        "parent_gate_status_changed": False,
        "models_lock_read": False,
        "models_lock_modified": False,
        "validation_configuration_tuned": False,
        "test_split_used": False,
        "full_validation_conclusion_allowed": True,
    }
    for key, expected in expected_interpretation.items():
        if interpretation.get(key) is not expected:
            raise ValueError(f"tiled diagnostic interpretation mismatch for {key}")
    parent_lineage = _mapping(parent.get("lineage"), "tiled parent.lineage")
    provenance = {
        "kind": "tiled_diagnostic_replay",
        "status": status["status"],
        "diagnostic_kind": TILED_DIAGNOSTIC_KIND,
        "registerable": False,
        "promotion_eligible": False,
        "parent_run_status": {
            "path": _relative(parent_path, workspace),
            "sha256": parent_sha256,
            "status": parent["status"],
        },
        "coordinate_space": "original_source_image_xyxy",
        "stitched_predictions": True,
    }
    return provenance, parent_lineage, prediction_reference


def build_audit(
    *,
    workspace: Path,
    predictions_path: Path,
    status_path: Path,
    component_coco_path: Path,
    materialization_summary_path: Path,
    component_manifest_path: Path,
    source_manifest_path: Path,
    split_manifest_path: Path,
    score_threshold: float = 0.0,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    score_threshold = _unit_interval(score_threshold, "score_threshold")
    iou_threshold = _unit_interval(iou_threshold, "iou_threshold", positive=True)
    inputs = {
        "predictions": predictions_path,
        "run_status": status_path,
        "component_coco": component_coco_path,
        "materialization_summary": materialization_summary_path,
        "component_manifest": component_manifest_path,
        "source_manifest": source_manifest_path,
        "split_manifest": split_manifest_path,
    }
    hashes = {name: _sha256(path) for name, path in inputs.items()}
    status = _read_object(status_path, "Component run status")
    if status.get("task") != "component":
        raise ValueError("run status task is not component")
    status_kind = status.get("status")
    evaluation_provenance: dict[str, Any]
    tiled_mode = status.get("kind") == TILED_DIAGNOSTIC_KIND
    run_lineage_override: Mapping[str, Any] | None = None
    status_predictions: Mapping[str, Any] | None = None
    if status_kind in FINAL_FORMAL_STATUSES:
        if status.get("formal_evidence") is not True or status.get("smoke") is not False:
            raise ValueError("Component run status is not formal non-smoke evidence")
        evaluation_provenance = {
            "kind": "formal_run",
            "status": status_kind,
        }
    elif status_kind == DIAGNOSTIC_STATUS and tiled_mode:
        (
            evaluation_provenance,
            run_lineage_override,
            status_predictions,
        ) = _validate_tiled_diagnostic_status(
            workspace=workspace,
            status=status,
            predictions_path=predictions_path,
            component_coco_path=component_coco_path,
            materialization_summary_path=materialization_summary_path,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
        )
    elif status_kind == DIAGNOSTIC_STATUS:
        if (
            status.get("formal_evidence") is not False
            or status.get("registerable") is not False
            or status.get("smoke") is not False
        ):
            raise ValueError("diagnostic Component status must be non-registerable")
        diagnostic_parent = _mapping(
            status.get("diagnostic_parent"), "status.diagnostic_parent"
        )
        parent_reference = _mapping(
            diagnostic_parent.get("run_status"),
            "status.diagnostic_parent.run_status",
        )
        parent_relative = _nonempty_string(
            parent_reference.get("path"), "diagnostic parent run-status path"
        )
        parent_path = _resolve_within(
            workspace,
            Path(parent_relative),
            "diagnostic parent run status",
            must_exist=True,
        )
        parent_sha256 = _sha256(parent_path)
        _expect_sha(
            parent_sha256,
            parent_reference.get("sha256"),
            "diagnostic parent run status",
        )
        parent_status = _read_object(parent_path, "diagnostic parent Component status")
        if (
            parent_status.get("task") != "component"
            or parent_status.get("status") not in FINAL_FORMAL_STATUSES
            or parent_status.get("formal_evidence") is not True
            or parent_status.get("smoke") is not False
            or not parent_status.get("finished_at_utc")
        ):
            raise ValueError("diagnostic parent is not a completed formal Component run")
        if parent_reference.get("status") != parent_status.get("status"):
            raise ValueError("diagnostic parent status label differs from parent bytes")
        if dict(_mapping(parent_status.get("lineage"), "parent.lineage")) != dict(
            _mapping(status.get("lineage"), "status.lineage")
        ):
            raise ValueError("diagnostic replay lineage differs from its formal parent")
        evaluation_provenance = {
            "kind": "diagnostic_replay",
            "status": status_kind,
            "registerable": False,
            "parent_run_status": {
                "path": _relative(parent_path, workspace),
                "sha256": parent_sha256,
                "status": parent_status["status"],
            },
        }
    else:
        raise ValueError("Component status is neither a completed formal run nor a verified diagnostic replay")
    if not status.get("finished_at_utc"):
        raise ValueError("Component run status lacks finished_at_utc")
    if status_predictions is None:
        status_predictions = _mapping(status.get("predictions"), "status.predictions")
    _expect_sha(
        hashes["predictions"],
        status_predictions.get("sha256"),
        "predictions/run status",
    )

    summary = _read_object(materialization_summary_path, "materialization summary")
    coco = _read_object(component_coco_path, "Component COCO")
    info = _mapping(coco.get("info"), "component_coco.info")
    binding = _nonempty_string(summary.get("binding_sha256"), "summary binding")
    if tiled_mode:
        tiled_materialization_reference = _mapping(
            _mapping(status.get("lineage"), "tiled status.lineage").get(
                "materialization_summary"
            ),
            "tiled status.lineage.materialization_summary",
        )
        if tiled_materialization_reference.get("binding_sha256") != binding:
            raise ValueError("tiled status/materialization binding mismatch")
    if info.get("binding_sha256") != binding:
        raise ValueError("Component COCO/materialization binding mismatch")
    summary_outputs = _mapping(summary.get("outputs"), "summary.outputs")
    summary_coco = _mapping(
        summary_outputs.get("component_coco.json"),
        "summary.outputs.component_coco.json",
    )
    _expect_sha(hashes["component_coco"], summary_coco.get("sha256"), "COCO/summary")
    summary_lineage = _mapping(summary.get("lineage"), "summary.lineage")
    info_lineage = _mapping(info.get("lineage"), "component_coco.info.lineage")
    if dict(info_lineage) != dict(summary_lineage):
        raise ValueError("Component COCO lineage differs from materialization summary")
    lineage_inputs = _mapping(summary_lineage.get("inputs"), "summary lineage inputs")
    for input_name, source_key in (
        ("component_manifest", "component_manifest"),
        ("source_manifest", "source_manifest"),
        ("split_manifest", "split_manifest"),
    ):
        expected = _mapping(
            lineage_inputs.get(input_name), f"summary lineage {input_name}"
        ).get("sha256")
        _expect_sha(hashes[source_key], expected, f"{source_key}/summary")

    run_lineage = (
        run_lineage_override
        if run_lineage_override is not None
        else _mapping(status.get("lineage"), "status.lineage")
    )
    expected_run_values = {
        "task": "component",
        "formal_evidence": True,
        "smoke": False,
        "binding_sha256": binding,
        "materialization_commit_id": binding,
        "materialization_commit_sha256": hashes["materialization_summary"],
        "source_manifest_sha256": hashes["source_manifest"],
        "split_manifest_sha256": hashes["split_manifest"],
        "task_manifest_sha256": hashes["component_manifest"],
    }
    for key, expected in expected_run_values.items():
        if run_lineage.get(key) != expected:
            raise ValueError(
                f"run lineage mismatch for {key}: "
                f"expected={expected!r}, actual={run_lineage.get(key)!r}"
            )

    split_rows = _read_jsonl(split_manifest_path, "split manifest")
    split_by_sample = {
        sample_id: _nonempty_string(row.get("split"), "split manifest split")
        for sample_id, row in _unique_index(
            split_rows, "sample_id", "split manifest"
        ).items()
    }
    if any(split not in {"train", "val", "test"} for split in split_by_sample.values()):
        raise ValueError("split manifest contains unsupported split")
    source_rows = _read_jsonl(source_manifest_path, "source manifest")
    source_by_sample = _unique_index(source_rows, "sample_id", "source manifest")
    if set(source_by_sample) != set(split_by_sample):
        raise ValueError("source/split manifest sample populations differ")
    component_rows = _read_jsonl(component_manifest_path, "component manifest")
    component_by_record = _unique_index(
        component_rows, "record_id", "component manifest"
    )
    val_component_by_record: dict[str, Mapping[str, Any]] = {}
    domain_by_sample: dict[str, str] = {}
    component_rows_by_split: dict[str, int] = defaultdict(int)
    for record_id, row in component_by_record.items():
        sample_id = _nonempty_string(row.get("sample_id"), "component sample_id")
        split = _nonempty_string(row.get("split"), "component split")
        if sample_id not in split_by_sample or split_by_sample[sample_id] != split:
            raise ValueError(f"component/split manifest mismatch at {record_id}")
        if row.get("state") != "active":
            raise ValueError(f"component manifest record is not active: {record_id}")
        source_domain = _nonempty_string(
            row.get("source_domain"), "component source_domain"
        )
        previous_domain = domain_by_sample.setdefault(sample_id, source_domain)
        if previous_domain != source_domain:
            raise ValueError(f"multiple source domains for sample {sample_id}")
        component_rows_by_split[split] += 1
        if split == "val":
            val_component_by_record[record_id] = row

    raw_categories = _sequence(coco.get("categories"), "component_coco.categories")
    categories: list[dict[str, Any]] = []
    seen_labels: set[int] = set()
    for raw_category in raw_categories:
        category = _mapping(raw_category, "COCO category")
        label = category.get("id")
        if type(label) is not int or label <= 0 or label in seen_labels:
            raise ValueError("COCO categories require unique positive IDs")
        seen_labels.add(label)
        official_raw = category.get("source_category_id")
        try:
            official_id = int(official_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"category {label} has invalid official ID") from exc
        categories.append(
            {
                "model_label": label,
                "official_category_id": official_id,
                "name": _nonempty_string(category.get("name"), "category name"),
            }
        )
    categories.sort(key=lambda row: int(row["model_label"]))
    category_by_label = {int(row["model_label"]): row for row in categories}

    raw_images = _sequence(coco.get("images"), "component_coco.images")
    image_by_id: dict[int, Mapping[str, Any]] = {}
    val_full_by_view: dict[str, Mapping[str, Any]] = {}
    for raw_image in raw_images:
        image = _mapping(raw_image, "COCO image")
        image_id = image.get("id")
        if type(image_id) is not int or image_id <= 0 or image_id in image_by_id:
            raise ValueError("COCO images require unique positive integer IDs")
        image_by_id[image_id] = image
        sample_id = _nonempty_string(image.get("sample_id"), "COCO image sample_id")
        split = _nonempty_string(image.get("split"), "COCO image split")
        if split_by_sample.get(sample_id) != split:
            raise ValueError(f"COCO/split manifest mismatch at image {image_id}")
        if image.get("binding_sha256") != binding:
            raise ValueError(f"COCO image {image_id} binding mismatch")
        if split == "val" and image.get("view_kind") == "full":
            view_id = _nonempty_string(image.get("view_id"), "COCO image view_id")
            if view_id in val_full_by_view:
                raise ValueError(f"duplicate validation full view: {view_id}")
            source = source_by_sample[sample_id]
            source_files = _mapping(source.get("files"), "source files")
            source_image = _mapping(source_files.get("image"), "source image")
            if image.get("source_image") != source_image.get("path"):
                raise ValueError(f"COCO/source image path mismatch at {view_id}")
            if image.get("source_image_sha256") != source_image.get("sha256"):
                raise ValueError(f"COCO/source image hash mismatch at {view_id}")
            val_full_by_view[view_id] = image
    expected_val_samples = {
        sample_id for sample_id, split in split_by_sample.items() if split == "val"
    }
    observed_val_samples = {
        str(image["sample_id"]) for image in val_full_by_view.values()
    }
    if observed_val_samples != expected_val_samples:
        raise ValueError("validation full views do not cover the frozen val samples")
    if len(observed_val_samples) != len(val_full_by_view):
        raise ValueError("validation materialization has multiple full views per sample")
    ordered_views = sorted(
        val_full_by_view,
        key=lambda view_id: int(val_full_by_view[view_id]["id"]),
    )
    view_order = {view_id: index for index, view_id in enumerate(ordered_views)}
    view_by_image_id = {
        int(image["id"]): view_id for view_id, image in val_full_by_view.items()
    }

    truths: list[dict[str, Any]] = []
    joined_records: set[str] = set()
    raw_annotations = _sequence(coco.get("annotations"), "component_coco.annotations")
    seen_annotation_ids: set[int] = set()
    for raw_annotation in raw_annotations:
        annotation = _mapping(raw_annotation, "COCO annotation")
        annotation_id = annotation.get("id")
        if (
            type(annotation_id) is not int
            or annotation_id <= 0
            or annotation_id in seen_annotation_ids
        ):
            raise ValueError("COCO annotations require unique positive IDs")
        seen_annotation_ids.add(annotation_id)
        image_id = annotation.get("image_id")
        if type(image_id) is not int or image_id not in image_by_id:
            raise ValueError(f"annotation {annotation_id} references an unknown image")
        if image_id not in view_by_image_id:
            continue
        if bool(annotation.get("ignore", 0)):
            raise ValueError("validation full views unexpectedly contain ignored truth")
        view_id = view_by_image_id[image_id]
        image = val_full_by_view[view_id]
        sample_id = str(image["sample_id"])
        record_id = _nonempty_string(
            annotation.get("source_record_id"), "annotation source_record_id"
        )
        if record_id in joined_records:
            raise ValueError(f"duplicate full-view source record: {record_id}")
        truth_manifest = val_component_by_record.get(record_id)
        if truth_manifest is None:
            raise ValueError(f"full-view annotation lacks val manifest row: {record_id}")
        if truth_manifest.get("sample_id") != sample_id:
            raise ValueError(f"annotation/manifest sample mismatch: {record_id}")
        label = annotation.get("category_id")
        if type(label) is not int or label not in category_by_label:
            raise ValueError(f"annotation {annotation_id} has invalid category")
        category = category_by_label[label]
        if (
            truth_manifest.get("category") != category["name"]
            or int(str(truth_manifest.get("category_id")))
            != category["official_category_id"]
        ):
            raise ValueError(f"annotation/manifest category mismatch: {record_id}")
        source_box = _sequence(annotation.get("source_bbox"), "annotation source_bbox")
        manifest_box = _mapping(truth_manifest.get("bbox"), "manifest bbox")
        expected_source_box = [
            float(manifest_box[key]) for key in ("x1", "y1", "x2", "y2")
        ]
        if [float(value) for value in source_box] != expected_source_box:
            raise ValueError(f"annotation/manifest source bbox mismatch: {record_id}")
        source_box_xyxy = _xyxy(source_box, f"annotation {annotation_id} source bbox")
        materialized_box = _xywh_to_xyxy(
            annotation.get("bbox"), f"annotation {annotation_id} bbox"
        )
        reported_area = _finite(annotation.get("area"), f"annotation {annotation_id} area")
        if not math.isclose(
            reported_area,
            _box_area(materialized_box),
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            raise ValueError(f"annotation {annotation_id} area differs from bbox")
        box = source_box_xyxy if tiled_mode else materialized_box
        area = _box_area(source_box_xyxy) if tiled_mode else reported_area
        joined_records.add(record_id)
        truths.append(
            {
                "annotation_id": annotation_id,
                "record_id": record_id,
                "sample_id": sample_id,
                "view_id": view_id,
                "label": label,
                "box": box,
                "area": area,
                "source_class": _nonempty_string(
                    source_by_sample[sample_id].get("source_class"),
                    "source_class",
                ),
                "source_domain": _nonempty_string(
                    truth_manifest.get("source_domain"), "source_domain"
                ),
            }
        )
    if joined_records != set(val_component_by_record):
        missing = sorted(set(val_component_by_record) - joined_records)
        extra = sorted(joined_records - set(val_component_by_record))
        raise ValueError(
            "full-view truth/val manifest coverage mismatch: "
            f"missing={missing[:4]}, extra={extra[:4]}"
        )

    prediction_rows = _read_jsonl(predictions_path, "Component predictions")
    predictions: list[dict[str, Any]] = []
    discarded_below_threshold = 0
    raw_prediction_count = 0
    if tiled_mode:
        prediction_by_sample = _unique_index(
            prediction_rows, "sample_id", "tiled Component predictions"
        )
        if set(prediction_by_sample) != expected_val_samples:
            unexpected = sorted(set(prediction_by_sample) - expected_val_samples)
            missing = sorted(expected_val_samples - set(prediction_by_sample))
            raise ValueError(
                "tiled predictions are not an exact frozen-val stitched population: "
                f"unexpected={unexpected[:4]}, missing={missing[:4]}"
            )
        category_by_name: dict[str, Mapping[str, Any]] = {}
        for category in categories:
            name = str(category["name"])
            if name in category_by_name:
                raise ValueError(f"duplicate Component category name {name!r}")
            category_by_name[name] = category
        total_tiles = 0
        total_candidates_before_nms = 0
        for view_id in ordered_views:
            image = val_full_by_view[view_id]
            sample_id = str(image["sample_id"])
            row = prediction_by_sample[sample_id]
            source_size = _sequence(
                image.get("source_size"), f"validation {sample_id} source_size"
            )
            if len(source_size) != 2:
                raise ValueError(f"validation {sample_id} source_size is invalid")
            width = int(source_size[0])
            height = int(source_size[1])
            if width <= 0 or height <= 0:
                raise ValueError(f"validation {sample_id} source_size is non-positive")
            shape = _sequence(row.get("shape"), f"tiled prediction {sample_id} shape")
            if list(shape) != [height, width]:
                raise ValueError(f"tiled prediction/source shape mismatch at {sample_id}")
            if row.get("source_image_sha256") != image.get("source_image_sha256"):
                raise ValueError(f"tiled prediction image hash mismatch at {sample_id}")
            tile_count = row.get("tile_count")
            candidate_count = row.get("candidate_count_before_cross_tile_nms")
            if type(tile_count) is not int or tile_count <= 0:
                raise ValueError(f"tiled prediction {sample_id} has invalid tile_count")
            if type(candidate_count) is not int or candidate_count < 0:
                raise ValueError(
                    f"tiled prediction {sample_id} has invalid pre-NMS candidate count"
                )
            sample_predictions = _sequence(
                row.get("predictions"), f"tiled prediction {sample_id} predictions"
            )
            if candidate_count < len(sample_predictions):
                raise ValueError(
                    f"tiled prediction {sample_id} has fewer pre-NMS than stitched rows"
                )
            total_tiles += tile_count
            total_candidates_before_nms += candidate_count
            raw_prediction_count += len(sample_predictions)
            source_class = _nonempty_string(
                source_by_sample[sample_id].get("source_class"), "source_class"
            )
            source_domain = domain_by_sample.get(sample_id, source_class)
            for prediction_index, raw_prediction in enumerate(sample_predictions):
                prediction = _mapping(raw_prediction, "tiled Component prediction")
                category_name = _nonempty_string(
                    prediction.get("category"), "tiled prediction category"
                )
                category = category_by_name.get(category_name)
                if category is None:
                    raise ValueError(
                        f"tiled prediction at {sample_id} has unknown category "
                        f"{category_name!r}"
                    )
                score = _unit_interval(
                    prediction.get("score"), "tiled prediction score"
                )
                if score < score_threshold:
                    raise ValueError(
                        "stitched prediction falls below its fixed adapter score threshold"
                    )
                box = _xyxy(
                    prediction.get("box"),
                    f"tiled prediction {sample_id}:{prediction_index} bbox",
                )
                if not (
                    0.0 <= box[0] < box[2] <= width
                    and 0.0 <= box[1] < box[3] <= height
                ):
                    raise ValueError(
                        f"tiled prediction lies outside source image at {sample_id}"
                    )
                _nonempty_string(
                    prediction.get("source_view_id"), "tiled prediction source_view_id"
                )
                predictions.append(
                    {
                        "sample_id": sample_id,
                        "view_id": view_id,
                        "label": int(category["model_label"]),
                        "box": box,
                        "area": _box_area(box),
                        "score": score,
                        "prediction_index": prediction_index,
                        "source_class": source_class,
                        "source_domain": source_domain,
                    }
                )
        tiled_boundary = _mapping(
            status.get("selection_boundary"), "tiled status.selection_boundary"
        )
        if (
            tiled_boundary.get("available_val_samples") != len(expected_val_samples)
            or tiled_boundary.get("selected_val_samples") != len(expected_val_samples)
            or list(
                _sequence(
                    tiled_boundary.get("selected_sample_ids"),
                    "tiled selected_sample_ids",
                )
            )
            != sorted(expected_val_samples)
        ):
            raise ValueError("tiled status does not cover the exact frozen validation split")
        if status_predictions.get("row_count") != len(prediction_rows):
            raise ValueError("tiled status prediction row count differs from artifact")
        performance = _mapping(status.get("performance"), "tiled status.performance")
        expected_performance = {
            "sample_count": len(prediction_rows),
            "tile_count": total_tiles,
            "candidate_count_before_cross_tile_nms": total_candidates_before_nms,
            "prediction_count_after_cross_tile_nms": raw_prediction_count,
        }
        for key, expected in expected_performance.items():
            if performance.get(key) != expected:
                raise ValueError(
                    f"tiled prediction/status performance mismatch for {key}"
                )
    else:
        prediction_by_view = _unique_index(
            prediction_rows, "view_id", "Component predictions"
        )
        if set(prediction_by_view) != set(val_full_by_view):
            unexpected = sorted(set(prediction_by_view) - set(val_full_by_view))
            missing = sorted(set(val_full_by_view) - set(prediction_by_view))
            raise ValueError(
                "predictions are not an exact frozen-val full-view population: "
                f"unexpected={unexpected[:4]}, missing={missing[:4]}"
            )
        for view_id in ordered_views:
            row = prediction_by_view[view_id]
            sample_id = str(val_full_by_view[view_id]["sample_id"])
            if row.get("sample_id") != sample_id:
                raise ValueError(f"prediction sample/view mismatch at {view_id}")
            boxes = _sequence(row.get("boxes"), f"prediction {view_id} boxes")
            scores = _sequence(row.get("scores"), f"prediction {view_id} scores")
            labels = _sequence(row.get("labels"), f"prediction {view_id} labels")
            if not (len(boxes) == len(scores) == len(labels)):
                raise ValueError(f"prediction arrays differ in length at {view_id}")
            raw_prediction_count += len(scores)
            source_class = _nonempty_string(
                source_by_sample[sample_id].get("source_class"), "source_class"
            )
            source_domain = domain_by_sample.get(sample_id, source_class)
            for prediction_index, (raw_box, raw_score, raw_label) in enumerate(
                zip(boxes, scores, labels)
            ):
                score = _unit_interval(raw_score, "prediction score")
                if type(raw_label) is not int or raw_label not in category_by_label:
                    raise ValueError(
                        f"prediction at {view_id} has unknown label {raw_label!r}"
                    )
                box = _xyxy(raw_box, f"prediction {view_id}:{prediction_index} bbox")
                if score < score_threshold:
                    discarded_below_threshold += 1
                    continue
                predictions.append(
                    {
                        "sample_id": sample_id,
                        "view_id": view_id,
                        "label": raw_label,
                        "box": box,
                        "area": _box_area(box),
                        "score": score,
                        "prediction_index": prediction_index,
                        "source_class": source_class,
                        "source_domain": source_domain,
                    }
                )

    overall = evaluate_stratum(
        truths,
        predictions,
        categories=categories,
        view_order=view_order,
        iou_threshold=iou_threshold,
    )
    source_metrics = _mapping(status.get("metrics"), "status.metrics")
    reproduction = {
        "bbox_map50": _close(overall["map50"], source_metrics.get("bbox_map50")),
        "precision50": _close(
            overall["precision50"], source_metrics.get("precision50")
        ),
        "recall50": _close(overall["recall50"], source_metrics.get("recall50")),
    }
    if "f1_50" in source_metrics:
        reproduction["f1_50"] = _close(
            overall["f1_50"], source_metrics.get("f1_50")
        )
    if not all(reproduction.values()):
        raise ValueError(
            "stratified evaluator does not reproduce completed-run metrics: "
            f"checks={reproduction}, computed={{'bbox_map50': {overall['map50']}, "
            f"'precision50': {overall['precision50']}, "
            f"'recall50': {overall['recall50']}}}, source={dict(source_metrics)}"
        )

    by_size: dict[str, dict[str, Any]] = {}
    for name, minimum, maximum in SIZE_BINS:
        result = evaluate_stratum(
            truths,
            predictions,
            categories=categories,
            view_order=view_order,
            iou_threshold=iou_threshold,
            area_range=(minimum, maximum),
        )
        by_size[name] = {
            (
                "area_range_source_px2"
                if tiled_mode
                else "area_range_view_px2"
            ): {
                "minimum_inclusive": minimum,
                "maximum_exclusive": maximum,
            },
            **_without_per_class(result),
            "per_class": result["per_class"],
        }
    if sum(int(row["support"]) for row in by_size.values()) != int(overall["support"]):
        raise RuntimeError("size bins do not partition validation truth")

    views_by_source_class: dict[str, set[str]] = defaultdict(set)
    views_by_source_domain: dict[str, set[str]] = defaultdict(set)
    for view_id, image in val_full_by_view.items():
        sample_id = str(image["sample_id"])
        source_class = _nonempty_string(
            source_by_sample[sample_id].get("source_class"), "source_class"
        )
        views_by_source_class[source_class].add(view_id)
        views_by_source_domain[domain_by_sample.get(sample_id, source_class)].add(view_id)

    def coverage_fields(result: Mapping[str, Any]) -> dict[str, Any]:
        support = int(result["support"])
        prediction_count = int(result["evaluated_predictions"])
        return {
            "ground_truth_coverage": float(result["recall50"]),
            "pred_to_gt_ratio": prediction_count / support if support else None,
        }

    def source_strata(groups: Mapping[str, set[str]]) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for name, view_ids in sorted(groups.items()):
            result = _without_per_class(
                evaluate_stratum(
                    truths,
                    predictions,
                    categories=categories,
                    view_order=view_order,
                    iou_threshold=iou_threshold,
                    view_ids=view_ids,
                )
            )
            sample_ids = sorted(
                str(val_full_by_view[view_id]["sample_id"])
                for view_id in view_ids
            )
            output[name] = {
                "sample_count": len(sample_ids),
                "sample_ids": sample_ids,
                **result,
                **coverage_fields(result),
            }
        return output

    by_sample: dict[str, dict[str, Any]] = {}
    for view_id in ordered_views:
        image = val_full_by_view[view_id]
        sample_id = str(image["sample_id"])
        result = _without_per_class(
            evaluate_stratum(
                truths,
                predictions,
                categories=categories,
                view_order=view_order,
                iou_threshold=iou_threshold,
                view_ids={view_id},
            )
        )
        source_class = _nonempty_string(
            source_by_sample[sample_id].get("source_class"), "source_class"
        )
        by_sample[sample_id] = {
            "source_class": source_class,
            "source_domain": domain_by_sample.get(sample_id, source_class),
            **result,
            **coverage_fields(result),
        }

    by_class = overall.pop("per_class")
    zero_recall_supported_classes = sorted(
        (
            {
                "name": str(row["name"]),
                "model_label": int(row["model_label"]),
                "official_category_id": row.get("official_category_id"),
                "support": int(row["support"]),
                "predictions": int(row["evaluated_predictions"]),
                "ap50": row.get("ap50"),
            }
            for row in by_class.values()
            if int(row["support"]) > 0 and float(row["recall50"]) == 0.0
        ),
        key=lambda row: (int(row["official_category_id"]), str(row["name"])),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": (
            "pcbparse_component_tiled_frozen_val_stratified_audit"
            if tiled_mode
            else "pcbparse_component_frozen_val_stratified_audit"
        ),
        **(
            {
                "diagnostic_only": True,
                "registerable": False,
                "promotion_eligible": False,
            }
            if tiled_mode
            else {}
        ),
        "selection_boundary": {
            "split": "val",
            "val_sample_rows_used": len(expected_val_samples),
            "val_full_view_rows_used": len(val_full_by_view),
            "val_truth_rows_used": len(truths),
            "val_prediction_rows_used": len(prediction_rows),
            **(
                {"val_prediction_sample_rows_used": len(prediction_rows)}
                if tiled_mode
                else {"val_prediction_view_rows_used": len(prediction_rows)}
            ),
            "prediction_population_unit": (
                "stitched_source_image" if tiled_mode else "materialized_full_view"
            ),
            "train_rows_used": 0,
            "train_images_decoded": 0,
            "test_rows_used": 0,
            "test_images_decoded": 0,
            "test_manifest_rows_used_for_metrics": 0,
            "thresholds_tuned": False,
            "score_threshold": score_threshold,
            "iou_threshold": iou_threshold,
            "purpose": "frozen-validation diagnosis and next-run gate design",
        },
        "evaluation_provenance": evaluation_provenance,
        "contract": {
            "matching": (
                "class-aware one-to-one IoU matching per view; predictions globally "
                "ranked by score within class with deterministic view/index ties"
            ),
            "ap": "101-point interpolated AP at IoU 0.5; mAP excludes zero-support classes",
            "size_basis": (
                "original source-image bbox area in pixels squared"
                if tiled_mode
                else "materialized 512x512 full-view bbox area in pixels squared"
            ),
            "size_matching": (
                "non-overlapping half-open bins; detections matched to out-of-bin truth "
                "and unmatched detections outside the bin are ignored for that bin"
            ),
            "metric_aggregation": "micro P/R/F1 across classes; macro AP50 across supported classes",
        },
        "sources": {
            name: {
                "path": _relative(path, workspace),
                "sha256": hashes[name],
                "verified": True,
            }
            for name, path in inputs.items()
        },
        "lineage_checks": {
            "materialization_binding_sha256": binding,
            "component_manifest_rows_by_split": dict(
                sorted(component_rows_by_split.items())
            ),
            "full_view_truth_manifest_joined": len(joined_records),
            **(
                {
                    "unexpected_prediction_rows": 0,
                    "missing_prediction_rows": 0,
                }
                if tiled_mode
                else {
                    "unexpected_prediction_views": 0,
                    "missing_prediction_views": 0,
                }
            ),
            "run_metric_reproduction": reproduction,
        },
        "population": {
            "categories": len(categories),
            "validation_samples": len(expected_val_samples),
            "validation_full_views": len(val_full_by_view),
            "ground_truth_boxes": len(truths),
            "raw_predictions": raw_prediction_count,
            "evaluated_predictions": len(predictions),
            "discarded_below_score_threshold": discarded_below_threshold,
        },
        "overall": overall,
        "by_class": by_class,
        "zero_recall_supported_classes": zero_recall_supported_classes,
        "by_size": by_size,
        "by_source_class": source_strata(views_by_source_class),
        "by_source_domain": source_strata(views_by_source_domain),
        "by_sample": by_sample,
    }
    canonical = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    report["audit_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    return report


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    workspace = arguments.workspace_root.expanduser().resolve(strict=True)
    paths = {
        "predictions_path": _resolve_within(
            workspace, arguments.predictions, "predictions", must_exist=True
        ),
        "status_path": _resolve_within(
            workspace, arguments.run_status, "run status", must_exist=True
        ),
        "component_coco_path": _resolve_within(
            workspace, arguments.component_coco, "Component COCO", must_exist=True
        ),
        "materialization_summary_path": _resolve_within(
            workspace,
            arguments.materialization_summary,
            "materialization summary",
            must_exist=True,
        ),
        "component_manifest_path": _resolve_within(
            workspace,
            arguments.component_manifest,
            "component manifest",
            must_exist=True,
        ),
        "source_manifest_path": _resolve_within(
            workspace, arguments.source_manifest, "source manifest", must_exist=True
        ),
        "split_manifest_path": _resolve_within(
            workspace, arguments.split_manifest, "split manifest", must_exist=True
        ),
    }
    output = _resolve_within(
        workspace, arguments.output, "audit output", must_exist=False
    )
    report = build_audit(
        workspace=workspace,
        score_threshold=arguments.score_threshold,
        iou_threshold=arguments.iou_threshold,
        **paths,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)
    print(f"audit_payload_sha256={report['audit_payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
