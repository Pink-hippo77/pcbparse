from __future__ import annotations

"""Characterize YOLO Component validation errors without touching test data."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import evaluate_component_tiled_validation as base  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "work/eval-component-yolo11n-v1-tiled512-val-cap128-20260914/"
            "predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=Path(
            "work/eval-component-yolo11n-v1-tiled512-val-cap128-20260914/"
            "diagnostic_status.json"
        ),
    )
    parser.add_argument(
        "--component-coco",
        type=Path,
        default=Path(
            "artifacts/training/materialized/dataset300-v2/component-pin/"
            "component_coco.json"
        ),
    )
    parser.add_argument(
        "--materialization-summary",
        type=Path,
        default=Path(
            "artifacts/training/materialized/dataset300-v2/component-pin/summary.json"
        ),
    )
    parser.add_argument(
        "--source-root", type=Path, default=Path("Dataset300/Dataset300")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/analysis/component_yolo11n_v1_validation_error_modes.json"
        ),
    )
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"prediction row {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise ValueError("prediction JSONL is empty")
    return rows


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _score_summary(values: Sequence[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p25": _quantile(values, 0.25),
        "median": _quantile(values, 0.5),
        "p75": _quantile(values, 0.75),
        "max": max(values) if values else None,
    }


def _best_overlap(
    box: Sequence[float], truth: Sequence[Mapping[str, Any]]
) -> tuple[float, str | None]:
    candidates = [
        (base._box_iou(box, item["box"]), str(item["category"])) for item in truth
    ]
    return max(candidates, default=(0.0, None), key=lambda item: (item[0], item[1] or ""))


def _classify(
    prediction_rows: Sequence[Mapping[str, Any]],
    ground_truth: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []
    misses: list[dict[str, Any]] = []
    for row in prediction_rows:
        sample_id = str(row["sample_id"])
        truth = list(ground_truth[sample_id])
        by_class: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
        for index, item in enumerate(truth):
            by_class[str(item["category"])].append((index, item))
        matched_truth: set[int] = set()
        predictions = [
            dict(base._mapping(item, "prediction"))
            for item in base._sequence(row.get("predictions"), "predictions")
        ]
        predictions.sort(
            key=lambda item: (
                -float(item["score"]),
                str(item["category"]),
                tuple(float(value) for value in item["box"]),
            )
        )
        for prediction in predictions:
            category = str(prediction["category"])
            box = [float(value) for value in prediction["box"]]
            candidates = []
            for truth_index, item in by_class.get(category, []):
                if truth_index in matched_truth:
                    continue
                candidates.append((base._box_iou(box, item["box"]), truth_index))
            best_same_iou, best_same_index = max(
                candidates, default=(-1.0, -1), key=lambda item: (item[0], -item[1])
            )
            if best_same_iou >= 0.5:
                matched_truth.add(best_same_index)
                outcome = "true_positive"
                nearest_iou = best_same_iou
                nearest_category = category
            else:
                nearest_iou, nearest_category = _best_overlap(box, truth)
                if nearest_iou >= 0.5:
                    outcome = (
                        "duplicate_same_class"
                        if nearest_category == category
                        else "class_confusion"
                    )
                elif nearest_iou >= 0.1:
                    outcome = "localization_or_partial_overlap"
                else:
                    outcome = "background_or_unlabelled"
            decisions.append(
                {
                    "sample_id": sample_id,
                    "category": category,
                    "score": float(prediction["score"]),
                    "box": box,
                    "outcome": outcome,
                    "nearest_ground_truth_category": nearest_category,
                    "nearest_ground_truth_iou": nearest_iou,
                    "source_view_id": str(prediction.get("source_view_id", "")),
                }
            )
        for truth_index, item in enumerate(truth):
            if truth_index not in matched_truth:
                misses.append(
                    {
                        "sample_id": sample_id,
                        "category": str(item["category"]),
                        "box": list(item["box"]),
                    }
                )
    return decisions, misses


def _axis_positions(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    values = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if values[-1] != last:
        values.append(last)
    return values


def _touches_internal_tile_border(
    decision: Mapping[str, Any],
    *,
    shape: Sequence[int],
    tile_size: int,
    overlap: int,
) -> bool:
    parts = str(decision.get("source_view_id", "")).split("_")
    if len(parts) != 3 or parts[0] != "tile":
        return False
    try:
        row = int(parts[1])
        column = int(parts[2])
    except ValueError:
        return False
    height, width = int(shape[0]), int(shape[1])
    x_positions = _axis_positions(width, tile_size, overlap)
    y_positions = _axis_positions(height, tile_size, overlap)
    if row >= len(y_positions) or column >= len(x_positions):
        return False
    left, top = x_positions[column], y_positions[row]
    tile_width = min(tile_size, width - left)
    tile_height = min(tile_size, height - top)
    x1, y1, x2, y2 = (float(value) for value in decision["box"])
    local = (x1 - left, y1 - top, x2 - left, y2 - top)
    margin = overlap / 2.0
    return (
        (left > 0 and local[0] <= margin)
        or (left + tile_width < width and tile_width - local[2] <= margin)
        or (top > 0 and local[1] <= margin)
        or (top + tile_height < height and tile_height - local[3] <= margin)
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workspace = Path(args.workspace_root).expanduser().resolve(strict=True)
    predictions_path = base._resolve_within(
        workspace, args.predictions, "validation predictions"
    )
    status_path = base._resolve_within(workspace, args.status, "validation status")
    coco_path = base._resolve_within(workspace, args.component_coco, "Component COCO")
    summary_path = base._resolve_within(
        workspace, args.materialization_summary, "materialization summary"
    )
    source_root = base._resolve_within(
        workspace, args.source_root, "Dataset300 source root", kind="directory"
    )
    output_path = base._resolve_within(
        workspace, args.output, "error-mode output", must_exist=False
    )
    summary = base._read_object(summary_path, "materialization summary")
    samples, categories, boundary = base._frozen_validation_samples(
        component_coco=base._read_object(coco_path, "Component COCO"),
        source_root=source_root,
        binding_sha256=str(summary["binding_sha256"]),
        max_val_samples=None,
    )
    rows = _read_jsonl(predictions_path)
    expected_ids = [str(item["sample_id"]) for item in samples]
    if [str(row.get("sample_id")) for row in rows] != expected_ids:
        raise ValueError("prediction population differs from frozen validation split")
    status = base._read_object(status_path, "validation status")
    status_prediction = base._mapping(
        base._mapping(status.get("artifacts"), "status artifacts").get("predictions"),
        "status predictions",
    )
    if status_prediction.get("sha256") != base._sha256(predictions_path):
        raise ValueError("validation status does not bind prediction bytes")
    truth = {
        str(item["sample_id"]): list(item["_ground_truth"]) for item in samples
    }
    decisions, misses = _classify(rows, truth)
    protocol = base._mapping(status.get("protocol"), "validation protocol")
    tile_size = int(protocol["tile_size"])
    tile_overlap = int(protocol["tile_overlap"])
    shape_by_sample = {
        str(row["sample_id"]): list(row["shape"])
        for row in rows
    }
    for decision in decisions:
        decision["touches_internal_tile_border"] = _touches_internal_tile_border(
            decision,
            shape=shape_by_sample[str(decision["sample_id"])],
            tile_size=tile_size,
            overlap=tile_overlap,
        )
    category_names = [str(item["name"]) for item in categories]
    per_class: dict[str, Any] = {}
    for category in category_names:
        owned = [item for item in decisions if item["category"] == category]
        positives = [item for item in owned if item["outcome"] == "true_positive"]
        false_positives = [item for item in owned if item["outcome"] != "true_positive"]
        outcomes = Counter(item["outcome"] for item in false_positives)
        confusions = Counter(
            str(item["nearest_ground_truth_category"])
            for item in false_positives
            if item["outcome"] == "class_confusion"
        )
        class_misses = [item for item in misses if item["category"] == category]
        per_class[category] = {
            "ground_truth": len(positives) + len(class_misses),
            "true_positives": len(positives),
            "false_positives": len(false_positives),
            "false_negatives": len(class_misses),
            "false_positive_modes": dict(sorted(outcomes.items())),
            "confused_with_ground_truth": dict(
                sorted(confusions.items(), key=lambda item: (-item[1], item[0]))
            ),
            "true_positive_scores": _score_summary(
                [float(item["score"]) for item in positives]
            ),
            "false_positive_scores": _score_summary(
                [float(item["score"]) for item in false_positives]
            ),
            "false_positives_touching_internal_tile_border": sum(
                bool(item["touches_internal_tile_border"])
                for item in false_positives
            ),
            "true_positives_touching_internal_tile_border": sum(
                bool(item["touches_internal_tile_border"])
                for item in positives
            ),
        }
    false_positive_rows = [
        item for item in decisions if item["outcome"] != "true_positive"
    ]
    samples_by_fp = Counter(item["sample_id"] for item in false_positive_rows)
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pcbparse_yolo_component_validation_error_modes",
        "status": "analysis_complete",
        "selection_only": True,
        "test_split_used": False,
        "test_rows_used": 0,
        "match_iou": 0.5,
        "summary": {
            "ground_truth": sum(len(items) for items in truth.values()),
            "predictions": len(decisions),
            "true_positives": sum(
                item["outcome"] == "true_positive" for item in decisions
            ),
            "false_positives": len(false_positive_rows),
            "false_negatives": len(misses),
            "false_positive_modes": dict(
                sorted(Counter(item["outcome"] for item in false_positive_rows).items())
            ),
            "internal_tile_border_audit": {
                "margin_px": tile_overlap / 2.0,
                "false_positives_touching_border": sum(
                    bool(item["touches_internal_tile_border"])
                    for item in false_positive_rows
                ),
                "true_positives_touching_border": sum(
                    item["outcome"] == "true_positive"
                    and bool(item["touches_internal_tile_border"])
                    for item in decisions
                ),
                "localization_false_positives_touching_border": sum(
                    item["outcome"] == "localization_or_partial_overlap"
                    and bool(item["touches_internal_tile_border"])
                    for item in false_positive_rows
                ),
            },
        },
        "priority_classes": {
            name: per_class[name]
            for name in ("Box", "NET_input", "Other", "C", "Gnd", "R", "PIN")
        },
        "per_class": per_class,
        "highest_false_positive_samples": [
            {"sample_id": sample_id, "false_positives": count}
            for sample_id, count in samples_by_fp.most_common(10)
        ],
        "selection_boundary": boundary,
        "lineage": {
            "predictions": {
                "path": base._relative(predictions_path, workspace),
                "sha256": base._sha256(predictions_path),
            },
            "validation_status": {
                "path": base._relative(status_path, workspace),
                "sha256": base._sha256(status_path),
            },
            "component_coco": {
                "path": base._relative(coco_path, workspace),
                "sha256": base._sha256(coco_path),
            },
        },
        "limitations": [
            "Error-mode labels are diagnostic categories derived from IoU, not human adjudication.",
            "Validation findings may guide a new training run but cannot serve as independent confirmation.",
        ],
    }
    report["analysis_payload_sha256"] = base._canonical_sha256(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base._atomic_write_json(output_path, report)
    print(output_path)
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    print(json.dumps(report["priority_classes"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
