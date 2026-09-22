from __future__ import annotations

"""Select conservative Component score thresholds per class on validation only.

The population floor remains fixed.  Only classes with adequate support are
eligible, recall loss is bounded, and long-tail classes stay protected.  The
result is diagnostic and must be replayed through real tiled inference before
it can be considered a candidate configuration.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import audit_component_stitched_threshold_sweep as sweep  # noqa: E402
import audit_yolo_component_score_threshold as global_audit  # noqa: E402
import evaluate_component_tiled_validation as tiled  # noqa: E402


PROTECTED_CATEGORIES = frozenset({"AMP", "ANT", "Other", "PIN", "Switch"})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
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
        "--source-root", type=Path, default=Path("Dataset300/Dataset300")
    )
    parser.add_argument("--models-lock", type=Path, default=Path("models/models.lock.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-ground-truth-support", type=int, default=50)
    parser.add_argument("--maximum-recall-drop", type=float, default=0.01)
    parser.add_argument("--minimum-f1-gain", type=float, default=0.005)
    parser.add_argument("--minimum-fp-reduction", type=int, default=3)
    return parser


def _combined_metric(
    annotated: list[dict[str, Any]],
    *,
    support: int,
    population_floor: float,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    true_positives = 0
    false_positives = 0
    for detection in annotated:
        threshold = float(
            thresholds.get(str(detection["category"]), population_floor)
        )
        if float(detection["score"]) < threshold:
            continue
        if bool(detection["is_true_positive"]):
            true_positives += 1
        else:
            false_positives += 1
    return sweep._metric_counts(true_positives, false_positives, support)


def _select_for_category(
    *,
    category: str,
    annotated: list[dict[str, Any]],
    truths: list[dict[str, Any]],
    population_floor: float,
    minimum_support: int,
    maximum_recall_drop: float,
    minimum_f1_gain: float,
    minimum_fp_reduction: int,
) -> tuple[dict[str, Any], float | None]:
    baseline = sweep._slice_metric(
        annotated, truths, threshold=population_floor, category=category
    )
    support = int(baseline["ground_truth_count"])
    protected = category in PROTECTED_CATEGORIES
    eligible = support >= minimum_support and not protected
    thresholds = sorted(
        {
            population_floor,
            *(
                float(row["score"])
                for row in annotated
                if str(row["category"]) == category
            ),
        }
    )
    candidates: list[tuple[float, dict[str, Any]]] = []
    if eligible:
        minimum_recall = max(0.0, float(baseline["recall50"]) - maximum_recall_drop)
        for threshold in thresholds:
            metric = sweep._slice_metric(
                annotated, truths, threshold=threshold, category=category
            )
            if float(metric["recall50"]) + 1e-12 >= minimum_recall:
                candidates.append((threshold, metric))
    if candidates:
        best_threshold, best_metric = max(
            candidates,
            key=lambda item: (
                float(item[1]["f1_50"]),
                float(item[1]["precision50"]),
                float(item[1]["recall50"]),
                -float(item[0]),
            ),
        )
    else:
        best_threshold, best_metric = population_floor, baseline
    delta = sweep._metric_delta(best_metric, baseline)
    fp_reduction = int(baseline["false_positives"]) - int(
        best_metric["false_positives"]
    )
    selected = (
        eligible
        and best_threshold > population_floor
        and float(delta["f1_50"]) >= minimum_f1_gain
        and fp_reduction >= minimum_fp_reduction
    )
    reason = "selected"
    if protected:
        reason = "protected_long_tail"
    elif support < minimum_support:
        reason = "insufficient_support"
    elif not selected:
        reason = "gain_or_fp_reduction_below_gate"
    return (
        {
            "ground_truth_support": support,
            "protected": protected,
            "eligible": eligible,
            "population_floor": baseline,
            "best_guarded": {
                "score_threshold": best_threshold,
                "metrics": best_metric,
                "delta": delta,
                "false_positive_reduction": fp_reduction,
            },
            "selected": selected,
            "selection_reason": reason,
            "candidate_threshold_count": len(thresholds),
        },
        best_threshold if selected else None,
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.minimum_ground_truth_support <= 0:
        raise ValueError("minimum ground-truth support must be positive")
    if not 0.0 <= args.maximum_recall_drop <= 1.0:
        raise ValueError("maximum recall drop must be in [0,1]")
    if not 0.0 <= args.minimum_f1_gain <= 1.0:
        raise ValueError("minimum F1 gain must be in [0,1]")
    if args.minimum_fp_reduction <= 0:
        raise ValueError("minimum FP reduction must be positive")

    workspace = Path(args.workspace_root).expanduser().resolve(strict=True)
    resolve = tiled._resolve_within
    status_path = resolve(workspace, args.status, "YOLO tiled status")
    predictions_path = resolve(workspace, args.predictions, "YOLO predictions")
    coco_path = resolve(workspace, args.component_coco, "Component COCO")
    summary_path = resolve(
        workspace, args.materialization_summary, "materialization summary"
    )
    source_root = resolve(
        workspace, args.source_root, "Dataset300 source root", kind="directory"
    )
    lock_path = resolve(workspace, args.models_lock, "models lock")
    output_path = resolve(workspace, args.output, "calibration output", must_exist=False)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite calibration audit: {output_path}")

    status = tiled._read_object(status_path, "YOLO tiled status")
    global_audit._validate_status(
        status,
        status_path=status_path,
        predictions_path=predictions_path,
        workspace=workspace,
    )
    protocol = tiled._mapping(status.get("protocol"), "status protocol")
    population_floor = tiled._finite(protocol.get("score_threshold"), "score floor")
    summary = tiled._read_object(summary_path, "materialization summary")
    binding = tiled._nonempty_string(summary.get("binding_sha256"), "binding")
    samples, categories, boundary = tiled._frozen_validation_samples(
        component_coco=tiled._read_object(coco_path, "Component COCO"),
        source_root=source_root,
        binding_sha256=binding,
        max_val_samples=None,
    )
    prediction_rows = sweep._read_jsonl(predictions_path, "YOLO predictions")
    detections = global_audit._flatten_predictions(
        prediction_rows,
        samples=samples,
        categories=categories,
        population_floor=population_floor,
    )
    truths = [
        {**truth, "sample_id": str(sample["sample_id"])}
        for sample in samples
        for truth in sample["_ground_truth"]
    ]
    annotated = sweep._annotate_matches(
        detections, truths, iou_threshold=float(protocol["match_iou"])
    )
    baseline = _combined_metric(
        annotated,
        support=len(truths),
        population_floor=population_floor,
        thresholds={},
    )
    source_metrics = tiled._mapping(status.get("metrics"), "status metrics")
    for key in (
        "true_positives",
        "false_positives",
        "false_negatives",
        "prediction_count",
        "ground_truth_count",
    ):
        if int(baseline[key]) != int(source_metrics[key]):
            raise ValueError(f"baseline does not reproduce source {key}")

    per_class: dict[str, Any] = {}
    selected_thresholds: dict[str, float] = {}
    for category_row in categories:
        category = str(category_row["name"])
        audit, selected_threshold = _select_for_category(
            category=category,
            annotated=annotated,
            truths=truths,
            population_floor=population_floor,
            minimum_support=args.minimum_ground_truth_support,
            maximum_recall_drop=args.maximum_recall_drop,
            minimum_f1_gain=args.minimum_f1_gain,
            minimum_fp_reduction=args.minimum_fp_reduction,
        )
        per_class[category] = audit
        if selected_threshold is not None:
            selected_thresholds[category] = selected_threshold
    selected_thresholds = dict(sorted(selected_thresholds.items()))
    forecast = _combined_metric(
        annotated,
        support=len(truths),
        population_floor=population_floor,
        thresholds=selected_thresholds,
    )

    lock_before = tiled._sha256(lock_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pcbparse_yolo_component_validation_per_class_threshold_calibration",
        "status": "validation_calibration_complete",
        "task": "component",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "population_floor": population_floor,
        "protected_categories": sorted(PROTECTED_CATEGORIES),
        "selection_gates": {
            "minimum_ground_truth_support": args.minimum_ground_truth_support,
            "maximum_absolute_recall_drop": args.maximum_recall_drop,
            "minimum_absolute_f1_gain": args.minimum_f1_gain,
            "minimum_false_positive_reduction": args.minimum_fp_reduction,
        },
        "selected_thresholds": selected_thresholds,
        "population_floor_metrics": baseline,
        "post_cap_forecast_metrics": forecast,
        "post_cap_forecast_delta": sweep._metric_delta(forecast, baseline),
        "per_class": per_class,
        "selection_boundary": {
            **boundary,
            "validation_selected_and_evaluated_on_same_split": True,
            "model_loaded": False,
            "inference_executed": False,
        },
        "sources": {
            "tiled_status": {
                "path": tiled._relative(status_path, workspace),
                "sha256": tiled._sha256(status_path),
            },
            "predictions": {
                "path": tiled._relative(predictions_path, workspace),
                "sha256": tiled._sha256(predictions_path),
            },
            "models_lock": {
                "path": tiled._relative(lock_path, workspace),
                "sha256_before": lock_before,
            },
        },
        "interpretation_boundary": {
            "test_split_used": False,
            "thresholds_written_to_production": False,
            "models_lock_modified": False,
            "real_tiled_replay_required": True,
            "independent_confirmation_required": True,
        },
        "limitations": [
            "Thresholds are selected and forecast on the same validation split.",
            "The forecast filters already capped predictions; a real tiled replay may refill capacity differently.",
            "Protected and low-support classes retain the shared population floor.",
        ],
    }
    lock_after = tiled._sha256(lock_path)
    if lock_before != lock_after:
        raise RuntimeError("models.lock.json changed during calibration")
    report["sources"]["models_lock"]["sha256_after"] = lock_after
    report["calibration_payload_sha256"] = tiled._canonical_sha256(report)
    tiled._atomic_write_json(output_path, report)
    print(output_path)
    print(json.dumps(selected_thresholds, ensure_ascii=False, sort_keys=True))
    print(json.dumps(report["post_cap_forecast_metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
