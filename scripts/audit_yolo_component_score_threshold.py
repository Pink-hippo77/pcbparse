from __future__ import annotations

"""Validation-only score calibration for stitched YOLO Component predictions.

The audit reuses saved predictions and never loads the model.  Candidate
thresholds are exact scores already present at the fixed 0.20 population
floor.  The selected threshold is validation-biased and cannot promote a model
without an independent confirmation run.
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
import evaluate_component_tiled_validation as tiled  # noqa: E402


EXPECTED_KIND = "pcbparse_yolo_component_tiled_frozen_val_diagnostic"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument(
        "--status",
        type=Path,
        default=Path(
            "work/eval-component-yolo11n-v1-tiled512-val-full-20260914/"
            "diagnostic_status.json"
        ),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "work/eval-component-yolo11n-v1-tiled512-val-full-20260914/"
            "predictions.jsonl"
        ),
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
        "--source-root", type=Path, default=Path("Dataset300/Dataset300")
    )
    parser.add_argument(
        "--models-lock", type=Path, default=Path("models/models.lock.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/component_yolo11n_v1_score_calibration.json"),
    )
    return parser


def _close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _validate_status(
    status: Mapping[str, Any],
    *,
    status_path: Path,
    predictions_path: Path,
    workspace: Path,
) -> None:
    payload = dict(status)
    claimed = payload.pop("diagnostic_payload_sha256", None)
    if claimed != tiled._canonical_sha256(payload):
        raise ValueError("YOLO tiled status canonical payload hash mismatch")
    expected = {
        "schema_version": 1,
        "kind": EXPECTED_KIND,
        "task": "component",
        "status": "diagnostic_evaluation_complete",
        "smoke": False,
        "registerable": False,
        "formal_evidence": False,
        "promotion_eligible": False,
        "blessing_authorized": False,
    }
    for key, value in expected.items():
        if status.get(key) != value:
            raise ValueError(f"YOLO tiled status mismatch for {key}")
    artifacts = tiled._mapping(status.get("artifacts"), "status.artifacts")
    prediction = tiled._mapping(artifacts.get("predictions"), "status predictions")
    expected_path = tiled._relative(predictions_path, workspace)
    if prediction.get("path") != expected_path:
        raise ValueError("prediction path differs from status")
    if prediction.get("sha256") != tiled._sha256(predictions_path):
        raise ValueError("prediction bytes differ from status")
    if status_path.stat().st_size <= 0:
        raise ValueError("YOLO tiled status is empty")


def _flatten_predictions(
    rows: list[dict[str, Any]],
    *,
    samples: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    population_floor: float,
) -> list[dict[str, Any]]:
    expected_ids = [str(sample["sample_id"]) for sample in samples]
    if [str(row.get("sample_id")) for row in rows] != expected_ids:
        raise ValueError("prediction population/order differs from frozen validation")
    category_names = {str(row["name"]) for row in categories}
    detections: list[dict[str, Any]] = []
    for row, sample in zip(rows, samples):
        sample_id = str(sample["sample_id"])
        for prediction_index, raw in enumerate(
            tiled._sequence(row.get("predictions"), "sample predictions")
        ):
            item = tiled._mapping(raw, "sample prediction")
            category = str(item.get("category"))
            if category not in category_names:
                raise ValueError(f"unknown prediction category {category!r}")
            score = tiled._finite(item.get("score"), "prediction score")
            if not population_floor <= score <= 1.0:
                raise ValueError("prediction score falls outside saved population")
            box = tiled._bbox(item.get("box"), "prediction box")
            detections.append(
                {
                    "sample_id": sample_id,
                    "category": category,
                    "score": score,
                    "box": list(box),
                    "prediction_index": prediction_index,
                    "source_view_id": str(item.get("source_view_id", "")),
                }
            )
    return detections


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
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
    output_path = resolve(
        workspace, args.output, "calibration output", must_exist=False
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite calibration audit: {output_path}")

    status = tiled._read_object(status_path, "YOLO tiled status")
    _validate_status(
        status,
        status_path=status_path,
        predictions_path=predictions_path,
        workspace=workspace,
    )
    protocol = tiled._mapping(status.get("protocol"), "status.protocol")
    population_floor = tiled._finite(protocol.get("score_threshold"), "score floor")
    summary = tiled._read_object(summary_path, "materialization summary")
    binding = tiled._nonempty_string(
        summary.get("binding_sha256"), "materialization binding"
    )
    samples, categories, boundary = tiled._frozen_validation_samples(
        component_coco=tiled._read_object(coco_path, "Component COCO"),
        source_root=source_root,
        binding_sha256=binding,
        max_val_samples=None,
    )
    rows = sweep._read_jsonl(predictions_path, "YOLO predictions")
    detections = _flatten_predictions(
        rows,
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
        detections,
        truths,
        iou_threshold=float(protocol["match_iou"]),
    )
    prefix = sweep._build_prefix(annotated, len(truths))
    thresholds = sorted({float(row["score"]) for row in detections})
    scan = [
        {
            "score_threshold": threshold,
            "score_boundary_origins": ["yolo11n-v1"],
            "metrics": {"yolo11n-v1": sweep._state_at(prefix, threshold)},
        }
        for threshold in thresholds
    ]
    best_row = sweep._best_row(scan, "yolo11n-v1")
    best_threshold = float(best_row["score_threshold"])
    floor_metrics = sweep._state_at(prefix, population_floor)
    source_metrics = tiled._mapping(status.get("metrics"), "status.metrics")
    for key in (
        "true_positives",
        "false_positives",
        "false_negatives",
        "prediction_count",
        "ground_truth_count",
    ):
        if int(source_metrics[key]) != int(floor_metrics[key]):
            raise ValueError(f"calibration does not reproduce source {key}")
    for key in ("precision50", "recall50", "f1_50"):
        if not _close(source_metrics[key], floor_metrics[key]):
            raise ValueError(f"calibration does not reproduce source {key}")

    per_class: dict[str, Any] = {}
    for category in categories:
        name = str(category["name"])
        per_class[name] = {
            "ground_truth_support": sum(
                1 for truth in truths if str(truth["category"]) == name
            ),
            "population_floor": sweep._slice_metric(
                annotated, truths, threshold=population_floor, category=name
            ),
            "f1_selected": sweep._slice_metric(
                annotated, truths, threshold=best_threshold, category=name
            ),
        }

    lock_sha256_before = tiled._sha256(lock_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pcbparse_yolo_component_validation_score_calibration",
        "status": "validation_calibration_complete",
        "task": "component",
        "diagnostic_only": True,
        "validation_calibration_only": True,
        "promotion_eligible": False,
        "registerable": False,
        "population_floor": {
            "score_threshold": population_floor,
            "metrics": floor_metrics,
        },
        "f1_selected": {
            "score_threshold": best_threshold,
            "metrics": best_row["metrics"]["yolo11n-v1"],
            "delta_from_population_floor": sweep._metric_delta(
                best_row["metrics"]["yolo11n-v1"], floor_metrics
            ),
        },
        "per_class": per_class,
        "threshold_scan": scan,
        "selection_boundary": {
            **boundary,
            "candidate_threshold_count": len(thresholds),
            "candidate_thresholds_sha256": tiled._canonical_sha256(thresholds),
            "thresholds_are_observed_prediction_scores": True,
            "threshold_selected_and_evaluated_on_same_validation_split": True,
            "selection_bias_present": True,
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
            "component_coco": {
                "path": tiled._relative(coco_path, workspace),
                "sha256": tiled._sha256(coco_path),
                "binding_sha256": binding,
            },
            "materialization_summary": {
                "path": tiled._relative(summary_path, workspace),
                "sha256": tiled._sha256(summary_path),
            },
            "models_lock": {
                "path": tiled._relative(lock_path, workspace),
                "sha256_before": lock_sha256_before,
            },
        },
        "interpretation_boundary": {
            "test_split_used": False,
            "model_loaded": False,
            "inference_executed": False,
            "threshold_written_to_config": False,
            "models_lock_modified": False,
            "independent_confirmation_required": True,
        },
        "limitations": [
            "Predictions below 0.20 were not saved and cannot be evaluated.",
            "The selected threshold is optimistically biased because selection and scoring use the same validation split.",
            "The sweep occurs after fixed cross-tile NMS and cannot tune NMS or tiling geometry.",
        ],
    }
    lock_sha256_after = tiled._sha256(lock_path)
    if lock_sha256_before != lock_sha256_after:
        raise RuntimeError("models.lock.json changed during calibration")
    report["sources"]["models_lock"]["sha256_after"] = lock_sha256_after
    report["calibration_payload_sha256"] = tiled._canonical_sha256(report)
    tiled._atomic_write_json(output_path, report)
    print(output_path)
    print(json.dumps(report["population_floor"], sort_keys=True))
    print(json.dumps(report["f1_selected"], sort_keys=True))
    print(f"calibration_payload_sha256={report['calibration_payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
