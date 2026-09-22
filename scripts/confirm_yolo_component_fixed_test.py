from __future__ import annotations

"""Evaluate a frozen YOLO Component candidate once on the independent test split.

The candidate contract fixes every inference parameter and decision gate before
test annotations are read.  This program performs no score, class-threshold,
tiling, or NMS search and never modifies ``models/models.lock.json``.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import confirm_component_calibrated_test as frozen_test  # noqa: E402
import evaluate_component_tiled_validation as base  # noqa: E402
import evaluate_yolo_component_tiled_validation as yolo_eval  # noqa: E402
from pcbparse.perception.yolo_component_adapter import factory  # noqa: E402


CANDIDATE_KIND = "pcbparse_yolo_component_fixed_test_candidate"
CONFIRMATION_KIND = "pcbparse_yolo_component_fixed_frozen_test_confirmation"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument(
        "--candidate-contract",
        type=Path,
        default=Path("configs/candidates/yolo_component_y11n_v1_fixed020_test.json"),
    )
    parser.add_argument(
        "--source-root", type=Path, default=Path("Dataset300/Dataset300")
    )
    parser.add_argument(
        "--models-lock", type=Path, default=Path("models/models.lock.json")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work/confirm-component-yolo11n-v1-fixed020-test"),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def _verify_entry(
    workspace: Path, raw: Any, label: str
) -> tuple[Path, dict[str, str]]:
    entry = base._mapping(raw, label)
    path = base._resolve_within(
        workspace,
        base._nonempty_string(entry.get("path"), f"{label}.path"),
        label,
    )
    expected = base._nonempty_string(entry.get("sha256"), f"{label}.sha256")
    actual = base._sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return path, {"path": base._relative(path, workspace), "sha256": actual}


def _fixed_protocol(
    candidate: Mapping[str, Any],
    pipeline: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    declared = base._mapping(candidate.get("protocol"), "candidate protocol")
    if declared.get("split") != "test":
        raise ValueError("candidate protocol is not frozen for the test split")
    derived = base._fixed_protocol(pipeline, thresholds, summary)
    expected = {
        "tile_size": int(derived["tile_size"]),
        "tile_overlap": int(derived["tile_overlap"]),
        "score_threshold": float(derived["score_threshold"]),
        "component_nms_iou": float(derived["component_nms_iou"]),
        "cross_tile_nms_iou": float(derived["component_nms_iou"]),
        "max_detections_per_tile": int(derived["max_detections_per_tile"]),
        "component_max_candidates": int(derived["component_max_candidates"]),
        "match_iou": float(derived["match_iou"]),
        "model_imgsz": 640,
    }
    for key, value in expected.items():
        if declared.get(key) != value:
            raise ValueError(
                f"candidate protocol {key} differs from the fixed replay: "
                f"{declared.get(key)!r} != {value!r}"
            )
    return {**derived, "model_imgsz": int(expected["model_imgsz"]), "split": "test"}


def _decision_gates(
    raw_gates: Any, metrics: Mapping[str, Any]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for raw in base._sequence(raw_gates, "independent test decision gates"):
        gate = base._mapping(raw, "independent test decision gate")
        metric = base._nonempty_string(gate.get("metric"), "gate metric")
        operator = base._nonempty_string(gate.get("operator"), "gate operator")
        if operator not in {"gte", "lte"}:
            raise ValueError(f"unsupported gate operator {operator!r}")
        threshold = base._finite(gate.get("threshold"), "gate threshold")
        value = base._finite(metrics.get(metric), f"metric {metric}")
        passed = value >= threshold if operator == "gte" else value <= threshold
        results.append(
            {
                "metric": metric,
                "operator": operator,
                "threshold": threshold,
                "value": value,
                "passed": passed,
                "basis": str(gate.get("basis", "predeclared candidate gate")),
            }
        )
    if not results:
        raise ValueError("candidate has no independent test decision gates")
    return results


def _metric_delta(
    metrics: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, float]:
    keys = ("bbox_map50", "precision50", "recall50", "f1_50")
    return {
        key: base._finite(metrics.get(key), f"candidate {key}")
        - base._finite(baseline.get(key), f"baseline {key}")
        for key in keys
    }


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    workspace = Path(arguments.workspace_root).expanduser().resolve(strict=True)
    candidate_path = base._resolve_within(
        workspace, arguments.candidate_contract, "candidate contract"
    )
    source_root = base._resolve_within(
        workspace, arguments.source_root, "Dataset300 source root", kind="directory"
    )
    models_lock_path = base._resolve_within(
        workspace, arguments.models_lock, "models lock"
    )
    output_dir = base._resolve_within(
        workspace,
        arguments.output_dir,
        "confirmation output directory",
        kind="directory",
        must_exist=False,
    )
    base._prepare_output(output_dir, workspace)

    lock_sha_before = base._sha256(models_lock_path)
    candidate = base._read_object(candidate_path, "YOLO candidate contract")
    if candidate.get("kind") != CANDIDATE_KIND or candidate.get("status") != "frozen_before_test":
        raise ValueError("candidate contract kind/status is not frozen")
    policy = base._mapping(candidate.get("test_policy"), "candidate test policy")
    required_false = (
        "threshold_search_on_test",
        "class_threshold_search_on_test",
        "test_sample_selection_by_label",
        "promotion_authorized",
        "models_lock_mutation_authorized",
    )
    if any(policy.get(key) is not False for key in required_false):
        raise ValueError("candidate test policy permits selection or mutation")

    checkpoint_path, checkpoint_source = _verify_entry(
        workspace, candidate.get("checkpoint"), "checkpoint"
    )
    category_path, category_source = _verify_entry(
        workspace, candidate.get("category_table"), "category table"
    )
    bound = base._mapping(candidate.get("bound_inputs"), "bound inputs")
    verified: dict[str, dict[str, str]] = {}
    resolved: dict[str, Path] = {}
    for name in (
        "component_coco",
        "materialization_summary",
        "pipeline_config",
        "thresholds_config",
        "training_config",
        "training_results",
        "training_log",
    ):
        resolved[name], verified[name] = _verify_entry(
            workspace, bound.get(name), name.replace("_", " ")
        )
    baseline_path, baseline_source = _verify_entry(
        workspace, candidate.get("baseline"), "baseline status"
    )

    summary = base._read_object(
        resolved["materialization_summary"], "materialization summary"
    )
    binding_sha256 = base._nonempty_string(
        summary.get("binding_sha256"), "materialization binding_sha256"
    )
    protocol = _fixed_protocol(
        candidate,
        base._read_object(resolved["pipeline_config"], "pipeline config"),
        base._read_object(resolved["thresholds_config"], "thresholds config"),
        summary,
    )
    component_coco = base._read_object(resolved["component_coco"], "Component COCO")
    samples, index_categories, selection = frozen_test._frozen_test_samples(
        component_coco=component_coco,
        source_root=source_root,
        binding_sha256=binding_sha256,
    )
    category_payload = base._read_object(category_path, "YOLO category table")
    yolo_categories = yolo_eval._category_rows(category_payload)
    yolo_eval._verify_category_binding(yolo_categories, index_categories)
    training_completion = yolo_eval._training_completion(
        resolved["training_results"], resolved["training_log"]
    )

    baseline_status = base._read_object(baseline_path, "baseline test status")
    baseline_selection = base._mapping(
        baseline_status.get("selection_boundary"), "baseline selection boundary"
    )
    if (
        baseline_selection.get("split") != "test"
        or baseline_selection.get("selected_sample_ids_sha256")
        != selection["selected_sample_ids_sha256"]
        or baseline_selection.get("selected_test_annotations")
        != selection["selected_test_annotations"]
    ):
        raise ValueError("candidate and baseline do not use the identical frozen test population")

    adapter = factory(
        checkpoint_path,
        {
            "task": "component",
            "sha256": checkpoint_source["sha256"],
            "category_table_path": str(category_path),
            "category_table_sha256": category_source["sha256"],
            "inference": {
                "device": str(arguments.device),
                "score_threshold": float(protocol["score_threshold"]),
                "nms_iou": float(protocol["component_nms_iou"]),
                "max_detections": int(protocol["max_detections_per_tile"]),
                "imgsz": int(protocol["model_imgsz"]),
            },
        },
    )
    predictions_path = output_dir / "predictions.jsonl"
    prediction_rows, performance = base._run_inference(
        samples=samples,
        adapter=adapter,
        protocol=protocol,
        predictions_path=predictions_path,
    )
    ground_truth_by_sample = {
        str(item["sample_id"]): list(item["_ground_truth"]) for item in samples
    }
    metrics = base._evaluate_predictions(
        prediction_rows,
        ground_truth_by_sample,
        index_categories,
        iou_threshold=float(protocol["match_iou"]),
    )
    baseline_metrics = dict(
        base._mapping(baseline_status.get("metrics"), "baseline test metrics")
    )
    gates = _decision_gates(candidate.get("independent_test_decision_gates"), metrics)
    gates_passed = all(item["passed"] for item in gates)

    lock_sha_after = base._sha256(models_lock_path)
    if lock_sha_before != lock_sha_after:
        raise RuntimeError("models.lock.json changed during frozen test confirmation")
    status: dict[str, Any] = {
        "schema_version": 1,
        "kind": CONFIRMATION_KIND,
        "task": "component",
        "status": "frozen_test_confirmation_complete",
        "formal_evidence": True,
        "gates_passed": gates_passed,
        "promotion_eligible": False,
        "blessing_authorized": False,
        "registerable": False,
        "finished_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "device": str(arguments.device),
        "adapter_version": adapter.version,
        "candidate_contract": {
            "path": base._relative(candidate_path, workspace),
            "sha256": base._sha256(candidate_path),
            "status_before_test": "frozen_before_test",
        },
        "protocol": protocol,
        "selection_boundary": selection,
        "metrics": metrics,
        "baseline": {
            **baseline_source,
            "name": str(base._mapping(candidate.get("baseline"), "baseline").get("name")),
            "metrics": baseline_metrics,
        },
        "delta_candidate_minus_baseline": _metric_delta(metrics, baseline_metrics),
        "decision_gates": gates,
        "checks": {
            "all_predeclared_gates_passed": gates_passed,
            "candidate_frozen_before_test": True,
            "identical_frozen_test_population_as_baseline": True,
            "test_threshold_search_count": 0,
            "test_class_threshold_search_count": 0,
            "test_label_dependent_sample_selection": False,
            "models_lock_unchanged": True,
        },
        "performance": performance,
        "training_completion": training_completion,
        "artifacts": {
            "predictions": {
                "path": base._relative(predictions_path, workspace),
                "sha256": base._sha256(predictions_path),
                "row_count": len(prediction_rows),
                "prediction_count": sum(
                    len(base._sequence(row.get("predictions"), "predictions"))
                    for row in prediction_rows
                ),
            }
        },
        "lineage": {
            "checkpoint": checkpoint_source,
            "category_table": category_source,
            "bound_inputs": verified,
            "materialization_binding_sha256": binding_sha256,
            "models_lock": {
                "path": base._relative(models_lock_path, workspace),
                "sha256_before": lock_sha_before,
                "sha256_after": lock_sha_after,
                "modified": False,
            },
        },
        "interpretation_boundary": {
            "independent_test_confirmation": True,
            "test_configuration_tuned": False,
            "test_split_used_for_selection": False,
            "direct_promotion_allowed": False,
            "models_lock_modified": False,
        },
        "limitations": [
            "The 30-image test split is an internal frozen Dataset300 holdout, not the organizer hidden set.",
            "This confirmation tests only the fixed global 0.20 score floor; the validation-selected 0.62955 threshold is not tested here.",
            "Passing evidence supports candidate packaging and end-to-end shadow evaluation, not automatic production promotion.",
        ],
    }
    status["confirmation_payload_sha256"] = base._canonical_sha256(status)
    status_path = output_dir / "fixed_test_status.json"
    base._atomic_write_json(status_path, status)
    print(status_path)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    print(f"gates_passed={json.dumps(gates_passed)}")
    print(f"confirmation_payload_sha256={status['confirmation_payload_sha256']}")
    return 0 if gates_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
