from __future__ import annotations

"""Run one hash-bound paired confirmation of Component border refinement.

The candidate contract, pipeline configurations, per-class thresholds, model,
test population, and decision gates are frozen before test labels are scored.
The program runs one baseline/candidate pair and never mutates models.lock.json.
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

import confirm_component_calibrated_test as calibrated  # noqa: E402
import evaluate_component_tiled_validation as base  # noqa: E402
import evaluate_yolo_component_tiled_validation as yolo_eval  # noqa: E402
from pcbparse.perception.yolo_component_adapter import factory  # noqa: E402


CONTRACT_KIND = "pcbparse_yolo_component_border_refinement_paired_test_candidate"
RESULT_KIND = "pcbparse_yolo_component_border_refinement_paired_test_confirmation"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument(
        "--candidate-contract",
        type=Path,
        default=Path(
            "configs/candidates/"
            "yolo_component_y11n_v2_border_refinement_test.json"
        ),
    )
    parser.add_argument(
        "--source-root", type=Path, default=Path("Dataset300/Dataset300")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "work/confirm-component-yolo11n-v2-border-refinement-test-20260916"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _verify_bound_inputs(
    workspace: Path, contract: Mapping[str, Any]
) -> dict[str, Path]:
    bound = base._mapping(contract.get("bound_inputs"), "contract.bound_inputs")
    paths: dict[str, Path] = {}
    for name, raw in bound.items():
        entry = base._mapping(raw, f"bound input {name}")
        path = base._resolve_within(
            workspace,
            base._nonempty_string(entry.get("path"), f"bound input {name}.path"),
            f"bound input {name}",
        )
        expected = base._nonempty_string(
            entry.get("sha256"), f"bound input {name}.sha256"
        ).lower()
        actual = base._sha256(path)
        if actual != expected:
            raise ValueError(f"bound input {name} SHA-256 mismatch")
        paths[str(name)] = path
    required = {
        "checkpoint",
        "category_table",
        "component_coco",
        "materialization_summary",
        "baseline_pipeline_config",
        "candidate_pipeline_config",
        "thresholds_config",
        "training_config",
        "training_results",
        "training_status",
        "per_class_threshold_calibration",
        "validation_refinement_ab",
        "models_lock",
    }
    if set(paths) != required:
        raise ValueError(
            "candidate contract bound input set differs from the paired-test contract"
        )
    return paths


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("kind") != CONTRACT_KIND
        or contract.get("status") != "frozen_before_test"
        or contract.get("task") != "component"
    ):
        raise ValueError("candidate contract identity is not frozen and final")
    policy = base._mapping(contract.get("test_policy"), "contract.test_policy")
    expected_false = {
        "threshold_search_on_test",
        "class_threshold_search_on_test",
        "geometry_search_on_test",
        "test_sample_selection_by_label",
        "post_test_parameter_tuning_allowed",
        "promotion_authorized",
        "models_lock_mutation_authorized",
    }
    if set(policy) != expected_false or any(policy[key] is not False for key in expected_false):
        raise ValueError("candidate contract does not prohibit test tuning and mutation")


def _validate_selection_evidence(
    calibration: Mapping[str, Any], validation_ab: Mapping[str, Any]
) -> None:
    selection = base._mapping(
        calibration.get("selection_boundary"), "calibration.selection_boundary"
    )
    interpretation = base._mapping(
        calibration.get("interpretation_boundary"),
        "calibration.interpretation_boundary",
    )
    if (
        calibration.get("kind")
        != "pcbparse_yolo_component_validation_per_class_threshold_calibration"
        or calibration.get("status") != "validation_calibration_complete"
        or selection.get("split") != "val"
        or selection.get("test_samples_loaded") != 0
        or selection.get("test_images_decoded") != 0
        or selection.get("test_annotations_scored") != 0
        or interpretation.get("test_split_used") is not False
    ):
        raise ValueError("per-class threshold selection is not validation-only")
    ab_boundary = base._mapping(
        validation_ab.get("interpretation_boundary"),
        "validation refinement interpretation boundary",
    )
    if (
        validation_ab.get("kind")
        != "component_box_internal_border_refinement_validation_ab"
        or validation_ab.get("status") != "validation_candidate_not_promoted"
        or ab_boundary.get("independent_test_confirmation_completed") is not False
        or ab_boundary.get("production_promotion_allowed") is not False
        or ab_boundary.get("models_lock_modified") is not False
    ):
        raise ValueError("border-refinement selection evidence is not validation-only")


def _validate_protocol_pair(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    declared: Mapping[str, Any],
) -> None:
    required_equal = {
        "tile_size",
        "tile_overlap",
        "tile_stride",
        "score_threshold",
        "component_nms_iou",
        "component_max_candidates",
        "component_internal_border_margin_px",
        "component_internal_border_confirmation_enabled",
        "component_internal_border_confirmation_iou",
        "component_internal_border_confirmation_metric",
        "component_internal_border_confirmation_categories",
        "match_iou",
        "max_detections_per_tile",
    }
    for key in required_equal:
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"baseline/candidate common protocol differs at {key}")
    if (
        baseline.get("component_internal_border_refinement_enabled") is not False
        or baseline.get("component_center_preferred_nms_enabled") is not False
        or candidate.get("component_internal_border_refinement_enabled") is not True
        or candidate.get("component_internal_border_refinement_categories") != ["Box"]
        or candidate.get("component_center_preferred_nms_enabled") is not True
    ):
        raise ValueError("baseline/candidate border-refinement roles drifted")
    expected = {
        "tile_size": baseline["tile_size"],
        "tile_overlap": baseline["tile_overlap"],
        "score_population_floor": baseline["score_threshold"],
        "component_nms_iou": baseline["component_nms_iou"],
        "component_max_candidates": baseline["component_max_candidates"],
        "max_detections_per_tile": baseline["max_detections_per_tile"],
        "match_iou": baseline["match_iou"],
        "model_imgsz": 640,
        "refinement_category": "Box",
    }
    if dict(declared) != expected:
        raise ValueError("candidate contract protocol differs from bound configurations")


def _paired_comparison(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    numeric = (
        "bbox_map50",
        "precision50",
        "recall50",
        "f1_50",
        "true_positives",
        "false_positives",
        "false_negatives",
        "prediction_count",
    )
    delta = {
        key: float(candidate[key]) - float(baseline[key]) for key in numeric
    }
    baseline_fp = int(baseline["false_positives"])
    if baseline_fp <= 0:
        raise ValueError("overall false-positive reduction ratio has zero denominator")
    baseline_box = base._mapping(
        base._mapping(baseline.get("per_class"), "baseline.per_class").get("Box"),
        "baseline Box metrics",
    )
    candidate_box = base._mapping(
        base._mapping(candidate.get("per_class"), "candidate.per_class").get("Box"),
        "candidate Box metrics",
    )
    baseline_box_fp = int(baseline_box["false_positives"])
    if baseline_box_fp <= 0:
        raise ValueError("Box false-positive reduction ratio has zero denominator")
    return {
        "delta": delta,
        "false_positive_reduction": baseline_fp - int(candidate["false_positives"]),
        "false_positive_reduction_ratio": (
            baseline_fp - int(candidate["false_positives"])
        )
        / baseline_fp,
        "box": {
            "baseline": dict(baseline_box),
            "candidate": dict(candidate_box),
            "delta_recall50": float(candidate_box["recall50"])
            - float(baseline_box["recall50"]),
            "false_positive_reduction": baseline_box_fp
            - int(candidate_box["false_positives"]),
            "false_positive_reduction_ratio": (
                baseline_box_fp - int(candidate_box["false_positives"])
            )
            / baseline_box_fp,
        },
    }


def _lookup_metric(comparison: Mapping[str, Any], dotted: str) -> float:
    current: Any = comparison
    for part in dotted.split("."):
        current = base._mapping(current, dotted).get(part)
    return base._finite(current, f"comparison metric {dotted}")


def _evaluate_gates(
    comparison: Mapping[str, Any], gates: Sequence[Any]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, raw in enumerate(gates):
        gate = base._mapping(raw, f"decision gate {index}")
        metric = base._nonempty_string(gate.get("metric"), "gate metric")
        operator = base._nonempty_string(gate.get("operator"), "gate operator")
        threshold = base._finite(gate.get("threshold"), "gate threshold")
        observed = _lookup_metric(comparison, metric)
        if operator == "gte":
            passed = observed >= threshold
        elif operator == "lte":
            passed = observed <= threshold
        else:
            raise ValueError(f"unsupported gate operator {operator!r}")
        results.append(
            {
                "metric": metric,
                "operator": operator,
                "threshold": threshold,
                "observed": observed,
                "passed": passed,
                "basis": gate.get("basis"),
            }
        )
    if not results:
        raise ValueError("candidate contract has no independent test gates")
    return results


def _lineage_entry(path: Path, workspace: Path) -> dict[str, str]:
    return {"path": base._relative(path, workspace), "sha256": base._sha256(path)}


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    workspace = Path(arguments.workspace_root).expanduser().resolve(strict=True)
    contract_path = base._resolve_within(
        workspace, arguments.candidate_contract, "candidate contract"
    )
    source_root = base._resolve_within(
        workspace, arguments.source_root, "Dataset300 source root", kind="directory"
    )
    output_dir = base._resolve_within(
        workspace,
        arguments.output_dir,
        "paired test output directory",
        kind="directory",
        must_exist=False,
    )

    contract = base._read_object(contract_path, "candidate contract")
    _validate_contract(contract)
    paths = _verify_bound_inputs(workspace, contract)
    base._prepare_output(output_dir, workspace)
    lock_before = base._sha256(paths["models_lock"])

    summary = base._read_object(
        paths["materialization_summary"], "materialization summary"
    )
    binding_sha256 = base._nonempty_string(
        summary.get("binding_sha256"), "materialization binding_sha256"
    )
    thresholds = base._read_object(paths["thresholds_config"], "thresholds config")
    baseline_protocol = base._fixed_protocol(
        base._read_object(
            paths["baseline_pipeline_config"], "baseline pipeline config"
        ),
        thresholds,
        summary,
    )
    candidate_protocol = base._fixed_protocol(
        base._read_object(
            paths["candidate_pipeline_config"], "candidate pipeline config"
        ),
        thresholds,
        summary,
    )
    _validate_protocol_pair(
        baseline_protocol,
        candidate_protocol,
        base._mapping(contract.get("protocol"), "contract.protocol"),
    )

    calibration_audit = base._read_object(
        paths["per_class_threshold_calibration"], "per-class threshold calibration"
    )
    validation_ab = base._read_object(
        paths["validation_refinement_ab"], "validation refinement A/B"
    )
    _validate_selection_evidence(calibration_audit, validation_ab)

    component_coco = base._read_object(paths["component_coco"], "Component COCO")
    samples, index_categories, selection_boundary = calibrated._frozen_test_samples(
        component_coco=component_coco,
        source_root=source_root,
        binding_sha256=binding_sha256,
    )
    population = base._mapping(contract.get("test_population"), "test population")
    if (
        population.get("split") != "test"
        or len(samples) != population.get("full_image_count")
        or selection_boundary.get("selected_test_annotations")
        != population.get("component_annotation_count")
    ):
        raise ValueError("frozen test population differs from the predeclared contract")

    category_payload = base._read_object(paths["category_table"], "category table")
    yolo_categories = yolo_eval._category_rows(category_payload)
    yolo_eval._verify_category_binding(yolo_categories, index_categories)
    completion = yolo_eval._training_completion(
        paths["training_results"],
        None,
        status_path=paths["training_status"],
        checkpoint_path=paths["checkpoint"],
        training_config_path=paths["training_config"],
    )

    checkpoint_sha256 = base._sha256(paths["checkpoint"])
    per_class_thresholds, verified_calibration = (
        yolo_eval._per_class_thresholds_from_audit(
            paths["per_class_threshold_calibration"],
            workspace=workspace,
            checkpoint_sha256=checkpoint_sha256,
            population_floor=float(baseline_protocol["score_threshold"]),
        )
    )
    adapter = factory(
        paths["checkpoint"],
        {
            "task": "component",
            "sha256": checkpoint_sha256,
            "category_table_path": str(paths["category_table"]),
            "category_table_sha256": base._sha256(paths["category_table"]),
            "inference": {
                "device": str(arguments.device),
                "score_threshold": float(baseline_protocol["score_threshold"]),
                "nms_iou": float(baseline_protocol["component_nms_iou"]),
                "max_detections": int(
                    baseline_protocol["max_detections_per_tile"]
                ),
                "imgsz": 640,
                "per_class_score_thresholds": per_class_thresholds,
            },
        },
    )

    baseline_predictions_path = output_dir / "baseline_predictions.jsonl"
    candidate_predictions_path = output_dir / "candidate_predictions.jsonl"
    baseline_rows, baseline_performance = base._run_inference(
        samples=samples,
        adapter=adapter,
        protocol=baseline_protocol,
        predictions_path=baseline_predictions_path,
    )
    candidate_rows, candidate_performance = base._run_inference(
        samples=samples,
        adapter=adapter,
        protocol=candidate_protocol,
        predictions_path=candidate_predictions_path,
    )
    ground_truth = {
        str(item["sample_id"]): list(item["_ground_truth"]) for item in samples
    }
    baseline_metrics = base._evaluate_predictions(
        baseline_rows,
        ground_truth,
        index_categories,
        iou_threshold=float(baseline_protocol["match_iou"]),
    )
    candidate_metrics = base._evaluate_predictions(
        candidate_rows,
        ground_truth,
        index_categories,
        iou_threshold=float(candidate_protocol["match_iou"]),
    )
    comparison = _paired_comparison(baseline_metrics, candidate_metrics)
    gate_results = _evaluate_gates(
        comparison,
        base._sequence(
            contract.get("independent_test_decision_gates"),
            "independent test decision gates",
        ),
    )
    all_gates_passed = all(item["passed"] for item in gate_results)

    lock_after = base._sha256(paths["models_lock"])
    if lock_before != lock_after:
        raise RuntimeError("models.lock.json changed during paired test")
    status: dict[str, Any] = {
        "schema_version": 1,
        "kind": RESULT_KIND,
        "task": "component",
        "status": "paired_test_passed" if all_gates_passed else "paired_test_failed",
        "finished_at_utc": _utc_now(),
        "candidate_name": contract["candidate_name"],
        "all_predeclared_gates_passed": all_gates_passed,
        "promotion_eligible": False,
        "promotion_authorized": False,
        "device": str(arguments.device),
        "adapter_version": adapter.version,
        "selection_boundary": selection_boundary,
        "protocol": {
            "baseline": baseline_protocol,
            "candidate": candidate_protocol,
            "model_imgsz": 640,
            "per_class_score_thresholds": per_class_thresholds,
        },
        "metrics": {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "comparison": comparison,
        },
        "decision_gates": gate_results,
        "performance": {
            "baseline": baseline_performance,
            "candidate": candidate_performance,
        },
        "training_completion": completion,
        "artifacts": {
            "baseline_predictions": {
                **_lineage_entry(baseline_predictions_path, workspace),
                "row_count": len(baseline_rows),
            },
            "candidate_predictions": {
                **_lineage_entry(candidate_predictions_path, workspace),
                "row_count": len(candidate_rows),
            },
        },
        "lineage": {
            "candidate_contract": _lineage_entry(contract_path, workspace),
            **{
                name: _lineage_entry(path, workspace)
                for name, path in paths.items()
                if name != "models_lock"
            },
            "models_lock": {
                "path": base._relative(paths["models_lock"], workspace),
                "sha256_before": lock_before,
                "sha256_after": lock_after,
                "modified": False,
            },
            "calibration_payload_sha256": verified_calibration[
                "calibration_payload_sha256"
            ],
        },
        "interpretation_boundary": {
            "single_predeclared_test_replay": True,
            "threshold_search_on_test": False,
            "geometry_search_on_test": False,
            "test_sample_selection_by_label": False,
            "post_test_parameter_tuning_allowed": False,
            "models_lock_modified": False,
            "direct_promotion_allowed": False,
            "test_set_is_organizer_hidden_set": False,
        },
        "limitations": [
            "Dataset300 test is a local frozen confirmation split, not the organizer hidden test set.",
            "The split was used by earlier project candidates; independence is specific to this v2 border-refinement decision.",
            "This run measures Component detection only; downstream Pin, OCR, Graph, and end-to-end behavior remain to be replayed.",
        ],
    }
    status["confirmation_payload_sha256"] = base._canonical_sha256(status)
    status_path = output_dir / "paired_test_status.json"
    base._atomic_write_json(status_path, status)
    print(status_path)
    print(
        json.dumps(
            {
                "status": status["status"],
                "baseline": {
                    key: baseline_metrics[key]
                    for key in ("precision50", "recall50", "f1_50", "bbox_map50", "false_positives")
                },
                "candidate": {
                    key: candidate_metrics[key]
                    for key in ("precision50", "recall50", "f1_50", "bbox_map50", "false_positives")
                },
                "comparison": comparison,
                "decision_gates": gate_results,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if all_gates_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
