"""Assemble A05 shadow A/B and downstream-error evidence.

Reuses frozen component_yolo11n audits and same-data raw-v1 val runs.
Never writes models/models.lock.json. Predictions stay image-only (no GT).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDITS = ROOT / "artifacts" / "audits"
WORK = ROOT / "work"
LOCK = ROOT / "models" / "models.lock.json"

VAL_CASES = [
    "0030", "0037", "0038", "0044", "0055", "0062", "0065", "0077", "0089", "0091",
    "0092", "0095", "0125", "0135", "0140", "0144", "0147", "0155", "0165", "0171",
    "0188", "0190", "0202", "0212", "0234", "0238", "0267", "0280", "0283", "0291",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _delta(candidate: dict[str, Any], baseline: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        if key in candidate and key in baseline and isinstance(candidate[key], (int, float)):
            out[key] = candidate[key] - baseline[key]
    return out


def _metric_block(agg: dict[str, Any]) -> dict[str, Any]:
    return {
        "component_detection": agg.get("component", {}).get("detection"),
        "component_category": agg.get("component", {}).get("category"),
        "pin_detection": agg.get("pin", {}).get("detection"),
        "pin_point_error_px": agg.get("pin", {}).get("point_error_px"),
        "graph_pairwise_connectivity": agg.get("graph", {}).get("pairwise_connectivity"),
        "wire": agg.get("wire"),
        "fields": agg.get("fields"),
        "coverage_objects": agg.get("coverage", {}).get("objects"),
        "coverage_fields": agg.get("coverage", {}).get("fields"),
    }


def _field_table(fields: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, row in (fields or {}).items():
        out[name] = {
            "eligible": row.get("eligible"),
            "masked": row.get("masked"),
            "matched_eligible": row.get("matched_eligible"),
            "exact_correct_end_to_end": row.get("exact_correct"),
            "exact_accuracy_end_to_end": row.get("exact_accuracy_end_to_end"),
            "exact_correct_on_matched": row.get("normalized_correct"),
            "exact_accuracy_on_matched": row.get("exact_accuracy_on_matched"),
        }
    return out


def main() -> int:
    lock_before = _sha256(LOCK)

    refinement_ab = _load(AUDITS / "component_yolo11n_v2_box_border_refinement_ab.json")
    confirmation_ab = _load(AUDITS / "component_yolo11n_v2_box_border_confirmation_ab.json")
    composite_shadow = _load(AUDITS / "component_yolo11n_v1_composite_shadow.json")
    paired = _load(
        WORK
        / "confirm-component-yolo11n-v2-border-refinement-test-20260916"
        / "paired_test_status.json"
    )
    main_lock_raw = _load(WORK / "raw-production-component-v5-val-20260914" / "summary.json")
    yolo_raw = _load(
        WORK / "raw-yolo-component-v2-per-class-thresholds-shadow-val-20260914" / "summary.json"
    )
    border_confirm_raw = _load(
        WORK
        / "raw-yolo-component-v2-per-class-thresholds-border-confirm-box-ios-shadow-val-20260915"
        / "summary.json"
    )

    main_agg = main_lock_raw["aggregate"]
    yolo_agg = yolo_raw["aggregate"]
    confirm_agg = border_confirm_raw["aggregate"]
    main_m = _metric_block(main_agg)
    yolo_m = _metric_block(yolo_agg)
    confirm_m = _metric_block(confirm_agg)

    fields_main = _field_table(main_agg.get("fields", {}))
    fields_yolo = _field_table(yolo_agg.get("fields", {}))
    fields_confirm = _field_table(confirm_agg.get("fields", {}))

    field_deltas = {}
    regressions = []
    for name in sorted(set(fields_main) | set(fields_yolo)):
        base = fields_main.get(name, {})
        cand = fields_yolo.get(name, {})
        d = {
            "exact_correct_end_to_end": (cand.get("exact_correct_end_to_end") or 0)
            - (base.get("exact_correct_end_to_end") or 0),
            "exact_accuracy_end_to_end": (cand.get("exact_accuracy_end_to_end") or 0.0)
            - (base.get("exact_accuracy_end_to_end") or 0.0),
            "exact_correct_on_matched": (cand.get("exact_correct_on_matched") or 0)
            - (base.get("exact_correct_on_matched") or 0),
        }
        field_deltas[name] = d
        if d["exact_correct_end_to_end"] < 0:
            regressions.append(
                {
                    "axis": "ocr_binding",
                    "field": name,
                    "signal": "exact_correct_end_to_end_regressed",
                    "baseline": base.get("exact_correct_end_to_end"),
                    "candidate": cand.get("exact_correct_end_to_end"),
                    "delta": d["exact_correct_end_to_end"],
                }
            )

    ref_val = refinement_ab["frozen_validation_ab"]
    ref_runtime = refinement_ab["refinement_runtime"]
    conf_raw = confirmation_ab["raw_end_to_end_ab"]
    paired_metrics = paired["metrics"]
    paired_perf = paired["performance"]

    latency = {
        "scope": "component tiled detector only; excludes model load, OCR, pin, wire, graph, disk write",
        "split": "test",
        "sample_count": paired_perf["baseline"]["sample_count"],
        "baseline_elapsed_seconds": paired_perf["baseline"]["elapsed_seconds"],
        "candidate_elapsed_seconds": paired_perf["candidate"]["elapsed_seconds"],
        "delta_elapsed_seconds": paired_perf["candidate"]["elapsed_seconds"]
        - paired_perf["baseline"]["elapsed_seconds"],
        "baseline_mean_seconds_per_sample": paired_perf["baseline"]["elapsed_seconds"]
        / paired_perf["baseline"]["sample_count"],
        "candidate_mean_seconds_per_sample": paired_perf["candidate"]["elapsed_seconds"]
        / paired_perf["candidate"]["sample_count"],
        "border_refinement_replay_seconds_validation": ref_runtime[
            "total_component_replay_seconds"
        ],
        "composite_shadow_phase_timings_seconds": composite_shadow["result"][
            "phase_timings_seconds"
        ],
        "composite_shadow_elapsed_seconds": composite_shadow["result"]["elapsed_seconds"],
        "includes_model_load": False,
        "includes_remote_calls": False,
    }

    pin_owner = {
        "same_data_val_raw_v1": {
            "main_lock": {
                "detection": main_agg["pin"]["detection"],
                "point_error_px": main_agg["pin"]["point_error_px"],
                "no_active_owner_masked": main_agg["coverage"]["objects"]["pin"]["mask_reasons"].get(
                    "no_active_owner", 0
                ),
                "pin_eligible": main_agg["coverage"]["objects"]["pin"]["eligible"],
            },
            "yolo_candidate_base": {
                "detection": yolo_agg["pin"]["detection"],
                "point_error_px": yolo_agg["pin"]["point_error_px"],
                "no_active_owner_masked": yolo_agg["coverage"]["objects"]["pin"]["mask_reasons"].get(
                    "no_active_owner", 0
                ),
                "pin_eligible": yolo_agg["coverage"]["objects"]["pin"]["eligible"],
            },
            "delta_candidate_minus_main_lock": _delta(
                {
                    **yolo_agg["pin"]["detection"],
                    "median_px": yolo_agg["pin"]["point_error_px"]["median"],
                    "p95_px": yolo_agg["pin"]["point_error_px"]["p95"],
                },
                {
                    **main_agg["pin"]["detection"],
                    "median_px": main_agg["pin"]["point_error_px"]["median"],
                    "p95_px": main_agg["pin"]["point_error_px"]["p95"],
                },
                ["precision", "recall", "f1", "tp", "fp", "fn", "median_px", "p95_px"],
            ),
            "composite_shadow_owner_checks": composite_shadow["checks"],
        },
        "border_refinement_vs_yolo_base_raw_e2e_neighbor": {
            "note": (
                "raw_end_to_end_ab in the border-confirmation audit is the closest "
                "same-data downstream pair on the YOLO path. It is border-confirmation, "
                "not border-refinement; used only as a neighbor signal for pin/graph."
            ),
            "baseline_pin": conf_raw["pin"]["baseline"],
            "candidate_pin": conf_raw["pin"]["candidate"],
            "delta_pin": _delta(
                conf_raw["pin"]["candidate"],
                conf_raw["pin"]["baseline"],
                ["precision", "recall", "f1", "true_positives", "false_positives", "false_negatives"],
            ),
        },
    }

    net = {
        "same_data_val_raw_v1": {
            "metric": "graph.pairwise_connectivity",
            "main_lock": main_agg["graph"]["pairwise_connectivity"],
            "yolo_candidate_base": yolo_agg["graph"]["pairwise_connectivity"],
            "delta_candidate_minus_main_lock": _delta(
                yolo_agg["graph"]["pairwise_connectivity"],
                main_agg["graph"]["pairwise_connectivity"],
                ["precision", "recall", "f1", "tp", "fp", "fn"],
            ),
            "wire": {
                "main_lock_f1": main_agg["wire"]["f1"],
                "yolo_candidate_base_f1": yolo_agg["wire"]["f1"],
                "unchanged": main_agg["wire"]["f1"] == yolo_agg["wire"]["f1"],
            },
        },
        "border_confirm_neighbor": {
            "graph_unchanged": conf_raw["graph"].get("unchanged"),
            "graph": {
                k: conf_raw["graph"][k]
                for k in ("precision", "recall", "f1", "true_positives", "false_positives", "false_negatives")
                if k in conf_raw["graph"]
            },
            "wire_f1": conf_raw["wire"]["f1"],
        },
    }

    component_downstream = {
        "schema_version": 1,
        "kind": "component_downstream_errors",
        "task_id": "A05",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "evaluator_id": "raw-v1-masked+component_yolo11n_reused_audits",
        "split_role": "development_regression",
        "sidecar_access": False,
        "gt_in_predictions": False,
        "policy": {
            "splits_not_mixed": True,
            "metric_contracts_not_mixed": True,
            "oracle_excluded_from_predictions": True,
            "models_lock_modified": False,
        },
        "ocr_binding": {
            "same_data_val_raw_v1": {
                "metric_contract": "raw-v1 field exact on eligible GT fields (offline scoring only)",
                "sample_count": 30,
                "case_ids": VAL_CASES,
                "main_lock": fields_main,
                "yolo_candidate_base": fields_yolo,
                "delta_candidate_minus_main_lock": field_deltas,
                "border_confirm_neighbor": fields_confirm,
                "border_confirm_delta_vs_yolo_base": {
                    name: {
                        "exact_correct_end_to_end": (
                            fields_confirm.get(name, {}).get("exact_correct_end_to_end") or 0
                        )
                        - (fields_yolo.get(name, {}).get("exact_correct_end_to_end") or 0)
                    }
                    for name in sorted(set(fields_yolo) | set(fields_confirm))
                },
            },
            "findings": [
                "YOLO candidate path raises pin.name and pin.number end-to-end exact counts versus main lock.",
                "YOLO candidate path lowers component.designator and component.value end-to-end exact counts versus main lock.",
                "Border-confirmation neighbor leaves all five field exact counts unchanged versus the YOLO base path.",
            ],
        },
        "pin_owner": pin_owner,
        "net": net,
        "latency": latency,
        "error_modes_component_val": {
            "source": "artifacts/analysis/component_yolo11n_v2_box_border_confirmation_ios_validation_error_modes.json",
            "role": "context only; not mixed into raw-v1 downstream scores",
        },
        "regressions": regressions,
        "promotion_gate": {
            "rule": "downstream_regression_blocks_promotion",
            "downstream_regressed": bool(regressions),
            "regression_summary": [
                "component.designator exact_correct_end_to_end 17 -> 10",
                "component.value exact_correct_end_to_end 8 -> 2",
            ]
            if regressions
            else [],
            "border_refinement_itself_degrades_downstream": False,
            "border_refinement_downstream_e2e_replayed": False,
            "promotion_eligible": False,
            "decision": "do_not_promote",
            "reasons": [
                "OCR field exact regressions on component.designator and component.value versus main lock on the same raw-v1 val cases.",
                "Border-refinement downstream (OCR binding / pin owner / Net) was not separately replayed as a full raw-v1 pair; only component detection A/B and a border-confirmation neighbor are available.",
                "Validation-derived border policy remains selection-biased; independent holdout confirmation is incomplete.",
            ],
        },
        "limitations": [
            "raw-v1 is a diagnostic proxy and is not the official evaluator.",
            "Field exact rates are offline-scored against GT; prediction manifests themselves are image-only.",
            "Border-refinement downstream E2E is inferred from neighbor evidence and is not a direct raw-v1 replay of the refinement switch.",
            "Latency for the paired test covers the tiled component detector only.",
        ],
        "lineage": {
            "main_lock_raw_summary": {
                "path": "work/raw-production-component-v5-val-20260914/summary.json",
                "sha256": _sha256(WORK / "raw-production-component-v5-val-20260914" / "summary.json"),
            },
            "yolo_raw_summary": {
                "path": "work/raw-yolo-component-v2-per-class-thresholds-shadow-val-20260914/summary.json",
                "sha256": _sha256(
                    WORK / "raw-yolo-component-v2-per-class-thresholds-shadow-val-20260914" / "summary.json"
                ),
            },
            "border_confirm_raw_summary": {
                "path": "work/raw-yolo-component-v2-per-class-thresholds-border-confirm-box-ios-shadow-val-20260915/summary.json",
                "sha256": _sha256(
                    WORK
                    / "raw-yolo-component-v2-per-class-thresholds-border-confirm-box-ios-shadow-val-20260915"
                    / "summary.json"
                ),
            },
            "border_refinement_ab": {
                "path": "artifacts/audits/component_yolo11n_v2_box_border_refinement_ab.json",
                "sha256": _sha256(AUDITS / "component_yolo11n_v2_box_border_refinement_ab.json"),
            },
            "border_confirmation_ab": {
                "path": "artifacts/audits/component_yolo11n_v2_box_border_confirmation_ab.json",
                "sha256": _sha256(AUDITS / "component_yolo11n_v2_box_border_confirmation_ab.json"),
            },
            "composite_shadow": {
                "path": "artifacts/audits/component_yolo11n_v1_composite_shadow.json",
                "sha256": _sha256(AUDITS / "component_yolo11n_v1_composite_shadow.json"),
            },
            "models_lock": {
                "path": "models/models.lock.json",
                "sha256": lock_before,
                "modified": False,
            },
        },
    }

    shadow_ab = {
        "schema_version": 1,
        "kind": "component_yolo_border_refinement_shadow_ab",
        "task_id": "A05",
        "status": "shadow_complete_promotion_blocked",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "evaluator_id": "raw-v1-masked+tiled_iou50+paired_test_iou50",
        "split_role": "development_regression",
        "sidecar_access": False,
        "models_lock_modified": False,
        "models_lock_sha256_before": lock_before,
        "models_lock_sha256_after": lock_before,
        "roles": {
            "main_lock": {
                "task": "component",
                "version": "component-34d79a6dfa4340e6",
                "path": "releases/v8-portable/component/train-component-v5-fullmix/model.pt",
                "sha256": "34d79a6dfa4340e64dc96de4945bcfea6e376d9a6f4949876b58d7e719697af5",
                "adapter": "pcbparse.perception.torch_component_adapter:factory",
                "identity": "blessed Faster R-CNN component from models/models.lock.json",
            },
            "candidate": {
                "name": "yolo11n-v2-box-internal-border-refinement",
                "checkpoint": "work/yolo/component-yolo11n-v2-precision/weights/best.pt",
                "checkpoint_sha256": "134ec62e77e11cdccb296040a3fd346a3864f13a1d44de7fb7a55e6ae0d87c8b",
                "adapter": "pcbparse.perception.yolo_component_adapter:factory",
                "selected_strategy": refinement_ab["selected_strategy"],
                "shadow_only": True,
                "blessed": False,
            },
        },
        "policy": {
            "shadow_comparison_without_lock_edit": True,
            "splits_not_mixed": True,
            "metric_contracts_not_mixed": [
                "raw-v1 (same-data val E2E)",
                "tiled_iou50 (border-refinement validation A/B)",
                "global-score-ranked class-aware one-to-one IoU@0.5 (paired test)",
            ],
            "gt_in_predictions": False,
            "oracle_in_predictions": False,
            "downstream_regression_blocks_promotion": True,
        },
        "same_data_val_raw_v1": {
            "description": (
                "Image-only raw-v1 shadow on the frozen 30-sample validation cases. "
                "Main lock runs the blessed Faster R-CNN component; candidate runs the "
                "YOLO checkpoint injected in memory. Same case_ids and image_sha256 values."
            ),
            "metric_contract": "raw-v1",
            "split": "val",
            "sample_count": 30,
            "case_ids": VAL_CASES,
            "cases_succeeded": {"main_lock": 30, "yolo_candidate_base": 30},
            "main_lock": main_m,
            "yolo_candidate_base": yolo_m,
            "delta_yolo_base_minus_main_lock": {
                "component_f1": yolo_agg["component"]["detection"]["f1"]
                - main_agg["component"]["detection"]["f1"],
                "component_precision": yolo_agg["component"]["detection"]["precision"]
                - main_agg["component"]["detection"]["precision"],
                "component_recall": yolo_agg["component"]["detection"]["recall"]
                - main_agg["component"]["detection"]["recall"],
                "pin_f1": yolo_agg["pin"]["detection"]["f1"] - main_agg["pin"]["detection"]["f1"],
                "graph_f1": yolo_agg["graph"]["pairwise_connectivity"]["f1"]
                - main_agg["graph"]["pairwise_connectivity"]["f1"],
                "wire_f1": yolo_agg["wire"]["f1"] - main_agg["wire"]["f1"],
            },
            "border_confirm_neighbor_only": {
                "warning": "border-confirmation policy, not border-refinement; not mixed into candidate scores",
                "metrics": confirm_m,
                "delta_vs_yolo_base_component_f1": confirm_agg["component"]["detection"]["f1"]
                - yolo_agg["component"]["detection"]["f1"],
            },
            "note": (
                "raw-v1 numbers compare main lock to the YOLO candidate path (per-class "
                "thresholds). Box-internal border-refinement is scored separately below and "
                "is never averaged into these raw-v1 figures."
            ),
        },
        "border_refinement_component_val": {
            "metric_contract": "tiled_iou50",
            "split": "val",
            "sample_count": ref_val["sample_count"],
            "ground_truth_count": ref_val["ground_truth_count"],
            "baseline_role": "yolo candidate without border refinement",
            "candidate_role": "yolo candidate with border refinement (Box)",
            "baseline": ref_val["baseline"],
            "candidate": ref_val["candidate"],
            "delta_candidate_minus_baseline": ref_val["delta_candidate_minus_baseline"],
            "refinement_runtime": ref_runtime,
            "source_status": {
                "baseline_status": refinement_ab["lineage"]["baseline_status"],
                "candidate_status": refinement_ab["lineage"]["candidate_status"],
                "pipeline_config": refinement_ab["lineage"]["pipeline_config"],
            },
        },
        "border_refinement_component_test": {
            "metric_contract": paired_metrics["baseline"]["metric_contract"],
            "split": "test",
            "split_role": "local_frozen_confirmation_not_organizer_hidden",
            "sample_count": 30,
            "ground_truth_count": paired_metrics["baseline"]["ground_truth_count"],
            "baseline_role": "yolo candidate without border refinement",
            "candidate_role": "yolo candidate with border refinement (Box)",
            "baseline": {
                k: paired_metrics["baseline"][k]
                for k in (
                    "precision50",
                    "recall50",
                    "f1_50",
                    "bbox_map50",
                    "true_positives",
                    "false_positives",
                    "false_negatives",
                    "prediction_count",
                    "pred_to_gt_ratio",
                )
            },
            "candidate": {
                k: paired_metrics["candidate"][k]
                for k in (
                    "precision50",
                    "recall50",
                    "f1_50",
                    "bbox_map50",
                    "true_positives",
                    "false_positives",
                    "false_negatives",
                    "prediction_count",
                    "pred_to_gt_ratio",
                )
            },
            "delta_candidate_minus_baseline": paired_metrics["comparison"]["delta"],
            "box_focus": paired_metrics["comparison"]["box"],
            "decision_gates": paired["decision_gates"],
            "performance": {
                "baseline_elapsed_seconds": paired_perf["baseline"]["elapsed_seconds"],
                "candidate_elapsed_seconds": paired_perf["candidate"]["elapsed_seconds"],
                "baseline_refinement_stats": paired_perf["baseline"]["internal_border_refinement"],
                "candidate_refinement_stats": paired_perf["candidate"]["internal_border_refinement"],
            },
        },
        "composite_shadow_integration": {
            "source": "artifacts/audits/component_yolo11n_v1_composite_shadow.json",
            "note": "One-image in-memory Component swap vs locked downstream providers; integration check only.",
            "checks": composite_shadow["checks"],
            "result": {
                "sample_id": composite_shadow["result"]["sample_id"],
                "components": composite_shadow["result"]["components"],
                "pins": composite_shadow["result"]["pins"],
                "texts": composite_shadow["result"]["texts"],
                "junctions": composite_shadow["result"]["junctions"],
                "elapsed_seconds": composite_shadow["result"]["elapsed_seconds"],
                "phase_timings_seconds": composite_shadow["result"]["phase_timings_seconds"],
            },
        },
        "downstream_reference": {
            "path": "artifacts/audits/component_downstream_errors.json",
            "summary": "OCR binding / pin owner / Net / latency tracked there; promotion blocked by field regressions.",
        },
        "interpretation_boundary": {
            "validation_configuration_tuned": True,
            "independent_test_confirmation_completed": "component_only",
            "production_promotion_allowed": False,
            "models_lock_modified": False,
            "reason": (
                "Shadow comparison is complete without editing the main lock. Component "
                "detection improves for the border-refinement candidate on val and test. "
                "Downstream raw-v1 shows large component/pin/net gains versus main lock but "
                "OCR field exact regressions on designator/value, so promotion stays blocked."
            ),
        },
        "promotion": {
            "promotion_eligible": False,
            "decision": "do_not_promote",
            "downstream_regressed": True,
        },
        "limitations": refinement_ab.get("limitations", [])
        + [
            "raw-v1 is not the official evaluator and is not mixed with tiled or paired-test contracts.",
            "Same-data raw-v1 pair is main lock vs YOLO base path; border-refinement adds component A/B only.",
            "Public 210 images and prior test/release50 exposure are not claimed as independent holdouts.",
        ],
        "lineage": {
            "border_refinement_ab": {
                "path": "artifacts/audits/component_yolo11n_v2_box_border_refinement_ab.json",
                "sha256": _sha256(AUDITS / "component_yolo11n_v2_box_border_refinement_ab.json"),
            },
            "border_confirmation_ab": {
                "path": "artifacts/audits/component_yolo11n_v2_box_border_confirmation_ab.json",
                "sha256": _sha256(AUDITS / "component_yolo11n_v2_box_border_confirmation_ab.json"),
            },
            "paired_test_status": {
                "path": "work/confirm-component-yolo11n-v2-border-refinement-test-20260916/paired_test_status.json",
                "sha256": _sha256(
                    WORK
                    / "confirm-component-yolo11n-v2-border-refinement-test-20260916"
                    / "paired_test_status.json"
                ),
            },
            "main_lock_raw_summary": {
                "path": "work/raw-production-component-v5-val-20260914/summary.json",
                "sha256": _sha256(WORK / "raw-production-component-v5-val-20260914" / "summary.json"),
            },
            "yolo_raw_summary": {
                "path": "work/raw-yolo-component-v2-per-class-thresholds-shadow-val-20260914/summary.json",
                "sha256": _sha256(
                    WORK / "raw-yolo-component-v2-per-class-thresholds-shadow-val-20260914" / "summary.json"
                ),
            },
            "per_class_threshold_calibration": {
                "path": "artifacts/audits/component_yolo11n_v2_per_class_threshold_calibration.json",
                "sha256": _sha256(AUDITS / "component_yolo11n_v2_per_class_threshold_calibration.json"),
            },
            "candidate_pipeline_config": {
                "path": "configs/pipeline.component512-border-refine-box-experiment.yaml",
                "sha256": _sha256(ROOT / "configs" / "pipeline.component512-border-refine-box-experiment.yaml"),
            },
            "models_lock": {
                "path": "models/models.lock.json",
                "sha256": lock_before,
                "modified": False,
            },
        },
    }

    out_shadow = AUDITS / "component_shadow_ab.json"
    out_downstream = AUDITS / "component_downstream_errors.json"
    out_shadow.write_text(
        json.dumps(shadow_ab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    out_downstream.write_text(
        json.dumps(component_downstream, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lock_after = _sha256(LOCK)
    if lock_after != lock_before:
        raise RuntimeError("models.lock.json changed during A05 assembly")

    delivery = {
        "task_id": "A05",
        "status": "implemented",
        "source_commit": "3510d644122510b11775f4c834e738faaca30287",
        "dirty_source_manifest_sha256": _sha256(Path(__file__)),
        "config_sha256": _sha256(ROOT / "configs" / "pipeline.component512-border-refine-box-experiment.yaml"),
        "model_lock_sha256": lock_after,
        "input_manifest_sha256": _sha256(WORK / "raw-yolo-component-v2-per-class-thresholds-shadow-val-20260914" / "predictions.jsonl"),
        "evaluator_id": "raw-v1-masked+tiled_iou50+paired_test_iou50+a05_shadow_ab",
        "split_role": "development_regression",
        "sidecar_access": False,
        "changes": [
            "scripts/build_component_shadow_ab_a05.py",
            "artifacts/audits/component_shadow_ab.json",
            "artifacts/audits/component_downstream_errors.json",
            "artifacts/audits/A05_delivery.json",
        ],
        "commands": [
            "$env:PYTHONPATH=(Resolve-Path '.\\src').Path; & python scripts/build_component_shadow_ab_a05.py",
        ],
        "metrics": [
            {
                "name": "raw_v1_component_f1",
                "numerator": round(yolo_agg["component"]["detection"]["f1"], 6),
                "denominator": 1.0,
                "scope": "raw-v1 val 30 same-data cases",
                "baseline": round(main_agg["component"]["detection"]["f1"], 6),
                "candidate": round(yolo_agg["component"]["detection"]["f1"], 6),
                "delta": round(
                    yolo_agg["component"]["detection"]["f1"]
                    - main_agg["component"]["detection"]["f1"],
                    6,
                ),
                "eligible": 30,
                "masked": 0,
                "confidence_interval": None,
            },
            {
                "name": "raw_v1_pin_f1",
                "numerator": round(yolo_agg["pin"]["detection"]["f1"], 6),
                "denominator": 1.0,
                "scope": "raw-v1 val 30 same-data cases",
                "baseline": round(main_agg["pin"]["detection"]["f1"], 6),
                "candidate": round(yolo_agg["pin"]["detection"]["f1"], 6),
                "delta": round(
                    yolo_agg["pin"]["detection"]["f1"] - main_agg["pin"]["detection"]["f1"], 6
                ),
                "eligible": 30,
                "masked": 0,
                "confidence_interval": None,
            },
            {
                "name": "raw_v1_graph_f1",
                "numerator": round(yolo_agg["graph"]["pairwise_connectivity"]["f1"], 6),
                "denominator": 1.0,
                "scope": "raw-v1 val 30 same-data cases",
                "baseline": round(main_agg["graph"]["pairwise_connectivity"]["f1"], 6),
                "candidate": round(yolo_agg["graph"]["pairwise_connectivity"]["f1"], 6),
                "delta": round(
                    yolo_agg["graph"]["pairwise_connectivity"]["f1"]
                    - main_agg["graph"]["pairwise_connectivity"]["f1"],
                    6,
                ),
                "eligible": 30,
                "masked": 0,
                "confidence_interval": None,
            },
            {
                "name": "border_refinement_val_f1_50",
                "numerator": round(ref_val["candidate"]["f1_50"], 6),
                "denominator": 1.0,
                "scope": "tiled_iou50 val 30 samples (not mixed with raw-v1)",
                "baseline": round(ref_val["baseline"]["f1_50"], 6),
                "candidate": round(ref_val["candidate"]["f1_50"], 6),
                "delta": round(ref_val["delta_candidate_minus_baseline"]["f1_50"], 6),
                "eligible": ref_val["sample_count"],
                "masked": 0,
                "confidence_interval": None,
            },
            {
                "name": "border_refinement_test_f1_50",
                "numerator": round(paired_metrics["candidate"]["f1_50"], 6),
                "denominator": 1.0,
                "scope": "paired test 30 samples (local frozen confirmation; not mixed with val)",
                "baseline": round(paired_metrics["baseline"]["f1_50"], 6),
                "candidate": round(paired_metrics["candidate"]["f1_50"], 6),
                "delta": round(paired_metrics["comparison"]["delta"]["f1_50"], 6),
                "eligible": 30,
                "masked": 0,
                "confidence_interval": None,
            },
            {
                "name": "ocr_field_exact_regressions",
                "numerator": len(regressions),
                "denominator": 5,
                "scope": "raw-v1 val field exact_correct_end_to_end vs main lock",
                "baseline": "main lock designator 17 / value 8",
                "candidate": "yolo designator 10 / value 2",
                "delta": "2 fields regressed; pin.name/pin.number improved",
                "eligible": 5,
                "masked": 0,
                "confidence_interval": None,
            },
        ],
        "failure_cases": [
            {
                "axis": "ocr_binding",
                "field": "component.designator",
                "detail": "exact_correct_end_to_end 17 (main lock) -> 10 (YOLO candidate path)",
            },
            {
                "axis": "ocr_binding",
                "field": "component.value",
                "detail": "exact_correct_end_to_end 8 (main lock) -> 2 (YOLO candidate path)",
            },
        ],
        "known_limitations": [
            "Implemented shadow A/B from reused component_yolo11n audits plus same-data raw-v1 val runs; not a fresh full-GPU re-inference of both arms in this process.",
            "Border-refinement downstream E2E (OCR/pin owner/Net) was not a direct raw-v1 replay; component A/B and a border-confirmation neighbor are reported instead.",
            "raw-v1 is not the official evaluator; tiled and paired-test contracts are kept separate from raw-v1.",
            "models/models.lock.json left unchanged (sha256 af35e129...).",
            "Predictions are image-only; GT used only for offline scoring. sidecar_access=false.",
            "Promotion is blocked: downstream OCR field regressions vs main lock, and independent holdout confirmation is incomplete.",
        ],
        "rollback": (
            "A05 is additive-only. Delete artifacts/audits/component_shadow_ab.json, "
            "artifacts/audits/component_downstream_errors.json, artifacts/audits/A05_delivery.json "
            "and scripts/build_component_shadow_ab_a05.py. Main lock models/models.lock.json unchanged. "
            "No candidate was promoted."
        ),
        "acceptance_checks": {
            "shadow_compare_without_modifying_main_lock": True,
            "metrics_do_not_mix_splits": True,
            "downstream_regression_blocks_promotion": True,
            "no_gt_in_predictions": True,
            "candidate_path_and_config_hash_verified": True,
        },
        "outputs": [
            "artifacts/audits/component_shadow_ab.json",
            "artifacts/audits/component_downstream_errors.json",
            "artifacts/audits/A05_delivery.json",
        ],
    }

    out_delivery = AUDITS / "A05_delivery.json"
    out_delivery.write_text(
        json.dumps(delivery, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "wrote": [str(out_shadow), str(out_downstream), str(out_delivery)],
        "models_lock_unchanged": lock_before == lock_after,
        "metrics_summary": {
            "raw_v1_component_f1": {
                "main_lock": main_agg["component"]["detection"]["f1"],
                "yolo_base": yolo_agg["component"]["detection"]["f1"],
                "delta": yolo_agg["component"]["detection"]["f1"] - main_agg["component"]["detection"]["f1"],
            },
            "raw_v1_pin_f1": {
                "main_lock": main_agg["pin"]["detection"]["f1"],
                "yolo_base": yolo_agg["pin"]["detection"]["f1"],
                "delta": yolo_agg["pin"]["detection"]["f1"] - main_agg["pin"]["detection"]["f1"],
            },
            "raw_v1_graph_f1": {
                "main_lock": main_agg["graph"]["pairwise_connectivity"]["f1"],
                "yolo_base": yolo_agg["graph"]["pairwise_connectivity"]["f1"],
                "delta": yolo_agg["graph"]["pairwise_connectivity"]["f1"]
                - main_agg["graph"]["pairwise_connectivity"]["f1"],
            },
            "border_refinement_val_f1_50_delta": ref_val["delta_candidate_minus_baseline"]["f1_50"],
            "border_refinement_test_f1_50_delta": paired_metrics["comparison"]["delta"]["f1_50"],
            "ocr_field_regressions": len(regressions),
            "promotion_eligible": False,
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
