"""Real text-box detection/identification/binding evaluation (OCR-01).

One-to-one geometric matching between predicted ``TextToken`` boxes and
ground-truth text boxes.  Unlike the owner-match proxy in ``error_funnel``,
this evaluator measures true text-box detection (did the detector find a box
at the GT text location?), recognition (exact / CER), field classification and
owner binding.  Objects without a trusted GT text box report ``not_measured``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from pcbparse.evaluation.masked_metrics import (
    _maximum_cardinality_matching,
    _natural_key,
    _normal_text,
)
from pcbparse.evaluation.raw_ground_truth import RawGroundTruth
from pcbparse.ir.models import CircuitIR, TextToken

TEXT_BOX_STAGES = (
    "box_detected",
    "text_recognized",
    "field_classified",
    "owner_bound",
)


def _bbox_iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _center_distance(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return math.hypot(acx - bcx, acy - bcy)


def _token_bbox(token: TextToken) -> tuple[float, float, float, float]:
    return (token.bbox.x1, token.bbox.y1, token.bbox.x2, token.bbox.y2)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance divided by ``len(reference)``; 0 for exact match."""
    if reference == hypothesis:
        return 0.0
    if not reference:
        return 1.0
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1] / len(reference)


def _gt_text_boxes(gt: RawGroundTruth) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for component in gt.components:
        for field, target in (
            ("designator", component.designator),
            ("value", component.value),
            ("search_name", component.search_name),
        ):
            if target.mask.eligible and target.value is not None and target.bbox is not None:
                boxes.append(
                    {
                        "object_id": f"{component.raw_id}.{field}",
                        "owner_kind": "component",
                        "owner_id": component.raw_id,
                        "field": field,
                        "gt_value": str(target.value),
                        "bbox": target.bbox,
                        "text_id": target.text_id,
                    }
                )
    for pin in gt.pins:
        for field, target in (("number", pin.number), ("name", pin.name)):
            if target.mask.eligible and target.value is not None and target.bbox is not None:
                boxes.append(
                    {
                        "object_id": f"{pin.raw_id}.{field}",
                        "owner_kind": "pin",
                        "owner_id": pin.raw_id,
                        "field": field,
                        "gt_value": str(target.value),
                        "bbox": target.bbox,
                        "text_id": target.text_id,
                    }
                )
    boxes.sort(key=lambda item: _natural_key(item["object_id"]))
    return boxes


def evaluate_text_boxes(
    ir: CircuitIR,
    gt: RawGroundTruth,
    *,
    iou_threshold: float = 0.3,
    center_tolerance_px: float = 12.0,
    pred_to_gt_component: Mapping[str, str] | None = None,
    pred_to_gt_pin: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Measure detection/identification/field/owner for every trusted GT text box.

    Returns a report with per-object stage verdicts and aggregate counts.
    GT text boxes without a trusted bbox are excluded with a reason and never
    counted as zeros.  Predicted tokens without a GT match are false positives.
    """
    gt_boxes = _gt_text_boxes(gt)
    excluded: list[dict[str, Any]] = []
    for component in gt.components:
        for field, target in (
            ("designator", component.designator),
            ("value", component.value),
            ("search_name", component.search_name),
        ):
            if not (target.mask.eligible and target.value is not None and target.bbox is not None):
                if target.mask.eligible and target.value is not None:
                    excluded.append(
                        {
                            "object_id": f"{component.raw_id}.{field}",
                            "reason": "no_trusted_gt_text_bbox",
                        }
                    )
    for pin in gt.pins:
        for field, target in (("number", pin.number), ("name", pin.name)):
            if not (target.mask.eligible and target.value is not None and target.bbox is not None):
                if target.mask.eligible and target.value is not None:
                    excluded.append(
                        {
                            "object_id": f"{pin.raw_id}.{field}",
                            "reason": "no_trusted_gt_text_bbox",
                        }
                    )

    tokens = sorted(ir.texts, key=lambda item: _natural_key(item.internal_id))
    candidates: list[tuple[int, int, float]] = []
    for pred_index, token in enumerate(tokens):
        pred_bbox = _token_bbox(token)
        for gt_index, target in enumerate(gt_boxes):
            score = _bbox_iou(pred_bbox, target["bbox"])
            distance = _center_distance(pred_bbox, target["bbox"])
            if score >= iou_threshold or distance <= center_tolerance_px:
                candidates.append((pred_index, gt_index, max(score, 0.01)))
    matches = _maximum_cardinality_matching(
        len(tokens),
        len(gt_boxes),
        candidates,
        left_keys=[item.internal_id for item in tokens],
        right_keys=[item["object_id"] for item in gt_boxes],
    )
    gt_to_pred: dict[int, int] = {}
    pred_to_gt: dict[int, int] = {}
    for left, right, _score in matches:
        gt_to_pred[right] = left
        pred_to_gt[left] = right

    objects: list[dict[str, Any]] = []
    for gt_index, target in enumerate(gt_boxes):
        pred_index = gt_to_pred.get(gt_index)
        token = tokens[pred_index] if pred_index is not None else None
        stages: dict[str, dict[str, Any]] = {}
        if token is None:
            stages["box_detected"] = {"status": "failed", "reason": "no_predicted_text_box_matched"}
            for stage in TEXT_BOX_STAGES[1:]:
                stages[stage] = {"status": "not_measured", "reason": "box_not_detected"}
        else:
            stages["box_detected"] = {"status": "passed"}
            raw_text = token.raw_text
            resolved_text = token.resolved_text
            recognized_text = resolved_text if resolved_text is not None else raw_text
            raw_exact = raw_text == target["gt_value"]
            exact = recognized_text == target["gt_value"]
            normalized = _normal_text(recognized_text) == _normal_text(target["gt_value"])
            cer = character_error_rate(target["gt_value"], recognized_text)
            raw_cer = character_error_rate(target["gt_value"], raw_text)
            post_processed = resolved_text is not None and resolved_text != raw_text
            if exact:
                stages["text_recognized"] = {
                    "status": "passed",
                    "cer": 0.0,
                    "raw_exact": raw_exact,
                    "raw_cer": raw_cer,
                    "post_processed": post_processed,
                }
            elif normalized:
                stages["text_recognized"] = {
                    "status": "failed",
                    "reason": "normalized_match_only",
                    "cer": cer,
                    "raw_exact": raw_exact,
                    "raw_cer": raw_cer,
                    "post_processed": post_processed,
                }
            else:
                stages["text_recognized"] = {
                    "status": "failed",
                    "reason": "text_mismatch",
                    "cer": cer,
                    "raw_exact": raw_exact,
                    "raw_cer": raw_cer,
                    "post_processed": post_processed,
                }
            predicted_field = token.details or ""
            expected_field = target["field"]
            field_aliases = {
                "designator": {"designator"},
                "value": {"value"},
                "search_name": {"searchname", "search_name", "name"},
                "number": {"pinnumber", "pin_number", "number"},
                "name": {"pinname", "pin_name", "name"},
            }
            if predicted_field.casefold() in {
                alias for alias in field_aliases.get(expected_field, {expected_field})
            }:
                stages["field_classified"] = {"status": "passed"}
            else:
                stages["field_classified"] = {
                    "status": "failed",
                    "reason": f"predicted_field={predicted_field or 'empty'}",
                }
            expected_owner = target["owner_id"]
            if target["owner_kind"] == "pin":
                raw_owner = token.pin_id
                if pred_to_gt_pin is not None and token.pin_id is not None:
                    predicted_owner = pred_to_gt_pin.get(
                        str(token.pin_id), str(token.pin_id)
                    )
                else:
                    predicted_owner = str(raw_owner) if raw_owner is not None else None
            else:
                raw_owner = token.component_id
                if pred_to_gt_component is not None and token.component_id is not None:
                    predicted_owner = pred_to_gt_component.get(
                        str(token.component_id), str(token.component_id)
                    )
                else:
                    predicted_owner = str(raw_owner) if raw_owner is not None else None
            if predicted_owner == expected_owner:
                stages["owner_bound"] = {"status": "passed"}
            else:
                stages["owner_bound"] = {
                    "status": "failed",
                    "reason": f"predicted_owner={predicted_owner or 'unbound'}",
                }
        first_failed = next(
            (stage for stage in TEXT_BOX_STAGES if stages.get(stage, {}).get("status") == "failed"),
            None,
        )
        objects.append(
            {
                "object_id": target["object_id"],
                "owner_kind": target["owner_kind"],
                "owner_id": target["owner_id"],
                "field": target["field"],
                "gt_value": target["gt_value"],
                "gt_text_id": target["text_id"],
                "matched_token_id": token.internal_id if token is not None else None,
                "first_failed_stage": first_failed,
                "stages": stages,
            }
        )

    false_positives = [
        {"token_id": tokens[index].internal_id, "raw_text": tokens[index].raw_text}
        for index in sorted(set(range(len(tokens))) - set(pred_to_gt))
    ]

    stage_rows: list[dict[str, Any]] = []
    for stage in TEXT_BOX_STAGES:
        passed = sum(1 for item in objects if item["stages"].get(stage, {}).get("status") == "passed")
        failed = sum(1 for item in objects if item["stages"].get(stage, {}).get("status") == "failed")
        measured = passed + failed
        stage_rows.append(
            {
                "stage": stage,
                "status": "measured" if measured else ("not_measured" if not gt_boxes else "measured"),
                "count": passed,
                "denominator": len(gt_boxes),
                "evaluated": measured,
            }
        )

    return {
        "schema_version": "text_box_eval_v1",
        "kind": "text_box_evaluation",
        "task_id": "OCR-01",
        "case_id": gt.case_id,
        "iou_threshold": iou_threshold,
        "center_tolerance_px": center_tolerance_px,
        "stages": stage_rows,
        "objects": objects,
        "denominator_excluded": excluded,
        "false_positives": false_positives,
        "false_positive_count": len(false_positives),
        "gt_box_count": len(gt_boxes),
        "predicted_token_count": len(tokens),
        "matched_count": len(gt_to_pred),
    }
