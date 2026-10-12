"""Detection-only evaluation from pre-recognition OCR candidates.

Evaluates text-box detection P/R using raw detector candidates with
independent stage states.  Detection evidence is never overwritten by
recognition results.  Coordinates are restored to original-image space
via the recorded transform.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pcbparse.evaluation.masked_metrics import (
    _maximum_cardinality_matching,
    _natural_key,
)
from pcbparse.evaluation.raw_ground_truth import RawGroundTruth
from pcbparse.evaluation.text_box_eval import _bbox_iou, _center_distance, _gt_text_boxes

STAGE_NAMES = ("detected", "geometry_accepted", "recognition_attempted", "recognition_accepted")


def _entry_bbox(entry: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    orig = entry.get("original_bbox")
    if orig is not None and len(orig) == 4:
        return tuple(orig)
    vb = entry.get("view_bbox")
    if vb is not None and len(vb) == 4:
        return tuple(vb)
    return None


def _stage_ok(entry: Mapping[str, Any], stage: str) -> bool:
    if stage == "detected":
        return True
    return bool(entry.get(stage, False))


def evaluate_detection_candidates(
    detection_trace: Sequence[Mapping[str, Any]],
    gt: RawGroundTruth,
    *,
    iou_threshold: float = 0.3,
    center_tolerance_px: float = 12.0,
) -> dict[str, Any]:
    """Measure detection P/R per stage from pre-recognition candidates.

    Each entry carries independent boolean flags: ``geometry_accepted``,
    ``recognition_attempted``, ``recognition_accepted``.  Detection TP is
    measured at the ``geometry_accepted`` stage and is unaffected by whether
    recognition succeeded.  ``original_bbox`` (restored via transform) is used
    for matching when available; ``view_bbox`` is used only as fallback.
    """
    gt_boxes = _gt_text_boxes(gt)
    all_entries = list(detection_trace)
    by_stage: dict[str, dict[str, Any]] = {}

    for stage in STAGE_NAMES:
        stage_entries = [e for e in all_entries if _stage_ok(e, stage)]
        candidates: list[tuple[int, int, float]] = []
        valid_entries = []
        for entry in stage_entries:
            bbox = _entry_bbox(entry)
            if bbox is not None:
                valid_entries.append((entry, bbox))
        for pred_index, (_entry, bbox) in enumerate(valid_entries):
            for gt_index, target in enumerate(gt_boxes):
                score = _bbox_iou(bbox, target["bbox"])
                distance = _center_distance(bbox, target["bbox"])
                if score >= iou_threshold or distance <= center_tolerance_px:
                    candidates.append((pred_index, gt_index, max(score, 0.01)))
        matches = _maximum_cardinality_matching(
            len(valid_entries),
            len(gt_boxes),
            candidates,
            left_keys=[
                f"{e.get('view_id')}:{e.get('detection_index')}" for e, _ in valid_entries
            ],
            right_keys=[item["object_id"] for item in gt_boxes],
        )
        matched_gt = {right for _left, right, _score in matches}
        matched_pred = {left for left, _right, _score in matches}
        pred_count = len(valid_entries)
        tp = len(matched_gt)
        by_stage[stage] = {
            "candidate_count": pred_count,
            "true_positive": tp,
            "false_positive": pred_count - len(matched_pred),
            "false_negative": len(gt_boxes) - tp,
            "precision": tp / pred_count if pred_count else None,
            "recall": tp / len(gt_boxes) if gt_boxes else None,
            "detected_object_ids": [gt_boxes[i]["object_id"] for i in sorted(matched_gt)],
            "missed_object_ids": [
                item["object_id"] for i, item in enumerate(gt_boxes) if i not in matched_gt
            ],
        }

    rejected = [
        {
            "view_id": e.get("view_id"),
            "detection_index": e.get("detection_index"),
            "level": e.get("level"),
            "batch_id": e.get("batch_id"),
            "reason": e.get("reason") or "unknown",
            "original_bbox": e.get("original_bbox"),
        }
        for e in all_entries
        if e.get("outcome") == "rejected"
    ]
    reject_reasons: dict[str, int] = {}
    for item in rejected:
        reason = item["reason"]
        reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

    level_stats: dict[str, dict[str, int]] = {}
    for e in all_entries:
        level = str(e.get("level") or "unknown")
        bucket = level_stats.setdefault(level, {"detected": 0, "geometry_accepted": 0, "rejected": 0})
        bucket["detected"] += 1
        if e.get("geometry_accepted"):
            bucket["geometry_accepted"] += 1
        if e.get("outcome") == "rejected":
            bucket["rejected"] += 1

    geometry = by_stage.get("geometry_accepted", {})
    return {
        "kind": "detection_evaluation",
        "task_id": "OCR-01-detection",
        "case_id": gt.case_id,
        "iou_threshold": iou_threshold,
        "center_tolerance_px": center_tolerance_px,
        "gt_box_count": len(gt_boxes),
        "total_candidates": len(all_entries),
        "stages": by_stage,
        "geometry_accepted_precision": geometry.get("precision"),
        "geometry_accepted_recall": geometry.get("recall"),
        "rejected": rejected,
        "reject_reasons": reject_reasons,
        "level_stats": level_stats,
        "note": (
            "Detection TP measured at geometry_accepted stage; recognition outcome "
            "never alters detection evidence.  original_bbox used for matching."
        ),
    }
