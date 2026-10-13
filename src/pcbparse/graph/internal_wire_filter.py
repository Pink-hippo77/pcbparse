from __future__ import annotations

import math
from typing import Any

from pcbparse.association.geometry import point_segment_distance


def _segment_box_fraction(segment: Any, boxes: list[tuple[float, float, float, float]]) -> float:
    start = (float(segment.start.x), float(segment.start.y))
    end = (float(segment.end.x), float(segment.end.y))
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-9 or not boxes:
        return 0.0

    intervals: list[tuple[float, float]] = []
    for x1, y1, x2, y2 in boxes:
        lo, hi, intersects = 0.0, 1.0, True
        for a, b in (
            (-dx, start[0] - x1),
            (dx, x2 - start[0]),
            (-dy, start[1] - y1),
            (dy, y2 - start[1]),
        ):
            if abs(a) < 1e-12:
                if b < 0:
                    intersects = False
                    break
            else:
                ratio = b / a
                if a < 0:
                    lo = max(lo, ratio)
                else:
                    hi = min(hi, ratio)
                if lo > hi:
                    intersects = False
                    break
        if intersects:
            lo, hi = max(0.0, lo), min(1.0, hi)
            if hi > lo:
                intervals.append((lo, hi))

    if not intervals:
        return 0.0
    intervals.sort()
    covered = 0.0
    lo, hi = intervals[0]
    for next_lo, next_hi in intervals[1:]:
        if next_lo <= hi:
            hi = max(hi, next_hi)
        else:
            covered += hi - lo
            lo, hi = next_lo, next_hi
    covered += hi - lo
    return min(1.0, covered)


def _endpoint_near(segment: Any, points: list[tuple[float, float]], tolerance: float) -> bool:
    start = (float(segment.start.x), float(segment.start.y))
    end = (float(segment.end.x), float(segment.end.y))
    return any(
        min(math.dist(start, point), math.dist(end, point)) <= tolerance
        for point in points
    )


def filter_internal_wire_segments(
    ir: Any,
    *,
    containment_threshold: float = 0.80,
    pin_radius: float = 10.0,
    label_radius: float = 8.0,
) -> dict[str, Any]:
    """Drop unanchored wire predictions mostly inside predicted component/OCR boxes.

    Call after an initial physical-graph build so pin/label-bearing nets can
    protect their segments. Rebuild physical/electrical nets after filtering.
    Evaluation tolerances are deliberately not used as graph snap radii.
    """
    if not 0.0 <= containment_threshold <= 1.0:
        raise ValueError("containment_threshold must be in [0, 1]")
    if pin_radius < 0.0 or label_radius < 0.0:
        raise ValueError("anchor radii must be non-negative")

    component_boxes = [
        (float(c.bbox.x1), float(c.bbox.y1), float(c.bbox.x2), float(c.bbox.y2))
        for c in ir.components
    ]
    text_boxes = [
        (float(t.bbox.x1), float(t.bbox.y1), float(t.bbox.x2), float(t.bbox.y2))
        for t in ir.texts
    ]
    pin_points = [(float(p.point.x), float(p.point.y)) for p in ir.pins]
    label_points = [
        (float(label.point.x), float(label.point.y))
        for label in ir.net_labels
        if label.point is not None
    ]
    anchored_segment_ids = {
        segment_id
        for net in ir.physical_nets
        if net.pins or net.labels
        for segment_id in net.segments
    }

    deleted: list[dict[str, Any]] = []
    kept = []
    for segment in ir.segments:
        component_fraction = _segment_box_fraction(segment, component_boxes)
        text_fraction = _segment_box_fraction(segment, text_boxes)
        mostly_internal = (
            component_fraction >= containment_threshold
            or text_fraction >= containment_threshold
        )
        near_pin = _endpoint_near(segment, pin_points, pin_radius)
        near_label = _endpoint_near(segment, label_points, label_radius)
        projected_label = any(
            point_segment_distance(label.point, segment) <= label_radius
            for label in ir.net_labels
            if label.point is not None
        )
        anchored = segment.internal_id in anchored_segment_ids
        repaired = bool(getattr(segment, "synthetic", False)) or any(
            evidence.source == "repair" for evidence in segment.evidence
        )
        if mostly_internal and not (
            near_pin or near_label or projected_label or anchored or repaired
        ):
            deleted.append(
                {
                    "segment_id": segment.internal_id,
                    "component_inside_fraction": round(component_fraction, 6),
                    "ocr_inside_fraction": round(text_fraction, 6),
                    "internal_source": (
                        "component"
                        if component_fraction >= text_fraction
                        else "ocr"
                    ),
                }
            )
        else:
            kept.append(segment)

    keep_ids = {segment.internal_id for segment in kept}
    ir.segments = kept
    for pin in ir.pins:
        pin.edge_ids = [segment_id for segment_id in pin.edge_ids if segment_id in keep_ids]

    junctions = []
    for junction in ir.junctions:
        junction.segment_ids = [
            segment_id
            for segment_id in junction.segment_ids
            if segment_id in keep_ids
        ]
        segment_count = len(set(junction.segment_ids))
        if segment_count == 0:
            continue
        if segment_count == 1:
            junction.kind = "endpoint"
        elif junction.kind == "connected_cross" and segment_count == 3:
            junction.kind = "t_junction"
        elif junction.kind in {"t_junction", "connected_cross"} and segment_count == 2:
            junction.kind = "bend"
        junctions.append(junction)
    ir.junctions = junctions

    report = {
        "input_segments": len(keep_ids) + len(deleted),
        "deleted_segments": len(deleted),
        "retained_segments": len(keep_ids),
        "deleted_inside_component": sum(
            item["component_inside_fraction"] >= containment_threshold
            for item in deleted
        ),
        "deleted_inside_ocr": sum(
            item["ocr_inside_fraction"] >= containment_threshold
            for item in deleted
        ),
        "deleted_examples": deleted[:10],
        "containment_threshold": containment_threshold,
        "pin_anchor_radius_px": pin_radius,
        "label_anchor_radius_px": label_radius,
        "protected_by_existing_anchored_physical_net": True,
        "protected_by_repair_or_synthetic_flag": True,
    }
    return report
