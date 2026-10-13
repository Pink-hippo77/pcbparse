"""Conservative ownership of image-wide visual Pin candidates."""
from __future__ import annotations

import math

from pcbparse.association.ocr_common import OCRAssociationConfig, upsert_evidence
from pcbparse.ir.models import BBox, CircuitIR, Evidence, Point


def boundary_distance(point: Point, bbox: BBox) -> float:
    dx = max(bbox.x1 - point.x, 0.0, point.x - bbox.x2)
    dy = max(bbox.y1 - point.y, 0.0, point.y - bbox.y2)
    if dx or dy:
        return math.hypot(dx, dy)
    return min(point.x - bbox.x1, bbox.x2 - point.x, point.y - bbox.y1, bbox.y2 - point.y)


def associate_pin_components(ir: CircuitIR, config: OCRAssociationConfig) -> dict:
    """Bind only a unique nearby detected boundary; preserve all existing owners.

    No keys, numbers, bases, or orientations are synthesized. Competing
    components within the configured distance margin leave the pin ownerless.
    """
    summary = {"enabled": config.pin_component_assignment_enabled, "assigned": 0,
               "ambiguous": 0, "unmatched": 0, "preserved": 0}
    if not config.pin_component_assignment_enabled:
        return summary
    for pin in ir.pins:
        if pin.component_id is not None:
            summary["preserved"] += 1
            continue
        candidates = sorted(
            (boundary_distance(pin.point, component.bbox), component.internal_id)
            for component in ir.components
        )
        candidates = [item for item in candidates if item[0] <= config.pin_component_max_distance_px]
        margin = candidates[1][0] - candidates[0][0] if len(candidates) > 1 else None
        status = "unmatched"
        if candidates:
            status = "ambiguous" if margin is not None and margin <= config.pin_component_unique_margin_px else "assigned"
        if status == "assigned":
            pin.component_id = candidates[0][1]
        summary[status] += 1
        upsert_evidence(pin.evidence, Evidence(
            evidence_id=f"pin-component:{pin.internal_id}", source="rule", view_id="original",
            metadata={"stage": "association", "policy": "bbox_boundary_distance_v1",
                      "status": status, "component_id": pin.component_id,
                      "margin_px": margin, "max_distance_px": config.pin_component_max_distance_px,
                      "candidates": [{"component_id": owner, "distance_px": distance}
                                     for distance, owner in candidates[:config.max_recorded_candidates]]},
        ))
    return summary
