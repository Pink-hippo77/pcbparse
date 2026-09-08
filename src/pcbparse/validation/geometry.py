from __future__ import annotations

import math
from collections.abc import Iterable

from pcbparse.ir.models import BBox, CircuitIR, Diagnostic, Point
from pcbparse.validation.common import error, stable_diagnostics


def _point_issues(
    point: Point,
    object_id: str,
    width: int,
    height: int,
    role: str,
) -> Iterable[Diagnostic]:
    if not math.isfinite(point.x) or not math.isfinite(point.y):
        yield error(
            "GEOM.NON_FINITE",
            f"{role} contains NaN or Infinity",
            object_id,
            stage="geometry",
        )
        return
    # Canonical geometry uses continuous canvas coordinates: pixel centres are
    # inside ``[0, width) x [0, height)``, while a wire/pin endpoint may lie on
    # the outer canvas edge itself.  Dataset/Golden ingestion and the raw
    # evaluator therefore accept ``x == width`` / ``y == height``.  Treating
    # these as integer pixel indices (and capping at ``width - 1``) made the
    # validator contradict the rest of the coordinate contract.
    if not (0.0 <= point.x <= width and 0.0 <= point.y <= height):
        yield error(
            "GEOM.OUT_OF_BOUNDS",
            f"{role} ({point.x}, {point.y}) is outside image {width}x{height}",
            object_id,
            stage="geometry",
        )


def _bbox_issues(
    bbox: BBox,
    object_id: str,
    width: int,
    height: int,
    role: str,
) -> Iterable[Diagnostic]:
    values = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
    if not all(math.isfinite(value) for value in values):
        yield error(
            "GEOM.NON_FINITE",
            f"{role} contains NaN or Infinity",
            object_id,
            stage="geometry",
        )
        return
    if bbox.x1 > bbox.x2 or bbox.y1 > bbox.y2:
        yield error(
            "GEOM.INVALID_BBOX_ORDER",
            f"{role} must satisfy x1 <= x2 and y1 <= y2",
            object_id,
            stage="geometry",
        )
    # BBoxes are half-open canvas extents.  The right/bottom edge may equal
    # the image width/height (as it does for a full-image detector box), while
    # the left/top edge must remain inside the canvas for a positive-area box.
    if not (
        0.0 <= bbox.x1 < width
        and 0.0 < bbox.x2 <= width
        and 0.0 <= bbox.y1 < height
        and 0.0 < bbox.y2 <= height
    ):
        yield error(
            "GEOM.OUT_OF_BOUNDS",
            f"{role} is outside image {width}x{height}",
            object_id,
            stage="geometry",
        )
    if bbox.x1 == bbox.x2 or bbox.y1 == bbox.y2:
        yield error(
            "GEOM.ZERO_AREA_BBOX",
            f"{role} has zero area",
            object_id,
            stage="geometry",
        )


def validate_geometry(ir: CircuitIR) -> list[Diagnostic]:
    issues: list[Diagnostic] = []
    width, height = ir.image.width, ir.image.height

    for component in ir.components:
        issues.extend(
            _bbox_issues(component.bbox, component.internal_id, width, height, "component bbox")
        )
    for text in ir.texts:
        issues.extend(_bbox_issues(text.bbox, text.internal_id, width, height, "text bbox"))
    for pin in ir.pins:
        issues.extend(_point_issues(pin.point, pin.internal_id, width, height, "pin point"))
        if pin.base is not None:
            issues.extend(_point_issues(pin.base, pin.internal_id, width, height, "pin base"))
    for segment in ir.segments:
        issues.extend(
            _point_issues(segment.start, segment.internal_id, width, height, "segment start")
        )
        issues.extend(
            _point_issues(segment.end, segment.internal_id, width, height, "segment end")
        )
        if (
            math.isfinite(segment.start.x)
            and math.isfinite(segment.start.y)
            and math.isfinite(segment.end.x)
            and math.isfinite(segment.end.y)
            and segment.start.x == segment.end.x
            and segment.start.y == segment.end.y
        ):
            issues.append(
                error(
                    "EDGE.ZERO_LENGTH",
                    "wire segment has identical start and end points",
                    segment.internal_id,
                    stage="geometry",
                )
            )
    for junction in ir.junctions:
        issues.extend(
            _point_issues(junction.point, junction.internal_id, width, height, "junction point")
        )
    for label in ir.net_labels:
        if label.point is not None:
            issues.extend(
                _point_issues(label.point, label.internal_id, width, height, "net-label point")
            )

    return stable_diagnostics(issues)


class GeometryValidator:
    def validate(self, ir: CircuitIR, ctx: object | None = None) -> list[Diagnostic]:
        del ctx
        return validate_geometry(ir)
