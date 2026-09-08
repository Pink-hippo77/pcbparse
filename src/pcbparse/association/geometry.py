from __future__ import annotations

import math

from pcbparse.ir.models import Point, WireSegment


def distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def closest_point_on_segment(point: Point, segment: WireSegment) -> Point:
    dx = segment.end.x - segment.start.x
    dy = segment.end.y - segment.start.y
    denominator = dx * dx + dy * dy
    if denominator <= 1e-12:
        return segment.start
    ratio = ((point.x - segment.start.x) * dx + (point.y - segment.start.y) * dy) / denominator
    ratio = max(0.0, min(1.0, ratio))
    return Point(x=segment.start.x + ratio * dx, y=segment.start.y + ratio * dy)


def point_segment_distance(point: Point, segment: WireSegment) -> float:
    return distance(point, closest_point_on_segment(point, segment))


def orientation_penalty(orientation: str, segment: WireSegment) -> float:
    if orientation == "UNKNOWN":
        return 0.0
    dx = abs(segment.end.x - segment.start.x)
    dy = abs(segment.end.y - segment.start.y)
    horizontal = dx >= dy
    expects_horizontal = orientation in {"L", "R"}
    return 0.0 if horizontal == expects_horizontal else 1.0

