from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from pcbparse.association.geometry import closest_point_on_segment, orientation_penalty, point_segment_distance
from pcbparse.ir.models import Pin, Point, WireSegment


@dataclass(frozen=True, slots=True)
class SnapDecision:
    status: str
    pin_id: str
    segment_id: str | None
    point: Point | None
    cost: float | None
    margin: float | None
    candidates: tuple[tuple[str, float], ...]


def snap_pin(
    pin: Pin,
    segments: list[WireSegment],
    *,
    line_width: float = 2.0,
    min_px: float = 2.0,
    max_px: float = 6.0,
    unique_margin: float = 1.5,
    segment_groups: dict[str, str] | None = None,
) -> SnapDecision:
    threshold = max(min_px, min(max_px, 1.5 * line_width))
    ranked: list[tuple[float, WireSegment, Point]] = []
    for segment in segments:
        projection = closest_point_on_segment(pin.point, segment)
        cost = point_segment_distance(pin.point, segment) + 2.0 * orientation_penalty(pin.orientation, segment)
        ranked.append((cost, segment, projection))
    ranked = _collapse_segment_groups(ranked, segment_groups)
    ranked.sort(key=lambda item: (item[0], item[1].internal_id))
    candidates = tuple((item[1].internal_id, item[0]) for item in ranked[:3])
    if not ranked:
        return SnapDecision("unmatched", pin.internal_id, None, None, None, None, candidates)
    best = ranked[0]
    second_cost = ranked[1][0] if len(ranked) > 1 else float("inf")
    margin = second_cost - best[0]
    if best[0] <= threshold and margin >= unique_margin:
        return SnapDecision("accepted", pin.internal_id, best[1].internal_id, best[2], best[0], margin, candidates)
    status = "ambiguous" if best[0] <= threshold else "unmatched"
    return SnapDecision(status, pin.internal_id, None, None, best[0], margin, candidates)


def snap_pins(
    pins: list[Pin],
    segments: list[WireSegment],
    *,
    line_width: float = 2.0,
    min_px: float = 2.0,
    max_px: float = 6.0,
    unique_margin: float = 1.5,
    segment_groups: dict[str, str] | None = None,
) -> list[SnapDecision]:
    """Snap pins using a spatial index and canonical physical candidates.

    Parallel/adjacent pieces already known to be one physical net must not count
    as competing second choices.  ``segment_groups`` maps each segment to the
    current union-find root and collapses them before the uniqueness margin is
    evaluated.
    """

    index = _SegmentSpatialIndex(segments, radius=max_px)
    decisions: list[SnapDecision] = []
    for pin in sorted(pins, key=lambda item: item.internal_id):
        local = index.query(pin.point)
        decisions.append(
            snap_pin(
                pin,
                local,
                line_width=line_width,
                min_px=min_px,
                max_px=max_px,
                unique_margin=unique_margin,
                segment_groups=segment_groups,
            )
        )
    return decisions


def _collapse_segment_groups(
    ranked: list[tuple[float, WireSegment, Point]],
    segment_groups: dict[str, str] | None,
) -> list[tuple[float, WireSegment, Point]]:
    if not segment_groups:
        return ranked
    best_by_group: dict[str, tuple[float, WireSegment, Point]] = {}
    for item in ranked:
        group = segment_groups.get(item[1].internal_id, item[1].internal_id)
        previous = best_by_group.get(group)
        if previous is None or (item[0], item[1].internal_id) < (previous[0], previous[1].internal_id):
            best_by_group[group] = item
    return list(best_by_group.values())


class _SegmentSpatialIndex:
    def __init__(self, segments: list[WireSegment], *, radius: float) -> None:
        self._cell_size = max(16.0, radius * 4.0, 1e-6)
        self._cells: dict[tuple[int, int], list[WireSegment]] = defaultdict(list)
        self._radius = max(0.0, radius)
        for segment in segments:
            x1 = min(segment.start.x, segment.end.x) - self._radius
            x2 = max(segment.start.x, segment.end.x) + self._radius
            y1 = min(segment.start.y, segment.end.y) - self._radius
            y2 = max(segment.start.y, segment.end.y) + self._radius
            for cx in range(math.floor(x1 / self._cell_size), math.floor(x2 / self._cell_size) + 1):
                for cy in range(math.floor(y1 / self._cell_size), math.floor(y2 / self._cell_size) + 1):
                    self._cells[(cx, cy)].append(segment)

    def query(self, point: Point) -> list[WireSegment]:
        cell = (
            math.floor(point.x / self._cell_size),
            math.floor(point.y / self._cell_size),
        )
        unique = {segment.internal_id: segment for segment in self._cells.get(cell, [])}
        return [unique[key] for key in sorted(unique)]