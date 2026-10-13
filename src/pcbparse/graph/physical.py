from __future__ import annotations

import bisect
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from pcbparse.association.geometry import point_segment_distance
from pcbparse.association.pin_wire import SnapDecision, snap_pins
from pcbparse.graph.union_find import UnionFind
from pcbparse.ir.models import CircuitIR, Evidence, PhysicalNet, Pin, Point, WireSegment

# Hard bound on longitudinal gap repair.  Gap tolerance may never grow without
# limit: that turns nearby parallel traces into one net.  Virtual gap repairs
# stay marked so they are never counted as visible solid wire.
HARD_MAX_GAP_REPAIR_PX = 12.0
# Lateral offset for "same supporting line" is always tightly bounded and is
# independent of the longitudinal gap tolerance.  Near-parallel traces a few
# pixels apart must not merge just because gap repair is enabled.
MAX_COLLINEAR_LATERAL_OFFSET_PX = 2.0


@dataclass(frozen=True, slots=True)
class ConnectionEvidence:
    """One connectivity decision shared by the builder and the validator.

    ``virtual`` is True only for gap-repair bridges.  A virtual bridge is not a
    visible solid wire and must stay separately marked in audits and net
    repair ratios.
    """

    left_id: str
    right_id: str
    rule: str
    confidence: float
    distance_px: float
    channel: str | None = None
    virtual: bool = False
    left_kind: str = "segment"
    right_kind: str = "segment"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "left_id": self.left_id,
            "right_id": self.right_id,
            "left_kind": self.left_kind,
            "right_kind": self.right_kind,
            "rule": self.rule,
            "confidence": self.confidence,
            "distance_px": self.distance_px,
            "channel": self.channel,
            "virtual": self.virtual,
            "metadata": dict(self.metadata),
        }

    def to_jsonl(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True, slots=True)
class ConnectionParams:
    """Tolerances shared by builder unions and validator adjacency."""

    endpoint_tolerance_px: float = 3.0
    collinear_merge_tolerance_px: float = 0.0
    collinear_gap_merge_tolerance_px: float = 0.0
    gap_repair_max_px: float = HARD_MAX_GAP_REPAIR_PX

    def __post_init__(self) -> None:
        for name in (
            "endpoint_tolerance_px",
            "collinear_merge_tolerance_px",
            "collinear_gap_merge_tolerance_px",
            "gap_repair_max_px",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite non-negative number")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.gap_repair_max_px > HARD_MAX_GAP_REPAIR_PX:
            raise ValueError(
                f"gap_repair_max_px cannot exceed the hard bound {HARD_MAX_GAP_REPAIR_PX}"
            )
        if self.collinear_gap_merge_tolerance_px > self.gap_repair_max_px:
            raise ValueError(
                "collinear_gap_merge_tolerance_px cannot exceed gap_repair_max_px "
                f"({self.gap_repair_max_px}); unbounded gap repair is forbidden"
            )

    @property
    def effective_gap_tolerance_px(self) -> float:
        return min(self.collinear_gap_merge_tolerance_px, self.gap_repair_max_px)


def collect_segment_connections(
    segments: Sequence[WireSegment],
    *,
    endpoint_tolerance_px: float = 3.0,
    collinear_merge_tolerance_px: float = 0.0,
    collinear_gap_merge_tolerance_px: float = 0.0,
    gap_repair_max_px: float = HARD_MAX_GAP_REPAIR_PX,
) -> list[ConnectionEvidence]:
    """Return the shared segment-segment connectivity evidence.

    Both :func:`build_physical_graph` and :func:`pcbparse.validation.graph.
    validate_graph` must use this function (or an equivalent call with the same
    parameters) so a connection accepted by the builder cannot be rejected by
    the validator.
    """

    params = ConnectionParams(
        endpoint_tolerance_px=endpoint_tolerance_px,
        collinear_merge_tolerance_px=collinear_merge_tolerance_px,
        collinear_gap_merge_tolerance_px=collinear_gap_merge_tolerance_px,
        gap_repair_max_px=gap_repair_max_px,
    )
    items = list(segments)
    evidence: list[ConnectionEvidence] = []
    seen: set[tuple[str, str, str]] = set()

    def _record(item: ConnectionEvidence) -> None:
        key = (item.left_id, item.right_id, item.rule)
        reverse = (item.right_id, item.left_id, item.rule)
        if key in seen or reverse in seen:
            return
        seen.add(key)
        evidence.append(item)

    endpoints: list[tuple[str, Point, str]] = []
    for segment in items:
        endpoints.append((segment.internal_id, segment.start, segment.channel))
        endpoints.append((segment.internal_id, segment.end, segment.channel))

    cells: dict[tuple[int, int, str], list[tuple[str, Point]]] = defaultdict(list)
    cell_size = max(params.endpoint_tolerance_px, 1e-6)
    for segment_id, point, channel in endpoints:
        cx = math.floor(point.x / cell_size)
        cy = math.floor(point.y / cell_size)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other_id, other_point in cells.get((cx + dx, cy + dy, channel), []):
                    if other_id == segment_id:
                        continue
                    distance = math.hypot(point.x - other_point.x, point.y - other_point.y)
                    if distance <= params.endpoint_tolerance_px:
                        left, right = sorted((segment_id, other_id))
                        _record(
                            ConnectionEvidence(
                                left_id=left,
                                right_id=right,
                                rule="endpoint_proximity",
                                confidence=1.0 - distance / max(params.endpoint_tolerance_px, 1e-6),
                                distance_px=distance,
                                channel=channel,
                                virtual=False,
                            )
                        )
        cells[(cx, cy, channel)].append((segment_id, point))

    by_polyline: dict[str, list[str]] = defaultdict(list)
    for segment in items:
        for polyline_id in segment.observed_polyline_ids:
            by_polyline[polyline_id].append(segment.internal_id)
    segment_index = {segment.internal_id: segment for segment in items}
    for polyline_id, segment_ids in by_polyline.items():
        for left_id, right_id, distance in _touching_pairs(
            segment_ids,
            segment_index,
            tolerance=params.endpoint_tolerance_px,
        ):
            _record(
                ConnectionEvidence(
                    left_id=left_id,
                    right_id=right_id,
                    rule="polyline_touch",
                    confidence=1.0,
                    distance_px=distance,
                    channel=segment_index[left_id].channel,
                    virtual=False,
                    metadata={"source_polyline_id": polyline_id},
                )
            )

    if params.collinear_merge_tolerance_px > 0 or params.effective_gap_tolerance_px > 0:
        cell_size = max(32.0, max(params.collinear_merge_tolerance_px, params.effective_gap_tolerance_px) * 4.0)
        cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        pair_seen: set[tuple[int, int]] = set()
        for index, segment in enumerate(items):
            pad = max(params.collinear_merge_tolerance_px, params.effective_gap_tolerance_px)
            min_x = min(segment.start.x, segment.end.x) - pad
            max_x = max(segment.start.x, segment.end.x) + pad
            min_y = min(segment.start.y, segment.end.y) - pad
            max_y = max(segment.start.y, segment.end.y) + pad
            candidates: set[int] = set()
            for cx in range(math.floor(min_x / cell_size), math.floor(max_x / cell_size) + 1):
                for cy in range(math.floor(min_y / cell_size), math.floor(max_y / cell_size) + 1):
                    candidates.update(cells.get((cx, cy), ()))
            for other_index in sorted(candidates):
                pair = (other_index, index)
                if pair in pair_seen:
                    continue
                pair_seen.add(pair)
                other = items[other_index]
                if other.channel != segment.channel:
                    continue
                left, right = sorted((other.internal_id, segment.internal_id))
                if params.collinear_merge_tolerance_px > 0 and segments_have_collinear_overlap(
                    other,
                    segment,
                    params.collinear_merge_tolerance_px,
                    allow_gap=False,
                ):
                    _record(
                        ConnectionEvidence(
                            left_id=left,
                            right_id=right,
                            rule="collinear_overlap",
                            confidence=1.0,
                            distance_px=0.0,
                            channel=segment.channel,
                            virtual=False,
                        )
                    )
                elif params.effective_gap_tolerance_px > 0 and segments_have_collinear_overlap(
                    other,
                    segment,
                    params.effective_gap_tolerance_px,
                    allow_gap=True,
                ):
                    gap = collinear_longitudinal_gap(other, segment)
                    _record(
                        ConnectionEvidence(
                            left_id=left,
                            right_id=right,
                            rule="collinear_gap_repair",
                            confidence=max(0.0, 1.0 - gap / max(params.effective_gap_tolerance_px, 1e-6)),
                            distance_px=gap,
                            channel=segment.channel,
                            virtual=True,
                            metadata={"gap_repair_max_px": params.gap_repair_max_px},
                        )
                    )
            for cx in range(math.floor(min_x / cell_size), math.floor(max_x / cell_size) + 1):
                for cy in range(math.floor(min_y / cell_size), math.floor(max_y / cell_size) + 1):
                    cells[(cx, cy)].append(index)

    by_source_edge: dict[str, list[str]] = defaultdict(list)
    for segment in items:
        if segment.source_edge_id:
            by_source_edge[segment.source_edge_id].append(segment.internal_id)
    for source_edge_id, segment_ids in by_source_edge.items():
        dataset_backed = all(
            any(item.source == "dataset" for item in segment_index[segment_id].evidence)
            for segment_id in segment_ids
            if segment_id in segment_index
        )
        for left_id, right_id, distance in _touching_pairs(
            segment_ids,
            segment_index,
            tolerance=params.endpoint_tolerance_px,
        ):
            rule = "dataset_source_edge" if dataset_backed else "source_edge_touch"
            _record(
                ConnectionEvidence(
                    left_id=left_id,
                    right_id=right_id,
                    rule=rule,
                    confidence=1.0,
                    distance_px=distance,
                    channel=segment_index[left_id].channel,
                    virtual=False,
                    metadata={"source_edge_id": source_edge_id, "dataset_backed": dataset_backed},
                )
            )
        if dataset_backed and len(segment_ids) > 1:
            anchor = sorted(segment_ids)[0]
            for other_id in sorted(segment_ids):
                if other_id == anchor:
                    continue
                _record(
                    ConnectionEvidence(
                        left_id=anchor,
                        right_id=other_id,
                        rule="dataset_source_edge_membership",
                        confidence=1.0,
                        distance_px=0.0,
                        channel=segment_index[anchor].channel if anchor in segment_index else None,
                        virtual=False,
                        metadata={"source_edge_id": source_edge_id, "dataset_backed": True},
                    )
                )

    return evidence


def evidence_adjacency(
    segment_ids: Sequence[str],
    evidence: Sequence[ConnectionEvidence],
) -> dict[str, set[str]]:
    """Adjacency map built only from shared connection evidence."""

    known = {segment_id for segment_id in segment_ids}
    adjacency: dict[str, set[str]] = {segment_id: set() for segment_id in known}
    for item in evidence:
        if item.left_kind != "segment" or item.right_kind != "segment":
            continue
        if item.left_id not in known or item.right_id not in known:
            continue
        adjacency[item.left_id].add(item.right_id)
        adjacency[item.right_id].add(item.left_id)
    return adjacency


def segments_are_connected(
    segment_ids: Sequence[str],
    evidence: Sequence[ConnectionEvidence],
    *,
    extra_groups: Sequence[set[str]] = (),
) -> bool:
    """True when every segment lies in one evidence-connected component."""

    available = list(dict.fromkeys(segment_ids))
    if len(available) < 2:
        return True
    adjacency = evidence_adjacency(available, evidence)
    for group in extra_groups:
        members = [item for item in group if item in adjacency]
        for source in members:
            for other in members:
                if source != other:
                    adjacency[source].add(other)
    visited = {available[0]}
    stack = [available[0]]
    while stack:
        current = stack.pop()
        for neighbor in adjacency[current] - visited:
            visited.add(neighbor)
            stack.append(neighbor)
    return len(visited) == len(available)


def filter_body_outline_wire_segments(
    ir: CircuitIR,
    *,
    tolerance_px: float = 4.0,
    coverage: float = 0.5,
    min_edge_span_px: float = 20.0,
) -> dict[str, Any]:
    """Remove wire segments that trace component body outlines (opt80 r08).

    A stroke running along a component bbox edge (both endpoints within
    ``tolerance_px`` of the edge line) whose collinear union covers at least
    ``coverage`` of the edge length is treated as the drawn body outline, not a
    wire: schematic body borders are routinely captured by the wire mask and,
    once vectorized, they fuse every pin stub that touches the body edge into
    one physical net (the dominant graph false-pair source on val30).

    Conservative guards: only edges at least ``min_edge_span_px`` long are
    considered, and at least ``coverage * span`` absolute length must be
    covered before suppression. Segments are removed from ``ir.segments`` (the
    same contract as the internal-wire filter); a report is returned for trace.
    """
    if tolerance_px < 0 or not math.isfinite(tolerance_px):
        raise ValueError("body outline tolerance must be finite and non-negative")
    if not 0.0 < coverage <= 1.0:
        raise ValueError("body outline coverage must be in (0, 1]")
    if min_edge_span_px < 0:
        raise ValueError("min_edge_span_px must be non-negative")
    suppressed: set[str] = set()
    for component in ir.components:
        x1, y1, x2, y2 = (
            component.bbox.x1,
            component.bbox.y1,
            component.bbox.x2,
            component.bbox.y2,
        )
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            continue
        edges = (
            ("v", x1, y1, y2, height),
            ("v", x2, y1, y2, height),
            ("h", y1, x1, x2, width),
            ("h", y2, x1, x2, width),
        )
        for axis, position, span_lo, span_hi, span in edges:
            if span < min_edge_span_px:
                continue
            covering: list[tuple[str, float, float]] = []
            for segment in ir.segments:
                if segment.internal_id in suppressed:
                    continue
                if axis == "v":
                    if (
                        abs(segment.start.x - position) > tolerance_px
                        or abs(segment.end.x - position) > tolerance_px
                    ):
                        continue
                    low, high = sorted((segment.start.y, segment.end.y))
                else:
                    if (
                        abs(segment.start.y - position) > tolerance_px
                        or abs(segment.end.y - position) > tolerance_px
                    ):
                        continue
                    low, high = sorted((segment.start.x, segment.end.x))
                clipped_low, clipped_high = max(low, span_lo), min(high, span_hi)
                if clipped_high > clipped_low:
                    covering.append(
                        (segment.internal_id, clipped_low, clipped_high)
                    )
            if not covering:
                continue
            intervals = sorted((low, high) for _, low, high in covering)
            merged: list[tuple[float, float]] = []
            for low, high in intervals:
                if merged and low <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], high))
                else:
                    merged.append((low, high))
            covered = sum(high - low for low, high in merged)
            if covered / span >= coverage:
                suppressed.update(segment_id for segment_id, _, _ in covering)
    if not suppressed:
        return {"suppressed": 0, "ids": []}
    removed = [s for s in ir.segments if s.internal_id not in suppressed]
    ir.segments = removed
    return {"suppressed": len(suppressed), "ids": sorted(suppressed)}


def filter_glyph_wire_segments(
    ir: CircuitIR,
    *,
    margin_px: float = 2.0,
) -> dict[str, Any]:
    """Remove glyph strokes that OCR text boxes explain (opt80 r08 residual).

    A wire segment whose BOTH endpoints lie inside the same detected text box
    (plus ``margin_px``) is a glyph stroke of that label, not a through wire:
    such strokes enter the wire mask from net labels and bridge otherwise
    separate wires. Segments that cross the text box boundary (label stubs and
    real wires under text) are kept. Junction points fully inside one text box
    are removed for the same reason. Evidence: offline replay on the r08
    candidate (work/module-opt80-20261007/r08/run-001/r08g-replay2.json) took
    graph pairs from TP40/FP180 to TP24/FP39.
    """
    if margin_px < 0 or not math.isfinite(margin_px):
        raise ValueError("glyph margin must be finite and non-negative")
    text_boxes = [
        (text.bbox.x1, text.bbox.y1, text.bbox.x2, text.bbox.y2) for text in ir.texts
    ]

    def _inside_one_box(point_a: Point, point_b: Point) -> bool:
        for x1, y1, x2, y2 in text_boxes:
            if all(
                x1 - margin_px <= p.x <= x2 + margin_px
                and y1 - margin_px <= p.y <= y2 + margin_px
                for p in (point_a, point_b)
            ):
                return True
        return False

    removed_segments = [
        s for s in ir.segments if _inside_one_box(s.start, s.end)
    ]
    removed_ids = {s.internal_id for s in removed_segments}
    kept_junctions = [
        j for j in ir.junctions if not _inside_one_box(j.point, j.point)
    ]
    removed_junctions = len(ir.junctions) - len(kept_junctions)
    if not removed_ids and not removed_junctions:
        return {"suppressed": 0, "junctions_suppressed": 0, "ids": []}
    ir.segments = [s for s in ir.segments if s.internal_id not in removed_ids]
    ir.junctions = kept_junctions
    return {
        "suppressed": len(removed_ids),
        "junctions_suppressed": removed_junctions,
        "ids": sorted(removed_ids),
    }


def _decor_point_seg_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _decor_seg_seg_dist(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx_: float, dy_: float,
) -> float:
    def _cross(ox: float, oy: float, px: float, py: float, qx: float, qy: float) -> float:
        return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)

    d1 = _cross(cx, cy, dx_, dy_, ax, ay)
    d2 = _cross(cx, cy, dx_, dy_, bx, by)
    d3 = _cross(ax, ay, bx, by, cx, cy)
    d4 = _cross(ax, ay, bx, by, dx_, dy_)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return 0.0
    return min(
        _decor_point_seg_dist(ax, ay, cx, cy, dx_, dy_),
        _decor_point_seg_dist(bx, by, cx, cy, dx_, dy_),
        _decor_point_seg_dist(cx, cy, ax, ay, bx, by),
        _decor_point_seg_dist(dx_, dy_, ax, ay, bx, by),
    )


def _decor_merged_lines(
    segments: Sequence[WireSegment],
    labels: Sequence[Any],
    boxes: Sequence[tuple[float, float, float, float]],
    *,
    lat_tol: float,
    gap_tol: float,
    axis_tol: float,
) -> list[dict[str, Any]]:
    """Merge collinear axis-aligned segments into line records with anchors.

    S09 task-7 (performance, semantics frozen): the greedy merge below is
    bit-for-bit the same decision procedure as before, but the "scan every
    item again until the cluster stops growing" loop is bounded to the items
    inside the leader's lateral band.  Three facts make that a pure constant
    factor win rather than a semantic change:

    * an item can join the cluster only when
      ``abs(pos2 - pos_leader) <= lat_tol`` (the original compares against the
      leader's ``pos``, never against a running mean), and
    * ``items`` is sorted by ``round(pos, 1)``, so those items are one
      contiguous run of the sorted list; and
    * ``cur_lo``/``cur_hi`` only ever widen, so the set of absorbable items is
      monotone in the scan - it is already maximal after the first pass, which
      is why the original's ``while changed`` outer loop never had to run a
      second time with new members.

    The label/box distance scan still visits members and references in their
    original order, so the ``min`` reductions see the same operands in the
    same sequence.
    """
    lines: list[dict[str, Any]] = []
    for orient in ("h", "v"):
        items: list[tuple[float, float, float, WireSegment]] = []
        for segment in segments:
            ax, ay = segment.start.x, segment.start.y
            bx, by = segment.end.x, segment.end.y
            if orient == "h" and abs(by - ay) <= axis_tol:
                items.append((min(ax, bx), max(ax, bx), (ay + by) / 2.0, segment))
            elif orient == "v" and abs(bx - ax) <= axis_tol:
                items.append((min(ay, by), max(ay, by), (ax + bx) / 2.0, segment))
        items.sort(key=lambda item: (round(item[2], 1), item[0]))
        # Reachability band for the cluster scan.  An item can only join the
        # cluster when ``abs(pos2 - pos_leader) <= lat_tol``, and the original
        # always compares against the leader's ``pos`` (never a running mean).
        # ``items`` is sorted by the geometric position (the ``round(pos, 1)``
        # key is monotone), so the eligible items form one contiguous range that
        # may include items *before* the leader - the original's fixed point
        # scans the whole list in index order, so a lower-index neighbour really
        # is absorbed.  Both edges come from bisection on the position column;
        # the ``abs(pos2 - pos) <= lat_tol`` test itself is still evaluated
        # inside the loop exactly as before.
        #
        # The rescan is kept verbatim: ``cur_lo``/``cur_hi`` widen as members
        # join, and widening can make an *earlier-indexed* candidate reachable
        # later (real 0092 has exactly that shape), so a single pass would drop
        # members.  Restricting the rescan to the band is the whole saving - the
        # scan the original wasted is the part that provably fails the lateral
        # test, and out-of-band items are never marked used by this cluster, so
        # the remaining clusters are seeded identically.
        pos_values = [item[2] for item in items]
        used = [False] * len(items)
        for index in range(len(items)):
            if used[index]:
                continue
            used[index] = True
            base_lo, base_hi, pos, segment = items[index]
            members = [segment]
            cur_lo, cur_hi = base_lo, base_hi
            first = bisect.bisect_left(pos_values, pos - lat_tol)
            if pos + lat_tol >= pos_values[-1]:
                last = len(items) - 1
            else:
                last = bisect.bisect_right(pos_values, pos + lat_tol) - 1
            changed = True
            while changed:
                changed = False
                for other_index in range(first, last + 1):
                    if used[other_index]:
                        continue
                    lo2, hi2, pos2, other = items[other_index]
                    if abs(pos2 - pos) <= lat_tol and not (
                        hi2 < cur_lo - gap_tol or lo2 > cur_hi + gap_tol
                    ):
                        used[other_index] = True
                        members.append(other)
                        cur_lo, cur_hi = min(cur_lo, lo2), max(cur_hi, hi2)
                        changed = True
            label_dist = 1e9
            comp_dist = 1e9
            for member in members:
                m_ax, m_ay = member.start.x, member.start.y
                m_bx, m_by = member.end.x, member.end.y
                # Cheap AABB rejection: ``label_dist`` only ever decreases, so a
                # label whose bounding box is already farther than the current
                # best cannot lower it.  ``_decor_point_seg_dist`` >= the
                # point-to-AABB distance, so the filter never drops the true
                # minimum, and the surviving calls still run in the original
                # order with the original ``min`` reduction.
                m_lo_x, m_hi_x = (m_ax, m_bx) if m_ax <= m_bx else (m_bx, m_ax)
                m_lo_y, m_hi_y = (m_ay, m_by) if m_ay <= m_by else (m_by, m_ay)
                if label_dist < 1e9:
                    for label in labels:
                        lp = label.point
                        dx = m_lo_x - lp.x if lp.x < m_lo_x else (
                            lp.x - m_hi_x if lp.x > m_hi_x else 0.0)
                        dy = m_lo_y - lp.y if lp.y < m_lo_y else (
                            lp.y - m_hi_y if lp.y > m_hi_y else 0.0)
                        if dx * dx + dy * dy > label_dist * label_dist:
                            continue
                        label_dist = min(
                            label_dist,
                            _decor_point_seg_dist(
                                lp.x, lp.y, m_ax, m_ay, m_bx, m_by,
                            ),
                        )
                else:
                    for label in labels:
                        label_dist = min(
                            label_dist,
                            _decor_point_seg_dist(
                                label.point.x, label.point.y,
                                m_ax, m_ay, m_bx, m_by,
                            ),
                        )
                if comp_dist < 1e9:
                    for x1, y1, x2, y2 in boxes:
                        dx = m_lo_x - x2 if x2 < m_lo_x else (
                            x1 - m_hi_x if x1 > m_hi_x else 0.0)
                        dy = m_lo_y - y2 if y2 < m_lo_y else (
                            y1 - m_hi_y if y1 > m_hi_y else 0.0)
                        if dx * dx + dy * dy > comp_dist * comp_dist:
                            continue
                        ex1, ey1 = x1, y1
                        ex2, ey2 = x2, y1
                        comp_dist = min(
                            comp_dist,
                            _decor_seg_seg_dist(m_ax, m_ay, m_bx, m_by,
                                                ex1, ey1, ex2, ey2),
                        )
                        ex1, ey1 = x2, y1
                        ex2, ey2 = x2, y2
                        comp_dist = min(
                            comp_dist,
                            _decor_seg_seg_dist(m_ax, m_ay, m_bx, m_by,
                                                ex1, ey1, ex2, ey2),
                        )
                        ex1, ey1 = x2, y2
                        ex2, ey2 = x1, y2
                        comp_dist = min(
                            comp_dist,
                            _decor_seg_seg_dist(m_ax, m_ay, m_bx, m_by,
                                                ex1, ey1, ex2, ey2),
                        )
                        ex1, ey1 = x1, y2
                        ex2, ey2 = x1, y1
                        comp_dist = min(
                            comp_dist,
                            _decor_seg_seg_dist(m_ax, m_ay, m_bx, m_by,
                                                ex1, ey1, ex2, ey2),
                        )
                else:
                    for x1, y1, x2, y2 in boxes:
                        for ex1, ey1, ex2, ey2 in (
                            (x1, y1, x2, y1), (x2, y1, x2, y2),
                            (x2, y2, x1, y2), (x1, y2, x1, y1),
                        ):
                            comp_dist = min(
                                comp_dist,
                                _decor_seg_seg_dist(m_ax, m_ay, m_bx, m_by,
                                                    ex1, ey1, ex2, ey2),
                            )
            lines.append({
                "orient": orient,
                "pos": pos,
                "lo": cur_lo,
                "hi": cur_hi,
                "span": cur_hi - cur_lo,
                "members": members,
                "label_dist": label_dist,
                "comp_dist": comp_dist,
            })
    return lines


def _decor_rect_outline_lines(
    lines: Sequence[dict[str, Any]],
    *,
    rect_min_area_px: float,
    rect_corner_tol_px: float,
    rect_min_cover: float,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Return the rectangle-outline quadruples the fold enumeration would emit.

    S09 task-7 (performance, semantics frozen).  The original walks every
    ``(h1, h2, v1, v2)`` quadruple.  Every bound below is a *necessary*
    condition of a test the original performs, so it can only remove quadruples
    the original would also have rejected; the original's tests are still
    evaluated, in the original order, on the candidates that survive.

    With ``x1 = v1.pos``, ``x2 = v2.pos``, ``y1 = h1.pos``, ``y2 = h2.pos``,
    ``w = x2 - x1`` and ``h = y2 - y1``, the original requires

      A) ``w * h >= rect_min_area_px``
      B) ``cover * w <= h1.span`` and ``cover * w <= h2.span``
      C) ``cover * h <= v1.span`` and ``cover * h <= v2.span``
      D) ``|h1.lo - x1| <= tol``, ``|h1.hi - x2| <= tol`` (likewise for ``h2``)
      E) ``|v.lo - y1| <= tol``, ``|v.hi - y2| <= tol`` (likewise for ``v2``)

    From B and C: ``w <= max(h1.span, h2.span) / cover`` (``span_dom``) and
    ``h <= max(v1.span, v2.span) / cover``.  From D: ``x1 >= h1.lo - tol`` and
    ``x2 <= h1.hi + tol``, so ``w <= h1.span + 2*tol``.  Combining A with the
    ``h <= 1200`` window gives ``w >= rect_min_area_px / 1200``.  Hence

      ``x1 >= h1.lo - tol``,  ``x2 <= h1.hi + tol``,
      ``w = x2 - x1`` with ``max(5, rect_min_area_px / 1200) <= w <= span_dom``

    which is exactly the two bisection windows used below: ``v1`` is searched in
    ``[h1.lo - tol, h1.hi + tol - w_min]`` and, for each ``v1``, ``v2`` in
    ``(v1.pos, h1.hi + tol]``.  E additionally needs ``v.hi >= y2 - tol`` for
    both verticals (the ``v_hi_bound`` test).

    Earlier revisions searched ``v1`` in a window derived only from ``h1.lo``,
    ``h1.hi`` and ``h2`` and then re-tested the width inside the inner loop;
    that still scanned a large fraction of ``vertical`` per horizontal pair and
    measured *no* speed-up at 1000 real segments.  The ``w`` window above is
    what actually bounds the work.
    """
    order = {id(line): index for index, line in enumerate(lines)}
    horizontal = sorted(
        [ln for ln in lines if ln["orient"] == "h" and ln["span"] >= 30],
        key=lambda ln: (ln["pos"], order[id(ln)]),
    )
    vertical = sorted(
        [ln for ln in lines if ln["orient"] == "v" and ln["span"] >= 30],
        key=lambda ln: (ln["pos"], order[id(ln)]),
    )
    if not horizontal or not vertical:
        return []

    v_positions = [ln["pos"] for ln in vertical]
    v_hi_all = [ln["hi"] for ln in vertical]
    v_spans = [ln["span"] for ln in vertical]
    cover = rect_min_cover
    tol = rect_corner_tol_px
    w_min = rect_min_area_px / 1200.0
    if w_min < 5.0:
        w_min = 5.0

    marked: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for h1_index, h1 in enumerate(horizontal):
        h1_pos, h1_lo, h1_hi, h1_span = h1["pos"], h1["lo"], h1["hi"], h1["span"]
        # D bounds x1 from below and x2 from above; A+B bound their difference.
        x1_lo = h1_lo - tol
        x2_hi = h1_hi + tol
        for h2_index in range(h1_index + 1, len(horizontal)):
            h2 = horizontal[h2_index]
            gap_h = h2["pos"] - h1_pos
            if gap_h > 1200.0:
                break                       # position-ordered: no later h2 qualifies
            if not gap_h > 5.0:
                continue
            # A leaves room for a rectangle only if w * gap_h can reach the area.
            if gap_h * (x2_hi - x1_lo) < rect_min_area_px:
                continue
            h2_span = h2["span"]
            span_dom = (h1_span if h1_span > h2_span else h2_span) / cover
            # A also needs w >= rect_min_area_px / gap_h, and gap_h <= 1200 turns
            # that into the global floor w_min.
            w_hi = x2_hi - x1_lo
            if span_dom < w_hi:
                w_hi = span_dom
            if w_hi < w_min:
                continue
            v_hi_bound = gap_h + h1_pos - tol      # E: v.hi >= y2 - tol
            v1_first = bisect.bisect_left(v_positions, x1_lo)
            v1_last = bisect.bisect_right(v_positions, x2_hi - w_min)
            if v1_last <= v1_first:
                continue
            for v1_index in range(v1_first, v1_last):
                v1 = vertical[v1_index]
                v1_pos = v1["pos"]
                # Only the upper endpoint is constrained by y2 - tol.
                # The old corner predicate permits lo > y2 - tol when the
                # vertical span is still at least 30 and both corners are
                # within tolerance.
                if not (v_hi_all[v1_index] >= v_hi_bound):
                    continue
                v2_lo = v1_pos + w_min
                # The legacy rectangle predicate also caps width at 1200 px.
                v2_hi = min(x2_hi, v1_pos + 1200.0)
                if v2_lo > v2_hi:
                    continue
                w_cap = span_dom
                w_room = x2_hi - v1_pos
                if w_room < w_cap:
                    w_cap = w_room
                # Include equality at the area-derived lower bound; the legacy
                # strict width > 5 predicate is checked inside the candidate loop.
                v2_first = bisect.bisect_left(v_positions, v2_lo, v1_index + 1)
                v2_last = bisect.bisect_right(v_positions, v2_hi, v2_first)
                for v2_index in range(v2_first, v2_last):
                    rect_width = v_positions[v2_index] - v1_pos
                    if rect_width > w_cap:
                        break
                    if not (v_hi_all[v2_index] >= v_hi_bound):
                        continue
                    v2 = vertical[v2_index]
                    # Enforce the complete legacy width range after subtraction;
                    # the indexed upper bound alone can round at translated coords.
                    if not (5.0 < rect_width <= 1200.0):
                        continue
                    if rect_width * gap_h < rect_min_area_px:
                        continue
                    x1, x2 = v1_pos, v2["pos"]
                    if not abs(h1_lo - x1) <= tol:
                        continue
                    if not abs(h1_hi - x2) <= tol:
                        continue
                    if not abs(h2["lo"] - x1) <= tol:
                        continue
                    if not abs(h2["hi"] - x2) <= tol:
                        continue
                    # ``v`` records carry a y-extent: the original tests
                    # ``|v.lo - y1| <= tol`` and ``|v.hi - y2| <= tol`` with
                    # ``y1 = h1.pos`` and ``y2 = h2.pos``.  The parentheses are
                    # load-bearing: without them Python evaluates
                    # ``(v.lo - gap_h) + h1_pos``.
                    y1 = h1_pos
                    y2 = gap_h + h1_pos
                    if not abs(v1["lo"] - y1) <= tol:
                        continue
                    if not abs(v1["hi"] - y2) <= tol:
                        continue
                    if not abs(v2["lo"] - y1) <= tol:
                        continue
                    if not abs(v2["hi"] - y2) <= tol:
                        continue
                    if (
                        h1_span < cover * rect_width
                        or h2_span < cover * rect_width
                        or v1["span"] < cover * gap_h
                        or v2["span"] < cover * gap_h
                    ):
                        continue
                    marked.append((h1, h2, v1, v2))
    return marked


def filter_decorative_wire_segments(
    ir: CircuitIR,
    *,
    canvas_width: int | None = None,
    canvas_height: int | None = None,
    span_fraction: float = 0.8,
    span_far_label_px: float = 20.0,
    span_far_comp_px: float = 20.0,
    rect_min_area_px: float = 500.0,
    rect_corner_tol_px: float = 15.0,
    rect_min_cover: float = 0.4,
    companion_max_gap_px: float = 40.0,
    companion_far_label_px: float = 20.0,
    companion_far_comp_px: float = 20.0,
    interior_inset_px: float = 5.0,
    merge_lat_tol_px: float = 4.0,
    merge_gap_tol_px: float = 30.0,
    axis_tol_px: float = 2.0,
) -> dict[str, Any]:
    """Remove sheet-furniture and symbol strokes from the wire segment set.

    opt80 r12: the wire mask also fires on page furniture (frames, section
    dividers, title-block tables) and component symbol outlines (chip body
    rectangles), and those strokes stay in ``ir.segments`` as the dominant
    wire false-positive pixel mass (four-layer attribution in
    work/module-opt80-20261007/r12-wire/r12a-attribution.json). Four
    structural signatures separate them from wires without image or GT
    access:

    * ``canvas_span``  - an axis-aligned line covering >= ``span_fraction`` of
      the canvas dimension while staying far from OCR labels and component
      bboxes (page frame / section divider);
    * ``companion``    - its parallel neighbour (double-line sheet borders);
    * ``rect_outline`` - merged lines closing a rectangle of at least
      ``rect_min_area_px`` (chip body outlines / symbol boxes; wire networks
      are cycle-free at this scale);
    * ``interior``     - a stroke with both endpoints at least
      ``interior_inset_px`` deep inside a component bbox (symbol strokes live
      inside bodies, wires attach at their edges).

    Conservative guards: every structural test is geometric, deterministic
    and keyed to canvas/component geometry, so real wires that merely run
    long or parallel are kept (the "far from label/component" gates). The
    same contract as :func:`filter_body_outline_wire_segments` applies:
    suppressed segments are removed from ``ir.segments`` and a report with
    per-layer counts is returned for trace. Offline replay on the r11
    candidate (work/module-opt80-20261007/r12-wire/r12i-t3.json) moves wire
    precision 0.543 -> 0.691 at recall 0.866 (F1 0.6685 -> 0.7687) and cuts
    graph pairwise FP 12 -> 8 while TP stays at 25.
    """
    for name, value in (("span_fraction", span_fraction), ("rect_min_cover", rect_min_cover)):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
    for name, value in (
        ("span_far_label_px", span_far_label_px),
        ("span_far_comp_px", span_far_comp_px),
        ("rect_min_area_px", rect_min_area_px),
        ("rect_corner_tol_px", rect_corner_tol_px),
        ("companion_max_gap_px", companion_max_gap_px),
        ("companion_far_label_px", companion_far_label_px),
        ("companion_far_comp_px", companion_far_comp_px),
        ("interior_inset_px", interior_inset_px),
        ("merge_lat_tol_px", merge_lat_tol_px),
        ("merge_gap_tol_px", merge_gap_tol_px),
        ("axis_tol_px", axis_tol_px),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    width = canvas_width if canvas_width is not None else ir.image.width
    height = canvas_height if canvas_height is not None else ir.image.height
    if width <= 0 or height <= 0:
        raise ValueError("canvas dimensions must be positive")

    segments = list(ir.segments)
    labels = [label for label in ir.net_labels if label.point is not None]
    boxes = [
        (c.bbox.x1, c.bbox.y1, c.bbox.x2, c.bbox.y2)
        for c in ir.components
        if c.bbox.x2 > c.bbox.x1 and c.bbox.y2 > c.bbox.y1
    ]
    lines = _decor_merged_lines(
        segments, labels, boxes,
        lat_tol=merge_lat_tol_px, gap_tol=merge_gap_tol_px, axis_tol=axis_tol_px,
    )
    layers: dict[str, set[str]] = {}

    def _mark(layer: str, segment: WireSegment) -> None:
        layers.setdefault(layer, set()).add(segment.internal_id)

    span_lines: list[dict[str, Any]] = []
    for line in lines:
        dim = width if line["orient"] == "h" else height
        if (
            line["span"] >= span_fraction * dim
            and line["label_dist"] > span_far_label_px
            and line["comp_dist"] > span_far_comp_px
        ):
            span_lines.append(line)
            for member in line["members"]:
                _mark("canvas_span", member)

    # S09 task-7: identity set instead of a list scan (the original used
    # ``line in span_lines``, an element-by-element ``==`` over dicts, which is
    # both O(len(span_lines)) and content-based).  Only the elements appended
    # below can ever be tested, so identity answers exactly the same question.
    span_line_ids = {id(line) for line in span_lines}

    for line in lines:
        if id(line) in span_line_ids:
            continue
        line_orient = line["orient"]
        line_pos, line_lo, line_hi = line["pos"], line["lo"], line["hi"]
        line_span = line["span"]
        # The two distance gates and the gap gate are per-line constants; the
        # original evaluated them with ``and`` in one chain, so hoisting them
        # ahead of the anchor loop (and leaving ``overlap`` inside) keeps the
        # same first-failing-condition and the same set of marked lines.
        if not (
            line["label_dist"] > companion_far_label_px
            and line["comp_dist"] > companion_far_comp_px
        ):
            continue
        half_span = 0.5 * line_span
        for anchor in span_lines:
            if line_orient != anchor["orient"]:
                continue
            if abs(line_pos - anchor["pos"]) > companion_max_gap_px:
                continue
            if min(line_hi, anchor["hi"]) - max(line_lo, anchor["lo"]) < half_span:
                continue
            if line_span < 0.25 * anchor["span"]:
                continue
            for member in line["members"]:
                _mark("companion", member)
            break

    for h1, h2, v1, v2 in _decor_rect_outline_lines(
        lines,
        rect_min_area_px=rect_min_area_px,
        rect_corner_tol_px=rect_corner_tol_px,
        rect_min_cover=rect_min_cover,
    ):
        for line in (h1, h2, v1, v2):
            for member in line["members"]:
                _mark("rect_outline", member)

    for segment in segments:
        seg_x1, seg_y1 = segment.start.x, segment.start.y
        seg_x2, seg_y2 = segment.end.x, segment.end.y
        seg_lo_x, seg_hi_x = (seg_x1, seg_x2) if seg_x1 <= seg_x2 else (seg_x2, seg_x1)
        seg_lo_y, seg_hi_y = (seg_y1, seg_y2) if seg_y1 <= seg_y2 else (seg_y2, seg_y1)
        for x1, y1, x2, y2 in boxes:
            # The original tests the two endpoints against the box; a box whose
            # x-range misses the segment's (or whose y-range does) cannot
            # contain both endpoints, so this only skips boxes whose endpoint
            # test was going to fail.
            if x2 < seg_lo_x or x1 > seg_hi_x or y2 < seg_lo_y or y1 > seg_hi_y:
                continue
            if (
                x1 <= seg_x1 <= x2 and y1 <= seg_y1 <= y2
                and x1 <= seg_x2 <= x2 and y1 <= seg_y2 <= y2
            ):
                inset = min(
                    seg_x1 - x1, x2 - seg_x1,
                    seg_y1 - y1, y2 - seg_y1,
                    seg_x2 - x1, x2 - seg_x2,
                    seg_y2 - y1, y2 - seg_y2,
                )
                if inset >= interior_inset_px:
                    _mark("interior", segment)
                break

    suppressed: set[str] = set()
    for ids in layers.values():
        suppressed |= ids
    if not suppressed:
        return {"suppressed": 0, "ids": [], "layers": {}, "junctions_suppressed": 0}
    ir.segments = [s for s in ir.segments if s.internal_id not in suppressed]
    # Derived-reference hygiene: every structure that stores segment ids must
    # forget the suppressed strokes, or ReferenceValidator blocks the output
    # with dangling-reference errors (the same contract as the palette filter).
    for pin in ir.pins:
        pin.edge_ids = [
            edge_id for edge_id in pin.edge_ids if edge_id not in suppressed
        ]
    minimum_branches = {"t_junction": 3, "connected_cross": 4}
    kept_junctions = []
    removed_junctions = 0
    for junction in ir.junctions:
        junction.segment_ids = [
            segment_id
            for segment_id in junction.segment_ids
            if segment_id not in suppressed
        ]
        branch_count = len(set(junction.segment_ids))
        minimum = minimum_branches.get(junction.kind)
        if branch_count == 0 or (minimum is not None and branch_count < minimum):
            removed_junctions += 1
            continue
        kept_junctions.append(junction)
    ir.junctions = kept_junctions
    return {
        "suppressed": len(suppressed),
        "ids": sorted(suppressed),
        "junctions_suppressed": removed_junctions,
        "layers": {name: len(ids) for name, ids in sorted(layers.items())},
    }


def build_physical_graph(
    ir: CircuitIR,
    *,
    endpoint_tolerance_px: float = 3.0,
    collinear_merge_tolerance_px: float = 0.0,
    collinear_gap_merge_tolerance_px: float = 0.0,
    pin_snap_min_px: float = 2.0,
    pin_snap_tolerance_px: float = 6.0,
    pin_point_projection_max_px: float = 0.0,
    unique_margin: float = 1.5,
    direct_pin_tolerance_px: float = 1.0,
    label_attach_tolerance_px: float = 8.0,
    gap_repair_max_px: float = HARD_MAX_GAP_REPAIR_PX,
    connection_evidence_out: list[ConnectionEvidence] | None = None,
) -> tuple[list[PhysicalNet], list[SnapDecision]]:
    if (
        endpoint_tolerance_px < 0
        or collinear_merge_tolerance_px < 0
        or collinear_gap_merge_tolerance_px < 0
        or pin_snap_min_px < 0
        or pin_snap_tolerance_px < 0
        or pin_point_projection_max_px < 0
    ):
        raise ValueError("graph tolerances must be non-negative")
    if not all(
        math.isfinite(value)
        for value in (
            endpoint_tolerance_px,
            collinear_merge_tolerance_px,
            collinear_gap_merge_tolerance_px,
            pin_point_projection_max_px,
        )
    ):
        raise ValueError("graph tolerances must be finite")
    if pin_snap_min_px > pin_snap_tolerance_px:
        raise ValueError("pin_snap_min_px cannot exceed pin_snap_tolerance_px")
    params = ConnectionParams(
        endpoint_tolerance_px=endpoint_tolerance_px,
        collinear_merge_tolerance_px=collinear_merge_tolerance_px,
        collinear_gap_merge_tolerance_px=collinear_gap_merge_tolerance_px,
        gap_repair_max_px=gap_repair_max_px,
    )
    uf = UnionFind()
    segment_tokens: dict[str, str] = {}
    for segment in ir.segments:
        token = f"segment:{segment.internal_id}"
        segment_tokens[segment.internal_id] = token
        uf.add(token)

    connection_evidence = collect_segment_connections(
        ir.segments,
        endpoint_tolerance_px=params.endpoint_tolerance_px,
        collinear_merge_tolerance_px=params.collinear_merge_tolerance_px,
        collinear_gap_merge_tolerance_px=params.collinear_gap_merge_tolerance_px,
        gap_repair_max_px=params.gap_repair_max_px,
    )
    for item in connection_evidence:
        if item.left_id in segment_tokens and item.right_id in segment_tokens:
            uf.union(segment_tokens[item.left_id], segment_tokens[item.right_id])

    for junction in ir.junctions:
        # Only semantic connection classes may bridge channel-separated
        # segments.  In particular, an ``endpoint`` classifier result is not
        # positive connectivity evidence even when its candidate patch happens
        # to list several nearby segments.  Treating every class except the
        # explicit negative classes as connected produced physical nets that
        # the validator (correctly) reported as disconnected.
        if junction.kind not in {"t_junction", "connected_cross", "bend"}:
            continue
        known = [segment_id for segment_id in junction.segment_ids if segment_id in segment_tokens]
        for segment_id in known[1:]:
            uf.union(segment_tokens[known[0]], segment_tokens[segment_id])
            left, right = sorted((known[0], segment_id))
            connection_evidence.append(
                ConnectionEvidence(
                    left_id=left,
                    right_id=right,
                    rule="junction_bridge",
                    confidence=junction.score,
                    distance_px=0.0,
                    virtual=False,
                    metadata={
                        "junction_id": junction.internal_id,
                        "junction_kind": junction.kind,
                    },
                )
            )

    by_source_edge: dict[str, list[str]] = defaultdict(list)
    for segment in ir.segments:
        if segment.source_edge_id:
            by_source_edge[segment.source_edge_id].append(segment.internal_id)

    explicit_pin_edges: set[str] = set()
    declared_pin_edges = {pin.internal_id for pin in ir.pins if pin.edge_ids}
    for pin in ir.pins:
        pin_token = f"pin:{pin.internal_id}"
        uf.add(pin_token)
        direct_segments = [segment_id for segment_id in pin.edge_ids if segment_id in segment_tokens]
        source_edges = [edge_id for edge_id in pin.edge_ids if edge_id in by_source_edge]
        for segment_id in direct_segments:
            segment = ir.segment_index()[segment_id]
            if _explicit_membership_is_trusted(pin, segment, pin_snap_tolerance_px):
                uf.union(pin_token, segment_tokens[segment_id])
                connection_evidence.append(
                    ConnectionEvidence(
                        left_id=pin.internal_id,
                        right_id=segment_id,
                        rule="pin_explicit_edge",
                        confidence=1.0,
                        distance_px=point_segment_distance(pin.point, segment),
                        channel=segment.channel,
                        virtual=False,
                        left_kind="pin",
                        right_kind="segment",
                    )
                )
        for edge_id in source_edges:
            segment = ir.segment_index()[by_source_edge[edge_id][0]]
            if _explicit_membership_is_trusted(pin, segment, pin_snap_tolerance_px):
                uf.union(pin_token, segment_tokens[segment.internal_id])
                connection_evidence.append(
                    ConnectionEvidence(
                        left_id=pin.internal_id,
                        right_id=segment.internal_id,
                        rule="pin_source_edge",
                        confidence=1.0,
                        distance_px=point_segment_distance(pin.point, segment),
                        channel=segment.channel,
                        virtual=False,
                        left_kind="pin",
                        right_kind="segment",
                        metadata={"source_edge_id": edge_id},
                    )
                )
        if any(
            _explicit_membership_is_trusted(pin, ir.segment_index()[segment_id], pin_snap_tolerance_px)
            for segment_id in direct_segments
        ) or any(
            _explicit_membership_is_trusted(
                pin,
                ir.segment_index()[by_source_edge[edge_id][0]],
                pin_snap_tolerance_px,
            )
            for edge_id in source_edges
        ):
            explicit_pin_edges.add(pin.internal_id)

    # Correct only a uniquely accepted, very small Pin-to-Wire offset.  This is
    # opt-in because the point is also part of the official Pin geometry.
    if pin_point_projection_max_px > 0:
        pre_snap_decisions = snap_pins(
            ir.pins,
            ir.segments,
            line_width=max(2.0, pin_snap_tolerance_px / 1.5),
            min_px=pin_snap_min_px,
            max_px=pin_snap_tolerance_px,
            unique_margin=unique_margin,
            segment_groups={
                segment_id: str(uf.find(token))
                for segment_id, token in segment_tokens.items()
            },
        )
        _project_pin_points_to_wire(
            ir.pins,
            ir.segment_index(),
            pre_snap_decisions,
            max_distance_px=pin_point_projection_max_px,
        )
    # A direct pin-to-pin connection has no wire pixels to snap.  Only pins from
    # distinct known components are eligible, and the tolerance is deliberately
    # tighter than the wire snapping tolerance.
    for left_id, right_id, distance in _direct_pin_pairs(
        ir.pins,
        tolerance=direct_pin_tolerance_px,
        pins_with_explicit_edges=declared_pin_edges,
    ):
        uf.union(f"pin:{left_id}", f"pin:{right_id}")
        connection_evidence.append(
            ConnectionEvidence(
                left_id=left_id,
                right_id=right_id,
                rule="direct_pin_contact",
                confidence=1.0,
                distance_px=distance,
                virtual=False,
                left_kind="pin",
                right_kind="pin",
            )
        )

    snap_decisions = snap_pins(
        ir.pins,
        ir.segments,
        line_width=max(2.0, pin_snap_tolerance_px / 1.5),
        min_px=pin_snap_min_px,
        max_px=pin_snap_tolerance_px,
        unique_margin=unique_margin,
        segment_groups={
            segment_id: str(uf.find(token))
            for segment_id, token in segment_tokens.items()
        },
    )
    for decision in snap_decisions:
        pin_token = f"pin:{decision.pin_id}"
        if (
            decision.pin_id not in explicit_pin_edges
            and not ir.pin_index()[decision.pin_id].edge_ids
            and decision.status == "accepted"
            and decision.segment_id
        ):
            uf.union(pin_token, segment_tokens[decision.segment_id])
            connection_evidence.append(
                ConnectionEvidence(
                    left_id=decision.pin_id,
                    right_id=decision.segment_id,
                    rule="pin_snap",
                    confidence=1.0 - min(1.0, (decision.cost or 0.0) / max(pin_snap_tolerance_px, 1e-6)),
                    distance_px=decision.cost or 0.0,
                    virtual=False,
                    left_kind="pin",
                    right_kind="segment",
                    metadata={"snap_margin": decision.margin},
                )
            )

    groups = uf.groups()
    nets: list[PhysicalNet] = []
    token_to_net: dict[str, str] = {}
    for index, group in enumerate(groups, start=1):
        pins = sorted(token.removeprefix("pin:") for token in group if str(token).startswith("pin:"))
        segments = sorted(token.removeprefix("segment:") for token in group if str(token).startswith("segment:"))
        if not pins and not segments:
            continue
        net_id = f"pnet_{index:05d}"
        member_ids = set(pins) | set(segments)
        internal_links = [
            item
            for item in connection_evidence
            if item.left_id in member_ids and item.right_id in member_ids
        ]
        virtual_links = [item for item in internal_links if item.virtual]
        repair_ratio = (len(virtual_links) / len(internal_links)) if internal_links else 0.0
        nets.append(
            PhysicalNet(
                internal_id=net_id,
                pins=pins,
                segments=segments,
                repair_ratio=repair_ratio,
            )
        )
        for token in group:
            token_to_net[str(token)] = net_id

    # Attach labels by source-edge hint, then by a unique nearby segment.
    source_edge_to_net: dict[str, str] = {}
    for segment in ir.segments:
        if segment.source_edge_id:
            net_id = token_to_net.get(segment_tokens[segment.internal_id])
            if net_id:
                source_edge_to_net[segment.source_edge_id] = net_id
    net_index = {net.internal_id: net for net in nets}
    for label in ir.net_labels:
        if label.attached_physical_net_id not in net_index:
            label.attached_physical_net_id = None
        hinted_edge = next((item for item in label.evidence if item.metadata.get("source_edge_id")), None)
        if hinted_edge:
            label.attached_physical_net_id = source_edge_to_net.get(str(hinted_edge.metadata["source_edge_id"]))
        if not label.attached_physical_net_id and label.point is not None:
            ranked = sorted(
                (
                    point_segment_distance(label.point, segment),
                    segment.internal_id,
                )
                for segment in ir.segments
            )
            if ranked:
                best_distance, best_segment_id = ranked[0]
                second_distance = ranked[1][0] if len(ranked) > 1 else float("inf")
                if best_distance <= label_attach_tolerance_px and second_distance - best_distance >= unique_margin:
                    label.attached_physical_net_id = token_to_net.get(segment_tokens[best_segment_id])
        if label.attached_physical_net_id in net_index:
            net_index[label.attached_physical_net_id].labels.append(label.internal_id)
            connection_evidence.append(
                ConnectionEvidence(
                    left_id=label.internal_id,
                    right_id=label.attached_physical_net_id,
                    rule="label_attach",
                    confidence=label.score,
                    distance_px=0.0,
                    virtual=False,
                    left_kind="label",
                    right_kind="physical_net",
                    metadata={"net_type": label.net_type, "resolved_text": label.resolved_text},
                )
            )
    for net in nets:
        net.labels = sorted(set(net.labels))

    # Virtual gap repairs are recorded separately from solid-wire evidence.
    for item in connection_evidence:
        if not item.virtual:
            continue
        for segment_id in (item.left_id, item.right_id):
            segment = ir.segment_index().get(segment_id)
            if segment is None:
                continue
            evidence_id = f"repair:virtual-gap:{item.left_id}:{item.right_id}:{item.rule}"
            if any(existing.evidence_id == evidence_id for existing in segment.evidence):
                continue
            segment.evidence.append(
                Evidence(
                    evidence_id=evidence_id,
                    source="repair",
                    view_id="original",
                    score=max(0.0, min(1.0, item.confidence)),
                    payload_ref=item.right_id if segment_id == item.left_id else item.left_id,
                    metadata={
                        "kind": "virtual_gap_repair",
                        "rule": item.rule,
                        "distance_px": item.distance_px,
                        "virtual": True,
                        "solid_wire": False,
                    },
                )
            )

    if connection_evidence_out is not None:
        connection_evidence_out.extend(connection_evidence)
    ir.metadata["connection_evidence"] = [item.as_dict() for item in connection_evidence]
    ir.metadata["connection_params"] = {
        "endpoint_tolerance_px": params.endpoint_tolerance_px,
        "collinear_merge_tolerance_px": params.collinear_merge_tolerance_px,
        "collinear_gap_merge_tolerance_px": params.collinear_gap_merge_tolerance_px,
        "gap_repair_max_px": params.gap_repair_max_px,
        "max_collinear_lateral_offset_px": MAX_COLLINEAR_LATERAL_OFFSET_PX,
    }
    return nets, snap_decisions


def _segment_endpoints(segment: WireSegment) -> tuple[Point, Point]:
    return segment.start, segment.end


def _segments_touch(left: WireSegment, right: WireSegment, tolerance: float) -> bool:
    return any(
        math.hypot(a.x - b.x, a.y - b.y) <= tolerance
        for a in _segment_endpoints(left)
        for b in _segment_endpoints(right)
    )


def _touching_pairs(
    segment_ids: list[str],
    segment_index: dict[str, WireSegment],
    *,
    tolerance: float,
) -> list[tuple[str, str, float]]:
    cells: dict[tuple[int, int], list[tuple[str, Point]]] = defaultdict(list)
    cell_size = max(tolerance, 1e-6)
    pairs: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()
    for segment_id in sorted(segment_ids):
        if segment_id not in segment_index:
            continue
        for point in _segment_endpoints(segment_index[segment_id]):
            cx = math.floor(point.x / cell_size)
            cy = math.floor(point.y / cell_size)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for other_id, other_point in cells.get((cx + dx, cy + dy), []):
                        if other_id == segment_id:
                            continue
                        distance = math.hypot(point.x - other_point.x, point.y - other_point.y)
                        if distance <= tolerance:
                            key = tuple(sorted((segment_id, other_id)))
                            if key not in seen:
                                seen.add(key)
                                pairs.append((key[0], key[1], distance))
            cells[(cx, cy)].append((segment_id, point))
    return pairs


def _union_touching_segments(
    segment_ids: list[str],
    segment_index: dict[str, WireSegment],
    segment_tokens: dict[str, str],
    uf: UnionFind,
    *,
    tolerance: float,
) -> None:
    for left_id, right_id, _distance in _touching_pairs(
        segment_ids,
        segment_index,
        tolerance=tolerance,
    ):
        uf.union(segment_tokens[left_id], segment_tokens[right_id])


def collinear_longitudinal_gap(left: WireSegment, right: WireSegment) -> float:
    """Non-negative longitudinal gap along the left supporting line."""

    lx = left.end.x - left.start.x
    ly = left.end.y - left.start.y
    length = math.hypot(lx, ly)
    if length <= 1e-6:
        return float("inf")
    ux = lx / length
    uy = ly / length
    left_min, left_max = 0.0, length
    right_start_projection = (right.start.x - left.start.x) * ux + (right.start.y - left.start.y) * uy
    right_end_projection = (right.end.x - left.start.x) * ux + (right.end.y - left.start.y) * uy
    right_min = min(right_start_projection, right_end_projection)
    right_max = max(right_start_projection, right_end_projection)
    overlap = min(left_max, right_max) - max(left_min, right_min)
    if overlap > 0.0:
        return 0.0
    return max(left_min, right_min) - min(left_max, right_max)


def segments_have_collinear_overlap(
    left: WireSegment,
    right: WireSegment,
    tolerance: float,
    *,
    allow_gap: bool = False,
) -> bool:
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("collinear tolerance must be finite and non-negative")
    lx = left.end.x - left.start.x
    ly = left.end.y - left.start.y
    length = math.hypot(lx, ly)
    if length <= 1e-6:
        return False
    ux = lx / length
    uy = ly / length
    # Lateral alignment is always tightly bounded.  Longitudinal gap tolerance
    # must never authorize a near-parallel lateral merge.
    alignment_tolerance = min(tolerance, MAX_COLLINEAR_LATERAL_OFFSET_PX)
    max_offset = 0.0
    for point in (right.start, right.end):
        cross = (point.x - left.start.x) * uy - (point.y - left.start.y) * ux
        max_offset = max(max_offset, abs(cross))
        if abs(cross) > alignment_tolerance:
            return False
    right_length = math.hypot(
        right.end.x - right.start.x,
        right.end.y - right.start.y,
    )
    if right_length <= 1e-6:
        return False
    left_min, left_max = 0.0, length
    right_start_projection = (
        (right.start.x - left.start.x) * ux
        + (right.start.y - left.start.y) * uy
    )
    right_end_projection = (
        (right.end.x - left.start.x) * ux
        + (right.end.y - left.start.y) * uy
    )
    right_min = min(right_start_projection, right_end_projection)
    right_max = max(right_start_projection, right_end_projection)
    overlap = min(left_max, right_max) - max(left_min, right_min)
    if overlap > 0.0:
        return True
    if not allow_gap:
        return False
    gap = max(left_min, right_min) - min(left_max, right_max)
    return gap >= 0.0 and math.hypot(gap, max_offset) <= tolerance


def _project_pin_points_to_wire(
    pins: list[Pin],
    segment_index: dict[str, WireSegment],
    decisions: list[SnapDecision],
    *,
    max_distance_px: float,
) -> int:
    projected = 0
    pin_index = {pin.internal_id: pin for pin in pins}
    for decision in decisions:
        if decision.status != "accepted" or not decision.segment_id:
            continue
        pin = pin_index.get(decision.pin_id)
        segment = segment_index.get(decision.segment_id)
        if pin is None or segment is None or pin.edge_ids:
            continue
        distance = point_segment_distance(pin.point, segment)
        if not (1e-6 < distance <= max_distance_px):
            continue
        projection = decision.point or pin.point
        dx = projection.x - pin.point.x
        dy = projection.y - pin.point.y
        if not (math.isfinite(dx) and math.isfinite(dy)):
            continue
        pin.point = projection
        if pin.base is not None:
            pin.base = Point(x=pin.base.x + dx, y=pin.base.y + dy)
        evidence_id = f"geometry:pin-projection:{pin.internal_id}:{segment.internal_id}"
        if not any(item.evidence_id == evidence_id for item in pin.evidence):
            pin.evidence.append(
                Evidence(
                    evidence_id=evidence_id,
                    source="geometry",
                    view_id="original",
                    score=max(0.0, min(1.0, 1.0 - distance / max_distance_px)),
                    payload_ref=segment.internal_id,
                    metadata={
                        "kind": "pin-point-projection",
                        "segment_id": segment.internal_id,
                        "distance_px": distance,
                        "max_distance_px": max_distance_px,
                    },
                )
            )
        projected += 1
    return projected

def _explicit_membership_is_trusted(pin: Pin, segment: WireSegment, tolerance: float) -> bool:
    if point_segment_distance(pin.point, segment) <= tolerance:
        return True
    # Dataset annotations are an explicit oracle/debug contract.  Learned and
    # heuristic providers must still satisfy geometry before a reference is used.
    return any(item.source == "dataset" for item in pin.evidence)


def _direct_pin_pairs(
    pins: list[Pin],
    *,
    tolerance: float,
    pins_with_explicit_edges: set[str] | None = None,
) -> list[tuple[str, str, float]]:
    if tolerance < 0:
        return []
    cells: dict[tuple[int, int], list[Pin]] = defaultdict(list)
    cell_size = max(tolerance, 1e-6)
    explicit = pins_with_explicit_edges or set()
    pairs: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()
    for pin in sorted(pins, key=lambda item: item.internal_id):
        if pin.component_id is None:
            continue
        cx = math.floor(pin.point.x / cell_size)
        cy = math.floor(pin.point.y / cell_size)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in cells.get((cx + dx, cy + dy), []):
                    if other.component_id == pin.component_id:
                        continue
                    if pin.internal_id in explicit or other.internal_id in explicit:
                        continue
                    distance = math.hypot(pin.point.x - other.point.x, pin.point.y - other.point.y)
                    if distance <= tolerance:
                        key = tuple(sorted((pin.internal_id, other.internal_id)))
                        if key not in seen:
                            seen.add(key)
                            pairs.append((key[0], key[1], distance))
        cells[(cx, cy)].append(pin)
    return pairs


def _union_direct_pins(
    pins: list[Pin],
    uf: UnionFind,
    *,
    tolerance: float,
    pins_with_explicit_edges: set[str] | None = None,
) -> None:
    for left_id, right_id, _distance in _direct_pin_pairs(
        pins,
        tolerance=tolerance,
        pins_with_explicit_edges=pins_with_explicit_edges,
    ):
        uf.union(f"pin:{left_id}", f"pin:{right_id}")
