from __future__ import annotations

import math
from collections import defaultdict

from pcbparse.association.geometry import point_segment_distance
from pcbparse.association.pin_wire import SnapDecision, snap_pins
from pcbparse.graph.union_find import UnionFind
from pcbparse.ir.models import CircuitIR, Evidence, PhysicalNet, Pin, Point, WireSegment


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
    uf = UnionFind()
    segment_tokens: dict[str, str] = {}
    endpoints: list[tuple[str, Point, str]] = []
    for segment in ir.segments:
        token = f"segment:{segment.internal_id}"
        segment_tokens[segment.internal_id] = token
        uf.add(token)
        endpoints.append((segment.internal_id, segment.start, segment.channel))
        endpoints.append((segment.internal_id, segment.end, segment.channel))

    # Grid-indexed endpoint clustering keeps this near linear and respects crossing channels.
    cells: dict[tuple[int, int, str], list[tuple[str, Point]]] = defaultdict(list)
    cell_size = max(endpoint_tolerance_px, 1e-6)
    for segment_id, point, channel in endpoints:
        cx = math.floor(point.x / cell_size)
        cy = math.floor(point.y / cell_size)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other_id, other_point in cells.get((cx + dx, cy + dy, channel), []):
                    if math.hypot(point.x - other_point.x, point.y - other_point.y) <= endpoint_tolerance_px:
                        uf.union(segment_tokens[segment_id], segment_tokens[other_id])
        cells[(cx, cy, channel)].append((segment_id, point))

    # A provider polyline may be split into strict two-point segments.  Membership
    # is only evidence of continuity when the member geometry actually touches;
    # this prevents a bad provider ID from collapsing distant runs into a giant net.
    by_polyline: dict[str, list[str]] = defaultdict(list)
    for segment in ir.segments:
        if segment.source_polyline_id:
            by_polyline[segment.source_polyline_id].append(segment.internal_id)
    for segment_ids in by_polyline.values():
        _union_touching_segments(
            segment_ids,
            ir.segment_index(),
            segment_tokens,
            uf,
            tolerance=endpoint_tolerance_px,
        )

    # Cross-tile wire predictions can overlap in their interiors without
    # sharing an endpoint. This is deliberately opt-in: a small geometric
    # error must not silently connect unrelated traces in the blessed path.
    if collinear_merge_tolerance_px > 0:
        _union_collinear_fragments(
            ir.segments,
            segment_tokens,
            uf,
            tolerance=collinear_merge_tolerance_px,
            allow_gap=False,
        )
    if collinear_gap_merge_tolerance_px > 0:
        _union_collinear_fragments(
            ir.segments,
            segment_tokens,
            uf,
            tolerance=collinear_gap_merge_tolerance_px,
            allow_gap=True,
        )

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

    # Preserve explicit source-edge connectivity as evidence, especially for dense golden paths.
    by_source_edge: dict[str, list[str]] = defaultdict(list)
    for segment in ir.segments:
        if segment.source_edge_id:
            by_source_edge[segment.source_edge_id].append(segment.internal_id)
    for segment_ids in by_source_edge.values():
        if all(
            any(item.source == "dataset" for item in ir.segment_index()[segment_id].evidence)
            for segment_id in segment_ids
        ):
            for segment_id in segment_ids[1:]:
                uf.union(segment_tokens[segment_ids[0]], segment_tokens[segment_id])
        else:
            _union_touching_segments(
                segment_ids,
                ir.segment_index(),
                segment_tokens,
                uf,
                tolerance=endpoint_tolerance_px,
            )

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
        for edge_id in source_edges:
            segment = ir.segment_index()[by_source_edge[edge_id][0]]
            if _explicit_membership_is_trusted(pin, segment, pin_snap_tolerance_px):
                uf.union(pin_token, segment_tokens[segment.internal_id])
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
    _union_direct_pins(
        ir.pins,
        uf,
        tolerance=direct_pin_tolerance_px,
        pins_with_explicit_edges=declared_pin_edges,
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

    groups = uf.groups()
    nets: list[PhysicalNet] = []
    token_to_net: dict[str, str] = {}
    for index, group in enumerate(groups, start=1):
        pins = sorted(token.removeprefix("pin:") for token in group if str(token).startswith("pin:"))
        segments = sorted(token.removeprefix("segment:") for token in group if str(token).startswith("segment:"))
        if not pins and not segments:
            continue
        net_id = f"pnet_{index:05d}"
        nets.append(PhysicalNet(internal_id=net_id, pins=pins, segments=segments))
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
    for net in nets:
        net.labels = sorted(set(net.labels))
    return nets, snap_decisions


def _segment_endpoints(segment: WireSegment) -> tuple[Point, Point]:
    return segment.start, segment.end


def _segments_touch(left: WireSegment, right: WireSegment, tolerance: float) -> bool:
    return any(
        math.hypot(a.x - b.x, a.y - b.y) <= tolerance
        for a in _segment_endpoints(left)
        for b in _segment_endpoints(right)
    )


def _union_touching_segments(
    segment_ids: list[str],
    segment_index: dict[str, WireSegment],
    segment_tokens: dict[str, str],
    uf: UnionFind,
    *,
    tolerance: float,
) -> None:
    cells: dict[tuple[int, int], list[tuple[str, Point]]] = defaultdict(list)
    cell_size = max(tolerance, 1e-6)
    for segment_id in sorted(segment_ids):
        for point in _segment_endpoints(segment_index[segment_id]):
            cx = math.floor(point.x / cell_size)
            cy = math.floor(point.y / cell_size)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for other_id, other_point in cells.get((cx + dx, cy + dy), []):
                        if math.hypot(point.x - other_point.x, point.y - other_point.y) <= tolerance:
                            uf.union(segment_tokens[segment_id], segment_tokens[other_id])
            cells[(cx, cy)].append((segment_id, point))


def _union_collinear_fragments(
    segments: list[WireSegment],
    segment_tokens: dict[str, str],
    uf: UnionFind,
    *,
    tolerance: float,
    allow_gap: bool,
) -> None:
    # Union same-channel segments whose interiors overlap, or whose endpoint
    # gap is explicitly enabled by a separate candidate control.
    # The spatial index avoids comparing every pair of long wire fragments.
    # A pair still needs the same channel and nearly identical supporting
    # lines. This never bridges crossing channels.
    if tolerance <= 0:
        return
    cell_size = max(32.0, tolerance * 4.0)
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    pair_seen: set[tuple[int, int]] = set()
    for index, segment in enumerate(segments):
        min_x = min(segment.start.x, segment.end.x) - tolerance
        max_x = max(segment.start.x, segment.end.x) + tolerance
        min_y = min(segment.start.y, segment.end.y) - tolerance
        max_y = max(segment.start.y, segment.end.y) + tolerance
        min_cx = math.floor(min_x / cell_size)
        max_cx = math.floor(max_x / cell_size)
        min_cy = math.floor(min_y / cell_size)
        max_cy = math.floor(max_y / cell_size)
        candidates: set[int] = set()
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                candidates.update(cells.get((cx, cy), ()))
        for other_index in sorted(candidates):
            pair = (other_index, index)
            if pair in pair_seen:
                continue
            pair_seen.add(pair)
            other = segments[other_index]
            if other.channel != segment.channel:
                continue
            if segments_have_collinear_overlap(
                other,
                segment,
                tolerance,
                allow_gap=allow_gap,
            ):
                uf.union(
                    segment_tokens[other.internal_id],
                    segment_tokens[segment.internal_id],
                )
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                cells[(cx, cy)].append(index)


def segments_have_collinear_overlap(
    left: WireSegment,
    right: WireSegment,
    tolerance: float,
    *,
    allow_gap: bool = False,
) -> bool:
    lx = left.end.x - left.start.x
    ly = left.end.y - left.start.y
    length = math.hypot(lx, ly)
    if length <= 1e-6:
        return False
    ux = lx / length
    uy = ly / length
    # Distance from the second supporting line to the first. Checking both
    # endpoints rejects parallel-but-separated traces.
    alignment_tolerance = min(tolerance, 1.5) if allow_gap else tolerance
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


def _union_direct_pins(
    pins: list[Pin],
    uf: UnionFind,
    *,
    tolerance: float,
    pins_with_explicit_edges: set[str] | None = None,
) -> None:
    if tolerance < 0:
        return
    cells: dict[tuple[int, int], list[Pin]] = defaultdict(list)
    cell_size = max(tolerance, 1e-6)
    explicit = pins_with_explicit_edges or set()
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
                    if math.hypot(pin.point.x - other.point.x, pin.point.y - other.point.y) <= tolerance:
                        uf.union(f"pin:{pin.internal_id}", f"pin:{other.internal_id}")
        cells[(cx, cy)].append(pin)
