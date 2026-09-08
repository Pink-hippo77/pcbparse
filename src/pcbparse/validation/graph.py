from __future__ import annotations

import math
from collections import Counter, defaultdict

from pcbparse.graph.physical import segments_have_collinear_overlap
from pcbparse.ir.models import CircuitIR, Diagnostic, WireSegment
from pcbparse.validation.common import error, stable_diagnostics, warning


def _point_key(x: float, y: float) -> tuple[float, float]:
    return (x, y)


def _segment_key(segment: WireSegment) -> tuple[object, ...]:
    endpoints = sorted(
        (_point_key(segment.start.x, segment.start.y), _point_key(segment.end.x, segment.end.y))
    )
    return (segment.channel, endpoints[0], endpoints[1])


def _has_disconnected_segments(
    segment_ids: list[str],
    index: dict[str, WireSegment],
    connected_junction_groups: list[set[str]],
    *,
    endpoint_tolerance_px: float = 3.0,
    collinear_merge_tolerance_px: float = 0.0,
    collinear_gap_merge_tolerance_px: float = 0.0,
) -> bool:
    available = [index[segment_id] for segment_id in dict.fromkeys(segment_ids) if segment_id in index]
    if len(available) < 2:
        return False
    endpoint_to_segments: dict[tuple[str, float, float], list[int]] = defaultdict(list)
    segment_positions: dict[str, int] = {}
    for idx, segment in enumerate(available):
        segment_positions[segment.internal_id] = idx
        endpoint_to_segments[
            (segment.channel, segment.start.x, segment.start.y)
        ].append(idx)
        endpoint_to_segments[
            (segment.channel, segment.end.x, segment.end.y)
        ].append(idx)
    adjacency: list[set[int]] = [set() for _ in available]
    for members in endpoint_to_segments.values():
        for source in members:
            adjacency[source].update(members)
    for group in connected_junction_groups:
        members = [segment_positions[item] for item in group if item in segment_positions]
        for source in members:
            adjacency[source].update(members)
    # Vectorization intentionally canonicalizes endpoints within a small local
    # tolerance; validator connectivity must use the same geometric contract.
    endpoints: list[tuple[int, str, float, float]] = []
    for idx, segment in enumerate(available):
        endpoints.extend(
            (
                (idx, segment.channel, segment.start.x, segment.start.y),
                (idx, segment.channel, segment.end.x, segment.end.y),
            )
        )
    cell_size = max(endpoint_tolerance_px, 1e-9)
    cells: dict[tuple[int, int, str], list[tuple[int, float, float]]] = defaultdict(list)
    for idx, channel, x, y in endpoints:
        cx = math.floor(x / cell_size)
        cy = math.floor(y / cell_size)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other_idx, other_x, other_y in cells.get((cx + dx, cy + dy, channel), []):
                    if math.hypot(x - other_x, y - other_y) <= endpoint_tolerance_px:
                        adjacency[idx].add(other_idx)
                        adjacency[other_idx].add(idx)
        cells[(cx, cy, channel)].append((idx, x, y))

    # Multiple strict two-point segments may retain one observed source edge or
    # source polyline. Those memberships are connectivity evidence, not merely labels.
    for attribute in ("source_edge_id", "source_polyline_id"):
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, segment in enumerate(available):
            value = getattr(segment, attribute)
            if value:
                groups[value].append(idx)
        for members in groups.values():
            for source in members:
                adjacency[source].update(members)

    # Match the builder's collinear connectivity: segments whose interiors
    # overlap (or whose gap is within tolerance) on the same channel are
    # connected even without touching endpoints.
    if collinear_merge_tolerance_px > 0 or collinear_gap_merge_tolerance_px > 0:
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                left = available[i]
                right = available[j]
                if left.channel != right.channel:
                    continue
                if collinear_merge_tolerance_px > 0 and segments_have_collinear_overlap(
                    left, right, collinear_merge_tolerance_px, allow_gap=False,
                ):
                    adjacency[i].add(j)
                    adjacency[j].add(i)
                elif collinear_gap_merge_tolerance_px > 0 and segments_have_collinear_overlap(
                    left, right, collinear_gap_merge_tolerance_px, allow_gap=True,
                ):
                    adjacency[i].add(j)
                    adjacency[j].add(i)

    visited = {0}
    stack = [0]
    while stack:
        current = stack.pop()
        for neighbor in adjacency[current] - visited:
            visited.add(neighbor)
            stack.append(neighbor)
    return len(visited) != len(available)


def _duplicates(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def validate_graph(
    ir: CircuitIR,
    *,
    collinear_merge_tolerance_px: float = 0.0,
    collinear_gap_merge_tolerance_px: float = 0.0,
) -> list[Diagnostic]:
    issues: list[Diagnostic] = []
    segment_index = ir.segment_index()
    connected_junction_groups = [
        set(junction.segment_ids)
        for junction in ir.junctions
        if junction.kind in {"t_junction", "connected_cross", "bend"}
    ]
    connected_junction_groups.extend(
        set(pin.edge_ids) for pin in ir.pins if len(set(pin.edge_ids)) > 1
    )

    segment_keys: dict[tuple[object, ...], list[str]] = defaultdict(list)
    for segment in ir.segments:
        segment_keys[_segment_key(segment)].append(segment.internal_id)
    for ids in segment_keys.values():
        if len(ids) > 1:
            issues.append(
                error(
                    "EDGE.DUPLICATE",
                    "wire segments are exact geometric duplicates in the same channel",
                    *sorted(ids),
                    stage="graph",
                )
            )

    for net in [*ir.physical_nets, *ir.electrical_nets]:
        for pin_id in _duplicates(net.pins):
            issues.append(
                error(
                    "NET.DUPLICATE_PIN",
                    f"net contains duplicate pin reference: {pin_id}",
                    net.internal_id,
                    pin_id,
                    stage="graph",
                )
            )
        for segment_id in _duplicates(net.segments):
            issues.append(
                error(
                    "NET.DUPLICATE_SEGMENT",
                    f"net contains duplicate segment reference: {segment_id}",
                    net.internal_id,
                    segment_id,
                    stage="graph",
                )
            )

    for net in ir.physical_nets:
        if _has_disconnected_segments(
            net.segments,
            segment_index,
            connected_junction_groups,
            collinear_merge_tolerance_px=collinear_merge_tolerance_px,
            collinear_gap_merge_tolerance_px=collinear_gap_merge_tolerance_px,
        ):
            issues.append(
                error(
                    "NET.DISCONNECTED_SEGMENTS",
                    "physical net contains more than one disconnected segment component",
                    net.internal_id,
                    stage="graph",
                )
            )

    electrical_owners: dict[str, list[str]] = defaultdict(list)
    for net in ir.electrical_nets:
        for pin_id in set(net.pins):
            electrical_owners[pin_id].append(net.internal_id)
        if not net.pins:
            issues.append(
                warning(
                    "NET.EMPTY_PIN_SET",
                    "electrical net has no pins and cannot contribute a HyperGraph pin set",
                    net.internal_id,
                    stage="graph",
                )
            )
    for pin_id, owners in sorted(electrical_owners.items()):
        if len(owners) > 1:
            issues.append(
                error(
                    "NET.PIN_IN_MULTIPLE_NETS",
                    f"pin belongs to multiple electrical nets: {pin_id}",
                    pin_id,
                    *sorted(owners),
                    stage="graph",
                )
            )

    for junction in ir.junctions:
        branch_count = len(set(junction.segment_ids))
        minimum = {"t_junction": 3, "connected_cross": 4}.get(junction.kind)
        if minimum is not None and branch_count < minimum:
            issues.append(
                error(
                    "JUNCTION.INSUFFICIENT_BRANCHES",
                    f"{junction.kind} requires at least {minimum} distinct segments",
                    junction.internal_id,
                    stage="graph",
                )
            )

    return stable_diagnostics(issues)


class GraphValidator:
    def __init__(
        self,
        *,
        collinear_merge_tolerance_px: float = 0.0,
        collinear_gap_merge_tolerance_px: float = 0.0,
    ) -> None:
        self.collinear_merge_tolerance_px = collinear_merge_tolerance_px
        self.collinear_gap_merge_tolerance_px = collinear_gap_merge_tolerance_px

    def validate(self, ir: CircuitIR, ctx: object | None = None) -> list[Diagnostic]:
        del ctx
        return validate_graph(
            ir,
            collinear_merge_tolerance_px=self.collinear_merge_tolerance_px,
            collinear_gap_merge_tolerance_px=self.collinear_gap_merge_tolerance_px,
        )
