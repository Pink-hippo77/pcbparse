from __future__ import annotations

from collections import Counter, defaultdict

from pcbparse.graph.physical import (
    HARD_MAX_GAP_REPAIR_PX,
    ConnectionEvidence,
    collect_segment_connections,
    evidence_adjacency,
)
from pcbparse.ir.models import CircuitIR, Diagnostic, ElectricalNet, WireSegment
from pcbparse.validation.common import error, info, stable_diagnostics, warning


def _point_key(x: float, y: float) -> tuple[float, float]:
    return (x, y)


def _segment_key(segment: WireSegment) -> tuple[object, ...]:
    endpoints = sorted(
        (_point_key(segment.start.x, segment.start.y), _point_key(segment.end.x, segment.end.y))
    )
    return (segment.channel, endpoints[0], endpoints[1])


def is_signal_flywire_electrical(net: ElectricalNet) -> bool:
    """True when SIGNAL label merge legally joins physically separate nets.

    A SIGNAL flywire is an electrical connection, not a solid wire.  Physical
    disconnect checks must not treat it as a broken PhysicalNet.
    """

    return net.net_type in {"SIGNAL", "UNKNOWN"} and len(net.physical_net_ids) > 1


def _has_disconnected_segments(
    segment_ids: list[str],
    index: dict[str, WireSegment],
    connected_junction_groups: list[set[str]],
    *,
    endpoint_tolerance_px: float = 3.0,
    collinear_merge_tolerance_px: float = 0.0,
    collinear_gap_merge_tolerance_px: float = 0.0,
    gap_repair_max_px: float | None = None,
    connection_evidence: list[ConnectionEvidence] | None = None,
) -> bool:
    available = [index[segment_id] for segment_id in dict.fromkeys(segment_ids) if segment_id in index]
    if len(available) < 2:
        return False

    # Shared evidence contract: the validator must use the same connectivity
    # decisions and parameters as the builder.  Duplicated adjacency logic is
    # how builder-connected nets used to be reported as disconnected.
    if connection_evidence is None:
        connection_evidence = collect_segment_connections(
            available,
            endpoint_tolerance_px=endpoint_tolerance_px,
            collinear_merge_tolerance_px=collinear_merge_tolerance_px,
            collinear_gap_merge_tolerance_px=collinear_gap_merge_tolerance_px,
            gap_repair_max_px=(
                gap_repair_max_px if gap_repair_max_px is not None else HARD_MAX_GAP_REPAIR_PX
            ),
        )
    available_ids = [segment.internal_id for segment in available]
    adjacency = evidence_adjacency(available_ids, connection_evidence)
    for group in connected_junction_groups:
        members = [item for item in group if item in adjacency]
        for source in members:
            for other in members:
                if source != other:
                    adjacency[source].add(other)

    visited = {available_ids[0]}
    stack = [available_ids[0]]
    while stack:
        current = stack.pop()
        for neighbor in adjacency[current] - visited:
            visited.add(neighbor)
            stack.append(neighbor)
    return len(visited) != len(available_ids)


def _duplicates(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def validate_graph(
    ir: CircuitIR,
    *,
    endpoint_tolerance_px: float = 3.0,
    collinear_merge_tolerance_px: float = 0.0,
    collinear_gap_merge_tolerance_px: float = 0.0,
    gap_repair_max_px: float | None = None,
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

    shared_evidence = collect_segment_connections(
        ir.segments,
        endpoint_tolerance_px=endpoint_tolerance_px,
        collinear_merge_tolerance_px=collinear_merge_tolerance_px,
        collinear_gap_merge_tolerance_px=collinear_gap_merge_tolerance_px,
        gap_repair_max_px=(
            gap_repair_max_px if gap_repair_max_px is not None else HARD_MAX_GAP_REPAIR_PX
        ),
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
            endpoint_tolerance_px=endpoint_tolerance_px,
            collinear_merge_tolerance_px=collinear_merge_tolerance_px,
            collinear_gap_merge_tolerance_px=collinear_gap_merge_tolerance_px,
            gap_repair_max_px=gap_repair_max_px,
            connection_evidence=shared_evidence,
        ):
            issues.append(
                error(
                    "NET.DISCONNECTED_SEGMENTS",
                    "physical net contains more than one disconnected segment component",
                    net.internal_id,
                    stage="graph",
                )
            )

    # Electrical nets may legally join physically separate PhysicalNets through
    # SIGNAL flywires.  Physical disconnect checks stay on PhysicalNet only.
    for net in ir.electrical_nets:
        if is_signal_flywire_electrical(net):
            issues.append(
                info(
                    "NET.SIGNAL_FLYWIRE",
                    "electrical net joins physically separate nets via SIGNAL labels",
                    net.internal_id,
                    *sorted(net.physical_net_ids),
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
        endpoint_tolerance_px: float = 3.0,
        collinear_merge_tolerance_px: float = 0.0,
        collinear_gap_merge_tolerance_px: float = 0.0,
        gap_repair_max_px: float | None = None,
    ) -> None:
        if endpoint_tolerance_px < 0:
            raise ValueError("endpoint_tolerance_px cannot be negative")
        self.endpoint_tolerance_px = endpoint_tolerance_px
        self.collinear_merge_tolerance_px = collinear_merge_tolerance_px
        self.collinear_gap_merge_tolerance_px = collinear_gap_merge_tolerance_px
        self.gap_repair_max_px = gap_repair_max_px

    def validate(self, ir: CircuitIR, ctx: object | None = None) -> list[Diagnostic]:
        del ctx
        return validate_graph(
            ir,
            endpoint_tolerance_px=self.endpoint_tolerance_px,
            collinear_merge_tolerance_px=self.collinear_merge_tolerance_px,
            collinear_gap_merge_tolerance_px=self.collinear_gap_merge_tolerance_px,
            gap_repair_max_px=self.gap_repair_max_px,
        )
