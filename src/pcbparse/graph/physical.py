from __future__ import annotations

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
