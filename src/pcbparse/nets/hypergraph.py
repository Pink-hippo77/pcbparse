from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from pcbparse.graph.union_find import UnionFind
from pcbparse.ir.models import CircuitIR, ElectricalNet, Evidence, NetLabel


@dataclass(frozen=True, slots=True)
class MetricCounts:
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 1.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 1.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2.0 * self.precision * self.recall / denominator if denominator else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "tp": self.true_positive,
            "fp": self.false_positive,
            "fn": self.false_negative,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True, slots=True)
class PinNetEvidence:
    """Per-pin evidence for physical membership and SIGNAL flywire merges."""

    pin_id: str
    physical_net_id: str
    electrical_net_id: str
    rule: str
    score: float
    label_ids: tuple[str, ...] = ()
    virtual: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pin_id": self.pin_id,
            "physical_net_id": self.physical_net_id,
            "electrical_net_id": self.electrical_net_id,
            "rule": self.rule,
            "score": self.score,
            "label_ids": list(self.label_ids),
            "virtual": self.virtual,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class NetView:
    """Comparable net projection: HyperGraph pin set plus NetLine edges."""

    net_id: str
    pins: frozenset[str]
    edges: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "net_id": self.net_id,
            "pins": sorted(self.pins),
            "edges": [
                [[left[0], left[1]], [right[0], right[1]]] for left, right in self.edges
            ],
        }


@dataclass(frozen=True, slots=True)
class MatchedNetComparison:
    """NetLine and NetHypergraph scored on the same assignment-matched pairs."""

    matched_pairs: int
    net_hypergraph: MetricCounts
    net_line: MetricCounts
    pair_details: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "matched_pairs": self.matched_pairs,
            "net_hypergraph": self.net_hypergraph.as_dict(),
            "net_line": self.net_line.as_dict(),
            "pair_details": [dict(item) for item in self.pair_details],
        }


def build_electrical_nets(
    ir: CircuitIR,
    *,
    merge_signal_labels: bool = True,
    merge_power_labels: bool = False,
    merge_ground_labels: bool = False,
    min_text_score: float = 0.75,
    merge_threshold: float = 0.85,
    evidence_out: list[PinNetEvidence] | None = None,
) -> list[ElectricalNet]:
    if not 0.0 <= merge_threshold <= 1.0:
        raise ValueError("merge_threshold must be in [0, 1]")
    physical_index = {net.internal_id: net for net in ir.physical_nets}
    uf = UnionFind(physical_index)
    labels_by_group: dict[tuple[str, str], list[NetLabel]] = defaultdict(list)
    for label in ir.net_labels:
        if label.score < min_text_score or not label.attached_physical_net_id:
            continue
        if label.attached_physical_net_id not in physical_index:
            continue
        labels_by_group[(label.net_type, label.resolved_text)].append(label)

    merge_events: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    for (net_type, resolved_text), labels in labels_by_group.items():
        enabled = (
            (net_type in {"SIGNAL", "UNKNOWN"} and merge_signal_labels)
            or (net_type == "POWER" and merge_power_labels)
            or (net_type == "GROUND" and merge_ground_labels)
        )
        if not enabled:
            continue
        if not labels or min(label.score for label in labels) < merge_threshold:
            continue
        physical_ids = sorted({label.attached_physical_net_id for label in labels if label.attached_physical_net_id})
        label_ids = tuple(sorted(label.internal_id for label in labels))
        for physical_id in physical_ids[1:]:
            uf.union(physical_ids[0], physical_id)
        if len(physical_ids) > 1:
            merge_events.append((net_type, tuple(physical_ids), label_ids))

    labels_by_physical: dict[str, list[NetLabel]] = defaultdict(list)
    for label in ir.net_labels:
        if label.attached_physical_net_id:
            labels_by_physical[label.attached_physical_net_id].append(label)

    pin_evidence: list[PinNetEvidence] = []
    electrical: list[ElectricalNet] = []
    for index, members_raw in enumerate(uf.groups(), start=1):
        members = [str(member) for member in members_raw]
        physical_nets = [physical_index[member] for member in members]
        pins = sorted({pin for net in physical_nets for pin in net.pins})
        segments = sorted({segment for net in physical_nets for segment in net.segments})
        labels = [label for member in members for label in labels_by_physical.get(member, [])]
        chosen_label = _resolved_label(labels)
        net_type = _resolved_type(labels)
        provenance = [f"physical:{member}" for member in sorted(members)]
        flywire = len(members) > 1
        if flywire and chosen_label:
            provenance.append(f"label_merge:{chosen_label}")
            provenance.append("signal_flywire_virtual")
        score_values = [net.score for net in physical_nets] + [label.score for label in labels]
        net_id = f"enet_{index:05d}"
        merge_label_ids = tuple(sorted(label.internal_id for label in labels))
        for pin_id in pins:
            owner_physical = next(
                (net.internal_id for net in physical_nets if pin_id in net.pins),
                members[0],
            )
            if flywire:
                rule = "signal_flywire_merge"
                virtual = True
            else:
                rule = "physical_membership"
                virtual = False
            item = PinNetEvidence(
                pin_id=pin_id,
                physical_net_id=owner_physical,
                electrical_net_id=net_id,
                rule=rule,
                score=min(score_values) if score_values else 1.0,
                label_ids=merge_label_ids,
                virtual=virtual,
                metadata={
                    "net_type": net_type,
                    "resolved_text": chosen_label,
                    "physical_net_ids": sorted(members),
                },
            )
            pin_evidence.append(item)
        for member in members:
            for pin_id in physical_index[member].pins:
                evidence_id = f"net:pin:{net_id}:{pin_id}:{member}"
                # Attach durable per-pin evidence on the IR pin objects too.
                pin_obj = next((pin for pin in ir.pins if pin.internal_id == pin_id), None)
                if pin_obj is None:
                    continue
                if any(existing.evidence_id == evidence_id for existing in pin_obj.evidence):
                    continue
                pin_obj.evidence.append(
                    Evidence(
                        evidence_id=evidence_id,
                        source="rule",
                        view_id="full",
                        score=min(score_values) if score_values else 1.0,
                        payload_ref=member,
                        metadata={
                            "kind": "pin_net_evidence",
                            "electrical_net_id": net_id,
                            "physical_net_id": member,
                            "rule": "signal_flywire_merge" if flywire else "physical_membership",
                            "virtual": flywire,
                            "label_ids": list(merge_label_ids),
                        },
                    )
                )
        electrical.append(
            ElectricalNet(
                internal_id=net_id,
                pins=pins,
                segments=segments,
                label=chosen_label,
                net_type=net_type,
                score=min(score_values) if score_values else 1.0,
                physical_net_ids=sorted(members),
                provenance=provenance,
            )
        )
    if evidence_out is not None:
        evidence_out.extend(pin_evidence)
    ir.metadata["pin_net_evidence"] = [item.as_dict() for item in pin_evidence]
    ir.metadata["label_merge_events"] = [
        {"net_type": net_type, "physical_net_ids": list(physical_ids), "label_ids": list(label_ids)}
        for net_type, physical_ids, label_ids in merge_events
    ]
    return electrical


def compare_matched_net_views(
    predicted: Sequence[NetView],
    expected: Sequence[NetView],
    *,
    line_tolerance_px: float = 5.0,
) -> MatchedNetComparison:
    """Score NetHypergraph and NetLine on the same matched-net assignment.

    Both views are reported together so a geometry-correct edge inside a wrong
    net still counts as an error under HyperGraph membership while NetLine
    keeps its own edge-level counts.
    """

    if line_tolerance_px < 0.0:
        raise ValueError("line_tolerance_px must be non-negative")
    # Import lazily: evaluation.official_metrics pulls the pipeline, which
    # itself imports this module at package import time.
    from pcbparse.evaluation.official_metrics import (
        _maximum_edge_matching,
        _maximum_weight_assignment,
    )

    predicted_items = sorted(predicted, key=_net_view_sort_key)
    expected_items = sorted(expected, key=_net_view_sort_key)
    predicted_pins = [item.pins for item in predicted_items]
    expected_pins = [item.pins for item in expected_items]
    weights = [
        [_pin_set_f1(left, right) for right in expected_pins]
        for left in predicted_pins
    ]
    assignments = [
        (left_index, right_index)
        for left_index, right_index in _maximum_weight_assignment(weights)
        if weights[left_index][right_index] > 0.0
    ]
    matched_predicted = {left for left, _right in assignments}
    matched_expected = {right for _left, right in assignments}
    hyper_tp = hyper_fp = hyper_fn = 0
    line_tp = line_fp = line_fn = 0
    pair_details: list[dict[str, Any]] = []

    for left_index, right_index in assignments:
        left = predicted_items[left_index]
        right = expected_items[right_index]
        left_pins = left.pins
        right_pins = right.pins
        member_tp = len(left_pins & right_pins)
        member_fp = len(left_pins - right_pins)
        member_fn = len(right_pins - left_pins)
        hyper_tp += member_tp
        hyper_fp += member_fp
        hyper_fn += member_fn
        matches = _maximum_edge_matching(
            [[{"x": a[0], "y": a[1]}, {"x": b[0], "y": b[1]}] for a, b in left.edges],
            [[{"x": a[0], "y": a[1]}, {"x": b[0], "y": b[1]}] for a, b in right.edges],
            line_tolerance_px,
        )
        line_tp += matches
        line_fp += len(left.edges) - matches
        line_fn += len(right.edges) - matches
        pair_details.append(
            {
                "predicted_net_id": left.net_id,
                "expected_net_id": right.net_id,
                "hypergraph_member_tp": member_tp,
                "hypergraph_member_fp": member_fp,
                "hypergraph_member_fn": member_fn,
                "line_edge_tp": matches,
                "line_edge_fp": len(left.edges) - matches,
                "line_edge_fn": len(right.edges) - matches,
            }
        )

    for index, item in enumerate(predicted_items):
        if index in matched_predicted:
            continue
        hyper_fp += len(item.pins)
        line_fp += len(item.edges)
    for index, item in enumerate(expected_items):
        if index in matched_expected:
            continue
        hyper_fn += len(item.pins)
        line_fn += len(item.edges)

    return MatchedNetComparison(
        matched_pairs=len(assignments),
        net_hypergraph=MetricCounts(hyper_tp, hyper_fp, hyper_fn),
        net_line=MetricCounts(line_tp, line_fp, line_fn),
        pair_details=tuple(pair_details),
    )


def electrical_nets_to_views(
    electrical_nets: Sequence[ElectricalNet],
    *,
    segment_edges: dict[str, tuple[tuple[float, float], tuple[float, float]]] | None = None,
) -> list[NetView]:
    """Project ElectricalNet objects into comparable HyperGraph/NetLine views."""

    views: list[NetView] = []
    for net in electrical_nets:
        edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
        if segment_edges:
            for segment_id in net.segments:
                edge = segment_edges.get(segment_id)
                if edge is not None:
                    edges.append(edge)
        views.append(
            NetView(
                net_id=net.internal_id,
                pins=frozenset(net.pins),
                edges=tuple(sorted(edges)),
            )
        )
    return views


def _net_view_sort_key(view: NetView) -> tuple[Any, ...]:
    return (tuple(sorted(view.pins)), view.edges, view.net_id)


def _pin_set_f1(left: frozenset[str], right: frozenset[str]) -> float:
    denominator = len(left) + len(right)
    return 2.0 * len(left & right) / denominator if denominator else 0.0


def _resolved_label(labels: list[NetLabel]) -> str | None:
    if not labels:
        return None
    ranked: dict[str, float] = defaultdict(float)
    for label in labels:
        ranked[label.resolved_text] += label.score
    return min(ranked, key=lambda text: (-ranked[text], text))


def _resolved_type(labels: list[NetLabel]) -> str:
    if not labels:
        return "UNKNOWN"
    ranked: dict[str, float] = defaultdict(float)
    for label in labels:
        ranked[label.net_type] += label.score
    return min(ranked, key=lambda text: (-ranked[text], text))
