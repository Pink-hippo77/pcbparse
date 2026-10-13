from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from pcbparse.graph.union_find import UnionFind
from pcbparse.ir.models import CircuitIR, ElectricalNet, Evidence, NetLabel


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
