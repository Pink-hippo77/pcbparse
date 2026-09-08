from __future__ import annotations

from collections import defaultdict

from pcbparse.graph.union_find import UnionFind
from pcbparse.ir.models import CircuitIR, ElectricalNet, NetLabel


def build_electrical_nets(
    ir: CircuitIR,
    *,
    merge_signal_labels: bool = True,
    merge_power_labels: bool = False,
    merge_ground_labels: bool = False,
    min_text_score: float = 0.75,
    merge_threshold: float = 0.85,
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

    for (net_type, _), labels in labels_by_group.items():
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
        for physical_id in physical_ids[1:]:
            uf.union(physical_ids[0], physical_id)

    labels_by_physical: dict[str, list[NetLabel]] = defaultdict(list)
    for label in ir.net_labels:
        if label.attached_physical_net_id:
            labels_by_physical[label.attached_physical_net_id].append(label)

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
        if len(members) > 1 and chosen_label:
            provenance.append(f"label_merge:{chosen_label}")
        score_values = [net.score for net in physical_nets] + [label.score for label in labels]
        electrical.append(
            ElectricalNet(
                internal_id=f"enet_{index:05d}",
                pins=pins,
                segments=segments,
                label=chosen_label,
                net_type=net_type,
                score=min(score_values) if score_values else 1.0,
                physical_net_ids=sorted(members),
                provenance=provenance,
            )
        )
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
