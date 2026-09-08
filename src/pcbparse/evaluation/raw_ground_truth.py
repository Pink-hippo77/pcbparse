from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pcbparse.graph.union_find import UnionFind


@dataclass(frozen=True, slots=True)
class ObjectMask:
    eligible: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TextTarget:
    value: str | None
    mask: ObjectMask


@dataclass(frozen=True, slots=True)
class RawComponent:
    raw_id: str
    bbox: tuple[float, float, float, float] | None
    category: str
    mask: ObjectMask
    designator: TextTarget
    value: TextTarget
    search_name: TextTarget


@dataclass(frozen=True, slots=True)
class RawPin:
    raw_id: str
    owner_raw_id: str | None
    tip: tuple[float, float] | None
    base: tuple[float, float] | None
    mask: ObjectMask
    number: TextTarget
    name: TextTarget


@dataclass(frozen=True, slots=True)
class RawWire:
    raw_id: str
    paths: tuple[tuple[tuple[float, float], ...], ...]
    mask: ObjectMask


@dataclass(frozen=True, slots=True)
class RawGroundTruth:
    policy_version: str
    case_id: str
    width: int
    height: int
    source_domain: str
    annotation_path: str | None
    components: tuple[RawComponent, ...]
    pins: tuple[RawPin, ...]
    wires: tuple[RawWire, ...]
    connectivity_nets: tuple[frozenset[str], ...]
    graph_total_connections: int
    graph_eligible_connections: int
    graph_mask_reasons: Mapping[str, int]

    def coverage(self) -> dict[str, Any]:
        return {
            "objects": {
                "component": _object_coverage(item.mask for item in self.components),
                "pin": _object_coverage(item.mask for item in self.pins),
                "wire": _object_coverage(item.mask for item in self.wires),
            },
            "fields": {
                "component.designator": _field_coverage(
                    item.designator for item in self.components if item.mask.eligible
                ),
                "component.value": _field_coverage(
                    item.value for item in self.components if item.mask.eligible
                ),
                "component.search_name": _field_coverage(
                    item.search_name for item in self.components if item.mask.eligible
                ),
                "pin.number": _field_coverage(
                    item.number for item in self.pins if item.mask.eligible
                ),
                "pin.name": _field_coverage(
                    item.name for item in self.pins if item.mask.eligible
                ),
            },
            "graph_connections": {
                "total": self.graph_total_connections,
                "eligible": self.graph_eligible_connections,
                "masked": self.graph_total_connections - self.graph_eligible_connections,
                "mask_reasons": dict(sorted(self.graph_mask_reasons.items())),
            },
        }


def build_raw_ground_truth(
    annotation: Mapping[str, Any],
    *,
    case_id: str,
    annotation_path: str | Path | None = None,
) -> RawGroundTruth:
    """Normalize CVJsonStd into raw-v1 targets without inventing identifiers.

    Missing or ambiguous text fields remain masked.  In particular, this code
    never creates the ``DATASET_*`` surrogate identifiers used by the debug
    provider, so the result is suitable for image-only accuracy evaluation.
    """

    canvas = _mapping(annotation.get("canvas"))
    width = _positive_int(canvas.get("width"), "canvas.width")
    height = _positive_int(canvas.get("height"), "canvas.height")
    raw_components = _mapping(annotation.get("components"))
    root_pins = _mapping(annotation.get("pins"))
    texts = _mapping(annotation.get("texts"))
    edges = _mapping(annotation.get("edges"))

    components: list[RawComponent] = []
    active_component_ids: set[str] = set()
    pin_owners: dict[str, set[str]] = defaultdict(set)
    for component_key, raw_value in sorted(raw_components.items(), key=_natural_key_item):
        component_id = str(component_key)
        component = _mapping(raw_value)
        reasons: list[str] = []
        if component.get("state") != "active":
            reasons.append("inactive_component")
        bbox = _bbox(component.get("location"), width=width, height=height)
        if bbox is None:
            reasons.append("invalid_component_bbox")
        mask = ObjectMask(not reasons, tuple(sorted(set(reasons))))
        if mask.eligible:
            active_component_ids.add(component_id)
        nested_pins = component.get("pins")
        if isinstance(nested_pins, Mapping):
            for pin_id in nested_pins:
                pin_owners[str(pin_id)].add(component_id)

        associated_refs = _associated_text_refs(component, texts)
        components.append(
            RawComponent(
                raw_id=component_id,
                bbox=bbox,
                category=str(component.get("category") or ""),
                mask=mask,
                designator=_target_from_refs(
                    associated_refs.get("designator", ()), texts, "component_designator"
                ),
                value=_target_from_refs(
                    associated_refs.get("value", ()), texts, "component_value"
                ),
                search_name=_target_from_refs(
                    associated_refs.get("searchname", ()), texts, "component_search_name"
                ),
            )
        )

    all_pin_ids = sorted(set(map(str, root_pins)) | set(pin_owners), key=_natural_key)
    pins: list[RawPin] = []
    for pin_id in all_pin_ids:
        raw_pin_value = root_pins.get(pin_id)
        raw_pin = _mapping(raw_pin_value)
        owners = sorted(
            (owner for owner in pin_owners.get(pin_id, set()) if owner in active_component_ids),
            key=_natural_key,
        )
        reasons = []
        if raw_pin_value is None:
            reasons.append("missing_root_pin")
        if not owners:
            reasons.append("no_active_owner")
        elif len(owners) > 1:
            reasons.append("multiple_active_owners")
        tip = _point(raw_pin.get("tip"), width=width, height=height)
        base = _point(raw_pin.get("base"), width=width, height=height)
        if tip is None:
            reasons.append("invalid_pin_tip")
        if base is None:
            reasons.append("invalid_pin_base")
        owner = owners[0] if len(owners) == 1 else None
        pins.append(
            RawPin(
                raw_id=pin_id,
                owner_raw_id=owner,
                tip=tip,
                base=base,
                mask=ObjectMask(not reasons, tuple(sorted(set(reasons)))),
                number=_target_from_refs(
                    _string_refs(
                        raw_pin,
                        "pinNumberByOCR",
                        "pinNumberByVLM",
                    ),
                    texts,
                    "pin_number",
                ),
                name=_target_from_refs(
                    _string_refs(
                        raw_pin,
                        "pinNameByOCR",
                        "pinNameByVLM",
                    ),
                    texts,
                    "pin_name",
                ),
            )
        )

    wires: list[RawWire] = []
    for edge_key, raw_value in sorted(edges.items(), key=_natural_key_item):
        edge = _mapping(raw_value)
        reasons = []
        if edge.get("state") != "active":
            reasons.append("inactive_edge")
        paths = _wire_paths(edge, width=width, height=height)
        if not paths:
            reasons.append("missing_wire_geometry")
        wires.append(
            RawWire(
                raw_id=str(edge_key),
                paths=paths,
                mask=ObjectMask(not reasons, tuple(sorted(set(reasons)))),
            )
        )

    eligible_pins = {item.raw_id: item for item in pins if item.mask.eligible}
    nets, graph_total, graph_eligible, graph_reasons = _raw_connectivity(
        edges=edges,
        root_pins=root_pins,
        texts=texts,
        eligible_pins=eligible_pins,
    )
    return RawGroundTruth(
        policy_version="raw-v1",
        case_id=str(case_id),
        width=width,
        height=height,
        source_domain=str(annotation.get("nowCls") or "unknown"),
        annotation_path=str(annotation_path) if annotation_path is not None else None,
        components=tuple(components),
        pins=tuple(pins),
        wires=tuple(wires),
        connectivity_nets=nets,
        graph_total_connections=graph_total,
        graph_eligible_connections=graph_eligible,
        graph_mask_reasons=dict(sorted(graph_reasons.items())),
    )


def _associated_text_refs(
    component: Mapping[str, Any], texts: Mapping[str, Any]
) -> dict[str, tuple[str, ...]]:
    result: dict[str, list[str]] = defaultdict(list)
    associated = component.get("associatedTexts")
    if isinstance(associated, Mapping):
        for raw_ref, embedded in associated.items():
            ref = str(raw_ref)
            root_text = texts.get(ref)
            value = root_text if isinstance(root_text, Mapping) else embedded
            if not isinstance(value, Mapping):
                continue
            detail = str(value.get("details") or "").strip().casefold()
            if detail in {"designator", "value", "searchname"}:
                result[detail].append(ref)
    for ref in _string_refs(
        component,
        "searchNameByOCR",
        "searchNameByVLM",
        "searchNameByLib",
    ):
        result["searchname"].append(ref)
    return {
        field: tuple(sorted(set(refs), key=_natural_key))
        for field, refs in result.items()
    }


def _target_from_refs(
    refs: Sequence[str], texts: Mapping[str, Any], field_name: str
) -> TextTarget:
    ordered = tuple(sorted(set(map(str, refs)), key=_natural_key))
    if not ordered:
        return TextTarget(None, ObjectMask(False, (f"missing_{field_name}",)))
    if len(ordered) > 1:
        return TextTarget(None, ObjectMask(False, (f"ambiguous_{field_name}",)))
    text = texts.get(ordered[0])
    if not isinstance(text, Mapping) or text.get("state") != "active":
        return TextTarget(None, ObjectMask(False, (f"unresolved_{field_name}_reference",)))
    content = text.get("content")
    if not isinstance(content, str) or content == "":
        return TextTarget(None, ObjectMask(False, (f"empty_{field_name}",)))
    return TextTarget(content, ObjectMask(True))


def _raw_connectivity(
    *,
    edges: Mapping[str, Any],
    root_pins: Mapping[str, Any],
    texts: Mapping[str, Any],
    eligible_pins: Mapping[str, RawPin],
) -> tuple[tuple[frozenset[str], ...], int, int, Counter[str]]:
    active_edges = {
        str(edge_id): _mapping(value)
        for edge_id, value in edges.items()
        if isinstance(value, Mapping) and value.get("state") == "active"
    }
    union = UnionFind(active_edges)
    memberships: dict[str, set[str]] = defaultdict(set)
    mask_reasons: Counter[str] = Counter()
    total_connections = 0
    eligible_connections = 0

    for pin_id, raw_value in root_pins.items():
        raw_pin = _mapping(raw_value)
        edge_id = raw_pin.get("edgeId")
        if isinstance(edge_id, str) and edge_id:
            total_connections += 1
            if str(pin_id) not in eligible_pins:
                mask_reasons["pin_edge_masked_pin"] += 1
            elif edge_id not in active_edges:
                mask_reasons["pin_edge_inactive_or_unknown_edge"] += 1
            else:
                memberships[str(pin_id)].add(edge_id)
                eligible_connections += 1

    for edge_id, edge in active_edges.items():
        connections = edge.get("connect")
        if not isinstance(connections, Mapping):
            if connections not in (None, {}):
                mask_reasons["invalid_edge_connect_container"] += 1
            continue
        for raw_pin_key, raw_connection in connections.items():
            total_connections += 1
            connection = _mapping(raw_connection)
            declared = connection.get("pinId")
            pin_id = str(declared) if isinstance(declared, str) and declared else str(raw_pin_key)
            if pin_id not in eligible_pins:
                reason = "connection_unknown_pin" if pin_id not in root_pins else "connection_masked_pin"
                mask_reasons[reason] += 1
                continue
            declared_owner = connection.get("cpntId")
            if (
                isinstance(declared_owner, str)
                and declared_owner
                and declared_owner != eligible_pins[pin_id].owner_raw_id
            ):
                mask_reasons["connection_owner_mismatch"] += 1
                continue
            memberships[pin_id].add(edge_id)
            eligible_connections += 1

    for edge_ids in memberships.values():
        ordered = sorted(edge_ids, key=_natural_key)
        for edge_id in ordered[1:]:
            union.union(ordered[0], edge_id)

    grouped_edges: dict[str, list[str]] = defaultdict(list)
    for edge_id in active_edges:
        grouped_edges[str(union.find(edge_id))].append(edge_id)
    physical_groups = [
        tuple(sorted(group, key=_natural_key)) for group in grouped_edges.values()
    ]
    physical_groups.sort(key=lambda group: _natural_key(group[0]))

    edge_to_group: dict[str, int] = {}
    for index, group in enumerate(physical_groups):
        for edge_id in group:
            edge_to_group[edge_id] = index
    pins_by_group: dict[int, set[str]] = defaultdict(set)
    for pin_id, edge_ids in memberships.items():
        for edge_id in edge_ids:
            if edge_id in edge_to_group:
                pins_by_group[edge_to_group[edge_id]].add(pin_id)

    labels_by_group: dict[int, set[str]] = defaultdict(set)
    for edge_id, edge in active_edges.items():
        reference = edge.get("netlabel")
        if not isinstance(reference, str) or not reference:
            continue
        text = texts.get(reference)
        if not isinstance(text, Mapping) or text.get("state") != "active":
            mask_reasons["unresolved_netlabel_reference"] += 1
            continue
        content = text.get("content")
        if isinstance(content, str) and content.strip():
            labels_by_group[edge_to_group[edge_id]].add(_normal_net_name(content))

    electrical: dict[str, set[str]] = defaultdict(set)
    for index, group in enumerate(physical_groups):
        labels = labels_by_group.get(index, set())
        if len(labels) == 1:
            label = next(iter(labels))
            key = f"signal:{label}" if _label_type(label) == "signal" else f"physical:{index}"
        else:
            if len(labels) > 1:
                mask_reasons["ambiguous_physical_net_labels"] += 1
            key = f"physical:{index}"
        electrical[key].update(pins_by_group.get(index, set()))
    nets = tuple(
        sorted(
            (frozenset(pins) for pins in electrical.values()),
            key=lambda pins: tuple(_natural_key(value) for value in sorted(pins, key=_natural_key)),
        )
    )
    return nets, total_connections, eligible_connections, mask_reasons


def _wire_paths(
    edge: Mapping[str, Any], *, width: int, height: int
) -> tuple[tuple[tuple[float, float], ...], ...]:
    result: list[tuple[tuple[float, float], ...]] = []
    raw_maps = edge.get("keyPointPaths")
    if isinstance(raw_maps, Mapping):
        for _geometry, raw_paths in sorted(raw_maps.items(), key=_natural_key_item):
            if not isinstance(raw_paths, Sequence) or isinstance(raw_paths, (str, bytes)):
                continue
            for raw_path in raw_paths:
                path = _path(raw_path, width=width, height=height)
                if len(path) >= 2:
                    result.append(path)
    if result:
        return tuple(result)
    key_points = edge.get("keyPoints")
    skeleton = key_points.get("skeleton") if isinstance(key_points, Mapping) else None
    if isinstance(skeleton, Mapping):
        for _geometry, raw_points in sorted(skeleton.items(), key=_natural_key_item):
            if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
                continue
            for raw_point in raw_points:
                point = _point(raw_point, width=width, height=height)
                if point is not None:
                    result.append((point,))
    return tuple(result)


def _path(value: Any, *, width: int, height: int) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    points = [_point(raw_point, width=width, height=height) for raw_point in value]
    # Dropping a bad interior point and joining the two surviving neighbours
    # would invent a line that is absent from the annotation.  Mask the path
    # instead of repairing it in the evaluator.
    if any(point is None for point in points):
        return ()
    return tuple(point for point in points if point is not None)


def _bbox(value: Any, *, width: int, height: int) -> tuple[float, float, float, float] | None:
    if not isinstance(value, Mapping):
        return None
    coordinates = [value.get(key) for key in ("x1", "y1", "x2", "y2")]
    if not all(_finite_number(item) for item in coordinates):
        return None
    x1, y1, x2, y2 = map(float, coordinates)
    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _point(value: Any, *, width: int, height: int) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None
    if not _finite_number(value[0]) or not _finite_number(value[1]):
        return None
    x, y = float(value[0]), float(value[1])
    if not (0.0 <= x <= width and 0.0 <= y <= height):
        return None
    return x, y


def _string_refs(value: Mapping[str, Any], *fields: str) -> tuple[str, ...]:
    refs = {
        str(value[field])
        for field in fields
        if isinstance(value.get(field), str) and str(value[field])
    }
    return tuple(sorted(refs, key=_natural_key))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _positive_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a positive integer")
    converted = int(value)
    if converted <= 0 or converted != value:
        raise ValueError(f"{location} must be a positive integer")
    return converted


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _normal_net_name(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _label_type(value: str) -> str:
    upper = value.upper()
    if upper in {"GND", "AGND", "DGND", "PGND", "VSS"} or upper.endswith("GND"):
        return "ground"
    if upper in {"VCC", "VDD", "VBAT", "+5V", "+3V3", "3V3", "5V"}:
        return "power"
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?V", upper):
        return "power"
    return "signal"


def _object_coverage(masks: Any) -> dict[str, Any]:
    items = list(masks)
    reasons: Counter[str] = Counter(
        reason for mask in items if not mask.eligible for reason in mask.reasons
    )
    eligible = sum(mask.eligible for mask in items)
    return {
        "total": len(items),
        "eligible": eligible,
        "masked": len(items) - eligible,
        "mask_reasons": dict(sorted(reasons.items())),
    }


def _field_coverage(targets: Any) -> dict[str, Any]:
    items = list(targets)
    reasons: Counter[str] = Counter(
        reason for target in items if not target.mask.eligible for reason in target.mask.reasons
    )
    eligible = sum(target.mask.eligible for target in items)
    return {
        "total": len(items),
        "eligible": eligible,
        "masked": len(items) - eligible,
        "mask_reasons": dict(sorted(reasons.items())),
    }


_NATURAL_PARTS = re.compile(r"(\d+)")


def _natural_key(value: object) -> tuple[tuple[int, object], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold(), part)
        for part in _NATURAL_PARTS.split(str(value))
        if part
    )


def _natural_key_item(item: tuple[Any, Any]) -> tuple[tuple[int, object], ...]:
    return _natural_key(item[0])


__all__ = [
    "ObjectMask",
    "RawComponent",
    "RawGroundTruth",
    "RawPin",
    "RawWire",
    "TextTarget",
    "build_raw_ground_truth",
]
