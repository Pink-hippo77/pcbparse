from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pcbparse.ir.models import CircuitIR, ElectricalNet, IRState, Point, WireSegment
from pcbparse.output.official_schema import (
    OfficialSchemaError,
    load_official_schema,
    split_official_pin_numbers,
    validate_official_output,
)


_NATURAL_PARTS = re.compile(r"(\d+)")


def natural_sort_key(value: object) -> tuple[tuple[object, ...], ...]:
    """Sort R2 before R10 while remaining deterministic for mixed identifiers."""

    text = "" if value is None else str(value)
    return tuple(
        # Preserve the original numeric token as a final tie-break.  Without
        # it, distinct legal strings such as R1 and R01 had identical keys and
        # inherited caller insertion order, violating deterministic output.
        # Only decimal digits match \d and can be converted by int(). Other
        # digit glyphs such as ² and ① remain literal identifier text.
        (0, int(part), part) if part.isdecimal() else (1, part.casefold(), part)
        for part in _NATURAL_PARTS.split(text)
        if part != ""
    )


def round_half_away_from_zero(value: float) -> int:
    if not math.isfinite(value):
        raise OfficialSchemaError("official coordinates cannot contain NaN or Infinity")
    if value >= 0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def _finite_coordinate(value: float) -> float:
    if not math.isfinite(value):
        raise OfficialSchemaError("official coordinates cannot contain NaN or Infinity")
    # Avoid a byte-level distinction between -0.0 and 0.0.
    return 0.0 if value == 0.0 else float(value)


def _point(point: Point, image_height: float) -> dict[str, float]:
    return {
        "x": _finite_coordinate(point.x),
        "y": _finite_coordinate(image_height - point.y),
    }


def _canonical_endpoints(
    segment: WireSegment, image_height: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    endpoints = sorted(
        (
            (
                _finite_coordinate(segment.start.x),
                _finite_coordinate(image_height - segment.start.y),
            ),
            (
                _finite_coordinate(segment.end.x),
                _finite_coordinate(image_height - segment.end.y),
            ),
        )
    )
    return endpoints[0], endpoints[1]


def _encoded_geometry(
    geometry: tuple[tuple[float, float], tuple[float, float]]
) -> list[dict[str, float]]:
    first, second = geometry
    return [{"x": first[0], "y": first[1]}, {"x": second[0], "y": second[1]}]


def _component_key_map(ir: CircuitIR) -> dict[str, str]:
    """Map internal_id -> official key without mutating the input IR.

    Missing designators are rejected: placeholder fill would hide OCR binding
    failures and fabricate official identifiers.
    """
    by_designator: dict[str, list[Any]] = defaultdict(list)
    for component in ir.components:
        if not isinstance(component.key, str) or not component.key.strip():
            raise OfficialSchemaError(
                f"component {component.internal_id} has no official key; "
                "placeholder fill is forbidden (prune upstream in degraded mode)"
            )
        by_designator[component.key].append(component)

    mapping: dict[str, str] = {}
    output_keys: set[str] = set()
    for designator in sorted(by_designator, key=natural_sort_key):
        members = sorted(
            by_designator[designator],
            key=lambda item: natural_sort_key(item.internal_id),
        )
        if len(members) == 1:
            candidates = (designator,)
        else:
            candidates = tuple(
                f"{designator}_{index}" for index in range(1, len(members) + 1)
            )
        for component, output_key in zip(members, candidates):
            if output_key in output_keys:
                raise OfficialSchemaError(
                    f"duplicate-designator numbering collides with official key {output_key!r}"
                )
            mapping[component.internal_id] = output_key
            output_keys.add(output_key)
    return mapping


def _pin_number_map(ir: CircuitIR, component_keys: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
    by_component: dict[str, list[Any]] = defaultdict(list)
    for pin in ir.pins:
        if not pin.component_id or pin.component_id not in component_keys:
            raise OfficialSchemaError(f"pin {pin.internal_id} references unknown component")
        by_component[pin.component_id].append(pin)

    output: dict[str, tuple[str, ...]] = {}
    for component_id, pins in sorted(
        by_component.items(), key=lambda item: natural_sort_key(component_keys[item[0]])
    ):
        ordered = sorted(pins, key=lambda item: natural_sort_key(item.internal_id))
        explicit = {
            pin.internal_id: split_official_pin_numbers(pin.number)
            for pin in ordered
        }
        used: set[str] = set()
        for pin in ordered:
            for number in explicit[pin.internal_id]:
                if number in used:
                    raise OfficialSchemaError(
                        f"duplicate pin number {number!r} for component {component_keys[component_id]}"
                    )
                used.add(number)
        for pin in ordered:
            numbers = explicit[pin.internal_id]
            if not numbers:
                raise OfficialSchemaError(
                    f"pin {pin.internal_id} has no explicit representable number; "
                    "auto-numbering is forbidden (prune upstream in degraded mode)"
                )
            output[pin.internal_id] = numbers
    return output


def _pin_refs(
    pin_id: str,
    pin_index: Mapping[str, Any],
    component_keys: Mapping[str, str],
    pin_numbers: Mapping[str, tuple[str, ...]],
    template: str,
) -> tuple[str, ...]:
    try:
        pin = pin_index[pin_id]
    except KeyError as exc:
        raise OfficialSchemaError(f"net references unknown pin: {pin_id}") from exc
    if not pin.component_id or pin.component_id not in component_keys:
        raise OfficialSchemaError(f"pin {pin_id} references unknown component")
    if pin_id not in pin_numbers:
        raise OfficialSchemaError(f"pin {pin_id} has no official number mapping")
    return tuple(
        template.format(
            component_key=component_keys[pin.component_id], pin_number=number
        )
        for number in pin_numbers[pin_id]
    )


def _net_signature(
    net: ElectricalNet,
    pin_index: Mapping[str, Any],
    component_keys: Mapping[str, str],
    pin_numbers: Mapping[str, tuple[str, ...]],
    segment_index: Mapping[str, WireSegment],
    pin_ref_template: str,
    image_height: float,
) -> tuple[object, ...]:
    refs = tuple(
        sorted(
            (
                ref
                for pin_id in set(net.pins)
                for ref in _pin_refs(
                    pin_id,
                    pin_index,
                    component_keys,
                    pin_numbers,
                    pin_ref_template,
                )
            ),
            key=natural_sort_key,
        )
    )
    segment_geometry: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for segment_id in set(net.segments):
        if segment_id not in segment_index:
            raise OfficialSchemaError(f"net references unknown segment: {segment_id}")
        segment_geometry.append(
            _canonical_endpoints(segment_index[segment_id], image_height)
        )
    segment_geometry.sort()
    return (
        tuple(natural_sort_key(ref) for ref in refs),
        tuple(segment_geometry),
        net.label or "",
        net.net_type,
        natural_sort_key(net.internal_id),
    )


def _net_zero_length_geometries(
    ir: CircuitIR,
    net: ElectricalNet,
    *,
    image_height: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Encode physical pin-only groups as the guide's point-like edges.

    The IR deliberately removes zero-length wire segments as invalid internal
    geometry.  Official pages 20, 22, 23 and 29 nevertheless require/permit a
    two-identical-point edge for direct-pin and flywire connection locations,
    so those edges are synthesized only at the serialization boundary.
    """

    pin_index = ir.pin_index()
    physical_index = {item.internal_id: item for item in ir.physical_nets}
    candidate_pin_ids: set[str] = set()
    if net.physical_net_ids:
        for physical_id in net.physical_net_ids:
            physical = physical_index.get(physical_id)
            if physical is None:
                raise OfficialSchemaError(
                    f"electrical net references unknown physical net: {physical_id}"
                )
            if not physical.segments:
                candidate_pin_ids.update(physical.pins)
    elif not net.segments:
        candidate_pin_ids.update(net.pins)

    points: set[tuple[float, float]] = set()
    for pin_id in candidate_pin_ids:
        pin = pin_index.get(pin_id)
        if pin is None:
            raise OfficialSchemaError(f"net references unknown pin: {pin_id}")
        encoded = _point(pin.point, image_height)
        points.add((encoded["x"], encoded["y"]))
    return [(point, point) for point in sorted(points)]


def _missing_designator_ids(ir: CircuitIR) -> list[str]:
    return [
        component.internal_id
        for component in ir.components
        if not isinstance(component.key, str) or not component.key.strip()
    ]


def _pin_number_stats(ir: CircuitIR) -> dict[str, int]:
    missing = 0
    explicit = 0
    for pin in ir.pins:
        numbers = split_official_pin_numbers(pin.number)
        if numbers:
            explicit += len(numbers)
        else:
            missing += 1
    return {"pin_number_missing": missing, "pin_explicit_numbers": explicit}


def _prune_stats(ir: CircuitIR) -> dict[str, Any]:
    empty: dict[str, Any] = {
        "pruned_components": 0,
        "pruned_pins": 0,
        "pins_dropped_missing_number": 0,
        "pins_dropped_unowned": 0,
        "missing_key_internal_ids": [],
        "nets_affected": 0,
        "nets_dropped": 0,
    }
    raw = ir.metadata.get("degraded_submission_prune")
    if not isinstance(raw, Mapping):
        return empty

    def _count(key: str) -> int:
        value = raw.get(key, 0)
        return int(value) if isinstance(value, (int, float)) else 0

    def _ids(key: str) -> list[str]:
        value = raw.get(key, [])
        return [str(item) for item in value] if isinstance(value, list) else []

    return {
        "pruned_components": _count("components_dropped"),
        "pruned_pins": _count("pins_dropped"),
        "pins_dropped_missing_number": _count("pins_dropped_missing_number"),
        "pins_dropped_unowned": _count("pins_dropped_unowned"),
        "missing_key_internal_ids": _ids("component_ids_dropped"),
        "nets_affected": _count("nets_affected"),
        "nets_dropped": _count("nets_dropped"),
    }


def _wire_provenance(ir: CircuitIR) -> dict[str, Any]:
    """Visible versus repaired wire provenance carried by the IR.

    Official ``edges`` are written from IR segment geometry (visible wire and
    mask-closing repaired spans).  Pure relation repairs (graph-level virtual
    gap evidence) influence connectivity only and never become edges.
    """

    repaired_segments = sorted(
        segment.internal_id
        for segment in ir.segments
        if any(item.source == "repair" for item in segment.evidence)
    )
    synthetic_segments = sorted(
        segment.internal_id for segment in ir.segments if segment.synthetic
    )
    breaks = ir.metadata.get("wire_breaks")
    return {
        "official_edges_source": "ir_segments_including_mask_closing_repairs",
        "relation_only_repairs_become_edges": False,
        "virtual_repaired_segments": repaired_segments,
        "virtual_repaired_segment_count": len(repaired_segments),
        "synthetic_segments": synthetic_segments,
        "synthetic_segment_count": len(synthetic_segments),
        "mask_closing_repairs": len(ir.repairs),
        "vectorizer_breaks": len(breaks) if isinstance(breaks, list) else 0,
    }


def _reject_error_diagnostics(ir: CircuitIR) -> None:
    """Entry self-check: never serialize an IR carrying ERROR diagnostics.

    The production orchestrator refuses to serialize after a failed validation,
    but that gate cannot replace the public entry self-check: a direct caller
    can hand in a ``VALIDATED``-state IR whose ``diagnostics`` still contain
    ERROR records.  Such contradictory IR must fail here instead of emitting an
    official submission.  WARNING and INFO diagnostics never block.
    """

    error_codes = sorted(
        f"{item.code}@{item.stage}"
        for item in ir.diagnostics
        if item.severity == "ERROR"
    )
    if error_codes:
        raise OfficialSchemaError(
            "serialization rejects IR carrying ERROR diagnostics "
            f"({len(error_codes)}): {error_codes[:5]}"
        )


def serialize_official_with_quality(
    ir: CircuitIR,
    profile: str | Path | Mapping[str, Any] | None = None,
    *,
    mode: str = "strict",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Serialize VALIDATED IR and report quality-state counts.

    Never mutates the input IR. ``mode`` is ``strict`` or ``degraded``.
    Missing component keys / pin numbers are rejected in both modes: the
    serializer never fabricates identifiers.  Upstream degraded pruning may drop
    unrepresentable objects and records the deleted IDs; the quality dict then
    separates explicit OCR-read pin numbers (``pin_explicit_numbers``) from
    pruned missing numbers (``pins_dropped_missing_number``) so reports never
    count invented numbers as OCR success.

    ``quality["status"]`` is exactly one of ``strict_success`` (non-empty output
    without pruning), ``degraded_nonempty`` (non-empty output after upstream
    pruning) or ``empty_result`` (empty container, optionally from a legitimate
    empty input). ``fallback_empty`` / ``failed_to_write`` are assigned later by
    the submission layer and never appear here.
    """
    if mode not in {"strict", "degraded"}:
        raise OfficialSchemaError(f"unsupported serialization mode {mode!r}")
    _reject_error_diagnostics(ir)
    missing_keys = _missing_designator_ids(ir)
    if missing_keys:
        raise OfficialSchemaError(
            "serialization rejects components without official key "
            f"(prune upstream in degraded mode): {missing_keys[:5]}"
        )
    pin_stats = _pin_number_stats(ir)
    if pin_stats["pin_number_missing"]:
        raise OfficialSchemaError(
            "serialization rejects pins without explicit representable number "
            f"(prune upstream in degraded mode): {pin_stats['pin_number_missing']} pin(s)"
        )
    output = serialize_official(ir, profile)
    prune = _prune_stats(ir)
    empty_output = not output["components"] and not output["pins"] and not output["nets"]
    degraded_marks = bool(prune["pruned_components"] or prune["pruned_pins"])
    if empty_output:
        status = "empty_result"
    elif degraded_marks:
        status = "degraded_nonempty"
    else:
        status = "strict_success"
    quality = {
        "mode": mode,
        "status": status,
        "strict_success": status == "strict_success",
        "degraded_nonempty": status == "degraded_nonempty",
        "empty_result": status == "empty_result",
        "fallback_empty": False,
        "failed_to_write": False,
        "pin_explicit_numbers": pin_stats["pin_explicit_numbers"],
        "pin_number_missing": pin_stats["pin_number_missing"],
        "pruned_components": prune["pruned_components"],
        "pruned_pins": prune["pruned_pins"],
        "pins_dropped_missing_number": prune["pins_dropped_missing_number"],
        "pins_dropped_unowned": prune["pins_dropped_unowned"],
        "missing_key_internal_ids": prune["missing_key_internal_ids"],
        "nets_affected": prune["nets_affected"],
        "nets_dropped": prune["nets_dropped"],
        "legitimate_empty_input": empty_output and not degraded_marks,
        "input_ir_mutated": False,
        "wire_provenance": _wire_provenance(ir),
    }
    return output, quality


def serialize_official(
    ir: CircuitIR,
    profile: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize a VALIDATED Circuit-IR into the exact official container shape.

    Official net ``edges`` are written from IR wire segment geometry: visible
    wire plus mask-closing repaired spans.  Pure relation repairs and virtual
    gap evidence may join candidate connectivity but never generate edges.
    """

    if ir.state != IRState.VALIDATED:
        raise OfficialSchemaError(
            f"official serialization requires VALIDATED IR, got {ir.state}"
        )
    _reject_error_diagnostics(ir)
    schema = load_official_schema(profile)

    image_height = float(ir.image.height)
    component_keys = _component_key_map(ir)
    pin_numbers = _pin_number_map(ir, component_keys)
    name_field = schema["component"]["fields"][1]
    type_aliases = schema["component"].get("type_aliases", {})

    components: dict[str, Any] = {}
    for component in sorted(
        ir.components,
        key=lambda item: natural_sort_key(component_keys[item.internal_id]),
    ):
        output_key = component_keys[component.internal_id]
        components[output_key] = {
            "type": type_aliases.get(component.category, component.category),
            name_field: component.name if component.name != "" else None,
            "value": component.value if component.value != "" else None,
            "bbox": [
                _finite_coordinate(component.bbox.x1),
                _finite_coordinate(image_height - component.bbox.y2),
                _finite_coordinate(component.bbox.x2),
                _finite_coordinate(image_height - component.bbox.y1),
            ],
        }

    pins_by_component: dict[str, list[Any]] = defaultdict(list)
    for pin in ir.pins:
        if not pin.component_id or pin.component_id not in component_keys:
            raise OfficialSchemaError(f"pin {pin.internal_id} references unknown component")
        pins_by_component[pin.component_id].append(pin)

    pins: dict[str, Any] = {}
    pin_key_template = schema["pin"]["key_template"]
    for component in sorted(
        ir.components,
        key=lambda item: natural_sort_key(component_keys[item.internal_id]),
    ):
        output_key = component_keys[component.internal_id]
        group: dict[str, Any] = {}
        for pin in sorted(
            pins_by_component[component.internal_id],
            key=lambda item: (
                tuple(natural_sort_key(number) for number in pin_numbers[item.internal_id]),
                natural_sort_key(item.internal_id),
            ),
        ):
            for number in pin_numbers[pin.internal_id]:
                pin_key = pin_key_template.format(number=number)
                if pin_key in group:
                    raise OfficialSchemaError(
                        f"duplicate pin key {pin_key} for component {output_key}"
                    )
                group[pin_key] = {
                    "pinname": pin.name,
                    "point": _point(pin.point, image_height),
                }
        pins[output_key] = dict(
            sorted(group.items(), key=lambda item: natural_sort_key(item[0]))
        )

    pin_index = ir.pin_index()
    segment_index = ir.segment_index()
    pin_ref_template = schema["net"]["hypergraph_pin_template"]
    ordered_nets = sorted(
        ir.electrical_nets,
        key=lambda net: _net_signature(
            net,
            pin_index,
            component_keys,
            pin_numbers,
            segment_index,
            pin_ref_template,
            image_height,
        ),
    )
    nets: dict[str, Any] = {}
    for index, net in enumerate(ordered_nets, start=1):
        refs = sorted(
            (
                ref
                for pin_id in set(net.pins)
                for ref in _pin_refs(
                    pin_id,
                    pin_index,
                    component_keys,
                    pin_numbers,
                    pin_ref_template,
                )
            ),
            key=natural_sort_key,
        )
        edge_geometries: list[
            tuple[tuple[float, float], tuple[float, float]]
        ] = []
        for segment_id in set(net.segments):
            if segment_id not in segment_index:
                raise OfficialSchemaError(f"net references unknown segment: {segment_id}")
            edge_geometries.append(
                _canonical_endpoints(segment_index[segment_id], image_height)
            )
        edge_geometries.extend(
            _net_zero_length_geometries(ir, net, image_height=image_height)
        )
        edge_geometries = sorted(set(edge_geometries))
        edges = {
            schema["net"]["edge_key_template"].format(index=edge_index): _encoded_geometry(
                geometry
            )
            for edge_index, geometry in enumerate(edge_geometries, start=1)
        }
        net_key = schema["net"]["key_template"].format(index=index)
        nets[net_key] = {
            "hyperGraph": f"({schema['net'].get('hypergraph_separator', ',').join(refs)})",
            "edges": edges,
        }

    output = {"components": components, "pins": pins, "nets": nets}
    validate_official_output(output, schema)
    return output


def official_json_bytes(
    output: Mapping[str, Any],
    profile: str | Path | Mapping[str, Any] | None = None,
) -> bytes:
    """Return canonical UTF-8 bytes used for both writing and hashing."""

    schema = load_official_schema(profile)
    validate_official_output(output, schema)
    canonical = _canonical_official_mapping(output, schema)
    text = json.dumps(
        canonical,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return (text + "\n").encode("utf-8")


def _canonical_official_mapping(
    output: Mapping[str, Any], schema: Mapping[str, Any]
) -> dict[str, Any]:
    """Rebuild a validated mapping in the one guide-v1 byte order.

    ``serialize_official`` already creates this order, but the public hashing
    and atomic-write APIs also accept caller-supplied mappings.  JSON object
    insertion order is not semantic, so hashing those mappings directly made
    equivalent submissions produce different digests.  Rebuilding the small
    fixed schema here makes the advertised canonical-byte contract real.
    """

    raw_components = output["components"]
    raw_pins = output["pins"]
    raw_nets = output["nets"]
    component_fields = list(schema["component"]["fields"])
    pin_fields = list(schema["pin"]["fields"])
    net_fields = list(schema["net"]["fields"])

    component_keys = sorted(raw_components, key=natural_sort_key)
    components: dict[str, Any] = {}
    pins: dict[str, Any] = {}
    for component_key in component_keys:
        raw_component = raw_components[component_key]
        components[component_key] = {
            field: raw_component[field] for field in component_fields
        }
        raw_group = raw_pins[component_key]
        group: dict[str, Any] = {}
        for pin_key in sorted(raw_group, key=natural_sort_key):
            raw_pin = raw_group[pin_key]
            pin_payload = {field: raw_pin[field] for field in pin_fields}
            raw_point = raw_pin["point"]
            pin_payload["point"] = {"x": raw_point["x"], "y": raw_point["y"]}
            group[pin_key] = pin_payload
        pins[component_key] = group

    nets: dict[str, Any] = {}
    for net_key in sorted(raw_nets, key=natural_sort_key):
        raw_net = raw_nets[net_key]
        edges: dict[str, Any] = {}
        raw_edges = raw_net["edges"]
        for edge_key in sorted(raw_edges, key=natural_sort_key):
            raw_edge = raw_edges[edge_key]
            edges[edge_key] = [
                {"x": point["x"], "y": point["y"]} for point in raw_edge
            ]
        net_payload = {field: raw_net[field] for field in net_fields}
        net_payload["edges"] = edges
        nets[net_key] = net_payload
    return {"components": components, "pins": pins, "nets": nets}


def official_json_hash(
    output: Mapping[str, Any],
    profile: str | Path | Mapping[str, Any] | None = None,
) -> str:
    return hashlib.sha256(official_json_bytes(output, profile)).hexdigest()


def atomic_write_official_json(
    output: Mapping[str, Any],
    destination: str | Path,
    profile: str | Path | Mapping[str, Any] | None = None,
) -> Path:
    """Validate, fsync, and atomically replace a submission JSON file."""

    payload = official_json_bytes(output, profile)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


class OfficialSerializer:
    def __init__(
        self,
        profile: str | Path | Mapping[str, Any] | None = None,
    ) -> None:
        self.profile = load_official_schema(profile)

    def serialize(self, ir: CircuitIR) -> dict[str, Any]:
        return serialize_official(ir, self.profile)

    def hash(self, ir: CircuitIR) -> str:
        return official_json_hash(self.serialize(ir), self.profile)

    def write(self, ir: CircuitIR, destination: str | Path) -> Path:
        return atomic_write_official_json(self.serialize(ir), destination, self.profile)


# Concise aliases used by the orchestrator pseudocode in the implementation manual.
serialize = serialize_official
atomic_write_json = atomic_write_official_json
deterministic_hash = official_json_hash
