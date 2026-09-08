from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Protocol

from pcbparse.ir.models import (
    BBox,
    CircuitIR,
    Diagnostic,
    IRState,
    Point,
    RepairRecord,
    WireSegment,
)
from pcbparse.validation.common import merge_diagnostics, stable_diagnostics


class Validator(Protocol):
    def validate(self, ir: CircuitIR, ctx: object | None = None) -> list[Diagnostic]: ...


def _clamp_near_boundary(value: float, upper: float, tolerance: float) -> float:
    if not math.isfinite(value):
        return value
    if -tolerance <= value < 0.0:
        return 0.0
    if upper < value <= upper + tolerance:
        return upper
    return value


def _dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _segment_geometry_key(segment: WireSegment) -> tuple[object, ...]:
    endpoints = sorted(
        ((segment.start.x, segment.start.y), (segment.end.x, segment.end.y))
    )
    return (segment.channel, endpoints[0], endpoints[1])


class _RepairLog:
    def __init__(self, ir: CircuitIR, config_hash: str) -> None:
        self.ir = ir
        self.config_hash = config_hash
        self._next = len(ir.repairs) + 1

    def add(
        self,
        rule_id: str,
        object_ids: Sequence[str],
        before: object,
        after: object,
        *,
        evidence_ids: Sequence[str] = (),
    ) -> None:
        self.ir.repairs.append(
            RepairRecord(
                repair_id=f"repair_{self._next:04d}",
                rule_id=rule_id,
                object_ids=list(object_ids),
                before=before,
                after=after,
                evidence_ids=list(evidence_ids),
                confidence=1.0,
                config_hash=self.config_hash,
            )
        )
        self._next += 1


def _clamp_point(point: Point, max_x: float, max_y: float, tolerance: float) -> bool:
    x = _clamp_near_boundary(point.x, max_x, tolerance)
    y = _clamp_near_boundary(point.y, max_y, tolerance)
    changed = x != point.x or y != point.y
    if changed:
        point.x = x
        point.y = y
    return changed


def _clamped_bbox(
    bbox: BBox, max_x: float, max_y: float, tolerance: float
) -> BBox | None:
    values = (
        _clamp_near_boundary(bbox.x1, max_x, tolerance),
        _clamp_near_boundary(bbox.y1, max_y, tolerance),
        _clamp_near_boundary(bbox.x2, max_x, tolerance),
        _clamp_near_boundary(bbox.y2, max_y, tolerance),
    )
    if values == (bbox.x1, bbox.y1, bbox.x2, bbox.y2):
        return None
    return BBox(x1=values[0], y1=values[1], x2=values[2], y2=values[3])


def _repair_coordinates(ir: CircuitIR, log: _RepairLog, tolerance: float) -> None:
    max_x = float(ir.image.width - 1)
    max_y = float(ir.image.height - 1)

    bbox_objects = [*ir.components, *ir.texts]
    for item in bbox_objects:
        before = item.bbox.model_dump(mode="json")
        clamped = _clamped_bbox(item.bbox, max_x, max_y, tolerance)
        if clamped is not None:
            item.bbox = clamped
            evidence_ids = [e.evidence_id for e in item.evidence]
            log.add(
                "REPAIR.CLAMP_COORDINATE",
                [item.internal_id],
                before,
                item.bbox.model_dump(mode="json"),
                evidence_ids=evidence_ids,
            )

    point_roles: list[tuple[object, str, Point]] = []
    for pin in ir.pins:
        point_roles.append((pin, "point", pin.point))
        if pin.base is not None:
            point_roles.append((pin, "base", pin.base))
    for segment in ir.segments:
        point_roles.extend(((segment, "start", segment.start), (segment, "end", segment.end)))
    for junction in ir.junctions:
        point_roles.append((junction, "point", junction.point))
    for label in ir.net_labels:
        if label.point is not None:
            point_roles.append((label, "point", label.point))

    for item, role, point in point_roles:
        before = {role: point.model_dump(mode="json")}
        if _clamp_point(point, max_x, max_y, tolerance):
            evidence_ids = [e.evidence_id for e in item.evidence]
            log.add(
                "REPAIR.CLAMP_COORDINATE",
                [item.internal_id],
                before,
                {role: point.model_dump(mode="json")},
                evidence_ids=evidence_ids,
            )


def _rewrite_segment_references(ir: CircuitIR, replacements: dict[str, str | None]) -> None:
    def rewrite(values: Iterable[str]) -> list[str]:
        rewritten = (
            replacements.get(value, value)
            for value in values
        )
        return _dedupe_preserving_order(value for value in rewritten if value is not None)

    for pin in ir.pins:
        pin.edge_ids = rewrite(pin.edge_ids)
    for junction in ir.junctions:
        junction.segment_ids = rewrite(junction.segment_ids)
    for net in ir.physical_nets:
        net.segments = rewrite(net.segments)
    for net in ir.electrical_nets:
        net.segments = rewrite(net.segments)


def _repair_segments(ir: CircuitIR, log: _RepairLog) -> None:
    zero_length: set[str] = set()
    groups: dict[tuple[object, ...], list[WireSegment]] = defaultdict(list)
    for segment in ir.segments:
        if segment.start.x == segment.end.x and segment.start.y == segment.end.y:
            zero_length.add(segment.internal_id)
        else:
            groups[_segment_geometry_key(segment)].append(segment)

    replacements: dict[str, str | None] = {segment_id: None for segment_id in zero_length}
    for segment in sorted(
        (item for item in ir.segments if item.internal_id in zero_length),
        key=lambda item: item.internal_id,
    ):
        log.add(
            "REPAIR.REMOVE_ZERO_LENGTH_SEGMENT",
            [segment.internal_id],
            segment.model_dump(mode="json"),
            None,
        )

    for segments in groups.values():
        if len(segments) < 2:
            continue
        # Prefer the strongest observed edge; the ID tie-break keeps the choice stable.
        canonical = min(segments, key=lambda item: (-item.score, item.internal_id))
        for duplicate in sorted(segments, key=lambda item: item.internal_id):
            if duplicate.internal_id == canonical.internal_id:
                continue
            replacements[duplicate.internal_id] = canonical.internal_id
            log.add(
                "REPAIR.REMOVE_DUPLICATE_SEGMENT",
                [duplicate.internal_id, canonical.internal_id],
                duplicate.model_dump(mode="json"),
                {"canonical_segment_id": canonical.internal_id},
            )

    if replacements:
        ir.segments = [
            segment for segment in ir.segments if segment.internal_id not in replacements
        ]
        _rewrite_segment_references(ir, replacements)


def _repair_duplicate_net_members(ir: CircuitIR, log: _RepairLog) -> None:
    for net in [*ir.physical_nets, *ir.electrical_nets]:
        unique_pins = _dedupe_preserving_order(net.pins)
        if unique_pins != net.pins:
            before = list(net.pins)
            net.pins = unique_pins
            log.add(
                "REPAIR.DEDUPLICATE_NET_PINS",
                [net.internal_id],
                before,
                list(net.pins),
            )
        unique_segments = _dedupe_preserving_order(net.segments)
        if unique_segments != net.segments:
            before = list(net.segments)
            net.segments = unique_segments
            log.add(
                "REPAIR.DEDUPLICATE_NET_SEGMENTS",
                [net.internal_id],
                before,
                list(net.segments),
            )


def repair_ir_once(
    ir: CircuitIR,
    *,
    coordinate_clamp_tolerance: float = 1.0,
    config_hash: str = "",
) -> CircuitIR:
    """Apply only the documented low-risk repairs to a deep copy of *ir*.

    Coordinates are clamped only when they are within the small configured
    tolerance of an image boundary.  The function deliberately does not infer
    component keys, pins, flywire merges, or power/ground scope.
    """

    if coordinate_clamp_tolerance < 0:
        raise ValueError("coordinate_clamp_tolerance must be non-negative")
    repaired = ir.model_copy(deep=True)
    log = _RepairLog(repaired, config_hash)
    _repair_coordinates(repaired, log, coordinate_clamp_tolerance)
    _repair_segments(repaired, log)
    _repair_duplicate_net_members(repaired, log)
    return repaired


def validate_and_repair_once(
    ir: CircuitIR,
    validators: Sequence[Validator] | None = None,
    *,
    coordinate_clamp_tolerance: float = 1.0,
    config_hash: str = "",
    ctx: object | None = None,
) -> tuple[CircuitIR, list[Diagnostic]]:
    """Perform exactly one repair pass, then a full deterministic validation."""

    if validators is None:
        from pcbparse.validation import default_validators

        validators = default_validators()
    repaired = repair_ir_once(
        ir,
        coordinate_clamp_tolerance=coordinate_clamp_tolerance,
        config_hash=config_hash,
    )
    validator_issues = stable_diagnostics(
        issue for validator in validators for issue in validator.validate(repaired, ctx)
    )
    issues = merge_diagnostics(repaired.diagnostics, validator_issues)
    repaired.diagnostics = issues
    has_errors = any(issue.severity == "ERROR" for issue in issues)
    if not has_errors:
        if repaired.state in {IRState.SEMANTIC, IRState.VALIDATED}:
            repaired.state = IRState.VALIDATED
    elif repaired.state in {IRState.VALIDATED, IRState.SERIALIZED}:
        repaired.state = IRState.SEMANTIC
    return repaired, issues
