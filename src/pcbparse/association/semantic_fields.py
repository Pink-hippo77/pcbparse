from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Mapping, Sequence

from pcbparse.association.ocr_common import (
    canonical_details,
    lexical_candidates,
    make_rule_evidence,
    point_bbox_distance,
    preferred_text,
)
from pcbparse.ir.models import Component, Pin, TextToken, WireSegment


"""Conservative semantic-field inference for untyped OCR tokens.

The OCR detector/recognizer intentionally produces only geometry and text:
it never guesses whether a token is a designator, value, pin number, pin name,
or net label.  ``associate_ocr`` therefore runs this rule stage first so the
component/pin/net-label passes can rank typed tokens.  Classification is
conservative: a token is typed only when its text shape AND its context
(OCR level, component ROI, proximity to components/pins/segments) agree.
Tokens that cannot be typed safely are left unclassified and never reach the
assignment passes.
"""


_DESIGNATOR = re.compile(r"^[A-Z]{1,4}[0-9]+[A-Z]?$")
_PIN_NUMBER = re.compile(r"^[0-9]{1,3}$")
_LEXEME = re.compile(r"^[^\s,()]+$")
_VALUEISH = re.compile(
    r"^(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*(?:[kKmMuUnNpP]?[A-Za-zΩµ%]*)?$"
)
_NETLIKE = re.compile(r"^[A-Za-z0-9_+.\-]+$")

_KNOWN_POWER = frozenset(
    {
        "VCC",
        "VDD",
        "VEE",
        "VBAT",
        "VIN",
        "VOUT",
        "AVCC",
        "DVCC",
        "GND",
        "GROUND",
        "AGND",
        "DGND",
        "PGND",
        "VSS",
        "VREF",
        "5V",
        "3.3V",
        "3V3",
        "12V",
        "VBUS",
    }
)


def infer_semantic_fields(
    components: Sequence[Component],
    pins: Sequence[Pin],
    texts: Sequence[TextToken],
    segments: Sequence[WireSegment],
    *,
    component_max_distance_factor: float = 2.0,
    pin_max_distance_px: float = 24.0,
) -> dict[str, Any]:
    """Assign conservative ``details`` labels to untyped OCR tokens in place.

    Existing (Golden or otherwise authoritative) ``details`` are never
    overwritten.  Returns a compact summary for the association report.
    """

    component_index = {component.internal_id: component for component in components}
    pin_index = {pin.internal_id: pin for pin in pins}
    segment_index = {segment.internal_id: segment for segment in segments}
    summary: dict[str, Any] = {
        "schema_version": 1,
        "classifier": "rule-v1",
        "classified": 0,
        "preserved_typed": 0,
        "left_unknown": 0,
        "candidate_rescued": 0,
        "by_field": {},
    }
    by_field: dict[str, int] = {}
    # The semantic pass is invoked after perception and can see hundreds of
    # objects on a single schematic. Spatial grids preserve the exact
    # distance checks below while avoiding a full scan for every OCR token.
    component_spatial_index = _build_component_spatial_index(
        components,
        component_max_distance_factor,
    )
    pin_spatial_index = _build_pin_spatial_index(pins, pin_max_distance_px)
    segment_spatial_index = _build_segment_spatial_index(segments, 18.0)

    for token in texts:
        if any(evidence.source == "dataset" for evidence in token.evidence):
            # Annotation-backed Golden/debug tokens carry authoritative
            # semantics (even when details is empty); never reinterpret them.
            summary["preserved_typed"] += 1
            continue
        existing = canonical_details(token.details)
        if existing:
            summary["preserved_typed"] += 1
            continue
        value = preferred_text(token)
        if not value:
            summary["left_unknown"] += 1
            continue
        metadata = _level_metadata(token)
        level = metadata.get("level", "global")
        side = metadata.get("side")
        roi_component_id = metadata.get("component_id")
        candidate_values = [value]
        for candidate in token.candidates:
            if (
                isinstance(candidate, str)
                and candidate.strip()
                and candidate.strip() not in candidate_values
            ):
                candidate_values.append(candidate.strip())

        inferred: str | None = None
        component_id: str | None = None
        selected_value = value
        selected_rank = 0
        raw_inferred: str | None = None
        for candidate_rank, candidate_value in enumerate(candidate_values):
            candidate_inferred, candidate_component_id = _classify_token_value(
                token,
                candidate_value,
                level=level,
                roi_component_id=roi_component_id,
                component_index=component_index,
                components=components,
                pins=pins,
                segments=segments,
                component_max_distance_factor=component_max_distance_factor,
                pin_max_distance_px=pin_max_distance_px,
                component_candidates=component_spatial_index.query(token.bbox),
                pin_candidates=pin_spatial_index.query(token.bbox),
                segment_candidates=segment_spatial_index.query(token.bbox),
            )
            if candidate_rank == 0:
                raw_inferred = candidate_inferred
            if candidate_inferred is None:
                continue
            if candidate_rank > 0 and not _allow_candidate_recovery(
                raw_inferred,
                candidate_inferred,
                level=level,
                raw_value=value,
                candidate_value=candidate_value,
                component_id=candidate_component_id,
            ):
                continue
            inferred = candidate_inferred
            component_id = candidate_component_id
            selected_value = candidate_value
            selected_rank = candidate_rank
            break

        if inferred is None:
            summary["left_unknown"] += 1
            continue
        token.details = inferred
        token.category = inferred
        if component_id and token.component_id is None:
            token.component_id = component_id
        by_field[inferred] = by_field.get(inferred, 0) + 1
        summary["classified"] += 1
        if selected_rank > 0:
            summary["candidate_rescued"] += 1
        _record_inference(
            token,
            inferred,
            level,
            side,
            roi_component_id,
            selected_value=selected_value,
            selected_rank=selected_rank,
        )

    summary["by_field"] = dict(sorted(by_field.items()))
    return summary


def _classify_token_value(
    token: TextToken,
    value: str,
    *,
    level: str,
    roi_component_id: str | None,
    component_index: Mapping[str, Component],
    components: Sequence[Component],
    pins: Sequence[Pin],
    segments: Sequence[WireSegment],
    component_max_distance_factor: float,
    pin_max_distance_px: float,
    component_candidates: Sequence[Component],
    pin_candidates: Sequence[Pin],
    segment_candidates: Sequence[WireSegment],
) -> tuple[str | None, str | None]:
    normalized = _normalize(value)
    if level == "side_strip":
        return _classify_side_strip(
            token,
            value,
            normalized,
            roi_component_id,
            component_index,
        )
    if level == "component":
        return _classify_component_roi(
            token,
            value,
            normalized,
            roi_component_id,
            component_index,
        )
    inferred, _context_component_id, _context_pin_id = _classify_global(
        token,
        value,
        normalized,
        components,
        pins,
        segments,
        component_max_distance_factor=component_max_distance_factor,
        pin_max_distance_px=pin_max_distance_px,
        component_candidates=component_candidates,
        pin_candidates=pin_candidates,
        segment_candidates=segment_candidates,
    )
    # Global proximity is sufficient to infer a semantic field, but it is not
    # explicit ownership evidence. The association stage resolves it later.
    return inferred, None


def _allow_candidate_recovery(
    raw_inferred: str | None,
    candidate_inferred: str,
    *,
    level: str,
    raw_value: str,
    candidate_value: str,
    component_id: str | None,
) -> bool:
    if raw_inferred is None:
        return True
    # A component-ROI crop often decodes a reference designator's leading
    # letter as blank and leaves only its number. Permit a candidate to
    # recover that specific case, but never let an alternative override an
    # already typed value, pin name, or net label.
    if (
        raw_inferred == "pinnumber"
        and candidate_inferred == "designator"
        and level == "component"
        and component_id is not None
        and len(_normalize(candidate_value)) > len(_normalize(raw_value))
        and any(character.isalpha() for character in _normalize(candidate_value))
    ):
        return True
    return False


def _classify_side_strip(
    token: TextToken,
    value: str,
    normalized: str,
    roi_component_id: str | None,
    component_index: Mapping[str, Component],
) -> tuple[str | None, str | None]:
    """Text captured at a component edge is almost always pin text."""

    if not _LEXEME.fullmatch(normalized) or len(normalized) > 16:
        return None, None
    component_id = roi_component_id if roi_component_id in component_index else None
    if _PIN_NUMBER.fullmatch(normalized):
        return "pinnumber", component_id
    if len(normalized) < 2:
        return None, component_id
    if _DESIGNATOR.fullmatch(normalized) and component_id is not None:
        # A symbol's own reference designator can appear on the strip view;
        # keep it as pin text only when it is not the component's own prefix.
        component = component_index.get(component_id)
        if component is not None and _prefix_compatible(component.category, normalized):
            return "pinname", component_id
    return "pinname", component_id


def _classify_component_roi(
    token: TextToken,
    value: str,
    normalized: str,
    roi_component_id: str | None,
    component_index: Mapping[str, Component],
) -> tuple[str | None, str | None]:
    """Text captured inside a component ROI is a designator or value."""

    component_id = roi_component_id if roi_component_id in component_index else None
    component = component_index.get(component_id) if component_id else None
    if _DESIGNATOR.fullmatch(normalized):
        if component is None or _prefix_compatible(component.category, normalized):
            return "designator", component_id
        return None, component_id
    if _PIN_NUMBER.fullmatch(normalized):
        # KiCad/Altium draw pin numbers inside the symbol body, so short
        # numeric tokens inside a component ROI are pin numbers, not values.
        return "pinnumber", component_id
    if _is_valueish(normalized):
        return "value", component_id
    return None, component_id


def _classify_global(
    token: TextToken,
    value: str,
    normalized: str,
    components: Sequence[Component],
    pins: Sequence[Pin],
    segments: Sequence[WireSegment],
    *,
    component_max_distance_factor: float,
    pin_max_distance_px: float,
    component_candidates: Sequence[Component] | None = None,
    pin_candidates: Sequence[Pin] | None = None,
    segment_candidates: Sequence[WireSegment] | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Classify free-floating tokens by text shape and spatial context."""

    nearest_component = _nearest_component(
        token,
        components,
        component_max_distance_factor,
        candidates=component_candidates,
    )
    nearest_pin = _nearest_pin(
        token,
        pins,
        pin_max_distance_px,
        candidates=pin_candidates,
    )

    if _DESIGNATOR.fullmatch(normalized):
        if nearest_component is not None:
            component = nearest_component[0]
            if _prefix_compatible(component.category, normalized):
                return "designator", component.internal_id, None
        return None, None, None

    if _is_valueish(normalized) and nearest_component is not None:
        # Unit-suffixed or decimal values are distinctive even when a pin is
        # nearby, so classify them before the pin-name fallback.
        return "value", nearest_component[0].internal_id, None

    if _PIN_NUMBER.fullmatch(normalized) and nearest_pin is not None:
        return "pinnumber", nearest_pin[0].component_id, nearest_pin[0].internal_id

    if nearest_pin is not None and _looks_like_pin_name(normalized):
        return "pinname", nearest_pin[0].component_id, nearest_pin[0].internal_id

    if _looks_like_netlabel(
        normalized
    ) and _near_segment(token, segments, candidates=segment_candidates):
        return "netlabel", None, None

    # Dataset300 searchName is used for grouped Box regions. Keep this rule
    # deliberately narrow: a title-like token must be near a Box, and cannot
    # also be a local pin or wire label. It remains ownerless for the regular
    # component association pass to resolve with its confidence/margin gates.
    if (
        nearest_component is not None
        and nearest_component[0].category.upper() == "BOX"
        and nearest_pin is None
        and _looks_like_search_name(normalized)
    ):
        return "searchname", nearest_component[0].internal_id, None

    return None, None, None


def _nearest_component(
    token: TextToken,
    components: Sequence[Component],
    component_max_distance_factor: float,
    *,
    candidates: Sequence[Component] | None = None,
) -> tuple[Component, float] | None:
    best: tuple[Component, float] | None = None
    for component in components if candidates is None else candidates:
        scale = max(component.bbox.width, component.bbox.height, 1.0)
        distance = _bbox_gap_distance(token.bbox, component.bbox)
        normalized = distance / scale
        if normalized > component_max_distance_factor:
            continue
        if best is None or normalized < best[1]:
            best = (component, normalized)
    return best


def _nearest_pin(
    token: TextToken,
    pins: Sequence[Pin],
    max_distance_px: float,
    *,
    candidates: Sequence[Pin] | None = None,
) -> tuple[Pin, float] | None:
    best: tuple[Pin, float] | None = None
    for pin in pins if candidates is None else candidates:
        distance = point_bbox_distance(pin.point, token.bbox)
        if distance > max_distance_px:
            continue
        if best is None or distance < best[1]:
            best = (pin, distance)
    return best


def _near_segment(
    token: TextToken,
    segments: Sequence[WireSegment],
    *,
    candidates: Sequence[WireSegment] | None = None,
) -> bool:
    from pcbparse.association.geometry import point_segment_distance

    center = token.bbox.center
    for segment in segments if candidates is None else candidates:
        if point_segment_distance(center, segment) <= 18.0:
            return True
    return False


class _SpatialGrid:
    """Small deterministic uniform grid for point/box proximity queries."""

    def __init__(self, *, cell_size: float = 64.0) -> None:
        if cell_size <= 0.0 or not math.isfinite(cell_size):
            raise ValueError("spatial grid cell_size must be positive and finite")
        self._cell_size = cell_size
        self._cells: dict[tuple[int, int], list[Any]] = {}
        self._order: dict[int, int] = {}
        self._next_order = 0

    def add(self, item: Any, x1: float, y1: float, x2: float, y2: float) -> None:
        left, right = sorted((float(x1), float(x2)))
        top, bottom = sorted((float(y1), float(y2)))
        if not all(math.isfinite(value) for value in (left, top, right, bottom)):
            return
        self._order.setdefault(id(item), self._next_order)
        self._next_order += 1
        for cell_x in range(self._cell(left), self._cell(right) + 1):
            for cell_y in range(self._cell(top), self._cell(bottom) + 1):
                self._cells.setdefault((cell_x, cell_y), []).append(item)

    def query(self, bbox: Any) -> list[Any]:
        if not all(
            math.isfinite(float(value))
            for value in (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
        ):
            return []
        left, right = sorted((float(bbox.x1), float(bbox.x2)))
        top, bottom = sorted((float(bbox.y1), float(bbox.y2)))
        found: dict[int, Any] = {}
        for cell_x in range(self._cell(left), self._cell(right) + 1):
            for cell_y in range(self._cell(top), self._cell(bottom) + 1):
                for item in self._cells.get((cell_x, cell_y), ()):
                    found.setdefault(id(item), item)
        return [
            item
            for _, item in sorted(
                found.items(),
                key=lambda pair: self._order.get(pair[0], 0),
            )
        ]

    def _cell(self, value: float) -> int:
        return math.floor(value / self._cell_size)


def _build_component_spatial_index(
    components: Sequence[Component],
    max_distance_factor: float,
) -> _SpatialGrid:
    grid = _SpatialGrid()
    for component in components:
        padding = max(
            component.bbox.width,
            component.bbox.height,
            1.0,
        ) * max_distance_factor
        grid.add(
            component,
            component.bbox.x1 - padding,
            component.bbox.y1 - padding,
            component.bbox.x2 + padding,
            component.bbox.y2 + padding,
        )
    return grid


def _build_pin_spatial_index(
    pins: Sequence[Pin],
    max_distance_px: float,
) -> _SpatialGrid:
    grid = _SpatialGrid()
    for pin in pins:
        grid.add(
            pin,
            pin.point.x - max_distance_px,
            pin.point.y - max_distance_px,
            pin.point.x + max_distance_px,
            pin.point.y + max_distance_px,
        )
    return grid


def _build_segment_spatial_index(
    segments: Sequence[WireSegment],
    max_distance_px: float,
) -> _SpatialGrid:
    grid = _SpatialGrid()
    for segment in segments:
        grid.add(
            segment,
            min(segment.start.x, segment.end.x) - max_distance_px,
            min(segment.start.y, segment.end.y) - max_distance_px,
            max(segment.start.x, segment.end.x) + max_distance_px,
            max(segment.start.y, segment.end.y) + max_distance_px,
        )
    return grid


def _record_inference(
    token: TextToken,
    details: str,
    level: str,
    side: str | None,
    roi_component_id: str | None,
    *,
    selected_value: str | None = None,
    selected_rank: int = 0,
) -> None:
    evidence = make_rule_evidence(
        evidence_id=f"semantic:{token.internal_id}",
        score=token.score,
        payload_ref=token.internal_id,
        metadata={
            "kind": "semantic-field-inference",
            "inferred_details": details,
            "classifier": "rule-v1",
            "level": level,
            "side": side,
            "crop_component_id": roi_component_id,
            "lexical_candidates": lexical_candidates(token),
            "selected_text": selected_value,
            "selected_candidate_rank": selected_rank,
            "candidate_recovery": selected_rank > 0,
        },
    )
    token.evidence.append(evidence)


def _level_metadata(token: TextToken) -> dict[str, str | None]:
    for evidence in token.evidence:
        metadata = evidence.metadata
        level = metadata.get("level")
        if level in {"global", "component", "side_strip"}:
            return {
                "level": str(level),
                "side": metadata.get("side"),
                "component_id": metadata.get("component_id"),
            }
    return {"level": "global", "side": None, "component_id": None}


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    return re.sub(r"\s+", " ", normalized)


def _is_valueish(normalized: str) -> bool:
    if not re.search(r"[0-9]", normalized):
        return False
    if not _VALUEISH.fullmatch(normalized) or len(normalized) > 16:
        return False
    if "." in normalized:
        return True
    # Unit-suffixed tokens (10k, 0.1uF, 4.7k, 220V, 100nF, 1K, 1M, 10µF).
    if re.search(r"[A-Za-zΩµ%]", normalized):
        return True
    # Long digit runs (>= 4) are almost never pin numbers.
    return len(re.sub(r"[^0-9]", "", normalized)) >= 4


def _looks_like_pin_name(normalized: str) -> bool:
    if not _LEXEME.fullmatch(normalized) or len(normalized) > 16:
        return False
    if len(normalized) < 2:
        return False
    if _DESIGNATOR.fullmatch(normalized):
        return True  # e.g. P14 next to a pin is treated as pin text contextually
    return any(character.isalpha() for character in normalized)


def _looks_like_search_name(normalized: str) -> bool:
    if not normalized or len(normalized) > 256 or not any(character.isalpha() for character in normalized):
        return False
    # Box search names are usually titles/part names, often containing spaces
    # or separators. Short plain words are left to the net-label/pin-name rules.
    return bool(re.search(r"\s|[/_-]", normalized)) or len(normalized) >= 8

def _looks_like_netlabel(normalized: str) -> bool:
    if not _NETLIKE.fullmatch(normalized) or len(normalized) < 2:
        return False
    upper = normalized.upper()
    if upper in _KNOWN_POWER:
        return True
    if normalized.startswith(("+", "-")) or "_" in normalized:
        return True
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?V", upper):
        return True
    if len(normalized) >= 3 and any(character.isdigit() for character in normalized):
        return False  # keep ambiguous alphanumerics for component/pin passes
    return True


def _prefix_compatible(category: str, designator: str) -> bool:
    normalized = category.upper()
    aliases = {
        "BOX": "*",
        "RESISTOR": "R",
        "CAPACITOR": "C",
        "INDUCTOR": "L",
        "DIODE": "D",
        "LED": "D",
        "IC": "U",
        "CONNECTOR": "J",
        "CON": "J",
        "JACK": "J",
        "PLUG": "J",
        "SOCKET": "J",
        "PORT": "J",
        "HEADER": "J",
        "TERMINAL": "J",
        "SWITCH": "SW",
        "RELAY": "K",
        "TRANSISTOR": "Q",
        "MOSFET": "Q",
        "BJT": "Q",
        "NPN": "Q",
        "PNP": "Q",
        "CRYSTAL": "Y",
        "XTAL": "Y",
        "FUSE": "F",
        "TESTPOINT": "TP",
        "VOLTAGE": "U",
        "REGULATOR": "U",
        "POWER": "P",
        "BATTERY": "B",
        "MOTOR": "M",
    }
    expected = aliases.get(normalized, normalized[:1])
    if expected == "*":
        return True
    prefix = re.match(r"^[A-Z]+", designator)
    if prefix is None:
        return False
    return prefix.group(0).startswith(expected)


def _bbox_gap_distance(left: Any, right: Any) -> float:
    dx = max(left.x1 - right.x2, right.x1 - left.x2, 0.0)
    dy = max(left.y1 - right.y2, right.y1 - left.y2, 0.0)
    return math.hypot(dx, dy)


__all__ = ["infer_semantic_fields"]
