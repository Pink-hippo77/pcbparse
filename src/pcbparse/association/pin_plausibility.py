"""Geometric pin-plausibility predicate (opt90 role-inference hygiene).

The same three physical rules as ``text_pin.prune_pin_precision`` —
deep-body placement, attachment-row misalignment, and bbox escape — evaluated
over ALL pin detections without the serializable-only gate and without any
mutation.

Why this exists: the opt90 S6 forensics showed that the semantic role
misroutes (designator-shaped texts classified as pinnumber/pinname) were
driven by pin detections that the precision prune later removes — 27 of 36
pin-role victims have no surviving pin within the classifier's own radius.
A pin that is geometrically implausible must not attract text roles before it
is pruned; this module supplies that exclusion set.

The predicate is intentionally standalone: ``prune_pin_precision`` keeps its
exact behaviour (its 21 tests gate against regression) and its own population
semantics (serializable pins only, row statistics on that population).  Here
the rows are formed over the full pin population because the question is
"is this detection plausible", not "should it be serialized".
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from pcbparse.ir.models import Component, Pin


def _body_edge_offset(pin: Pin, box) -> float | None:
    if box is None or pin.base is None:
        return None
    if pin.orientation in ("T", "B"):
        edge = box.y1 if pin.orientation == "T" else box.y2
        sign = -1.0 if pin.orientation == "T" else 1.0
        return (pin.base.y - edge) * sign
    if pin.orientation in ("L", "R"):
        edge = box.x1 if pin.orientation == "L" else box.x2
        sign = -1.0 if pin.orientation == "L" else 1.0
        return (pin.base.x - edge) * sign
    return None


def _tip_box_escape(pin: Pin, box) -> float:
    dx = max(box.x1 - pin.point.x, 0.0, pin.point.x - box.x2)
    dy = max(box.y1 - pin.point.y, 0.0, pin.point.y - box.y2)
    diagonal = math.hypot(box.x2 - box.x1, box.y2 - box.y1) or 1.0
    return math.hypot(dx, dy) / diagonal


def geometry_junk_pin_ids(
    pins: Sequence[Pin],
    components: Sequence[Component],
    *,
    deep_body_px: float = 11.0,
    row_misalign_px: float = 3.5,
    bbox_escape_ratio: float = 0.15,
    row_consensus_min_pins: int = 2,
) -> set[str]:
    """Return ids of pins failing the physical plausibility rules."""

    deep_limit = float(deep_body_px)
    row_limit = float(row_misalign_px)
    escape_ratio = float(bbox_escape_ratio)
    if not math.isfinite(deep_limit) or deep_limit < 0:
        raise ValueError("deep_body_px must be a non-negative finite number")
    if not math.isfinite(row_limit) or row_limit < 0:
        raise ValueError("row_misalign_px must be a non-negative finite number")
    if not math.isfinite(escape_ratio) or not 0 < escape_ratio <= 1:
        raise ValueError("bbox_escape_ratio must be in (0, 1]")
    if type(row_consensus_min_pins) is not int or row_consensus_min_pins < 1:
        raise ValueError("row_consensus_min_pins must be a positive integer")

    boxes = {item.internal_id: item.bbox for item in components if item.bbox is not None}
    rows: dict[tuple[str | None, str], list[tuple[str, float]]] = defaultdict(list)
    for pin in sorted(pins, key=lambda item: item.internal_id):
        key = (str(pin.component_id) if pin.component_id else None, str(pin.orientation))
        axis_value = pin.point.y if key[1] in ("T", "B") else pin.point.x
        rows[key].append((pin.internal_id, float(axis_value)))
    row_deviation: dict[str, float] = {}
    row_size: dict[str, int] = {}
    for members in rows.values():
        for member_id, _value in members:
            row_size[member_id] = len(members)
        if len(members) < row_consensus_min_pins:
            continue
        values = sorted(value for _member_id, value in members)
        median = values[len(values) // 2]
        for member_id, value in members:
            row_deviation[member_id] = abs(value - median)

    junk: set[str] = set()
    for pin in sorted(pins, key=lambda item: item.internal_id):
        box = boxes.get(str(pin.component_id)) if pin.component_id else None
        edge_offset = _body_edge_offset(pin, box)
        if (
            edge_offset is not None
            and row_size.get(pin.internal_id, 0) >= row_consensus_min_pins
            and edge_offset < -deep_limit
        ):
            junk.add(pin.internal_id)
            continue
        deviation = row_deviation.get(pin.internal_id)
        if deviation is not None and deviation > row_limit:
            junk.add(pin.internal_id)
            continue
        if box is not None and _tip_box_escape(pin, box) > escape_ratio:
            junk.add(pin.internal_id)
    return junk


__all__ = ["geometry_junk_pin_ids"]
