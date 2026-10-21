"""Suppress IC border/glyph strokes only with contrary image color evidence."""
from __future__ import annotations

from collections import Counter
import math
from statistics import median

import numpy as np

from pcbparse.association.ocr_common import canonical_details
from pcbparse.graph.palette_wire import PaletteWireContext, _hue_distance, _neighborhood, _short_pin_lead
from pcbparse.ir.models import CircuitIR


def _pin_base_outline_sides(component, owned):
    """Propose side bands from consistent opposing observed pin bases.

    Detection boxes may include the numbered pin strip. These proposals only
    explain existing strokes; component boxes and pin coordinates stay fixed.
    """
    b = component.bbox
    sides = []
    for low, high, axis in (("L", "R", "v"), ("T", "B", "h")):
        groups = []
        for orientation in (low, high):
            pins = [p for p in owned if p.orientation == orientation and p.base is not None]
            if len(pins) < 3:
                break
            bases = [p.base.x if axis == "v" else p.base.y for p in pins]
            tips = [p.point.x if axis == "v" else p.point.y for p in pins]
            along = [p.point.y if axis == "v" else p.point.x for p in pins]
            position = median(bases)
            inward = 1 if orientation in {"L", "T"} else -1
            span = b.height if axis == "v" else b.width
            edge = (b.x1 if orientation == "L" else b.x2 if orientation == "R"
                    else b.y1 if orientation == "T" else b.y2)
            width = b.width if axis == "v" else b.height
            if (sum(abs(x - position) <= 3. for x in bases) < .8 * len(pins)
                    or sum((base - tip) * inward > 0 for base, tip in zip(bases, tips)) < .8 * len(pins)
                    or max(along) - min(along) < .4 * span
                    or abs(position - edge) > .2 * width):
                break
            groups.append(position)
        if len(groups) == 2 and groups[1] - groups[0] >= .5 * (b.width if axis == "v" else b.height):
            lo, hi = (b.y1, b.y2) if axis == "v" else (b.x1, b.x2)
            sides.extend((axis, position, lo, hi) for position in groups)
    return sides


def filter_ic_artifact_segments(ir: CircuitIR, context: PaletteWireContext, *,
                               refine_outline_enabled: bool = False,
                               refine_component_ids: set[str] | None = None) -> dict:
    """Keep geometry unless a stable wire palette contradicts an IC artifact.

    Ordinary symbols and gray/uncertain palettes abstain. Pin lead directions,
    visible wire-colored pixels and explicit edge memberships are protected.
    Neither coordinates nor component/pin fields are changed.
    """
    report = {"method": "ic_artifact_color_support_v1", "gt_access": False,
              "removed_segment_ids": [], "removed_invalid_junction_ids": [],
              "component_palettes": {}, "decisions": []}
    pins = ir.pin_index()
    grouped = {}
    for pin in ir.pins:
        grouped.setdefault(pin.component_id, []).append(pin)
    explicit = {edge for pin in ir.pins for edge in pin.edge_ids}
    candidates = {}
    for component in ir.components:
        owned = grouped.get(component.internal_id, [])
        if (component.category.upper() not in {"BOX", "BOXIC", "BOX_IC", "IC"}
                or component.score < .7 or len(owned) < 4
                or min(component.bbox.width, component.bbox.height) < 20.):
            continue
        votes = [v for v in context.palette.get("votes", [])
                 if v["component_id"] == component.internal_id]
        counts = Counter(v["hue"] for v in votes)
        color = None
        if counts:
            winner = max(counts, key=lambda h: (sum(n for c, n in counts.items()
                                                  if _hue_distance(h, c) <= 1), -h))
            support = sum(n for c, n in counts.items() if _hue_distance(winner, c) <= 1)
            if support >= 4 and support / len(votes) >= .8:
                color = winner
        if color is None and context.palette.get("enabled"):
            color = context.palette["wire_hue_bin"]
        if color is None:
            continue
        report["component_palettes"][component.internal_id] = color
        b = component.bbox
        tolerance = max(4., min(8., min(b.width, b.height) * .05))
        # Only sufficiently long, covered IC body sides explain a border.
        sides = [("v", b.x1, b.y1, b.y2, "ic_body_border"), ("v", b.x2, b.y1, b.y2, "ic_body_border"),
                 ("h", b.y1, b.x1, b.x2, "ic_body_border"), ("h", b.y2, b.x1, b.x2, "ic_body_border")]
        refine = refine_outline_enabled and (refine_component_ids is None or component.internal_id in refine_component_ids)
        for axis, estimate, lo, hi in (_pin_base_outline_sides(component, owned) if refine else []):
            # Bases are learned anchors, not exact body edges. Locate an
            # actually observed long stroke nearby instead of widening the
            # removal band (which could include a column of pin digits).
            width = b.width if axis == "v" else b.height
            columns = {}
            for segment in ir.segments:
                a, z = segment.start, segment.end
                cross = (a.x, z.x) if axis == "v" else (a.y, z.y)
                if abs(cross[0] - cross[1]) > 1.:
                    continue
                position = round(sum(cross) / 2.)
                if abs(position - estimate) > min(20., .12 * width):
                    continue
                along = sorted((a.y, z.y) if axis == "v" else (a.x, z.x))
                left, right = max(lo, along[0]), min(hi, along[1])
                if right > left:
                    columns.setdefault(position, []).append((left, right))
            observed = []
            for position, intervals in columns.items():
                union = []
                for left, right in sorted(intervals):
                    if union and left <= union[-1][1]:
                        union[-1][1] = max(union[-1][1], right)
                    else:
                        union.append([left, right])
                covered = sum(right - left for left, right in union)
                if covered >= .65 * (hi - lo):
                    observed.append((-covered, abs(position - estimate), position))
            if observed:
                position = min(observed)[2]
                native = (b.x1 if estimate < b.center.x else b.x2) if axis == "v" else (b.y1 if estimate < b.center.y else b.y2)
                if abs(position - native) > tolerance:
                    sides.append((axis, position, lo, hi, "ic_corrected_body_border"))
        for axis, position, lo, hi, rule in sides:
            if hi - lo < 20.:
                continue
            members = []
            for segment in ir.segments:
                a, z = segment.start, segment.end
                cross = (a.x, z.x) if axis == "v" else (a.y, z.y)
                along = sorted((a.y, z.y) if axis == "v" else (a.x, z.x))
                if max(abs(x - position) for x in cross) > tolerance:
                    continue
                left, right = max(lo, along[0]), min(hi, along[1])
                if right > left:
                    members.append((left, right, segment.internal_id))
            union = []
            for left, right, _ in sorted(members):
                if union and left <= union[-1][1]:
                    union[-1][1] = max(union[-1][1], right)
                else:
                    union.append([left, right])
            if sum(right - left for left, right in union) >= .5 * (hi - lo):
                for _, _, sid in members:
                    candidates.setdefault(sid, []).append((color, rule))
        for text in ir.texts:
            pin = pins.get(text.pin_id)
            if (canonical_details(text.details) not in {"pinnumber", "pinname"}
                    or text.score < .9 or pin is None or pin.component_id != component.internal_id):
                continue
            if pin.orientation not in {"L", "R", "T", "B"}:
                continue
            t = text.bbox
            # Restrict glyph evidence to the IC and its immediate pin strip.
            if not (b.x1 - 24 <= t.center.x <= b.x2 + 24
                    and b.y1 - 24 <= t.center.y <= b.y2 + 24):
                continue
            for segment in ir.segments:
                a, z = segment.start, segment.end
                length = math.hypot(z.x - a.x, z.y - a.y)
                if length < 3.:
                    continue
                parallel = (abs(z.x - a.x) <= .2 * length if pin.orientation in {"L", "R"}
                            else abs(z.y - a.y) <= .2 * length)
                if parallel and all(t.x1 - 1 <= q.x <= t.x2 + 1 and t.y1 - 1 <= q.y <= t.y2 + 1
                                    for q in (a, z)):
                    candidates.setdefault(segment.internal_id, []).append((color, "ic_pin_glyph"))
    removed = set()
    for segment in ir.segments:
        if segment.internal_id not in candidates:
            continue
        closing_border = (refine_outline_enabled
            and any(rule == "ic_corrected_body_border" for _, rule in candidates[segment.internal_id])
            and any(e.source == "repair" and e.metadata.get("rule") == "mask_closing_virtual_gap"
                    and e.metadata.get("virtual") is True and e.metadata.get("solid_wire") is False
                    for e in segment.evidence))
        if segment.synthetic and not closing_border:
            continue
        if segment.internal_id in explicit or segment.source_edge_id in explicit:
            continue
        if _short_pin_lead(segment, context.protection_pins, context.policy):
            continue
        a, b = segment.start, segment.end
        length = math.hypot(b.x - a.x, b.y - a.y)
        if length < 3.:
            continue
        count = max(4, min(128, math.ceil(length) + 1))
        observed = ink_count = 0
        color_positions = Counter()
        for t in np.linspace(0., 1., count):
            point = (a.x + t * (b.x - a.x), a.y + t * (b.y - a.y))
            colors, ink = _neighborhood(context.bins, context.ink,
                point, context.policy.radius_px)
            observed += bool(ink)
            ink_count += ink
            for color, _ in candidates[segment.internal_id]:
                # Count a wire-colored position once even with duplicate evidence.
                along = [_neighborhood(context.bins, context.ink,
                    (point[0] + direction * 3 * (b.x - a.x) / length,
                     point[1] + direction * 3 * (b.y - a.y) / length),
                    context.policy.radius_px)[0] for direction in (-1, 1)]
                # A perpendicular wire touching the border does not make the
                # border conductive. Require wire color along the stroke.
                if (any(_hue_distance(c, color) <= 1 for c in colors)
                        and all(any(_hue_distance(c, color) <= 1 for c in side) for side in along)):
                    color_positions[(float(t), color)] = 1
        if observed / count < .6 or ink_count < 8:
            continue
        colors = {color for color, _ in candidates[segment.internal_id]}
        # Conflicting local wire palettes must all reject this stroke.
        wire_fraction = max(sum(n for (_, c), n in color_positions.items() if c == color) / count
                            for color in colors)
        if wire_fraction > .05:
            continue
        removed.add(segment.internal_id)
        report["decisions"].append({"segment_id": segment.internal_id, "wire_positions_fraction": wire_fraction,
                                    "rules": sorted({rule for _, rule in candidates[segment.internal_id]})})
    if not removed:
        ir.metadata["ic_artifact_filter"] = report
        return report
    ir.segments = [s for s in ir.segments if s.internal_id not in removed]
    minimum = {"t_junction": 3, "connected_cross": 4, "bend": 2}
    invalid = set()
    for junction in ir.junctions:
        if not removed.intersection(junction.segment_ids):
            continue
        junction.segment_ids = [sid for sid in junction.segment_ids if sid not in removed]
        if not junction.segment_ids or len(set(junction.segment_ids)) < minimum.get(junction.kind, 0):
            invalid.add(junction.internal_id)
    ir.junctions = [j for j in ir.junctions if j.internal_id not in invalid]
    report["removed_segment_ids"] = sorted(removed)
    report["removed_invalid_junction_ids"] = sorted(invalid)
    ir.metadata["ic_artifact_filter"] = report
    return report

