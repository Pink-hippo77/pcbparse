"""Filter symbol/text strokes only when image-supported wire color is stable.

The perceived pin geometry supplies anchor rays before OCR association or
degraded pruning. OCR boxes cannot censor these pixels. This is a conservative
color-image refinement, with abstention on grayscale and mixed palettes.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pcbparse.ir.models import CircuitIR, Pin, WireSegment
from pcbparse.pipeline.context import stable_file_hash


@dataclass(frozen=True)
class PalettePolicy:
    minimum_anchor_pins: int = 6
    minimum_anchor_components: int = 3
    minimum_anchor_agreement: float = .75
    minimum_chromatic_fraction: float = .65
    maximum_wire_color_fraction: float = .08
    radius_px: int = 1
    pin_protection_px: float = 3.
    pin_lead_max_length_px: float = 24.


@dataclass(frozen=True)
class PaletteWireContext:
    bins: np.ndarray
    ink: np.ndarray
    palette: dict[str, Any]
    protection_pins: tuple[Pin, ...]
    policy: PalettePolicy


def hue_map(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(rgb, dtype=np.float32) / 255.
    high = values.max(axis=2)
    low = values.min(axis=2)
    delta = high - low
    chromatic = (delta >= .20) & (low < .80)
    denominator = np.maximum(delta, 1e-6)
    r, g, b = np.moveaxis(values, -1, 0)
    hue = np.zeros(high.shape, dtype=np.float32)
    red = (high == r) & chromatic
    green = (high == g) & chromatic & ~red
    blue = chromatic & ~red & ~green
    hue[red] = ((g[red] - b[red]) / denominator[red]) % 6.
    hue[green] = (b[green] - r[green]) / denominator[green] + 2.
    hue[blue] = (r[blue] - g[blue]) / denominator[blue] + 4.
    bins = np.rint(hue * 2).astype(np.int8) % 12
    bins[~chromatic] = -1
    return bins, low < .80


def _neighborhood(bins, ink, point, radius):
    x, y = round(point[0]), round(point[1])
    height, width = bins.shape
    if not (0 <= x < width and 0 <= y < height):
        return Counter(), 0
    slices = (slice(max(0, y-radius), min(height, y+radius+1)),
              slice(max(0, x-radius), min(width, x+radius+1)))
    colors = bins[slices][ink[slices]]
    return Counter(int(v) for v in colors if v >= 0), int(ink[slices].sum())


def _hue_distance(left, right):
    return min((left-right) % 12, (right-left) % 12)


def estimate_palette(ir: CircuitIR, bins, ink, policy: PalettePolicy) -> dict[str, Any]:
    components = ir.component_index()
    boxes = [c.bbox for c in ir.components]
    directions = {"L": (-1, 0), "R": (1, 0), "T": (0, -1), "B": (0, 1)}
    votes = []
    for pin in sorted(ir.pins, key=lambda p: p.internal_id):
        if pin.component_id not in components or pin.orientation not in directions:
            continue
        dx, dy = directions[pin.orientation]
        counts = Counter()
        observed_positions = sample_positions = 0
        for distance in range(4, 29, 2):
            x, y = pin.point.x+distance*dx, pin.point.y+distance*dy
            if any(b.x1 <= x <= b.x2 and b.y1 <= y <= b.y2 for b in boxes):
                continue
            sample_positions += 1
            local, _ = _neighborhood(bins, ink, (x, y), policy.radius_px)
            if local:
                observed_positions += 1
                counts.update(local)
        if observed_positions < 3 or sample_positions < 4 or not counts:
            continue
        color, _ = max(counts.items(), key=lambda item: (item[1], -item[0]))
        support = sum(n for h, n in counts.items() if _hue_distance(h, color) <= 1) / sum(counts.values())
        if support < .8 or observed_positions/sample_positions < .4:
            continue
        votes.append({"pin_id": pin.internal_id, "component_id": pin.component_id,
                      "hue": color, "positions": observed_positions, "support": support})
    if not votes:
        return {"enabled": False, "reason": "no_reliable_chromatic_pin_rays", "votes": []}
    counts = Counter(v["hue"] for v in votes)
    color, _ = max(counts.items(), key=lambda item: (
        sum(n for h, n in counts.items() if _hue_distance(h, item[0]) <= 1), -item[0]))
    supporting = [v for v in votes if _hue_distance(v["hue"], color) <= 1]
    owners = {v["component_id"] for v in supporting}
    agreement = len(supporting)/len(votes)
    enabled = (len(supporting) >= policy.minimum_anchor_pins
               and len(owners) >= policy.minimum_anchor_components
               and agreement >= policy.minimum_anchor_agreement)
    return {"enabled": enabled,
            "reason": "stable_pin_ray_color" if enabled else "insufficient_anchor_agreement",
            "wire_hue_bin": color, "supporting_pins": len(supporting),
            "supporting_components": len(owners), "agreement": agreement, "votes": votes}


def prepare_palette_wire(source: str | Path, ir: CircuitIR) -> PaletteWireContext:
    """Read only the actual input image and bind pixels to canonical IR geometry."""
    if stable_file_hash(source) != ir.image.sha256:
        raise ValueError("palette wire input image hash does not match Circuit-IR")
    with Image.open(source) as bitmap:
        if bitmap.size != (ir.image.width, ir.image.height):
            raise ValueError("palette wire input image dimensions do not match Circuit-IR")
        rgb = np.asarray(bitmap.convert("RGB"))
    bins, ink = hue_map(rgb)
    policy = PalettePolicy()
    return PaletteWireContext(bins, ink, estimate_palette(ir, bins, ink, policy),
                              tuple(p.model_copy(deep=True) for p in ir.pins), policy)


def _short_pin_lead(segment: WireSegment, pins, policy: PalettePolicy) -> bool:
    a, b = segment.start, segment.end
    dx, dy = b.x-a.x, b.y-a.y
    length = math.hypot(dx, dy)
    if length > policy.pin_lead_max_length_px or length < 1e-6:
        return False
    for pin in pins:
        aligned = ((pin.orientation in {"L", "R"} and abs(dy) <= length*.15)
                   or (pin.orientation in {"T", "B"} and abs(dx) <= length*.15))
        if not aligned:
            continue
        t = max(0., min(1., ((pin.point.x-a.x)*dx+(pin.point.y-a.y)*dy)/(length*length)))
        if math.hypot(pin.point.x-(a.x+t*dx), pin.point.y-(a.y+t*dy)) <= policy.pin_protection_px:
            return True
    return False


def apply_palette_wire_filter(ir: CircuitIR, context: PaletteWireContext) -> dict[str, Any]:
    """Refine existing segments before graph construction; never invent geometry."""
    bins, ink, policy, palette = context.bins, context.ink, context.policy, context.palette
    report = {"method": "pin_ray_palette_wire_support_v2", "gt_access": False, "palette": palette,
              "removed_segment_ids": [], "removed_invalid_junction_ids": [],
              "protected_pin_leads": 0, "segment_decisions": []}
    if not palette["enabled"]:
        ir.metadata["palette_wire_filter"] = report
        return report
    color = palette["wire_hue_bin"]
    explicit_edges = {sid for p in ir.pins for sid in p.edge_ids}
    removed = set()
    for segment in sorted(ir.segments, key=lambda s: s.internal_id):
        if segment.synthetic:
            continue
        a, b = segment.start, segment.end
        length = math.hypot(b.x-a.x, b.y-a.y)
        if length < 3:
            continue
        count = max(4, min(128, math.ceil(length)+1))
        colors = Counter()
        ink_count = observed_positions = 0
        for t in np.linspace(0., 1., count):
            local, total = _neighborhood(bins, ink, (a.x+t*(b.x-a.x), a.y+t*(b.y-a.y)), policy.radius_px)
            colors.update(local)
            ink_count += total
            observed_positions += bool(total)
        chromatic = sum(colors.values())
        if ink_count < 8 or observed_positions/count < .5 or chromatic/ink_count < policy.minimum_chromatic_fraction:
            continue
        wire = sum(n for h, n in colors.items() if _hue_distance(h, color) <= 1)/max(1, chromatic)
        if wire > policy.maximum_wire_color_fraction:
            continue
        if _short_pin_lead(segment, context.protection_pins, policy):
            report["protected_pin_leads"] += 1
            continue
        if segment.internal_id in explicit_edges or segment.source_edge_id in explicit_edges:
            continue
        removed.add(segment.internal_id)
        report["segment_decisions"].append({"segment_id": segment.internal_id, "length_px": length,
            "chromatic_fraction": chromatic/ink_count, "wire_color_fraction": wire,
            "dominant_hue_bin": max(colors, key=lambda h: (colors[h], -h))})
    ir.segments = [s for s in ir.segments if s.internal_id not in removed]
    minimum = {"t_junction": 3, "connected_cross": 4, "bend": 2}
    invalid = set()
    for junction in ir.junctions:
        junction.segment_ids = [sid for sid in junction.segment_ids if sid not in removed]
        if not junction.segment_ids or len(set(junction.segment_ids)) < minimum.get(junction.kind, 0):
            invalid.add(junction.internal_id)
    ir.junctions = [j for j in ir.junctions if j.internal_id not in invalid]
    report["removed_segment_ids"] = sorted(removed)
    report["removed_invalid_junction_ids"] = sorted(invalid)
    ir.metadata["palette_wire_filter"] = report
    return report
