"""Measure short connector outlines missed by the ordinary IC outline scan.

Only an existing detector proposal, a separately observed connector reference,
a complete outline and multiple exterior strokes can authorize a candidate.
The caller still has to read every printed terminal number from the image.
"""
from __future__ import annotations

import math

from pcbparse.ir.models import BBox
from pcbparse.perception.symbol_body import _box_rectangle


def compact_connector_outlines(ir, rgb, palette, ctx=None):
    from pcbparse.perception.ic_visual_readback import _CONNECTOR_REFERENCE, _reference, terminal_strokes

    bins, ink = palette
    rectangles, records = [], []
    for index, component in enumerate(ir.components):
        if ctx is not None and index % 16 == 0:
            ctx.time_budget.check(stage="compact_connector_outlines")
        box = component.body_bbox or component.bbox
        if (component.category.upper() not in {"BOX", "IC", "BOXIC", "BOX_IC"}
                or component.score < .75 or not 12 <= box.height < 38
                or not 14 <= box.width <= 120
                or not any(e.source == "detector" for e in component.evidence)):
            continue
        # The detector includes exterior leads. A further padded ROI makes a
        # real short body too small relative to its crop and admits neighbors.
        x0, y0 = max(0, math.floor(box.x1)), max(0, math.floor(box.y1))
        x2, y2 = min(rgb.shape[1], math.ceil(box.x2)+1), min(rgb.shape[0], math.ceil(box.y2)+1)
        if x2 <= x0 or y2 <= y0:
            continue
        colors, visible = bins[y0:y2, x0:x2], ink[y0:y2, x0:x2]
        choices = []
        for hue in (-1, *range(12)):
            mask = visible & ((colors < 0) if hue < 0 else (colors == hue))
            local = _box_rectangle(mask)
            if local is None:
                continue
            a, b, c, d = local
            body = BBox(x1=a+x0, y1=b+y0, x2=c+x0, y2=d+y0)
            if not 12 <= body.height < 38 or not 14 <= body.width <= 120:
                continue
            overlap = (max(0., min(box.x2, body.x2)-max(box.x1, body.x1))
                       * max(0., min(box.y2, body.y2)-max(box.y1, body.y1)))
            if overlap/max(1., box.width*box.height+body.width*body.height-overlap) < .4:
                continue
            reference = _reference(ir, body)
            if (reference is None or reference.score < .95
                    or not _CONNECTOR_REFERENCE.fullmatch(reference.raw_text.strip())
                    or component.key and component.key != reference.raw_text.strip()):
                continue
            rectangle = (int(body.x1), int(body.y1), int(body.x2), int(body.y2), hue)
            strokes = terminal_strokes(rgb, rectangle, palette=palette, short_colored_leads=True)
            if not 2 <= len(strokes) <= 6 or len({side for side, _, _ in strokes}) != 1:
                continue
            positions = sorted(p for _, p, _ in strokes)
            if min(b-a for a, b in zip(positions, positions[1:])) < 6:
                continue
            choices.append((rectangle, reference, strokes))
        if len(choices) != 1:
            continue
        rectangle, reference, strokes = choices[0]
        if rectangle in rectangles:
            continue
        rectangles.append(rectangle)
        records.append({"component_id": component.internal_id, "reference": reference.raw_text,
                        "body": list(rectangle[:4]), "strokes": [list(s) for s in strokes]})
    ir.metadata["compact_connector_outlines"] = {"candidates": records, "gt_access": False}
    return tuple(rectangles)
