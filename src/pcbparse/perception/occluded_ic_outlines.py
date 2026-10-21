"""Recover a detected IC outline whose single horizontal edge meets its title.

Only a local detector ROI is searched. Three measured edges, an independent
external reference and a complete OCR title covering every missing edge pixel
are required; this module supplies neither terminals nor their identities.
"""
from __future__ import annotations

import math
import re

import numpy as np

from pcbparse.ir.models import BBox


def _iou(a, b):
    area = max(0., min(a.x2,b.x2)-max(a.x1,b.x1))*max(0., min(a.y2,b.y2)-max(a.y1,b.y1))
    return area/max(1., a.width*a.height+b.width*b.height-area)


def _title_covers_gap(ir, body, y, visible):
    from pcbparse.perception.ic_visual_readback import _QUANTITY
    missing = np.flatnonzero(~visible)+int(body.x1)
    if len(missing) < 3:
        return None
    choices = []
    for token in ir.texts:
        value, b = token.raw_text.strip(), token.bbox
        if (token.score < .95 or not 5 <= len(value) <= 64
                or not re.fullmatch(r'[A-Za-z0-9_+./() \-]+', value)
                or not re.search(r'[A-Za-z]', value) or not re.search(r'\d{2,}', value)
                or _QUANTITY.fullmatch(value)
                or b.width > body.width*1.1 or b.height > min(32., body.height*.5)
                or not b.y1-2 <= y <= b.y2+2
                or min(b.x2,body.x2)-max(b.x1,body.x1) < .8*b.width):
            continue
        # A nearby title does not license a gap elsewhere on the same edge.
        if np.all((missing >= b.x1-2) & (missing <= b.x2+2)):
            choices.append(token)
    if not choices:
        return None
    return max(choices, key=lambda t:(t.score, len(t.raw_text), t.internal_id))


def occluded_ic_outlines(ir, palette, rectangles, *, check=None):
    """Return unique, independently supported supplements to closed outlines."""
    # Import locally: the ordinary readback calls this optional supplement.
    from pcbparse.perception.ic_visual_readback import _IC, _reference, _vertical_edges

    bins, ink = palette
    measured = [BBox(x1=r[0],y1=r[1],x2=r[2],y2=r[3]) for r in rectangles]
    found, observations = [], {}
    owners = [c for c in ir.components if c.category.upper() in _IC and c.score >= .75
              and c.body_bbox is None and any(e.source == 'detector' for e in c.evidence)]
    for component in owners[:128]:
        if check:
            check()
        b = component.bbox
        if b.width*b.height > .5*ir.image.width*ir.image.height:
            continue
        if any(_iou(b, body) >= .4 for body in measured):
            continue
        pad = 8
        x0,y0 = max(0,math.floor(b.x1)-pad),max(0,math.floor(b.y1)-pad)
        x3,y3 = min(ink.shape[1],math.ceil(b.x2)+pad),min(ink.shape[0],math.ceil(b.y2)+pad)
        local_ink, local_bins = ink[y0:y3,x0:x3], bins[y0:y3,x0:x3]
        candidates = []
        for hue in sorted(int(h) for h in np.unique(local_bins[local_ink])):
            if check:
                check()
            mask = local_ink & (local_bins == hue)
            if np.count_nonzero(mask) < 100:
                continue
            edges = _vertical_edges(mask)
            for i,(left,a,z) in enumerate(edges):
                if check and i % 64 == 0:
                    check()
                for right,c,d in edges[i+1:]:
                    if right-left < 38 or abs(a-c) > 3 or abs(z-d) > 3:
                        continue
                    top,bottom = round((a+c)/2),round((z+d)/2)-1
                    if bottom-top < 38:
                        continue
                    body = BBox(x1=left+x0,y1=top+y0,x2=right+x0,y2=bottom+y0)
                    if _iou(b,body) < .5 or body.width*body.height > .5*ir.image.width*ir.image.height:
                        continue
                    # Both vertical edges must themselves remain visible.
                    if min(mask[top:bottom+1,left].mean(),mask[top:bottom+1,right].mean()) < .95:
                        continue
                    horizontal = [mask[max(0,y-2):y+3,left:right+1].any(0) for y in (top,bottom)]
                    coverage = [float(v.mean()) for v in horizontal]
                    if sum(v >= .95 for v in coverage) != 1 or min(coverage) < .5:
                        continue
                    side = int(coverage[1] < .95)
                    title = _title_covers_gap(ir,body,body.y2 if side else body.y1,horizontal[side])
                    if title is None:
                        continue
                    reference = _reference(ir,body)
                    if (reference is None or reference.score < .95
                            or component.key and component.key != reference.raw_text.strip()):
                        continue
                    # Overlapping detector objects must not claim one body.
                    if sum(_iou(other.bbox,body) >= .5 for other in owners) != 1:
                        continue
                    rectangle = (int(body.x1),int(body.y1),int(body.x2),int(body.y2),hue)
                    if any(max(abs(u-v) for u,v in zip(rectangle[:4],old[0][:4])) < 5 for old in candidates):
                        continue
                    candidates.append((rectangle,{'kind':'title-occluded-detected-ic-outline',
                        'component_id':component.internal_id,'reference_token':reference.internal_id,
                        'title_token':title.internal_id,'occluded_edge':'B' if side else 'T',
                        'horizontal_coverage':coverage,'gt_access':False}))
        # Competing outlines leave the original detector observation intact.
        if len(candidates) == 1:
            rectangle, evidence = candidates[0]
            found.append(rectangle)
            observations[rectangle] = evidence
    return found, observations
