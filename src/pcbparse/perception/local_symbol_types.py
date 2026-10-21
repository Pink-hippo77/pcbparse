"""Verify small diode detections using their local strokes and printed prefix."""
from __future__ import annotations

import math
import re

from pcbparse.ir.models import BBox, Evidence
from pcbparse.perception.passive_visual_readback import _contains
from pcbparse.perception.two_terminal_readback import symbol_candidates

_REFERENCE = re.compile(r'D\d+[A-Z]?$')
_ALIASES = {'D': {'D', 'DIODE'}}
_REVIEWABLE = {'R', 'RESISTOR', 'C', 'CAPACITOR', 'L', 'INDUCTOR', 'D', 'DIODE',
               'CRYSTAL', 'FUSE', 'SWITCH', 'BOX', 'IC', 'OTHER'}


def _same_body(first, second):
    return min(_contains(first, second), _contains(second, first)) >= .65


def refine_local_symbol_types(ir, rgb, ctx=None):
    """Change type only after independent local shape confirmation.

    Existing detector boxes and pins remain intact. Printed IDs authorize a
    type hypothesis, never new geometry or an ID assignment. Competing bodies,
    conflicting type hypotheses and already supported categories abstain.
    Rectangles and coil outlines cannot resolve the type distinctions needed
    here; this pass supports only a diode's triangle/bar with two axial leads.
    """
    proposals = []
    checked = set()
    shape_cache = {}
    references = [t for t in ir.texts if math.isfinite(t.score) and t.score >= .98
                  and _REFERENCE.fullmatch(t.raw_text.strip())
                  and not any(e.source == 'dataset' for e in t.evidence)]
    for token in references:
        if ctx is not None and ctx.time_budget.remaining_seconds < 30:
            break
        reference, text = token.raw_text.strip(), token.bbox
        kind = reference[0]
        choices = []
        for component in ir.components:
            b = component.bbox
            if (component.category.upper() not in _REVIEWABLE or component.score < .3
                    or max(b.width, b.height) > 100 or min(b.width, b.height) < 5
                    or component.key and component.key != reference):
                continue
            gap = math.hypot(max(b.x1-text.x2, text.x1-b.x2, 0.),
                             max(b.y1-text.y2, text.y1-b.y2, 0.))
            if gap > max(12., min(25., 1.5*text.height)):
                continue
            rank = gap + .15*math.hypot(text.center.x-b.center.x, text.center.y-b.center.y)
            choices.append((rank, component.internal_id, component))
        choices.sort(key=lambda row: row[:2])
        if not choices:
            continue
        rank, _, nearest = choices[0]
        group = [c for _, _, c in choices if _same_body(nearest.bbox, c.bbox)]
        if any(r-rank < 4 and c not in group for r, _, c in choices):
            continue
        # Cross-class detector duplicates are competitors for the same body.
        # An existing compatible detection protects that body's current type.
        if any(c.category.upper() in _ALIASES[kind] and _same_body(nearest.bbox, c.bbox)
               for c in ir.components):
            continue
        component = min(group, key=lambda c: (-c.score, c.internal_id))
        b = component.bbox
        cache_key = (component.internal_id, kind)
        checked.add(cache_key)
        if cache_key not in shape_cache:
            region = BBox(x1=max(0., b.x1-8), y1=max(0., b.y1-8),
                          x2=min(ir.image.width, b.x2+8), y2=min(ir.image.height, b.y2+8))
            shapes = symbol_candidates(rgb, region, kind)
            shape_cache[cache_key] = [body for body, _, _ in shapes
                                      if _contains(b, body) >= .8 and _contains(body, b) >= .2]
        shapes = shape_cache[cache_key]
        if len(shapes) != 1:
            continue
        body = shapes[0]
        if body.x1 < text.center.x < body.x2 and body.y1 < text.center.y < body.y2:
            continue
        proposals.append((component, kind, token, body))

    updates = []
    for component in sorted(ir.components, key=lambda c: c.internal_id):
        rows = [row for row in proposals if row[0] is component]
        if len({row[1] for row in rows}) != 1:
            continue
        kind = rows[0][1]
        row = {'component_id': component.internal_id, 'previous_category': component.category,
               'category': kind, 'references': sorted({t.raw_text.strip() for _, _, t, _ in rows}),
               'token_ids': sorted({t.internal_id for _, _, t, _ in rows}),
               'visible_body': rows[0][3].model_dump(), 'gt_access': False,
               'geometry_changed': False, 'pins_created': 0, 'identity_assigned': False}
        component.category = kind
        component.evidence.append(Evidence(evidence_id='local-symbol-type:'+component.internal_id,
                                           source='geometry', metadata=row))
        updates.append(row)
    report = {'updates': updates, 'checked_hypotheses': len(checked), 'gt_access': False,
              'geometry_changed': False, 'pins_created': 0, 'identity_assigned': False}
    ir.metadata['local_symbol_types'] = report
    return report
