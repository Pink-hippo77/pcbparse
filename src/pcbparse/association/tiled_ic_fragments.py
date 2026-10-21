"""Reject clipped duplicate IC banks with independent image and OCR evidence."""
from __future__ import annotations

import math
import re

from pcbparse.perception.symbol_body import readback_rgb

_BOX = {'BOX', 'IC', 'BOXIC', 'BOX_IC'}


def _tile_cut(component, body):
    """An unchanged detector box must start at a real interior tile edge."""
    b = component.bbox
    for e in component.evidence:
        if e.source != 'detector' or e.metadata.get('touches_internal_tile_border') is not True:
            continue
        binding = e.metadata.get('local_to_global_binding', {})
        local = binding.get('local_geometry', {}).get('values', [])
        original = binding.get('original_geometry', {}).get('values', [])
        size = binding.get('view_size', [])
        if (binding.get('contract') != 'image_view_local_to_original_v1'
                or len(local) != 4 or len(original) != 4 or len(size) != 2
                or any(abs(a-z) > .001 for a, z in zip(original, (b.x1,b.y1,b.x2,b.y2)))):
            continue
        # This pass handles partial left/right banks only. Top/bottom and
        # corner fragments keep their original predictions until verified.
        if abs(local[0]) <= 1 and body.x1+8 < b.x1 < body.x2-8:
            side, far = 'R', abs(b.x2-body.x2)
        elif abs(local[2]-size[0]) <= 1 and body.x1+8 < b.x2 < body.x2-8:
            side, far = 'L', abs(b.x1-body.x1)
        else:
            continue
        if (far <= 32 and abs(b.y1-body.y1) <= 32 and abs(b.y2-body.y2) <= 32):
            return side
    return None


def _observed_rows(ir, parent, side):
    body = parent.body_bbox
    measured_body = any(e.source == 'geometry'
        and e.metadata.get('method') == 'visible-symbol-body-v2'
        and e.metadata.get('gt_access') is False
        and e.metadata.get('body_bbox') == body.model_dump() for e in parent.evidence)
    reference = any(e.source == 'geometry' and e.evidence_id.startswith('visual-ic-evidence:')
        and e.metadata.get('gt_access') is False and e.metadata.get('reference') == parent.key
        for e in parent.evidence)
    if not measured_body or not reference:
        return []
    rows = []
    for pin in ir.pins:
        if pin.component_id != parent.internal_id or pin.orientation != side:
            continue
        for e in reversed(pin.evidence):
            row = e.metadata
            if (e.source != 'geometry' or row.get('kind') != 'visible-stroke-and-text-lanes'
                    or row.get('gt_access') is not False or row.get('image_sha256') != ir.image.sha256):
                continue
            axis, tip = row.get('position'), row.get('tip')
            if (type(axis) not in {int,float} or type(tip) not in {int,float}
                    or not math.isfinite(axis) or not math.isfinite(tip)):
                break
            rows.append((float(axis), float(tip), pin.name, row.get('name_score', 0.)))
            break
    return sorted(rows)


def _reference_is_signal_echo(ir, component, parent, rows, side):
    """A lost dot in P4.4 may look like a connector reference P44."""
    if not component.key or not re.fullmatch(r'[A-Z]+\d+', component.key):
        return False
    tokens = [t for t in ir.texts if t.component_id == component.internal_id
              and t.pin_id is None and t.raw_text.strip() == component.key
              and t.details == 'designator' and t.score >= .95]
    if not tokens:
        return False
    body = parent.body_bbox
    for token in tokens:
        t = token.bbox
        gap = t.x1-body.x2 if side == 'R' else body.x1-t.x2
        if not 0 <= gap <= 48:
            continue
        for axis, _, name, score in rows:
            pitch = min([abs(a-axis) for a,_,_,_ in rows if a != axis] + [20.])
            if score < .95 or not axis-min(12., pitch+2) <= t.center.y <= axis+2:
                continue
            if any('.' in part and part.replace('.', '') == component.key
                   for part in name.split('/')):
                return True
    return False


def _same_visible_bank(ink, body, pins, rows, side):
    """Every fragment terminal must lie beside an already measured lead.

    The fragment's learned endpoint may overshoot a no-connect mark. Check
    pixels along the independently measured lead, not that spurious overshoot.
    This never repositions or joins either terminal.
    """
    if len(pins) < 3 or any(p.orientation != side for p in pins):
        return False
    if max(p.point.y for p in pins)-min(p.point.y for p in pins) < 20:
        return False
    used = set()
    for pin in pins:
        choices = sorted((abs(axis-pin.point.y), axis, tip)
                         for axis,tip,_,_ in rows)
        if not choices or choices[0][0] > 5:
            return False
        _, axis, tip = choices[0]
        if axis in used:
            return False
        used.add(axis)
        if side == 'R':
            if not body.x2 < tip <= pin.point.x <= min(body.x2+32,tip+16):
                return False
            left, right = round(body.x2), round(tip)
        else:
            if not max(body.x1-32,tip-16) <= pin.point.x <= tip < body.x1:
                return False
            left, right = round(tip), round(body.x1)
        y = round(axis)
        if not (0 <= left < right < ink.shape[1] and 1 <= y < ink.shape[0]-1):
            return False
        if not ink[y-1:y+2,left:right+1].any(axis=0).all():
            return False
    return True


def reject_tiled_ic_fragments(ir, path, ctx=None):
    """Remove a proven duplicate detector fragment, never merge pin identities.

    Containment alone cannot reject a symbol. Require an unchanged interior
    tile cut, a separately measured complete IC, a reference that echoes a
    dotted signal on that IC, and a one-to-one visibly shared terminal bank.
    Any separate measured symbol or ambiguous parent keeps the prediction.
    """
    report = ir.metadata.setdefault('tiled_ic_fragments', {'removed': [], 'gt_access': False})
    hypotheses = []
    for c in ir.components:
        if c.category.upper() not in _BOX or c.body_bbox is not None:
            continue
        if any(e.source in {'geometry','qwen'} for e in c.evidence):
            continue
        choices = []
        b = c.bbox
        for parent in ir.components:
            body = parent.body_bbox
            if parent is c or body is None or not parent.key or parent.category.upper() not in _BOX:
                continue
            if not (body.x1 < b.center.x < body.x2 and body.y1 < b.center.y < body.y2):
                continue
            overlap = max(0.,min(b.x2,body.x2)-max(b.x1,body.x1))*max(0.,min(b.y2,body.y2)-max(b.y1,body.y1))
            if overlap/max(1.,b.width*b.height) < .65:
                continue
            side = _tile_cut(c, body)
            rows = _observed_rows(ir,parent,side) if side else []
            if len(rows) < 3 or not _reference_is_signal_echo(ir,c,parent,rows,side):
                continue
            choices.append((parent,rows,side))
        if len(choices) == 1:
            hypotheses.append((c,*choices[0]))
    if not hypotheses:
        return report
    from pcbparse.graph.palette_wire import hue_map
    _, ink = hue_map(readback_rgb(ir,path))
    removed_components, removed_pins = set(), set()
    for c,parent,rows,side in hypotheses:
        if ctx is not None:
            ctx.time_budget.check(stage='tiled_ic_fragment_review')
        pins = [p for p in ir.pins if p.component_id == c.internal_id]
        if not _same_visible_bank(ink,parent.body_bbox,pins,rows,side):
            continue
        removed_components.add(c.internal_id)
        removed_pins.update(p.internal_id for p in pins)
        # Keep only the observed detector geometry for a palette sensitivity
        # check. These proposals never re-enter the IR, wire obstacles, pin
        # ownership or official output after rejection.
        report.setdefault('palette_observations', []).append({
            'component': {'internal_id': c.internal_id, 'category': c.category,
                          'bbox': c.bbox.model_dump()},
            'pins': [{'internal_id': p.internal_id, 'component_id': p.component_id,
                      'point': p.point.model_dump(), 'orientation': p.orientation}
                     for p in pins]})
        report['removed'].append({'component_id':c.internal_id,'reference':c.key,
            'parent_id':parent.internal_id,'parent_reference':parent.key,
            'pin_ids':[p.internal_id for p in pins], 'reason':'clipped_duplicate_visible_bank',
            'image_sha256':ir.image.sha256,'coordinates_changed':False,'gt_access':False})
    ir.components[:] = [c for c in ir.components if c.internal_id not in removed_components]
    ir.pins[:] = [p for p in ir.pins if p.internal_id not in removed_pins]
    for token in ir.texts:
        if token.component_id in removed_components or token.pin_id in removed_pins:
            token.component_id, token.pin_id = None, None
            token.details = token.category = 'unknown'
    for net in [*ir.physical_nets,*ir.electrical_nets]:
        net.pins = [pid for pid in net.pins if pid not in removed_pins]
    return report
