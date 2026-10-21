"""Recover two-terminal symbols from a printed reference and visible strokes."""
from __future__ import annotations

import math
import re

import numpy as np

from pcbparse.graph.palette_wire import hue_map
from pcbparse.ir.models import BBox, Component, Evidence, Pin, Point
from pcbparse.perception.ic_visual_readback import _runs
from pcbparse.perception.passive_visual_readback import _contains
from pcbparse.perception.rail_visual_readback import _bars, _stem
from pcbparse.perception.symbol_body import groups, connected_masks, readback_rgb

_REF = re.compile(r'([RL])\d+[A-Z]?$')
_QUANTITY = {'R': re.compile(r'\d+(?:[.,]\d+)?\s*(?:[kKmMgGΩ]|[kKmMgG]?Ohm)?$'),
             'L': re.compile(r'\d+(?:[.,]\d+)?\s*[pnumµμ]?H$')}
_IC = {'BOX', 'IC', 'BOXIC', 'BOX_IC'}


def recover_printed_switch_types(ir):
    """Resolve an Other symbol only when its printed reference and model agree.

    S/SW alone does not establish a symbol type. A separate, unique SW-PB
    model label must identify the same detected body. Geometry and existing
    pin labels are preserved; neighboring and conflicting owners abstain.
    """
    candidates = [c for c in ir.components if c.category.upper() in {'OTHER', 'SWITCH'}]
    references = [t for t in ir.texts if t.score >= .98 and re.fullmatch(r'(?:SW|S)\d+[A-Z]?', t.raw_text)]
    models = [t for t in ir.texts if t.score >= .98 and t.raw_text.strip().upper() == 'SW-PB']

    def owner(token):
        choices = []
        t = token.bbox
        font = max(6., t.height)
        for c in candidates:
            b = c.body_bbox or c.bbox
            if not (font <= max(b.width, b.height) <= 8*font and min(b.width, b.height) <= 4*font):
                continue
            gap = math.hypot(max(b.x1-t.x2, 0., t.x1-b.x2), max(b.y1-t.y2, 0., t.y1-b.y2))
            if gap > font or not b.x1-font <= t.center.x <= b.x2+font:
                continue
            choices.append((gap+.15*math.hypot(b.center.x-t.center.x, b.center.y-t.center.y), c.internal_id, c))
        choices.sort(key=lambda r: r[:2])
        if not choices or len(choices) > 1 and choices[1][0]-choices[0][0] < 4:
            return None
        return choices[0][2]

    reference_owners = [(t, owner(t)) for t in references]
    model_owners = [(t, owner(t)) for t in models]
    updates = []
    for c in candidates:
        if c.category.upper() != 'OTHER':
            continue
        refs = [t for t, target in reference_owners if target is c]
        labels = [t for t, target in model_owners if target is c]
        if len({t.raw_text for t in refs}) != 1 or not labels:
            continue
        reference = max(refs, key=lambda t: (t.score, t.internal_id))
        if c.key and c.key != reference.raw_text:
            continue
        if any(other is not c and other.key == reference.raw_text for other in ir.components):
            continue
        model = max(labels, key=lambda t: (t.score, t.internal_id))
        row = {'component_id': c.internal_id, 'previous_category': c.category, 'category': 'Switch',
               'reference': reference.raw_text, 'reference_token': reference.internal_id,
               'model_token': model.internal_id, 'model': model.raw_text, 'gt_access': False,
               'geometry_changed': False, 'pins_created': 0}
        c.category = 'Switch'
        c.key, c.key_score = reference.raw_text, float(reference.score)
        reference.component_id, reference.category, reference.details = c.internal_id, 'designator', 'designator'
        c.evidence.append(Evidence(evidence_id='printed-switch-type:'+c.internal_id, source='ocr', metadata=row))
        updates.append(row)
    ir.metadata['printed_switch_types'] = {'updates': updates, 'gt_access': False}
    return updates


def _rectangles(mask):
    """An empty rectangular resistor with two central external leads."""
    found = []
    for transpose in (False, True):
        m = mask.T if transpose else mask
        lines = _runs(m.T, 12)
        for i, (x1, a, b) in enumerate(lines):
            for x2, c, d in lines[i+1:]:
                width = x2-x1
                if not 5 <= width <= 35 or max(abs(a-c), abs(b-d)) > 2:
                    continue
                y1, y2 = round((a+c)/2), round((b+d)/2)-1
                if not 1.35*width <= y2-y1 <= 6*width:
                    continue
                if min(m[max(0,y-1):y+2,x1:x2+1].any(0).mean() for y in (y1,y2)) < .9:
                    continue
                axis = round((x1+x2)/2)
                if m[y1+3:y2-2,max(0,axis-1):axis+2].any(1).mean() > .35:
                    continue
                top, bottom = _stem(m,[y1,y1,x1,x2],-1), _stem(m,[y2,y2,x1,x2],1)
                if min(top[0],bottom[0]) < 4:
                    continue
                found.append(((x1,y1,x2,y2), [(axis,top[1]),(axis,bottom[1])], transpose))
    return found


def _coils(mask):
    """Require repeated same-side arches and straight leads on both ends."""
    found = []
    for transpose in (False, True):
        m = mask.T if transpose else mask
        for left,top,local in connected_masks(m):
            h,w = local.shape
            if not 4 <= h <= 30 or not max(25,3*h) <= w <= 250:
                continue
            if not local.any(0).all():
                continue
            curve = np.array([np.flatnonzero(col).mean() for col in local.T])
            end = 3
            baseline = float(np.median(np.r_[curve[:end],curve[-end:]]))
            if max(abs(curve[:end]-baseline).max(),abs(curve[-end:]-baseline).max()) > 1.5:
                continue
            delta = baseline-curve
            if abs(delta.min()) > abs(delta.max()):
                delta = -delta
            arches = groups(np.flatnonzero(delta >= max(2, .3*h)))
            if not 3 <= len(arches) <= 8 or min(len(g) for g in arches) < 2:
                continue
            centers = np.array([np.mean(g) for g in arches])
            gaps = np.diff(centers)
            if gaps.max() > 1.8*gaps.min() or arches[0][0] < 3 or arches[-1][-1] > w-4:
                continue
            a,b = left+arches[0][0]-2,left+arches[-1][-1]+2
            fixed = round(top+baseline)
            found.append(((a,top,b,top+h-1),[(left,fixed),(left+w-1,fixed)],transpose))
    return found


def _diodes(mask):
    found=[]
    for transpose in (False,True):
        m=mask.T if transpose else mask
        bars=_bars(m)
        for i,a in enumerate(bars):
            for b in bars[i+1:]:
                gap=b[0]-a[1];width=a[3]-a[2]
                if not 10<=width<=50 or not .35*width<=gap<=1.8*width:continue
                if max(abs(a[k]-b[k]) for k in (2,3))>3:continue
                x1,x2=min(a[2],b[2]),max(a[3],b[3])
                inner=m[a[1]+2:b[0]-1,x1:x2+1]
                if not len(inner) or inner.any(1).mean()<.8:continue
                spans=np.array([np.ptp(np.flatnonzero(row)) if row.any() else 0 for row in inner])
                if spans.max()<width*.65 or spans.min()>width*.35:continue
                top,bottom=_stem(m,a,-1),_stem(m,b,1)
                if min(top[0],bottom[0])<4:continue
                found.append(((x1,a[0],x2,b[1]),[(top[2],top[1]),(bottom[2],bottom[1])],transpose))
    return found


def symbol_candidates(rgb, region, kind, text_boxes=()):
    h,w = rgb.shape[:2]
    x0,y0 = max(0,math.floor(region.x1)),max(0,math.floor(region.y1))
    x2,y2 = min(w,math.ceil(region.x2)),min(h,math.ceil(region.y2))
    bins,ink = hue_map(rgb[y0:y2,x0:x2])
    masks = [(hue, (bins==hue)&ink) for hue in range(-1,12)]
    found = []
    for hue,mask in masks:
        if mask.sum() < 15:
            continue
        if hue < 0:
            for t in text_boxes:
                a,b = max(0,math.floor(t.x1)-x0),max(0,math.floor(t.y1)-y0)
                c,d = min(mask.shape[1],math.ceil(t.x2)-x0),min(mask.shape[0],math.ceil(t.y2)-y0)
                if a<c and b<d:
                    mask[b:d,a:c] = False
        detector = _rectangles if kind=='R' else (_coils if kind=='L' else _diodes)
        for bounds,terminals,transpose in detector(mask):
            a,b,c,d = bounds
            # Detectors measure their leads along a known axis. A diode body
            # can be square or wider than its lead axis, so its aspect ratio
            # cannot establish terminal direction. Restore the measured axis
            # alongside the coordinates when transposing the detector view.
            horizontal = (kind == 'L') != transpose
            if transpose:
                a,b,c,d = b,a,d,c
                terminals = [(v,u) for u,v in terminals]
            body = BBox(x1=x0+a,y1=y0+b,x2=x0+c,y2=y0+d)
            points = [Point(x=x0+u,y=y0+v) for u,v in terminals]
            if hue < 0:
                # Gray wire/symbol boundaries are invisible. Use the local
                # lead near the body, without following a wire to crop edges.
                for index,p in enumerate(points):
                    sign = -1 if index==0 else 1
                    if horizontal:p.x = (body.x1 if index==0 else body.x2)+sign*10
                    else:p.y = (body.y1 if index==0 else body.y2)+sign*10
            if not any(max(abs(getattr(body,k)-getattr(old[0],k)) for k in ('x1','y1','x2','y2'))<4 for old in found):
                found.append((body,points,'LR' if horizontal else 'TB'))
    return found


def _complete_diode_references(references):
    """A contained OCR prefix cannot displace a complete printed reference."""
    return [token for token in references if not any(
        token.raw_text != other.raw_text and (
            other.raw_text.startswith(token.raw_text) and _contains(other.bbox, token.bbox) > .55
            or token.score < .98 <= other.score and other.bbox.width > token.bbox.width
            and .75 <= other.bbox.height/max(1.,token.bbox.height) <= 1.33
            and _contains(other.bbox, token.bbox) > .8
        ) for other in references)]


def _read_diode_fields(ir,rgb):
    report=[]
    for c in ir.components:
        if c.category.upper() not in {'D','DIODE','LED'}:continue
        prefix='LED' if c.category.upper()=='LED' else 'D'
        references=[t for t in ir.texts if t.score>=.85 and re.fullmatch(prefix+r'\d+[A-Z]?',t.raw_text)
                    and math.hypot(t.bbox.center.x-c.bbox.center.x,t.bbox.center.y-c.bbox.center.y)<max(70,c.bbox.width+c.bbox.height)]
        references=_complete_diode_references(references)
        if not references:continue
        token=min(references,key=lambda t:math.hypot(t.bbox.center.x-c.bbox.center.x,t.bbox.center.y-c.bbox.center.y))
        font=max(10,min(30,token.bbox.height))
        models=[]
        for t in ir.texts:
            if t.score<.85 or not re.fullmatch(r'(?:\d+[A-Z][A-Z0-9.-]{2,}|[A-Z]{2,}[A-Z0-9.-]*\d[A-Z0-9.-]*)',t.raw_text):continue
            if t.raw_text==token.raw_text or re.fullmatch(r'(?:[A-Z]{1,4}|LED)\d+',t.raw_text):continue
            if abs(t.bbox.x1-token.bbox.x1)>font or not 0<t.bbox.center.y-token.bbox.center.y<font*2.5:continue
            models.append((t.score+.005*len(t.raw_text),t))
        if models:
            model=max(models,key=lambda v:v[0])[1]
            c.name=model.raw_text;model.component_id=c.internal_id;model.category=model.details='model'
        elif prefix=='LED':c.name=token.raw_text
        c.key=token.raw_text;c.key_score=token.score;token.component_id=c.internal_id;token.category=token.details='designator'
        choices=symbol_candidates(rgb,c.bbox,'D')
        if choices:
            body,points,sides=min(choices,key=lambda v:math.hypot(v[0].center.x-c.bbox.center.x,v[0].center.y-c.bbox.center.y))
            c.body_bbox=body
            ir.pins=[p for p in ir.pins if p.component_id!=c.internal_id]
            ir.pins.extend(Pin(internal_id=f'visual-diode-pin:{c.internal_id}:{s}',component_id=c.internal_id,point=p,orientation=s)
                           for s,p in zip(sides,points))
        row={'component_id':c.internal_id,'method':'visible-diode-reference-model-and-strokes','name':c.name,'gt_access':False}
        c.evidence.append(Evidence(evidence_id='visual-diode:'+c.internal_id,source='geometry',metadata=row));report.append(row)
    return report


def recover_two_terminal_symbols(ir,path,ctx=None,*,rgb=None,occluded_rectangles=False):
    if rgb is None:
        rgb=readback_rgb(ir,path)
    glyphs = [t.bbox for t in ir.texts if t.score>=.75]
    proposals = []
    for token in ir.texts:
        match = _REF.fullmatch(token.raw_text)
        if token.score<.9 or match is None:
            continue
        if ctx:ctx.time_budget.check(stage='two_terminal_readback')
        kind,t = match[1],token.bbox
        font = max(10,min(30,t.height))
        region = BBox(x1=max(0,t.x1-4*font),y1=max(0,t.y1-2*font),
                      x2=min(ir.image.width,t.x2+7*font),y2=min(ir.image.height,t.y2+4*font))
        choices = []
        if occluded_rectangles:
            from pcbparse.perception.printed_rectangle_retry import retry_printed_rectangles
            symbols=retry_printed_rectangles(ir,rgb,region,kind,glyphs,symbol_candidates)
        else:
            symbols=symbol_candidates(rgb,region,kind,glyphs)
        for body,points,sides in symbols:
            if any(c.category.upper() in _IC and c.body_bbox is not None
                   and _contains(c.body_bbox,body)>.8 and c.body_bbox.width>body.width*2 for c in ir.components):
                continue
            gap = math.hypot(max(body.x1-t.x2,0,t.x1-body.x2),max(body.y1-t.y2,0,t.y1-body.y2))
            if gap>font*1.7:continue
            rank = gap+.15*math.hypot(body.center.x-t.center.x,body.center.y-t.center.y)
            choices.append((rank,body,points,sides))
        choices.sort(key=lambda x:x[0])
        if choices and (len(choices)==1 or choices[1][0]-choices[0][0]>5):
            proposals.append((choices[0][0],token,kind,*choices[0][1:]))
    claimed=[];removed=set();report=[]
    for _,token,kind,body,points,sides in sorted(proposals,key=lambda x:x[0]):
        if any(math.hypot(body.center.x-b.center.x,body.center.y-b.center.y)<6 for b in claimed):continue
        claimed.append(body)
        owners=[c for c in ir.components if c.internal_id not in removed
                and c.category.upper() in {kind,'RESISTOR' if kind=='R' else 'INDUCTOR','BOX','FUSE'}
                and not (c.body_bbox is not None and c.category.upper() in _IC)
                and _contains(c.bbox,body)>.65]
        owner=min(owners,key=lambda c:math.hypot(c.bbox.center.x-body.center.x,c.bbox.center.y-body.center.y)) if owners else None
        if owner is None:
            owner=Component(internal_id='visual-rl:'+token.internal_id,category=kind,
                bbox=BBox(x1=min(body.x1,*[p.x for p in points]),y1=min(body.y1,*[p.y for p in points]),
                          x2=max(body.x2,*[p.x for p in points]),y2=max(body.y2,*[p.y for p in points])),score=token.score)
            ir.components.append(owner)
        removed.update(c.internal_id for c in owners if c is not owner)
        owner.category=kind;owner.body_bbox=body;owner.key=owner.name=token.raw_text;owner.key_score=token.score
        token.component_id=owner.internal_id;token.category=token.details='designator'
        quantities=[]
        for value in ir.texts:
            b=value.bbox
            if value.score<.85 or not _QUANTITY[kind].fullmatch(value.raw_text):continue
            if abs(b.x1-token.bbox.x1)>font or not 0<b.center.y-token.bbox.center.y<font*3:continue
            gap=math.hypot(max(body.x1-b.x2,0,b.x1-body.x2),max(body.y1-b.y2,0,b.y1-body.y2))
            if gap<=font*2.2:quantities.append((value.score+.002*len(value.raw_text),value))
        if quantities:
            value=max(quantities,key=lambda x:x[0])[1]
            owner.value=value.raw_text;value.component_id=owner.internal_id;value.category=value.details='value'
        cid=owner.internal_id
        ir.pins=[p for p in ir.pins if p.component_id!=cid]
        ir.pins.extend(Pin(internal_id=f'visual-rl-pin:{cid}:{side}',component_id=cid,point=p,orientation=side)
                       for side,p in zip(sides,points))
        row={'method':'printed-reference-and-two-terminal-strokes','kind':kind,'component_id':cid,
             'reference':token.raw_text,'body_bbox':body.model_dump(),'gt_access':False}
        owner.evidence.append(Evidence(evidence_id='visual-rl:'+cid,source='geometry',metadata=row));report.append(row)
    recover_printed_switch_types(ir)
    report.extend(_read_diode_fields(ir,rgb))
    ir.components=[c for c in ir.components if c.internal_id not in removed]
    ir.pins=[p for p in ir.pins if p.component_id not in removed]
    valid={p.internal_id for p in ir.pins}
    for token in ir.texts:
        if token.component_id in removed:token.component_id=None
        if token.pin_id and token.pin_id not in valid:token.pin_id=None
    ir.metadata['visual_two_terminal_readback']={'observations':report,'removed_duplicates':sorted(removed)}
    return ir.metadata['visual_two_terminal_readback']
