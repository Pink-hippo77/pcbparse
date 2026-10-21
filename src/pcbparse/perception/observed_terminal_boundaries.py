"""Retain a terminal boundary observed independently of the learned tip.

Only previously verified IC outlines, capacitor plates or two-terminal R/L
symbols qualify. The colored stroke must terminate at a continuously visible,
separately colored wire.
This never supplies component identities, pin numbers or unobserved leads.
"""
from collections import Counter
import math

import numpy as np

from pcbparse.graph.palette_wire import hue_map
from pcbparse.ir.models import Evidence, Point
from pcbparse.perception.ic_visual_readback import _colored_lead_end

NORMALS={'L':(-1,0),'R':(1,0),'T':(0,-1),'B':(0,1)}


def _outline_hue(bins,ink,body):
    x1,y1,x2,y2=map(round,(body.x1,body.y1,body.x2,body.y2))
    h,w=ink.shape
    if not (0<=x1<x2<w and 0<=y1<y2<h):
        return None
    mask=np.zeros((y2-y1+1,x2-x1+1),dtype=bool)
    mask[:2,:]=True;mask[-2:,:]=True;mask[:,:2]=True;mask[:,-2:]=True
    local=bins[y1:y2+1,x1:x2+1]
    values=local[mask & ink[y1:y2+1,x1:x2+1] & (local>=0)]
    if len(values)<8:
        return None
    hue,count=Counter(map(int,values)).most_common(1)[0]
    return hue if count/len(values)>=.8 else None


def refine_observed_terminal_boundaries(ir,rgb,ctx=None):
    report={'updates':[],'measurements':[],'gt_access':False,'new_pins':0,'identities_changed':False,
            'method':'independently-measured-symbol-wire-boundaries'}
    if (rgb.max(2).astype(np.int16)-rgb.min(2).astype(np.int16)).max(initial=0)<50:
        ir.metadata['observed_terminal_boundaries']=report
        return report
    bins,ink=hue_map(rgb)
    index=ir.component_index()
    by_owner={}
    for pin in ir.pins:
        by_owner.setdefault(pin.component_id,[]).append(pin)
    for component in ir.components:
        if ctx is not None:
            ctx.time_budget.check(stage='observed_terminal_boundaries')
        body=component.body_bbox
        if body is None:
            continue
        is_cap=component.category.upper() in {'C','CAPACITOR'} and any(
            e.metadata.get('method')=='visible-capacitor-plates-and-reference' for e in component.evidence)
        is_ic=component.category.upper() in {'BOX','IC','BOXIC','BOX_IC'} and any(
            e.evidence_id.startswith('visual-ic-evidence:') for e in component.evidence)
        is_rl=component.category.upper() in {'R','RESISTOR','L','INDUCTOR'} and any(
            e.metadata.get('method')=='printed-reference-and-two-terminal-strokes'
            for e in component.evidence)
        owned=by_owner.get(component.internal_id,[])
        if is_rl:
            axis='TB' if body.height>body.width else 'LR'
            cross=[p.point.x if axis=='TB' else p.point.y for p in owned]
            center=(body.x1+body.x2)/2 if axis=='TB' else (body.y1+body.y2)/2
            width=body.width if axis=='TB' else body.height
            # A verified printed reference is insufficient if the two model
            # terminals disagree about the observed symbol's central axis.
            is_rl=(len(owned)==2 and {p.orientation for p in owned}==set(axis)
                   and max(cross)-min(cross)<=3
                   and all(abs(p-center)<=max(3,width*.2) for p in cross))
        if not (is_cap or is_ic or is_rl):
            continue
        hue=_outline_hue(bins,ink,body)
        if hue is None:
            continue
        for pin in owned:
            side=pin.orientation
            if side not in NORMALS:
                continue
            horizontal=side in 'LR'
            fixed=round({'L':body.x1,'R':body.x2,'T':body.y1,'B':body.y2}[side])
            position=round(pin.point.y if horizontal else pin.point.x)
            if is_ic:
                observations=[e.metadata for e in pin.evidence
                    if e.metadata.get('kind')=='visible-stroke-and-text-lanes' and e.metadata.get('side')==side]
                if not observations or observations[-1].get('decorated_lead'):
                    continue
                row=observations[-1]
                if abs(position-row['position'])>3:
                    continue
                position=round(row['position'])
            elif is_cap and side not in ('LR' if body.height>body.width else 'TB'):
                continue
            normal=-1 if side in 'LT' else 1
            length=_colored_lead_end(bins,ink,fixed,position,normal,horizontal,hue)
            if length is None:
                continue
            target=Point(x=fixed+normal*length,y=position) if horizontal else Point(x=position,y=fixed+normal*length)
            distance=math.hypot(pin.point.x-target.x,pin.point.y-target.y)
            # A strong local measurement may exceed the old fixed 24px cap on
            # scaled capacitor symbols; constrain it by the observed plate.
            limit=max(24.,1.5*(body.height if horizontal else body.width)) if is_cap else max(24.,length*.75)
            outward=(pin.point.x-fixed)*normal if horizontal else (pin.point.y-fixed)*normal
            if distance>limit or outward<2:
                continue
            if any(c.internal_id!=component.internal_id and c.body_bbox is not None
                   and c.body_bbox.x1<target.x<c.body_bbox.x2
                   and c.body_bbox.y1<target.y<c.body_bbox.y2 for c in ir.components):
                continue
            row={'pin_id':pin.internal_id,'component_id':pin.component_id,'previous':pin.point.model_dump(),
                 'point':target.model_dump(),'symbol_hue':hue,'lead_pixels':length,
                 'gt_access':False,'identities_changed':False}
            pin.evidence.append(Evidence(evidence_id='observed-boundary:'+pin.internal_id,source='geometry',
                metadata={'kind':'independently-measured-symbol-wire-boundary',**row}))
            report['measurements'].append(row)
            # A tip can already agree with the pixel measurement. Preserve
            # its evidence too: long lead trimming needs the measured extent,
            # not a requirement that the tip was previously wrong.
            if distance<.25:
                continue
            pin.point=target
            pin.base=Point(x=fixed,y=position) if horizontal else Point(x=position,y=fixed)
            report['updates'].append(row)
    ir.metadata['observed_terminal_boundaries']=report
    return report
