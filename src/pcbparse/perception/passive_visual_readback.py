"""Recover capacitor plates only when a nearby printed reference agrees.

The reference supplies identity; two separate, parallel plates with external
central leads supply geometry. Neither a text label nor a detector box alone
is sufficient to create a component.
"""
from __future__ import annotations

import math
import re

import numpy as np

from pcbparse.graph.palette_wire import hue_map
from pcbparse.ir.models import BBox, Component, Evidence, Pin, Point
from pcbparse.perception.rail_visual_readback import _bars, _stem
from pcbparse.perception.symbol_body import readback_rgb

_REF = re.compile(r'C\d+[A-Z]?$')
_VALUE = re.compile(r'(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*(?:[pnumkµμ]?F|[pnumµμ])?$')
_PRINTED_QUANTITY = re.compile(
    r'(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*(?:[pnumkKMµμ]?(?:F|H|V|A|Ohm|Ω)|[pnumµμ])'
    r'(?:/(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*V)?$', re.I)


def plate_pairs(rgb, region):
    h,w=rgb.shape[:2]
    x0,y0=max(0,math.floor(region.x1)),max(0,math.floor(region.y1))
    x2,y2=min(w,math.ceil(region.x2)),min(h,math.ceil(region.y2))
    bins,ink=hue_map(rgb[y0:y2,x0:x2])
    found=[]
    masks=[ink & (bins<0)]+[(bins==hue)&ink for hue in range(12) if (bins==hue).sum()>=15]
    for mask in masks:
        for transpose in (False,True):
            m=mask.T if transpose else mask
            bars=_bars(m)
            for i,a in enumerate(bars):
                length=a[3]-a[2]
                if not 7<=length<=95:continue
                for b in bars[i+1:]:
                    gap=b[0]-a[1]
                    if not 2<=gap<=min(25,length*.9):continue
                    if max(abs(a[k]-b[k]) for k in (2,3))>max(2,.12*length):continue
                    if gap==2:
                        # Low-resolution plates may have exactly one empty
                        # pixel column between them. Require a long pair and
                        # an entirely empty separator across their shared
                        # extent, so thin closed rectangles cannot qualify.
                        lo,hi=max(a[2],b[2]),min(a[3],b[3])
                        if length<12 or m[a[1]+1:b[0],lo:hi+1].any():continue
                    first,second=_stem(m,a,-1),_stem(m,b,1)
                    if min(first[0],second[0])<4:continue
                    axis=round((a[2]+a[3]+b[2]+b[3])/4)
                    # A visible dielectric gap distinguishes plates from a
                    # continuous wire or the center stroke of a symbol.
                    if m[a[1]+1:b[0],max(0,axis-1):axis+2].any(1).mean()>.25:continue
                    left,right=min(a[2],b[2]),max(a[3],b[3])
                    top,bottom=(a[0]+a[1])/2,(b[0]+b[1])/2
                    lead=min(15,max(5,length*.25))
                    ends=(top-min(first[0],lead),bottom+min(second[0],lead))
                    # A newly recovered narrow dielectric may have a thick
                    # plate. Its outer pixel row belongs to the symbol body,
                    # not to a wire on the other side of the capacitor.
                    body_top,body_bottom=(a[0],b[1]) if gap==2 else (top,bottom)
                    if transpose:
                        body=BBox(x1=x0+body_top,y1=y0+left,x2=x0+body_bottom,y2=y0+right)
                        points=[Point(x=x0+v,y=y0+axis) for v in ends]
                    else:
                        body=BBox(x1=x0+left,y1=y0+body_top,x2=x0+right,y2=y0+body_bottom)
                        points=[Point(x=x0+axis,y=y0+v) for v in ends]
                    if not any(max(abs(getattr(old[0],k)-getattr(body,k)) for k in ('x1','y1','x2','y2'))<3 for old in found):
                        found.append((body,points,'LR' if transpose else 'TB'))
    return found


def _contains(a,b):
    area=max(0,min(a.x2,b.x2)-max(a.x1,b.x1))*max(0,min(a.y2,b.y2)-max(a.y1,b.y1))
    return area/max(1,b.width*b.height)


def _capacitor_label_stack(reference,texts):
    """Use an independently printed second value line to bound the search.

    A reference above a value can be several text heights above the symbol.
    Cutting the image at a fixed distance from that first line hides plates.
    """
    b=reference.bbox;font=max(8.,min(35.,b.height));choices=[]
    for token in texts:
        t=token.bbox
        if (token is reference or token.score<.9 or not _PRINTED_QUANTITY.fullmatch(token.raw_text.strip())
                or not -.1*font<=t.y1-b.y2<=1.25*font or t.y2-b.y2>2*font
                or abs(t.center.x-b.center.x)>.75*font):
            continue
        choices.append((max(0.,t.y1-b.y2)+abs(t.center.x-b.center.x),token.internal_id,t))
    choices.sort(key=lambda r:r[:2])
    if not choices or len(choices)>1 and choices[1][0]-choices[0][0]<.2*font:
        return b
    t=choices[0][2]
    return BBox(x1=min(b.x1,t.x1),y1=b.y1,x2=max(b.x2,t.x2),y2=max(b.y2,t.y2))


def recover_visual_capacitors(ir,path,ctx=None,*,rgb=None,occluded_plates=False):
    if rgb is None:
        rgb=readback_rgb(ir,path)
    refs=[t for t in ir.texts if t.score>=.85 and _REF.fullmatch(t.raw_text)]
    # Overlapping short OCR fragments cannot override the complete reference.
    refs=[t for t in refs if not any(t.raw_text!=v.raw_text and v.raw_text.startswith(t.raw_text)
          and _contains(v.bbox,t.bbox)>.55 for v in refs)]
    proposals=[]
    for token in refs:
        if ctx:ctx.time_budget.check(stage='passive_readback')
        t=token.bbox;font=max(8,min(35,t.height))
        label=_capacitor_label_stack(token,ir.texts) if occluded_plates else t
        region=BBox(x1=max(0,label.x1-3*font),y1=max(0,label.y1-2*font),
                    x2=min(ir.image.width,label.x2+3*font),y2=min(ir.image.height,label.y2+3*font))
        choices=[]
        symbols=plate_pairs(rgb,region)
        if occluded_plates and not symbols:
            from pcbparse.perception.clipped_capacitors import clipped_plate_pairs
            symbols=clipped_plate_pairs(rgb,region,[v.bbox for v in ir.texts if v.score>=.5])
        for body,points,sides in symbols:
            gap=math.hypot(max(body.x1-label.x2,0,label.x1-body.x2),max(body.y1-label.y2,0,label.y1-body.y2))
            if gap>font*1.8:continue
            center=math.hypot(body.center.x-t.center.x,body.center.y-t.center.y)
            # Horizontal print normally starts beside the symbol. Penalize
            # a plate on the far side of the complete text, which otherwise
            # steals references in closely spaced capacitor banks.
            far_side=max(0,body.center.x-t.x1) if sides=='TB' and t.width>t.height else 0
            choices.append((gap+.18*center+.5*far_side,body,points,sides))
        choices.sort(key=lambda c:c[0])
        if not choices or (len(choices)>1 and choices[1][0]-choices[0][0]<5):continue
        rank,body,points,sides=choices[0]
        proposals.append((rank-token.score,token,body,points,sides))
    claimed=[];removed=set();report=[];quantity_fragments={}
    for _,reference,body,points,sides in sorted(proposals,key=lambda c:c[0]):
        if any(math.hypot(body.center.x-b.center.x,body.center.y-b.center.y)<6 for b in claimed):continue
        claimed.append(body)
        candidates=[]
        for c in ir.components:
            if c.internal_id in removed or c.category.upper() not in {'C','CAPACITOR','BOX','IC','BOXIC','BOX_IC'}:continue
            if any(e.metadata.get('method')=='visible-ic-outline-and-terminal-labels' for e in c.evidence):continue
            if c.category.upper() in {'BOX','IC','BOXIC','BOX_IC'} and c.body_bbox is not None:continue
            if _contains(c.bbox,body)<.65:continue
            candidates.append(c)
        passive=[c for c in candidates if c.category.upper() in {'C','CAPACITOR'}]
        existing=min(passive,key=lambda c:math.hypot(c.bbox.center.x-body.center.x,c.bbox.center.y-body.center.y)) if passive else None
        cid=existing.internal_id if existing else 'visual-cap:'+reference.internal_id
        if existing is None:
            bounds=BBox(x1=min(body.x1,*[p.x for p in points]),y1=min(body.y1,*[p.y for p in points]),
                        x2=max(body.x2,*[p.x for p in points]),y2=max(body.y2,*[p.y for p in points]))
            existing=Component(internal_id=cid,category='C',bbox=bounds,score=reference.score)
            ir.components.append(existing)
        removed.update(c.internal_id for c in candidates if c.internal_id!=cid)
        existing.category='C';existing.body_bbox=body
        existing.key=existing.name=reference.raw_text;existing.key_score=reference.score
        reference.component_id=cid;reference.details=reference.category='designator'
        # Prefer a complete printed quantity over a contained OCR fragment.
        quantities=[]
        for t in ir.texts:
            if t.score<.8 or not _VALUE.fullmatch(t.raw_text):continue
            gap=math.hypot(max(body.x1-t.bbox.x2,0,t.bbox.x1-body.x2),max(body.y1-t.bbox.y2,0,t.bbox.y1-body.y2))
            if gap>max(35,reference.bbox.height*2):continue
            # A clipped view of C16 or 0.47uF/16V may read only "16" or
            # "0". The enclosing observation does not authorize that shorter
            # string as a component quantity, even if its OCR score is high.
            parents=[v for v in ir.texts if v is not t and v.score>=.9
                     and len(v.raw_text)>len(t.raw_text) and t.raw_text in v.raw_text
                     and (v.raw_text==reference.raw_text or _PRINTED_QUANTITY.fullmatch(v.raw_text))
                     and .6*t.bbox.height<=v.bbox.height<=1.8*t.bbox.height
                     and _contains(v.bbox,t.bbox)>.6]
            if parents:
                # Rejected crop fragments must not become a neighboring
                # terminal number/name in the later semantic pass. Keep the
                # complete observation and its literal spelling instead.
                if not t.details and t.pin_id is None and not any(e.source=='dataset' for e in t.evidence):
                    parent=min(parents,key=lambda v:(-len(v.raw_text),-v.score,v.internal_id))
                    quantity_fragments[t.internal_id]={'raw_text':t.raw_text,'complete_text_id':parent.internal_id,
                        'complete_raw_text':parent.raw_text,'gt_access':False}
                continue
            aligned=(abs(t.bbox.x1-reference.bbox.x1)<reference.bbox.height*.8
                     and reference.bbox.height*.5<=t.bbox.center.y-reference.bbox.center.y<reference.bbox.height*1.8)
            if not aligned:continue
            quantities.append((bool(re.search(r'[F]$',t.raw_text)),len(t.raw_text),t.score,t))
        if quantities:
            value=max(quantities,key=lambda q:q[:3])[-1]
            existing.value=value.raw_text
            value.component_id=cid;value.details=value.category='value'
        row={'method':'visible-capacitor-plates-and-reference','component_id':cid,'reference':reference.raw_text,
             'reference_token':reference.internal_id,'body_bbox':body.model_dump(),'gt_access':False}
        existing.evidence.append(Evidence(evidence_id='visual-cap:'+cid,source='geometry',metadata=row))
        # Repeated symbol proposals often carry plate endpoints as terminals.
        # Two measured central leads retain the two distinct physical sides.
        ir.pins=[p for p in ir.pins if p.component_id!=cid]
        ir.pins.extend(Pin(internal_id=f'visual-cap-pin:{cid}:{side}',component_id=cid,point=point,orientation=side)
                       for side,point in zip(sides,points))
        report.append(row)
    ir.components=[c for c in ir.components if c.internal_id not in removed]
    ir.pins=[p for p in ir.pins if p.component_id not in removed]
    valid={p.internal_id for p in ir.pins}
    for token in ir.texts:
        if token.component_id in removed:token.component_id=None
        if token.pin_id and token.pin_id not in valid:token.pin_id=None
    ir.texts=[t for t in ir.texts if t.internal_id not in quantity_fragments]
    ir.metadata['visual_capacitor_readback']={'observations':report,'removed_duplicates':sorted(removed),
        'quantity_fragments':quantity_fragments}
    return ir.metadata['visual_capacitor_readback']
