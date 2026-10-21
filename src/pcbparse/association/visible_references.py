"""Recover missing ordinary references from unambiguous external glyphs."""
from __future__ import annotations

import math
import re

from pcbparse.association.reference_prefixes import BOX_REFERENCE_PATTERN, DESIGNATOR_PATTERN, designator_compatible
from pcbparse.ir.models import Evidence

_PATTERN=re.compile(DESIGNATOR_PATTERN)
_SUPPORTED={'R','RESISTOR','C','CAPACITOR','L','INDUCTOR','D','DIODE','LED',
            'Q','BJT','MOSFET','TRANSISTOR','SWITCH','FUSE','CRYSTAL','BUZZER','OTHER','M3螺丝',
            'PIN','TESTPOINT'}
_BOX_REFERENCE=re.compile(BOX_REFERENCE_PATTERN)
_AMPLIFIER_REFERENCE=re.compile(r'(?:U|IC|A)\d+(?:[A-Z]|\.\d+)?$')


def recover_amplifier_references(ir):
    """Restore a unique printed reference on/next to an amplifier symbol.

    Multi-unit U8.1/U8.2 strings are references, and are not pin numbers.
    This pass accepts observed text only; ambiguous neighboring owners abstain.
    """
    amps=[c for c in ir.components if c.category.upper() in {'AMP','OPAMP','AMPLIFIER'}]
    available=[c for c in amps if not c.key]
    used={c.key for c in ir.components if c.key};proposals=[]
    for token in ir.texts:
        value=token.raw_text.strip()
        if token.score<.95 or not _AMPLIFIER_REFERENCE.fullmatch(value) or value in used:continue
        t=token.bbox;choices=[]
        for c in amps:
            b=c.bbox
            dx=max(b.x1-t.x2,0,t.x1-b.x2);dy=max(b.y1-t.y2,0,t.y1-b.y2)
            gap=math.hypot(dx,dy)
            if gap>max(18.,1.5*t.height):continue
            # Text on the opposite side of another symbol is not evidence
            # for this amplifier, even if an expanded detector ROI is nearby.
            if not (b.x1-t.width*.5<=t.center.x<=b.x2+t.width*.5):continue
            rank=gap+.12*math.hypot(t.center.x-b.center.x,t.center.y-b.center.y)
            choices.append((rank,c.internal_id,c))
        choices.sort(key=lambda r:r[:2])
        if not choices or len(choices)>1 and choices[1][0]-choices[0][0]<4:continue
        rank,_,c=choices[0]
        if c not in available:continue
        proposals.append((rank,value,token,c))
    updates=[]
    for rank,value,token,c in sorted(proposals,key=lambda r:(r[0],r[1],r[2].internal_id)):
        if c.key or value in used:continue
        if any(other is c and v!=value and r<=rank+4 for r,v,_,other in proposals):continue
        c.key,c.key_score=value,float(token.score)
        token.component_id,token.pin_id=c.internal_id,None
        token.details=token.category='designator'
        row={'component_id':c.internal_id,'reference':value,'token_id':token.internal_id,'gt_access':False}
        c.evidence.append(Evidence(evidence_id='amplifier-reference:'+c.internal_id,source='rule',
            metadata={'kind':'visible-amplifier-reference',**row}))
        updates.append(row);used.add(value)
    ir.metadata['amplifier_reference_recovery']={'updates':updates,'gt_access':False}
    return updates


def _box_reference_gap(body, text, *, lateral=False):
    gaps=[]
    if body.x1-.5*text.width<=text.center.x<=body.x2+.5*text.width:
        upper=(text.y1<body.y1+2 and text.y2<=body.y1+min(text.height,.15*body.height))
        lower=(text.y2>body.y2-2 and text.y1>=body.y2-min(text.height,.15*body.height))
        gap=min(abs(text.center.y-body.y1),abs(text.center.y-body.y2))
        if (upper or lower) and gap<=max(12.,1.5*text.height):gaps.append(gap)
    # Side-mounted references sit beside the top corner. Interior pin labels
    # and labels along the middle of a pin bank are not component evidence.
    if lateral and body.y1-.5*text.height<=text.center.y<=body.y1+min(.35*body.height,2*text.height):
        left=(text.x1<body.x1+2 and text.x2<=body.x1+min(text.width,.15*body.width))
        right=(text.x2>body.x2-2 and text.x1>=body.x2-min(text.width,.15*body.width))
        gap=min(abs(text.center.x-body.x1),abs(text.center.x-body.x2))
        if (left or right) and gap<=max(12.,1.5*text.width):gaps.append(gap)
    return min(gaps) if gaps else None


def recover_box_edge_references(ir):
    """Recover a unique printed edge reference, including a side top corner.

    Includes H/USB connector references and bottom-mounted module references.
    Multiple competing references or references already owned elsewhere abstain.
    """
    boxes=[c for c in ir.components if c.category.upper() in {'BOX','IC','BOXIC','BOX_IC'}]
    proposals=[]
    for c in boxes:
        # Independently measured IC reference/outline evidence is protected.
        if any(e.evidence_id.startswith('visual-ic-evidence:') for e in c.evidence):continue
        b=c.body_bbox or c.bbox;choices=[]
        for token in ir.texts:
            value=token.raw_text.strip();t=token.bbox
            if token.score<.95 or not _BOX_REFERENCE.fullmatch(value):continue
            gap=_box_reference_gap(b,t,lateral=not c.key)
            if gap is None:continue
            choices.append((value,token,gap))
        if len({v for v,_,_ in choices})!=1:continue
        value,token,gap=max(choices,key=lambda v:v[1].score)
        if value==c.key:continue
        if any(other is not c and other.key==value for other in ir.components):continue
        # A label equidistant from two detected boxes has no unique owner.
        competing=False
        for other in boxes:
            if other is c:continue
            ob=other.body_bbox or other.bbox
            d=_box_reference_gap(ob,token.bbox,lateral=True)
            if d is not None and d<=gap+3:competing=True;break
        if not competing:proposals.append((c,value,token))
    counts={value:sum(v==value for _,v,_ in proposals) for _,value,_ in proposals}
    updates=[]
    for c,value,token in proposals:
        if counts[value]!=1:continue
        row={'component_id':c.internal_id,'previous_key':c.key,'reference':value,
             'token_id':token.internal_id,'gt_access':False}
        c.key,c.key_score=value,float(token.score)
        token.component_id,token.pin_id,token.details,token.category=c.internal_id,None,'designator','designator'
        c.evidence.append(Evidence(evidence_id='box-edge-reference:'+c.internal_id,source='rule',
                                  metadata={'kind':'visible-box-edge-reference',**row}))
        updates.append(row)
    ir.metadata['box_edge_reference_recovery']={'updates':updates,'gt_access':False}
    return updates


def recover_visible_references(ir):
    """Keep existing keys; never infer a reference that OCR did not observe."""
    available=[c for c in ir.components if not c.key and c.category.upper() in _SUPPORTED]
    used={c.key for c in ir.components if c.key}
    boxes=[c.body_bbox or c.bbox for c in ir.components if c.category.upper() in {'BOX','IC'}
           and c.key and re.fullmatch(r'(?:U|IC)\d+[A-Z]?',c.key)]
    proposals=[]
    for token in ir.texts:
        value=token.raw_text.strip()
        if token.score<.95 or not _PATTERN.fullmatch(value) or value in used:
            continue
        point=token.bbox.center
        if any(b.x1<=point.x<=b.x2 and b.y1<=point.y<=b.y2 for b in boxes):
            continue
        choices=[]
        for component in available:
            if not designator_compatible(component.category,value):continue
            b=component.body_bbox or component.bbox
            if b.x1<point.x<b.x2 and b.y1<point.y<b.y2:continue
            dx=max(b.x1-token.bbox.x2,token.bbox.x1-b.x2,0.)
            dy=max(b.y1-token.bbox.y2,token.bbox.y1-b.y2,0.)
            gap=math.hypot(dx,dy)
            if gap>min(30.,max(12.,.7*max(b.width,b.height))):continue
            rank=gap+.15*math.hypot(point.x-b.center.x,point.y-b.center.y)
            choices.append((rank,component.internal_id,component))
        choices.sort(key=lambda r:r[:2])
        if not choices or len(choices)>1 and choices[1][0]-choices[0][0]<4.:
            continue
        rank,cid,component=choices[0]
        proposals.append((rank,value,token.internal_id,component,token))
    updates=[];assigned=set()
    for rank,value,tid,component,token in sorted(proposals,key=lambda r:r[:3]):
        if component.internal_id in assigned or value in used:continue
        competing=[p for p in proposals if p[3].internal_id==component.internal_id and p[1]!=value]
        if any(p[0]<=rank+4. for p in competing):continue
        before={'component_id':token.component_id,'pin_id':token.pin_id,'details':token.details}
        component.key,component.key_score=value,float(token.score)
        token.component_id,token.pin_id=component.internal_id,None
        token.details=token.category='designator'
        row={'component_id':component.internal_id,'reference':value,'token_id':tid,
             'previous_token_binding':before,'distance_rank_px':rank,'gt_access':False}
        component.evidence.append(Evidence(evidence_id='visible-reference:'+component.internal_id,source='rule',
            metadata={'kind':'external-visible-reference-recovery',**row}))
        updates.append(row);assigned.add(component.internal_id);used.add(value)
    report={'updates':updates,'coordinates_changed':False,'new_components':0,'new_pins':0,'gt_access':False}
    ir.metadata['visible_reference_recovery']=report
    return report
