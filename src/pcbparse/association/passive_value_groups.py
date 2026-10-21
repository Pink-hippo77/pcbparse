"""Recover literal passive quantities through observed printed reference groups.

Only existing component values and their evidence change. Unit conversion,
reference creation and geometry repair do not belong in this pass.
"""
from __future__ import annotations
from collections import defaultdict
import math,re
from pcbparse.ir.models import Evidence
from pcbparse.association.reference_prefixes import DESIGNATOR_PATTERN,designator_compatible

_REFERENCE=re.compile(DESIGNATOR_PATTERN)
_NUMBER=r'(?:\d+(?:[.,]\d+)?|[.,]\d+)'
_QUANTITIES={
    'R':re.compile(_NUMBER+r'\s*(?:[kKMmGg]?(?:Ω|Ω|[oO]hm[sS]?)|[kKMmGgRr])$'),
    'C':re.compile(_NUMBER+r'\s*(?:[pPnNuUµμmM]?[Ff]|[pPnNuUµμmM])$'),
    'L':re.compile(_NUMBER+r'\s*[pPnNuUµμmM]?[Hh]$'),
}
_KINDS={'R':'R','RESISTOR':'R','C':'C','CAPACITOR':'C','L':'L','INDUCTOR':'L'}

def _gap(a,b):
    return math.hypot(max(a.x1-b.x2,b.x1-a.x2,0.),max(a.y1-b.y2,b.y1-a.y2,0.))

def _group_geometry(reference,value):
    a,b=reference.bbox,value.bbox
    font=max(a.height,b.height)
    if min(a.height,b.height)<4 or not .65<=b.height/a.height<=1.55:
        return None
    overlap=max(0.,min(a.y2,b.y2)-max(a.y1,b.y1))/min(a.height,b.height)
    # Horizontal CAD annotations: R3 22R. A quantity above a different
    # reference must not be borrowed through a generic nearest-body rule.
    if (overlap>=.65 and b.x1>=a.x2-1. and b.x1-a.x2<=2.5*font):
        return {'layout':'same-line','rank':max(0,b.x1-a.x2)/font+
            2*abs(a.center.y-b.center.y)/font,'font':font}
    # Stacked CAD annotations: reference on top, quantity immediately below.
    aligned=min(abs(a.x1-b.x1),abs(a.center.x-b.center.x),abs(a.x2-b.x2))
    gap=b.y1-a.y2
    if (b.center.y-a.center.y>=.5*font and -.35*font<=gap<=.8*font and aligned<=.5*font):
        return {'layout':'stacked','rank':.6+max(0,gap)/font+2*aligned/font,'font':font}
    return None

def recover_passive_values(ir,*,replace=False):
    components=ir.component_index()
    # Include references for already keyed neighbours as competing anchors.
    references=[t for t in ir.texts if math.isfinite(t.score) and t.score>=.95 and _REFERENCE.fullmatch(t.raw_text.strip())
        and not any(e.source=='dataset' for e in t.evidence)]
    anchors=[]
    for c in ir.components:
        kind=_KINDS.get(c.category.upper())
        if not kind or not c.key or not designator_compatible(c.category,c.key):
            continue
        body=c.body_bbox or c.bbox
        for t in references:
            if t.raw_text.strip()!=c.key or _gap(t.bbox,body)>min(32.,max(14.,max(body.width,body.height))):
                continue
            anchors.append((c,kind,t))
    boxes=[c.body_bbox for c in ir.components if c.category.upper()in{'BOX','IC','BOXIC','BOX_IC'}
        and c.body_bbox is not None and any(e.evidence_id.startswith('visual-ic-evidence:')for e in c.evidence)]
    candidates=[]
    for token in ir.texts:
        value=token.raw_text.strip()
        if not math.isfinite(token.score) or token.score<.95 or any(e.source=='dataset'for e in token.evidence):
            continue
        if any(b.x1<token.bbox.center.x<b.x2 and b.y1<token.bbox.center.y<b.y2 for b in boxes):
            continue
        choices=[]
        for c,kind,reference in anchors:
            if not _QUANTITIES[kind].fullmatch(value):
                continue
            group=_group_geometry(reference,token)
            if group is None:
                continue
            body=c.body_bbox or c.bbox
            distance=_gap(body,token.bbox)
            if distance>min(36.,max(16.,group['font']*1.25,max(body.width,body.height)*.8)):
                continue
            score=group['rank']+.35*distance/group['font']
            choices.append((score,c.internal_id,reference.internal_id,c,reference,group))
        # Duplicate OCR views of a reference do not create competing owners.
        best={}
        for row in choices:
            if row[1]not in best or row[:3]<best[row[1]][:3]:best[row[1]]=row
        ranked=sorted(best.values(),key=lambda r:r[:3])
        if not ranked or len(ranked)>1 and ranked[1][0]-ranked[0][0]<.45:
            continue
        score,_,_,component,reference,group=ranked[0]
        # An unowned but closer printed reference is also a competitor.
        rivals=[]
        for other in references:
            if (other.raw_text.strip()==component.key
                    or not designator_compatible(component.category,other.raw_text.strip())):
                continue
            alternative=_group_geometry(other,token)
            if alternative and alternative['rank']<=group['rank']+.35:
                rivals.append(other.internal_id)
        if rivals:
            continue
        candidates.append((score,component,reference,token,group))
    by_owner=defaultdict(list)
    for row in candidates:by_owner[row[1].internal_id].append(row)
    updates=[];ambiguous=[]
    for cid,rows in sorted(by_owner.items()):
        component=components[cid]
        # Multiple agreeing views can support one literal. Conflicting
        # readings of the same glyphs are unresolved rather than rewritten.
        # An enclosing observation may add a cropped unit suffix, e.g.
        # 5.1k -> 5.1kΩ. Different digits or distinct locations stay ambiguous.
        complete=[]
        for row in rows:
            token=row[3];value=token.raw_text.strip();b=token.bbox
            if all(value.startswith(other[3].raw_text.strip())and(
                other[3].raw_text.strip()==value or(
                    b.width>=other[3].bbox.width+1 and
                    max(0,min(b.x2,other[3].bbox.x2)-max(b.x1,other[3].bbox.x1))*
                    max(0,min(b.y2,other[3].bbox.y2)-max(b.y1,other[3].bbox.y1))>=
                    .65*other[3].bbox.width*other[3].bbox.height))for other in rows):
                complete.append(row)
        if complete:rows=complete
        values={r[3].raw_text.strip()for r in rows}
        if len(values)!=1:
            ambiguous.append({'component_id':cid,'values':sorted(values)})
            continue
        score,_,reference,token,group=min(rows,key=lambda r:(r[0],-r[3].score,r[3].internal_id))
        value=token.raw_text.strip()
        if component.value==value or component.value and not replace:
            continue
        old=component.value
        if old and _QUANTITIES[_KINDS[component.category.upper()]].fullmatch(old.strip()):
            if not value.startswith(old.strip())or len(value)<=len(old.strip()):
                continue
        component.value=value
        row={'component_id':cid,'reference':component.key,'previous_value':old,'observed_value':value,
            'reference_token':reference.internal_id,'value_token':token.internal_id,
            'layout':group['layout'],'group_rank':score,'gt_access':False,
            'value_bbox':token.bbox.model_dump(),'reference_bbox':reference.bbox.model_dump()}
        component.evidence.append(Evidence(evidence_id='grouped-passive-value:'+cid,source='geometry',
            metadata={'kind':'printed-reference-and-quantity-group',**row}))
        updates.append(row)
    result={'method':'printed-reference-and-quantity-group-v1','updates':updates,'ambiguous':ambiguous,
        'coordinates_changed':False,'new_objects':0,'token_fields_changed':False,'gt_access':False}
    ir.metadata['grouped_passive_values']=result
    return result
