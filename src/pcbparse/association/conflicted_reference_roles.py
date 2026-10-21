"""Recover reference roles lost when overlapping OCR crops disagree on owner."""
from __future__ import annotations

import re

from pcbparse.association.ocr_common import bbox_gap_distance, preferred_text
from pcbparse.association.reference_prefixes import BOX_REFERENCE_PATTERN, DESIGNATOR_PATTERN, designator_compatible
from pcbparse.ir.models import Evidence

_REFERENCE=re.compile(DESIGNATOR_PATTERN)
_BOX_TYPES={'BOX','BOXIC','BOX_IC','IC'}
_BOX_REFERENCE=re.compile(BOX_REFERENCE_PATTERN)
_RESISTOR_ARRAY=re.compile(r'(?:RES(?:ISTOR)?|R)[ _-]*(?:PACK|ARRAY|NETWORK)[0-9]*',re.I)
_RAILS={'V','GND','GROUND','VCC','VDD','VSS','POWER'}


def _contains(body,point):
    return body.x1<=point.x<=body.x2 and body.y1<=point.y<=body.y2


def _array_caption(component,texts):
    body=component.body_bbox
    if body is None:
        return None
    captions=[]
    for t in texts:
        if (t.score<.90 or t.details not in {'','searchname'}
                or not _RESISTOR_ARRAY.fullmatch(t.raw_text)):
            continue
        b=t.bbox
        overlap=max(0.,min(body.x2,b.x2)-max(body.x1,b.x1))
        gap=min(abs(b.y1-body.y2),abs(b.y2-body.y1))
        if (overlap>=.8*min(body.width,b.width) and gap<=max(6.,.5*b.height)
                and (b.center.y<body.y1 or b.center.y>body.y2)):
            captions.append(t)
    return captions[0] if len(captions)==1 else None


def recover_conflicted_reference_roles(components,texts):
    """Use type and exterior placement, retaining uncertain crop interpretations.

    A crop halo does not prove a field role. Only untyped observations whose
    owner was explicitly cleared by the OCR merger are reconsidered. A real
    nearby compatible symbol must win uniquely; BGA/body text and nearby
    incompatible symbol contexts remain untouched.
    """
    index={c.internal_id:c for c in components}
    updates=[]
    for t in texts:
        if (t.details or t.component_id is not None or t.pin_id is not None or t.score<.90
                or any(e.source=='dataset'for e in t.evidence)):
            continue
        value=preferred_text(t)
        if value is None or not _REFERENCE.fullmatch(value):
            continue
        contexts=set();levels=set()
        for e in t.evidence:
            m=e.metadata
            if e.source=='ocr'and m.get('component_context_conflict')is True:
                contexts.update(cid for cid in m.get('component_context_candidates',[])if cid in index)
                levels.add(m.get('level'))
        if not contexts:
            continue
        b=t.bbox;choices=[];incompatible=[]
        # Internal package labels must not become references on another part.
        if any(c.category.upper()in _BOX_TYPES and _contains(c.body_bbox or c.bbox,b.center)
               for c in components):
            continue
        for cid in sorted(contexts):
            c=index[cid];category=c.category.upper();body=c.body_bbox or c.bbox
            gap=bbox_gap_distance(b,body)
            if c.score>=.75 and not designator_compatible(c.category,value):
                incompatible.append(gap)
            # A passive detector ROI may include part of its printed label.
            # A measured body is stronger evidence than that approximate ROI.
            inside=(_contains(body,b.center) if c.body_bbox is not None or category in _BOX_TYPES
                    else body.x1<=b.x1<=b.x2<=body.x2 and body.y1<=b.y1<=b.y2<=body.y2)
            if (category in _RAILS or not designator_compatible(c.category,value)
                    or inside or (c.key and c.key!=value)
                    or gap>min(24.,max(body.width,body.height)*.5,max(14.,1.5*b.height))):
                continue
            caption=None
            if category in _BOX_TYPES:
                # Lateral alphanumeric text is commonly a package pin label.
                if not (b.center.y<body.y1 or b.center.y>body.y2) or not body.x1<=b.center.x<=body.x2:
                    continue
                if not _BOX_REFERENCE.fullmatch(value):
                    if not re.fullmatch(r'(?:R|RN|RP)[0-9]+[A-Z]?',value):
                        continue
                    caption=_array_caption(c,texts)
                    if caption is None:
                        continue
            choices.append((gap,cid,caption))
        choices.sort(key=lambda r:(r[0],r[1]))
        if (not choices or (len(choices)>1 and choices[1][0]-choices[0][0]<=2.)
                or (incompatible and min(incompatible)<=choices[0][0])):
            continue
        gap,cid,caption=choices[0]
        t.details=t.category='designator';t.component_id=cid
        update={'text_id':t.internal_id,'raw_text':t.raw_text,'component_id':cid,
                'candidate_contexts':sorted(contexts),'original_levels':sorted(v for v in levels if isinstance(v,str)),
                'body_gap_px':gap,'array_caption_id':caption.internal_id if caption is not None else None,
                'gt_access':False,'coordinates_changed':False}
        t.evidence.append(Evidence(evidence_id='association:conflicted-reference-role:'+t.internal_id,
            source='rule',score=t.score,metadata={'kind':'conflicting-crop-reference-role',**update}))
        updates.append(update)
    return updates
