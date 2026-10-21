"""Resolve crop-truncated observations and unambiguous visible IC titles."""
from __future__ import annotations

import re
from pcbparse.ir.models import BBox, Evidence, TextToken

_NUMERIC_QUANTITY=re.compile(r'^(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*[A-Za-zΩµμ%]+$')


def _clipped(token):
    observations=[]
    for e in token.evidence:
        if e.source!='ocr':
            continue
        b=e.metadata.get('local_to_global_binding',{})
        values=b.get('local_geometry',{}).get('values')
        size=b.get('view_size')
        if values and size and len(values)==4:
            observations.append(min(values[0],values[1],size[0]-values[2],size[1]-values[3])<=2.)
    return bool(observations) and all(observations)


def clean_ocr_observations(ir):
    removed={}
    for short in ir.texts:
        if not _clipped(short) or not short.raw_text or short.details:
            continue
        a=short.bbox
        for full in ir.texts:
            if full is short or full.score<.90 or _clipped(full):
                continue
            if not (0<len(full.raw_text)-len(short.raw_text)<=8 and short.raw_text in full.raw_text):
                continue
            b=full.bbox
            overlap=max(0.,min(a.x2,b.x2)-max(a.x1,b.x1))*max(0.,min(a.y2,b.y2)-max(a.y1,b.y1))
            if overlap/max(1.,a.width*a.height)<.75 or not .6<=a.height/max(1.,b.height)<=1.6:
                continue
            removed[short.internal_id]={'raw_text':short.raw_text,'replacement_id':full.internal_id,
                'replacement_raw_text':full.raw_text,'reason':'crop_border_clipped_substring_of_same_pixels'}
            break
    ir.texts=[t for t in ir.texts if t.internal_id not in removed]
    ir.metadata['ocr_crop_cleanup']={'gt_access':False,'removed':removed}
    return removed


def classify_visible_ic_titles(ir):
    assigned={}
    references=re.compile(r'^(?:R|C|L|D|Q|U|J|P|H|IC|CN|SW|FB|TP)\d+[A-Z]?$')
    for t in ir.texts:
        value=t.raw_text
        if (t.details or t.score<.90 or not re.fullmatch(r'[A-Za-z0-9_+./() \-]{5,64}',value)
                or not re.search(r'[A-Za-z]',value) or not re.search(r'\d{2,}',value)
                or references.fullmatch(value.upper()) or _NUMERIC_QUANTITY.fullmatch(value)):
            continue
        box=t.bbox
        choices=[]
        for c in ir.components:
            if c.category.upper() not in {'BOX','IC','BOXIC','BOX_IC'}:
                continue
            b=c.bbox
            edge=min(abs(box.center.y-b.y1),abs(box.center.y-b.y2))
            if edge>max(16.,box.height) or box.center.y<b.y1-20 or box.center.y>b.y2+20:
                continue
            # A printed part number can extend past a narrow symbol's left or
            # right edge. Require substantial horizontal overlap and placement
            # outside the top/bottom edge, rather than containment in the body.
            # Detector artifacts on the title row must not turn that exterior
            # text into a pin function. Interior text retains the stricter
            # original containment and pin-row checks below.
            overlap=max(0.,min(box.x2,b.x2)-max(box.x1,b.x1))
            exterior=(box.center.y<=b.y1+.15*box.height
                      or box.center.y>=b.y2-.15*box.height)
            if (exterior and overlap>=.55*min(box.width,b.width)
                    and box.width<=4*b.width):
                choices.append(c)
                continue
            if box.x1<b.x1-6 or box.x2>b.x2+6 or box.width>b.width*1.15:
                continue
            # Titles must not occupy an observed pin-label row/column.
            aligned=False
            for p in ir.pins:
                if p.component_id!=c.internal_id:
                    continue
                if p.orientation in {'L','R'}:
                    aligned|=box.y1-3<=p.point.y<=box.y2+3
                elif p.orientation in {'T','B'}:
                    # A top/bottom terminal's column extends across the whole
                    # symbol. Only nearby text can be its function label;
                    # distant centered model text must not be vetoed by it.
                    body=c.body_bbox or b
                    gap=(box.y1-body.y1 if p.orientation=='T' else body.y2-box.y2)
                    aligned|=(box.x1-3<=p.point.x<=box.x2+3
                              and gap<=max(16.,box.height))
            if not aligned:
                choices.append(c)
        if len(choices)!=1:
            continue
        t.details=t.category='searchname'
        t.component_id=choices[0].internal_id
        assigned[t.internal_id]=t.component_id
        t.evidence.append(Evidence(evidence_id='association:visible-ic-title:'+t.internal_id,
            source='rule',score=t.score,metadata={'kind':'visible-ic-title','gt_access':False,
                'raw_text':value,'component_id':t.component_id}))
    continuations=_join_title_continuations(ir,assigned)
    ir.metadata['visible_ic_titles']={'gt_access':False,'assignments':assigned,
                                     'continuations':continuations}
    return assigned


def _join_title_continuations(ir,assigned):
    """Join two observed model alternatives separated by a printed slash.

    One line must already have an unambiguous title role. Both lines must
    be inside the same body, centered, adjacent, and repeat the model stem.
    Pin rows alone cannot establish this role. No characters are inferred.
    """
    records=[]
    components=ir.component_index()
    for first in list(ir.texts):
        owner=assigned.get(first.internal_id)
        if owner is None or not first.raw_text.endswith('/'):
            continue
        stem=first.raw_text[:-1]
        if not re.fullmatch(r'[A-Za-z]+[0-9]{2,}[A-Za-z0-9]*',stem):
            continue
        body=components[owner].bbox
        a=first.bbox
        if not (body.x1<=a.x1<a.x2<=body.x2 and body.y1<=a.y1<a.y2<=body.y2):
            continue
        candidates=[]
        for second in ir.texts:
            if (second is first or second.score<.90 or second.pin_id is not None
                    or (second.details and assigned.get(second.internal_id)!=owner)
                    or second.component_id not in {None,owner}
                    or not re.fullmatch(re.escape(stem)+r'[-/][A-Za-z0-9-]+',second.raw_text)):
                continue
            b=second.bbox
            if (body.x1<=b.x1<b.x2<=body.x2 and body.y1<=b.y1<b.y2<=body.y2
                    and 0<=b.y1-a.y2<=.5*max(a.height,b.height)
                    and .7<=a.height/max(1.,b.height)<=1.4
                    and abs(a.center.x-b.center.x)<=.5*min(a.height,b.height)
                    and abs(a.center.x-body.center.x)<=.15*body.width
                    and abs(b.center.x-body.center.x)<=.15*body.width):
                candidates.append(second)
        if len(candidates)!=1:
            continue
        second=candidates[0];b=second.bbox
        identity='visible-title-continuation:'+first.internal_id+':'+second.internal_id
        record={'component_id':owner,'source_text_ids':[first.internal_id,second.internal_id],
                'raw_lines':[first.raw_text,second.raw_text],'joined_text':first.raw_text+second.raw_text,
                'text_id':identity,'gt_access':False}
        combined=TextToken(internal_id=identity,raw_text=record['joined_text'],
            category='searchname',details='searchname',component_id=owner,
            bbox=BBox(x1=min(a.x1,b.x1),y1=a.y1,x2=max(a.x2,b.x2),y2=b.y2),
            score=min(first.score,second.score),evidence=[*first.evidence,*second.evidence,
                Evidence(evidence_id='association:'+identity,source='rule',
                         score=min(first.score,second.score),metadata={'kind':'visible-title-continuation',**record})])
        ir.texts=[t for t in ir.texts if t.internal_id not in {first.internal_id,second.internal_id}]
        ir.texts.append(combined)
        assigned.pop(first.internal_id);assigned.pop(second.internal_id,None);assigned[identity]=owner
        records.append(record)
    return records
