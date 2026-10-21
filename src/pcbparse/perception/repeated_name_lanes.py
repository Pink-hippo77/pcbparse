"""Recover repeated observed pin-name forms mistaken for body references."""
from collections import defaultdict
import math
import re
from pcbparse.association.reference_prefixes import BOX_REFERENCE_PATTERN
from pcbparse.ir.models import Evidence

REFERENCE=re.compile(BOX_REFERENCE_PATTERN)
PARTS=re.compile(r'([A-Z]{1,8})([0-9]{1,4})([A-Z]?)')


def refine_repeated_pin_name_lanes(ir,ctx=None):
    report={'updates':[],'gt_access':False,'numbers_changed':False,'coordinates_changed':False,'new_pins':0}
    if ctx is not None and ctx.time_budget.remaining_seconds<25:
        report['skipped']='time_reserve'
        ir.metadata['repeated_pin_name_lanes']=report
        return report
    texts={t.internal_id:t for t in ir.texts}
    for c in ir.components:
        if ctx:ctx.time_budget.check(stage='repeated_pin_name_lanes')
        if c.category.upper() not in {'BOX','IC','BOXIC','BOX_IC'} or c.body_bbox is None:
            continue
        body=c.body_bbox
        rows={}
        for p in ir.pins:
            if p.component_id!=c.internal_id or p.orientation not in {'L','R','T','B'} or not p.number:
                continue
            observations=[e.metadata for e in p.evidence if e.source=='geometry'
                and e.metadata.get('kind')=='visible-stroke-and-text-lanes' and e.metadata.get('gt_access') is False]
            if not observations:continue
            row=observations[-1]
            if row.get('number_score',0)<.95:continue
            position=row.get('position')
            if type(position) not in {float,int} or not math.isfinite(position):continue
            rows[p.internal_id]=(p,row,position)
        proposals=[]
        for pid,(p,row,position) in rows.items():
            horizontal=p.orientation in {'L','R'}
            peers=[(q.internal_id,qpos)for q,qrow,qpos in rows.values()if q.orientation==p.orientation]
            pitch=min([abs(qpos-position)for qid,qpos in peers if qid!=pid]+[40.])
            choices=[]
            for t in ir.texts:
                value=t.raw_text.strip();b=t.bbox
                parts=PARTS.fullmatch(value)
                if (t.score<.95 or not parts or not REFERENCE.fullmatch(value)
                    or value in {c.key,c.name} or t.component_id not in {None,c.internal_id}
                    or t.pin_id not in {None,pid}
                    or not (body.x1<=b.x1<b.x2<=body.x2 and body.y1<=b.y1<b.y2<=body.y2)):
                    continue
                axis=b.center.y if horizontal else b.center.x
                gap={'L':b.x1-body.x1,'R':body.x2-b.x2,'T':b.y1-body.y1,'B':body.y2-b.y2}[p.orientation]
                width=body.width if horizontal else body.height
                if gap>max(12.,.18*width) or abs(axis-position)>min(.48*pitch+1.,.65*b.height+1.):
                    continue
                distances=sorted((abs(axis-qpos),qid)for qid,qpos in peers)
                if distances[0][1]!=pid or len(distances)>1 and distances[1][0]-distances[0][0]<min(2.,.2*pitch):
                    continue
                choices.append((t.score,-gap,-abs(axis-position),t.internal_id,parts.group(1),t))
            choices.sort(key=lambda z:z[:4],reverse=True)
            if not choices:continue
            best=choices[0];other=next((z for z in choices if z[5].raw_text.strip()!=best[5].raw_text.strip()),None)
            if other is not None and best[0]-other[0]<=.03:continue
            proposals.append((p,row,best[4],best[5]))
        groups=defaultdict(list)
        for p,row,prefix,t in proposals:groups[(p.orientation,prefix)].append((p,row,t))
        for (side,prefix),members in groups.items():
            if len(members)<3 or len({t.raw_text.strip()for _,_,t in members})<3:continue
            support=[t.internal_id for _,_,t in members]
            for p,row,t in members:
                value=t.raw_text.strip()
                if p.name==value:continue
                oldscore=row.get('name_score',0.)
                if p.name and row.get('name_token') is not None:
                    old=texts.get(row['name_token'])
                    if old is None or oldscore>=.98 or t.score<max(.99,oldscore+.02):continue
                    oldgap={'L':old.bbox.x1-body.x1,'R':body.x2-old.bbox.x2,
                            'T':old.bbox.y1-body.y1,'B':body.y2-old.bbox.y2}[side]
                    gap={'L':t.bbox.x1-body.x1,'R':body.x2-t.bbox.x2,
                         'T':t.bbox.y1-body.y1,'B':body.y2-t.bbox.y2}[side]
                    if oldgap-gap<max(4.,.75*t.bbox.height):continue
                elif p.name and oldscore>=.98:
                    continue
                update={'pin_id':p.internal_id,'previous_name':p.name,'name':value,'token_id':t.internal_id,
                        'side':side,'observed_prefix':prefix,'support_tokens':support,
                        'method':'repeated-observed-reference-shaped-pin-names','gt_access':False}
                p.name=value
                t.component_id,t.pin_id=c.internal_id,p.internal_id
                t.details=t.category='pinname'
                p.evidence.append(Evidence(evidence_id='repeated-pin-name:'+p.internal_id,source='geometry',metadata=update))
                report['updates'].append(update)
    ir.metadata['repeated_pin_name_lanes']=report
    return report
