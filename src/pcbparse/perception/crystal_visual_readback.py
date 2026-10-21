"""Recover crystal ownership only from a printed reference, plates and observed leads."""
from __future__ import annotations
import math
import re
import numpy as np
from pcbparse.graph.palette_wire import hue_map
from pcbparse.ir.models import BBox, Evidence, Point
from pcbparse.perception.ic_visual_readback import _runs
from pcbparse.perception.symbol_body import groups

_REFERENCE=re.compile(r'[XY]\d+[A-Z]?$')
_CANDIDATES={'BOX','IC','BOXIC','BOX_IC','CRYSTAL','XTAL','OTHER','C','CAPACITOR'}

def _shapes(mask,ink=None):
    found=[]
    for transposed in (False,True):
        m=mask.T if transposed else mask
        all_ink=(ink.T if transposed else ink) if ink is not None else m
        edges=_runs(m.T,10)
        for i,(x1,a,b) in enumerate(edges):
            for x2,c,d in edges[i+1:]:
                w=x2-x1
                if not 4<=w<=45 or max(abs(a-c),abs(b-d))>2:continue
                y1,y2=round((a+c)/2),round((b+d)/2)-1
                h=y2-y1
                if not 1.2*w<=h<=6*w:continue
                if min(m[max(0,y-1):y+2,x1:x2+1].any(0).mean() for y in (y1,y2))<.9:continue
                interior=m[y1+3:y2-2,x1+3:x2-2]
                if not interior.size or interior.mean()>.12:continue
                axis=round((y1+y2)/2)
                sides=[]
                for sign in (-1,1):
                    candidates=[]
                    columns={}
                    for xx in range(max(0,x1-round(.55*h)-5),min(m.shape[1],x2+round(.55*h)+6)):
                        gap=x1-xx if sign<0 else xx-x2
                        if not 2<=gap<=max(5,.55*h):continue
                        lo=max(0,y1-round(.15*h));hi=min(m.shape[0],y2+round(.15*h)+1)
                        ys=np.flatnonzero(m[lo:hi,xx])+lo
                        splits=np.split(ys,np.flatnonzero(np.diff(ys)>max(3,.2*h))+1)
                        splits=[g for g in splits if len(g) and abs((g[0]+g[-1])/2-axis)<=max(2,.12*h)]
                        if len(splits)!=1:continue
                        ys=splits[0]
                        if len(ys)<.4*h:continue
                        if not .5*h<=ys[-1]-ys[0]<=1.3*h or np.diff(ys).max(initial=0)>max(3,.2*h):continue
                        if abs((ys[0]+ys[-1])/2-axis)>max(2,.12*h):continue
                        columns[xx]=(int(ys[0]),int(ys[-1]))
                    for g in groups(np.asarray(sorted(columns),dtype=int)):
                        if len(g)>4:continue
                        p=[int(g[0]),int(g[-1]),min(columns[x][0] for x in g),max(columns[x][1] for x in g)]
                        gap=x1-p[1] if sign<0 else p[0]-x2
                        gapmask=m[p[2]+2:p[3]-1,p[1]+1:x1] if sign<0 else m[p[2]+2:p[3]-1,x2+1:p[0]]
                        if not gapmask.size or gapmask.mean()>.15:continue
                        edge=p[0] if sign<0 else p[1]
                        steps=0
                        for step in range(1,5):
                            xx=edge+sign*step
                            if not 0<=xx<m.shape[1] or not all_ink[max(0,axis-1):axis+2,xx].any():break
                            steps=step
                        if steps>=4:candidates.append((gap,p))
                    if len(candidates)!=1:break
                    sides.append(candidates[0])
                if len(sides)!=2:continue
                body=(sides[0][1][0],min(y1,sides[0][1][2],sides[1][1][2]),
                      sides[1][1][1],max(y2,sides[0][1][3],sides[1][1][3]))
                if transposed:
                    body=(body[1],body[0],body[3],body[2])
                if not any(max(abs(a-b) for a,b in zip(body,r[0]))<4 for r in found):
                    found.append((body,'TB' if transposed else 'LR'))
    return found


def crystal_shapes(rgb,region):
    x,y=max(0,math.floor(region.x1)),max(0,math.floor(region.y1))
    xx,yy=min(rgb.shape[1],math.ceil(region.x2)),min(rgb.shape[0],math.ceil(region.y2))
    bins,ink=hue_map(rgb[y:yy,x:xx])
    out=[]
    for hue in range(-1,12):
        mask=ink&(bins==hue)
        if mask.sum()<25:continue
        for b,axis in _shapes(mask,ink):
            out.append({'body':[b[0]+x,b[1]+y,b[2]+x,b[3]+y],
                        'axis':axis,'hue':hue})
    return out



def _supported_terminals(rgb, body, axis, pins):
    """Require exactly one observed exterior tip on each measured lead."""
    ink=rgb.min(2)<180
    horizontal=axis=='LR'
    fixed=(body.x1,body.x2) if horizontal else (body.y1,body.y2)
    center=body.center.y if horizontal else body.center.x
    groups=[[],[]]
    interior=[]
    for pin in pins:
        p=pin.point
        if body.x1<=p.x<=body.x2 and body.y1<=p.y<=body.y2:
            interior.append(pin)
            continue
        along,lateral=(p.x,p.y) if horizontal else (p.y,p.x)
        side=0 if along<fixed[0] else 1 if along>fixed[1] else None
        if side is None or pin.score<.3 or abs(lateral-center)>3:
            return None
        distance=abs(along-fixed[side])
        if not 2<=distance<=40:return None
        support=[]
        for position in np.linspace(fixed[side],along,max(3,math.ceil(distance)+1)):
            x,y=(round(position),round(lateral)) if horizontal else (round(lateral),round(position))
            if not 1<=x<ink.shape[1]-1 or not 1<=y<ink.shape[0]-1:return None
            support.append(ink[y-1:y+2,x-1:x+2].any())
        if np.mean(support)<.85:return None
        groups[side].append(pin)
    if [len(g) for g in groups]!=[1,1]:return None
    return [g[0] for g in groups],interior


def _overlap(a,b):
    area=max(0,min(a.x2,b.x2)-max(a.x1,b.x1))*max(0,min(a.y2,b.y2)-max(a.y1,b.y1))
    return area/max(1,b.width*b.height)


def recover_visual_crystals(ir,rgb,ctx=None):
    """Resolve duplicated/misclassified existing owners without inventing tips.

    References alone never determine type. Two parallel plates, an empty
    rectangle, continuous exterior leads and two unique model tips must agree.
    A third exterior terminal or ambiguous model tips leave the owner intact.
    """
    if 'visual_crystal_readback' in ir.metadata:
        return ir.metadata['visual_crystal_readback']
    references=[t for t in ir.texts if t.score>=.95 and _REFERENCE.fullmatch(t.raw_text)]
    proposals=[]
    for token in references:
        if ctx:ctx.time_budget.check(stage='crystal_readback')
        b=token.bbox;font=max(10,min(30,b.height))
        region=BBox(x1=max(0,b.x1-5*font),y1=max(0,b.y1-2*font),
                    x2=min(ir.image.width,b.x2+5*font),y2=min(ir.image.height,b.y2+6*font))
        choices=[]
        for shape in crystal_shapes(rgb,region):
            body=BBox(x1=shape['body'][0],y1=shape['body'][1],x2=shape['body'][2],y2=shape['body'][3])
            gap=math.hypot(max(body.x1-b.x2,0,b.x1-body.x2),max(body.y1-b.y2,0,b.y1-body.y2))
            if gap>3*font:continue
            rank=gap+.15*math.hypot(body.center.x-b.center.x,body.center.y-b.center.y)
            choices.append((rank,body,shape['axis']))
        choices.sort(key=lambda row:row[0])
        if not choices or len(choices)>1 and choices[1][0]-choices[0][0]<5:continue
        rank,body,axis=choices[0]
        proposals.append((rank,token,body,axis))
    removed=set();removed_pins=set();claimed=[];updates=[]
    for rank,token,body,axis in sorted(proposals,key=lambda row:(row[0],row[1].internal_id)):
        if any(other.raw_text!=token.raw_text and other_rank<=rank+4
               and math.dist((body.center.x,body.center.y),(other_body.center.x,other_body.center.y))<6
               for other_rank,other,other_body,_ in proposals):continue
        if any(math.dist((body.center.x,body.center.y),(b.center.x,b.center.y))<6 for b in claimed):continue
        candidates=[]
        for c in ir.components:
            if c.internal_id in removed or c.category.upper() not in _CANDIDATES:continue
            if c.key and c.key!=token.raw_text:continue
            if any(e.metadata.get('method')=='visible-ic-outline-and-terminal-labels' for e in c.evidence):continue
            box=c.body_bbox or c.bbox
            if (_overlap(box,body)<.65 or max(box.width,box.height)>2.5*max(body.width,body.height)
                    or math.dist((box.center.x,box.center.y),(body.center.x,body.center.y))>max(5,.3*max(body.width,body.height))):continue
            candidates.append(c)
        if not candidates:continue
        # Every duplicate must describe this same small body. A large IC or
        # separately keyed component cannot be consumed by a symbol repair.
        if any(other.key==token.raw_text and other not in candidates for other in ir.components):continue
        if any(_overlap(a.bbox,b.bbox)<.65 or _overlap(b.bbox,a.bbox)<.65
               for a in candidates for b in candidates):continue
        observations=[]
        for c in candidates:
            pins=[p for p in ir.pins if p.component_id==c.internal_id]
            supported=_supported_terminals(rgb,body,axis,pins)
            if supported:observations.append((c,supported))
        if not observations:continue
        c,(pins,interior)=min(observations,key=lambda row:(row[0].category.upper() not in {'CRYSTAL','XTAL'},-row[0].score,row[0].internal_id))
        duplicates=[other for other in candidates if other is not c]
        # Preserve independently populated fields rather than discarding a
        # conflicting duplicate before association has resolved it.
        if any((other.name and other.name!=c.name) or (other.value and other.value!=c.value) for other in duplicates):continue
        if token.component_id and token.component_id not in {v.internal_id for v in candidates}:continue
        previous=c.category
        c.category='Crystal';c.key=token.raw_text;c.key_score=token.score;c.body_bbox=body
        token.component_id=c.internal_id;token.pin_id=None;token.details=token.category='designator'
        removed.update(other.internal_id for other in duplicates)
        removed_pins.update(p.internal_id for p in interior)
        for pin,side in zip(pins,axis):
            pin.orientation=side
            boundary={'L':body.x1,'R':body.x2,'T':body.y1,'B':body.y2}[side]
            pin.base=Point(x=boundary,y=pin.point.y) if axis=='LR' else Point(x=pin.point.x,y=boundary)
        row={'method':'visible-crystal-plates-reference-and-model-leads','component_id':c.internal_id,
             'previous_category':previous,'reference':token.raw_text,'reference_token':token.internal_id,
             'body_bbox':body.model_dump(),'pin_ids':[p.internal_id for p in pins],
             'removed_duplicate_components':[v.internal_id for v in duplicates],
             'removed_interior_pins':[p.internal_id for p in interior],
             'pin_coordinates_changed':False,'new_pins':0,'gt_access':False}
        c.evidence.append(Evidence(evidence_id='visual-crystal:'+c.internal_id,source='geometry',metadata=row))
        updates.append(row);claimed.append(body)
    ir.components=[c for c in ir.components if c.internal_id not in removed]
    ir.pins=[p for p in ir.pins if p.component_id not in removed and p.internal_id not in removed_pins]
    valid={p.internal_id for p in ir.pins}
    for token in ir.texts:
        if token.component_id in removed:token.component_id=None
        if token.pin_id and token.pin_id not in valid:token.pin_id=None
    report={'updates':updates,'removed_duplicates':sorted(removed),'new_pins':0,
            'coordinates_changed':False,'gt_access':False}
    ir.metadata['visual_crystal_readback']=report
    return report
