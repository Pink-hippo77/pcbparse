"""Fuse model tips with a body supported by interior pin numbers.

The midpoint is an estimated lead endpoint, not an observed color boundary.
This drawing-style prior is optional and preserves terminal identities.
"""
import math
import re
import numpy as np
from pcbparse.ir.models import BBox, Evidence, Point
from pcbparse.perception.ic_visual_readback import closed_rectangles
from pcbparse.perception.passive_visual_readback import _contains

NORMALS={'L':(-1,0),'R':(1,0),'T':(0,-1),'B':(0,1)}


def refine_interior_number_tips(ir,rgb,ctx=None,*,rectangles=None):
    report={'updates':[],'gt_access':False,'new_pins':0,'identities_assigned':False}
    if ctx is not None and ctx.time_budget.remaining_seconds<30:
        report['skipped']='time_reserve'
        ir.metadata['interior_number_tip_refinement']=report
        return report
    check=(lambda:ctx.time_budget.check(stage='interior_number_tips')) if ctx is not None else None
    if rectangles is None:
        rectangles=closed_rectangles(rgb,check=check)
    gray=(rgb.min(2)<180)&((rgb.max(2).astype(np.int16)-rgb.min(2).astype(np.int16))<30)
    texts=[t for t in ir.texts if t.score>=.95 and re.fullmatch(r'[1-9]\d{0,3}',t.raw_text.strip())]
    for c in ir.components:
        if check:check()
        if c.category.upper() not in {'BOX','IC','BOXIC','BOX_IC'} or c.score<.5:
            continue
        choices=[]
        for x1,y1,x2,y2,hue in rectangles:
            b=BBox(x1=x1,y1=y1,x2=x2,y2=y2)
            if (hue<0 and min(b.width,b.height)>=38 and _contains(c.bbox,b)>.8
                    and max(abs(getattr(c.bbox,k)-getattr(b,k)) for k in ['x1','y1','x2','y2'])<=20
                    and b.width*b.height>=.4*c.bbox.width*c.bbox.height):
                choices.append(b)
        if len(choices)!=1:
            continue
        body=choices[0]
        proposals=[]
        for p in ir.pins:
            if p.component_id!=c.internal_id or p.orientation not in NORMALS or p.internal_id.startswith('visual-'):
                continue
            nx,ny=NORMALS[p.orientation]
            edge={'L':body.x1,'R':body.x2,'T':body.y1,'B':body.y2}[p.orientation]
            outward=(p.point.x-edge)*nx if nx else (p.point.y-edge)*ny
            if not 2<=outward<=18:
                continue
            axis=p.point.y if nx else p.point.x
            inside=[];outside=[]
            for t in texts:
                b=t.bbox;font=max(6.,min(30.,b.height))
                lateral=abs((b.center.y if nx else b.center.x)-axis)
                if lateral>max(3.,.3*font):
                    continue
                if _contains(body,b)>.95:
                    gap={'L':b.x1-body.x1,'R':body.x2-b.x2,'T':b.y1-body.y1,'B':body.y2-b.y2}[p.orientation]
                    if 0<=gap<=max(5.,.4*font):
                        inside.append((lateral,t.internal_id,t))
                else:
                    center_out=(b.center.x-edge)*nx if nx else (b.center.y-edge)*ny
                    if 0<center_out<=max(30.,font*2):outside.append(t)
            inside.sort(key=lambda r:r[:2])
            if outside or not inside or len(inside)>1 and inside[1][0]-inside[0][0]<3:
                continue
            target=Point(x=(edge+p.point.x)/2 if nx else p.point.x,
                         y=(edge+p.point.y)/2 if ny else p.point.y)
            supported=True
            for fraction in np.linspace(0.,1.,max(3,math.ceil(outward*2))):
                x=round(target.x+(p.point.x-target.x)*fraction);y=round(target.y+(p.point.y-target.y)*fraction)
                if not (1<=x<gray.shape[1]-1 and 1<=y<gray.shape[0]-1 and gray[y-1:y+2,x-1:x+2].any()):
                    supported=False;break
            if supported:proposals.append((p,target,inside[0][2]))
        # Multiple independently printed numbers establish this drawing style.
        if len({t.internal_id for _,_,t in proposals})<3:
            continue
        for p,target,t in proposals:
            row={'pin_id':p.internal_id,'previous':p.point.model_dump(),'point':target.model_dump(),
                 'number_token':t.internal_id,'printed_number':t.raw_text,'body':body.model_dump(),
                 'gt_access':False,'identities_assigned':False,
                 'method':'model-and-interior-number-body-midpoint','estimated_coordinate':True}
            p.point=target
            p.evidence.append(Evidence(evidence_id='interior-number-tip:'+p.internal_id,source='geometry',metadata=row))
            report['updates'].append(row)
    ir.metadata['interior_number_tip_refinement']=report
    return report
