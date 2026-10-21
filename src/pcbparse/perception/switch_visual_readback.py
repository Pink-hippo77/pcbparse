"""Recover printed switches from measured contacts, leads and actuator shapes.

Each contact needs a hollow pad, an outward stem, and a measured transition
to a different common wire color, or a measured open contact and actuator.
No contacts or numbers are filled by order.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from itertools import combinations, permutations

import numpy as np

from pcbparse.graph.palette_wire import hue_map
from pcbparse.ir.models import BBox, Component, Evidence, Pin, Point
from pcbparse.perception.ic_visual_readback import _runs
from pcbparse.perception.passive_visual_readback import _contains
from pcbparse.perception.rail_visual_readback import _bars, _stem

_REFERENCE = re.compile(r'(?:SW|S)\d+[A-Z]?$')


def _pads(mask, font):
    bars = _bars(mask, maximum_thickness=max(3, round(font*.2)))
    if len(bars) > 160:
        return []
    pads = []
    tolerance = max(1, round(font*.08))
    for i, a in enumerate(bars):
        for b in bars[i+1:]:
            width, height = a[3]-a[2], b[1]-a[0]
            if not (5 <= width <= 2.5*font and 3 <= height <= 1.2*font and 1.3 <= width/max(1,height) <= 5):
                continue
            if max(abs(a[k]-b[k]) for k in (2,3)) > tolerance:
                continue
            left,right,top,bottom = min(a[2],b[2]),max(a[3],b[3]),a[0],b[1]
            if min(mask[top:bottom+1,max(0,x-1):x+2].any(1).mean() for x in (left,right)) < .9:
                continue
            interior = mask[a[1]+1:b[0],left+2:right-1]
            if not interior.size or interior.mean() > .2:
                continue
            bounds = (left,top,right,bottom)
            if any(max(abs(v-z) for v,z in zip(bounds,old['bounds'])) <= tolerance for old in pads):
                continue
            pads.append({'bounds':bounds, 'bars':(a,b), 'x':(left+right)/2, 'y':(top+bottom)/2})
    return pads if len(pads) <= 64 else []


def _rows(pads, tolerance):
    rows = []
    for pad in sorted(pads,key=lambda p:(p['y'],p['x'])):
        if rows and abs(pad['y']-np.mean([p['y'] for p in rows[-1]])) <= tolerance:
            rows[-1].append(pad)
        else:
            rows.append([pad])
    return [sorted(row,key=lambda p:p['x']) for row in rows]


def _wire_after(bins, ink, point, normal, hue):
    votes = []
    for step in range(1,7):
        x,y = round(point[0]+normal[0]*step),round(point[1]+normal[1]*step)
        if not (1 <= x < bins.shape[1]-1 and 1 <= y < bins.shape[0]-1):
            return None
        xs = x+np.arange(-1,2)*normal[1]
        ys = y+np.arange(-1,2)*normal[0]
        colors = bins[ys,xs][ink[ys,xs]]
        foreign = [int(v) for v in colors if v >= 0 and min((int(v)-hue)%12,(hue-int(v))%12) >= 2]
        if foreign:
            votes.append(Counter(foreign).most_common(1)[0][0])
    if len(votes) < 4:
        return None
    wire,count = Counter(votes).most_common(1)[0]
    if count < 4:
        return None
    return wire


def _outward_contact(mask, bins, ink, pad, direction, hue, font):
    maximum_length = math.ceil(max(20,3*font))
    length,end,axis = _stem(mask,pad['bars'][0 if direction < 0 else 1],direction,
                            maximum_length=maximum_length+1)
    if not 4 <= length <= maximum_length:
        return None
    wire = _wire_after(bins,ink,(axis,end),(0,direction),hue)
    if wire is None:
        return None
    base = pad['bounds'][1 if direction < 0 else 3]
    return {'point':(axis,end), 'base':(axis,base), 'wire_hue':wire}


def grid_switch_candidates(rgb, reference, ctx=None):
    """Return only fully measured grids surrounding an observed reference."""
    font = max(8.,min(32.,reference.height))
    x0,y0 = max(0,math.floor(reference.x1-6*font)),max(0,math.floor(reference.y1-5*font))
    x2,y2 = min(rgb.shape[1],math.ceil(reference.x2+6*font)),min(rgb.shape[0],math.ceil(reference.y2+5*font))
    bins,ink = hue_map(rgb[y0:y2,x0:x2])
    found = []
    for hue in range(12):
        mask = (bins==hue)&ink
        if mask.sum() < 60:
            continue
        for transpose in (False,True):
            if ctx:
                ctx.time_budget.check(stage='switch_pad_readback')
            m,colors,occupied = (mask.T,bins.T,ink.T) if transpose else (mask,bins,ink)
            rx,ry = (reference.center.y-y0,reference.center.x-x0) if transpose else (reference.center.x-x0,reference.center.y-y0)
            rows = _rows(_pads(m,font),max(2.,font*.16))
            for upper,lower in combinations(rows,2):
                if len(upper) != len(lower) or not 2 <= len(upper) <= 6:
                    continue
                ux,lx = np.array([p['x'] for p in upper]),np.array([p['x'] for p in lower])
                if max(abs(ux-lx)) > max(2.,font*.15):
                    continue
                widths = np.array([p['bounds'][2]-p['bounds'][0] for p in upper+lower])
                heights = np.array([p['bounds'][3]-p['bounds'][1] for p in upper+lower])
                spacing = np.diff((ux+lx)/2)
                if (widths.max() > 1.3*widths.min() or heights.max() > 1.4*heights.min()
                        or spacing.min() < 1.3*widths.max() or spacing.max() > 1.3*spacing.min()):
                    continue
                top,bottom = max(p['bounds'][3] for p in upper),min(p['bounds'][1] for p in lower)
                if not (top < ry < bottom and widths.mean() <= bottom-top+heights.mean() <= 5*font
                        and ux[0] <= rx <= ux[-1]):
                    continue
                contacts = [_outward_contact(m,colors,occupied,pad,direction,hue,font)
                            for row,direction in [(upper,-1),(lower,1)] for pad in row]
                if any(p is None for p in contacts) or len({p['wire_hue'] for p in contacts}) != 1:
                    continue
                pads = upper+lower
                restored = []
                boxes = []
                for i,(pad,contact) in enumerate(zip(pads,contacts)):
                    a,b,c,d = pad['bounds']
                    point,base = contact['point'],contact['base']
                    if transpose:
                        a,b,c,d = b,a,d,c
                        point,base = point[::-1],base[::-1]
                    boxes.append((a+x0,b+y0,c+x0,d+y0))
                    side = ('L' if i<len(upper) else 'R') if transpose else ('T' if i<len(upper) else 'B')
                    restored.append({'point':(point[0]+x0,point[1]+y0), 'base':(base[0]+x0,base[1]+y0), 'side':side})
                body = BBox(x1=min(b[0] for b in boxes),y1=min(b[1] for b in boxes),
                            x2=max(b[2] for b in boxes),y2=max(b[3] for b in boxes))
                if not any(max(abs(a-b) for a,b in zip(body.model_dump().values(),old['body'].model_dump().values())) < 3 for old in found):
                    found.append({'body':body, 'pads':boxes, 'contacts':restored,
                                  'symbol_hue':hue,'wire_hue':contacts[0]['wire_hue']})
    return found


def open_switch_candidates(rgb, reference, ctx=None):
    """Measure two open collinear contacts and a separate raised actuator."""
    font = max(8.,min(32.,reference.height))
    x0,y0 = max(0,math.floor(reference.x1-4*font)),max(0,math.floor(reference.y1-2*font))
    x2,y2 = min(rgb.shape[1],math.ceil(reference.x2+4*font)),min(rgb.shape[0],math.ceil(reference.y2+5*font))
    bins,ink = hue_map(rgb[y0:y2,x0:x2]);found=[]
    for hue in range(12):
        mask=(bins==hue)&ink
        if mask.sum()<20:
            continue
        for transpose in (False,True):
            if ctx:
                ctx.time_budget.check(stage='switch_contact_readback')
            m,colors,occupied=(mask.T,bins.T,ink.T) if transpose else (mask,bins,ink)
            bars=_bars(m,maximum_thickness=max(3,round(font*.2)))
            if len(bars)>160:
                continue
            for a,b in combinations(bars,2):
                a,b=sorted((a,b),key=lambda v:v[2])
                axis=(a[0]+a[1]+b[0]+b[1])/4
                gap=b[2]-a[3]
                if (abs(a[0]-b[0])>1 or abs(a[1]-b[1])>1 or not 4<=gap<=2.5*font
                        or min(a[3]-a[2],b[3]-b[2])<4 or max(a[3]-a[2],b[3]-b[2])>4*font):
                    continue
                if m[max(0,round(axis)-1):round(axis)+2,a[3]+2:b[2]-1].any(0).mean()>.2:
                    continue
                actuators=[bar for bar in bars if 2<=abs((bar[0]+bar[1])/2-axis)<=1.5*font
                           and abs(bar[2]-a[3])<=font and abs(bar[3]-b[2])<=font
                           and bar[2]<=a[3]+.25*gap and bar[3]>=b[2]-.25*gap]
                if not actuators:
                    continue
                left,right=(a[2],axis),(b[3],axis)
                wire=_wire_after(colors,occupied,left,(-1,0),hue)
                if wire is None or _wire_after(colors,occupied,right,(1,0),hue)!=wire:
                    continue
                bar=min(actuators,key=lambda v:abs((v[0]+v[1])/2-axis))
                bx1,by1,bx2,by2=min(a[3],bar[2]),min(a[0],bar[0]),max(b[2],bar[3]),max(a[1],bar[1])
                points,bases=[left,right],[(a[3],axis),(b[2],axis)]
                if transpose:
                    bx1,by1,bx2,by2=by1,bx1,by2,bx2
                    points,bases=[p[::-1] for p in points],[p[::-1] for p in bases]
                body=BBox(x1=bx1+x0,y1=by1+y0,x2=bx2+x0,y2=by2+y0)
                t=reference
                distance=math.hypot(max(body.x1-t.x2,0,t.x1-body.x2),max(body.y1-t.y2,0,t.y1-body.y2))
                if distance>1.3*font or math.hypot(body.center.x-t.center.x,body.center.y-t.center.y)>3*font:
                    continue
                contacts=[{'point':(p[0]+x0,p[1]+y0),'base':(b[0]+x0,b[1]+y0),'side':side}
                          for p,b,side in zip(points,bases,'TB' if transpose else 'LR')]
                if not any(max(abs(a-b) for a,b in zip(body.model_dump().values(),old['body'].model_dump().values()))<3 for old in found):
                    found.append({'body':body,'pads':[],'contacts':contacts,'symbol_hue':hue,'wire_hue':wire})
    return found


def _rings(mask, font):
    rows = {}
    for y,a,b in _runs(mask,1):
        rows.setdefault(y,[]).append((a,b))
    rings = []
    for y,runs in rows.items():
        for (_,b),(c,_) in zip(runs,runs[1:]):
            left,right=b-1,c;width=right-left;axis=round((left+right)/2)
            if not 3 <= width <= 1.4*font:
                continue
            def closes(yy):
                return mask[yy,left:right+1].mean() >= .5 and mask[yy,max(left,axis-1):min(right+1,axis+2)].any()
            tops=[yy for yy in range(max(0,y-int(1.5*width)),y) if closes(yy)]
            bottoms=[yy for yy in range(y+1,min(mask.shape[0],y+int(1.5*width)+1)) if closes(yy)]
            if not tops or not bottoms:
                continue
            top,bottom=max(tops),min(bottoms)
            if not .6*width <= bottom-top <= 1.5*width:
                continue
            if min(mask[top:bottom+1,max(0,x-1):x+2].any(1).mean() for x in (left,right)) < .75:
                continue
            interior=mask[top+1:bottom,left+1:right]
            bounds=(left,top,right,bottom)
            if not interior.size or interior.mean()>.25:
                continue
            if not any(max(abs(a-b) for a,b in zip(bounds,old))<=2 for old in rings):
                rings.append(bounds)
    return rings if len(rings)<=64 else []


def lever_switch_candidates(rgb, reference, ctx=None):
    """Require two hollow round contacts and a measured inclined open arm."""
    font=max(8.,min(32.,reference.height))
    x0,y0=max(0,math.floor(reference.x1-4*font)),max(0,math.floor(reference.y1-2*font))
    x2,y2=min(rgb.shape[1],math.ceil(reference.x2+4*font)),min(rgb.shape[0],math.ceil(reference.y2+5*font))
    bins,ink=hue_map(rgb[y0:y2,x0:x2]);found=[]
    for hue in range(12):
        mask=(bins==hue)&ink
        if mask.sum()<20:
            continue
        for transpose in (False,True):
            if ctx:
                ctx.time_budget.check(stage='switch_contact_readback')
            m,colors,occupied=(mask.T,bins.T,ink.T) if transpose else (mask,bins,ink)
            for a,b in combinations(_rings(m,font),2):
                a,b=sorted((a,b),key=lambda p:p[0])
                ax,ay=(a[0]+a[2])/2,(a[1]+a[3])/2
                bx,by=(b[0]+b[2])/2,(b[1]+b[3])/2
                if (abs(ay-by)>max(1.,.1*font) or not 2*max(a[2]-a[0],b[2]-b[0])<=bx-ax<=4*font
                        or max(a[2]-a[0],b[2]-b[0])>1.4*min(a[2]-a[0],b[2]-b[0])):
                    continue
                baseline=(ay+by)/2
                gap=m[round(baseline),a[2]+1:b[0]]
                if not gap.size or gap.mean()>.2:
                    continue
                contacts=[]
                for ring,direction,edge in [(a,-1,a[0]),(b,1,b[2])]:
                    maximum_length=math.ceil(4*font)
                    length,end,axis=_stem(m.T,[edge,edge,ring[1],ring[3]],direction,
                                          maximum_length=maximum_length+1)
                    wire=_wire_after(colors,occupied,(end,axis),(direction,0),hue)
                    if not 4<=length<=maximum_length or wire is None:
                        break
                    contacts.append({'point':(end,axis),'base':(edge,axis),'wire_hue':wire})
                if len(contacts)!=2 or contacts[0]['wire_hue']!=contacts[1]['wire_hue']:
                    continue
                arm=None
                for direction in (-1,1):
                    low,high=(max(0,math.floor(baseline-1.5*font)),math.floor(baseline-1)) if direction<0 else (math.ceil(baseline+2),min(m.shape[0],math.ceil(baseline+1.5*font)))
                    xs=[];ys=[]
                    for x in range(round(ax),round(bx)+1):
                        hits=np.flatnonzero(m[low:high,x])
                        if len(hits):
                            xs.append(x);ys.append(low+int(hits[0] if direction<0 else hits[-1]))
                    if len(xs)<.9*(round(bx)-round(ax)+1):
                        continue
                    slope,intercept=np.polyfit(xs,ys,1)
                    if not .1<=abs(slope)<=1. or np.max(np.abs(np.array(ys)-(np.array(xs)*slope+intercept)))>1.2:
                        continue
                    first,last=abs(baseline-ys[0]),abs(baseline-ys[-1]);radius=max(a[3]-a[1],b[3]-b[1])/2
                    if min(first,last)>radius+2 or max(first,last)<radius+3:
                        continue
                    arm=(min(ys),max(ys));break
                if arm is None:
                    continue
                bx1,by1,bx2,by2=a[0],min(a[1],b[1],arm[0]),b[2],max(a[3],b[3],arm[1])
                if transpose:
                    bx1,by1,bx2,by2=by1,bx1,by2,bx2
                    for contact in contacts:
                        contact['point']=contact['point'][::-1];contact['base']=contact['base'][::-1]
                body=BBox(x1=bx1+x0,y1=by1+y0,x2=bx2+x0,y2=by2+y0)
                distance=math.hypot(max(body.x1-reference.x2,0,reference.x1-body.x2),max(body.y1-reference.y2,0,reference.y1-body.y2))
                if distance>1.3*font or math.hypot(body.center.x-reference.center.x,body.center.y-reference.center.y)>3*font:
                    continue
                restored=[{'point':(p['point'][0]+x0,p['point'][1]+y0),'base':(p['base'][0]+x0,p['base'][1]+y0),'side':side}
                          for p,side in zip(contacts,'TB' if transpose else 'LR')]
                if not any(max(abs(a-b) for a,b in zip(body.model_dump().values(),old['body'].model_dump().values()))<3 for old in found):
                    found.append({'body':body,'pads':[],'contacts':restored,'symbol_hue':hue,'wire_hue':contacts[0]['wire_hue']})
    return found


def _near_measured_lead(pin, contact, tolerance):
    """A model tip may lie anywhere along the independently measured lead."""
    ax,ay = contact['base']
    bx,by = contact['point']
    dx,dy = bx-ax,by-ay
    length_squared = dx*dx+dy*dy
    if not length_squared:
        return False
    t = max(0.,min(1.,((pin.point.x-ax)*dx+(pin.point.y-ay)*dy)/length_squared))
    return math.hypot(pin.point.x-(ax+t*dx),pin.point.y-(ay+t*dy)) <= tolerance


def recover_visual_switches(ir, rgb, ctx=None):
    updates = []
    for token in sorted(ir.texts,key=lambda t:t.internal_id):
        if token.score < .98 or not _REFERENCE.fullmatch(token.raw_text):
            continue
        if any(c.key == token.raw_text for c in ir.components):
            continue
        choices = grid_switch_candidates(rgb,token.bbox,ctx)
        method = 'visible-switch-pad-grid-and-leads'
        if not choices:
            choices = open_switch_candidates(rgb,token.bbox,ctx)
            method = 'visible-open-switch-contacts-and-actuator'
        if not choices:
            choices = lever_switch_candidates(rgb,token.bbox,ctx)
            method = 'visible-open-switch-round-contacts-and-lever'
        if len(choices) != 1:
            continue
        grid = choices[0]
        body = grid['body']
        # A uniquely overlapping, unkeyed two-terminal resistor proposal can
        # be a misclassified switch. Require agreement of both measured ends
        # before repairing that owner. Box/IC and conflicting owners remain
        # protected, and no duplicate component or terminal is created.
        overlaps = [c for c in ir.components if _contains(c.body_bbox or c.bbox,body) > .25]
        reused = None
        ordered_pins = None
        if overlaps:
            if len(overlaps) != 1 or len(grid['contacts']) != 2:
                continue
            candidate = overlaps[0]
            owned = [p for p in ir.pins if p.component_id == candidate.internal_id]
            bounds = candidate.body_bbox or candidate.bbox
            font = max(8.,min(32.,token.bbox.height))
            if (candidate.key or candidate.name or candidate.value
                    or candidate.category.upper() not in {'R','RESISTOR','SWITCH'} or len(owned) != 2
                    or _contains(bounds,body) < .7 or bounds.width*bounds.height > 4*body.width*body.height
                    or math.hypot(bounds.center.x-body.center.x,bounds.center.y-body.center.y) > font):
                continue
            orderings = [order for order in permutations(owned) if all(
                _near_measured_lead(pin,contact,max(2.,font*.15))
                for pin,contact in zip(order,grid['contacts']))]
            if len(orderings) != 1:
                continue
            reused,ordered_pins = candidate,orderings[0]
        # Competing printed switch references inside the same contact grid
        # cannot establish a unique owner, even when only one was read first.
        if any(other is not token and other.score >= .98 and _REFERENCE.fullmatch(other.raw_text)
               and other.raw_text != token.raw_text and body.x1 <= other.bbox.center.x <= body.x2
               and body.y1 <= other.bbox.center.y <= body.y2 for other in ir.texts):
            continue
        cid = reused.internal_id if reused is not None else 'visual-switch:'+token.internal_id
        points = [p['point'] for p in grid['contacts']]
        bounds = BBox(x1=min(body.x1,*[p[0] for p in points]),y1=min(body.y1,*[p[1] for p in points]),
                      x2=max(body.x2,*[p[0] for p in points]),y2=max(body.y2,*[p[1] for p in points]))
        row = {'method':method,'component_id':cid,'reference':token.raw_text,
               'reference_token':token.internal_id,'pads':grid['pads'],'contacts':grid['contacts'],
               'symbol_hue':grid['symbol_hue'],'wire_hue':grid['wire_hue'],'gt_access':False,'numbers_inferred':False}
        if reused is None:
            component = Component(internal_id=cid,key=token.raw_text,key_score=token.score,category='Switch',
                                  bbox=bounds,body_bbox=body,score=token.score)
            ir.components.append(component)
        else:
            component = reused
            row.update(reused_existing_owner=True,previous_category=component.category,
                       previous_bbox=component.bbox.model_dump())
            component.key,component.key_score,component.category = token.raw_text,token.score,'Switch'
            component.bbox,component.body_bbox = bounds,body
        component.evidence.append(Evidence(evidence_id=cid,source='geometry',metadata=row))
        for i,contact in enumerate(grid['contacts']):
            if ordered_pins is None:
                pin = Pin(internal_id=f'visual-switch-pin:{cid}:{i}',component_id=cid,
                          point=Point(x=contact['point'][0],y=contact['point'][1]),
                          base=Point(x=contact['base'][0],y=contact['base'][1]),orientation=contact['side'],score=token.score)
                ir.pins.append(pin)
            else:
                pin = ordered_pins[i]
                pin.point = Point(x=contact['point'][0],y=contact['point'][1])
                pin.base = Point(x=contact['base'][0],y=contact['base'][1])
                pin.orientation = contact['side']
            pin.evidence.append(Evidence(evidence_id=pin.internal_id,source='geometry',metadata={
                'method':method,'gt_access':False}))
        token.component_id,token.category,token.details = cid,'designator','designator'
        token.pin_id = None
        updates.append(row)
    ir.metadata['visual_switch_readback'] = {'updates':updates,'gt_access':False}
    return updates
