"""Refine terminals only at an observed symbol-to-wire color transition."""
from __future__ import annotations

from collections import Counter
import math

import numpy as np

from pcbparse.graph.palette_wire import PalettePolicy, estimate_palette, hue_map
from pcbparse.ir.models import Evidence, Point

NORMALS = {'L':(-1,0), 'R':(1,0), 'T':(0,-1), 'B':(0,1)}


def restore_monochrome_passive_tips(ir, rgb, observations, *, allow_color=False, observation_source='primary'):
    """Use a measured model tip on the same visible lead, rather than a fixed length."""
    chroma=rgb.max(2).astype(np.int16)-rgb.min(2).astype(np.int16)
    report={'enabled':bool(allow_color or (chroma>30).mean()<.001),'updates':[],'gt_access':False}
    if not report['enabled']:
        ir.metadata['monochrome_terminal_fusion']=report
        return report
    ink=rgb.min(2)<180; h,w=ink.shape
    components=ir.component_index();used=set()
    for pin in ir.pins:
        if not pin.internal_id.startswith(('visual-cap-pin:','visual-rl-pin:')) or pin.orientation not in NORMALS:
            continue
        component=components[pin.component_id];body=component.body_bbox
        if body is None:
            continue
        nx,ny=NORMALS[pin.orientation]
        fixed={'L':body.x1,'R':body.x2,'T':body.y1,'B':body.y2}[pin.orientation]
        choices=[]
        for old in observations:
            if old['id'] in used or old['owner']!=pin.component_id or old['orientation']!=pin.orientation or old['score']<.5:
                continue
            x,y=old['point']; lateral=abs((x-pin.point.x)*ny-(y-pin.point.y)*nx)
            outward=(x-fixed)*nx if nx else (y-fixed)*ny
            distance=math.hypot(x-pin.point.x,y-pin.point.y)
            if lateral>3 or not 2<=outward<=40 or distance>24:
                continue
            a,b=(fixed,y) if nx else (x,fixed)
            samples=[]
            for fraction in np.linspace(0,1,max(3,math.ceil(outward))):
                px,py=round(a+(x-a)*fraction),round(b+(y-b)*fraction)
                if 1<=px<w-1 and 1<=py<h-1:
                    samples.append(ink[py-1:py+2,px-1:px+2].any())
            if not samples or np.mean(samples)<.85:
                continue
            choices.append((.25*distance+3*lateral-2*old['score'],old['id'],old))
        if not choices:
            continue
        _,oid,old=min(choices,key=lambda c:c[:2]);used.add(oid)
        previous=pin.point.model_dump();pin.point=Point(x=old['point'][0],y=old['point'][1])
        pin.base=Point(x=fixed,y=pin.point.y) if nx else Point(x=pin.point.x,y=fixed)
        row={'pin_id':pin.internal_id,'previous':previous,'point':pin.point.model_dump(),
             'model_observation':oid,'model_score':old['score'],'gt_access':False}
        pin.evidence.append(Evidence(evidence_id='model-terminal-fusion:'+pin.internal_id+':'+observation_source,source='geometry',
                                    metadata={'kind':'visible-lead-model-terminal-fusion',**row}))
        report['updates'].append(row)
    ir.metadata['monochrome_terminal_fusion']=report
    return report


def refine_color_terminals(ir, rgb):
    if (rgb.max(2).astype(np.int16)-rgb.min(2).astype(np.int16)).max(initial=0)<50:
        report={'enabled':False,'updates':[],'gt_access':False}
        ir.metadata['color_terminal_refinement']=report
        return report
    bins, ink = hue_map(rgb)
    # This vote only proposes a color for a separately verified lead. Raster
    # topology keeps its own, stricter palette gate.
    palette = estimate_palette(ir, bins, ink, PalettePolicy(minimum_anchor_agreement=.65))
    report = {'enabled':palette['enabled'], 'scope':'capacitor', 'updates':[], 'gt_access':False,
              'palette':{k:palette[k] for k in ('wire_hue_bin','agreement','supporting_pins') if k in palette}}
    if not palette['enabled']:
        ir.metadata['color_terminal_refinement'] = report
        return report
    hue = palette['wire_hue_bin']
    wire = ink & (bins >= 0) & (np.minimum((bins-hue)%12, (hue-bins)%12) <= 1)
    h,w = ink.shape
    components = ir.component_index()
    for pin in ir.pins:
        c = components.get(pin.component_id)
        if c is None or c.body_bbox is None or pin.orientation not in NORMALS:
            continue
        # Other symbol families already have independent visual readback.
        # Apply this repair only to capacitor tips whose plate-based readback
        # otherwise uses a fixed lead length.
        if c.category.upper() not in {'C','CAPACITOR'}:
            continue
        b = c.body_bbox
        nx,ny = NORMALS[pin.orientation]
        horizontal = nx != 0
        lateral = pin.point.y if horizontal else pin.point.x
        lo,hi = (b.y1,b.y2) if horizontal else (b.x1,b.x2)
        if not lo-2 <= lateral <= hi+2:
            continue
        fixed = {'L':b.x1,'R':b.x2,'T':b.y1,'B':b.y2}[pin.orientation]
        best = []
        for axis in range(round(lateral)-2, round(lateral)+3):
            xs = np.array([round(fixed)+nx*t if horizontal else axis for t in range(61)])
            ys = np.array([axis if horizontal else round(fixed)+ny*t for t in range(61)])
            valid = (xs>=0)&(xs<w)&(ys>=0)&(ys<h)
            if not valid.all():
                end = next((i for i,v in enumerate(valid) if not v),61)
                xs,ys = xs[:end],ys[:end]
            if len(xs)<10:
                continue
            on_wire = wire[ys,xs]
            offsets=np.arange(-6,7)
            cross_x=xs[:,None]-ny*offsets; cross_y=ys[:,None]+nx*offsets
            in_image=(cross_x>=0)&(cross_x<w)&(cross_y>=0)&(cross_y<h)
            cross=wire[np.clip(cross_y,0,h-1),np.clip(cross_x,0,w-1)] & in_image
            transverse=cross.sum(1)
            starts = [t for t in range(4,min(46,len(xs)-4))
                      if ink[ys[t],xs[t]] and ((on_wire[t] and on_wire[t:t+5].sum()>=4) or transverse[t]>=6)
                      and not on_wire[max(0,t-3):t].any()]
            for t in starts:
                # The inward stroke must be continuous and have one color
                # distinct from the independently inferred wire palette.
                colors = bins[ys[1:t],xs[1:t]]
                occupied = ink[ys[1:t],xs[1:t]]
                foreign = colors[occupied & (colors>=0) & ~on_wire[1:t]]
                if occupied.mean()<.8 or len(foreign)<3:
                    continue
                common,count = Counter(map(int,foreign)).most_common(1)[0]
                if count/max(1,t-1)<.65:
                    continue
                if (bins[ys[t],xs[t]] == common and
                        not (cross[t,:6].sum()>=2 and cross[t,7:].sum()>=2)):
                    continue
                x,y = int(xs[t]),int(ys[t])
                distance = math.hypot(x-pin.point.x,y-pin.point.y)
                if distance>24:
                    continue
                # Another symbol along the ray is not this terminal's lead.
                if any(other.internal_id!=c.internal_id and other.body_bbox is not None
                       and other.body_bbox.x1<x<other.body_bbox.x2
                       and other.body_bbox.y1<y<other.body_bbox.y2 for other in ir.components):
                    continue
                best.append((abs(axis-lateral),distance,t,x,y,int(common)))
        if not best:
            continue
        _,distance,t,x,y,symbol_hue = min(best)
        if distance<.25:
            continue
        previous = pin.point.model_dump()
        pin.point = Point(x=x,y=y)
        pin.base = Point(x=fixed,y=y) if horizontal else Point(x=x,y=fixed)
        row = {'pin_id':pin.internal_id,'previous':previous,'point':pin.point.model_dump(),
               'wire_hue':int(hue),'symbol_hue':symbol_hue,'lead_pixels':t,'gt_access':False}
        pin.evidence.append(Evidence(evidence_id='color-terminal:'+pin.internal_id,source='geometry',
                                    metadata={'kind':'visible-symbol-wire-color-transition',**row}))
        report['updates'].append(row)
    ir.metadata['color_terminal_refinement'] = report
    return report
