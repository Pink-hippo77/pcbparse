"""Verify detected power/ground symbols against visible bars and their stem.

A complete T or a centered,
decreasing stack of bars must be visible; nearby OCR supplies the rail name.
This pass runs before association, so duplicate proposals can be removed
without retaining conflicting terminal owners or electrical connections.
"""
from __future__ import annotations

import math
import re

import numpy as np

from pcbparse.graph.palette_wire import hue_map
from pcbparse.ir.models import BBox, Component, Evidence, Pin, Point
from pcbparse.perception.symbol_body import groups, connected_masks, readback_rgb

_SUPPLY = re.compile(r'(?:V[A-Z][A-Z0-9_]*|[+-]?\d+(?:\.\d+)?V(?:\d+)?|\d+V\d+)$')
_GROUND = re.compile(r'(?:[A-Z0-9_]*GND[A-Z0-9_]*|VSS)$')
_RAIL = {'GND', 'GROUND', 'V', 'POWER', 'POWERPORT'}


def _bars(mask, *, maximum_thickness=3):
    result=[]
    for y,row in enumerate(mask):
        for run in groups(np.flatnonzero(row)):
            if len(run)<5:continue
            x1,x2=run[0],run[-1]
            old=next((b for b in result if b[1]+1==y
                      and abs(b[2]+b[3]-x1-x2)<=4
                      and min(b[3],x2)-max(b[2],x1)>=.75*min(b[3]-b[2],x2-x1)),None)
            if old is None:result.append([y,y,x1,x2])
            else:old[1]=y;old[2]=min(old[2],x1);old[3]=max(old[3],x2)
    return [b for b in result if b[1]-b[0]<=maximum_thickness]


def _stem(mask,bar,direction,*,maximum_length=70):
    center=round((bar[2]+bar[3])/2)
    edge=bar[0] if direction<0 else bar[1]
    count=0
    for step in range(1,maximum_length+1):
        y=edge+direction*step
        if not 1<=y<mask.shape[0]-1:break
        if not mask[y,max(0,center-1):center+2].any():break
        count=step
    return count,edge+direction*count,center


def _short_color_stem_supported(bins, ink, stem, direction, hue):
    """A three-pixel stem must end at a separate, observed wire pigment."""
    if stem[0] != 3:
        return False
    _, end, axis = stem
    samples = []
    for step in (1, 2):
        y = end+direction*step
        if not (0 <= y < bins.shape[0] and 0 <= axis < bins.shape[1] and ink[y, axis]):
            return False
        colors = bins[y, max(0, axis-2):axis+3]
        visible = ink[y, max(0, axis-2):axis+3]
        samples.append({int(wire) for wire in colors[visible] if wire >= 0
                        and min((int(wire)-hue)%12, (hue-int(wire))%12) >= 2})
    return bool(samples[0] & samples[1])


def visible_triangle_ground(rgb, box):
    """Measure a closed ground triangle and its external stem in any rotation.

    Callers must independently verify a nearby ground name. A triangular
    amplifier or a diode must never authorize a global ground connection.
    """
    h, w = rgb.shape[:2]
    x0, y0 = max(0, math.floor(box.x1)-4), max(0, math.floor(box.y1)-4)
    x2, y2 = min(w, math.ceil(box.x2)+5), min(h, math.ceil(box.y2)+5)
    bins, ink = hue_map(rgb[y0:y2, x0:x2])
    if not ink.size:
        return None
    height, width = ink.shape
    candidates = []
    for hue in range(12):
        mask = ink & (bins == hue)
        if mask.sum() < 20:
            continue
        for rotation in range(4):
            local = np.rot90(mask, rotation)
            def restore(x, y):
                if rotation == 1: x, y = width-1-y, x
                elif rotation == 2: x, y = width-1-x, height-1-y
                elif rotation == 3: x, y = y, height-1-x
                return x+x0, y+y0
            for bar in _bars(local):
                length = bar[3]-bar[2]
                if not 8 <= length <= 45:
                    continue
                stem, end, center = _stem(local, bar, -1)
                if not 4 <= stem <= 65:
                    continue
                base = bar[1]
                for depth in range(max(6, round(length*.35)), min(40, round(length*1.3))+1):
                    if base+depth+2 >= local.shape[0]:
                        break
                    supported = []
                    for step in range(1, depth+1):
                        half = length*.5*(1-step/depth)
                        yy = base+step
                        for xx in (round(center-half), round(center+half)):
                            supported.append(local[yy, max(0, xx-1):xx+2].any())
                    if np.mean(supported) < .90:
                        continue
                    # The vertex ends here; a rectangle, crossing or continuing
                    # stroke cannot masquerade as the taper of a closed triangle.
                    if local[base+depth+2:min(local.shape[0], base+depth+5), max(0, center-1):center+2].any():
                        continue
                    tip = Point(x=restore(center, end)[0], y=restore(center, end)[1])
                    corners = [restore(x, y) for x in (bar[2], bar[3]) for y in (bar[0], base+depth)]
                    body = BBox(x1=min(p[0] for p in corners), y1=min(p[1] for p in corners),
                                x2=max(p[0] for p in corners), y2=max(p[1] for p in corners))
                    shift = math.hypot(body.center.x-box.center.x, body.center.y-box.center.y)
                    candidates.append(((-shift, float(np.mean(supported))), {
                        'body': body, 'point': tip, 'side': 'TRBL'[rotation], 'bars': 3,
                        'hue': hue, 'triangle': True}))
    return max(candidates, key=lambda row: row[0])[1] if candidates else None


def visible_rail_shape(rgb,box,*,monochrome=False,text_boxes=()):
    height,width=rgb.shape[:2]
    x0=max(0,math.floor(box.x1)-4);y0=max(0,math.floor(box.y1)-4)
    x2=min(width,math.ceil(box.x2)+5);y2=min(height,math.ceil(box.y2)+5)
    bins,ink=hue_map(rgb[y0:y2,x0:x2])
    if monochrome:
        for t in text_boxes:
            # OCR boxes may include the symbol's underline at their lower
            # boundary. Remove glyph interiors while preserving that stroke.
            a,b=max(0,math.floor(t.x1)-x0),max(0,math.floor(t.y1)-y0)
            c,d=min(ink.shape[1],math.ceil(t.x2)-x0),min(ink.shape[0],math.ceil(t.y2)-3-y0)
            if a<c and b<d:ink[b:d,a:c]=False
    candidates=[]
    # A monochrome connected wire has no visible color transition at its
    # terminal. Leave that geometry to the existing detector for now.
    for hue in ([-1] if monochrome else range(12)):
        mask=(bins==hue)&ink
        if mask.sum()<10:continue
        for transpose in (False,True):
            m=mask.T if transpose else mask
            bars=_bars(m)
            for bar in bars:
                length=bar[3]-bar[2]
                if not 5<=length<=max(55,min(m.shape)*1.2):continue
                center=(bar[2]+bar[3])/2
                aligned=[b for b in bars if abs((b[2]+b[3])/2-center)<=2
                         and abs((b[0]+b[1]-bar[0]-bar[1])/2)<=max(14,length)]
                aligned.sort(key=lambda b:b[0])
                widths=[b[3]-b[2] for b in aligned]
                ground=False
                if len(aligned) in {3,4}:
                    gaps=np.diff([(b[0]+b[1])/2 for b in aligned])
                    decreasing=all(a>=b+2 for a,b in zip(widths,widths[1:]))
                    increasing=all(a+2<=b for a,b in zip(widths,widths[1:]))
                    ground=bool((decreasing or increasing) and min(gaps)>=2 and max(gaps)<=max(10,length*.7))
                if ground:
                    widest=max(aligned,key=lambda b:b[3]-b[2])
                    if bar is not widest:continue
                    direction=-1 if widest is aligned[0] else 1
                    stem=_stem(m,widest,direction)
                    chosen=aligned
                elif len(aligned)==1:
                    options=[(*_stem(m,bar,d),d) for d in (-1,1)]
                    count,tip,axis,direction=max(options)
                    opposite=_stem(m,bar,-direction)[0]
                    if opposite>2:continue
                    stem=(count,tip,axis);chosen=[bar]
                else:continue
                if stem[0]<4:
                    colors=bins.T if transpose else bins
                    visible=ink.T if transpose else ink
                    if monochrome or len(chosen)!=1 or not _short_color_stem_supported(colors,visible,stem,direction,hue):
                        continue
                a=min(b[2] for b in chosen);b=min(b[0] for b in chosen)
                c=max(b[3] for b in chosen);d=max(b[1] for b in chosen)
                if transpose:
                    body=BBox(x1=x0+b,y1=y0+a,x2=x0+max(b+1,d),y2=y0+c)
                    point=Point(x=x0+stem[1],y=y0+stem[2])
                    side='L' if direction<0 else 'R'
                else:
                    body=BBox(x1=x0+a,y1=y0+b,x2=x0+c,y2=y0+max(b+1,d))
                    point=Point(x=x0+stem[2],y=y0+stem[1])
                    side='T' if direction<0 else 'B'
                shift=math.hypot(body.center.x-box.center.x,body.center.y-box.center.y)
                if monochrome:
                    # A gray wire and its symbol lead have no color boundary.
                    # Keep the anchor close to the observed stem, outside the
                    # bars, without following the wire to an arbitrary crop.
                    offset=min(stem[0],10)
                    if transpose:point=Point(x=x0+(bar[0] if direction<0 else bar[1])+direction*offset,y=y0+stem[2])
                    else:point=Point(x=x0+stem[2],y=y0+(bar[0] if direction<0 else bar[1])+direction*offset)
                rank=(int(ground),-shift,stem[0])
                candidates.append((rank,{'body':body,'point':point,'side':side,'bars':len(chosen),'hue':hue}))
    return max(candidates,key=lambda item:item[0])[1] if candidates else None


def _overlap(a,b):
    area=max(0,min(a.x2,b.x2)-max(a.x1,b.x1))*max(0,min(a.y2,b.y2)-max(a.y1,b.y1))
    return area/max(1,a.width*a.height)


def _rail_obstacle_reason(ir, shape, bins=None):
    """An omitted rail cannot be a measured device edge or printed glyph."""
    body = shape['body']
    for component in ir.components:
        if component.category.upper() in _RAIL or component.body_bbox is None:
            continue
        if _overlap(body, component.body_bbox) > .25:
            return 'inside_measured_symbol_body'
    for token in ir.texts:
        if token.score < .75 or not token.raw_text.strip():
            continue
        b = token.bbox
        # Some text detectors include the supply underline at the box edge.
        # Only the glyph interior vetoes a proposal, preserving that boundary.
        if b.width <= 2 or b.height <= 3:
            continue
        interior = BBox(x1=b.x1+1,y1=b.y1+1,x2=b.x2-1,
                        y2=b.y2-min(3,b.height*.2))
        if _overlap(body,interior) > .5:
            if bins is not None:
                # OCR can enclose a symbol with its printed name. Different
                # symbol/text pigments independently distinguish that case
                # from a T-shaped glyph masquerading as a supply.
                x1,y1=max(0,math.floor(b.x1)),max(0,math.floor(b.y1))
                x2,y2=min(bins.shape[1],math.ceil(b.x2)),min(bins.shape[0],math.ceil(b.y2))
                colors=bins[y1:y2,x1:x2].copy()
                point=shape.get('point',body.center)
                left,top=min(body.x1,point.x),min(body.y1,point.y)
                right,bottom=max(body.x2,point.x),max(body.y2,point.y)
                a,z=max(0,math.floor(left)-x1-1),max(0,math.floor(top)-y1-1)
                c,d=min(colors.shape[1],math.ceil(right)-x1+2),min(colors.shape[0],math.ceil(bottom)-y1+2)
                if a<c and z<d:
                    region=colors[z:d,a:c]
                    region[region==shape['hue']]=-1
                values=colors[colors>=0]
                counts=np.bincount(values,minlength=12) if len(values) else np.zeros(12,dtype=int)
                alternate=[]
                for hue in np.flatnonzero(counts>=8):
                    if hue==shape['hue']:continue
                    yy,xx=np.nonzero(colors==hue)
                    # A wire through an oversized OCR box is not a second
                    # text pigment. Require ink spread in both dimensions.
                    if xx.max()-xx.min()>=3 and yy.max()-yy.min()>=3:
                        alternate.append(int(counts[hue]))
                if alternate and max(alternate)>counts[shape['hue']]:
                    continue
            return 'inside_visible_text'
    return None


def refine_monochrome_ground_ports(ir, rgb):
    """Reject extra ground ports only when decreasing bars verify one stem.

    Keep the measured detector tip. A grayscale stem has no symbol/wire color
    boundary, so its end must not be guessed from a fixed lead length.
    """
    chroma=rgb.max(2).astype(np.int16)-rgb.min(2).astype(np.int16)
    report={'enabled':bool((chroma>30).mean()<.001),'updates':[],'gt_access':False}
    if not report['enabled']:
        ir.metadata['monochrome_ground_ports']=report
        return report
    ink=rgb.min(2)<180;h,w=ink.shape
    rejected=set()
    normals={'L':(-1,0),'R':(1,0),'T':(0,-1),'B':(0,1)}
    by_owner={}
    for pin in ir.pins:by_owner.setdefault(pin.component_id,[]).append(pin)
    for component in ir.components:
        if component.category.upper() not in {'GND','GROUND'}:continue
        shape=visible_rail_shape(rgb,component.bbox,monochrome=True)
        if shape is None or shape['bars']<3:continue
        body=shape['body'];side=shape['side'];nx,ny=normals[side]
        axis=shape['point'].y if nx else shape['point'].x
        edge={'L':body.x1,'R':body.x2,'T':body.y1,'B':body.y2}[side]
        owned=by_owner.get(component.internal_id,[]);choices=[]
        for pin in owned:
            x,y=pin.point.x,pin.point.y
            lateral=abs((y if nx else x)-axis)
            outward=((x if nx else y)-edge)*(nx or ny)
            if lateral>3 or not 2<=outward<=40:continue
            samples=[]
            for t in np.linspace(0,1,max(3,math.ceil(outward))):
                px=round(edge+(x-edge)*t) if nx else round(x)
                py=round(y) if nx else round(edge+(y-edge)*t)
                if 1<=px<w-1 and 1<=py<h-1:samples.append(ink[py-1:py+2,px-1:px+2].any())
            if not samples or np.mean(samples)<.85:continue
            # The existing same-side prediction is more reliable than a
            # stronger response on a neighboring wire or on a ground bar.
            rank=(int(pin.orientation==side),-lateral,float(pin.score))
            choices.append((rank,pin.internal_id,pin))
        if not choices:continue
        keep=max(choices,key=lambda c:(c[0],c[1]))[2]
        remove=[p.internal_id for p in owned if p is not keep]
        rejected.update(remove)
        component.body_bbox=body
        keep.orientation=side
        row={'component_id':component.internal_id,'kept_pin':keep.internal_id,
             'removed_pins':remove,'side':side,'bars':shape['bars'],
             'body_bbox':body.model_dump(),'tip_preserved':True,'gt_access':False}
        keep.evidence.append(Evidence(evidence_id='single-ground-port:'+component.internal_id,
            source='geometry',metadata={'kind':'visible-monochrome-ground-single-port',**row}))
        report['updates'].append(row)
    ir.pins=[p for p in ir.pins if p.internal_id not in rejected]
    for token in ir.texts:
        if token.pin_id in rejected:token.pin_id=None
    ir.metadata['monochrome_ground_ports']=report
    return report


def _visible_name(ir,rgb,body,engine,ctx,ground_only=False,read_cache=None):
    from pcbparse.perception.rail_name_context import interior_terminal_label,local_rail_name_box,label_contains_symbol_stroke
    choices=[]
    pending={}
    observations=[]
    read_cache={} if read_cache is None else read_cache
    def valid(value):return bool(_GROUND.fullmatch(value) or (not ground_only and _SUPPLY.fullmatch(value)))
    for token in ir.texts:
        b=token.bbox
        gap=math.hypot(max(body.x1-b.x2,0,b.x1-body.x2),max(body.y1-b.y2,0,b.y1-body.y2))
        if gap>28 or _overlap(b,body)>.25 or interior_terminal_label(ir,token):continue
        value=token.raw_text.strip().upper()
        region=local_rail_name_box(ir,body,token,split=not valid(value))
        if region is None:continue
        if label_contains_symbol_stroke(body,region,rgb):continue
        gap=math.hypot(max(body.x1-region.x2,0,region.x1-body.x2),max(body.y1-region.y2,0,region.y1-body.y2))
        if token.score>=.75 and valid(value):
            choices.append((token.score-.006*gap,value,token.internal_id,0,region,float(token.score),'ocr_token'))
        elif engine is not None and (token.score<.99 or valid(value.replace(' ',''))):
            # A confident detector can merge adjacent names or omit an
            # underscore. Re-read the local pixels rather than normalizing
            # the spelling or silently assigning the default ground name.
            bounds=(max(0,math.floor(region.x1)-2),max(0,math.floor(region.y1)-2),
                    math.ceil(region.x2)+3,math.ceil(region.y2)+3)
            crop=rgb[bounds[1]:bounds[3],bounds[0]:bounds[2]]
            for rotation in (0,1,3):
                rotated=np.rot90(crop,rotation).copy()
                if not rotated.size:continue
                key=(token.internal_id,rotation,bounds)
                if key not in read_cache:pending[key]=rotated
                observations.append((key,gap,token.internal_id,rotation,region,.95 if token.score>=.99 else .8))
    if pending:
        from pcbparse.perception.ic_visual_readback import _read_batch
        read_cache.update(_read_batch(engine,pending,ctx))
    for key,gap,tid,rotation,region,threshold in observations:
        value,score=read_cache[key]
        value=value.upper()
        if score>=threshold and valid(value):
            choices.append((score-.006*gap,value,tid,rotation,region,float(score),'localized_pixel_read'))
    # A GND fragment from the same pixels must not compete with a complete
    # observed SWD_GND or AGND name. Independent neighboring labels remain
    # distinct and still cause abstention when ownership is ambiguous.
    complete=[]
    for choice in choices:
        a=choice[4]
        if any(len(other[1])>=len(choice[1])+2 and choice[1] in other[1]
                and other[5]>=.95
                and max(0,min(a.x2,other[4].x2)-max(a.x1,other[4].x1))
                    *max(0,min(a.y2,other[4].y2)-max(a.y1,other[4].y1))
                    /max(1,a.width*a.height)>=.7 for other in choices):
            continue
        complete.append(choice)
    choices=sorted(complete,key=lambda row:row[:4],reverse=True)
    if not choices:return None
    if any(c[1]!=choices[0][1] and choices[0][0]-c[0]<.06 for c in choices[1:]):return None
    return choices[0]


def _missing_color_rails(ir,rgb,engine,ctx,read_cache,triangle_grounds=False):
    """Seed omitted rails from actual isolated colored strokes, then read names."""
    bins,ink=hue_map(rgb)
    candidates=[]
    # Text glyphs can resemble T-shaped supplies. Restrict proposals to
    # colors independently observed on verified non-rail symbol bodies.
    symbol_hues=set()
    for c in ir.components:
        if c.body_bbox is None or c.category.upper() in _RAIL:continue
        b=c.body_bbox
        a,z=max(0,math.floor(b.x1)),max(0,math.floor(b.y1))
        d,e=min(rgb.shape[1],math.ceil(b.x2)+1),min(rgb.shape[0],math.ceil(b.y2)+1)
        local=bins[z:e,a:d];colors=local[local>=0]
        if len(colors)>=8:symbol_hues.add(int(np.bincount(colors,minlength=12).argmax()))
    for hue in sorted(symbol_hues):
        mask=(bins==hue)&ink
        if mask.sum()<10:continue
        for left,top,local in connected_masks(mask):
            h,w=local.shape
            if not 5<=max(h,w)<=70:continue
            region=BBox(x1=max(0,left-8),y1=max(0,top-8),
                        x2=min(rgb.shape[1],left+w+8),y2=min(rgb.shape[0],top+h+8))
            if any(c.category.upper() in {'BOX','IC','BOXIC','BOX_IC'} and c.body_bbox is not None
                   and _overlap(region,c.body_bbox)>.75 for c in ir.components):continue
            shape=visible_rail_shape(rgb,region)
            if triangle_grounds and (shape is None or shape['bars'] < 3):
                shape=visible_triangle_ground(rgb,region) or shape
            if shape is None:continue
            body=shape['body']
            if _rail_obstacle_reason(ir,shape,bins) is not None:continue
            if any(max(abs(getattr(body,k)-getattr(s['body'],k)) for k in ('x1','y1','x2','y2'))<=3 for s in candidates):continue
            # Missing supply symbols need a printed name. The three decreasing
            # bars independently identify ground even when its label was missed.
            if any(c.category.upper() in _RAIL and math.hypot((c.body_bbox or c.bbox).center.x-body.center.x,
                    (c.body_bbox or c.bbox).center.y-body.center.y)<18 for c in ir.components):continue
            name=_visible_name(ir,rgb,body,engine,ctx,shape['bars']>=3,read_cache)
            if shape.get('triangle') and name is None:continue
            if name is None and shape['bars']<3:continue
            cid=f'visual-color-rail:{round(body.center.x)}:{round(body.center.y)}'
            point=shape['point']
            c=Component(internal_id=cid,category='Gnd' if shape['bars']>=3 else 'V',score=.95,
                        bbox=BBox(x1=min(body.x1,point.x),y1=min(body.y1,point.y),
                                  x2=max(body.x2,point.x),y2=max(body.y2,point.y)))
            ir.components.append(c);candidates.append(shape)


def recover_visual_rails(ir,path,engine=None,ctx=None,*,rgb=None,triangle_grounds=False):
    if rgb is None:
        rgb=readback_rgb(ir,path)
    read_cache={}
    # A printed exterior supply label can seed a missing monochrome symbol,
    # but the same bars/stem check must independently verify its shape.
    chroma=rgb.max(2).astype(float)-rgb.min(2).astype(float)
    monochrome=bool((chroma>30).mean()<.001)
    gray_shapes={}
    if not monochrome:
        _missing_color_rails(ir,rgb,engine,ctx,read_cache,triangle_grounds)
    if monochrome:
        glyph_boxes=[t.bbox for t in ir.texts if t.score>=.75 and len(re.sub(r'\W','',t.raw_text))>=2]
        for token in ir.texts:
            value=token.raw_text.strip().upper();t=token.bbox
            if token.score<.9 or not (_GROUND.fullmatch(value) or _SUPPLY.fullmatch(value)):continue
            if any((c.body_bbox or c.bbox).x1<t.center.x<(c.body_bbox or c.bbox).x2
                   and (c.body_bbox or c.bbox).y1<t.center.y<(c.body_bbox or c.bbox).y2
                   for c in ir.components if c.category.upper() in {'BOX','IC','BOXIC','BOX_IC'}):continue
            font=max(8,min(35,t.height))
            region=BBox(x1=max(0,t.x1-font),y1=max(0,t.y1-3*font),
                        x2=min(ir.image.width,t.x2+font),y2=min(ir.image.height,t.y2+font))
            shape=visible_rail_shape(rgb,region,monochrome=True,text_boxes=glyph_boxes)
            if shape is None:continue
            body=shape['body']
            gap=math.hypot(max(body.x1-t.x2,0,t.x1-body.x2),max(body.y1-t.y2,0,t.y1-body.y2))
            if gap>font or (_GROUND.fullmatch(value) and shape['bars']<3):continue
            owners=[c for c in ir.components if c.category.upper() in _RAIL
                    and math.hypot(c.bbox.center.x-body.center.x,c.bbox.center.y-body.center.y)<max(20,font)]
            component=min(owners,key=lambda c:math.hypot(c.bbox.center.x-body.center.x,c.bbox.center.y-body.center.y)) if owners else None
            if component is None:
                component=Component(internal_id='visual-rail:'+token.internal_id,category='Gnd' if shape['bars']>=3 else 'V',
                    bbox=BBox(x1=min(body.x1,shape['point'].x),y1=min(body.y1,shape['point'].y),
                              x2=max(body.x2,shape['point'].x),y2=max(body.y2,shape['point'].y)),score=token.score)
                ir.components.append(component)
            gray_shapes[component.internal_id]=shape
    observed=[]
    for component in ir.components:
        if component.category.upper() not in _RAIL:continue
        if ctx:ctx.time_budget.check(stage='rail_readback')
        shape=gray_shapes.get(component.internal_id) if monochrome else visible_rail_shape(rgb,component.bbox)
        if triangle_grounds and not monochrome and component.category.upper() in {'GND','GROUND'} and (shape is None or shape['bars'] < 3):
            shape=visible_triangle_ground(rgb,component.bbox) or shape
        if shape is None:continue
        name=_visible_name(ir,rgb,shape['body'],engine,ctx,shape['bars']>=3,read_cache)
        if shape.get('triangle') and name is None:continue
        if shape['bars']==1 and name is None:continue
        category='Gnd' if shape['bars']>=3 or (name and _GROUND.fullmatch(name[1])) else 'V'
        row={'component_id':component.internal_id,'method':'visible-ground-triangle-and-name' if shape.get('triangle') else 'visible-rail-bars-and-stem',
             'previous_category':component.category,'category':category,'bars':shape['bars'],
             'body_bbox':shape['body'].model_dump(),'point':shape['point'].model_dump(),
             'hue':shape['hue'],'gt_access':False}
        component.category=category;component.body_bbox=shape['body']
        if name:
            component.value=name[1]
            row.update(name=name[1],text_token=name[2],text_rotation=name[3])
            row.update(name_bbox=name[4].model_dump(),name_score=name[5],name_source=name[6])
        elif category=='Gnd':component.value='GND'
        component.evidence.append(Evidence(evidence_id='visual-rail:'+component.internal_id,source='geometry',metadata=row))
        observed.append((component,shape,row))
    removed=set()
    for i,(component,shape,row) in enumerate(observed):
        if component.internal_id in removed:continue
        for other,other_shape,_ in observed[i+1:]:
            if max(abs(getattr(shape['body'],k)-getattr(other_shape['body'],k)) for k in ['x1','y1','x2','y2'])<=3:
                removed.add(other.internal_id)
    for component,shape,row in observed:
        if component.internal_id in removed:continue
        owned=[p for p in ir.pins if p.component_id==component.internal_id]
        # The color transition on the visible stem supplies the single rail
        # terminal; extra detections on individual bars must not short nets.
        keep=min(owned,key=lambda p:math.hypot(p.point.x-shape['point'].x,p.point.y-shape['point'].y)) if owned else None
        ir.pins=[p for p in ir.pins if p.component_id!=component.internal_id]
        if keep is None:keep=Pin(internal_id='visual-rail-pin:'+component.internal_id,component_id=component.internal_id,point=shape['point'])
        keep.point=shape['point'];keep.orientation=shape['side'];keep.name='';keep.number=None
        keep.evidence.append(Evidence(evidence_id='visual-rail-terminal:'+component.internal_id,source='geometry',metadata=row))
        ir.pins.append(keep)
    ir.components=[c for c in ir.components if c.internal_id not in removed]
    ir.pins=[p for p in ir.pins if p.component_id not in removed]
    valid_pins={p.internal_id for p in ir.pins}
    for token in ir.texts:
        if token.component_id in removed:token.component_id=None
        if token.pin_id and token.pin_id not in valid_pins:token.pin_id=None
    ir.metadata['visual_rail_readback']={'observations':[r for _,_,r in observed],'removed_duplicates':sorted(removed)}
    return ir.metadata['visual_rail_readback']
