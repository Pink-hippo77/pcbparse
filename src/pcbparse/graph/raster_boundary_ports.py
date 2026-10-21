"""Experimental terminal recovery from raster strokes, boxes and wire probabilities."""
from __future__ import annotations

import math

import numpy as np

from pcbparse.graph.raster_trace import assign_pins, box, point_box_distance, sanitize_geometry


def recover_boundary_ports(image, probability, points, geometry, min_probability=.35, length=12):
    geometry = sanitize_geometry(geometry)
    gray = image.mean(axis=2)
    h,w = gray.shape
    result = [dict(p) for p in points]
    existing,_=assign_pins(points,geometry,radius=16)
    nc = [box(t.get('location')) for t in geometry['texts'].values()
          if t.get('state')!='deleted' and str(t.get('content') or '').strip().lower().replace('×','x') in ('x','nc','noconnect')]
    for cid,c in geometry['components'].items():
        if c.get('state')=='deleted' or str(c.get('category') or '').lower() not in ('box','c','r'):
            continue
        outer,body=box(c.get('location')),box(c.get('realBBox'))
        if outer is None or body is None:continue
        x1,y1,x2,y2=map(round,outer)
        for side,lo,hi,fixed,normal in (('top',x1,x2,y1,(0,-1)),('bottom',x1,x2,y2,(0,1)),
                                       ('left',y1,y2,x1,(-1,0)),('right',y1,y2,x2,(1,0))):
            candidates=[]
            for coord in range(lo+2,hi-1):
                x,y=(coord,fixed) if side in ('top','bottom') else (fixed,coord)
                if not 1<=x<w-1 or not 1<=y<h-1:continue
                if any(b and point_box_distance(x,y,b)<=6 for b in nc):continue
                if gray[y-1:y+2,x-1:x+2].min()>=180:continue
                samples=[];support=[]
                for step in range(2,length+2):
                    px,py=x+step*normal[0],y+step*normal[1]
                    if 1<=px<w-1 and 1<=py<h-1:
                        samples.append(gray[py-1:py+2,px-1:px+2].min()<180)
                        support.append(float(probability[py-1:py+2,px-1:px+2].max()))
                if len(samples)<length or np.mean(samples)<.85 or np.mean(support)<min_probability:continue
                candidates.append((coord,float(np.mean(support)),float(gray[y,x]),x,y))
            # One terminal per contiguous stroke intersection. Parallel nearby
            # leads separated by even one unsupported coordinate stay distinct.
            groups=[]
            for candidate in candidates:
                if not groups or candidate[0]>groups[-1][-1][0]+1:groups.append([])
                groups[-1].append(candidate)
            for group in groups:
                best=min(group,key=lambda q:(q[2],abs(q[0]-(group[0][0]+group[-1][0])/2)))
                _,score,_,x,y=best
                if any(math.hypot(p['x']-x,p['y']-y)<=(3 if p.get('source')=='wire_supported_boundary' else 6) for p in result):continue
                duplicate=False
                for old in existing:
                    if old['component_id']!=cid or old['side']!=side:continue
                    lateral=abs((old['x']-x)*normal[1]-(old['y']-y)*normal[0])
                    distance=math.hypot(old['x']-x,old['y']-y)
                    if lateral>2 or distance>20:continue
                    visible=[]
                    for fraction in np.linspace(0,1,max(2,math.ceil(distance)+1)):
                        px=round(x+(old['x']-x)*fraction);py=round(y+(old['y']-y)*fraction)
                        if 1<=px<w-1 and 1<=py<h-1:visible.append(gray[py-1:py+2,px-1:px+2].min()<180)
                    if visible and np.mean(visible)>=.85:
                        duplicate=True;break
                if duplicate:continue
                result.append({'x':x,'y':y,'score':score,'area':1,'source':'wire_supported_boundary',
                               'symbol_hint':cid,'side_hint':side})
    return sorted(result,key=lambda p:(p['y'],p['x']))
