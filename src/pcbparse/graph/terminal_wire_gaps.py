"""Recover visible axial gaps between existing wire fragments at a terminal."""
from __future__ import annotations

import math
import numpy as np


def recover_terminal_wire_gaps(wire, image, probability, body_blocked, pins, geometry, *, check=None, max_length=24):
    """Join existing model fragments along a supported terminal's outward ray.

    The measured/detected pin and its position stay unchanged. Every added
    pixel must belong to one connected ink corridor, outside bodies and OCR
    glyphs. The far fragment needs independent strong model support; nearby
    endpoints alone cannot authorize a connection.
    """
    from pcbparse.graph.raster_trace import box, label_pixels, nearest_labels, text_stroke_obstacles

    report={'method':'visible-terminal-wire-gaps','joins':[],'gt_access':False,
            'pin_coordinates_changed':False}
    if max_length <= 0 or not pins:
        return wire,report
    labels,components=label_pixels(wire,check=check)
    visible=(image.min(2)<180)&~body_blocked&~text_stroke_obstacles(image,geometry)
    h,w=wire.shape
    result=wire.copy();parents=list(range(len(components)))
    def find(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    for pin in pins:
        if check:
            check()
        if not pin.get('source_pin_id') or not pin.get('outward_normal'):
            continue
        component=geometry['components'].get(pin.get('component_id'))
        if component is None:
            continue
        body,outer=box(component.get('realBBox')),box(component.get('location'))
        if body is None or outer is None or body==outer:
            continue
        category=str(component.get('category','')).upper()
        if not (pin.get('owner_verified')is True or category in {'R','C','L','CRYSTAL','XTAL','D','DIODE'}and pin.get('score',0)>=.75):
            continue
        nx,ny=pin['outward_normal'];x0,y0=float(pin['x']),float(pin['y'])
        if (nx,ny)not in {(-1,0),(1,0),(0,-1),(0,1)}:
            continue
        distance={'left':body[0]-x0,'right':x0-body[2],'top':body[1]-y0,'bottom':y0-body[3]}.get(pin.get('side'),-math.inf)
        if not 1 <= distance <= 64:
            continue
        start=nearest_labels(labels,x0,y0,3,(nx,ny),1.5)
        start=[r for r in start if len(components[r[1]])>=3]
        if not start or len(start)>1 and start[1][0]-start[0][0]<=1:
            continue
        first=start[0][1]
        # Only start on the actual local fragment, never bridge a blank from
        # a rounded point to a different nearby line.
        paths={}
        missing=False
        for step in range(max_length+1):
            x,y=round(x0+nx*step),round(y0+ny*step)
            current={}
            for offset in (-1,0,1):
                px,py=x-ny*offset,y+nx*offset
                if not 0<=px<w or not 0<=py<h or not visible[py,px]:
                    continue
                if step==0:
                    if int(labels[py,px])==first:
                        current[(px,py)]=[(px,py)]
                else:
                    previous=[(p,path)for p,path in paths.items()if max(abs(px-p[0]),abs(py-p[1]))<=1]
                    if previous:
                        _,path=min(previous,key=lambda row:(abs(row[0][0]-px)+abs(row[0][1]-py),row[0]))
                        current[(px,py)]=[*path,(px,py)]
            if not current:
                break
            paths=current
            targets={int(labels[py,px])for px,py in paths if labels[py,px]and find(int(labels[py,px]))!=find(first)}
            if targets:
                if len(targets)!=1 or not missing:
                    break
                target=next(iter(targets));pixels=components[target]
                near=pixels[(np.abs(pixels[:,1]-x)<=4)&(np.abs(pixels[:,0]-y)<=4)]
                if len(pixels)<8 or len(near)<3 or np.count_nonzero(probability[near[:,0],near[:,1]]>=.4)<3:
                    break
                candidates=[path for (px,py),path in paths.items()if int(labels[py,px])==target]
                path=min(candidates,key=lambda path:(sum(abs(px-x0-nx*i)+abs(py-y0-ny*i)for i,(px,py)in enumerate(path)),path))
                # Third fragments inside the corridor make the contact
                # ambiguous. Keep the existing crossing path unchanged.
                occupied={find(int(labels[py,px]))for px,py in path if labels[py,px]}
                if not occupied <= {find(first),find(target)}:
                    break
                xs,ys=zip(*path);result[list(ys),list(xs)]=True
                parents[find(target)]=find(first)
                report['joins'].append({'pin_id':pin['source_pin_id'],'fragment_labels':[first,target],
                    'start':list(path[0]),'end':list(path[-1]),'visible_pixels':len(path)})
                break
            if not any(labels[py,px]for px,py in paths):
                missing=True
    return result,report
