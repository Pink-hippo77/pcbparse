"""Recover short, fully visible achromatic jogs between traced wire ends."""
from __future__ import annotations

from collections import defaultdict
import numpy as np


def recover_short_wire_jogs(skeleton,image,blocked,max_gap=6,*,check=None):
    report={'method':'visible-short-wire-jogs','joins':[],'gt_access':False}
    if max_gap<=0:
        return skeleton,report
    h,w=skeleton.shape
    padded=np.pad(skeleton,1)
    degree=sum(padded[1+dy:1+dy+h,1+dx:1+dx+w].astype(np.uint8)
               for dy in (-1,0,1)for dx in (-1,0,1)if dx or dy)
    endpoints=[]
    for y,x in zip(*np.nonzero(skeleton&(degree==1))):
        if check:check()
        normals=[]
        for nx,ny in [(-1,0),(1,0),(0,-1),(0,1)]:
            xx=x-nx*np.arange(1,4);yy=y-ny*np.arange(1,4)
            if ((xx>=0)&(xx<w)&(yy>=0)&(yy<h)).all()and skeleton[yy,xx].all():
                normals.append((nx,ny))
        if len(normals)==1:
            endpoints.append((int(x),int(y),*normals[0]))
    # Endpoints need an existing axial stem, not a nearby isolated pixel.
    size=max(1,3*max_gap);grid=defaultdict(list)
    for i,(x,y,*_)in enumerate(endpoints):grid[(x//size,y//size)].append(i)
    dark=(image.min(2)<180)&(image.max(2).astype(np.int16)-image.min(2)<=45)&~blocked
    proposals=[];partners=defaultdict(set)
    for i,(x,y,nx,ny)in enumerate(endpoints):
        if check:check()
        for gx in range(x//size-1,x//size+2):
            for gy in range(y//size-1,y//size+2):
                for j in grid.get((gx,gy),()):
                    if j<=i:continue
                    bx,by,bnx,bny=endpoints[j]
                    if (bnx,bny)!=(-nx,-ny):continue
                    along=(bx-x)*nx+(by-y)*ny
                    across=abs((bx-x)*ny-(by-y)*nx)
                    if not (1<=along<=max_gap and 2<=across<=2*max_gap and along+across<=3*max_gap):
                        continue
                    a=(x,y)if nx else(y,x);b=(bx,by)if nx else(by,bx)
                    routes=[]
                    for bend in range(min(a[0],b[0]),max(a[0],b[0])+1):
                        # Every added pixel must belong to a visible route.
                        # Never join two nearby stubs through blank space.
                        pixels=set((u,a[1])for u in range(min(a[0],bend),max(a[0],bend)+1))
                        pixels.update((bend,v)for v in range(min(a[1],b[1]),max(a[1],b[1])+1))
                        pixels.update((u,b[1])for u in range(min(b[0],bend),max(b[0],bend)+1))
                        xy=sorted(pixels if nx else{(v,u)for u,v in pixels})
                        xs=np.array([p[0]for p in xy]);ys=np.array([p[1]for p in xy])
                        if not dark[ys,xs].all():continue
                        interior=[(px,py)for px,py in xy if (px,py)not in{(x,y),(bx,by)}]
                        # A third traced stroke makes the missing route
                        # ambiguous; crossings remain with the normal pass.
                        if any(skeleton[py,px]for px,py in interior):continue
                        routes.append((bend,xy))
                    if not routes or routes[-1][0]-routes[0][0]>2:continue
                    route=routes[len(routes)//2]
                    proposals.append((i,j,route[0],route[1]));partners[i].add(j);partners[j].add(i)
    result=skeleton.copy()
    for i,j,bend,xy in proposals:
        if len(partners[i])!=1 or len(partners[j])!=1:continue
        xs=[p[0]for p in xy];ys=[p[1]for p in xy]
        result[ys,xs]=True
        report['joins'].append({'endpoints':[list(endpoints[i][:2]),list(endpoints[j][:2])],
                               'bend_axis':bend,'visible_pixels':len(xy)})
    return result,report
