"""Image-supported symbol-body boxes, separate from lead-inclusive detections."""
from __future__ import annotations
from pathlib import Path
from itertools import combinations
import hashlib
import math
import numpy as np
from PIL import Image
from pcbparse.graph.palette_wire import hue_map
from pcbparse.ir.models import BBox,Evidence


def readback_rgb(ir, path):
    """Verify and decode one source image for all visual readback passes."""
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != ir.image.sha256:
        raise ValueError('visual readback image hash mismatch')
    with Image.open(path) as bitmap:
        if bitmap.size != (ir.image.width, ir.image.height):
            raise ValueError('visual readback image dimensions mismatch')
        return np.asarray(bitmap.convert('RGB'))

def connected_masks(mask):
    """Yield local masks and offsets without a separate morphology runtime."""
    from pcbparse.graph.raster_trace import label_pixels
    labels,components=label_pixels(mask)
    for coordinates in components[1:]:
        y1,x1=coordinates.min(0);y2,x2=coordinates.max(0)+1
        yield int(x1),int(y1),labels[y1:y2,x1:x2]==labels[coordinates[0,0],coordinates[0,1]]


def groups(indices):
    out=[]
    for i in indices:
        if out and i==out[-1][-1]+1:out[-1].append(int(i))
        else:out.append([int(i)])
    return out

def _box_rectangle(mask):
    h,w=mask.shape
    xs=[g for g in groups(np.flatnonzero(mask.sum(0)>=max(12,.55*h))) if len(g)<=max(4,.04*w)]
    ys=[g for g in groups(np.flatnonzero(mask.sum(1)>=max(12,.55*w))) if len(g)<=max(4,.04*h)]
    candidates=[]
    for xpair in combinations(xs,2):
        x1,x2=[round(float(np.median(g))) for g in xpair]
        if x2-x1<max(16,.6*w):continue
        for ypair in combinations(ys,2):
            y1,y2=[round(float(np.median(g))) for g in ypair]
            if y2-y1<max(16,.6*h):continue
            edges=[mask[y1:y2+1,max(0,x-1):x+2].any(1).mean() for x in (x1,x2)]
            edges += [mask[max(0,y-1):y+2,x1:x2+1].any(0).mean() for y in (y1,y2)]
            if min(edges)<.85:continue
            rank=((x2-x1)*(y2-y1),sum(edges),-abs((x1+x2)/2-w/2)-abs((y1+y2)/2-h/2))
            candidates.append((rank,(x1,y1,x2,y2)))
    return max(candidates)[1] if candidates else None

def _capacitor(mask):
    candidates=[]
    for vertical,m in [(False,mask),(True,mask.T)]:
        sums=m.sum(1);peak=int(sums.max(initial=0))
        if peak<5:continue
        runs=groups(np.flatnonzero(sums>=max(5,.7*peak)))
        if len(runs)!=2 or any(len(g)>max(4,.2*m.shape[0]) for g in runs):continue
        y1,y2=[float(np.median(g)) for g in runs]
        if not 4<=y2-y1<=3*peak:continue
        bounds=[]
        for g in runs:
            x=np.flatnonzero(m[g].any(0));bounds.append((int(x[0]),int(x[-1])))
        if max(abs(bounds[0][i]-bounds[1][i]) for i in (0,1))>max(3,.2*peak):continue
        x1=min(x[0] for x in bounds);x2=max(x[1] for x in bounds)
        if x2-x1<4:continue
        bbox=(y1,x1,y2,x2) if vertical else (x1,y1,x2,y2)
        candidates.append((peak,bbox))
    return max(candidates)[1] if candidates else None

def _resistor(mask):
    ys,xs=np.nonzero(mask)
    if len(xs)<12:return None
    horizontal=xs.max()-xs.min()>=ys.max()-ys.min()
    m=mask if horizontal else mask.T
    spans=np.zeros(m.shape[1])
    for x in range(m.shape[1]):
        y=np.flatnonzero(m[:,x])
        if len(y)>=2:spans[x]=y[-1]-y[0]
    peak=float(spans.max(initial=0))
    if peak<4:return None
    columns=np.flatnonzero(spans>=max(4,.65*peak))
    if len(columns)<2:return None
    x1,x2=int(columns[0]),int(columns[-1])
    y=np.flatnonzero(m[:,x1:x2+1].any(1));y1,y2=int(y[0]),int(y[-1])
    if x2-x1<5 or not .12<=(y2-y1)/max(1,x2-x1)<=1.2:return None
    return (x1,y1,x2,y2) if horizontal else (y1,x1,y2,x2)

def estimate_body(rgb,component):
    b=component.bbox;h,w=rgb.shape[:2]
    x1=max(0,math.floor(b.x1)-2);y1=max(0,math.floor(b.y1)-2)
    x2=min(w,math.ceil(b.x2)+3);y2=min(h,math.ceil(b.y2)+3)
    bins,ink=hue_map(rgb[y1:y2,x1:x2])
    colors=bins[bins>=0]
    if colors.size<12:return None
    counts=np.bincount(colors,minlength=12);winner=int(counts.argmax())
    support=counts[winner]/len(colors)
    category=component.category.upper()
    if category in {'BOX','IC','BOXIC','BOX_IC'}:
        rectangles=[]
        for hue in np.flatnonzero(counts>=12):
            candidate=_box_rectangle((bins==hue)&ink)
            if candidate is not None:
                a,z,c,d=candidate
                rectangles.append(((c-a)*(d-z),int(hue),candidate))
        if not rectangles:return None
        _,winner,result=max(rectangles)
        support=counts[winner]/len(colors)
    else:
        if support<.5:return None
        mask=(bins==winner)&ink
        if category in {'R','RESISTOR'}:result=_resistor(mask)
        elif category in {'C','CAPACITOR'}:result=_capacitor(mask)
        else:return None
    if result is None:return None
    a,z,c,d=result
    if not (2<c-a<=b.width+5 and 2<d-z<=b.height+5):return None
    body=BBox(x1=x1+a,y1=y1+z,x2=x1+c,y2=y1+d)
    return body,{'hue':winner,'chromatic_support':float(support),'method':'visible-symbol-body-v2'}

def refine_component_body_boxes(ir,path):
    rgb=readback_rgb(ir,path)
    report=[]
    for c in ir.components:
        result=estimate_body(rgb,c)
        if result is None:continue
        body,evidence=result
        # A plate-only estimate must not erase a previously validated symbol
        # extent. Require the fresh plate observation to agree with the one
        # used by the independent detector/stem consensus.
        if c.category.upper() in {'C','CAPACITOR'} and c.body_bbox is not None:
            consensus=next((e.metadata for e in reversed(c.evidence)
                if e.source=='geometry'
                and e.metadata.get('kind')=='detector-and-visible-plate-stem-consensus'
                and e.metadata.get('symbol_bbox')==c.body_bbox.model_dump()),None)
            if consensus and all(abs(getattr(body,k)-consensus['plate_bbox'][k])<=2
                                 for k in ('x1','y1','x2','y2')):
                evidence={**evidence,'method':'detector-and-visible-plate-stem-consensus',
                          'measured_plate_bbox':body.model_dump()}
                body=c.body_bbox
        original=c.bbox.model_dump()
        c.body_bbox=body
        row={'component_id':c.internal_id,'detection_bbox':original,'body_bbox':body.model_dump(),**evidence,'gt_access':False}
        c.evidence.append(Evidence(evidence_id='geometry:body-bbox:'+c.internal_id,source='geometry',metadata=row))
        report.append(row)
    ir.metadata['symbol_body_boxes']=report
    return report
