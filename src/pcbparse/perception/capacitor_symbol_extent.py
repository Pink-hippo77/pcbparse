"""Keep the observed symbol extent instead of collapsing it to the plates.

The independent detector supplies scale. Both extended ends must remain on
the capacitor's continuous same-color central stems, before any wire boundary.
"""
import math

from pcbparse.graph.palette_wire import hue_map
from pcbparse.ir.models import BBox,Evidence
from pcbparse.perception.observed_terminal_boundaries import _outline_hue


def restore_capacitor_symbol_extents(ir,rgb,ctx=None):
    bins,ink=hue_map(rgb)
    report={'updates':[],'gt_access':False,'method':'detector-and-visible-plate-stem-consensus'}
    for c in ir.components:
        if ctx is not None:
            ctx.time_budget.check(stage='capacitor_symbol_extent')
        plate=c.body_bbox
        if c.category.upper() not in {'C','CAPACITOR'} or plate is None or c.score<.85:
            continue
        if not any(e.metadata.get('method')=='visible-capacitor-plates-and-reference' for e in c.evidence):
            continue
        hue=_outline_hue(bins,ink,plate)
        if hue is None:
            continue
        observations=[]
        for e in c.evidence:
            if e.source!='detector':
                continue
            v=e.metadata.get('local_to_global_binding',{}).get('original_geometry',{}).get('values')
            if isinstance(v,list)and len(v)==4:
                observations.append(BBox(**dict(zip(('x1','y1','x2','y2'),v))))
        if not observations:
            continue
        model=observations[-1]
        horizontal=plate.height>plate.width
        width=plate.height if horizontal else plate.width
        transverse=model.height if horizontal else model.width
        center=plate.center.y if horizontal else plate.center.x
        model_center=model.center.y if horizontal else model.center.x
        lo,hi=(model.x1,model.x2) if horizontal else (model.y1,model.y2)
        a,b=(plate.x1,plate.x2) if horizontal else (plate.y1,plate.y2)
        if (not .8*width<=transverse<=1.5*width or abs(center-model_center)>max(3.,.15*width)
                or not lo<=a<b<=hi or hi-lo<1.5*(b-a) or hi-lo>2*width):
            continue
        supported=True
        for start,stop in ((math.ceil(lo),math.floor(a)),(math.ceil(b),math.floor(hi))):
            samples=[]
            for axis in range(start,stop+1):
                x,y=(axis,round(center)) if horizontal else (round(center),axis)
                if not 1<=x<ink.shape[1]-1 or not 1<=y<ink.shape[0]-1:
                    supported=False;break
                colors=bins[y-1:y+2,x] if horizontal else bins[y,x-1:x+2]
                samples.append(bool((colors==hue).any()))
            if len(samples)<2 or sum(samples)/len(samples)<.95:
                supported=False;break
        if not supported:
            continue
        extent=(BBox(x1=lo,y1=plate.y1,x2=hi,y2=plate.y2) if horizontal else
                BBox(x1=plate.x1,y1=lo,x2=plate.x2,y2=hi))
        row={'component_id':c.internal_id,'plate_bbox':plate.model_dump(),
            'detector_bbox':model.model_dump(),'symbol_bbox':extent.model_dump(),'gt_access':False}
        c.body_bbox=extent
        c.evidence.append(Evidence(evidence_id='capacitor-symbol-extent:'+c.internal_id,source='geometry',
            metadata={'kind':'detector-and-visible-plate-stem-consensus',**row}))
        report['updates'].append(row)
    ir.metadata['capacitor_symbol_extent']=report
    return report
