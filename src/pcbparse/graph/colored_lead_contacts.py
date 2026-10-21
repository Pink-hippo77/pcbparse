"""Connect existing terminals over visible symbol-to-wire leads."""
from __future__ import annotations
import numpy as np
from pcbparse.graph.palette_wire import hue_map
from pcbparse.ir.models import Evidence

def observed_lead_contacts(ir,rgb,ctx=None,*,palette=None):
    policy=ir.metadata.get('raster_palette',{});wire_hue=policy.get('wire_hue_bin')
    if not policy.get('enabled')or wire_hue is None:return []
    bins,ink=palette if palette is not None else hue_map(rgb);h,w=ink.shape
    components=ir.component_index();membership={p:n for n in ir.electrical_nets for p in n.pins}
    physical={p:n for n in ir.physical_nets for p in n.pins}
    segments={s.internal_id:s for s in ir.segments}
    choices=[]
    normals={'L':(-1,0),'R':(1,0),'T':(0,-1),'B':(0,1)}
    def similar(a,b):return a>=0 and b>=0 and min((a-b)%12,(b-a)%12)<=1
    for p in ir.pins:
        if ctx:ctx.time_budget.check(stage="observed_lead_contacts")
        owner=components[p.component_id];normal=normals.get(p.orientation)
        pn=physical.get(p.internal_id);net=membership.get(p.internal_id)
        if (normal is None or pn is None or net is None or pn.segments or len(net.pins)!=1
                or owner.category.upper()not in {'AMP','BOX','R','C','L','CRYSTAL','XTAL','MOSFET','DIODE','D'}
                or owner.score<.9 or p.score<.5 or p.base is None):continue
        if any(e.metadata.get('kind')=='visible-no-connect-marker' for e in p.evidence):continue
        nx,ny=normal
        body=owner.body_bbox or owner.bbox
        outward={'L':body.x1-p.point.x,'R':p.point.x-body.x2,
                 'T':body.y1-p.point.y,'B':p.point.y-body.y2}[p.orientation]
        if outward < -1:continue
        if (p.point.x-p.base.x)*nx+(p.point.y-p.base.y)*ny<3:continue
        if abs((p.point.x-p.base.x)*ny-(p.point.y-p.base.y)*nx)>2:continue
        paths=[]
        for step in range(34):
            x,y=round(p.point.x+nx*step),round(p.point.y+ny*step)
            if not 1<=x<w-1 or not 1<=y<h-1:break
            if any((c.body_bbox or c.bbox).x1+.5<x<(c.body_bbox or c.bbox).x2-.5
                   and(c.body_bbox or c.bbox).y1+.5<y<(c.body_bbox or c.bbox).y2-.5
                   for c in ir.components if c.internal_id!=p.component_id):break
            local=[]
            for offset in (-1,0,1):
                px,py=x-ny*offset,y+nx*offset
                if ink[py,px]and bins[py,px]>=0:local.append(int(bins[py,px]))
            if not local and not (ink[y-1:y+2,x].any()if nx else ink[y,x-1:x+2].any()):break
            paths.append(local or [-1])
        if len(paths)<8:continue
        initial=[b for local in paths[:3]for b in local if b>=0 and not similar(b,wire_hue)]
        if not initial:continue
        symbol=max(set(initial),key=initial.count)
        if not all(any(similar(b,symbol)for b in local)for local in paths[:3]):continue
        transition=next((i for i in range(3,len(paths)-2)
                         if all(any(similar(b,wire_hue)for b in local)for local in paths[i:i+3])),None)
        if transition is None:continue
        prefix=[any(similar(b,symbol)for b in local)for local in paths[:transition]]
        if sum(prefix)<len(prefix)-2 or not all(prefix[:3]):continue
        if any(any(b>=0 and not similar(b,symbol)for b in local)for local in paths[:transition]):continue
        candidates={}
        for other in ir.physical_nets:
            if other.internal_id==pn.internal_id:continue
            for sid in other.segments:
                s=segments[sid];a,b=s.start,s.end
                dx,dy=b.x-a.x,b.y-a.y;length=dx*dx+dy*dy
                if not length:continue
                tx,ty=p.point.x+nx*transition,p.point.y+ny*transition
                f=max(0.,min(1.,((tx-a.x)*dx+(ty-a.y)*dy)/length))
                x,y=a.x+f*dx,a.y+f*dy
                along=(x-p.point.x)*nx+(y-p.point.y)*ny
                lateral=abs((x-p.point.x)*ny-(y-p.point.y)*nx)
                if transition-2<=along<=min(32,len(paths)-1)and lateral<=1.5:
                    candidates[other.internal_id]=(along,other,x,y)
        if len(candidates)!=1:continue
        along,other,x,y=next(iter(candidates.values()))
        target=membership.get(other.pins[0])if other.pins else next((n for n in ir.electrical_nets if other.internal_id in n.physical_net_ids),None)
        if target is None or target.internal_id==net.internal_id:continue
        if net.label and target.label and net.label!=target.label:continue
        choices.append({'pin_id':p.internal_id,'owner':owner.key,'category':owner.category,
                        'source_net':net.internal_id,'target_net':target.internal_id,'physical_net':other.internal_id,
                        'position':p.point.model_dump(),'attachment':{'x':x,'y':y},'distance':along,
                        'symbol_hue':symbol,'wire_hue':wire_hue})
    return choices


def join_observed_colored_leads(ir,rgb,ctx=None,*,palette=None):
    """Preserve every pin and wire; add only independently visible contacts."""
    contacts=observed_lead_contacts(ir,rgb,ctx,palette=palette)
    electrical={n.internal_id:n for n in ir.electrical_nets}
    physical={n.internal_id:n for n in ir.physical_nets}
    pins=ir.pin_index()
    updates=[]
    for contact in contacts:
        source=electrical.get(contact['source_net']);target=electrical.get(contact['target_net'])
        if source is None or target is None or source.internal_id==target.internal_id:continue
        pid=contact['pin_id']
        # Recheck ownership after earlier contacts; no already joined network
        # is merged on the basis of a proposal made while it was isolated.
        if source.pins!=[pid] or source.segments:continue
        old_physical=[physical[n]for n in source.physical_net_ids if n in physical]
        if len(old_physical)!=1 or old_physical[0].pins!=[pid] or old_physical[0].segments:continue
        destination=physical.get(contact['physical_net'])
        if destination is None or destination.internal_id not in target.physical_net_ids:continue
        destination.pins.append(pid);target.pins.append(pid)
        if source.label and not target.label:target.label=source.label
        ir.physical_nets.remove(old_physical[0]);ir.electrical_nets.remove(source)
        electrical.pop(source.internal_id,None);physical.pop(old_physical[0].internal_id,None)
        pins[pid].evidence.append(Evidence(evidence_id='visible-colored-lead:'+pid,
            source='geometry',metadata={'kind':'visible-colored-lead-contact','gt_access':False,
                'image_sha256':ir.image.sha256,**contact}))
        target.provenance.append('visible-colored-lead-contact')
        updates.append(contact)
    report={'method':'visible-colored-lead-contacts','gt_access':False,'contacts':updates,
            'pin_coordinates_changed':False,'pin_identities_changed':False,'wire_segments_changed':False}
    ir.metadata['colored_lead_contacts']=report
    return report

