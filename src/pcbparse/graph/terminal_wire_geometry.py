"""Exclude symbol leads from wire paths using already detected terminal tips."""
from __future__ import annotations

from collections import Counter
import math

from pcbparse.ir.models import Evidence, Point


def _verified_crystal_tip(pin, components):
    owner = components.get(pin.component_id)
    if owner is None or owner.category.upper() not in {'CRYSTAL', 'XTAL'} or pin.score < .3:
        return False
    # The crystal reader independently checks the printed reference, plates,
    # exactly two observed exterior tips and continuous lead pixels. Keep the
    # original model score; this evidence only permits trimming the same
    # terminal's symbol lead on its own physical stroke.
    return any(e.metadata.get('method') == 'visible-crystal-plates-reference-and-model-leads'
               and e.metadata.get('component_id') == owner.internal_id
               and pin.internal_id in e.metadata.get('pin_ids', ())
               and e.metadata.get('new_pins') == 0
               and e.metadata.get('pin_coordinates_changed') is False
               for e in owner.evidence)


def _observed_lead_limit(pin):
    """Only a current, independently measured color boundary may exceed 24px."""
    limit=24.
    if pin.base is None:
        return limit
    extent=math.hypot(pin.point.x-pin.base.x,pin.point.y-pin.base.y)
    for evidence in pin.evidence:
        row=evidence.metadata
        if row.get('kind')!='independently-measured-symbol-wire-boundary':
            continue
        point=row.get('point',{})
        length=row.get('lead_pixels',0)
        if (row.get('pin_id')==pin.internal_id and row.get('component_id')==pin.component_id
                and 5<=length<=120 and abs(extent-length)<=2
                and math.hypot(point.get('x',math.inf)-pin.point.x,
                               point.get('y',math.inf)-pin.point.y)<=.25):
            limit=max(limit,length+3.)
    return limit


def trim_terminal_leads(ir):
    """Trim only free ends that extend inside the corresponding pin's lead.

    Electrical membership, pin positions and branch vertices are immutable.
    A terminal must be on the same physical stroke, and its outward ray must
    agree with the endpoint-to-wire direction. No wire is extended or added.
    """
    pins, segments = ir.pin_index(), {s.internal_id:s for s in ir.segments}
    components = {c.internal_id:c for c in ir.components}
    normals = {'L':(-1.,0.), 'R':(1.,0.), 'T':(0.,-1.), 'B':(0.,1.)}
    updates = []
    for net in ir.physical_nets:
        lines = [segments[sid] for sid in net.segments if sid in segments]
        degree = Counter((round(p.x, 3), round(p.y, 3)) for s in lines for p in (s.start, s.end))
        candidates = [pins[pid] for pid in net.pins if pid in pins
                      and (pins[pid].score >= .75 or _verified_crystal_tip(pins[pid], components))
                      and pins[pid].orientation in normals]
        limits={pin.internal_id:_observed_lead_limit(pin) for pin in candidates}
        for line in lines:
            for field, other in (('start','end'), ('end','start')):
                a, b = getattr(line,field), getattr(line,other)
                if degree[(round(a.x,3),round(a.y,3))] != 1:
                    continue
                dx, dy = b.x-a.x, b.y-a.y
                length = math.hypot(dx,dy)
                if length < 3:
                    continue
                choices = []
                for pin in candidates:
                    nx, ny = normals[pin.orientation]
                    if (dx*nx+dy*ny)/length < .97:
                        continue
                    tx, ty = pin.point.x-a.x, pin.point.y-a.y
                    along = (tx*dx+ty*dy)/length
                    lateral = abs(tx*dy-ty*dx)/length
                    if not (1.5 < along <= min(limits[pin.internal_id], length-1.5) and lateral <= 3.0):
                        continue
                    # The old wire end must lie towards the symbol from this
                    # pin, never beside an unrelated external branch.
                    if tx*nx+ty*ny <= 0:
                        continue
                    choices.append((lateral,along,pin.internal_id,pin))
                if not choices:
                    continue
                choices.sort(key=lambda c:c[:3])
                if len(choices)>1 and choices[1][0]-choices[0][0] < 1 and choices[1][2]!=choices[0][2]:
                    continue
                lateral,along,pid,pin = choices[0]
                previous = a.model_dump()
                point = Point(x=a.x+dx*along/length,y=a.y+dy*along/length)
                setattr(line,field,point)
                row = {'segment_id':line.internal_id,'endpoint':field,'pin_id':pid,
                       'previous_point':previous,'point':point.model_dump(),
                       'trimmed_pixels':along,'lateral_error':lateral,'gt_access':False}
                line.evidence.append(Evidence(evidence_id=f'terminal-wire:{line.internal_id}:{field}',
                    source='geometry',metadata={'kind':'observed-terminal-wire-boundary',**row}))
                updates.append(row)
    ir.metadata['terminal_wire_geometry'] = {'trimmed_ends':len(updates),'updates':updates,
                                             'pins_changed':0,'nets_changed':0,'gt_access':False}
    return updates
