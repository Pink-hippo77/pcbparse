"""Restore continuous wire geometry across a classified non-junction crossing.

Raster tracing cuts a small square to keep perpendicular strokes separate.
That cut is a topology device, not two visible wire endpoints. Only opposing,
free, collinear ends already in the same electrical net may be joined here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import math

from pcbparse.ir.models import Evidence, WireSegment


def restore_crossing_wires(ir, crossings, ctx=None):
    segments = ir.segment_index()
    owners = defaultdict(set)
    physical = defaultdict(set)
    degree = Counter()
    buckets = defaultdict(list)
    cell_size = 8

    def xy(p): return (round(p.x, 4), round(p.y, 4))

    def index(line, net_id):
        for field in ('start', 'end'):
            p = getattr(line, field)
            degree[(net_id, *xy(p))] += 1
            buckets[(math.floor(p.x/cell_size), math.floor(p.y/cell_size))].append((line.internal_id, field))

    for net in ir.electrical_nets:
        for sid in net.segments:
            owners[sid].add(net.internal_id)
    for net in ir.physical_nets:
        for sid in net.segments:
            physical[sid].add(net.internal_id)
    for sid, line in segments.items():
        if len(owners[sid]) == 1:
            index(line, next(iter(owners[sid])))

    replacements = {}
    physical_parents = {n.internal_id:n.internal_id for n in ir.physical_nets}

    def root(pid):
        while physical_parents[pid] != pid:
            physical_parents[pid] = physical_parents[physical_parents[pid]]
            pid = physical_parents[pid]
        return pid

    def candidates(x, y, radius, axis, sign):
        result = []
        cx, cy = math.floor(x/cell_size), math.floor(y/cell_size)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for sid, field in buckets.get((cx+dx, cy+dy), ()):
                    if sid not in segments or len(owners[sid]) != 1 or len(physical[sid]) != 1:
                        continue
                    line = segments[sid]
                    # This repair only reverses the explicitly recorded raster
                    # cut, never generic gaps or non-raster observations.
                    if not any(e.metadata.get('kind') in {'raster-model-trace', 'non-junction-wire-continuation'}
                               for e in line.evidence):
                        continue
                    p = getattr(line, field)
                    q = line.end if field == 'start' else line.start
                    net_id = next(iter(owners[sid]))
                    if degree[(net_id, *xy(p))] != 1:
                        continue
                    along = (p.x-x) if axis == 0 else (p.y-y)
                    lateral = (p.y-y) if axis == 0 else (p.x-x)
                    outward = (q.x-p.x) if axis == 0 else (q.y-p.y)
                    drift = (q.y-p.y) if axis == 0 else (q.x-p.x)
                    if not (radius <= sign*along <= radius+2 and abs(lateral) <= 1.5
                            and sign*outward >= 4 and abs(drift) <= 1.5):
                        continue
                    result.append((sid, field, p, q, net_id))
        return result

    updates = []
    for ci, crossing in enumerate(sorted(crossings, key=lambda c:(c['y'], c['x']))):
        if ctx and ci % 128 == 0:
            ctx.time_budget.check(stage='crossing_wire_geometry')
        if crossing.get('connected') is not False or crossing.get('decision') not in {
                'straight_through_no_dot', 'learned_image_crossing'}:
            continue
        if crossing.get('review_required') and not (
                crossing.get('decision') == 'learned_image_crossing'
                and 0 <= crossing.get('connection_probability', 1.) <= .1):
            # Asymmetric ink near a small dot is not enough evidence for a
            # continuous non-junction stroke. Preserve the original split.
            continue
        x, y = crossing['x'], crossing['y']
        radius = crossing.get('cut_radius', 3)
        if radius not in (1, 2, 3) or not all(math.isfinite(v) for v in (x, y)):
            continue
        # A terminal at the crossing is a distinct boundary, even if the
        # crossing classifier disagrees. Do not erase that boundary.
        if any(math.hypot(p.point.x-x, p.point.y-y) <= radius+2 for p in ir.pins):
            continue
        for axis in (0, 1):
            left = candidates(x, y, radius, axis, -1)
            right = candidates(x, y, radius, axis, 1)
            if len(left) != 1 or len(right) != 1:
                continue
            a, b = left[0], right[0]
            if a[0] == b[0] or a[4] != b[4]:
                continue
            if (abs(a[2].y-b[2].y) if axis == 0 else abs(a[2].x-b[2].x)) > 1.5:
                continue
            old_ids = sorted((a[0], b[0]))
            sid = 'crossing-wire:' + sha256(('\0'.join(old_ids)).encode()).hexdigest()[:24]
            old = [segments[key] for key in old_ids]
            detail = {'kind':'non-junction-wire-continuation', 'x':x, 'y':y, 'axis':axis,
                      'cut_radius':radius, 'previous_segments':old_ids, 'gt_access':False}
            line = WireSegment(internal_id=sid, start=a[3].model_copy(), end=b[3].model_copy(),
                score=min(s.score for s in old), source_polyline_ids=sorted({p for s in old for p in s.observed_polyline_ids}),
                evidence=[Evidence(evidence_id=sid, source='geometry', metadata=detail)])
            pa, pb = root(next(iter(physical[a[0]]))), root(next(iter(physical[b[0]])))
            dest, other = sorted((pa, pb))
            physical_parents[other] = dest
            for key in old_ids:
                old_line = segments.pop(key)
                for p in (old_line.start, old_line.end):
                    degree[(a[4], *xy(p))] -= 1
                replacements[key] = sid
            segments[sid] = line
            owners[sid] = {a[4]}; physical[sid] = {dest}
            index(line, a[4])
            updates.append({'segment_id':sid, **detail})

    def replacement(sid):
        while sid in replacements:
            sid = replacements[sid]
        return sid

    if updates:
        ir.segments = list(segments.values())
        groups = {}
        for net in ir.physical_nets:
            dest = root(net.internal_id)
            if dest not in groups:
                groups[dest] = net.model_copy(deep=True)
                groups[dest].internal_id = dest
                groups[dest].pins = []; groups[dest].segments = []; groups[dest].labels = []
            group = groups[dest]
            group.pins = sorted(set(group.pins) | set(net.pins))
            group.segments = sorted(set(group.segments) | {replacement(sid) for sid in net.segments})
            group.labels = sorted(set(group.labels) | set(net.labels))
            group.score = min(group.score, net.score)
            group.repair_ratio = max(group.repair_ratio, net.repair_ratio)
        ir.physical_nets = list(groups.values())
        for net in ir.electrical_nets:
            net.segments = sorted({replacement(sid) for sid in net.segments})
            net.physical_net_ids = sorted({root(pid) for pid in net.physical_net_ids})
    result = {'restored_axes':len(updates), 'updates':updates,
              'pin_fields_changed':False, 'electrical_membership_changed':False, 'gt_access':False}
    ir.metadata['crossing_wire_geometry'] = result
    return result
