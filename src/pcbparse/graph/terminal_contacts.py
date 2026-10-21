"""Verify direct contact with a measured, single-terminal colored rail.

Two touching symbol leads need not have a separately drawn wire between them.
This pass requires opposing measured terminals and continuous same-color ink;
proximity or equal text alone never authorizes a connection.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np

from pcbparse.graph.palette_wire import hue_map
from pcbparse.ir.models import Evidence

_NORMALS = {'L': (-1, 0), 'R': (1, 0), 'T': (0, -1), 'B': (0, 1)}
_RAILS = {'GND', 'GROUND', 'V', 'POWER', 'POWERPORT'}
_METHODS = {'visible-rail-bars-and-stem', 'visible-ground-triangle-and-name'}


def _rail_observation(component, pin):
    if component.category.upper() not in _RAILS:
        return None
    for evidence in reversed(pin.evidence):
        row = evidence.metadata
        point = row.get('point')
        if (row.get('method') in _METHODS and row.get('gt_access') is False
                and isinstance(row.get('hue'), int) and 0 <= row['hue'] < 12
                and isinstance(point, dict) and set(point) == {'x', 'y'}
                and math.dist((point['x'], point['y']), (pin.point.x, pin.point.y)) <= 1.):
            return row
    return None


def _continuous_contact(a, b, hue, bins, ink, bodies):
    nx, ny = _NORMALS[a.orientation]
    dx, dy = b.point.x-a.point.x, b.point.y-a.point.y
    distance = math.hypot(dx, dy)
    if distance > 8. or abs(dx*ny-dy*nx) > 1.5 or dx*nx+dy*ny < -1.:
        return False
    h, w = ink.shape
    samples = {}
    for fraction in np.linspace(0., 1., max(2, math.ceil(distance*2)+1)):
        x, y = a.point.x+dx*fraction, a.point.y+dy*fraction
        # A symbol outline cannot provide the route through its interior.
        if any(body.x1+.5 < x < body.x2-.5 and body.y1+.5 < y < body.y2-.5 for body in bodies):
            return False
        px, py = round(x), round(y)
        if not 1 <= px < w-1 or not 1 <= py < h-1:
            return False
        # Sample only across the lead, never jump a blank along its axis.
        local = bins[py, px-1:px+2] if ny else bins[py-1:py+2, px]
        visible = ink[py, px-1:px+2] if ny else ink[py-1:py+2, px]
        same = (local >= 0) & (np.minimum((local-hue) % 12, (hue-local) % 12) <= 1)
        if not visible.any():
            return False
        samples[(px, py)] = bool((visible & same).any())
    colored = list(samples.values())
    # A number glyph can desaturate up to two pixels of the colored lead.
    # Require color on both ends and a continuously occupied short corridor.
    longest_gap = gap = 0
    for value in colored:
        gap = 0 if value else gap+1
        longest_gap = max(longest_gap, gap)
    return colored[0] and colored[-1] and np.mean(colored) >= .6 and longest_gap <= 2


def join_touching_rail_terminals(ir, rgb, ctx=None):
    report = {'method': 'opposing-measured-rail-contact-v1', 'contacts': [],
              'ambiguous_rails': [], 'gt_access': False, 'coordinates_changed': False}
    counts = Counter(p.component_id for p in ir.pins)
    components = ir.component_index()
    membership = {pid:i for i,n in enumerate(ir.electrical_nets) for pid in n.pins}
    attached = {p for n in ir.physical_nets if n.segments for p in n.pins}
    eligible = [p for p in ir.pins if p.component_id in components and p.orientation in _NORMALS
                and p.internal_id in membership
                and not any(e.metadata.get('kind') == 'visible-no-connect-marker' for e in p.evidence)]
    measured = [p for p in eligible if components[p.component_id].category.upper() not in _RAILS
                and any(e.metadata.get('kind') == 'visible-stroke-and-text-lanes'
                        and e.metadata.get('gt_access') is False for e in p.evidence)]
    rails = [(p, _rail_observation(components[p.component_id], p)) for p in eligible
             if counts[p.component_id] == 1]
    rails = [(p, row) for p,row in rails if row is not None]
    if not rails or not measured:
        ir.metadata['direct_rail_contacts'] = report
        return report
    bins, ink = hue_map(rgb)
    parents = list(range(len(ir.electrical_nets)))
    labels = [{n.label} if n.label else set() for n in ir.electrical_nets]
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for rail, observation in rails:
        if ctx:
            ctx.time_budget.check(stage='direct_rail_contacts')
        choices = []
        normal = _NORMALS[rail.orientation]
        for other in measured:
            if other.component_id == rail.component_id or _NORMALS[other.orientation] != tuple(-v for v in normal):
                continue
            # A traced wire already supplies attachment; require a terminal
            # for which the missing symbol-to-symbol contact matters.
            if rail.internal_id in attached and other.internal_id in attached:
                continue
            bodies = [components[p.component_id].body_bbox or components[p.component_id].bbox for p in (rail, other)]
            if _continuous_contact(rail, other, observation['hue'], bins, ink, bodies):
                choices.append(other)
        if len(choices) > 1:
            report['ambiguous_rails'].append(rail.internal_id)
            continue
        if not choices:
            continue
        other = choices[0]
        a, b = find(membership[rail.internal_id]), find(membership[other.internal_id])
        if a == b or len(labels[a] | labels[b]) > 1:
            continue
        left, right = sorted((a, b))
        parents[right] = left
        labels[left] |= labels[right]
        row = {'rail_pin':rail.internal_id, 'other_pin':other.internal_id,
               'rail_component':rail.component_id, 'other_component':other.component_id,
               'distance_px':math.dist((rail.point.x,rail.point.y),(other.point.x,other.point.y)),
               'hue':observation['hue'], 'gt_access':False}
        report['contacts'].append(row)
        other.evidence.append(Evidence(evidence_id='direct-rail-contact:'+other.internal_id+':'+rail.internal_id,
            source='geometry', metadata={'kind':report['method'], **row}))
    grouped = defaultdict(list)
    for i, net in enumerate(ir.electrical_nets):
        grouped[find(i)].append(net)
    merged = []
    for root, nets in grouped.items():
        first = nets[0]
        if len(nets) > 1:
            first.pins = sorted({p for n in nets for p in n.pins})
            first.segments = sorted({s for n in nets for s in n.segments})
            first.physical_net_ids = sorted({p for n in nets for p in n.physical_net_ids})
            first.provenance = sorted({p for n in nets for p in n.provenance} | {report['method']})
            first.label = next(iter(labels[root])) if labels[root] else None
            if any(n.net_type == 'GROUND' for n in nets):
                first.net_type = 'GROUND'
        merged.append(first)
    ir.electrical_nets = merged
    ir.metadata['direct_rail_contacts'] = report
    return report
