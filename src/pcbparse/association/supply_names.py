"""Resolve sheet supply names from visible text and detected ground symbols."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from pcbparse.ir.models import CircuitIR, Evidence

_GROUND = re.compile(r'^(?:[A-Z0-9_]*GND[A-Z0-9_]*|VSS)$')
_SUPPLY = re.compile(r'^(?:V[A-Z][A-Z0-9_]*|[+-]?\d+(?:\.\d+)?V(?:\d+)?|\d+V\d+)$')


def bind_supply_names(ir: CircuitIR, geometry: dict[str, Any]) -> dict[str, str]:
    """Q&A 1013: same-named ground/supply symbols share a logical net.

    Ground is a symbol class with a conventional default name. Explicit AGND,
    DGND, etc. override that default. A voltage/power symbol needs a visible
    name and one observed terminal; a two-terminal voltage source is excluded.
    """
    counts = Counter(p.component_id for p in ir.pins)
    boxes = [c for c in ir.components if c.category.lower() == 'box']
    names = {}
    for c in ir.components:
        category = c.category.lower()
        if category not in {'gnd', 'ground', 'v', 'power', 'powerport'}:
            continue
        if category not in {'gnd', 'ground'} and counts[c.internal_id] != 1:
            continue
        pattern = _GROUND if category in {'gnd', 'ground'} else _SUPPLY
        visual=[e.metadata for e in c.evidence if e.metadata.get('method') in {'visible-rail-bars-and-stem','visible-ground-triangle-and-name'}
                and e.metadata.get('name') and pattern.fullmatch(e.metadata['name'])]
        if visual:
            name=visual[-1]['name']
            c.value=name;names[c.internal_id]=name
            if c.internal_id in geometry['components']:geometry['components'][c.internal_id]['net_name']=name
            continue
        candidates = []
        for t in ir.texts:
            value = (t.resolved_text or t.raw_text).strip().upper()
            if t.score < .75 or not pattern.fullmatch(value):
                continue
            if t.pin_id is not None or (t.component_id is not None and t.component_id != c.internal_id):
                continue
            point = t.bbox.center
            if any(b.bbox.x1 < point.x < b.bbox.x2 and b.bbox.y1 < point.y < b.bbox.y2 for b in boxes):
                continue
            dx = max(c.bbox.x1-point.x, 0, point.x-c.bbox.x2)
            dy = max(c.bbox.y1-point.y, 0, point.y-c.bbox.y2)
            distance = math.hypot(dx, dy)
            if distance <= 24:
                candidates.append((distance, value, t.internal_id))
        candidates.sort()
        if candidates:
            nearest, name, _ = candidates[0]
            if any(v != name and d-nearest <= 4 for d,v,_ in candidates[1:]):
                continue
            tokens = [tid for d,v,tid in candidates if v == name and d-nearest <= 4]
            method = 'visible-ocr-supply-name'
        elif category in {'gnd', 'ground'}:
            name, tokens, method = 'GND', [], 'detected-ground-symbol-default'
        else:
            continue
        names[c.internal_id] = name
        previous = c.value
        c.value = name
        c.evidence.append(Evidence(evidence_id='association:supply-name:'+c.internal_id, source='rule',
            metadata={'kind':'supply-net-name','method':method,'name':name,'token_ids':tokens,
                      'previous_value':previous,'gt_access':False}))
        if c.internal_id in geometry['components']:
            geometry['components'][c.internal_id]['net_name'] = name
    ir.metadata['supply_net_names'] = names
    return names
