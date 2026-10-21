"""Use an observed passive reference as its output Name when no name exists."""
from __future__ import annotations

import re
from pcbparse.ir.models import Component, Evidence

_PREFIXES={'R':'R','RESISTOR':'R','C':'C','CAPACITOR':'C','L':'L','INDUCTOR':'L'}


def complete_passive_reference_names(components: list[Component]) -> dict[str,str]:
    updates={}
    for component in components:
        prefix=_PREFIXES.get(component.category.upper())
        if (prefix is None or component.name or not isinstance(component.key,str)
                or not re.fullmatch(prefix+r'[0-9]+[A-Z]?',component.key)):
            continue
        component.name=component.key
        updates[component.internal_id]=component.name
        component.evidence.append(Evidence(
            evidence_id='association:passive-reference-name:'+component.internal_id,
            source='rule',score=component.key_score,
            metadata={'kind':'observed-passive-reference-as-name','component_key':component.key,
                      'name':component.name,'new_reference_created':False,'gt_access':False}))
    return updates
