"""Fill missing identities of observed pins using an exact static-library match."""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from pcbparse.ir.models import CircuitIR, Evidence
from pcbparse.knowledge.library_index import normalize_model_name, split_variant_suffix


@lru_cache(maxsize=4)
def _load(path: str, expected_sha256: str) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if not re.fullmatch(r'[a-f0-9]{64}', expected_sha256) or digest != expected_sha256:
        raise ValueError('pin library SHA256 mismatch')
    records = json.loads(raw)
    if not isinstance(records, dict):
        raise ValueError('pin library must be a mapping')
    return records, digest


def _number_is_explicit(value: str | None) -> bool:
    return bool(value and re.fullmatch(r'(?:\d{1,4}|[A-Z]{1,2}\d{1,3}|\d{1,3}[A-Z]{1,2})', value))


def complete_library_pin_numbers(ir: CircuitIR, config: dict) -> dict:
    options = config.get('knowledge', {})
    path = options.get('pin_library_path')
    if path is None:
        return {'enabled': False}
    records, digest = _load(str(Path(path).resolve()), options.get('pin_library_sha256', ''))
    updates = []
    rejected = []
    for component in ir.components:
        if component.category.upper() not in {'BOX', 'IC', 'BOXIC', 'BOX_IC'} or not component.name:
            continue
        title = normalize_model_name(component.name)
        matches = [(k, v) for k, v in records.items()
                   if title in {normalize_model_name(k), normalize_model_name(split_variant_suffix(k)[0])}]
        if not matches:
            continue
        # Variants must agree on every name used. Do not choose a package by score.
        maps = []
        for _, record in matches:
            names = defaultdict(set)
            for row in record.get('pins', []):
                name, number = normalize_model_name(row.get('name', '')), str(row.get('number', ''))
                if name and number:
                    names[name].add(number)
            maps.append(names)
        consensus = {}
        for name in set.intersection(*(set(m) for m in maps)):
            numbers = set.union(*(m[name] for m in maps))
            if len(numbers) == 1:
                consensus[name] = next(iter(numbers))
        pins = [p for p in ir.pins if p.component_id == component.internal_id]
        supported = [p for p in pins if normalize_model_name(p.name) in consensus]
        if len({normalize_model_name(p.name) for p in supported}) < 2:
            continue
        conflicts = [p.internal_id for p in supported if _number_is_explicit(p.number)
                     and p.number != consensus[normalize_model_name(p.name)]]
        if conflicts:
            rejected.append({'component_id': component.internal_id, 'reason': 'visible_number_conflict', 'pins': conflicts})
            continue
        used = {p.number for p in pins if _number_is_explicit(p.number)}
        # Repeated names cannot supply an unseen unique number. A one-character
        # OCR confusion is resolvable when all exact-model variants agree on
        # the possibilities and only one unused visible-number candidate fits.
        for pin in pins:
            name=normalize_model_name(pin.name)
            variants=[m.get(name,set()) for m in maps]
            if not _number_is_explicit(pin.number) or not variants or len(variants[0])<2:
                continue
            allowed=variants[0]
            if any(v!=allowed for v in variants[1:]) or pin.number in allowed:
                continue
            remaining=[n for n in allowed-used if len(n)==len(pin.number)
                       and sum(a!=b for a,b in zip(n,pin.number))==1]
            if len(remaining)!=1:
                continue
            before=pin.number;pin.number=remaining[0];used.add(pin.number)
            row={'pin_id':pin.internal_id,'component_id':component.internal_id,'model':component.name,
                 'visible_name':pin.name,'previous_number':before,'number':pin.number,
                 'library_keys':[k for k,_ in matches],'library_sha256':digest,'gt_access':False,
                 'method':'exact-model-and-one-character-visible-number'}
            pin.evidence.append(Evidence(evidence_id='association:library-pin-correction:'+pin.internal_id,
                                         source='rule',metadata=row))
            updates.append(row)
        proposals = defaultdict(list)
        for pin in supported:
            if not _number_is_explicit(pin.number):
                proposals[consensus[normalize_model_name(pin.name)]].append(pin)
        for number, targets in sorted(proposals.items()):
            if number in used or len(targets) != 1:
                continue
            pin = targets[0]
            before = pin.number
            pin.number = number
            row = {'pin_id': pin.internal_id, 'component_id': component.internal_id,
                   'model': component.name, 'visible_name': pin.name, 'previous_number': before,
                   'number': number, 'library_keys': [k for k, _ in matches],
                   'library_sha256': digest, 'gt_access': False}
            pin.evidence.append(Evidence(evidence_id='association:library-pin-number:'+pin.internal_id,
                                         source='rule', metadata={'kind': 'exact-model-visible-pin-consensus', **row}))
            updates.append(row)
    report = {'enabled': True, 'library_sha256': digest, 'updates': updates, 'rejected': rejected,
              'new_pins': 0, 'coordinates_changed': False, 'gt_access': False}
    ir.metadata['library_pin_numbers'] = report
    return report
