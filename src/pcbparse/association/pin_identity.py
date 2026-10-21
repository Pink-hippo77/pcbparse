from __future__ import annotations

from collections import defaultdict
from typing import Any

from pcbparse.ir.models import CircuitIR, Evidence
from pcbparse.output.official_schema import load_official_schema, split_official_pin_numbers


# Names such as 1A, 2Y, /RESET and ~OE are printed pin labels too. A bare
# number is not sufficient evidence of a name in a separate interior lane.
PIN_NAME_PATTERN = r'(?=.{1,40}$)(?=.*[A-Za-z])[A-Za-z0-9_+./()~!#\[\]\- ]+'


def assign_unscored_pin_numbers(ir: CircuitIR) -> dict[str, Any]:
    """Give detected non-Box pins stable output IDs, as permitted by Q&A 1013.

    These are serialization identities, not recovered OCR labels. Box pin
    numbers remain evidence-dependent. No geometry, ownership or existing
    number is changed, and no additional pins are created.
    """
    schema = load_official_schema()["component"]
    aliases = schema["type_aliases"]
    allowed = set(schema["allowed_types"]) - {"box"}
    components = ir.component_index()
    groups: dict[str, list[Any]] = defaultdict(list)
    for pin in ir.pins:
        owner = components.get(pin.component_id)
        if owner is None or not isinstance(owner.key, str) or not owner.key.strip():
            continue
        if aliases.get(owner.category, owner.category) in allowed:
            groups[owner.internal_id].append(pin)

    assignments: dict[str, str] = {}
    for component_id in sorted(groups):
        pins = groups[component_id]
        used = {number for pin in pins for number in split_official_pin_numbers(pin.number)}
        next_number = 1
        for pin in sorted(pins, key=lambda item: (item.point.x, item.point.y, item.internal_id)):
            if split_official_pin_numbers(pin.number):
                continue
            while str(next_number) in used:
                next_number += 1
            number = str(next_number)
            previous = pin.number
            pin.number = number
            used.add(number)
            assignments[pin.internal_id] = number
            pin.evidence.append(Evidence(
                evidence_id=f"association:unscored-pin-number:{pin.internal_id}",
                source="rule",
                payload_ref=pin.internal_id,
                metadata={
                    "kind": "unscored-pin-output-identity",
                    "previous_number": previous,
                    "output_number": number,
                    "ocr_recovered": False,
                    "qna_reference": "1013: non-Box pin number/name are not scored",
                },
            ))

    # Count from evidence so repeated calls stay idempotent and auditable.
    assigned = {
        pin.internal_id: pin.number
        for pin in ir.pins
        if any(e.metadata.get("kind") == "unscored-pin-output-identity" for e in pin.evidence)
    }
    report = {"assigned_count": len(assigned), "assignments": dict(sorted(assigned.items())),
              "new_assignments": len(assignments), "box_numbers_inferred": 0}
    ir.metadata["unscored_pin_numbering"] = report
    return report
