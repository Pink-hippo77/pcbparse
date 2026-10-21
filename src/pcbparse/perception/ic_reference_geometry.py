"""Read unusual box designators only beside a measured multi-terminal body.

The prefix is not a component classifier. A closed outline, several observed
strokes and the caller's independent pin-text checks authorize box recovery.
Ordinary two-terminal LEDs/resistors never qualify through this helper.
"""
from __future__ import annotations

import re

from pcbparse.association.reference_prefixes import BOX_REFERENCE_PATTERN, DESIGNATOR_PATTERN
from pcbparse.association.visible_references import _box_reference_gap

_DESIGNATOR = re.compile(DESIGNATOR_PATTERN)
_BOX_DESIGNATOR = re.compile(BOX_REFERENCE_PATTERN)
_ARRAY_DESIGNATOR = re.compile(r'(?:LED|RP|RN|RA|LCD|DISP|DS)\d+[A-Z]?$')
_PIN_ROLES = {'pinname', 'pinnumber', 'pin_name', 'pin_number', 'netlabel', 'netname'}


def shape_supported_reference(ir, body, strokes):
    """Return one independent exterior reference; abstain on competing text."""
    if (len(strokes) < 4
            or len({(side, position) for side, position, _ in strokes}) != len(strokes)):
        return None
    return exterior_reference_candidate(ir, body)


def exterior_reference_candidate(ir, body):
    """Cheap text screen; this alone never authorizes a component or pin."""
    if (body.width < 12 or body.height < 12
            or body.width*body.height > .5*ir.image.width*ir.image.height):
        return None
    choices = []
    for token in ir.texts:
        value, text = token.raw_text.strip(), token.bbox
        if (token.score < .95 or not _DESIGNATOR.fullmatch(value)
                or token.pin_id is not None or token.details.casefold() in _PIN_ROLES
                or token.category.casefold() in _PIN_ROLES):
            continue
        known_box = _BOX_DESIGNATOR.fullmatch(value) or _ARRAY_DESIGNATOR.fullmatch(value)
        if not known_box and not any(c.key == value for c in ir.components):
            # An arbitrary part number such as PW7052 is not evidence of a
            # designator. Existing independent ownership may support other
            # spellings; a prefix alone cannot authorize a box.
            continue
        # A pin lane just inside the top/bottom corner is not a title lane.
        exterior = (text.y2 <= body.y1+1 or text.y1 >= body.y2-1
                    or text.x2 <= body.x1+1 or text.x1 >= body.x2-1)
        if not exterior or _box_reference_gap(body, text, lateral=True) is None:
            continue
        # Do not steal a reference already belonging to a neighboring symbol.
        conflicting = False
        for component in ir.components:
            other = component.body_bbox or component.bbox
            overlap = (max(0., min(body.x2, other.x2)-max(body.x1, other.x1))
                       * max(0., min(body.y2, other.y2)-max(body.y1, other.y1)))
            same_body = overlap/max(1., min(body.width*body.height, other.width*other.height)) >= .5
            if component.key and component.key != value and same_body:
                conflicting = True
                break
            if (component.key == value or component.internal_id == token.component_id) and not same_body:
                conflicting = True
                break
        # A shorter crop of a complete model/title is not a separate printed
        # designator. Keep its full OCR observation as a competing explanation.
        for other in ir.texts:
            full = other.raw_text.strip()
            if (other is token or len(full) <= len(value) or value not in full
                    or other.score < max(.95, token.score-.03)):
                continue
            b = other.bbox
            overlap = (max(0., min(text.x2,b.x2)-max(text.x1,b.x1))
                       * max(0., min(text.y2,b.y2)-max(text.y1,b.y1)))
            if overlap/max(1.,text.width*text.height) >= .8:
                conflicting = True
                break
        if not conflicting:
            choices.append(token)
    if len({t.raw_text.strip() for t in choices}) != 1:
        return None
    return max(choices, key=lambda t: (t.score, t.internal_id))
