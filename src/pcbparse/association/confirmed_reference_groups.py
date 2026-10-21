"""Keep overlapping OCR views of an independently confirmed reference together."""
from __future__ import annotations

import math

from pcbparse.association.reference_prefixes import designator_compatible
from pcbparse.ir.models import Evidence

_IC = {'BOX', 'IC', 'BOXIC', 'BOX_IC'}


def reserve_confirmed_reference_groups(ir):
    """Bind only the same visible glyphs, without changing any raw OCR text.

    Local readback already requires two high-confidence recognitions and a
    unique compatible body. A second OCR view of those pixels must not become
    an adjacent component's pin label merely because its ROI overlaps them.
    """
    components = ir.component_index()
    boxes = [c.body_bbox or c.bbox for c in ir.components if c.category.upper() in _IC]
    anchors = []
    for token in ir.texts:
        owner = components.get(token.component_id)
        if (owner is None or not math.isfinite(token.score) or token.score < .98
                or any(e.source == 'dataset' for e in token.evidence)
                or token.raw_text.strip() != owner.key
                or not designator_compatible(owner.category, owner.key)):
            continue
        if not any(e.source == 'ocr' and e.metadata.get('gt_access') is False
                   and e.metadata.get('method') == 'local-color-reference-readback-v1'
                   and e.metadata.get('observed_reference') == owner.key
                   and e.metadata.get('component_id') == owner.internal_id
                   and e.metadata.get('confirmation_score', 0) >= .98
                   for e in token.evidence):
            continue
        if any(b.x1 <= token.bbox.center.x <= b.x2 and b.y1 <= token.bbox.center.y <= b.y2 for b in boxes):
            continue
        anchors.append((owner, token))
    updates = []
    for token in ir.texts:
        if any(e.source == 'dataset' for e in token.evidence):
            continue
        if any(b.x1 <= token.bbox.center.x <= b.x2 and b.y1 <= token.bbox.center.y <= b.y2 for b in boxes):
            continue
        candidates = []
        for owner, anchor in anchors:
            if token.raw_text.strip() != anchor.raw_text.strip():
                continue
            a, b = anchor.bbox, token.bbox
            if min(a.width, a.height, b.width, b.height) <= 0 or not .5 <= b.height / a.height <= 2:
                continue
            overlap = max(0., min(a.x2, b.x2) - max(a.x1, b.x1)) * max(0., min(a.y2, b.y2) - max(a.y1, b.y1))
            # Both views must cover substantially the same glyphs. Containment
            # by a much larger text region does not establish that identity.
            if (overlap / min(a.width * a.height, b.width * b.height) >= .8
                    and overlap / max(a.width * a.height, b.width * b.height) >= .6):
                candidates.append((owner, anchor))
        if len({owner.internal_id for owner, _ in candidates}) != 1:
            continue
        owner, anchor = min(candidates, key=lambda row: row[1].internal_id)
        if (token.details == token.category == 'designator' and token.component_id == owner.internal_id
                and token.pin_id is None):
            continue
        row = {'token_id': token.internal_id, 'anchor_token_id': anchor.internal_id,
               'component_id': owner.internal_id, 'reference': owner.key,
               'previous_component_id': token.component_id, 'previous_pin_id': token.pin_id,
               'previous_details': token.details, 'gt_access': False}
        token.component_id, token.pin_id = owner.internal_id, None
        token.details = token.category = 'designator'
        token.evidence.append(Evidence(evidence_id='confirmed-reference-group:' + token.internal_id,
            source='geometry', metadata={'kind': 'same-glyphs-as-confirmed-reference', **row}))
        updates.append(row)
    report = {'updates': updates, 'anchors': len(anchors), 'raw_text_changed': False,
              'coordinates_changed': False, 'gt_access': False}
    ir.metadata['confirmed_reference_groups'] = report
    return report
