"""Use paired visible labels to distinguish pin functions from body fields.

This module supplies classification evidence only. It never writes owners,
reference keys, pin numbers, text spellings or geometry.
"""
from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Sequence

from pcbparse.association.ocr_common import point_bbox_distance, preferred_text
from pcbparse.association.text_pin import _pin_side
from pcbparse.ir.models import Component, Pin, TextToken


_BODY_CATEGORIES = frozenset({'BOX', 'IC', 'U', 'REGULATOR', 'VREG', 'LDO'})
_REFERENCE_PREFIXES = frozenset({
    'R', 'C', 'L', 'D', 'Q', 'U', 'J', 'SW', 'FB', 'Y', 'X', 'TP', 'F',
    'K', 'RN', 'LED', 'JP', 'CN', 'CON',
})
_NAME = re.compile(r'[A-Za-z0-9_+./() \-]+')
_REFERENCE = re.compile(r'([A-Z]+)[0-9]+[A-Z]?')
_NUMBER = re.compile(r'[0-9]{1,3}')


class BodyPinLabelContext:
    """One-image index of the visible number/name pairs on component edges."""

    def __init__(self, components: Sequence[Component], pins: Sequence[Pin],
                 texts: Sequence[TextToken], maximum_distance: float) -> None:
        self.components = {component.internal_id: component for component in components}
        self.pins = tuple(sorted(pins, key=lambda pin: pin.internal_id))
        self.maximum_distance = maximum_distance
        self.numbers = tuple(
            token for token in sorted(texts, key=lambda token: token.internal_id)
            if _NUMBER.fullmatch(unicodedata.normalize('NFKC', preferred_text(token) or ''))
        )

    def hint(self, token: TextToken, value: str) -> dict[str, Any] | None:
        if not value or len(value) > 32 or value.isdecimal():
            return None
        if not _NAME.fullmatch(value) or not any(character.isalpha() for character in value):
            return None
        # Long numeric part identifiers must retain their original path.
        if re.search(r'[0-9]{4,}', value):
            return None
        reference = _REFERENCE.fullmatch(value)
        if reference and reference.group(1) in _REFERENCE_PREFIXES:
            return None
        box = token.bbox
        center = box.center
        text_scale = max(2.0, min(box.width, box.height) * .5)
        choices = []
        for pin in self.pins:
            component = self.components.get(pin.component_id)
            if component is None or component.category.upper() not in _BODY_CATEGORIES:
                continue
            body = component.bbox
            if not (body.x1 <= box.x1 <= box.x2 <= body.x2
                    and body.y1 <= box.y1 <= box.y2 <= body.y2):
                continue
            side = _pin_side(pin)
            if side == 'UNKNOWN':
                continue
            horizontal = side in {'L', 'R'}
            axis = pin.point.y if horizontal else pin.point.x
            low, high = (box.y1, box.y2) if horizontal else (box.x1, box.x2)
            gap = max(low - axis, axis - high, 0.0)
            if gap > text_scale:
                continue
            inward = {'L': center.x - pin.point.x, 'R': pin.point.x - center.x,
                      'T': center.y - pin.point.y, 'B': pin.point.y - center.y}[side]
            if inward < 0:
                continue
            # The component title occupies a different region from edge labels.
            if (side == 'L' and center.x > body.center.x
                    or side == 'R' and center.x < body.center.x
                    or side == 'T' and center.y > body.center.y
                    or side == 'B' and center.y < body.center.y):
                continue
            distance = self._distance(pin, box)
            if distance > self.maximum_distance:
                continue
            edge_gap = {'L': abs(pin.point.x - body.x1), 'R': abs(pin.point.x - body.x2),
                        'T': abs(pin.point.y - body.y1), 'B': abs(pin.point.y - body.y2)}[side]
            if edge_gap > min(36.0, max(12.0, min(body.width, body.height) * .1)):
                continue
            for number in self.numbers:
                if number.internal_id == token.internal_id:
                    continue
                if number.component_id and number.component_id != component.internal_id:
                    continue
                peer_scale = max(2.0, min(number.bbox.width, number.bbox.height) * .5)
                peer_low, peer_high = ((number.bbox.y1, number.bbox.y2) if horizontal
                                      else (number.bbox.x1, number.bbox.x2))
                peer_gap = max(peer_low - axis, axis - peer_high, 0.0)
                if peer_gap > peer_scale:
                    continue
                name_axis = center.y if horizontal else center.x
                number_axis = number.bbox.center.y if horizontal else number.bbox.center.x
                if abs(name_axis - number_axis) > text_scale + peer_scale + 2.0:
                    continue
                peer_distance = self._distance(pin, number.bbox)
                if peer_distance > self.maximum_distance:
                    continue
                choices.append((gap + peer_gap, distance + peer_distance, component.internal_id,
                                pin.internal_id, number.internal_id))
        if not choices:
            return None
        chosen = min(choices)
        return {'component_id': chosen[2], 'pin_id': chosen[3], 'number_token_id': chosen[4],
                'lane_gap': chosen[0], 'pair_distance': chosen[1]}

    @staticmethod
    def _distance(pin: Pin, box) -> float:
        return min(point_bbox_distance(pin.point, box),
                   point_bbox_distance(pin.base, box) if pin.base else math.inf)


__all__ = ['BodyPinLabelContext']
