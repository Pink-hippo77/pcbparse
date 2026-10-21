"""Dictionary-constrained decoding and confusion-pair snapping for OCR text.

For short technical strings (1-15 chars), free-form OCR output often contains
confusion-pair errors (``Rl``→``R1``, ``O``→``0``, ``CC13``→``C13``).  This
module provides deterministic correction via:

1. **Closed-set dictionary snapping** — when the OCR text is within edit
   distance 1-2 of a known valid entry (designator pattern, net name, pin
   number), snap to the entry.
2. **Grammar-constrained correction** — when the text almost matches a
   grammar pattern (designator, pin number, value), apply confusion-pair
   substitution to make it valid.
3. **Component-aware prefix correction** — when a component category is
   known, use it to resolve prefix confusions (``0R1``→``OR1`` for a resistor
   with prefix ``R``).

Inspired by PCBnet (arXiv:2608.27923) and constrained CTC decoding
(arXiv:2502.09026).  All corrections are deterministic and record rewrite
traces.  ``raw_text`` is never modified.
"""

from __future__ import annotations

import re
from typing import Any

from pcbparse.association.ocr_postprocess import (
    _CONFUSION_TO_DIGIT,
    _CONFUSION_TO_LETTER,
    _DESIGNATOR_PREFIXES,
    _KNOWN_NET_NAMES,
)


# ---------------------------------------------------------------------------
# Closed-set dictionaries
# ---------------------------------------------------------------------------

# Common net / power / signal names in PCB schematics.
NET_DICTIONARY: frozenset[str] = _KNOWN_NET_NAMES | frozenset({
    "VCC", "VDD", "VEE", "VSS", "VREF", "VBAT", "VIN", "VOUT",
    "AVCC", "DVCC", "AGND", "DGND", "PGND", "GND", "GROUND",
    "VBUS", "VDDIO",
    "3V3", "5V", "1V8", "12V", "3.3V", "2.5V", "1.2V",
    "CLK", "SDA", "SCL", "TX", "RX", "MOSI", "MISO", "SCK",
    "RESET", "NRESET", "RST", "NRST", "INT", "NINT",
    "NC", "EP", "D+", "D-", "DP", "DM",
    "AIN", "AOUT", "BIN", "BOUT",
    "EN", "NEN", "CS", "NCS", "RD", "WR", "ALE",
})

# Designator prefixes with their canonical form.
_DESIGNATOR_PREFIX_MAP: dict[str, str] = {
    "R": "R", "C": "C", "L": "L", "D": "D", "Q": "Q",
    "U": "U", "J": "J", "P": "P", "S": "S", "SW": "SW",
    "FB": "FB", "Y": "Y", "X": "X", "TP": "TP", "F": "F",
    "K": "K", "M": "M", "B": "B", "N": "N", "E": "E",
    "H": "H", "A": "A", "G": "G", "V": "V", "O": "O", "I": "I",
}

from pcbparse.association.reference_prefixes import DESIGNATOR_PATTERN
_DESIGNATOR_RE = re.compile(DESIGNATOR_PATTERN)
_PIN_NUMBER_RE = re.compile(r"^[0-9]{1,3}$")
_ALNUM_PIN_RE = re.compile(r"^(?:[A-Z]{1,2}[0-9]{1,2}|[0-9]{1,2}[A-Z]{1,2})$")
_VALUE_RE = re.compile(
    r"^(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*[kKmMuUnNpPFfHhVvAaWwRrΩµ%]*$"
)


def _edit_distance_confusion(a: str, b: str) -> int | None:
    """Compute edit distance, treating confusion-pair substitutions as cost 0.

    Returns the confusion-aware edit distance, or None if the distance
    exceeds 2 (too far to snap safely).
    """
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 2:
        return None
    la, lb = len(a), len(b)
    # DP table with confusion-aware substitution cost.
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            sub_cost = _substitution_cost(a[i - 1], b[j - 1])
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + sub_cost,  # substitution
            )
    result = dp[la][lb]
    return result if result <= 2 else None


def _substitution_cost(a: str, b: str) -> int:
    """Cost 0 for confusion-pair substitutions, 1 for others."""
    if a == b:
        return 0
    if _CONFUSION_TO_DIGIT.get(a) == b or _CONFUSION_TO_DIGIT.get(a.upper()) == b:
        return 0
    if _CONFUSION_TO_LETTER.get(a) == b or _CONFUSION_TO_LETTER.get(a.upper()) == b:
        return 0
    return 1


def snap_to_net(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Snap a near-match to a known net name via confusion-aware edit distance.

    Returns ``(snapped_name, rewrite_records)`` or ``(None, [])`` when no
    safe snap exists (edit distance > 1 for short names, > 2 for long).
    """
    stripped = text.strip().upper()
    if not stripped or stripped in NET_DICTIONARY:
        return None, []
    max_dist = 1 if len(stripped) <= 4 else 2
    best: tuple[int, str] | None = None
    for name in sorted(NET_DICTIONARY):
        dist = _edit_distance_confusion(stripped, name)
        if dist is not None and dist <= max_dist:
            if best is None or dist < best[0]:
                best = (dist, name)
    if best is None:
        return None, []
    dist, name = best
    return name, [{
        "rule": "net_dictionary_snap",
        "original": text,
        "corrected": name,
        "edit_distance": dist,
    }]


def snap_designator_with_category(
    text: str,
    category: str,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Use a component category to correct a designator's prefix.

    When the prefix is incompatible with the category, try confusion-pair
    substitution to find a compatible prefix.  E.g. for a resistor (category
    ``RESISTOR``, expected prefix ``R``): ``0R1`` has prefix ``0`` which maps
    to ``O`` (not ``R``), so no snap.  But ``8R1`` has prefix ``8``→``B`` which
    is also not ``R``, so no snap.  However ``RR8`` is handled by the
    doubled-prefix collapse in ``_canonical_designator``.
    """
    from pcbparse.association.text_component import _prefix_compatible

    stripped = text.strip().upper()
    match = re.fullmatch(r"([A-Za-z])([0-9A-Za-z]*)", stripped)
    if not match:
        return None, []
    prefix, suffix = match.group(1), match.group(2)
    if _prefix_compatible(category, prefix):
        return None, []
    # Try confusion-pair correction on the prefix.
    fixed = _CONFUSION_TO_LETTER.get(prefix)
    if fixed and fixed != prefix and _prefix_compatible(category, fixed):
        return f"{fixed}{suffix}", [{
            "rule": "designator_category_prefix_snap",
            "original": text,
            "corrected": f"{fixed}{suffix}",
            "confused_char": prefix,
            "corrected_char": fixed,
            "category": category,
        }]
    return None, []


def snap_pin_number(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Snap a near-match pin number via confusion-pair correction.

    Handles single-char confusions (``O``→``0``, ``l``→``1``, ``S``→``5``)
    and common multi-char patterns.
    """
    stripped = text.strip()
    if not stripped:
        return None, []
    if _PIN_NUMBER_RE.fullmatch(stripped):
        return None, []
    # Try direct confusion correction.
    chars = list(stripped)
    corrections: list[dict[str, Any]] = []
    for i, ch in enumerate(chars):
        if ch.isdigit():
            continue
        fixed = _CONFUSION_TO_DIGIT.get(ch) or _CONFUSION_TO_DIGIT.get(ch.upper())
        if fixed:
            corrections.append({
                "position": i,
                "original": ch,
                "corrected": fixed,
                "context": "pinnumber_snap",
            })
            chars[i] = fixed
    corrected = "".join(chars)
    if _PIN_NUMBER_RE.fullmatch(corrected) and corrections:
        return corrected, [{
            "rule": "pin_number_confusion_snap",
            "original": text,
            "corrected": corrected,
            "confusion_pairs": corrections,
        }]
    # Try alnum pin correction.
    if _ALNUM_PIN_RE.fullmatch(stripped.upper()):
        return None, []
    chars = list(stripped)
    corrections = []
    for i, ch in enumerate(chars):
        if ch.isalnum():
            continue
        fixed = _CONFUSION_TO_DIGIT.get(ch) or _CONFUSION_TO_DIGIT.get(ch.upper()) or _CONFUSION_TO_LETTER.get(ch)
        if fixed:
            corrections.append({
                "position": i,
                "original": ch,
                "corrected": fixed,
                "context": "alnum_pin_snap",
            })
            chars[i] = fixed
    corrected = "".join(chars)
    if (_ALNUM_PIN_RE.fullmatch(corrected) or _PIN_NUMBER_RE.fullmatch(corrected)) and corrections:
        return corrected, [{
            "rule": "alnum_pin_confusion_snap",
            "original": text,
            "corrected": corrected,
            "confusion_pairs": corrections,
        }]
    return None, []


def constrained_correct(
    text: str,
    *,
    context: str | None = None,
    category: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Apply dictionary-constrained correction to a single OCR string.

    Tries in order:
    1. Category-aware designator prefix snap (if category is known).
    2. Net name dictionary snap (for net-like strings).
    3. Pin number confusion snap (for pin-like strings).
    4. Designator confusion snap (for designator-like strings).

    Returns ``(corrected, rewrite_records)``.  When no correction is applied,
    the original string is returned with an empty list.
    """
    if not text or not text.strip():
        return text, []
    stripped = text.strip()

    # 1. Category-aware designator prefix snap.
    if category:
        snapped, records = snap_designator_with_category(stripped, category)
        if snapped:
            return snapped, records

    # 2. Net name dictionary snap.
    if context in (None, "netlabel", "netlike"):
        snapped, records = snap_to_net(stripped)
        if snapped:
            return snapped, records

    # 3. Pin number confusion snap.
    if context in (None, "pinnumber", "alnum_pin"):
        snapped, records = snap_pin_number(stripped)
        if snapped:
            return snapped, records

    # 4. Designator confusion snap (suffix correction).
    if context in (None, "designator"):
        from pcbparse.association.ocr_postprocess import _validate_designator

        corrected, records = _validate_designator(stripped)
        if records and corrected != stripped:
            return corrected, [{
                "rule": "designator_confusion_snap",
                "original": text,
                "corrected": corrected,
                "confusion_pairs": records,
            }]

    return text, []


__all__ = [
    "NET_DICTIONARY",
    "constrained_correct",
    "snap_designator_with_category",
    "snap_pin_number",
    "snap_to_net",
]
