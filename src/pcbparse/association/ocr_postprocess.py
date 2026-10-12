"""Deterministic OCR text post-processing for engineering schematics.

Implements two correction strategies inspired by PCBnet (arXiv:2608.27923):

1. **Confusion-pair correction** — common OCR character substitutions
   (``1/l/I``, ``O/0``, ``8/B``, ``5/S``, ``Z/2``) are resolved using
   format-grammar context (designator vs pin number vs value).

2. **Format grammar constraints** — recognized text is validated against
   expected patterns for designators, pin numbers, values, and net labels.
   Invalid forms are corrected when a unique legal alternative exists via
   confusion-pair substitution; otherwise the original text is preserved.

All corrections are deterministic and record a rewrite trace so the original
spelling stays recoverable in ``Evidence`` metadata.  ``raw_text`` is never
modified.
"""

from __future__ import annotations

import re
from typing import Any

from pcbparse.ir.models import TextToken


# ---------------------------------------------------------------------------
# Confusion-pair tables (domain-specific, PCBnet-inspired)
# ---------------------------------------------------------------------------
# Each entry maps a wrong character to its likely intended character in a
# given context.  Contexts: "digit" (pin numbers), "letter" (designator
# prefixes), "any" (general).
_CONFUSION_TO_DIGIT: dict[str, str] = {
    "O": "0", "o": "0",
    "D": "0",
    "I": "1", "l": "1", "|": "1", "!": "1",
    "Z": "2", "z": "2",
    "S": "5", "s": "5",
    "G": "6", "b": "6",
    "T": "7",
    "B": "8",
    "g": "9", "q": "9",
}

_CONFUSION_TO_LETTER: dict[str, str] = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
    "|": "I",
    "!": "I",
}

# Designator prefixes valid in PCB schematics.
_DESIGNATOR_PREFIXES = frozenset(
    "RCLDQUJTPSWFBYXMNHEKAGVOI"
)

# Value unit suffixes (case-insensitive match).
_VALUE_UNITS = re.compile(
    r"(?:[kKmMuUnNpP]?[FfHhΩohm%]|[kKmMuUnNpP]|[Vv]|[Aa]|[Ww]|[Hh]|[Rr]|[Ff])$"
)

# Net label / power patterns.
_KNOWN_NET_NAMES = frozenset({
    "VCC", "VDD", "VEE", "VBAT", "VIN", "VOUT", "AVCC", "DVCC",
    "GND", "GROUND", "AGND", "DGND", "PGND", "VSS", "VREF",
    "5V", "3.3V", "3V3", "12V", "VBUS", "VDDIO",
    "CLK", "SDA", "SCL", "TX", "RX", "RESET", "NRESET", "NC", "EP",
})

_DESIGNATOR_RE = re.compile(r"^[A-Z]{1,4}[0-9]+[A-Z]?$")
_PIN_NUMBER_RE = re.compile(r"^[0-9]{1,3}$")
_ALNUM_PIN_RE = re.compile(r"^(?:[A-Z]{1,2}[0-9]{1,2}|[0-9]{1,2}[A-Z]{1,2})$")
_VALUE_RE = re.compile(
    r"^(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*[kKmMuUnNpPFfHhVvAaWwRrΩµ%]*$"
)
_NETLIKE_RE = re.compile(r"^[A-Za-z0-9_+.\-]+$")


def _classify_context(text: str) -> str:
    """Return the most likely field context for a raw OCR string."""
    stripped = text.strip()
    if not stripped:
        return "unknown"
    if _PIN_NUMBER_RE.fullmatch(stripped):
        return "pinnumber"
    if _DESIGNATOR_RE.fullmatch(stripped):
        return "designator"
    if _ALNUM_PIN_RE.fullmatch(stripped):
        return "alnum_pin"
    if _VALUE_RE.fullmatch(stripped):
        return "value"
    if stripped.upper() in _KNOWN_NET_NAMES:
        return "netlabel"
    # Single characters that are common digit confusions are likely pin numbers.
    if len(stripped) == 1 and stripped in _CONFUSION_TO_DIGIT:
        return "pinnumber"
    if _NETLIKE_RE.fullmatch(stripped) and len(stripped) >= 2:
        return "netlike"
    return "unknown"


def _correct_confusions(text: str, context: str) -> tuple[str, list[dict[str, Any]]]:
    """Apply confusion-pair corrections based on the detected context.

    Returns ``(corrected, rewrite_records)``.  When no correction is applied
    the original string is returned unchanged.  Net labels and net-like strings
    are never corrected — they are alphabetic identifiers where letter→digit
    substitutions would be wrong.
    """
    if context in ("unknown", "netlabel", "netlike"):
        return text, []
    corrections: list[dict[str, Any]] = []
    chars = list(text)
    for i, ch in enumerate(chars):
        replacement = None
        if context in ("pinnumber", "alnum_pin"):
            replacement = _CONFUSION_TO_DIGIT.get(ch)
            # For alnum pins, only fix the digit part (positions after first letter run).
            if context == "alnum_pin" and replacement:
                # Only convert if this position is expected to be a digit.
                prefix_len = 0
                for j, c in enumerate(text):
                    if c.isalpha():
                        prefix_len = j + 1
                    else:
                        break
                if i < prefix_len:
                    replacement = None
        elif context == "designator":
            # Fix the numeric suffix (after the letter prefix).
            if ch.isdigit() or ch in _CONFUSION_TO_DIGIT:
                replacement = _CONFUSION_TO_DIGIT.get(ch)
            elif ch in _CONFUSION_TO_LETTER and i == 0:
                # First char should be a letter in designators.
                replacement = _CONFUSION_TO_LETTER.get(ch)
        elif context in ("value", "netlabel", "netlike"):
            # For values, digits should be digits.
            if ch in _CONFUSION_TO_DIGIT and not ch.isdigit():
                replacement = _CONFUSION_TO_DIGIT.get(ch)
        if replacement and replacement != ch:
            corrections.append({
                "position": i,
                "original": ch,
                "corrected": replacement,
                "context": context,
            })
            chars[i] = replacement
    if not corrections:
        return text, []
    return "".join(chars), corrections


def _validate_designator(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Validate and correct a designator string.

    Applies confusion-pair correction to the numeric suffix and validates
    the prefix is a known designator letter.  Handles cases like ``RI``→``R1``
    where a letter in the suffix is actually a confused digit.
    """
    stripped = text.strip().upper()
    if _DESIGNATOR_RE.fullmatch(stripped):
        # A complete letter prefix plus numeric suffix is already legal.
        # Converting letters within GAIN1/VDD1/PB10, or the suffix of U5B,
        # destroys valid visible text. Semantic context resolves their field.
        return stripped, []
    match = re.fullmatch(r"([A-Za-z])([0-9A-Za-z]*)", stripped)
    if not match:
        return text, []
    prefix, suffix = match.group(1), match.group(2)
    if not suffix:
        return stripped, []
    corrections: list[dict[str, Any]] = []
    # Correct prefix: first char must be a valid designator letter.
    prefix_chars = list(prefix)
    if prefix_chars and prefix_chars[0] not in _DESIGNATOR_PREFIXES:
        fixed = _CONFUSION_TO_LETTER.get(prefix_chars[0])
        if fixed and fixed in _DESIGNATOR_PREFIXES:
            corrections.append({
                "position": 0,
                "original": prefix_chars[0],
                "corrected": fixed,
                "context": "designator_prefix",
            })
            prefix_chars[0] = fixed
    # Correct suffix: digits (and optional trailing letter).  Confused letters
    # in the suffix are corrected to digits.
    suffix_chars = list(suffix)
    for i, ch in enumerate(suffix_chars):
        if ch.isdigit():
            continue
        if i == len(suffix_chars) - 1 and ch.isalpha() and ch not in _CONFUSION_TO_DIGIT:
            continue  # trailing letter is valid (e.g. U5A) unless it's a confusion
        fixed = _CONFUSION_TO_DIGIT.get(ch)
        if fixed:
            corrections.append({
                "position": len(prefix) + i,
                "original": ch,
                "corrected": fixed,
                "context": "designator_suffix",
            })
            suffix_chars[i] = fixed
    corrected = "".join(prefix_chars) + "".join(suffix_chars)
    if corrected != stripped:
        return corrected, corrections
    return stripped, []


def _validate_pin_number(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Validate and correct a pin number string (pure digits or alnum)."""
    stripped = text.strip().upper()
    if _PIN_NUMBER_RE.fullmatch(stripped):
        return stripped, []
    # Try confusion correction to digits (check both cases).
    original = text.strip()
    chars = list(original)
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
                "context": "pinnumber",
            })
            chars[i] = fixed
    corrected = "".join(chars)
    if _PIN_NUMBER_RE.fullmatch(corrected):
        return corrected, corrections
    # Alphanumeric pin: letter+digit or digit+letter.
    if _ALNUM_PIN_RE.fullmatch(stripped):
        return stripped, []
    # Try alnum correction.
    chars = list(original)
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
                "context": "alnum_pin",
            })
            chars[i] = fixed
    corrected = "".join(chars)
    if _ALNUM_PIN_RE.fullmatch(corrected) or _PIN_NUMBER_RE.fullmatch(corrected):
        return corrected, corrections
    return stripped, []


def _validate_value(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Validate and correct a component value string."""
    stripped = text.strip()
    if _VALUE_RE.fullmatch(stripped):
        return stripped, []
    # Try confusion correction: characters that map to digits (O/0, I/1, etc.)
    # are corrected.  Only accept the result if it forms a valid value.
    chars = list(stripped)
    corrections: list[dict[str, Any]] = []
    for i, ch in enumerate(chars):
        if ch.isdigit() or ch in ".,":
            continue
        fixed = _CONFUSION_TO_DIGIT.get(ch) or _CONFUSION_TO_DIGIT.get(ch.upper())
        if fixed and fixed != ch:
            corrections.append({
                "position": i,
                "original": ch,
                "corrected": fixed,
                "context": "value",
            })
            chars[i] = fixed
    corrected = "".join(chars)
    if corrections and _VALUE_RE.fullmatch(corrected):
        return corrected, corrections
    return stripped, []


def correct_ocr_text(
    text: str,
    *,
    category_hint: str | None = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Correct a single OCR string using confusion pairs and format grammar.

    ``category_hint`` is an optional semantic field hint (``"designator"``,
    ``"pinnumber"``, ``"value"``, ``"netlabel"``) from the classifier.  When
    provided, it overrides automatic context detection.

    Returns ``(corrected_text, detected_context, rewrite_records)``.
    """
    if not text or not text.strip():
        return text, "unknown", []
    context = category_hint or _classify_context(text)
    # Net labels and unknown strings are never corrected.
    if context in ("netlabel", "netlike", "unknown"):
        return text, context, []
    # Apply format-specific validation first (stronger than generic confusion).
    if context == "designator":
        corrected, records = _validate_designator(text)
    elif context in ("pinnumber", "alnum_pin"):
        corrected, records = _validate_pin_number(text)
    elif context == "value":
        corrected, records = _validate_value(text)
    else:
        corrected, records = _correct_confusions(text, context)
    # Dictionary-constrained correction: snap to known valid entries.
    if not records:
        from pcbparse.association.ocr_dictionary import constrained_correct

        corrected2, records2 = constrained_correct(text, context=context)
        if records2 and corrected2 != text:
            return corrected2, context, records2
    return corrected, context, records


def postprocess_tokens(
    tokens: list[TextToken],
    *,
    enabled: bool = True,
) -> dict[str, Any]:
    """Apply confusion-pair and format-grammar corrections to OCR tokens in place.

    Only ``resolved_text`` is updated (when a correction is found).  ``raw_text``
    is never modified.  A rewrite trace is stored in each token's evidence.

    Returns a summary dict with correction statistics.
    """
    if not enabled:
        return {"enabled": False, "corrected": 0, "total": 0, "by_context": {}}
    corrected_count = 0
    total = 0
    by_context: dict[str, int] = {}
    for token in tokens:
        raw = token.raw_text
        if not raw or not raw.strip():
            continue
        total += 1
        hint = token.details if token.details else None
        if hint not in ("designator", "pinnumber", "value", "netlabel"):
            hint = None
        corrected, context, records = correct_ocr_text(raw, category_hint=hint)
        if records and corrected != raw:
            token.resolved_text = corrected
            corrected_count += 1
            by_context[context] = by_context.get(context, 0) + 1
            from pcbparse.association.ocr_common import make_rule_evidence, upsert_evidence

            evidence = make_rule_evidence(
                evidence_id=f"ocr-postprocess:{token.internal_id}",
                score=token.score,
                payload_ref=token.internal_id,
                metadata={
                    "kind": "ocr-text-correction",
                    "original_text": raw,
                    "corrected_text": corrected,
                    "context": context,
                    "confusion_pairs": records,
                    "rule": "confusion_pair_format_grammar",
                },
            )
            upsert_evidence(token.evidence, evidence)
    return {
        "enabled": True,
        "corrected": corrected_count,
        "total": total,
        "by_context": dict(sorted(by_context.items())),
    }


__all__ = [
    "correct_ocr_text",
    "postprocess_tokens",
]
