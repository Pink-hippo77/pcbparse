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

import os
import re
from copy import deepcopy
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

from pcbparse.association.reference_prefixes import DESIGNATOR_PATTERN
_DESIGNATOR_RE = re.compile(DESIGNATOR_PATTERN)
_PIN_NUMBER_RE = re.compile(r"^[0-9]{1,3}$")
_ALNUM_PIN_RE = re.compile(r"^(?:[A-Z]{1,2}[0-9]{1,2}|[0-9]{1,2}[A-Z]{1,2})$")
_VALUE_RE = re.compile(
    r"^(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*[kKmMuUnNpPFfHhVvAaWwRrΩµμ%]*$"
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


# ---------------------------------------------------------------------------
# Co-located duplicate / fragment token reconciliation (opt90-ocr, goal C)
# ---------------------------------------------------------------------------
# Multi-view OCR (image tiles + component crops) can emit several observations
# of one physical string: a complete read next to a truncated fragment or a
# degraded misread at the same location.  Such duplicate observations are
# defects -- text-box evaluation matches one token per ground-truth box and a
# degraded observation can steal that slot from the complete read.
#
# ``reconcile_colocated_text_tokens`` merges same-region observations into one
# surviving token (deterministic winner: longer raw text -> higher recognition
# score -> tighter box -> earlier internal_id), drops the degraded observations
# and records the merge as Evidence on the survivor.  Text is never rewritten,
# no string is ever fabricated, and genuinely distinct nearby strings (region
# overlap below the threshold and no containment) are always preserved.
# Flag-gated, default OFF (see ``_reconcile_flag``).
_REGION_OVERLAP_MIN = 0.5
_CONTAIN_MARGIN_PX = 2.0
_EXTENT_RATIO_MIN = 0.3


def _reconcile_bbox(token: TextToken) -> tuple[float, float, float, float]:
    return (token.bbox.x1, token.bbox.y1, token.bbox.x2, token.bbox.y2)


def _overlap_fraction_smaller(a: tuple[float, float, float, float],
                              b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0.0 else 0.0


def _contains_box(outer: tuple[float, float, float, float],
                  inner: tuple[float, float, float, float]) -> bool:
    return (
        outer[0] - _CONTAIN_MARGIN_PX <= inner[0]
        and outer[1] - _CONTAIN_MARGIN_PX <= inner[1]
        and inner[2] <= outer[2] + _CONTAIN_MARGIN_PX
        and inner[3] <= outer[3] + _CONTAIN_MARGIN_PX
    )


def _comparable_extent(a: tuple[float, float, float, float],
                       b: tuple[float, float, float, float]) -> bool:
    """Two observations of one physical string span comparable text extents.

    An over-merged read that swallowed several strings has a much larger box
    than any single-string token inside it; such a pair is NOT a duplicate.
    """
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    larger = max(area_a, area_b)
    if larger <= 0.0:
        return False
    return min(area_a, area_b) / larger >= _EXTENT_RATIO_MIN


def _same_region(a: tuple[float, float, float, float],
                 b: tuple[float, float, float, float]) -> bool:
    if not _comparable_extent(a, b):
        return False
    if _overlap_fraction_smaller(a, b) >= _REGION_OVERLAP_MIN:
        return True
    return _contains_box(a, b) or _contains_box(b, a)


def _recognition_score(token: TextToken) -> float:
    for evidence in token.evidence:
        metadata = getattr(evidence, "metadata", None)
        if isinstance(metadata, dict) and "recognition_score" in metadata:
            try:
                return float(metadata["recognition_score"])
            except (TypeError, ValueError):
                continue
    return float(token.score)


def _observation_quality(token: TextToken) -> tuple[float, int, float]:
    """Deterministic quality of one observation.

    Recognition confidence first (rounded to two decimals as a noise band, so
    short-string score inflation does not decide), then longer raw text (a
    fragment is the shorter observation), then tighter box.
    """
    box = _reconcile_bbox(token)
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return (round(_recognition_score(token), 2), len((token.raw_text or "").strip()), -area)


def _reconcile_flag(duplicate_reconcile: bool | None) -> bool:
    if duplicate_reconcile is not None:
        return bool(duplicate_reconcile)
    value = os.environ.get("PCBPARSE_OCR_DUPLICATE_RECONCILE", "0")
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def reconcile_colocated_text_tokens(
    tokens: list[TextToken],
    *,
    enabled: bool = True,
) -> dict[str, Any]:
    """Merge same-region observations and retain their original payloads in IR evidence."""
    if not enabled:
        return {"enabled": False, "dropped": 0}
    from pcbparse.association.ocr_common import make_rule_evidence, upsert_evidence

    def source_payload(token: TextToken) -> list[dict[str, Any]]:
        return [
            {
                "evidence_id": item.evidence_id,
                "source": item.source,
                "view_id": item.view_id,
                "score": item.score,
                "payload_ref": item.payload_ref,
                "metadata": deepcopy(item.metadata),
            }
            for item in token.evidence
            if item.metadata.get("kind") != "ocr-colocate-reconcile"
        ]

    def snapshot(token: TextToken) -> dict[str, Any]:
        return {
            "token_id": token.internal_id,
            "raw_text": token.raw_text,
            "resolved_text": token.resolved_text,
            "candidates": list(token.candidates),
            "category": token.category,
            "details": token.details,
            "bbox": list(_reconcile_bbox(token)),
            "confidence": token.score,
            "source_object_id": token.source_object_id,
            "component_id": token.component_id,
            "pin_id": token.pin_id,
            "sources": source_payload(token),
        }

    def inherited(token: TextToken) -> list[dict[str, Any]]:
        members: list[dict[str, Any]] = []
        for item in token.evidence:
            metadata = item.metadata
            if metadata.get("kind") != "ocr-colocate-reconcile":
                continue
            observations = metadata.get("observations")
            if isinstance(observations, list):
                members.extend(
                    deepcopy(member)
                    for member in observations
                    if isinstance(member, dict)
                )
                continue
            # Preserve legacy loser history as far as the old evidence allows.
            for old in metadata.get("dropped", []):
                if not isinstance(old, dict):
                    continue
                members.append(
                    {
                        "token_id": old.get("token_id"),
                        "raw_text": old.get("raw_text"),
                        "bbox": deepcopy(old.get("bbox")),
                        "confidence": old.get("confidence"),
                        "sources": deepcopy(old.get("sources", [])),
                    }
                )
        return members

    ordered = sorted(tokens, key=lambda item: item.internal_id)
    original = {item.internal_id: item for item in ordered}
    dropped: dict[str, dict[str, Any]] = {}
    changed = True
    while changed:
        changed = False
        for index, left in enumerate(ordered):
            if left.internal_id in dropped:
                continue
            for right in ordered[index + 1:]:
                # Once left is dropped it cannot absorb further candidates.
                if left.internal_id in dropped:
                    break
                if right.internal_id in dropped:
                    continue
                left_box = _reconcile_bbox(left)
                right_box = _reconcile_bbox(right)
                if not _same_region(left_box, right_box):
                    continue
                if _observation_quality(left) >= _observation_quality(right):
                    winner, loser = left, right
                else:
                    winner, loser = right, left
                dropped[loser.internal_id] = {
                    "token_id": loser.internal_id,
                    "raw_text": loser.raw_text,
                    "merged_into": winner.internal_id,
                    "winner_raw_text": winner.raw_text,
                    "region_overlap": round(
                        _overlap_fraction_smaller(left_box, right_box), 4
                    ),
                }
                changed = True

    def final_owner(token_id: str) -> str:
        seen: set[str] = set()
        while token_id in dropped and token_id not in seen:
            seen.add(token_id)
            token_id = dropped[token_id]["merged_into"]
        return token_id

    survivors = [
        item for item in ordered if item.internal_id not in dropped
    ]
    grouped_ids: dict[str, list[str]] = {}
    for token_id in dropped:
        grouped_ids.setdefault(final_owner(token_id), []).append(token_id)

    members_by_owner: dict[str, list[TextToken]] = {}
    for item in ordered:
        members_by_owner.setdefault(final_owner(item.internal_id), []).append(item)

    for winner in survivors:
        members: dict[str, dict[str, Any]] = {}
        for item in members_by_owner.get(winner.internal_id, []):
            for old in inherited(item):
                member_id = old.get("token_id")
                if isinstance(member_id, str):
                    members[member_id] = old
            members[item.internal_id] = snapshot(item)
        if len(members) <= 1 or not grouped_ids.get(winner.internal_id):
            continue

        dropped_records = []
        for token_id in sorted(grouped_ids[winner.internal_id]):
            item = original.get(token_id)
            if item is None:
                continue
            dropped_records.append(
                {
                    **dropped[token_id],
                    "bbox": list(_reconcile_bbox(item)),
                    "confidence": item.score,
                    "sources": source_payload(item),
                }
            )
        evidence = make_rule_evidence(
            evidence_id=f"ocr-colocate-reconcile:{winner.internal_id}",
            score=winner.score,
            payload_ref=winner.internal_id,
            metadata={
                "kind": "ocr-colocate-reconcile",
                "rule": "same_region_duplicate_observation_v1",
                "observations": [
                    members[member_id] for member_id in sorted(members)
                ],
                "dropped": dropped_records,
            },
        )
        upsert_evidence(winner.evidence, evidence)

    tokens[:] = survivors
    final_ids = {item.internal_id for item in survivors}
    merges = {
        winner_id: sorted(token_ids)
        for winner_id, token_ids in sorted(grouped_ids.items())
        if winner_id in final_ids
    }
    return {
        "enabled": True,
        "dropped": len(dropped),
        "merges": merges,
    }

def postprocess_tokens(
    tokens: list[TextToken],
    *,
    enabled: bool = True,
    duplicate_reconcile: bool | None = None,
) -> dict[str, Any]:
    """Apply confusion-pair and format-grammar corrections to OCR tokens in place.

    Only ``resolved_text`` is updated (when a correction is found).  ``raw_text``
    is never modified.  A rewrite trace is stored in each token's evidence.

    When ``duplicate_reconcile`` is true (or ``None`` and the environment switch
    ``PCBPARSE_OCR_DUPLICATE_RECONCILE`` is on) same-region duplicate/fragment
    observations are merged first via :func:`reconcile_colocated_text_tokens`.
    Default OFF; the off-state is byte-identical to the previous behaviour.

    Returns a summary dict with correction statistics.
    """
    if not enabled:
        return {"enabled": False, "corrected": 0, "total": 0, "by_context": {}}
    reconcile_summary = (
        reconcile_colocated_text_tokens(tokens) if _reconcile_flag(duplicate_reconcile)
        else None
    )
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
    summary: dict[str, Any] = {
        "enabled": True,
        "corrected": corrected_count,
        "total": total,
        "by_context": dict(sorted(by_context.items())),
    }
    if reconcile_summary is not None:
        summary["colocate_reconcile"] = reconcile_summary
    return summary


__all__ = [
    "correct_ocr_text",
    "postprocess_tokens",
    "reconcile_colocated_text_tokens",
]
