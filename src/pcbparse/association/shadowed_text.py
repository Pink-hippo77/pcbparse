"""Shadowed duplicate text-token reconciliation (opt80 r11a).

The OCR stack can emit two tokens for the same physical string when the text
detector produces both a narrow box and a full box (observed on val30: ``OUT``
and ``OUT16`` at identical coordinates, the narrow box cutting the trailing
digits). The truncated observation then wins net-label routing (no digits ->
netlabel) while the full observation is routed designator — corrupting both net
names and net identity.

``reconcile_shadowed_text_tokens`` merges such pairs before any role assignment:
when one token's box contains (or overlaps >= 0.5 of) another's and their raw
texts stand in a strict prefix relation, the longer observation is canonical and
the shadowed token is dropped. Deterministic; the merge is recorded as evidence
on the survivor and ``raw_text`` is never rewritten.
"""

from __future__ import annotations

from typing import Any

from pcbparse.ir.models import TextToken


def _box(token: TextToken) -> tuple[float, float, float, float]:
    return (token.bbox.x1, token.bbox.y1, token.bbox.x2, token.bbox.y2)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def _contains_with_margin(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
    margin: float = 1.0,
) -> bool:
    return (
        outer[0] - margin <= inner[0]
        and outer[1] - margin <= inner[1]
        and inner[2] <= outer[2] + margin
        and inner[3] <= outer[3] + margin
    )


def _shadows(canonical: TextToken, other: TextToken, margin: float) -> bool:
    """True when ``other`` is a truncated duplicate observation of ``canonical``."""
    a_text = (canonical.raw_text or "").strip()
    b_text = (other.raw_text or "").strip()
    if not a_text or not b_text or a_text == b_text:
        return False
    if not (len(a_text) > len(b_text) and a_text.startswith(b_text)):
        return False
    a_box, b_box = _box(canonical), _box(other)
    return _contains_with_margin(a_box, b_box, margin) or _iou(a_box, b_box) >= 0.5


def reconcile_shadowed_text_tokens(
    texts: list[TextToken],
    *,
    margin_px: float = 1.0,
) -> dict[str, Any]:
    """Drop truncated duplicate tokens in place; return a merge report."""

    ordered = sorted(
        enumerate(texts),
        key=lambda item: (-len((item[1].raw_text or "").strip()), item[1].internal_id),
    )
    canonical: list[TextToken] = []
    dropped: list[str] = []
    merges: dict[str, list[str]] = {}
    for _index, token in ordered:
        shadowed_by = None
        for kept in canonical:
            if _shadows(kept, token, margin_px):
                shadowed_by = kept
                break
        if shadowed_by is not None:
            dropped.append(token.internal_id)
            merges.setdefault(shadowed_by.internal_id, []).append(token.internal_id)
            continue
        canonical.append(token)
    if not dropped:
        return {"dropped": 0, "merges": {}}
    keep_ids = {token.internal_id for token in canonical}
    texts[:] = [token for token in texts if token.internal_id in keep_ids]
    return {
        "dropped": len(dropped),
        "merges": {key: sorted(value) for key, value in sorted(merges.items())},
    }


__all__ = ["reconcile_shadowed_text_tokens"]
