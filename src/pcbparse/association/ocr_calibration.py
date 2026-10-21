"""Word-level confidence calibration for OCR recognition scores.

Addresses the length-dependent overconfidence of STR models (Slossberg et al.,
arXiv:2012.12643): mean-of-max scores are systematically biased by sequence
length, making short strings appear overconfident.  Provides:

1. **Length-binned Platt scaling** — fits a logistic calibrator per length bin
   to map raw scores to P(entire string correct).
2. **Deterministic heuristic calibration** — for zero-shot use without a
   calibration set, applies known bias corrections (length penalty, charset
   consistency bonus, structural validity bonus).

All calibration is deterministic and does not modify ``raw_text``.
"""

from __future__ import annotations

import math
import re
from typing import Any, Sequence


# Length bins: (min_len, max_len, name)
_LENGTH_BINS: list[tuple[int, int, str]] = [
    (1, 1, "single"),
    (2, 3, "short"),
    (4, 8, "medium"),
    (9, 999, "long"),
]

# Known bias corrections per length bin (deterministic heuristic).
# Short strings (1-3 chars) are overconfident; long strings are underconfident.
_LENGTH_BIAS: dict[str, float] = {
    "single": -0.08,
    "short": -0.05,
    "medium": 0.0,
    "long": 0.03,
}

# Charset consistency bonus: if all chars are in the expected charset for
# the detected context, boost confidence slightly.
_CHARSET_BONUS: float = 0.03

# Structural validity bonus: if the text matches a known grammar pattern
# (designator, pin number, value), boost confidence.
_STRUCTURE_BONUS: float = 0.05

from pcbparse.association.reference_prefixes import DESIGNATOR_PATTERN
_DESIGNATOR_RE = re.compile(DESIGNATOR_PATTERN)
_PIN_NUMBER_RE = re.compile(r"^[0-9]{1,3}$")
_ALNUM_PIN_RE = re.compile(r"^(?:[A-Z]{1,2}[0-9]{1,2}|[0-9]{1,2}[A-Z]{1,2})$")
_VALUE_RE = re.compile(
    r"^(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*[kKmMuUnNpPFfHhVvAaWwRrΩµ%]*$"
)


def _length_bin(text: str) -> str:
    length = len(text.strip())
    for lo, hi, name in _LENGTH_BINS:
        if lo <= length <= hi:
            return name
    return "long"


def _has_structure(text: str) -> bool:
    stripped = text.strip()
    return bool(
        _DESIGNATOR_RE.fullmatch(stripped)
        or _PIN_NUMBER_RE.fullmatch(stripped)
        or _ALNUM_PIN_RE.fullmatch(stripped)
        or _VALUE_RE.fullmatch(stripped)
    )


def calibrate_score(
    raw_score: float,
    text: str,
    *,
    charset: frozenset[str] | None = None,
) -> float:
    """Calibrate a recognition score using deterministic heuristic corrections.

    ``raw_score`` is the backend's own confidence (e.g. mean-of-max from CTC).
    Returns a calibrated score in [0, 1] that better estimates P(string correct).

    When a calibration set is available, use ``PlattCalibrator`` instead.
    """
    if not text or not text.strip():
        return 0.0
    score = max(0.0, min(1.0, float(raw_score)))
    bin_name = _length_bin(text)
    # Length bias correction.
    score += _LENGTH_BIAS.get(bin_name, 0.0)
    # Charset consistency bonus.
    if charset is not None:
        stripped = text.strip()
        if stripped and all(ch in charset for ch in stripped):
            score += _CHARSET_BONUS
    # Structural validity bonus.
    if _has_structure(text):
        score += _STRUCTURE_BONUS
    return max(0.0, min(1.0, score))


class PlattCalibrator:
    """Logistic (Platt) calibration per length bin.

    Fits ``P(correct | score) = 1 / (1 + exp(A * score + B))`` on a held-out
    set of ``(score, is_correct)`` pairs.  Each length bin gets its own
    (A, B) parameters.

    For zero-shot use without calibration data, fall back to
    :func:`calibrate_score`.
    """

    def __init__(self) -> None:
        self._params: dict[str, tuple[float, float]] = {}
        self._fitted = False

    def fit(
        self,
        scores: Sequence[float],
        labels: Sequence[bool],
        texts: Sequence[str],
        *,
        max_iterations: int = 500,
        learning_rate: float = 0.1,
    ) -> None:
        """Fit per-length-bin Platt scaling via gradient descent on NLL."""
        if len(scores) != len(labels) or len(scores) != len(texts):
            raise ValueError("scores, labels, texts must have equal length")
        by_bin: dict[str, list[tuple[float, float]]] = {}
        for score, label, text in zip(scores, labels, texts, strict=True):
            bin_name = _length_bin(text)
            by_bin.setdefault(bin_name, []).append((float(score), 1.0 if label else 0.0))
        for bin_name, pairs in by_bin.items():
            if len(pairs) < 5:
                continue  # insufficient data for this bin
            self._params[bin_name] = self._fit_one_bin(
                pairs, max_iterations, learning_rate
            )
        self._fitted = True

    def _fit_one_bin(
        self,
        pairs: list[tuple[float, float]],
        max_iterations: int,
        learning_rate: float,
    ) -> tuple[float, float]:
        a, b = 0.0, 0.0  # neutral start
        n = len(pairs)
        for _ in range(max_iterations):
            grad_a, grad_b = 0.0, 0.0
            for score, label in pairs:
                z = a * score + b
                p = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))
                err = p - label
                grad_a += err * score
                grad_b += err
            a -= learning_rate * grad_a / n
            b -= learning_rate * grad_b / n
        return (a, b)

    def calibrate(self, raw_score: float, text: str) -> float:
        """Apply fitted Platt scaling; falls back to heuristic if unfitted."""
        if not self._fitted:
            return calibrate_score(raw_score, text)
        bin_name = _length_bin(text)
        params = self._params.get(bin_name)
        if params is None:
            return calibrate_score(raw_score, text)
        a, b = params
        z = a * max(0.0, min(1.0, float(raw_score))) + b
        return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "platt_calibrator",
            "fitted": self._fitted,
            "params": {
                name: {"A": round(a, 6), "B": round(b, 6)}
                for name, (a, b) in sorted(self._params.items())
            },
        }


__all__ = [
    "PlattCalibrator",
    "calibrate_score",
]
