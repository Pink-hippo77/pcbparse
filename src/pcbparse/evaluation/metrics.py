from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Hashable, Iterable


@dataclass(frozen=True, slots=True)
class PRF1:
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 1.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 1.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2.0 * self.precision * self.recall / denominator if denominator else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "tp": self.true_positive,
            "fp": self.false_positive,
            "fn": self.false_negative,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def set_prf1(predicted: Iterable[Hashable], expected: Iterable[Hashable]) -> PRF1:
    predicted_set = set(predicted)
    expected_set = set(expected)
    return PRF1(
        true_positive=len(predicted_set & expected_set),
        false_positive=len(predicted_set - expected_set),
        false_negative=len(expected_set - predicted_set),
    )


def connectivity_prf1(
    predicted_nets: Iterable[Iterable[Hashable]],
    expected_nets: Iterable[Iterable[Hashable]],
) -> PRF1:
    return set_prf1(_connectivity_pairs(predicted_nets), _connectivity_pairs(expected_nets))


def _connectivity_pairs(nets: Iterable[Iterable[Hashable]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for net in nets:
        pins = sorted({str(pin) for pin in net})
        pairs.update(combinations(pins, 2))
    return pairs

