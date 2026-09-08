from __future__ import annotations

import math
from dataclasses import dataclass, field

from pcbparse.pipeline.budget import TimeBudget, TokenBudget


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    value: str
    score: float


@dataclass(frozen=True, slots=True)
class AmbiguityTask:
    task_id: str
    task_type: str
    candidates: tuple[Candidate, ...]
    primary_score: float
    graph_inconsistency_score: float = 0.0
    downstream_value_score: float = 0.0
    hardcase_prior: float = 0.0
    crop_path: str | None = None
    context: dict[str, str] = field(default_factory=dict)

    @property
    def candidate_entropy(self) -> float:
        scores = [max(0.0, candidate.score) for candidate in self.candidates]
        total = sum(scores)
        if total <= 0.0 or len(scores) <= 1:
            return 0.0
        probabilities = [score / total for score in scores]
        entropy = -sum(probability * math.log(probability) for probability in probabilities if probability > 0.0)
        return entropy / math.log(len(scores))


@dataclass(frozen=True, slots=True)
class Decision:
    task_id: str
    selected_id: str
    confidence: float
    source: str
    reason: str = ""

    def validate_against(self, task: AmbiguityTask) -> None:
        allowed = {candidate.candidate_id for candidate in task.candidates} | {"UNKNOWN"}
        if self.selected_id not in allowed:
            raise ValueError(f"reasoner created out-of-set candidate {self.selected_id!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("decision confidence must be in [0, 1]")


class ValueAwareRouter:
    def __init__(self, threshold: float = 0.72, min_time_reserve_seconds: float = 25.0) -> None:
        self.threshold = threshold
        self.min_time_reserve_seconds = min_time_reserve_seconds

    def ambiguity_score(self, task: AmbiguityTask) -> float:
        score = (
            0.30 * (1.0 - _clamp01(task.primary_score))
            + 0.25 * _clamp01(task.candidate_entropy)
            + 0.20 * _clamp01(task.graph_inconsistency_score)
            + 0.15 * _clamp01(task.downstream_value_score)
            + 0.10 * _clamp01(task.hardcase_prior)
        )
        return _clamp01(score)

    def should_route(
        self,
        task: AmbiguityTask,
        token_budget: TokenBudget,
        time_budget: TimeBudget,
        *,
        estimated_input_tokens: int,
    ) -> bool:
        return (
            self.ambiguity_score(task) >= self.threshold
            and token_budget.remaining_calls > 0
            and token_budget.remaining_input_tokens >= estimated_input_tokens
            and time_budget.remaining_seconds >= self.min_time_reserve_seconds
            and 1 < len(task.candidates) <= 5
        )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))

