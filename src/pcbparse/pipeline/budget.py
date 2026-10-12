from __future__ import annotations

import time
import math
from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    pass


@dataclass(slots=True)
class TimeBudget:
    total_seconds: float
    started_at: float = field(default_factory=time.monotonic)
    deadline_at: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.total_seconds) or self.total_seconds < 0:
            raise ValueError("total_seconds must be finite and non-negative")
        if self.deadline_at is not None and not math.isfinite(self.deadline_at):
            raise ValueError("deadline_at must be finite")

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)

    @property
    def remaining_seconds(self) -> float:
        deadline = self.started_at + self.total_seconds
        if self.deadline_at is not None:
            deadline = min(deadline, self.deadline_at)
        return max(0.0, deadline - time.monotonic())

    def check(self, *, reserve_seconds: float = 0.0, stage: str = "pipeline") -> None:
        if self.remaining_seconds <= 0.0 or self.remaining_seconds < reserve_seconds:
            raise BudgetExceeded(
                f"{stage} time budget exhausted: remaining={self.remaining_seconds:.3f}s, reserve={reserve_seconds:.3f}s"
            )


@dataclass(slots=True)
class TokenBudget:
    max_calls: int
    max_input_tokens: int
    max_output_tokens_per_call: int
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def remaining_calls(self) -> int:
        return max(0, self.max_calls - self.calls)

    @property
    def remaining_input_tokens(self) -> int:
        return max(0, self.max_input_tokens - self.input_tokens)

    def reserve(self, estimated_input_tokens: int, estimated_output_tokens: int) -> None:
        if self.remaining_calls < 1:
            raise BudgetExceeded("Qwen call budget exhausted")
        if estimated_input_tokens > self.remaining_input_tokens:
            raise BudgetExceeded("Qwen input token budget exhausted")
        if estimated_output_tokens > self.max_output_tokens_per_call:
            raise BudgetExceeded("Qwen output token request exceeds per-call limit")
        self.calls += 1
        self.input_tokens += estimated_input_tokens

    def record_output(self, output_tokens: int) -> None:
        if output_tokens < 0:
            raise ValueError("output_tokens must be non-negative")
        self.output_tokens += output_tokens
