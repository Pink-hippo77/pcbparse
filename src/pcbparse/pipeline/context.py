from __future__ import annotations

import hashlib
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

from pcbparse.config import config_hash
from pcbparse.pipeline.budget import BudgetExceeded, TimeBudget, TokenBudget


@dataclass(slots=True)
class StageTrace:
    stage: str
    started_at: float
    elapsed_seconds: float
    status: str
    object_count: int | None = None
    message: str = ""


@dataclass(slots=True)
class RunContext:
    run_id: str
    image_id: str
    config_hash: str
    time_budget: TimeBudget
    token_budget: TokenBudget
    artifact_policy: str = "error"
    model_versions: dict[str, str] = field(default_factory=dict)
    random_seed: int = 20260812
    trace: list[StageTrace] = field(default_factory=list)

    @classmethod
    def start(cls, image_path: str | Path, config: dict[str, object]) -> "RunContext":
        path = Path(image_path)
        run_config = config.get("run", {}) if isinstance(config.get("run"), dict) else {}
        qwen_config = config.get("qwen", {}) if isinstance(config.get("qwen"), dict) else {}
        return cls(
            run_id=uuid.uuid4().hex,
            image_id=path.stem,
            config_hash=config_hash(config),
            time_budget=TimeBudget(float(run_config.get("max_image_seconds", 120.0))),
            token_budget=TokenBudget(
                max_calls=int(qwen_config.get("max_calls_per_image", 0)),
                max_input_tokens=int(qwen_config.get("max_input_tokens_per_image", 0)),
                max_output_tokens_per_call=int(qwen_config.get("max_output_tokens_per_call", 0)),
            ),
            artifact_policy=str(run_config.get("artifact_policy", "error")),
            random_seed=int(run_config.get("seed", 20260812)),
        )

    @contextmanager
    def stage(self, name: str) -> Iterator[dict[str, int | str | None]]:
        started = time.monotonic()
        details: dict[str, int | str | None] = {"object_count": None, "message": ""}
        status = "ok"
        try:
            self.time_budget.check(stage=name)
            yield details
        except Exception:
            status = "error"
            raise
        finally:
            elapsed = time.monotonic() - started
            self.trace.append(
                StageTrace(
                    stage=name,
                    started_at=started,
                    elapsed_seconds=elapsed,
                    status=status,
                    object_count=details["object_count"] if isinstance(details["object_count"], int) else None,
                    message=str(details["message"] or ""),
                )
            )
            if status == "ok" and self.time_budget.elapsed_seconds >= self.time_budget.total_seconds:
                raise BudgetExceeded(
                    f"{name} exceeded the image time budget of {self.time_budget.total_seconds:.3f}s"
                )

    def manifest(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "image_id": self.image_id,
            "config_hash": self.config_hash,
            "random_seed": self.random_seed,
            "model_versions": dict(sorted(self.model_versions.items())),
            "time": {
                "elapsed_seconds": self.time_budget.elapsed_seconds,
                "remaining_seconds": self.time_budget.remaining_seconds,
            },
            "tokens": {
                "calls": self.token_budget.calls,
                "input_tokens": self.token_budget.input_tokens,
                "output_tokens": self.token_budget.output_tokens,
            },
            "trace": [asdict(item) for item in self.trace],
        }


def stable_file_hash(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
