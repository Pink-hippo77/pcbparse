from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from typing import Any

from pcbparse.reasoning.router import AmbiguityTask, Decision


@dataclass(frozen=True, slots=True)
class CachedDecision:
    cache_key: str
    decision: Decision

    def to_dict(self) -> dict[str, Any]:
        return {"cache_key": self.cache_key, "decision": asdict(self.decision)}


class ReasoningCache:
    """Small, thread-safe cache for deterministic closed-set decisions.

    The cache is deliberately value-only: it cannot store or return geometry.
    Persisted Circuit-IR metadata serves as the audit record, while this cache
    removes duplicate reasoner calls within a serving process.
    """

    def __init__(self) -> None:
        self._records: dict[str, Decision] = {}
        self._lock = threading.RLock()

    def get(self, cache_key: str) -> Decision | None:
        with self._lock:
            decision = self._records.get(cache_key)
            if decision is None:
                return None
            return Decision(**asdict(decision))

    def put(self, cache_key: str, decision: Decision) -> None:
        # Reconstructing also protects the cache from accidental mutable or
        # duck-typed objects supplied by an adapter.
        clean = Decision(**asdict(decision))
        with self._lock:
            self._records[cache_key] = clean

    def discard(self, cache_key: str) -> None:
        with self._lock:
            self._records.pop(cache_key, None)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


def reasoning_cache_key(
    image_hash: str,
    task: AmbiguityTask,
    reasoner_version: str,
) -> str:
    """Hash exactly the identity required for safe cache reuse."""

    payload = {
        "image_sha256": str(image_hash),
        "reasoner_version": str(reasoner_version),
        "task": {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "context": dict(sorted(task.context.items())),
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "value": candidate.value,
                    "score": round(float(candidate.score), 12),
                }
                for candidate in sorted(task.candidates, key=lambda item: item.candidate_id)
            ],
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# Compatibility alias with the subject first, useful to callers and tests.
stable_cache_key = reasoning_cache_key
DEFAULT_REASONING_CACHE = ReasoningCache()


__all__ = [
    "CachedDecision",
    "DEFAULT_REASONING_CACHE",
    "ReasoningCache",
    "reasoning_cache_key",
    "stable_cache_key",
]
