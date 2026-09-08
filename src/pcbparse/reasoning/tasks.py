from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pcbparse.ir.models import CircuitIR, Evidence
from pcbparse.reasoning.router import AmbiguityTask, Candidate


_COMPONENT_FIELD_ALIASES = {
    "designator": "key",
    "key": "key",
    "value": "value",
    "name": "name",
    "searchName": "name",
    "searchname": "name",
}
_PIN_FIELD_ALIASES = {"number": "number", "name": "name"}
_AMBIGUOUS_DECISIONS = {
    "ambiguous",
    "component_number_conflict",
    "global_key_conflict",
    "low_margin",
    "unresolved",
}


def generate_ambiguity_tasks(
    ir: CircuitIR,
    *,
    max_candidates: int = 5,
) -> list[AmbiguityTask]:
    """Build deterministic, closed-set reasoning tasks from existing evidence.

    Candidate values are copied only from evidence emitted by upstream OCR /
    association stages.  The sole exception is an ``ambiguous`` crossing, for
    which the IR schema itself defines the two legal resolutions.  A field that
    already has a value is deliberately not made into a task.
    """

    if not 2 <= max_candidates <= 5:
        raise ValueError("max_candidates must be in [2, 5]")

    tasks: list[AmbiguityTask] = []
    for component in sorted(ir.components, key=lambda item: item.internal_id):
        empty_fields = {
            "key": not _has_value(component.key),
            "value": not _has_value(component.value),
            "name": not _has_value(component.name),
        }
        for field in ("key", "value", "name"):
            if not empty_fields[field]:
                continue
            candidates, ambiguity = _field_candidates(
                component.evidence,
                owner_kind="component",
                requested_field=field,
                aliases=_COMPONENT_FIELD_ALIASES,
                max_candidates=max_candidates,
            )
            if len(candidates) < 2:
                continue
            tasks.append(
                _field_task(
                    owner_kind="component",
                    owner_id=component.internal_id,
                    field=field,
                    candidates=candidates,
                    ambiguity=ambiguity,
                )
            )

    for pin in sorted(ir.pins, key=lambda item: item.internal_id):
        empty_fields = {
            "number": not _has_value(pin.number),
            "name": not _has_value(pin.name),
        }
        for field in ("number", "name"):
            if not empty_fields[field]:
                continue
            candidates, ambiguity = _field_candidates(
                pin.evidence,
                owner_kind="pin",
                requested_field=field,
                aliases=_PIN_FIELD_ALIASES,
                max_candidates=max_candidates,
            )
            if len(candidates) < 2:
                continue
            tasks.append(
                _field_task(
                    owner_kind="pin",
                    owner_id=pin.internal_id,
                    field=field,
                    candidates=candidates,
                    ambiguity=ambiguity,
                )
            )

    for junction in sorted(ir.junctions, key=lambda item: item.internal_id):
        if junction.kind != "ambiguous":
            continue
        task_id = f"reasoning:junction:{junction.internal_id}:kind"
        candidates = tuple(
            Candidate(candidate_id=kind, value=kind, score=0.5)
            for kind in ("connected_cross", "non_connected_cross")
        )
        tasks.append(
            AmbiguityTask(
                task_id=task_id,
                task_type="junction.kind",
                candidates=candidates,
                primary_score=0.0,
                graph_inconsistency_score=1.0,
                downstream_value_score=1.0,
                hardcase_prior=1.0,
                context={
                    "owner_kind": "junction",
                    "object_id": junction.internal_id,
                    "field": "kind",
                    "current_value": "ambiguous",
                },
            )
        )

    return sorted(tasks, key=lambda item: item.task_id)


# A readable alias for callers that think in terms of construction rather than
# generation.  Keeping one implementation avoids divergent task semantics.
build_ambiguity_tasks = generate_ambiguity_tasks


def _field_task(
    *,
    owner_kind: str,
    owner_id: str,
    field: str,
    candidates: tuple[Candidate, ...],
    ambiguity: bool,
) -> AmbiguityTask:
    task_id = f"reasoning:{owner_kind}:{owner_id}:{field}"
    downstream_value = 1.0 if field in {"key", "number"} else 0.55
    return AmbiguityTask(
        task_id=task_id,
        task_type=f"{owner_kind}.{field}",
        candidates=candidates,
        primary_score=max(candidate.score for candidate in candidates),
        graph_inconsistency_score=1.0 if ambiguity else 0.0,
        downstream_value_score=downstream_value,
        hardcase_prior=1.0 if ambiguity else 0.0,
        context={
            "owner_kind": owner_kind,
            "object_id": owner_id,
            "field": field,
            "current_value": "",
        },
    )


def _field_candidates(
    evidence: Sequence[Evidence],
    *,
    owner_kind: str,
    requested_field: str,
    aliases: Mapping[str, str],
    max_candidates: int,
) -> tuple[tuple[Candidate, ...], bool]:
    values: dict[str, tuple[str, float]] = {}
    ambiguity = False
    for item in sorted(evidence, key=lambda value: value.evidence_id):
        metadata = item.metadata
        raw_field = metadata.get("field")
        if not isinstance(raw_field, str) or aliases.get(raw_field, aliases.get(raw_field.casefold())) != requested_field:
            continue
        kind = str(metadata.get("kind", "")).casefold()
        if kind and owner_kind.casefold() not in kind:
            continue
        decision = str(metadata.get("decision", "")).casefold()
        ambiguity = ambiguity or decision in _AMBIGUOUS_DECISIONS
        for value, score in _candidate_values(metadata.get("candidates"), default_score=item.score):
            _keep_candidate(values, value, score)
        # Some adapters put the lexical list directly on field evidence.
        for value, score in _candidate_values(
            metadata.get("lexical_candidates"),
            default_score=item.score,
        ):
            _keep_candidate(values, value, score)

    ranked = sorted(values.values(), key=lambda item: (-item[1], item[0].casefold(), item[0]))
    result = tuple(
        Candidate(
            candidate_id=_candidate_id(owner_kind, requested_field, value),
            value=value,
            score=score,
        )
        for value, score in ranked[:max_candidates]
    )
    return result, ambiguity


def _candidate_values(raw: Any, *, default_score: float) -> Iterable[tuple[str, float]]:
    if not isinstance(raw, (list, tuple)):
        return
    for entry in raw:
        if isinstance(entry, str):
            yield entry, _score(default_score)
            continue
        if not isinstance(entry, Mapping):
            continue
        score = _score(entry.get("score", default_score))
        value = entry.get("value")
        if isinstance(value, str):
            yield value, score
        lexical = entry.get("lexical_candidates")
        if isinstance(lexical, (list, tuple)):
            for alternative in lexical:
                if isinstance(alternative, str):
                    yield alternative, score


def _keep_candidate(
    values: dict[str, tuple[str, float]],
    value: str,
    score: float,
) -> None:
    normalized = value.strip()
    if not normalized:
        return
    key = normalized.casefold()
    prior = values.get(key)
    if prior is None or score > prior[1] or (score == prior[1] and normalized < prior[0]):
        values[key] = (normalized, score)


def _candidate_id(owner_kind: str, field: str, value: str) -> str:
    digest = hashlib.sha256(
        f"{owner_kind}\0{field}\0{value}".encode("utf-8")
    ).hexdigest()[:16]
    return f"candidate:{digest}"


def _score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _has_value(value: str | None) -> bool:
    return isinstance(value, str) and bool(value.strip())


__all__ = ["build_ambiguity_tasks", "generate_ambiguity_tasks"]
