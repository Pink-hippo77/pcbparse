from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pcbparse.ir.models import CircuitIR, Diagnostic, Evidence
from pcbparse.pipeline.budget import BudgetExceeded
from pcbparse.pipeline.context import RunContext
from pcbparse.reasoning.cache import (
    DEFAULT_REASONING_CACHE,
    ReasoningCache,
    reasoning_cache_key,
)
from pcbparse.reasoning.qwen_client import DisabledReasoner, HTTPQwenReasoner, Reasoner
from pcbparse.reasoning.router import AmbiguityTask, Decision, ValueAwareRouter
from pcbparse.reasoning.tasks import generate_ambiguity_tasks


REASONING_STAGE = "reasoning.selective"
_METADATA_KEY = "selective_reasoning"


def run_selective_reasoning(
    ir: CircuitIR,
    config: Mapping[str, Any],
    ctx: RunContext,
    reasoner: Reasoner | None = None,
) -> CircuitIR:
    """Resolve selected ambiguities without allowing generative IR edits.

    The reasoner sees candidate IDs and context, never the mutable Circuit-IR.
    Its decision can fill only an explicitly empty semantic field, or resolve an
    ``ambiguous`` crossing to one of the two schema-defined crossing kinds.
    Geometry, object counts, and high-confidence/existing values are immutable.
    """

    qwen_config = _qwen_config(config)
    if not bool(qwen_config.get("enabled", False)):
        return ir

    active_reasoner: Reasoner = reasoner if reasoner is not None else DisabledReasoner()
    reasoner_version = _reasoner_version(active_reasoner, qwen_config, ctx)
    config_fingerprint = _config_fingerprint(qwen_config)
    pre_state_signature = _reasoning_state_signature(ir)
    previous = ir.metadata.get(_METADATA_KEY)
    if (
        isinstance(previous, Mapping)
        and previous.get("completed") is True
        and previous.get("reasoner_version") == reasoner_version
        and previous.get("config_fingerprint") == config_fingerprint
        and previous.get("post_state_signature") == pre_state_signature
    ):
        return ir

    # Reprocessing after an actual upstream/config change replaces this stage's
    # own diagnostics and ledger rather than multiplying them.
    ir.diagnostics = [item for item in ir.diagnostics if item.stage != REASONING_STAGE]

    max_candidates = int(qwen_config.get("max_candidates_per_task", 5))
    max_candidates = max(2, min(5, max_candidates))
    tasks = generate_ambiguity_tasks(ir, max_candidates=max_candidates)
    router = ValueAwareRouter(
        threshold=float(qwen_config.get("route_threshold", 0.72)),
        min_time_reserve_seconds=float(qwen_config.get("min_time_reserve_seconds", 25.0)),
    )
    cache_enabled = bool(qwen_config.get("cache", True))
    cache = _cache_from_config(qwen_config)
    image_hash, image_hash_source = _image_hash(ir)
    minimum_confidence = _clamp01(qwen_config.get("decision_min_confidence", 0.0))

    ranked_tasks = sorted(
        tasks,
        key=lambda task: (-router.ambiguity_score(task), task.task_id),
    )
    ledger: list[dict[str, Any]] = []
    applied = 0
    routed = 0
    rejected = 0
    cache_hits = 0

    for task in ranked_tasks:
        route_score = router.ambiguity_score(task)
        estimated_input_tokens = _estimated_input_tokens(task)
        cache_key = reasoning_cache_key(image_hash, task, reasoner_version)
        row = _ledger_base(
            task,
            route_score=route_score,
            estimated_input_tokens=estimated_input_tokens,
            cache_key=cache_key,
        )

        # Junction crossing kind is a visual decision.  The text-only HTTP
        # reasoner (send_images=False) has no crop to inspect, so routing
        # would only produce UNKNOWN and waste the call budget.  Capable
        # fixture/rule reasoners keep the ability to resolve junctions.
        if (
            task.task_type == "junction.kind"
            and not task.crop_path
            and isinstance(active_reasoner, HTTPQwenReasoner)
            and not active_reasoner.send_images
        ):
            row["status"] = "not_routed"
            row["reason"] = "junction resolution requires a visual crop"
            ledger.append(row)
            continue

        # A cached decision has no call/token cost, but it still has to satisfy
        # the configured value threshold.  This prevents a stale policy from
        # silently widening the set of routed tasks.
        threshold_eligible = (
            route_score >= router.threshold and 1 < len(task.candidates) <= 5
        )
        cached = cache.get(cache_key) if cache_enabled and threshold_eligible else None
        if cached is not None:
            decision = cached
            row["cache_hit"] = True
            row["routed"] = True
            routed += 1
            cache_hits += 1
        elif router.should_route(
            task,
            ctx.token_budget,
            ctx.time_budget,
            estimated_input_tokens=estimated_input_tokens,
        ):
            row["routed"] = True
            routed += 1
            decision, failure = _choose_with_accounting(
                active_reasoner,
                task,
                ctx,
                estimated_input_tokens=estimated_input_tokens,
            )
            if failure is not None:
                row["status"] = "reasoner_failed"
                row["reason"] = failure
                rejected += 1
                _append_diagnostic(
                    ir,
                    severity="WARNING",
                    code="REASONING.REASONER_FAILED",
                    message=f"task {task.task_id} failed closed: {failure}",
                    object_ids=[task.context.get("object_id", task.task_id)],
                )
                ledger.append(row)
                continue
        else:
            row["status"] = "not_routed"
            row["reason"] = _not_routed_reason(
                task,
                router,
                ctx,
                estimated_input_tokens=estimated_input_tokens,
            )
            ledger.append(row)
            continue

        validation_error = _decision_error(decision, task)
        if validation_error is not None:
            if cache_enabled and bool(row["cache_hit"]):
                cache.discard(cache_key)
            row["status"] = "decision_rejected"
            row["reason"] = validation_error
            row["decision"] = _decision_dict(decision)
            rejected += 1
            _append_diagnostic(
                ir,
                severity="WARNING",
                code="REASONING.DECISION_REJECTED",
                message=f"task {task.task_id} rejected: {validation_error}",
                object_ids=[task.context.get("object_id", task.task_id)],
            )
            ledger.append(row)
            continue

        if cache_enabled and not bool(row["cache_hit"]):
            cache.put(cache_key, decision)
        row["decision"] = _decision_dict(decision)
        if decision.selected_id == "UNKNOWN":
            row["status"] = "unknown"
            row["reason"] = decision.reason or "reasoner selected UNKNOWN"
            _append_diagnostic(
                ir,
                severity="INFO",
                code="REASONING.UNKNOWN",
                message=f"task {task.task_id} remains unresolved",
                object_ids=[task.context.get("object_id", task.task_id)],
            )
            ledger.append(row)
            continue
        if decision.confidence < minimum_confidence:
            row["status"] = "below_decision_threshold"
            row["reason"] = (
                f"decision confidence {decision.confidence:.6f} is below "
                f"{minimum_confidence:.6f}"
            )
            _append_diagnostic(
                ir,
                severity="INFO",
                code="REASONING.DECISION_BELOW_THRESHOLD",
                message=f"task {task.task_id} remains unresolved below decision threshold",
                object_ids=[task.context.get("object_id", task.task_id)],
            )
            ledger.append(row)
            continue

        candidate = next(
            item for item in task.candidates if item.candidate_id == decision.selected_id
        )
        status, reason = _safe_apply(ir, task, candidate.value, decision, cache_key)
        row["status"] = status
        row["reason"] = reason
        if status == "applied":
            row["applied_value"] = candidate.value
            applied += 1
            _append_diagnostic(
                ir,
                severity="INFO",
                code="REASONING.APPLIED",
                message=f"task {task.task_id} selected a supplied candidate",
                object_ids=[task.context.get("object_id", task.task_id)],
            )
        else:
            rejected += 1
            _append_diagnostic(
                ir,
                severity="WARNING",
                code="REASONING.WRITE_PROTECTED",
                message=f"task {task.task_id} was not applied: {reason}",
                object_ids=[task.context.get("object_id", task.task_id)],
            )
        ledger.append(row)

    _append_diagnostic(
        ir,
        severity="INFO",
        code="REASONING.SUMMARY",
        message=(
            f"generated={len(tasks)}, routed={routed}, applied={applied}, "
            f"rejected={rejected}, cache_hits={cache_hits}"
        ),
        object_ids=[],
    )
    ir.metadata[_METADATA_KEY] = {
        "version": "1.0",
        "completed": True,
        "reasoner_version": reasoner_version,
        "config_fingerprint": config_fingerprint,
        "image_hash": image_hash,
        "image_hash_source": image_hash_source,
        "cache_enabled": cache_enabled,
        "generated_tasks": len(tasks),
        "routed_tasks": routed,
        "applied_tasks": applied,
        "rejected_tasks": rejected,
        "cache_hits": cache_hits,
        "ledger": ledger,
        "post_state_signature": "",  # populated after all evidence writes
    }
    ir.metadata[_METADATA_KEY]["post_state_signature"] = _reasoning_state_signature(ir)
    return ir


def _qwen_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = config.get("qwen")
    return nested if isinstance(nested, Mapping) else config


def _cache_from_config(config: Mapping[str, Any]) -> ReasoningCache:
    supplied = config.get("cache_store")
    return supplied if isinstance(supplied, ReasoningCache) else DEFAULT_REASONING_CACHE


def _choose_with_accounting(
    reasoner: Reasoner,
    task: AmbiguityTask,
    ctx: RunContext,
    *,
    estimated_input_tokens: int,
) -> tuple[Decision, str | None]:
    before_calls = ctx.token_budget.calls
    decision: Decision | None = None
    failure: str | None = None
    try:
        decision = reasoner.choose(task, ctx.token_budget)
    except Exception as exc:  # third-party model adapters must fail closed
        failure = type(exc).__name__
    finally:
        if ctx.token_budget.calls == before_calls:
            try:
                ctx.token_budget.reserve(
                    estimated_input_tokens,
                    min(64, max(0, ctx.token_budget.max_output_tokens_per_call)),
                )
            except BudgetExceeded as exc:
                failure = failure or str(exc)
    if failure is not None:
        return Decision(task.task_id, "UNKNOWN", 0.0, "reasoner_error", failure), failure
    if not isinstance(decision, Decision):
        return (
            Decision(task.task_id, "UNKNOWN", 0.0, "reasoner_error", "invalid decision type"),
            "invalid decision type",
        )
    return decision, None


def _decision_error(decision: Decision, task: AmbiguityTask) -> str | None:
    if decision.task_id != task.task_id:
        return f"decision task id {decision.task_id!r} does not match routed task"
    try:
        decision.validate_against(task)
    except ValueError as exc:
        return str(exc)
    return None


def _safe_apply(
    ir: CircuitIR,
    task: AmbiguityTask,
    value: str,
    decision: Decision,
    cache_key: str,
) -> tuple[str, str]:
    owner_kind = task.context.get("owner_kind", "")
    object_id = task.context.get("object_id", "")
    field = task.context.get("field", "")
    if owner_kind == "component" and field in {"key", "value", "name"}:
        owner = next((item for item in ir.components if item.internal_id == object_id), None)
        if owner is None:
            return "write_rejected", "component no longer exists"
        current = getattr(owner, field)
        if isinstance(current, str) and current.strip():
            return "write_rejected", "component field is already populated"
        setattr(owner, field, value)
        _upsert_reasoning_evidence(owner.evidence, task, decision, cache_key, value)
        return "applied", "filled empty component field"
    if owner_kind == "pin" and field in {"number", "name"}:
        owner = next((item for item in ir.pins if item.internal_id == object_id), None)
        if owner is None:
            return "write_rejected", "pin no longer exists"
        current = getattr(owner, field)
        if isinstance(current, str) and current.strip():
            return "write_rejected", "pin field is already populated"
        setattr(owner, field, value)
        _upsert_reasoning_evidence(owner.evidence, task, decision, cache_key, value)
        return "applied", "filled empty pin field"
    if owner_kind == "junction" and field == "kind":
        owner = next((item for item in ir.junctions if item.internal_id == object_id), None)
        if owner is None:
            return "write_rejected", "junction no longer exists"
        if owner.kind != "ambiguous":
            return "write_rejected", "junction is no longer ambiguous"
        if value not in {"connected_cross", "non_connected_cross"}:
            return "write_rejected", "candidate is not a legal crossing resolution"
        owner.kind = value  # type: ignore[assignment]
        _upsert_reasoning_evidence(owner.evidence, task, decision, cache_key, value)
        return "applied", "resolved ambiguous crossing"
    return "write_rejected", "task target is outside the reasoning write allowlist"


def _upsert_reasoning_evidence(
    evidence: list[Evidence],
    task: AmbiguityTask,
    decision: Decision,
    cache_key: str,
    value: str,
) -> None:
    item = Evidence(
        evidence_id=f"reasoning:{cache_key[:24]}",
        source="qwen",
        view_id="original",
        score=decision.confidence,
        payload_ref=task.task_id,
        metadata={
            "kind": "selective-reasoning",
            "selected_id": decision.selected_id,
            "selected_value": value,
            "decision_source": decision.source,
            "candidate_ids": [candidate.candidate_id for candidate in task.candidates],
            "cache_key": cache_key,
        },
    )
    for index, existing in enumerate(evidence):
        if existing.evidence_id == item.evidence_id:
            evidence[index] = item
            return
    evidence.append(item)


def _ledger_base(
    task: AmbiguityTask,
    *,
    route_score: float,
    estimated_input_tokens: int,
    cache_key: str,
) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "object_id": task.context.get("object_id", ""),
        "field": task.context.get("field", ""),
        "route_score": round(route_score, 8),
        "estimated_input_tokens": estimated_input_tokens,
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "value": candidate.value,
                "score": round(candidate.score, 8),
            }
            for candidate in task.candidates
        ],
        "cache_key": cache_key,
        "cache_hit": False,
        "routed": False,
        "status": "pending",
        "reason": "",
        "decision": None,
    }


def _decision_dict(decision: Decision) -> dict[str, Any]:
    return {
        "task_id": decision.task_id,
        "selected_id": decision.selected_id,
        "confidence": round(float(decision.confidence), 8),
        "source": decision.source,
        "reason": decision.reason,
    }


def _not_routed_reason(
    task: AmbiguityTask,
    router: ValueAwareRouter,
    ctx: RunContext,
    *,
    estimated_input_tokens: int,
) -> str:
    if router.ambiguity_score(task) < router.threshold:
        return "below route threshold"
    if not 1 < len(task.candidates) <= 5:
        return "candidate count outside [2, 5]"
    if ctx.token_budget.remaining_calls < 1:
        return "call budget exhausted"
    if ctx.token_budget.remaining_input_tokens < estimated_input_tokens:
        return "input token budget exhausted"
    if ctx.time_budget.remaining_seconds < router.min_time_reserve_seconds:
        return "time reserve reached"
    return "router declined task"


def _estimated_input_tokens(task: AmbiguityTask) -> int:
    payload = {
        "task": task.task_type,
        "context": task.context,
        "candidates": [
            {"id": item.candidate_id, "value": item.value, "score": item.score}
            for item in task.candidates
        ],
    }
    size = len(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    )
    return max(1, (size + 3) // 4)


def _reasoner_version(
    reasoner: Reasoner,
    config: Mapping[str, Any],
    ctx: RunContext,
) -> str:
    for key in ("reasoner_version", "model_version", "version", "model"):
        value = config.get(key)
        if isinstance(value, (str, int, float)) and str(value):
            return str(value)
    ctx_version = ctx.model_versions.get("qwen")
    if ctx_version:
        return str(ctx_version)
    for attribute in ("version", "model_version", "model"):
        value = getattr(reasoner, attribute, None)
        if isinstance(value, (str, int, float)) and str(value):
            return str(value)
    cls = type(reasoner)
    return f"{cls.__module__}.{cls.__qualname__}:unversioned"


def _image_hash(ir: CircuitIR) -> tuple[str, str]:
    declared = ir.image.sha256.strip().lower()
    if declared:
        return declared, "ir.image.sha256"
    path = Path(ir.image.path)
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest(), "image_file"
    payload = {
        "image_id": ir.image.image_id,
        "path": ir.image.path,
        "width": ir.image.width,
        "height": ir.image.height,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), "image_identity_fallback"


def _reasoning_state_signature(ir: CircuitIR) -> str:
    def upstream_evidence(items: list[Evidence]) -> list[dict[str, Any]]:
        return [
            item.model_dump(mode="json")
            for item in sorted(items, key=lambda value: value.evidence_id)
            if item.source != "qwen"
        ]

    payload = {
        "components": [
            {
                "id": item.internal_id,
                "key": item.key,
                "value": item.value,
                "name": item.name,
                "evidence": upstream_evidence(item.evidence),
            }
            for item in sorted(ir.components, key=lambda value: value.internal_id)
        ],
        "pins": [
            {
                "id": item.internal_id,
                "number": item.number,
                "name": item.name,
                "evidence": upstream_evidence(item.evidence),
            }
            for item in sorted(ir.pins, key=lambda value: value.internal_id)
        ],
        "junctions": [
            {
                "id": item.internal_id,
                "kind": item.kind,
                "evidence": upstream_evidence(item.evidence),
            }
            for item in sorted(ir.junctions, key=lambda value: value.internal_id)
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    safe = {
        str(key): value
        for key, value in config.items()
        if key != "cache_store" and isinstance(value, (str, int, float, bool, type(None), list, tuple, dict))
    }
    encoded = json.dumps(
        safe,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _append_diagnostic(
    ir: CircuitIR,
    *,
    severity: str,
    code: str,
    message: str,
    object_ids: list[str],
) -> None:
    ir.diagnostics.append(
        Diagnostic(
            severity=severity,  # type: ignore[arg-type]
            code=code,
            message=message,
            object_ids=object_ids,
            stage=REASONING_STAGE,
        )
    )


def _clamp01(value: Any) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, converted))


__all__ = ["REASONING_STAGE", "run_selective_reasoning"]
