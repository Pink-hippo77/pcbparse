from __future__ import annotations

import json
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pcbparse.association import associate_ocr
from pcbparse.graph.physical import build_physical_graph
from pcbparse.ir.models import CircuitIR, Diagnostic, IRState
from pcbparse.nets.hypergraph import build_electrical_nets
from pcbparse.output.official_serializer import (
    atomic_write_official_json,
    official_json_hash,
    serialize_official,
)
from pcbparse.pipeline.context import RunContext
from pcbparse.reasoning.orchestrator import run_selective_reasoning
from pcbparse.reasoning.qwen_client import (
    CandidateOnlyReasoner,
    DisabledReasoner,
    HTTPQwenReasoner,
)
from pcbparse.validation import (
    GeometryValidator,
    GraphValidator,
    ReferenceValidator,
    SchemaValidator,
)
from pcbparse.validation.common import merge_diagnostics, stable_diagnostics
from pcbparse.validation.repair import repair_ir_once


class PipelineError(RuntimeError):
    def __init__(self, message: str, diagnostics: list[Diagnostic] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or []


class IRProvider(Protocol):
    def build_ir(self, input_path: str | Path, ctx: RunContext | None = None) -> CircuitIR: ...


@dataclass(slots=True)
class InferenceResult:
    input_path: Path
    output_path: Path
    output: dict[str, Any]
    output_hash: str
    ir: CircuitIR
    diagnostics: list[Diagnostic]
    context: RunContext


def _submission_mode(config: dict[str, Any]) -> str:
    run = config.get("run", {})
    mode = str(run.get("submission_mode", "strict")).strip().lower()
    return mode if mode in {"strict", "degraded"} else "strict"


def _is_debug_execution(config: dict[str, Any]) -> bool:
    """True when the pipeline runs in debug/non-production mode."""
    run = config.get("run", {})
    profile = str(run.get("profile", "")).strip().lower()
    return profile == "debug"


def _prune_unrepresentable(ir: Any) -> None:
    """Drop objects the official schema cannot represent (never guess)."""
    valid_components = {
        item.internal_id
        for item in ir.components
        if isinstance(item.key, str) and item.key.strip()
    }
    total_components = len(ir.components)
    valid_pins = {
        item.internal_id
        for item in ir.pins
        # The guide explicitly supports empty/noisy pinNumber values through
        # deterministic pin_1, pin_2, ... conversion.  Only ownerless pins or
        # pins whose owner was pruned remain unrepresentable.
        if item.component_id in valid_components
    }
    total_pins = len(ir.pins)
    ir.components = [item for item in ir.components if item.internal_id in valid_components]
    ir.pins = [item for item in ir.pins if item.internal_id in valid_pins]
    for net in ir.physical_nets:
        net.pins = [pid for pid in net.pins if pid in valid_pins]
    for net in ir.electrical_nets:
        net.pins = [pid for pid in net.pins if pid in valid_pins]
    ir.electrical_nets = [net for net in ir.electrical_nets if net.pins]
    ir.metadata["degraded_submission_prune"] = {
        "components_dropped": total_components - len(ir.components),
        "pins_dropped": total_pins - len(ir.pins),
    }


_QUALITY_ERROR_CODES = frozenset(
    {
        "NET.DISCONNECTED_SEGMENTS",
        "JUNCTION.INSUFFICIENT_BRANCHES",
        "COMPONENT.MISSING_KEY",
    }
)


def _downgrade_quality_errors(
    diagnostics: list[Any],
) -> list[Any]:
    """In degraded mode, quality gaps are warnings; schema/ref/geometry stay hard."""
    for item in diagnostics:
        if item.severity == "ERROR" and item.code in _QUALITY_ERROR_CODES:
            item.severity = "WARNING"
    return diagnostics


def infer_one(
    input_path: str | Path,
    output_path: str | Path,
    *,
    provider: IRProvider,
    config: dict[str, Any],
    official_profile: str | Path | dict[str, Any] | None = None,
) -> InferenceResult:
    source = Path(input_path)
    ctx = RunContext.start(source, config)
    try:
        return _infer_one(
            input_path,
            output_path,
            provider=provider,
            config=config,
            official_profile=official_profile,
            _context=ctx,
        )
    except Exception as exc:
        setattr(exc, "context_manifest", ctx.manifest())
        # Attach the partial IR so failure artifacts can be saved for replay.
        # The PipelineError from _infer_one already carries partial_ir; don't
        # overwrite it with None.
        if not hasattr(exc, "partial_ir") or getattr(exc, "partial_ir", None) is None:
            partial_ir = getattr(exc, "_partial_ir", None)
            if partial_ir is None:
                for stage_data in reversed(ctx._stages if hasattr(ctx, '_stages') else []):
                    if isinstance(stage_data, dict) and "ir" in stage_data:
                        partial_ir = stage_data["ir"]
                        break
            setattr(exc, "partial_ir", partial_ir)
        raise


def _infer_one(
    input_path: str | Path,
    output_path: str | Path,
    *,
    provider: IRProvider,
    config: dict[str, Any],
    official_profile: str | Path | dict[str, Any] | None = None,
    _context: RunContext | None = None,
) -> InferenceResult:
    source = Path(input_path)
    destination = Path(output_path)
    ctx = _context or RunContext.start(source, config)

    with ctx.stage("perception") as trace:
        ir = provider.build_ir(source, ctx)
        if ir.state == IRState.RAW:
            ir.state = IRState.PERCEIVED
        if ir.state != IRState.PERCEIVED:
            raise PipelineError(f"provider returned unexpected state {ir.state}; expected PERCEIVED")
        declared_versions = ir.metadata.get("model_versions", {})
        if isinstance(declared_versions, dict):
            ctx.model_versions.update(
                {
                    str(task): str(version)
                    for task, version in declared_versions.items()
                    if str(task) and str(version)
                }
            )
        provider_version = getattr(provider, "version", None)
        if provider_version:
            ctx.model_versions.setdefault("provider", str(provider_version))
        trace["object_count"] = len(ir.components) + len(ir.pins) + len(ir.segments)

    with ctx.stage("association") as trace:
        associate_ocr(ir, config)
        ir.state = IRState.ASSOCIATED
        trace["object_count"] = len(ir.components) + len(ir.pins)
        report = ir.metadata.get("ocr_association", {})
        if isinstance(report, dict):
            diagnostics = report.get("diagnostics", {})
            if isinstance(diagnostics, dict):
                trace["message"] = f"ocr_association_diagnostics={diagnostics.get('total', 0)}"

    if _submission_mode(config) == "degraded":
        # Degraded output is explicit and local to this image. Drop objects
        # that cannot be represented safely; pin numbers may still be assigned
        # deterministically at serialization time, but ownership is never guessed.
        _prune_unrepresentable(ir)

    # Junction decisions change physical connectivity, so selective reasoning
    # must finish before either graph is materialized.  Running it later would
    # leave physical/electrical nets stale even though the junction object had
    # changed kind.
    with ctx.stage("selective_reasoning") as trace:
        reasoner = _reasoner_from_config(config)
        run_selective_reasoning(ir, config, ctx, reasoner)
        report = ir.metadata.get("selective_reasoning", {})
        trace["object_count"] = (
            int(report.get("generated_tasks", 0)) if isinstance(report, dict) else 0
        )
        if isinstance(report, dict):
            trace["message"] = (
                f"routed={report.get('routed_tasks', 0)},"
                f"applied={report.get('applied_tasks', 0)},"
                f"cache_hits={report.get('cache_hits', 0)}"
            )

    graph_config = config.get("graph", {})
    association_config = config.get("association", {})
    with ctx.stage("physical_graph") as trace:
        ir.physical_nets, snap_decisions = build_physical_graph(
            ir,
            endpoint_tolerance_px=float(graph_config.get("endpoint_merge_tolerance_px", 3.0)),
            collinear_merge_tolerance_px=float(
                graph_config.get("collinear_merge_tolerance_px", 0.0)
            ),
            collinear_gap_merge_tolerance_px=float(
                graph_config.get("collinear_gap_merge_tolerance_px", 0.0)
            ),
            pin_snap_min_px=float(
                association_config.get("pin_wire_snap_min_px", 2.0)
            ),
            pin_snap_tolerance_px=float(
                association_config.get(
                    "pin_wire_snap_max_px",
                    graph_config.get("pin_snap_tolerance_px", 6.0),
                )
            ),
            pin_point_projection_max_px=float(graph_config.get("pin_point_projection_max_px", 0.0)),
            unique_margin=float(association_config.get("unique_margin", 1.5)),
        )
        ir.metadata["snap_decisions"] = [
            {
                "pin_id": decision.pin_id,
                "status": decision.status,
                "segment_id": decision.segment_id,
                "cost": decision.cost,
                "margin": decision.margin,
            }
            for decision in snap_decisions
        ]
        ir.state = IRState.GRAPHED
        trace["object_count"] = len(ir.physical_nets)

    label_config = config.get("labels", {})
    with ctx.stage("electrical_graph") as trace:
        ir.electrical_nets = build_electrical_nets(
            ir,
            merge_signal_labels=bool(label_config.get("merge_signal_labels", True)),
            merge_power_labels=bool(label_config.get("merge_power_labels", False)),
            merge_ground_labels=bool(label_config.get("merge_ground_labels", False)),
            min_text_score=float(label_config.get("min_text_score", 0.75)),
            merge_threshold=float(label_config.get("merge_threshold", 0.85)),
        )
        ir.state = IRState.SEMANTIC
        trace["object_count"] = len(ir.electrical_nets)

    with ctx.stage("validation") as trace:
        validators = (
            SchemaValidator(),
            ReferenceValidator(),
            GeometryValidator(),
            GraphValidator(
                endpoint_tolerance_px=float(
                    graph_config.get("endpoint_merge_tolerance_px", 3.0)
                ),
                collinear_merge_tolerance_px=float(
                    graph_config.get("collinear_merge_tolerance_px", 0.0)
                ),
                collinear_gap_merge_tolerance_px=float(
                    graph_config.get("collinear_gap_merge_tolerance_px", 0.0)
                ),
            ),
        )
        repaired = repair_ir_once(ir, config_hash=ctx.config_hash)
        validator_issues = stable_diagnostics(
            issue for validator in validators for issue in validator.validate(repaired, ctx)
        )
        diagnostics = merge_diagnostics(repaired.diagnostics, validator_issues)
        repaired.diagnostics = diagnostics
        ir = repaired
        if _submission_mode(config) == "degraded":
            diagnostics = _downgrade_quality_errors(diagnostics)
            ir.diagnostics = diagnostics
            remaining_errors = [
                item for item in diagnostics if item.severity == "ERROR"
            ]
            if not remaining_errors and ir.state in {
                IRState.SEMANTIC,
                IRState.VALIDATED,
            }:
                ir.state = IRState.VALIDATED
        errors = [item for item in diagnostics if item.severity == "ERROR"]
        trace["object_count"] = len(diagnostics)
        if errors:
            if _is_debug_execution(config):
                # Debug mode: log quality errors but don't block serialization.
                # Debug providers (e.g., dataset_debug) naturally produce IR
                # with quality gaps; blocking them prevents debug output.
                ir.state = IRState.VALIDATED
            else:
                err = PipelineError(f"validation failed with {len(errors)} error(s)", diagnostics)
                err.partial_ir = ir
                raise err
        elif ir.state in {IRState.SEMANTIC, IRState.VALIDATED}:
            ir.state = IRState.VALIDATED

    with ctx.stage("serialization") as trace:
        output = serialize_official(ir, official_profile)
        output_hash = official_json_hash(output, official_profile)
        atomic_write_official_json(output, destination, official_profile)
        trace["object_count"] = len(output["components"]) + len(output["pins"]) + len(output["nets"])
        ir.state = IRState.SERIALIZED

    return InferenceResult(
        input_path=source,
        output_path=destination,
        output=output,
        output_hash=output_hash,
        ir=ir,
        diagnostics=diagnostics,
        context=ctx,
    )


def _reasoner_from_config(config: dict[str, Any]):
    qwen = config.get("qwen", {})
    if not isinstance(qwen, dict) or not bool(qwen.get("enabled", False)):
        return DisabledReasoner()
    backend = str(qwen.get("backend", "disabled")).strip().lower()
    if backend == "candidate_only":
        return CandidateOnlyReasoner(
            minimum_score=float(qwen.get("candidate_min_score", 0.75)),
            minimum_margin=float(qwen.get("candidate_min_margin", 0.10)),
        )
    if backend == "http":
        endpoint = str(qwen.get("endpoint", "")).strip()
        model = str(qwen.get("model", "")).strip()
        if not endpoint or not model:
            raise PipelineError(
                "qwen backend=http requires non-empty qwen.endpoint and qwen.model"
            )
        return HTTPQwenReasoner(
            endpoint=endpoint,
            model=model,
            timeout_seconds=float(qwen.get("timeout_seconds", 12.0)),
            api_key=(str(qwen["api_key"]) if qwen.get("api_key") else None),
            enable_thinking=bool(qwen.get("enable_thinking", False)),
            send_images=bool(qwen.get("send_images", False)),
        )
    if backend == "disabled":
        raise PipelineError(
            "qwen.enabled=true requires an explicit qwen.backend of candidate_only or http"
        )
    raise PipelineError(f"unsupported qwen backend {backend!r}")


def infer_batch(
    inputs: list[str | Path],
    output_dir: str | Path,
    *,
    provider: IRProvider,
    config: dict[str, Any],
    official_profile: str | Path | dict[str, Any] | None = None,
    continue_on_error: bool = True,
) -> tuple[list[InferenceResult], list[dict[str, Any]]]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    ordered_inputs = sorted((Path(item) for item in inputs), key=lambda path: path.as_posix())
    duplicate_stems = sorted(stem for stem, count in Counter(path.stem for path in ordered_inputs).items() if count > 1)
    if duplicate_stems:
        raise ValueError(
            "batch input stems would overwrite output files; duplicate stems="
            f"{duplicate_stems}"
        )
    results: list[InferenceResult] = []
    errors: list[dict[str, Any]] = []
    for input_item in ordered_inputs:
        output_path = destination / f"{input_item.stem}.json"
        try:
            results.append(
                infer_one(
                    input_item,
                    output_path,
                    provider=provider,
                    config=config,
                    official_profile=official_profile,
                )
            )
        except Exception as exc:
            record = {
                "input": str(input_item),
                "output": str(output_path),
                "error_type": type(exc).__name__,
                "message": str(exc),
                "diagnostics": [item.model_dump(mode="json") for item in getattr(exc, "diagnostics", [])],
            }
            errors.append(record)
            if not continue_on_error:
                raise
    if errors:
        error_path = destination / "error_manifest.json"
        payload = json.dumps(errors, ensure_ascii=False, indent=2, sort_keys=True)
        temporary = error_path.with_suffix(".json.tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(error_path)
    return results, errors
