from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from pcbparse.association.netlabel_text import associate_netlabel_texts
from pcbparse.association.ocr_common import (
    ASSOCIATION_STAGE,
    OCRAssociationConfig,
    canonical_details,
    point_bbox_distance,
    preferred_text,
)
from pcbparse.association.ocr_postprocess import postprocess_tokens
from pcbparse.association.semantic_fields import infer_semantic_fields
from pcbparse.association.text_component import (
    associate_component_texts,
    resolve_component_keys,
)
from pcbparse.association.text_pin import associate_pin_texts
from pcbparse.ir.models import CircuitIR
from pcbparse.knowledge.model_catalog import load_model_catalog


def associate_ocr(
    ir: CircuitIR,
    config: OCRAssociationConfig | Mapping[str, Any] | None = None,
) -> CircuitIR:
    """Run all OCR association passes on a Circuit-IR in place.

    The API deliberately leaves ``ir.state`` unchanged so an orchestrator can
    place it within its own state transition and timing boundary.  Existing
    component/pin values are preserved; only uniquely supported empty fields
    are assigned.  A compact report is written to
    ``ir.metadata['ocr_association']`` and detailed rankings live in Evidence.
    """

    resolved = OCRAssociationConfig.coerce(config)
    model_names = (
        load_model_catalog(resolved.component_model_catalog_path, resolved.component_model_catalog_sha256)
        if resolved.component_model_catalog_path is not None else frozenset()
    )
    # Re-running the stage should not multiply its diagnostics.  Diagnostics
    # from Golden/debug, perception, graph, and validation remain untouched.
    ir.diagnostics = [
        diagnostic
        for diagnostic in ir.diagnostics
        if diagnostic.stage != ASSOCIATION_STAGE
    ]
    before = len(ir.diagnostics)
    postprocess_summary = postprocess_tokens(ir.texts)
    semantic_options: dict[str, Any] = {}
    if resolved.pin_body_component_roi_context_enabled:
        semantic_options['pin_body_component_roi_context_enabled'] = True
    semantic_summary = infer_semantic_fields(
        ir.components,
        ir.pins,
        ir.texts,
        ir.segments,
        component_max_distance_factor=float(resolved.component_max_distance_factor),
        pin_max_distance_px=float(resolved.pin_max_distance_px),
        known_model_names=model_names,
        pin_body_name_context_enabled=resolved.pin_body_name_context_enabled,
        **semantic_options,
    )
    semantic_summary["model_catalog"] = {
        "enabled": bool(model_names), "names": len(model_names),
        "sha256": resolved.component_model_catalog_sha256,
    }
    pin_ocr_gate_summary = _gate_pin_candidates(ir, resolved)
    component_summary = associate_component_texts(
        ir.components,
        ir.texts,
        ir.diagnostics,
        resolved,
    )
    # The per-token association above is intentionally conservative and may
    # leave a designator unresolved when nearby OCR hypotheses compete.  A
    # second, global one-to-one pass can safely recover only assignments that
    # are regex-valid, component-compatible, unused, and separated from the
    # next-best target by the same configured margin.
    global_key_assignments = resolve_component_keys(
        ir.components,
        ir.texts,
        max_distance_factor=float(resolved.component_max_distance_factor),
        unique_margin=float(resolved.component_unique_margin),
        min_score=float(resolved.component_min_score),
    )
    component_summary['global_key_assignments'] = dict(sorted(global_key_assignments.items()))
    component_summary['global_key_assignment_count'] = len(global_key_assignments)
    pin_summary = associate_pin_texts(
        ir.pins,
        ir.texts,
        ir.diagnostics,
        resolved,
    )
    netlabel_summary = associate_netlabel_texts(ir, resolved)
    stage_diagnostics = ir.diagnostics[before:]
    ir.metadata["ocr_association"] = {
        "schema_version": 1,
        "config": resolved.to_dict(),
        "postprocess": postprocess_summary,
        "semantic_fields": semantic_summary,
        "pin_ocr_gate": pin_ocr_gate_summary,
        "component": component_summary,
        "pin": pin_summary,
        "netlabel": netlabel_summary,
        "diagnostics": {
            "total": len(stage_diagnostics),
            "by_code": dict(sorted(Counter(item.code for item in stage_diagnostics).items())),
        },
        "state_unchanged": True,
    }
    return ir

def _gate_pin_candidates(
    ir: CircuitIR,
    config: OCRAssociationConfig,
) -> dict[str, Any]:
    """Keep visual Pin candidates that have local OCR number evidence."""
    candidate_count = len(ir.pins)
    if not config.pin_ocr_gate_enabled:
        return {
            "enabled": False,
            "candidate_count": candidate_count,
            "retained_count": candidate_count,
            "removed_count": 0,
            "eligible_token_count": 0,
            "reason": "disabled",
        }
    tokens = [
        token
        for token in ir.texts
        if canonical_details(token.details) == "pinnumber"
        and preferred_text(token)
        and float(token.score) >= float(config.pin_ocr_gate_min_text_score)
    ]
    if not tokens:
        return {
            "enabled": True,
            "candidate_count": candidate_count,
            "retained_count": candidate_count,
            "removed_count": 0,
            "eligible_token_count": 0,
            "reason": "no_eligible_pin_number_tokens",
        }
    retained = []
    for pin in ir.pins:
        supported = any(
            (token.component_id is None or token.component_id == pin.component_id)
            and point_bbox_distance(pin.point, token.bbox) <= float(config.pin_ocr_gate_max_distance_px)
            for token in tokens
        )
        if supported:
            retained.append(pin)
    ir.pins[:] = retained
    return {
        "enabled": True,
        "candidate_count": candidate_count,
        "retained_count": len(retained),
        "removed_count": candidate_count - len(retained),
        "eligible_token_count": len(tokens),
        "max_distance_px": float(config.pin_ocr_gate_max_distance_px),
        "min_text_score": float(config.pin_ocr_gate_min_text_score),
        "reason": "applied",
    }

__all__ = ["OCRAssociationConfig", "associate_ocr"]
