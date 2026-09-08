from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any

from pcbparse.association.geometry import closest_point_on_segment, point_segment_distance
from pcbparse.association.ocr_common import (
    OCRAssociationConfig,
    append_diagnostic,
    canonical_details,
    lexical_candidates,
    make_rule_evidence,
    preferred_text,
    upsert_evidence,
)
from pcbparse.ir.models import CircuitIR, NetLabel


def normalize_netlabel_text(value: str) -> str:
    """Apply Unicode and whitespace canonicalization without semantic edits."""

    normalized = unicodedata.normalize("NFKC", value).strip()
    return re.sub(r"\s+", " ", normalized)


def _known_net_type(value: str) -> str:
    normalized = re.sub(r"\s+", "", value).upper()
    if normalized in {"0", "GND", "GROUND", "AGND", "DGND", "PGND", "VSS"}:
        return "GROUND"
    if normalized.endswith("_GND"):
        return "GROUND"
    if (
        normalized in {"VCC", "VDD", "VEE", "VBAT", "VIN", "VOUT", "AVCC", "DVCC"}
        or re.fullmatch(r"[+-]?\d+(?:\.\d+)?V", normalized)
        or normalized.startswith(("VCC_", "VDD_", "+", "-"))
    ):
        return "POWER"
    return "SIGNAL" if normalized else "UNKNOWN"


def _source_text_ids(label: NetLabel) -> set[str]:
    result: set[str] = set()
    for evidence in label.evidence:
        source = evidence.metadata.get("source_text_id")
        if isinstance(source, str) and source:
            result.add(source)
    return result


def associate_netlabel_texts(
    ir: CircuitIR,
    config: OCRAssociationConfig,
) -> dict[str, Any]:
    """Convert typed net-label OCR into NetLabel objects and segment hints."""

    existing_sources: dict[str, NetLabel] = {}
    for label in ir.net_labels:
        for source_text_id in _source_text_ids(label):
            existing_sources.setdefault(source_text_id, label)
    label_ids = {label.internal_id for label in ir.net_labels}

    segment_to_physical: dict[str, list[str]] = defaultdict(list)
    for physical_net in ir.physical_nets:
        for segment_id in physical_net.segments:
            segment_to_physical[segment_id].append(physical_net.internal_id)

    relevant_tokens = 0
    created = 0
    accepted_segments = 0
    ambiguous = 0
    unmatched = 0
    preserved = 0

    for token in sorted(ir.texts, key=lambda item: item.internal_id):
        if canonical_details(token.details) != "netlabel":
            continue
        relevant_tokens += 1
        source_text_id = token.source_object_id or token.internal_id
        existing = existing_sources.get(source_text_id)
        if existing is not None:
            preserved += 1
            prior = next(
                (
                    item
                    for item in existing.evidence
                    if item.metadata.get("kind") == "text-to-netlabel"
                    and item.metadata.get("source_text_id") == source_text_id
                    and item.metadata.get("decision") != "preserved_existing"
                ),
                None,
            )
            prior_decision = prior.metadata.get("decision") if prior is not None else None
            if prior_decision == "ambiguous":
                candidates = prior.metadata.get("segment_candidates", [])
                segment_ids = [
                    str(candidate["segment_id"])
                    for candidate in candidates[:2]
                    if isinstance(candidate, dict) and candidate.get("segment_id")
                ] if isinstance(candidates, list) else []
                ambiguous += 1
                append_diagnostic(
                    ir.diagnostics,
                    severity="WARNING",
                    code="ASSOCIATION.NETLABEL_AMBIGUOUS",
                    message=f"net label {token.internal_id} is near multiple wire segments",
                    object_ids=[token.internal_id, *segment_ids],
                )
            elif prior_decision in {"unmatched", "below_text_threshold"}:
                unmatched += 1
                append_diagnostic(
                    ir.diagnostics,
                    severity="WARNING",
                    code="ASSOCIATION.NETLABEL_UNMATCHED",
                    message=(
                        f"net label {token.internal_id} has no unique nearby wire "
                        f"segment ({prior_decision})"
                    ),
                    object_ids=[token.internal_id],
                )
            evidence = make_rule_evidence(
                evidence_id=f"association:netlabel-existing:{token.internal_id}",
                score=existing.score,
                payload_ref=token.internal_id,
                metadata={
                    "kind": "text-to-netlabel",
                    "decision": "preserved_existing",
                    "net_label_id": existing.internal_id,
                    "source_text_id": source_text_id,
                },
            )
            upsert_evidence(token.evidence, evidence)
            continue

        raw_value = preferred_text(token)
        normalized = normalize_netlabel_text(raw_value or "")
        if not normalized:
            unmatched += 1
            append_diagnostic(
                ir.diagnostics,
                severity="WARNING",
                code="ASSOCIATION.NETLABEL_EMPTY",
                message=f"net-label text {token.internal_id} is empty after normalization",
                object_ids=[token.internal_id],
            )
            continue

        center = token.bbox.center
        ranked = sorted(
            (
                (point_segment_distance(center, segment), segment)
                for segment in ir.segments
            ),
            key=lambda item: (item[0], item[1].internal_id),
        )
        candidate_records = [
            {
                "segment_id": segment.internal_id,
                "distance_px": round(float(distance), 8),
                "source_edge_id": segment.source_edge_id,
            }
            for distance, segment in ranked[: config.max_recorded_candidates]
        ]
        best_distance = ranked[0][0] if ranked else None
        second_distance = ranked[1][0] if len(ranked) > 1 else None
        distance_margin = (
            second_distance - best_distance
            if best_distance is not None and second_distance is not None
            else None
        )
        if token.score < config.netlabel_min_text_score:
            status = "below_text_threshold"
        elif best_distance is None or best_distance > config.netlabel_max_distance_px:
            status = "unmatched"
        elif distance_margin is not None and distance_margin < config.netlabel_unique_margin_px:
            status = "ambiguous"
        else:
            status = "accepted"

        accepted_segment = ranked[0][1] if status == "accepted" else None
        attached_physical: str | None = None
        if accepted_segment is not None:
            physical_candidates = sorted(
                set(segment_to_physical.get(accepted_segment.internal_id, ()))
            )
            if len(physical_candidates) == 1:
                attached_physical = physical_candidates[0]
            accepted_segments += 1
        elif status == "ambiguous":
            ambiguous += 1
            append_diagnostic(
                ir.diagnostics,
                severity="WARNING",
                code="ASSOCIATION.NETLABEL_AMBIGUOUS",
                message=f"net label {token.internal_id} is near multiple wire segments",
                object_ids=[token.internal_id, *[item[1].internal_id for item in ranked[:2]]],
            )
        else:
            unmatched += 1
            append_diagnostic(
                ir.diagnostics,
                severity="WARNING",
                code="ASSOCIATION.NETLABEL_UNMATCHED",
                message=f"net label {token.internal_id} has no unique nearby wire segment ({status})",
                object_ids=[token.internal_id],
            )

        base_id = f"net-label:{token.internal_id}"
        label_id = base_id
        suffix = 2
        while label_id in label_ids:
            label_id = f"{base_id}:{suffix}"
            suffix += 1
        label_ids.add(label_id)
        textual_type = _known_net_type(normalized)
        evidence = make_rule_evidence(
            evidence_id=f"association:netlabel:{token.internal_id}",
            score=token.score,
            payload_ref=token.internal_id,
            metadata={
                "kind": "text-to-netlabel",
                "decision": status,
                "source_text_id": source_text_id,
                "raw_text": raw_value,
                "normalized_text": normalized,
                "lexical_candidates": lexical_candidates(token),
                "textual_net_type": textual_type,
                "accepted_segment_id": (
                    accepted_segment.internal_id if accepted_segment is not None else None
                ),
                "attached_physical_net_id": attached_physical,
                "distance_margin_px": distance_margin,
                "thresholds": {
                    "min_text_score": config.netlabel_min_text_score,
                    "max_distance_px": config.netlabel_max_distance_px,
                    "unique_margin_px": config.netlabel_unique_margin_px,
                },
                "segment_candidates": candidate_records,
                "input_evidence_ids": [item.evidence_id for item in token.evidence],
            },
        )
        label = NetLabel(
            internal_id=label_id,
            instance_text=raw_value or "",
            resolved_text=normalized,
            net_type=textual_type if status == "accepted" else "UNKNOWN",  # type: ignore[arg-type]
            point=(
                closest_point_on_segment(center, accepted_segment)
                if accepted_segment is not None
                else center
            ),
            attached_physical_net_id=attached_physical,
            score=token.score,
            evidence=[evidence],
        )
        ir.net_labels.append(label)
        existing_sources[source_text_id] = label
        created += 1
        token_link = make_rule_evidence(
            evidence_id=f"association:netlabel-link:{token.internal_id}",
            score=token.score,
            payload_ref=label_id,
            metadata={
                "kind": "text-to-netlabel-link",
                "decision": status,
                "net_label_id": label_id,
                "accepted_segment_id": (
                    accepted_segment.internal_id if accepted_segment is not None else None
                ),
            },
        )
        upsert_evidence(token.evidence, token_link)

    return {
        "relevant_tokens": relevant_tokens,
        "created": created,
        "preserved_existing": preserved,
        "accepted_segments": accepted_segments,
        "ambiguous": ambiguous,
        "unmatched": unmatched,
    }


__all__ = ["associate_netlabel_texts", "normalize_netlabel_text"]
