from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

from pcbparse.association.ocr_common import (
    OCRAssociationConfig,
    RankedAssociationCandidate,
    append_diagnostic,
    bbox_gap_distance,
    canonical_details,
    choose_unique_target,
    field_choice,
    lexical_candidates,
    make_rule_evidence,
    preferred_text,
    upsert_evidence,
)
from pcbparse.ir.models import Component, Diagnostic, TextToken


_DESIGNATOR = re.compile(r"^[A-Z]{1,4}[0-9]+[A-Z]?$")
# All-letter designators (RUN, VCC, GND, VDD, etc.) are valid when they
# belong to a recognized power/signal category.  2-6 letters keeps the bar
# high enough to avoid random OCR noise while covering known labels.
_ALL_LETTER_DESIGNATOR = re.compile(r"^[A-Z]{2,6}$")

# Categories where all-letter designators are legitimate.
_ALL_LETTER_COMPATIBLE_CATEGORIES = frozenset({
    "VREG", "VOLTAGE_REGULATOR",
    "LDO", "DCDC", "DC-DC",
    "CONNECTOR", "HEADER", "Jack", "Jack",
    "ESD", "TVS", "PROTECTION",
    "CRYSTAL", "OSCILLATOR", "OSC",
    "SWITCH", "BUTTON",
    "LED", "LIGHT",
    "FUSE", "FERRITE",
    "BUS", "SIGNAL", "POWER", "GROUND",
    "VCC", "VDD", "VSS", "VDDIO", "VBUS",
    "BOX", "GENERIC", "UNKNOWN",
})


def _is_valid_designator(value: str, category: str = "") -> bool:
    """Check whether *value* is a conservative, visually-grounded designator.

    Standard alphanumeric designators (R1, C10, U5A) are always accepted.
    All-letter designators (RUN, VCC, GND) are accepted only when the
    component category is compatible — this prevents arbitrary OCR noise
    from becoming a key.
    """
    if _DESIGNATOR.fullmatch(value):
        return True
    if _ALL_LETTER_DESIGNATOR.fullmatch(value):
        normalized = category.upper().strip()
        return normalized in _ALL_LETTER_COMPATIBLE_CATEGORIES or normalized.startswith(("V", "G", "BUS", "SIG"))
    return False


def resolve_component_keys(
    components: list[Component],
    texts: list[TextToken],
    *,
    max_distance_factor: float = 2.0,
    unique_margin: float = 0.10,
    min_score: float = 0.0,
) -> dict[str, str]:
    """Assign unique designators from OCR tokens.

    The function accepts both standard alphanumeric designators (R1, C10)
    and legitimate all-letter designators (RUN, VCC) for compatible
    categories.  When multiple components each have independent evidence for
    the same visible designator, the visible text is preserved on all of
    them; the serializer handles KEY_1/KEY_2 suffixing for official output.
    """

    component_index = {component.internal_id: component for component in components}
    candidates: list[tuple[TextToken, str, str | None]] = []
    for text in texts:
        value = preferred_text(text)
        if value and canonical_details(text.details) == "designator":
            owner = str(text.component_id) if text.component_id is not None else None
            # An explicit owner is a hard constraint.  In particular, never
            # let a later global pass move an OCR token across components.
            if owner is None or owner in component_index:
                category = component_index[owner].category if owner and owner in component_index else ""
                if _is_valid_designator(value, category):
                    candidates.append((text, value, owner))
    # Explicit component-owned OCR is stronger than a duplicate global token with the same spelling.
    explicit_values = {
        _canonical_designator(value, component_index[owner].category).casefold()
        for _, value, owner in candidates
        if owner is not None and owner in component_index
    }
    scored: list[tuple[float, str, str]] = []
    if (
        not isinstance(min_score, (int, float))
        or isinstance(min_score, bool)
        or not math.isfinite(float(min_score))
        or not 0.0 <= float(min_score) <= 1.0
    ):
        raise ValueError("min_score must be finite in [0, 1]")
    for component in components:
        scale = max(component.bbox.width, component.bbox.height, 1.0)
        for token, value, owner in candidates:
            if owner is not None and owner != component.internal_id:
                continue
            canonical_value = _canonical_designator(value, component.category)
            if owner is None and canonical_value.casefold() in explicit_values:
                continue
            distance = bbox_gap_distance(component.bbox, token.bbox)
            normalized = distance / scale
            if normalized > max_distance_factor:
                continue
            prefix = re.match(r"^[A-Z]+", canonical_value)
            compatibility = 1.0 if prefix and _prefix_compatible(component.category, prefix.group(0)) else 0.0
            proximity = max(0.0, 1.0 - normalized / max_distance_factor)
            score = (
                float(token.score)
                if owner is not None
                else 0.65 * float(token.score) + 0.25 * proximity + 0.10 * compatibility
            )
            scored.append((min(1.0, score), component.internal_id, canonical_value))
    assignments: dict[str, str] = {}
    # Keys already assigned by a prior pass (e.g., from explicit component-owned OCR)
    # are locked.  However, different components may legitimately share the same
    # visible designator (e.g., two resistors both labelled "R1").  The official
    # serializer handles KEY_1/KEY_2 suffixing; we only need to prevent assigning
    # the same key to the same component twice.
    used_keys: set[str] = {component.key for component in components if component.key}
    # Resolve the remaining image-wide keys in score order.  Multiple components
    # may receive the same designator when each has independent evidence; the
    # serializer deterministically adds suffixes for official output.
    missing_components = {
        component.internal_id
        for component in components
        if not component.key
    }
    edge_candidates: dict[tuple[str, str], float] = {}
    for score, component_id, key in scored:
        if key in used_keys:
            continue
        pair = (component_id, key)
        edge_candidates[pair] = max(edge_candidates.get(pair, -1.0), score)
    ordered_edges = sorted(
        (
            (score, component_id, key)
            for (component_id, key), score in edge_candidates.items()
            if component_id in missing_components
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    assigned_components: set[str] = set()
    # Track newly assigned keys so a second component with independent evidence
    # can also receive the same visible designator.
    newly_used_keys: set[str] = set()
    for score, component_id, key in ordered_edges:
        if component_id in assigned_components:
            continue
        if score < float(min_score):
            continue
        competing_targets = sorted(
            (
                competing_score
                for (competing_component, competing_key), competing_score
                in edge_candidates.items()
                if competing_key == key
                and competing_component != component_id
            ),
            reverse=True,
        )
        if competing_targets and score - competing_targets[0] < unique_margin:
            continue
        alternatives = sorted(
            (
                alternative_score
                for (alternative_component, alternative_key), alternative_score
                in edge_candidates.items()
                if alternative_component == component_id
                and alternative_key != key
                and alternative_key not in used_keys
                and alternative_key not in newly_used_keys
            ),
            reverse=True,
        )
        if alternatives and score - alternatives[0] < unique_margin:
            continue
        component = next(
            item
            for item in components
            if item.internal_id == component_id
        )
        component.key = key
        component.key_score = score
        assignments[component_id] = key
        assigned_components.add(component_id)
        used_keys.add(key)
    return assignments


def _prefix_compatible(category: str, prefix: str) -> bool:
    normalized = category.upper()
    aliases = {
        "RESISTOR": "R",
        "CAPACITOR": "C",
        "INDUCTOR": "L",
        "DIODE": "D",
        "IC": "U",
        "BOX": "*",
    }
    expected = aliases.get(normalized, normalized[:1])
    if expected == "*":
        return True
    return prefix.startswith(expected)


def _canonical_designator(value: str, category: str) -> str:
    """Correct only the observed doubled-prefix OCR failure mode.

    The recognizer frequently emits ``CC13``/``RR8`` for capacitor/resistor
    references. Collapse the repeated first letter only when the component
    category independently supports that prefix; all other OCR strings remain
    unchanged. This avoids inventing identifiers for generic Box detections.
    """

    normalized = value.strip().upper()
    match = re.fullmatch(r"([A-Z])\1([0-9]+[A-Z]?)", normalized)
    if match is None:
        return value
    prefix = match.group(1)
    if not _prefix_compatible(category, prefix):
        return value
    return f"{prefix}{match.group(2)}"


_COMPONENT_TEXT_FIELDS = {
    "designator": "designator",
    "refdes": "designator",
    "reference": "designator",
    "value": "value",
    "searchname": "searchName",
    "partname": "searchName",
}


def associate_component_texts(
    components: list[Component],
    texts: list[TextToken],
    diagnostics: list[Diagnostic],
    config: OCRAssociationConfig,
) -> dict[str, Any]:
    """Associate typed OCR tokens to component fields without guessing.

    Existing fields, including Golden ``DATASET_*`` debug surrogates, are
    immutable inputs to this step.  Every target ranking and unresolved field
    choice is retained as rule evidence.
    """

    component_index = {component.internal_id: component for component in components}
    rows_by_field: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    relevant_tokens = 0
    accepted_targets = 0
    ambiguous_targets = 0

    for token in sorted(texts, key=lambda item: item.internal_id):
        field = _COMPONENT_TEXT_FIELDS.get(canonical_details(token.details))
        value = preferred_text(token)
        if field is None or value is None:
            continue
        relevant_tokens += 1
        explicit = token.component_id in component_index
        pool = (
            [component_index[str(token.component_id)]]
            if explicit
            else sorted(components, key=lambda item: item.internal_id)
        )
        candidates: list[RankedAssociationCandidate] = []
        candidate_values: dict[str, str] = {}
        for component in pool:
            candidate_value = (
                _canonical_designator(value, component.category)
                if field == "designator"
                else value
            )
            candidate_values[component.internal_id] = candidate_value
            scale = max(component.bbox.width, component.bbox.height, 1.0)
            gap = bbox_gap_distance(token.bbox, component.bbox)
            normalized_distance = gap / scale
            if not explicit and normalized_distance > config.component_max_distance_factor:
                continue
            semantic = 0.0
            if field == "designator":
                prefix = re.match(r"^[A-Z]+", candidate_value)
                semantic = (
                    1.0
                    if prefix and _prefix_compatible(component.category, prefix.group(0))
                    else 0.0
                )
            proximity = max(
                0.0,
                1.0 - normalized_distance / config.component_max_distance_factor,
            )
            score = (
                float(token.score)
                if explicit
                else 0.70 * float(token.score) + 0.25 * proximity + 0.05 * semantic
            )
            candidates.append(
                RankedAssociationCandidate(
                    target_id=component.internal_id,
                    score=min(1.0, score),
                    distance=gap,
                    explicit=explicit,
                    semantic_compatibility=semantic,
                )
            )
        candidates.sort(key=lambda item: (-item.score, item.distance, item.target_id))
        status, best, margin = choose_unique_target(
            candidates,
            min_score=config.component_min_score,
            unique_margin=config.component_unique_margin,
        )
        if status == "accepted" and best is not None:
            accepted_targets += 1
            if token.component_id is None and (explicit or config.bind_global_component_targets):
                token.component_id = best.target_id
        elif status == "ambiguous":
            ambiguous_targets += 1
            append_diagnostic(
                diagnostics,
                severity="INFO",
                code="ASSOCIATION.COMPONENT_TARGET_AMBIGUOUS",
                message=f"text {token.internal_id} has no unique component target",
                object_ids=[token.internal_id, *[item.target_id for item in candidates[:2]]],
            )

        eligible = not (field == "designator" and not any(
            _is_valid_designator(value, comp.category)
            for comp in pool
        ))
        if field == "designator" and not eligible:
            append_diagnostic(
                diagnostics,
                severity="INFO",
                code="ASSOCIATION.DESIGNATOR_FORMAT_REJECTED",
                message=f"text {token.internal_id} is not a conservative reference designator: {value!r}",
                object_ids=[token.internal_id],
            )

        candidate_records = [item.to_dict() for item in candidates]
        token_evidence = make_rule_evidence(
            evidence_id=f"association:component-target:{token.internal_id}",
            score=best.score if best is not None else token.score,
            payload_ref=token.internal_id,
            metadata={
                "kind": "text-to-component",
                "field": field,
                "value": value,
                "lexical_candidates": lexical_candidates(token),
                "decision": status,
                "accepted_target_id": best.target_id if status == "accepted" and best else None,
                "target_margin": margin,
                "thresholds": {
                    "min_score": config.component_min_score,
                    "unique_margin": config.component_unique_margin,
                    "max_distance_factor": config.component_max_distance_factor,
                },
                "target_candidates": candidate_records[: config.max_recorded_candidates],
            },
        )
        upsert_evidence(token.evidence, token_evidence)

        for rank, candidate in enumerate(candidates, start=1):
            rows_by_field[(candidate.target_id, field)].append(
                {
                    "text_id": token.internal_id,
                    "source_object_id": token.source_object_id,
                    "value": candidate_values.get(candidate.target_id, value),
                    "lexical_candidates": lexical_candidates(token),
                    "score": candidate.score,
                    "distance": candidate.distance,
                    "explicit": candidate.explicit,
                    "target_rank": rank,
                    "target_status": status if rank == 1 else "alternative_target",
                    "eligible": eligible,
                    "token_evidence_id": token_evidence.evidence_id,
                }
            )

    evaluations: dict[tuple[str, str], dict[str, Any]] = {}
    for component in sorted(components, key=lambda item: item.internal_id):
        for field in ("designator", "value", "searchName"):
            rows = sorted(
                rows_by_field.get((component.internal_id, field), []),
                key=lambda item: (-float(item["score"]), str(item["text_id"])),
            )
            if not rows:
                continue
            existing = (
                component.key
                if field == "designator"
                else component.value
                if field == "value"
                else component.name
            )
            if existing:
                decision, chosen, score, margin = "preserved_existing", str(existing), 1.0, None
            else:
                # Component/side-strip OCR carries an explicit component owner.
                # When a designator candidate has that provenance, unowned
                # global OCR from the same image is a weaker duplicate view and
                # must not override it during field selection. Other fields
                # retain the existing mixed-candidate behavior.
                choice_rows = rows
                if field == "designator":
                    explicit_rows = [
                        row
                        for row in rows
                        if bool(row.get("explicit"))
                        and row.get("target_status") == "accepted"
                        and row.get("eligible") is not False
                    ]
                    if explicit_rows:
                        choice_rows = explicit_rows
                decision, chosen, score, margin = field_choice(
                    choice_rows,
                    min_score=config.component_min_score,
                    unique_margin=config.component_unique_margin,
                )
            evaluations[(component.internal_id, field)] = {
                "component": component,
                "field": field,
                "rows": rows,
                "existing": existing or None,
                "decision": decision,
                "chosen": chosen,
                "score": score,
                "margin": margin,
            }

    # A reference designator is a global image key.  Tentative assignments that
    # collide with an existing or another tentative key are all withheld.
    used_designators = {
        component.key.casefold() for component in components if component.key
    }
    tentative_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (component_id, field), evaluation in evaluations.items():
        if field != "designator" or evaluation["decision"] != "accepted":
            continue
        chosen = evaluation["chosen"]
        if isinstance(chosen, str):
            tentative_by_key[chosen.casefold()].append(evaluation)
    for normalized, group in tentative_by_key.items():
        explicit_group = [
            evaluation
            for evaluation in group
            if any(
                bool(row.get("explicit"))
                and row.get("target_status") == "accepted"
                and str(row.get("value", "")).casefold() == normalized
                for row in evaluation["rows"]
            )
        ]
        if explicit_group:
            for evaluation in group:
                if evaluation not in explicit_group:
                    evaluation["decision"] = "global_key_conflict"
                    evaluation["chosen"] = None
            group = explicit_group
        if normalized in used_designators:
            for evaluation in group:
                evaluation["decision"] = "global_key_conflict"
                evaluation["chosen"] = None
        elif len(group) > 1:
            # Multiple components claim the same visible designator.  Allow it
            # when every member has independent explicit OCR evidence — the
            # serializer deterministically adds _1/_2 suffixes for official output.
            all_explicit = all(
                any(
                    bool(row.get("explicit"))
                    and row.get("target_status") == "accepted"
                    and row.get("eligible") is not False
                    and str(row.get("value", "")).casefold() == normalized
                    for row in evaluation["rows"]
                )
                for evaluation in group
            )
            if not all_explicit:
                for evaluation in group:
                    evaluation["decision"] = "global_key_conflict"
                    evaluation["chosen"] = None

    assignments: dict[str, dict[str, str]] = defaultdict(dict)
    ambiguous_fields = 0
    for (component_id, field), evaluation in sorted(evaluations.items()):
        component = evaluation["component"]
        decision = str(evaluation["decision"])
        chosen = evaluation["chosen"]
        if decision == "accepted" and isinstance(chosen, str):
            if field == "designator":
                component.key = chosen
                component.key_score = float(evaluation["score"])
            elif field == "value":
                component.value = chosen
            else:
                component.name = chosen
            assignments[component_id][field] = chosen
        elif decision in {"ambiguous", "global_key_conflict"}:
            ambiguous_fields += 1
            append_diagnostic(
                diagnostics,
                severity="INFO",
                code=(
                    "ASSOCIATION.COMPONENT_KEY_CONFLICT"
                    if decision == "global_key_conflict"
                    else "ASSOCIATION.COMPONENT_FIELD_AMBIGUOUS"
                ),
                message=f"component {component_id} field {field} remains unresolved ({decision})",
                object_ids=[component_id, *[str(row["text_id"]) for row in evaluation["rows"][:3]]],
            )

        evidence = make_rule_evidence(
            evidence_id=f"association:component-field:{component_id}:{field}",
            score=float(evaluation["score"]),
            payload_ref=component_id,
            metadata={
                "kind": "component-field-resolution",
                "field": field,
                "decision": decision,
                "existing_value": evaluation["existing"],
                "chosen_value": chosen,
                "field_margin": evaluation["margin"],
                "thresholds": {
                    "min_score": config.component_min_score,
                    "unique_margin": config.component_unique_margin,
                    "max_distance_factor": config.component_max_distance_factor,
                },
                "candidates": evaluation["rows"][: config.max_recorded_candidates],
            },
        )
        upsert_evidence(component.evidence, evidence)

    return {
        "relevant_tokens": relevant_tokens,
        "accepted_targets": accepted_targets,
        "ambiguous_targets": ambiguous_targets,
        "assigned_fields": sum(len(value) for value in assignments.values()),
        "ambiguous_fields": ambiguous_fields,
        "assignments": {key: dict(value) for key, value in sorted(assignments.items())},
    }


__all__ = ["associate_component_texts", "resolve_component_keys"]
