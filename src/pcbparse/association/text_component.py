from __future__ import annotations

from pcbparse.association.reference_prefixes import DESIGNATOR_PATTERN, prefix_compatible

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
    point_bbox_distance,
    preferred_text,
    resolved_text,
    upsert_evidence,
)
from pcbparse.ir.models import Component, Diagnostic, TextToken


_DESIGNATOR = re.compile(DESIGNATOR_PATTERN)
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


def _assign_qna_special_coordinate_keys(components: list[Component]) -> list[str]:
    """Assign the official coordinate identity to otherwise keyless VCC/GND symbols.

    Q&A states that VCC/GND use the empty-set key and match by coordinates;
    GND instance numbering does not affect scoring. This rule never touches
    visible OCR text, pin numbers, or components with an existing key.
    """
    assigned: list[str] = []
    for component in components:
        if component.category not in {"Gnd", "V"}:
            continue
        if isinstance(component.key, str) and component.key.strip():
            continue
        component.key = "\u2205"
        upsert_evidence(
            component.evidence,
            make_rule_evidence(
                evidence_id=f"association:qna-special-coordinate-key:{component.internal_id}",
                score=float(component.score),
                payload_ref=component.internal_id,
                metadata={
                    "kind": "qna-special-coordinate-identity",
                    "official_key": "\u2205",
                    "component_category": component.category,
                    "matching": "coordinate",
                    "qna_reference": "VCC/GND coordinate matching; GND instance name is unscored",
                },
            ),
        )
        assigned.append(component.internal_id)
    return assigned


def resolve_component_keys(
    components: list[Component],
    texts: list[TextToken],
    *,
    max_distance_factor: float = 2.0,
    unique_margin: float = 0.10,
    min_score: float = 0.0,
    alternative_guard: str = "unique_margin",
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
    rewrite_by_token: dict[str, dict[str, Any]] = {}
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
                    _, rewrite = _canonical_designator(value, category)
                    if rewrite is not None:
                        rewrite_by_token[text.internal_id] = rewrite
    # Explicit component-owned OCR is stronger than a duplicate global token with the same spelling.
    explicit_values = {
        _canonical_designator(value, component_index[owner].category)[0].casefold()
        for _, value, owner in candidates
        if owner is not None and owner in component_index
    }
    scored: list[tuple[float, str, str, bool]] = []
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
            canonical_value, rewrite = _canonical_designator(value, component.category)
            if rewrite is not None:
                rewrite_by_token.setdefault(token.internal_id, rewrite)
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
            scored.append((min(1.0, score), component.internal_id, canonical_value, owner is not None))
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
    # Track which (component, key) pairs have explicit owner evidence
    explicit_owner_edges: set[tuple[str, str]] = set()
    for score, component_id, key, has_explicit_owner in scored:
        if key in used_keys:
            continue
        pair = (component_id, key)
        edge_candidates[pair] = max(edge_candidates.get(pair, -1.0), score)
        if has_explicit_owner:
            explicit_owner_edges.add(pair)
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
    # Track keys that were assigned from explicit owner evidence — duplicate
    # designators are allowed when both components have independent explicit evidence.
    explicitly_assigned_keys: set[str] = set()
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
            # Allow duplicate designators when the current component has
            # explicit owner evidence AND all competing assignments also came
            # from explicit owner evidence.  This means two components each
            # independently observed the same visible label — the serializer
            # handles KEY_1/KEY_2 suffixing.
            current_has_explicit = (component_id, key) in explicit_owner_edges
            all_competitors_explicit = all(
                (comp_id, key) in explicit_owner_edges
                for comp_id, comp_key in edge_candidates
                if comp_key == key and comp_id != component_id
            )
            if current_has_explicit and all_competitors_explicit:
                pass  # Allow duplicate
            else:
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
        # The default guard defers an assignment whenever the component still
        # has a near-tie alternative candidate.  Because this loop keeps going,
        # the deferred edge is *not* a real abstention: the component is left to
        # be filled by a strictly worse edge later in score order.  Under
        # ``argument_max`` the component keeps its best argument unless another
        # candidate for the *same* component is strictly better.
        if alternative_guard == "argument_max":
            if alternatives and score < alternatives[0]:
                continue
        elif alternatives and score - alternatives[0] < unique_margin:
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
        newly_used_keys.add(key)
        # Record designator rewrite evidence while keeping raw token text.
        for token, value, owner in candidates:
            if owner is not None and owner != component_id:
                continue
            canonical_value, rewrite = _canonical_designator(value, component.category)
            if canonical_value != key:
                continue
            if rewrite is not None:
                make = make_rule_evidence(
                    evidence_id=f"association:designator-rewrite:{token.internal_id}",
                    score=score,
                    payload_ref=token.internal_id,
                    metadata={
                        "kind": "designator-rewrite",
                        "component_id": component_id,
                        "assigned_key": key,
                        "token_raw_text": token.raw_text,
                        **rewrite,
                    },
                )
                upsert_evidence(token.evidence, make)
                rewrite_by_token[token.internal_id] = rewrite
            break
        if (component_id, key) in explicit_owner_edges:
            explicitly_assigned_keys.add(key)
    return assignments


def _prefix_compatible(category: str, prefix: str) -> bool:
    return prefix_compatible(category, prefix)


def _canonical_designator(value: str, category: str) -> tuple[str, dict[str, Any] | None]:
    """Correct observed OCR failure modes using component category as constraint.

    Handles two correction strategies (PCBnet component-aware pairing):
    1. **Doubled-prefix collapse** — ``CC13``→``C13``, ``RR8``→``R8`` when the
       component category supports that prefix.
    2. **Confusion-pair prefix correction** — when the prefix is incompatible
       with the component category, try confusion-pair substitution (``0``→``O``,
       ``1``→``I``, ``8``→``B``, etc.) to find a compatible prefix.

    Returns ``(canonical, rewrite_record)``.  When a rewrite is applied the
    record captures the raw input so the original spelling stays recoverable;
    ``TextToken.raw_text`` is never modified by this helper.
    """

    stripped = value.strip()
    uppered = stripped.upper()
    # Strategy 1: doubled-prefix collapse.
    match = re.fullmatch(r"([A-Z])\1([0-9]+[A-Z]?)", uppered)
    if match is not None:
        prefix = match.group(1)
        if _prefix_compatible(category, prefix):
            canonical = f"{prefix}{match.group(2)}"
            if canonical != value:
                return canonical, {
                    "raw_value": value,
                    "canonical_value": canonical,
                    "rule": "doubled_prefix_collapse",
                    "strip_applied": stripped != value,
                    "upper_applied": uppered != stripped,
                }
            return canonical, None
    # Strategy 2: component-aware confusion-pair prefix correction.
    match2 = re.fullmatch(r"([A-Za-z])([0-9]+[A-Z]?)", uppered)
    if match2 is not None:
        prefix = match2.group(1)
        suffix = match2.group(2)
        if not _prefix_compatible(category, prefix):
            from pcbparse.association.ocr_postprocess import _CONFUSION_TO_LETTER

            corrected_prefix = _CONFUSION_TO_LETTER.get(prefix)
            if (
                corrected_prefix
                and corrected_prefix != prefix
                and _prefix_compatible(category, corrected_prefix)
            ):
                canonical = f"{corrected_prefix}{suffix}"
                return canonical, {
                    "raw_value": value,
                    "canonical_value": canonical,
                    "rule": "component_aware_confusion_correction",
                    "confused_char": prefix,
                    "corrected_char": corrected_prefix,
                    "category": category,
                    "strip_applied": stripped != value,
                    "upper_applied": uppered != stripped,
                }
    if stripped != value:
        return stripped, {
            "raw_value": value,
            "canonical_value": stripped,
            "rule": "outer_strip_only",
            "strip_applied": True,
            "upper_applied": False,
        }
    return value, None


_COMPONENT_TEXT_FIELDS = {
    "designator": "designator",
    "refdes": "designator",
    "reference": "designator",
    "value": "value",
    "searchname": "searchName",
    "partname": "searchName",
}


def _is_box_upper_reference(component: Component, token: TextToken, value: str) -> bool:
    if component.category.upper() not in {"BOX", "BOXIC", "BOX_IC", "IC"}:
        return False
    if token.score < .6 or not re.fullmatch(r"(?:U|IC|J|P|CN|CON|JP)[0-9]+[A-Z]?", value):
        return False
    body, box = component.bbox, token.bbox
    return (box.y1 < body.y1 and box.y2 <= body.y1 + min(box.height, body.height * .15)
            and body.x1 - box.width * .5 <= box.center.x <= body.x2 + box.width * .5
            and bbox_gap_distance(box, body) <= max(body.width, body.height) * .25)


def _box_external_designator_rows(
    component: Component, rows: list[dict[str, Any]], text_index: dict[str, TextToken],
) -> list[dict[str, Any]]:
    """A unique visible upper-edge reference outranks numeric body pin text.

    Only accepted candidates for this very component are reconsidered. No
    explicit owner is moved and no text is manufactured or rewritten.
    """
    if component.category.upper() not in {"BOX", "BOXIC", "BOX_IC", "IC"}:
        return []
    anchors = []
    for row in rows:
        token = text_index[str(row["text_id"])]
        if row.get("target_status") != "accepted" or row.get("eligible") is False:
            continue
        if _is_box_upper_reference(component, token, str(row.get("value", ""))):
            anchors.append(row)
    # Multiple different upper-edge readings remain under the original rules.
    return anchors if len({row["value"] for row in anchors}) == 1 else []


def _designator_center_tiebreak(
    candidates: list[RankedAssociationCandidate],
    best: RankedAssociationCandidate,
    token: TextToken,
    component_index: dict[str, Component],
) -> tuple[RankedAssociationCandidate, dict[str, Any] | None]:
    """Use center geometry only when the original target ranking is tied.

    Overlapping text boxes can have zero gap to two adjacent bodies, leaving
    stable IDs to decide ownership. Different categories, body axes or rows
    remain under the original ranking; no score, margin or owner is invented.
    """
    tied = [item for item in candidates if item.score == best.score and item.distance == best.distance]
    bodies = [component_index[item.target_id] for item in tied]
    categories = {body.category.upper() for body in bodies}
    axes = {(body.bbox.width > body.bbox.height) - (body.bbox.width < body.bbox.height) for body in bodies}
    if (
        len(tied) < 2
        or len(categories) != 1
        or "BOX" in categories
        or any(item.explicit or item.semantic_compatibility != 1.0 for item in tied)
        or len(axes) != 1
        or 0 in axes
        or max(body.bbox.y1 for body in bodies) >= min(body.bbox.y2 for body in bodies)
    ):
        return best, None
    ranked = sorted(
        ((point_bbox_distance(token.bbox.center, component_index[item.target_id].bbox), item) for item in tied),
        key=lambda item: (item[0], item[1].target_id),
    )
    if ranked[0][0] == ranked[1][0] or ranked[0][1].target_id == best.target_id:
        return best, None
    winner = ranked[0][1]
    candidates.remove(winner)
    candidates.insert(0, winner)
    return winner, {
        "method": "aligned_designator_center_exact_tie",
        "previous_target_id": best.target_id,
        "chosen_target_id": winner.target_id,
        "original_score": best.score,
        "original_bbox_gap": best.distance,
        "center_to_bbox_distances": {item.target_id: distance for distance, item in ranked},
    }


def _cross_view_designator_rows(component, rows, text_index):
    """Resolve a strong global reading of the same glyphs as a weak ROI reading.

    ROI ownership remains fixed. Only content confidence is reconsidered;
    unrelated global text cannot displace explicit component-owned evidence.
    """
    explicit = [r for r in rows if r.get("explicit") and r.get("target_status") == "accepted"
                and r.get("eligible") is not False]
    if not explicit:
        return []
    winners = []
    for row in rows:
        if row.get("explicit") or row.get("target_status") != "accepted" or row.get("eligible") is False:
            continue
        token = text_index[str(row["text_id"])]
        value = str(row["value"])
        prefix = re.match(r"^[A-Z]+", value)
        if (token.score < .95 or not _DESIGNATOR.fullmatch(value) or not prefix
                or not _prefix_compatible(component.category, prefix.group(0))):
            continue
        competing = [r for r in explicit if r["value"] != value]
        if not competing:
            continue
        for other in competing:
            rival = text_index[str(other["text_id"])]
            a, b = token.bbox, rival.bbox
            overlap = max(0., min(a.x2, b.x2) - max(a.x1, b.x1)) * max(0., min(a.y2, b.y2) - max(a.y1, b.y1))
            small, large = sorted((a.width * a.height, b.width * b.height))
            if (token.score - rival.score < .10 or small <= 0 or large <= 0
                    or overlap / small < .8 or small / large < .25):
                break
        else:
            winners.append(row)
    return winners if len({r["value"] for r in winners}) == 1 else []


def _punctuation_designator_target(token, value, components):
    """Recover an observed colon-prefixed ordinary reference near a unique body."""
    match = re.fullmatch(r":([RCLDQ][0-9]+[A-Z]?)", value)
    if (not match or token.score < .9 or token.pin_id is not None
            or any(e.source == "dataset" for e in token.evidence)
            or not any(e.metadata.get("kind") == "semantic-field-inference" for e in token.evidence)):
        return None
    # Side-strip OCR commonly observes IC pin names: never reinterpret it.
    if any(e.metadata.get("level") == "side_strip" for e in token.evidence):
        return None
    reference = match.group(1)
    targets = []
    for component in components:
        if (component.category.upper() in {"BOX", "BOXIC", "BOX_IC", "IC"}
                or (token.component_id is not None and token.component_id != component.internal_id)
                or not _prefix_compatible(component.category, reference[0])):
            continue
        b, center = component.bbox, token.bbox.center
        if b.x1 <= center.x <= b.x2 and b.y1 <= center.y <= b.y2:
            continue
        gap = bbox_gap_distance(token.bbox, b)
        if gap <= min(24., max(b.width, b.height) * .5):
            targets.append((gap, component.internal_id, component))
    targets.sort(key=lambda r: (r[0], r[1]))
    if not targets or (len(targets) > 1 and targets[1][0] - targets[0][0] <= 2.):
        return None
    return reference, targets[0][2]


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
    text_index = {token.internal_id: token for token in texts}
    rows_by_field: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    relevant_tokens = 0
    accepted_targets = 0
    ambiguous_targets = 0
    rewrite_records: list[dict[str, Any]] = []

    for token in sorted(texts, key=lambda item: item.internal_id):
        field = _COMPONENT_TEXT_FIELDS.get(canonical_details(token.details))
        value = preferred_text(token)
        resolved = resolved_text(token)
        recovery_targets = []
        punctuation_recovery = None
        if config.component_punctuation_designator_enabled and value:
            punctuation_recovery = _punctuation_designator_target(token, value, components)
            if punctuation_recovery is not None:
                value, target = punctuation_recovery
                recovery_targets = [target]
                field = "designator"
                original_details = token.details
                token.details = token.category = "designator"
        if (config.box_external_designator_preference_enabled and field is None and value
                and not any(item.source == "dataset" for item in token.evidence)):
            recovery_targets = [component for component in components
                                if (token.component_id is None or token.component_id == component.internal_id)
                                and _is_box_upper_reference(component, token, value)]
            if len(recovery_targets) == 1:
                field = "designator"
            else:
                recovery_targets = []
        if field is None or value is None:
            continue
        relevant_tokens += 1
        explicit = token.component_id in component_index
        pool = (
            recovery_targets if recovery_targets else
            [component_index[str(token.component_id)]]
            if explicit
            else sorted(components, key=lambda item: item.internal_id)
        )
        candidates: list[RankedAssociationCandidate] = []
        candidate_values: dict[str, str] = {}
        candidate_rewrites: dict[str, dict[str, Any] | None] = {}
        for component in pool:
            if field == "designator":
                candidate_value, rewrite = _canonical_designator(value, component.category)
                if rewrite is not None:
                    rewrite_records.append(
                        {
                            "text_id": token.internal_id,
                            "component_id": component.internal_id,
                            "field": field,
                            **rewrite,
                        }
                    )
            else:
                candidate_value, rewrite = value, None
            candidate_values[component.internal_id] = candidate_value
            candidate_rewrites[component.internal_id] = rewrite
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
        geometry_tiebreak = None
        if (
            config.component_designator_center_tiebreak_enabled
            and field == "designator"
            and status == "accepted"
            and best is not None
            and not explicit
        ):
            best, geometry_tiebreak = _designator_center_tiebreak(
                candidates, best, token, component_index
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
        token_rewrite = next(
            (
                rewrite
                for rewrite in rewrite_records
                if rewrite.get("text_id") == token.internal_id
            ),
            None,
        )
        token_evidence = make_rule_evidence(
            evidence_id=f"association:component-target:{token.internal_id}",
            score=best.score if best is not None else token.score,
            payload_ref=token.internal_id,
            metadata={
                "kind": "text-to-component",
                "field": field,
                # Bind both spellings; output-facing "value" stays the raw form.
                "value": value,
                "raw_text": token.raw_text,
                "resolved_value": resolved,
                "lexical_candidates": lexical_candidates(token),
                "rewrite": token_rewrite,
                **({"role_recovery": "colon_prefixed_ordinary_reference", "original_details": original_details,
                    "raw_value": token.raw_text} if punctuation_recovery else
                   {"role_recovery": "box_upper_edge_reference", "original_details": token.details} if recovery_targets else {}),
                "decision": status,
                "accepted_target_id": best.target_id if status == "accepted" and best else None,
                "target_margin": margin,
                "thresholds": {
                    "min_score": config.component_min_score,
                    "unique_margin": config.component_unique_margin,
                    "max_distance_factor": config.component_max_distance_factor,
                },
                "target_candidates": candidate_records[: config.max_recorded_candidates],
                **({"geometry_tiebreak": geometry_tiebreak} if geometry_tiebreak is not None else {}),
            },
        )
        upsert_evidence(token.evidence, token_evidence)

        for rank, candidate in enumerate(candidates, start=1):
            rows_by_field[(candidate.target_id, field)].append(
                {
                    "text_id": token.internal_id,
                    "source_object_id": token.source_object_id,
                    "raw_text": token.raw_text,
                    "value": candidate_values.get(candidate.target_id, value),
                    "raw_value": value,
                    "resolved_value": resolved,
                    "rewrite": candidate_rewrites.get(candidate.target_id),
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
            selection_rule = None
            if existing:
                decision, chosen, score, margin = "preserved_existing", str(existing), 1.0, None
            else:
                # Component/side-strip OCR carries an explicit component owner.
                # When a designator candidate has that provenance, unowned
                # global OCR from the same image is a weaker duplicate view and
                # must not override it during field selection. Other fields
                # retain the existing mixed-candidate behavior.
                choice_rows = rows
                if field == "designator" and config.box_external_designator_preference_enabled:
                    anchors = _box_external_designator_rows(component, rows, text_index)
                    if anchors:
                        choice_rows = anchors
                        selection_rule = "unique_box_upper_edge_reference"
                if field == "designator" and not config.key_evidence_pool_enabled:
                    explicit_rows = [
                        row
                        for row in choice_rows
                        if bool(row.get("explicit"))
                        and row.get("target_status") == "accepted"
                        and row.get("eligible") is not False
                    ]
                    if explicit_rows:
                        cross_view = (_cross_view_designator_rows(component, choice_rows, text_index)
                                      if config.component_cross_view_designator_enabled and selection_rule is None else [])
                        if cross_view:
                            choice_rows = cross_view
                            selection_rule = "strong_same_glyph_global_reference"
                        else:
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
                **({"selection_rule": selection_rule} if selection_rule else {}),
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
                **({"selection_rule": evaluation["selection_rule"]} if evaluation.get("selection_rule") else {}),
                "thresholds": {
                    "min_score": config.component_min_score,
                    "unique_margin": config.component_unique_margin,
                    "max_distance_factor": config.component_max_distance_factor,
                },
                "candidates": evaluation["rows"][: config.max_recorded_candidates],
            },
        )
        upsert_evidence(component.evidence, evidence)

    _assign_qna_special_coordinate_keys(components)
    return {
        "relevant_tokens": relevant_tokens,
        "accepted_targets": accepted_targets,
        "ambiguous_targets": ambiguous_targets,
        "assigned_fields": sum(len(value) for value in assignments.values()),
        "ambiguous_fields": ambiguous_fields,
        "designator_rewrites": rewrite_records,
        "assignments": {key: dict(value) for key, value in sorted(assignments.items())},
    }


__all__ = ["associate_component_texts", "resolve_component_keys"]
