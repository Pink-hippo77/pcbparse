from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import math
import re
from typing import Any

from pcbparse.association.ocr_common import (
    OCRAssociationConfig,
    append_diagnostic,
    canonical_details,
    field_choice,
    lexical_candidates,
    make_rule_evidence,
    point_bbox_distance,
    preferred_text,
    resolved_text,
    upsert_evidence,
)
from pcbparse.association.pin_assignment import (
    PinAssignmentCandidate,
    solve_pin_assignment_group,
)
from pcbparse.ir.models import BBox, Component, Diagnostic, Pin, TextToken


_PIN_TEXT_FIELDS = {
    "pinnumber": "number",
    "pindesignator": "number",
    "pinname": "name",
}
# Pure-digit pin numbers stay on the conservative path.  Alphanumeric pin
# labels (A5, B12, USB_D-) use this separate lexeme and are only accepted
# through pin-context matching, never by loosening _PIN_NUMBER alone.
_NUMBER_LEXEME = re.compile(r"^[^\s,()]+$")
_ALNUM_PIN_NUMBER = re.compile(r"^(?:[A-Z]{1,2}[0-9]{1,3}|[0-9]{1,3}[A-Z]{1,2})$")
_SIDES = {"L", "R", "T", "B"}


def associate_pin_texts(
    pins: list[Pin],
    texts: list[TextToken],
    diagnostics: list[Diagnostic],
    config: OCRAssociationConfig,
) -> dict[str, Any]:
    """Conservatively resolve typed pin OCR with global one-to-one matching.

    Matching is performed independently for each field and physical
    component-side group.  This prevents a locally attractive token from
    consuming the only viable pin for a neighbouring token.  Existing pin
    fields always win and all rejected candidates remain available in
    Evidence for debugging.
    """

    pin_index = {pin.internal_id: pin for pin in pins}
    token_index = {token.internal_id: token for token in texts}
    contexts: dict[str, dict[str, Any]] = {}
    grouped_edges: dict[
        tuple[str, tuple[str | None, str]], list[PinAssignmentCandidate]
    ] = defaultdict(list)
    relevant_tokens = 0

    # Candidate construction is deterministic, but neither input list is
    # sorted or replaced: callers retain their original object order.
    for token in sorted(texts, key=lambda item: item.internal_id):
        field = _PIN_TEXT_FIELDS.get(canonical_details(token.details))
        value = preferred_text(token)
        resolved = resolved_text(token)
        if field is None or value is None:
            continue
        # Bind raw and resolved spellings for matching; output keeps raw.
        bound_values = lexical_candidates(token)
        relevant_tokens += 1
        explicit = token.pin_id in pin_index
        if explicit:
            pool = [pin_index[str(token.pin_id)]]
        elif token.component_id:
            pool = sorted(
                (pin for pin in pins if pin.component_id == token.component_id),
                key=lambda item: item.internal_id,
            )
        else:
            pool = sorted(pins, key=lambda item: item.internal_id)

        candidates = [
            candidate
            for pin in pool
            if (
                candidate := _score_candidate(
                    token,
                    pin,
                    field=field,
                    value=value,
                    bound_values=bound_values,
                    explicit=explicit,
                    config=config,
                )
            )
            is not None
        ]
        candidates.sort(key=_candidate_sort_key)
        context: dict[str, Any] = {
            "token": token,
            "field": field,
            "value": value,
            "raw_text": token.raw_text,
            "resolved_value": resolved,
            "bound_values": bound_values,
            "explicit": explicit,
            "candidates": candidates,
            "status": "unmatched",
            "selected_pin_id": None,
            "target_margin": None,
            "group_margin": None,
            "assignment_method": None,
            "group_total_score": None,
            "group_next_best_score": None,
        }
        contexts[token.internal_id] = context

        eligible = [item for item in candidates if item.score >= config.pin_min_score]
        if not candidates:
            context["status"] = "unmatched"
            continue
        if not eligible:
            context["status"] = "below_threshold"
            continue

        if explicit:
            selected_group = eligible[0].group_key
        else:
            group_best: dict[tuple[str | None, str], PinAssignmentCandidate] = {}
            for candidate in eligible:
                previous = group_best.get(candidate.group_key)
                if previous is None or _candidate_sort_key(candidate) < _candidate_sort_key(previous):
                    group_best[candidate.group_key] = candidate
            ranked_groups = sorted(
                group_best.items(),
                key=lambda item: (*_candidate_sort_key(item[1]), _group_sort_key(item[0])),
            )
            selected_group = ranked_groups[0][0]
            if len(ranked_groups) > 1:
                group_margin = ranked_groups[0][1].score - ranked_groups[1][1].score
                context["group_margin"] = group_margin
                if group_margin < config.pin_unique_margin:
                    context["status"] = "ambiguous"
                    continue

        selected_edges = [item for item in eligible if item.group_key == selected_group]
        if not selected_edges:
            context["status"] = "unmatched"
            continue
        context["selected_group"] = selected_group
        grouped_edges[(field, selected_group)].extend(selected_edges)

    # Solve the complete field/group objective before accepting any target.
    for group_key in sorted(grouped_edges, key=lambda item: (item[0], _group_sort_key(item[1]))):
        group_candidates = grouped_edges[group_key]
        result = solve_pin_assignment_group(
            group_candidates,
            minimum_score=config.pin_min_score,
        )
        token_ids = sorted({item.token_id for item in group_candidates})
        for token_id in token_ids:
            context = contexts[token_id]
            context["assignment_method"] = result.method
            context["group_total_score"] = result.total_score
            context["group_next_best_score"] = result.next_best_score
            pin_id = result.assignments.get(token_id)
            if pin_id is None:
                context["status"] = "ambiguous"
                continue
            margin = result.pair_margins.get((token_id, pin_id), 0.0)
            context["selected_pin_id"] = pin_id
            context["target_margin"] = margin
            context["status"] = (
                "accepted" if margin >= config.pin_unique_margin else "ambiguous"
            )

    rows_by_field: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    accepted_targets = 0
    ambiguous_targets = 0
    unknown_targets = 0
    for token_id in sorted(contexts):
        context = contexts[token_id]
        token = context["token"]
        field = str(context["field"])
        value = str(context["value"])
        candidates: list[PinAssignmentCandidate] = context["candidates"]
        status = str(context["status"])
        selected_pin_id = context["selected_pin_id"]
        selected = next(
            (item for item in candidates if item.pin_id == selected_pin_id),
            candidates[0] if candidates else None,
        )

        if status == "accepted" and selected_pin_id is not None:
            accepted_targets += 1
            selected_pin = pin_index[selected_pin_id]
            if token.pin_id is None:
                token.pin_id = selected_pin_id
            if token.component_id is None and selected_pin.component_id is not None:
                token.component_id = selected_pin.component_id
        elif status == "ambiguous":
            ambiguous_targets += 1
            append_diagnostic(
                diagnostics,
                severity="INFO",
                code="ASSOCIATION.PIN_TARGET_AMBIGUOUS",
                message=f"text {token_id} has no unique global pin target",
                object_ids=[token_id, *[item.pin_id for item in candidates[:2]]],
            )
        else:
            unknown_targets += 1
            append_diagnostic(
                diagnostics,
                severity="INFO",
                code="ASSOCIATION.PIN_TARGET_UNKNOWN",
                message=f"text {token_id} pin target remains UNKNOWN ({status})",
                object_ids=[token_id, *[item.pin_id for item in candidates[:1]]],
            )

        candidate_records = [
            _candidate_record(
                item,
                rank=rank,
                status=(
                    "accepted"
                    if status == "accepted" and item.pin_id == selected_pin_id
                    else status
                    if rank == 1 and status != "accepted"
                    else "alternative_target"
                ),
                assignment_method=context["assignment_method"],
            )
            for rank, item in enumerate(candidates, start=1)
        ]
        token_evidence = make_rule_evidence(
            evidence_id=f"association:pin-target:{token_id}",
            score=selected.score if selected is not None else token.score,
            payload_ref=token_id,
            metadata={
                "kind": "text-to-pin-global",
                "field": field,
                "value": value,
                "raw_text": context["raw_text"],
                "resolved_value": context["resolved_value"],
                "lexical_candidates": lexical_candidates(token),
                "decision": status,
                "accepted_target_id": (
                    selected_pin_id if status == "accepted" else None
                ),
                "target_margin": context["target_margin"],
                "group_margin": context["group_margin"],
                "assignment_method": context["assignment_method"],
                "group_total_score": context["group_total_score"],
                "group_next_best_score": context["group_next_best_score"],
                "thresholds": {
                    "min_score": config.pin_min_score,
                    "unique_margin": config.pin_unique_margin,
                    "max_distance_px": config.pin_max_distance_px,
                },
                "target_candidates": candidate_records[: config.max_recorded_candidates],
            },
        )
        upsert_evidence(token.evidence, token_evidence)

        for record in candidate_records:
            rows_by_field[(str(record["target_id"]), field)].append(
                {
                    "text_id": token_id,
                    "source_object_id": token.source_object_id,
                    "raw_text": context["raw_text"],
                    "value": value,
                    "resolved_value": context["resolved_value"],
                    "lexical_candidates": lexical_candidates(token),
                    "score": record["score"],
                    "distance": record["distance"],
                    "target_rank": record["target_rank"],
                    "target_status": record["target_status"],
                    "global_margin": (
                        context["target_margin"]
                        if record["target_id"] == selected_pin_id
                        else None
                    ),
                    "assignment_method": context["assignment_method"],
                    "score_components": record["score_components"],
                    "token_evidence_id": token_evidence.evidence_id,
                }
            )

    evaluations: dict[tuple[str, str], dict[str, Any]] = {}
    for pin in sorted(pins, key=lambda item: item.internal_id):
        for field in ("number", "name"):
            rows = sorted(
                rows_by_field.get((pin.internal_id, field), []),
                key=lambda item: (-float(item["score"]), str(item["text_id"])),
            )
            if not rows:
                continue
            existing = pin.number if field == "number" else pin.name
            if existing:
                decision, chosen, score, margin = "preserved_existing", str(existing), 1.0, None
            else:
                decision, chosen, score, margin = field_choice(
                    rows,
                    min_score=config.pin_min_score,
                    unique_margin=config.pin_unique_margin,
                )
                if decision == "no_accepted_candidate" and any(
                    row["target_status"] == "ambiguous" for row in rows
                ):
                    decision = "ambiguous"
                    score = float(rows[0]["score"])
                    margin = rows[0].get("global_margin")
            evaluations[(pin.internal_id, field)] = {
                "pin": pin,
                "field": field,
                "rows": rows,
                "existing": existing or None,
                "decision": decision,
                "chosen": chosen,
                "score": score,
                "margin": margin,
            }

    # Pin numbers are unique only within one component.  Withhold all
    # tentative collisions, including collisions with pre-existing numbers.
    used_numbers: dict[str | None, set[str]] = defaultdict(set)
    for pin in pins:
        if pin.number:
            used_numbers[pin.component_id].add(pin.number.casefold())
    tentative_numbers: dict[tuple[str | None, str], list[dict[str, Any]]] = defaultdict(list)
    for (_, field), evaluation in evaluations.items():
        if field != "number" or evaluation["decision"] != "accepted":
            continue
        chosen = evaluation["chosen"]
        pin = evaluation["pin"]
        if isinstance(chosen, str):
            tentative_numbers[(pin.component_id, chosen.casefold())].append(evaluation)
    for (component_id, normalized), group in tentative_numbers.items():
        if normalized in used_numbers[component_id] or len(group) > 1:
            for evaluation in group:
                evaluation["decision"] = "component_number_conflict"
                evaluation["chosen"] = None

    assignments: dict[str, dict[str, str]] = defaultdict(dict)
    ambiguous_fields = 0
    for (pin_id, field), evaluation in sorted(evaluations.items()):
        pin = evaluation["pin"]
        decision = str(evaluation["decision"])
        chosen = evaluation["chosen"]
        if decision == "accepted" and isinstance(chosen, str):
            if field == "number":
                pin.number = chosen
            else:
                pin.name = chosen
            assignments[pin_id][field] = chosen
        elif decision in {"ambiguous", "component_number_conflict"}:
            ambiguous_fields += 1
            append_diagnostic(
                diagnostics,
                severity="INFO",
                code=(
                    "ASSOCIATION.PIN_NUMBER_CONFLICT"
                    if decision == "component_number_conflict"
                    else "ASSOCIATION.PIN_FIELD_AMBIGUOUS"
                ),
                message=f"pin {pin_id} field {field} remains unresolved ({decision})",
                object_ids=[pin_id, *[str(row["text_id"]) for row in evaluation["rows"][:3]]],
            )

        evidence = make_rule_evidence(
            evidence_id=f"association:pin-field:{pin_id}:{field}",
            score=float(evaluation["score"]),
            payload_ref=pin_id,
            metadata={
                "kind": "pin-field-resolution",
                "field": field,
                "decision": decision,
                "existing_value": evaluation["existing"],
                "chosen_value": chosen,
                "field_margin": evaluation["margin"],
                "thresholds": {
                    "min_score": config.pin_min_score,
                    "unique_margin": config.pin_unique_margin,
                    "max_distance_px": config.pin_max_distance_px,
                },
                "candidates": evaluation["rows"][: config.max_recorded_candidates],
            },
        )
        upsert_evidence(pin.evidence, evidence)

    return {
        "relevant_tokens": relevant_tokens,
        "accepted_targets": accepted_targets,
        "ambiguous_targets": ambiguous_targets,
        "unknown_targets": unknown_targets,
        "assigned_fields": sum(len(value) for value in assignments.values()),
        "ambiguous_fields": ambiguous_fields,
        "assignments": {key: dict(value) for key, value in sorted(assignments.items())},
    }


def _score_candidate(
    token: TextToken,
    pin: Pin,
    *,
    field: str,
    value: str,
    bound_values: list[str],
    explicit: bool,
    config: OCRAssociationConfig,
) -> PinAssignmentCandidate | None:
    distances = [point_bbox_distance(pin.point, token.bbox)]
    if pin.base is not None:
        distances.append(point_bbox_distance(pin.base, token.bbox))
    distance = min(distances)
    if not explicit and distance > config.pin_max_distance_px:
        return None

    side = _pin_side(pin)
    proximity = max(0.0, 1.0 - distance / config.pin_max_distance_px)
    side_score = _side_compatibility(pin, token, side, config.pin_max_distance_px)
    details_score = 1.0 if _PIN_TEXT_FIELDS.get(canonical_details(token.details)) == field else 0.0
    # Score against every bound spelling (raw_text + resolved_text + candidates).
    lexical_score = max(
        (_lexical_compatibility(field, candidate) for candidate in bound_values),
        default=0.0,
    )
    if explicit:
        score = 0.90 * float(token.score) + 0.05 * details_score + 0.05 * lexical_score
    else:
        score = (
            0.65 * float(token.score)
            + 0.22 * proximity
            + 0.08 * side_score
            + 0.05 * lexical_score
        )
    center = token.bbox.center
    token_axis = center.y if side in {"L", "R", "UNKNOWN"} else center.x
    pin_axis = pin.point.y if side in {"L", "R", "UNKNOWN"} else pin.point.x
    score_components = {
        "ocr": float(token.score),
        "proximity": proximity,
        "side_orientation": side_score,
        "details": details_score,
        "lexical": lexical_score,
    }
    # Long function labels can be closer to a neighbouring pin in Euclidean
    # distance. Use their row as additional ranking evidence. Number matching
    # and explicit targets retain their existing paths.
    row_penalty = float(config.pin_name_row_penalty)
    if field == "name" and not explicit and side != "UNKNOWN" and row_penalty > 0:
        low, high = ((token.bbox.y1, token.bbox.y2) if side in {"L", "R"}
                     else (token.bbox.x1, token.bbox.x2))
        gap = max(low - pin_axis, pin_axis - high, 0.0)
        scale = max(2.0, min(token.bbox.width, token.bbox.height) * 0.5)
        alignment = max(0.0, 1.0 - gap / scale)
        score = max(0.0, score - row_penalty * (1.0 - alignment))
        score_components["row_alignment"] = alignment
    return PinAssignmentCandidate(
        token_id=token.internal_id,
        pin_id=pin.internal_id,
        component_id=pin.component_id,
        side=side,
        score=min(1.0, max(0.0, score)),
        distance=distance,
        token_axis=token_axis,
        pin_axis=pin_axis,
        explicit=explicit,
        score_components=score_components,
    )


def _pin_side(pin: Pin) -> str:
    if pin.orientation in _SIDES:
        return pin.orientation
    if pin.base is None:
        return "UNKNOWN"
    dx = pin.point.x - pin.base.x
    dy = pin.point.y - pin.base.y
    if dx == 0 and dy == 0:
        return "UNKNOWN"
    if abs(dx) >= abs(dy):
        return "R" if dx > 0 else "L"
    return "B" if dy > 0 else "T"


def _side_compatibility(
    pin: Pin,
    token: TextToken,
    side: str,
    maximum_distance: float,
) -> float:
    if side == "UNKNOWN":
        return 0.75
    center = token.bbox.center
    inward = {
        "R": pin.point.x - center.x,
        "L": center.x - pin.point.x,
        "B": pin.point.y - center.y,
        "T": center.y - pin.point.y,
    }[side]
    if inward >= 0:
        return 1.0
    if inward >= -0.25 * maximum_distance:
        return 0.6
    return 0.2


def _lexical_compatibility(field: str, value: str) -> float:
    normalized = value.strip()
    if not normalized:
        return 0.0
    if field == "number":
        if _NUMBER_LEXEME.fullmatch(normalized):
            # Pure digits and alphanumeric pin labels (A5, USB_D-) are both
            # valid numbers; the latter use the separate _ALNUM_PIN_NUMBER path.
            return 1.0
        return 0.25
    printable = sum(not character.isspace() for character in normalized)
    return 1.0 if printable else 0.0


def _candidate_sort_key(
    candidate: PinAssignmentCandidate,
) -> tuple[float, float, str, str]:
    return (-candidate.score, candidate.distance, candidate.pin_id, candidate.token_id)


def _group_sort_key(group: tuple[str | None, str]) -> tuple[str, str]:
    return group[0] or "", group[1]


def _candidate_record(
    candidate: PinAssignmentCandidate,
    *,
    rank: int,
    status: str,
    assignment_method: str | None,
) -> dict[str, Any]:
    return {
        "target_id": candidate.pin_id,
        "score": round(float(candidate.score), 8),
        "distance": round(float(candidate.distance), 8),
        "explicit": candidate.explicit,
        "semantic_compatibility": round(
            float((candidate.score_components or {}).get("details", 0.0)), 8
        ),
        "component_id": candidate.component_id,
        "side": candidate.side,
        "target_rank": rank,
        "target_status": status,
        "assignment_method": assignment_method,
        "score_components": {
            key: round(float(value), 8)
            for key, value in sorted((candidate.score_components or {}).items())
        },
    }


def prune_pin_precision(
    pins: list[Pin],
    components: "Sequence[Component]",
    *,
    enabled: bool = False,
    deep_body_px: float = 11.0,
    row_misalign_px: float = 3.5,
    bbox_escape_ratio: float = 0.15,
) -> dict[str, Any]:
    """Deterministically drop physically implausible pin detections.

    Disabled unless ``enabled`` is true (production default is off).  All
    decisions are made on a frozen snapshot of the inputs; the union of the
    three physical rules below is then removed from ``pins`` in place, the
    same list contract the OCR gate uses:

    * ``deep_body`` -- the pin base sits more than ``deep_body_px`` inside
      the owner component body edge on its orientation side, leaving the
      whole stub detached from the body outline.  Evaluated only with row
      consensus (two or more pins on the same attachment row); on a
      singleton row the body edge is not a trustworthy anchor, so the pin
      is kept (this also preserves the pins that carry net pairs).
    * ``row_misalign`` -- inside a (component, orientation) attachment row
      the pin deviates from the row median by more than ``row_misalign_px``.
    * ``bbox_escape`` -- the tip lies outside the owner bounding box by
      more than ``bbox_escape_ratio`` of the box diagonal.

    The rules read detection geometry only; ground truth never participates.
    Pins without a resolved number are never pruned: the serializer gate
    rejects them anyway, and letting them vote would distort the row
    statistics of the serializable population the thresholds were tuned on.
    The returned summary is JSON-safe and deterministic for identical input,
    so it can live in ``ocr_association`` metadata and byte-stable output.
    """

    if type(enabled) is not bool:
        raise ValueError("enabled must be a boolean")
    deep_limit = float(deep_body_px)
    if not math.isfinite(deep_limit) or deep_limit < 0:
        raise ValueError("deep_body_px must be a non-negative finite number")
    row_limit = float(row_misalign_px)
    if not math.isfinite(row_limit) or row_limit < 0:
        raise ValueError("row_misalign_px must be a non-negative finite number")
    escape_ratio = float(bbox_escape_ratio)
    if not math.isfinite(escape_ratio) or not 0 < escape_ratio <= 1:
        raise ValueError("bbox_escape_ratio must be in (0, 1]")

    ordered = sorted(pins, key=lambda item: item.internal_id)
    # Only serializable candidates (resolved pin number) take part in the
    # decision.  Candidates the serializer gate will drop anyway (missing
    # number / unowned) must not distort row statistics or receive verdicts;
    # the rules were calibrated on the serialized pin population.
    eligible = [item for item in ordered if item.number]
    candidate_count = len(ordered)
    thresholds = {
        "deep_body_px": deep_limit,
        "row_misalign_px": row_limit,
        "bbox_escape_ratio": escape_ratio,
        "row_consensus_min_pins": 2,
    }
    if not enabled:
        return {
            "enabled": False,
            "candidate_count": candidate_count,
            "eligible_count": len(eligible),
            "retained_count": candidate_count,
            "removed_count": 0,
            "removed_by_rule": {"deep_body": 0, "row_misalign": 0, "bbox_escape": 0},
            "removed_ids": [],
            "thresholds": thresholds,
            "reason": "disabled",
        }

    boxes: dict[str, BBox] = {
        item.internal_id: item.bbox for item in components if item.bbox is not None
    }

    # Row statistics use the frozen snapshot so that removing one pin can
    # never shift the median and cascade into further removals.
    rows: dict[tuple[str | None, str], list[tuple[str, float]]] = defaultdict(list)
    for pin in eligible:
        key = (str(pin.component_id) if pin.component_id else None, str(pin.orientation))
        axis_value = pin.point.y if key[1] in ("T", "B") else pin.point.x
        rows[key].append((pin.internal_id, float(axis_value)))
    row_deviation: dict[str, float] = {}
    row_size: dict[str, int] = {}
    for members in rows.values():
        for member_id, _value in members:
            row_size[member_id] = len(members)
        if len(members) < 2:
            continue
        values = sorted(value for _member_id, value in members)
        median = values[len(values) // 2]
        for member_id, value in members:
            row_deviation[member_id] = abs(value - median)

    removed_by_rule = {"deep_body": 0, "row_misalign": 0, "bbox_escape": 0}
    removed_ids: list[str] = []
    for pin in eligible:
        box = boxes.get(str(pin.component_id)) if pin.component_id else None
        fired = False
        edge_offset = _body_edge_offset(pin, box)
        if (
            edge_offset is not None
            and row_size.get(pin.internal_id, 0) >= 2
            and edge_offset < -deep_limit
        ):
            removed_by_rule["deep_body"] += 1
            fired = True
        deviation = row_deviation.get(pin.internal_id)
        if deviation is not None and deviation > row_limit:
            removed_by_rule["row_misalign"] += 1
            fired = True
        if box is not None and _tip_box_escape(pin, box) > escape_ratio:
            removed_by_rule["bbox_escape"] += 1
            fired = True
        if fired:
            removed_ids.append(pin.internal_id)

    if removed_ids:
        doomed = set(removed_ids)
        pins[:] = [item for item in pins if item.internal_id not in doomed]
    return {
        "enabled": True,
        "candidate_count": candidate_count,
        "eligible_count": len(eligible),
        "retained_count": len(pins),
        "removed_count": len(removed_ids),
        "removed_by_rule": removed_by_rule,
        "removed_ids": removed_ids,
        "thresholds": thresholds,
        "reason": "applied",
    }


def _body_edge_offset(pin: Pin, box: "BBox | None") -> float | None:
    """Signed distance of the pin base from the owner body edge.

    Negative values place the base inside the body (the direction a real
    stub grows from); returns ``None`` when geometry is unavailable.
    """

    if box is None or pin.base is None:
        return None
    if pin.orientation in ("T", "B"):
        edge = box.y1 if pin.orientation == "T" else box.y2
        sign = -1.0 if pin.orientation == "T" else 1.0
        return (pin.base.y - edge) * sign
    if pin.orientation in ("L", "R"):
        edge = box.x1 if pin.orientation == "L" else box.x2
        sign = -1.0 if pin.orientation == "L" else 1.0
        return (pin.base.x - edge) * sign
    return None


def _tip_box_escape(pin: Pin, box: BBox) -> float:
    """Distance from the tip to the owner bounding box, in diagonals."""

    dx = max(box.x1 - pin.point.x, 0.0, pin.point.x - box.x2)
    dy = max(box.y1 - pin.point.y, 0.0, pin.point.y - box.y2)
    diagonal = math.hypot(box.x2 - box.x1, box.y2 - box.y1) or 1.0
    return math.hypot(dx, dy) / diagonal


__all__ = ["associate_pin_texts", "prune_pin_precision"]
