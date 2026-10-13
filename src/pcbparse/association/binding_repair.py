"""Flag-gated binding repair for the OCR association stage (S02).

Diagnosis that motivates the design (work/opt90-bind/REPLAY.md has the full
evidence chain): on the run-001 stack every component key comes from the
per-token pass; ``resolve_component_keys`` assigns **zero** keys because its
``used_keys`` filter forbids reusing any visible designator that another
component already carries.  A keyless component therefore stays keyless even
when its own visible designator token is captured inside its own bounding box
-- and ``_prune_unrepresentable`` then deletes it.

Two mechanisms, both default-off, deterministic and GT-free:

``role_tolerant_binding_enabled``
    A keyless component may carry a designator-grammar token
    (``^[A-Z]{1,4}[0-9]+[A-Z]?$``) that is *visually contained* in its bounding
    box or was captured in its own component ROI, regardless of the role the
    semantic pass gave that token (``pinnumber`` / ``pinname`` / empty).  The
    key is always the token's own visible text; nothing is invented.

``binding_swap_repair_enabled``
    A component that already carries a key may swap it for a contained token
    that is *less contested* -- contained by fewer other compatible components
    -- which repairs ROI-ownership transpositions where a component's key came
    from a label that geometrically belongs to its neighbour.  Only strict
    improvements are applied.

Both mechanisms only rebind existing visible text: never fabricate a key, never
auto-number, never read ground truth.
"""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

from pcbparse.association.text_component import _canonical_designator, _prefix_compatible
from pcbparse.association.ocr_common import (
    bbox_gap_distance,
    make_rule_evidence,
    preferred_text,
    upsert_evidence,
)
from pcbparse.ir.models import Component, TextToken

_DESIGNATOR = re.compile(r"^[A-Z]{1,4}[0-9]+[A-Z]?$")

# Weighting mirrors ``resolve_component_keys`` so a repair cannot outrank
# evidence the main pass already trusts more.
_TEXT_WEIGHT = 0.65
_PROXIMITY_WEIGHT = 0.25
_COMPATIBILITY_WEIGHT = 0.10


def _serializable_categories() -> frozenset[str]:
    """Component categories the official schema can serialize (uppercased).

    Measured hazard: letting a component whose category is absent from
    ``component.allowed_types`` survive the keyless prune turns a degraded but
    valid output into an ``OfficialSchemaError`` (case 0037, category ``M``).
    The repair therefore never hands a key to an unrepresentable component; the
    serializer alias gap itself is reported separately, not patched here.
    """
    try:
        from pcbparse.output.official_schema import DEFAULT_OFFICIAL_SCHEMA

        profile = DEFAULT_OFFICIAL_SCHEMA
        component = profile.get("component", {}) if isinstance(profile, dict) else {}
        allowed = component.get("allowed_types") or []
        aliases = component.get("type_aliases") or {}
    except Exception:  # pragma: no cover - defensive: never block association
        return frozenset()
    names = {str(item).strip().upper() for item in allowed if str(item).strip()}
    names |= {str(item).strip().upper() for item in aliases if str(item).strip()}
    return frozenset(names)


_SERIALIZABLE = _serializable_categories()


def _representable(component: Component) -> bool:
    if not _SERIALIZABLE:
        return True
    return component.category.strip().upper() in _SERIALIZABLE


def _has_dataset_evidence(token: TextToken) -> bool:
    return any(evidence.source == "dataset" for evidence in token.evidence)


def _key_of(component: Component) -> str:
    return (component.key or "").strip() if isinstance(component.key, str) else ""


def _compatible(component: Component, text: str, *, token: TextToken) -> bool:
    """Category/prefix compatibility, with the BOX wildcard guarded.

    ``_prefix_compatible`` treats BOX as a wildcard, which would let any BOX
    region absorb any token.  Following the precedent in
    ``semantic_fields._classify_global`` the wildcard only applies when the
    token was captured inside that very component's ROI.
    """
    if component.category.upper() == "BOX":
        return str(token.component_id or "") == component.internal_id
    return _prefix_compatible(component.category, text)


def _touches(component: Component, token: TextToken) -> bool:
    """True when the token's box intersects or touches the component's box."""
    return bbox_gap_distance(component.bbox, token.bbox) <= 0.0


def _roi_component_id(token: TextToken) -> str | None:
    """Component whose ROI crop produced this token (perception provenance).

    ``token.component_id`` is only filled by the semantic pass, and pin-role
    tokens are never visited by ``associate_component_texts``, so the ROI
    provenance recorded by the OCR view is the only reliable ownership signal
    available when the repair runs.
    """
    for evidence in token.evidence:
        metadata = evidence.metadata or {}
        if metadata.get("level") == "component" and metadata.get("component_id"):
            return str(metadata["component_id"])
    return None


def _owned_by(token: TextToken, component: Component) -> bool:
    if token.component_id is not None and str(token.component_id) == component.internal_id:
        return True
    return _roi_component_id(token) == component.internal_id


def _evidence_ok(component: Component, text: str, *, token: TextToken) -> bool:
    """Prefix compatibility, relaxed when the visible token is grounded.

    A detected category is a model output and is frequently wrong (a capacitor
    detected as ``Battary``, a switch as ``Other``).  When a designator-grammar
    token is geometrically contained in the component box -- optionally also
    captured by that component's ROI -- the read text outranks the category
    prefix; prefix compatibility then only remains a ranking preference.

    NOTE: the relaxation is evaluated at repair time, when pin-role tokens have
    no ``component_id`` yet, so containment is the only ownership signal that is
    always available.
    """
    if _compatible(component, text, token=token):
        return True
    if component.category.upper() == "BOX":
        return False
    return _touches(component, token) or _owned_by(token, component)


def _roi_component_id(token: TextToken) -> str | None:
    """Component whose ROI crop produced this token (perception provenance).

    ``token.component_id`` is only filled by the semantic pass, and pin-role
    tokens are never visited by ``associate_component_texts``, so the ROI
    provenance recorded by the OCR view is the only reliable ownership signal
    available when the repair runs.
    """
    for evidence in token.evidence:
        metadata = evidence.metadata or {}
        if metadata.get("level") == "component" and metadata.get("component_id"):
            return str(metadata["component_id"])
    return None


def _collect_tokens(texts: Sequence[TextToken]) -> list[tuple[TextToken, str, str]]:
    """``(token, grammar_text, visible_text)`` for designator-grammar tokens."""
    collected: list[tuple[TextToken, str, str]] = []
    for token in sorted(texts, key=lambda item: item.internal_id):
        if _has_dataset_evidence(token):
            continue
        value = preferred_text(token)
        if not value:
            continue
        visible = value.strip()
        grammar = visible.upper()
        if not _DESIGNATOR.fullmatch(grammar):
            continue
        collected.append((token, grammar, visible))
    return collected


def _proximity_score(component: Component, token: TextToken, max_distance_factor: float) -> float:
    scale = max(component.bbox.width, component.bbox.height, 1.0)
    normalized = bbox_gap_distance(component.bbox, token.bbox) / scale
    return max(0.0, 1.0 - normalized / max(max_distance_factor, 1e-9))


def _contest_count(
    components: Sequence[Component],
    text: str,
    token: TextToken,
    owner_id: str,
) -> int:
    """How many *other* compatible components visually contain this token."""
    count = 0
    for component in components:
        if component.internal_id == owner_id:
            continue
        if not _compatible(component, text, token=token):
            continue
        if _touches(component, token):
            count += 1
    return count


def _bind_keyless_components(
    components: Sequence[Component],
    tokens: Sequence[tuple[TextToken, str, str]],
    *,
    max_distance_factor: float,
    min_score: float,
) -> dict[str, Any]:
    keyless = [component for component in components
               if not _key_of(component) and _representable(component)]
    if not keyless or not tokens:
        return {"candidates": 0, "assigned": 0, "assignments": {}, "rejected": 0}

    candidates: list[tuple[int, float, float, str, str, TextToken, Component, str, str]] = []
    rejected = 0
    key_index = {_key_of(component).casefold(): component
                 for component in components if _key_of(component)}
    for component in keyless:
        for token, text, raw in tokens:
            owned = _owned_by(token, component)
            grounded = _touches(component, token) or owned
            duplicate_grounded = False
            if not grounded:
                # A second detection of the same physical part can sit a few
                # pixels away from a label whose owner already carries that
                # exact designator; such a duplicate reading is still the same
                # visible text and is allowed inside the same distance gate.
                owner = key_index.get(text.casefold())
                gap_px = bbox_gap_distance(component.bbox, token.bbox)
                scale_px = max(component.bbox.width, component.bbox.height, 1.0)
                duplicate_grounded = (
                    owner is not None
                    and owner.internal_id != component.internal_id
                    and _compatible(owner, text, token=token)
                    and gap_px / scale_px <= max_distance_factor
                )
                grounded = duplicate_grounded
                if not grounded:
                    rejected += 1
                    continue
            if not (_evidence_ok(component, text, token=token) or duplicate_grounded):
                continue
            gap = bbox_gap_distance(component.bbox, token.bbox)
            scale = max(component.bbox.width, component.bbox.height, 1.0)
            if gap / scale > max_distance_factor:
                rejected += 1
                continue
            score = min(1.0, _TEXT_WEIGHT * float(token.score)
                        + _PROXIMITY_WEIGHT * _proximity_score(component, token, max_distance_factor)
                        + _COMPATIBILITY_WEIGHT)
            if score < min_score:
                rejected += 1
                continue
            # ROI-owned evidence outranks plain containment; a prefix-compatible
            # reading outranks a category-relaxed one; then confidence, then
            # distance, then stable ids.
            prefix_ok = _compatible(component, text, token=token)
            candidates.append((0 if owned else 1, 0 if prefix_ok else 1, -score, gap,
                               token.internal_id, component.internal_id,
                               token, component, text, raw))
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5]))

    used_tokens: set[str] = set()
    assigned: dict[str, str] = {}
    for _own, _prefix, _neg, gap, token_id, component_id, token, component, text, raw in candidates:
        if component_id in assigned or token_id in used_tokens:
            continue
        canonical, _rewrite = _canonical_designator(raw, component.category)
        if not canonical:
            continue
        component.key = canonical
        component.key_score = float(token.score)
        assigned[component_id] = canonical
        used_tokens.add(token_id)
        if token.component_id is None:
            token.component_id = component_id
        upsert_evidence(
            component.evidence,
            make_rule_evidence(
                evidence_id=f"association:role-tolerant-key:{component_id}",
                score=float(token.score),
                payload_ref=component_id,
                metadata={
                    "kind": "role-tolerant-binding",
                    "component_id": component_id,
                    "assigned_key": canonical,
                    "token_id": token_id,
                    "token_raw_text": token.raw_text,
                    "token_details": token.details,
                    "token_owned_by_component": _owned_by(token, component),
                    "gap_px": round(gap, 4),
                    "min_score": min_score,
                    "rule": "contained-visible-designator-for-keyless-component",
                },
            ),
        )
    return {"candidates": len(candidates), "assigned": len(assigned),
            "rejected": rejected, "assignments": dict(sorted(assigned.items()))}


def _nearest_token_for_text(
    tokens: Sequence[tuple[TextToken, str, str]],
    component: Component,
    text: str,
) -> tuple[TextToken, str] | None:
    best: tuple[float, str, TextToken, str] | None = None
    for token, token_text, _raw in tokens:
        if token_text.casefold() != text.casefold():
            continue
        if not _compatible(component, token_text, token=token):
            continue
        if not _touches(component, token):
            continue
        candidate = (bbox_gap_distance(component.bbox, token.bbox), token.internal_id,
                     token, token_text)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        return None
    return best[2], best[3]


def _repair_existing_keys(
    components: Sequence[Component],
    tokens: Sequence[tuple[TextToken, str, str]],
    *,
    max_distance_factor: float,
    min_gain_px: float,
) -> dict[str, Any]:
    """Prefer the least-contested contained designator for a keyed component."""
    swaps: list[dict[str, Any]] = []
    rounds = 0
    for _round in range(2):
        changed_this_round = 0
        for component in sorted(components, key=lambda item: item.internal_id):
            current_key = _key_of(component)
            if not current_key:
                continue
            current = _nearest_token_for_text(tokens, component, current_key)
            current_rank = (10 ** 6, 0.0) if current is None else (
                _contest_count(components, current[1], current[0], component.internal_id),
                -float(current[0].score),
            )
            best: tuple[tuple[int, float], TextToken, str, str] | None = None
            for token, text, _raw in tokens:
                if text.casefold() == current_key.casefold():
                    continue
                if not _compatible(component, text, token=token):
                    continue
                if not _touches(component, token):
                    continue
                rank = (_contest_count(components, text, token, component.internal_id),
                        -float(token.score))
                if best is None or (rank, token.internal_id) < (best[0], best[1].internal_id):
                    best = (rank, token, text, _raw)
            if best is None:
                continue
            rank, token, _text, raw = best
            gap = bbox_gap_distance(component.bbox, token.bbox)
            gain = (current_rank[0] - rank[0]) * max(
                max(component.bbox.width, component.bbox.height, 1.0), 1.0)
            if rank >= current_rank or gain < min_gain_px:
                continue
            previous = current_key
            canonical, _rewrite = _canonical_designator(raw, component.category)
            if not canonical or canonical == previous:
                continue
            component.key = canonical
            changed_this_round += 1
            swaps.append({
                "component_id": component.internal_id,
                "previous_key": previous,
                "assigned_key": canonical,
                "token_id": token.internal_id,
                "previous_contest": current_rank[0],
                "contest": rank[0],
                "gap_px": round(gap, 4),
            })
            upsert_evidence(
                component.evidence,
                make_rule_evidence(
                    evidence_id=f"association:swap-repair-key:{component.internal_id}",
                    score=float(component.key_score or 0.0),
                    payload_ref=component.internal_id,
                    metadata={
                        "kind": "binding-swap-repair",
                        "component_id": component.internal_id,
                        "previous_key": previous,
                        "assigned_key": canonical,
                        "token_id": token.internal_id,
                        "token_raw_text": raw,
                        "previous_contest": current_rank[0],
                        "contest": rank[0],
                        "gap_px": round(gap, 4),
                        "min_gain_px": min_gain_px,
                        "rule": "least-contested-contained-designator",
                    },
                ),
            )
        rounds += 1
        if changed_this_round == 0:
            break
    return {"swaps": len(swaps), "rounds": rounds, "records": swaps}


def repair_binding(
    components: Sequence[Component],
    texts: Sequence[TextToken],
    *,
    enabled: bool,
    role_tolerant_enabled: bool,
    swap_enabled: bool,
    max_distance_factor: float,
    min_score: float,
    min_gain_px: float,
) -> dict[str, Any] | None:
    """Apply the enabled repair mechanisms; return ``None`` when nothing runs.

    Returning ``None`` keeps the disabled pipeline byte-identical: no component
    field, token field, evidence entry or metadata entry is written.
    """
    if not enabled or not (role_tolerant_enabled or swap_enabled):
        return None
    tokens = _collect_tokens(texts)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "role_tolerant_binding_enabled": bool(role_tolerant_enabled),
        "binding_swap_repair_enabled": bool(swap_enabled),
        "designator_grammar_tokens": len(tokens),
        "max_distance_factor": max_distance_factor,
        "min_score": min_score,
        "min_gain_px": min_gain_px,
    }
    if role_tolerant_enabled:
        summary["role_tolerant"] = _bind_keyless_components(
            components, tokens,
            max_distance_factor=max_distance_factor,
            min_score=min_score,
        )
    if swap_enabled:
        summary["swap_repair"] = _repair_existing_keys(
            components, tokens,
            max_distance_factor=max_distance_factor,
            min_gain_px=min_gain_px,
        )
    return summary


__all__ = ["repair_binding"]
