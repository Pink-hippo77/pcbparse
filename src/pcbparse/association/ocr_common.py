from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from pcbparse.ir.models import BBox, Diagnostic, Evidence, Point, TextToken


ASSOCIATION_STAGE = "association.ocr"


@dataclass(frozen=True, slots=True)
class OCRAssociationConfig:
    """Conservative thresholds for OCR-to-geometry association."""

    component_max_distance_factor: float = 1.5
    component_min_score: float = 0.75
    component_unique_margin: float = 0.10
    # Global OCR remains reassignable until field resolution unless disabled.
    bind_global_component_targets: bool = True
    pin_max_distance_px: float = 36.0
    pin_component_assignment_enabled: bool = False
    pin_component_max_distance_px: float = 64.0
    pin_component_unique_margin_px: float = 3.0
    pin_min_score: float = 0.60
    pin_unique_margin: float = 0.10
    pin_body_name_context_enabled: bool = False
    pin_body_component_roi_context_enabled: bool = False
    pin_name_row_penalty: float = 0.0
    pin_ocr_gate_enabled: bool = False
    pin_ocr_gate_max_distance_px: float = 3.0
    pin_ocr_gate_min_text_score: float = 0.50
    netlabel_max_distance_px: float = 18.0
    netlabel_unique_margin_px: float = 3.0
    netlabel_min_text_score: float = 0.60
    max_recorded_candidates: int = 5
    # S02 binding-funnel repair (diagnostic-driven, default off).  Both flags
    # are inert when false: the association stage writes no extra metadata and
    # no component/token field, so the disabled pipeline stays byte-identical.
    role_tolerant_binding_enabled: bool = False
    binding_swap_repair_enabled: bool = False
    binding_swap_min_gain_px: float = 1.0
    key_argument_max_enabled: bool = False
    # When true the designator field pools explicit (ROI-owned) and global
    # evidence in one score comparison instead of discarding the global rows as
    # soon as one explicit row exists.  The two provenance paths score on
    # different scales (raw token confidence vs. composite), so the default
    # filter can hand a component a strictly worse reading.
    key_evidence_pool_enabled: bool = False
    component_model_catalog_path: str | None = None
    component_model_catalog_sha256: str | None = None

    def __post_init__(self) -> None:
        positive = {
            "component_max_distance_factor": self.component_max_distance_factor,
            "pin_max_distance_px": self.pin_max_distance_px,
            "pin_component_max_distance_px": self.pin_component_max_distance_px,
            "pin_ocr_gate_max_distance_px": self.pin_ocr_gate_max_distance_px,
            "netlabel_max_distance_px": self.netlabel_max_distance_px,
            "max_recorded_candidates": self.max_recorded_candidates,
        }
        for name, value in positive.items():
            if float(value) <= 0:
                raise ValueError(f"{name} must be positive")
        margins = {
            "component_unique_margin": self.component_unique_margin,
            "pin_unique_margin": self.pin_unique_margin,
            "pin_component_unique_margin_px": self.pin_component_unique_margin_px,
            "netlabel_unique_margin_px": self.netlabel_unique_margin_px,
        }
        for name, value in margins.items():
            if float(value) < 0:
                raise ValueError(f"{name} cannot be negative")
        for name, value in {
            "component_min_score": self.component_min_score,
            "pin_min_score": self.pin_min_score,
            "pin_ocr_gate_min_text_score": self.pin_ocr_gate_min_text_score,
            "netlabel_min_text_score": self.netlabel_min_text_score,
        }.items():
            if not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if type(self.pin_ocr_gate_enabled) is not bool:
            raise ValueError("pin_ocr_gate_enabled must be a boolean")
        if type(self.pin_component_assignment_enabled) is not bool:
            raise ValueError("pin_component_assignment_enabled must be a boolean")
        for value in (self.pin_component_max_distance_px, self.pin_component_unique_margin_px):
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError("pin component distances must be finite numbers")
        if type(self.pin_body_name_context_enabled) is not bool:
            raise ValueError("pin_body_name_context_enabled must be a boolean")
        if type(self.pin_body_component_roi_context_enabled) is not bool:
            raise ValueError("pin_body_component_roi_context_enabled must be a boolean")
        if self.pin_body_component_roi_context_enabled and not self.pin_body_name_context_enabled:
            raise ValueError("component ROI pin context requires pin_body_name_context_enabled")
        row_penalty = float(self.pin_name_row_penalty)
        if isinstance(self.pin_name_row_penalty, bool) or not math.isfinite(row_penalty) or not 0 <= row_penalty <= 1:
            raise ValueError("pin_name_row_penalty must be a finite number in [0, 1]")
        if type(self.bind_global_component_targets) is not bool:
            raise ValueError("bind_global_component_targets must be a boolean")
        if type(self.role_tolerant_binding_enabled) is not bool:
            raise ValueError("role_tolerant_binding_enabled must be a boolean")
        if type(self.binding_swap_repair_enabled) is not bool:
            raise ValueError("binding_swap_repair_enabled must be a boolean")
        if type(self.key_argument_max_enabled) is not bool:
            raise ValueError("key_argument_max_enabled must be a boolean")
        if type(self.key_evidence_pool_enabled) is not bool:
            raise ValueError("key_evidence_pool_enabled must be a boolean")
        if (
            isinstance(self.binding_swap_min_gain_px, bool)
            or not math.isfinite(float(self.binding_swap_min_gain_px))
            or float(self.binding_swap_min_gain_px) < 0.0
        ):
            raise ValueError("binding_swap_min_gain_px must be a finite non-negative number")
        path, digest = self.component_model_catalog_path, self.component_model_catalog_sha256
        if (path is None) != (digest is None):
            raise ValueError("model catalog path and sha256 must be provided together")
        if path is not None:
            if not isinstance(path, str) or not path.strip():
                raise ValueError("model catalog path must be nonempty")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("model catalog sha256 must be lowercase SHA256")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def coerce(
        cls,
        value: "OCRAssociationConfig | Mapping[str, Any] | None",
    ) -> "OCRAssociationConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("OCR association config must be a mapping or OCRAssociationConfig")
        source: Mapping[str, Any] = value
        for key in ("ocr_association", "association"):
            nested = source.get(key)
            if isinstance(nested, Mapping):
                source = nested
                break
        aliases = {
            "max_component_text_distance_factor": "component_max_distance_factor",
        }
        fields = cls.__dataclass_fields__
        kwargs: dict[str, Any] = {}
        for raw_key, raw_value in source.items():
            key = aliases.get(str(raw_key), str(raw_key))
            if key in fields:
                kwargs[key] = raw_value
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class RankedAssociationCandidate:
    target_id: str
    score: float
    distance: float
    explicit: bool = False
    semantic_compatibility: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "score": round(float(self.score), 8),
            "distance": round(float(self.distance), 8),
            "explicit": self.explicit,
            "semantic_compatibility": round(float(self.semantic_compatibility), 8),
        }


def canonical_details(details: str) -> str:
    return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKC", details).casefold())


def match_key(value: str) -> str:
    """Identity key for cross-view matching only.

    NFKC + casefold + whitespace collapse.  Never write this back into
    ``raw_text`` / field outputs: ``10k`` / ``10K`` / ``10 k`` stay distinct
    original strings even though they share a match key.
    """

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def preferred_text(token: TextToken) -> str | None:
    """Return the output-facing string for a token.

    Prefers ``raw_text`` so the original spelling (``10k`` / ``10K`` /
    ``10 k``) is never silently rewritten by a resolved/normalized form.
    Only outer whitespace is stripped; case and internal spaces are kept.
    """

    for value in (token.raw_text, token.resolved_text):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def resolved_text(token: TextToken) -> str | None:
    """Return the resolved/normalized hypothesis when present."""

    value = token.resolved_text
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def lexical_candidates(token: TextToken) -> list[str]:
    """Bind raw_text, resolved_text and OCR candidates for matching.

    All three spellings are retained so a match can consider the original
    string and any resolved form without collapsing them into one output.
    """

    values: list[str] = []
    for value in (token.raw_text, token.resolved_text, *token.candidates):
        if isinstance(value, str):
            normalized = value.strip()
            if normalized and normalized not in values:
                values.append(normalized)
    return values


def bbox_gap_distance(left: BBox, right: BBox) -> float:
    dx = max(left.x1 - right.x2, right.x1 - left.x2, 0.0)
    dy = max(left.y1 - right.y2, right.y1 - left.y2, 0.0)
    return math.hypot(dx, dy)


def point_bbox_distance(point: Point, bbox: BBox) -> float:
    dx = max(bbox.x1 - point.x, point.x - bbox.x2, 0.0)
    dy = max(bbox.y1 - point.y, point.y - bbox.y2, 0.0)
    return math.hypot(dx, dy)


def choose_unique_target(
    candidates: Sequence[RankedAssociationCandidate],
    *,
    min_score: float,
    unique_margin: float,
) -> tuple[str, RankedAssociationCandidate | None, float | None]:
    ranked = sorted(candidates, key=lambda item: (-item.score, item.distance, item.target_id))
    if not ranked:
        return "unmatched", None, None
    best = ranked[0]
    if best.score < min_score:
        return "below_threshold", best, None
    if len(ranked) == 1:
        return "accepted", best, None
    margin = best.score - ranked[1].score
    if margin < unique_margin:
        return "ambiguous", best, margin
    return "accepted", best, margin


def make_rule_evidence(
    *,
    evidence_id: str,
    score: float,
    payload_ref: str,
    metadata: Mapping[str, Any],
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source="rule",
        view_id="original",
        score=max(0.0, min(1.0, float(score))),
        payload_ref=payload_ref,
        metadata=dict(metadata),
    )


def upsert_evidence(evidence: list[Evidence], item: Evidence) -> None:
    for index, existing in enumerate(evidence):
        if existing.evidence_id == item.evidence_id:
            evidence[index] = item
            return
    evidence.append(item)


def append_diagnostic(
    diagnostics: list[Diagnostic],
    *,
    severity: str,
    code: str,
    message: str,
    object_ids: Sequence[str],
) -> None:
    diagnostics.append(
        Diagnostic(
            severity=severity,  # type: ignore[arg-type]
            code=code,
            message=message,
            object_ids=list(object_ids),
            stage=ASSOCIATION_STAGE,
        )
    )


def field_choice(
    candidates: Sequence[Mapping[str, Any]],
    *,
    min_score: float,
    unique_margin: float,
) -> tuple[str, str | None, float, float | None]:
    """Choose one exact value string from accepted token-target proposals.

    Distinct originals (``10k`` / ``10K`` / ``10 k``) compete as separate
    candidates so the winner keeps its own spelling.  Case/space variants
    that score within ``unique_margin`` are rejected as ambiguous rather
    than silently normalized into one form.
    """

    best_by_value: dict[str, float] = {}
    for candidate in candidates:
        if candidate.get("target_status") != "accepted":
            continue
        if candidate.get("eligible") is False:
            continue
        value = candidate.get("value")
        score = candidate.get("score")
        if not isinstance(value, str) or not isinstance(score, (int, float)):
            continue
        # Exact-string identity: never casefold/space-collapse the output.
        best_by_value[value] = max(best_by_value.get(value, -1.0), float(score))
    if not best_by_value:
        return "no_accepted_candidate", None, 0.0, None
    ranked = sorted(best_by_value.items(), key=lambda item: (-item[1], item[0]))
    best_value, best_score = ranked[0]
    if best_score < min_score:
        return "below_threshold", None, best_score, None
    if len(ranked) == 1:
        return "accepted", best_value, best_score, None
    margin = best_score - ranked[1][1]
    if margin < unique_margin:
        return "ambiguous", None, best_score, margin
    return "accepted", best_value, best_score, margin


__all__ = [
    "ASSOCIATION_STAGE",
    "OCRAssociationConfig",
    "RankedAssociationCandidate",
    "append_diagnostic",
    "bbox_gap_distance",
    "canonical_details",
    "choose_unique_target",
    "field_choice",
    "lexical_candidates",
    "make_rule_evidence",
    "match_key",
    "point_bbox_distance",
    "preferred_text",
    "resolved_text",
    "upsert_evidence",
]
