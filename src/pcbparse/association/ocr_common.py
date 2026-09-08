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
    pin_max_distance_px: float = 24.0
    pin_min_score: float = 0.75
    pin_unique_margin: float = 0.10
    pin_ocr_gate_enabled: bool = False
    pin_ocr_gate_max_distance_px: float = 3.0
    pin_ocr_gate_min_text_score: float = 0.50
    netlabel_max_distance_px: float = 18.0
    netlabel_unique_margin_px: float = 3.0
    netlabel_min_text_score: float = 0.60
    max_recorded_candidates: int = 5

    def __post_init__(self) -> None:
        positive = {
            "component_max_distance_factor": self.component_max_distance_factor,
            "pin_max_distance_px": self.pin_max_distance_px,
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
        if type(self.bind_global_component_targets) is not bool:
            raise ValueError("bind_global_component_targets must be a boolean")

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


def preferred_text(token: TextToken) -> str | None:
    for value in (token.resolved_text, token.raw_text):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def lexical_candidates(token: TextToken) -> list[str]:
    values: list[str] = []
    for value in (token.resolved_text, token.raw_text, *token.candidates):
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
    """Choose one distinct value from accepted token-target proposals."""

    best_by_value: dict[str, float] = {}
    display_by_normalized: dict[str, str] = {}
    for candidate in candidates:
        if candidate.get("target_status") != "accepted":
            continue
        if candidate.get("eligible") is False:
            continue
        value = candidate.get("value")
        score = candidate.get("score")
        if not isinstance(value, str) or not isinstance(score, (int, float)):
            continue
        normalized = value.casefold()
        display_by_normalized.setdefault(normalized, value)
        best_by_value[normalized] = max(best_by_value.get(normalized, -1.0), float(score))
    if not best_by_value:
        return "no_accepted_candidate", None, 0.0, None
    ranked = sorted(best_by_value.items(), key=lambda item: (-item[1], item[0]))
    best_normalized, best_score = ranked[0]
    if best_score < min_score:
        return "below_threshold", None, best_score, None
    if len(ranked) == 1:
        return "accepted", display_by_normalized[best_normalized], best_score, None
    margin = best_score - ranked[1][1]
    if margin < unique_margin:
        return "ambiguous", None, best_score, margin
    return "accepted", display_by_normalized[best_normalized], best_score, margin


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
    "point_bbox_distance",
    "preferred_text",
    "upsert_evidence",
]
