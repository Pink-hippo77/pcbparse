from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pcbparse.io import ComponentView, ImageView
from pcbparse.ir.models import BBox, Point, TextToken
from pcbparse.perception.coordinate_binding import bind_bbox_evidence
from pcbparse.perception.protocols import OCRProvider


_SIDE_ORDER = ("L", "R", "T", "B")
_LEVEL_PRIORITY = {"global": 0, "component": 1, "side_strip": 2}


@dataclass(frozen=True, slots=True)
class MultilevelOCRConfig:
    """Scheduling and cross-view suppression settings for hierarchical OCR.

    Limits count views, rather than provider calls.  A limit of zero disables
    that level while ``None`` means no limit.  Input ordering never affects
    which views are selected.
    """

    global_batch_size: int = 8
    component_batch_size: int = 16
    strip_batch_size: int = 32
    max_global_views: int | None = None
    max_component_views: int | None = None
    max_strip_views: int | None = None
    nms_iou_threshold: float = 0.5
    nms_center_distance_px: float = 3.0
    # Conflicting OCR strings are merged only when the caller explicitly opts
    # in.  A stricter overlap gate protects adjacent glyphs in dense layouts.
    merge_conflicting_text: bool = False
    conflicting_text_iou_threshold: float = 0.85
    require_four_side_strips: bool = True

    def __post_init__(self) -> None:
        for name in ("global_batch_size", "component_batch_size", "strip_batch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("max_global_views", "max_component_views", "max_strip_views"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")
        if not 0.0 <= self.nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold must be in [0, 1]")
        if self.nms_center_distance_px < 0.0:
            raise ValueError("nms_center_distance_px cannot be negative")
        if type(self.merge_conflicting_text) is not bool:
            raise ValueError("merge_conflicting_text must be a boolean")
        if not 0.0 <= self.conflicting_text_iou_threshold <= 1.0:
            raise ValueError("conflicting_text_iou_threshold must be in [0, 1]")


# Keep both spellings importable; callers commonly capitalize "MultiLevel".
MultiLevelOCRConfig = MultilevelOCRConfig


@dataclass(frozen=True, slots=True)
class _ScheduledView:
    view: ImageView
    level: str
    component_id: str | None = None
    side: str | None = None


@dataclass(slots=True)
class _RestoredToken:
    token: TextToken
    provider_local_id: str
    source_view_id: str
    level: str
    component_id: str | None
    side: str | None


def run_multilevel_ocr(
    provider: OCRProvider,
    global_views: Sequence[ImageView],
    component_views: Sequence[ComponentView],
    ctx: Any,
    config: MultilevelOCRConfig | Mapping[str, Any] | None = None,
) -> list[TextToken]:
    """Run global-tile, component-ROI and four-side-strip OCR.

    Provider boxes are interpreted only in the single view named by their
    evidence and are restored to original-image coordinates.  Context such as
    the OCR level and component ROI is provenance only: this function never
    guesses a token category, component assignment, pin assignment, or
    resolved text.
    """

    settings = _coerce_config(config)
    levels = _build_schedule(global_views, component_views, settings)
    restored: list[_RestoredToken] = []
    for scheduled, batch_size in levels:
        for batch in _batches(scheduled, batch_size):
            allowed = {item.view.view_id: item for item in batch}
            produced = provider.recognize([item.view for item in batch], ctx)
            if not isinstance(produced, list):
                raise TypeError("OCRProvider.recognize must return list[TextToken]")
            for token in produced:
                if not isinstance(token, TextToken):
                    raise TypeError("OCRProvider.recognize returned a non-TextToken value")
                restored.append(_restore_token(token, allowed))

    kept = _deduplicate(restored, settings)
    output = sorted((item.token for item in kept), key=_output_sort_key)
    for index, token in enumerate(output, start=1):
        token.internal_id = f"text:{index:06d}"
    return output


def _coerce_config(
    config: MultilevelOCRConfig | Mapping[str, Any] | None,
) -> MultilevelOCRConfig:
    if config is None:
        return MultilevelOCRConfig()
    if isinstance(config, MultilevelOCRConfig):
        return config
    if isinstance(config, Mapping):
        return MultilevelOCRConfig(**dict(config))
    raise TypeError("config must be MultilevelOCRConfig, a mapping, or None")


def _build_schedule(
    global_views: Sequence[ImageView],
    component_views: Sequence[ComponentView],
    config: MultilevelOCRConfig,
) -> list[tuple[list[_ScheduledView], int]]:
    global_items = [
        _ScheduledView(view=view, level="global")
        for view in sorted(global_views, key=lambda item: item.view_id)
    ]
    ordered_components = sorted(
        component_views,
        key=lambda item: (
            item.component_bbox.y1,
            item.component_bbox.x1,
            item.component_bbox.y2,
            item.component_bbox.x2,
            item.component_id,
        ),
    )
    component_items: list[_ScheduledView] = []
    strip_items: list[_ScheduledView] = []
    component_ids: set[str] = set()
    for item in ordered_components:
        if item.component_id in component_ids:
            raise ValueError(f"duplicate component OCR context {item.component_id}")
        component_ids.add(item.component_id)
        component_items.append(
            _ScheduledView(
                view=item.full_roi,
                level="component",
                component_id=item.component_id,
            )
        )
        side_keys = set(item.side_strips)
        expected = set(_SIDE_ORDER)
        if config.require_four_side_strips and side_keys != expected:
            missing = sorted(expected - side_keys)
            extra = sorted(side_keys - expected)
            raise ValueError(
                f"component {item.component_id} side strips must be exactly L/R/T/B; "
                f"missing={missing}, extra={extra}"
            )
        for side in _SIDE_ORDER:
            view = item.side_strips.get(side)
            if view is not None:
                strip_items.append(
                    _ScheduledView(
                        view=view,
                        level="side_strip",
                        component_id=item.component_id,
                        side=side,
                    )
                )

    # A view ID is the provider's only coordinate-frame handle.  Reusing one
    # anywhere in the schedule would make restoration ambiguous, even if a
    # configured cap happens to omit one of the colliding entries.
    all_items = [*global_items, *component_items, *strip_items]
    identifiers: set[str] = set()
    for item in all_items:
        identifier = item.view.view_id
        if identifier in identifiers:
            raise ValueError(f"duplicate OCR view_id {identifier}")
        identifiers.add(identifier)

    global_items = _limit(global_items, config.max_global_views)
    component_items = _limit(component_items, config.max_component_views)
    strip_items = _limit(strip_items, config.max_strip_views)
    return [
        (global_items, config.global_batch_size),
        (component_items, config.component_batch_size),
        (strip_items, config.strip_batch_size),
    ]


def _limit(items: list[_ScheduledView], maximum: int | None) -> list[_ScheduledView]:
    return items if maximum is None else items[:maximum]


def _batches(items: list[_ScheduledView], size: int) -> list[list[_ScheduledView]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def _restore_token(
    token: TextToken,
    allowed: dict[str, _ScheduledView],
) -> _RestoredToken:
    source_views = {evidence.view_id for evidence in token.evidence}
    if not source_views:
        raise ValueError(f"OCR token {token.internal_id} has no source view evidence")
    if len(source_views) != 1:
        raise ValueError(
            f"OCR token {token.internal_id} has ambiguous source views: {sorted(source_views)}"
        )
    source_view_id = next(iter(source_views))
    scheduled = allowed.get(source_view_id)
    if scheduled is None:
        raise ValueError(
            f"OCR token {token.internal_id} references view {source_view_id} "
            "outside its provider batch"
        )

    item = token.model_copy(deep=True)
    provider_local_id = item.internal_id
    local_bbox = item.bbox.model_copy(deep=True)
    item.bbox = _bbox_to_original(item.bbox, scheduled.view)
    for evidence in item.evidence:
        evidence.metadata.setdefault("provider_local_id", provider_local_id)
        evidence.metadata["level"] = scheduled.level
        evidence.metadata["side"] = scheduled.side
        evidence.metadata["component_id"] = scheduled.component_id
        evidence.metadata["candidates"] = list(item.candidates)
    bind_bbox_evidence(
        item.evidence,
        view=scheduled.view,
        local_bbox=local_bbox,
        original_bbox=item.bbox,
        object_kind="text",
    )
    item.evidence.sort(
        key=lambda evidence: (
            evidence.evidence_id,
            evidence.source,
            -evidence.score,
            str(evidence.payload_ref or ""),
        )
    )
    return _RestoredToken(
        token=item,
        provider_local_id=provider_local_id,
        source_view_id=source_view_id,
        level=scheduled.level,
        component_id=scheduled.component_id,
        side=scheduled.side,
    )


def _bbox_to_original(bbox: BBox, view: ImageView) -> BBox:
    corners = [
        view.to_original(Point(x=bbox.x1, y=bbox.y1)),
        view.to_original(Point(x=bbox.x2, y=bbox.y1)),
        view.to_original(Point(x=bbox.x1, y=bbox.y2)),
        view.to_original(Point(x=bbox.x2, y=bbox.y2)),
    ]
    return BBox(
        x1=min(point.x for point in corners),
        y1=min(point.y for point in corners),
        x2=max(point.x for point in corners),
        y2=max(point.y for point in corners),
    )


def _deduplicate(
    restored: list[_RestoredToken],
    config: MultilevelOCRConfig,
) -> list[_RestoredToken]:
    ranked = sorted(restored, key=_nms_rank_key)
    kept: list[_RestoredToken] = []
    for candidate in ranked:
        normalized = _normalize_text(candidate.token.resolved_text or candidate.token.raw_text)
        duplicate = next(
            (
                existing
                for existing in kept
                if normalized
                and (
                    (
                        _normalize_text(
                            existing.token.resolved_text or existing.token.raw_text
                        )
                        == normalized
                        and (
                            _iou(existing.token.bbox, candidate.token.bbox)
                            >= config.nms_iou_threshold
                            or _center_distance(
                                existing.token.bbox, candidate.token.bbox
                            )
                            <= config.nms_center_distance_px
                        )
                    )
                    or (
                        config.merge_conflicting_text
                        and _same_ocr_context(existing, candidate)
                        and _normalize_text(
                            existing.token.resolved_text or existing.token.raw_text
                        )
                        != normalized
                        and _iou(existing.token.bbox, candidate.token.bbox)
                        >= config.conflicting_text_iou_threshold
                    )
                )
            ),
            None,
        )
        if duplicate is None:
            kept.append(candidate)
            continue
        _merge_suppressed_duplicate(duplicate, candidate)
    return kept


def _same_ocr_context(left: _RestoredToken, right: _RestoredToken) -> bool:
    """Return whether two boxes came from the same hierarchical OCR context."""

    return (
        left.level == right.level
        and left.component_id == right.component_id
        and left.side == right.side
    )


def _merge_suppressed_duplicate(winner: _RestoredToken, suppressed: _RestoredToken) -> None:
    normalized_winner = _normalize_text(
        winner.token.resolved_text or winner.token.raw_text
    )
    normalized_suppressed = _normalize_text(
        suppressed.token.resolved_text or suppressed.token.raw_text
    )
    variant_values = [*winner.token.candidates, *suppressed.token.candidates]
    if normalized_winner and normalized_suppressed != normalized_winner:
        variant_values.append(suppressed.token.raw_text)
    winner.token.candidates = _stable_unique(variant_values)
    record = {
        "provider_local_id": suppressed.provider_local_id,
        "view_id": suppressed.source_view_id,
        "level": suppressed.level,
        "side": suppressed.side,
        "component_id": suppressed.component_id,
        "score": suppressed.token.score,
        "raw_text": suppressed.token.raw_text,
        "candidates": list(suppressed.token.candidates),
    }
    for evidence in winner.token.evidence:
        evidence.metadata["candidates"] = list(winner.token.candidates)
    metadata = winner.token.evidence[0].metadata
    records = metadata.setdefault("suppressed_duplicates", [])
    if not isinstance(records, list):
        raise ValueError("OCR evidence metadata.suppressed_duplicates must be a list")
    records.append(record)
    records.sort(
        key=lambda item: (
            str(item["level"]),
            str(item["component_id"] or ""),
            str(item["side"] or ""),
            str(item["view_id"]),
            str(item["provider_local_id"]),
        )
    )
    owners = {
        str(value)
        for value in metadata.get("component_context_candidates", [])
        if isinstance(value, str) and value
    }
    owners.update(
        value
        for value in (winner.component_id, suppressed.component_id)
        if isinstance(value, str) and value
    )
    if len(owners) > 1:
        # The same physical glyph was observed through overlapping ROIs owned
        # by different components.  Retaining the highest-score ROI as an
        # explicit owner would force downstream association to that component.
        # Preserve every context as evidence, but clear the authoritative
        # owner so the normal geometry + uniqueness rules decide safely.
        candidates = sorted(owners)
        winner.component_id = None
        for evidence in winner.token.evidence:
            evidence.metadata["component_id"] = None
            evidence.metadata["component_context_conflict"] = True
            evidence.metadata["component_context_candidates"] = candidates


def _stable_unique(values: Sequence[str]) -> list[str]:
    return sorted(set(values), key=lambda value: (_normalize_text(value), value))


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _nms_rank_key(item: _RestoredToken) -> tuple[Any, ...]:
    bbox = item.token.bbox
    return (
        -item.token.score,
        -_LEVEL_PRIORITY[item.level],
        bbox.width * bbox.height,
        _normalize_text(item.token.resolved_text or item.token.raw_text),
        bbox.y1,
        bbox.x1,
        bbox.y2,
        bbox.x2,
        item.source_view_id,
        item.provider_local_id,
    )


def _output_sort_key(token: TextToken) -> tuple[Any, ...]:
    return (
        token.bbox.y1,
        token.bbox.x1,
        token.bbox.y2,
        token.bbox.x2,
        _normalize_text(token.resolved_text or token.raw_text),
        token.raw_text,
        token.internal_id,
    )


def _iou(left: BBox, right: BBox) -> float:
    width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = width * height
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0.0 else 0.0


def _center_distance(left: BBox, right: BBox) -> float:
    return math.hypot(left.center.x - right.center.x, left.center.y - right.center.y)


__all__ = ["MultiLevelOCRConfig", "MultilevelOCRConfig", "run_multilevel_ocr"]
