from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pcbparse.io import (
    ImageView,
    crop_image_view,
    load_image,
    make_component_view,
    tile_image_view,
)
from pcbparse.ir.models import BBox, CircuitIR, Component, Evidence, IRState, Junction, Pin, Point, RepairRecord, TextToken, WireSegment
from pcbparse.perception.protocols import ComponentProvider, FullImagePinProvider, JunctionProvider, OCRProvider, PinProvider, WireMask, WireProvider
from pcbparse.perception.coordinate_binding import (
    bind_bbox_evidence,
    bind_point_evidence,
    bind_segment_evidence,
)
from pcbparse.perception.ocr_multilevel import MultilevelOCRConfig, run_multilevel_ocr
from pcbparse.perception.wire_vectorizer import (
    SkeletonPolyline,
    WireVectorizerConfig,
    WireJunctionCandidate,
    deduplicate_polylines,
    vectorize_wire_mask,
)
from pcbparse.pipeline.context import RunContext


@dataclass(slots=True)
class CompositeProviderConfig:
    tile_size: int = 1024
    tile_overlap: int = 160
    component_tile_size: int | None = None
    component_tile_overlap: int | None = None
    # Providers with ``predict_many`` receive at most this many Component
    # views per inference call. Providers without it retain sequential calls.
    component_inference_batch_size: int = 1
    wire_tile_size: int | None = None
    wire_tile_overlap: int | None = None
    # OCR detector is trained at 512px; tiling at the same size preserves text
    # resolution instead of downscaling 1024px tiles by 2x.
    ocr_tile_size: int = 512
    ocr_tile_overlap: int = 160
    component_nms_iou: float = 0.55
    component_center_preferred_nms_enabled: bool = False
    component_center_preference_score_bonus: float = 1.0
    component_internal_border_margin_px: float = 32.0
    component_internal_border_confirmation_enabled: bool = False
    component_internal_border_confirmation_iou: float = 0.30
    component_internal_border_confirmation_metric: str = "iou"
    component_internal_border_confirmation_categories: tuple[str, ...] = ()
    # Optional second pass for detections truncated by an internal tile edge.
    # A crop centred on the candidate gives the detector full local context;
    # only a same-class overlapping result is allowed to replace the original.
    component_internal_border_refinement_enabled: bool = False
    component_internal_border_refinement_iou: float = 0.50
    component_internal_border_refinement_categories: tuple[str, ...] = ()
    component_internal_border_refinement_max_candidates: int = 16
    component_internal_border_refinement_batch_size: int = 1
    # Optional second scale for categories whose symbols become too small in
    # the primary tiles. Recovery candidates are merged through the same
    # class-aware NMS and full-image cap as the primary pass.
    component_targeted_recovery_enabled: bool = False
    component_targeted_recovery_categories: tuple[str, ...] = ()
    component_targeted_recovery_tile_size: int = 256
    component_targeted_recovery_tile_overlap: int = 32
    component_targeted_recovery_min_dimension_px: int = 768
    component_targeted_recovery_batch_size: int = 1
    component_roi_padding_factor: float = 0.15
    # Legacy shared absolute padding.  The task-specific values below fall
    # back to this field so existing configurations keep identical geometry.
    component_roi_padding_px: float = 0.0
    ocr_component_roi_padding_px: float | None = None
    pin_component_roi_padding_px: float | None = None
    # Backwards-compatible name: this is now the skeleton path pixel minimum,
    # not a scanline-run threshold.
    wire_minimum_run_px: int = 4
    wire_rdp_epsilon_px: float = 0.75
    wire_gap_closing_radius_px: int = 0
    wire_maximum_hole_area_px: int = 0
    wire_maximum_component_pixels: int = 0
    wire_tile_dedup_tolerance_px: float = 1.0
    # Learned Junction remains quarantined until all six real validation
    # classes are reviewed and its graph-level false-positive gate passes.
    # Deterministic vectorizer endpoints/T-junctions remain available.
    learned_junctions_enabled: bool = False
    ocr_global_batch_size: int = 8
    ocr_component_batch_size: int = 16
    ocr_strip_batch_size: int = 32
    ocr_nms_iou: float = 0.5
    ocr_nms_center_distance_px: float = 3.0
    # Conflicting strings from the same high-overlap OCR view are merged only
    # by explicit opt-in; the stricter IoU gate protects adjacent glyphs.
    ocr_merge_conflicting_text: bool = False
    ocr_conflicting_text_iou: float = 0.85
    # Component detections are the fan-out point for both OCR and Pin. Keep
    # this gate in the composite provider as a second line of defence: an
    # adapter may expose low-score candidates for calibration, but production
    # must not create an unbounded number of downstream ROIs.
    component_candidate_min_score: float = 0.0
    component_max_candidates: int | None = None
    # Image-only dense-layout fallback. The primary component pass remains
    # authoritative unless it emits a sparse, non-Box-heavy result. This
    # avoids using dataset/source labels while recovering small components in
    # layouts whose first pass is visibly under-populated.
    component_adaptive_fallback_enabled: bool = False
    component_fallback_tile_size: int = 256
    component_fallback_tile_overlap: int = 32
    component_fallback_trigger_max_count: int = 24
    component_fallback_trigger_max_box_fraction: float = 0.20
    component_fallback_trigger_max_bbox_area_fraction: float = 0.02
    # Runtime-only Pin decoder overrides. The model lock remains authoritative
    # for artifact identity; overrides are explicit in the pipeline config and
    # included in the resolved perception trace.
    pin_inference_overrides: dict[str, Any] = field(default_factory=dict)
    # Runtime-only Wire decoder overrides. The model lock remains authoritative
    # for artifact identity; the threshold is explicit in the pipeline trace.
    wire_inference_overrides: dict[str, Any] = field(default_factory=dict)
    # Runtime-only OCR decoder overrides. Model identity remains in the lock;
    # detector threshold and candidate recovery are explicit in the resolved
    # perception trace. Candidate recovery is disabled by default.
    ocr_inference_overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            type(self.component_inference_batch_size) is not int
            or self.component_inference_batch_size <= 0
        ):
            raise ValueError("component_inference_batch_size must be a positive integer")
        nms_iou = self.component_nms_iou
        if type(nms_iou) not in {int, float} or not np.isfinite(float(nms_iou)) or not 0.0 <= float(nms_iou) <= 1.0:
            raise ValueError("component_nms_iou must be finite in [0,1]")
        if type(self.component_center_preferred_nms_enabled) is not bool:
            raise ValueError("component_center_preferred_nms_enabled must be a boolean")
        if (
            type(self.component_center_preference_score_bonus) not in {int, float}
            or not np.isfinite(float(self.component_center_preference_score_bonus))
            or not 0.0 <= float(self.component_center_preference_score_bonus) <= 1.0
        ):
            raise ValueError(
                "component_center_preference_score_bonus must be finite in [0,1]"
            )
        if (
            type(self.component_internal_border_margin_px) not in {int, float}
            or not np.isfinite(float(self.component_internal_border_margin_px))
            or float(self.component_internal_border_margin_px) < 0.0
        ):
            raise ValueError("component_internal_border_margin_px must be finite and non-negative")
        if type(self.component_internal_border_confirmation_enabled) is not bool:
            raise ValueError(
                "component_internal_border_confirmation_enabled must be a boolean"
            )
        confirmation_iou = self.component_internal_border_confirmation_iou
        if (
            type(confirmation_iou) not in {int, float}
            or not np.isfinite(float(confirmation_iou))
            or not 0.0 <= float(confirmation_iou) <= 1.0
        ):
            raise ValueError(
                "component_internal_border_confirmation_iou must be finite in [0,1]"
            )
        if self.component_internal_border_confirmation_metric not in {
            "iou",
            "intersection_over_smaller",
        }:
            raise ValueError(
                "component_internal_border_confirmation_metric must be "
                "'iou' or 'intersection_over_smaller'"
            )
        confirmation_categories = self.component_internal_border_confirmation_categories
        if not isinstance(confirmation_categories, (list, tuple)) or any(
            not isinstance(item, str) or not item.strip()
            for item in confirmation_categories
        ):
            raise ValueError(
                "component_internal_border_confirmation_categories must contain non-empty strings"
            )
        self.component_internal_border_confirmation_categories = tuple(
            dict.fromkeys(item.strip() for item in confirmation_categories)
        )
        if type(self.component_internal_border_refinement_enabled) is not bool:
            raise ValueError(
                "component_internal_border_refinement_enabled must be a boolean"
            )
        refinement_iou = self.component_internal_border_refinement_iou
        if (
            type(refinement_iou) not in {int, float}
            or not np.isfinite(float(refinement_iou))
            or not 0.0 <= float(refinement_iou) <= 1.0
        ):
            raise ValueError(
                "component_internal_border_refinement_iou must be finite in [0,1]"
            )
        refinement_categories = self.component_internal_border_refinement_categories
        if not isinstance(refinement_categories, (list, tuple)) or any(
            not isinstance(item, str) or not item.strip()
            for item in refinement_categories
        ):
            raise ValueError(
                "component_internal_border_refinement_categories must contain non-empty strings"
            )
        self.component_internal_border_refinement_categories = tuple(
            dict.fromkeys(item.strip() for item in refinement_categories)
        )
        if (
            type(self.component_internal_border_refinement_max_candidates) is not int
            or self.component_internal_border_refinement_max_candidates <= 0
        ):
            raise ValueError(
                "component_internal_border_refinement_max_candidates must be a positive integer"
            )
        if (
            type(self.component_internal_border_refinement_batch_size) is not int
            or self.component_internal_border_refinement_batch_size <= 0
        ):
            raise ValueError(
                "component_internal_border_refinement_batch_size must be a positive integer"
            )
        if type(self.component_targeted_recovery_enabled) is not bool:
            raise ValueError("component_targeted_recovery_enabled must be a boolean")
        recovery_categories = self.component_targeted_recovery_categories
        if not isinstance(recovery_categories, (list, tuple)) or any(
            not isinstance(item, str) or not item.strip()
            for item in recovery_categories
        ):
            raise ValueError(
                "component_targeted_recovery_categories must contain non-empty strings"
            )
        self.component_targeted_recovery_categories = tuple(
            dict.fromkeys(item.strip() for item in recovery_categories)
        )
        if (
            type(self.component_targeted_recovery_tile_size) is not int
            or self.component_targeted_recovery_tile_size <= 0
        ):
            raise ValueError(
                "component_targeted_recovery_tile_size must be a positive integer"
            )
        if (
            type(self.component_targeted_recovery_tile_overlap) is not int
            or not 0
            <= self.component_targeted_recovery_tile_overlap
            < self.component_targeted_recovery_tile_size
        ):
            raise ValueError(
                "component_targeted_recovery_tile_overlap must satisfy 0 <= overlap < tile_size"
            )
        if (
            type(self.component_targeted_recovery_min_dimension_px) is not int
            or self.component_targeted_recovery_min_dimension_px <= 0
        ):
            raise ValueError(
                "component_targeted_recovery_min_dimension_px must be a positive integer"
            )
        if (
            type(self.component_targeted_recovery_batch_size) is not int
            or self.component_targeted_recovery_batch_size <= 0
        ):
            raise ValueError(
                "component_targeted_recovery_batch_size must be a positive integer"
            )
        threshold = self.component_candidate_min_score
        if type(threshold) not in {int, float} or not np.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("component_candidate_min_score must be finite in [0,1]")
        if self.component_max_candidates is not None and (
            type(self.component_max_candidates) is not int
            or self.component_max_candidates <= 0
        ):
            raise ValueError("component_max_candidates must be a positive integer")
        if type(self.ocr_merge_conflicting_text) is not bool:
            raise ValueError("ocr_merge_conflicting_text must be a boolean")
        if (
            type(self.ocr_conflicting_text_iou) not in {int, float}
            or not np.isfinite(float(self.ocr_conflicting_text_iou))
            or not 0.0 <= float(self.ocr_conflicting_text_iou) <= 1.0
        ):
            raise ValueError("ocr_conflicting_text_iou must be finite in [0,1]")
        if type(self.component_adaptive_fallback_enabled) is not bool:
            raise ValueError("component_adaptive_fallback_enabled must be a boolean")
        if type(self.component_fallback_tile_size) is not int or self.component_fallback_tile_size <= 0:
            raise ValueError("component_fallback_tile_size must be a positive integer")
        if type(self.component_fallback_tile_overlap) is not int or not 0 <= self.component_fallback_tile_overlap < self.component_fallback_tile_size:
            raise ValueError("component_fallback_tile_overlap must satisfy 0 <= overlap < tile_size")
        if type(self.component_fallback_trigger_max_count) is not int or self.component_fallback_trigger_max_count <= 0:
            raise ValueError("component_fallback_trigger_max_count must be a positive integer")
        if (
            type(self.component_fallback_trigger_max_box_fraction) not in {int, float}
            or not np.isfinite(float(self.component_fallback_trigger_max_box_fraction))
            or not 0.0 <= float(self.component_fallback_trigger_max_box_fraction) <= 1.0
        ):
            raise ValueError("component_fallback_trigger_max_box_fraction must be finite in [0,1]")
        if (
            type(self.component_fallback_trigger_max_bbox_area_fraction) not in {int, float}
            or not np.isfinite(float(self.component_fallback_trigger_max_bbox_area_fraction))
            or not 0.0 <= float(self.component_fallback_trigger_max_bbox_area_fraction) <= 1.0
        ):
            raise ValueError("component_fallback_trigger_max_bbox_area_fraction must be finite in [0,1]")
        if type(self.wire_minimum_run_px) is not int or self.wire_minimum_run_px <= 0:
            raise ValueError("wire_minimum_run_px must be a positive integer")
        for value, name in (
            (self.wire_gap_closing_radius_px, "wire_gap_closing_radius_px"),
            (self.wire_maximum_hole_area_px, "wire_maximum_hole_area_px"),
            (self.wire_maximum_component_pixels, "wire_maximum_component_pixels"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.pin_inference_overrides, dict):
            raise ValueError("pin_inference_overrides must be an object")
        allowed_pin_keys = {
            "score_threshold",
            "max_detections_per_component",
            "box_margin_px",
            "box_prior_mode",
            "nms_kernel",
            "nms_radius_heatmap",
            "base_length_pixels",
            "pin_budget_mode",
            "pin_budget_floor",
            "pin_budget_pitch_px",
        }
        unknown_pin_keys = sorted(set(self.pin_inference_overrides) - allowed_pin_keys)
        if unknown_pin_keys:
            raise ValueError(f"unsupported Pin inference override(s): {unknown_pin_keys}")
        self.pin_inference_overrides = dict(self.pin_inference_overrides)
        if not isinstance(self.wire_inference_overrides, dict):
            raise ValueError('wire_inference_overrides must be an object')
        unknown_wire_keys = sorted(set(self.wire_inference_overrides) - {'threshold'})
        if unknown_wire_keys:
            raise ValueError(f'unsupported Wire inference override(s): {unknown_wire_keys}')
        if 'threshold' in self.wire_inference_overrides:
            threshold = self.wire_inference_overrides['threshold']
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not np.isfinite(float(threshold))
                or not 0.0 <= float(threshold) <= 1.0
            ):
                raise ValueError('wire threshold override must be finite in [0,1]')
        self.wire_inference_overrides = dict(self.wire_inference_overrides)
        if not isinstance(self.ocr_inference_overrides, dict):
            raise ValueError("ocr_inference_overrides must be an object")
        unknown_ocr_keys = sorted(
            set(self.ocr_inference_overrides)
            - {
                "detector_threshold",
                "detector_min_area",
                "recognition_candidate_count",
                "vertical_rotation_enabled",
                "recognizer_backend",
            }
        )
        if unknown_ocr_keys:
            raise ValueError(f"unsupported OCR inference override(s): {unknown_ocr_keys}")
        if "detector_threshold" in self.ocr_inference_overrides:
            threshold = self.ocr_inference_overrides["detector_threshold"]
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not np.isfinite(float(threshold))
                or not 0.0 <= float(threshold) <= 1.0
            ):
                raise ValueError("ocr detector threshold override must be finite in [0,1]")
        if "detector_min_area" in self.ocr_inference_overrides:
            min_area = self.ocr_inference_overrides["detector_min_area"]
            if type(min_area) is not int or min_area <= 0:
                raise ValueError(
                    "ocr detector min area override must be a positive integer"
                )
        if "recognition_candidate_count" in self.ocr_inference_overrides:
            candidate_count = self.ocr_inference_overrides["recognition_candidate_count"]
            if type(candidate_count) is not int or candidate_count < 0:
                raise ValueError(
                    "ocr recognition candidate count must be a non-negative integer"
                )
        if "vertical_rotation_enabled" in self.ocr_inference_overrides:
            if type(self.ocr_inference_overrides["vertical_rotation_enabled"]) is not bool:
                raise ValueError("ocr vertical_rotation_enabled must be a boolean")
        if "recognizer_backend" in self.ocr_inference_overrides:
            backend_value = self.ocr_inference_overrides["recognizer_backend"]
            if not isinstance(backend_value, (str, dict)):
                raise ValueError(
                    "ocr recognizer_backend must be a string or object"
                )
        self.ocr_inference_overrides = dict(self.ocr_inference_overrides)

    def to_dict(self) -> dict[str, Any]:
        """Return the fully resolved, JSON-safe perception configuration."""

        payload = asdict(self)
        payload["component_tile_size"] = self.resolved_component_tile_size
        payload["component_tile_overlap"] = self.resolved_component_tile_overlap
        payload["wire_tile_size"] = self.resolved_wire_tile_size
        payload["wire_tile_overlap"] = self.resolved_wire_tile_overlap
        payload["ocr_component_roi_padding_px"] = self.resolved_ocr_component_roi_padding_px
        payload["pin_component_roi_padding_px"] = self.resolved_pin_component_roi_padding_px
        payload["component_internal_border_confirmation_categories"] = list(
            self.component_internal_border_confirmation_categories
        )
        payload["component_internal_border_refinement_categories"] = list(
            self.component_internal_border_refinement_categories
        )
        payload["component_targeted_recovery_categories"] = list(
            self.component_targeted_recovery_categories
        )
        return payload

    @property
    def resolved_component_tile_size(self) -> int:
        return self.tile_size if self.component_tile_size is None else self.component_tile_size

    @property
    def resolved_component_tile_overlap(self) -> int:
        return self.tile_overlap if self.component_tile_overlap is None else self.component_tile_overlap

    @property
    def resolved_wire_tile_size(self) -> int:
        return self.tile_size if self.wire_tile_size is None else self.wire_tile_size

    @property
    def resolved_wire_tile_overlap(self) -> int:
        return self.tile_overlap if self.wire_tile_overlap is None else self.wire_tile_overlap

    @property
    def resolved_ocr_component_roi_padding_px(self) -> float:
        return (
            self.component_roi_padding_px
            if self.ocr_component_roi_padding_px is None
            else self.ocr_component_roi_padding_px
        )

    @property
    def resolved_pin_component_roi_padding_px(self) -> float:
        return (
            self.component_roi_padding_px
            if self.pin_component_roi_padding_px is None
            else self.pin_component_roi_padding_px
        )


class CompositePerceptionProvider:
    """Connect independent learned/heuristic providers through Circuit-IR."""

    version = "composite-v1"

    def __init__(
        self,
        *,
        component_provider: ComponentProvider,
        ocr_provider: OCRProvider,
        pin_provider: PinProvider,
        wire_provider: WireProvider,
        junction_provider: JunctionProvider | None = None,
        config: CompositeProviderConfig | None = None,
    ) -> None:
        self.component_provider = component_provider
        self.ocr_provider = ocr_provider
        self.pin_provider = pin_provider
        self.wire_provider = wire_provider
        self.junction_provider = junction_provider
        self.config = config or CompositeProviderConfig()

    @property
    def model_task_plan(self) -> dict[str, Any]:
        """Describe exactly which learned adapters this provider can execute."""

        learned_junction_required = self.config.learned_junctions_enabled
        learned_junction_loaded = self.junction_provider is not None
        base_tasks = {"component", "ocr", "pin", "wire"}
        required_tasks = base_tasks | ({"junction"} if learned_junction_required else set())
        loaded_tasks = base_tasks | ({"junction"} if learned_junction_loaded else set())
        return {
            "contract": "production_model_task_plan_v1",
            "junction_mode": (
                "learned_model_plus_deterministic_vectorizer"
                if learned_junction_required
                else "deterministic_vectorizer"
            ),
            "required_model_tasks": sorted(required_tasks),
            "loaded_model_tasks": sorted(loaded_tasks),
            "learned_junction_model_required": learned_junction_required,
            "learned_junction_model_loaded": learned_junction_loaded,
            "learned_junction_model_invoked": (
                learned_junction_required and learned_junction_loaded
            ),
        }

    def build_ir(self, input_path: str | Path, ctx: RunContext | None = None) -> CircuitIR:
        _phase_timings: dict[str, float] = {}
        _t0 = time.monotonic()
        original = load_image(input_path)
        _phase_timings["image_load"] = time.monotonic() - _t0
        component_tiles = tile_image_view(
            original,
            tile_size=self.config.resolved_component_tile_size,
            overlap=self.config.resolved_component_tile_overlap,
        )
        wire_views = tile_image_view(
            original,
            tile_size=self.config.resolved_wire_tile_size,
            overlap=self.config.resolved_wire_tile_overlap,
        )
        wire_view_index = {view.view_id: view for view in wire_views}
        wire_view_index[original.view_id] = original

        # --- Phase: Component Detection ---
        _t_component = time.monotonic()
        component_candidates: list[Component] = []
        raw_component_count = 0
        score_filtered_component_count = 0
        component_predictions, _ = _predict_component_views(
            self.component_provider,
            component_tiles,
            ctx,
            batch_size=self.config.component_internal_border_refinement_batch_size,
        )
        for view, predicted in zip(component_tiles, component_predictions):
            restored = _restore_components(
                predicted,
                view,
                original_size=original.size,
                internal_border_margin_px=self.config.component_internal_border_margin_px,
            )
            raw_component_count += len(restored)
            gated = _filter_component_candidates(
                restored,
                min_score=self.config.component_candidate_min_score,
            )
            score_filtered_component_count += len(restored) - len(gated)
            component_candidates.extend(gated)
        component_candidates, border_refinement = _refine_internal_border_components(
            self.component_provider,
            component_candidates,
            original,
            ctx,
            enabled=self.config.component_internal_border_refinement_enabled,
            categories=self.config.component_internal_border_refinement_categories,
            tile_size=self.config.resolved_component_tile_size,
            match_iou=self.config.component_internal_border_refinement_iou,
            max_candidates=(
                self.config.component_internal_border_refinement_max_candidates
            ),
            min_score=self.config.component_candidate_min_score,
            internal_border_margin_px=(
                self.config.component_internal_border_margin_px
            ),
            batch_size=self.config.component_inference_batch_size,
        )
        confirmed_component_candidates = _filter_unconfirmed_internal_border_components(
            component_candidates,
            enabled=self.config.component_internal_border_confirmation_enabled,
            categories=self.config.component_internal_border_confirmation_categories,
            confirmation_iou=self.config.component_internal_border_confirmation_iou,
            confirmation_metric=self.config.component_internal_border_confirmation_metric,
        )
        border_confirmation_filtered_component_count = (
            len(component_candidates) - len(confirmed_component_candidates)
        )
        components = _class_aware_nms(
            confirmed_component_candidates,
            self.config.component_nms_iou,
            prefer_non_border=self.config.component_center_preferred_nms_enabled,
            center_preference_score_bonus=(
                self.config.component_center_preference_score_bonus
            ),
        )
        nms_component_count = len(components)
        primary_component_count = len(components)
        primary_box_count = sum(
            str(item.category).casefold() == "box" for item in components
        )
        primary_box_fraction = primary_box_count / max(1, primary_component_count)
        primary_bbox_area = sum(
            max(0.0, float(item.bbox.x2) - float(item.bbox.x1))
            * max(0.0, float(item.bbox.y2) - float(item.bbox.y1))
            for item in components
        )
        primary_bbox_area_fraction = primary_bbox_area / max(
            1.0, float(original.width) * float(original.height)
        )
        fallback_triggered = (
            self.config.component_adaptive_fallback_enabled
            and primary_component_count <= self.config.component_fallback_trigger_max_count
            and primary_box_fraction <= self.config.component_fallback_trigger_max_box_fraction
            and primary_bbox_area_fraction <= self.config.component_fallback_trigger_max_bbox_area_fraction
        )
        fallback_raw_component_count = 0
        fallback_score_filtered_component_count = 0
        fallback_border_confirmation_filtered_component_count = 0
        fallback_border_refinement = _empty_border_refinement_stats()
        if fallback_triggered:
            fallback_candidates: list[Component] = []
            fallback_tiles = tile_image_view(
                original,
                tile_size=self.config.component_fallback_tile_size,
                overlap=self.config.component_fallback_tile_overlap,
            )
            fallback_predictions, _ = _predict_component_views(
                self.component_provider,
                fallback_tiles,
                ctx,
                batch_size=self.config.component_inference_batch_size,
            )
            for view, predicted in zip(fallback_tiles, fallback_predictions):
                restored = _restore_components(
                    predicted,
                    view,
                    original_size=original.size,
                    internal_border_margin_px=self.config.component_internal_border_margin_px,
                )
                fallback_raw_component_count += len(restored)
                gated = _filter_component_candidates(
                    restored,
                    min_score=self.config.component_candidate_min_score,
                )
                fallback_score_filtered_component_count += len(restored) - len(gated)
                fallback_candidates.extend(gated)
            fallback_candidates, fallback_border_refinement = (
                _refine_internal_border_components(
                    self.component_provider,
                    fallback_candidates,
                    original,
                    ctx,
                    enabled=(
                        self.config.component_internal_border_refinement_enabled
                    ),
                    categories=(
                        self.config.component_internal_border_refinement_categories
                    ),
                    tile_size=self.config.component_fallback_tile_size,
                    match_iou=(
                        self.config.component_internal_border_refinement_iou
                    ),
                    max_candidates=(
                        self.config.component_internal_border_refinement_max_candidates
                    ),
                    min_score=self.config.component_candidate_min_score,
                    internal_border_margin_px=(
                        self.config.component_internal_border_margin_px
                    ),
                    batch_size=(
                        self.config.component_internal_border_refinement_batch_size
                    ),
                )
            )
            confirmed_fallback_candidates = _filter_unconfirmed_internal_border_components(
                fallback_candidates,
                enabled=self.config.component_internal_border_confirmation_enabled,
                categories=self.config.component_internal_border_confirmation_categories,
                confirmation_iou=self.config.component_internal_border_confirmation_iou,
                confirmation_metric=(
                    self.config.component_internal_border_confirmation_metric
                ),
            )
            fallback_border_confirmation_filtered_component_count = (
                len(fallback_candidates) - len(confirmed_fallback_candidates)
            )
            components = _class_aware_nms(
                confirmed_fallback_candidates,
                self.config.component_nms_iou,
                prefer_non_border=self.config.component_center_preferred_nms_enabled,
                center_preference_score_bonus=(
                    self.config.component_center_preference_score_bonus
                ),
            )
            nms_component_count = len(components)
        components, targeted_recovery = _recover_targeted_components(
            self.component_provider,
            components,
            original,
            ctx,
            enabled=self.config.component_targeted_recovery_enabled,
            categories=self.config.component_targeted_recovery_categories,
            tile_size=self.config.component_targeted_recovery_tile_size,
            tile_overlap=self.config.component_targeted_recovery_tile_overlap,
            min_dimension_px=(
                self.config.component_targeted_recovery_min_dimension_px
            ),
            batch_size=self.config.component_targeted_recovery_batch_size,
            min_score=self.config.component_candidate_min_score,
            internal_border_margin_px=(
                self.config.component_internal_border_margin_px
            ),
            nms_iou=self.config.component_nms_iou,
            prefer_non_border=self.config.component_center_preferred_nms_enabled,
            center_preference_score_bonus=(
                self.config.component_center_preference_score_bonus
            ),
        )
        nms_component_count = len(components)
        cap_filtered_component_count = 0
        if self.config.component_max_candidates is not None:
            ranked = sorted(
                components,
                key=lambda item: (
                    -item.score,
                    item.bbox.y1,
                    item.bbox.x1,
                    item.bbox.y2,
                    item.bbox.x2,
                    item.category,
                    item.internal_id,
                ),
            )
            kept = ranked[: self.config.component_max_candidates]
            cap_filtered_component_count = len(ranked) - len(kept)
            components = sorted(
                kept,
                key=lambda item: (
                    item.bbox.y1,
                    item.bbox.x1,
                    item.category,
                    item.internal_id,
                ),
            )
        _assign_global_ids(components, "component")

        ocr_component_views = [
            make_component_view(
                original,
                component,
                padding=max(
                    max(component.bbox.width, component.bbox.height)
                    * self.config.component_roi_padding_factor,
                    self.config.resolved_ocr_component_roi_padding_px,
                ),
                view_id_prefix="component-ocr",
            )
            for component in components
        ]
        pin_component_views = [
            make_component_view(
                original,
                component,
                padding=max(
                    max(component.bbox.width, component.bbox.height)
                    * self.config.component_roi_padding_factor,
                    self.config.resolved_pin_component_roi_padding_px,
                ),
                view_id_prefix="component-pin",
            )
            for component in components
        ]
        ocr_views = tile_image_view(
            original,
            tile_size=self.config.ocr_tile_size,
            overlap=self.config.ocr_tile_overlap,
        )
        _phase_timings["component_detect"] = time.monotonic() - _t_component
        # --- Phase: OCR ---
        _t_ocr = time.monotonic()
        texts = run_multilevel_ocr(
            self.ocr_provider,
            ocr_views,
            ocr_component_views,
            ctx,
            MultilevelOCRConfig(
                global_batch_size=self.config.ocr_global_batch_size,
                component_batch_size=self.config.ocr_component_batch_size,
                strip_batch_size=self.config.ocr_strip_batch_size,
                nms_iou_threshold=self.config.ocr_nms_iou,
                nms_center_distance_px=self.config.ocr_nms_center_distance_px,
                merge_conflicting_text=self.config.ocr_merge_conflicting_text,
                conflicting_text_iou_threshold=self.config.ocr_conflicting_text_iou,
            ),
        )
        _phase_timings["ocr_recognize"] = time.monotonic() - _t_ocr
        # --- Phase: Pin Detection ---
        _t_pin = time.monotonic()
        pin_view_index = {
            view.view_id: view
            for item in pin_component_views
            for view in [item.full_roi, *item.side_strips.values()]
        }
        pin_view_owners = {
            view.view_id: item.component_id
            for item in pin_component_views
            for view in [item.full_roi, *item.side_strips.values()]
        }
        full_image_pins = isinstance(self.pin_provider, FullImagePinProvider)
        if full_image_pins:
            pins = _restore_pins(
                self.pin_provider.predict_image(original, ctx),
                {original.view_id: original},
                {},
                unowned_views=frozenset({original.view_id}),
            )
            component_ids = {item.internal_id for item in components}
            if any(pin.component_id is not None and pin.component_id not in component_ids for pin in pins):
                raise ValueError("full-image Pin provider references an unknown component")
        else:
            pins = _restore_pins(
                self.pin_provider.predict(pin_component_views, ctx),
                pin_view_index,
                pin_view_owners,
            )
        _assign_global_ids(pins, "pin")
        _phase_timings["pin_detect"] = time.monotonic() - _t_pin
        # --- Phase: Wire Segmentation ---
        _t_wire = time.monotonic()
        wire_masks = self.wire_provider.segment(wire_views, ctx)
        segments, vectorized_junctions, wire_lineage = _vectorize_masks(
            wire_masks,
            wire_view_index,
            self.config,
        )
        model_junctions = (
            _restore_junctions(
                self.junction_provider.classify(wire_views, segments, ctx),
                wire_view_index,
            )
            if (
                self.junction_provider is not None
                and self.config.learned_junctions_enabled
            )
            else []
        )
        # A learned classifier is authoritative where it emits a candidate;
        # deterministic skeleton endpoints/T/crosses remain available as graph
        # topology evidence elsewhere.
        junctions = _merge_junctions(
            vectorized_junctions,
            model_junctions,
            tolerance=self.config.wire_tile_dedup_tolerance_px,
        )
        _assign_global_ids(junctions, "junction")
        _phase_timings["wire_segment"] = time.monotonic() - _t_wire
        _phase_timings["total_perception"] = time.monotonic() - _t0
        image_info = original.image_info(image_id=ctx.image_id if ctx else Path(input_path).stem)
        return CircuitIR(
            state=IRState.PERCEIVED,
            image=image_info,
            transforms=original.transform.to_records(),
            components=components,
            pins=pins,
            texts=texts,
            segments=segments,
            junctions=junctions,
            repairs=wire_lineage["repairs"],
            metadata={
                "provider": {"name": type(self).__name__, "version": self.version},
                "perception_config": self.config.to_dict(),
                "model_task_plan": self.model_task_plan,
                "model_versions": {
                    "component": self.component_provider.version,
                    "ocr": self.ocr_provider.version,
                    "pin": self.pin_provider.version,
                    "wire": self.wire_provider.version,
                    **(
                        {"junction": self.junction_provider.version}
                        if (
                            self.junction_provider is not None
                            and self.config.learned_junctions_enabled
                        )
                        else {}
                    ),
                },
                "quarantined_model_versions": (
                    {"junction": self.junction_provider.version}
                    if (
                        self.junction_provider is not None
                        and not self.config.learned_junctions_enabled
                    )
                    else {}
                ),
                "views": [view.view_id for view in wire_views],
                "component_views": [view.view_id for view in component_tiles],
                "wire_views": [view.view_id for view in wire_views],
                "wire_breaks": wire_lineage["breaks"],
                "ocr_component_views": [
                    view.full_roi.view_id for view in ocr_component_views
                ],
                "pin_component_views": [
                    view.full_roi.view_id for view in pin_component_views
                ],
                "pin_inference_scope": "full_image" if full_image_pins else "component_roi",
                "component_candidate_gate": {
                    "raw_count": raw_component_count,
                    "score_threshold": self.config.component_candidate_min_score,
                    "score_filtered_count": score_filtered_component_count,
                    "border_confirmation_filtered_count": (
                        border_confirmation_filtered_component_count
                    ),
                    "internal_border_refinement": border_refinement,
                    "primary_count": primary_component_count,
                    "primary_box_fraction": primary_box_fraction,
                    "primary_bbox_area_fraction": primary_bbox_area_fraction,
                    "adaptive_fallback": {
                        "enabled": self.config.component_adaptive_fallback_enabled,
                        "triggered": fallback_triggered,
                        "tile_size": self.config.component_fallback_tile_size,
                        "tile_overlap": self.config.component_fallback_tile_overlap,
                        "trigger_max_count": self.config.component_fallback_trigger_max_count,
                        "trigger_max_box_fraction": self.config.component_fallback_trigger_max_box_fraction,
                        "trigger_max_bbox_area_fraction": self.config.component_fallback_trigger_max_bbox_area_fraction,
                        "raw_count": fallback_raw_component_count,
                        "score_filtered_count": fallback_score_filtered_component_count,
                        "border_confirmation_filtered_count": (
                            fallback_border_confirmation_filtered_component_count
                        ),
                        "internal_border_refinement": fallback_border_refinement,
                    },
                    "targeted_recovery": targeted_recovery,
                    "after_nms_count": nms_component_count,
                    "max_candidates": self.config.component_max_candidates,
                    "cap_filtered_count": cap_filtered_component_count,
                    "final_count": len(components),
                },
                "phase_timings_seconds": {
                    k: round(v, 6) for k, v in _phase_timings.items()
                },
            },
        )


def _restore_components(
    candidates: list[Component],
    view: ImageView,
    *,
    original_size: tuple[int, int] | None = None,
    internal_border_margin_px: float = 0.0,
) -> list[Component]:
    restored: list[Component] = []
    for candidate in candidates:
        item = candidate.model_copy(deep=True)
        local_bbox = item.bbox.model_copy(deep=True)
        corners = [
            view.to_original(Point(x=item.bbox.x1, y=item.bbox.y1)),
            view.to_original(Point(x=item.bbox.x2, y=item.bbox.y1)),
            view.to_original(Point(x=item.bbox.x1, y=item.bbox.y2)),
            view.to_original(Point(x=item.bbox.x2, y=item.bbox.y2)),
        ]
        item.bbox = BBox(
            x1=min(point.x for point in corners),
            y1=min(point.y for point in corners),
            x2=max(point.x for point in corners),
            y2=max(point.y for point in corners),
        )
        if not item.evidence:
            item.evidence = [
                Evidence(
                    evidence_id=f"component-provider:{view.view_id}:{item.internal_id}",
                    source="detector",
                    view_id=view.view_id,
                    score=item.score,
                )
            ]
        for evidence in item.evidence:
            evidence.metadata.setdefault("provider_local_id", candidate.internal_id)
            touches_border, border_distance = _internal_tile_border_state(
                local_bbox,
                view=view,
                original_size=original_size,
                margin_px=internal_border_margin_px,
            )
            evidence.metadata["touches_internal_tile_border"] = touches_border
            evidence.metadata["internal_tile_border_distance_px"] = border_distance
        bind_bbox_evidence(
            item.evidence,
            view=view,
            local_bbox=local_bbox,
            original_bbox=item.bbox,
            object_kind="component",
        )
        restored.append(item)
    return restored


def _restore_pins(
    pins: list[Pin],
    views: dict[str, ImageView],
    view_owners: dict[str, str],
    *,
    unowned_views: frozenset[str] = frozenset(),
) -> list[Pin]:
    output: list[Pin] = []
    for pin in pins:
        item = pin.model_copy(deep=True)
        local_point = item.point.model_copy(deep=True)
        local_base = item.base.model_copy(deep=True) if item.base is not None else None
        view = _source_view(item.evidence, views, object_id=item.internal_id, required=True)
        if view is not None:
            expected_owner = view_owners.get(view.view_id)
            if expected_owner is None and view.view_id not in unowned_views:
                raise ValueError(
                    f"pin {item.internal_id} source view has no component owner"
                )
            if expected_owner is not None and item.component_id is None:
                # The ComponentView is an explicit inference context, so this
                # is provenance restoration rather than semantic guessing.
                item.component_id = expected_owner
            elif expected_owner is not None and item.component_id != expected_owner:
                raise ValueError(
                    f"pin {item.internal_id} component owner {item.component_id!r} "
                    f"conflicts with source view owner {expected_owner!r}"
                )
            if expected_owner is not None:
                for evidence in item.evidence:
                    evidence.metadata.setdefault("component_id", expected_owner)
            item.point = view.to_original(item.point)
            if item.base is not None:
                item.base = view.to_original(item.base)
            bind_point_evidence(
                item.evidence,
                view=view,
                local_point=local_point,
                original_point=item.point,
                object_kind="pin",
                local_base=local_base,
                original_base=item.base,
            )
        output.append(item)
    return output


def _source_view(
    evidence: list[Evidence],
    views: dict[str, ImageView],
    *,
    object_id: str,
    required: bool = False,
) -> ImageView | None:
    identifiers = {item.view_id for item in evidence}
    if not identifiers:
        if required:
            raise ValueError(f"object {object_id} has no source view evidence")
        return None
    if len(identifiers) != 1:
        raise ValueError(f"object {object_id} has ambiguous source views: {sorted(identifiers)}")
    identifier = next(iter(identifiers))
    if identifier not in views:
        raise ValueError(f"object {object_id} references unknown source view {identifier}")
    return views[identifier]


def _restore_junctions(junctions: list[Junction], views: dict[str, ImageView]) -> list[Junction]:
    output: list[Junction] = []
    for junction in junctions:
        item = junction.model_copy(deep=True)
        local_point = item.point.model_copy(deep=True)
        view = _source_view(item.evidence, views, object_id=item.internal_id, required=True)
        if view is not None:
            item.point = view.to_original(item.point)
            bind_point_evidence(
                item.evidence,
                view=view,
                local_point=local_point,
                original_point=item.point,
                object_kind="junction",
            )
        output.append(item)
    return output


def _assign_global_ids(items: list[Any], prefix: str) -> None:
    ordered = sorted(items, key=_object_sort_key)
    for index, item in enumerate(ordered, start=1):
        local_id = item.internal_id
        item.internal_id = f"{prefix}:{index:06d}"
        for evidence in item.evidence:
            evidence.metadata.setdefault("provider_local_id", local_id)


def _object_sort_key(item: Any) -> tuple[Any, ...]:
    bbox = getattr(item, "bbox", None)
    point = getattr(item, "point", None)
    if bbox is not None:
        geometry = (bbox.y1, bbox.x1, bbox.y2, bbox.x2)
    elif point is not None:
        geometry = (point.y, point.x, point.y, point.x)
    else:
        geometry = (0.0, 0.0, 0.0, 0.0)
    return (*geometry, str(getattr(item, "category", "")), str(item.internal_id))


def _class_aware_nms(
    candidates: list[Component],
    threshold: float,
    *,
    prefer_non_border: bool = False,
    center_preference_score_bonus: float = 1.0,
) -> list[Component]:
    kept: list[Component] = []
    ranked = sorted(
        candidates,
        key=lambda item: (
            -(
                item.score
                + (
                    center_preference_score_bonus
                    if prefer_non_border and not _touches_internal_tile_border(item)
                    else 0.0
                )
            ),
            _touches_internal_tile_border(item) if prefer_non_border else False,
            -item.score,
            item.category,
            item.internal_id,
        ),
    )
    for candidate in ranked:
        if any(
            existing.category == candidate.category and _iou(existing.bbox, candidate.bbox) > threshold
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda item: (item.bbox.y1, item.bbox.x1, item.category, item.internal_id))


def _touches_internal_tile_border(item: Component) -> bool:
    return any(
        evidence.metadata.get("touches_internal_tile_border") is True
        for evidence in item.evidence
    )


def _empty_border_refinement_stats() -> dict[str, int]:
    return {
        "eligible_count": 0,
        "attempted_count": 0,
        "raw_refinement_count": 0,
        "matched_count": 0,
        "replaced_count": 0,
    }


def _predict_component_views(
    provider: ComponentProvider,
    views: Sequence[ImageView],
    ctx: RunContext | None,
    *,
    batch_size: int,
) -> tuple[list[list[Component]], int]:
    """Predict Component views in bounded batches when the provider supports it."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("component inference batch_size must be a positive integer")
    if not views:
        return [], 0
    predict_many = getattr(provider, "predict_many", None)
    if not callable(predict_many) or batch_size == 1:
        output = []
        for view in views:
            if ctx is not None:
                ctx.time_budget.check(stage="component_view")
            output.append(provider.predict(view, ctx))
            if ctx is not None:
                ctx.time_budget.check(stage="component_view")
        return output, len(views)

    output: list[list[Component]] = []
    call_count = 0
    for start in range(0, len(views), batch_size):
        if ctx is not None:
            ctx.time_budget.check(stage="component_batch")
        batch = list(views[start : start + batch_size])
        predicted = predict_many(batch, ctx)
        if ctx is not None:
            ctx.time_budget.check(stage="component_batch")
        call_count += 1
        if not isinstance(predicted, (list, tuple)) or len(predicted) != len(batch):
            raise RuntimeError(
                "batched Component provider returned invalid result cardinality"
            )
        for index, rows in enumerate(predicted):
            if not isinstance(rows, (list, tuple)):
                raise RuntimeError(
                    "batched Component provider result must contain candidate arrays"
                )
            if any(not isinstance(item, Component) for item in rows):
                raise RuntimeError(
                    f"batched Component provider emitted an invalid candidate at batch index {index}"
                )
            output.append(list(rows))
    return output, call_count


def _empty_targeted_recovery_stats() -> dict[str, Any]:
    return {
        "triggered": False,
        "tile_count": 0,
        "inference_call_count": 0,
        "raw_candidate_count": 0,
        "targeted_candidate_count": 0,
        "added_after_nms_count": 0,
        "candidate_count_after_nms": 0,
    }


def _recover_targeted_components(
    provider: ComponentProvider,
    primary: list[Component],
    original: ImageView,
    ctx: RunContext | None,
    *,
    enabled: bool,
    categories: tuple[str, ...],
    tile_size: int,
    tile_overlap: int,
    min_dimension_px: int,
    batch_size: int,
    min_score: float,
    internal_border_margin_px: float,
    nms_iou: float,
    prefer_non_border: bool,
    center_preference_score_bonus: float,
) -> tuple[list[Component], dict[str, Any]]:
    """Recover configured small-symbol classes from a higher-resolution tiling."""

    stats = _empty_targeted_recovery_stats()
    stats["candidate_count_after_nms"] = len(primary)
    if (
        not enabled
        or not categories
        or max(original.width, original.height) < min_dimension_px
    ):
        return list(primary), stats

    views = tile_image_view(original, tile_size=tile_size, overlap=tile_overlap)
    predictions, call_count = _predict_component_views(
        provider,
        views,
        ctx,
        batch_size=batch_size,
    )
    targeted_names = set(categories)
    recovered: list[Component] = []
    raw_count = 0
    for view, rows in zip(views, predictions):
        restored = _restore_components(
            rows,
            view,
            original_size=original.size,
            internal_border_margin_px=internal_border_margin_px,
        )
        gated = _filter_component_candidates(restored, min_score=min_score)
        raw_count += len(gated)
        recovered.extend(item for item in gated if item.category in targeted_names)

    merged = _class_aware_nms(
        [*primary, *recovered],
        nms_iou,
        prefer_non_border=prefer_non_border,
        center_preference_score_bonus=center_preference_score_bonus,
    )
    stats.update(
        {
            "triggered": True,
            "tile_count": len(views),
            "inference_call_count": call_count,
            "raw_candidate_count": raw_count,
            "targeted_candidate_count": len(recovered),
            "added_after_nms_count": max(0, len(merged) - len(primary)),
            "candidate_count_after_nms": len(merged),
        }
    )
    return merged, stats


def _refine_internal_border_components(
    provider: ComponentProvider,
    candidates: list[Component],
    original: ImageView,
    ctx: RunContext | None,
    *,
    enabled: bool,
    categories: tuple[str, ...],
    tile_size: int,
    match_iou: float,
    max_candidates: int,
    min_score: float,
    internal_border_margin_px: float,
    batch_size: int = 1,
) -> tuple[list[Component], dict[str, int]]:
    """Re-detect selected internal-edge candidates in centred image crops.

    The original candidate is retained unless the centred pass returns a
    same-class box with sufficient intersection over the smaller box.  This
    keeps the feature recall-safe while allowing a complete, non-truncated
    box to replace the edge detection before confirmation and NMS.
    """

    stats = _empty_border_refinement_stats()
    if not enabled or not categories or not candidates:
        return list(candidates), stats

    targeted = set(categories)
    eligible = [
        item
        for item in candidates
        if item.category in targeted and _touches_internal_tile_border(item)
    ]
    stats["eligible_count"] = len(eligible)
    selected = sorted(
        eligible,
        key=lambda item: (
            -item.score,
            item.bbox.y1,
            item.bbox.x1,
            item.category,
            item.internal_id,
        ),
    )[:max_candidates]
    replacements: dict[int, Component] = {}
    crop_groups: dict[
        tuple[int, int, int, int], tuple[BBox, list[Component]]
    ] = {}
    for candidate in selected:
        crop_box = _centred_component_crop_box(
            candidate.bbox,
            image_size=original.size,
            tile_size=tile_size,
        )
        crop_key = (
            int(crop_box.x1),
            int(crop_box.y1),
            int(crop_box.x2),
            int(crop_box.y2),
        )
        if crop_key not in crop_groups:
            crop_groups[crop_key] = (crop_box, [])
        crop_groups[crop_key][1].append(candidate)

    refinement_jobs: list[tuple[ImageView, list[Component]]] = []
    for index, (_, (crop_box, crop_candidates)) in enumerate(sorted(crop_groups.items())):
        refinement_view = crop_image_view(
            original,
            crop_box,
            view_id=f"component-refine:{index:04d}",
        )
        refinement_jobs.append((refinement_view, crop_candidates))
    predicted_jobs, _ = _predict_component_views(
        provider,
        [item[0] for item in refinement_jobs],
        ctx,
        batch_size=batch_size,
    )
    for (refinement_view, crop_candidates), predicted in zip(
        refinement_jobs, predicted_jobs
    ):
        stats["attempted_count"] += 1
        restored = _restore_components(
            predicted,
            refinement_view,
            original_size=original.size,
            internal_border_margin_px=internal_border_margin_px,
        )
        gated = _filter_component_candidates(restored, min_score=min_score)
        stats["raw_refinement_count"] += len(gated)
        for candidate in crop_candidates:
            matches = [
                (
                    _confirmation_overlap(
                        candidate.bbox,
                        item.bbox,
                        metric="intersection_over_smaller",
                    ),
                    item,
                )
                for item in gated
                if item.category == candidate.category
            ]
            matches = [item for item in matches if item[0] >= match_iou]
            if not matches:
                continue
            stats["matched_count"] += 1
            overlap, replacement = max(
                matches,
                key=lambda item: (
                    not _touches_internal_tile_border(item[1]),
                    item[0],
                    item[1].score,
                    item[1].bbox.width * item[1].bbox.height,
                    -item[1].bbox.y1,
                    -item[1].bbox.x1,
                ),
            )
            replacement = replacement.model_copy(deep=True)
            for evidence in replacement.evidence:
                evidence.metadata["internal_border_refinement"] = True
                evidence.metadata["refines_component_local_id"] = candidate.internal_id
                evidence.metadata[
                    "refinement_match_intersection_over_smaller"
                ] = overlap
            replacements[id(candidate)] = replacement

    stats["replaced_count"] = len(replacements)
    return [replacements.get(id(item), item) for item in candidates], stats


def _centred_component_crop_box(
    bbox: BBox,
    *,
    image_size: tuple[int, int],
    tile_size: int,
) -> BBox:
    image_width, image_height = image_size
    crop_width = min(int(tile_size), int(image_width))
    crop_height = min(int(tile_size), int(image_height))
    center_x = (float(bbox.x1) + float(bbox.x2)) / 2.0
    center_y = (float(bbox.y1) + float(bbox.y2)) / 2.0
    left = max(0, min(image_width - crop_width, round(center_x - crop_width / 2.0)))
    top = max(0, min(image_height - crop_height, round(center_y - crop_height / 2.0)))
    return BBox(
        x1=float(left),
        y1=float(top),
        x2=float(left + crop_width),
        y2=float(top + crop_height),
    )


def _filter_unconfirmed_internal_border_components(
    candidates: list[Component],
    *,
    enabled: bool,
    categories: tuple[str, ...],
    confirmation_iou: float,
    confirmation_metric: str = "iou",
) -> list[Component]:
    if not enabled or not categories:
        return list(candidates)
    targeted = set(categories)
    kept: list[Component] = []
    source_views = {
        id(item): {evidence.view_id for evidence in item.evidence}
        for item in candidates
    }
    for candidate in candidates:
        if (
            candidate.category not in targeted
            or not _touches_internal_tile_border(candidate)
        ):
            kept.append(candidate)
            continue
        candidate_views = source_views[id(candidate)]
        confirmed = any(
            other is not candidate
            and other.category == candidate.category
            and candidate_views.isdisjoint(source_views[id(other)])
            and _confirmation_overlap(
                candidate.bbox,
                other.bbox,
                metric=confirmation_metric,
            )
            >= confirmation_iou
            for other in candidates
        )
        if confirmed:
            kept.append(candidate)
    return kept


def _confirmation_overlap(left: BBox, right: BBox, *, metric: str) -> float:
    if metric == "iou":
        return _iou(left, right)
    if metric != "intersection_over_smaller":
        raise ValueError(f"unsupported border confirmation metric: {metric}")
    width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    smaller_area = min(left.width * left.height, right.width * right.height)
    return width * height / smaller_area if smaller_area > 0.0 else 0.0


def _internal_tile_border_state(
    bbox: BBox,
    *,
    view: ImageView,
    original_size: tuple[int, int] | None,
    margin_px: float,
) -> tuple[bool, float | None]:
    if original_size is None:
        return False, None
    crop = view.metadata.get("crop_box")
    if not isinstance(crop, list) or len(crop) != 4:
        return False, None
    left, top, right, bottom = (float(value) for value in crop)
    original_width, original_height = original_size
    distances: list[float] = []
    if left > 0.0:
        distances.append(float(bbox.x1))
    if right < float(original_width):
        distances.append(float(view.width) - float(bbox.x2))
    if top > 0.0:
        distances.append(float(bbox.y1))
    if bottom < float(original_height):
        distances.append(float(view.height) - float(bbox.y2))
    if not distances:
        return False, None
    distance = max(0.0, min(distances))
    return distance <= float(margin_px), distance


def _iou(left: BBox, right: BBox) -> float:
    width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = width * height
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0.0 else 0.0


def _filter_component_candidates(
    candidates: list[Component],
    *,
    min_score: float,
) -> list[Component]:
    """Apply the downstream fan-out gate without mutating provider results."""

    if not isinstance(min_score, (int, float)) or isinstance(min_score, bool):
        raise ValueError("component_candidate_min_score must be a number")
    threshold = float(min_score)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("component_candidate_min_score must be finite in [0,1]")
    return [item for item in candidates if item.score >= threshold]


def _vectorize_masks(
    masks: list[WireMask],
    views: dict[str, ImageView],
    config: CompositeProviderConfig,
) -> tuple[list[WireSegment], list[Junction], dict[str, Any]]:
    polylines: list[SkeletonPolyline] = []
    junction_candidates: list[tuple[WireJunctionCandidate, str, list[Evidence], float]] = []
    vectorizer_config = WireVectorizerConfig(
        gap_closing_radius=config.wire_gap_closing_radius_px,
        maximum_hole_area=config.wire_maximum_hole_area_px,
        maximum_component_pixels=config.wire_maximum_component_pixels,
        minimum_component_pixels=2,
        minimum_polyline_pixels=max(2, config.wire_minimum_run_px),
        rdp_epsilon=config.wire_rdp_epsilon_px,
    )
    polyline_metadata: dict[
        str, tuple[float, list[Evidence], str, tuple[tuple[float, float], ...]]
    ] = {}
    virtual_repairs: list[tuple[Any, str, float]] = []
    vectorizer_breaks: list[tuple[Any, str]] = []
    for mask_index, wire_mask in enumerate(sorted(masks, key=lambda item: item.view_id)):
        view = views.get(wire_mask.view_id)
        if view is None:
            raise ValueError(f"wire mask references unknown view {wire_mask.view_id}")
        array = np.asarray(wire_mask.mask)
        if array.shape != (view.height, view.width):
            raise ValueError(
                f"wire mask shape {array.shape} does not match view {wire_mask.view_id} {(view.height, view.width)}"
            )
        result = vectorize_wire_mask(
            array,
            config=vectorizer_config,
            id_namespace=f"mask-{mask_index:04d}-{wire_mask.view_id}",
        )
        for repair in result.virtual_repairs:
            virtual_repairs.append((repair, wire_mask.view_id, wire_mask.score))
        for record in result.breaks:
            vectorizer_breaks.append((record, wire_mask.view_id))
        local_to_global: dict[str, str] = {}
        for polyline in result.polylines:
            points = tuple(
                (restored.x, restored.y)
                for restored in (
                    view.to_original(Point(x=x, y=y)) for x, y in polyline.points
                )
            )
            global_polyline = SkeletonPolyline(
                internal_id=polyline.internal_id,
                points=points,
                pixel_count=polyline.pixel_count,
                length=sum(
                    ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                    for (x1, y1), (x2, y2) in zip(points, points[1:])
                ),
                virtual_repair_ids=polyline.virtual_repair_ids,
            )
            polylines.append(global_polyline)
            local_to_global[polyline.internal_id] = global_polyline.internal_id
            polyline_metadata[global_polyline.internal_id] = (
                wire_mask.score,
                list(wire_mask.evidence),
                wire_mask.view_id,
                tuple((float(x), float(y)) for x, y in polyline.points),
            )
        for junction in result.junctions:
            junction_candidates.append(
                (junction, wire_mask.view_id, list(wire_mask.evidence), wire_mask.score)
            )

    kept = deduplicate_polylines(
        polylines,
        tolerance=config.wire_tile_dedup_tolerance_px,
    )
    segments: list[WireSegment] = []
    polyline_to_segments: dict[str, list[str]] = {}
    for polyline_index, polyline in enumerate(kept, start=1):
        score, evidence, source_view, local_points = polyline_metadata[
            polyline.internal_id
        ]
        if len(local_points) != len(polyline.points):
            raise ValueError(
                f"polyline {polyline.internal_id} local/global point counts differ"
            )
        source_image_view = views[source_view]
        segment_ids: list[str] = []
        for segment_index, ((start, end), (local_start, local_end)) in enumerate(
            zip(
                zip(polyline.points, polyline.points[1:]),
                zip(local_points, local_points[1:]),
            ),
            start=1,
        ):
            if start == end:
                continue
            segment_id = f"mask-segment:{polyline_index:06d}:{segment_index:04d}"
            segment_ids.append(segment_id)
            dx, dy = end[0] - start[0], end[1] - start[1]
            # Channel identity is used to keep an unproven crossing split.
            # Collapsing both diagonal slopes into one generic channel made an
            # X-shaped non-connected crossing self-connect before the Junction
            # classifier could veto it.  The sign of dx*dy is invariant to
            # endpoint order and distinguishes the two undirected diagonals.
            channel = (
                "horizontal"
                if abs(dy) <= 1e-9
                else "vertical"
                if abs(dx) <= 1e-9
                else "diagonal_positive"
                if dx * dy > 0
                else "diagonal_negative"
            )
            segment = WireSegment(
                internal_id=segment_id,
                start=Point(x=start[0], y=start[1]),
                end=Point(x=end[0], y=end[1]),
                score=score,
                source_polyline_id=polyline.internal_id,
                synthetic=bool(polyline.virtual_repair_ids),
                channel=channel,
                evidence=[
                    *evidence,
                    Evidence(
                        evidence_id=f"wire-vectorizer:{segment_id}",
                        source="geometry",
                        view_id=source_view,
                        score=score,
                        metadata={"vectorizer": "skeleton-pixel-graph-rdp-v1"},
                    ),
                    *(
                        Evidence(
                            evidence_id=f"wire-virtual-repair:{segment_id}:{repair_id}",
                            source="repair",
                            view_id=source_view,
                            score=score,
                            payload_ref=repair_id,
                            metadata={
                                "rule": "mask_closing_virtual_gap",
                                "virtual": True,
                                "solid_wire": False,
                                "segment_id": segment_id,
                            },
                        )
                        for repair_id in polyline.virtual_repair_ids
                    ),
                ],
            )
            bind_segment_evidence(
                segment.evidence,
                view=source_image_view,
                local_start=Point(x=local_start[0], y=local_start[1]),
                local_end=Point(x=local_end[0], y=local_end[1]),
                original_start=segment.start,
                original_end=segment.end,
            )
            segments.append(segment)
        polyline_to_segments[polyline.internal_id] = segment_ids

    segment_to_polyline = {
        segment_id: polyline_id
        for polyline_id, segment_ids in polyline_to_segments.items()
        for segment_id in segment_ids
    }
    kept_by_id = {item.internal_id: item for item in kept}

    def restore(point: Any, view_id: str) -> list[float] | None:
        if point is None:
            return None
        restored = views[view_id].to_original(Point(x=float(point[0]), y=float(point[1])))
        return [float(restored.x), float(restored.y)]

    lineage_breaks = [
        {
            **record.as_dict(),
            "view_id": view_id,
            "left_point": restore(record.metadata.get("left_point"), view_id),
            "right_point": restore(record.metadata.get("right_point"), view_id),
            "segment_ids": sorted(
                {
                    segment_id
                    for polyline_id in (record.left_polyline_id, record.right_polyline_id)
                    for segment_id in polyline_to_segments.get(polyline_id, [])
                }
            ),
        }
        for record, view_id in vectorizer_breaks
    ]
    lineage_repairs: list[RepairRecord] = []
    for repair, view_id, score in virtual_repairs:
        affected_segment_ids = sorted(
            {
                segment_id
                for segment_id, polyline_id in segment_to_polyline.items()
                if repair.repair_id in kept_by_id[polyline_id].virtual_repair_ids
            }
        )
        lineage_repairs.append(
            RepairRecord(
                repair_id=repair.repair_id,
                rule_id="mask_closing_virtual_gap",
                object_ids=affected_segment_ids,
                before={
                    "left_point": restore(repair.left_point, view_id),
                    "right_point": restore(repair.right_point, view_id),
                    "bridge_gap_px": float(repair.bridge_gap_px),
                },
                after={
                    "bridge_point": restore(repair.bridge_point, view_id),
                    "added_pixels": int(repair.added_pixels),
                    "closing_radius": int(repair.closing_radius),
                },
                evidence_ids=[
                    f"wire-virtual-repair:{segment_id}:{repair.repair_id}"
                    for segment_id in affected_segment_ids
                ],
                confidence=score,
            )
        )
    lineage = {
        "repairs": lineage_repairs,
        "breaks": lineage_breaks,
        "segment_to_polyline": segment_to_polyline,
    }

    junctions: list[Junction] = []
    kept_polyline_ids = set(polyline_to_segments)
    for candidate, view_id, evidence, score in junction_candidates:
        view = views[view_id]
        restored = view.to_original(Point(x=candidate.point[0], y=candidate.point[1]))
        member_ids = [
            member for member in candidate.polyline_ids if member in kept_polyline_ids
        ]
        segment_ids = sorted(
            {
                segment_id
                for member in member_ids
                for segment_id in polyline_to_segments.get(member, [])
                if _segment_touches_point(
                    next(segment for segment in segments if segment.internal_id == segment_id),
                    restored,
                    tolerance=max(1.5, config.wire_tile_dedup_tolerance_px),
                )
            }
        )
        minimum = {"t_junction": 3, "connected_cross": 4}.get(candidate.kind, 1)
        if len(segment_ids) < minimum:
            continue
        # Four-way raster intersections do not by themselves prove electrical
        # connectivity (a bridge/no-dot crossing may look identical).  Keep the
        # geometry but require the learned junction provider to decide.
        kind = "ambiguous" if candidate.kind == "connected_cross" else candidate.kind
        junction = Junction(
            internal_id=candidate.internal_id,
            point=restored,
            kind=kind,
            segment_ids=segment_ids,
            score=score,
            evidence=[
                *evidence,
                Evidence(
                    evidence_id=f"wire-vectorizer:{candidate.internal_id}",
                    source="geometry",
                    view_id=view_id,
                    score=score,
                    metadata={"vectorizer": "skeleton-pixel-graph-rdp-v1"},
                ),
            ],
        )
        bind_point_evidence(
            junction.evidence,
            view=view,
            local_point=Point(x=candidate.point[0], y=candidate.point[1]),
            original_point=junction.point,
            object_kind="junction",
        )
        junctions.append(junction)
    return segments, _merge_junctions([], junctions, tolerance=config.wire_tile_dedup_tolerance_px), lineage


def _segment_touches_point(segment: WireSegment, point: Point, *, tolerance: float) -> bool:
    return min(
        ((segment.start.x - point.x) ** 2 + (segment.start.y - point.y) ** 2) ** 0.5,
        ((segment.end.x - point.x) ** 2 + (segment.end.y - point.y) ** 2) ** 0.5,
    ) <= tolerance


def _merge_junctions(
    vectorized: list[Junction],
    model: list[Junction],
    *,
    tolerance: float,
) -> list[Junction]:
    output: list[Junction] = []
    # Classifier results always enter first and therefore own the semantic kind;
    # skeleton candidates contribute branch IDs/evidence at the same point.
    ranked = [
        *sorted(model, key=lambda item: (-item.score, item.point.y, item.point.x, item.internal_id)),
        *sorted(vectorized, key=lambda item: (-item.score, item.point.y, item.point.x, item.internal_id)),
    ]
    for candidate in ranked:
        duplicate = next(
            (
                item
                for item in output
                if ((item.point.x - candidate.point.x) ** 2 + (item.point.y - candidate.point.y) ** 2) ** 0.5
                <= tolerance
            ),
            None,
        )
        if duplicate is None:
            output.append(candidate)
        else:
            duplicate.segment_ids = sorted(set(duplicate.segment_ids) | set(candidate.segment_ids))
            duplicate.evidence.extend(candidate.evidence)
    return sorted(output, key=_object_sort_key)
