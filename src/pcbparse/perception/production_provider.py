from __future__ import annotations

import importlib
import json
import math
from collections.abc import Mapping
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any

from pcbparse.perception.composite_provider import CompositePerceptionProvider, CompositeProviderConfig
from pcbparse.service.readiness import (
    REQUIRED_MODEL_TASKS,
    ReadinessGateError,
    readiness_report,
)


DETERMINISTIC_JUNCTION_MODEL_TASKS = frozenset(REQUIRED_MODEL_TASKS - {"junction"})


class ProductionProviderError(RuntimeError):
    """A verified model lock cannot be turned into executable providers."""


class ProductionPerceptionProvider(CompositePerceptionProvider):
    """Fail-closed composition of the configured executable model adapters."""

    provider_kind = "production"
    execution_mode = "production"
    version = "production-composite-v2"


def composite_provider_config_from_pipeline(
    pipeline_config: Mapping[str, Any] | None,
) -> CompositeProviderConfig:
    """Resolve production tiling from a loaded pipeline configuration.

    Older callers that do not supply a pipeline configuration keep the stable
    provider defaults.  When an ``image`` section is present, its tiling values
    are type- and range-checked instead of being silently ignored.
    """

    defaults = CompositeProviderConfig()
    if pipeline_config is None:
        return defaults
    if not isinstance(pipeline_config, Mapping):
        raise ValueError("pipeline config must be an object")
    raw_image = pipeline_config.get("image")
    if raw_image is None:
        raw_image = {}
    if not isinstance(raw_image, Mapping):
        raise ValueError("pipeline image section must be an object")
    raw_graph = pipeline_config.get("graph")
    if raw_graph is None:
        raw_graph = {}
    if not isinstance(raw_graph, Mapping):
        raise ValueError("pipeline graph section must be an object")
    raw_pin = pipeline_config.get("pin")
    if raw_pin is None:
        raw_pin = {}
    raw_wire = pipeline_config.get('wire')
    if raw_wire is None:
        raw_wire = {}
    if not isinstance(raw_wire, Mapping):
        raise ValueError('pipeline wire section must be an object')
    if not isinstance(raw_pin, Mapping):
        raise ValueError("pipeline pin section must be an object")
    tile_size = _strict_int(raw_image, "tile_size", defaults.tile_size)
    tile_overlap = _strict_int(raw_image, "tile_overlap", defaults.tile_overlap)
    component_tile_size = _optional_strict_int(raw_image, "component_tile_size")
    component_tile_overlap = _optional_strict_int(raw_image, "component_tile_overlap")
    component_inference_batch_size = _strict_positive_int(
        raw_image,
        "component_inference_batch_size",
        defaults.component_inference_batch_size,
    )
    wire_tile_size = _optional_strict_int(raw_image, "wire_tile_size")
    wire_tile_overlap = _optional_strict_int(raw_image, "wire_tile_overlap")
    component_candidate_min_score = _strict_unit_interval_number(
        raw_image,
        "component_candidate_min_score",
        defaults.component_candidate_min_score,
    )
    component_nms_iou = _strict_unit_interval_number(
        raw_image,
        "component_nms_iou",
        defaults.component_nms_iou,
    )
    component_center_preferred_nms_enabled = _strict_image_bool(
        raw_image,
        "component_center_preferred_nms_enabled",
        defaults.component_center_preferred_nms_enabled,
    )
    component_center_preference_score_bonus = _strict_unit_interval_number(
        raw_image,
        "component_center_preference_score_bonus",
        defaults.component_center_preference_score_bonus,
    )
    component_internal_border_margin_px = _strict_nonnegative_number(
        raw_image,
        "component_internal_border_margin_px",
        defaults.component_internal_border_margin_px,
    )
    component_internal_border_confirmation_enabled = _strict_image_bool(
        raw_image,
        "component_internal_border_confirmation_enabled",
        defaults.component_internal_border_confirmation_enabled,
    )
    component_internal_border_confirmation_iou = _strict_unit_interval_number(
        raw_image,
        "component_internal_border_confirmation_iou",
        defaults.component_internal_border_confirmation_iou,
    )
    component_internal_border_confirmation_metric = _strict_choice(
        raw_image,
        "component_internal_border_confirmation_metric",
        defaults.component_internal_border_confirmation_metric,
        {"iou", "intersection_over_smaller"},
    )
    component_internal_border_confirmation_categories = _strict_string_tuple(
        raw_image,
        "component_internal_border_confirmation_categories",
        defaults.component_internal_border_confirmation_categories,
    )
    component_internal_border_refinement_enabled = _strict_image_bool(
        raw_image,
        "component_internal_border_refinement_enabled",
        defaults.component_internal_border_refinement_enabled,
    )
    component_internal_border_refinement_iou = _strict_unit_interval_number(
        raw_image,
        "component_internal_border_refinement_iou",
        defaults.component_internal_border_refinement_iou,
    )
    component_internal_border_refinement_categories = _strict_string_tuple(
        raw_image,
        "component_internal_border_refinement_categories",
        defaults.component_internal_border_refinement_categories,
    )
    component_internal_border_refinement_max_candidates = _strict_positive_int(
        raw_image,
        "component_internal_border_refinement_max_candidates",
        defaults.component_internal_border_refinement_max_candidates,
    )
    component_internal_border_refinement_batch_size = _strict_positive_int(
        raw_image,
        "component_internal_border_refinement_batch_size",
        defaults.component_internal_border_refinement_batch_size,
    )
    component_targeted_recovery_enabled = _strict_image_bool(
        raw_image,
        "component_targeted_recovery_enabled",
        defaults.component_targeted_recovery_enabled,
    )
    component_targeted_recovery_categories = _strict_string_tuple(
        raw_image,
        "component_targeted_recovery_categories",
        defaults.component_targeted_recovery_categories,
    )
    component_targeted_recovery_tile_size = _strict_positive_int(
        raw_image,
        "component_targeted_recovery_tile_size",
        defaults.component_targeted_recovery_tile_size,
    )
    component_targeted_recovery_tile_overlap = _strict_nonnegative_int(
        raw_image,
        "component_targeted_recovery_tile_overlap",
        defaults.component_targeted_recovery_tile_overlap,
    )
    component_targeted_recovery_min_dimension_px = _strict_positive_int(
        raw_image,
        "component_targeted_recovery_min_dimension_px",
        defaults.component_targeted_recovery_min_dimension_px,
    )
    component_targeted_recovery_batch_size = _strict_positive_int(
        raw_image,
        "component_targeted_recovery_batch_size",
        defaults.component_targeted_recovery_batch_size,
    )
    component_max_candidates = _optional_strict_positive_int(
        raw_image,
        "component_max_candidates",
    )
    component_adaptive_fallback_enabled = _strict_image_bool(
        raw_image,
        "component_adaptive_fallback_enabled",
        defaults.component_adaptive_fallback_enabled,
    )
    component_fallback_tile_size = _strict_int(
        raw_image, "component_fallback_tile_size", defaults.component_fallback_tile_size
    )
    component_fallback_tile_overlap = _strict_int(
        raw_image, "component_fallback_tile_overlap", defaults.component_fallback_tile_overlap
    )
    component_fallback_trigger_max_count = _strict_int(
        raw_image, "component_fallback_trigger_max_count", defaults.component_fallback_trigger_max_count
    )
    component_fallback_trigger_max_box_fraction = _strict_unit_interval_number(
        raw_image,
        "component_fallback_trigger_max_box_fraction",
        defaults.component_fallback_trigger_max_box_fraction,
    )
    component_fallback_trigger_max_bbox_area_fraction = _strict_unit_interval_number(
        raw_image,
        "component_fallback_trigger_max_bbox_area_fraction",
        defaults.component_fallback_trigger_max_bbox_area_fraction,
    )
    component_roi_padding_px = _strict_nonnegative_number(
        raw_image,
        "component_roi_padding_px",
        defaults.component_roi_padding_px,
    )
    ocr_component_roi_padding_px = _optional_strict_nonnegative_number(
        raw_image,
        "ocr_component_roi_padding_px",
    )
    ocr_merge_conflicting_text = _strict_image_bool(
        raw_image,
        "ocr_merge_conflicting_text",
        defaults.ocr_merge_conflicting_text,
    )
    ocr_conflicting_text_iou = _strict_unit_interval_number(
        raw_image,
        "ocr_conflicting_text_iou",
        defaults.ocr_conflicting_text_iou,
    )
    pin_component_roi_padding_px = _optional_strict_nonnegative_number(
        raw_image,
        "pin_component_roi_padding_px",
    )
    wire_minimum_run_px = _strict_positive_int(raw_graph, "wire_minimum_run_px", defaults.wire_minimum_run_px)
    wire_rdp_epsilon_px = _strict_nonnegative_number(
        raw_graph,
        "rdp_epsilon_px",
        defaults.wire_rdp_epsilon_px,
    )
    wire_gap_closing_radius_px = _strict_nonnegative_int(
        raw_graph,
        "wire_gap_closing_radius_px",
        defaults.wire_gap_closing_radius_px,
    )
    wire_maximum_hole_area_px = _strict_nonnegative_int(
        raw_graph,
        "wire_maximum_hole_area_px",
        defaults.wire_maximum_hole_area_px,
    )
    wire_maximum_component_pixels = _strict_nonnegative_int(
        raw_graph,
        "wire_maximum_component_pixels",
        defaults.wire_maximum_component_pixels,
    )
    learned_junctions_enabled = _strict_bool(
        raw_graph,
        "learned_junctions_enabled",
        defaults.learned_junctions_enabled,
    )
    pin_inference_overrides = _parse_pin_inference_overrides(raw_pin)
    wire_inference_overrides = _parse_wire_inference_overrides(raw_wire)
    raw_ocr = pipeline_config.get("ocr")
    if raw_ocr is None:
        raw_ocr = {}
    if not isinstance(raw_ocr, Mapping):
        raise ValueError("pipeline ocr section must be an object")
    ocr_inference_overrides = _parse_ocr_inference_overrides(raw_ocr)
    # OCR batch sizes — configurable to control GPU memory and throughput.
    ocr_global_batch_size = _optional_strict_positive_int(
        raw_ocr, "global_batch_size",
    ) or defaults.ocr_global_batch_size
    ocr_component_batch_size = _optional_strict_positive_int(
        raw_ocr, "component_batch_size",
    ) or defaults.ocr_component_batch_size
    ocr_strip_batch_size = _optional_strict_positive_int(
        raw_ocr, "strip_batch_size",
    ) or defaults.ocr_strip_batch_size
    _validate_tile_pair(
        "component_fallback_tile",
        component_fallback_tile_size,
        component_fallback_tile_overlap,
    )
    _validate_tile_pair(
        "component_targeted_recovery_tile",
        component_targeted_recovery_tile_size,
        component_targeted_recovery_tile_overlap,
    )
    _validate_tile_pair("tile", tile_size, tile_overlap)
    _validate_tile_pair(
        "component_tile",
        tile_size if component_tile_size is None else component_tile_size,
        tile_overlap if component_tile_overlap is None else component_tile_overlap,
    )
    _validate_tile_pair(
        "wire_tile",
        tile_size if wire_tile_size is None else wire_tile_size,
        tile_overlap if wire_tile_overlap is None else wire_tile_overlap,
    )
    return CompositeProviderConfig(
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        component_tile_size=component_tile_size,
        component_tile_overlap=component_tile_overlap,
        component_inference_batch_size=component_inference_batch_size,
        component_nms_iou=component_nms_iou,
        component_center_preferred_nms_enabled=component_center_preferred_nms_enabled,
        component_center_preference_score_bonus=component_center_preference_score_bonus,
        component_internal_border_margin_px=component_internal_border_margin_px,
        component_internal_border_confirmation_enabled=(
            component_internal_border_confirmation_enabled
        ),
        component_internal_border_confirmation_iou=(
            component_internal_border_confirmation_iou
        ),
        component_internal_border_confirmation_metric=(
            component_internal_border_confirmation_metric
        ),
        component_internal_border_confirmation_categories=(
            component_internal_border_confirmation_categories
        ),
        component_internal_border_refinement_enabled=(
            component_internal_border_refinement_enabled
        ),
        component_internal_border_refinement_iou=(
            component_internal_border_refinement_iou
        ),
        component_internal_border_refinement_categories=(
            component_internal_border_refinement_categories
        ),
        component_internal_border_refinement_max_candidates=(
            component_internal_border_refinement_max_candidates
        ),
        component_internal_border_refinement_batch_size=(
            component_internal_border_refinement_batch_size
        ),
        component_targeted_recovery_enabled=component_targeted_recovery_enabled,
        component_targeted_recovery_categories=(
            component_targeted_recovery_categories
        ),
        component_targeted_recovery_tile_size=(
            component_targeted_recovery_tile_size
        ),
        component_targeted_recovery_tile_overlap=(
            component_targeted_recovery_tile_overlap
        ),
        component_targeted_recovery_min_dimension_px=(
            component_targeted_recovery_min_dimension_px
        ),
        component_targeted_recovery_batch_size=(
            component_targeted_recovery_batch_size
        ),
        wire_tile_size=wire_tile_size,
        wire_tile_overlap=wire_tile_overlap,
        component_candidate_min_score=component_candidate_min_score,
        component_max_candidates=component_max_candidates,
        component_adaptive_fallback_enabled=component_adaptive_fallback_enabled,
        component_fallback_tile_size=component_fallback_tile_size,
        component_fallback_tile_overlap=component_fallback_tile_overlap,
        component_fallback_trigger_max_count=component_fallback_trigger_max_count,
        component_fallback_trigger_max_box_fraction=component_fallback_trigger_max_box_fraction,
        component_fallback_trigger_max_bbox_area_fraction=component_fallback_trigger_max_bbox_area_fraction,
        component_roi_padding_px=component_roi_padding_px,
        ocr_component_roi_padding_px=ocr_component_roi_padding_px,
        ocr_merge_conflicting_text=ocr_merge_conflicting_text,
        ocr_conflicting_text_iou=ocr_conflicting_text_iou,
        ocr_global_batch_size=ocr_global_batch_size,
        ocr_component_batch_size=ocr_component_batch_size,
        ocr_strip_batch_size=ocr_strip_batch_size,
        pin_component_roi_padding_px=pin_component_roi_padding_px,
        wire_minimum_run_px=wire_minimum_run_px,
        wire_rdp_epsilon_px=wire_rdp_epsilon_px,
        wire_gap_closing_radius_px=wire_gap_closing_radius_px,
        wire_maximum_hole_area_px=wire_maximum_hole_area_px,
        wire_maximum_component_pixels=wire_maximum_component_pixels,
        learned_junctions_enabled=learned_junctions_enabled,
        pin_inference_overrides=pin_inference_overrides,
        wire_inference_overrides=wire_inference_overrides,
        ocr_inference_overrides=ocr_inference_overrides,
    )


def resolved_perception_config(provider: Any) -> dict[str, Any] | None:
    """Return a provider's actual perception settings when it exposes them."""

    config = getattr(provider, "config", None)
    if isinstance(config, CompositeProviderConfig):
        return config.to_dict()
    return None


def required_model_tasks_for_config(
    config: CompositeProviderConfig | None,
) -> frozenset[str]:
    """Return the exact learned-model task set for one production configuration.

    Junction topology is always produced.  With the learned junction classifier
    disabled, it comes from the deterministic wire vectorizer and no Junction
    artifact is permitted in the lock or loaded into the process.
    """

    resolved = config or CompositeProviderConfig()
    return (
        REQUIRED_MODEL_TASKS
        if resolved.learned_junctions_enabled
        else DETERMINISTIC_JUNCTION_MODEL_TASKS
    )


def load_production_provider(
    models_lock: str | Path,
    *,
    config: CompositeProviderConfig | None = None,
) -> ProductionPerceptionProvider:
    """Load adapters only after every declared model byte has been verified.

    Each model entry names an ``adapter`` as ``package.module:factory``.  The
    factory receives ``(resolved_model_path, entry_copy)`` and returns the task
    protocol implementation.  This keeps framework-specific loading outside the
    core while giving CLI/Docker one real production entry point.
    """

    lock_path = Path(models_lock).expanduser().resolve()
    resolved_config = config or CompositeProviderConfig()
    required_tasks = required_model_tasks_for_config(resolved_config)
    report = readiness_report(
        provider_name="production",
        models_lock=lock_path,
        required_tasks=required_tasks,
        require_lineage=True,
    )
    if not report["ready"]:
        raise ReadinessGateError(report)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    entries = {str(entry["task"]): entry for entry in lock["models"]}
    loaded: dict[str, Any] = {}
    load_order = tuple(
        task
        for task in ("component", "ocr", "pin", "wire", "junction")
        if task in required_tasks
    )
    for task in load_order:
        entry = entries[task]
        adapter = entry.get("adapter")
        if not isinstance(adapter, str) or ":" not in adapter:
            raise ProductionProviderError(
                f"verified task {task!r} must declare adapter='package.module:factory'"
            )
        module_name, attribute_name = adapter.rsplit(":", 1)
        try:
            factory = getattr(importlib.import_module(module_name), attribute_name)
        except (ImportError, AttributeError) as exc:
            raise ProductionProviderError(
                f"cannot import adapter {adapter!r} for task {task!r}: {type(exc).__name__}"
            ) from exc
        if not callable(factory):
            raise ProductionProviderError(f"adapter {adapter!r} for task {task!r} is not callable")
        model_path = _resolve_model_path(lock_path, str(entry["path"]))
        try:
            provider = factory(model_path, dict(entry))
            if task == "pin":
                provider = _apply_pin_inference_overrides(
                    provider,
                    resolved_config.pin_inference_overrides,
                )
            if task == 'wire':
                provider = _apply_wire_inference_overrides(
                    provider,
                    resolved_config.wire_inference_overrides,
                )
            if task == "ocr":
                provider = _apply_ocr_inference_overrides(
                    provider,
                    resolved_config.ocr_inference_overrides,
                )
        except Exception as exc:
            raise ProductionProviderError(
                f"adapter {adapter!r} failed to load task {task!r}: {type(exc).__name__}: {exc}"
            ) from exc
        _verify_provider_contract(provider, task)
        loaded[task] = provider
    return ProductionPerceptionProvider(
        component_provider=loaded["component"],
        ocr_provider=loaded["ocr"],
        pin_provider=loaded["pin"],
        wire_provider=loaded["wire"],
        junction_provider=loaded.get("junction"),
        config=resolved_config,
    )


def _apply_pin_inference_overrides(provider: Any, overrides: Mapping[str, Any]) -> Any:
    if not overrides:
        return provider
    current = getattr(provider, "config", None)
    if current is None or not is_dataclass(current):
        raise ProductionProviderError(
            "pipeline pin overrides require a dataclass-backed Pin adapter config"
        )
    try:
        provider.config = replace(current, **dict(overrides))
    except (TypeError, ValueError) as exc:
        raise ProductionProviderError(
            f"invalid pipeline Pin inference override: {exc}"
        ) from exc
    return provider


def _apply_wire_inference_overrides(provider: Any, overrides: Mapping[str, Any]) -> Any:
    if not overrides:
        return provider
    if not hasattr(provider, 'threshold'):
        raise ProductionProviderError(
            'pipeline wire overrides require a threshold-backed Wire adapter'
        )
    try:
        provider.threshold = float(overrides['threshold'])
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ProductionProviderError(
            f'invalid pipeline Wire inference override: {exc}'
        ) from exc
    return provider


def _parse_pin_inference_overrides(section: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
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
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"pipeline pin contains unsupported key(s): {unknown}")
    result: dict[str, Any] = {}
    if "score_threshold" in section:
        result["score_threshold"] = _strict_unit_interval_number(
            section, "score_threshold", 0.0
        )
    if "max_detections_per_component" in section:
        value = section["max_detections_per_component"]
        if type(value) is not int or value <= 0:
            raise ValueError(
                "pipeline pin.max_detections_per_component must be a positive integer"
            )
        result["max_detections_per_component"] = value
    if "pin_budget_mode" in section:
        value = section["pin_budget_mode"]
        if not isinstance(value, str) or value not in {"fixed", "edge_length_v1"}:
            raise ValueError(
                "pipeline pin.pin_budget_mode must be fixed or edge_length_v1"
            )
        result["pin_budget_mode"] = value
    if "pin_budget_floor" in section:
        value = section["pin_budget_floor"]
        if type(value) is not int or value <= 0:
            raise ValueError(
                "pipeline pin.pin_budget_floor must be a positive integer"
            )
        result["pin_budget_floor"] = value
    if "pin_budget_pitch_px" in section:
        result["pin_budget_pitch_px"] = _strict_nonnegative_number(
            section, "pin_budget_pitch_px", 0.0
        )
        if float(result["pin_budget_pitch_px"]) <= 0:
            raise ValueError(
                "pipeline pin.pin_budget_pitch_px must be strictly positive"
            )
    if "box_margin_px" in section:
        result["box_margin_px"] = _strict_nonnegative_number(
            section, "box_margin_px", 0.0
        )
    if "box_prior_mode" in section:
        value = section["box_prior_mode"]
        if not isinstance(value, str) or value not in {
            "filled_rect_v1",
            "boundary_band_v2",
        }:
            raise ValueError(
                "pipeline pin.box_prior_mode must be filled_rect_v1 or boundary_band_v2"
            )
        result["box_prior_mode"] = value
    if "nms_kernel" in section:
        value = section["nms_kernel"]
        if type(value) is not int or value <= 0 or value % 2 != 1:
            raise ValueError("pipeline pin.nms_kernel must be a positive odd integer")
        result["nms_kernel"] = value
    if "nms_radius_heatmap" in section:
        result["nms_radius_heatmap"] = _strict_nonnegative_number(
            section, "nms_radius_heatmap", 0.0
        )
    if "base_length_pixels" in section:
        value = _strict_nonnegative_number(section, "base_length_pixels", 0.0)
        if value <= 0.0:
            raise ValueError("pipeline pin.base_length_pixels must be positive")
        result["base_length_pixels"] = value
    return result


def _parse_wire_inference_overrides(section: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {'threshold'}
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f'pipeline wire contains unsupported key(s): {unknown}')
    if 'threshold' not in section:
        return {}
    return {'threshold': _strict_unit_interval_number(section, 'threshold', 0.5)}


def _parse_ocr_inference_overrides(section: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "detector_threshold",
        "detector_min_area",
        "recognition_candidate_count",
        # Batch sizes are consumed by CompositeProviderConfig below rather
        # than by the OCR model adapter.  They still belong to the strict
        # pipeline OCR schema and must not be rejected as unknown keys.
        "global_batch_size",
        "component_batch_size",
        "strip_batch_size",
        "vertical_rotation_enabled",
        "recognizer_backend",
        "reread_enabled",
        "reread_selection",
    }
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"pipeline ocr contains unsupported key(s): {unknown}")
    result: dict[str, Any] = {}
    if "detector_threshold" in section:
        result["detector_threshold"] = _strict_unit_interval_number(
            section, "detector_threshold", 0.5
        )
    if "detector_min_area" in section:
        value = section["detector_min_area"]
        if type(value) is not int or value <= 0:
            raise ValueError(
                "pipeline ocr.detector_min_area must be a positive integer"
            )
        result["detector_min_area"] = value
    if "recognition_candidate_count" in section:
        value = section["recognition_candidate_count"]
        if type(value) is not int or value < 0:
            raise ValueError(
                "pipeline ocr.recognition_candidate_count must be a non-negative integer"
            )
        result["recognition_candidate_count"] = value
    if "vertical_rotation_enabled" in section:
        value = section["vertical_rotation_enabled"]
        if type(value) is not bool:
            raise ValueError(
                "pipeline ocr.vertical_rotation_enabled must be a boolean"
            )
        result["vertical_rotation_enabled"] = value
    if "recognizer_backend" in section:
        value = section["recognizer_backend"]
        if not isinstance(value, (str, Mapping)):
            raise ValueError(
                "pipeline ocr.recognizer_backend must be a string or object"
            )
        result["recognizer_backend"] = value
    if "reread_enabled" in section:
        value = section["reread_enabled"]
        if type(value) is not bool:
            raise ValueError("pipeline ocr.reread_enabled must be a boolean")
        result["reread_enabled"] = value
    if "reread_selection" in section:
        value = section["reread_selection"]
        if not isinstance(value, str):
            raise ValueError("pipeline ocr.reread_selection must be a string")
        result["reread_selection"] = value
    return result


def _apply_ocr_inference_overrides(provider: Any, overrides: Mapping[str, Any]) -> Any:
    if not overrides:
        return provider
    if (
        "detector_threshold" in overrides
        and not hasattr(provider, "detector_threshold")
    ):
        raise ProductionProviderError(
            "pipeline OCR overrides require a detector_threshold-backed OCR adapter"
        )
    if (
        "detector_min_area" in overrides
        and not hasattr(provider, "detector_min_area")
    ):
        raise ProductionProviderError(
            "pipeline OCR overrides require a detector_min_area-backed OCR adapter"
        )
    if (
        "recognition_candidate_count" in overrides
        and not hasattr(provider, "recognition_candidate_count")
    ):
        raise ProductionProviderError(
            "pipeline OCR candidate overrides require a candidate-capable OCR adapter"
        )
    if (
        "vertical_rotation_enabled" in overrides
        and not hasattr(provider, "vertical_rotation_enabled")
    ):
        raise ProductionProviderError(
            "pipeline OCR vertical rotation requires a rotation-capable OCR adapter"
        )
    if "recognizer_backend" in overrides and not hasattr(
        provider, "_recognizer_backend"
    ):
        raise ProductionProviderError(
            "pipeline OCR recognizer_backend requires a backend-capable OCR adapter"
        )
    if "reread_enabled" in overrides and not hasattr(provider, "reread_enabled"):
        raise ProductionProviderError(
            "pipeline OCR reread_enabled requires a reread-capable OCR adapter"
        )
    if "reread_selection" in overrides and not hasattr(provider, "reread_selection"):
        raise ProductionProviderError(
            "pipeline OCR reread_selection requires a reread-capable OCR adapter"
        )
    try:
        if "detector_threshold" in overrides:
            provider.detector_threshold = float(overrides["detector_threshold"])
        if "detector_min_area" in overrides:
            provider.detector_min_area = int(overrides["detector_min_area"])
        if "recognition_candidate_count" in overrides:
            provider.recognition_candidate_count = int(
                overrides["recognition_candidate_count"]
            )
        if "vertical_rotation_enabled" in overrides:
            provider.vertical_rotation_enabled = bool(
                overrides["vertical_rotation_enabled"]
            )
        if "recognizer_backend" in overrides:
            from pcbparse.perception.torch_ocr_adapter import _load_recognizer_backend

            provider._recognizer_backend = _load_recognizer_backend(
                overrides["recognizer_backend"]
            )
            base_version = provider.version.split("+")[0]
            provider.version = (
                f"{base_version}+{provider._recognizer_backend.version}"
            )
        if "reread_enabled" in overrides:
            provider.reread_enabled = overrides["reread_enabled"]
        if "reread_selection" in overrides:
            provider.reread_selection = str(overrides["reread_selection"])
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ProductionProviderError(
            f"invalid pipeline OCR inference override: {exc}"
        ) from exc
    return provider


def _resolve_model_path(lock_path: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return (path if path.is_absolute() else lock_path.parent / path).resolve()


def _strict_int(section: Mapping[str, Any], key: str, default: int) -> int:
    value = section[key] if key in section else default
    if type(value) is not int:
        raise ValueError(f"pipeline image.{key} must be an integer")
    return value


def _strict_positive_int(section: Mapping[str, Any], key: str, default: int) -> int:
    value = section[key] if key in section else default
    if type(value) is not int or value <= 0:
        raise ValueError(f"pipeline graph.{key} must be positive")
    return value

def _strict_image_bool(section: Mapping[str, Any], key: str, default: bool) -> bool:
    value = section[key] if key in section else default
    if type(value) is not bool:
        raise ValueError(f"pipeline image.{key} must be a boolean")
    return value

def _strict_bool(section: Mapping[str, Any], key: str, default: bool) -> bool:
    value = section[key] if key in section else default
    if type(value) is not bool:
        raise ValueError(f"pipeline graph.{key} must be a boolean")
    return value


def _strict_nonnegative_int(
    section: Mapping[str, Any], key: str, default: int
) -> int:
    value = section[key] if key in section else default
    if type(value) is not int or value < 0:
        raise ValueError(f"pipeline graph.{key} must be a non-negative integer")
    return value


def _optional_strict_int(section: Mapping[str, Any], key: str) -> int | None:
    if key not in section:
        return None
    value = section[key]
    if type(value) is not int:
        raise ValueError(f"pipeline image.{key} must be an integer")
    return value


def _optional_strict_positive_int(section: Mapping[str, Any], key: str) -> int | None:
    value = _optional_strict_int(section, key)
    if value is not None and value <= 0:
        raise ValueError(f"pipeline image.{key} must be positive")
    return value


def _strict_unit_interval_number(
    section: Mapping[str, Any], key: str, default: float
) -> float:
    resolved = _strict_nonnegative_number(section, key, default)
    if resolved > 1.0:
        raise ValueError(f"pipeline image.{key} must be in [0,1]")
    return resolved


def _validate_tile_pair(label: str, tile_size: int, tile_overlap: int) -> None:
    if tile_size <= 0:
        raise ValueError(f"pipeline image.{label}_size must be positive")
    if tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError(
            f"pipeline image.{label}_overlap must satisfy 0 <= overlap < {label}_size"
        )


def _strict_nonnegative_number(
    section: Mapping[str, Any], key: str, default: float
) -> float:
    value = section[key] if key in section else default
    if type(value) not in {int, float}:
        raise ValueError(f"pipeline image.{key} must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"pipeline image.{key} must be a finite number")
    if resolved < 0:
        raise ValueError(f"pipeline image.{key} must be non-negative")
    return resolved


def _strict_string_tuple(
    section: Mapping[str, Any],
    key: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    value = section[key] if key in section else list(default)
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(
            f"pipeline image.{key} must contain non-empty strings"
        )
    return tuple(dict.fromkeys(item.strip() for item in value))


def _strict_choice(
    section: Mapping[str, Any],
    key: str,
    default: str,
    choices: set[str],
) -> str:
    value = section[key] if key in section else default
    if not isinstance(value, str) or value not in choices:
        raise ValueError(
            f"pipeline image.{key} must be one of {sorted(choices)}"
        )
    return value


def _optional_strict_nonnegative_number(
    section: Mapping[str, Any], key: str
) -> float | None:
    if key not in section:
        return None
    return _strict_nonnegative_number(section, key, 0.0)


def _verify_provider_contract(provider: Any, task: str) -> None:
    required_method = {
        "component": "predict",
        "ocr": "recognize",
        "pin": "predict",
        "wire": "segment",
        "junction": "classify",
    }[task]
    if not isinstance(getattr(provider, "version", None), str) or not provider.version:
        raise ProductionProviderError(f"task {task!r} provider has no non-empty version")
    if not callable(getattr(provider, required_method, None)):
        raise ProductionProviderError(
            f"task {task!r} provider does not implement {required_method}()"
        )


__all__ = [
    "DETERMINISTIC_JUNCTION_MODEL_TASKS",
    "ProductionPerceptionProvider",
    "ProductionProviderError",
    "composite_provider_config_from_pipeline",
    "load_production_provider",
    "required_model_tasks_for_config",
    "resolved_perception_config",
]
