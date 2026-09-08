from __future__ import annotations

import copy


def valid_inference_contract(task: str) -> dict[str, object]:
    """Return one complete task contract for model-lock readiness fixtures."""

    contracts: dict[str, dict[str, object]] = {
        "component": {
            "device": "auto",
            "score_threshold": 0.0,
            "max_detections": 300,
        },
        "junction": {
            "device": "auto",
            "input_size": 64,
            "patch_size_original_px": 64.0,
            "score_threshold": 0.55,
            "candidate_merge_tolerance_px": 2.0,
            "segment_touch_tolerance_px": 2.5,
            "maximum_candidates": 16384,
            "batch_size": 128,
        },
        "ocr": {
            "device": "auto",
            "detector_threshold": 0.5,
            "detector_min_area": 4,
            "recognition_threshold": 0.0,
            "detector_batch_size": 4,
            "recognition_batch_size": 32,
            "max_detections_per_view": 512,
            "max_recognition_width": 4096,
        },
        "pin": {
            "device": "auto",
            "input_size": [384, 384],
            "output_stride": 4,
            "score_threshold": 0.15,
            "nms_kernel": 3,
            "nms_radius_heatmap": 2.0,
            "max_detections_per_component": 128,
            "base_length_pixels": 8.0,
            "box_margin_px": 0.0,
            "box_prior_mode": "filled_rect_v1",
        },
        "wire": {
            "device": "auto",
            "threshold": 0.5,
            "batch_size": 4,
            "output_kind": "logits",
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        },
    }
    return copy.deepcopy(contracts[task])


__all__ = ["valid_inference_contract"]
