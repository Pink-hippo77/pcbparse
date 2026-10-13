from __future__ import annotations

"""Production OCRProvider adapter for a unified PyTorch OCR bundle.

The detector consumes fixed-size RGB batches.  Its connected components are
mapped back into each ``ImageView``'s local coordinates, cropped from the
original pixels, and passed through the dynamic-width CRNN.  This adapter does
not infer semantic categories, details, ownership, or a resolved spelling.
"""

import hashlib
import importlib
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from pcbparse.io import ImageView
from pcbparse.ir.models import BBox, Evidence, TextToken
from pcbparse.perception.protocols import ProviderError
from pcbparse.training.torch_ocr_model import (
    DetectionBox,
    LoadedOCRBundle,
    ctc_beam_decode,
    extract_text_boxes,
    greedy_ctc_decode,
    load_ocr_bundle,
    recognition_output_lengths,
)


class TorchOCRAdapterError(ProviderError):
    """A frozen OCR bundle or inference result is unsafe to consume."""


@dataclass(slots=True)
class _CropCandidate:
    view_index: int
    detection_index: int
    view: ImageView
    box: tuple[float, float, float, float]
    detector_box: DetectionBox
    crop_box: tuple[int, int, int, int]
    tensor: Any | None = None
    width: int = 0
    crop_sha256: str = ""
    orientation_degrees: int = 0
    orientation_candidates_degrees: tuple[int, ...] = (0,)
    np_crop: Any | None = None
    variant_name: str = "base"
    pre_upscale: int = 1


# --- Multi-crop candidate re-read (opt80 r05) ---------------------------------
# Deterministic recognition-time augmentation: every detected text box is also
# read from a few fixed crop variants (small padding / min-size expansion /
# pre-upscale smoothing).  Exactly one candidate per detector box is kept, chosen
# by a fixed selection policy.  Disabled by default; when disabled the pipeline
# takes the original single-crop path unchanged.

_REREAD_SELECTION_POLICIES = (
    "consensus_grammar_original",
    "consensus_score",
    "score_max",
)

_DESIGNATOR_STRUCTURE_RE = __import__("re").compile(r"^[A-Z]{1,4}[0-9]+[A-Z]?$")


def _reread_variant_crops(
    box: tuple[float, float, float, float],
    view_width: int,
    view_height: int,
) -> list[tuple[str, tuple[int, int, int, int], int]]:
    """Fixed crop variants around a detected box: (name, crop_box, pre_upscale)."""

    x1, y1, x2, y2 = box
    variants: list[tuple[str, tuple[int, int, int, int], int]] = []

    def clip(vbox: tuple[float, float, float, float]) -> tuple[int, int, int, int] | None:
        ix1 = max(0, min(view_width - 1, int(math.floor(vbox[0]))))
        iy1 = max(0, min(view_height - 1, int(math.floor(vbox[1]))))
        ix2 = max(ix1 + 1, min(view_width, int(math.ceil(vbox[2]))))
        iy2 = max(iy1 + 1, min(view_height, int(math.ceil(vbox[3]))))
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        return (ix1, iy1, ix2, iy2)

    padded = clip((x1 - 2.0, y1 - 2.0, x2 + 2.0, y2 + 2.0))
    if padded is not None:
        variants.append(("pad2", padded, 1))
        variants.append(("pad2x3", padded, 3))
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_h = max(y2 - y1, 16.0) / 2.0
    half_w = max(x2 - x1, 8.0) / 2.0
    expanded = clip((center_x - half_w, center_y - half_h, center_x + half_w, center_y + half_h))
    if expanded is not None:
        variants.append(("min16", expanded, 1))
    return variants


def _reread_candidate_key(
    item: tuple[_CropCandidate, str, float, list[str], int, float],
) -> tuple[int, int]:
    candidate = item[0]
    return (candidate.view_index, candidate.detection_index)


def _reread_select(
    pool: list[tuple[_CropCandidate, str, float, list[str], int, float]],
    policy: str,
) -> tuple[_CropCandidate, str, float, list[str], int, float]:
    """Deterministic single-candidate selection across crop variants."""

    if policy == "score_max":
        return max(pool, key=lambda item: (item[2], item[0].variant_name))
    counts: dict[str, int] = {}
    for item in pool:
        counts[item[1]] = counts.get(item[1], 0) + 1
    top = max(counts.values())
    tied = [item for item in pool if counts[item[1]] == top]
    if policy == "consensus_score":
        return max(tied, key=lambda item: (item[2], item[0].variant_name))
    # consensus_grammar_original: majority first, then structurally valid text,
    # then the original detector-box read, then recognition score.
    return max(
        tied,
        key=lambda item: (
            1 if _text_has_structure(item[1]) else 0,
            1 if item[0].variant_name == "base" else 0,
            item[2],
            item[0].variant_name,
        ),
    )


def _select_reread_results(
    recognized: Sequence[tuple[_CropCandidate, str, float, list[str], int, float]],
    policy: str,
) -> tuple[
    list[tuple[_CropCandidate, str, float, list[str], int, float]],
    dict[tuple[int, int], list[dict[str, Any]]],
]:
    """Keep one candidate per detector box and record the considered variants."""

    groups: dict[tuple[int, int], list[tuple[_CropCandidate, str, float, list[str], int, float]]] = {}
    for item in recognized:
        groups.setdefault(_reread_candidate_key(item), []).append(item)
    selected: list[tuple[_CropCandidate, str, float, list[str], int, float]] = []
    meta: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for key in sorted(groups):
        pool = sorted(
            groups[key],
            key=lambda item: (item[0].variant_name, item[0].orientation_degrees),
        )
        chosen = _reread_select(pool, policy)
        selected.append(chosen)
        meta[key] = [
            {
                "variant": item[0].variant_name,
                "orientation_degrees": item[0].orientation_degrees,
                "pre_upscale": item[0].pre_upscale,
                "text": item[1],
                "recognition_score": item[2],
                "selected": item is chosen,
            }
            for item in pool
        ]
    return selected, meta


class TorchOCRAdapter:
    """Deterministic two-stage OCR provider for local ``ImageView`` coordinates."""

    def __init__(
        self,
        bundle: LoadedOCRBundle,
        *,
        device: str,
        detector_threshold: float,
        detector_min_area: int,
        recognition_threshold: float,
        detector_batch_size: int,
        recognition_batch_size: int,
        max_detections_per_view: int,
        max_recognition_width: int,
        recognition_candidate_count: int = 0,
        vertical_rotation_enabled: bool = False,
        recognizer_backend: Any | None = None,
        reread_enabled: bool = False,
        reread_selection: str = "consensus_grammar_original",
    ) -> None:
        self._bundle = bundle
        self._detector = bundle.detector
        self._recognizer = bundle.recognizer
        self._charset = bundle.charset
        self._preprocess = dict(bundle.preprocess)
        self._device = device
        self._detector_threshold = _probability(
            detector_threshold, "detector_threshold"
        )
        self._recognition_threshold = _probability(
            recognition_threshold, "recognition_threshold"
        )
        for value, name in (
            (detector_min_area, "detector_min_area"),
            (detector_batch_size, "detector_batch_size"),
            (recognition_batch_size, "recognition_batch_size"),
            (max_detections_per_view, "max_detections_per_view"),
            (max_recognition_width, "max_recognition_width"),
        ):
            if type(value) is not int or value <= 0:
                raise TorchOCRAdapterError(f"{name} must be a positive integer")
        if type(recognition_candidate_count) is not int or recognition_candidate_count < 0:
            raise TorchOCRAdapterError(
                "recognition_candidate_count must be a non-negative integer"
            )
        if type(vertical_rotation_enabled) is not bool:
            raise TorchOCRAdapterError(
                "vertical_rotation_enabled must be a boolean"
            )
        if type(reread_enabled) is not bool:
            raise TorchOCRAdapterError("reread_enabled must be a boolean")
        if reread_selection not in _REREAD_SELECTION_POLICIES:
            raise TorchOCRAdapterError(
                f"unsupported reread_selection policy {reread_selection!r}"
            )
        self._detector_min_area = detector_min_area
        self._detector_batch_size = detector_batch_size
        self._recognition_batch_size = recognition_batch_size
        self._max_detections_per_view = max_detections_per_view
        self._max_recognition_width = max_recognition_width
        self._recognition_candidate_count = recognition_candidate_count
        self._vertical_rotation_enabled = vertical_rotation_enabled
        self._reread_enabled = reread_enabled
        self._reread_selection = reread_selection
        self._recognizer_backend = recognizer_backend
        self._last_detection_trace: list[dict[str, Any]] = []
        self._detection_trace_log: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._detector.eval()
        self._recognizer.eval()
        backend_tag = (
            getattr(recognizer_backend, "version", "custom")
            if recognizer_backend is not None
            else None
        )
        self.version = (
            f"torch-ocr/{bundle.version}+{backend_tag}"
            if backend_tag
            else f"torch-ocr/{bundle.version}"
        )

    @property
    def checkpoint_sha256(self) -> str:
        return self._bundle.checkpoint_sha256

    @property
    def charset(self) -> tuple[str, ...]:
        return self._charset.characters

    @property
    def detector_threshold(self) -> float:
        return self._detector_threshold

    @detector_threshold.setter
    def detector_threshold(self, value: float) -> None:
        self._detector_threshold = _probability(value, "detector_threshold")

    @property
    def detector_min_area(self) -> int:
        return self._detector_min_area

    @detector_min_area.setter
    def detector_min_area(self, value: int) -> None:
        if type(value) is not int or value <= 0:
            raise TorchOCRAdapterError("detector_min_area must be a positive integer")
        self._detector_min_area = value

    @property
    def recognition_candidate_count(self) -> int:
        return self._recognition_candidate_count

    @recognition_candidate_count.setter
    def recognition_candidate_count(self, value: int) -> None:
        if type(value) is not int or value < 0:
            raise TorchOCRAdapterError(
                "recognition_candidate_count must be a non-negative integer"
            )
        self._recognition_candidate_count = value

    @property
    def vertical_rotation_enabled(self) -> bool:
        return self._vertical_rotation_enabled

    @vertical_rotation_enabled.setter
    def vertical_rotation_enabled(self, value: bool) -> None:
        if type(value) is not bool:
            raise TorchOCRAdapterError("vertical_rotation_enabled must be a boolean")
        self._vertical_rotation_enabled = value

    @property
    def reread_enabled(self) -> bool:
        return self._reread_enabled

    @reread_enabled.setter
    def reread_enabled(self, value: bool) -> None:
        if type(value) is not bool:
            raise TorchOCRAdapterError("reread_enabled must be a boolean")
        self._reread_enabled = value

    @property
    def reread_selection(self) -> str:
        return self._reread_selection

    @reread_selection.setter
    def reread_selection(self, value: str) -> None:
        if value not in _REREAD_SELECTION_POLICIES:
            raise TorchOCRAdapterError(
                f"unsupported reread_selection policy {value!r}"
            )
        self._reread_selection = value

    def recognize(self, views: list[ImageView], ctx: Any = None) -> list[TextToken]:
        if not isinstance(views, list) or any(not isinstance(view, ImageView) for view in views):
            raise TypeError("TorchOCRAdapter.recognize expects list[ImageView]")
        if not views:
            return []
        identifiers = [view.view_id for view in views]
        if len(set(identifiers)) != len(identifiers):
            raise TorchOCRAdapterError(
                "OCR input view_id values must be unique for unambiguous evidence"
            )
        detection_trace: list[dict[str, Any]] = []
        _check_budget(ctx, "ocr_detector:before")
        torch = _require_torch()
        detector_height, detector_width = self._preprocess["detector_size"]
        candidates: list[_CropCandidate] = []
        with self._lock, torch.inference_mode():
            for batch_start in range(0, len(views), self._detector_batch_size):
                _check_budget(ctx, "ocr_detector_batch:before")
                batch_views = views[batch_start : batch_start + self._detector_batch_size]
                images = torch.stack(
                    [
                        _detector_tensor(
                            view, torch, height=detector_height, width=detector_width
                        )
                        for view in batch_views
                    ],
                    dim=0,
                ).to(self._device)
                logits = _tensor_output(self._detector(images), torch)
                _check_budget(ctx, "ocr_detector_batch:after")
                if tuple(logits.shape) != (
                    len(batch_views), 1, detector_height, detector_width
                ):
                    raise TorchOCRAdapterError(
                        "OCR detector output shape differs from configured detector_size"
                    )
                probabilities = torch.sigmoid(logits).detach().cpu().numpy()
                for offset, view in enumerate(batch_views):
                    local = extract_text_boxes(
                        probabilities[offset, 0],
                        threshold=self._detector_threshold,
                        min_area=self._detector_min_area,
                    )
                    # Apply a score cap first, then restore reading order.
                    local = sorted(
                        local,
                        key=lambda box: (-box.score, box.y1, box.x1, box.y2, box.x2),
                    )[: self._max_detections_per_view]
                    local.sort(key=lambda box: (box.y1, box.x1, box.y2, box.x2, -box.score))
                    view_index = batch_start + offset
                    for detection_index, detector_box in enumerate(local):
                        x1 = max(0.0, min(float(view.width), detector_box.x1 * view.width / detector_width))
                        y1 = max(0.0, min(float(view.height), detector_box.y1 * view.height / detector_height))
                        x2 = max(0.0, min(float(view.width), detector_box.x2 * view.width / detector_width))
                        y2 = max(0.0, min(float(view.height), detector_box.y2 * view.height / detector_height))
                        trace_entry: dict[str, Any] = {
                            "view_id": view.view_id,
                            "view_width": view.width,
                            "view_height": view.height,
                            "detection_index": detection_index,
                            "detector_bbox": [
                                detector_box.x1, detector_box.y1,
                                detector_box.x2, detector_box.y2,
                            ],
                            "detector_score": detector_box.score,
                            "view_bbox": [x1, y1, x2, y2],
                            "original_bbox": _view_bbox_to_original(x1, y1, x2, y2, view),
                            "transform": view.transform.to_records(),
                            "stage": "detected",
                            "geometry_accepted": False,
                            "recognition_attempted": False,
                            "recognition_accepted": False,
                            "outcome": "kept",
                            "reason": None,
                        }
                        if x2 <= x1 or y2 <= y1:
                            trace_entry["outcome"] = "rejected"
                            trace_entry["reason"] = "degenerate_box"
                            detection_trace.append(trace_entry)
                            continue
                        if not _box_has_source_pixel_support(x1, y1, x2, y2):
                            trace_entry["outcome"] = "rejected"
                            trace_entry["reason"] = "no_source_pixel_support"
                            detection_trace.append(trace_entry)
                            continue
                        trace_entry["geometry_accepted"] = True
                        padding = self._preprocess["crop_padding"]
                        crop_box = (
                            max(0, int(math.floor(x1)) - padding),
                            max(0, int(math.floor(y1)) - padding),
                            min(view.width, int(math.ceil(x2)) + padding),
                            min(view.height, int(math.ceil(y2)) + padding),
                        )
                        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                            trace_entry["outcome"] = "rejected"
                            trace_entry["reason"] = "empty_crop"
                            detection_trace.append(trace_entry)
                            continue
                        trace_entry["crop_box"] = list(crop_box)
                        candidate = _CropCandidate(
                            view_index=view_index,
                            detection_index=detection_index,
                            view=view,
                            box=(x1, y1, x2, y2),
                            detector_box=detector_box,
                            crop_box=crop_box,
                        )
                        tensor, width, crop_sha256 = _recognition_tensor(
                            view, crop_box, torch,
                            height=self._preprocess["recognition_height"],
                        )
                        is_vertical = _is_vertical_crop(crop_box)
                        is_wide = _is_wide_crop(crop_box)
                        orientation_candidates = (
                            (0, 90)
                            if self._vertical_rotation_enabled and (is_vertical or is_wide)
                            else (0,)
                        )
                        candidate.orientation_candidates_degrees = orientation_candidates
                        if width <= self._max_recognition_width:
                            candidate.tensor = tensor
                            candidate.width = width
                            candidate.crop_sha256 = crop_sha256
                            if self._recognizer_backend is not None:
                                candidate.np_crop = _crop_to_np(view, crop_box, rotation_degrees=0)
                            candidates.append(candidate)
                            trace_entry["outcome"] = "kept"
                            trace_entry["recognition_attempted"] = True
                            trace_entry["recognition_width"] = width
                            detection_trace.append(trace_entry)
                        else:
                            trace_entry["outcome"] = "rejected"
                            trace_entry["reason"] = "over_max_recognition_width"
                            trace_entry["recognition_width"] = width
                            detection_trace.append(trace_entry)
                        if orientation_candidates == (0, 90):
                            rotated_tensor, rotated_width, rotated_sha256 = _recognition_tensor(
                                view,
                                crop_box,
                                torch,
                                height=self._preprocess["recognition_height"],
                                rotation_degrees=90,
                            )
                            if rotated_width <= self._max_recognition_width:
                                rotated_np = (
                                    _crop_to_np(view, crop_box, rotation_degrees=90)
                                    if self._recognizer_backend is not None
                                    else None
                                )
                                candidates.append(
                                    _CropCandidate(
                                        view_index=view_index,
                                        detection_index=detection_index,
                                        view=view,
                                        box=(x1, y1, x2, y2),
                                        detector_box=detector_box,
                                        crop_box=crop_box,
                                        tensor=rotated_tensor,
                                        width=rotated_width,
                                        crop_sha256=rotated_sha256,
                                        orientation_degrees=90,
                                        orientation_candidates_degrees=orientation_candidates,
                                        np_crop=rotated_np,
                                    )
                                )
                        if self._reread_enabled and width <= self._max_recognition_width:
                            for variant_name, variant_box, pre_upscale in _reread_variant_crops(
                                (x1, y1, x2, y2), view.width, view.height
                            ):
                                v_tensor, v_width, v_sha256 = _recognition_tensor(
                                    view,
                                    variant_box,
                                    torch,
                                    height=self._preprocess["recognition_height"],
                                    pre_upscale=pre_upscale,
                                )
                                if v_width > self._max_recognition_width:
                                    continue
                                candidates.append(
                                    _CropCandidate(
                                        view_index=view_index,
                                        detection_index=detection_index,
                                        view=view,
                                        box=(x1, y1, x2, y2),
                                        detector_box=detector_box,
                                        crop_box=variant_box,
                                        tensor=v_tensor,
                                        width=v_width,
                                        crop_sha256=v_sha256,
                                        orientation_candidates_degrees=(0,),
                                        np_crop=(
                                            _crop_to_np(view, variant_box, pre_upscale=pre_upscale)
                                            if self._recognizer_backend is not None
                                            else None
                                        ),
                                        variant_name=variant_name,
                                        pre_upscale=pre_upscale,
                                    )
                                )
        _check_budget(ctx, "ocr_detector:after")
        self._last_detection_trace = detection_trace
        if not candidates:
            _record_version(ctx, self.version)
            return []

        _check_budget(ctx, "ocr_recognizer:before")
        recognized: list[
            tuple[_CropCandidate, str, float, list[str], int, float]
        ] = []
        # Re-read variants are recognized in their own batches: the base crops
        # must keep the exact flag-off batch layout so their reads stay
        # bit-identical (batch composition must not perturb the original reads).
        base_candidates = [c for c in candidates if c.variant_name == "base"]
        variant_candidates = [c for c in candidates if c.variant_name != "base"]
        if self._recognizer_backend is not None:
            for group in (base_candidates, variant_candidates):
                if group:
                    recognized.extend(
                        self._recognize_with_backend(group, detection_trace, ctx=ctx)
                    )
        else:
            recognition_batches = _recognition_batches(
                base_candidates, self._recognition_batch_size
            )
            if variant_candidates:
                recognition_batches = recognition_batches + _recognition_batches(
                    variant_candidates, self._recognition_batch_size
                )
            with self._lock, torch.inference_mode():
                for batch_index, batch in enumerate(recognition_batches):
                    _check_budget(ctx, "ocr_recognition_batch:before")
                    maximum_width = max(candidate.width for candidate in batch)
                    padding_ratio = _padding_ratio(batch, maximum_width)
                    height = self._preprocess["recognition_height"]
                    padded = torch.full(
                        (len(batch), 1, height, maximum_width),
                        float(self._preprocess["recognition_pad_value"]),
                        dtype=torch.float32,
                    )
                    widths = torch.tensor(
                        [candidate.width for candidate in batch], dtype=torch.long
                    )
                    for index, candidate in enumerate(batch):
                        if candidate.tensor is None:
                            raise TorchOCRAdapterError("internal OCR crop tensor is missing")
                        padded[index, :, :, : candidate.width] = candidate.tensor
                    raw_logits = self._recognizer(
                        padded.to(self._device), widths.to(self._device)
                    )
                    logits = _tensor_output(raw_logits, torch)
                    input_lengths = recognition_output_lengths(widths)
                    texts, scores = greedy_ctc_decode(logits, input_lengths, self._charset)
                    candidate_lists = (
                        ctc_beam_decode(
                            logits,
                            input_lengths,
                            self._charset,
                            candidate_count=self._recognition_candidate_count,
                        )
                        if self._recognition_candidate_count > 0
                        else [[] for _ in batch]
                    )
                    for candidate, text, score, raw_candidates in zip(
                        batch, texts, scores, candidate_lists, strict=True
                    ):
                        if not text or score < self._recognition_threshold:
                            for entry in detection_trace:
                                if (
                                    candidate.variant_name == "base"
                                    and entry.get("view_id") == candidate.view.view_id
                                    and entry.get("detection_index") == candidate.detection_index
                                    and entry.get("recognition_attempted")
                                ):
                                    entry["recognition_accepted"] = False
                                    entry["outcome"] = "rejected"
                                    entry["reason"] = (
                                        "empty_recognition" if not text else "below_recognition_threshold"
                                    )
                                    entry["rec_score"] = score
                                    entry["raw_text"] = text or None
                            continue
                        alternatives = [
                            value for value in raw_candidates if value != text
                        ][: self._recognition_candidate_count]
                        recognized.append(
                            (
                                candidate,
                                text,
                                score,
                                alternatives,
                                batch_index,
                                padding_ratio,
                            )
                        )
                        for entry in detection_trace:
                            if (
                                candidate.variant_name == "base"
                                and entry.get("view_id") == candidate.view.view_id
                                and entry.get("detection_index") == candidate.detection_index
                                and entry.get("recognition_attempted")
                            ):
                                entry["recognition_accepted"] = True
                                entry["raw_text"] = text
                                entry["rec_score"] = score
                                entry["candidates"] = list(alternatives)
        _check_budget(ctx, "ocr_recognizer:after")
        recognized = _select_orientation_results(recognized)
        reread_meta: dict[tuple[int, int], list[dict[str, Any]]] = {}
        if self._reread_enabled:
            recognized, reread_meta = _select_reread_results(
                recognized, self._reread_selection
            )

        recognized.sort(
            key=lambda item: (
                item[0].view_index,
                item[0].box[1],
                item[0].box[0],
                item[0].box[3],
                item[0].box[2],
                item[1],
            )
        )
        occurrence: dict[str, int] = {}
        output: list[TextToken] = []
        for (
            candidate,
            text,
            recognition_score,
            text_candidates,
            recognition_batch_index,
            recognition_padding_ratio,
        ) in recognized:
            detector_score = candidate.detector_box.score
            raw_score = math.sqrt(max(0.0, detector_score * recognition_score))
            # Word-level calibration: correct length-dependent overconfidence.
            from pcbparse.association.ocr_calibration import calibrate_score

            score = calibrate_score(raw_score, text)
            image_identity = candidate.view.sha256 or hashlib.sha256(
                candidate.view.image.convert("RGB").tobytes()
            ).hexdigest()
            signature = (
                f"{self._bundle.checkpoint_sha256}|{image_identity}|{candidate.view.view_id}|"
                + "|".join(f"{value:.4f}" for value in candidate.box)
                + f"|{text}"
            )
            digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:20]
            duplicate_index = occurrence.get(digest, 0)
            occurrence[digest] = duplicate_index + 1
            suffix = "" if duplicate_index == 0 else f":{duplicate_index}"
            token_id = f"ocr-local:{digest}{suffix}"
            evidence_id = (
                f"ocr:{self._bundle.checkpoint_sha256[:16]}:{digest}:{duplicate_index}"
            )
            x1, y1, x2, y2 = candidate.box
            metadata: dict[str, Any] = {
                "architecture": {
                    "detector": "tiny_ocr_unet",
                    "recognizer": "tiny_crnn_bilstm_ctc",
                },
                "coordinate_system": "image_view_local_xyxy",
                "crop_source": "original_view_pixels",
                "crop_sha256": candidate.crop_sha256,
                "orientation_degrees": candidate.orientation_degrees,
                "orientation_candidates_degrees": list(
                    candidate.orientation_candidates_degrees
                ),
                "orientation_policy": (
                    "identity_plus_90_vertical_v1"
                    if len(candidate.orientation_candidates_degrees) > 1
                    else "identity_only_v1"
                ),
                "transform_chain": candidate.view.transform.to_records(),
                "detector_score": detector_score,
                "recognition_score": recognition_score,
                "detector_component_area": candidate.detector_box.area,
                "detector_index": candidate.detection_index,
                "crop_box": list(candidate.crop_box),
                "recognition_width": candidate.width,
                "recognition_batch_index": recognition_batch_index,
                "recognition_padding_ratio": recognition_padding_ratio,
                "checkpoint_epoch": self._bundle.epoch,
                "checkpoint_global_step": self._bundle.global_step,
                "semantic_fields_inferred": False,
                "recognition_candidate_count": self._recognition_candidate_count,
                "recognition_candidates": list(text_candidates),
            }
            if self._reread_enabled:
                metadata["reread"] = {
                    "policy": self._reread_selection,
                    "selected_variant": candidate.variant_name,
                    "pre_upscale": candidate.pre_upscale,
                    "variants": reread_meta.get(
                        (candidate.view_index, candidate.detection_index), []
                    ),
                }
            output.append(
                TextToken(
                    internal_id=token_id,
                    raw_text=text,
                    candidates=text_candidates,
                    resolved_text=None,
                    category="unknown",
                    details="",
                    bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    score=max(0.0, min(1.0, score)),
                    source_object_id=None,
                    component_id=None,
                    pin_id=None,
                    evidence=[
                        Evidence(
                            evidence_id=evidence_id,
                            source="ocr",
                            view_id=candidate.view.view_id,
                            score=max(0.0, min(1.0, score)),
                            payload_ref=f"sha256:{self._bundle.checkpoint_sha256}",
                            metadata=metadata,
                        )
                    ],
                )
            )
        _record_version(ctx, self.version)
        return output

    def _recognize_with_backend(
        self,
        candidates: list[_CropCandidate],
        detection_trace: list[dict[str, Any]],
        *, ctx: Any = None,
    ) -> list[tuple[_CropCandidate, str, float, list[str], int, float]]:
        """Recognize crops via an external ``RecognizerBackend``."""
        backend = self._recognizer_backend
        np_crops = [
            c.np_crop if c.np_crop is not None else _crop_to_np(c.view, c.crop_box, rotation_degrees=c.orientation_degrees)
            for c in candidates
        ]
        from pcbparse.perception.recognizer_backend import PaddleRecognizerBackend
        if isinstance(backend, PaddleRecognizerBackend):
            results = backend.recognize_crops(
                np_crops, budget_check=lambda: _check_budget(ctx, "external_ocr_batch"),
            )
        else:
            _check_budget(ctx, "external_ocr:before")
            results = backend.recognize_crops(np_crops)
            _check_budget(ctx, "external_ocr:after")
        recognized: list[tuple[_CropCandidate, str, float, list[str], int, float]] = []
        for candidate, (text, score) in zip(candidates, results, strict=True):
            if not text or score < self._recognition_threshold:
                for entry in detection_trace:
                    if (
                        entry.get("view_id") == candidate.view.view_id
                        and entry.get("detection_index") == candidate.detection_index
                        and entry.get("recognition_attempted")
                    ):
                        entry["recognition_accepted"] = False
                        entry["outcome"] = "rejected"
                        entry["reason"] = (
                            "empty_recognition" if not text else "below_recognition_threshold"
                        )
                        entry["rec_score"] = score
                        entry["raw_text"] = text or None
                continue
            recognized.append((candidate, text, score, [], 0, 0.0))
            for entry in detection_trace:
                if (
                    entry.get("view_id") == candidate.view.view_id
                    and entry.get("detection_index") == candidate.detection_index
                    and entry.get("recognition_attempted")
                ):
                    entry["recognition_accepted"] = True
                    entry["raw_text"] = text
                    entry["rec_score"] = score
                    entry["candidates"] = []
        return recognized


def factory(path: str | Path, entry: Mapping[str, Any]) -> TorchOCRAdapter:
    """Load a hash-bound unified bundle as an ``OCRProvider``."""

    if not isinstance(entry, Mapping):
        raise TypeError("OCR model-lock entry must be a mapping")
    task = entry.get("task")
    if task is not None and task != "ocr":
        raise TorchOCRAdapterError(f"Torch OCR adapter cannot load task {task!r}")
    torch = _require_torch()
    raw_device = _entry_value(entry, "device", default="cpu")
    if not isinstance(raw_device, str) or not raw_device.strip():
        raise TorchOCRAdapterError("OCR adapter device must be a non-empty string")
    device = raw_device.strip().lower()
    if device == "auto":
        device = "cuda:0" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
    try:
        resolved_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise TorchOCRAdapterError(f"invalid OCR adapter device {device!r}") from exc
    if resolved_device.type == "cuda" and not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
        raise TorchOCRAdapterError("OCR adapter requested CUDA but CUDA is unavailable")
    expected_sha = entry.get("sha256")
    if expected_sha is not None and not isinstance(expected_sha, str):
        raise TorchOCRAdapterError("OCR model-lock sha256 must be a string")
    expected_lineage = entry.get("lineage")
    if expected_lineage is not None and not isinstance(expected_lineage, Mapping):
        raise TorchOCRAdapterError("OCR model-lock lineage must be a mapping")
    try:
        bundle = load_ocr_bundle(
            path,
            device=resolved_device,
            expected_sha256=expected_sha,
            expected_lineage=expected_lineage,
            strict=True,
        )
    except Exception as exc:
        if isinstance(exc, TorchOCRAdapterError):
            raise
        raise TorchOCRAdapterError(str(exc)) from exc
    preprocess = bundle.preprocess
    adapter = TorchOCRAdapter(
        bundle,
        device=str(resolved_device),
        detector_threshold=_entry_number(
            entry,
            "detector_threshold",
            default=preprocess["detector_threshold"],
        ),
        detector_min_area=_entry_integer(
            entry, "detector_min_area", default=preprocess["detector_min_area"]
        ),
        recognition_threshold=_entry_number(
            entry, "recognition_threshold", default=0.0
        ),
        detector_batch_size=_entry_integer(entry, "detector_batch_size", default=4),
        recognition_batch_size=_entry_integer(
            entry, "recognition_batch_size", default=32
        ),
        max_detections_per_view=_entry_integer(
            entry, "max_detections_per_view", default=512
        ),
        max_recognition_width=_entry_integer(
            entry, "max_recognition_width", default=4096
        ),
        recognition_candidate_count=_entry_nonnegative_integer(
            entry, "recognition_candidate_count", default=0
        ),
        reread_enabled=_entry_value(entry, "reread_enabled", default=False),
        reread_selection=str(
            _entry_value(entry, "reread_selection", default="consensus_grammar_original")
        ),
    )
    backend_spec = _entry_value(entry, "recognizer_backend", default=None)
    if backend_spec is not None:
        adapter._recognizer_backend = _load_recognizer_backend(backend_spec)
        adapter.version = (
            f"torch-ocr/{bundle.version}+{adapter._recognizer_backend.version}"
        )
    return adapter


load = factory


def _detector_tensor(
    view: ImageView, torch: Any, *, height: int, width: int
) -> Any:
    image = view.image.convert("RGB").resize(
        (width, height), Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32).copy() / np.float32(255.0)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _recognition_tensor(
    view: ImageView,
    crop_box: tuple[int, int, int, int],
    torch: Any,
    *,
    height: int,
    rotation_degrees: int = 0,
    pre_upscale: int = 1,
) -> tuple[Any, int, str]:
    if rotation_degrees not in {0, 90}:
        raise TorchOCRAdapterError(
            "OCR crop rotation_degrees must be 0 or 90"
        )
    if type(pre_upscale) is not int or pre_upscale < 1:
        raise TorchOCRAdapterError("OCR crop pre_upscale must be a positive integer")
    source_crop = view.image.crop(crop_box).convert("L")
    crop = source_crop
    if rotation_degrees == 90:
        crop = source_crop.rotate(90, expand=True, resample=Image.Resampling.BILINEAR)
    if pre_upscale > 1:
        crop = crop.resize(
            (crop.width * pre_upscale, crop.height * pre_upscale),
            Image.Resampling.LANCZOS,
        )
    try:
        crop_sha256 = _image_crop_sha256(crop)
        width = max(1, int(math.floor(crop.width * height / crop.height + 0.5)))
        resized = crop.resize((width, height), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32).copy() / np.float32(255.0)
        return (
            torch.from_numpy(array).unsqueeze(0).contiguous(),
            width,
            crop_sha256,
        )
    finally:
        crop.close()
        if crop is not source_crop:
            source_crop.close()


def _image_crop_sha256(crop: Image.Image) -> str:
    header = f"{crop.mode}:{crop.width}x{crop.height}\0".encode("ascii")
    return hashlib.sha256(header + crop.tobytes()).hexdigest()


def _is_vertical_crop(crop_box: tuple[int, int, int, int]) -> bool:
    width = crop_box[2] - crop_box[0]
    height = crop_box[3] - crop_box[1]
    return width > 0 and height >= width * 1.25


def _is_wide_crop(crop_box: tuple[int, int, int, int]) -> bool:
    width = crop_box[2] - crop_box[0]
    height = crop_box[3] - crop_box[1]
    return height > 0 and width >= height * 3.0


def _crop_to_np(
    view: ImageView,
    crop_box: tuple[int, int, int, int],
    *,
    rotation_degrees: int = 0,
    pre_upscale: int = 1,
) -> Any:
    """Extract a crop as a grayscale numpy array for external recognizer backends."""
    if type(pre_upscale) is not int or pre_upscale < 1:
        raise TorchOCRAdapterError("OCR crop pre_upscale must be a positive integer")
    source_crop = view.image.crop(crop_box).convert("L")
    crop = source_crop
    if rotation_degrees == 90:
        crop = source_crop.rotate(90, expand=True, resample=Image.Resampling.BILINEAR)
    if pre_upscale > 1:
        crop = crop.resize(
            (crop.width * pre_upscale, crop.height * pre_upscale),
            Image.Resampling.LANCZOS,
        )
    try:
        return np.asarray(crop, dtype=np.uint8).copy()
    finally:
        crop.close()
        if crop is not source_crop:
            source_crop.close()


def _select_orientation_results(
    recognized: Sequence[tuple[_CropCandidate, str, float, list[str], int, float]],
) -> list[tuple[_CropCandidate, str, float, list[str], int, float]]:
    """Keep the best orientation for each detector box.

    Uses calibrated confidence (recognition score + text quality signals)
    rather than raw recognition score alone.  When scores are close, prefers
    the text that matches a known grammar pattern (designator, pin number,
    value) and prefers 0° over 90° as a deterministic tie-breaker.
    """
    selected: dict[tuple[int, int, str], tuple[_CropCandidate, str, float, list[str], int, float]] = {}
    for item in recognized:
        candidate = item[0]
        text = item[1]
        score = item[2]
        key = (candidate.view_index, candidate.detection_index, candidate.variant_name)
        current = selected.get(key)
        if current is None:
            selected[key] = item
            continue
        current_text = current[1]
        current_score = current[2]
        current_candidate = current[0]
        item_quality = _orientation_quality(text, score)
        current_quality = _orientation_quality(current_text, current_score)
        if item_quality > current_quality + 1e-9 or (
            abs(item_quality - current_quality) <= 1e-9
            and candidate.orientation_degrees < current_candidate.orientation_degrees
        ):
            selected[key] = item
    return list(selected.values())


def _orientation_quality(text: str, score: float) -> float:
    """Combined quality score for orientation selection.

    Weights: 70% recognition score + 20% structure validity + 10% charset.
    """
    base = max(0.0, min(1.0, float(score)))
    structure = 1.0 if _text_has_structure(text) else 0.0
    charset = 1.0 if _text_is_clean_charset(text) else 0.0
    return 0.70 * base + 0.20 * structure + 0.10 * charset


_TEXT_STRUCTURE_PATTERNS = (
    __import__("re").compile(r"^[A-Z]{1,4}[0-9]+[A-Z]?$"),          # designator
    __import__("re").compile(r"^[0-9]{1,3}$"),                       # pin number
    __import__("re").compile(r"^(?:[A-Z]{1,2}[0-9]{1,2}|[0-9]{1,2}[A-Z]{1,2})$"),  # alnum pin
    __import__("re").compile(r"^(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*[kKmMuUnNpPFfHhVvAaWwRrΩµμ%]*$"),  # value
)


def _text_has_structure(text: str) -> bool:
    stripped = text.strip()
    if any(p.fullmatch(stripped) for p in _TEXT_STRUCTURE_PATTERNS):
        return True
    # Known net/power names are structurally valid.
    from pcbparse.association.ocr_dictionary import NET_DICTIONARY

    return stripped.upper() in NET_DICTIONARY


def _text_is_clean_charset(text: str) -> bool:
    """Whether all chars are common ASCII alphanumeric/symbols (no noise)."""
    stripped = text.strip()
    return bool(stripped) and all(
        ch.isalnum() or ch in "._+-/Ωµ%," for ch in stripped
    )


def _recognition_batches(
    candidates: Sequence[_CropCandidate],
    batch_size: int,
) -> list[list[_CropCandidate]]:
    """Group normalized-width crops to reduce dynamic-width padding.

    Sorting only changes the recognizer batch layout.  The caller restores
    deterministic view/box order before emitting tokens, so this remains
    output-compatible with the legacy provider while making padding measurable.
    """

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.width,
            candidate.view_index,
            candidate.box[1],
            candidate.box[0],
            candidate.box[3],
            candidate.box[2],
            candidate.detection_index,
        ),
    )
    return [
        ordered[start : start + batch_size]
        for start in range(0, len(ordered), batch_size)
    ]


def _padding_ratio(batch: Sequence[_CropCandidate], maximum_width: int) -> float:
    if not batch or maximum_width <= 0:
        return 0.0
    useful_width = sum(candidate.width for candidate in batch)
    total_width = len(batch) * maximum_width
    return max(0.0, min(1.0, 1.0 - useful_width / total_width))


def _tensor_output(value: Any, torch: Any) -> Any:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1 and torch.is_tensor(value[0]):
        return value[0]
    if isinstance(value, Mapping):
        for key in ("logits", "output"):
            if torch.is_tensor(value.get(key)):
                return value[key]
    raise TorchOCRAdapterError("OCR model output is not one logits tensor")


def _entry_number(entry: Mapping[str, Any], name: str, *, default: float) -> float:
    value = _entry_value(entry, name, default=default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TorchOCRAdapterError(f"OCR entry {name} must be numeric")
    return float(value)


def _entry_integer(entry: Mapping[str, Any], name: str, *, default: int) -> int:
    value = _entry_value(entry, name, default=default)
    if type(value) is not int or value <= 0:
        raise TorchOCRAdapterError(f"OCR entry {name} must be a positive integer")
    return value


def _entry_nonnegative_integer(
    entry: Mapping[str, Any], name: str, *, default: int
) -> int:
    value = _entry_value(entry, name, default=default)
    if type(value) is not int or value < 0:
        raise TorchOCRAdapterError(
            f"OCR entry {name} must be a non-negative integer"
        )
    return value


def _entry_value(entry: Mapping[str, Any], name: str, *, default: Any) -> Any:
    inference = entry.get("inference")
    if inference is not None and not isinstance(inference, Mapping):
        raise TorchOCRAdapterError("OCR model-lock inference must be an object")
    nested = inference if isinstance(inference, Mapping) else {}
    if name in entry and name in nested and entry[name] != nested[name]:
        raise TorchOCRAdapterError(
            f"OCR model-lock inference.{name} conflicts with top-level {name}"
        )
    if name in entry:
        return entry[name]
    return nested.get(name, default)


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TorchOCRAdapterError(f"{name} must be numeric in [0,1]")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise TorchOCRAdapterError(f"{name} must be finite in [0,1]")
    return number


def _view_bbox_to_original(
    x1: float, y1: float, x2: float, y2: float, view: Any
) -> list[float]:
    """Restore a view-local bbox to original-image coordinates via the transform."""
    from pcbparse.ir.models import Point

    corners = [
        view.to_original(Point(x=x1, y=y1)),
        view.to_original(Point(x=x2, y=y1)),
        view.to_original(Point(x=x1, y=y2)),
        view.to_original(Point(x=x2, y=y2)),
    ]
    return [
        min(p.x for p in corners),
        min(p.y for p in corners),
        max(p.x for p in corners),
        max(p.y for p in corners),
    ]


def _box_has_source_pixel_support(
    x1: float, y1: float, x2: float, y2: float
) -> bool:
    """Return whether a mapped detector box spans a real source pixel."""

    values = (x1, y1, x2, y2)
    return (
        all(math.isfinite(value) for value in values)
        and x2 - x1 >= 1.0
        and y2 - y1 >= 1.0
    )


def _check_budget(ctx: Any, stage: str) -> None:
    if ctx is None:
        return
    budget = getattr(ctx, "time_budget", None)
    check = getattr(budget, "check", None)
    if callable(check):
        check(stage=stage)


def _record_version(ctx: Any, version: str) -> None:
    if ctx is None:
        return
    versions = getattr(ctx, "model_versions", None)
    if isinstance(versions, dict):
        versions["ocr"] = version


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        raise TorchOCRAdapterError(
            "PyTorch is required for the OCR adapter; use the project .venv-train runtime"
        ) from exc


def _load_recognizer_backend(spec: Any) -> Any:
    """Instantiate a ``RecognizerBackend`` from a string or mapping spec.

    A plain string is interpreted as a PaddleOCR model name (e.g.
    ``"en_PP-OCRv5_mobile_rec"``).  A mapping with ``type``/``class`` keys
    selects a fully-qualified adapter class.
    """
    if isinstance(spec, str):
        from pcbparse.perception.recognizer_backend import PaddleRecognizerBackend

        return PaddleRecognizerBackend(model_name=spec)
    if isinstance(spec, Mapping):
        class_path = spec.get("class") or spec.get("type")
        if not isinstance(class_path, str) or not class_path:
            raise TorchOCRAdapterError(
                "recognizer_backend mapping requires a 'class' or 'type' key"
            )
        module_name, _, attr_name = class_path.rpartition(":")
        if not module_name or not attr_name:
            raise TorchOCRAdapterError(
                f"recognizer_backend class must be 'module:attr', got {class_path!r}"
            )
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, attr_name)
        except (ImportError, AttributeError) as exc:
            raise TorchOCRAdapterError(
                f"cannot import recognizer_backend class {class_path!r}"
            ) from exc
        kwargs = {k: v for k, v in spec.items() if k not in ("class", "type")}
        return cls(**kwargs)
    raise TorchOCRAdapterError(
        f"recognizer_backend must be a string or mapping, got {type(spec).__name__}"
    )


__all__ = ["TorchOCRAdapter", "TorchOCRAdapterError", "factory", "load"]
