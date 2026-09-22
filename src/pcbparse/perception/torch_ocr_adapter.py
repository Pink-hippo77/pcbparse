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
        self._detector_min_area = detector_min_area
        self._detector_batch_size = detector_batch_size
        self._recognition_batch_size = recognition_batch_size
        self._max_detections_per_view = max_detections_per_view
        self._max_recognition_width = max_recognition_width
        self._recognition_candidate_count = recognition_candidate_count
        self._vertical_rotation_enabled = vertical_rotation_enabled
        self._lock = threading.RLock()
        self._detector.eval()
        self._recognizer.eval()
        self.version = f"torch-ocr/{bundle.version}"

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
        _check_budget(ctx, "ocr_detector:before")
        torch = _require_torch()
        detector_height, detector_width = self._preprocess["detector_size"]
        candidates: list[_CropCandidate] = []
        with self._lock, torch.inference_mode():
            for batch_start in range(0, len(views), self._detector_batch_size):
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
                        if x2 <= x1 or y2 <= y1:
                            continue
                        # Component and side-strip views can be much smaller
                        # than the fixed detector canvas. A component mapping
                        # to less than one source pixel is an upsampling
                        # artefact, not a resolvable text crop. Reject it before
                        # padding lets the recognizer hallucinate a character.
                        if not _box_has_source_pixel_support(x1, y1, x2, y2):
                            continue
                        padding = self._preprocess["crop_padding"]
                        crop_box = (
                            max(0, int(math.floor(x1)) - padding),
                            max(0, int(math.floor(y1)) - padding),
                            min(view.width, int(math.ceil(x2)) + padding),
                            min(view.height, int(math.ceil(y2)) + padding),
                        )
                        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                            continue
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
                        orientation_candidates = (
                            (0, 90)
                            if self._vertical_rotation_enabled and is_vertical
                            else (0,)
                        )
                        candidate.orientation_candidates_degrees = orientation_candidates
                        if width <= self._max_recognition_width:
                            candidate.tensor = tensor
                            candidate.width = width
                            candidate.crop_sha256 = crop_sha256
                            candidates.append(candidate)
                        if orientation_candidates == (0, 90):
                            rotated_tensor, rotated_width, rotated_sha256 = _recognition_tensor(
                                view,
                                crop_box,
                                torch,
                                height=self._preprocess["recognition_height"],
                                rotation_degrees=90,
                            )
                            if rotated_width <= self._max_recognition_width:
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
                                    )
                                )
        _check_budget(ctx, "ocr_detector:after")
        if not candidates:
            _record_version(ctx, self.version)
            return []

        _check_budget(ctx, "ocr_recognizer:before")
        recognized: list[
            tuple[_CropCandidate, str, float, list[str], int, float]
        ] = []
        with self._lock, torch.inference_mode():
            for batch_index, batch in enumerate(
                _recognition_batches(candidates, self._recognition_batch_size)
            ):
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
        _check_budget(ctx, "ocr_recognizer:after")
        recognized = _select_orientation_results(recognized)

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
            score = math.sqrt(max(0.0, detector_score * recognition_score))
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
                            metadata={
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
                            },
                        )
                    ],
                )
            )
        _record_version(ctx, self.version)
        return output


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
    return TorchOCRAdapter(
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
    )


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
) -> tuple[Any, int, str]:
    if rotation_degrees not in {0, 90}:
        raise TorchOCRAdapterError(
            "OCR crop rotation_degrees must be 0 or 90"
        )
    source_crop = view.image.crop(crop_box).convert("L")
    crop = source_crop
    if rotation_degrees == 90:
        crop = source_crop.rotate(90, expand=True, resample=Image.Resampling.BILINEAR)
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


def _select_orientation_results(
    recognized: Sequence[tuple[_CropCandidate, str, float, list[str], int, float]],
) -> list[tuple[_CropCandidate, str, float, list[str], int, float]]:
    """Keep the highest-confidence orientation for each detector box."""

    selected: dict[tuple[int, int], tuple[_CropCandidate, str, float, list[str], int, float]] = {}
    for item in recognized:
        candidate = item[0]
        key = (candidate.view_index, candidate.detection_index)
        current = selected.get(key)
        if current is None:
            selected[key] = item
            continue
        current_candidate = current[0]
        if item[2] > current[2] + 1e-9 or (
            abs(item[2] - current[2]) <= 1e-9
            and candidate.orientation_degrees < current_candidate.orientation_degrees
        ):
            selected[key] = item
    return list(selected.values())


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


__all__ = ["TorchOCRAdapter", "TorchOCRAdapterError", "factory", "load"]
