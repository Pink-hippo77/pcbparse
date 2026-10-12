"""Unified recognizer backend protocol for torch and PaddleOCR.

Provides a single ``recognize_crops`` interface that abstracts away
framework-specific preprocessing, charset, and decoding.  Circuit-IR is
never touched here.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np


class RecognizerBackend(Protocol):
    """Unified crop-level text recognition interface."""

    version: str

    def recognize_crops(
        self, crops: Sequence[np.ndarray]
    ) -> list[tuple[str | None, float]]:
        """Recognize text in grayscale/RGB crop arrays.

        Returns one ``(text, score)`` per crop.  ``text`` is ``None`` when
        recognition produced no output.  ``score`` is the backend's own
        confidence and is NOT calibrated across backends.
        """
        ...


class TorchRecognizerBackend:
    """Adapter around the existing torch CRNN recognizer."""

    def __init__(self, bundle: Any, *, device: str = "cpu") -> None:
        import torch
        from pcbparse.training.torch_ocr_model import (
            greedy_ctc_decode,
            recognition_output_lengths,
        )

        self._bundle = bundle
        self._torch = torch
        self._greedy = greedy_ctc_decode
        self._lengths = recognition_output_lengths
        self._device = device
        self._height = bundle.preprocess["recognition_height"]
        self._pad_value = bundle.preprocess["recognition_pad_value"]
        self._charset = bundle.charset
        self._recognizer = bundle.recognizer
        self.version = f"torch/{bundle.version}"

    def recognize_crops(
        self, crops: Sequence[np.ndarray]
    ) -> list[tuple[str | None, float]]:
        torch = self._torch
        results: list[tuple[str | None, float]] = []
        for crop_array in crops:
            if crop_array is None or crop_array.size == 0:
                results.append((None, 0.0))
                continue
            from PIL import Image

            if crop_array.ndim == 3:
                img = Image.fromarray(crop_array).convert("L")
            else:
                img = Image.fromarray(crop_array)
            h, w = img.height, img.width
            if h <= 0 or w <= 0:
                results.append((None, 0.0))
                continue
            target_h = self._height
            target_w = max(1, int(round(w * target_h / h)))
            resized = img.resize((target_w, target_h), Image.Resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.float32).copy() / 255.0
            tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).contiguous()
            widths = torch.tensor([target_w], dtype=torch.long)
            with torch.inference_mode():
                logits = self._recognizer(tensor.to(self._device), widths.to(self._device))
            texts, scores = self._greedy(logits, self._lengths(widths), self._charset)
            text = texts[0] if texts else None
            score = float(scores[0]) if scores else 0.0
            results.append((text if text else None, score))
        return results


class PaddleRecognizerBackend:
    """Adapter around PaddleOCR TextRecognition."""

    def __init__(
        self, model_name: str = "en_PP-OCRv5_mobile_rec", *,
        batch_size: int = 16, cache_size: int = 4096, model_dir: str | None = None,
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")
        if model_dir is not None and not Path(model_dir).is_dir():
            raise ValueError(f"recognizer model_dir is not a directory: {model_dir}")
        from paddleocr import TextRecognition

        kwargs = {"model_dir": model_dir} if model_dir is not None else {}
        self._model = TextRecognition(model_name=model_name, **kwargs)
        self.version = f"paddle/{model_name}"
        self.batch_size = batch_size
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple, tuple[str | None, float]] = OrderedDict()
        self.last_report: dict[str, Any] = {}

    def recognize_crops(
        self, crops: Sequence[np.ndarray], *, budget_check: Callable[[], None] | None = None,
    ) -> list[tuple[str | None, float]]:
        results: list[tuple[str | None, float]] = [(None, 0.0)] * len(crops)
        pending: dict[tuple, tuple[np.ndarray, list[int]]] = {}
        report: dict[str, Any] = {"input_count": len(crops), "cache_hits": 0,
                                  "duplicate_inputs": 0, "model_batches": 0,
                                  "invalid_crops": 0, "errors": []}
        self.last_report = report
        for index, crop_array in enumerate(crops):
            if not isinstance(crop_array, np.ndarray) or crop_array.size == 0:
                report["invalid_crops"] += 1
                continue
            # Paddle's resize operator indexes the channel axis. Torch adapter
            # crops are grayscale, so preserve their pixels in three channels.
            if crop_array.ndim == 2:
                crop_array = np.repeat(crop_array[:, :, None], 3, axis=2)
            elif crop_array.ndim == 3 and crop_array.shape[2] == 1:
                crop_array = np.repeat(crop_array, 3, axis=2)
            if crop_array.ndim != 3 or crop_array.shape[2] != 3:
                report["invalid_crops"] += 1
                continue
            key = (crop_array.shape, crop_array.dtype.str,
                   hashlib.sha256(crop_array.tobytes()).hexdigest())
            if key in self._cache:
                results[index] = self._cache[key]
                self._cache.move_to_end(key)
                report["cache_hits"] += 1
            elif key in pending:
                pending[key][1].append(index)
                report["duplicate_inputs"] += 1
            else:
                pending[key] = (crop_array, [index])
        items = list(pending.items())
        report["unique_uncached_crops"] = len(items)
        for start in range(0, len(items), self.batch_size):
            if budget_check is not None:
                budget_check()  # Budget errors propagate; never become empty OCR.
            batch = items[start:start + self.batch_size]
            try:
                report["model_batches"] += 1
                output = list(self._model.predict(input=[item[1][0] for item in batch],
                                                  batch_size=self.batch_size))
                if len(output) != len(batch):
                    raise ValueError(f"recognizer returned {len(output)} results for {len(batch)} crops")
                parsed = []
                for res in output:
                    text = res.get("rec_text")
                    if text is not None and not isinstance(text, str):
                        raise TypeError(f"rec_text must be str, got {type(text).__name__}")
                    score = res.get("rec_score", 0.0)
                    if not isinstance(score, (int, float)) or isinstance(score, bool):
                        raise TypeError("rec_score must be numeric")
                    score = float(score)
                    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                        raise ValueError("rec_score must be finite and in [0, 1]")
                    parsed.append((text or None, score))
            except Exception as exc:
                report["errors"].append({"indices": [i for _, (_, indices) in batch for i in indices],
                                         "type": type(exc).__name__, "message": str(exc)})
                continue  # Failed batches are not cached and can recover next call.
            if budget_check is not None:
                budget_check()
            for (key, (_, indices)), result in zip(batch, parsed, strict=True):
                for index in indices:
                    results[index] = result
                if self.cache_size:
                    self._cache[key] = result
                    while len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)
        return results


def crop_to_array(view_image: Any, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    """Extract a crop from a PIL image as a numpy array."""
    crop = view_image.crop(crop_box)
    return np.array(crop)
