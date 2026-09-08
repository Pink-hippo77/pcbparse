from __future__ import annotations

"""Production ``ComponentProvider`` adapter for PCBParse torch checkpoints.

Torch remains a delayed dependency: importing this module in the core runtime
does not import torch.  The production provider factory calls :func:`factory`
only after ``models.lock`` has verified the artifact bytes.
"""

import hashlib
import math
import threading
from pathlib import Path
from typing import Any, Mapping

from pcbparse.io import ImageView
from pcbparse.ir.models import BBox, Component, Evidence
from pcbparse.training.torch_component_model import (
    LoadedComponentCheckpoint,
    TorchComponentArtifactError,
    load_checkpoint,
)


class TorchComponentAdapter:
    """Run a frozen Faster R-CNN artifact in ``ImageView`` local coordinates."""

    def __init__(
        self,
        checkpoint: LoadedComponentCheckpoint,
        *,
        device: str,
        score_threshold: float,
        max_detections: int,
    ) -> None:
        self._checkpoint = checkpoint
        self._model = checkpoint.model
        self._device_name = device
        self._score_threshold = _threshold(score_threshold)
        if type(max_detections) is not int or max_detections <= 0:
            raise ValueError("max_detections must be a positive integer")
        self._max_detections = max_detections
        self._categories = {
            int(item["model_label"]): dict(item)
            for item in checkpoint.category_table
        }
        self._lock = threading.RLock()
        # TorchVision otherwise discards low-score candidates before the
        # adapter's artifact-configured threshold can be applied.  NMS and the
        # checkpoint's detections-per-image cap remain part of the model.
        roi_heads = getattr(self._model, "roi_heads", None)
        if roi_heads is None or not hasattr(roi_heads, "score_thresh"):
            raise TorchComponentArtifactError("loaded Component model lacks ROI score threshold")
        roi_heads.score_thresh = 0.0
        self._model.eval()
        self.version = (
            f"torch-component/{checkpoint.architecture}/"
            f"{checkpoint.checkpoint_sha256[:12]}"
        )

    @property
    def category_table(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._checkpoint.category_table)

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint.checkpoint_sha256

    def predict(self, image: ImageView, ctx: Any) -> list[Component]:
        """Return deterministic local ``xyxy`` detections for one image view."""

        if not isinstance(image, ImageView):
            raise TypeError("TorchComponentAdapter.predict expects an ImageView")
        _check_budget(ctx, "component_detector:before")
        torch = _require_torch()
        tensor = _image_tensor(image, torch).to(self._device_name)
        with self._lock, torch.inference_mode():
            raw_outputs = self._model([tensor])
        _check_budget(ctx, "component_detector:after")
        if not isinstance(raw_outputs, (list, tuple)) or len(raw_outputs) != 1:
            raise RuntimeError("Component detector returned invalid output cardinality")
        prediction = raw_outputs[0]
        if not isinstance(prediction, Mapping):
            raise RuntimeError("Component detector output must be a mapping")
        try:
            boxes = prediction["boxes"].detach().cpu().tolist()
            scores = prediction["scores"].detach().cpu().tolist()
            labels = prediction["labels"].detach().cpu().tolist()
        except (KeyError, AttributeError, TypeError) as exc:
            raise RuntimeError("Component detector lacks boxes/scores/labels tensors") from exc
        if not (len(boxes) == len(scores) == len(labels)):
            raise RuntimeError("Component detector boxes/scores/labels cardinality mismatch")

        candidates: list[tuple[float, int, tuple[float, float, float, float], int]] = []
        for source_index, (raw_box, raw_score, raw_label) in enumerate(
            zip(boxes, scores, labels)
        ):
            score = float(raw_score)
            label = int(raw_label)
            if not math.isfinite(score) or score < self._score_threshold:
                continue
            if label not in self._categories:
                # Background and any out-of-contract classifier output are
                # deliberately rejected rather than mapped to UNKNOWN.
                continue
            if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
                continue
            values = tuple(float(item) for item in raw_box)
            if any(not math.isfinite(item) for item in values):
                continue
            x1 = min(float(image.width), max(0.0, values[0]))
            y1 = min(float(image.height), max(0.0, values[1]))
            x2 = min(float(image.width), max(0.0, values[2]))
            y2 = min(float(image.height), max(0.0, values[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            candidates.append((score, label, (x1, y1, x2, y2), source_index))
        candidates.sort(
            key=lambda item: (
                -item[0],
                item[1],
                item[2][1],
                item[2][0],
                item[2][3],
                item[2][2],
                item[3],
            )
        )
        candidates = candidates[: self._max_detections]
        image_identity = image.sha256 or hashlib.sha256(
            image.image.convert("RGB").tobytes()
        ).hexdigest()
        output: list[Component] = []
        occurrence: dict[str, int] = {}
        for score, label, box, source_index in candidates:
            category = self._categories[label]
            signature = (
                f"{image_identity}|{image.view_id}|{label}|"
                + "|".join(f"{coordinate:.4f}" for coordinate in box)
            )
            digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:20]
            duplicate_index = occurrence.get(digest, 0)
            occurrence[digest] = duplicate_index + 1
            local_id = (
                f"component-local:{digest}"
                if duplicate_index == 0
                else f"component-local:{digest}:{duplicate_index}"
            )
            evidence_id = (
                f"detector:{self._checkpoint.checkpoint_sha256[:16]}:{digest}:"
                f"{duplicate_index}"
            )
            output.append(
                Component(
                    internal_id=local_id,
                    key=None,
                    category=str(category["name"]),
                    name="",
                    value="",
                    bbox=BBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                    score=max(0.0, min(1.0, score)),
                    key_score=0.0,
                    evidence=[
                        Evidence(
                            evidence_id=evidence_id,
                            source="detector",
                            view_id=image.view_id,
                            score=max(0.0, min(1.0, score)),
                            payload_ref=f"sha256:{self._checkpoint.checkpoint_sha256}",
                            metadata={
                                "architecture": self._checkpoint.architecture,
                                "official_category_id": int(category["official_id"]),
                                "model_label": label,
                                "coordinate_system": "image_view_local_xyxy",
                                "source_detection_index": source_index,
                                "checkpoint_epoch": self._checkpoint.epoch,
                            },
                        )
                    ],
                )
            )
        if ctx is not None:
            versions = getattr(ctx, "model_versions", None)
            if isinstance(versions, dict):
                versions["component"] = self.version
        return output


def factory(path: str | Path, entry: Mapping[str, Any]) -> TorchComponentAdapter:
    """Production-provider factory with hash, device, and threshold gates."""

    if not isinstance(entry, Mapping):
        raise TypeError("Component model-lock entry must be a mapping")
    task = entry.get("task")
    if task is not None and task != "component":
        raise ValueError(f"Torch Component adapter cannot load task {task!r}")
    torch = _require_torch()
    raw_device = _entry_value(entry, "device", default="cpu")
    if not isinstance(raw_device, str) or not raw_device.strip():
        raise ValueError("Component adapter device must be a non-empty string")
    device = raw_device.strip().lower()
    if device == "auto":
        device = "cuda:0" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
    try:
        resolved_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid Component adapter device {device!r}") from exc
    if resolved_device.type == "cuda" and not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
        raise RuntimeError("Component adapter requested CUDA but CUDA is unavailable")
    expected_sha = entry.get("sha256")
    if expected_sha is not None and not isinstance(expected_sha, str):
        raise ValueError("Component model-lock sha256 must be a string")
    checkpoint = load_checkpoint(
        path,
        device=resolved_device,
        expected_sha256=expected_sha,
        strict=True,
    )
    score_threshold = _entry_value(entry, "score_threshold", default=0.25)
    max_detections = _entry_value(entry, "max_detections", default=300)
    return TorchComponentAdapter(
        checkpoint,
        device=str(resolved_device),
        score_threshold=_threshold(score_threshold),
        max_detections=max_detections,
    )


def _entry_value(entry: Mapping[str, Any], name: str, *, default: Any) -> Any:
    inference = entry.get("inference")
    if inference is not None and not isinstance(inference, Mapping):
        raise ValueError("Component model-lock inference must be an object")
    nested = inference if isinstance(inference, Mapping) else {}
    if name in entry and name in nested and entry[name] != nested[name]:
        raise ValueError(
            f"Component model-lock inference.{name} conflicts with top-level {name}"
        )
    if name in entry:
        return entry[name]
    return nested.get(name, default)


def _image_tensor(image: ImageView, torch: Any) -> Any:
    rgb = image.image.convert("RGB")
    width, height = rgb.size
    data = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
    return (
        data.reshape(height, width, 3)
        .permute(2, 0, 1)
        .contiguous()
        .to(dtype=torch.float32)
        .div_(255.0)
    )


def _threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("score_threshold must be a number in [0,1]")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("score_threshold must be a finite number in [0,1]")
    return number


def _check_budget(ctx: Any, stage: str) -> None:
    if ctx is None:
        return
    budget = getattr(ctx, "time_budget", None)
    check = getattr(budget, "check", None)
    if callable(check):
        check(stage=stage)


def _require_torch() -> Any:
    try:
        import importlib

        return importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "PyTorch is required to construct the production Component adapter; "
            "use the project .venv-train runtime"
        ) from exc


__all__ = ["TorchComponentAdapter", "factory"]
