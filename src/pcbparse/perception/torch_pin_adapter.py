from __future__ import annotations

"""Production PinProvider adapter for exported PyTorch heatmap models.

The module is safe to import without torch or onnxruntime.  Optional runtimes
are resolved by :func:`create_pin_provider`, after the production model-lock
gate has already verified artifact bytes.
"""

import hashlib
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from pcbparse.io import ComponentView, ImageView
from pcbparse.ir.models import Evidence, Pin, Point
from pcbparse.training.torch_pin_model import (
    TorchPinDependencyError,
    decode_pin_heatmaps,
    load_pin_checkpoint,
    split_pin_outputs,
)


class TorchPinAdapterError(RuntimeError):
    """Raised when an artifact cannot safely implement the PinProvider contract."""


@dataclass(frozen=True, slots=True)
class PinAdapterConfig:
    input_size: tuple[int, int] = (512, 512)
    output_stride: int = 4
    score_threshold: float = 0.3
    nms_kernel: int = 3
    nms_radius_heatmap: float = 1.0
    max_detections_per_component: int = 256
    base_length_pixels: float = 8.0
    box_margin_px: float = 0.0
    box_prior_mode: str = "filled_rect_v1"
    pin_budget_mode: str = "fixed"
    pin_budget_floor: int = 12
    pin_budget_pitch_px: float = 8.0
    device: str = "auto"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.input_size, tuple)
            or len(self.input_size) != 2
            or any(type(value) is not int or value <= 0 for value in self.input_size)
        ):
            raise ValueError("input_size must be a positive (width, height) tuple")
        if self.output_stride != 4:
            raise ValueError("Pin adapter output_stride must be 4")
        if not _finite(self.score_threshold) or not 0 <= self.score_threshold <= 1:
            raise ValueError("score_threshold must be in [0, 1]")
        if type(self.nms_kernel) is not int or self.nms_kernel <= 0 or self.nms_kernel % 2 != 1:
            raise ValueError("nms_kernel must be a positive odd integer")
        if not _finite(self.nms_radius_heatmap) or self.nms_radius_heatmap < 0:
            raise ValueError("nms_radius_heatmap must be finite and non-negative")
        if type(self.max_detections_per_component) is not int or self.max_detections_per_component <= 0:
            raise ValueError("max_detections_per_component must be positive")
        if not _finite(self.base_length_pixels) or self.base_length_pixels <= 0:
            raise ValueError("base_length_pixels must be finite and positive")
        if not _finite(self.box_margin_px) or self.box_margin_px < 0:
            raise ValueError("box_margin_px must be finite and non-negative")
        if self.box_prior_mode not in {"filled_rect_v1", "boundary_band_v2"}:
            raise ValueError(
                "box_prior_mode must be filled_rect_v1 or boundary_band_v2"
            )
        if self.pin_budget_mode not in {"fixed", "edge_length_v1"}:
            raise ValueError("pin_budget_mode must be fixed or edge_length_v1")
        if type(self.pin_budget_floor) is not int or self.pin_budget_floor <= 0:
            raise ValueError("pin_budget_floor must be a positive integer")
        if self.pin_budget_floor > self.max_detections_per_component:
            raise ValueError("pin_budget_floor cannot exceed max_detections_per_component")
        if not _finite(self.pin_budget_pitch_px) or self.pin_budget_pitch_px <= 0:
            raise ValueError("pin_budget_pitch_px must be finite and positive")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string")

    def pin_budget_for(
        self, box_rect: tuple[float, float, float, float] | None
    ) -> int:
        """Visible-edge-length driven per-component decode budget.

        ``fixed`` reproduces the legacy hard cap exactly.  ``edge_length_v1``
        budgets one pin slot per ``pin_budget_pitch_px`` inference-canvas
        pixels of the component bounding-box perimeter (pins sit on that
        rectangle and the decoder NMS cannot resolve two pins closer than
        ``output_stride * nms_radius_heatmap`` pixels), clamped to
        ``[pin_budget_floor, max_detections_per_component]``.  Small parts
        therefore keep a tight budget while dense ICs are not silently
        truncated at the global cap.
        """

        if self.pin_budget_mode == "fixed":
            return self.max_detections_per_component
        if box_rect is None:
            return self.pin_budget_floor
        try:
            x1, y1, x2, y2 = (float(value) for value in box_rect)
        except (TypeError, ValueError):
            return self.pin_budget_floor
        perimeter = 2.0 * (abs(x2 - x1) + abs(y2 - y1))
        if not math.isfinite(perimeter) or perimeter <= 0:
            return self.pin_budget_floor
        budget = math.ceil(perimeter / self.pin_budget_pitch_px)
        return max(self.pin_budget_floor, min(self.max_detections_per_component, budget))

    @classmethod
    def from_entry(cls, entry: Mapping[str, Any]) -> "PinAdapterConfig":
        merged: dict[str, Any] = {}
        for key in ("inference", "adapter_config"):
            candidate = entry.get(key)
            if candidate is not None:
                if not isinstance(candidate, Mapping):
                    raise TorchPinAdapterError(f"model entry {key!r} must be an object")
                merged.update(candidate)
        direct_names = {
            "input_size",
            "output_stride",
            "score_threshold",
            "nms_kernel",
            "nms_radius_heatmap",
            "max_detections_per_component",
            "base_length_pixels",
            "box_margin_px",
            "box_prior_mode",
            "pin_budget_mode",
            "pin_budget_floor",
            "pin_budget_pitch_px",
            "device",
        }
        merged.update({key: entry[key] for key in direct_names if key in entry})
        if "input_size" in merged:
            raw = merged["input_size"]
            if type(raw) is int:
                merged["input_size"] = (raw, raw)
            elif isinstance(raw, (list, tuple)) and len(raw) == 2:
                merged["input_size"] = (raw[0], raw[1])
        try:
            return cls(**merged)
        except (TypeError, ValueError) as exc:
            raise TorchPinAdapterError(f"invalid Pin adapter configuration: {exc}") from exc


@dataclass(frozen=True, slots=True)
class _LetterboxGeometry:
    input_size: tuple[int, int]
    resized_size: tuple[int, int]
    padding: tuple[int, int, int, int]
    source_size: tuple[int, int]

    @property
    def scale_x(self) -> float:
        return self.resized_size[0] / self.source_size[0]

    @property
    def scale_y(self) -> float:
        return self.resized_size[1] / self.source_size[1]


# The model export has a dynamic batch dimension. Keep the micro-batch bounded
# so a high component count cannot turn one image into an unbounded GPU peak.
PIN_INFERENCE_MICRO_BATCH_SIZE = 16


class TorchPinProvider:
    """Decode one full ROI per component into local-coordinate Pin candidates."""

    def __init__(
        self,
        backend: Any,
        *,
        version: str,
        config: PinAdapterConfig | None = None,
    ) -> None:
        if not isinstance(version, str) or not version:
            raise ValueError("version must be non-empty")
        self._backend = backend
        self.version = version
        self.config = config or PinAdapterConfig()
        self._lock = threading.Lock()

    def predict(self, component_views: list[ComponentView], ctx: Any) -> list[Pin]:
        output: list[Pin] = []
        seen_components: set[str] = set()
        for component_view in component_views:
            if component_view.component_id in seen_components:
                raise TorchPinAdapterError(
                    f"duplicate ComponentView component_id {component_view.component_id!r}"
                )
            seen_components.add(component_view.component_id)

        for batch in _batches(component_views, PIN_INFERENCE_MICRO_BATCH_SIZE):
            tensors: list[Any] = []
            geometries: list[_LetterboxGeometry] = []
            box_rects: list[tuple[float, float, float, float] | None] = []
            for component_view in batch:
                _check_budget(ctx)
                inference_tensor, geometry = _view_tensor(
                    component_view.full_roi, self.config.input_size
                )
                tensors.append(inference_tensor)
                geometries.append(geometry)
                box_rects.append(_component_box_rect_heatmap(component_view, geometry))
            budgets = [self.config.pin_budget_for(rect) for rect in box_rects]

            torch = _require_torch()
            inference_batch = torch.cat(tensors, dim=0)
            with self._lock:
                raw_outputs = self._backend.run(inference_batch)
            stride = float(self.config.output_stride)
            predictions = decode_pin_heatmaps(
                raw_outputs,
                score_threshold=self.config.score_threshold,
                nms_kernel=self.config.nms_kernel,
                nms_radius=self.config.nms_radius_heatmap,
                max_detections=budgets,
                box_rects=[
                    tuple(value / stride for value in box_rect)
                    if box_rect is not None
                    else (0.0, 0.0, 0.0, 0.0)
                    for box_rect in box_rects
                ],
                box_margin=self.config.box_margin_px / stride,
                box_prior_mode=self.config.box_prior_mode,
            )
            if len(predictions) != len(batch):
                raise TorchPinAdapterError(
                    "batched Pin inference returned an unexpected batch cardinality"
                )
            for component_view, component_predictions, geometry in zip(
                batch,
                predictions,
                geometries,
            ):
                output.extend(
                    self._materialize_component_pins(
                        component_view,
                        component_predictions,
                        geometry=geometry,
                    )
                )
        if ctx is not None and isinstance(getattr(ctx, "model_versions", None), dict):
            ctx.model_versions["pin"] = self.version
        return output

    def _materialize_component_pins(
        self,
        component_view: ComponentView,
        detections: list[dict[str, Any]],
        *,
        geometry: _LetterboxGeometry,
    ) -> list[Pin]:
        view = component_view.full_roi
        inference_width, inference_height = geometry.input_size
        pad_left, pad_top, _, _ = geometry.padding
        resized_width, resized_height = geometry.resized_size
        result: list[Pin] = []
        for rank, detection in enumerate(detections):
            inference_x = float(detection["x"]) * self.config.output_stride
            inference_y = float(detection["y"]) * self.config.output_stride
            # Padding has no source-image counterpart.  Rejecting maxima there
            # matches the training visibility contract and avoids border pins
            # fabricated from the white letterbox canvas.
            if not (
                pad_left <= inference_x < pad_left + resized_width
                and pad_top <= inference_y < pad_top + resized_height
            ):
                continue
            point_x = min(
                max((inference_x - pad_left) / geometry.scale_x, 0.0),
                max(0.0, view.width - 1.0),
            )
            point_y = min(
                max((inference_y - pad_top) / geometry.scale_y, 0.0),
                max(0.0, view.height - 1.0),
            )
            raw_unit_x, raw_unit_y = detection["orientation"]
            local_x = float(raw_unit_x) / geometry.scale_x
            local_y = float(raw_unit_y) / geometry.scale_y
            magnitude = math.hypot(local_x, local_y)
            if magnitude > 1e-8:
                unit_x, unit_y = local_x / magnitude, local_y / magnitude
                base = Point(
                    x=point_x - unit_x * self.config.base_length_pixels,
                    y=point_y - unit_y * self.config.base_length_pixels,
                )
            else:
                unit_x, unit_y = 0.0, 0.0
                base = None
            score = min(1.0, max(0.0, float(detection["score"])))
            identity_material = (
                f"{self.version}\0{component_view.component_id}\0{view.view_id}\0"
                f"{rank}\0{point_x:.6f}\0{point_y:.6f}"
            ).encode("utf-8")
            identity = hashlib.sha256(identity_material).hexdigest()[:20]
            result.append(
                Pin(
                    internal_id=f"pin-model:{identity}",
                    component_id=component_view.component_id,
                    number=None,
                    name="",
                    point=Point(x=point_x, y=point_y),
                    base=base,
                    orientation=_orientation_label(unit_x, unit_y),
                    score=score,
                    evidence=[
                        Evidence(
                            evidence_id=f"pin-model-evidence:{identity}",
                            source="detector",
                            view_id=view.view_id,
                            score=score,
                            metadata={
                                "model_version": self.version,
                                "heatmap_anchor": list(detection["anchor"]),
                                "heatmap_point": [
                                    float(detection["x"]),
                                    float(detection["y"]),
                                ],
                                "orientation_unit": [unit_x, unit_y],
                                "inference_size": [inference_width, inference_height],
                                "letterbox_resized_size": [resized_width, resized_height],
                                "letterbox_padding": list(geometry.padding),
                                "semantic_fields_generated": False,
                            },
                        )
                    ],
                )
            )
        return result


class _TorchBackend:
    def __init__(self, model: Any, device: str) -> None:
        torch = _require_torch()
        self.device = _resolve_torch_device(device, torch)
        self.model = model.to(self.device).eval()
        self.torch = torch

    def run(self, image: Any) -> dict[str, Any]:
        with self.torch.inference_mode():
            value = self.model(image.to(self.device, dtype=self.torch.float32))
        return {
            key: tensor.detach().to("cpu")
            for key, tensor in split_pin_outputs(value).items()
        }


class _ONNXBackend:
    def __init__(self, path: Path, device: str) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise TorchPinDependencyError("ONNX Pin inference requires onnxruntime") from exc
        requested = device.lower()
        available = set(ort.get_available_providers())
        if requested in {"cuda", "gpu"}:
            if "CUDAExecutionProvider" not in available:
                raise TorchPinAdapterError("CUDA ONNX provider was requested but is unavailable")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif requested == "auto" and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif requested in {"auto", "cpu"}:
            providers = ["CPUExecutionProvider"]
        else:
            raise TorchPinAdapterError(f"unsupported ONNX device {device!r}")
        try:
            self.session = ort.InferenceSession(str(path), providers=providers)
        except Exception as exc:
            raise TorchPinAdapterError(
                f"cannot load ONNX Pin artifact: {type(exc).__name__}: {exc}"
            ) from exc

        # Verify the session actually loaded the requested providers.  ORT
        # silently falls back to CPU when a requested provider fails to
        # initialize (e.g., missing CUDA DLLs), so get_available_providers()
        # alone is not sufficient evidence of GPU execution.
        active_providers = set(self.session.get_providers())
        if requested in {"cuda", "gpu"} and "CUDAExecutionProvider" not in active_providers:
            raise TorchPinAdapterError(
                f"CUDA was explicitly requested but ONNX session only has "
                f"providers {sorted(active_providers)}; check CUDA/driver/DLL compatibility"
            )
        if requested == "auto" and "CUDAExecutionProvider" in available and "CUDAExecutionProvider" not in active_providers:
            # In auto mode, log a warning but don't fail — the user didn't demand GPU.
            import logging
            logging.getLogger("pcbparse.pin").warning(
                "ONNX Pin: CUDA was available but session fell back to %s; "
                "check CUDA/driver/DLL compatibility",
                sorted(active_providers),
            )

        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise TorchPinAdapterError("ONNX Pin artifact must expose exactly one input")
        self.input_name = inputs[0].name
        self.output_names = [item.name for item in self.session.get_outputs()]
        required = {"heatmap_logits", "offset", "orientation"}
        if set(self.output_names) != required:
            raise TorchPinAdapterError(
                f"ONNX Pin outputs must be {sorted(required)}, got {self.output_names}"
            )
        self.requested_device = requested
        self.active_providers = sorted(active_providers)

    def run(self, image: Any) -> dict[str, Any]:
        torch = _require_torch()
        arrays = self.session.run(
            self.output_names,
            {self.input_name: image.detach().cpu().numpy().astype(np.float32, copy=False)},
        )
        return {
            name: torch.from_numpy(value)
            for name, value in zip(self.output_names, arrays)
        }


def create_pin_provider(model_path: str | Path, entry: Mapping[str, Any]) -> TorchPinProvider:
    """Factory used by ``ProductionPerceptionProvider`` model-lock entries."""

    path = Path(model_path).expanduser().resolve(strict=True)
    if not isinstance(entry, Mapping):
        raise TorchPinAdapterError("model entry must be an object")
    if entry.get("task") not in (None, "pin"):
        raise TorchPinAdapterError("model entry task is not pin")
    expected_hash = entry.get("sha256")
    if expected_hash is not None:
        if not _is_sha256(expected_hash):
            raise TorchPinAdapterError("model entry sha256 is invalid")
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            raise TorchPinAdapterError("Pin artifact SHA-256 differs from model entry")
    config = PinAdapterConfig.from_entry(entry)
    artifact_format = str(
        entry.get("artifact_format")
        or entry.get("export_format")
        or entry.get("format")
        or ("onnx" if path.suffix.lower() == ".onnx" else "auto")
    ).lower()
    if artifact_format == "onnx":
        backend: Any = _ONNXBackend(path, config.device)
    elif artifact_format in {"torchscript", "jit", "auto"}:
        torch = _require_torch()
        device = _resolve_torch_device(config.device, torch)
        try:
            model = torch.jit.load(str(path), map_location=device)
            backend = _TorchBackend(model, str(device))
        except Exception as script_error:
            if artifact_format != "auto":
                raise TorchPinAdapterError(
                    f"cannot load TorchScript Pin artifact: {type(script_error).__name__}: {script_error}"
                ) from script_error
            try:
                model, _ = load_pin_checkpoint(path, map_location=device)
                backend = _TorchBackend(model, str(device))
            except Exception as checkpoint_error:
                raise TorchPinAdapterError(
                    "Pin artifact is neither valid TorchScript nor a valid Pin checkpoint: "
                    f"{type(checkpoint_error).__name__}: {checkpoint_error}"
                ) from checkpoint_error
    elif artifact_format in {"checkpoint", "pytorch"}:
        model, _ = load_pin_checkpoint(path, map_location="cpu")
        backend = _TorchBackend(model, config.device)
    else:
        raise TorchPinAdapterError(f"unsupported Pin artifact format {artifact_format!r}")
    version = str(
        entry.get("version")
        or entry.get("model_version")
        or f"pin-{_sha256_file(path)[:12]}"
    )
    return TorchPinProvider(backend, version=version, config=config)


# Explicit alias makes model-lock configuration readable while retaining the
# conventional ``create_*_provider`` factory name.
factory = create_pin_provider
TorchPinAdapter = TorchPinProvider


def _batches(items: list[ComponentView], size: int) -> list[list[ComponentView]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def _component_box_rect_heatmap(
    component_view: Any, geometry: _LetterboxGeometry
) -> tuple[float, float, float, float] | None:
    """Map the component bbox into inference-canvas pixels.

    Pins in schematic ROIs lie exactly on the component bounding-box
    rectangle, so this rectangle is a powerful geometric prior for pruning
    diffuse false-positive peaks.  Callers divide by the output stride to get
    heatmap units.
    """
    view = component_view.full_roi
    bbox = getattr(component_view, "component_bbox", None)
    if view is None or bbox is None:
        return None
    try:
        top_left = view.to_view(Point(x=bbox.x1, y=bbox.y1))
        bottom_right = view.to_view(Point(x=bbox.x2, y=bbox.y2))
        x1v, y1v = float(top_left.x), float(top_left.y)
        x2v, y2v = float(bottom_right.x), float(bottom_right.y)
    except (AttributeError, TypeError, ValueError):
        return None
    pad_left, pad_top, _, _ = geometry.padding
    return (
        pad_left + x1v * geometry.scale_x,
        pad_top + y1v * geometry.scale_y,
        pad_left + x2v * geometry.scale_x,
        pad_top + y2v * geometry.scale_y,
    )


def _view_tensor(
    view: ImageView, size: tuple[int, int]
) -> tuple[Any, _LetterboxGeometry]:
    torch = _require_torch()
    image = view.image.convert("RGB")
    output_width, output_height = size
    scale = min(output_width / image.width, output_height / image.height)
    resized_width = max(1, min(output_width, int(round(image.width * scale))))
    resized_height = max(1, min(output_height, int(round(image.height * scale))))
    if image.size != (resized_width, resized_height):
        image = image.resize(
            (resized_width, resized_height), resample=Image.Resampling.BILINEAR
        )
    pad_left = (output_width - resized_width) // 2
    pad_top = (output_height - resized_height) // 2
    pad_right = output_width - resized_width - pad_left
    pad_bottom = output_height - resized_height - pad_top
    canvas = Image.new("RGB", size, (255, 255, 255))
    canvas.paste(image, (pad_left, pad_top))
    array = np.asarray(canvas, dtype=np.float32).copy()
    if array.shape != (size[1], size[0], 3):
        raise TorchPinAdapterError(f"decoded Pin ROI has unexpected shape {array.shape}")
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().div_(255.0).unsqueeze(0)
    return tensor, _LetterboxGeometry(
        input_size=size,
        resized_size=(resized_width, resized_height),
        padding=(pad_left, pad_top, pad_right, pad_bottom),
        source_size=(view.width, view.height),
    )


def _resolve_torch_device(raw: str, torch: Any) -> Any:
    normalized = raw.lower()
    if normalized == "auto":
        return torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    if normalized in {"cuda", "gpu"}:
        if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
            raise TorchPinAdapterError("CUDA was requested for Pin inference but is unavailable")
        return torch.device("cuda")
    if normalized == "cpu" or normalized.startswith("cuda:"):
        device = torch.device(normalized)
        if device.type == "cuda" and not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
            raise TorchPinAdapterError("CUDA was requested for Pin inference but is unavailable")
        return device
    raise TorchPinAdapterError(f"unsupported torch device {raw!r}")


def _orientation_label(x: float, y: float) -> str:
    if math.hypot(x, y) <= 1e-8:
        return "UNKNOWN"
    if abs(x) >= abs(y):
        return "R" if x >= 0 else "L"
    return "B" if y >= 0 else "T"


def _check_budget(ctx: Any) -> None:
    budget = getattr(ctx, "time_budget", None) if ctx is not None else None
    check = getattr(budget, "check", None)
    if callable(check):
        check(stage="pin_provider")


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise TorchPinDependencyError(
            "TorchScript/checkpoint Pin inference requires the optional PyTorch runtime"
        ) from exc
    return torch


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


__all__ = [
    "PinAdapterConfig",
    "TorchPinAdapter",
    "TorchPinAdapterError",
    "TorchPinProvider",
    "create_pin_provider",
    "factory",
]
