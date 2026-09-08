from __future__ import annotations

"""Production ``JunctionProvider`` for the six-class Torch baseline.

Candidate coordinates are derived only from wire geometry.  Every patch is
owned by exactly one :class:`ImageView`; the returned local coordinate and its
single Evidence row let ``CompositePerceptionProvider`` restore it exactly
once.  Low-confidence decisions remain ``ambiguous``.
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
from pcbparse.ir.models import Evidence, Junction, Point, WireSegment
from pcbparse.perception.protocols import ProviderError
from pcbparse.training.torch_junction_model import (
    JunctionCheckpointError,
    TorchJunctionRuntimeError,
    load_junction_checkpoint,
)
from pcbparse.training.torch_wire_junction import JUNCTION_CLASSES


class TorchJunctionAdapterError(ProviderError):
    """The junction artifact, geometry, or inference violates its contract."""


@dataclass(frozen=True, slots=True)
class JunctionAdapterConfig:
    input_size: int = 64
    patch_size_original_px: float = 64.0
    score_threshold: float = 0.55
    candidate_merge_tolerance_px: float = 2.0
    segment_touch_tolerance_px: float = 2.5
    maximum_candidates: int = 16384
    batch_size: int = 128
    device: str = "auto"

    def __post_init__(self) -> None:
        if type(self.input_size) is not int or self.input_size < 24:
            raise ValueError("junction input_size must be an integer >= 24")
        for name in (
            "patch_size_original_px",
            "score_threshold",
            "candidate_merge_tolerance_px",
            "segment_touch_tolerance_px",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.patch_size_original_px <= 0:
            raise ValueError("patch_size_original_px must be positive")
        if not 0 <= self.score_threshold <= 1:
            raise ValueError("score_threshold must be in [0,1]")
        if self.candidate_merge_tolerance_px <= 0 or self.segment_touch_tolerance_px <= 0:
            raise ValueError("junction geometry tolerances must be positive")
        if type(self.maximum_candidates) is not int or self.maximum_candidates <= 0:
            raise ValueError("maximum_candidates must be positive")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")

    @classmethod
    def from_entry(cls, entry: Mapping[str, Any]) -> "JunctionAdapterConfig":
        merged: dict[str, Any] = {}
        for nested_name in ("inference", "adapter_config"):
            nested = entry.get(nested_name)
            if nested is not None:
                if not isinstance(nested, Mapping):
                    raise TorchJunctionAdapterError(f"model entry {nested_name!r} must be an object")
                merged.update(nested)
        names = {
            "input_size",
            "patch_size_original_px",
            "score_threshold",
            "candidate_merge_tolerance_px",
            "segment_touch_tolerance_px",
            "maximum_candidates",
            "batch_size",
            "device",
        }
        merged.update({name: entry[name] for name in names if name in entry})
        try:
            return cls(**merged)
        except (TypeError, ValueError) as exc:
            raise TorchJunctionAdapterError(f"invalid junction adapter configuration: {exc}") from exc


@dataclass(frozen=True, slots=True)
class JunctionPatch:
    original_point: Point
    local_point: Point
    view: ImageView
    segment_ids: tuple[str, ...]
    tensor: np.ndarray
    patch_box_local: tuple[float, float, float, float]


class TorchJunctionProvider:
    """Hash-bound TorchScript/checkpoint/ONNX six-class provider."""

    def __init__(self, artifact_path: str | Path, entry: Mapping[str, Any] | None = None) -> None:
        self.artifact_path = Path(artifact_path).expanduser().resolve()
        if not self.artifact_path.is_file() or self.artifact_path.stat().st_size <= 0:
            raise TorchJunctionAdapterError(f"junction artifact is missing or empty: {self.artifact_path}")
        self.entry = dict(entry or {})
        task = self.entry.get("task")
        if task is not None and task != "junction":
            raise TorchJunctionAdapterError(f"junction adapter received task {task!r}")
        self.artifact_sha256 = _sha256_file(self.artifact_path)
        expected_sha = self.entry.get("sha256")
        if expected_sha is not None and expected_sha != self.artifact_sha256:
            raise TorchJunctionAdapterError("junction artifact SHA-256 differs from model entry")
        declared_classes = self.entry.get("class_names")
        if declared_classes is not None and tuple(declared_classes) != JUNCTION_CLASSES:
            raise TorchJunctionAdapterError("junction model entry class order is incompatible")
        self.config = JunctionAdapterConfig.from_entry(self.entry)
        self._backend_kind, self._backend, self.device = self._load_backend(self.config.device)
        declared_version = self.entry.get("version")
        self.version = (
            str(declared_version)
            if isinstance(declared_version, str) and declared_version.strip()
            else f"tiny-junction-{self.artifact_sha256[:16]}"
        )
        self._lock = threading.RLock()

    def classify(
        self,
        views: list[ImageView],
        segments: list[WireSegment],
        ctx: Any = None,
    ) -> list[Junction]:
        _check_budget(ctx)
        if not views or not segments:
            if ctx is not None and isinstance(getattr(ctx, "model_versions", None), dict):
                ctx.model_versions["junction"] = self.version
            return []
        if any(not isinstance(view, ImageView) for view in views):
            raise TorchJunctionAdapterError("junction views must be ImageView objects")
        if any(not isinstance(segment, WireSegment) for segment in segments):
            raise TorchJunctionAdapterError("junction segments must be WireSegment objects")
        patches = build_junction_patches(views, segments, config=self.config)
        if len(patches) > self.config.maximum_candidates:
            raise TorchJunctionAdapterError(
                f"junction candidate count {len(patches)} exceeds maximum {self.config.maximum_candidates}"
            )
        output: list[Junction] = []
        with self._lock:
            for offset in range(0, len(patches), self.config.batch_size):
                _check_budget(ctx)
                chunk = patches[offset : offset + self.config.batch_size]
                inputs = np.stack([item.tensor for item in chunk], axis=0)
                logits = self._infer(inputs)
                if logits.shape != (len(chunk), len(JUNCTION_CLASSES)) or not np.isfinite(logits).all():
                    raise TorchJunctionAdapterError("junction model must return finite N,6 logits")
                probabilities = _softmax(logits)
                for patch, probability in zip(chunk, probabilities):
                    label_id = int(np.argmax(probability))
                    score = float(probability[label_id])
                    raw_kind = JUNCTION_CLASSES[label_id]
                    kind = raw_kind if score >= self.config.score_threshold else "ambiguous"
                    identity = _candidate_identity(
                        self.version,
                        self.artifact_sha256,
                        patch.original_point,
                        patch.segment_ids,
                    )
                    evidence = Evidence(
                        evidence_id=f"junction-model:{identity}",
                        source="detector",
                        view_id=patch.view.view_id,
                        score=max(0.0, min(1.0, score)),
                        payload_ref=str(self.artifact_path),
                        metadata={
                            "task": "junction",
                            "model_version": self.version,
                            "artifact_sha256": self.artifact_sha256,
                            "runtime": self._backend_kind,
                            "device": self.device,
                            "coordinate_space": "image_view_local",
                            "original_candidate_point": [patch.original_point.x, patch.original_point.y],
                            "local_candidate_point": [patch.local_point.x, patch.local_point.y],
                            "patch_box_local": list(patch.patch_box_local),
                            "patch_input_size": self.config.input_size,
                            "raw_class": raw_kind,
                            "raw_class_id": label_id,
                            "score_threshold": self.config.score_threshold,
                            "probabilities": {
                                name: float(probability[index])
                                for index, name in enumerate(JUNCTION_CLASSES)
                            },
                            "unique_source_view": True,
                            "segment_ids": list(patch.segment_ids),
                        },
                    )
                    output.append(
                        Junction(
                            internal_id=f"junction-model:{identity}",
                            point=patch.local_point,
                            kind=kind,
                            segment_ids=list(patch.segment_ids),
                            score=max(0.0, min(1.0, score)),
                            evidence=[evidence],
                        )
                    )
        if ctx is not None and isinstance(getattr(ctx, "model_versions", None), dict):
            ctx.model_versions["junction"] = self.version
        return sorted(
            output,
            key=lambda item: (
                item.evidence[0].metadata["original_candidate_point"][1],
                item.evidence[0].metadata["original_candidate_point"][0],
                item.internal_id,
            ),
        )

    def _load_backend(self, requested_device: str) -> tuple[str, Any, str]:
        artifact_format = str(_option(self.entry, "artifact_format", "export_format", "format", default="")).strip().lower()
        if artifact_format in {"jit", "torch", "ts"}:
            artifact_format = "torchscript"
        if not artifact_format:
            if self.artifact_path.suffix.lower() == ".onnx":
                artifact_format = "onnx"
            elif self.artifact_path.suffix.lower() in {".ckpt", ".checkpoint", ".pth"}:
                artifact_format = "checkpoint"
            else:
                artifact_format = "torchscript_or_checkpoint"
        if artifact_format == "onnx":
            return self._load_onnx(requested_device)
        if artifact_format not in {"torchscript", "checkpoint", "torchscript_or_checkpoint"}:
            raise TorchJunctionAdapterError(f"unsupported junction artifact format {artifact_format!r}")
        torch = _require_torch()
        device = _torch_device(torch, requested_device)
        if artifact_format in {"torchscript", "torchscript_or_checkpoint"}:
            try:
                model = torch.jit.load(str(self.artifact_path), map_location=device)
            except Exception as script_error:
                if artifact_format == "torchscript":
                    raise TorchJunctionAdapterError(f"cannot load junction TorchScript: {type(script_error).__name__}: {script_error}") from script_error
            else:
                model.eval()
                return "torchscript", model, str(device)
        expected_lineage = self.entry.get("lineage")
        try:
            loaded = load_junction_checkpoint(
                self.artifact_path,
                map_location=device,
                expected_lineage=expected_lineage if isinstance(expected_lineage, Mapping) else None,
            )
        except (JunctionCheckpointError, TorchJunctionRuntimeError) as exc:
            raise TorchJunctionAdapterError(str(exc)) from exc
        loaded.model.to(device)
        loaded.model.eval()
        return "torch-checkpoint", loaded.model, str(device)

    def _load_onnx(self, requested_device: str) -> tuple[str, Any, str]:
        try:
            runtime = importlib.import_module("onnxruntime")
        except ImportError as exc:
            raise TorchJunctionAdapterError("onnxruntime is required for junction ONNX") from exc
        available = set(runtime.get_available_providers())
        request = requested_device.strip().lower()
        if request == "auto":
            request = "cuda" if "CUDAExecutionProvider" in available else "cpu"
        if request.startswith("cuda"):
            if "CUDAExecutionProvider" not in available:
                raise TorchJunctionAdapterError("junction ONNX requests unavailable CUDA provider")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif request == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            raise TorchJunctionAdapterError("junction ONNX device must be auto/cpu/cuda")
        try:
            session = runtime.InferenceSession(str(self.artifact_path), providers=providers)
        except Exception as exc:
            raise TorchJunctionAdapterError(f"cannot load junction ONNX: {type(exc).__name__}: {exc}") from exc
        if len(session.get_inputs()) != 1 or not session.get_outputs():
            raise TorchJunctionAdapterError("junction ONNX must expose one input and one or more outputs")
        return "onnxruntime", session, request

    def _infer(self, inputs: np.ndarray) -> np.ndarray:
        if self._backend_kind == "onnxruntime":
            try:
                result = self._backend.run(None, {self._backend.get_inputs()[0].name: inputs})[0]
            except Exception as exc:
                raise TorchJunctionAdapterError(f"junction ONNX inference failed: {type(exc).__name__}: {exc}") from exc
            return np.asarray(result, dtype=np.float32)
        torch = _require_torch()
        tensor = torch.from_numpy(inputs).to(self.device)
        try:
            with torch.inference_mode():
                result = self._backend(tensor)
        except Exception as exc:
            raise TorchJunctionAdapterError(f"junction Torch inference failed: {type(exc).__name__}: {exc}") from exc
        if isinstance(result, (tuple, list)) and result:
            result = result[0]
        if isinstance(result, Mapping):
            result = result.get("logits")
        if not torch.is_tensor(result):
            raise TorchJunctionAdapterError("junction model did not return logits tensor")
        return result.detach().to("cpu").float().numpy()


def build_junction_patches(
    views: Sequence[ImageView],
    segments: Sequence[WireSegment],
    *,
    config: JunctionAdapterConfig | None = None,
) -> list[JunctionPatch]:
    """Create one uniquely-owned raster patch per stable geometry candidate."""

    policy = config or JunctionAdapterConfig()
    view_ids = [view.view_id for view in views]
    if len(set(view_ids)) != len(view_ids):
        raise TorchJunctionAdapterError("junction views contain duplicate view_id values")
    segment_ids = [segment.internal_id for segment in segments]
    if len(set(segment_ids)) != len(segment_ids):
        raise TorchJunctionAdapterError("junction segments contain duplicate internal_id values")
    points: list[Point] = []
    for segment in segments:
        points.extend((segment.start, segment.end))
    points.extend(_segment_intersections(segments, cell_size=max(32.0, policy.patch_size_original_px)))
    candidates = _merge_points(points, policy.candidate_merge_tolerance_px)
    patches: list[JunctionPatch] = []
    for original in candidates:
        touching = tuple(
            sorted(
                segment.internal_id
                for segment in segments
                if _point_segment_distance(original, segment) <= policy.segment_touch_tolerance_px
            )
        )
        if not touching:
            continue
        owner = _select_view_owner(views, original)
        if owner is None:
            continue
        local = owner.to_view(original)
        half = policy.patch_size_original_px / 2.0
        # The configured extent is in canonical/original pixels.  Mapping all
        # four corners keeps scaled or rotated ImageViews honest; treating the
        # number as local pixels would silently change model context.
        mapped_corners = [
            owner.to_view(Point(x=original.x + dx, y=original.y + dy))
            for dx in (-half, half)
            for dy in (-half, half)
        ]
        box = (
            min(point.x for point in mapped_corners),
            min(point.y for point in mapped_corners),
            max(point.x for point in mapped_corners),
            max(point.y for point in mapped_corners),
        )
        tensor = _extract_tensor(owner, box, policy.input_size)
        patches.append(
            JunctionPatch(
                original_point=original,
                local_point=local,
                view=owner,
                segment_ids=touching,
                tensor=tensor,
                patch_box_local=box,
            )
        )
    return sorted(patches, key=lambda item: (item.original_point.y, item.original_point.x, item.segment_ids))


def _segment_intersections(segments: Sequence[WireSegment], *, cell_size: float) -> list[Point]:
    cells: dict[tuple[int, int], list[int]] = {}
    for index, segment in enumerate(segments):
        minimum_x, maximum_x = sorted((segment.start.x, segment.end.x))
        minimum_y, maximum_y = sorted((segment.start.y, segment.end.y))
        for cell_x in range(math.floor(minimum_x / cell_size), math.floor(maximum_x / cell_size) + 1):
            for cell_y in range(math.floor(minimum_y / cell_size), math.floor(maximum_y / cell_size) + 1):
                cells.setdefault((cell_x, cell_y), []).append(index)
    pairs: set[tuple[int, int]] = set()
    for members in cells.values():
        ordered = sorted(set(members))
        for position, left in enumerate(ordered):
            for right in ordered[position + 1 :]:
                pairs.add((left, right))
    output: list[Point] = []
    for left, right in sorted(pairs):
        intersection = _proper_intersection(segments[left], segments[right])
        if intersection is not None:
            output.append(intersection)
    return output


def _proper_intersection(left: WireSegment, right: WireSegment) -> Point | None:
    p = np.array((left.start.x, left.start.y), dtype=np.float64)
    r = np.array((left.end.x - left.start.x, left.end.y - left.start.y), dtype=np.float64)
    q = np.array((right.start.x, right.start.y), dtype=np.float64)
    s = np.array((right.end.x - right.start.x, right.end.y - right.start.y), dtype=np.float64)
    cross_rs = float(r[0] * s[1] - r[1] * s[0])
    if abs(cross_rs) <= 1.0e-9:
        return None
    delta = q - p
    t = float((delta[0] * s[1] - delta[1] * s[0]) / cross_rs)
    u = float((delta[0] * r[1] - delta[1] * r[0]) / cross_rs)
    if -1.0e-9 <= t <= 1.0 + 1.0e-9 and -1.0e-9 <= u <= 1.0 + 1.0e-9:
        value = p + t * r
        return Point(x=float(value[0]), y=float(value[1]))
    return None


def _merge_points(points: Sequence[Point], tolerance: float) -> list[Point]:
    # Deterministic single-link grouping with grid neighbours.  The arithmetic
    # mean eliminates segment-order bias while stable sorting fixes IDs.
    ordered = sorted(points, key=lambda item: (item.y, item.x))
    clusters: list[list[Point]] = []
    centers: list[Point] = []
    for point in ordered:
        selected: int | None = None
        selected_distance = math.inf
        for index, center in enumerate(centers):
            distance = math.hypot(point.x - center.x, point.y - center.y)
            if distance <= tolerance and distance < selected_distance:
                selected = index
                selected_distance = distance
        if selected is None:
            clusters.append([point])
            centers.append(point)
        else:
            clusters[selected].append(point)
            values = clusters[selected]
            centers[selected] = Point(
                x=sum(item.x for item in values) / len(values),
                y=sum(item.y for item in values) / len(values),
            )
    return sorted(centers, key=lambda item: (item.y, item.x))


def _point_segment_distance(point: Point, segment: WireSegment) -> float:
    dx = segment.end.x - segment.start.x
    dy = segment.end.y - segment.start.y
    length_squared = dx * dx + dy * dy
    if length_squared <= 1.0e-12:
        return math.hypot(point.x - segment.start.x, point.y - segment.start.y)
    fraction = ((point.x - segment.start.x) * dx + (point.y - segment.start.y) * dy) / length_squared
    fraction = max(0.0, min(1.0, fraction))
    nearest_x = segment.start.x + fraction * dx
    nearest_y = segment.start.y + fraction * dy
    return math.hypot(point.x - nearest_x, point.y - nearest_y)


def _select_view_owner(views: Sequence[ImageView], point: Point) -> ImageView | None:
    candidates: list[tuple[float, str, ImageView]] = []
    for view in views:
        local = view.to_view(point)
        if 0 <= local.x < view.width and 0 <= local.y < view.height:
            margin = min(local.x, local.y, view.width - 1.0 - local.x, view.height - 1.0 - local.y)
            candidates.append((-margin, view.view_id, view))
    return min(candidates, default=(0.0, "", None))[2]


def _extract_tensor(view: ImageView, box: tuple[float, float, float, float], output_size: int) -> np.ndarray:
    left, top, right, bottom = box
    width = max(1, int(math.ceil(right - left)))
    height = max(1, int(math.ceil(bottom - top)))
    canvas = Image.new("RGB", (width, height), "white")
    source_left = max(0, math.floor(left))
    source_top = max(0, math.floor(top))
    source_right = min(view.width, math.ceil(right))
    source_bottom = min(view.height, math.ceil(bottom))
    if source_right > source_left and source_bottom > source_top:
        crop = view.image.convert("RGB").crop((source_left, source_top, source_right, source_bottom))
        destination = (int(round(source_left - left)), int(round(source_top - top)))
        canvas.paste(crop, destination)
    canvas = canvas.resize((output_size, output_size), Image.Resampling.LANCZOS)
    values = np.asarray(canvas, dtype=np.float32).copy() / 255.0
    return np.ascontiguousarray(np.transpose(values, (2, 0, 1)), dtype=np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=1, keepdims=True)


def _candidate_identity(version: str, artifact_sha: str, point: Point, segment_ids: Sequence[str]) -> str:
    material = f"{version}\0{artifact_sha}\0{point.x:.6f}\0{point.y:.6f}\0{'|'.join(segment_ids)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _option(entry: Mapping[str, Any], *names: str, default: Any) -> Any:
    nested = entry.get("inference")
    inference = nested if isinstance(nested, Mapping) else {}
    for name in names:
        if name in entry:
            return entry[name]
        if name in inference:
            return inference[name]
    return default


def _torch_device(torch: Any, requested: str) -> Any:
    value = requested.strip().lower()
    if value == "auto":
        value = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise TorchJunctionAdapterError(f"invalid junction device {requested!r}") from exc
    if device.type == "cuda" and not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
        raise TorchJunctionAdapterError("junction entry requests unavailable CUDA")
    if device.type not in {"cpu", "cuda"}:
        raise TorchJunctionAdapterError("junction device must be CPU or CUDA")
    return device


def _check_budget(ctx: Any) -> None:
    checker = getattr(ctx, "check_budget", None) if ctx is not None else None
    if callable(checker):
        checker()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise TorchJunctionAdapterError("PyTorch is required for junction Torch artifacts") from exc


def create_junction_provider(model_path: str | Path, entry: Mapping[str, Any]) -> TorchJunctionProvider:
    return TorchJunctionProvider(model_path, entry)


def load(model_path: str | Path, entry: Mapping[str, Any]) -> TorchJunctionProvider:
    return create_junction_provider(model_path, entry)


def factory(model_path: str | Path, entry: Mapping[str, Any]) -> TorchJunctionProvider:
    return create_junction_provider(model_path, entry)


# Model registry convention also accepts the shorter Adapter class name.
TorchJunctionAdapter = TorchJunctionProvider


__all__ = [
    "JunctionAdapterConfig",
    "JunctionPatch",
    "TorchJunctionAdapter",
    "TorchJunctionAdapterError",
    "TorchJunctionProvider",
    "build_junction_patches",
    "create_junction_provider",
    "factory",
    "load",
]
