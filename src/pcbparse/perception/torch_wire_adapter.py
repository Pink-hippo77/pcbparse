from __future__ import annotations

"""Production WireProvider adapter for Torch checkpoints and exported models.

No torch/onnx runtime is imported until a provider is constructed.  Segmentation
stays in each :class:`ImageView`'s local coordinates; the composite provider is
the single owner of coordinate restoration and vectorisation.
"""

import hashlib
import importlib
import math
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pcbparse.io import ImageView
from pcbparse.ir.models import Evidence
from pcbparse.perception.protocols import ProviderError, WireMask
from pcbparse.training.torch_wire_model import (
    TorchWireRuntimeError,
    WireCheckpointError,
    load_wire_checkpoint,
)


class TorchWireAdapterError(ProviderError):
    """A wire artifact or inference result violates the provider contract."""


class TorchWireAdapter:
    """Batched, thread-safe learned wire-mask provider."""

    def __init__(
        self,
        artifact_path: str | Path,
        entry: Mapping[str, Any] | None = None,
        *,
        threshold: float | None = None,
        batch_size: int | None = None,
        device: str | None = None,
    ) -> None:
        self.artifact_path = Path(artifact_path).expanduser().resolve()
        if not self.artifact_path.is_file() or self.artifact_path.stat().st_size <= 0:
            raise TorchWireAdapterError(
                f"wire model artifact is missing or empty: {self.artifact_path}"
            )
        self.entry = dict(entry or {})
        task = self.entry.get("task")
        if task is not None and task != "wire":
            raise TorchWireAdapterError(
                f"wire adapter received an artifact declared for task {task!r}"
            )
        self.artifact_sha256 = _sha256_file(self.artifact_path)
        expected_sha = self.entry.get("sha256")
        if expected_sha is not None and expected_sha != self.artifact_sha256:
            raise TorchWireAdapterError(
                "wire artifact SHA-256 differs from the verified model entry"
            )
        self.threshold = _finite_probability(
            threshold
            if threshold is not None
            else _option(self.entry, "threshold", "wire_threshold", default=0.5),
            "wire threshold",
        )
        selected_batch_size = (
            batch_size
            if batch_size is not None
            else _option(self.entry, "batch_size", "inference_batch_size", default=4)
        )
        if type(selected_batch_size) is not int or selected_batch_size <= 0:
            raise TorchWireAdapterError("wire inference batch_size must be positive")
        self.batch_size = selected_batch_size
        self.output_kind = str(
            _option(self.entry, "output_kind", default="logits")
        ).strip().lower()
        if self.output_kind not in {"logits", "probabilities"}:
            raise TorchWireAdapterError(
                "wire output_kind must be 'logits' or 'probabilities'"
            )
        self.mean = _normalization_triplet(
            _option(self.entry, "mean", default=(0.0, 0.0, 0.0)), "mean"
        )
        self.std = _normalization_triplet(
            _option(self.entry, "std", default=(1.0, 1.0, 1.0)), "std"
        )
        if any(value <= 0 for value in self.std):
            raise TorchWireAdapterError("wire normalization std must be positive")
        requested_device = device or str(_option(self.entry, "device", default="auto"))
        self._backend_kind, self._backend, self.device = self._load_backend(
            requested_device
        )
        declared_version = self.entry.get("version")
        self.version = (
            str(declared_version)
            if isinstance(declared_version, str) and declared_version.strip()
            else f"tiny-unet-{self.artifact_sha256[:16]}"
        )
        self._lock = threading.RLock()

    def segment(self, views: list[ImageView], ctx: Any = None) -> list[WireMask]:
        """Return one local-coordinate binary mask for every input view."""

        _check_budget(ctx)
        if not views:
            return []
        if any(not isinstance(view, ImageView) for view in views):
            raise TorchWireAdapterError("wire segment inputs must be ImageView objects")

        # Same-shape grouping preserves true batching without padding pixels that
        # would alter border predictions.  Results are restored to caller order.
        groups: dict[tuple[int, int], list[tuple[int, ImageView]]] = {}
        for index, view in enumerate(views):
            if view.width <= 0 or view.height <= 0:
                raise TorchWireAdapterError(f"wire view {view.view_id!r} is empty")
            groups.setdefault((view.height, view.width), []).append((index, view))
        output: list[WireMask | None] = [None] * len(views)
        with self._lock:
            for group in groups.values():
                for offset in range(0, len(group), self.batch_size):
                    _check_budget(ctx)
                    chunk = group[offset : offset + self.batch_size]
                    inputs = np.stack(
                        [self._image_tensor(view) for _, view in chunk], axis=0
                    )
                    probabilities = self._infer(inputs)
                    if probabilities.shape != (
                        len(chunk),
                        1,
                        chunk[0][1].height,
                        chunk[0][1].width,
                    ):
                        raise TorchWireAdapterError(
                            "wire model output shape does not match the input view batch"
                        )
                    if not np.isfinite(probabilities).all():
                        raise TorchWireAdapterError(
                            "wire model returned non-finite probabilities"
                        )
                    probabilities = np.clip(probabilities, 0.0, 1.0)
                    for batch_index, (input_index, view) in enumerate(chunk):
                        probability = probabilities[batch_index, 0]
                        binary = np.ascontiguousarray(
                            probability >= self.threshold, dtype=np.bool_
                        )
                        confidence = float(
                            np.mean(np.maximum(probability, 1.0 - probability))
                        )
                        evidence_id = _evidence_id(
                            self.version,
                            self.artifact_sha256,
                            view,
                            self.threshold,
                        )
                        evidence = Evidence(
                            evidence_id=evidence_id,
                            source="detector",
                            view_id=view.view_id,
                            score=max(0.0, min(1.0, confidence)),
                            payload_ref=str(self.artifact_path),
                            metadata={
                                "task": "wire",
                                "model_version": self.version,
                                "artifact_sha256": self.artifact_sha256,
                                "runtime": self._backend_kind,
                                "device": self.device,
                                "threshold": self.threshold,
                                "output_kind": self.output_kind,
                                "local_mask_shape": [view.height, view.width],
                                "coordinate_space": "image_view_local",
                                "vectorized": False,
                            },
                        )
                        output[input_index] = WireMask(
                            view_id=view.view_id,
                            mask=binary,
                            score=max(0.0, min(1.0, confidence)),
                            evidence=[evidence],
                        )
        if any(item is None for item in output):
            raise TorchWireAdapterError("wire adapter failed to emit every input view")
        return [item for item in output if item is not None]

    def _load_backend(self, requested_device: str) -> tuple[str, Any, str]:
        artifact_format = str(
            _option(self.entry, "artifact_format", "export_format", "format", default="")
        ).strip().lower()
        if artifact_format in {"jit", "torch", "ts"}:
            artifact_format = "torchscript"
        suffix = self.artifact_path.suffix.lower()
        if not artifact_format:
            if suffix == ".onnx":
                artifact_format = "onnx"
            elif suffix in {".ckpt", ".checkpoint", ".pth"}:
                artifact_format = "checkpoint"
            else:
                artifact_format = "torchscript_or_checkpoint"
        if artifact_format == "onnx":
            return self._load_onnx(requested_device)
        if artifact_format not in {
            "torchscript",
            "checkpoint",
            "torchscript_or_checkpoint",
        }:
            raise TorchWireAdapterError(
                f"unsupported wire artifact format {artifact_format!r}"
            )
        torch = _require_torch()
        selected_device = _torch_device(torch, requested_device)
        if artifact_format in {"torchscript", "torchscript_or_checkpoint"}:
            try:
                # Windows user and temporary directories may both contain
                # Unicode characters. A Python stream avoids the native
                # TorchScript loader's filename handling entirely.
                with self.artifact_path.open("rb") as stream:
                    module = torch.jit.load(stream, map_location=selected_device)
                module.eval()
                return "torchscript", module, str(selected_device)
            except OSError as exc:
                raise TorchWireAdapterError(
                    f"cannot read wire TorchScript artifact: {type(exc).__name__}: {exc}"
                ) from exc
            except Exception as script_error:
                if artifact_format == "torchscript":
                    raise TorchWireAdapterError(
                        f"cannot load wire TorchScript artifact: {type(script_error).__name__}: {script_error}"
                    ) from script_error
        expected_lineage = self.entry.get("lineage")
        try:
            loaded = load_wire_checkpoint(
                self.artifact_path,
                map_location=selected_device,
                expected_lineage=(
                    expected_lineage if isinstance(expected_lineage, Mapping) else None
                ),
            )
        except (WireCheckpointError, TorchWireRuntimeError) as exc:
            raise TorchWireAdapterError(str(exc)) from exc
        loaded.model.to(selected_device)
        loaded.model.eval()
        return "torch-checkpoint", loaded.model, str(selected_device)

    def _load_onnx(self, requested_device: str) -> tuple[str, Any, str]:
        try:
            runtime = importlib.import_module("onnxruntime")
        except ImportError as exc:
            raise TorchWireAdapterError(
                "onnxruntime is required for the declared wire ONNX artifact"
            ) from exc
        available = set(runtime.get_available_providers())
        request = requested_device.strip().lower()
        if request == "auto":
            request = "cuda" if "CUDAExecutionProvider" in available else "cpu"
        if request.startswith("cuda"):
            if "CUDAExecutionProvider" not in available:
                raise TorchWireAdapterError(
                    "wire ONNX entry requests CUDA but CUDAExecutionProvider is unavailable"
                )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            selected = request
        elif request == "cpu":
            providers = ["CPUExecutionProvider"]
            selected = "cpu"
        else:
            raise TorchWireAdapterError(
                "wire ONNX device must be 'auto', 'cpu', or 'cuda[:index]'"
            )
        try:
            session = runtime.InferenceSession(
                str(self.artifact_path), providers=providers
            )
        except Exception as exc:
            raise TorchWireAdapterError(
                f"cannot load wire ONNX artifact: {type(exc).__name__}: {exc}"
            ) from exc
        if len(session.get_inputs()) != 1 or not session.get_outputs():
            raise TorchWireAdapterError(
                "wire ONNX model must expose one image input and at least one output"
            )
        return "onnxruntime", session, selected

    def _image_tensor(self, view: ImageView) -> np.ndarray:
        image = np.asarray(view.image.convert("RGB"), dtype=np.float32).copy()
        image = np.transpose(image / 255.0, (2, 0, 1))
        mean = np.asarray(self.mean, dtype=np.float32)[:, None, None]
        std = np.asarray(self.std, dtype=np.float32)[:, None, None]
        return np.ascontiguousarray((image - mean) / std, dtype=np.float32)

    def _infer(self, inputs: np.ndarray) -> np.ndarray:
        if self._backend_kind == "onnxruntime":
            input_name = self._backend.get_inputs()[0].name
            try:
                result = self._backend.run(None, {input_name: inputs})[0]
            except Exception as exc:
                raise TorchWireAdapterError(
                    f"wire ONNX inference failed: {type(exc).__name__}: {exc}"
                ) from exc
            values = np.asarray(result, dtype=np.float32)
        else:
            torch = _require_torch()
            tensor = torch.from_numpy(inputs).to(self.device)
            try:
                with torch.inference_mode():
                    output = self._backend(tensor)
            except Exception as exc:
                raise TorchWireAdapterError(
                    f"wire PyTorch inference failed: {type(exc).__name__}: {exc}"
                ) from exc
            output = _extract_tensor(output, torch)
            values = output.detach().to("cpu").float().numpy()
        if values.ndim == 3:
            values = values[:, None]
        if values.ndim != 4 or values.shape[1] != 1:
            raise TorchWireAdapterError("wire model output must have N,1,H,W shape")
        if self.output_kind == "logits":
            # Stable sigmoid avoids overflow warnings on large negative logits.
            positive = values >= 0
            probabilities = np.empty_like(values, dtype=np.float32)
            probabilities[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
            exponential = np.exp(values[~positive])
            probabilities[~positive] = exponential / (1.0 + exponential)
            return probabilities
        return values


def create_wire_provider(
    model_path: str | Path, entry: Mapping[str, Any]
) -> TorchWireAdapter:
    """Factory used by ``ProductionPerceptionProvider`` model-lock entries."""

    return TorchWireAdapter(model_path, entry)


# Concise alias for model-card/registry tooling which conventionally uses load().
def load(model_path: str | Path, entry: Mapping[str, Any]) -> TorchWireAdapter:
    return create_wire_provider(model_path, entry)


def factory(model_path: str | Path, entry: Mapping[str, Any]) -> TorchWireAdapter:
    """Registry-friendly alias shared with the other learned adapters."""

    return create_wire_provider(model_path, entry)


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise TorchWireAdapterError(
            "PyTorch is required for the declared wire checkpoint/TorchScript artifact"
        ) from exc


def _torch_device(torch: Any, requested: str) -> Any:
    value = requested.strip().lower()
    if value == "auto":
        value = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
    try:
        selected = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise TorchWireAdapterError(f"invalid wire torch device {requested!r}") from exc
    if selected.type == "cuda" and not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
        raise TorchWireAdapterError("wire entry requests CUDA but torch CUDA is unavailable")
    if selected.type not in {"cpu", "cuda"}:
        raise TorchWireAdapterError("wire torch device must be CPU or CUDA")
    return selected


def _extract_tensor(output: Any, torch: Any) -> Any:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if isinstance(output, dict) and torch.is_tensor(output.get("logits")):
        return output["logits"]
    raise TorchWireAdapterError("wire model did not return a logits tensor")


def _option(entry: Mapping[str, Any], *names: str, default: Any) -> Any:
    inference = entry.get("inference")
    nested = inference if isinstance(inference, Mapping) else {}
    for name in names:
        if name in entry:
            return entry[name]
        if name in nested:
            return nested[name]
    return default


def _finite_probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TorchWireAdapterError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise TorchWireAdapterError(f"{name} must be finite and in [0, 1]")
    return result


def _normalization_triplet(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise TorchWireAdapterError(f"wire normalization {name} must contain three values")
    output: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TorchWireAdapterError(f"wire normalization {name} must be numeric")
        converted = float(item)
        if not math.isfinite(converted):
            raise TorchWireAdapterError(f"wire normalization {name} must be finite")
        output.append(converted)
    return output[0], output[1], output[2]


def _evidence_id(
    version: str,
    artifact_sha256: str,
    view: ImageView,
    threshold: float,
) -> str:
    payload = "\0".join(
        (
            version,
            artifact_sha256,
            view.view_id,
            view.sha256,
            f"{threshold:.12g}",
            str(view.width),
            str(view.height),
        )
    ).encode("utf-8")
    return "wire-model:" + hashlib.sha256(payload).hexdigest()[:24]


def _check_budget(ctx: Any) -> None:
    budget = getattr(ctx, "time_budget", None) if ctx is not None else None
    check = getattr(budget, "check", None)
    if callable(check):
        check(stage="wire_provider")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "TorchWireAdapter",
    "TorchWireAdapterError",
    "create_wire_provider",
    "factory",
    "load",
]
