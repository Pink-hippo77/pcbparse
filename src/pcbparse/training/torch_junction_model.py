from __future__ import annotations

"""Six-class PyTorch junction classifier and auditable artifact lifecycle."""

import hashlib
import importlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from pcbparse.training.torch_wire_junction import JUNCTION_CLASSES


CHECKPOINT_SCHEMA = "pcbparse.junction.torch-checkpoint.v1"
EXPORT_SCHEMA = "pcbparse.junction.torch-export-consistency.v1"
_MODEL_CLASS: type[Any] | None = None


class TorchJunctionRuntimeError(RuntimeError):
    """A junction model operation cannot be completed safely."""


class JunctionCheckpointError(TorchJunctionRuntimeError):
    """A checkpoint is corrupt, incompatible, or has wrong lineage."""


class JunctionExportError(TorchJunctionRuntimeError):
    """An export is missing or differs from eager inference."""


@dataclass(frozen=True, slots=True)
class JunctionModelConfig:
    input_channels: int = 3
    base_channels: int = 16
    class_names: tuple[str, ...] = JUNCTION_CLASSES
    dropout: float = 0.1
    input_size: int = 64

    def __post_init__(self) -> None:
        if self.input_channels != 3:
            raise ValueError("junction input_channels must be 3")
        if type(self.base_channels) is not int or self.base_channels < 4:
            raise ValueError("junction base_channels must be an integer >= 4")
        if tuple(self.class_names) != JUNCTION_CLASSES:
            raise ValueError("junction class order must match the frozen six-class schema")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)) or not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("junction dropout must be in [0, 1)")
        if type(self.input_size) is not int or self.input_size < 24:
            raise ValueError("junction input_size must be an integer >= 24")


@dataclass(frozen=True, slots=True)
class JunctionCheckpointInfo:
    path: str
    sha256: str
    bytes: int
    model_version: str
    epoch: int
    global_step: int
    lineage: dict[str, Any]


@dataclass(slots=True)
class LoadedJunctionCheckpoint:
    model: Any
    epoch: int
    global_step: int
    lineage: dict[str, Any]
    metrics: dict[str, float]
    checkpoint_sha256: str
    model_version: str
    optimizer_loaded: bool
    path: Path


@dataclass(frozen=True, slots=True)
class JunctionExportReport:
    schema: str
    path: str
    format: Literal["torchscript", "onnx"]
    artifact_sha256: str
    artifact_bytes: int
    sample_count: int
    input_shape: tuple[int, int, int, int]
    max_abs_error: float
    mean_abs_error: float
    tolerance: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TinyJunctionClassifier:
    """Lazy facade returning an ordinary compact ``torch.nn.Module``."""

    def __new__(
        cls,
        config: JunctionModelConfig | Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> Any:
        if config is not None and overrides:
            raise ValueError("pass either config or keyword overrides, not both")
        if config is None:
            resolved = JunctionModelConfig(**overrides)
        elif isinstance(config, JunctionModelConfig):
            resolved = config
        elif isinstance(config, Mapping):
            raw = dict(config)
            if "class_names" in raw:
                raw["class_names"] = tuple(raw["class_names"])
            resolved = JunctionModelConfig(**raw)
        else:
            raise TypeError("config must be JunctionModelConfig, a mapping, or None")
        return _model_class()(resolved)


def build_junction_model(
    config: JunctionModelConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    return TinyJunctionClassifier(config, **overrides)


def _model_class() -> type[Any]:
    global _MODEL_CLASS
    if _MODEL_CLASS is not None:
        return _MODEL_CLASS
    torch = _require_torch()
    nn = torch.nn

    class _ConvBlock(nn.Module):
        def __init__(self, inputs: int, outputs: int, *, stride: int) -> None:
            super().__init__()
            groups = max(1, math.gcd(outputs, 8))
            self.layers = nn.Sequential(
                nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(groups, outputs),
                nn.SiLU(inplace=True),
                nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
                nn.GroupNorm(groups, outputs),
                nn.SiLU(inplace=True),
            )

        def forward(self, inputs: Any) -> Any:
            return self.layers(inputs)

    class _TinyJunctionClassifier(nn.Module):
        def __init__(self, config: JunctionModelConfig) -> None:
            super().__init__()
            width = config.base_channels
            self.features = nn.Sequential(
                _ConvBlock(config.input_channels, width, stride=1),
                _ConvBlock(width, width * 2, stride=2),
                _ConvBlock(width * 2, width * 4, stride=2),
                _ConvBlock(width * 4, width * 6, stride=2),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.dropout = nn.Dropout(float(config.dropout))
            self.classifier = nn.Linear(width * 6, len(config.class_names))
            self.model_config = {
                "name": "tiny_junction_classifier",
                "input_channels": config.input_channels,
                "base_channels": config.base_channels,
                "class_names": list(config.class_names),
                "dropout": float(config.dropout),
                "input_size": config.input_size,
            }

        def forward(self, inputs: Any) -> Any:
            values = self.features(inputs).flatten(1)
            return self.classifier(self.dropout(values))

    _TinyJunctionClassifier.__name__ = "TinyJunctionClassifier"
    _TinyJunctionClassifier.__qualname__ = "TinyJunctionClassifier"
    _MODEL_CLASS = _TinyJunctionClassifier
    return _MODEL_CLASS


def junction_class_weights_from_counts(
    counts: Mapping[str, int],
    *,
    power: float = 0.5,
    require_complete: bool = True,
) -> Any:
    """Return mean-one inverse-frequency weights in frozen class order."""

    torch = _require_torch()
    if isinstance(power, bool) or not isinstance(power, (int, float)) or not math.isfinite(float(power)) or float(power) < 0:
        raise ValueError("power must be finite and non-negative")
    unknown = set(counts) - set(JUNCTION_CLASSES)
    if unknown:
        raise ValueError(f"unknown junction classes: {sorted(unknown)}")
    values: list[float] = []
    missing: list[str] = []
    for name in JUNCTION_CLASSES:
        count = counts.get(name, 0)
        if type(count) is not int or count < 0:
            raise ValueError(f"class count for {name!r} must be a non-negative integer")
        if count == 0:
            missing.append(name)
            values.append(0.0)
        else:
            values.append(float(count) ** -float(power))
    if require_complete and missing:
        raise ValueError("six-class loss is missing supervision for: " + ", ".join(missing))
    nonzero = [value for value in values if value > 0.0]
    if not nonzero:
        raise ValueError("at least one class count must be positive")
    scale = len(nonzero) / sum(nonzero)
    return torch.tensor([value * scale for value in values], dtype=torch.float32)


def junction_loss(
    logits: Any,
    labels: Any,
    *,
    class_weights: Any | None = None,
    label_smoothing: float = 0.0,
) -> Any:
    torch = _require_torch()
    if not torch.is_tensor(logits) or logits.ndim != 2 or logits.shape[1] != len(JUNCTION_CLASSES):
        raise ValueError("junction logits must have shape N,6")
    if not torch.is_tensor(labels) or labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("junction labels must have shape N")
    if isinstance(label_smoothing, bool) or not isinstance(label_smoothing, (int, float)) or not 0 <= float(label_smoothing) < 1:
        raise ValueError("label_smoothing must be in [0,1)")
    labels = labels.to(device=logits.device, dtype=torch.long)
    if bool(((labels < 0) | (labels >= len(JUNCTION_CLASSES))).any()):
        raise ValueError("junction labels contain an out-of-schema class ID")
    weight = None
    if class_weights is not None:
        if not torch.is_tensor(class_weights) or tuple(class_weights.shape) != (len(JUNCTION_CLASSES),):
            raise ValueError("class_weights must be a six-element tensor")
        weight = class_weights.to(device=logits.device, dtype=logits.dtype)
        if bool((weight < 0).any()) or not bool(torch.isfinite(weight).all()):
            raise ValueError("class_weights must be finite and non-negative")
    return torch.nn.functional.cross_entropy(
        logits,
        labels,
        weight=weight,
        label_smoothing=float(label_smoothing),
    )


def junction_train_step(
    model: Any,
    batch: Mapping[str, Any],
    optimizer: Any,
    *,
    device: str | Any = "cpu",
    class_weights: Any | None = None,
    label_smoothing: float = 0.0,
    zero_grad: bool = True,
) -> dict[str, float]:
    torch = _require_torch()
    image, label = _batch(batch, device)
    model.to(device)
    model.train()
    if zero_grad:
        optimizer.zero_grad(set_to_none=True)
    logits = model(image)
    loss = junction_loss(
        logits,
        label,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
    )
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        accuracy = (logits.argmax(dim=1) == label).float().mean()
    return {"loss": float(loss.detach().cpu()), "accuracy": float(accuracy.detach().cpu())}


def junction_confusion_matrix(logits: Any, labels: Any) -> Any:
    torch = _require_torch()
    if not torch.is_tensor(logits) or logits.ndim != 2 or logits.shape[1] != len(JUNCTION_CLASSES):
        raise ValueError("junction logits must have shape N,6")
    if not torch.is_tensor(labels) or labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("junction labels must have shape N")
    labels = labels.to(device=logits.device, dtype=torch.long)
    prediction = logits.argmax(dim=1)
    if bool(((labels < 0) | (labels >= len(JUNCTION_CLASSES))).any()):
        raise ValueError("junction labels contain an out-of-schema class ID")
    indices = labels * len(JUNCTION_CLASSES) + prediction
    return torch.bincount(indices, minlength=36).reshape(6, 6)


def junction_metrics_from_confusion(confusion: Any) -> dict[str, float]:
    torch = _require_torch()
    values = torch.as_tensor(confusion, dtype=torch.float64)
    if tuple(values.shape) != (6, 6) or bool((values < 0).any()) or not bool(torch.isfinite(values).all()):
        raise ValueError("confusion matrix must be finite non-negative 6x6")
    true_positive = values.diag()
    support = values.sum(dim=1)
    predicted = values.sum(dim=0)
    recall = true_positive / support.clamp_min(1.0)
    precision = true_positive / predicted.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-12)
    observed = support > 0
    result: dict[str, float] = {
        "accuracy": float(true_positive.sum() / values.sum().clamp_min(1.0)),
        "macro_f1": float(f1.mean()),
        "macro_f1_observed": float(f1[observed].mean()) if bool(observed.any()) else 0.0,
        "observed_class_count": float(observed.sum()),
    }
    for index, name in enumerate(JUNCTION_CLASSES):
        result[f"recall/{name}"] = float(recall[index])
        result[f"precision/{name}"] = float(precision[index])
        result[f"f1/{name}"] = float(f1[index])
        result[f"support/{name}"] = float(support[index])
    return result


def evaluate_junction_model(
    model: Any,
    batches: Any,
    *,
    device: str | Any = "cpu",
    maximum_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate all examples and return six-class metrics plus predictions."""

    torch = _require_torch()
    if maximum_batches is not None and (type(maximum_batches) is not int or maximum_batches <= 0):
        raise ValueError("maximum_batches must be a positive integer or None")
    model.to(device)
    model.eval()
    confusion = torch.zeros((6, 6), dtype=torch.long)
    predictions: list[dict[str, Any]] = []
    losses: list[float] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(batches):
            if maximum_batches is not None and batch_index >= maximum_batches:
                break
            image, label = _batch(batch, device)
            logits = model(image)
            loss = junction_loss(logits, label)
            probability = torch.softmax(logits, dim=1)
            predicted = probability.argmax(dim=1)
            confusion += junction_confusion_matrix(logits, label).to("cpu")
            losses.append(float(loss.detach().cpu()))
            record_ids = batch.get("record_id", [f"batch-{batch_index}:{i}" for i in range(image.shape[0])])
            for item_index in range(image.shape[0]):
                prediction_id = int(predicted[item_index].detach().cpu())
                truth_id = int(label[item_index].detach().cpu())
                predictions.append(
                    {
                        "record_id": str(record_ids[item_index]),
                        "truth_id": truth_id,
                        "truth": JUNCTION_CLASSES[truth_id],
                        "prediction_id": prediction_id,
                        "prediction": JUNCTION_CLASSES[prediction_id],
                        "score": float(probability[item_index, prediction_id].detach().cpu()),
                        "correct": prediction_id == truth_id,
                    }
                )
    if not predictions:
        raise ValueError("junction evaluation received no examples")
    metrics = junction_metrics_from_confusion(confusion)
    metrics["loss"] = sum(losses) / len(losses)
    return {"metrics": metrics, "confusion_matrix": confusion.tolist(), "predictions": predictions}


def save_junction_checkpoint(
    path: str | Path,
    model: Any,
    *,
    optimizer: Any | None = None,
    epoch: int = 0,
    global_step: int = 0,
    lineage: Mapping[str, Any],
    metrics: Mapping[str, float] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> JunctionCheckpointInfo:
    """Atomically save state, exact lineage, and finite validation metrics."""

    torch = _require_torch()
    if type(epoch) is not int or epoch < 0 or type(global_step) is not int or global_step < 0:
        raise ValueError("epoch/global_step must be non-negative integers")
    lineage_copy = _json_mapping(lineage, "lineage", require_nonempty=True)
    metrics_copy = _finite_metrics(metrics or {})
    extra_copy = _json_mapping(extra or {}, "extra", require_nonempty=False)
    model_config = getattr(model, "model_config", None)
    if not isinstance(model_config, Mapping) or model_config.get("name") != "tiny_junction_classifier":
        raise JunctionCheckpointError("checkpoint requires TinyJunctionClassifier model_config")
    if tuple(model_config.get("class_names", ())) != JUNCTION_CLASSES:
        raise JunctionCheckpointError("checkpoint class schema differs from frozen order")
    document = {
        "schema": CHECKPOINT_SCHEMA,
        "model": _json_mapping(model_config, "model_config", require_nonempty=True),
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "lineage": lineage_copy,
        "lineage_sha256": _canonical_sha256(lineage_copy),
        "metrics": metrics_copy,
        "extra": extra_copy,
        "torch_version": str(torch.__version__),
    }
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(document, temporary)
        if temporary.stat().st_size <= 0:
            raise JunctionCheckpointError("torch produced an empty checkpoint")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    digest = _sha256_file(destination)
    return JunctionCheckpointInfo(
        path=str(destination),
        sha256=digest,
        bytes=destination.stat().st_size,
        model_version=f"tiny-junction-{digest[:16]}",
        epoch=epoch,
        global_step=global_step,
        lineage=lineage_copy,
    )


def load_junction_checkpoint(
    path: str | Path,
    *,
    map_location: str | Any = "cpu",
    expected_lineage: Mapping[str, Any] | None = None,
    model: Any | None = None,
    optimizer: Any | None = None,
    strict: bool = True,
) -> LoadedJunctionCheckpoint:
    torch = _require_torch()
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise JunctionCheckpointError(f"checkpoint is missing or empty: {source}")
    try:
        try:
            document = torch.load(source, map_location=map_location, weights_only=False)
        except TypeError:
            document = torch.load(source, map_location=map_location)
    except Exception as exc:
        raise JunctionCheckpointError(f"cannot deserialize junction checkpoint: {type(exc).__name__}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != CHECKPOINT_SCHEMA:
        raise JunctionCheckpointError("junction checkpoint schema is incompatible")
    lineage = _json_mapping(document.get("lineage"), "lineage", require_nonempty=True)
    if document.get("lineage_sha256") != _canonical_sha256(lineage):
        raise JunctionCheckpointError("checkpoint lineage digest mismatch")
    if expected_lineage is not None and lineage != _json_mapping(expected_lineage, "expected_lineage", require_nonempty=True):
        raise JunctionCheckpointError("checkpoint lineage differs from requested frozen lineage")
    model_spec = document.get("model")
    if not isinstance(model_spec, Mapping) or model_spec.get("name") != "tiny_junction_classifier":
        raise JunctionCheckpointError("checkpoint model specification is invalid")
    if tuple(model_spec.get("class_names", ())) != JUNCTION_CLASSES:
        raise JunctionCheckpointError("checkpoint class order is invalid")
    if model is None:
        try:
            model = build_junction_model(
                input_channels=int(model_spec["input_channels"]),
                base_channels=int(model_spec["base_channels"]),
                class_names=tuple(model_spec["class_names"]),
                dropout=float(model_spec["dropout"]),
                input_size=int(model_spec["input_size"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise JunctionCheckpointError("checkpoint model parameters are invalid") from exc
    state = document.get("state_dict")
    if not isinstance(state, Mapping):
        raise JunctionCheckpointError("checkpoint state_dict is missing")
    try:
        model.load_state_dict(state, strict=strict)
    except Exception as exc:
        raise JunctionCheckpointError(f"checkpoint weights are incompatible: {type(exc).__name__}: {exc}") from exc
    optimizer_loaded = False
    if optimizer is not None and document.get("optimizer_state_dict") is not None:
        try:
            optimizer.load_state_dict(document["optimizer_state_dict"])
        except Exception as exc:
            raise JunctionCheckpointError(f"optimizer state is incompatible: {type(exc).__name__}: {exc}") from exc
        optimizer_loaded = True
    epoch = document.get("epoch")
    global_step = document.get("global_step")
    if type(epoch) is not int or epoch < 0 or type(global_step) is not int or global_step < 0:
        raise JunctionCheckpointError("checkpoint epoch/global_step are invalid")
    digest = _sha256_file(source)
    return LoadedJunctionCheckpoint(
        model=model,
        epoch=epoch,
        global_step=global_step,
        lineage=lineage,
        metrics=_finite_metrics(document.get("metrics", {})),
        checkpoint_sha256=digest,
        model_version=f"tiny-junction-{digest[:16]}",
        optimizer_loaded=optimizer_loaded,
        path=source,
    )


def export_junction_model(
    model: Any,
    path: str | Path,
    *,
    format: Literal["torchscript", "onnx"] = "torchscript",
    input_shape: Sequence[int] = (2, 3, 64, 64),
    device: str | Any = "cpu",
    tolerance: float | None = None,
    verify: bool = True,
    opset_version: int = 17,
) -> JunctionExportReport:
    """Atomically export and compare eager logits on a fixed ramp batch."""

    torch = _require_torch()
    normalized_format = str(format).lower()
    if normalized_format not in {"torchscript", "onnx"}:
        raise ValueError("format must be 'torchscript' or 'onnx'")
    shape = tuple(int(value) for value in input_shape)
    if len(shape) != 4 or any(value <= 0 for value in shape) or shape[1] != 3:
        raise ValueError("input_shape must be positive NCHW RGB")
    maximum_error = float(tolerance) if tolerance is not None else (1e-5 if normalized_format == "torchscript" else 2e-4)
    if not math.isfinite(maximum_error) or maximum_error < 0:
        raise ValueError("tolerance must be finite and non-negative")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = (
        tempfile.gettempdir()
        if normalized_format == "torchscript"
        else destination.parent
    )
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=f".{normalized_format}.tmp", dir=temporary_directory)
    os.close(descriptor)
    temporary = Path(temporary_name)
    original_training = bool(model.training)
    try:
        original_device = next(model.parameters()).device
    except StopIteration:
        original_device = torch.device("cpu")
    example = torch.linspace(0.0, 1.0, steps=int(np.prod(shape)), dtype=torch.float32, device=device).reshape(shape)
    try:
        model.to(device)
        model.eval()
        with torch.inference_mode():
            reference = model(example).detach().to("cpu").numpy()
        if reference.shape != (shape[0], 6):
            raise JunctionExportError("junction model does not emit N,6 logits")
        if normalized_format == "torchscript":
            traced = torch.jit.trace(model, example, strict=True)
            traced.save(str(temporary))
        else:
            options = dict(
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["image"],
                output_names=["logits"],
                dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            )
            try:
                torch.onnx.export(model, example, str(temporary), dynamo=False, **options)
            except TypeError:
                torch.onnx.export(model, example, str(temporary), **options)
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise JunctionExportError("junction export produced no bytes")
        if verify:
            candidate = _run_exported(temporary, normalized_format, example)
            if candidate.shape != reference.shape:
                raise JunctionExportError("junction export output shape differs from eager")
            error = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
            max_abs = float(error.max(initial=0.0))
            mean_abs = float(error.mean()) if error.size else 0.0
            if not math.isfinite(max_abs) or max_abs > maximum_error:
                raise JunctionExportError(f"junction {normalized_format} max_abs_error {max_abs:.8g} exceeds {maximum_error:.8g}")
        else:
            max_abs = mean_abs = 0.0
        try:
            os.replace(temporary, destination)
        except OSError:
            # torch.jit.save cannot write non-ASCII (e.g. Chinese) paths on
            # Windows; stage in the ASCII %TEMP% directory then copy.
            shutil.copy2(temporary, destination)
            temporary.unlink(missing_ok=True)
    except JunctionExportError:
        raise
    except Exception as exc:
        raise JunctionExportError(f"cannot export junction model: {type(exc).__name__}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
        model.to(original_device)
        model.train(original_training)
    digest = _sha256_file(destination)
    return JunctionExportReport(
        schema=EXPORT_SCHEMA,
        path=str(destination),
        format=normalized_format,  # type: ignore[arg-type]
        artifact_sha256=digest,
        artifact_bytes=destination.stat().st_size,
        sample_count=shape[0],
        input_shape=shape,
        max_abs_error=max_abs,
        mean_abs_error=mean_abs,
        tolerance=maximum_error,
        passed=True,
    )


def verify_junction_export(
    model: Any,
    path: str | Path,
    *,
    format: Literal["torchscript", "onnx"],
    inputs: Any,
    tolerance: float | None = None,
) -> dict[str, float | bool]:
    torch = _require_torch()
    maximum_error = float(tolerance) if tolerance is not None else (1e-5 if format == "torchscript" else 2e-4)
    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    with torch.inference_mode():
        reference = model(inputs.to(device)).detach().to("cpu").numpy()
    resolved_path = Path(path).resolve()
    if format == "torchscript":
        # torch.jit.load cannot open non-ASCII (e.g. Chinese) paths on Windows.
        descriptor, staged_name = tempfile.mkstemp(
            prefix=".junction-verify-", suffix=".pt", dir=tempfile.gettempdir()
        )
        os.close(descriptor)
        staged = Path(staged_name)
        staged.unlink()
        try:
            shutil.copy2(resolved_path, staged)
            candidate = _run_exported(staged, format, inputs)
        finally:
            staged.unlink(missing_ok=True)
    else:
        candidate = _run_exported(resolved_path, format, inputs)
    if candidate.shape != reference.shape:
        raise JunctionExportError("junction export output shape differs from eager")
    error = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    max_abs = float(error.max(initial=0.0))
    mean_abs = float(error.mean()) if error.size else 0.0
    if not math.isfinite(max_abs) or max_abs > maximum_error:
        raise JunctionExportError(f"junction {format} max_abs_error {max_abs:.8g} exceeds {maximum_error:.8g}")
    return {"max_abs_error": max_abs, "mean_abs_error": mean_abs, "tolerance": maximum_error, "passed": True}


def _run_exported(path: Path, format: str, inputs: Any) -> np.ndarray:
    torch = _require_torch()
    if format == "torchscript":
        model = torch.jit.load(str(path), map_location="cpu")
        model.eval()
        with torch.inference_mode():
            return model(inputs.detach().to("cpu")).detach().to("cpu").numpy()
    try:
        runtime = importlib.import_module("onnxruntime")
    except ImportError as exc:
        raise JunctionExportError("onnxruntime is required to verify junction ONNX") from exc
    session = runtime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    result = session.run(None, {session.get_inputs()[0].name: inputs.detach().to("cpu").numpy()})[0]
    return np.asarray(result)


def _batch(batch: Mapping[str, Any], device: str | Any) -> tuple[Any, Any]:
    torch = _require_torch()
    if not isinstance(batch, Mapping) or "image" not in batch or "label" not in batch:
        raise ValueError("junction batch requires image and label")
    image = batch["image"]
    label = batch["label"]
    if not torch.is_tensor(image) or image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("junction batch image must be NCHW RGB")
    if not torch.is_tensor(label) or label.ndim != 1 or label.shape[0] != image.shape[0]:
        raise ValueError("junction batch label must have shape N")
    return image.to(device=device, dtype=torch.float32), label.to(device=device, dtype=torch.long)


def _json_mapping(value: Any, name: str, *, require_nonempty: bool) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    try:
        copied = json.loads(json.dumps(dict(value), ensure_ascii=False, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only finite JSON values") from exc
    if require_nonempty and not copied:
        raise ValueError(f"{name} cannot be empty")
    return copied


def _finite_metrics(value: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("metrics must be a mapping")
    output: dict[str, float] = {}
    for name, raw in value.items():
        if not isinstance(name, str) or not name or isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            raise ValueError("metric names must be non-empty and values finite numeric")
        output[name] = float(raw)
    return output


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


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
        raise TorchJunctionRuntimeError("PyTorch is required for junction model operations") from exc


__all__ = [
    "CHECKPOINT_SCHEMA",
    "EXPORT_SCHEMA",
    "JunctionCheckpointError",
    "JunctionCheckpointInfo",
    "JunctionExportError",
    "JunctionExportReport",
    "JunctionModelConfig",
    "LoadedJunctionCheckpoint",
    "TinyJunctionClassifier",
    "TorchJunctionRuntimeError",
    "build_junction_model",
    "evaluate_junction_model",
    "export_junction_model",
    "junction_class_weights_from_counts",
    "junction_confusion_matrix",
    "junction_loss",
    "junction_metrics_from_confusion",
    "junction_train_step",
    "load_junction_checkpoint",
    "save_junction_checkpoint",
    "verify_junction_export",
]
