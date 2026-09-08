from __future__ import annotations

"""Trainable PyTorch wire-segmentation model and artifact lifecycle.

The module is intentionally importable without PyTorch.  Framework imports are
performed only when a model, loss, training step, checkpoint, or export is
actually requested.  This keeps dataset audits and production readiness checks
usable in the lightweight runtime.

The frozen :class:`~pcbparse.training.torch_wire_junction.WireTorchDataset`
contract is used directly: ``target`` is a one-pixel wire centreline and
``ignore`` is one for pixels which must not become either positive or negative
supervision.  Every loss term applies the valid mask before reduction.
"""

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


CHECKPOINT_SCHEMA = "pcbparse.wire.torch-checkpoint.v1"
EXPORT_SCHEMA = "pcbparse.wire.torch-export-consistency.v1"
_TINY_UNET_CLASS: type[Any] | None = None


class TorchWireRuntimeError(RuntimeError):
    """A PyTorch wire operation cannot be completed safely."""


class WireCheckpointError(TorchWireRuntimeError):
    """A checkpoint is corrupt, incompatible, or has the wrong lineage."""


class WireExportError(TorchWireRuntimeError):
    """A model export failed or disagrees with eager inference."""


@dataclass(frozen=True, slots=True)
class WireLossConfig:
    """Weights for centreline-aware segmentation supervision."""

    bce_weight: float = 0.45
    dice_weight: float = 0.35
    cldice_weight: float = 0.20
    positive_weight: float | None = None
    skeleton_iterations: int = 12
    epsilon: float = 1.0e-6

    def validate(self) -> "WireLossConfig":
        values = (self.bce_weight, self.dice_weight, self.cldice_weight)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("wire loss weights must be finite and non-negative")
        if sum(values) <= 0:
            raise ValueError("at least one wire loss weight must be positive")
        if self.positive_weight is not None and (
            not math.isfinite(self.positive_weight) or self.positive_weight <= 0
        ):
            raise ValueError("positive_weight must be finite and positive")
        if self.skeleton_iterations <= 0:
            raise ValueError("skeleton_iterations must be positive")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        return self


@dataclass(frozen=True, slots=True)
class WireCheckpointInfo:
    path: str
    sha256: str
    bytes: int
    model_version: str
    epoch: int
    global_step: int
    lineage: dict[str, Any]


@dataclass(slots=True)
class LoadedWireCheckpoint:
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
class WireExportReport:
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


class TinyUNet:
    """Lazy constructor for the real ``torch.nn.Module`` Tiny U-Net.

    ``TinyUNet(...)`` returns an ordinary ``nn.Module``.  The facade exists so
    importing this file never imports torch; optimizers, scripting, and
    ``isinstance(model, torch.nn.Module)`` all operate on the returned object.
    """

    def __new__(
        cls,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 16,
    ) -> Any:
        implementation = _tiny_unet_implementation()
        return implementation(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
        )


def build_tiny_unet(
    *, in_channels: int = 3, out_channels: int = 1, base_channels: int = 16
) -> Any:
    """Build the default real wire model (base width 16)."""

    return TinyUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
    )


def _tiny_unet_implementation() -> type[Any]:
    global _TINY_UNET_CLASS
    if _TINY_UNET_CLASS is not None:
        return _TINY_UNET_CLASS
    torch = _require_torch()
    nn = torch.nn
    functional = torch.nn.functional

    class _ConvBlock(nn.Module):
        def __init__(self, input_channels: int, output_channels: int) -> None:
            super().__init__()
            groups = max(1, math.gcd(8, output_channels))
            self.layers = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                nn.GroupNorm(groups, output_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                nn.GroupNorm(groups, output_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, inputs: Any) -> Any:
            return self.layers(inputs)

    class _TinyUNet(nn.Module):
        def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            base_channels: int = 16,
        ) -> None:
            super().__init__()
            if in_channels <= 0 or out_channels <= 0 or base_channels <= 0:
                raise ValueError("TinyUNet channel counts must be positive")
            self.encoder1 = _ConvBlock(in_channels, base_channels)
            self.encoder2 = _ConvBlock(base_channels, base_channels * 2)
            self.bottleneck = _ConvBlock(base_channels * 2, base_channels * 4)
            self.decoder2 = _ConvBlock(base_channels * 6, base_channels * 2)
            self.decoder1 = _ConvBlock(base_channels * 3, base_channels)
            self.head = nn.Conv2d(base_channels, out_channels, 1)
            self.model_config = {
                "name": "tiny_unet",
                "in_channels": int(in_channels),
                "out_channels": int(out_channels),
                "base_channels": int(base_channels),
            }

        def forward(self, inputs: Any) -> Any:
            encoder1 = self.encoder1(inputs)
            encoder2 = self.encoder2(
                functional.max_pool2d(encoder1, kernel_size=2, stride=2, ceil_mode=True)
            )
            bottleneck = self.bottleneck(
                functional.max_pool2d(encoder2, kernel_size=2, stride=2, ceil_mode=True)
            )
            decoder2 = functional.interpolate(
                bottleneck,
                size=encoder2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            decoder2 = self.decoder2(torch.cat((decoder2, encoder2), dim=1))
            decoder1 = functional.interpolate(
                decoder2,
                size=encoder1.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            decoder1 = self.decoder1(torch.cat((decoder1, encoder1), dim=1))
            return self.head(decoder1)

    _TinyUNet.__name__ = "TinyUNet"
    _TinyUNet.__qualname__ = "TinyUNet"
    _TinyUNet_CLASS = _TinyUNet
    return _TinyUNet_CLASS


def compute_wire_loss(
    logits: Any,
    target: Any,
    ignore: Any | None = None,
    *,
    valid: Any | None = None,
    config: WireLossConfig | None = None,
) -> dict[str, Any]:
    """Return differentiable, fully masked BCE/Dice/soft-clDice terms."""

    torch = _require_torch()
    functional = torch.nn.functional
    policy = (config or WireLossConfig()).validate()
    logits, target, valid_mask = _normalise_supervision(
        logits, target, ignore=ignore, valid=valid
    )
    target_masked = target * valid_mask
    probabilities = torch.sigmoid(logits) * valid_mask
    valid_pixels = valid_mask.sum()

    positive_weight = None
    if policy.positive_weight is not None:
        positive_weight = torch.as_tensor(
            policy.positive_weight, dtype=logits.dtype, device=logits.device
        )
    pixel_bce = functional.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=positive_weight,
    )
    bce = (pixel_bce * valid_mask).sum() / valid_pixels.clamp_min(1.0)

    dimensions = tuple(range(1, logits.ndim))
    intersection = (probabilities * target_masked).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + target_masked.sum(dim=dimensions)
    valid_per_sample = valid_mask.sum(dim=dimensions) > 0
    dice_per_sample = 1.0 - (
        (2.0 * intersection + policy.epsilon) / (denominator + policy.epsilon)
    )
    dice = _masked_sample_mean(dice_per_sample, valid_per_sample, logits)

    predicted_skeleton = soft_skeletonize(
        probabilities,
        iterations=policy.skeleton_iterations,
        valid=valid_mask,
    )
    target_skeleton = soft_skeletonize(
        target_masked,
        iterations=policy.skeleton_iterations,
        valid=valid_mask,
    )
    predicted_skeleton_sum = predicted_skeleton.sum(dim=dimensions)
    target_skeleton_sum = target_skeleton.sum(dim=dimensions)
    topology_precision = (
        (predicted_skeleton * target_masked).sum(dim=dimensions) + policy.epsilon
    ) / (predicted_skeleton_sum + policy.epsilon)
    topology_recall = (
        (target_skeleton * probabilities).sum(dim=dimensions) + policy.epsilon
    ) / (target_skeleton_sum + policy.epsilon)
    cldice_score = (
        2.0 * topology_precision * topology_recall + policy.epsilon
    ) / (topology_precision + topology_recall + policy.epsilon)
    # Empty targets do not define a topology; BCE and Dice still supervise them.
    topology_samples = valid_per_sample & (target_masked.sum(dim=dimensions) > 0)
    cldice = _masked_sample_mean(1.0 - cldice_score, topology_samples, logits)

    weight_sum = policy.bce_weight + policy.dice_weight + policy.cldice_weight
    total = (
        policy.bce_weight * bce
        + policy.dice_weight * dice
        + policy.cldice_weight * cldice
    ) / weight_sum
    # When a batch is entirely ignored, all terms and gradients are exactly zero.
    total = torch.where(valid_pixels > 0, total, logits.sum() * 0.0)
    return {
        "loss": total,
        "bce": bce,
        "dice": dice,
        "cldice": cldice,
        "valid_pixels": valid_pixels,
    }


def masked_wire_loss(
    logits: Any,
    target: Any,
    ignore: Any | None = None,
    *,
    valid: Any | None = None,
    config: WireLossConfig | None = None,
) -> Any:
    """Convenience scalar form of :func:`compute_wire_loss`."""

    return compute_wire_loss(
        logits, target, ignore, valid=valid, config=config
    )["loss"]


def soft_skeletonize(inputs: Any, *, iterations: int = 12, valid: Any | None = None) -> Any:
    """Differentiable morphology used by soft-clDice.

    The valid mask is re-applied after every neighbourhood operation so an
    ignored prediction cannot leak into a supervised topology term.
    """

    torch = _require_torch()
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    image = inputs.clamp(0.0, 1.0)
    valid_mask = torch.ones_like(image) if valid is None else valid.to(
        device=image.device, dtype=image.dtype
    )
    if valid_mask.shape != image.shape:
        valid_mask = torch.broadcast_to(valid_mask, image.shape)
    image = image * valid_mask
    opened = _soft_open(image, valid_mask)
    skeleton = torch.relu(image - opened) * valid_mask
    for _ in range(iterations - 1):
        image = _soft_erode(image) * valid_mask
        opened = _soft_open(image, valid_mask)
        delta = torch.relu(image - opened) * valid_mask
        skeleton = skeleton + torch.relu(delta - skeleton * delta)
    return skeleton * valid_mask


def wire_centerline_metrics(
    logits_or_probabilities: Any,
    target: Any,
    ignore: Any | None = None,
    *,
    valid: Any | None = None,
    from_logits: bool = True,
    threshold: float = 0.5,
    tolerance_px: int = 1,
    skeleton_iterations: int = 32,
) -> dict[str, float]:
    """Compute masked centreline P/R/F1 and a connectivity auxiliary score."""

    torch = _require_torch()
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if tolerance_px < 0:
        raise ValueError("tolerance_px must be non-negative")
    values, target, valid_mask = _normalise_supervision(
        logits_or_probabilities, target, ignore=ignore, valid=valid
    )
    probabilities = torch.sigmoid(values) if from_logits else values.clamp(0.0, 1.0)
    predicted = (probabilities >= threshold).to(values.dtype) * valid_mask
    expected = (target >= 0.5).to(values.dtype) * valid_mask
    predicted_skeleton = (
        soft_skeletonize(
            predicted,
            iterations=skeleton_iterations,
            valid=valid_mask,
        )
        > 0.5
    ).to(values.dtype)
    target_skeleton = (
        soft_skeletonize(
            expected,
            iterations=skeleton_iterations,
            valid=valid_mask,
        )
        > 0.5
    ).to(values.dtype)
    if tolerance_px:
        kernel = tolerance_px * 2 + 1
        predicted_neighbourhood = torch.nn.functional.max_pool2d(
            predicted_skeleton, kernel, stride=1, padding=tolerance_px
        ) * valid_mask
        target_neighbourhood = torch.nn.functional.max_pool2d(
            target_skeleton, kernel, stride=1, padding=tolerance_px
        ) * valid_mask
    else:
        predicted_neighbourhood = predicted_skeleton
        target_neighbourhood = target_skeleton
    predicted_count = predicted_skeleton.sum()
    target_count = target_skeleton.sum()
    precision_hits = (predicted_skeleton * target_neighbourhood).sum()
    recall_hits = (target_skeleton * predicted_neighbourhood).sum()
    precision = (
        precision_hits / predicted_count
        if float(predicted_count.detach().cpu()) > 0
        else torch.ones((), device=values.device, dtype=values.dtype)
    )
    recall = (
        recall_hits / target_count
        if float(target_count.detach().cpu()) > 0
        else torch.ones((), device=values.device, dtype=values.dtype)
    )
    denominator = precision + recall
    f1 = (
        2.0 * precision * recall / denominator
        if float(denominator.detach().cpu()) > 0
        else torch.zeros((), device=values.device, dtype=values.dtype)
    )

    predicted_cpu = predicted_skeleton.detach().to("cpu").numpy().astype(bool)
    target_cpu = target_skeleton.detach().to("cpu").numpy().astype(bool)
    valid_cpu = valid_mask.detach().to("cpu").numpy() >= 0.5
    predicted_components = 0
    target_components = 0
    component_count_error_sum = 0.0
    connectivity_scores: list[float] = []
    for batch_index in range(predicted_cpu.shape[0]):
        prediction_count = _connected_component_count(
            predicted_cpu[batch_index, 0] & valid_cpu[batch_index, 0]
        )
        expected_count = _connected_component_count(
            target_cpu[batch_index, 0] & valid_cpu[batch_index, 0]
        )
        predicted_components += prediction_count
        target_components += expected_count
        component_count_error_sum += abs(prediction_count - expected_count)
        if prediction_count == expected_count == 0:
            connectivity_scores.append(1.0)
        else:
            connectivity_scores.append(
                max(
                    0.0,
                    1.0
                    - abs(prediction_count - expected_count)
                    / max(prediction_count, expected_count, 1),
                )
            )
    return {
        "centerline_precision": float(precision.detach().cpu()),
        "centerline_recall": float(recall.detach().cpu()),
        "centerline_f1": float(f1.detach().cpu()),
        # Alias matches the training config's blessed metric gate name.
        "skeleton_f1": float(f1.detach().cpu()),
        # Raw numerators and denominators are part of the metric contract so a
        # caller can aggregate an entire validation population as a true micro
        # metric.  Averaging per-tile P/R/F1 gives empty and short-line tiles
        # the same weight as dense tiles and can materially overstate quality.
        "precision_hits": float(precision_hits.detach().cpu()),
        "recall_hits": float(recall_hits.detach().cpu()),
        "predicted_skeleton_pixels": float(predicted_count.detach().cpu()),
        "target_skeleton_pixels": float(target_count.detach().cpu()),
        "connectivity_score": float(np.mean(connectivity_scores))
        if connectivity_scores
        else 1.0,
        "connectivity_score_sum": float(sum(connectivity_scores)),
        "predicted_components": float(predicted_components),
        "target_components": float(target_components),
        "component_count_error": float(
            component_count_error_sum / len(connectivity_scores)
        )
        if connectivity_scores
        else 0.0,
        "component_count_error_sum": float(component_count_error_sum),
    }


def wire_train_step(
    model: Any,
    batch: Mapping[str, Any],
    optimizer: Any,
    *,
    device: str | Any = "cpu",
    loss_config: WireLossConfig | None = None,
    threshold: float = 0.5,
    tolerance_px: int = 1,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    """Run one real optimization step over a ``wire_collate`` batch."""

    torch = _require_torch()
    if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
        raise ValueError("max_grad_norm must be finite and positive")
    model.to(device)
    model.train()
    image, target, ignore = _batch_tensors(batch, device)
    optimizer.zero_grad(set_to_none=True)
    logits = _extract_logits(model(image))
    terms = compute_wire_loss(logits, target, ignore, config=loss_config)
    terms["loss"].backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    result = {
        name: float(value.detach().cpu())
        for name, value in terms.items()
        if name != "valid_pixels"
    }
    result["valid_pixels"] = float(terms["valid_pixels"].detach().cpu())
    result.update(
        wire_centerline_metrics(
            logits.detach(),
            target,
            ignore,
            threshold=threshold,
            tolerance_px=tolerance_px,
        )
    )
    return result


def wire_eval_step(
    model: Any,
    batch: Mapping[str, Any],
    *,
    device: str | Any = "cpu",
    loss_config: WireLossConfig | None = None,
    threshold: float = 0.5,
    tolerance_px: int = 1,
) -> dict[str, float]:
    """Run a no-gradient validation step with centreline/topology metrics."""

    torch = _require_torch()
    model.to(device)
    model.eval()
    image, target, ignore = _batch_tensors(batch, device)
    with torch.inference_mode():
        logits = _extract_logits(model(image))
        terms = compute_wire_loss(logits, target, ignore, config=loss_config)
    result = {
        name: float(value.detach().cpu())
        for name, value in terms.items()
        if name != "valid_pixels"
    }
    result["valid_pixels"] = float(terms["valid_pixels"].detach().cpu())
    result.update(
        wire_centerline_metrics(
            logits,
            target,
            ignore,
            threshold=threshold,
            tolerance_px=tolerance_px,
        )
    )
    return result


def save_wire_checkpoint(
    path: str | Path,
    model: Any,
    *,
    optimizer: Any | None = None,
    epoch: int = 0,
    global_step: int = 0,
    lineage: Mapping[str, Any],
    metrics: Mapping[str, float] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> WireCheckpointInfo:
    """Atomically save weights with immutable JSON-safe data lineage."""

    torch = _require_torch()
    if epoch < 0 or global_step < 0:
        raise ValueError("epoch and global_step must be non-negative")
    lineage_copy = _json_mapping(lineage, "lineage", require_nonempty=True)
    metrics_copy = _finite_metrics(metrics or {})
    extra_copy = _json_mapping(extra or {}, "extra", require_nonempty=False)
    model_spec = getattr(model, "model_config", None)
    if not isinstance(model_spec, dict) or model_spec.get("name") != "tiny_unet":
        raise WireCheckpointError(
            "wire checkpoint requires a TinyUNet with serializable model_config"
        )
    document = {
        "schema": CHECKPOINT_SCHEMA,
        "model": _json_mapping(model_spec, "model_config", require_nonempty=True),
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "lineage": lineage_copy,
        "lineage_sha256": _canonical_sha256(lineage_copy),
        "metrics": metrics_copy,
        "extra": extra_copy,
        "torch_version": str(torch.__version__),
    }
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(document, temporary)
        if temporary.stat().st_size <= 0:
            raise WireCheckpointError("torch produced an empty checkpoint")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    digest = _sha256_file(destination)
    return WireCheckpointInfo(
        path=str(destination),
        sha256=digest,
        bytes=destination.stat().st_size,
        model_version=f"tiny-unet-{digest[:16]}",
        epoch=int(epoch),
        global_step=int(global_step),
        lineage=lineage_copy,
    )


def load_wire_checkpoint(
    path: str | Path,
    *,
    map_location: str | Any = "cpu",
    expected_lineage: Mapping[str, Any] | None = None,
    model: Any | None = None,
    optimizer: Any | None = None,
    strict: bool = True,
) -> LoadedWireCheckpoint:
    """Load a checkpoint and fail closed on schema or lineage mismatch."""

    torch = _require_torch()
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise WireCheckpointError(f"wire checkpoint is missing or empty: {source}")
    try:
        try:
            document = torch.load(source, map_location=map_location, weights_only=False)
        except TypeError:  # PyTorch <2.6
            document = torch.load(source, map_location=map_location)
    except Exception as exc:
        raise WireCheckpointError(
            f"cannot deserialize wire checkpoint {source}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(document, dict) or document.get("schema") != CHECKPOINT_SCHEMA:
        raise WireCheckpointError("wire checkpoint schema is missing or incompatible")
    lineage = _json_mapping(document.get("lineage"), "lineage", require_nonempty=True)
    if document.get("lineage_sha256") != _canonical_sha256(lineage):
        raise WireCheckpointError("wire checkpoint lineage digest does not match lineage")
    if expected_lineage is not None:
        expected = _json_mapping(
            expected_lineage, "expected_lineage", require_nonempty=True
        )
        if expected != lineage:
            raise WireCheckpointError(
                "wire checkpoint lineage differs from the requested frozen lineage"
            )
    model_spec = document.get("model")
    if not isinstance(model_spec, dict) or model_spec.get("name") != "tiny_unet":
        raise WireCheckpointError("wire checkpoint model specification is invalid")
    if model is None:
        try:
            model = build_tiny_unet(
                in_channels=int(model_spec["in_channels"]),
                out_channels=int(model_spec["out_channels"]),
                base_channels=int(model_spec["base_channels"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise WireCheckpointError("wire checkpoint model channels are invalid") from exc
    state = document.get("state_dict")
    if not isinstance(state, dict):
        raise WireCheckpointError("wire checkpoint state_dict is missing")
    try:
        model.load_state_dict(state, strict=strict)
    except Exception as exc:
        raise WireCheckpointError(
            f"wire checkpoint weights are incompatible: {type(exc).__name__}: {exc}"
        ) from exc
    optimizer_loaded = False
    optimizer_state = document.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
        except Exception as exc:
            raise WireCheckpointError(
                f"wire optimizer state is incompatible: {type(exc).__name__}: {exc}"
            ) from exc
        optimizer_loaded = True
    epoch = document.get("epoch")
    global_step = document.get("global_step")
    if type(epoch) is not int or epoch < 0 or type(global_step) is not int or global_step < 0:
        raise WireCheckpointError("wire checkpoint epoch/global_step are invalid")
    metrics = _finite_metrics(document.get("metrics", {}))
    digest = _sha256_file(source)
    return LoadedWireCheckpoint(
        model=model,
        epoch=epoch,
        global_step=global_step,
        lineage=lineage,
        metrics=metrics,
        checkpoint_sha256=digest,
        model_version=f"tiny-unet-{digest[:16]}",
        optimizer_loaded=optimizer_loaded,
        path=source,
    )


def export_wire_model(
    model: Any,
    path: str | Path,
    *,
    format: Literal["torchscript", "onnx"] = "torchscript",
    input_shape: Sequence[int] = (1, 3, 128, 128),
    device: str | Any = "cpu",
    tolerance: float | None = None,
    verify: bool = True,
    opset_version: int = 17,
) -> WireExportReport:
    """Export and compare eager vs TorchScript/ONNX output on fixed inputs."""

    torch = _require_torch()
    normalized_format = str(format).lower()
    if normalized_format not in {"torchscript", "onnx"}:
        raise ValueError("format must be 'torchscript' or 'onnx'")
    shape = tuple(int(value) for value in input_shape)
    if len(shape) != 4 or any(value <= 0 for value in shape) or shape[1] != 3:
        raise ValueError("input_shape must be positive NCHW with three channels")
    maximum_error = (
        float(tolerance)
        if tolerance is not None
        else (1.0e-5 if normalized_format == "torchscript" else 2.0e-4)
    )
    if not math.isfinite(maximum_error) or maximum_error < 0:
        raise ValueError("tolerance must be finite and non-negative")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = (
        tempfile.gettempdir()
        if normalized_format == "torchscript"
        else destination.parent
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=f".{normalized_format}.tmp", dir=temporary_directory
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    original_training = bool(model.training)
    try:
        original_device = next(model.parameters()).device
    except StopIteration:
        original_device = torch.device("cpu")
    # A deterministic non-constant input exposes coordinate/order mistakes.
    number_of_values = int(np.prod(shape))
    example = torch.linspace(
        0.0, 1.0, steps=number_of_values, dtype=torch.float32, device=device
    ).reshape(shape)
    try:
        model.to(device)
        model.eval()
        with torch.inference_mode():
            reference = _extract_logits(model(example)).detach().to("cpu").numpy()
        if normalized_format == "torchscript":
            traced = torch.jit.trace(model, example, strict=True)
            traced.save(str(temporary))
        else:
            try:
                torch.onnx.export(
                    model,
                    example,
                    str(temporary),
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=["image"],
                    output_names=["logits"],
                    dynamic_axes={
                        "image": {0: "batch", 2: "height", 3: "width"},
                        "logits": {0: "batch", 2: "height", 3: "width"},
                    },
                    dynamo=False,
                )
            except TypeError:  # Older torch has no ``dynamo`` argument.
                torch.onnx.export(
                    model,
                    example,
                    str(temporary),
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=["image"],
                    output_names=["logits"],
                    dynamic_axes={
                        "image": {0: "batch", 2: "height", 3: "width"},
                        "logits": {0: "batch", 2: "height", 3: "width"},
                    },
                )
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise WireExportError("wire export produced no artifact bytes")
        if verify:
            candidate = _run_exported(temporary, normalized_format, example)
            if candidate.shape != reference.shape:
                raise WireExportError(
                    f"wire export output shape {candidate.shape} differs from eager {reference.shape}"
                )
            absolute = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
            max_abs = float(absolute.max(initial=0.0))
            mean_abs = float(absolute.mean()) if absolute.size else 0.0
            if not math.isfinite(max_abs) or max_abs > maximum_error:
                raise WireExportError(
                    f"wire {normalized_format} max_abs_error {max_abs:.8g} exceeds {maximum_error:.8g}"
                )
        else:
            max_abs = 0.0
            mean_abs = 0.0
        try:
            os.replace(temporary, destination)
        except OSError:
            # torch.jit.save cannot write non-ASCII (e.g. Chinese) paths on
            # Windows; the artifact is staged in the ASCII %TEMP% directory and
            # copied here.  Cross-volume replacement is also handled here.
            shutil.copy2(temporary, destination)
            temporary.unlink(missing_ok=True)
    except WireExportError:
        raise
    except Exception as exc:
        raise WireExportError(
            f"cannot export wire model as {normalized_format}: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
        model.to(original_device)
        model.train(original_training)
    digest = _sha256_file(destination)
    return WireExportReport(
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


def verify_wire_export(
    model: Any,
    path: str | Path,
    *,
    format: Literal["torchscript", "onnx"],
    inputs: Any,
    tolerance: float | None = None,
) -> dict[str, float | bool]:
    """Independently compare an existing export with eager model outputs."""

    torch = _require_torch()
    maximum_error = tolerance if tolerance is not None else (
        1.0e-5 if format == "torchscript" else 2.0e-4
    )
    if not math.isfinite(maximum_error) or maximum_error < 0:
        raise ValueError("tolerance must be finite and non-negative")
    model.eval()
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    with torch.inference_mode():
        reference = (
            _extract_logits(model(inputs.to(model_device))).detach().to("cpu").numpy()
        )
    resolved_path = Path(path).resolve()
    if format == "torchscript":
        # torch.jit.load cannot open non-ASCII (e.g. Chinese) paths on Windows.
        descriptor, staged_name = tempfile.mkstemp(
            prefix=".wire-verify-", suffix=".pt", dir=tempfile.gettempdir()
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
        raise WireExportError("export output shape differs from eager output")
    absolute = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    max_abs = float(absolute.max(initial=0.0))
    mean_abs = float(absolute.mean()) if absolute.size else 0.0
    passed = math.isfinite(max_abs) and max_abs <= maximum_error
    if not passed:
        raise WireExportError(
            f"wire {format} max_abs_error {max_abs:.8g} exceeds {maximum_error:.8g}"
        )
    return {
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "tolerance": float(maximum_error),
        "passed": True,
    }


def _run_exported(path: Path, format: str, inputs: Any) -> np.ndarray:
    torch = _require_torch()
    if format == "torchscript":
        module = torch.jit.load(str(path), map_location="cpu")
        module.eval()
        with torch.inference_mode():
            result = _extract_logits(module(inputs.detach().to("cpu")))
        return result.detach().to("cpu").numpy()
    try:
        runtime = importlib.import_module("onnxruntime")
    except ImportError as exc:
        raise WireExportError(
            "onnxruntime is required to verify a wire ONNX export"
        ) from exc
    session = runtime.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: inputs.detach().to("cpu").numpy()})[0]
    return np.asarray(result)


def _normalise_supervision(
    values: Any,
    target: Any,
    *,
    ignore: Any | None,
    valid: Any | None,
) -> tuple[Any, Any, Any]:
    torch = _require_torch()
    if not torch.is_tensor(values) or not torch.is_tensor(target):
        raise TypeError("wire logits/probabilities and target must be torch tensors")
    if values.ndim == 3:
        values = values.unsqueeze(1)
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if values.ndim != 4 or target.ndim != 4 or values.shape != target.shape:
        raise ValueError("wire logits/probabilities and target must have matching NCHW shape")
    target = target.to(device=values.device, dtype=values.dtype).clamp(0.0, 1.0)
    if valid is not None and ignore is not None:
        supplied_valid = valid.to(device=values.device, dtype=values.dtype)
        supplied_ignore = ignore.to(device=values.device, dtype=values.dtype)
        if supplied_valid.ndim == 3:
            supplied_valid = supplied_valid.unsqueeze(1)
        if supplied_ignore.ndim == 3:
            supplied_ignore = supplied_ignore.unsqueeze(1)
        if supplied_valid.shape != values.shape or supplied_ignore.shape != values.shape:
            raise ValueError("wire valid/ignore masks must match logits shape")
        if not torch.allclose(supplied_valid, 1.0 - supplied_ignore, atol=1.0e-6):
            raise ValueError("wire valid and ignore masks contradict each other")
        valid_mask = supplied_valid
    elif valid is not None:
        valid_mask = valid.to(device=values.device, dtype=values.dtype)
    elif ignore is not None:
        ignored = ignore.to(device=values.device, dtype=values.dtype)
        if ignored.ndim == 3:
            ignored = ignored.unsqueeze(1)
        valid_mask = 1.0 - ignored
    else:
        valid_mask = torch.ones_like(target)
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    if valid_mask.shape != values.shape:
        raise ValueError("wire valid/ignore mask must match logits shape")
    valid_mask = valid_mask.clamp(0.0, 1.0)
    return values, target, valid_mask


def _soft_erode(inputs: Any) -> Any:
    functional = _require_torch().nn.functional
    vertical = -functional.max_pool2d(-inputs, (3, 1), stride=1, padding=(1, 0))
    horizontal = -functional.max_pool2d(-inputs, (1, 3), stride=1, padding=(0, 1))
    return _require_torch().minimum(vertical, horizontal)


def _soft_dilate(inputs: Any) -> Any:
    return _require_torch().nn.functional.max_pool2d(inputs, 3, stride=1, padding=1)


def _soft_open(inputs: Any, valid: Any) -> Any:
    return _soft_dilate(_soft_erode(inputs) * valid) * valid


def _masked_sample_mean(values: Any, selected: Any, reference: Any) -> Any:
    selected_float = selected.to(values.dtype)
    return (
        (values * selected_float).sum() / selected_float.sum().clamp_min(1.0)
        if bool(selected.any())
        else reference.sum() * 0.0
    )


def _batch_tensors(batch: Mapping[str, Any], device: Any) -> tuple[Any, Any, Any]:
    try:
        image = batch["image"].to(device)
        target = batch["target"].to(device)
        ignore = batch["ignore"].to(device)
    except (KeyError, AttributeError) as exc:
        raise ValueError("wire batch requires tensor image/target/ignore fields") from exc
    return image, target, ignore


def _extract_logits(output: Any) -> Any:
    torch = _require_torch()
    if torch.is_tensor(output):
        logits = output
    elif isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        logits = output[0]
    elif isinstance(output, dict) and torch.is_tensor(output.get("logits")):
        logits = output["logits"]
    else:
        raise TorchWireRuntimeError("wire model must return a tensor or a 'logits' tensor")
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise TorchWireRuntimeError("wire model output must have shape N,1,H,W")
    return logits


def _connected_component_count(mask: np.ndarray) -> int:
    if mask.ndim != 2:
        raise ValueError("connectivity mask must be two-dimensional")
    remaining = np.asarray(mask, dtype=bool).copy()
    height, width = remaining.shape
    count = 0
    for y, x in np.argwhere(remaining):
        y = int(y)
        x = int(x)
        if not remaining[y, x]:
            continue
        count += 1
        remaining[y, x] = False
        stack = [(y, x)]
        while stack:
            current_y, current_x = stack.pop()
            for next_y in range(max(0, current_y - 1), min(height, current_y + 2)):
                for next_x in range(max(0, current_x - 1), min(width, current_x + 2)):
                    if remaining[next_y, next_x]:
                        remaining[next_y, next_x] = False
                        stack.append((next_y, next_x))
    return count


def _json_mapping(value: Any, name: str, *, require_nonempty: bool) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WireCheckpointError(f"{name} must be a JSON object")
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise WireCheckpointError(f"{name} must be finite JSON data") from exc
    if not isinstance(decoded, dict) or (require_nonempty and not decoded):
        qualifier = "non-empty " if require_nonempty else ""
        raise WireCheckpointError(f"{name} must be a {qualifier}JSON object")
    return decoded


def _finite_metrics(metrics: Any) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        raise WireCheckpointError("checkpoint metrics must be an object")
    output: dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key:
            raise WireCheckpointError("checkpoint metric names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise WireCheckpointError(f"checkpoint metric {key!r} must be finite")
        output[key] = float(value)
    return output


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


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
        raise TorchWireRuntimeError(
            "PyTorch is required for wire model training/inference; install the training runtime"
        ) from exc


__all__ = [
    "CHECKPOINT_SCHEMA",
    "EXPORT_SCHEMA",
    "LoadedWireCheckpoint",
    "TinyUNet",
    "TorchWireRuntimeError",
    "WireCheckpointError",
    "WireCheckpointInfo",
    "WireExportError",
    "WireExportReport",
    "WireLossConfig",
    "build_tiny_unet",
    "compute_wire_loss",
    "export_wire_model",
    "load_wire_checkpoint",
    "masked_wire_loss",
    "save_wire_checkpoint",
    "soft_skeletonize",
    "verify_wire_export",
    "wire_centerline_metrics",
    "wire_eval_step",
    "wire_train_step",
]
