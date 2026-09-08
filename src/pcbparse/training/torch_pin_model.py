from __future__ import annotations

"""Small, dependency-lazy PyTorch model for Pin keypoint estimation.

The materialized :class:`~pcbparse.training.torch_component_pin.PinKeypointDataset`
contract is deliberately mirrored here: one heatmap logit, two sub-pixel
offsets and a two-dimensional ``base -> tip`` orientation unit vector at an
output stride of four.  Importing this module never imports torch; the optional
runtime is loaded only when a model, loss, step, checkpoint, or export is used.
"""

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_CHECKPOINT_SCHEMA = "pcbparse-pin-heatmap-checkpoint-v1"
_HEX = frozenset("0123456789abcdef")
_MODEL_CLASS: type[Any] | None = None
_EXPORT_WRAPPER_CLASS: type[Any] | None = None


class TorchPinDependencyError(RuntimeError):
    """Raised when an explicitly requested optional ML runtime is unavailable."""


class PinCheckpointError(RuntimeError):
    """Raised when a checkpoint is corrupt or has untrusted lineage."""


class PinExportError(RuntimeError):
    """Raised when an exported graph cannot reproduce eager model outputs."""


@dataclass(frozen=True, slots=True)
class PinModelConfig:
    input_channels: int = 3
    base_channels: int = 16
    output_stride: int = 4

    def __post_init__(self) -> None:
        if self.input_channels != 3:
            raise ValueError("PinModelConfig.input_channels must be 3 (RGB)")
        if type(self.base_channels) is not int or self.base_channels < 2:
            raise ValueError("PinModelConfig.base_channels must be an integer >= 2")
        if self.output_stride != 4:
            raise ValueError(
                "TinyPinHeatmapNet implements output_stride=4, matching PinKeypointDataset"
            )


@dataclass(frozen=True, slots=True)
class PinLossConfig:
    heatmap_loss: str = "focal"
    heatmap_weight: float = 1.0
    offset_weight: float = 1.0
    orientation_weight: float = 0.5
    focal_alpha: float = 2.0
    focal_beta: float = 4.0
    smooth_l1_beta: float = 1.0
    # Count-aware regularization: penalize the total soft heatmap response
    # beyond ``count_target_multiplier * gt_count`` per view.  This directly
    # discourages the diffuse false-positive peaks observed on validation.
    count_loss_weight: float = 0.0
    count_target_multiplier: float = 4.0

    def __post_init__(self) -> None:
        if self.heatmap_loss not in {"focal", "bce", "weighted_bce"}:
            raise ValueError("heatmap_loss must be 'focal', 'bce', or 'weighted_bce'")
        for name in ("heatmap_weight", "offset_weight", "orientation_weight"):
            value = getattr(self, name)
            if not _finite_number(value) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("focal_alpha", "focal_beta", "smooth_l1_beta"):
            value = getattr(self, name)
            if not _finite_number(value) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not _finite_number(self.count_loss_weight) or float(self.count_loss_weight) < 0:
            raise ValueError("count_loss_weight must be finite and non-negative")
        if not _finite_number(self.count_target_multiplier) or float(self.count_target_multiplier) <= 0:
            raise ValueError("count_target_multiplier must be finite and positive")


class TinyPinHeatmapNet:
    """Dependency-lazy constructor returning a real ``torch.nn.Module``.

    The implementation is a compact U-Net/hourglass: two stride-two encoder
    stages, a stride-eight bottleneck, and a skip-connected decoder at stride
    four.  Returning the actual module from ``__new__`` keeps core imports safe
    on machines that intentionally omit PyTorch.
    """

    def __new__(
        cls,
        config: PinModelConfig | Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> Any:
        if config is not None and overrides:
            raise ValueError("pass either config or keyword overrides, not both")
        if config is None:
            resolved = PinModelConfig(**overrides)
        elif isinstance(config, PinModelConfig):
            resolved = config
        elif isinstance(config, Mapping):
            resolved = PinModelConfig(**dict(config))
        else:
            raise TypeError("config must be PinModelConfig, a mapping, or None")
        return _pin_model_class()(resolved)


def split_pin_outputs(value: Any) -> dict[str, Any]:
    """Normalize eager, TorchScript, or ONNX-like outputs to three tensors."""

    if isinstance(value, Mapping):
        required = ("heatmap_logits", "offset", "orientation")
        if any(name not in value for name in required):
            raise ValueError(f"Pin model output mapping must contain {required}")
        result = {name: value[name] for name in required}
    elif isinstance(value, (tuple, list)) and len(value) == 3:
        result = {
            "heatmap_logits": value[0],
            "offset": value[1],
            "orientation": value[2],
        }
    else:
        raise TypeError("Pin model output must be a mapping or a three-tensor sequence")
    _validate_output_shapes(result)
    return result


def pin_losses(
    outputs: Mapping[str, Any] | Sequence[Any],
    targets: Mapping[str, Any],
    config: PinLossConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute heatmap, masked offset and masked orientation losses."""

    torch = _require_torch()
    functional = torch.nn.functional
    resolved = _loss_config(config)
    prediction = split_pin_outputs(outputs)
    required = ("heatmap", "offset", "orientation", "regression_mask")
    if any(name not in targets for name in required):
        raise ValueError(f"Pin targets must contain {required}")
    heatmap = targets["heatmap"].to(
        device=prediction["heatmap_logits"].device,
        dtype=prediction["heatmap_logits"].dtype,
    )
    offset = targets["offset"].to(
        device=prediction["offset"].device, dtype=prediction["offset"].dtype
    )
    orientation = targets["orientation"].to(
        device=prediction["orientation"].device,
        dtype=prediction["orientation"].dtype,
    )
    mask = targets["regression_mask"].to(
        device=prediction["offset"].device, dtype=prediction["offset"].dtype
    )
    expected = {
        "heatmap": prediction["heatmap_logits"].shape,
        "offset": prediction["offset"].shape,
        "orientation": prediction["orientation"].shape,
    }
    actual = {
        "heatmap": heatmap.shape,
        "offset": offset.shape,
        "orientation": orientation.shape,
    }
    if expected != actual or mask.shape != heatmap.shape:
        raise ValueError(
            f"Pin target/output shapes differ: expected={expected}, actual={actual}, "
            f"mask={tuple(mask.shape)}"
        )

    if resolved.heatmap_loss == "bce":
        # Use weighted BCE with higher weight for positive samples
        # The heatmap is very sparse (~1% positive), so we need higher weight
        pos_weight = torch.full_like(heatmap, 100.0)  # 100x weight for positive samples
        heatmap_loss = functional.binary_cross_entropy_with_logits(
            prediction["heatmap_logits"], heatmap, pos_weight=pos_weight, reduction="mean"
        )
    elif resolved.heatmap_loss == "weighted_bce":
        # Alternative: weighted BCE with dynamic weight based on positive ratio
        pos_count = heatmap.sum().clamp_min(1.0)
        neg_count = (1.0 - heatmap).sum().clamp_min(1.0)
        pos_weight_value = neg_count / pos_count  # Balance the classes
        pos_weight = torch.full_like(heatmap, min(pos_weight_value.item(), 100.0))
        heatmap_loss = functional.binary_cross_entropy_with_logits(
            prediction["heatmap_logits"], heatmap, pos_weight=pos_weight, reduction="mean"
        )
    else:
        heatmap_loss = _modified_focal_loss(
            prediction["heatmap_logits"],
            heatmap,
            alpha=resolved.focal_alpha,
            beta=resolved.focal_beta,
        )

    expanded_mask = mask.expand_as(offset)
    offset_values = functional.smooth_l1_loss(
        prediction["offset"], offset, reduction="none", beta=resolved.smooth_l1_beta
    )
    offset_denominator = expanded_mask.sum().clamp_min(1.0)
    offset_loss = (offset_values * expanded_mask).sum() / offset_denominator

    target_norm = torch.linalg.vector_norm(orientation, dim=1, keepdim=True)
    valid_orientation = mask * (target_norm > 1e-8).to(mask.dtype)
    predicted_unit = functional.normalize(
        prediction["orientation"], p=2, dim=1, eps=1e-8
    )
    target_unit = functional.normalize(orientation, p=2, dim=1, eps=1e-8)
    cosine = (predicted_unit * target_unit).sum(dim=1, keepdim=True)
    orientation_loss = (
        ((1.0 - cosine).clamp_min(0.0) * valid_orientation).sum()
        / valid_orientation.sum().clamp_min(1.0)
    )
    count_loss = torch.zeros((), dtype=prediction["heatmap_logits"].dtype, device=prediction["heatmap_logits"].device)
    if resolved.count_loss_weight > 0:
        soft_count = functional.sigmoid(prediction["heatmap_logits"]).sum(
            dim=(1, 2, 3)
        )
        gt_count = mask.sum(dim=(1, 2, 3)).to(soft_count.dtype)
        excess = (soft_count - resolved.count_target_multiplier * gt_count).clamp_min(0.0)
        spatial_cells = max(
            1,
            int(prediction["heatmap_logits"].shape[2])
            * int(prediction["heatmap_logits"].shape[3]),
        )
        count_loss = (excess / spatial_cells).mean()
    total = (
        resolved.heatmap_weight * heatmap_loss
        + resolved.offset_weight * offset_loss
        + resolved.orientation_weight * orientation_loss
        + resolved.count_loss_weight * count_loss
    )
    return {
        "loss": total,
        "heatmap_loss": heatmap_loss,
        "offset_loss": offset_loss,
        "orientation_loss": orientation_loss,
        "count_loss": count_loss,
        "positive_count": mask.sum(),
    }


def pin_train_step(
    model: Any,
    batch: tuple[Any, Mapping[str, Any]],
    optimizer: Any,
    *,
    device: str | Any = "cpu",
    loss_config: PinLossConfig | Mapping[str, Any] | None = None,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    """Run one optimizer step and return detached scalar evidence."""

    torch = _require_torch()
    if max_grad_norm is not None and (
        not _finite_number(max_grad_norm) or float(max_grad_norm) <= 0
    ):
        raise ValueError("max_grad_norm must be finite and positive")
    images, raw_targets = batch
    model.train()
    images = images.to(device=device, dtype=torch.float32)
    targets = _move_dense_targets(raw_targets, device)
    optimizer.zero_grad(set_to_none=True)
    losses = pin_losses(model(images), targets, loss_config)
    losses["loss"].backward()
    if max_grad_norm is not None:
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(max_grad_norm)
        )
    else:
        gradient_norm = _gradient_l2_norm(model.parameters(), torch)
    optimizer.step()
    result = _scalar_losses(losses)
    result["gradient_norm"] = float(gradient_norm.detach().cpu())
    return result


def pin_eval_step(
    model: Any,
    batch: tuple[Any, Mapping[str, Any]],
    *,
    device: str | Any = "cpu",
    loss_config: PinLossConfig | Mapping[str, Any] | None = None,
    score_threshold: float = 0.3,
    heatmap_stride: int = 4,
) -> dict[str, float]:
    """Run one validation step with loss, PCK@2/3/5 and count metrics."""

    torch = _require_torch()
    images, raw_targets = batch
    model.eval()
    with torch.no_grad():
        images = images.to(device=device, dtype=torch.float32)
        targets = _move_dense_targets(raw_targets, device)
        outputs = split_pin_outputs(model(images))
        losses = pin_losses(outputs, targets, loss_config)
        metrics = compute_pin_metrics(
            outputs,
            raw_targets,
            heatmap_stride=heatmap_stride,
            score_threshold=score_threshold,
        )
    return {**_scalar_losses(losses), **metrics}


def decode_pin_heatmaps(
    outputs: Mapping[str, Any] | Sequence[Any],
    *,
    score_threshold: float = 0.3,
    nms_kernel: int = 3,
    nms_radius: float = 1.0,
    max_detections: int = 256,
    box_rects: Sequence[tuple[float, float, float, float]] | None = None,
    box_margin: float = 0.0,
    box_prior_mode: str = "filled_rect_v1",
) -> list[list[dict[str, Any]]]:
    """Decode local maxima, sub-pixel offsets, and normalized orientations.

    Coordinates are returned in heatmap units.  A caller that works in image
    pixels must multiply them by the dataset/model output stride (four).
    """

    torch = _require_torch()
    if not _finite_number(score_threshold) or not 0 <= float(score_threshold) <= 1:
        raise ValueError("score_threshold must be in [0, 1]")
    if type(nms_kernel) is not int or nms_kernel <= 0 or nms_kernel % 2 != 1:
        raise ValueError("nms_kernel must be a positive odd integer")
    if not _finite_number(nms_radius) or float(nms_radius) < 0:
        raise ValueError("nms_radius must be finite and non-negative")
    if type(max_detections) is not int or max_detections <= 0:
        raise ValueError("max_detections must be a positive integer")
    if not _finite_number(box_margin) or float(box_margin) < 0:
        raise ValueError("box_margin must be finite and non-negative")
    if box_prior_mode not in {"filled_rect_v1", "boundary_band_v2"}:
        raise ValueError(
            "box_prior_mode must be filled_rect_v1 or boundary_band_v2"
        )
    value = split_pin_outputs(outputs)
    logits = value["heatmap_logits"]
    scores = torch.sigmoid(logits)
    pooled = torch.nn.functional.max_pool2d(
        scores, kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2
    )
    maxima = (scores >= pooled) & (scores >= float(score_threshold))
    if box_rects is not None and box_margin > 0 and len(box_rects) != int(scores.shape[0]):
        raise ValueError("box_rects must contain one rectangle per batch item")
    result: list[list[dict[str, Any]]] = []
    for batch_index in range(scores.shape[0]):
        ys, xs = torch.where(maxima[batch_index, 0])
        if xs.numel() == 0:
            result.append([])
            continue
        candidate_scores = scores[batch_index, 0, ys, xs]
        order = torch.argsort(candidate_scores, descending=True, stable=True)
        detections: list[dict[str, Any]] = []
        for raw_index in order.tolist():
            anchor_x = int(xs[raw_index])
            anchor_y = int(ys[raw_index])
            offset_x = float(value["offset"][batch_index, 0, anchor_y, anchor_x])
            offset_y = float(value["offset"][batch_index, 1, anchor_y, anchor_x])
            point_x = anchor_x + offset_x
            point_y = anchor_y + offset_y
            if box_rects is not None and box_margin > 0:
                x1, y1, x2, y2 = box_rects[batch_index]
                distance = (
                    _point_to_rect_boundary_distance
                    if box_prior_mode == "boundary_band_v2"
                    else _point_to_rect_distance
                )
                if (
                    distance(
                        point_x,
                        point_y,
                        x1,
                        y1,
                        x2,
                        y2,
                    )
                    > float(box_margin)
                ):
                    continue
            if any(
                math.hypot(point_x - item["x"], point_y - item["y"])
                <= float(nms_radius)
                for item in detections
            ):
                continue
            unit_x = float(
                value["orientation"][batch_index, 0, anchor_y, anchor_x]
            )
            unit_y = float(
                value["orientation"][batch_index, 1, anchor_y, anchor_x]
            )
            magnitude = math.hypot(unit_x, unit_y)
            if magnitude > 1e-8:
                unit_x, unit_y = unit_x / magnitude, unit_y / magnitude
            else:
                unit_x, unit_y = 0.0, 0.0
            detections.append(
                {
                    "x": point_x,
                    "y": point_y,
                    "score": float(candidate_scores[raw_index]),
                    "orientation": (unit_x, unit_y),
                    "anchor": (anchor_x, anchor_y),
                }
            )
            if len(detections) >= max_detections:
                break
        result.append(detections)
    return result


def compute_pin_metrics(
    outputs: Mapping[str, Any] | Sequence[Any],
    targets: Mapping[str, Any],
    *,
    heatmap_stride: int = 4,
    score_threshold: float = 0.08,
    pck_thresholds: Iterable[float] = (2.0, 3.0, 5.0),
    nms_kernel: int = 3,
    nms_radius: float = 1.0,
    max_detections: int = 256,
    box_rects: Sequence[tuple[float, float, float, float]] | None = None,
    box_margin_px: float = 0.0,
    box_prior_mode: str = "filled_rect_v1",
) -> dict[str, float]:
    """Compute anonymous-keypoint PCK and count quality for one batch.

    Thresholds are input-image pixels (not heatmap cells).  Since pins have no
    semantic identity at this stage, each threshold uses maximum-cardinality
    one-to-one geometric matching.
    """

    if type(heatmap_stride) is not int or heatmap_stride <= 0:
        raise ValueError("heatmap_stride must be a positive integer")
    thresholds = tuple(float(value) for value in pck_thresholds)
    if not thresholds or any(not math.isfinite(value) or value <= 0 for value in thresholds):
        raise ValueError("pck_thresholds must contain finite positive values")
    if box_rects is not None:
        normalized = split_pin_outputs(outputs)
        if len(box_rects) != int(normalized["heatmap_logits"].shape[0]):
            raise ValueError("box_rects must contain one rectangle per batch item")
    if not _finite_number(box_margin_px) or float(box_margin_px) < 0:
        raise ValueError("box_margin_px must be finite and non-negative")
    box_rects_heatmap = (
        None
        if box_rects is None
        else [
            tuple(float(value) / heatmap_stride for value in rect)
            for rect in box_rects
        ]
    )
    predictions = decode_pin_heatmaps(
        outputs,
        score_threshold=score_threshold,
        nms_kernel=nms_kernel,
        nms_radius=nms_radius,
        max_detections=max_detections,
        box_rects=box_rects_heatmap,
        box_margin=box_margin_px / heatmap_stride,
        box_prior_mode=box_prior_mode,
    )
    ground_truth = _ground_truth_points(targets, len(predictions))
    total_gt = sum(len(items) for items in ground_truth)
    total_pred = sum(len(items) for items in predictions)
    absolute_count_error = sum(
        abs(len(predicted) - len(expected))
        for predicted, expected in zip(predictions, ground_truth)
    )
    exact_counts = sum(
        len(predicted) == len(expected)
        for predicted, expected in zip(predictions, ground_truth)
    )
    output: dict[str, float] = {
        "gt_count": float(total_gt),
        "pred_count": float(total_pred),
        "count_mae": float(absolute_count_error / max(1, len(predictions))),
        "count_exact_accuracy": float(exact_counts / max(1, len(predictions))),
    }
    matches_by_threshold: dict[float, int] = {}
    for threshold in thresholds:
        threshold_heatmap = threshold / heatmap_stride
        matched = sum(
            _maximum_threshold_matches(predicted, expected, threshold_heatmap)
            for predicted, expected in zip(predictions, ground_truth)
        )
        matches_by_threshold[threshold] = matched
        suffix = str(int(threshold)) if threshold.is_integer() else str(threshold)
        output[f"pck@{suffix}"] = float(
            matched / total_gt if total_gt else (1.0 if total_pred == 0 else 0.0)
        )
    largest = max(thresholds)
    largest_matches = matches_by_threshold[largest]
    output["detection_precision@max_pck"] = float(
        largest_matches / total_pred if total_pred else (1.0 if total_gt == 0 else 0.0)
    )
    output["detection_recall@max_pck"] = float(
        largest_matches / total_gt if total_gt else 1.0
    )
    return output


def save_pin_checkpoint(
    path: str | Path,
    model: Any,
    *,
    optimizer: Any | None = None,
    epoch: int = 0,
    global_step: int = 0,
    lineage: Mapping[str, Any],
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically save weights plus immutable dataset-lineage evidence."""

    torch = _require_torch()
    target = Path(path).expanduser().resolve()
    if type(epoch) is not int or epoch < 0 or type(global_step) is not int or global_step < 0:
        raise ValueError("epoch and global_step must be non-negative integers")
    normalized_lineage = _validated_lineage(lineage)
    model_config = getattr(model, "pin_model_config", None)
    if not isinstance(model_config, Mapping):
        raise ValueError("model does not expose pin_model_config")
    payload = {
        "schema": _CHECKPOINT_SCHEMA,
        "task": "pin",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "torch_version": str(torch.__version__),
        "model_config": dict(model_config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "lineage": normalized_lineage,
        "lineage_sha256": _sha256_json(normalized_lineage),
        "metrics": _json_mapping(metrics or {}, "metrics"),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(target),
        "sha256": _sha256_file(target),
        "lineage_sha256": payload["lineage_sha256"],
        "epoch": epoch,
        "global_step": global_step,
    }


def load_pin_checkpoint(
    path: str | Path,
    *,
    model: Any | None = None,
    optimizer: Any | None = None,
    expected_lineage: Mapping[str, Any] | None = None,
    map_location: str | Any = "cpu",
    strict: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Load a checkpoint only after schema and lineage hashes are verified."""

    torch = _require_torch()
    source = Path(path).expanduser().resolve(strict=True)
    try:
        try:
            payload = torch.load(source, map_location=map_location, weights_only=True)
        except TypeError:  # pragma: no cover - compatibility with older torch
            payload = torch.load(source, map_location=map_location)
    except Exception as exc:
        raise PinCheckpointError(f"cannot load Pin checkpoint: {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != _CHECKPOINT_SCHEMA:
        raise PinCheckpointError("unsupported Pin checkpoint schema")
    if payload.get("task") != "pin":
        raise PinCheckpointError("checkpoint task is not pin")
    lineage = payload.get("lineage")
    try:
        normalized_lineage = _validated_lineage(lineage)
    except (TypeError, ValueError) as exc:
        raise PinCheckpointError(f"invalid checkpoint lineage: {exc}") from exc
    if payload.get("lineage_sha256") != _sha256_json(normalized_lineage):
        raise PinCheckpointError("checkpoint lineage SHA-256 does not match its payload")
    if expected_lineage is not None:
        expected = _validated_lineage(expected_lineage)
        if _sha256_json(expected) != payload["lineage_sha256"]:
            raise PinCheckpointError("checkpoint lineage differs from expected lineage")
    config = payload.get("model_config")
    if not isinstance(config, dict):
        raise PinCheckpointError("checkpoint lacks model_config")
    if model is None:
        model = TinyPinHeatmapNet(config)
    try:
        model.load_state_dict(payload["model_state_dict"], strict=strict)
        if optimizer is not None:
            optimizer_state = payload.get("optimizer_state_dict")
            if optimizer_state is None:
                raise PinCheckpointError("checkpoint has no optimizer state")
            optimizer.load_state_dict(optimizer_state)
    except PinCheckpointError:
        raise
    except Exception as exc:
        raise PinCheckpointError(f"checkpoint state is incompatible: {exc}") from exc
    return model, payload


def export_pin_model(
    model: Any,
    path: str | Path,
    sample_inputs: Any | Sequence[Any],
    *,
    export_format: str = "torchscript",
    opset: int = 17,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Export TorchScript or ONNX and numerically verify every supplied sample."""

    torch = _require_torch()
    if export_format not in {"torchscript", "onnx"}:
        raise ValueError("export_format must be 'torchscript' or 'onnx'")
    if type(opset) is not int or opset < 11:
        raise ValueError("opset must be an integer >= 11")
    if not _finite_number(tolerance) or float(tolerance) < 0:
        raise ValueError("tolerance must be finite and non-negative")
    samples = _normalize_export_samples(sample_inputs, torch)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    wrapper_class = _export_wrapper_class()
    training = bool(model.training)
    parameters = list(model.parameters())
    original_device = parameters[0].device if parameters else torch.device("cpu")
    wrapper = wrapper_class(model.to("cpu").eval()).eval()
    cpu_samples = [sample.detach().to(device="cpu", dtype=torch.float32) for sample in samples]
    try:
        with torch.no_grad():
            references = [tuple(item.detach().cpu() for item in wrapper(sample)) for sample in cpu_samples]
        if export_format == "torchscript":
            exported = torch.jit.trace(wrapper, cpu_samples[0], strict=True, check_trace=True)
            exported.save(str(target))
            loaded = torch.jit.load(str(target), map_location="cpu").eval()
            with torch.no_grad():
                candidates = [tuple(item.detach().cpu() for item in loaded(sample)) for sample in cpu_samples]
            backend = "torch.jit.trace"
        else:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise TorchPinDependencyError(
                    "ONNX verification requires onnxruntime"
                ) from exc
            torch.onnx.export(
                wrapper,
                cpu_samples[0],
                str(target),
                input_names=["image"],
                output_names=["heatmap_logits", "offset", "orientation"],
                dynamic_axes={
                    "image": {0: "batch", 2: "height", 3: "width"},
                    "heatmap_logits": {0: "batch", 2: "heatmap_height", 3: "heatmap_width"},
                    "offset": {0: "batch", 2: "heatmap_height", 3: "heatmap_width"},
                    "orientation": {0: "batch", 2: "heatmap_height", 3: "heatmap_width"},
                },
                opset_version=opset,
                do_constant_folding=True,
                dynamo=False,
            )
            session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
            candidates = []
            for sample in cpu_samples:
                arrays = session.run(None, {"image": sample.numpy()})
                candidates.append(tuple(torch.from_numpy(item) for item in arrays))
            backend = f"onnxruntime-{ort.__version__}"
        maximum_error = 0.0
        for reference, candidate in zip(references, candidates):
            if len(reference) != len(candidate):
                raise PinExportError("export output count differs from eager model")
            for expected, actual in zip(reference, candidate):
                if expected.shape != actual.shape:
                    raise PinExportError(
                        f"export output shape differs: {tuple(expected.shape)} != {tuple(actual.shape)}"
                    )
                if expected.numel():
                    maximum_error = max(
                        maximum_error,
                        float((expected - actual.to(expected.dtype)).abs().max()),
                    )
        if not math.isfinite(maximum_error) or maximum_error > float(tolerance):
            target.unlink(missing_ok=True)
            raise PinExportError(
                f"export maximum absolute error {maximum_error:.8g} exceeds {float(tolerance):.8g}"
            )
    finally:
        model.to(original_device)
        model.train(training)
    return {
        "path": str(target),
        "format": export_format,
        "export_backend": backend,
        "sample_count": len(samples),
        "maximum_absolute_error": maximum_error,
        "tolerance": float(tolerance),
        "sha256": _sha256_file(target),
    }


def _pin_model_class() -> type[Any]:
    global _MODEL_CLASS
    if _MODEL_CLASS is not None:
        return _MODEL_CLASS
    torch = _require_torch()
    nn = torch.nn
    functional = torch.nn.functional

    def groups(channels: int) -> int:
        for value in (8, 4, 2):
            if channels % value == 0:
                return value
        return 1

    class ConvBlock(nn.Module):
        def __init__(self, input_channels: int, output_channels: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                nn.GroupNorm(groups(output_channels), output_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                nn.GroupNorm(groups(output_channels), output_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, value: Any) -> Any:
            return self.layers(value)

    class _TinyPinHeatmapNet(nn.Module):
        def __init__(self, config: PinModelConfig) -> None:
            super().__init__()
            base = config.base_channels
            self.pin_model_config = asdict(config)
            self.stem = ConvBlock(config.input_channels, base)
            self.down_one = nn.Sequential(
                nn.Conv2d(base, base * 2, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(groups(base * 2), base * 2),
                nn.SiLU(inplace=True),
                ConvBlock(base * 2, base * 2),
            )
            self.down_two = nn.Sequential(
                nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(groups(base * 4), base * 4),
                nn.SiLU(inplace=True),
                ConvBlock(base * 4, base * 4),
            )
            self.bottleneck = nn.Sequential(
                nn.Conv2d(base * 4, base * 8, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(groups(base * 8), base * 8),
                nn.SiLU(inplace=True),
                ConvBlock(base * 8, base * 8),
            )
            self.decoder = ConvBlock(base * 12, base * 4)
            self.head = nn.Conv2d(base * 4, 5, 1)
            nn.init.constant_(self.head.bias[0], -2.19)

        def forward(self, image: Any) -> dict[str, Any]:
            first = self.stem(image)
            second = self.down_one(first)
            skip = self.down_two(second)
            bottom = self.bottleneck(skip)
            bottom = functional.interpolate(
                bottom, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            features = self.decoder(torch.cat((skip, bottom), dim=1))
            raw = self.head(features)
            return {
                "heatmap_logits": raw[:, 0:1],
                "offset": torch.sigmoid(raw[:, 1:3]),
                "orientation": functional.normalize(raw[:, 3:5], p=2, dim=1, eps=1e-8),
            }

    _MODEL_CLASS = _TinyPinHeatmapNet
    return _MODEL_CLASS


def _export_wrapper_class() -> type[Any]:
    global _EXPORT_WRAPPER_CLASS
    if _EXPORT_WRAPPER_CLASS is not None:
        return _EXPORT_WRAPPER_CLASS
    torch = _require_torch()

    class _PinExportWrapper(torch.nn.Module):
        def __init__(self, model: Any) -> None:
            super().__init__()
            self.model = model

        def forward(self, image: Any) -> tuple[Any, Any, Any]:
            value = self.model(image)
            return value["heatmap_logits"], value["offset"], value["orientation"]

    _EXPORT_WRAPPER_CLASS = _PinExportWrapper
    return _EXPORT_WRAPPER_CLASS


def _modified_focal_loss(logits: Any, target: Any, *, alpha: float, beta: float) -> Any:
    torch = _require_torch()
    probability = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    positives = (target >= 1.0 - 1e-6).to(logits.dtype)
    negatives = (target < 1.0 - 1e-6).to(logits.dtype)
    negative_weights = torch.pow(1.0 - target, beta)
    positive_loss = (
        torch.log(probability) * torch.pow(1.0 - probability, alpha) * positives
    )
    negative_loss = (
        torch.log(1.0 - probability)
        * torch.pow(probability, alpha)
        * negative_weights
        * negatives
    )
    positive_count = positives.sum()
    combined = -(positive_loss.sum() + negative_loss.sum())
    return torch.where(positive_count > 0, combined / positive_count.clamp_min(1.0), -negative_loss.sum())


def _ground_truth_points(targets: Mapping[str, Any], batch_size: int) -> list[list[tuple[float, float]]]:
    raw_points = targets.get("tip_points_heatmap")
    raw_visibility = targets.get("visibility")
    if not isinstance(raw_points, Sequence) or len(raw_points) != batch_size:
        raise ValueError("targets.tip_points_heatmap must contain one tensor per batch item")
    if not isinstance(raw_visibility, Sequence) or len(raw_visibility) != batch_size:
        raise ValueError("targets.visibility must contain one tensor per batch item")
    output: list[list[tuple[float, float]]] = []
    for points, visibility in zip(raw_points, raw_visibility):
        if len(points) != len(visibility):
            raise ValueError("Pin target points and visibility lengths differ")
        items: list[tuple[float, float]] = []
        for point, visible in zip(points, visibility):
            if int(visible) != 2:
                continue
            x, y = float(point[0]), float(point[1])
            if x >= 0 and y >= 0 and math.isfinite(x) and math.isfinite(y):
                items.append((x, y))
        output.append(items)
    return output


def _point_to_rect_distance(
    x: float, y: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)


def _point_to_rect_boundary_distance(
    x: float, y: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    """Distance to the rectangle perimeter, both inside and outside.

    Pin tips in the frozen Dataset300 labels lie on a component boundary.  A
    point-to-filled-rectangle distance is zero everywhere inside the box and
    therefore cannot reject interior heatmap maxima.  Outside points retain
    the ordinary Euclidean distance to the nearest edge/corner.
    """

    if x1 <= x <= x2 and y1 <= y <= y2:
        return min(x - x1, x2 - x, y - y1, y2 - y)
    return _point_to_rect_distance(x, y, x1, y1, x2, y2)


def _maximum_threshold_matches(
    predicted: Sequence[Mapping[str, Any]],
    expected: Sequence[tuple[float, float]],
    threshold: float,
) -> int:
    adjacency: list[list[int]] = []
    for item in predicted:
        choices = [
            index
            for index, point in enumerate(expected)
            if math.hypot(float(item["x"]) - point[0], float(item["y"]) - point[1])
            <= threshold
        ]
        choices.sort(
            key=lambda index: math.hypot(
                float(item["x"]) - expected[index][0],
                float(item["y"]) - expected[index][1],
            )
        )
        adjacency.append(choices)
    owner: dict[int, int] = {}

    def augment(prediction_index: int, seen: set[int]) -> bool:
        for expected_index in adjacency[prediction_index]:
            if expected_index in seen:
                continue
            seen.add(expected_index)
            if expected_index not in owner or augment(owner[expected_index], seen):
                owner[expected_index] = prediction_index
                return True
        return False

    return sum(augment(index, set()) for index in range(len(predicted)))


def _move_dense_targets(targets: Mapping[str, Any], device: Any) -> dict[str, Any]:
    output = dict(targets)
    for name in ("heatmap", "offset", "orientation", "regression_mask"):
        if name not in output or not hasattr(output[name], "to"):
            raise ValueError(f"Pin batch lacks tensor target {name!r}")
        output[name] = output[name].to(device=device)
    return output


def _gradient_l2_norm(parameters: Iterable[Any], torch: Any) -> Any:
    values = [parameter.grad.detach().norm(2) for parameter in parameters if parameter.grad is not None]
    if not values:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(values), ord=2)


def _scalar_losses(losses: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: float(value.detach().cpu())
        for name, value in losses.items()
        if hasattr(value, "detach")
    }


def _loss_config(value: PinLossConfig | Mapping[str, Any] | None) -> PinLossConfig:
    if value is None:
        return PinLossConfig()
    if isinstance(value, PinLossConfig):
        return value
    if isinstance(value, Mapping):
        return PinLossConfig(**dict(value))
    raise TypeError("loss config must be PinLossConfig, a mapping, or None")


def _validate_output_shapes(outputs: Mapping[str, Any]) -> None:
    shapes = {name: getattr(value, "shape", None) for name, value in outputs.items()}
    if any(shape is None or len(shape) != 4 for shape in shapes.values()):
        raise ValueError(f"Pin outputs must be BCHW tensors, got {shapes}")
    heatmap = tuple(shapes["heatmap_logits"])
    offset = tuple(shapes["offset"])
    orientation = tuple(shapes["orientation"])
    if heatmap[1] != 1 or offset[1] != 2 or orientation[1] != 2:
        raise ValueError(f"Pin output channels must be 1/2/2, got {shapes}")
    if heatmap[0] != offset[0] or heatmap[0] != orientation[0] or heatmap[2:] != offset[2:] or heatmap[2:] != orientation[2:]:
        raise ValueError(f"Pin output batch/spatial shapes differ: {shapes}")


def _validated_lineage(value: Any) -> dict[str, Any]:
    lineage = _json_mapping(value, "lineage")
    binding = lineage.get("binding_sha256")
    if not _is_sha256(binding):
        raise ValueError("lineage.binding_sha256 must be a lowercase SHA-256")
    for key, item in lineage.items():
        if key.endswith("_sha256") and not _is_sha256(item):
            raise ValueError(f"lineage.{key} must be a lowercase SHA-256")
    return lineage


def _json_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    try:
        encoded = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON-safe: {exc}") from exc
    if not isinstance(decoded, dict):
        raise TypeError(f"{label} must encode to an object")
    return decoded


def _normalize_export_samples(value: Any | Sequence[Any], torch: Any) -> list[Any]:
    if torch.is_tensor(value):
        samples = [value]
    elif isinstance(value, Sequence) and value and all(torch.is_tensor(item) for item in value):
        samples = list(value)
    else:
        raise TypeError("sample_inputs must be a tensor or non-empty tensor sequence")
    for sample in samples:
        if sample.ndim != 4 or sample.shape[1] != 3:
            raise ValueError("each export sample must have shape [B, 3, H, W]")
    return samples


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise TorchPinDependencyError(
            "Pin heatmap training/inference requires the optional PyTorch runtime"
        ) from exc
    return torch


def _sha256_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


__all__ = [
    "PinCheckpointError",
    "PinExportError",
    "PinLossConfig",
    "PinModelConfig",
    "TinyPinHeatmapNet",
    "TorchPinDependencyError",
    "compute_pin_metrics",
    "decode_pin_heatmaps",
    "export_pin_model",
    "load_pin_checkpoint",
    "pin_eval_step",
    "pin_losses",
    "pin_train_step",
    "save_pin_checkpoint",
    "split_pin_outputs",
]
