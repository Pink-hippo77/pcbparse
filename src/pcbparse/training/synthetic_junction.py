from __future__ import annotations

"""Deterministic six-class junction patches for *training only*.

The official freeze contains trustworthy endpoint, T-junction, and bend
annotations, but intentionally leaves raster crosses as ``UNKNOWN``.  This
module does not reinterpret those rows.  It supplies auditable synthetic
examples for the three absent classes and can be concatenated with the
trusted real training dataset.  Synthetic rows are structurally forbidden on
validation and test splits.
"""

import hashlib
import importlib
import io
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from pcbparse.training.torch_wire_junction import JUNCTION_CLASSES


SYNTHETIC_POLICY = "pcbparse.synthetic-junction.v1"


class SyntheticJunctionError(ValueError):
    """A synthetic supervision request violates the frozen policy."""


@dataclass(frozen=True, slots=True)
class SyntheticJunctionConfig:
    patch_size: int = 64
    supersample: int = 4
    seed: int = 20260813
    minimum_line_width: float = 1.1
    maximum_line_width: float = 3.4
    maximum_center_jitter: float = 4.0
    maximum_rotation_degrees: float = 180.0
    maximum_gaussian_noise: float = 5.0
    minimum_dot_scale: float = 1.8
    maximum_dot_scale: float = 2.8
    minimum_bridge_scale: float = 2.2
    maximum_bridge_scale: float = 3.8

    def __post_init__(self) -> None:
        if type(self.patch_size) is not int or self.patch_size < 24:
            raise SyntheticJunctionError("patch_size must be an integer >= 24")
        if type(self.supersample) is not int or self.supersample < 2:
            raise SyntheticJunctionError("supersample must be an integer >= 2")
        if type(self.seed) is not int:
            raise SyntheticJunctionError("seed must be an integer")
        for name in (
            "minimum_line_width",
            "maximum_line_width",
            "maximum_center_jitter",
            "maximum_rotation_degrees",
            "maximum_gaussian_noise",
            "minimum_dot_scale",
            "maximum_dot_scale",
            "minimum_bridge_scale",
            "maximum_bridge_scale",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise SyntheticJunctionError(f"{name} must be finite")
        if not 0.5 <= self.minimum_line_width <= self.maximum_line_width:
            raise SyntheticJunctionError("line-width range is invalid")
        if self.maximum_center_jitter < 0 or self.maximum_rotation_degrees < 0:
            raise SyntheticJunctionError("jitter and rotation must be non-negative")
        if self.maximum_gaussian_noise < 0:
            raise SyntheticJunctionError("maximum_gaussian_noise must be non-negative")
        if not 1.0 < self.minimum_dot_scale <= self.maximum_dot_scale:
            raise SyntheticJunctionError("connected-dot scale range is invalid")
        if not 1.2 < self.minimum_bridge_scale <= self.maximum_bridge_scale:
            raise SyntheticJunctionError("non-connected bridge range is invalid")

    @property
    def policy_sha256(self) -> str:
        return _canonical_sha256({"policy": SYNTHETIC_POLICY, **asdict(self)})


@dataclass(frozen=True, slots=True)
class SyntheticJunctionAsset:
    image: Image.Image
    label: str
    label_id: int
    record_id: str
    image_sha256: str
    metadata: dict[str, Any]


def generate_synthetic_junction_patch(
    label: str,
    index: int,
    *,
    config: SyntheticJunctionConfig | Mapping[str, Any] | None = None,
) -> SyntheticJunctionAsset:
    """Render one deterministic anti-aliased RGB patch and its audit record.

    ``connected_cross`` is two uninterrupted strokes plus an explicit filled
    node dot. ``non_connected_cross`` has an uninterrupted overpass and a
    white underpass gap wider than the stroke; it never contains a node dot.
    These visual contracts are recorded in every returned asset.
    """

    policy = _config(config)
    if label not in JUNCTION_CLASSES:
        raise SyntheticJunctionError(f"unknown junction class {label!r}")
    if type(index) is not int or index < 0:
        raise SyntheticJunctionError("index must be a non-negative integer")
    local_seed = _derived_seed(policy.seed, label, index, policy.policy_sha256)
    rng = np.random.default_rng(local_seed)
    scale = policy.supersample
    size = policy.patch_size
    canvas_size = size * scale
    background = int(rng.integers(241, 256))
    ink = int(rng.integers(0, 49))
    line_width = float(rng.uniform(policy.minimum_line_width, policy.maximum_line_width))
    line_width_hi = max(2, int(round(line_width * scale)))
    jitter = rng.uniform(-policy.maximum_center_jitter, policy.maximum_center_jitter, size=2)
    center = (
        (size / 2.0 + float(jitter[0])) * scale,
        (size / 2.0 + float(jitter[1])) * scale,
    )
    rotation = float(rng.uniform(-policy.maximum_rotation_degrees, policy.maximum_rotation_degrees))
    radius = float(rng.uniform(size * 0.37, size * 0.49)) * scale

    image = Image.new("L", (canvas_size, canvas_size), color=background)
    draw = ImageDraw.Draw(image)
    visual_contract: dict[str, Any]
    parameters: dict[str, Any] = {
        "seed": local_seed,
        "center": [center[0] / scale, center[1] / scale],
        "rotation_degrees": rotation,
        "line_width": line_width,
        "background": background,
        "ink": ink,
    }

    def endpoint(angle: float, distance: float = radius) -> tuple[float, float]:
        radians = math.radians(angle)
        return center[0] + math.cos(radians) * distance, center[1] + math.sin(radians) * distance

    def stroke(angle_a: float, angle_b: float) -> None:
        draw.line(
            [endpoint(angle_a), endpoint(angle_b)],
            fill=ink,
            width=line_width_hi,
            joint="curve",
        )

    def arm(angle: float, *, inner: float = 0.0) -> None:
        outer = endpoint(angle)
        radians = math.radians(angle)
        inner_point = (
            center[0] + math.cos(radians) * inner,
            center[1] + math.sin(radians) * inner,
        )
        draw.line([inner_point, outer], fill=ink, width=line_width_hi)

    if label == "endpoint":
        arm(rotation)
        visual_contract = {"center_degree": 1, "node_dot": False, "bridge_gap": False}
    elif label == "t_junction":
        for angle in (rotation, rotation + 90.0, rotation + 180.0):
            arm(angle)
        visual_contract = {"center_degree": 3, "node_dot": False, "bridge_gap": False}
    elif label == "bend":
        bend_angle = float(rng.uniform(48.0, 132.0))
        arm(rotation)
        arm(rotation + bend_angle)
        parameters["bend_angle_degrees"] = bend_angle
        visual_contract = {
            "center_degree": 2,
            "non_collinear": True,
            "node_dot": False,
            "bridge_gap": False,
        }
    elif label == "connected_cross":
        stroke(rotation, rotation + 180.0)
        stroke(rotation + 90.0, rotation + 270.0)
        dot_scale = float(rng.uniform(policy.minimum_dot_scale, policy.maximum_dot_scale))
        dot_radius = 0.5 * line_width_hi * dot_scale
        draw.ellipse(
            [
                center[0] - dot_radius,
                center[1] - dot_radius,
                center[0] + dot_radius,
                center[1] + dot_radius,
            ],
            fill=ink,
        )
        parameters["node_dot_diameter_over_line_width"] = dot_scale
        visual_contract = {
            "center_degree": 4,
            "two_uninterrupted_strokes": True,
            "node_dot": True,
            "bridge_gap": False,
        }
    elif label == "non_connected_cross":
        # Draw the underpass first, erase a centre halo, then redraw the
        # overpass.  The underpass gap remains visibly wider than either line.
        stroke(rotation + 90.0, rotation + 270.0)
        bridge_scale = float(rng.uniform(policy.minimum_bridge_scale, policy.maximum_bridge_scale))
        gap_radius = 0.5 * line_width_hi * bridge_scale
        draw.ellipse(
            [
                center[0] - gap_radius,
                center[1] - gap_radius,
                center[0] + gap_radius,
                center[1] + gap_radius,
            ],
            fill=background,
        )
        stroke(rotation, rotation + 180.0)
        parameters["underpass_gap_diameter_over_line_width"] = bridge_scale
        visual_contract = {
            "center_degree": 0,
            "overpass_uninterrupted": True,
            "underpass_visibly_gapped": True,
            "node_dot": False,
            "bridge_gap": True,
        }
    else:  # noise
        # Distractors deliberately avoid the central acceptance disc.
        exclusion = size * 0.14 * scale
        shape_count = int(rng.integers(1, 5))
        for _ in range(shape_count):
            angle = float(rng.uniform(0.0, 360.0))
            distance = float(rng.uniform(size * 0.22, size * 0.46)) * scale
            shape_center = (
                center[0] + math.cos(math.radians(angle)) * distance,
                center[1] + math.sin(math.radians(angle)) * distance,
            )
            length = float(rng.uniform(size * 0.05, size * 0.18)) * scale
            direction = float(rng.uniform(0.0, 360.0))
            delta = (
                math.cos(math.radians(direction)) * length,
                math.sin(math.radians(direction)) * length,
            )
            draw.line(
                [
                    (shape_center[0] - delta[0], shape_center[1] - delta[1]),
                    (shape_center[0] + delta[0], shape_center[1] + delta[1]),
                ],
                fill=ink,
                width=max(1, int(round(line_width_hi * rng.uniform(0.6, 1.3)))),
            )
        draw.ellipse(
            [center[0] - exclusion, center[1] - exclusion, center[0] + exclusion, center[1] + exclusion],
            fill=background,
        )
        visual_contract = {
            "center_degree": 0,
            "central_exclusion_radius": exclusion / scale,
            "node_dot": False,
            "bridge_gap": False,
        }

    # Blur at high resolution, then Lanczos downsample.  Both emulate raster
    # capture while retaining a deterministic, inspectable rendering path.
    blur_radius = float(rng.uniform(0.0, 0.32)) * scale
    if blur_radius > 0.0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32)
    noise_sigma = float(rng.uniform(0.0, policy.maximum_gaussian_noise))
    if noise_sigma > 0.0:
        array = array + rng.normal(0.0, noise_sigma, size=array.shape)
    # Low-amplitude brightness/contrast variation broadens scan domains.
    contrast = float(rng.uniform(0.92, 1.08))
    brightness = float(rng.uniform(-4.0, 4.0))
    array = np.clip((array - 127.5) * contrast + 127.5 + brightness, 0, 255).astype(np.uint8)
    rgb = Image.fromarray(array, mode="L").convert("RGB")
    payload = io.BytesIO()
    rgb.save(payload, format="PNG", optimize=False, compress_level=9)
    image_sha = hashlib.sha256(payload.getvalue()).hexdigest()
    record_material = f"{SYNTHETIC_POLICY}\0{policy.policy_sha256}\0{label}\0{index}\0{image_sha}"
    record_id = "synthetic-junction:" + hashlib.sha256(record_material.encode("utf-8")).hexdigest()[:24]
    metadata = {
        "schema": SYNTHETIC_POLICY,
        "policy_sha256": policy.policy_sha256,
        "label": label,
        "label_id": JUNCTION_CLASSES.index(label),
        "index": index,
        "image_sha256": image_sha,
        "parameters": {**parameters, "noise_sigma": noise_sigma, "contrast": contrast, "brightness": brightness},
        "visual_contract": visual_contract,
        "provenance": "deterministic_synthetic_train_only",
        "raw_unknown_crosses_used": False,
    }
    return SyntheticJunctionAsset(
        image=rgb,
        label=label,
        label_id=JUNCTION_CLASSES.index(label),
        record_id=record_id,
        image_sha256=image_sha,
        metadata=metadata,
    )


class SyntheticJunctionDataset:
    """On-demand deterministic patches, hard-gated to ``split='train'``."""

    def __init__(
        self,
        *,
        samples_per_class: int | Mapping[str, int] = 256,
        classes: Sequence[str] = JUNCTION_CLASSES,
        config: SyntheticJunctionConfig | Mapping[str, Any] | None = None,
        split: str = "train",
    ) -> None:
        if split != "train":
            raise SyntheticJunctionError(
                "synthetic junction supervision is train-only; validation/test must remain real"
            )
        self.split = split
        self.config = _config(config)
        selected = tuple(classes)
        if not selected or len(set(selected)) != len(selected) or any(name not in JUNCTION_CLASSES for name in selected):
            raise SyntheticJunctionError("classes must be a unique non-empty subset of JUNCTION_CLASSES")
        if isinstance(samples_per_class, Mapping):
            unknown = set(samples_per_class) - set(selected)
            if unknown:
                raise SyntheticJunctionError(f"counts supplied for unselected classes: {sorted(unknown)}")
            counts = {name: samples_per_class.get(name, 0) for name in selected}
        else:
            counts = {name: samples_per_class for name in selected}
        if any(type(value) is not int or value < 0 for value in counts.values()) or not any(counts.values()):
            raise SyntheticJunctionError("synthetic class counts must be non-negative integers with a positive total")
        self.class_counts = {name: int(counts.get(name, 0)) for name in JUNCTION_CLASSES}
        self.records: list[dict[str, Any]] = []
        for label in JUNCTION_CLASSES:
            for index in range(self.class_counts[label]):
                self.records.append(
                    {
                        "label": label,
                        "label_id": JUNCTION_CLASSES.index(label),
                        "index": index,
                        "split": "train",
                        "label_status": "synthetic_train_only",
                    }
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = _require_torch()
        row = self.records[index]
        asset = generate_synthetic_junction_patch(
            row["label"], row["index"], config=self.config
        )
        array = np.asarray(asset.image, dtype=np.float32).copy()
        tensor = torch.from_numpy(np.transpose(array / 255.0, (2, 0, 1)))
        return {
            "image": tensor,
            "label": torch.tensor(asset.label_id, dtype=torch.long),
            "label_name": asset.label,
            "record_id": asset.record_id,
            "sample_id": asset.record_id,
            "split": "train",
            "transform": {
                "kind": "synthetic_identity",
                "patch_size": [self.config.patch_size, self.config.patch_size],
            },
            "source": {
                "kind": "deterministic_synthetic",
                "policy": SYNTHETIC_POLICY,
                "policy_sha256": self.config.policy_sha256,
                "image_sha256": asset.image_sha256,
                "metadata": asset.metadata,
                "raw_unknown_crosses_used": False,
            },
        }


class MixedJunctionTrainingDataset:
    """Concatenate trusted real training rows with train-only synthetic rows."""

    def __init__(self, real_dataset: Any, synthetic_dataset: SyntheticJunctionDataset) -> None:
        if getattr(real_dataset, "split", None) != "train":
            raise SyntheticJunctionError("real junction dataset must select split='train'")
        if synthetic_dataset.split != "train":
            raise SyntheticJunctionError("synthetic junction dataset must be train-only")
        # ``JunctionTorchDataset.records`` already excludes every UNKNOWN row.
        real_records = getattr(real_dataset, "records", None)
        if not isinstance(real_records, list):
            raise SyntheticJunctionError("real dataset must expose frozen records")
        if any(row.get("label_status") != "trusted_gt" or row.get("label") not in JUNCTION_CLASSES for row in real_records):
            raise SyntheticJunctionError("real dataset contains non-trusted or UNKNOWN supervision")
        self.real_dataset = real_dataset
        self.synthetic_dataset = synthetic_dataset
        self.split = "train"
        self.records = [*real_records, *synthetic_dataset.records]
        self.class_counts = dict(Counter(row["label"] for row in self.records))
        self.lineage = {
            "real_run_id": getattr(getattr(real_dataset, "metadata", None), "run_id", None),
            "real_records": len(real_dataset),
            "synthetic_policy": SYNTHETIC_POLICY,
            "synthetic_policy_sha256": synthetic_dataset.config.policy_sha256,
            "synthetic_records": len(synthetic_dataset),
            "raw_unknown_crosses_used": False,
            "split": "train",
        }

    def __len__(self) -> int:
        return len(self.real_dataset) + len(self.synthetic_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < len(self.real_dataset):
            item = self.real_dataset[index]
            if item.get("label_name") not in JUNCTION_CLASSES:
                raise SyntheticJunctionError("real dataset yielded non-six-class supervision")
            return item
        return self.synthetic_dataset[index - len(self.real_dataset)]


def build_mixed_junction_training_dataset(
    real_dataset: Any,
    *,
    samples_per_class: int | Mapping[str, int] = 256,
    synthetic_classes: Sequence[str] = (
        "connected_cross",
        "non_connected_cross",
        "noise",
    ),
    config: SyntheticJunctionConfig | Mapping[str, Any] | None = None,
) -> MixedJunctionTrainingDataset:
    """Default mix: real trusted first three classes + synthetic absent three."""

    synthetic = SyntheticJunctionDataset(
        samples_per_class=samples_per_class,
        classes=synthetic_classes,
        config=config,
        split="train",
    )
    return MixedJunctionTrainingDataset(real_dataset, synthetic)


def _config(value: SyntheticJunctionConfig | Mapping[str, Any] | None) -> SyntheticJunctionConfig:
    if value is None:
        return SyntheticJunctionConfig()
    if isinstance(value, SyntheticJunctionConfig):
        return value
    if isinstance(value, Mapping):
        return SyntheticJunctionConfig(**dict(value))
    raise TypeError("config must be SyntheticJunctionConfig, a mapping, or None")


def _derived_seed(seed: int, label: str, index: int, policy_sha: str) -> int:
    material = f"{seed}\0{label}\0{index}\0{policy_sha}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to consume synthetic junction tensors") from exc


__all__ = [
    "MixedJunctionTrainingDataset",
    "SYNTHETIC_POLICY",
    "SyntheticJunctionAsset",
    "SyntheticJunctionConfig",
    "SyntheticJunctionDataset",
    "SyntheticJunctionError",
    "build_mixed_junction_training_dataset",
    "generate_synthetic_junction_patch",
]
