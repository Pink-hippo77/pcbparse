"""Image-only Pin inference from EDA_pin_net_graph_integration_20261012_v2.

The original FP32 networks and quantized, conservative fusion are retained.
No package training, annotation, graph, or Net code is imported at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from pcbparse.io import ComponentView, ImageView
from pcbparse.ir.models import Evidence, Pin, Point
from pcbparse.perception.protocols import ProviderError
from pcbparse.pipeline.context import stable_file_hash


class PinPackageError(ProviderError):
    pass


@dataclass(frozen=True, slots=True)
class PinPackageConfig:
    patch_size: int = 512
    stride: int = 384
    primary_threshold: float = 0.20
    multitask_threshold: float = 0.45
    multitask_min_area: int = 40
    branch_merge_radius: float = 7.0

    def __post_init__(self) -> None:
        if type(self.patch_size) is not int or self.patch_size < 16 or self.patch_size % 16:
            raise ValueError("Pin package patch_size must be a positive multiple of 16")
        if type(self.stride) is not int or not 0 < self.stride <= self.patch_size:
            raise ValueError("Pin package stride must be in (0, patch_size]")
        for name in ("primary_threshold", "multitask_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or not 0 < value <= 1:
                raise ValueError(f"Pin package {name} must be in (0, 1]")
        if type(self.multitask_min_area) is not int or self.multitask_min_area < 2:
            raise ValueError("Pin package multitask_min_area must be an integer >= 2")
        if isinstance(self.branch_merge_radius, bool) or not math.isfinite(float(self.branch_merge_radius)) or self.branch_merge_radius < 0:
            raise ValueError("Pin package branch_merge_radius must be finite and nonnegative")


def _verified_asset(root: Path, record: dict[str, Any]) -> Path:
    if set(record) != {"path", "sha256"} or not isinstance(record["path"], str):
        raise PinPackageError("Pin package asset must declare path and sha256")
    relative = Path(record["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise PinPackageError("Pin package asset path must stay inside its bundle")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise PinPackageError("Pin package asset is missing or outside its bundle")
    digest = record["sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise PinPackageError("Pin package asset requires lowercase SHA256")
    if stable_file_hash(path) != digest:
        raise PinPackageError(f"Pin package asset SHA256 mismatch: {relative}")
    return path


def verify_descriptor(path: str | Path, expected_sha256: str) -> tuple[dict, dict[str, Path], PinPackageConfig]:
    descriptor_path = Path(path).resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or stable_file_hash(descriptor_path) != expected_sha256:
        raise PinPackageError("Pin package descriptor SHA256 mismatch")
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    if descriptor.get("schema_version") != 1 or descriptor.get("task") != "pin" or descriptor.get("inference_scope") != "full_image":
        raise PinPackageError("unsupported Pin package descriptor contract")
    if descriptor.get("status") != "candidate" or descriptor.get("training_lineage_verified") is not False:
        raise PinPackageError("unverified package must remain an explicit candidate")
    weights = descriptor.get("weights", {})
    if not isinstance(weights, dict) or set(weights) != {"primary", "multitask"}:
        raise PinPackageError("Pin package requires exactly primary and multitask assets")
    assets = {name: _verified_asset(descriptor_path.parent, record) for name, record in weights.items()}
    config = PinPackageConfig(**descriptor["policy"])
    return descriptor, assets, config


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    # Preserve the package's traversal and FP32 centroid reduction order.
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    output = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        stack = [(int(y), int(x))]
        seen[y, x] = True
        pixels = []
        while stack:
            cy, cx = stack.pop()
            pixels.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not dy and not dx:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        output.append(np.asarray(pixels, dtype=np.int32))
    return output


def _points(probability: np.ndarray, threshold: float, min_area: int, source: str) -> list[dict]:
    output = []
    for component in _connected_components(probability >= threshold):
        if len(component) < min_area:
            continue
        ys, xs = component[:, 0], component[:, 1]
        weights = probability[ys, xs]
        output.append({"x": float(np.average(xs, weights=weights)),
                       "y": float(np.average(ys, weights=weights)),
                       "area": len(component), "score": float(weights.max()), "source": source})
    return output


def fuse_points(primary_u8: np.ndarray, multitask_u8: np.ndarray, config: PinPackageConfig) -> list[dict]:
    if primary_u8.dtype != np.uint8 or multitask_u8.dtype != np.uint8 or primary_u8.ndim != 2 or primary_u8.shape != multitask_u8.shape:
        raise PinPackageError("Pin package maps must be equally sized uint8 2D arrays")
    primary = _points(primary_u8.astype(np.float32) / 255, config.primary_threshold, 2, "primary")
    secondary = _points(multitask_u8.astype(np.float32) / 255, config.multitask_threshold, config.multitask_min_area, "multitask_supplement")
    output = list(primary)
    for point in secondary:
        if not any(math.hypot(point["x"] - other["x"], point["y"] - other["y"]) <= config.branch_merge_radius for other in primary):
            output.append(point)
    return sorted(output, key=lambda point: (point["y"], point["x"]))


def _origins(length: int, patch_size: int, stride: int) -> list[int]:
    last = max(length - patch_size, 0)
    values = list(range(0, last + 1, stride))
    if values[-1] != last:
        values.append(last)
    return values


class PinPackageAdapter:
    def __init__(self, descriptor_path: str | Path, *, sha256: str, device: str = "cpu") -> None:
        # Verify both artifacts before importing torch or deserializing weights.
        self.descriptor, assets, self.config = verify_descriptor(descriptor_path, sha256)
        import torch
        from pcbparse.perception.pin_package_network import MultiTaskUNet, UNet
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"} or self.device.type == "cuda" and not torch.cuda.is_available():
            raise PinPackageError(f"unsupported or unavailable Pin package device: {device}")
        self.torch = torch
        self.primary = UNet()
        self.multitask = MultiTaskUNet()
        for name, model in (("primary", self.primary), ("multitask", self.multitask)):
            checkpoint = torch.load(assets[name], map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.to(self.device).eval()
        self.descriptor_sha256 = sha256
        self.version = f"pin-package-v2-{sha256[:12]}"

    def probabilities(self, image: np.ndarray, ctx: Any = None) -> tuple[np.ndarray, np.ndarray]:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not all(image.shape[:2]):
            raise PinPackageError("Pin package input must be nonempty RGB uint8")
        height, width = image.shape[:2]
        maps = []
        with self.torch.inference_mode():
            for model, multitask in ((self.primary, False), (self.multitask, True)):
                total = np.zeros((height, width), dtype=np.float32)
                counts = np.zeros_like(total)
                for top in _origins(height, self.config.patch_size, self.config.stride):
                    for left in _origins(width, self.config.patch_size, self.config.stride):
                        if ctx is not None:
                            ctx.time_budget.check(stage="pin_package_patch")
                        patch = image[top:top + self.config.patch_size, left:left + self.config.patch_size]
                        ph, pw = patch.shape[:2]
                        padded = np.zeros((self.config.patch_size, self.config.patch_size, 3), dtype=np.uint8)
                        padded[:ph, :pw] = patch
                        tensor = self.torch.from_numpy(padded.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
                        logits = model(tensor.to(self.device))
                        if multitask:
                            logits = logits[1]
                        probability = self.torch.sigmoid(logits).cpu().numpy()[0, 0]
                        total[top:top + ph, left:left + pw] += probability[:ph, :pw]
                        counts[top:top + ph, left:left + pw] += 1.0
                maps.append((np.clip(total / np.maximum(counts, 1.0), 0, 1) * 255).astype(np.uint8))
        return maps[0], maps[1]

    def predict_image(self, image: ImageView, ctx: Any) -> list[Pin]:
        primary, multitask = self.probabilities(np.asarray(image.image.convert("RGB"), dtype=np.uint8), ctx)
        points = fuse_points(primary, multitask, self.config)
        return [Pin(
            internal_id=f"package-pin-{index:06d}", point=Point(x=point["x"], y=point["y"]),
            score=point["score"], evidence=[Evidence(
                evidence_id=f"package-pin:{index:06d}", source="detector", view_id=image.view_id,
                score=point["score"], metadata={"model_version": self.version,
                    "descriptor_sha256": self.descriptor_sha256, "branch": point["source"],
                    "area": point["area"], "coordinate_system": "source_view_pixels",
                    "training_lineage_verified": False, "image_only": True},
            )],
        ) for index, point in enumerate(points)]

    def predict(self, component_views: list[ComponentView], ctx: Any) -> list[Pin]:
        """Protocol-compatible ROI fallback; Composite uses predict_image once."""
        output = []
        for component in component_views:
            pins = self.predict_image(component.full_roi, ctx)
            for pin in pins:
                pin.component_id = component.component_id
                pin.internal_id = f"{component.component_id}:{pin.internal_id}"
            output.extend(pins)
        return output
