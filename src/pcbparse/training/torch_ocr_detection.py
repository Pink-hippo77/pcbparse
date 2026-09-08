from __future__ import annotations

"""Frozen Dataset300 OCR-detection data for PyTorch.

The OCR materializer intentionally permits ``materialize_images=false``.  This
dataset therefore reconstructs each tile lazily from the hash-bound source
image, never from annotation side channels.  PyTorch is imported only when the
dataset is constructed or a sample is tensorized.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from pcbparse.training.torch_ocr import TorchOCRDataError, require_torch


VALID_SPLITS = ("train", "val", "test")


class TorchOCRDetectionDataError(TorchOCRDataError):
    """A frozen OCR detection tile is missing, corrupt, or ambiguous."""


class TorchOCRDetectionDataset:
    """Map-style OCR detector dataset backed by ``detection_index.jsonl``.

    ``resize`` may be one positive integer (a square, default 512), a
    ``(height, width)`` pair, or ``None`` to retain the frozen tile dimensions.
    The returned ``target`` and ``ignore`` tensors have shape ``[1,H,W]``.
    Ignore regions are removed from ``target`` so a pixel can never provide
    contradictory positive and ignored supervision.
    """

    def __init__(
        self,
        materialization_root: str | Path,
        *,
        split: str,
        source_root: str | Path,
        resize: int | Sequence[int] | None = 512,
        target_shrink_ratio: float = 0.0,
        verify_image_hashes: bool = True,
        max_records: int | None = None,
    ) -> None:
        require_torch()
        if split not in VALID_SPLITS:
            raise TorchOCRDetectionDataError(
                f"split must be one of {VALID_SPLITS}, got {split!r}"
            )
        if type(verify_image_hashes) is not bool:
            raise TorchOCRDetectionDataError("verify_image_hashes must be boolean")
        if (
            isinstance(target_shrink_ratio, bool)
            or not isinstance(target_shrink_ratio, (int, float))
            or not math.isfinite(float(target_shrink_ratio))
            or not 0.0 <= float(target_shrink_ratio) < 0.5
        ):
            raise TorchOCRDetectionDataError(
                "target_shrink_ratio must be finite and in [0, 0.5)"
            )
        if max_records is not None and (
            type(max_records) is not int or max_records <= 0
        ):
            raise TorchOCRDetectionDataError("max_records must be a positive integer")

        self.root = Path(materialization_root).expanduser().resolve(strict=True)
        self.source_root = Path(source_root).expanduser().resolve(strict=True)
        if not self.source_root.is_dir():
            raise TorchOCRDetectionDataError(
                f"source_root is not a directory: {self.source_root}"
            )
        self.split = split
        self.resize = _normalise_resize(resize)
        self.target_shrink_ratio = float(target_shrink_ratio)
        self.verify_image_hashes = verify_image_hashes
        self._verified_sources: dict[str, tuple[str, int, int]] = {}

        manifest = _read_json_object(self.root / "manifest.json")
        _validate_manifest(manifest)
        _verify_committed_artifact(
            self.root, manifest, "detection_index.jsonl"
        )
        _verify_committed_artifact(self.root, manifest, "report.json")
        report = _read_json_object(self.root / "report.json")
        policy = report.get("policy")
        if not isinstance(policy, dict):
            raise TorchOCRDetectionDataError(
                "materialization report lacks a policy object"
            )
        if _canonical_sha256(policy) != manifest.get("policy_sha256"):
            raise TorchOCRDetectionDataError(
                "materialization policy hash does not match manifest"
            )
        _validate_detection_policy(policy)
        rows = _read_jsonl(self.root / "detection_index.jsonl", "tile_id")
        expected_count = manifest["counts"].get("detection_tiles")
        if expected_count != len(rows):
            raise TorchOCRDetectionDataError(
                "detection index count does not match materialization manifest"
            )
        selected: list[dict[str, Any]] = []
        for row in rows:
            _validate_detection_row(row)
            if row["split"] == split:
                selected.append(row)
        selected.sort(key=lambda row: row["tile_id"])
        if max_records is not None:
            selected = selected[:max_records]
        if not selected:
            raise TorchOCRDetectionDataError(
                f"no OCR detection tiles for split {split!r}"
            )
        self.records = tuple(selected)
        self.manifest = manifest
        self.policy = dict(policy)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = require_torch()
        row = self.records[index]
        tile = self._load_source_tile(row)
        try:
            original_width, original_height = tile.size
            target_array, ignore_array, boxes, ignore_boxes = _rasterize_annotations(
                row,
                width=original_width,
                height=original_height,
                target_shrink_ratio=self.target_shrink_ratio,
            )
            if self.resize is None:
                output_height, output_width = original_height, original_width
                resized = tile.convert("RGB")
            else:
                output_height, output_width = self.resize
                resized = tile.convert("RGB").resize(
                    (output_width, output_height), Image.Resampling.BILINEAR
                )
                target_array = _resize_mask(
                    target_array, width=output_width, height=output_height
                )
                ignore_array = _resize_mask(
                    ignore_array, width=output_width, height=output_height
                )
                # Nearest-neighbour rasterization is deterministic, but apply
                # the dominance rule again after scaling so even edge rounding
                # can never create contradictory supervision.
                target_array[ignore_array != 0] = 0
                scale_x = output_width / original_width
                scale_y = output_height / original_height
                boxes = [_scale_box(box, scale_x, scale_y) for box in boxes]
                ignore_boxes = [
                    _scale_box(box, scale_x, scale_y) for box in ignore_boxes
                ]
            rgb = np.asarray(resized, dtype=np.float32).copy() / np.float32(255.0)
            image_tensor = (
                torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
            )
            target_tensor = torch.from_numpy(target_array.astype(np.float32, copy=False))
            ignore_tensor = torch.from_numpy(ignore_array.astype(np.float32, copy=False))
            box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            ignore_box_tensor = torch.tensor(
                ignore_boxes, dtype=torch.float32
            ).reshape(-1, 4)
        finally:
            tile.close()

        return {
            "image": image_tensor,
            "target": target_tensor.unsqueeze(0).contiguous(),
            "ignore": ignore_tensor.unsqueeze(0).contiguous(),
            "boxes": box_tensor,
            "ignore_boxes": ignore_box_tensor,
            "tile_id": row["tile_id"],
            "sample_id": row["sample_id"],
            "metadata": {
                "tile_id": row["tile_id"],
                "sample_id": row["sample_id"],
                "split": row["split"],
                "source_domain": row.get("source_domain"),
                "source_image": row["source_image"],
                "source_image_sha256": row["source_image_sha256"],
                "source_box": dict(row["source_box"]),
                "frozen_output_size": list(row["output_size"]),
                "tensor_size": [output_height, output_width],
                "target_shrink_ratio": self.target_shrink_ratio,
                "positive_count": row["positive_count"],
                "ignored_count": row["ignored_count"],
                "empty_tile": row["empty_tile"],
                "spec_sha256": row["spec_sha256"],
            },
        }

    def _load_source_tile(self, row: dict[str, Any]) -> Image.Image:
        source = _safe_child(self.source_root, row["source_image"])
        if self.verify_image_hashes:
            stat = source.stat()
            identity = (
                row["source_image_sha256"],
                int(stat.st_size),
                int(stat.st_mtime_ns),
            )
            if self._verified_sources.get(str(source)) != identity:
                actual = _sha256_file(source)
                if actual != row["source_image_sha256"]:
                    raise TorchOCRDetectionDataError(
                        f"source image for {row['tile_id']} SHA-256 mismatch: "
                        f"expected {row['source_image_sha256']}, got {actual}"
                    )
                self._verified_sources[str(source)] = identity
        decoded = _decode_image(source)
        box = row["source_box"]
        left, top, right, bottom = (
            box["left"],
            box["top"],
            box["right"],
            box["bottom"],
        )
        if right > decoded.width or bottom > decoded.height:
            decoded.close()
            raise TorchOCRDetectionDataError(
                f"source_box for {row['tile_id']} exceeds decoded source image"
            )
        try:
            crop = decoded.crop((left, top, right, bottom)).convert("RGB")
        finally:
            decoded.close()
        output_width, output_height = row["output_size"]
        if crop.size == (output_width, output_height):
            return crop
        if crop.width > output_width or crop.height > output_height:
            crop.close()
            raise TorchOCRDetectionDataError(
                f"source tile for {row['tile_id']} is larger than frozen output_size"
            )
        canvas = Image.new("RGB", (output_width, output_height), (255, 255, 255))
        canvas.paste(crop, (0, 0))
        crop.close()
        return canvas


def detection_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stack fixed-size detector tensors while retaining variable box lists."""

    torch = require_torch()
    if not batch:
        raise TorchOCRDetectionDataError("cannot collate an empty detection batch")
    shapes = [tuple(item["image"].shape) for item in batch]
    if len(set(shapes)) != 1:
        raise TorchOCRDetectionDataError(
            "detector images must share a shape; configure a fixed resize"
        )
    for item in batch:
        if tuple(item["target"].shape) != (1, shapes[0][1], shapes[0][2]):
            raise TorchOCRDetectionDataError("target shape differs from detector image")
        if tuple(item["ignore"].shape) != tuple(item["target"].shape):
            raise TorchOCRDetectionDataError("ignore shape differs from target")
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "target": torch.stack([item["target"] for item in batch], dim=0),
        "ignore": torch.stack([item["ignore"] for item in batch], dim=0),
        "boxes": [item["boxes"] for item in batch],
        "ignore_boxes": [item["ignore_boxes"] for item in batch],
        "tile_ids": [item["tile_id"] for item in batch],
        "sample_ids": [item["sample_id"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


def _normalise_resize(
    value: int | Sequence[int] | None,
) -> tuple[int, int] | None:
    if value is None:
        return None
    if type(value) is int:
        if value <= 0:
            raise TorchOCRDetectionDataError("resize must be positive")
        return value, value
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TorchOCRDetectionDataError(
            "resize must be an integer, (height,width), or None"
        )
    items = tuple(value)
    if len(items) != 2 or any(type(item) is not int or item <= 0 for item in items):
        raise TorchOCRDetectionDataError(
            "resize pair must contain positive integer height and width"
        )
    return int(items[0]), int(items[1])


def _rasterize_annotations(
    row: dict[str, Any],
    *,
    width: int,
    height: int,
    target_shrink_ratio: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[list[float]], list[list[float]]]:
    if (
        isinstance(target_shrink_ratio, bool)
        or not isinstance(target_shrink_ratio, (int, float))
        or not math.isfinite(float(target_shrink_ratio))
        or not 0.0 <= float(target_shrink_ratio) < 0.5
    ):
        raise TorchOCRDetectionDataError(
            "target_shrink_ratio must be finite and in [0, 0.5)"
        )
    shrink_ratio = float(target_shrink_ratio)
    target = np.zeros((height, width), dtype=np.uint8)
    ignore = np.zeros((height, width), dtype=np.uint8)
    boxes: list[list[float]] = []
    ignore_boxes: list[list[float]] = []
    for annotation in row["annotations"]:
        points = annotation["points"]
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        x1 = max(0, min(width, int(math.floor(min(xs)))))
        y1 = max(0, min(height, int(math.floor(min(ys)))))
        x2 = max(0, min(width, int(math.ceil(max(xs)))))
        y2 = max(0, min(height, int(math.ceil(max(ys)))))
        if x2 <= x1 or y2 <= y1:
            raise TorchOCRDetectionDataError(
                f"annotation {annotation['record_id']} rasterizes to zero area"
            )
        box = [float(x1), float(y1), float(x2), float(y2)]
        if annotation["ignore"]:
            ignore[y1:y2, x1:x2] = 1
            ignore_boxes.append(box)
        else:
            box_width = x2 - x1
            box_height = y2 - y1
            margin_x = min(
                int(math.floor(box_width * shrink_ratio)),
                max(0, (box_width - 1) // 2),
            )
            margin_y = min(
                int(math.floor(box_height * shrink_ratio)),
                max(0, (box_height - 1) // 2),
            )
            target[
                y1 + margin_y : y2 - margin_y,
                x1 + margin_x : x2 - margin_x,
            ] = 1
            boxes.append(box)
    target[ignore != 0] = 0
    return target, ignore, boxes, ignore_boxes


def _resize_mask(mask: np.ndarray, *, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(mask, mode="L")
    try:
        resized = image.resize((width, height), Image.Resampling.NEAREST)
        return (np.asarray(resized, dtype=np.uint8).copy() != 0).astype(np.uint8)
    finally:
        image.close()


def _scale_box(box: Sequence[float], scale_x: float, scale_y: float) -> list[float]:
    return [
        float(box[0]) * scale_x,
        float(box[1]) * scale_y,
        float(box[2]) * scale_x,
        float(box[3]) * scale_y,
    ]


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise TorchOCRDetectionDataError("unsupported OCR materialization manifest")
    if manifest.get("kind") != "pcbparse_ocr_paddle_materialization":
        raise TorchOCRDetectionDataError("not an OCR materialization manifest")
    if manifest.get("complete") is not True:
        raise TorchOCRDetectionDataError("OCR materialization is incomplete")
    if not isinstance(manifest.get("counts"), dict):
        raise TorchOCRDetectionDataError("manifest lacks counts")
    if not isinstance(manifest.get("artifacts"), dict):
        raise TorchOCRDetectionDataError("manifest lacks artifact hashes")
    if not _is_sha256(manifest.get("policy_sha256")):
        raise TorchOCRDetectionDataError("manifest lacks a valid policy SHA-256")


def _validate_detection_policy(policy: dict[str, Any]) -> None:
    expected = {
        "bbox_points": "top_left_top_right_bottom_right_bottom_left_clockwise",
        "partial_detection_box": "clip_intersection_and_mark_###_ignore",
        "empty_detection_tiles_retained": True,
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise TorchOCRDetectionDataError(
                f"unsupported frozen detection policy {key}={policy.get(key)!r}"
            )
    tile_size = policy.get("detection_tile_size")
    overlap = policy.get("detection_tile_overlap")
    if type(tile_size) is not int or tile_size <= 0:
        raise TorchOCRDetectionDataError("policy detection_tile_size is invalid")
    if type(overlap) is not int or overlap < 0 or overlap >= tile_size:
        raise TorchOCRDetectionDataError("policy detection_tile_overlap is invalid")


def _validate_detection_row(row: dict[str, Any]) -> None:
    tile_id = row.get("tile_id")
    if row.get("schema_version") != 1 or not isinstance(tile_id, str) or not tile_id:
        raise TorchOCRDetectionDataError("detection row has invalid schema/tile_id")
    if row.get("split") not in VALID_SPLITS:
        raise TorchOCRDetectionDataError(f"tile {tile_id} has invalid split")
    for key in ("sample_id", "source_image", "spec_sha256"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise TorchOCRDetectionDataError(f"tile {tile_id} lacks {key}")
    if not _is_sha256(row.get("source_image_sha256")) or not _is_sha256(
        row.get("spec_sha256")
    ):
        raise TorchOCRDetectionDataError(f"tile {tile_id} has invalid SHA-256 metadata")
    source_box = row.get("source_box")
    if not isinstance(source_box, dict):
        raise TorchOCRDetectionDataError(f"tile {tile_id} lacks source_box")
    values: list[int] = []
    for key in ("left", "top", "right", "bottom"):
        value = source_box.get(key)
        if type(value) is not int:
            raise TorchOCRDetectionDataError(f"tile {tile_id} source_box is not integer")
        values.append(value)
    left, top, right, bottom = values
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise TorchOCRDetectionDataError(f"tile {tile_id} has invalid source_box")
    output_size = row.get("output_size")
    valid_size = row.get("valid_size")
    if (
        not isinstance(output_size, list)
        or len(output_size) != 2
        or any(type(value) is not int or value <= 0 for value in output_size)
    ):
        raise TorchOCRDetectionDataError(f"tile {tile_id} has invalid output_size")
    if valid_size != [right - left, bottom - top]:
        raise TorchOCRDetectionDataError(f"tile {tile_id} valid_size/source_box mismatch")
    if valid_size[0] > output_size[0] or valid_size[1] > output_size[1]:
        raise TorchOCRDetectionDataError(f"tile {tile_id} valid_size exceeds output_size")
    annotations = row.get("annotations")
    if not isinstance(annotations, list):
        raise TorchOCRDetectionDataError(f"tile {tile_id} lacks annotations")
    positive = 0
    ignored = 0
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise TorchOCRDetectionDataError(f"tile {tile_id} annotation is not an object")
        if type(annotation.get("ignore")) is not bool:
            raise TorchOCRDetectionDataError(f"tile {tile_id} annotation lacks ignore flag")
        if not isinstance(annotation.get("record_id"), str):
            raise TorchOCRDetectionDataError(f"tile {tile_id} annotation lacks record_id")
        points = annotation.get("points")
        if not isinstance(points, list) or len(points) != 4:
            raise TorchOCRDetectionDataError(f"tile {tile_id} annotation is not a quadrilateral")
        for point in points:
            if (
                not isinstance(point, list)
                or len(point) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in point
                )
            ):
                raise TorchOCRDetectionDataError(f"tile {tile_id} has invalid polygon point")
        xs = {float(point[0]) for point in points}
        ys = {float(point[1]) for point in points}
        if len(xs) != 2 or len(ys) != 2:
            raise TorchOCRDetectionDataError(
                f"tile {tile_id} annotation must be an axis-aligned rectangle"
            )
        minimum_x, maximum_x = min(xs), max(xs)
        minimum_y, maximum_y = min(ys), max(ys)
        expected_points = [
            [minimum_x, minimum_y],
            [maximum_x, minimum_y],
            [maximum_x, maximum_y],
            [minimum_x, maximum_y],
        ]
        actual_points = [
            [float(point[0]), float(point[1])] for point in points
        ]
        if actual_points != expected_points:
            raise TorchOCRDetectionDataError(
                f"tile {tile_id} annotation points violate frozen clockwise order"
            )
        if (
            minimum_x < 0
            or minimum_y < 0
            or maximum_x > output_size[0]
            or maximum_y > output_size[1]
        ):
            raise TorchOCRDetectionDataError(
                f"tile {tile_id} annotation lies outside frozen output_size"
            )
        if annotation["ignore"]:
            ignored += 1
        else:
            positive += 1
    if row.get("positive_count") != positive or row.get("ignored_count") != ignored:
        raise TorchOCRDetectionDataError(f"tile {tile_id} annotation counts mismatch")
    if row.get("empty_tile") is not (positive == 0):
        raise TorchOCRDetectionDataError(f"tile {tile_id} empty_tile flag mismatch")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise TorchOCRDetectionDataError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TorchOCRDetectionDataError(f"JSON root is not an object: {path}")
    return value


def _read_jsonl(path: Path, identifier: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise TorchOCRDetectionDataError(f"cannot open {path}: {exc}") from exc
    with handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw, parse_constant=_reject)
            except (json.JSONDecodeError, ValueError) as exc:
                raise TorchOCRDetectionDataError(
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise TorchOCRDetectionDataError(f"row {line_number} is not an object")
            identity = value.get(identifier)
            if not isinstance(identity, str) or not identity or identity in seen:
                raise TorchOCRDetectionDataError(
                    f"row {line_number} has missing/duplicate {identifier}"
                )
            seen.add(identity)
            rows.append(value)
    return rows


def _verify_committed_artifact(root: Path, manifest: dict[str, Any], relative: str) -> None:
    expected = manifest["artifacts"].get(relative)
    if not _is_sha256(expected):
        raise TorchOCRDetectionDataError(f"manifest lacks SHA-256 for {relative}")
    path = _safe_child(root, relative)
    actual = _sha256_file(path)
    if actual != expected:
        raise TorchOCRDetectionDataError(
            f"committed artifact {relative} SHA-256 mismatch: expected {expected}, got {actual}"
        )


def _safe_child(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise TorchOCRDetectionDataError(f"path must be relative: {relative!r}")
    resolved = (root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise TorchOCRDetectionDataError(f"path escapes root: {relative!r}") from exc
    if not resolved.is_file():
        raise TorchOCRDetectionDataError(f"file does not exist: {resolved}")
    return resolved


def _decode_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as probe:
            if not probe.format:
                raise TorchOCRDetectionDataError(f"image format is unknown: {path}")
            probe.verify()
        with Image.open(path) as opened:
            opened.load()
            return opened.copy()
    except TorchOCRDetectionDataError:
        raise
    except (OSError, ValueError) as exc:
        raise TorchOCRDetectionDataError(f"cannot decode image {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


__all__ = [
    "TorchOCRDetectionDataError",
    "TorchOCRDetectionDataset",
    "detection_collate",
]
