from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from PIL import Image

from pcbparse.data.manifest import write_json, write_jsonl


_SCHEMA_VERSION = 1
_MATERIALIZER_VERSION = "component-pin-index-v1"
_SPLITS = frozenset({"train", "val", "test"})
_SIDES = ("L", "R", "T", "B")


@dataclass(frozen=True, slots=True)
class ComponentPinMaterializeConfig:
    """Geometry and augmentation contract for Component/Pin training indexes.

    Images are decoded lazily by default.  Component tiles remain at source
    scale and are padded to ``tile_size``.  Full-image and Pin views use a
    deterministic letterbox transform whose exact forward and inverse matrices
    are written into the index.
    """

    tile_size: int = 512
    tile_overlap: int = 64
    component_full_size: int = 512
    pin_view_size: int = 512
    pin_roi_margin_px: int = 64
    pin_strip_outside_px: int = 64
    pin_strip_inside_px: int = 16
    pin_strip_tangent_margin_px: int = 24
    heatmap_stride: int = 4
    heatmap_sigma: float = 2.0
    heatmap_truncate_sigma: float = 3.0
    bbox_jitter_max_fraction: float = 0.12
    bbox_jitter_max_pixels: int = 24
    bbox_jitter_probability: float = 0.8
    augmentation_seed: int = 20260813

    def __post_init__(self) -> None:
        integer_positive = (
            "tile_size",
            "component_full_size",
            "pin_view_size",
            "pin_roi_margin_px",
            "pin_strip_outside_px",
            "pin_strip_inside_px",
            "pin_strip_tangent_margin_px",
            "heatmap_stride",
            "bbox_jitter_max_pixels",
        )
        for field_name in integer_positive:
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if type(self.tile_overlap) is not int or not 0 <= self.tile_overlap < self.tile_size:
            raise ValueError("tile_overlap must satisfy 0 <= overlap < tile_size")
        if self.pin_roi_margin_px < max(
            self.pin_strip_outside_px, self.pin_strip_tangent_margin_px
        ):
            raise ValueError(
                "pin_roi_margin_px must contain the configured side-strip extent"
            )
        for field_name in ("heatmap_sigma", "heatmap_truncate_sigma"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and positive")
        if not 0.0 <= self.bbox_jitter_max_fraction <= 1.0:
            raise ValueError("bbox_jitter_max_fraction must be in [0, 1]")
        if not 0.0 <= self.bbox_jitter_probability <= 1.0:
            raise ValueError("bbox_jitter_probability must be in [0, 1]")
        if type(self.augmentation_seed) is not int or self.augmentation_seed < 0:
            raise ValueError("augmentation_seed must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ComponentPinMaterializationResult:
    component_index: Path
    pin_index: Path
    summary: Path
    binding_sha256: str
    component_view_count: int
    pin_component_count: int
    crop_count: int


def materialize_component_pin_training_data(
    workspace_root: str | Path,
    output_root: str | Path,
    *,
    derived_root: str | Path = "data/derived/dataset300-v2",
    source_root: str | Path = "Dataset300/Dataset300",
    source_manifest: str | Path = "data/manifests/dataset300.jsonl",
    split_manifest: str | Path = "data/splits/dataset300_split.jsonl",
    config: ComponentPinMaterializeConfig | None = None,
    sample_ids: Iterable[str] | None = None,
    write_crops: bool = False,
    verify_source_hashes: bool = True,
) -> ComponentPinMaterializationResult:
    """Build deterministic, hash-bound Component and Pin training indexes.

    The source dataset and derived manifests are opened read-only.  Index files
    are replaced atomically, and ``summary.json`` is published last as a commit
    marker containing the exact hashes of both indexes.  ``sample_ids`` exists
    for smoke tests and experiments; lineage validation still covers the full
    source, split, Component, and Pin manifests.
    """

    settings = config or ComponentPinMaterializeConfig()
    workspace = Path(workspace_root).expanduser().resolve(strict=True)
    derived = _resolve_input(workspace, derived_root, "derived_root")
    source = _resolve_input(workspace, source_root, "source_root")
    source_manifest_path = _resolve_input(
        workspace, source_manifest, "source_manifest"
    )
    split_manifest_path = _resolve_input(workspace, split_manifest, "split_manifest")
    destination = Path(output_root).expanduser().resolve(strict=False)
    _validate_output_location(destination, source, derived)

    summary_path = derived / "summary.json"
    summary = _read_json_object(summary_path)
    task_files = summary.get("task_files")
    if not isinstance(task_files, dict):
        raise ValueError("derived summary.task_files must be an object")
    component_path = _safe_child(
        derived, task_files.get("component_detection"), "component task manifest"
    )
    pin_path = _safe_child(derived, task_files.get("pin_keypoints"), "pin task manifest")

    source_rows = _read_unique_jsonl(source_manifest_path, "source manifest")
    split_rows = _read_unique_jsonl(split_manifest_path, "split manifest")
    component_rows = list(_iter_jsonl(component_path, "component manifest"))
    pin_rows = list(_iter_jsonl(pin_path, "pin manifest"))
    source_by_sample, split_by_sample, components_by_sample, pins_by_owner = _validate_lineage(
        summary=summary,
        source_rows=source_rows,
        split_rows=split_rows,
        component_rows=component_rows,
        pin_rows=pin_rows,
        split_manifest_sha256=_sha256_file(split_manifest_path),
    )

    selected = _select_samples(sample_ids, set(source_by_sample))
    source_image_hashes = _validate_selected_images(
        selected,
        source_by_sample,
        source,
        verify_hashes=verify_source_hashes,
    )
    config_payload = {
        "schema_version": _SCHEMA_VERSION,
        "materializer_version": _MATERIALIZER_VERSION,
        **asdict(settings),
        "component_tile_partial_box_policy": "clip_and_ignore",
        "component_full_view_policy": "letterbox_retain_all_boxes",
        "pin_decode_policy": "lazy_source_crop_letterbox",
        "crop_encoding": "rgb_png_v1" if write_crops else None,
    }
    materialize_config_sha = _sha256_json(config_payload)
    lineage = {
        "derivation_version": summary.get("derivation_version"),
        "derivation_config_sha256": summary.get("config_hash"),
        "materialize_config_sha256": materialize_config_sha,
        "selection_sha256": _sha256_json(selected),
        "selected_sample_count": len(selected),
        "inputs": {
            "summary": _input_binding(summary_path, workspace),
            "source_manifest": _input_binding(source_manifest_path, workspace),
            "split_manifest": _input_binding(split_manifest_path, workspace),
            "component_manifest": _input_binding(component_path, workspace),
            "pin_manifest": _input_binding(pin_path, workspace),
        },
    }
    binding_sha = _sha256_json(lineage)

    categories, category_ids = _component_categories(component_rows)
    component_index = _build_component_index(
        selected=selected,
        source_by_sample=source_by_sample,
        components_by_sample=components_by_sample,
        source_image_hashes=source_image_hashes,
        settings=settings,
        categories=categories,
        category_ids=category_ids,
        lineage=lineage,
        binding_sha256=binding_sha,
    )
    pin_index = _build_pin_index(
        selected=selected,
        source_by_sample=source_by_sample,
        components_by_sample=components_by_sample,
        pins_by_owner=pins_by_owner,
        source_image_hashes=source_image_hashes,
        settings=settings,
        binding_sha256=binding_sha,
    )

    crop_manifest: list[dict[str, str]] = []
    if write_crops:
        crop_manifest = _write_all_crops(
            component_index["images"], pin_index, source, destination
        )

    destination.mkdir(parents=True, exist_ok=True)
    component_output = destination / "component_coco.json"
    pin_output = destination / "pin_index.jsonl"
    summary_output = destination / "summary.json"
    write_json(component_index, component_output)
    write_jsonl(pin_index, pin_output)
    output_hashes = {
        "component_coco.json": _sha256_file(component_output),
        "pin_index.jsonl": _sha256_file(pin_output),
    }
    materialization_summary = {
        "schema_version": _SCHEMA_VERSION,
        "materializer_version": _MATERIALIZER_VERSION,
        "binding_sha256": binding_sha,
        "lineage": lineage,
        "config": config_payload,
        "selection": selected,
        "counts": {
            "component_views": len(component_index["images"]),
            "component_annotations": len(component_index["annotations"]),
            "pin_components": len(pin_index),
            "pin_views": sum(len(record["views"]) for record in pin_index),
            "pin_targets": sum(len(record["pins"]) for record in pin_index),
            "written_crops": len(crop_manifest),
        },
        "outputs": {
            name: {"sha256": digest} for name, digest in sorted(output_hashes.items())
        },
        "crop_manifest_sha256": _sha256_json(crop_manifest) if crop_manifest else None,
    }
    # This is the commit marker and must always be published last.
    write_json(materialization_summary, summary_output)
    return ComponentPinMaterializationResult(
        component_index=component_output,
        pin_index=pin_output,
        summary=summary_output,
        binding_sha256=binding_sha,
        component_view_count=len(component_index["images"]),
        pin_component_count=len(pin_index),
        crop_count=len(crop_manifest),
    )


def decode_indexed_view(
    view: Mapping[str, Any], source_root: str | Path
) -> Image.Image:
    """Decode one indexed view on demand according to its recorded transform."""

    source_image = view.get("source_image")
    if not isinstance(source_image, str) or not source_image:
        raise ValueError("indexed view lacks source_image")
    root = Path(source_root).expanduser().resolve(strict=True)
    image_path = _safe_child(root, source_image, "indexed source image")
    crop = _bbox_values(view.get("crop_bbox"), "view.crop_bbox")
    output_size = view.get("output_size")
    resized_size = view.get("resized_size")
    padding = view.get("padding")
    if not (
        isinstance(output_size, list)
        and len(output_size) == 2
        and all(type(value) is int and value > 0 for value in output_size)
    ):
        raise ValueError("view.output_size must contain two positive integers")
    if not (
        isinstance(resized_size, list)
        and len(resized_size) == 2
        and all(type(value) is int and value > 0 for value in resized_size)
    ):
        raise ValueError("view.resized_size must contain two positive integers")
    if not (
        isinstance(padding, list)
        and len(padding) == 4
        and all(type(value) is int and value >= 0 for value in padding)
    ):
        raise ValueError("view.padding must contain four non-negative integers")
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        region = image.crop(tuple(int(value) for value in crop))
    if region.size != tuple(resized_size):
        region = region.resize(tuple(resized_size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", tuple(output_size), (255, 255, 255))
    canvas.paste(region, (padding[0], padding[1]))
    return canvas


def transform_point(
    point: Sequence[float], matrix: Sequence[Sequence[float]]
) -> tuple[float, float]:
    """Apply a recorded 3x3 affine matrix (useful to verify round trips)."""

    if len(point) != 2 or len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("point must have length 2 and matrix must be 3x3")
    x, y = float(point[0]), float(point[1])
    denominator = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
    if abs(denominator) < 1e-15:
        raise ValueError("affine transform produced a point at infinity")
    return (
        (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / denominator,
        (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / denominator,
    )


def _validate_lineage(
    *,
    summary: dict[str, Any],
    source_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    pin_rows: list[dict[str, Any]],
    split_manifest_sha256: str,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[tuple[str, str], list[dict[str, Any]]],
]:
    derivation_version = summary.get("derivation_version")
    config_sha = summary.get("config_hash")
    if not isinstance(derivation_version, str) or not derivation_version:
        raise ValueError("derived summary lacks derivation_version")
    if not _is_sha256(config_sha):
        raise ValueError("derived summary.config_hash must be a SHA-256 digest")
    summary_config = summary.get("config")
    if not isinstance(summary_config, dict):
        raise ValueError("derived summary.config must be an object")
    if summary_config.get("split_manifest_sha256") != split_manifest_sha256:
        raise ValueError("split manifest SHA-256 does not match derived lineage")

    source_by_sample = {str(row["sample_id"]): row for row in source_rows}
    split_by_sample = {str(row["sample_id"]): row for row in split_rows}
    if set(source_by_sample) != set(split_by_sample):
        raise ValueError("source and split manifests have different sample coverage")
    # Keep the source manifest immutable on disk while enriching the in-memory
    # records with the authoritative split used by the two output indexes.
    source_by_sample = {
        sample_id: {**record, "split": split_by_sample[sample_id].get("split")}
        for sample_id, record in source_by_sample.items()
    }
    expected_sample_count = summary.get("sample_count")
    if expected_sample_count != len(source_by_sample):
        raise ValueError(
            f"summary sample_count={expected_sample_count!r} but source has "
            f"{len(source_by_sample)} samples"
        )
    groups: dict[str, set[str]] = defaultdict(set)
    for sample_id, split_record in split_by_sample.items():
        split = split_record.get("split")
        if split not in _SPLITS:
            raise ValueError(f"sample {sample_id!r} has invalid split {split!r}")
        source_record = source_by_sample[sample_id]
        source_group = source_record.get("duplicate_group")
        split_group = split_record.get("duplicate_group")
        if source_group != split_group:
            raise ValueError(f"sample {sample_id!r} duplicate_group changed in split manifest")
        groups[str(source_group)].add(str(split))
    leaking = sorted(group for group, splits in groups.items() if len(splits) > 1)
    if leaking:
        raise ValueError(f"duplicate groups cross split boundaries: {leaking[:10]}")

    expected_counts = summary.get("counts")
    if not isinstance(expected_counts, dict):
        raise ValueError("derived summary.counts must be an object")
    if expected_counts.get("component_detection") != len(component_rows):
        raise ValueError("component manifest count does not match derived summary")
    if expected_counts.get("pin_keypoints") != len(pin_rows):
        raise ValueError("pin manifest count does not match derived summary")

    components_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    component_by_owner: dict[tuple[str, str], dict[str, Any]] = {}
    component_record_ids: set[str] = set()
    for record in component_rows:
        _validate_derived_record(
            record,
            task="component_detection",
            derivation_version=derivation_version,
            config_sha=config_sha,
            source_by_sample=source_by_sample,
            split_by_sample=split_by_sample,
        )
        sample_id = record["sample_id"]
        record_id = record.get("record_id")
        component_key = record.get("component_key")
        component_id = record.get("component_id")
        if not all(
            isinstance(value, str) and value
            for value in (record_id, component_key, component_id)
        ):
            raise ValueError(f"component record in sample {sample_id!r} lacks stable IDs")
        category = record.get("category")
        category_id = record.get("category_id")
        if not isinstance(category, str) or not category:
            raise ValueError(f"component {record_id!r} lacks a category")
        if isinstance(category_id, bool) or not isinstance(category_id, (str, int)):
            raise ValueError(f"component {record_id!r} lacks a stable category_id")
        if record_id in component_record_ids:
            raise ValueError(f"duplicate component record_id {record_id!r}")
        component_record_ids.add(record_id)
        owner_key = (sample_id, component_key)
        if owner_key in component_by_owner:
            raise ValueError(f"duplicate component key {component_key!r} in {sample_id!r}")
        bbox = _bbox_values(record.get("bbox"), f"component {record_id} bbox")
        canvas = source_by_sample[sample_id].get("canvas")
        width, height = _canvas_size(canvas, sample_id)
        if not (0 <= bbox[0] < bbox[2] <= width and 0 <= bbox[1] < bbox[3] <= height):
            raise ValueError(f"component {record_id!r} bbox is outside its canvas")
        component_by_owner[owner_key] = record
        components_by_sample[sample_id].append(record)

    pins_by_owner: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    pin_record_ids: set[str] = set()
    for record in pin_rows:
        _validate_derived_record(
            record,
            task="pin_keypoints",
            derivation_version=derivation_version,
            config_sha=config_sha,
            source_by_sample=source_by_sample,
            split_by_sample=split_by_sample,
        )
        sample_id = record["sample_id"]
        record_id = record.get("record_id")
        owner_key_value = record.get("owner_component_key")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"pin record in sample {sample_id!r} lacks record_id")
        if not all(
            isinstance(record.get(field), str) and record.get(field)
            for field in ("pin_id", "pin_key")
        ):
            raise ValueError(f"pin {record_id!r} lacks stable pin_id/pin_key")
        if record_id in pin_record_ids:
            raise ValueError(f"duplicate pin record_id {record_id!r}")
        pin_record_ids.add(record_id)
        if not isinstance(owner_key_value, str) or not owner_key_value:
            raise ValueError(f"pin {record_id!r} lacks owner_component_key")
        owner = component_by_owner.get((sample_id, owner_key_value))
        if owner is None:
            raise ValueError(f"pin {record_id!r} references a missing component")
        if record.get("owner_component_id") != owner.get("component_id"):
            raise ValueError(f"pin {record_id!r} owner_component_id disagrees with owner")
        _point_values(record.get("tip"), f"pin {record_id} tip")
        _point_values(record.get("base"), f"pin {record_id} base")
        pins_by_owner[(sample_id, owner_key_value)].append(record)

    for rows in components_by_sample.values():
        rows.sort(key=lambda item: str(item["record_id"]))
    for rows in pins_by_owner.values():
        rows.sort(key=lambda item: str(item["record_id"]))
    return source_by_sample, split_by_sample, components_by_sample, pins_by_owner


def _validate_derived_record(
    record: dict[str, Any],
    *,
    task: str,
    derivation_version: str,
    config_sha: str,
    source_by_sample: dict[str, dict[str, Any]],
    split_by_sample: dict[str, dict[str, Any]],
) -> None:
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or sample_id not in source_by_sample:
        raise ValueError(f"{task} record references unknown sample {sample_id!r}")
    if record.get("task") != task:
        raise ValueError(f"record {record.get('record_id')!r} has wrong task")
    if record.get("derivation_version") != derivation_version:
        raise ValueError(f"record {record.get('record_id')!r} changed derivation version")
    if record.get("config_hash") != config_sha:
        raise ValueError(f"record {record.get('record_id')!r} changed config hash")
    split = split_by_sample[sample_id]["split"]
    if record.get("split") != split:
        raise ValueError(f"record {record.get('record_id')!r} changed split")
    source_record = source_by_sample[sample_id]
    if record.get("duplicate_group") != source_record.get("duplicate_group"):
        raise ValueError(f"record {record.get('record_id')!r} changed duplicate group")
    files = source_record.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"source manifest sample {sample_id!r} lacks files")
    for field, role in (("source_image", "image"), ("source_json", "json")):
        metadata = files.get(role)
        expected = metadata.get("path") if isinstance(metadata, dict) else None
        if record.get(field) != expected:
            raise ValueError(f"record {record.get('record_id')!r} changed {field}")


def _validate_selected_images(
    selected: list[str],
    source_by_sample: dict[str, dict[str, Any]],
    source_root: Path,
    *,
    verify_hashes: bool,
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for sample_id in selected:
        record = source_by_sample[sample_id]
        width, height = _canvas_size(record.get("canvas"), sample_id)
        files = record.get("files")
        image_metadata = files.get("image") if isinstance(files, dict) else None
        if not isinstance(image_metadata, dict):
            raise ValueError(f"source sample {sample_id!r} lacks image metadata")
        relative = image_metadata.get("path")
        image_path = _safe_child(source_root, relative, f"source image {sample_id}")
        if not image_path.is_file():
            raise ValueError(f"source image is missing: {image_path}")
        with Image.open(image_path) as image:
            if image.size != (width, height):
                raise ValueError(
                    f"source image {sample_id!r} dimensions {image.size} disagree with "
                    f"manifest {(width, height)}"
                )
            image.verify()
        expected_sha = image_metadata.get("sha256")
        if not _is_sha256(expected_sha):
            raise ValueError(f"source image {sample_id!r} lacks a valid SHA-256")
        actual_sha = _sha256_file(image_path) if verify_hashes else str(expected_sha)
        if actual_sha != expected_sha:
            raise ValueError(f"source image {sample_id!r} SHA-256 mismatch")
        hashes[sample_id] = actual_sha
    return hashes


def _build_component_index(
    *,
    selected: list[str],
    source_by_sample: dict[str, dict[str, Any]],
    components_by_sample: dict[str, list[dict[str, Any]]],
    source_image_hashes: dict[str, str],
    settings: ComponentPinMaterializeConfig,
    categories: list[dict[str, Any]],
    category_ids: dict[tuple[str, str], int],
    lineage: dict[str, Any],
    binding_sha256: str,
) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for sample_id in selected:
        source_record = source_by_sample[sample_id]
        width, height = _canvas_size(source_record["canvas"], sample_id)
        source_image = source_record["files"]["image"]["path"]
        view_specs = [
            _view_spec(
                view_id=f"component:{sample_id}:full",
                view_kind="full",
                sample_id=sample_id,
                split=source_record.get("split"),
                source_image=source_image,
                image_sha256=source_image_hashes[sample_id],
                crop_bbox=(0, 0, width, height),
                output_size=(settings.component_full_size, settings.component_full_size),
                letterbox=True,
            )
        ]
        x_starts = _axis_starts(width, settings.tile_size, settings.tile_overlap)
        y_starts = _axis_starts(height, settings.tile_size, settings.tile_overlap)
        for row_index, top in enumerate(y_starts):
            for column_index, left in enumerate(x_starts):
                view_specs.append(
                    _view_spec(
                        view_id=(
                            f"component:{sample_id}:tile:{row_index:04d}:{column_index:04d}"
                        ),
                        view_kind="tile",
                        sample_id=sample_id,
                        split=source_record.get("split"),
                        source_image=source_image,
                        image_sha256=source_image_hashes[sample_id],
                        crop_bbox=(
                            left,
                            top,
                            min(width, left + settings.tile_size),
                            min(height, top + settings.tile_size),
                        ),
                        output_size=(settings.tile_size, settings.tile_size),
                        letterbox=False,
                    )
                )
        for view in view_specs:
            image_id = len(images) + 1
            image_record = {"id": image_id, "binding_sha256": binding_sha256, **view}
            images.append(image_record)
            crop_bbox = _bbox_values(view["crop_bbox"], "component crop")
            for component in components_by_sample.get(sample_id, []):
                source_bbox = _bbox_values(component["bbox"], "component bbox")
                intersection = _intersection(source_bbox, crop_bbox)
                if intersection is None:
                    continue
                partial = not _same_bbox(intersection, source_bbox)
                if view["view_kind"] == "full":
                    partial = False
                transformed = _transform_bbox(intersection, view["transform"]["forward"])
                category_key = (
                    str(component.get("category")), str(component.get("category_id"))
                )
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_ids[category_key],
                    "bbox": _xyxy_to_xywh(transformed),
                    "area": _round8(
                        max(0.0, transformed[2] - transformed[0])
                        * max(0.0, transformed[3] - transformed[1])
                    ),
                    "iscrowd": 1 if partial else 0,
                    "ignore": 1 if partial else 0,
                    "segmentation": [],
                    "partial_box": partial,
                    "partial_box_policy": "clip_and_ignore" if partial else "complete",
                    "component_id": component["component_id"],
                    "component_key": component["component_key"],
                    "source_record_id": component["record_id"],
                    "source_category": component.get("category"),
                    "source_category_id": component.get("category_id"),
                    "source_bbox": _bbox_list(source_bbox),
                    "source_raw_bbox": component.get("raw_bbox"),
                    "source_real_bbox": component.get("real_bbox"),
                    "source_issue_flags": list(component.get("issue_flags", [])),
                    "tile_intersection_bbox": _bbox_list(intersection),
                    "visible_fraction": _round8(
                        _bbox_area(intersection) / max(_bbox_area(source_bbox), 1e-12)
                    ),
                }
                annotations.append(annotation)
                annotation_id += 1
    return {
        "info": {
            "description": "PCB Component detection: deterministic full+tile index",
            "version": _MATERIALIZER_VERSION,
            "schema_version": _SCHEMA_VERSION,
            "binding_sha256": binding_sha256,
            "lineage": lineage,
            "partial_box_policy": (
                "Tile intersections are clipped for geometry but marked ignore=1 and "
                "iscrowd=1; complete boxes remain normal labels; full views retain every box."
            ),
        },
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }


def _build_pin_index(
    *,
    selected: list[str],
    source_by_sample: dict[str, dict[str, Any]],
    components_by_sample: dict[str, list[dict[str, Any]]],
    pins_by_owner: dict[tuple[str, str], list[dict[str, Any]]],
    source_image_hashes: dict[str, str],
    settings: ComponentPinMaterializeConfig,
    binding_sha256: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    record_number = 0
    for sample_id in selected:
        source_record = source_by_sample[sample_id]
        width, height = _canvas_size(source_record["canvas"], sample_id)
        source_image = source_record["files"]["image"]["path"]
        split = str(source_record.get("split"))
        for component in components_by_sample.get(sample_id, []):
            record_number += 1
            component_bbox = _bbox_values(component["bbox"], "component bbox")
            owner_key = (sample_id, str(component["component_key"]))
            pins = pins_by_owner.get(owner_key, [])
            roi_bbox = _expand_bbox(
                component_bbox, settings.pin_roi_margin_px, width, height
            )
            view_windows = [("roi", None, roi_bbox), *_side_windows(
                component_bbox, width, height, settings
            )]
            digest = hashlib.sha256(
                f"{sample_id}\0{component['component_key']}".encode("utf-8")
            ).hexdigest()[:20]
            views: list[dict[str, Any]] = []
            for kind, side, crop_bbox in view_windows:
                suffix = "roi" if side is None else f"strip-{side}"
                view = _view_spec(
                    view_id=f"pin:{sample_id}:{digest}:{suffix}",
                    view_kind=kind,
                    sample_id=sample_id,
                    split=split,
                    source_image=source_image,
                    image_sha256=source_image_hashes[sample_id],
                    crop_bbox=crop_bbox,
                    output_size=(settings.pin_view_size, settings.pin_view_size),
                    letterbox=True,
                )
                view["side"] = side
                view["targets"] = [
                    _pin_view_target(pin, view, settings) for pin in pins
                ]
                views.append(view)
            jitter_seed = hashlib.sha256(
                (
                    f"{settings.augmentation_seed}\0{sample_id}\0"
                    f"{component['component_key']}"
                ).encode("utf-8")
            ).hexdigest()
            original_pins = [_original_pin_target(pin) for pin in pins]
            output.append(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "materializer_version": _MATERIALIZER_VERSION,
                    "binding_sha256": binding_sha256,
                    "record_id": f"pin-component:{sample_id}:{digest}",
                    "record_number": record_number,
                    "sample_id": sample_id,
                    "split": split,
                    "duplicate_group": source_record.get("duplicate_group"),
                    "source_image": source_image,
                    "source_image_sha256": source_image_hashes[sample_id],
                    "component_id": component["component_id"],
                    "component_key": component["component_key"],
                    "source_component_record_id": component["record_id"],
                    "component_bbox": _bbox_list(component_bbox),
                    "component_raw_bbox": component.get("raw_bbox"),
                    "component_category": component.get("category"),
                    "negative_component": not bool(pins),
                    "pins": original_pins,
                    "bbox_jitter": {
                        "policy": "runtime_independent_edge_uniform_v1",
                        "enabled": split == "train",
                        "applies_to": ["train"],
                        "probability": settings.bbox_jitter_probability,
                        "max_edge_offset_fraction": settings.bbox_jitter_max_fraction,
                        "max_edge_offset_pixels": settings.bbox_jitter_max_pixels,
                        "clamp_to_canvas": True,
                        "minimum_result_size_pixels": 2,
                        "seed_sha256": jitter_seed,
                        "nominal_component_bbox": _bbox_list(component_bbox),
                        "nominal_roi_bbox": _bbox_list(roi_bbox),
                        "materialized_jitter_applied": False,
                    },
                    "views": views,
                }
            )
    return output


def _pin_view_target(
    pin: dict[str, Any],
    view: dict[str, Any],
    settings: ComponentPinMaterializeConfig,
) -> dict[str, Any]:
    tip = _point_values(pin["tip"], "pin tip")
    base = _point_values(pin["base"], "pin base")
    matrix = view["transform"]["forward"]
    tip_view = transform_point(tip, matrix)
    base_view = transform_point(base, matrix)
    width, height = view["output_size"]
    crop = _bbox_values(view["crop_bbox"], "pin view crop")
    pad_left, pad_top, _, _ = view["padding"]
    resized_width, resized_height = view["resized_size"]
    # Letterbox padding is not source imagery.  Requiring membership in both
    # the source crop and resized content prevents a nearby out-of-strip pin
    # from being mislabeled visible merely because its extrapolated coordinate
    # happens to land on the white padding canvas.
    visible = (
        crop[0] <= tip[0] < crop[2]
        and crop[1] <= tip[1] < crop[3]
        and pad_left <= tip_view[0] < pad_left + resized_width
        and pad_top <= tip_view[1] < pad_top + resized_height
        and 0 <= tip_view[0] < width
        and 0 <= tip_view[1] < height
    )
    heatmap_point = (
        [_round8(tip_view[0] / settings.heatmap_stride), _round8(tip_view[1] / settings.heatmap_stride)]
        if visible
        else None
    )
    vector = (tip_view[0] - base_view[0], tip_view[1] - base_view[1])
    vector_norm = math.hypot(*vector)
    unit = (
        [_round8(vector[0] / vector_norm), _round8(vector[1] / vector_norm)]
        if vector_norm > 1e-12
        else [0.0, 0.0]
    )
    return {
        "pin_id": pin.get("pin_id"),
        "pin_key": pin.get("pin_key"),
        "source_record_id": pin.get("record_id"),
        "tip_original": [_round8(tip[0]), _round8(tip[1])],
        "base_original": [_round8(base[0]), _round8(base[1])],
        "tip_view": [_round8(tip_view[0]), _round8(tip_view[1])] if visible else None,
        "base_view": [_round8(base_view[0]), _round8(base_view[1])],
        "visibility": 2 if visible else 0,
        "visibility_convention": "coco_keypoint_0_absent_2_visible",
        "orientation_original": pin.get("orientation"),
        "orientation_view": {
            "vector": [_round8(vector[0]), _round8(vector[1])],
            "unit_vector": unit,
            "angle_degrees_image_y_down": _round8(
                math.degrees(math.atan2(vector[1], vector[0])) % 360.0
            ) if vector_norm > 1e-12 else None,
        },
        "heatmap": {
            "rasterized": False,
            "target_type": "gaussian_2d_peak_1",
            "output_stride": settings.heatmap_stride,
            "output_size": [
                int(math.ceil(view["output_size"][0] / settings.heatmap_stride)),
                int(math.ceil(view["output_size"][1] / settings.heatmap_stride)),
            ],
            "point": heatmap_point,
            "sigma_heatmap_pixels": settings.heatmap_sigma,
            "truncate_sigma": settings.heatmap_truncate_sigma,
            "ignore_when_visibility_zero": True,
        },
    }


def _original_pin_target(pin: dict[str, Any]) -> dict[str, Any]:
    return {
        "pin_id": pin.get("pin_id"),
        "pin_key": pin.get("pin_key"),
        "source_record_id": pin.get("record_id"),
        "tip": pin.get("tip"),
        "base": pin.get("base"),
        "orientation": pin.get("orientation"),
        "edge_id": pin.get("edge_id"),
        "label_refs": pin.get("label_refs"),
        "missing_label_mask": pin.get("missing_label_mask"),
        "source_issue_flags": list(pin.get("issue_flags", [])),
    }


def _view_spec(
    *,
    view_id: str,
    view_kind: str,
    sample_id: str,
    split: Any,
    source_image: str,
    image_sha256: str,
    crop_bbox: Sequence[int | float],
    output_size: tuple[int, int],
    letterbox: bool,
) -> dict[str, Any]:
    x1, y1, x2, y2 = (int(value) for value in crop_bbox)
    crop_width = x2 - x1
    crop_height = y2 - y1
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(f"view {view_id!r} has an empty crop")
    output_width, output_height = output_size
    if letterbox:
        scale = min(output_width / crop_width, output_height / crop_height)
        resized_width = max(1, min(output_width, int(round(crop_width * scale))))
        resized_height = max(1, min(output_height, int(round(crop_height * scale))))
        pad_left = (output_width - resized_width) // 2
        pad_top = (output_height - resized_height) // 2
    else:
        resized_width = crop_width
        resized_height = crop_height
        if resized_width > output_width or resized_height > output_height:
            raise ValueError(f"non-letterboxed view {view_id!r} exceeds its output canvas")
        pad_left = 0
        pad_top = 0
    pad_right = output_width - resized_width - pad_left
    pad_bottom = output_height - resized_height - pad_top
    scale_x = resized_width / crop_width
    scale_y = resized_height / crop_height
    translate_x = pad_left - scale_x * x1
    translate_y = pad_top - scale_y * y1
    forward = [
        [_round12(scale_x), 0.0, _round12(translate_x)],
        [0.0, _round12(scale_y), _round12(translate_y)],
        [0.0, 0.0, 1.0],
    ]
    inverse = [
        [_round12(1.0 / scale_x), 0.0, _round12(-translate_x / scale_x)],
        [0.0, _round12(1.0 / scale_y), _round12(-translate_y / scale_y)],
        [0.0, 0.0, 1.0],
    ]
    return {
        "view_id": view_id,
        "view_kind": view_kind,
        "sample_id": sample_id,
        "split": split,
        "source_image": source_image,
        "source_image_sha256": image_sha256,
        "crop_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "source_size": [crop_width, crop_height],
        "resized_size": [resized_width, resized_height],
        "output_size": [output_width, output_height],
        "padding": [pad_left, pad_top, pad_right, pad_bottom],
        "transform": {
            "convention": "original_xy_to_view_xy_homogeneous",
            "forward": forward,
            "inverse": inverse,
            "roundtrip_tolerance_pixels": 1e-8,
        },
        "decode": {
            "mode": "lazy",
            "color_mode": "RGB",
            "resample": "bilinear",
            "padding_rgb": [255, 255, 255],
        },
    }


def _side_windows(
    bbox: tuple[float, float, float, float],
    canvas_width: int,
    canvas_height: int,
    settings: ComponentPinMaterializeConfig,
) -> list[tuple[str, str, tuple[int, int, int, int]]]:
    x1, y1, x2, y2 = bbox
    outside = settings.pin_strip_outside_px
    inside = settings.pin_strip_inside_px
    tangent = settings.pin_strip_tangent_margin_px
    windows = {
        "L": (x1 - outside, y1 - tangent, x1 + inside, y2 + tangent),
        "R": (x2 - inside, y1 - tangent, x2 + outside, y2 + tangent),
        "T": (x1 - tangent, y1 - outside, x2 + tangent, y1 + inside),
        "B": (x1 - tangent, y2 - inside, x2 + tangent, y2 + outside),
    }
    output: list[tuple[str, str, tuple[int, int, int, int]]] = []
    for side in _SIDES:
        output.append(
            (
                "side_strip",
                side,
                _clip_integer_bbox(windows[side], canvas_width, canvas_height),
            )
        )
    return output


def _write_all_crops(
    component_views: list[dict[str, Any]],
    pin_records: list[dict[str, Any]],
    source_root: Path,
    output_root: Path,
) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    for view in component_views:
        path = Path("crops") / "component" / f"{int(view['id']):08d}.png"
        digest = _write_view_crop(view, source_root, output_root / path)
        view["materialized_crop"] = {"path": path.as_posix(), "sha256": digest}
        manifest.append({"path": path.as_posix(), "sha256": digest})
    pin_view_number = 0
    for record in pin_records:
        for view in record["views"]:
            pin_view_number += 1
            path = Path("crops") / "pin" / f"{pin_view_number:08d}.png"
            digest = _write_view_crop(view, source_root, output_root / path)
            view["materialized_crop"] = {"path": path.as_posix(), "sha256": digest}
            manifest.append({"path": path.as_posix(), "sha256": digest})
    return manifest


def _write_view_crop(
    view: dict[str, Any], source_root: Path, destination: Path
) -> str:
    rendered = decode_indexed_view(view, source_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as handle:
            rendered.save(handle, format="PNG", optimize=False, compress_level=6)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        rendered.close()
    return _sha256_file(destination)


def _component_categories(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], int]]:
    pairs = {(str(record.get("category")), str(record.get("category_id"))) for record in records}

    def sort_key(pair: tuple[str, str]) -> tuple[int, int | str, str]:
        try:
            return (0, int(pair[1]), pair[0])
        except ValueError:
            return (1, pair[1], pair[0])

    ordered = sorted(pairs, key=sort_key)
    mapping = {pair: index for index, pair in enumerate(ordered, start=1)}
    categories = [
        {
            "id": mapping[pair],
            "name": pair[0],
            "supercategory": "pcb_component",
            "source_category_id": pair[1],
        }
        for pair in ordered
    ]
    return categories, mapping


def _select_samples(requested: Iterable[str] | None, available: set[str]) -> list[str]:
    if requested is None:
        return sorted(available)
    values = list(requested)
    if not values:
        raise ValueError("sample_ids cannot be empty")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("sample_ids must contain non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError("sample_ids contains duplicates")
    unknown = sorted(set(values) - available)
    if unknown:
        raise ValueError(f"sample_ids contains unknown samples: {unknown[:10]}")
    return sorted(values)


def _axis_starts(length: int, size: int, overlap: int) -> list[int]:
    if length <= size:
        return [0]
    stride = size - overlap
    starts = list(range(0, length - size + 1, stride))
    final = length - size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _expand_bbox(
    bbox: tuple[float, float, float, float], margin: int, width: int, height: int
) -> tuple[int, int, int, int]:
    return _clip_integer_bbox(
        (bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin),
        width,
        height,
    )


def _clip_integer_bbox(
    bbox: Sequence[float], width: int, height: int
) -> tuple[int, int, int, int]:
    x1 = max(0, min(width, int(math.floor(bbox[0]))))
    y1 = max(0, min(height, int(math.floor(bbox[1]))))
    x2 = max(0, min(width, int(math.ceil(bbox[2]))))
    y2 = max(0, min(height, int(math.ceil(bbox[3]))))
    # Components may touch a canvas edge.  Keep every side view decodable even
    # when the nominal outside half is fully clipped away.
    if x2 <= x1:
        x1, x2 = max(0, min(width - 1, x1)), max(1, min(width, x1 + 1))
    if y2 <= y1:
        y1, y2 = max(0, min(height - 1, y1)), max(1, min(height, y1 + 1))
    return x1, y1, x2, y2


def _intersection(
    first: Sequence[float], second: Sequence[float]
) -> tuple[float, float, float, float] | None:
    result = (
        max(first[0], second[0]),
        max(first[1], second[1]),
        min(first[2], second[2]),
        min(first[3], second[3]),
    )
    return result if result[2] > result[0] and result[3] > result[1] else None


def _transform_bbox(
    bbox: Sequence[float], matrix: Sequence[Sequence[float]]
) -> tuple[float, float, float, float]:
    points = [
        transform_point((bbox[0], bbox[1]), matrix),
        transform_point((bbox[2], bbox[1]), matrix),
        transform_point((bbox[0], bbox[3]), matrix),
        transform_point((bbox[2], bbox[3]), matrix),
    ]
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def _same_bbox(first: Sequence[float], second: Sequence[float]) -> bool:
    return all(abs(float(a) - float(b)) <= 1e-9 for a, b in zip(first, second))


def _bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _xyxy_to_xywh(bbox: Sequence[float]) -> list[float]:
    return [
        _round8(bbox[0]),
        _round8(bbox[1]),
        _round8(bbox[2] - bbox[0]),
        _round8(bbox[3] - bbox[1]),
    ]


def _bbox_list(bbox: Sequence[float]) -> list[float]:
    return [_round8(value) for value in bbox]


def _bbox_values(value: Any, label: str) -> tuple[float, float, float, float]:
    if isinstance(value, Mapping):
        raw = [value.get(key) for key in ("x1", "y1", "x2", "y2")]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 4:
        raw = list(value)
    else:
        raise ValueError(f"{label} must be x1/y1/x2/y2")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in raw
    ):
        raise ValueError(f"{label} contains non-finite coordinates")
    result = tuple(float(item) for item in raw)
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"{label} is empty or inverted")
    return result  # type: ignore[return-value]


def _point_values(value: Any, label: str) -> tuple[float, float]:
    if not (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            for item in value
        )
    ):
        raise ValueError(f"{label} must contain two finite coordinates")
    return float(value[0]), float(value[1])


def _canvas_size(value: Any, sample_id: str) -> tuple[int, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"source sample {sample_id!r} lacks canvas")
    width, height = value.get("width"), value.get("height")
    if type(width) is not int or width <= 0 or type(height) is not int or height <= 0:
        raise ValueError(f"source sample {sample_id!r} has invalid canvas")
    return width, height


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _read_unique_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = list(_iter_jsonl(path, label))
    identifiers: set[str] = set()
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{label} row lacks sample_id")
        if sample_id in identifiers:
            raise ValueError(f"{label} has duplicate sample_id {sample_id!r}")
        identifiers.add(sample_id)
    return rows


def _iter_jsonl(path: Path, label: str) -> Iterator[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"{label} contains blank line {line_number}")
                value = json.loads(
                    line,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
                if not isinstance(value, dict):
                    raise ValueError(f"{label} line {line_number} is not an object")
                yield value
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc


def _input_binding(path: Path, workspace: Path) -> dict[str, str]:
    return {"path": path.relative_to(workspace).as_posix(), "sha256": _sha256_file(path)}


def _resolve_input(workspace: Path, raw: str | Path, label: str) -> Path:
    requested = Path(raw).expanduser()
    resolved = (requested if requested.is_absolute() else workspace / requested).resolve(
        strict=True
    )
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"{label} escapes workspace: {raw!r}") from exc
    return resolved


def _safe_child(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} path must be a non-empty string")
    child = (root / Path(raw)).resolve(strict=False)
    try:
        child.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{label} path escapes its root: {raw!r}") from exc
    return child


def _validate_output_location(output: Path, source: Path, derived: Path) -> None:
    for protected, label in ((source, "source dataset"), (derived, "derived manifests")):
        if output == protected or _is_relative_to(output, protected):
            raise ValueError(f"output_root cannot be inside the read-only {label}")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _round8(value: float) -> float:
    return round(float(value), 8)


def _round12(value: float) -> float:
    return round(float(value), 12)


__all__ = [
    "ComponentPinMaterializationResult",
    "ComponentPinMaterializeConfig",
    "decode_indexed_view",
    "materialize_component_pin_training_data",
    "transform_point",
]
