from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from pcbparse.training.component_pin_materialize import (
    decode_indexed_view,
    transform_point,
)
from pcbparse.training.torch_component_model import normalize_category_table


_MATERIALIZER_VERSION = "component-pin-index-v1"
_VALID_SPLITS = frozenset({"train", "val", "test"})
_VALID_COMPONENT_VIEWS = frozenset({"full", "tile"})
_VALID_PIN_MODES = frozenset({"roi", "side", "all", "L", "R", "T", "B"})


class TorchDependencyError(RuntimeError):
    """Raised only when a torch-backed dataset is used without PyTorch."""


class ComponentDetectionDataset:
    """Lazy Component detection reader for ``component_coco.json``.

    The class deliberately does not inherit from ``torch.utils.data.Dataset``:
    PyTorch DataLoader only requires ``__len__`` and ``__getitem__``, while a
    plain Python base keeps the core package importable in environments where
    torch is not installed.  Ignored/partial tile annotations are excluded by
    default and their count remains available in each target.
    """

    def __init__(
        self,
        component_index: str | Path,
        source_root: str | Path,
        *,
        splits: str | Iterable[str] | None = None,
        view_kinds: str | Iterable[str] | None = None,
        filter_ignore: bool = True,
        include_empty: bool = True,
        expected_binding_sha256: str | None = None,
        verify_commit: bool = True,
    ) -> None:
        _require_torch()
        self.index_path = Path(component_index).expanduser().resolve(strict=True)
        self.source_root = Path(source_root).expanduser().resolve(strict=True)
        self.filter_ignore = bool(filter_ignore)
        selected_splits = _normalize_selection(splits, _VALID_SPLITS, "splits")
        selected_views = _normalize_selection(
            view_kinds, _VALID_COMPONENT_VIEWS, "view_kinds"
        )
        payload = _read_json_object(self.index_path)
        info = payload.get("info")
        if not isinstance(info, dict):
            raise ValueError("component index lacks info")
        if info.get("version") != _MATERIALIZER_VERSION:
            raise ValueError("component index materializer version is unsupported")
        binding = info.get("binding_sha256")
        _validate_binding(binding, expected_binding_sha256)
        self.binding_sha256 = str(binding)
        if verify_commit:
            _verify_commit(self.index_path, "component_coco.json", self.binding_sha256)

        raw_images = payload.get("images")
        raw_annotations = payload.get("annotations")
        raw_categories = payload.get("categories")
        if not isinstance(raw_images, list) or not isinstance(raw_annotations, list):
            raise ValueError("component index images/annotations must be arrays")
        if not isinstance(raw_categories, list) or not raw_categories:
            raise ValueError("component index categories must be a non-empty array")
        category_ids: set[int] = set()
        for category in raw_categories:
            if not isinstance(category, dict) or type(category.get("id")) is not int:
                raise ValueError("component category lacks integer id")
            if category["id"] in category_ids:
                raise ValueError(f"duplicate component category id {category['id']}")
            category_ids.add(category["id"])
        self.categories = normalize_category_table(raw_categories)
        canonical_by_official = {
            int(category["official_id"]): category for category in self.categories
        }
        category_id_to_model_label: dict[int, int] = {}
        for category in raw_categories:
            raw_category_id = int(category["id"])
            raw_official_id = category.get(
                "official_id",
                category.get("source_category_id", raw_category_id),
            )
            try:
                official_id = int(raw_official_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"component category {raw_category_id} has invalid official id"
                ) from exc
            canonical = canonical_by_official.get(official_id)
            if canonical is None:
                raise ValueError(
                    f"component category {raw_category_id} lacks a canonical mapping"
                )
            category_id_to_model_label[raw_category_id] = int(
                canonical["model_label"]
            )
        self.official_to_model_label = {
            int(item["official_id"]): int(item["model_label"])
            for item in self.categories
        }
        self.model_label_to_official = {
            int(item["model_label"]): int(item["official_id"])
            for item in self.categories
        }
        self._category_id_to_model_label = category_id_to_model_label

        image_by_id: dict[int, dict[str, Any]] = {}
        for image in raw_images:
            _validate_component_image(image, self.binding_sha256)
            image_id = int(image["id"])
            if image_id in image_by_id:
                raise ValueError(f"duplicate component image id {image_id}")
            image_by_id[image_id] = image
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        annotation_ids: set[int] = set()
        for annotation in raw_annotations:
            if not isinstance(annotation, dict):
                raise ValueError("component annotation must be an object")
            annotation_id = annotation.get("id")
            image_id = annotation.get("image_id")
            category_id = annotation.get("category_id")
            if type(annotation_id) is not int or annotation_id in annotation_ids:
                raise ValueError("component annotations contain an invalid/duplicate id")
            annotation_ids.add(annotation_id)
            if type(image_id) is not int or image_id not in image_by_id:
                raise ValueError(f"annotation {annotation_id} references unknown image")
            if type(category_id) is not int or category_id not in category_ids:
                raise ValueError(f"annotation {annotation_id} references unknown category")
            _validate_coco_bbox(annotation.get("bbox"), f"annotation {annotation_id}")
            annotations_by_image[image_id].append(annotation)
        for rows in annotations_by_image.values():
            rows.sort(key=lambda item: int(item["id"]))

        selected: list[dict[str, Any]] = []
        for image_id in sorted(image_by_id):
            image = image_by_id[image_id]
            if selected_splits is not None and image.get("split") not in selected_splits:
                continue
            if selected_views is not None and image.get("view_kind") not in selected_views:
                continue
            usable = [
                annotation
                for annotation in annotations_by_image.get(image_id, [])
                if not self.filter_ignore or not _annotation_is_ignored(annotation)
            ]
            if include_empty or usable:
                selected.append(image)
        self._images = tuple(selected)
        self._annotations = annotations_by_image

    def __len__(self) -> int:
        return len(self._images)

    def class_balanced_sampling_weights(
        self,
        *,
        power: float = 0.5,
        maximum_weight: float = 8.0,
    ) -> tuple[float, ...]:
        """Return deterministic train-image weights for long-tailed labels.

        Each image receives the largest inverse-frequency weight among its
        usable labels.  Common-class and empty images retain weight 1, while a
        finite cap prevents a handful of tail annotations from taking over an
        epoch.  The method reads only the frozen index; it does not decode
        images or alter validation sampling.
        """

        if (
            isinstance(power, bool)
            or not isinstance(power, (int, float))
            or not math.isfinite(float(power))
            or not 0.0 <= float(power) <= 1.0
        ):
            raise ValueError("power must be finite and in [0, 1]")
        if (
            isinstance(maximum_weight, bool)
            or not isinstance(maximum_weight, (int, float))
            or not math.isfinite(float(maximum_weight))
            or float(maximum_weight) < 1.0
        ):
            raise ValueError("maximum_weight must be finite and >= 1")
        usable_labels_by_image: dict[int, tuple[int, ...]] = {}
        label_counts: dict[int, int] = defaultdict(int)
        for image in self._images:
            image_id = int(image["id"])
            labels = tuple(
                self._category_id_to_model_label[int(annotation["category_id"])]
                for annotation in self._annotations.get(image_id, [])
                if not self.filter_ignore or not _annotation_is_ignored(annotation)
            )
            usable_labels_by_image[image_id] = labels
            for label in labels:
                label_counts[label] += 1
        maximum_count = max(label_counts.values(), default=1)
        selected_power = float(power)
        selected_cap = float(maximum_weight)
        output: list[float] = []
        for image in self._images:
            labels = usable_labels_by_image[int(image["id"])]
            weight = max(
                (
                    (maximum_count / label_counts[label]) ** selected_power
                    for label in set(labels)
                ),
                default=1.0,
            )
            output.append(min(selected_cap, max(1.0, float(weight))))
        return tuple(output)

    def full_view_mixture_sampling_weights(
        self,
        *,
        full_fraction: float,
        base_weights: Sequence[float] | None = None,
    ) -> tuple[float, ...]:
        """Reweight train views to one declared expected full-view fraction.

        The operation preserves all relative weights within the full and tile
        groups, so it composes with class-balanced sampling.  It reads only the
        materialized train index and is intended for a replacement sampler;
        validation remains sequential and unweighted.
        """

        if (
            isinstance(full_fraction, bool)
            or not isinstance(full_fraction, (int, float))
            or not math.isfinite(float(full_fraction))
            or not 0.0 < float(full_fraction) < 1.0
        ):
            raise ValueError("full_fraction must be finite and in (0, 1)")
        if base_weights is None:
            weights = [1.0] * len(self._images)
        else:
            if len(base_weights) != len(self._images):
                raise ValueError("base_weights length differs from Component views")
            weights = []
            for value in base_weights:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                ):
                    raise ValueError("base_weights must be finite positive numbers")
                weights.append(float(value))
        full_total = sum(
            weight
            for image, weight in zip(self._images, weights)
            if image.get("view_kind") == "full"
        )
        tile_total = sum(
            weight
            for image, weight in zip(self._images, weights)
            if image.get("view_kind") == "tile"
        )
        if full_total <= 0.0 or tile_total <= 0.0:
            raise ValueError("Component view mixture requires both full and tile views")
        selected_fraction = float(full_fraction)
        full_multiplier = (
            selected_fraction * tile_total
            / ((1.0 - selected_fraction) * full_total)
        )
        return tuple(
            weight * full_multiplier
            if image.get("view_kind") == "full"
            else weight
            for image, weight in zip(self._images, weights)
        )

    def sampling_view_kind_summary(
        self, weights: Sequence[float]
    ) -> dict[str, dict[str, float | int]]:
        if len(weights) != len(self._images):
            raise ValueError("weights length differs from Component views")
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for image, raw_weight in zip(self._images, weights):
            value = float(raw_weight)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("weights must be finite and positive")
            kind = str(image.get("view_kind"))
            totals[kind] += value
            counts[kind] += 1
        grand_total = sum(totals.values())
        return {
            kind: {
                "view_count": counts[kind],
                "weight_sum": totals[kind],
                "expected_fraction": totals[kind] / grand_total,
            }
            for kind in sorted(totals)
        }

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        image_record = self._images[_normalize_index(index, len(self))]
        torch = _require_torch()
        decoded = decode_indexed_view(image_record, self.source_root)
        try:
            image_tensor = _pil_to_tensor(decoded, torch)
        finally:
            decoded.close()
        all_annotations = self._annotations.get(int(image_record["id"]), [])
        ignored_count = sum(_annotation_is_ignored(item) for item in all_annotations)
        annotations = [
            item
            for item in all_annotations
            if not self.filter_ignore or not _annotation_is_ignored(item)
        ]
        boxes: list[list[float]] = []
        labels: list[int] = []
        official_category_ids: list[int] = []
        areas: list[float] = []
        crowds: list[int] = []
        annotation_ids: list[int] = []
        component_ids: list[str] = []
        component_keys: list[str] = []
        for annotation in annotations:
            _validate_component_transform(annotation, image_record)
            x, y, width, height = (float(value) for value in annotation["bbox"])
            boxes.append([x, y, x + width, y + height])
            model_label = self._category_id_to_model_label[int(annotation["category_id"])]
            labels.append(model_label)
            official_category_ids.append(self.model_label_to_official[model_label])
            areas.append(float(annotation.get("area", width * height)))
            crowds.append(int(annotation.get("iscrowd", 0)))
            annotation_ids.append(int(annotation["id"]))
            component_ids.append(str(annotation.get("component_id", "")))
            component_keys.append(str(annotation.get("component_key", "")))
        box_tensor = torch.tensor(boxes, dtype=torch.float32)
        if box_tensor.numel() == 0:
            box_tensor = torch.empty((0, 4), dtype=torch.float32)
        target = {
            "boxes": box_tensor,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "official_category_ids": torch.tensor(
                official_category_ids, dtype=torch.int64
            ),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(crowds, dtype=torch.int64),
            "annotation_ids": torch.tensor(annotation_ids, dtype=torch.int64),
            "image_id": torch.tensor([int(image_record["id"])], dtype=torch.int64),
            "component_ids": tuple(component_ids),
            "component_keys": tuple(component_keys),
            "ignored_annotation_count": ignored_count,
            "view_id": str(image_record["view_id"]),
            "view_kind": str(image_record["view_kind"]),
            "sample_id": str(image_record["sample_id"]),
            "split": str(image_record["split"]),
            "binding_sha256": self.binding_sha256,
            "output_size": tuple(int(value) for value in image_record["output_size"]),
        }
        return image_tensor, target


@dataclass(frozen=True, slots=True)
class _PinOffset:
    offset: int
    length: int
    view_index: int
    view_id: str


class PinKeypointDataset:
    """Streaming Pin view reader for ``pin_index.jsonl``.

    Initialization stores only byte offsets and selected view indices.  Each
    item re-opens and seeks the JSONL file, making the dataset safe under both
    forked and spawned DataLoader workers without retaining the 100+ MiB index
    in Python objects.
    """

    def __init__(
        self,
        pin_index: str | Path,
        source_root: str | Path,
        *,
        view_mode: str = "roi",
        splits: str | Iterable[str] | None = None,
        sample_ids: Iterable[str] | None = None,
        include_negative: bool = True,
        heatmap_stride: int | None = None,
        expected_binding_sha256: str | None = None,
        verify_commit: bool = True,
    ) -> None:
        _require_torch()
        self.index_path = Path(pin_index).expanduser().resolve(strict=True)
        self.source_root = Path(source_root).expanduser().resolve(strict=True)
        normalized_mode = _normalize_pin_mode(view_mode)
        selected_splits = _normalize_selection(splits, _VALID_SPLITS, "splits")
        selected_samples = _normalize_sample_ids(sample_ids)
        summary: dict[str, Any] | None = None
        if verify_commit:
            summary = _read_json_object(self.index_path.parent / "summary.json")
            summary_binding = summary.get("binding_sha256")
            _validate_binding(summary_binding, expected_binding_sha256)
            _verify_commit(
                self.index_path,
                "pin_index.jsonl",
                str(summary_binding),
                summary=summary,
            )
            expected_binding_sha256 = str(summary_binding)
        if heatmap_stride is not None and (
            type(heatmap_stride) is not int or heatmap_stride <= 0
        ):
            raise ValueError("heatmap_stride must be a positive integer")
        summary_stride: Any = None
        if summary is not None and isinstance(summary.get("config"), dict):
            summary_stride = summary["config"].get("heatmap_stride")
        if summary_stride is not None and (
            type(summary_stride) is not int or summary_stride <= 0
        ):
            raise ValueError("materialization summary heatmap_stride is invalid")
        if heatmap_stride is not None and summary_stride is not None and heatmap_stride != summary_stride:
            raise ValueError("requested heatmap_stride differs from materialization summary")
        self.heatmap_stride = int(
            heatmap_stride if heatmap_stride is not None else summary_stride or 4
        )

        entries: list[_PinOffset] = []
        record_ids: set[str] = set()
        binding: str | None = None
        with self.index_path.open("rb") as handle:
            line_number = 0
            while True:
                offset = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                line_number += 1
                if not raw_line.strip():
                    raise ValueError(f"pin index contains blank line {line_number}")
                record = _loads_object(raw_line, f"pin index line {line_number}")
                _validate_pin_record(record, line_number)
                record_binding = str(record["binding_sha256"])
                if binding is None:
                    binding = record_binding
                    _validate_binding(binding, expected_binding_sha256)
                elif record_binding != binding:
                    raise ValueError("pin index contains mixed binding_sha256 values")
                record_id = str(record["record_id"])
                if record_id in record_ids:
                    raise ValueError(f"pin index contains duplicate record_id {record_id!r}")
                record_ids.add(record_id)
                if selected_splits is not None and record["split"] not in selected_splits:
                    continue
                if selected_samples is not None and record["sample_id"] not in selected_samples:
                    continue
                if not include_negative and bool(record.get("negative_component")):
                    continue
                for view_index in _selected_pin_view_indices(record["views"], normalized_mode):
                    view = record["views"][view_index]
                    entries.append(
                        _PinOffset(
                            offset=offset,
                            length=len(raw_line),
                            view_index=view_index,
                            view_id=str(view["view_id"]),
                        )
                    )
        if binding is None:
            raise ValueError("pin index is empty")
        self.binding_sha256 = binding
        self._entries = tuple(entries)
        stat = self.index_path.stat()
        self._indexed_size = stat.st_size
        self._indexed_mtime_ns = stat.st_mtime_ns
        self.view_mode = normalized_mode

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        entry = self._entries[_normalize_index(index, len(self))]
        stat = self.index_path.stat()
        if stat.st_size != self._indexed_size or stat.st_mtime_ns != self._indexed_mtime_ns:
            raise RuntimeError(
                "pin_index.jsonl changed after its byte-offset index was built; "
                "recreate PinKeypointDataset"
            )
        with self.index_path.open("rb") as handle:
            handle.seek(entry.offset)
            raw_line = handle.read(entry.length)
        record = _loads_object(raw_line, f"pin record at byte {entry.offset}")
        if record.get("binding_sha256") != self.binding_sha256:
            raise RuntimeError("pin record binding changed after indexing")
        views = record.get("views")
        if not isinstance(views, list) or entry.view_index >= len(views):
            raise RuntimeError("pin view layout changed after indexing")
        view = views[entry.view_index]
        if view.get("view_id") != entry.view_id:
            raise RuntimeError("pin view identity changed after indexing")
        torch = _require_torch()
        decoded = decode_indexed_view(view, self.source_root)
        try:
            image_tensor = _pil_to_tensor(decoded, torch)
        finally:
            decoded.close()
        target = _build_pin_tensors(
            record, view, torch, default_heatmap_stride=self.heatmap_stride
        )
        return image_tensor, target


def component_detection_collate(
    batch: Sequence[tuple[Any, dict[str, Any]]],
) -> tuple[Any, list[dict[str, Any]]]:
    """Stack fixed-size Component images and retain variable detection targets."""

    torch = _require_torch()
    if not batch:
        raise ValueError("cannot collate an empty Component batch")
    images, targets = zip(*batch)
    _require_same_image_shape(images, "Component")
    return torch.stack(list(images), dim=0), list(targets)


def pin_keypoint_collate(
    batch: Sequence[tuple[Any, dict[str, Any]]],
) -> tuple[Any, dict[str, Any]]:
    """Stack Pin images/dense targets while keeping variable instances as lists."""

    torch = _require_torch()
    if not batch:
        raise ValueError("cannot collate an empty Pin batch")
    images, targets = zip(*batch)
    _require_same_image_shape(images, "Pin")
    dense_keys = ("heatmap", "offset", "orientation", "regression_mask")
    for key in dense_keys:
        shapes = {tuple(target[key].shape) for target in targets}
        if len(shapes) != 1:
            raise ValueError(f"Pin target {key} shapes differ within the batch: {shapes}")
    metadata_keys = (
        "record_id",
        "view_id",
        "view_kind",
        "side",
        "sample_id",
        "split",
        "component_id",
        "component_key",
        "binding_sha256",
        "collision_count",
        "box_rect_view",
        "view_output_size",
    )
    output: dict[str, Any] = {
        key: torch.stack([target[key] for target in targets], dim=0)
        for key in dense_keys
    }
    output.update(
        {
            "visibility": [target["visibility"] for target in targets],
            "tip_points": [target["tip_points"] for target in targets],
            "tip_points_heatmap": [
                target["tip_points_heatmap"] for target in targets
            ],
            "orientation_instances": [
                target["orientation_instances"] for target in targets
            ],
            "pin_ids": [target["pin_ids"] for target in targets],
            "metadata": [
                {key: target[key] for key in metadata_keys} for target in targets
            ],
        }
    )
    return torch.stack(list(images), dim=0), output


def _build_pin_tensors(
    record: dict[str, Any],
    view: dict[str, Any],
    torch: Any,
    *,
    default_heatmap_stride: int,
) -> dict[str, Any]:
    _validate_view_transform(view)
    raw_targets = view.get("targets")
    if not isinstance(raw_targets, list):
        raise ValueError(f"Pin view {view.get('view_id')!r} targets must be an array")
    if raw_targets:
        descriptor = raw_targets[0].get("heatmap")
        if not isinstance(descriptor, dict):
            raise ValueError("Pin target lacks heatmap descriptor")
        output_size = _positive_pair(descriptor.get("output_size"), "heatmap.output_size")
    else:
        output_width, output_height = _positive_pair(
            view.get("output_size"), "view.output_size"
        )
        # Frozen materializer default; negative records contain no per-Pin
        # descriptor, so derive the same stride from the schema contract.
        stride = default_heatmap_stride
        output_size = (
            int(math.ceil(output_width / stride)),
            int(math.ceil(output_height / stride)),
        )
    heatmap_width, heatmap_height = output_size
    heatmap = np.zeros((1, heatmap_height, heatmap_width), dtype=np.float32)
    offset = np.zeros((2, heatmap_height, heatmap_width), dtype=np.float32)
    orientation = np.zeros((2, heatmap_height, heatmap_width), dtype=np.float32)
    regression_mask = np.zeros((1, heatmap_height, heatmap_width), dtype=np.float32)
    visibility_values: list[int] = []
    tip_points: list[list[float]] = []
    tip_points_heatmap: list[list[float]] = []
    orientation_instances: list[list[float]] = []
    pin_ids: list[str] = []
    collisions = 0
    for target_number, raw_target in enumerate(raw_targets):
        if not isinstance(raw_target, dict):
            raise ValueError("Pin view target must be an object")
        _validate_pin_target_transform(raw_target, view)
        visibility = raw_target.get("visibility")
        if visibility not in (0, 2):
            raise ValueError("Pin visibility must use COCO values 0 or 2")
        visibility_values.append(int(visibility))
        pin_ids.append(str(raw_target.get("pin_id", "")))
        orientation_value = _orientation_unit(raw_target)
        orientation_instances.append(list(orientation_value))
        descriptor = raw_target.get("heatmap")
        if not isinstance(descriptor, dict):
            raise ValueError("Pin target lacks heatmap descriptor")
        descriptor_stride = descriptor.get("output_stride")
        if descriptor_stride != default_heatmap_stride:
            raise ValueError(
                "Pin target output_stride differs from the dataset heatmap contract"
            )
        if _positive_pair(descriptor.get("output_size"), "heatmap.output_size") != output_size:
            raise ValueError("Pin targets in one view disagree on heatmap output_size")
        point = descriptor.get("point")
        if visibility == 0:
            if point is not None or raw_target.get("tip_view") is not None:
                raise ValueError("invisible Pin target must not expose a view/heatmap point")
            tip_points.append([-1.0, -1.0])
            tip_points_heatmap.append([-1.0, -1.0])
            continue
        center = _finite_pair(point, "heatmap.point")
        if not (0 <= center[0] < heatmap_width and 0 <= center[1] < heatmap_height):
            raise ValueError("visible Pin heatmap point lies outside the heatmap")
        tip_view = _finite_pair(raw_target.get("tip_view"), "tip_view")
        tip_points.append([tip_view[0], tip_view[1]])
        tip_points_heatmap.append([center[0], center[1]])
        sigma = _positive_float(descriptor.get("sigma_heatmap_pixels"), "heatmap sigma")
        truncate = _positive_float(descriptor.get("truncate_sigma"), "heatmap truncate")
        _draw_gaussian(heatmap[0], center, sigma, truncate)
        anchor_x = min(heatmap_width - 1, max(0, int(math.floor(center[0]))))
        anchor_y = min(heatmap_height - 1, max(0, int(math.floor(center[1]))))
        if regression_mask[0, anchor_y, anchor_x] != 0:
            collisions += 1
            continue
        regression_mask[0, anchor_y, anchor_x] = 1.0
        offset[:, anchor_y, anchor_x] = (
            center[0] - anchor_x,
            center[1] - anchor_y,
        )
        orientation[:, anchor_y, anchor_x] = orientation_value

    component_bbox = record.get("component_bbox")
    transform_forward = view.get("transform", {}).get("forward")
    if (
        isinstance(component_bbox, (list, tuple))
        and len(component_bbox) == 4
        and isinstance(transform_forward, (list, tuple))
        and len(transform_forward) == 3
    ):
        x1v, y1v = transform_point(
            [component_bbox[0], component_bbox[1]], transform_forward
        )
        x2v, y2v = transform_point(
            [component_bbox[2], component_bbox[3]], transform_forward
        )
        box_rect_view: tuple[float, float, float, float] | None = (
            float(x1v),
            float(y1v),
            float(x2v),
            float(y2v),
        )
    else:
        box_rect_view = None
    raw_output_size = view.get("output_size")
    if isinstance(raw_output_size, (list, tuple)) and len(raw_output_size) == 2:
        view_output_size: tuple[int, int] | None = (
            int(raw_output_size[0]),
            int(raw_output_size[1]),
        )
    else:
        view_output_size = None

    return {
        "heatmap": torch.from_numpy(heatmap),
        "offset": torch.from_numpy(offset),
        "orientation": torch.from_numpy(orientation),
        "regression_mask": torch.from_numpy(regression_mask),
        "visibility": torch.tensor(visibility_values, dtype=torch.int64),
        "tip_points": torch.tensor(tip_points, dtype=torch.float32).reshape(-1, 2),
        "tip_points_heatmap": torch.tensor(
            tip_points_heatmap, dtype=torch.float32
        ).reshape(-1, 2),
        "orientation_instances": torch.tensor(
            orientation_instances, dtype=torch.float32
        ).reshape(-1, 2),
        "pin_ids": tuple(pin_ids),
        "record_id": str(record["record_id"]),
        "view_id": str(view["view_id"]),
        "view_kind": str(view["view_kind"]),
        "side": view.get("side"),
        "sample_id": str(record["sample_id"]),
        "split": str(record["split"]),
        "component_id": str(record["component_id"]),
        "component_key": str(record["component_key"]),
        "binding_sha256": str(record["binding_sha256"]),
        "collision_count": collisions,
        "box_rect_view": box_rect_view,
        "view_output_size": view_output_size,
    }


def _draw_gaussian(
    canvas: np.ndarray,
    center: tuple[float, float],
    sigma: float,
    truncate_sigma: float,
) -> None:
    radius = max(1, int(math.ceil(sigma * truncate_sigma)))
    left = max(0, int(math.floor(center[0] - radius)))
    right = min(canvas.shape[1] - 1, int(math.ceil(center[0] + radius)))
    top = max(0, int(math.floor(center[1] - radius)))
    bottom = min(canvas.shape[0] - 1, int(math.ceil(center[1] + radius)))
    xs = np.arange(left, right + 1, dtype=np.float32)
    ys = np.arange(top, bottom + 1, dtype=np.float32)
    squared = (xs[None, :] - center[0]) ** 2 + (ys[:, None] - center[1]) ** 2
    values = np.exp(-squared / (2.0 * sigma * sigma)).astype(np.float32)
    region = canvas[top : bottom + 1, left : right + 1]
    np.maximum(region, values, out=region)


def _validate_component_image(image: Any, binding_sha256: str) -> None:
    if not isinstance(image, dict):
        raise ValueError("component image entry must be an object")
    if type(image.get("id")) is not int or image["id"] <= 0:
        raise ValueError("component image id must be a positive integer")
    if image.get("binding_sha256") != binding_sha256:
        raise ValueError("component image binding differs from index binding")
    if image.get("view_kind") not in _VALID_COMPONENT_VIEWS:
        raise ValueError("component image has unsupported view_kind")
    if image.get("split") not in _VALID_SPLITS:
        raise ValueError("component image has invalid split")
    _validate_view_transform(image)


def _validate_component_transform(
    annotation: dict[str, Any], image: dict[str, Any]
) -> None:
    source_bbox = annotation.get("tile_intersection_bbox")
    if not (
        isinstance(source_bbox, list)
        and len(source_bbox) == 4
        and all(_finite_number(value) for value in source_bbox)
    ):
        raise ValueError(f"annotation {annotation.get('id')} lacks source intersection bbox")
    matrix = image["transform"]["forward"]
    upper_left = transform_point(source_bbox[:2], matrix)
    lower_right = transform_point(source_bbox[2:], matrix)
    expected = (
        upper_left[0],
        upper_left[1],
        lower_right[0] - upper_left[0],
        lower_right[1] - upper_left[1],
    )
    tolerance = max(
        float(image["transform"].get("roundtrip_tolerance_pixels", 0.0)), 1e-6
    )
    if any(abs(float(actual) - expected_value) > tolerance for actual, expected_value in zip(annotation["bbox"], expected)):
        raise ValueError(
            f"annotation {annotation.get('id')} bbox disagrees with its recorded transform"
        )


def _validate_pin_record(record: dict[str, Any], line_number: int) -> None:
    if record.get("materializer_version") != _MATERIALIZER_VERSION:
        raise ValueError(f"pin index line {line_number} has unsupported materializer version")
    _validate_binding(record.get("binding_sha256"), None)
    for field in ("record_id", "sample_id", "component_id", "component_key"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise ValueError(f"pin index line {line_number} lacks {field}")
    if record.get("split") not in _VALID_SPLITS:
        raise ValueError(f"pin index line {line_number} has invalid split")
    views = record.get("views")
    if not isinstance(views, list) or len(views) != 5:
        raise ValueError(f"pin index line {line_number} must contain five views")
    roi = [view for view in views if isinstance(view, dict) and view.get("view_kind") == "roi"]
    sides = {
        view.get("side")
        for view in views
        if isinstance(view, dict) and view.get("view_kind") == "side_strip"
    }
    if len(roi) != 1 or sides != {"L", "R", "T", "B"}:
        raise ValueError(f"pin index line {line_number} must contain ROI and L/R/T/B")
    view_ids: set[str] = set()
    for view in views:
        if not isinstance(view, dict) or not isinstance(view.get("view_id"), str):
            raise ValueError(f"pin index line {line_number} contains an invalid view")
        if view["view_id"] in view_ids:
            raise ValueError(f"pin index line {line_number} contains duplicate view_id")
        view_ids.add(view["view_id"])
        _validate_view_transform(view)


def _validate_pin_target_transform(target: dict[str, Any], view: dict[str, Any]) -> None:
    tip_original = _finite_pair(target.get("tip_original"), "tip_original")
    base_original = _finite_pair(target.get("base_original"), "base_original")
    matrix = view["transform"]["forward"]
    expected_tip = transform_point(tip_original, matrix)
    expected_base = transform_point(base_original, matrix)
    tolerance = max(
        float(view["transform"].get("roundtrip_tolerance_pixels", 0.0)), 1e-6
    )
    base_view = _finite_pair(target.get("base_view"), "base_view")
    if _distance(expected_base, base_view) > tolerance:
        raise ValueError(f"Pin {target.get('pin_id')!r} base_view violates transform")
    if target.get("visibility") == 2:
        tip_view = _finite_pair(target.get("tip_view"), "tip_view")
        if _distance(expected_tip, tip_view) > tolerance:
            raise ValueError(f"Pin {target.get('pin_id')!r} tip_view violates transform")
        descriptor = target.get("heatmap")
        if not isinstance(descriptor, dict):
            raise ValueError("visible Pin lacks heatmap descriptor")
        stride = descriptor.get("output_stride")
        if type(stride) is not int or stride <= 0:
            raise ValueError("Pin heatmap output_stride must be positive")
        point = _finite_pair(descriptor.get("point"), "heatmap.point")
        expected_point = (tip_view[0] / stride, tip_view[1] / stride)
        if _distance(point, expected_point) > tolerance:
            raise ValueError(f"Pin {target.get('pin_id')!r} heatmap point violates transform")


def _validate_view_transform(view: Mapping[str, Any]) -> None:
    transform = view.get("transform")
    crop = view.get("crop_bbox")
    if not isinstance(transform, dict):
        raise ValueError(f"view {view.get('view_id')!r} lacks transform")
    forward = _matrix(transform.get("forward"), "forward transform")
    inverse = _matrix(transform.get("inverse"), "inverse transform")
    if not isinstance(crop, dict):
        raise ValueError(f"view {view.get('view_id')!r} lacks crop_bbox")
    corners = (
        (crop.get("x1"), crop.get("y1")),
        (crop.get("x2"), crop.get("y2")),
    )
    tolerance = max(float(transform.get("roundtrip_tolerance_pixels", 0.0)), 1e-6)
    for corner in corners:
        point = _finite_pair(corner, "crop corner")
        restored = transform_point(transform_point(point, forward), inverse)
        if _distance(point, restored) > tolerance:
            raise ValueError(f"view {view.get('view_id')!r} transform is not reversible")
    _positive_pair(view.get("output_size"), "view.output_size")


def _selected_pin_view_indices(views: list[dict[str, Any]], mode: str) -> list[int]:
    if mode == "all":
        return list(range(len(views)))
    if mode == "roi":
        return [index for index, view in enumerate(views) if view.get("view_kind") == "roi"]
    if mode == "side":
        return [
            index for index, view in enumerate(views) if view.get("view_kind") == "side_strip"
        ]
    return [index for index, view in enumerate(views) if view.get("side") == mode]


def _orientation_unit(target: dict[str, Any]) -> tuple[float, float]:
    orientation = target.get("orientation_view")
    if not isinstance(orientation, dict):
        raise ValueError("Pin target lacks orientation_view")
    unit = _finite_pair(orientation.get("unit_vector"), "orientation unit_vector")
    magnitude = math.hypot(*unit)
    if magnitude <= 1e-12:
        return 0.0, 0.0
    if abs(magnitude - 1.0) > 1e-4:
        raise ValueError("non-zero Pin orientation unit_vector is not normalized")
    return unit[0] / magnitude, unit[1] / magnitude


def _verify_commit(
    index_path: Path,
    output_name: str,
    binding_sha256: str,
    *,
    summary: dict[str, Any] | None = None,
) -> None:
    value = summary or _read_json_object(index_path.parent / "summary.json")
    if value.get("materializer_version") != _MATERIALIZER_VERSION:
        raise ValueError("materialization summary version is unsupported")
    if value.get("binding_sha256") != binding_sha256:
        raise ValueError("materialization summary binding differs from index")
    outputs = value.get("outputs")
    metadata = outputs.get(output_name) if isinstance(outputs, dict) else None
    expected_hash = metadata.get("sha256") if isinstance(metadata, dict) else None
    if not _is_sha256(expected_hash):
        raise ValueError(f"materialization summary lacks {output_name} SHA-256")
    actual_hash = _sha256_file(index_path)
    if actual_hash != expected_hash:
        raise ValueError(f"{output_name} SHA-256 differs from materialization summary")


def _validate_binding(actual: Any, expected: str | None) -> None:
    if not _is_sha256(actual):
        raise ValueError("training index binding_sha256 must be a lowercase SHA-256")
    if expected is not None:
        if not _is_sha256(expected):
            raise ValueError("expected_binding_sha256 must be a lowercase SHA-256")
        if actual != expected:
            raise ValueError("training index binding_sha256 differs from expected binding")


def _pil_to_tensor(image: Any, torch: Any) -> Any:
    array = np.asarray(image, dtype=np.float32).copy()
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"decoded image must be RGB HWC, got {array.shape}")
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().div_(255.0)


def _normalize_selection(
    value: str | Iterable[str] | None,
    allowed: frozenset[str],
    label: str,
) -> frozenset[str] | None:
    if value is None:
        return None
    values = [value] if isinstance(value, str) else list(value)
    if not values:
        raise ValueError(f"{label} cannot be empty")
    invalid = sorted(set(values) - allowed)
    if invalid:
        raise ValueError(f"{label} contains unsupported values: {invalid}")
    return frozenset(values)


def _normalize_pin_mode(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("view_mode must be roi, side, all, or L/R/T/B")
    normalized = value.lower() if value.lower() in {"roi", "side", "all"} else value.upper()
    if normalized not in _VALID_PIN_MODES:
        raise ValueError("view_mode must be roi, side, all, or L/R/T/B")
    return normalized


def _normalize_sample_ids(value: Iterable[str] | None) -> frozenset[str] | None:
    if value is None:
        return None
    values = list(value)
    if not values or any(not isinstance(item, str) or not item for item in values):
        raise ValueError("sample_ids must contain non-empty strings")
    return frozenset(values)


def _normalize_index(index: int, length: int) -> int:
    if not isinstance(index, int):
        raise TypeError("dataset index must be an integer")
    normalized = index + length if index < 0 else index
    if normalized < 0 or normalized >= length:
        raise IndexError("dataset index out of range")
    return normalized


def _require_same_image_shape(images: Sequence[Any], label: str) -> None:
    shapes = {tuple(image.shape) for image in images}
    if len(shapes) != 1:
        raise ValueError(f"{label} image shapes differ within the batch: {shapes}")


def _validate_coco_bbox(value: Any, label: str) -> tuple[float, float, float, float]:
    result = _finite_quad(value, f"{label} bbox")
    if result[2] <= 0 or result[3] <= 0:
        raise ValueError(f"{label} bbox width/height must be positive")
    return result


def _annotation_is_ignored(annotation: dict[str, Any]) -> bool:
    return bool(annotation.get("ignore", 0))


def _positive_pair(value: Any, label: str) -> tuple[int, int]:
    if not (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and all(type(item) is int and item > 0 for item in value)
    ):
        raise ValueError(f"{label} must contain two positive integers")
    return int(value[0]), int(value[1])


def _finite_pair(value: Any, label: str) -> tuple[float, float]:
    if not (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and all(_finite_number(item) for item in value)
    ):
        raise ValueError(f"{label} must contain two finite numbers")
    return float(value[0]), float(value[1])


def _finite_quad(value: Any, label: str) -> tuple[float, float, float, float]:
    if not (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 4
        and all(_finite_number(item) for item in value)
    ):
        raise ValueError(f"{label} must contain four finite numbers")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _matrix(value: Any, label: str) -> list[list[float]]:
    if not (
        isinstance(value, list)
        and len(value) == 3
        and all(
            isinstance(row, list)
            and len(row) == 3
            and all(_finite_number(item) for item in row)
            for row in value
        )
    ):
        raise ValueError(f"{label} must be a finite 3x3 matrix")
    return [[float(item) for item in row] for row in value]


def _positive_float(value: Any, label: str) -> float:
    if not _finite_number(value) or float(value) <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


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


def _loads_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _require_torch() -> Any:
    try:
        import torch
    except (ModuleNotFoundError, ImportError, OSError) as exc:
        raise TorchDependencyError(
            "PyTorch is required for Component/Pin training datasets but is not "
            "installed or loadable. Use the workspace .venv-train interpreter or "
            "install a compatible torch build; the core pcbparse package remains "
            "torch-optional."
        ) from exc
    return torch


__all__ = [
    "ComponentDetectionDataset",
    "PinKeypointDataset",
    "TorchDependencyError",
    "component_detection_collate",
    "pin_keypoint_collate",
]
