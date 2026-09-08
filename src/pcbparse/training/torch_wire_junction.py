from __future__ import annotations

"""Lazy PyTorch datasets for frozen Wire/Junction materializations.

This module deliberately does not import :mod:`torch` at module import time.
Index inspection and training-gate checks therefore work in the lightweight
runtime.  Tensor datasets, collate functions, and sampler helpers call
``_require_torch`` and fail with an actionable error when PyTorch is absent.
"""

import hashlib
import importlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from PIL import Image


JUNCTION_CLASSES: tuple[str, ...] = (
    "endpoint",
    "t_junction",
    "bend",
    "connected_cross",
    "non_connected_cross",
    "noise",
)
_ALLOWED_SPLITS = frozenset({"train", "val", "test"})


class TorchUnavailableError(RuntimeError):
    """PyTorch-backed functionality was requested without a PyTorch runtime."""


class FrozenDataError(ValueError):
    """The materialized data is incomplete, corrupt, or violates its contract."""


class DatasetGateError(RuntimeError):
    """A trainer attempted to cross an explicit supervision gate."""


@dataclass(frozen=True, slots=True)
class WireDatasetMetadata:
    run_id: str
    split: str | None
    total_manifest_records: int
    selected_records: int
    eligible_records: int
    excluded_ineligible_records: int
    excluded_other_splits: int
    eligible_by_split: dict[str, int]
    source_materialization_ok: bool

    @property
    def wire_training_ready(self) -> bool:
        return self.source_materialization_ok and self.eligible_records > 0


@dataclass(frozen=True, slots=True)
class JunctionDatasetMetadata:
    run_id: str
    split: str | None
    total_manifest_records: int
    selected_records: int
    trusted_records: int
    excluded_unknown_records: int
    excluded_other_splits: int
    class_counts: dict[str, int]
    missing_classes: tuple[str, ...]
    source_training_gate_ok: bool
    source_gate_blockers: tuple[dict[str, Any], ...]

    @property
    def six_class_training_ready(self) -> bool:
        return (
            self.source_training_gate_ok
            and not self.missing_classes
            and self.excluded_unknown_records == 0
        )

    def require_training_ready(self) -> None:
        if self.six_class_training_ready:
            return
        details: list[str] = []
        if self.missing_classes:
            details.append("missing classes: " + ", ".join(self.missing_classes))
        if self.excluded_unknown_records:
            details.append(
                f"pending UNKNOWN records: {self.excluded_unknown_records}"
            )
        if not self.source_training_gate_ok:
            codes = sorted(
                {
                    str(item.get("code", "UNKNOWN_GATE_BLOCKER"))
                    for item in self.source_gate_blockers
                }
            )
            details.append("source gate blockers: " + ", ".join(codes))
        raise DatasetGateError(
            "junction six-class training is blocked; " + "; ".join(details)
        )


@dataclass(slots=True)
class _Freeze:
    run_dir: Path
    summary: dict[str, Any]
    wire_rows: list[dict[str, Any]]
    junction_rows: list[dict[str, Any]]


def inspect_wire_dataset(
    run_dir: str | Path, *, split: str | None = "train"
) -> WireDatasetMetadata:
    split = _validated_split(split)
    freeze = _load_freeze(run_dir, need_wire=True, need_junction=False)
    return _wire_metadata(freeze, split)


def inspect_junction_dataset(
    run_dir: str | Path, *, split: str | None = "train"
) -> JunctionDatasetMetadata:
    split = _validated_split(split)
    freeze = _load_freeze(run_dir, need_wire=False, need_junction=True)
    return _junction_metadata(freeze, split)


class WireTorchDataset:
    """Read lazy source-image tiles plus frozen target/ignore masks.

    Records with ``supervision.eligible=false`` are always removed.  There is
    intentionally no override: admitting those rows would turn an unknown wire
    path into negative supervision.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        source_root: str | Path,
        split: str | None = "train",
    ) -> None:
        self._torch = _require_torch()
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.source_root = Path(source_root).expanduser().resolve()
        if not self.source_root.is_dir():
            raise FrozenDataError(f"source_root is not a directory: {self.source_root}")
        self.split = _validated_split(split)
        self._freeze = _load_freeze(
            self.run_dir, need_wire=True, need_junction=False
        )
        self.metadata = _wire_metadata(self._freeze, self.split)
        self.records = [
            row
            for row in self._freeze.wire_rows
            if _selected_split(row, self.split)
            and row["supervision"]["eligible"] is True
        ]
        self._verified: set[tuple[Path, str]] = set()
        semantics = self._freeze.summary.get("wire", {}).get("mask_semantics", {})
        foreground = semantics.get("foreground")
        background = semantics.get("background")
        if type(foreground) is not int or not 0 <= foreground <= 255:
            raise FrozenDataError("summary wire foreground is invalid")
        if type(background) is not int or not 0 <= background <= 255:
            raise FrozenDataError("summary wire background is invalid")
        if foreground == background:
            raise FrozenDataError("summary wire foreground equals background")
        self._foreground = foreground
        self._background = background

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        source = row["source"]
        image_path = _resolve_within(self.source_root, source.get("image"), "source image")
        self._verify_once(image_path, source.get("image_sha256"), "source image")
        tile_bbox = _bbox(row.get("tile_bbox"), row["record_id"])
        expected_size = _size(row.get("tile_size"), row["record_id"])
        if (tile_bbox[2] - tile_bbox[0], tile_bbox[3] - tile_bbox[1]) != expected_size:
            raise FrozenDataError(f"wire row {row['record_id']!r} tile bbox/size mismatch")
        with Image.open(image_path) as opened:
            opened.load()
            if tile_bbox[0] < 0 or tile_bbox[1] < 0 or tile_bbox[2] > opened.width or tile_bbox[3] > opened.height:
                raise FrozenDataError(
                    f"wire row {row['record_id']!r} tile bbox is outside source image"
                )
            image = opened.convert("RGB").crop(tile_bbox)

        target_info = row["target_mask"]
        ignore_info = row["ignore_mask"]
        target_path = _resolve_within(
            self.run_dir, target_info.get("path"), "wire target mask"
        )
        ignore_path = _resolve_within(
            self.run_dir, ignore_info.get("path"), "wire ignore mask"
        )
        self._verify_once(target_path, target_info.get("sha256"), "wire target mask")
        self._verify_once(ignore_path, ignore_info.get("sha256"), "wire ignore mask")
        target = _load_mask(target_path, expected_size)
        ignore = _load_mask(ignore_path, expected_size)
        image_tensor = _rgb_tensor(image, self._torch)
        target_tensor = _binary_mask_tensor(
            target, self._foreground, self._background, self._torch
        )
        ignore_tensor = _binary_mask_tensor(
            ignore, self._foreground, self._background, self._torch
        )
        for name, info, tensor in (
            ("target", target_info, target_tensor),
            ("ignore", ignore_info, ignore_tensor),
        ):
            declared_pixels = info.get("foreground_pixels")
            actual_pixels = int(tensor.sum().item())
            if type(declared_pixels) is not int or declared_pixels != actual_pixels:
                raise FrozenDataError(
                    f"wire row {row['record_id']!r} {name} foreground count "
                    f"mismatch: declared {declared_pixels!r}, actual {actual_pixels}"
                )
        if bool(self._torch.any((target_tensor > 0.5) & (ignore_tensor > 0.5))):
            raise FrozenDataError(
                f"wire row {row['record_id']!r} target overlaps ignore supervision"
            )
        return {
            "image": image_tensor,
            "target": target_tensor,
            "ignore": ignore_tensor,
            "valid": 1.0 - ignore_tensor,
            "record_id": row["record_id"],
            "sample_id": row["sample_id"],
            "split": row["split"],
            "tile_bbox": row["tile_bbox"],
            "transform": row["transform"],
            "source": source,
        }

    def _verify_once(self, path: Path, expected: Any, kind: str) -> None:
        if not _is_sha256(expected):
            raise FrozenDataError(f"{kind} lacks a valid SHA-256: {path}")
        key = (path, expected)
        if key in self._verified:
            return
        actual = _sha256_file(path)
        if actual != expected:
            raise FrozenDataError(
                f"{kind} SHA-256 mismatch for {path}: expected {expected}, actual {actual}"
            )
        self._verified.add(key)


class JunctionTorchDataset:
    """Read trusted six-class patches; UNKNOWN is unconditionally excluded."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        split: str | None = "train",
        require_training_ready: bool = False,
    ) -> None:
        self._torch = _require_torch()
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.split = _validated_split(split)
        self._freeze = _load_freeze(
            self.run_dir, need_wire=False, need_junction=True
        )
        self.metadata = _junction_metadata(self._freeze, self.split)
        if require_training_ready:
            self.metadata.require_training_ready()
        self.records = [
            row
            for row in self._freeze.junction_rows
            if _selected_split(row, self.split)
            and row["label_status"] == "trusted_gt"
            and row["label"] in JUNCTION_CLASSES
        ]
        self._verified: set[tuple[Path, str]] = set()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        patch_info = row["patch"]
        patch_path = _resolve_within(
            self.run_dir, patch_info.get("path"), "junction patch"
        )
        expected_hash = patch_info.get("sha256")
        if not _is_sha256(expected_hash):
            raise FrozenDataError(
                f"junction patch lacks a valid SHA-256: {patch_path}"
            )
        key = (patch_path, expected_hash)
        if key not in self._verified:
            actual = _sha256_file(patch_path)
            if actual != expected_hash:
                raise FrozenDataError(
                    f"junction patch SHA-256 mismatch for {patch_path}: "
                    f"expected {expected_hash}, actual {actual}"
                )
            self._verified.add(key)
        expected_size = _size(patch_info.get("size"), row["record_id"])
        with Image.open(patch_path) as opened:
            opened.load()
            image = opened.convert("RGB")
        if image.size != expected_size:
            raise FrozenDataError(
                f"junction row {row['record_id']!r} patch size mismatch"
            )
        label = row["label"]
        label_id = row["label_id"]
        expected_id = JUNCTION_CLASSES.index(label)
        if label_id != expected_id:
            raise FrozenDataError(
                f"junction row {row['record_id']!r} label ID changed after indexing"
            )
        return {
            "image": _rgb_tensor(image, self._torch),
            "label": self._torch.tensor(label_id, dtype=self._torch.long),
            "label_name": label,
            "record_id": row["record_id"],
            "sample_id": row["sample_id"],
            "split": row["split"],
            "transform": row["transform"],
            "source": row["source"],
        }

    def require_training_ready(self) -> None:
        self.metadata.require_training_ready()


def wire_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable edge tiles; padding is always ignored supervision."""

    torch = _require_torch()
    if not batch:
        raise ValueError("wire_collate requires a non-empty batch")
    _validate_tensor_items(batch, "image", channels=3)
    _validate_tensor_items(batch, "target", channels=1)
    _validate_tensor_items(batch, "ignore", channels=1)
    max_height = max(int(item["image"].shape[-2]) for item in batch)
    max_width = max(int(item["image"].shape[-1]) for item in batch)
    images = torch.zeros(
        (len(batch), 3, max_height, max_width), dtype=batch[0]["image"].dtype
    )
    targets = torch.zeros(
        (len(batch), 1, max_height, max_width), dtype=batch[0]["target"].dtype
    )
    ignores = torch.ones(
        (len(batch), 1, max_height, max_width), dtype=batch[0]["ignore"].dtype
    )
    shapes: list[list[int]] = []
    for index, item in enumerate(batch):
        height, width = (int(item["image"].shape[-2]), int(item["image"].shape[-1]))
        if tuple(item["target"].shape[-2:]) != (height, width) or tuple(
            item["ignore"].shape[-2:]
        ) != (height, width):
            raise FrozenDataError("wire batch item image/mask shapes differ")
        images[index, :, :height, :width] = item["image"]
        targets[index, :, :height, :width] = item["target"]
        ignores[index, :, :height, :width] = item["ignore"]
        shapes.append([height, width])
    return {
        "image": images,
        "target": targets,
        "ignore": ignores,
        "valid": 1.0 - ignores,
        "original_shapes": torch.tensor(shapes, dtype=torch.long),
        "record_id": [item["record_id"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
        "transform": [item["transform"] for item in batch],
        "source": [item["source"] for item in batch],
    }


def junction_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pad junction patches and stack integer labels."""

    torch = _require_torch()
    if not batch:
        raise ValueError("junction_collate requires a non-empty batch")
    _validate_tensor_items(batch, "image", channels=3)
    max_height = max(int(item["image"].shape[-2]) for item in batch)
    max_width = max(int(item["image"].shape[-1]) for item in batch)
    images = torch.zeros(
        (len(batch), 3, max_height, max_width), dtype=batch[0]["image"].dtype
    )
    shapes: list[list[int]] = []
    for index, item in enumerate(batch):
        height, width = (int(item["image"].shape[-2]), int(item["image"].shape[-1]))
        images[index, :, :height, :width] = item["image"]
        shapes.append([height, width])
    return {
        "image": images,
        "label": torch.stack([item["label"] for item in batch]),
        "original_shapes": torch.tensor(shapes, dtype=torch.long),
        "record_id": [item["record_id"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
        "label_name": [item["label_name"] for item in batch],
        "transform": [item["transform"] for item in batch],
        "source": [item["source"] for item in batch],
    }


def junction_sample_weights(
    dataset: JunctionTorchDataset,
    *,
    power: float = 1.0,
    normalize_mean: bool = True,
) -> Any:
    """Return inverse-frequency per-sample weights for trusted rows only."""

    torch = _require_torch()
    if not isinstance(power, (int, float)) or isinstance(power, bool) or not math.isfinite(float(power)) or power < 0:
        raise ValueError("power must be a finite non-negative number")
    if not dataset.records:
        raise DatasetGateError("cannot compute junction weights for an empty dataset")
    counts = Counter(row["label"] for row in dataset.records)
    values = [float(counts[row["label"]]) ** -float(power) for row in dataset.records]
    if normalize_mean:
        mean = sum(values) / len(values)
        values = [value / mean for value in values]
    return torch.tensor(values, dtype=torch.double)


def junction_class_weights(
    dataset: JunctionTorchDataset,
    *,
    power: float = 1.0,
    require_complete: bool = True,
) -> Any:
    """Return six class weights; default gate refuses missing classes."""

    torch = _require_torch()
    if require_complete:
        dataset.require_training_ready()
    if not isinstance(power, (int, float)) or isinstance(power, bool) or not math.isfinite(float(power)) or power < 0:
        raise ValueError("power must be a finite non-negative number")
    counts = Counter(row["label"] for row in dataset.records)
    values = [
        float(counts[name]) ** -float(power) if counts[name] else 0.0
        for name in JUNCTION_CLASSES
    ]
    nonzero = [value for value in values if value > 0]
    if nonzero:
        scale = len(nonzero) / sum(nonzero)
        values = [value * scale for value in values]
    return torch.tensor(values, dtype=torch.float32)


def make_junction_weighted_sampler(
    dataset: JunctionTorchDataset,
    *,
    num_samples: int | None = None,
    replacement: bool = True,
    power: float = 1.0,
    require_complete: bool = True,
    generator: Any = None,
) -> Any:
    """Construct ``WeightedRandomSampler`` without starting a training loop."""

    torch = _require_torch()
    if require_complete:
        dataset.require_training_ready()
    weights = junction_sample_weights(dataset, power=power)
    count = len(dataset) if num_samples is None else num_samples
    if type(count) is not int or count <= 0:
        raise ValueError("num_samples must be a positive integer")
    return torch.utils.data.WeightedRandomSampler(
        weights,
        num_samples=count,
        replacement=bool(replacement),
        generator=generator,
    )


def _load_freeze(
    run_dir: str | Path,
    *,
    need_wire: bool,
    need_junction: bool,
) -> _Freeze:
    root = Path(run_dir).expanduser().resolve()
    summary_path = root / "summary.json"
    summary = _read_json(summary_path)
    if summary.get("complete") is not True:
        raise FrozenDataError("materialization summary is not a complete commit marker")
    if summary.get("kind") != "wire_junction_materialization":
        raise FrozenDataError("summary kind is not wire_junction_materialization")
    run_id = summary.get("run_id")
    if not _is_sha256(run_id):
        raise FrozenDataError("summary run_id is not a SHA-256")
    outputs = summary.get("outputs")
    if not isinstance(outputs, dict):
        raise FrozenDataError("summary lacks output hash bindings")
    wire_rows: list[dict[str, Any]] = []
    junction_rows: list[dict[str, Any]] = []
    if need_wire:
        wire_path = _bound_output(root, outputs, "wire_tiles")
        wire_rows = _read_jsonl_unique(wire_path)
        _validate_wire_rows(wire_rows, run_id)
    if need_junction:
        junction_path = _bound_output(root, outputs, "junction_annotations")
        schema_path = _bound_output(root, outputs, "junction_label_schema")
        _validate_label_schema(_read_json(schema_path))
        junction_rows = _read_jsonl_unique(junction_path)
        _validate_junction_rows(junction_rows, run_id, summary)
    return _Freeze(root, summary, wire_rows, junction_rows)


def _bound_output(root: Path, outputs: dict[str, Any], name: str) -> Path:
    metadata = outputs.get(name)
    if not isinstance(metadata, dict):
        raise FrozenDataError(f"summary lacks output binding {name!r}")
    path = _resolve_within(root, metadata.get("path"), f"output {name}")
    if not path.is_file():
        raise FrozenDataError(f"bound output does not exist: {path}")
    expected_bytes = metadata.get("bytes")
    if type(expected_bytes) is not int or path.stat().st_size != expected_bytes:
        raise FrozenDataError(f"bound output byte count mismatch: {path}")
    expected_hash = metadata.get("sha256")
    if not _is_sha256(expected_hash):
        raise FrozenDataError(f"bound output lacks SHA-256: {path}")
    actual = _sha256_file(path)
    if actual != expected_hash:
        raise FrozenDataError(
            f"bound output SHA-256 mismatch for {path}: expected {expected_hash}, actual {actual}"
        )
    return path


def _validate_wire_rows(rows: list[dict[str, Any]], run_id: str) -> None:
    for row in rows:
        identity = row["record_id"]
        split = row.get("split")
        if split not in _ALLOWED_SPLITS:
            raise FrozenDataError(f"wire row {identity!r} has invalid split")
        supervision = row.get("supervision")
        if not isinstance(supervision, dict) or type(supervision.get("eligible")) is not bool:
            raise FrozenDataError(
                f"wire row {identity!r} lacks boolean supervision.eligible"
            )
        lineage = row.get("lineage")
        if not isinstance(lineage, dict) or lineage.get("run_id") != run_id:
            raise FrozenDataError(f"wire row {identity!r} lineage run_id mismatch")
        for field in ("target_mask", "ignore_mask", "source", "transform"):
            if not isinstance(row.get(field), dict):
                raise FrozenDataError(f"wire row {identity!r} lacks {field}")
        _bbox(row.get("tile_bbox"), identity)
        _size(row.get("tile_size"), identity)


def _validate_junction_rows(
    rows: list[dict[str, Any]], run_id: str, summary: dict[str, Any]
) -> None:
    global_counts: Counter[str] = Counter()
    split_counts: dict[str, Counter[str]] = {
        split: Counter() for split in _ALLOWED_SPLITS
    }
    for row in rows:
        identity = row["record_id"]
        split = row.get("split")
        if split not in _ALLOWED_SPLITS:
            raise FrozenDataError(f"junction row {identity!r} has invalid split")
        lineage = row.get("lineage")
        if not isinstance(lineage, dict) or lineage.get("run_id") != run_id:
            raise FrozenDataError(f"junction row {identity!r} lineage run_id mismatch")
        label = row.get("label")
        status = row.get("label_status")
        label_id = row.get("label_id")
        if label == "UNKNOWN":
            if status != "needs_review" or label_id != -1:
                raise FrozenDataError(
                    f"junction row {identity!r} UNKNOWN is not fail-closed"
                )
        elif label in JUNCTION_CLASSES:
            if status != "trusted_gt" or label_id != JUNCTION_CLASSES.index(label):
                raise FrozenDataError(
                    f"junction row {identity!r} trusted label contract mismatch"
                )
        else:
            raise FrozenDataError(f"junction row {identity!r} has unknown label {label!r}")
        if not isinstance(row.get("patch"), dict):
            raise FrozenDataError(f"junction row {identity!r} lacks patch metadata")
        global_counts[label] += 1
        split_counts[split][label] += 1
    declared = summary.get("junction_class_counts")
    if not isinstance(declared, dict):
        raise FrozenDataError("summary lacks junction class counts")
    expected_global = {
        **{name: int(global_counts[name]) for name in JUNCTION_CLASSES},
        "UNKNOWN": int(global_counts["UNKNOWN"]),
    }
    declared_trusted = declared.get("trusted")
    if not isinstance(declared_trusted, dict) or {
        name: declared_trusted.get(name) for name in JUNCTION_CLASSES
    } != {name: expected_global[name] for name in JUNCTION_CLASSES}:
        raise FrozenDataError("summary trusted junction counts differ from manifest")
    if declared.get("UNKNOWN") != expected_global["UNKNOWN"]:
        raise FrozenDataError("summary UNKNOWN junction count differs from manifest")
    declared_by_split = declared.get("by_split")
    if not isinstance(declared_by_split, dict):
        raise FrozenDataError("summary lacks junction counts by split")
    for split in _ALLOWED_SPLITS:
        actual = {
            **{name: int(split_counts[split][name]) for name in JUNCTION_CLASSES},
            "UNKNOWN": int(split_counts[split]["UNKNOWN"]),
        }
        declared_split = declared_by_split.get(split)
        if declared_split is None and not any(actual.values()):
            continue
        if actual != declared_split:
            raise FrozenDataError(
                f"summary junction counts for split {split!r} differ from manifest"
            )
    gate = summary.get("training_gate")
    if not isinstance(gate, dict) or gate.get("required_classes") != list(
        JUNCTION_CLASSES
    ):
        raise FrozenDataError("summary junction training gate class order is invalid")


def _validate_label_schema(schema: dict[str, Any]) -> None:
    classes = schema.get("classes")
    if not isinstance(classes, list) or [
        item.get("name") if isinstance(item, dict) else None for item in classes
    ] != list(JUNCTION_CLASSES):
        raise FrozenDataError("junction label schema class order mismatch")
    if [item.get("id") for item in classes] != list(range(len(JUNCTION_CLASSES))):
        raise FrozenDataError("junction label schema IDs mismatch")
    cross = schema.get("cross_policy")
    if not isinstance(cross, dict) or cross.get("automatic_label") != "UNKNOWN" or cross.get("needs_review") is not True:
        raise FrozenDataError("junction label schema cross policy is not fail-closed")


def _wire_metadata(freeze: _Freeze, split: str | None) -> WireDatasetMetadata:
    selected = [row for row in freeze.wire_rows if _selected_split(row, split)]
    eligible = [row for row in selected if row["supervision"]["eligible"] is True]
    by_split = Counter(
        row["split"]
        for row in freeze.wire_rows
        if row["supervision"]["eligible"] is True
    )
    materialization = freeze.summary.get("materialization", {})
    return WireDatasetMetadata(
        run_id=freeze.summary["run_id"],
        split=split,
        total_manifest_records=len(freeze.wire_rows),
        selected_records=len(selected),
        eligible_records=len(eligible),
        excluded_ineligible_records=len(selected) - len(eligible),
        excluded_other_splits=len(freeze.wire_rows) - len(selected),
        eligible_by_split={name: int(by_split[name]) for name in sorted(_ALLOWED_SPLITS)},
        source_materialization_ok=materialization.get("ok") is True,
    )


def _junction_metadata(
    freeze: _Freeze, split: str | None
) -> JunctionDatasetMetadata:
    selected = [row for row in freeze.junction_rows if _selected_split(row, split)]
    trusted = [row for row in selected if row["label_status"] == "trusted_gt"]
    counts = Counter(row["label"] for row in trusted)
    missing = tuple(name for name in JUNCTION_CLASSES if counts[name] == 0)
    gate = freeze.summary.get("training_gate")
    if not isinstance(gate, dict):
        raise FrozenDataError("summary lacks training_gate")
    blockers = gate.get("blockers")
    if not isinstance(blockers, list) or any(not isinstance(item, dict) for item in blockers):
        raise FrozenDataError("summary training_gate.blockers is invalid")
    return JunctionDatasetMetadata(
        run_id=freeze.summary["run_id"],
        split=split,
        total_manifest_records=len(freeze.junction_rows),
        selected_records=len(selected),
        trusted_records=len(trusted),
        excluded_unknown_records=sum(row["label"] == "UNKNOWN" for row in selected),
        excluded_other_splits=len(freeze.junction_rows) - len(selected),
        class_counts={name: int(counts[name]) for name in JUNCTION_CLASSES},
        missing_classes=missing,
        source_training_gate_ok=gate.get("ok") is True,
        source_gate_blockers=tuple(blockers),
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FrozenDataError(f"required freeze file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise FrozenDataError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FrozenDataError(f"JSON root is not an object: {path}")
    return value


def _read_jsonl_unique(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=_reject_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise FrozenDataError(
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise FrozenDataError(f"JSONL row at {path}:{line_number} is not an object")
            identity = row.get("record_id")
            if not isinstance(identity, str) or not identity:
                raise FrozenDataError(f"JSONL row at {path}:{line_number} lacks record_id")
            if identity in seen:
                raise FrozenDataError(f"duplicate record_id {identity!r} in {path}")
            seen.add(identity)
            rows.append(row)
    return rows


def _rgb_tensor(image: Image.Image, torch: Any) -> Any:
    width, height = image.size
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return (
        data.reshape(height, width, 3)
        .permute(2, 0, 1)
        .contiguous()
        .to(dtype=torch.float32)
        .div_(255.0)
    )


def _binary_mask_tensor(
    image: Image.Image, foreground: int, background: int, torch: Any
) -> Any:
    width, height = image.size
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(
        height, width
    )
    valid = (data == foreground) | (data == background)
    if not bool(torch.all(valid)):
        values = sorted(int(value) for value in torch.unique(data).tolist())
        raise FrozenDataError(
            f"mask contains values outside foreground/background: {values}"
        )
    return (data == foreground).to(dtype=torch.float32).unsqueeze(0)


def _load_mask(path: Path, expected_size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as opened:
        opened.load()
        image = opened.convert("L")
    if image.size != expected_size:
        raise FrozenDataError(
            f"mask size {image.size} differs from declared tile size {expected_size}: {path}"
        )
    return image


def _validate_tensor_items(
    batch: Sequence[dict[str, Any]], field: str, *, channels: int
) -> None:
    for item in batch:
        tensor = item.get(field)
        if tensor is None or getattr(tensor, "ndim", None) != 3 or int(tensor.shape[0]) != channels:
            raise FrozenDataError(
                f"batch field {field!r} must contain [C,H,W] tensors with C={channels}"
            )


def _validated_split(split: str | None) -> str | None:
    if split is not None and split not in _ALLOWED_SPLITS:
        raise ValueError("split must be train, val, test, or None")
    return split


def _selected_split(row: dict[str, Any], split: str | None) -> bool:
    return split is None or row["split"] == split


def _bbox(value: Any, identity: str) -> tuple[int, int, int, int]:
    if not isinstance(value, dict):
        raise FrozenDataError(f"record {identity!r} lacks tile_bbox")
    values: list[int] = []
    for key in ("x1", "y1", "x2", "y2"):
        item = value.get(key)
        if type(item) is not int:
            raise FrozenDataError(f"record {identity!r} tile_bbox.{key} is not an integer")
        values.append(item)
    if values[0] < 0 or values[1] < 0 or values[2] <= values[0] or values[3] <= values[1]:
        raise FrozenDataError(f"record {identity!r} has invalid tile_bbox")
    return values[0], values[1], values[2], values[3]


def _size(value: Any, identity: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2 or any(
        type(item) is not int or item <= 0 for item in value
    ):
        raise FrozenDataError(f"record {identity!r} has invalid image size")
    return value[0], value[1]


def _resolve_within(root: Path, relative: Any, kind: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise FrozenDataError(f"{kind} has an invalid relative path")
    path = (root / Path(relative)).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise FrozenDataError(f"{kind} path escapes its root: {relative!r}") from exc
    return path


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        raise TorchUnavailableError(
            "PyTorch is required for tensor datasets/collate/samplers but is not "
            "available in this Python runtime. Use the project training environment "
            "or install a compatible torch build; freeze inspection remains available "
            "without torch."
        ) from exc


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value.lower()
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


__all__ = [
    "DatasetGateError",
    "FrozenDataError",
    "JUNCTION_CLASSES",
    "JunctionDatasetMetadata",
    "JunctionTorchDataset",
    "TorchUnavailableError",
    "WireDatasetMetadata",
    "WireTorchDataset",
    "inspect_junction_dataset",
    "inspect_wire_dataset",
    "junction_class_weights",
    "junction_collate",
    "junction_sample_weights",
    "make_junction_weighted_sampler",
    "wire_collate",
]
