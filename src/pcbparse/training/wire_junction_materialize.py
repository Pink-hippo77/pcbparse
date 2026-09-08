from __future__ import annotations

"""Materialize trustworthy Wire and Junction training inputs.

The Dataset300-v2 manifests intentionally contain lazy JSON Pointers instead of
copying dense edge coordinates.  This module resolves only ``keyPointPaths``
and ``keyPoints/skeleton`` from the source annotation, creates one-pixel wire
targets, and crops junction review patches.  The raw ``coordinates`` member is
lexically replaced by ``null`` before JSON decoding, so it is never expanded
into a Python list.

Publication is content-addressed.  A complete run is first written to a hidden
staging directory and then renamed into ``<output>/runs/<run_id>``.  The
``summary.json`` file is the commit marker and ``latest.json`` is replaced
atomically afterwards.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from PIL import Image, ImageDraw


PinLinePolicy = Literal["ignore", "positive", "negative"]

JUNCTION_CLASSES: tuple[str, ...] = (
    "endpoint",
    "t_junction",
    "bend",
    "connected_cross",
    "non_connected_cross",
    "noise",
)
_SAFE_JUNCTION_MAP = {
    "endpoint": "endpoint",
    "diverge": "t_junction",
    "turn": "bend",
}
_ALLOWED_SPLITS = frozenset({"train", "val", "test"})


class MaterializationError(ValueError):
    """Raised when source lineage or geometry cannot be trusted."""


@dataclass(frozen=True, slots=True)
class WireJunctionMaterializeConfig:
    tile_size: int = 512
    geometry_types: tuple[str, ...] = ("wire",)
    target_edge_categories: tuple[str, ...] = ("wire",)
    pin_line_policy: PinLinePolicy = "ignore"
    pin_ignore_width: int = 3
    ignore_bbox_margin: int = 0
    mask_foreground: int = 255
    mask_background: int = 0

    def __post_init__(self) -> None:
        if type(self.tile_size) is not int or self.tile_size <= 0:
            raise ValueError("tile_size must be a positive integer")
        if not self.geometry_types or any(
            not isinstance(value, str) or not value for value in self.geometry_types
        ):
            raise ValueError("geometry_types must contain non-empty strings")
        if not self.target_edge_categories or any(
            not isinstance(value, str) or not value
            for value in self.target_edge_categories
        ):
            raise ValueError("target_edge_categories must contain non-empty strings")
        if self.pin_line_policy not in {"ignore", "positive", "negative"}:
            raise ValueError("pin_line_policy must be ignore, positive, or negative")
        if type(self.pin_ignore_width) is not int or self.pin_ignore_width <= 0:
            raise ValueError("pin_ignore_width must be a positive integer")
        if type(self.ignore_bbox_margin) is not int or self.ignore_bbox_margin < 0:
            raise ValueError("ignore_bbox_margin must be a non-negative integer")
        for name, value in (
            ("mask_foreground", self.mask_foreground),
            ("mask_background", self.mask_background),
        ):
            if type(value) is not int or not 0 <= value <= 255:
                raise ValueError(f"{name} must be an integer in [0, 255]")
        if self.mask_foreground == self.mask_background:
            raise ValueError("mask_foreground and mask_background must differ")


@dataclass(frozen=True, slots=True)
class MaterializationResult:
    run_id: str
    run_dir: Path
    summary_path: Path
    summary: dict[str, Any]

    @property
    def training_ready(self) -> bool:
        return bool(self.summary.get("training_gate", {}).get("ok"))


@dataclass(slots=True)
class _InputRows:
    wire: list[dict[str, Any]]
    junction: list[dict[str, Any]]
    source: dict[str, dict[str, Any]]
    split: dict[str, dict[str, Any]]
    derivation_version: str
    config_hash: str
    hashes: dict[str, str]


@dataclass(slots=True)
class _EdgeGeometry:
    record_id: str
    edge_key: str
    paths: list[list[tuple[float, float]]]
    skeleton: list[tuple[float, float]]
    bbox: tuple[float, float, float, float] | None


def _load_review_overrides(path: str | Path) -> dict[str, str]:
    """Load human-reviewed junction labels from the labeler progress CSV."""
    import csv as _csv

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise MaterializationError(f"review override file not found: {source}")
    overrides: dict[str, str] = {}
    with source.open(encoding="utf-8", newline="") as handle:
        for row in _csv.DictReader(handle):
            status = str(row.get("review_status", "")).strip().lower()
            label = str(row.get("reviewed_label", "")).strip()
            record_id = str(row.get("record_id", "")).strip()
            if status not in {"labeled", "reviewed"} or not label:
                continue
            if label not in JUNCTION_CLASSES:
                raise MaterializationError(
                    f"review override for {record_id!r} has invalid label {label!r}"
                )
            overrides[record_id] = label
    return overrides



def materialize_wire_junction(
    *,
    wire_manifest: str | Path,
    junction_manifest: str | Path,
    source_manifest: str | Path,
    split_manifest: str | Path,
    source_root: str | Path,
    output_root: str | Path,
    config: WireJunctionMaterializeConfig | None = None,
    sample_ids: Iterable[str] | None = None,
    expected_derivation_version: str | None = None,
    expected_config_hash: str | None = None,
    review_overrides: Mapping[str, str] | None = None,
) -> MaterializationResult:
    """Materialize masks and patches while preserving verifiable lineage.

    The materializer validates *all* input manifest rows before applying an
    optional sample filter.  A false junction training gate is a successful
    materialization result: it records missing labels and pending review rather
    than inventing connected/non-connected/noise supervision.
    """

    cfg = config or WireJunctionMaterializeConfig()
    wire_path = Path(wire_manifest).expanduser().resolve()
    junction_path = Path(junction_manifest).expanduser().resolve()
    source_manifest_path = Path(source_manifest).expanduser().resolve()
    split_manifest_path = Path(split_manifest).expanduser().resolve()
    source_root_path = Path(source_root).expanduser().resolve()
    output_root_path = Path(output_root).expanduser().resolve(strict=False)

    inputs = _load_and_validate_inputs(
        wire_path=wire_path,
        junction_path=junction_path,
        source_manifest_path=source_manifest_path,
        split_manifest_path=split_manifest_path,
        expected_derivation_version=expected_derivation_version,
        expected_config_hash=expected_config_hash,
    )
    selected = _select_samples(sample_ids, inputs.source)
    config_document = {
        "schema_version": 1,
        **asdict(cfg),
        "geometry_types": list(cfg.geometry_types),
        "target_edge_categories": list(cfg.target_edge_categories),
        "selected_sample_ids": selected,
    }
    config_hash = _sha256_bytes(_canonical_bytes(config_document))
    run_document = {
        "schema_version": 1,
        "kind": "wire_junction_materialization",
        "inputs": inputs.hashes,
        "source_path_policy": "source_manifest_paths_relative_to_source_root",
        "derivation_version": inputs.derivation_version,
        "derived_config_hash": inputs.config_hash,
        "materializer_config_sha256": config_hash,
        "selected_sample_ids": selected,
    }
    run_id = _sha256_bytes(_canonical_bytes(run_document))
    runs_root = output_root_path / "runs"
    final_dir = runs_root / run_id
    output_root_path.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    existing = _existing_result(final_dir, run_id)
    if existing is not None:
        _publish_latest(output_root_path, existing)
        return existing

    staging = Path(
        tempfile.mkdtemp(prefix=".wire-junction-", dir=str(output_root_path))
    )
    try:
        result = _materialize_into(
            staging=staging,
            final_dir=final_dir,
            run_id=run_id,
            run_document=run_document,
            materializer_config=config_document,
            materializer_config_hash=config_hash,
            inputs=inputs,
            selected=selected,
            source_root=source_root_path,
            cfg=cfg,
            review_overrides=review_overrides,
        )
        try:
            os.replace(staging, final_dir)
        except OSError:
            raced = _existing_result(final_dir, run_id)
            if raced is None:
                raise
            shutil.rmtree(staging, ignore_errors=True)
            result = raced
        result = MaterializationResult(
            run_id=run_id,
            run_dir=final_dir,
            summary_path=final_dir / "summary.json",
            summary=result.summary,
        )
        _publish_latest(output_root_path, result)
        return result
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_and_validate_inputs(
    *,
    wire_path: Path,
    junction_path: Path,
    source_manifest_path: Path,
    split_manifest_path: Path,
    expected_derivation_version: str | None,
    expected_config_hash: str | None,
) -> _InputRows:
    for path in (wire_path, junction_path, source_manifest_path, split_manifest_path):
        if not path.is_file():
            raise MaterializationError(f"required input file does not exist: {path}")

    wire = _read_jsonl_unique(wire_path, "record_id")
    junction = _read_jsonl_unique(junction_path, "record_id")
    source_rows = _read_jsonl_unique(source_manifest_path, "sample_id")
    split_rows = _read_jsonl_unique(split_manifest_path, "sample_id")
    if not wire:
        raise MaterializationError("wire manifest is empty")
    if not junction:
        raise MaterializationError("junction manifest is empty")

    source = {row["sample_id"]: row for row in source_rows}
    split = {row["sample_id"]: row for row in split_rows}
    if set(source) != set(split):
        raise MaterializationError(
            "source and split manifests have different sample sets: "
            f"missing_in_split={sorted(set(source) - set(split))[:10]}, "
            f"unexpected_in_split={sorted(set(split) - set(source))[:10]}"
        )

    group_splits: dict[str, set[str]] = defaultdict(set)
    for sample_id, source_row in sorted(source.items()):
        split_name = split[sample_id].get("split")
        if split_name not in _ALLOWED_SPLITS:
            raise MaterializationError(
                f"sample {sample_id!r} has invalid split {split_name!r}"
            )
        source_split = source_row.get("split")
        if source_split is not None and source_split != split_name:
            raise MaterializationError(
                f"sample {sample_id!r} split differs between manifests"
            )
        group = source_row.get("duplicate_group")
        if not isinstance(group, str) or not group:
            raise MaterializationError(
                f"source sample {sample_id!r} lacks duplicate_group"
            )
        group_splits[group].add(split_name)
    leaking = {key: sorted(value) for key, value in group_splits.items() if len(value) > 1}
    if leaking:
        first = sorted(leaking)[0]
        raise MaterializationError(
            f"duplicate group {first!r} leaks across splits {leaking[first]}"
        )

    lineage_values: set[tuple[str, str]] = set()
    sample_metadata: dict[str, tuple[Any, ...]] = {}
    for expected_task, rows in (("wire_paths", wire), ("junction_patches", junction)):
        for row in rows:
            if row.get("task") != expected_task:
                raise MaterializationError(
                    f"record {row.get('record_id')!r} has task {row.get('task')!r}, "
                    f"expected {expected_task!r}"
                )
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or sample_id not in source:
                raise MaterializationError(
                    f"record {row.get('record_id')!r} references unknown sample {sample_id!r}"
                )
            derivation = row.get("derivation_version")
            derived_hash = row.get("config_hash")
            if not isinstance(derivation, str) or not derivation:
                raise MaterializationError("derived record lacks derivation_version")
            if not isinstance(derived_hash, str) or len(derived_hash) != 64:
                raise MaterializationError("derived record lacks a SHA-256 config_hash")
            lineage_values.add((derivation, derived_hash))
            expected_split = split[sample_id]["split"]
            if row.get("split") != expected_split:
                raise MaterializationError(
                    f"record {row['record_id']!r} split {row.get('split')!r} does not "
                    f"match split manifest {expected_split!r}"
                )
            files = source[sample_id].get("files")
            if not isinstance(files, dict):
                raise MaterializationError(f"source sample {sample_id!r} lacks files")
            json_info = files.get("json")
            image_info = files.get("image")
            if not isinstance(json_info, dict) or not isinstance(image_info, dict):
                raise MaterializationError(
                    f"source sample {sample_id!r} lacks image/json file metadata"
                )
            expected_metadata = (
                json_info.get("path"),
                image_info.get("path"),
                source[sample_id].get("source_class"),
                source[sample_id].get("duplicate_group"),
            )
            row_metadata = (
                row.get("source_json"),
                row.get("source_image"),
                row.get("source_domain"),
                row.get("duplicate_group"),
            )
            if row_metadata != expected_metadata:
                raise MaterializationError(
                    f"record {row['record_id']!r} source metadata differs from source manifest"
                )
            previous = sample_metadata.setdefault(sample_id, row_metadata)
            if previous != row_metadata:
                raise MaterializationError(
                    f"derived rows disagree on metadata for sample {sample_id!r}"
                )
    if len(lineage_values) != 1:
        raise MaterializationError(
            f"derived manifests contain {len(lineage_values)} lineage pairs; expected exactly one"
        )
    derivation_version, config_hash = next(iter(lineage_values))
    if expected_derivation_version is not None and derivation_version != expected_derivation_version:
        raise MaterializationError(
            f"derivation version {derivation_version!r} does not match expected "
            f"{expected_derivation_version!r}"
        )
    if expected_config_hash is not None and config_hash != expected_config_hash:
        raise MaterializationError(
            f"derived config hash {config_hash!r} does not match expected {expected_config_hash!r}"
        )
    return _InputRows(
        wire=wire,
        junction=junction,
        source=source,
        split=split,
        derivation_version=derivation_version,
        config_hash=config_hash,
        hashes={
            "wire_manifest_sha256": _sha256_file(wire_path),
            "junction_manifest_sha256": _sha256_file(junction_path),
            "source_manifest_sha256": _sha256_file(source_manifest_path),
            "split_manifest_sha256": _sha256_file(split_manifest_path),
        },
    )


def _select_samples(
    requested: Iterable[str] | None, source: dict[str, dict[str, Any]]
) -> list[str]:
    if requested is None:
        return sorted(source)
    selected: set[str] = set()
    for sample_id in requested:
        if not isinstance(sample_id, str) or not sample_id:
            raise MaterializationError("sample_ids must contain non-empty strings")
        selected.add(sample_id)
    unknown = sorted(selected - set(source))
    if unknown:
        raise MaterializationError(f"unknown selected sample IDs: {unknown[:20]}")
    if not selected:
        raise MaterializationError("sample_ids cannot be empty")
    return sorted(selected)


def _materialize_into(
    *,
    staging: Path,
    final_dir: Path,
    run_id: str,
    run_document: dict[str, Any],
    materializer_config: dict[str, Any],
    materializer_config_hash: str,
    inputs: _InputRows,
    selected: list[str],
    source_root: Path,
    cfg: WireJunctionMaterializeConfig,
    review_overrides: Mapping[str, str] | None = None,
) -> MaterializationResult:
    wire_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    junction_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in inputs.wire:
        if row["sample_id"] in selected:
            wire_by_sample[row["sample_id"]].append(row)
    for row in inputs.junction:
        if row["sample_id"] in selected:
            junction_by_sample[row["sample_id"]].append(row)
    for rows in (wire_by_sample, junction_by_sample):
        for values in rows.values():
            values.sort(key=lambda item: item["record_id"])

    wire_tiles: list[dict[str, Any]] = []
    junction_annotations: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    junction_counts: Counter[str] = Counter()
    junction_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    source_stats: list[dict[str, Any]] = []
    asset_entries: list[dict[str, Any]] = []
    wire_source_records = 0
    wire_skipped_categories: Counter[str] = Counter()
    wire_unusable_sparse_records: list[str] = []

    for sample_id in selected:
        source_row = inputs.source[sample_id]
        split_name = inputs.split[sample_id]["split"]
        files = source_row["files"]
        json_info = files["json"]
        image_info = files["image"]
        json_path = _resolve_source_file(source_root, json_info["path"])
        image_path = _resolve_source_file(source_root, image_info["path"])
        json_sha = _verify_source_hash(json_path, json_info, sample_id, "json")
        image_sha = _verify_source_hash(image_path, image_info, sample_id, "image")
        annotation, dense_stats = _load_annotation_without_dense_coordinates(json_path)
        canvas = _validate_canvas(annotation, source_row, sample_id)

        with Image.open(image_path) as opened:
            opened.load()
            if opened.size != (canvas["width"], canvas["height"]):
                raise MaterializationError(
                    f"sample {sample_id!r} image size {opened.size} differs from canvas "
                    f"{(canvas['width'], canvas['height'])}"
                )
            source_image = opened.convert("RGB")

        geometries: list[_EdgeGeometry] = []
        sample_unusable_sparse_records: list[str] = []
        for row in wire_by_sample.get(sample_id, []):
            category = row.get("category")
            if category not in cfg.target_edge_categories:
                wire_skipped_categories[str(category)] += 1
                continue
            geometry = _resolve_wire_geometry(row, annotation, cfg.geometry_types)
            if geometry.bbox is None:
                sample_unusable_sparse_records.append(row["record_id"])
                wire_unusable_sparse_records.append(row["record_id"])
            else:
                geometries.append(geometry)
            wire_source_records += 1
        ignore_regions = _collect_ignore_regions(annotation, canvas, cfg)
        sample_tiles, sample_assets = _write_wire_tiles(
            staging=staging,
            sample_id=sample_id,
            split_name=split_name,
            source_row=source_row,
            json_sha=json_sha,
            image_sha=image_sha,
            canvas=canvas,
            geometries=geometries,
            unusable_sparse_record_ids=sample_unusable_sparse_records,
            ignore_regions=ignore_regions,
            run_id=run_id,
            cfg=cfg,
        )
        wire_tiles.extend(sample_tiles)
        asset_entries.extend(sample_assets)

        for row in junction_by_sample.get(sample_id, []):
            _validate_junction_sources(row, annotation)
            label, review_reason, review_candidates = _safe_junction_label(row)
            human_label = (review_overrides or {}).get(row["record_id"])
            if human_label is not None:
                label = human_label
                review_reason = "human_reviewed"
                review_candidates = []
            patch_rel = _junction_patch_relative_path(row)
            patch_path = staging / patch_rel
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            patch, transform = _junction_patch(source_image, row)
            _save_png(patch, patch_path)
            patch_sha = _sha256_file(patch_path)
            asset_entries.append(
                {
                    "path": patch_rel.as_posix(),
                    "sha256": patch_sha,
                    "bytes": patch_path.stat().st_size,
                    "kind": "junction_patch",
                }
            )
            label_key = label if label is not None else "UNKNOWN"
            junction_counts[label_key] += 1
            junction_by_split[split_name][label_key] += 1
            annotation_record = {
                "schema_version": 1,
                "record_id": row["record_id"],
                "sample_id": sample_id,
                "split": split_name,
                "label": label_key,
                "label_id": JUNCTION_CLASSES.index(label) if label is not None else -1,
                "label_status": "trusted_gt" if label is not None else "needs_review",
                "review_reason": review_reason,
                "review_candidates": review_candidates,
                "candidate_types": row.get("candidate_types"),
                "center": row.get("center"),
                "patch": {
                    "path": patch_rel.as_posix(),
                    "sha256": patch_sha,
                    "size": [patch.width, patch.height],
                },
                "transform": transform,
                "source": {
                    "image": row["source_image"],
                    "image_sha256": image_sha,
                    "json": row["source_json"],
                    "json_sha256": json_sha,
                    "source_refs": row.get("source_refs", []),
                },
                "lineage": {
                    "run_id": run_id,
                    "derived_record_id": row["record_id"],
                    "derivation_version": inputs.derivation_version,
                    "derived_config_hash": inputs.config_hash,
                },
            }
            junction_annotations.append(annotation_record)
            if label is None:
                review_queue.append(
                    {
                        "schema_version": 1,
                        "record_id": row["record_id"],
                        "sample_id": sample_id,
                        "split": split_name,
                        "patch_path": patch_rel.as_posix(),
                        "source_image": row["source_image"],
                        "center": row.get("center"),
                        "candidate_types": row.get("candidate_types"),
                        "reason": review_reason,
                        "allowed_labels": list(JUNCTION_CLASSES),
                        "suggested_labels": review_candidates,
                        "review_status": "pending",
                    }
                )

        source_stats.append(
            {
                "sample_id": sample_id,
                "split": split_name,
                "json_sha256": json_sha,
                "image_sha256": image_sha,
                "dense_coordinates_loaded": False,
                "dense_coordinate_fields_stripped": dense_stats["fields"],
                "dense_coordinate_source_bytes_skipped": dense_stats["bytes"],
                "wire_manifest_records": len(wire_by_sample.get(sample_id, [])),
                "junction_manifest_records": len(junction_by_sample.get(sample_id, [])),
                "tile_count": len(sample_tiles),
            }
        )

    wire_tiles.sort(key=lambda item: item["record_id"])
    junction_annotations.sort(key=lambda item: item["record_id"])
    review_queue.sort(key=lambda item: item["record_id"])
    asset_entries.sort(key=lambda item: item["path"])

    output_paths = {
        "wire_tiles": staging / "wire_tiles.jsonl",
        "junction_annotations": staging / "junction_annotations.jsonl",
        "junction_review_queue": staging / "junction_review_queue.jsonl",
        "junction_review_template": staging / "junction_review_template.csv",
        "junction_label_schema": staging / "junction_label_schema.json",
        "asset_manifest": staging / "asset_manifest.jsonl",
    }
    _write_jsonl(wire_tiles, output_paths["wire_tiles"])
    _write_jsonl(junction_annotations, output_paths["junction_annotations"])
    _write_jsonl(review_queue, output_paths["junction_review_queue"])
    _write_review_template(review_queue, output_paths["junction_review_template"])
    _write_json(_junction_label_schema(), output_paths["junction_label_schema"])
    _write_jsonl(asset_entries, output_paths["asset_manifest"])

    six_counts = {name: int(junction_counts.get(name, 0)) for name in JUNCTION_CLASSES}
    by_split = {
        split_name: {
            **{name: int(counter.get(name, 0)) for name in JUNCTION_CLASSES},
            "UNKNOWN": int(counter.get("UNKNOWN", 0)),
        }
        for split_name, counter in sorted(junction_by_split.items())
    }
    train_counter = junction_by_split.get("train", Counter())
    missing_train = [name for name in JUNCTION_CLASSES if train_counter.get(name, 0) == 0]
    missing_global = [name for name in JUNCTION_CLASSES if junction_counts.get(name, 0) == 0]
    blockers: list[dict[str, Any]] = []
    for name in missing_train:
        blockers.append(
            {
                "code": "JUNCTION_CLASS_MISSING_FROM_TRAIN",
                "class": name,
                "message": f"trusted train labels contain no {name!r} examples",
            }
        )
    if review_queue:
        blockers.append(
            {
                "code": "JUNCTION_REVIEW_PENDING",
                "count": len(review_queue),
                "message": "UNKNOWN/cross candidates require human review before training",
            }
        )

    output_hashes = {
        name: {
            "path": path.relative_to(staging).as_posix(),
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for name, path in sorted(output_paths.items())
    }
    asset_tree_hash = _asset_tree_hash(asset_entries)
    summary = {
        "schema_version": 1,
        "kind": "wire_junction_materialization",
        "complete": True,
        "run_id": run_id,
        "materialization": {
            "ok": True,
            "selected_sample_count": len(selected),
            "wire_tile_count": len(wire_tiles),
            "wire_source_record_count": wire_source_records,
            "junction_annotation_count": len(junction_annotations),
            "review_queue_count": len(review_queue),
        },
        "training_gate": {
            "ok": not blockers,
            "required_classes": list(JUNCTION_CLASSES),
            "required_split": "train",
            "missing_classes_in_train": missing_train,
            "missing_classes_global": missing_global,
            "blockers": blockers,
        },
        "junction_class_counts": {
            "trusted": six_counts,
            "UNKNOWN": int(junction_counts.get("UNKNOWN", 0)),
            "by_split": by_split,
        },
        "wire": {
            "mask_semantics": {
                "foreground": cfg.mask_foreground,
                "background": cfg.mask_background,
                "centerline_width_pixels": 1,
                "ignore_precedence": "ignore pixels are cleared from the positive target",
                "geometry_sources": ["keyPointPaths", "keyPoints/skeleton"],
                "dense_coordinates_loaded": False,
                "pin_line_policy": cfg.pin_line_policy,
            },
            "skipped_edge_categories": dict(sorted(wire_skipped_categories.items())),
            "unusable_sparse_geometry_record_count": len(wire_unusable_sparse_records),
            "unusable_sparse_geometry_record_ids": sorted(wire_unusable_sparse_records),
            "unusable_sparse_policy": "mark every tile from the affected sample ineligible and fully ignored",
        },
        "lineage": run_document,
        "materializer_config": materializer_config,
        "materializer_config_sha256": materializer_config_hash,
        "input_manifest_counts": {
            "wire_paths": len(inputs.wire),
            "junction_patches": len(inputs.junction),
            "source_samples": len(inputs.source),
            "selected_wire_paths": sum(len(value) for value in wire_by_sample.values()),
            "selected_junction_patches": sum(
                len(value) for value in junction_by_sample.values()
            ),
        },
        "sources": source_stats,
        "outputs": output_hashes,
        "assets": {
            "count": len(asset_entries),
            "bytes": sum(int(item["bytes"]) for item in asset_entries),
            "tree_sha256": asset_tree_hash,
        },
        "publication": {
            "content_addressed_run_dir": (Path("runs") / run_id).as_posix(),
            "commit_marker": "summary.json",
            "atomic_directory_publish": True,
        },
    }
    summary_path = staging / "summary.json"
    _write_json(summary, summary_path)
    return MaterializationResult(
        run_id=run_id,
        run_dir=staging,
        summary_path=summary_path,
        summary=summary,
    )


def _resolve_wire_geometry(
    row: dict[str, Any], annotation: dict[str, Any], geometry_types: Sequence[str]
) -> _EdgeGeometry:
    edge_key = row.get("edge_key")
    if not isinstance(edge_key, str) or not edge_key:
        raise MaterializationError(f"wire record {row['record_id']!r} lacks edge_key")
    edges = annotation.get("edges")
    if not isinstance(edges, dict) or edge_key not in edges:
        raise MaterializationError(
            f"wire record {row['record_id']!r} references missing raw edge {edge_key!r}"
        )
    edge = edges[edge_key]
    if not isinstance(edge, dict):
        raise MaterializationError(f"raw edge {edge_key!r} is not an object")
    if edge.get("coordinates") is not None:
        raise MaterializationError(
            f"dense coordinates unexpectedly survived filtering for edge {edge_key!r}"
        )
    path_ref = row.get("key_point_paths_ref")
    skeleton_ref = row.get("skeleton_ref")
    if not isinstance(path_ref, dict) or not isinstance(skeleton_ref, dict):
        raise MaterializationError(
            f"wire record {row['record_id']!r} lacks geometry references"
        )
    paths_value = _resolve_json_pointer(annotation, path_ref.get("json_pointer"))
    skeleton_value = _resolve_json_pointer(annotation, skeleton_ref.get("json_pointer"))
    if paths_value is not edge.get("keyPointPaths"):
        raise MaterializationError(
            f"wire record {row['record_id']!r} keyPointPaths pointer targets another object"
        )
    key_points = edge.get("keyPoints")
    if not isinstance(key_points, dict) or skeleton_value is not key_points.get("skeleton"):
        raise MaterializationError(
            f"wire record {row['record_id']!r} skeleton pointer targets another object"
        )
    paths = _extract_paths(paths_value, geometry_types, row["record_id"])
    skeleton = _extract_points(skeleton_value, geometry_types, row["record_id"])
    _validate_geometry_summary(path_ref.get("geometry"), paths_value, "path", row)
    _validate_geometry_summary(skeleton_ref.get("geometry"), skeleton_value, "point", row)
    all_points = [point for path in paths for point in path] + skeleton
    return _EdgeGeometry(
        record_id=row["record_id"],
        edge_key=edge_key,
        paths=paths,
        skeleton=skeleton,
        bbox=_points_bbox(all_points),
    )


def _validate_geometry_summary(
    declared: Any,
    raw: Any,
    kind: Literal["path", "point"],
    row: dict[str, Any],
) -> None:
    if not isinstance(declared, dict) or not isinstance(raw, dict):
        raise MaterializationError(
            f"wire record {row['record_id']!r} has invalid {kind} geometry summary"
        )
    for geometry, summary in declared.items():
        if not isinstance(summary, dict):
            raise MaterializationError("geometry summary entry is not an object")
        value = raw.get(geometry)
        if kind == "path":
            actual_path_count = len(value) if isinstance(value, list) else 0
            actual_point_count = (
                sum(len(path) for path in value if isinstance(path, list))
                if isinstance(value, list)
                else 0
            )
            if summary.get("path_count") != actual_path_count or summary.get(
                "point_count"
            ) != actual_point_count:
                raise MaterializationError(
                    f"wire record {row['record_id']!r} keyPointPaths summary mismatch"
                )
        else:
            actual = len(value) if isinstance(value, list) else 0
            if summary.get("point_count") != actual:
                raise MaterializationError(
                    f"wire record {row['record_id']!r} skeleton summary mismatch"
                )


def _write_wire_tiles(
    *,
    staging: Path,
    sample_id: str,
    split_name: str,
    source_row: dict[str, Any],
    json_sha: str,
    image_sha: str,
    canvas: dict[str, int],
    geometries: list[_EdgeGeometry],
    unusable_sparse_record_ids: list[str],
    ignore_regions: list[dict[str, Any]],
    run_id: str,
    cfg: WireJunctionMaterializeConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []
    for y1 in range(0, canvas["height"], cfg.tile_size):
        y2 = min(canvas["height"], y1 + cfg.tile_size)
        for x1 in range(0, canvas["width"], cfg.tile_size):
            x2 = min(canvas["width"], x1 + cfg.tile_size)
            tile_bbox = (x1, y1, x2, y2)
            width = x2 - x1
            height = y2 - y1
            target = Image.new("L", (width, height), cfg.mask_background)
            target_draw = ImageDraw.Draw(target)
            edge_record_ids: list[str] = []
            for geometry in geometries:
                if geometry.bbox is None or not _bbox_intersects(geometry.bbox, tile_bbox):
                    continue
                used = False
                for path in geometry.paths:
                    if not _bbox_intersects(_points_bbox(path), tile_bbox):
                        continue
                    local = [(_pixel_x(px) - x1, _pixel_y(py) - y1) for px, py in path]
                    if len(local) == 1:
                        target_draw.point(local[0], fill=cfg.mask_foreground)
                    else:
                        target_draw.line(local, fill=cfg.mask_foreground, width=1)
                    used = True
                for px, py in geometry.skeleton:
                    ix, iy = _pixel_x(px), _pixel_y(py)
                    if x1 <= ix < x2 and y1 <= iy < y2:
                        target_draw.point((ix - x1, iy - y1), fill=cfg.mask_foreground)
                        used = True
                if used:
                    edge_record_ids.append(geometry.record_id)

            ignore = Image.new("L", (width, height), cfg.mask_background)
            ignore_draw = ImageDraw.Draw(ignore)
            tile_regions: list[dict[str, Any]] = []
            if unusable_sparse_record_ids:
                ignore_draw.rectangle(
                    (0, 0, width - 1, height - 1), fill=cfg.mask_foreground
                )
                target_draw.rectangle(
                    (0, 0, width - 1, height - 1), fill=cfg.mask_background
                )
                tile_regions.append(
                    {
                        "kind": "sample_unusable_sparse_geometry",
                        "source_record_ids": sorted(unusable_sparse_record_ids),
                        "image_bbox": [x1, y1, x2, y2],
                        "tile_bbox": [0, 0, width, height],
                    }
                )
            else:
                for region in ignore_regions:
                    clipped = _draw_ignore_region(
                        region, ignore_draw, target_draw, tile_bbox, cfg
                    )
                    if clipped is not None:
                        tile_regions.append(clipped)
                if cfg.pin_line_policy == "positive":
                    for region in ignore_regions:
                        if region["kind"] != "pin_line":
                            continue
                        _draw_positive_pin(region, target_draw, tile_bbox, cfg)

            # Ignore always wins when the positive target and an exclusion overlap.
            target.paste(cfg.mask_background, mask=ignore)
            tile_id = f"y{y1:06d}_x{x1:06d}"
            target_rel = Path("wire") / "target_masks" / sample_id / f"{tile_id}.png"
            ignore_rel = Path("wire") / "ignore_masks" / sample_id / f"{tile_id}.png"
            target_path = staging / target_rel
            ignore_path = staging / ignore_rel
            target_path.parent.mkdir(parents=True, exist_ok=True)
            ignore_path.parent.mkdir(parents=True, exist_ok=True)
            _save_png(target, target_path)
            _save_png(ignore, ignore_path)
            target_sha = _sha256_file(target_path)
            ignore_sha = _sha256_file(ignore_path)
            for rel, path, sha, kind in (
                (target_rel, target_path, target_sha, "wire_target_mask"),
                (ignore_rel, ignore_path, ignore_sha, "wire_ignore_mask"),
            ):
                assets.append(
                    {
                        "path": rel.as_posix(),
                        "sha256": sha,
                        "bytes": path.stat().st_size,
                        "kind": kind,
                    }
                )
            files = source_row["files"]
            records.append(
                {
                    "schema_version": 1,
                    "record_id": f"{sample_id}:wire_tile:{tile_id}",
                    "sample_id": sample_id,
                    "split": split_name,
                    "tile_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "tile_size": [width, height],
                    "target_mask": {
                        "path": target_rel.as_posix(),
                        "sha256": target_sha,
                        "foreground_pixels": _mask_pixel_count(target, cfg.mask_foreground),
                    },
                    "ignore_mask": {
                        "path": ignore_rel.as_posix(),
                        "sha256": ignore_sha,
                        "foreground_pixels": _mask_pixel_count(ignore, cfg.mask_foreground),
                        "regions": tile_regions,
                    },
                    "transform": {
                        "coordinate_convention": "pixel_xy_half_open_bbox",
                        "tile_to_image": [
                            [1.0, 0.0, float(x1)],
                            [0.0, 1.0, float(y1)],
                            [0.0, 0.0, 1.0],
                        ],
                        "image_to_tile": [
                            [1.0, 0.0, float(-x1)],
                            [0.0, 1.0, float(-y1)],
                            [0.0, 0.0, 1.0],
                        ],
                    },
                    "source": {
                        "image": files["image"]["path"],
                        "image_sha256": image_sha,
                        "json": files["json"]["path"],
                        "json_sha256": json_sha,
                        "image_crop_included": False,
                        "derived_wire_record_ids": sorted(edge_record_ids),
                    },
                    "supervision": {
                        "eligible": not unusable_sparse_record_ids,
                        "ineligible_reason": (
                            "one_or_more_wire_records_lack_keyPointPaths_and_skeleton"
                            if unusable_sparse_record_ids
                            else None
                        ),
                        "unusable_sparse_record_ids": sorted(
                            unusable_sparse_record_ids
                        ),
                    },
                    "lineage": {
                        "run_id": run_id,
                        "geometry_sources": ["keyPointPaths", "keyPoints/skeleton"],
                        "dense_coordinates_loaded": False,
                    },
                }
            )
    return records, assets


def _collect_ignore_regions(
    annotation: dict[str, Any],
    canvas: dict[str, int],
    cfg: WireJunctionMaterializeConfig,
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for field, kind in (("components", "component_bbox"), ("texts", "text_bbox")):
        values = annotation.get(field)
        if not isinstance(values, dict):
            raise MaterializationError(f"raw annotation {field!r} is not an object")
        for key, value in sorted(values.items()):
            if not isinstance(value, dict) or value.get("state") not in (None, "active"):
                continue
            bbox = _bbox_from_value(value.get("location"), cfg.ignore_bbox_margin)
            if bbox is None:
                continue
            clipped = _clip_bbox_to_canvas(bbox, canvas)
            if clipped is None:
                continue
            regions.append(
                {"kind": kind, "source_id": str(key), "bbox": list(clipped)}
            )
    pins = annotation.get("pins")
    if not isinstance(pins, dict):
        raise MaterializationError("raw annotation 'pins' is not an object")
    for key, value in sorted(pins.items()):
        if not isinstance(value, dict):
            continue
        tip = _point(value.get("tip"))
        base = _point(value.get("base"))
        if tip is None or base is None:
            continue
        regions.append(
            {
                "kind": "pin_line",
                "source_id": str(key),
                "points": [list(tip), list(base)],
                "policy": cfg.pin_line_policy,
            }
        )
    regions.sort(key=lambda value: _canonical_text(value))
    return regions


def _draw_ignore_region(
    region: dict[str, Any],
    ignore_draw: ImageDraw.ImageDraw,
    target_draw: ImageDraw.ImageDraw,
    tile_bbox: tuple[int, int, int, int],
    cfg: WireJunctionMaterializeConfig,
) -> dict[str, Any] | None:
    x1, y1, x2, y2 = tile_bbox
    if region["kind"] in {"component_bbox", "text_bbox"}:
        rx1, ry1, rx2, ry2 = region["bbox"]
        clipped = (
            max(x1, _pixel_floor(rx1)),
            max(y1, _pixel_floor(ry1)),
            min(x2, _pixel_ceil(rx2)),
            min(y2, _pixel_ceil(ry2)),
        )
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            return None
        local = (
            clipped[0] - x1,
            clipped[1] - y1,
            clipped[2] - x1 - 1,
            clipped[3] - y1 - 1,
        )
        ignore_draw.rectangle(local, fill=cfg.mask_foreground)
        target_draw.rectangle(local, fill=cfg.mask_background)
        return {
            "kind": region["kind"],
            "source_id": region["source_id"],
            "image_bbox": list(clipped),
            "tile_bbox": [local[0], local[1], local[2] + 1, local[3] + 1],
        }
    if region["kind"] == "pin_line" and cfg.pin_line_policy == "ignore":
        points = [
            (_pixel_x(point[0]) - x1, _pixel_y(point[1]) - y1)
            for point in region["points"]
        ]
        line_bbox = _points_bbox([tuple(point) for point in region["points"]])
        if not _bbox_intersects_expanded(line_bbox, tile_bbox, cfg.pin_ignore_width):
            return None
        ignore_draw.line(points, fill=cfg.mask_foreground, width=cfg.pin_ignore_width)
        target_draw.line(points, fill=cfg.mask_background, width=cfg.pin_ignore_width)
        return {
            "kind": "pin_line",
            "source_id": region["source_id"],
            "policy": "ignore",
            "width": cfg.pin_ignore_width,
            "image_points": region["points"],
            "tile_points": [list(point) for point in points],
        }
    return None


def _draw_positive_pin(
    region: dict[str, Any],
    target_draw: ImageDraw.ImageDraw,
    tile_bbox: tuple[int, int, int, int],
    cfg: WireJunctionMaterializeConfig,
) -> None:
    line_bbox = _points_bbox([tuple(point) for point in region["points"]])
    if not _bbox_intersects_expanded(line_bbox, tile_bbox, 1):
        return
    x1, y1, _, _ = tile_bbox
    points = [
        (_pixel_x(point[0]) - x1, _pixel_y(point[1]) - y1)
        for point in region["points"]
    ]
    target_draw.line(points, fill=cfg.mask_foreground, width=1)


def _safe_junction_label(
    row: dict[str, Any],
) -> tuple[str | None, str | None, list[str]]:
    raw_types = row.get("candidate_types")
    if not isinstance(raw_types, list) or any(
        not isinstance(value, str) for value in raw_types
    ):
        raise MaterializationError(
            f"junction record {row['record_id']!r} has invalid candidate_types"
        )
    types = sorted(set(raw_types))
    unknown_types = sorted(set(types) - ({"cross"} | set(_SAFE_JUNCTION_MAP)))
    if unknown_types:
        return None, f"unsupported_candidate_types:{','.join(unknown_types)}", list(
            JUNCTION_CLASSES
        )
    if "cross" in types:
        return (
            None,
            "cross_connectivity_not_encoded_by_gt",
            ["connected_cross", "non_connected_cross", "noise"],
        )
    mapped = sorted({_SAFE_JUNCTION_MAP[value] for value in types})
    if len(mapped) != 1:
        return None, "multiple_or_missing_trusted_gt_types", list(JUNCTION_CLASSES)
    return mapped[0], None, []


def _validate_junction_sources(row: dict[str, Any], annotation: dict[str, Any]) -> None:
    center = _point(row.get("center"))
    if center is None:
        raise MaterializationError(
            f"junction record {row['record_id']!r} has invalid center"
        )
    refs = row.get("source_refs")
    if not isinstance(refs, list) or not refs:
        raise MaterializationError(
            f"junction record {row['record_id']!r} has no source_refs"
        )
    for source_ref in refs:
        if not isinstance(source_ref, dict):
            raise MaterializationError("junction source_ref is not an object")
        value = _resolve_json_pointer(annotation, source_ref.get("json_pointer"))
        if source_ref.get("kind") == "cross":
            if not isinstance(value, dict) or not any(
                _contains_point(points, center) for points in value.values()
            ):
                raise MaterializationError(
                    f"junction record {row['record_id']!r} cross center is absent from skeleton"
                )
        else:
            point = _point(value)
            if point != center:
                raise MaterializationError(
                    f"junction record {row['record_id']!r} source point differs from center"
                )


def _junction_patch(
    source_image: Image.Image, row: dict[str, Any]
) -> tuple[Image.Image, dict[str, Any]]:
    patch_size = row.get("patch_size")
    if type(patch_size) is not int or patch_size <= 0:
        raise MaterializationError(
            f"junction record {row['record_id']!r} has invalid patch_size"
        )
    raw = _integral_bbox(row.get("raw_patch_bbox"), "raw_patch_bbox", row)
    clipped = _integral_bbox(row.get("patch_bbox"), "patch_bbox", row)
    padding = row.get("padding")
    if not isinstance(padding, dict) or any(
        type(padding.get(name)) is not int
        for name in ("left", "top", "right", "bottom")
    ):
        raise MaterializationError(
            f"junction record {row['record_id']!r} has invalid padding"
        )
    if raw[2] - raw[0] != patch_size or raw[3] - raw[1] != patch_size:
        raise MaterializationError(
            f"junction record {row['record_id']!r} raw patch size mismatch"
        )
    expected_clipped = (
        max(0, raw[0]),
        max(0, raw[1]),
        min(source_image.width, raw[2]),
        min(source_image.height, raw[3]),
    )
    if clipped != expected_clipped:
        raise MaterializationError(
            f"junction record {row['record_id']!r} clipped patch bbox mismatch"
        )
    expected_padding = {
        "left": clipped[0] - raw[0],
        "top": clipped[1] - raw[1],
        "right": raw[2] - clipped[2],
        "bottom": raw[3] - clipped[3],
    }
    if padding != expected_padding:
        raise MaterializationError(
            f"junction record {row['record_id']!r} padding mismatch"
        )
    patch = Image.new("RGB", (patch_size, patch_size), (255, 255, 255))
    if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
        crop = source_image.crop(clipped)
        patch.paste(crop, (padding["left"], padding["top"]))
    transform = {
        "coordinate_convention": "pixel_xy_half_open_bbox",
        "patch_to_image": [
            [1.0, 0.0, float(raw[0])],
            [0.0, 1.0, float(raw[1])],
            [0.0, 0.0, 1.0],
        ],
        "image_to_patch": [
            [1.0, 0.0, float(-raw[0])],
            [0.0, 1.0, float(-raw[1])],
            [0.0, 0.0, 1.0],
        ],
        "raw_patch_bbox": {"x1": raw[0], "y1": raw[1], "x2": raw[2], "y2": raw[3]},
        "clipped_source_bbox": {
            "x1": clipped[0],
            "y1": clipped[1],
            "x2": clipped[2],
            "y2": clipped[3],
        },
        "padding": padding,
    }
    return patch, transform


def _junction_label_schema() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "classes": [
            {"id": index, "name": name} for index, name in enumerate(JUNCTION_CLASSES)
        ],
        "unknown": {"id": -1, "name": "UNKNOWN", "trainable": False},
        "trusted_gt_mapping": {
            "endpoint": "endpoint",
            "diverge": "t_junction",
            "turn": "bend",
        },
        "cross_policy": {
            "automatic_label": "UNKNOWN",
            "needs_review": True,
            "reason": "skeleton overlap does not encode electrical connectivity",
            "never_auto_map_to": ["connected_cross", "non_connected_cross"],
        },
        "training_gate": {
            "requires_at_least_one_trusted_train_example_per_class": True,
            "requires_zero_pending_UNKNOWN": True,
        },
    }


def _write_review_template(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "record_id",
                "sample_id",
                "split",
                "patch_path",
                "source_image",
                "center_x",
                "center_y",
                "candidate_types",
                "suggested_labels",
                "review_status",
                "reviewed_label",
                "reviewer",
                "notes",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            center = record.get("center") or [None, None]
            writer.writerow(
                {
                    "record_id": record["record_id"],
                    "sample_id": record["sample_id"],
                    "split": record["split"],
                    "patch_path": record["patch_path"],
                    "source_image": record["source_image"],
                    "center_x": center[0],
                    "center_y": center[1],
                    "candidate_types": "|".join(record.get("candidate_types", [])),
                    "suggested_labels": "|".join(record.get("suggested_labels", [])),
                    "review_status": "pending",
                    "reviewed_label": "",
                    "reviewer": "",
                    "notes": "",
                }
            )


def _load_annotation_without_dense_coordinates(
    path: Path,
) -> tuple[dict[str, Any], dict[str, int]]:
    source_text = path.read_text(encoding="utf-8")
    filtered, stats = _strip_json_object_values(source_text, {"coordinates"})
    try:
        value = json.loads(filtered, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise MaterializationError(f"invalid source JSON after dense-field filter: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MaterializationError(f"source JSON root is not an object: {path}")
    return value, stats


def _strip_json_object_values(
    source: str, keys: set[str]
) -> tuple[str, dict[str, int]]:
    """Replace selected object values with null without decoding those values."""

    output: list[str] = []
    cursor = 0
    index = 0
    fields = 0
    skipped_bytes = 0
    length = len(source)
    while index < length:
        if source[index] != '"':
            index += 1
            continue
        string_end = _scan_json_string(source, index)
        after = _skip_whitespace(source, string_end)
        if after >= length or source[after] != ":":
            index = string_end
            continue
        try:
            key = json.loads(source[index:string_end])
        except json.JSONDecodeError as exc:
            raise MaterializationError(f"invalid JSON string while filtering: {exc}") from exc
        value_start = _skip_whitespace(source, after + 1)
        if key not in keys:
            index = string_end
            continue
        value_end = _scan_json_value(source, value_start)
        output.append(source[cursor:value_start])
        output.append("null")
        cursor = value_end
        index = value_end
        fields += 1
        skipped_bytes += value_end - value_start
    output.append(source[cursor:])
    return "".join(output), {"fields": fields, "bytes": skipped_bytes}


def _scan_json_string(source: str, start: int) -> int:
    index = start + 1
    escaped = False
    while index < len(source):
        char = source[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return index + 1
        index += 1
    raise MaterializationError("unterminated JSON string")


def _scan_json_value(source: str, start: int) -> int:
    if start >= len(source):
        raise MaterializationError("missing JSON value")
    first = source[start]
    if first == '"':
        return _scan_json_string(source, start)
    if first in "[{":
        stack = [first]
        index = start + 1
        while index < len(source) and stack:
            char = source[index]
            if char == '"':
                index = _scan_json_string(source, index)
                continue
            if char in "[{":
                stack.append(char)
            elif char in "]}":
                expected = "[" if char == "]" else "{"
                if not stack or stack[-1] != expected:
                    raise MaterializationError("unbalanced JSON while filtering dense values")
                stack.pop()
            index += 1
        if stack:
            raise MaterializationError("unterminated JSON value while filtering")
        return index
    index = start
    while index < len(source) and source[index] not in ",]}\r\n\t ":
        index += 1
    if index == start:
        raise MaterializationError("empty JSON value")
    return index


def _skip_whitespace(source: str, index: int) -> int:
    while index < len(source) and source[index] in " \t\r\n":
        index += 1
    return index


def _resolve_json_pointer(root: Any, pointer: Any) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise MaterializationError(f"invalid JSON Pointer {pointer!r}")
    value = root
    for raw_token in pointer.split("/")[1:]:
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            if token not in value:
                raise MaterializationError(f"JSON Pointer {pointer!r} has missing token {token!r}")
            value = value[token]
        elif isinstance(value, list):
            try:
                index = int(token)
            except ValueError as exc:
                raise MaterializationError(
                    f"JSON Pointer {pointer!r} has non-integer list token {token!r}"
                ) from exc
            if index < 0 or index >= len(value):
                raise MaterializationError(f"JSON Pointer {pointer!r} index is out of range")
            value = value[index]
        else:
            raise MaterializationError(f"JSON Pointer {pointer!r} traverses a scalar")
    return value


def _extract_paths(
    value: Any, geometry_types: Sequence[str], record_id: str
) -> list[list[tuple[float, float]]]:
    if not isinstance(value, dict):
        raise MaterializationError(f"wire record {record_id!r} keyPointPaths is not an object")
    output: list[list[tuple[float, float]]] = []
    for geometry in geometry_types:
        raw_paths = value.get(geometry, [])
        if not isinstance(raw_paths, list):
            raise MaterializationError(
                f"wire record {record_id!r} path geometry {geometry!r} is not a list"
            )
        for raw_path in raw_paths:
            if not isinstance(raw_path, list):
                raise MaterializationError(f"wire record {record_id!r} contains a non-list path")
            path: list[tuple[float, float]] = []
            for raw_point in raw_path:
                point = _point(raw_point)
                if point is None:
                    raise MaterializationError(
                        f"wire record {record_id!r} contains an invalid path point"
                    )
                path.append(point)
            if path:
                output.append(path)
    return output


def _extract_points(
    value: Any, geometry_types: Sequence[str], record_id: str
) -> list[tuple[float, float]]:
    if not isinstance(value, dict):
        raise MaterializationError(f"wire record {record_id!r} skeleton is not an object")
    output: list[tuple[float, float]] = []
    for geometry in geometry_types:
        raw_points = value.get(geometry, [])
        if not isinstance(raw_points, list):
            raise MaterializationError(
                f"wire record {record_id!r} skeleton geometry {geometry!r} is not a list"
            )
        for raw_point in raw_points:
            point = _point(raw_point)
            if point is None:
                raise MaterializationError(
                    f"wire record {record_id!r} contains an invalid skeleton point"
                )
            output.append(point)
    return output


def _validate_canvas(
    annotation: dict[str, Any], source_row: dict[str, Any], sample_id: str
) -> dict[str, int]:
    raw = annotation.get("canvas")
    manifest = source_row.get("canvas")
    if not isinstance(raw, dict) or not isinstance(manifest, dict):
        raise MaterializationError(f"sample {sample_id!r} lacks canvas metadata")
    width = raw.get("width")
    height = raw.get("height")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise MaterializationError(f"sample {sample_id!r} has invalid raw canvas")
    if manifest.get("width") != width or manifest.get("height") != height:
        raise MaterializationError(
            f"sample {sample_id!r} raw canvas differs from source manifest"
        )
    return {"width": width, "height": height}


def _verify_source_hash(
    path: Path, metadata: dict[str, Any], sample_id: str, kind: str
) -> str:
    if not path.is_file():
        raise MaterializationError(
            f"sample {sample_id!r} {kind} source file does not exist: {path}"
        )
    actual = _sha256_file(path)
    expected = metadata.get("sha256")
    if not isinstance(expected, str) or actual != expected:
        raise MaterializationError(
            f"sample {sample_id!r} {kind} SHA-256 mismatch: expected {expected!r}, actual {actual}"
        )
    expected_bytes = metadata.get("bytes")
    if type(expected_bytes) is int and path.stat().st_size != expected_bytes:
        raise MaterializationError(
            f"sample {sample_id!r} {kind} byte count differs from source manifest"
        )
    return actual


def _resolve_source_file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise MaterializationError("source manifest contains an invalid relative path")
    child = (root / Path(relative)).resolve(strict=False)
    try:
        child.relative_to(root)
    except ValueError as exc:
        raise MaterializationError(f"source path escapes source root: {relative!r}") from exc
    return child


def _read_jsonl_unique(path: Path, key: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line, parse_constant=_reject_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise MaterializationError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise MaterializationError(f"JSONL row at {path}:{line_number} is not an object")
            identity = value.get(key)
            if not isinstance(identity, str) or not identity:
                raise MaterializationError(
                    f"JSONL row at {path}:{line_number} lacks string {key!r}"
                )
            if identity in seen:
                raise MaterializationError(f"duplicate {key} {identity!r} in {path}")
            seen.add(identity)
            records.append(value)
    return records


def _point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    if not _finite_number(value[0]) or not _finite_number(value[1]):
        return None
    return float(value[0]), float(value[1])


def _contains_point(values: Any, expected: tuple[float, float]) -> bool:
    if not isinstance(values, list):
        return False
    return any(_point(value) == expected for value in values)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _points_bbox(
    points: Sequence[tuple[float, float]],
) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs) + 1.0, max(ys) + 1.0


def _bbox_from_value(value: Any, margin: int) -> tuple[float, float, float, float] | None:
    if not isinstance(value, dict):
        return None
    values = [value.get(key) for key in ("x1", "y1", "x2", "y2")]
    if not all(_finite_number(item) for item in values):
        return None
    x1, y1, x2, y2 = (float(item) for item in values)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1 - margin, y1 - margin, x2 + margin, y2 + margin


def _clip_bbox_to_canvas(
    bbox: tuple[float, float, float, float], canvas: dict[str, int]
) -> tuple[float, float, float, float] | None:
    clipped = (
        max(0.0, bbox[0]),
        max(0.0, bbox[1]),
        min(float(canvas["width"]), bbox[2]),
        min(float(canvas["height"]), bbox[3]),
    )
    return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else None


def _bbox_intersects(
    bbox: tuple[float, float, float, float] | None,
    tile: tuple[int, int, int, int],
) -> bool:
    if bbox is None:
        return False
    return bbox[0] < tile[2] and bbox[2] > tile[0] and bbox[1] < tile[3] and bbox[3] > tile[1]


def _bbox_intersects_expanded(
    bbox: tuple[float, float, float, float] | None,
    tile: tuple[int, int, int, int],
    amount: int,
) -> bool:
    if bbox is None:
        return False
    expanded = (bbox[0] - amount, bbox[1] - amount, bbox[2] + amount, bbox[3] + amount)
    return _bbox_intersects(expanded, tile)


def _integral_bbox(
    value: Any, field: str, row: dict[str, Any]
) -> tuple[int, int, int, int]:
    if not isinstance(value, dict):
        raise MaterializationError(f"junction record {row['record_id']!r} lacks {field}")
    output: list[int] = []
    for key in ("x1", "y1", "x2", "y2"):
        raw = value.get(key)
        if not _finite_number(raw) or not float(raw).is_integer():
            raise MaterializationError(
                f"junction record {row['record_id']!r} {field}.{key} is not integral"
            )
        output.append(int(raw))
    return output[0], output[1], output[2], output[3]


def _pixel_x(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def _pixel_y(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def _pixel_floor(value: float) -> int:
    return int(math.floor(float(value)))


def _pixel_ceil(value: float) -> int:
    return int(math.ceil(float(value)))


def _junction_patch_relative_path(row: dict[str, Any]) -> Path:
    digest = _sha256_bytes(row["record_id"].encode("utf-8"))[:24]
    return Path("junction") / "patches" / row["sample_id"] / f"{digest}.png"


def _mask_pixel_count(image: Image.Image, foreground: int) -> int:
    return int(image.histogram()[foreground])


def _save_png(image: Image.Image, path: Path) -> None:
    image.save(path, format="PNG", compress_level=9)


def _asset_tree_hash(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry["bytes"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _existing_result(final_dir: Path, run_id: str) -> MaterializationResult | None:
    summary_path = final_dir / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(summary, dict) or summary.get("run_id") != run_id or summary.get("complete") is not True:
        return None
    return MaterializationResult(run_id, final_dir, summary_path, summary)


def _publish_latest(output_root: Path, result: MaterializationResult) -> None:
    pointer = {
        "schema_version": 1,
        "run_id": result.run_id,
        "run_dir": result.run_dir.relative_to(output_root).as_posix(),
        "summary": result.summary_path.relative_to(output_root).as_posix(),
        "summary_sha256": _sha256_file(result.summary_path),
        "training_ready": result.training_ready,
    }
    _atomic_write(output_root / "latest.json", _pretty_json(pointer))


def _write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    content = "".join(_canonical_text(record) + "\n" for record in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_pretty_json(value), encoding="utf-8", newline="\n")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _canonical_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_bytes(value: Any) -> bytes:
    return _canonical_text(value).encode("utf-8")


def _pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize Dataset300-v2 wire masks and junction review patches"
    )
    parser.add_argument("--wire-manifest", required=True)
    parser.add_argument("--junction-manifest", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument(
        "--pin-line-policy", choices=("ignore", "positive", "negative"), default="ignore"
    )
    parser.add_argument("--pin-ignore-width", type=int, default=3)
    parser.add_argument("--ignore-bbox-margin", type=int, default=0)
    parser.add_argument("--expected-derivation-version")
    parser.add_argument("--expected-config-hash")
    parser.add_argument("--review-override", help="junction_review_progress.csv with human labels")
    parser.add_argument(
        "--require-train-ready",
        action="store_true",
        help="return exit code 2 when six-class junction training remains blocked",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        result = materialize_wire_junction(
            wire_manifest=args.wire_manifest,
            junction_manifest=args.junction_manifest,
            source_manifest=args.source_manifest,
            split_manifest=args.split_manifest,
            source_root=args.source_root,
            output_root=args.output_root,
            config=WireJunctionMaterializeConfig(
                tile_size=args.tile_size,
                pin_line_policy=args.pin_line_policy,
                pin_ignore_width=args.pin_ignore_width,
                ignore_bbox_margin=args.ignore_bbox_margin,
            ),
            sample_ids=args.sample_ids,
            expected_derivation_version=args.expected_derivation_version,
            expected_config_hash=args.expected_config_hash,
            review_overrides=(
                _load_review_overrides(args.review_override)
                if args.review_override
                else None
            ),
        )
    except (MaterializationError, OSError, ValueError) as exc:
        print(_canonical_text({"ok": False, "error": str(exc)}))
        return 1
    print(
        _canonical_text(
            {
                "ok": True,
                "run_id": result.run_id,
                "run_dir": str(result.run_dir),
                "summary": str(result.summary_path),
                "training_ready": result.training_ready,
                "training_blockers": result.summary["training_gate"]["blockers"],
            }
        )
    )
    if args.require_train_ready and not result.training_ready:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main(argv)
    raise SystemExit(main())
