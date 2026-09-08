from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


VALID_SPLITS = ("train", "val", "test")


class OCRMaterializationError(ValueError):
    """Raised when OCR training data cannot be materialized without leakage."""


@dataclass(frozen=True, slots=True)
class OCRMaterializationOptions:
    """Deterministic policies used to turn ``ocr_text`` into Paddle labels."""

    padding: int = 2
    max_text_length: int = 64
    tile_size: int = 1024
    tile_overlap: int = 160
    sample_limit: int | None = None
    materialize_images: bool = True
    verify_all_source_hashes: bool = True

    def validate(self) -> None:
        if type(self.padding) is not int or self.padding < 0:
            raise OCRMaterializationError("padding must be a non-negative integer")
        if type(self.max_text_length) is not int or self.max_text_length <= 0:
            raise OCRMaterializationError("max_text_length must be a positive integer")
        if type(self.tile_size) is not int or self.tile_size <= 0:
            raise OCRMaterializationError("tile_size must be a positive integer")
        if (
            type(self.tile_overlap) is not int
            or self.tile_overlap < 0
            or self.tile_overlap >= self.tile_size
        ):
            raise OCRMaterializationError(
                "tile_overlap must satisfy 0 <= tile_overlap < tile_size"
            )
        if self.sample_limit is not None and (
            type(self.sample_limit) is not int or self.sample_limit <= 0
        ):
            raise OCRMaterializationError("sample_limit must be a positive integer")
        if type(self.materialize_images) is not bool:
            raise OCRMaterializationError("materialize_images must be boolean")
        if type(self.verify_all_source_hashes) is not bool:
            raise OCRMaterializationError("verify_all_source_hashes must be boolean")


def materialize_ocr_training_data(
    *,
    ocr_manifest: str | Path,
    derived_summary: str | Path,
    source_root: str | Path,
    source_manifest: str | Path,
    split_manifest: str | Path,
    output_root: str | Path,
    options: OCRMaterializationOptions | None = None,
) -> dict[str, Any]:
    """Build deterministic recognition crops and 1024px detection tiles.

    The source text is never stripped, normalized, or truncated. Records that
    cannot be represented safely in a line-oriented Paddle recognition label
    (empty/non-string content, TAB/CR/LF, or more than ``max_text_length``
    code points) are closed out as explicit exclusions in the index and report.
    Detection boxes for unsafe text are emitted as ``###`` ignore regions.

    JSON/label files and PNGs are replaced atomically. ``manifest.json`` is
    written last and is the commit marker for a complete output generation.
    When ``materialize_images`` is false, the same indices and labels are
    generated, but image paths are plans rather than existing artifacts.
    """

    policy = options or OCRMaterializationOptions()
    policy.validate()
    ocr_path = Path(ocr_manifest).expanduser().resolve(strict=True)
    summary_path = Path(derived_summary).expanduser().resolve(strict=True)
    source_base = Path(source_root).expanduser().resolve(strict=True)
    source_manifest_path = Path(source_manifest).expanduser().resolve(strict=True)
    split_manifest_path = Path(split_manifest).expanduser().resolve(strict=True)
    destination = Path(output_root).expanduser().resolve(strict=False)

    summary = _read_json_object(summary_path)
    ocr_rows = _read_jsonl(ocr_path, identifier="record_id")
    source_rows = _read_jsonl(source_manifest_path, identifier="sample_id")
    split_rows = _read_jsonl(split_manifest_path, identifier="sample_id")
    lineage = _validate_lineage(
        summary=summary,
        summary_path=summary_path,
        ocr_rows=ocr_rows,
        source_rows=source_rows,
        split_rows=split_rows,
        ocr_path=ocr_path,
        source_base=source_base,
        source_manifest_path=source_manifest_path,
        split_manifest_path=split_manifest_path,
    )

    source_by_id = {row["sample_id"]: row for row in source_rows}
    split_by_id = {row["sample_id"]: row for row in split_rows}
    texts_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ocr_rows:
        texts_by_sample[row["sample_id"]].append(row)
    for rows in texts_by_sample.values():
        rows.sort(key=lambda row: row["record_id"])

    all_sample_ids = sorted(source_by_id)
    selected_ids = (
        all_sample_ids[: policy.sample_limit]
        if policy.sample_limit is not None
        else all_sample_ids
    )
    actual_source_hashes: dict[str, str] = {}
    ids_to_hash = all_sample_ids if policy.verify_all_source_hashes else selected_ids
    for sample_id in ids_to_hash:
        source = source_by_id[sample_id]
        image_entry = _image_entry(source, sample_id)
        image_path = _safe_source_path(source_base, image_entry["path"])
        actual_hash = _sha256_file(image_path)
        if actual_hash != image_entry["sha256"]:
            raise OCRMaterializationError(
                f"source image SHA-256 mismatch for sample {sample_id}: "
                f"expected {image_entry['sha256']}, got {actual_hash}"
            )
        actual_source_hashes[sample_id] = actual_hash
    lineage["source_images_sha256_verified"] = len(ids_to_hash)
    lineage["source_image_verification_scope"] = (
        "all_source_manifest_samples"
        if policy.verify_all_source_hashes
        else "selected_samples_only"
    )

    recognition_index: list[dict[str, Any]] = []
    detection_index: list[dict[str, Any]] = []
    rec_label_rows: dict[str, list[tuple[str, str]]] = {
        split: [] for split in VALID_SPLITS
    }
    det_label_rows: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {
        split: [] for split in VALID_SPLITS
    }
    image_hashes: list[tuple[str, str]] = []
    excluded: list[dict[str, Any]] = []
    oversize: list[dict[str, Any]] = []
    eligible_characters: dict[str, set[str]] = {
        split: set() for split in VALID_SPLITS
    }
    length_counts: Counter[int] = Counter()
    rec_strata: dict[str, Counter[str]] = {
        "split": Counter(),
        "source_domain": Counter(),
        "category": Counter(),
        "details": Counter(),
        "vertical_long": Counter(),
    }
    det_strata: dict[str, Counter[str]] = {
        "split": Counter(),
        "source_domain": Counter(),
        "empty_tile": Counter(),
    }

    destination.mkdir(parents=True, exist_ok=True)
    # Invalidate a previous commit marker before replacing any member file.
    # Readers must only consume a generation whose final manifest says true.
    _atomic_json(
        destination / "manifest.json",
        {
            "schema_version": 1,
            "kind": "pcbparse_ocr_paddle_materialization",
            "complete": False,
        },
    )
    for sample_id in selected_ids:
        source = source_by_id[sample_id]
        split = split_by_id[sample_id]["split"]
        image_entry = _image_entry(source, sample_id)
        image_path = _safe_source_path(source_base, image_entry["path"])
        if sample_id not in actual_source_hashes:
            actual_hash = _sha256_file(image_path)
            if actual_hash != image_entry["sha256"]:
                raise OCRMaterializationError(
                    f"source image SHA-256 mismatch for sample {sample_id}"
                )
            actual_source_hashes[sample_id] = actual_hash
        image, actual_format = _decode_image_by_magic(image_path)
        width, height = image.size
        _validate_decoded_source(
            sample_id=sample_id,
            source=source,
            image_entry=image_entry,
            width=width,
            height=height,
            actual_format=actual_format,
        )
        rows = texts_by_sample.get(sample_id, [])

        for row in rows:
            bbox = _bbox(row, width=width, height=height)
            content = row.get("content")
            reason = _recognition_exclusion_reason(
                row, content, max_text_length=policy.max_text_length
            )
            text_length = len(content) if isinstance(content, str) else None
            if text_length is not None:
                length_counts[text_length] += 1
            vertical_long = _bbox_height(bbox) > _bbox_width(bbox)
            crop_box = _padded_crop_box(
                bbox, width=width, height=height, padding=policy.padding
            )
            output_relative = _recognition_image_path(split, row["record_id"])
            base_index = {
                "schema_version": 1,
                "record_id": row["record_id"],
                "sample_id": sample_id,
                "split": split,
                "source_domain": row.get("source_domain"),
                "source_image": image_entry["path"],
                "source_image_sha256": actual_source_hashes[sample_id],
                "source_format": actual_format,
                "bbox": bbox,
                "crop_box": crop_box,
                "padding": policy.padding,
                "transform": {
                    "source_to_crop": [
                        1,
                        0,
                        -crop_box["left"],
                        0,
                        1,
                        -crop_box["top"],
                    ],
                    "crop_to_source": [
                        1,
                        0,
                        crop_box["left"],
                        0,
                        1,
                        crop_box["top"],
                    ],
                },
                "content": content,
                "content_sha256": _text_sha256(content),
                "text_length_codepoints": text_length,
                "vertical_long": vertical_long,
                "normalization_applied": False,
                "output_image": output_relative,
            }
            base_index["spec_sha256"] = _canonical_sha256(base_index)
            if reason is not None:
                base_index.update(
                    {
                        "status": "excluded",
                        "exclusion_reason": reason,
                        "output_image_sha256": None,
                    }
                )
                excluded.append(
                    {
                        "record_id": row["record_id"],
                        "sample_id": sample_id,
                        "split": split,
                        "reason": reason,
                        "text_length_codepoints": text_length,
                    }
                )
                recognition_index.append(base_index)
                continue

            assert isinstance(content, str)
            output_hash: str | None = None
            if policy.materialize_images:
                crop = image.crop(
                    (
                        crop_box["left"],
                        crop_box["top"],
                        crop_box["right"],
                        crop_box["bottom"],
                    )
                ).convert("RGB")
                output_hash = _atomic_save_png(destination / output_relative, crop)
                image_hashes.append((output_relative, output_hash))
            base_index.update(
                {
                    "status": "included",
                    "exclusion_reason": None,
                    "output_image_sha256": output_hash,
                }
            )
            recognition_index.append(base_index)
            rec_label_rows[split].append((output_relative, content))
            eligible_characters[split].update(content)
            rec_strata["split"][split] += 1
            rec_strata["source_domain"][str(row.get("source_domain", "unknown"))] += 1
            rec_strata["category"][str(row.get("category", "unknown"))] += 1
            rec_strata["details"][str(row.get("details", "unknown"))] += 1
            rec_strata["vertical_long"][str(vertical_long).lower()] += 1

        for top in _tile_starts(height, policy.tile_size, policy.tile_overlap):
            for left in _tile_starts(width, policy.tile_size, policy.tile_overlap):
                valid_right = min(width, left + policy.tile_size)
                valid_bottom = min(height, top + policy.tile_size)
                tile_relative = _detection_image_path(split, sample_id, top, left)
                annotations: list[dict[str, Any]] = []
                positive_count = 0
                ignored_count = 0
                for row in rows:
                    if row.get("state") != "active":
                        continue
                    bbox = _bbox(row, width=width, height=height)
                    intersection = _intersection(
                        bbox,
                        left=left,
                        top=top,
                        right=valid_right,
                        bottom=valid_bottom,
                    )
                    if intersection is None:
                        continue
                    fully_contained = _fully_contained(
                        bbox,
                        left=left,
                        top=top,
                        right=valid_right,
                        bottom=valid_bottom,
                    )
                    content = row.get("content")
                    unsafe_reason = _line_safety_exclusion_reason(content)
                    ignored = not fully_contained or unsafe_reason is not None
                    label_bbox = bbox if fully_contained else intersection
                    points = _clockwise_tile_points(label_bbox, left=left, top=top)
                    annotation: dict[str, Any] = {
                        "transcription": "###" if ignored else content,
                        "points": points,
                        "record_id": row["record_id"],
                        "ignore": ignored,
                    }
                    if ignored:
                        annotation["ignore_reason"] = (
                            "partial_intersection"
                            if not fully_contained
                            else unsafe_reason
                        )
                        ignored_count += 1
                    else:
                        positive_count += 1
                    annotations.append(annotation)

                annotations.sort(key=lambda value: value["record_id"])
                tile_hash: str | None = None
                if policy.materialize_images:
                    tile = Image.new(
                        "RGB", (policy.tile_size, policy.tile_size), (255, 255, 255)
                    )
                    source_crop = image.crop(
                        (left, top, valid_right, valid_bottom)
                    ).convert("RGB")
                    tile.paste(source_crop, (0, 0))
                    tile_hash = _atomic_save_png(destination / tile_relative, tile)
                    image_hashes.append((tile_relative, tile_hash))
                tile_index = {
                    "schema_version": 1,
                    "tile_id": f"{sample_id}:{top}:{left}",
                    "sample_id": sample_id,
                    "split": split,
                    "source_domain": source.get("source_class"),
                    "source_image": image_entry["path"],
                    "source_image_sha256": actual_source_hashes[sample_id],
                    "source_format": actual_format,
                    "source_box": {
                        "left": left,
                        "top": top,
                        "right": valid_right,
                        "bottom": valid_bottom,
                    },
                    "valid_size": [valid_right - left, valid_bottom - top],
                    "output_size": [policy.tile_size, policy.tile_size],
                    "padding": {
                        "right": policy.tile_size - (valid_right - left),
                        "bottom": policy.tile_size - (valid_bottom - top),
                    },
                    "transform": {
                        "source_to_tile": [1, 0, -left, 0, 1, -top],
                        "tile_to_source": [1, 0, left, 0, 1, top],
                    },
                    "output_image": tile_relative,
                    "output_image_sha256": tile_hash,
                    "positive_count": positive_count,
                    "ignored_count": ignored_count,
                    "empty_tile": positive_count == 0,
                    "annotations": annotations,
                }
                tile_index["spec_sha256"] = _canonical_sha256(
                    {key: value for key, value in tile_index.items() if key != "output_image_sha256"}
                )
                detection_index.append(tile_index)
                det_label_rows[split].append((tile_relative, annotations))
                det_strata["split"][split] += 1
                det_strata["source_domain"][str(source.get("source_class", "unknown"))] += 1
                det_strata["empty_tile"][str(not annotations).lower()] += 1

        for row in rows:
            if row.get("state") != "active":
                continue
            bbox = _bbox(row, width=width, height=height)
            if _bbox_width(bbox) > policy.tile_size or _bbox_height(bbox) > policy.tile_size:
                oversize.append(
                    {
                        "record_id": row["record_id"],
                        "sample_id": sample_id,
                        "split": split,
                        "width": _bbox_width(bbox),
                        "height": _bbox_height(bbox),
                        "tile_size": policy.tile_size,
                    }
                )
        image.close()

    recognition_index.sort(key=lambda row: row["record_id"])
    detection_index.sort(key=lambda row: row["tile_id"])
    excluded.sort(key=lambda row: row["record_id"])
    oversize.sort(key=lambda row: row["record_id"])
    for split in VALID_SPLITS:
        rec_label_rows[split].sort(key=lambda item: item[0])
        det_label_rows[split].sort(key=lambda item: item[0])

    train_charset = sorted(eligible_characters["train"], key=ord)
    all_charset = set().union(*eligible_characters.values())
    oov_by_split: dict[str, dict[str, Any]] = {}
    for split in ("val", "test"):
        oov = sorted(eligible_characters[split] - eligible_characters["train"], key=ord)
        oov_records = sorted(
            row["record_id"]
            for row in recognition_index
            if row["split"] == split
            and row["status"] == "included"
            and isinstance(row["content"], str)
            and any(character in set(oov) for character in row["content"])
        )
        oov_by_split[split] = {
            "characters": oov,
            "codepoints": [f"U+{ord(character):04X}" for character in oov],
            "character_count": len(oov),
            "record_count": len(oov_records),
            "record_ids": oov_records,
        }

    exclusion_counts = Counter(row["reason"] for row in excluded)
    report: dict[str, Any] = {
        "schema_version": 1,
        "policy": {
            "raw_text": "preserved_without_strip_normalization_or_truncation",
            "unsafe_line_content": "exclude_record_and_ignore_detection_box",
            "max_text_length": policy.max_text_length,
            "overlength": "exclude_not_truncate",
            "recognition_padding": policy.padding,
            "detection_tile_size": policy.tile_size,
            "detection_tile_overlap": policy.tile_overlap,
            "partial_detection_box": "clip_intersection_and_mark_###_ignore",
            "empty_detection_tiles_retained": True,
            "bbox_points": "top_left_top_right_bottom_right_bottom_left_clockwise",
        },
        "selection": {
            "available_samples": len(all_sample_ids),
            "selected_samples": len(selected_ids),
            "selected_sample_ids": selected_ids,
            "sample_limit": policy.sample_limit,
        },
        "recognition": {
            "indexed": len(recognition_index),
            "included": sum(len(rec_label_rows[split]) for split in VALID_SPLITS),
            "excluded": len(excluded),
            "included_by_split": {
                split: len(rec_label_rows[split]) for split in VALID_SPLITS
            },
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
            "excluded_records": excluded,
            "vertical_long_included": rec_strata["vertical_long"].get("true", 0),
        },
        "detection": {
            "tiles": len(detection_index),
            "tiles_by_split": {
                split: len(det_label_rows[split]) for split in VALID_SPLITS
            },
            "empty_tiles": sum(row["empty_tile"] for row in detection_index),
            "positive_boxes": sum(row["positive_count"] for row in detection_index),
            "ignored_boxes": sum(row["ignored_count"] for row in detection_index),
            "oversize_count": len(oversize),
            "oversize_records": oversize,
        },
        "charset": {
            "training_characters": train_charset,
            "training_codepoints": [f"U+{ord(character):04X}" for character in train_charset],
            "training_character_count": len(train_charset),
            "all_selected_character_count": len(all_charset),
            "oov_by_split": oov_by_split,
        },
        "lengths": {
            "unit": "unicode_codepoints",
            "maximum_allowed": policy.max_text_length,
            "observed_maximum": max(length_counts, default=0),
            "histogram": {
                str(length): count for length, count in sorted(length_counts.items())
            },
        },
        "stratification": {
            "recognition": {
                name: dict(sorted(counter.items()))
                for name, counter in rec_strata.items()
            },
            "detection_tiles": {
                name: dict(sorted(counter.items()))
                for name, counter in det_strata.items()
            },
        },
        "lineage": lineage,
        "materialize_images": policy.materialize_images,
    }

    artifact_paths: list[Path] = []
    recognition_path = destination / "recognition_index.jsonl"
    detection_path = destination / "detection_index.jsonl"
    _atomic_jsonl(recognition_path, recognition_index)
    _atomic_jsonl(detection_path, detection_index)
    artifact_paths.extend((recognition_path, detection_path))
    for split in VALID_SPLITS:
        rec_path = destination / f"rec_{split}.txt"
        det_path = destination / f"det_{split}.txt"
        _atomic_text(
            rec_path,
            "".join(f"{path}\t{content}\n" for path, content in rec_label_rows[split]),
        )
        _atomic_text(
            det_path,
            "".join(
                f"{path}\t{_canonical_json(_paddle_annotations(annotations))}\n"
                for path, annotations in det_label_rows[split]
            ),
        )
        artifact_paths.extend((rec_path, det_path))
    charset_path = destination / "charset.txt"
    _atomic_text(charset_path, "".join(f"{character}\n" for character in train_charset))
    report_path = destination / "report.json"
    _atomic_json(report_path, report)
    artifact_paths.extend((charset_path, report_path))

    artifact_hashes = {
        path.relative_to(destination).as_posix(): _sha256_file(path)
        for path in sorted(artifact_paths)
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pcbparse_ocr_paddle_materialization",
        "complete": True,
        "materialize_images": policy.materialize_images,
        "lineage": lineage,
        "policy_sha256": _canonical_sha256(report["policy"]),
        "selection_sha256": _canonical_sha256(report["selection"]),
        "counts": {
            "recognition_index": len(recognition_index),
            "recognition_included": report["recognition"]["included"],
            "recognition_excluded": len(excluded),
            "detection_tiles": len(detection_index),
            "detection_oversize": len(oversize),
        },
        "artifacts": artifact_hashes,
        "image_set": {
            "count": len(image_hashes),
            "sha256": _canonical_sha256(sorted(image_hashes))
            if policy.materialize_images
            else None,
        },
    }
    manifest_path = destination / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest


def _validate_lineage(
    *,
    summary: dict[str, Any],
    summary_path: Path,
    ocr_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    ocr_path: Path,
    source_base: Path,
    source_manifest_path: Path,
    split_manifest_path: Path,
) -> dict[str, Any]:
    summary_count = summary.get("counts", {}).get("ocr_text")
    if summary_count != len(ocr_rows):
        raise OCRMaterializationError(
            f"derived summary OCR count mismatch: expected {summary_count!r}, "
            f"read {len(ocr_rows)}"
        )
    task_file = summary.get("task_files", {}).get("ocr_text")
    if isinstance(task_file, str) and Path(task_file).name != ocr_path.name:
        raise OCRMaterializationError(
            f"derived summary binds OCR task to {task_file!r}, not {ocr_path.name!r}"
        )
    config_hash = summary.get("config_hash")
    derivation_version = summary.get("derivation_version")
    if not _is_sha256(config_hash):
        raise OCRMaterializationError("derived summary config_hash is not SHA-256")
    if not isinstance(derivation_version, str) or not derivation_version:
        raise OCRMaterializationError("derived summary lacks derivation_version")
    expected_split_hash = summary.get("config", {}).get("split_manifest_sha256")
    actual_split_hash = _sha256_file(split_manifest_path)
    if expected_split_hash != actual_split_hash:
        raise OCRMaterializationError(
            "split manifest SHA-256 does not match derived summary lineage"
        )
    if summary.get("sample_count") != len(source_rows):
        raise OCRMaterializationError(
            "derived summary sample_count does not match source manifest"
        )

    source_by_id = {row["sample_id"]: row for row in source_rows}
    split_by_id = {row["sample_id"]: row for row in split_rows}
    if set(source_by_id) != set(split_by_id):
        missing = sorted(set(source_by_id) - set(split_by_id))
        extra = sorted(set(split_by_id) - set(source_by_id))
        raise OCRMaterializationError(
            f"split/source sample coverage mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )

    group_splits: dict[str, set[str]] = defaultdict(set)
    for sample_id in sorted(source_by_id):
        source = source_by_id[sample_id]
        split_row = split_by_id[sample_id]
        split = split_row.get("split")
        if split not in VALID_SPLITS:
            raise OCRMaterializationError(
                f"sample {sample_id} has unsupported split {split!r}"
            )
        source_group = source.get("duplicate_group")
        split_group = split_row.get("duplicate_group", source_group)
        if not isinstance(source_group, str) or not source_group:
            raise OCRMaterializationError(
                f"source sample {sample_id} lacks duplicate_group"
            )
        if split_group != source_group:
            raise OCRMaterializationError(
                f"duplicate_group mismatch for sample {sample_id}"
            )
        group_splits[source_group].add(split)
        image_entry = _image_entry(source, sample_id)
        _safe_source_path(source_base, image_entry["path"])

    leaking = {
        group: sorted(splits)
        for group, splits in group_splits.items()
        if len(splits) != 1
    }
    if leaking:
        first = sorted(leaking)[0]
        raise OCRMaterializationError(
            f"duplicate group leakage for {first}: {leaking[first]}"
        )

    for row in ocr_rows:
        record_id = row.get("record_id")
        sample_id = row.get("sample_id")
        if row.get("task") != "ocr_text":
            raise OCRMaterializationError(f"record {record_id} is not task ocr_text")
        if row.get("config_hash") != config_hash:
            raise OCRMaterializationError(f"record {record_id} config_hash mismatch")
        if row.get("derivation_version") != derivation_version:
            raise OCRMaterializationError(
                f"record {record_id} derivation_version mismatch"
            )
        if sample_id not in source_by_id:
            raise OCRMaterializationError(
                f"record {record_id} references unknown sample {sample_id!r}"
            )
        source = source_by_id[sample_id]
        split_row = split_by_id[sample_id]
        if row.get("split") != split_row["split"]:
            raise OCRMaterializationError(f"record {record_id} split mismatch")
        if row.get("duplicate_group") != source.get("duplicate_group"):
            raise OCRMaterializationError(
                f"record {record_id} duplicate_group mismatch"
            )
        if row.get("source_image") != _image_entry(source, sample_id)["path"]:
            raise OCRMaterializationError(
                f"record {record_id} source_image lineage mismatch"
            )
        if row.get("source_domain") != source.get("source_class"):
            raise OCRMaterializationError(
                f"record {record_id} source_domain lineage mismatch"
            )
        if row.get("normalization_applied") is not False:
            raise OCRMaterializationError(
                f"record {record_id} does not certify strict raw text"
            )
        content = row.get("content")
        if isinstance(content, str):
            # These records are excluded later, never normalized. Scanning the
            # full manifest here ensures a sample limit cannot hide bad labels.
            _line_safety_exclusion_reason(content)
        canvas = source.get("canvas")
        if not isinstance(canvas, dict):
            image = _image_entry(source, sample_id)
            canvas = {"width": image.get("width"), "height": image.get("height")}
        _bbox(
            row,
            width=_positive_int(canvas.get("width"), f"{sample_id} canvas width"),
            height=_positive_int(canvas.get("height"), f"{sample_id} canvas height"),
        )

    return {
        "derivation_version": derivation_version,
        "dataset_config_hash": config_hash,
        "derived_summary_sha256": _sha256_file(summary_path),
        "ocr_manifest_sha256": _sha256_file(ocr_path),
        "source_manifest_sha256": _sha256_file(source_manifest_path),
        "split_manifest_sha256": actual_split_hash,
        "source_sample_count": len(source_rows),
        "ocr_record_count": len(ocr_rows),
        "duplicate_group_leakage": 0,
    }


def _recognition_exclusion_reason(
    row: dict[str, Any], content: Any, *, max_text_length: int
) -> str | None:
    if row.get("state") != "active":
        return "inactive_text"
    unsafe = _line_safety_exclusion_reason(content)
    if unsafe is not None:
        return unsafe
    assert isinstance(content, str)
    if len(content) > max_text_length:
        return "over_max_text_length"
    return None


def _line_safety_exclusion_reason(content: Any) -> str | None:
    if not isinstance(content, str):
        return "non_string_content"
    if content == "":
        return "empty_content"
    if "\t" in content:
        return "contains_tab"
    if "\r" in content:
        return "contains_carriage_return"
    if "\n" in content:
        return "contains_line_feed"
    return None


def _paddle_annotations(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "transcription": annotation["transcription"],
            "points": annotation["points"],
        }
        for annotation in annotations
    ]


def _recognition_image_path(split: str, record_id: str) -> str:
    digest = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
    return f"rec/{split}/{digest}.png"


def _detection_image_path(split: str, sample_id: str, top: int, left: int) -> str:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]
    return f"det/{split}/{digest}_{top:08d}_{left:08d}.png"


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    values = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if values[-1] != last:
        values.append(last)
    return values


def _bbox(row: dict[str, Any], *, width: int, height: int) -> dict[str, int | float]:
    record_id = row.get("record_id", "<unknown>")
    raw = row.get("bbox")
    if not isinstance(raw, dict):
        raise OCRMaterializationError(f"record {record_id} lacks a valid bbox")
    result: dict[str, int | float] = {}
    for key in ("x1", "y1", "x2", "y2"):
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise OCRMaterializationError(
                f"record {record_id} bbox {key} is not finite numeric"
            )
        number = float(value)
        if not math.isfinite(number):
            raise OCRMaterializationError(
                f"record {record_id} bbox {key} is not finite numeric"
            )
        result[key] = int(number) if number.is_integer() else number
    if result["x2"] <= result["x1"] or result["y2"] <= result["y1"]:
        raise OCRMaterializationError(f"record {record_id} bbox has zero/negative area")
    if (
        result["x1"] < 0
        or result["y1"] < 0
        or result["x2"] > width
        or result["y2"] > height
    ):
        raise OCRMaterializationError(
            f"record {record_id} bbox is outside the declared canvas"
        )
    return result


def _bbox_width(bbox: dict[str, int | float]) -> int | float:
    return bbox["x2"] - bbox["x1"]


def _bbox_height(bbox: dict[str, int | float]) -> int | float:
    return bbox["y2"] - bbox["y1"]


def _padded_crop_box(
    bbox: dict[str, int | float], *, width: int, height: int, padding: int
) -> dict[str, int]:
    left = max(0, math.floor(bbox["x1"]) - padding)
    top = max(0, math.floor(bbox["y1"]) - padding)
    right = min(width, math.ceil(bbox["x2"]) + padding)
    bottom = min(height, math.ceil(bbox["y2"]) + padding)
    if right <= left or bottom <= top:
        raise OCRMaterializationError("padded recognition crop has zero area")
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def _fully_contained(
    bbox: dict[str, int | float], *, left: int, top: int, right: int, bottom: int
) -> bool:
    return (
        bbox["x1"] >= left
        and bbox["y1"] >= top
        and bbox["x2"] <= right
        and bbox["y2"] <= bottom
    )


def _intersection(
    bbox: dict[str, int | float], *, left: int, top: int, right: int, bottom: int
) -> dict[str, int | float] | None:
    x1 = max(bbox["x1"], left)
    y1 = max(bbox["y1"], top)
    x2 = min(bbox["x2"], right)
    y2 = min(bbox["y2"], bottom)
    if x2 <= x1 or y2 <= y1:
        return None
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _clockwise_tile_points(
    bbox: dict[str, int | float], *, left: int, top: int
) -> list[list[int | float]]:
    x1 = _clean_number(bbox["x1"] - left)
    y1 = _clean_number(bbox["y1"] - top)
    x2 = _clean_number(bbox["x2"] - left)
    y2 = _clean_number(bbox["y2"] - top)
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _clean_number(value: int | float) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number


def _image_entry(source: dict[str, Any], sample_id: str) -> dict[str, Any]:
    files = source.get("files")
    image = files.get("image") if isinstance(files, dict) else None
    if not isinstance(image, dict):
        raise OCRMaterializationError(f"source sample {sample_id} lacks image metadata")
    path = image.get("path")
    digest = image.get("sha256")
    if not isinstance(path, str) or not path:
        raise OCRMaterializationError(f"source sample {sample_id} lacks image path")
    if not _is_sha256(digest):
        raise OCRMaterializationError(
            f"source sample {sample_id} image SHA-256 is invalid"
        )
    return image


def _safe_source_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise OCRMaterializationError(f"source image path must be relative: {relative!r}")
    resolved = (root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise OCRMaterializationError(
            f"source image path escapes source root: {relative!r}"
        ) from exc
    if not resolved.is_file():
        raise OCRMaterializationError(f"source image does not exist: {resolved}")
    return resolved


def _decode_image_by_magic(path: Path) -> tuple[Image.Image, str]:
    try:
        with Image.open(path) as probe:
            actual_format = probe.format
            probe.verify()
        with Image.open(path) as opened:
            opened.load()
            image = opened.copy()
    except (OSError, ValueError) as exc:
        raise OCRMaterializationError(f"cannot decode source image {path}: {exc}") from exc
    if not actual_format:
        image.close()
        raise OCRMaterializationError(f"image magic did not identify a format: {path}")
    return image, str(actual_format).upper()


def _validate_decoded_source(
    *,
    sample_id: str,
    source: dict[str, Any],
    image_entry: dict[str, Any],
    width: int,
    height: int,
    actual_format: str,
) -> None:
    expected_width = image_entry.get("width")
    expected_height = image_entry.get("height")
    canvas = source.get("canvas")
    if isinstance(canvas, dict):
        expected_width = canvas.get("width", expected_width)
        expected_height = canvas.get("height", expected_height)
    if width != expected_width or height != expected_height:
        raise OCRMaterializationError(
            f"decoded size mismatch for sample {sample_id}: "
            f"expected {expected_width}x{expected_height}, got {width}x{height}"
        )
    expected_format = image_entry.get("detected_format")
    if isinstance(expected_format, str) and expected_format.upper() != actual_format:
        raise OCRMaterializationError(
            f"decoded format mismatch for sample {sample_id}: "
            f"expected {expected_format}, got {actual_format}"
        )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise OCRMaterializationError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OCRMaterializationError(f"JSON root is not an object: {path}")
    return value


def _read_jsonl(path: Path, *, identifier: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw, parse_constant=_reject_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise OCRMaterializationError(
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise OCRMaterializationError(
                    f"JSONL row is not an object at {path}:{line_number}"
                )
            value = row.get(identifier)
            if not isinstance(value, str) or not value:
                raise OCRMaterializationError(
                    f"JSONL row at {path}:{line_number} lacks {identifier}"
                )
            if value in seen:
                raise OCRMaterializationError(
                    f"duplicate {identifier} {value!r} in {path}"
                )
            seen.add(value)
            rows.append(row)
    return rows


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _text_sha256(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save_png(path: Path, image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    payload = buffer.getvalue()
    _atomic_bytes(path, payload)
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_text(path, "".join(f"{_canonical_json(row)}\n" for row in rows))


def _atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(path, value.encode("utf-8"))


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise OCRMaterializationError(f"{label} must be a positive integer")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


__all__ = [
    "OCRMaterializationError",
    "OCRMaterializationOptions",
    "materialize_ocr_training_data",
]
