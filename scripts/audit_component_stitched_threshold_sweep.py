from __future__ import annotations

"""Audit score thresholds on frozen, already-stitched Component predictions.

This module is deliberately inference-free and torch-free.  It verifies the
completed v4/v5 tiled diagnostic artifacts and their formal parent lineage,
then calibrates score thresholds on the common frozen validation population.
Every scan candidate is an exact score observed in one of the supplied
prediction artifacts.  The result is validation-calibration-only evidence: it
cannot register, bless, promote, or modify a model.
"""

import argparse
import hashlib
import json
import math
import os
import tempfile
from bisect import bisect_right
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TILED_KIND = "pcbparse_component_tiled_frozen_val_diagnostic"
TILED_STATUS = "diagnostic_evaluation_complete"
FORMAL_PARENT_STATUSES = frozenset({"candidate_evidence_ready", "gate_failed"})
DEFAULT_POPULATION_FLOOR = 0.20
DEFAULT_MATCH_IOU = 0.50
DEFAULT_NMS_IOU = 0.55


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Hash-bound validation-only score sweep over existing stitched "
            "Component predictions; no model is loaded or executed."
        )
    )
    parser.add_argument("--workspace-root", type=Path, default=root)
    parser.add_argument(
        "--candidate",
        nargs=3,
        action="append",
        metavar=("LABEL", "STATUS", "PREDICTIONS"),
        help=(
            "Repeat for each candidate. Defaults to the frozen v4 and v5 "
            "tiled512 CPU validation replays."
        ),
    )
    parser.add_argument(
        "--component-coco",
        type=Path,
        default=Path(
            "artifacts/training/materialized/dataset300-v2/"
            "component-pin/component_coco.json"
        ),
    )
    parser.add_argument(
        "--materialization-summary",
        type=Path,
        default=Path(
            "artifacts/training/materialized/dataset300-v2/"
            "component-pin/summary.json"
        ),
    )
    parser.add_argument(
        "--component-manifest",
        type=Path,
        default=Path("data/derived/dataset300-v2/component_detection.jsonl"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/manifests/dataset300.jsonl"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/splits/dataset300_split.jsonl"),
    )
    parser.add_argument(
        "--models-lock",
        type=Path,
        default=Path("models/models.lock.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/audits/component_v4_v5_stitched_threshold_sweep.json"
        ),
    )
    parser.add_argument(
        "--population-floor", type=float, default=DEFAULT_POPULATION_FLOOR
    )
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_MATCH_IOU)
    parser.add_argument("--major-class-count", type=int, default=10)
    return parser


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {label} JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                try:
                    row = json.loads(
                        raw,
                        object_pairs_hook=_unique_object,
                        parse_constant=_reject_constant,
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid {label} JSONL at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"{label} JSONL row is not an object at {path}:{line_number}"
                    )
                rows.append(row)
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 in {label} JSONL at {path}: {exc}") from exc
    if not rows:
        raise ValueError(f"{label} JSONL is empty: {path}")
    return rows


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an array")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _unit_interval(value: Any, label: str, *, positive: bool = False) -> float:
    result = _finite(value, label)
    if result > 1.0 or (result <= 0.0 if positive else result < 0.0):
        interval = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{label} must be in {interval}")
    return result


def _positive_integer(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _xyxy(value: Any, label: str) -> tuple[float, float, float, float]:
    coordinates = _sequence(value, label)
    if len(coordinates) != 4:
        raise ValueError(f"{label} must contain four coordinates")
    x1, y1, x2, y2 = (_finite(item, label) for item in coordinates)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{label} must have positive area")
    return x1, y1, x2, y2


def _sha256(path: Path) -> str:
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


def _resolve_within(
    workspace: Path,
    raw: Path | str,
    label: str,
    *,
    must_exist: bool = True,
) -> Path:
    path = Path(raw).expanduser()
    resolved = (path if path.is_absolute() else workspace / path).resolve(strict=False)
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside workspace: {resolved}") from exc
    if must_exist and (not resolved.is_file() or resolved.stat().st_size <= 0):
        raise FileNotFoundError(f"{label} is missing or empty: {resolved}")
    return resolved


def _relative(path: Path, workspace: Path) -> str:
    return path.relative_to(workspace).as_posix()


def _expect_sha(actual: str, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} expected SHA-256 is missing or invalid")
    if actual != expected.lower():
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected.lower()}, actual={actual}"
        )


def _verified_reference(
    *,
    workspace: Path,
    reference: Mapping[str, Any],
    expected_path: Path | None,
    label: str,
) -> tuple[Path, str]:
    path = _resolve_within(
        workspace,
        _nonempty_string(reference.get("path"), f"{label}.path"),
        label,
    )
    if expected_path is not None and path != expected_path:
        raise ValueError(
            f"{label} path mismatch: expected={expected_path}, actual={path}"
        )
    actual = _sha256(path)
    _expect_sha(actual, reference.get("sha256"), label)
    return path, actual


def _unique_index(
    rows: Iterable[Mapping[str, Any]], key: str, label: str
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        identity = _nonempty_string(row.get(key), f"{label}.{key}")
        if identity in output:
            raise ValueError(f"duplicate {label} {key}: {identity}")
        output[identity] = row
    return output


def _close(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=tolerance
        )
    except (TypeError, ValueError):
        return False


def _source_entry(path: Path, workspace: Path, sha256: str) -> dict[str, Any]:
    return {
        "path": _relative(path, workspace),
        "sha256": sha256,
        "verified": True,
    }


def _load_frozen_truth(
    *,
    workspace: Path,
    component_coco_path: Path,
    materialization_summary_path: Path,
    component_manifest_path: Path,
    source_manifest_path: Path,
    split_manifest_path: Path,
) -> dict[str, Any]:
    paths = {
        "component_coco": component_coco_path,
        "materialization_summary": materialization_summary_path,
        "component_manifest": component_manifest_path,
        "source_manifest": source_manifest_path,
        "split_manifest": split_manifest_path,
    }
    hashes = {name: _sha256(path) for name, path in paths.items()}
    summary = _read_object(materialization_summary_path, "materialization summary")
    coco = _read_object(component_coco_path, "Component COCO")
    binding = _nonempty_string(summary.get("binding_sha256"), "summary binding")
    info = _mapping(coco.get("info"), "Component COCO info")
    if info.get("binding_sha256") != binding:
        raise ValueError("Component COCO/materialization binding mismatch")
    summary_outputs = _mapping(summary.get("outputs"), "summary.outputs")
    coco_output = _mapping(
        summary_outputs.get("component_coco.json"),
        "summary.outputs.component_coco.json",
    )
    _expect_sha(hashes["component_coco"], coco_output.get("sha256"), "COCO/summary")
    summary_lineage = _mapping(summary.get("lineage"), "summary.lineage")
    if dict(_mapping(info.get("lineage"), "COCO info.lineage")) != dict(
        summary_lineage
    ):
        raise ValueError("Component COCO lineage differs from materialization summary")
    lineage_inputs = _mapping(summary_lineage.get("inputs"), "summary lineage inputs")
    for lineage_name, source_name in (
        ("component_manifest", "component_manifest"),
        ("source_manifest", "source_manifest"),
        ("split_manifest", "split_manifest"),
    ):
        reference = _mapping(
            lineage_inputs.get(lineage_name), f"summary lineage {lineage_name}"
        )
        _verified_reference(
            workspace=workspace,
            reference=reference,
            expected_path=paths[source_name],
            label=f"summary {lineage_name}",
        )

    split_rows = _read_jsonl(split_manifest_path, "split manifest")
    split_by_sample = _unique_index(split_rows, "sample_id", "split manifest")
    duplicate_group_splits: dict[str, set[str]] = defaultdict(set)
    for sample_id, row in split_by_sample.items():
        if row.get("split") not in {"train", "val", "test"}:
            raise ValueError(f"unsupported split for sample {sample_id}")
        duplicate_group = _nonempty_string(
            row.get("duplicate_group"), "split manifest duplicate_group"
        )
        duplicate_group_splits[duplicate_group].add(str(row["split"]))
    leaking_groups = sorted(
        group for group, splits in duplicate_group_splits.items() if len(splits) != 1
    )
    if leaking_groups:
        raise ValueError(f"duplicate groups cross splits: {leaking_groups[:4]}")
    source_rows = _read_jsonl(source_manifest_path, "source manifest")
    source_by_sample = _unique_index(source_rows, "sample_id", "source manifest")
    if set(source_by_sample) != set(split_by_sample):
        raise ValueError("source and split manifest sample populations differ")

    component_rows = _read_jsonl(component_manifest_path, "component manifest")
    component_by_record = _unique_index(
        component_rows, "record_id", "component manifest"
    )
    domain_by_sample: dict[str, str] = {}
    val_component_records: set[str] = set()
    for record_id, row in component_by_record.items():
        sample_id = _nonempty_string(row.get("sample_id"), "component sample_id")
        split = _nonempty_string(row.get("split"), "component split")
        split_row = split_by_sample.get(sample_id)
        if split_row is None or split_row.get("split") != split:
            raise ValueError(f"component/split mismatch at {record_id}")
        if row.get("state") != "active":
            raise ValueError(f"component record is not active: {record_id}")
        domain = _nonempty_string(row.get("source_domain"), "component source_domain")
        previous = domain_by_sample.setdefault(sample_id, domain)
        if previous != domain:
            raise ValueError(f"multiple source domains for sample {sample_id}")
        if split == "val":
            val_component_records.add(record_id)

    categories: list[dict[str, Any]] = []
    category_by_id: dict[int, dict[str, Any]] = {}
    category_names: set[str] = set()
    for raw in _sequence(coco.get("categories"), "COCO categories"):
        category = _mapping(raw, "COCO category")
        category_id = _positive_integer(category.get("id"), "COCO category id")
        if category_id in category_by_id:
            raise ValueError(f"duplicate COCO category id {category_id}")
        name = _nonempty_string(category.get("name"), "COCO category name")
        if name in category_names:
            raise ValueError(f"duplicate COCO category name {name!r}")
        category_names.add(name)
        try:
            official_id = int(category.get("source_category_id"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid official category id for {name}") from exc
        item = {
            "index_id": category_id,
            "official_category_id": official_id,
            "name": name,
        }
        categories.append(item)
        category_by_id[category_id] = item
    categories.sort(key=lambda row: (int(row["official_category_id"]), row["name"]))

    val_samples = sorted(
        sample_id
        for sample_id, row in split_by_sample.items()
        if row.get("split") == "val"
    )
    val_sample_set = set(val_samples)
    image_ids: set[int] = set()
    val_image_by_id: dict[int, Mapping[str, Any]] = {}
    sample_metadata: dict[str, dict[str, Any]] = {}
    val_tile_count_by_sample: dict[str, int] = defaultdict(int)
    for raw in _sequence(coco.get("images"), "COCO images"):
        image = _mapping(raw, "COCO image")
        image_id = _positive_integer(image.get("id"), "COCO image id")
        if image_id in image_ids:
            raise ValueError(f"duplicate COCO image id {image_id}")
        image_ids.add(image_id)
        sample_id = _nonempty_string(image.get("sample_id"), "COCO image sample_id")
        split_row = split_by_sample.get(sample_id)
        if split_row is None or image.get("split") != split_row.get("split"):
            raise ValueError(f"COCO/split mismatch at image {image_id}")
        if image.get("binding_sha256") != binding:
            raise ValueError(f"COCO image binding mismatch at image {image_id}")
        if image.get("split") == "val" and image.get("view_kind") == "tile":
            val_tile_count_by_sample[sample_id] += 1
        if image.get("split") != "val" or image.get("view_kind") != "full":
            continue
        if sample_id in sample_metadata:
            raise ValueError(f"duplicate validation full view for {sample_id}")
        size = _sequence(image.get("source_size"), "validation source_size")
        if len(size) != 2:
            raise ValueError(f"validation source_size is invalid for {sample_id}")
        width = _positive_integer(size[0], "validation source width")
        height = _positive_integer(size[1], "validation source height")
        source = source_by_sample[sample_id]
        source_class = _nonempty_string(source.get("source_class"), "source_class")
        domain = domain_by_sample.get(sample_id)
        if domain is None or domain != source_class:
            raise ValueError(f"source/component domain mismatch for {sample_id}")
        source_image = _mapping(
            _mapping(source.get("files"), "source files").get("image"),
            "source image",
        )
        if (
            image.get("source_image") != source_image.get("path")
            or image.get("source_image_sha256") != source_image.get("sha256")
            or source_image.get("width") != width
            or source_image.get("height") != height
        ):
            raise ValueError(f"COCO/source image binding mismatch for {sample_id}")
        val_image_by_id[image_id] = image
        sample_metadata[sample_id] = {
            "sample_id": sample_id,
            "source_class": source_class,
            "source_domain": domain,
            "source_image": str(image["source_image"]),
            "source_image_sha256": str(image["source_image_sha256"]),
            "width": width,
            "height": height,
            "image_id": image_id,
        }
    if set(sample_metadata) != val_sample_set:
        raise ValueError("COCO full views do not cover the exact frozen val split")
    if set(val_tile_count_by_sample) != val_sample_set:
        raise ValueError("COCO tiles do not cover the exact frozen val split")

    truths: list[dict[str, Any]] = []
    selected_annotation_ids: set[int] = set()
    joined_records: set[str] = set()
    for raw in _sequence(coco.get("annotations"), "COCO annotations"):
        annotation = _mapping(raw, "COCO annotation")
        image_id = annotation.get("image_id")
        if image_id not in val_image_by_id:
            continue
        annotation_id = _positive_integer(
            annotation.get("id"), "validation annotation id"
        )
        if annotation_id in selected_annotation_ids:
            raise ValueError(f"duplicate validation annotation id {annotation_id}")
        selected_annotation_ids.add(annotation_id)
        if annotation.get("ignore") not in (0, False, None):
            raise ValueError("validation full-view truth unexpectedly ignored")
        if annotation.get("iscrowd") not in (0, False, None):
            raise ValueError("validation full-view truth unexpectedly marked crowd")
        if annotation.get("partial_box") not in (False, 0, None):
            raise ValueError("validation full-view truth unexpectedly partial")
        category_id = _positive_integer(
            annotation.get("category_id"), "validation category id"
        )
        category = category_by_id.get(category_id)
        if category is None:
            raise ValueError("validation annotation references unknown category")
        sample_id = str(val_image_by_id[int(image_id)]["sample_id"])
        metadata = sample_metadata[sample_id]
        box = _xyxy(annotation.get("source_bbox"), "validation source_bbox")
        if not (
            0.0 <= box[0] < box[2] <= metadata["width"]
            and 0.0 <= box[1] < box[3] <= metadata["height"]
        ):
            raise ValueError(f"validation truth outside source image for {sample_id}")
        if annotation.get("source_category") not in (None, category["name"]):
            raise ValueError("validation source category mapping changed")
        record_id = _nonempty_string(
            annotation.get("source_record_id"), "validation source_record_id"
        )
        manifest_row = component_by_record.get(record_id)
        if manifest_row is None or manifest_row.get("sample_id") != sample_id:
            raise ValueError(f"truth/component manifest join failed at {record_id}")
        if (
            manifest_row.get("category") != category["name"]
            or int(str(manifest_row.get("category_id")))
            != int(category["official_category_id"])
            or manifest_row.get("source_domain") != metadata["source_domain"]
        ):
            raise ValueError(f"truth/component manifest mapping changed at {record_id}")
        manifest_box = _mapping(manifest_row.get("bbox"), "component manifest bbox")
        if list(box) != [
            float(manifest_box[key]) for key in ("x1", "y1", "x2", "y2")
        ]:
            raise ValueError(f"truth/component bbox mismatch at {record_id}")
        joined_records.add(record_id)
        truths.append(
            {
                "annotation_id": annotation_id,
                "sample_id": sample_id,
                "category": category["name"],
                "official_category_id": int(category["official_category_id"]),
                "box": list(box),
                "source_class": metadata["source_class"],
                "source_domain": metadata["source_domain"],
            }
        )
    if joined_records != val_component_records:
        missing = sorted(val_component_records - joined_records)
        extra = sorted(joined_records - val_component_records)
        raise ValueError(
            "validation truth/component coverage mismatch: "
            f"missing={missing[:4]}, extra={extra[:4]}"
        )
    truths.sort(key=lambda row: int(row["annotation_id"]))
    truth_payload = [
        {
            "annotation_id": row["annotation_id"],
            "sample_id": row["sample_id"],
            "category": row["category"],
            "official_category_id": row["official_category_id"],
            "box": row["box"],
            "source_domain": row["source_domain"],
        }
        for row in truths
    ]
    return {
        "binding_sha256": binding,
        "categories": categories,
        "category_names": {str(row["name"]) for row in categories},
        "sample_ids": val_samples,
        "sample_metadata": sample_metadata,
        "val_tile_count_by_sample": dict(val_tile_count_by_sample),
        "truths": truths,
        "truth_population_sha256": _canonical_sha256(truth_payload),
        "hashes": hashes,
        "sources": {
            name: _source_entry(path, workspace, hashes[name])
            for name, path in paths.items()
        },
    }


def _validate_parent_lineage(
    *,
    parent: Mapping[str, Any],
    dataset: Mapping[str, Any],
    label: str,
) -> None:
    lineage = _mapping(parent.get("lineage"), f"{label} parent.lineage")
    expected = {
        "task": "component",
        "formal_evidence": True,
        "smoke": False,
        "binding_sha256": dataset["binding_sha256"],
        "materialization_commit_id": dataset["binding_sha256"],
        "materialization_commit_sha256": dataset["hashes"][
            "materialization_summary"
        ],
        "source_manifest_sha256": dataset["hashes"]["source_manifest"],
        "split_manifest_sha256": dataset["hashes"]["split_manifest"],
        "task_manifest_sha256": dataset["hashes"]["component_manifest"],
    }
    for key, value in expected.items():
        if lineage.get(key) != value:
            raise ValueError(
                f"{label} parent lineage mismatch for {key}: "
                f"expected={value!r}, actual={lineage.get(key)!r}"
            )


def _validate_candidate(
    *,
    workspace: Path,
    label: str,
    status_path: Path,
    predictions_path: Path,
    dataset: Mapping[str, Any],
    population_floor: float,
    iou_threshold: float,
) -> dict[str, Any]:
    status_sha256 = _sha256(status_path)
    predictions_sha256 = _sha256(predictions_path)
    status = _read_object(status_path, f"{label} tiled status")
    payload_copy = dict(status)
    claimed_payload = payload_copy.pop("diagnostic_payload_sha256", None)
    if claimed_payload != _canonical_sha256(payload_copy):
        raise ValueError(f"{label} tiled status canonical payload hash mismatch")
    expected_flags = {
        "schema_version": 1,
        "kind": TILED_KIND,
        "task": "component",
        "status": TILED_STATUS,
        "formal_evidence": False,
        "registerable": False,
        "promotion_eligible": False,
        "blessing_authorized": False,
        "smoke": False,
        "device": "cpu",
        "cuda_visible_devices_forced": "",
        "deterministic_algorithms": True,
    }
    for key, expected in expected_flags.items():
        if status.get(key) != expected:
            raise ValueError(
                f"{label} tiled status mismatch for {key}: "
                f"expected={expected!r}, actual={status.get(key)!r}"
            )
    if not status.get("finished_at_utc"):
        raise ValueError(f"{label} tiled status is not finished")
    protocol = _mapping(status.get("protocol"), f"{label} protocol")
    expected_protocol = {
        "tile_size": 512,
        "tile_overlap": 64,
        "tile_stride": 448,
        "max_detections_per_tile": 300,
        "thresholds_searched": False,
        "tiling_configs_searched": False,
        "nms_configs_searched": False,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"{label} tiled protocol mismatch for {key}")
    if not _close(protocol.get("score_threshold"), population_floor):
        raise ValueError(f"{label} stored population floor differs from requested floor")
    if not _close(protocol.get("match_iou"), iou_threshold):
        raise ValueError(f"{label} match IoU differs from requested IoU")
    if not _close(protocol.get("component_nms_iou"), DEFAULT_NMS_IOU):
        raise ValueError(f"{label} NMS IoU differs from fixed stitched protocol")

    artifacts = _mapping(status.get("artifacts"), f"{label} status.artifacts")
    prediction_reference = _mapping(
        artifacts.get("predictions"), f"{label} status predictions"
    )
    _verified_reference(
        workspace=workspace,
        reference=prediction_reference,
        expected_path=predictions_path,
        label=f"{label} predictions",
    )
    if prediction_reference.get("sha256") != predictions_sha256:
        raise ValueError(f"{label} prediction reference changed during verification")

    lineage = _mapping(status.get("lineage"), f"{label} status.lineage")
    for key, expected_path, source_key in (
        ("component_coco", None, "component_coco"),
        ("materialization_summary", None, "materialization_summary"),
    ):
        reference = _mapping(lineage.get(key), f"{label} lineage.{key}")
        path, digest = _verified_reference(
            workspace=workspace,
            reference=reference,
            expected_path=None,
            label=f"{label} {key}",
        )
        expected_source = dataset["sources"][source_key]
        if (
            _relative(path, workspace) != expected_source["path"]
            or digest != expected_source["sha256"]
        ):
            raise ValueError(f"{label} {key} differs from supplied frozen truth")
    materialization_reference = _mapping(
        lineage.get("materialization_summary"),
        f"{label} lineage.materialization_summary",
    )
    if materialization_reference.get("binding_sha256") != dataset["binding_sha256"]:
        raise ValueError(f"{label} materialization binding mismatch")

    parent_reference = _mapping(
        lineage.get("parent_status"), f"{label} lineage.parent_status"
    )
    parent_path, parent_sha256 = _verified_reference(
        workspace=workspace,
        reference=parent_reference,
        expected_path=None,
        label=f"{label} parent status",
    )
    parent = _read_object(parent_path, f"{label} parent status")
    if (
        parent.get("task") != "component"
        or parent.get("status") not in FORMAL_PARENT_STATUSES
        or parent.get("formal_evidence") is not True
        or parent.get("smoke") is not False
        or not parent.get("finished_at_utc")
    ):
        raise ValueError(f"{label} parent is not a completed formal Component run")
    if parent_reference.get("status") != parent.get("status"):
        raise ValueError(f"{label} parent status label/bytes mismatch")
    _validate_parent_lineage(parent=parent, dataset=dataset, label=label)

    checkpoint_reference = _mapping(
        lineage.get("checkpoint"), f"{label} lineage.checkpoint"
    )
    parent_artifact = _mapping(parent.get("artifact"), f"{label} parent.artifact")
    parent_artifact_path = _resolve_within(
        workspace,
        _nonempty_string(parent_artifact.get("path"), f"{label} parent artifact path"),
        f"{label} parent checkpoint",
    )
    checkpoint_path, checkpoint_sha256 = _verified_reference(
        workspace=workspace,
        reference=checkpoint_reference,
        expected_path=parent_artifact_path,
        label=f"{label} checkpoint",
    )
    _expect_sha(checkpoint_sha256, parent_artifact.get("sha256"), f"{label} parent artifact")

    ancillary_sources: dict[str, dict[str, Any]] = {}
    for key in (
        "parent_predictions",
        "pipeline_config",
        "thresholds_config",
        "training_config",
    ):
        reference = _mapping(lineage.get(key), f"{label} lineage.{key}")
        path = _resolve_within(
            workspace,
            _nonempty_string(reference.get("path"), f"{label} {key}.path"),
            f"{label} {key}",
        )
        declared_sha256 = _nonempty_string(
            reference.get("sha256"), f"{label} {key}.sha256"
        ).lower()
        if len(declared_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in declared_sha256
        ):
            raise ValueError(f"{label} {key} declared SHA-256 is invalid")
        current_sha256 = _sha256(path)
        ancillary_sources[key] = {
            "path": _relative(path, workspace),
            "declared_historical_sha256": declared_sha256,
            "current_sha256": current_sha256,
            "current_bytes_match_declared_history": current_sha256
            == declared_sha256,
            "used_for_threshold_metrics": False,
        }
        if key == "parent_predictions" and reference.get(
            "used_for_tiled_metrics"
        ) is not False:
            raise ValueError(f"{label} parent predictions usage boundary changed")

    boundary = _mapping(status.get("selection_boundary"), f"{label} boundary")
    expected_samples = list(dataset["sample_ids"])
    if (
        boundary.get("split") != "val"
        or boundary.get("available_val_samples") != len(expected_samples)
        or boundary.get("selected_val_samples") != len(expected_samples)
        or list(
            _sequence(
                boundary.get("selected_sample_ids"), f"{label} selected sample IDs"
            )
        )
        != expected_samples
        or boundary.get("selected_sample_ids_sha256")
        != _canonical_sha256(expected_samples)
        or boundary.get("selected_validation_annotations")
        != len(dataset["truths"])
    ):
        raise ValueError(f"{label} does not cover the exact frozen val population")
    for key in (
        "train_samples_loaded",
        "train_images_decoded",
        "train_annotations_scored",
        "test_samples_loaded",
        "test_images_decoded",
        "test_annotations_scored",
    ):
        if boundary.get(key) != 0:
            raise ValueError(f"{label} boundary requires {key}=0")
    interpretation = _mapping(
        status.get("interpretation_boundary"), f"{label} interpretation boundary"
    )
    expected_interpretation = {
        "training_gate_reused": False,
        "parent_gate_status_changed": False,
        "models_lock_read": False,
        "models_lock_modified": False,
        "validation_configuration_tuned": False,
        "test_split_used": False,
        "full_validation_conclusion_allowed": True,
    }
    for key, expected in expected_interpretation.items():
        if interpretation.get(key) is not expected:
            raise ValueError(f"{label} interpretation mismatch for {key}")

    rows = _read_jsonl(predictions_path, f"{label} predictions")
    if prediction_reference.get("row_count") != len(rows):
        raise ValueError(f"{label} prediction row count/status mismatch")
    prediction_by_sample = _unique_index(rows, "sample_id", f"{label} predictions")
    if list(prediction_by_sample) != expected_samples:
        raise ValueError(f"{label} prediction row order/population is not frozen val")
    detections: list[dict[str, Any]] = []
    total_tiles = 0
    total_before_nms = 0
    for sample_id in expected_samples:
        row = prediction_by_sample[sample_id]
        metadata = dataset["sample_metadata"][sample_id]
        if row.get("source_image_sha256") != metadata["source_image_sha256"]:
            raise ValueError(f"{label} source image hash mismatch for {sample_id}")
        if list(_sequence(row.get("shape"), f"{label} shape")) != [
            metadata["height"],
            metadata["width"],
        ]:
            raise ValueError(f"{label} source shape mismatch for {sample_id}")
        tile_count = _positive_integer(row.get("tile_count"), f"{label} tile_count")
        if tile_count != dataset["val_tile_count_by_sample"][sample_id]:
            raise ValueError(f"{label} tile count/COCO index mismatch for {sample_id}")
        before_nms = _nonnegative_integer(
            row.get("candidate_count_before_cross_tile_nms"),
            f"{label} pre-NMS count",
        )
        sample_predictions = _sequence(
            row.get("predictions"), f"{label} sample predictions"
        )
        if before_nms < len(sample_predictions):
            raise ValueError(f"{label} pre-NMS count is smaller than stitched output")
        total_tiles += tile_count
        total_before_nms += before_nms
        for prediction_index, raw in enumerate(sample_predictions):
            item = _mapping(raw, f"{label} prediction")
            category = _nonempty_string(item.get("category"), f"{label} category")
            if category not in dataset["category_names"]:
                raise ValueError(f"{label} prediction has unknown category {category!r}")
            score = _unit_interval(item.get("score"), f"{label} prediction score")
            if score < population_floor:
                raise ValueError(f"{label} prediction falls below stored population floor")
            box = _xyxy(item.get("box"), f"{label} prediction bbox")
            if not (
                0.0 <= box[0] < box[2] <= metadata["width"]
                and 0.0 <= box[1] < box[3] <= metadata["height"]
            ):
                raise ValueError(f"{label} prediction outside source image")
            source_view_id = _nonempty_string(
                item.get("source_view_id"), f"{label} source_view_id"
            )
            detections.append(
                {
                    "sample_id": sample_id,
                    "category": category,
                    "score": score,
                    "box": list(box),
                    "prediction_index": prediction_index,
                    "source_view_id": source_view_id,
                    "source_class": metadata["source_class"],
                    "source_domain": metadata["source_domain"],
                }
            )
    performance = _mapping(status.get("performance"), f"{label} performance")
    expected_performance = {
        "sample_count": len(rows),
        "tile_count": total_tiles,
        "candidate_count_before_cross_tile_nms": total_before_nms,
        "prediction_count_after_cross_tile_nms": len(detections),
    }
    for key, expected in expected_performance.items():
        if performance.get(key) != expected:
            raise ValueError(f"{label} prediction/performance mismatch for {key}")
    return {
        "label": label,
        "status": status,
        "status_path": status_path,
        "status_sha256": status_sha256,
        "predictions_path": predictions_path,
        "predictions_sha256": predictions_sha256,
        "parent_status_path": parent_path,
        "parent_status_sha256": parent_sha256,
        "parent_status": parent,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "ancillary_sources": ancillary_sources,
        "detections": detections,
        "protocol": dict(protocol),
    }


def _box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0.0:
        return 0.0
    left_area = (float(left[2]) - float(left[0])) * (
        float(left[3]) - float(left[1])
    )
    right_area = (float(right[2]) - float(right[0])) * (
        float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _annotate_matches(
    detections: Sequence[Mapping[str, Any]],
    truths: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """Freeze TP/FP flags under the production global-score-ranked contract."""

    gt_by_class_sample: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    detections_by_class: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for truth in sorted(truths, key=lambda row: int(row["annotation_id"])):
        gt_by_class_sample[str(truth["category"])][str(truth["sample_id"])].append(
            truth
        )
    for detection in detections:
        detections_by_class[str(detection["category"])].append(detection)
    annotated: list[dict[str, Any]] = []
    for category in sorted(detections_by_class):
        ground_truth = gt_by_class_sample.get(category, {})
        matched = {
            sample_id: [False] * len(rows) for sample_id, rows in ground_truth.items()
        }
        ranked = sorted(
            detections_by_class[category],
            key=lambda row: (
                -float(row["score"]),
                str(row["sample_id"]),
                int(row["prediction_index"]),
            ),
        )
        for detection in ranked:
            sample_id = str(detection["sample_id"])
            candidates = ground_truth.get(sample_id, [])
            best_index = -1
            best_iou = -1.0
            for gt_index, truth in enumerate(candidates):
                if matched[sample_id][gt_index]:
                    continue
                overlap = _box_iou(detection["box"], truth["box"])
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = gt_index
            is_tp = best_index >= 0 and best_iou >= iou_threshold
            if is_tp:
                matched[sample_id][best_index] = True
            annotated.append({**dict(detection), "is_true_positive": bool(is_tp)})
    annotated.sort(
        key=lambda row: (
            -float(row["score"]),
            str(row["category"]),
            str(row["sample_id"]),
            int(row["prediction_index"]),
        )
    )
    return annotated


def _metric_counts(tp: int, fp: int, support: int) -> dict[str, Any]:
    fn = support - tp
    if min(tp, fp, fn) < 0:
        raise RuntimeError("invalid TP/FP/FN accounting")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / support if support else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "prediction_count": tp + fp,
        "ground_truth_count": support,
        "precision50": precision,
        "recall50": recall,
        "f1_50": f1,
    }


def _build_prefix(
    annotated: Sequence[Mapping[str, Any]], support: int
) -> dict[str, Any]:
    ranked = sorted(
        annotated,
        key=lambda row: (
            -float(row["score"]),
            str(row["category"]),
            str(row["sample_id"]),
            int(row["prediction_index"]),
        ),
    )
    scores_desc: list[float] = []
    states_desc: list[dict[str, Any]] = []
    tp = fp = 0
    index = 0
    while index < len(ranked):
        score = float(ranked[index]["score"])
        while index < len(ranked) and float(ranked[index]["score"]) == score:
            if ranked[index]["is_true_positive"]:
                tp += 1
            else:
                fp += 1
            index += 1
        scores_desc.append(score)
        states_desc.append(_metric_counts(tp, fp, support))
    return {
        "scores_desc": scores_desc,
        "negative_scores": [-score for score in scores_desc],
        "states_desc": states_desc,
        "support": support,
    }


def _state_at(prefix: Mapping[str, Any], threshold: float) -> dict[str, Any]:
    negative_scores = prefix["negative_scores"]
    index = bisect_right(negative_scores, -float(threshold)) - 1
    if index < 0:
        return _metric_counts(0, 0, int(prefix["support"]))
    return dict(prefix["states_desc"][index])


def _boundary_thresholds(
    candidates: Mapping[str, Mapping[str, Any]], population_floor: float
) -> tuple[list[float], dict[float, list[str]]]:
    origins: dict[float, set[str]] = defaultdict(set)
    for label, candidate in candidates.items():
        for detection in candidate["detections"]:
            score = _unit_interval(detection["score"], f"{label} score boundary")
            if score < population_floor:
                raise ValueError(f"{label} score boundary falls below population floor")
            origins[score].add(label)
    if not origins:
        raise ValueError("stitched prediction population contains no score boundaries")
    thresholds = sorted(origins)
    return thresholds, {
        score: sorted(labels) for score, labels in origins.items()
    }


def _metric_delta(
    newer: Mapping[str, Any], older: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        key: newer[key] - older[key]
        for key in (
            "true_positives",
            "false_positives",
            "false_negatives",
            "prediction_count",
            "precision50",
            "recall50",
            "f1_50",
        )
    }


def _best_row(
    rows: Sequence[Mapping[str, Any]], label: str
) -> Mapping[str, Any]:
    owned = [row for row in rows if label in row["score_boundary_origins"]]
    if not owned:
        raise ValueError(f"no owned score boundary exists for {label}")

    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        metric = row["metrics"][label]
        tp = int(metric["true_positives"])
        fp = int(metric["false_positives"])
        support = int(metric["ground_truth_count"])
        fn = support - tp
        f1 = Fraction(2 * tp, 2 * tp + fp + fn) if 2 * tp + fp + fn else Fraction()
        precision = Fraction(tp, tp + fp) if tp + fp else Fraction()
        recall = Fraction(tp, support) if support else Fraction()
        return f1, precision, recall, float(row["score_threshold"])

    return max(owned, key=key)


def _slice_metric(
    annotated: Sequence[Mapping[str, Any]],
    truths: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    sample_ids: set[str] | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    def selected(row: Mapping[str, Any]) -> bool:
        return (
            (sample_ids is None or str(row["sample_id"]) in sample_ids)
            and (category is None or str(row["category"]) == category)
        )

    support = sum(1 for truth in truths if selected(truth))
    tp = fp = 0
    for detection in annotated:
        if float(detection["score"]) < threshold or not selected(detection):
            continue
        if detection["is_true_positive"]:
            tp += 1
        else:
            fp += 1
    return _metric_counts(tp, fp, support)


def _comparison_block(
    *,
    candidates: Mapping[str, Mapping[str, Any]],
    truths: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float],
    sample_ids: set[str] | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    metrics = {
        label: _slice_metric(
            candidate["annotated"],
            truths,
            threshold=thresholds[label],
            sample_ids=sample_ids,
            category=category,
        )
        for label, candidate in candidates.items()
    }
    labels = list(candidates)
    output: dict[str, Any] = {
        "thresholds": dict(thresholds),
        "metrics": metrics,
    }
    if len(labels) == 2:
        output[f"delta_{labels[1]}_minus_{labels[0]}"] = _metric_delta(
            metrics[labels[1]], metrics[labels[0]]
        )
    return output


def _snapshot(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: _sha256(path) for name, path in sorted(paths.items())}


def build_audit(
    *,
    workspace: Path,
    candidate_specs: Sequence[tuple[str, Path, Path]],
    component_coco_path: Path,
    materialization_summary_path: Path,
    component_manifest_path: Path,
    source_manifest_path: Path,
    split_manifest_path: Path,
    models_lock_path: Path,
    population_floor: float = DEFAULT_POPULATION_FLOOR,
    iou_threshold: float = DEFAULT_MATCH_IOU,
    major_class_count: int = 10,
    implementation_path: Path | None = None,
) -> dict[str, Any]:
    population_floor = _unit_interval(population_floor, "population floor")
    iou_threshold = _unit_interval(iou_threshold, "IoU threshold", positive=True)
    if major_class_count <= 0:
        raise ValueError("major_class_count must be positive")
    if len(candidate_specs) < 2:
        raise ValueError("at least two candidates are required for a comparison sweep")
    labels = [label for label, _status, _predictions in candidate_specs]
    if any(not label for label in labels) or len(labels) != len(set(labels)):
        raise ValueError("candidate labels must be unique non-empty strings")

    direct_source_paths: dict[str, Path] = {
        "component_coco": component_coco_path,
        "materialization_summary": materialization_summary_path,
        "component_manifest": component_manifest_path,
        "source_manifest": source_manifest_path,
        "split_manifest": split_manifest_path,
        "models_lock": models_lock_path,
    }
    if implementation_path is not None:
        direct_source_paths["audit_implementation"] = implementation_path
    for label, status_path, predictions_path in candidate_specs:
        direct_source_paths[f"{label}_diagnostic_status"] = status_path
        direct_source_paths[f"{label}_stitched_predictions"] = predictions_path
    direct_before = _snapshot(direct_source_paths)

    dataset = _load_frozen_truth(
        workspace=workspace,
        component_coco_path=component_coco_path,
        materialization_summary_path=materialization_summary_path,
        component_manifest_path=component_manifest_path,
        source_manifest_path=source_manifest_path,
        split_manifest_path=split_manifest_path,
    )
    candidates: dict[str, dict[str, Any]] = {}
    for label, status_path, predictions_path in candidate_specs:
        candidates[label] = _validate_candidate(
            workspace=workspace,
            label=label,
            status_path=status_path,
            predictions_path=predictions_path,
            dataset=dataset,
            population_floor=population_floor,
            iou_threshold=iou_threshold,
        )
    first_protocol = candidates[labels[0]]["protocol"]
    for label in labels[1:]:
        if candidates[label]["protocol"] != first_protocol:
            raise ValueError("candidate tiled protocols differ")

    for candidate in candidates.values():
        annotated = _annotate_matches(
            candidate["detections"],
            dataset["truths"],
            iou_threshold=iou_threshold,
        )
        candidate["annotated"] = annotated
        candidate["prefix"] = _build_prefix(annotated, len(dataset["truths"]))
        reproduced = _state_at(candidate["prefix"], population_floor)
        source_metrics = _mapping(
            candidate["status"].get("metrics"),
            f"{candidate['label']} source metrics",
        )
        integer_checks = {
            "true_positives": reproduced["true_positives"],
            "false_positives": reproduced["false_positives"],
            "false_negatives": reproduced["false_negatives"],
            "prediction_count": reproduced["prediction_count"],
            "ground_truth_count": reproduced["ground_truth_count"],
        }
        for key, actual in integer_checks.items():
            if source_metrics.get(key) != actual:
                raise ValueError(
                    f"{candidate['label']} threshold evaluator does not reproduce {key}"
                )
        for key in ("precision50", "recall50", "f1_50"):
            if not _close(source_metrics.get(key), reproduced[key]):
                raise ValueError(
                    f"{candidate['label']} threshold evaluator does not reproduce {key}"
                )
        candidate["population_floor_metric"] = reproduced

    thresholds, threshold_origins = _boundary_thresholds(
        candidates, population_floor
    )
    sweep_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        metrics = {
            label: _state_at(candidates[label]["prefix"], threshold)
            for label in labels
        }
        row: dict[str, Any] = {
            "score_threshold": threshold,
            "score_boundary_origins": threshold_origins[threshold],
            "metrics": metrics,
        }
        if len(labels) == 2:
            row[f"delta_{labels[1]}_minus_{labels[0]}"] = _metric_delta(
                metrics[labels[1]], metrics[labels[0]]
            )
        sweep_rows.append(row)
    best_rows = {label: _best_row(sweep_rows, label) for label in labels}
    best_thresholds = {
        label: float(best_rows[label]["score_threshold"]) for label in labels
    }
    floor_thresholds = {label: population_floor for label in labels}
    floor_comparison = _comparison_block(
        candidates=candidates,
        truths=dataset["truths"],
        thresholds=floor_thresholds,
    )
    calibrated_comparison = _comparison_block(
        candidates=candidates,
        truths=dataset["truths"],
        thresholds=best_thresholds,
    )
    cross_threshold_comparisons = {
        owner: _comparison_block(
            candidates=candidates,
            truths=dataset["truths"],
            thresholds={label: best_thresholds[owner] for label in labels},
        )
        for owner in labels
    }

    samples_by_domain: dict[str, set[str]] = defaultdict(set)
    for sample_id, metadata in dataset["sample_metadata"].items():
        samples_by_domain[str(metadata["source_domain"])].add(sample_id)
    by_domain: dict[str, Any] = {}
    for domain, sample_ids in sorted(samples_by_domain.items()):
        domain_floor = _comparison_block(
            candidates=candidates,
            truths=dataset["truths"],
            thresholds=floor_thresholds,
            sample_ids=sample_ids,
        )
        domain_best = _comparison_block(
            candidates=candidates,
            truths=dataset["truths"],
            thresholds=best_thresholds,
            sample_ids=sample_ids,
        )
        by_domain[domain] = {
            "sample_count": len(sample_ids),
            "sample_ids": sorted(sample_ids),
            "population_floor": domain_floor,
            "candidate_specific_best": domain_best,
            "calibration_impact_by_candidate": {
                label: _metric_delta(
                    domain_best["metrics"][label],
                    domain_floor["metrics"][label],
                )
                for label in labels
            },
        }

    category_support: dict[str, int] = defaultdict(int)
    official_by_category = {
        str(row["name"]): int(row["official_category_id"])
        for row in dataset["categories"]
    }
    for truth in dataset["truths"]:
        category_support[str(truth["category"])] += 1
    major_categories = sorted(
        category_support,
        key=lambda name: (
            -category_support[name],
            official_by_category[name],
            name,
        ),
    )[:major_class_count]
    by_major_class: dict[str, Any] = {}
    for category in major_categories:
        class_floor = _comparison_block(
            candidates=candidates,
            truths=dataset["truths"],
            thresholds=floor_thresholds,
            category=category,
        )
        class_best = _comparison_block(
            candidates=candidates,
            truths=dataset["truths"],
            thresholds=best_thresholds,
            category=category,
        )
        by_major_class[category] = {
            "official_category_id": official_by_category[category],
            "ground_truth_support": category_support[category],
            "population_floor": class_floor,
            "candidate_specific_best": class_best,
            "calibration_impact_by_candidate": {
                label: _metric_delta(
                    class_best["metrics"][label],
                    class_floor["metrics"][label],
                )
                for label in labels
            },
        }

    source_paths: dict[str, Path] = dict(direct_source_paths)
    protected_before = dict(direct_before)
    for label, candidate in candidates.items():
        source_paths[f"{label}_parent_status"] = candidate["parent_status_path"]
        protected_before[f"{label}_parent_status"] = candidate[
            "parent_status_sha256"
        ]
        source_paths[f"{label}_checkpoint"] = candidate["checkpoint_path"]
        protected_before[f"{label}_checkpoint"] = candidate["checkpoint_sha256"]
        for name, entry in candidate["ancillary_sources"].items():
            source_paths[f"{label}_{name}"] = _resolve_within(
                workspace, entry["path"], f"{label} {name}"
            )
            protected_before[f"{label}_{name}"] = entry["current_sha256"]

    candidate_evidence: dict[str, Any] = {}
    for label, candidate in candidates.items():
        best_metric = dict(best_rows[label]["metrics"][label])
        floor_metric = dict(candidate["population_floor_metric"])
        candidate_evidence[label] = {
            "diagnostic_status": _source_entry(
                candidate["status_path"], workspace, candidate["status_sha256"]
            ),
            "stitched_predictions": _source_entry(
                candidate["predictions_path"],
                workspace,
                candidate["predictions_sha256"],
            ),
            "formal_parent_status": {
                **_source_entry(
                    candidate["parent_status_path"],
                    workspace,
                    candidate["parent_status_sha256"],
                ),
                "status": candidate["parent_status"]["status"],
                "formal_evidence": True,
            },
            "checkpoint_bytes_verified_not_loaded": _source_entry(
                candidate["checkpoint_path"],
                workspace,
                candidate["checkpoint_sha256"],
            ),
            "historical_lineage_references_not_used_for_metrics": candidate[
                "ancillary_sources"
            ],
            "stored_prediction_count": len(candidate["detections"]),
            "observed_score_boundary_count": len(
                set(float(row["score"]) for row in candidate["detections"])
            ),
            "minimum_observed_score": min(
                float(row["score"]) for row in candidate["detections"]
            ),
            "maximum_observed_score": max(
                float(row["score"]) for row in candidate["detections"]
            ),
            "population_floor_metric": floor_metric,
            "best_validation_calibration": {
                "score_threshold": best_thresholds[label],
                "score_is_owned_prediction_boundary": True,
                "metrics": best_metric,
                "delta_from_population_floor": _metric_delta(
                    best_metric, floor_metric
                ),
            },
        }

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pcbparse_component_stitched_validation_score_threshold_sweep",
        "status": "validation_calibration_audit_complete",
        "task": "component",
        "diagnostic_only": True,
        "validation_calibration_only": True,
        "formal_evidence": False,
        "registerable": False,
        "promotion_eligible": False,
        "blessing_authorized": False,
        "recommended_for_direct_promotion": False,
        "execution_boundary": {
            "assessment_role": "validation-calibration-only",
            "prediction_reuse_only": True,
            "existing_prediction_artifacts_only": True,
            "model_artifact_deserialized": False,
            "checkpoint_loaded": False,
            "model_inference_executed": False,
            "inference_executed": False,
            "training_executed": False,
            "torch_imported_by_audit": False,
            "train_rows_used": 0,
            "test_rows_used": 0,
            "train_split_used": False,
            "test_split_used": False,
            "validation_rows_used": len(dataset["sample_ids"]),
            "models_lock_read_for_integrity_snapshot_only": True,
            "models_lock_modified": False,
            "parent_status_modified": False,
            "threshold_written_to_config": False,
            "source_artifacts_modified": [],
        },
        "contract": {
            "prediction_population": (
                "already-stitched, class-aware-NMS outputs saved by the fixed "
                "512/64 validation replay at adapter score>=0.20"
            ),
            "matching": (
                "per-class globally score-ranked one-to-one greedy matching; "
                "ties by sample_id then frozen prediction_index; IoU>=0.5 is TP"
            ),
            "scan_boundaries": (
                "exact IEEE-754 scores observed in either supplied prediction "
                "artifact; no rounding, interpolation, grid, or invented cutoff"
            ),
            "same_score_rule": "all predictions sharing a score enter as one group",
            "per_candidate_best_domain": (
                "only score boundaries observed in that candidate's own predictions"
            ),
            "best_tie_break": (
                "exact rational maximum F1, then precision, then recall, then "
                "higher owned score threshold"
            ),
            "aggregation": "micro precision/recall/F1 across classes at IoU 0.5",
        },
        "selection_boundary": {
            "split": "val",
            "sample_count": len(dataset["sample_ids"]),
            "sample_ids": list(dataset["sample_ids"]),
            "sample_ids_sha256": _canonical_sha256(dataset["sample_ids"]),
            "truth_box_count": len(dataset["truths"]),
            "truth_population_sha256": dataset["truth_population_sha256"],
            "available_prediction_population_floor": population_floor,
            "population_floor_is_scan_candidate": population_floor in threshold_origins,
            "population_below_floor_observable": False,
            "match_iou": iou_threshold,
            "candidate_threshold_count": len(thresholds),
            "candidate_thresholds_sha256": _canonical_sha256(thresholds),
            "minimum_candidate_threshold": thresholds[0],
            "maximum_candidate_threshold": thresholds[-1],
            "all_thresholds_are_observed_prediction_scores": True,
            "threshold_search_performed_on_validation": True,
            "threshold_search_performed_on_test": False,
            "thresholds_selected_and_evaluated_on_same_validation_split": True,
            "selection_bias_present": True,
            "post_nms_stitched_population_only": True,
        },
        "frozen_truth_and_split": {
            "binding_sha256": dataset["binding_sha256"],
            "truth_population_sha256": dataset["truth_population_sha256"],
            "sources": dataset["sources"],
        },
        "candidate_evidence": candidate_evidence,
        "population_floor_comparison": floor_comparison,
        "candidate_specific_best_comparison": calibrated_comparison,
        "cross_candidate_best_threshold_comparisons": cross_threshold_comparisons,
        "by_source_domain": by_domain,
        "major_class_selection": {
            "rule": (
                f"top {major_class_count} validation classes by frozen truth support; "
                "ties by official_category_id then name"
            ),
            "selected_classes": major_categories,
        },
        "by_major_class": by_major_class,
        "threshold_sweep": sweep_rows,
        "protected_files_before": protected_before,
        "protected_files_after": None,
        "protected_files_unchanged": None,
        "interpretation_boundary": {
            "allowed_use": (
                "validation-only calibration evidence for a future, explicitly "
                "configured replay or pre-registered evaluation"
            ),
            "direct_models_lock_update_allowed": False,
            "direct_candidate_promotion_allowed": False,
            "test_set_claim_allowed": False,
            "unbiased_generalization_claim_allowed": False,
            "score_below_0_20_claim_allowed": False,
            "independent_confirmation_required_before_promotion": True,
            "warning": (
                "The maximizing threshold was selected on frozen validation and "
                "is optimistically biased for that same validation population."
            ),
        },
        "limitations": [
            "Predictions below score 0.20 were never saved and cannot be calibrated by this audit.",
            "The sweep occurs after the fixed tiled replay and cross-tile NMS; it does not retune tiling or NMS.",
            "Best-threshold metrics are validation-selected and cannot be treated as independent promotion evidence.",
            "No test images, test truth, model inference, training, registration, or blessing is used.",
        ],
    }
    protected_after = _snapshot(source_paths)
    if protected_before != protected_after:
        changed = sorted(
            key
            for key in set(protected_before) | set(protected_after)
            if protected_before.get(key) != protected_after.get(key)
        )
        raise RuntimeError(f"protected source bytes changed during audit: {changed}")
    report["protected_files_after"] = protected_after
    report["protected_files_unchanged"] = True
    report["audit_payload_sha256"] = _canonical_sha256(report)
    return report


def _atomic_write_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    destination_created = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            with path.open("x", encoding="utf-8", newline="\n") as destination:
                destination_created = True
                destination.write(Path(temporary).read_text(encoding="utf-8"))
                destination.flush()
                os.fsync(destination.fileno())
        except Exception:
            if destination_created and path.exists():
                path.unlink()
            raise
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    workspace = arguments.workspace_root.expanduser().resolve(strict=True)
    raw_specs = arguments.candidate or [
        (
            "v4",
            "work/eval-component-v4-tiled512-cpu-full/diagnostic_status.json",
            "work/eval-component-v4-tiled512-cpu-full/predictions.jsonl",
        ),
        (
            "v5",
            "work/eval-component-v5-tiled512-cpu-full/diagnostic_status.json",
            "work/eval-component-v5-tiled512-cpu-full/predictions.jsonl",
        ),
    ]
    candidate_specs = [
        (
            str(label),
            _resolve_within(workspace, status, f"{label} diagnostic status"),
            _resolve_within(workspace, predictions, f"{label} predictions"),
        )
        for label, status, predictions in raw_specs
    ]
    component_coco_path = _resolve_within(
        workspace, arguments.component_coco, "Component COCO"
    )
    materialization_summary_path = _resolve_within(
        workspace, arguments.materialization_summary, "materialization summary"
    )
    component_manifest_path = _resolve_within(
        workspace, arguments.component_manifest, "component manifest"
    )
    source_manifest_path = _resolve_within(
        workspace, arguments.source_manifest, "source manifest"
    )
    split_manifest_path = _resolve_within(
        workspace, arguments.split_manifest, "split manifest"
    )
    models_lock_path = _resolve_within(
        workspace, arguments.models_lock, "models lock"
    )
    output_path = _resolve_within(
        workspace, arguments.output, "audit output", must_exist=False
    )
    implementation_path = Path(__file__).resolve(strict=True)
    report = build_audit(
        workspace=workspace,
        candidate_specs=candidate_specs,
        component_coco_path=component_coco_path,
        materialization_summary_path=materialization_summary_path,
        component_manifest_path=component_manifest_path,
        source_manifest_path=source_manifest_path,
        split_manifest_path=split_manifest_path,
        models_lock_path=models_lock_path,
        population_floor=arguments.population_floor,
        iou_threshold=arguments.iou_threshold,
        major_class_count=arguments.major_class_count,
        implementation_path=implementation_path,
    )
    _atomic_write_new(output_path, report)
    print(output_path)
    print(f"audit_payload_sha256={report['audit_payload_sha256']}")
    print(f"audit_file_sha256={_sha256(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
