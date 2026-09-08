from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import audit_dataset
from .hashing import sha256_file
from .manifest import read_jsonl, write_json, write_jsonl


TASK_NAMES = (
    "component_detection",
    "pin_keypoints",
    "ocr_text",
    "wire_paths",
    "junction_patches",
)
TASK_FILES = {task: f"{task}.jsonl" for task in TASK_NAMES}


@dataclass(slots=True)
class DerivedDatasetResult:
    records: dict[str, list[dict[str, Any]]]
    summary: dict[str, Any]


def derive_training_datasets(
    src: str | Path,
    *,
    version: str,
    patch_size: int = 64,
    split_manifest: str | Path | None = None,
    expected_count: int = 300,
) -> DerivedDatasetResult:
    """Derive five deterministic training manifests from CVJsonStd JSON.

    Dense wire coordinates and image patches are deliberately not copied. Wire
    manifests carry JSON Pointer references, and junction records describe the
    patch window that a downstream loader should crop lazily.
    """

    if not isinstance(version, str) or not version.strip():
        raise ValueError("version must be a non-empty string")
    if type(patch_size) is not int or patch_size <= 0:
        raise ValueError("patch_size must be a positive integer")
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")

    source_root = Path(src).expanduser().resolve(strict=False)
    source_audit = audit_dataset(source_root, expected_count=expected_count)
    if not source_audit.ok:
        errors = [
            issue
            for issue in source_audit.report["issues"]
            if issue["severity"] == "error"
        ]
        first = errors[0] if errors else {"message": "unknown source audit error"}
        raise ValueError(
            f"source dataset audit failed with {len(errors)} error(s): "
            f"{first['message']} Action: {first.get('action', 'repair the dataset')}"
        )

    sample_ids = {record["sample_id"] for record in source_audit.records}
    split_path = (
        Path(split_manifest).expanduser().resolve(strict=False)
        if split_manifest is not None
        else None
    )
    split_assignments = _load_split_assignments(split_path, sample_ids)
    if split_assignments is not None:
        _validate_duplicate_split_groups(source_audit.records, split_assignments)

    config: dict[str, Any] = {
        "schema_version": 1,
        "derivation_version": version,
        "tasks": list(TASK_NAMES),
        "expected_sample_count": expected_count,
        "patch_size": patch_size,
        "bbox_policy": "preserve_raw_and_clip_to_canvas",
        "ocr_policy": "strict_raw_no_normalization",
        "pin_policy": "single_active_owner_non_orphan",
        "wire_policy": "lazy_json_pointer_no_dense_coordinates",
        "junction_policy": "sparse_keypoints_plus_shared_skeleton_crosses",
        "split_manifest_sha256": sha256_file(split_path) if split_path else None,
    }
    config_hash = hashlib.sha256(_canonical_bytes(config)).hexdigest()

    task_records: dict[str, list[dict[str, Any]]] = {
        task: [] for task in TASK_NAMES
    }
    skipped: Counter[str] = Counter()
    domains: Counter[str] = Counter()

    for sample_manifest in sorted(
        source_audit.records, key=lambda record: record["sample_id"]
    ):
        sample_id = sample_manifest["sample_id"]
        source_domain = sample_manifest["source_class"]
        domains[source_domain] += 1
        json_relative_path = sample_manifest["files"]["json"]["path"]
        json_path = source_root / Path(json_relative_path)
        annotation = _read_strict_json(json_path)
        canvas = sample_manifest["canvas"]
        common = {
            "schema_version": 1,
            "derivation_version": version,
            "config_hash": config_hash,
            "sample_id": sample_id,
            "source_domain": source_domain,
            "source_json": json_relative_path,
            "source_image": sample_manifest["files"]["image"]["path"],
            "duplicate_group": sample_manifest["duplicate_group"],
        }
        if split_assignments is not None:
            common["split"] = split_assignments[sample_id]

        components = annotation["components"]
        component_records, component_skips = _derive_components(
            common, components, canvas
        )
        task_records["component_detection"].extend(component_records)
        skipped.update(component_skips)

        pin_records, pin_skips = _derive_pins(
            common, annotation["pins"], components, canvas
        )
        task_records["pin_keypoints"].extend(pin_records)
        skipped.update(pin_skips)

        text_records, text_skips = _derive_texts(
            common, annotation["texts"], canvas
        )
        task_records["ocr_text"].extend(text_records)
        skipped.update(text_skips)

        active_edges: dict[str, dict[str, Any]] = {}
        wire_records: list[dict[str, Any]] = []
        for edge_key, edge in sorted(annotation["edges"].items()):
            if not isinstance(edge, dict):
                skipped["invalid_edge_objects"] += 1
                continue
            if edge.get("state") != "active":
                skipped["inactive_edges"] += 1
                continue
            active_edges[edge_key] = edge
            wire_records.append(_derive_wire_record(common, edge_key, edge))
        task_records["wire_paths"].extend(wire_records)

        junction_records, junction_skips = _derive_junctions(
            common, active_edges, canvas, patch_size
        )
        task_records["junction_patches"].extend(junction_records)
        skipped.update(junction_skips)

    for records in task_records.values():
        records.sort(key=lambda record: record["record_id"])

    issue_flag_counts: dict[str, dict[str, int]] = {}
    split_counts: dict[str, dict[str, int]] = {}
    junction_type_counts: Counter[str] = Counter()
    for task, records in task_records.items():
        flag_counter: Counter[str] = Counter()
        task_split_counter: Counter[str] = Counter()
        for record in records:
            flag_counter.update(record.get("issue_flags", []))
            if "split" in record:
                task_split_counter[record["split"]] += 1
            if task == "junction_patches":
                junction_type_counts.update(record["candidate_types"])
        issue_flag_counts[task] = dict(sorted(flag_counter.items()))
        if split_assignments is not None:
            split_counts[task] = dict(sorted(task_split_counter.items()))

    counts = {task: len(task_records[task]) for task in TASK_NAMES}
    summary: dict[str, Any] = {
        "schema_version": 1,
        "derivation_version": version,
        "config": config,
        "config_hash": config_hash,
        "config_hash_algorithm": "sha256",
        "source_root": str(source_root),
        "split_manifest": str(split_path) if split_path else None,
        "sample_count": len(source_audit.records),
        "source_domains": dict(sorted(domains.items())),
        "counts": counts,
        "total_records": sum(counts.values()),
        "task_files": dict(TASK_FILES),
        "issue_flags": issue_flag_counts,
        "skipped": dict(sorted(skipped.items())),
        "junction_candidate_types": dict(sorted(junction_type_counts.items())),
        "split_counts": split_counts,
        "source_audit": {
            "errors": source_audit.report["summary"]["errors"],
            "warnings": source_audit.report["summary"]["warnings"],
            "issues_by_code": source_audit.report["summary"]["issues_by_code"],
        },
    }
    return DerivedDatasetResult(records=task_records, summary=summary)


def write_derived_datasets(
    result: DerivedDatasetResult, out: str | Path
) -> dict[str, Path]:
    """Atomically publish each JSONL file, then the summary commit marker."""

    output_root = Path(out).expanduser().resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for task in TASK_NAMES:
        path = output_root / TASK_FILES[task]
        write_jsonl(result.records[task], path)
        written[task] = path
    summary_path = output_root / "summary.json"
    # The summary is written last: its presence describes one complete set of
    # atomically replaced task files and contains their exact record counts.
    write_json(result.summary, summary_path)
    written["summary"] = summary_path
    return written


def _read_strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r}")

    value = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if not isinstance(value, dict):
        raise ValueError(f"annotation root is not an object: {path}")
    return value


def _load_split_assignments(
    split_path: Path | None, sample_ids: set[str]
) -> dict[str, str] | None:
    if split_path is None:
        return None
    records = read_jsonl(split_path)
    assignments: dict[str, str] = {}
    for record in records:
        split = record.get("split")
        if not isinstance(split, str) or not split:
            raise ValueError(
                f"split manifest record {record['sample_id']!r} lacks a string split"
            )
        assignments[record["sample_id"]] = split
    actual_ids = set(assignments)
    missing = sorted(sample_ids - actual_ids)
    extra = sorted(actual_ids - sample_ids)
    if missing or extra:
        raise ValueError(
            "split manifest sample coverage mismatch: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return assignments


def _validate_duplicate_split_groups(
    source_records: list[dict[str, Any]], assignments: dict[str, str]
) -> None:
    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in source_records:
        group_splits[record["duplicate_group"]].add(assignments[record["sample_id"]])
    leaking = {
        group_id: sorted(splits)
        for group_id, splits in group_splits.items()
        if len(splits) > 1
    }
    if leaking:
        first_group = sorted(leaking)[0]
        raise ValueError(
            f"split manifest leaks duplicate group {first_group}: {leaking[first_group]}"
        )


def _base_record(
    common: dict[str, Any], task: str, record_id: str
) -> dict[str, Any]:
    record = dict(common)
    record.update({"task": task, "record_id": record_id, "issue_flags": []})
    return record


def _derive_components(
    common: dict[str, Any],
    components: dict[str, Any],
    canvas: dict[str, int],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for component_key, component in sorted(components.items()):
        if not isinstance(component, dict):
            skipped["invalid_component_objects"] += 1
            continue
        if component.get("state") != "active":
            skipped["inactive_components"] += 1
            continue
        raw_bbox, bbox, flags = _clip_bbox(
            component.get("location"), canvas, "bbox"
        )
        raw_real_bbox, real_bbox, real_flags = _clip_bbox(
            component.get("realBBox"), canvas, "real_bbox"
        )
        record = _base_record(
            common,
            "component_detection",
            f"{common['sample_id']}:component:{component_key}",
        )
        record.update(
            {
                "component_key": component_key,
                "component_id": component.get("cpntId"),
                "state": component.get("state"),
                "category": component.get("category"),
                "category_id": component.get("categoryId"),
                "raw_bbox": raw_bbox,
                "bbox": bbox,
                "raw_real_bbox": raw_real_bbox,
                "real_bbox": real_bbox,
                "rotation": component.get("rotation"),
                "pin_ids": sorted(
                    component.get("pins", {})
                    if isinstance(component.get("pins"), dict)
                    else []
                ),
                "associated_text_ids": sorted(
                    component.get("associatedTexts", {})
                    if isinstance(component.get("associatedTexts"), dict)
                    else []
                ),
            }
        )
        record["issue_flags"] = sorted(set(flags + real_flags))
        records.append(record)
    return records, skipped


def _derive_pins(
    common: dict[str, Any],
    root_pins: dict[str, Any],
    components: dict[str, Any],
    canvas: dict[str, int],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    owners: dict[str, list[str]] = defaultdict(list)
    for component_key, component in sorted(components.items()):
        if not isinstance(component, dict):
            continue
        nested_pins = component.get("pins")
        if isinstance(nested_pins, dict):
            for pin_key in nested_pins:
                owners[pin_key].append(component_key)

    nested_only = set(owners) - set(root_pins)
    skipped["nested_pins_missing_root_entry"] += len(nested_only)
    for pin_key, pin in sorted(root_pins.items()):
        pin_owners = sorted(set(owners.get(pin_key, [])))
        if not pin_owners:
            skipped["unowned_root_pins"] += 1
            continue
        if len(pin_owners) != 1:
            skipped["multi_owned_pins"] += 1
            continue
        owner_key = pin_owners[0]
        owner = components.get(owner_key)
        if not isinstance(owner, dict) or owner.get("state") != "active":
            skipped["pins_with_inactive_owner"] += 1
            continue
        if not isinstance(pin, dict):
            skipped["invalid_pin_objects"] += 1
            continue
        tip = _point(pin.get("tip"))
        base = _point(pin.get("base"))
        if tip is None or base is None:
            skipped["invalid_pin_geometry"] += 1
            continue

        flags: list[str] = []
        if not _point_in_canvas(tip, canvas):
            flags.append("tip_out_of_bounds")
        if not _point_in_canvas(base, canvas):
            flags.append("base_out_of_bounds")
        orientation = _pin_orientation(tip, base)
        if orientation["label"] == "overlap":
            flags.append("degenerate_orientation")

        label_refs = {
            "pin_number_by_ocr": pin.get("pinNumberByOCR"),
            "pin_number_by_vlm": pin.get("pinNumberByVLM"),
            "pin_name_by_ocr": pin.get("pinNameByOCR"),
            "pin_name_by_vlm": pin.get("pinNameByVLM"),
        }
        missing_mask = {
            key: value is None or value == "" for key, value in label_refs.items()
        }
        record = _base_record(
            common,
            "pin_keypoints",
            f"{common['sample_id']}:pin:{pin_key}",
        )
        record.update(
            {
                "pin_key": pin_key,
                "pin_id": pin.get("pinId"),
                "owner_component_key": owner_key,
                "owner_component_id": owner.get("cpntId"),
                "owner_category": owner.get("category"),
                "edge_id": pin.get("edgeId"),
                "tip": list(tip),
                "base": list(base),
                "orientation": orientation,
                "label_refs": label_refs,
                "missing_label_mask": missing_mask,
            }
        )
        record["issue_flags"] = sorted(flags)
        records.append(record)
    return records, skipped


def _derive_texts(
    common: dict[str, Any],
    texts: dict[str, Any],
    canvas: dict[str, int],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for text_key, text in sorted(texts.items()):
        if not isinstance(text, dict):
            skipped["invalid_text_objects"] += 1
            continue
        raw_bbox, bbox, flags = _clip_bbox(text.get("location"), canvas, "bbox")
        record = _base_record(
            common, "ocr_text", f"{common['sample_id']}:text:{text_key}"
        )
        record.update(
            {
                "text_key": text_key,
                "text_id": text.get("tid"),
                "state": text.get("state"),
                "category": text.get("category"),
                "category_id": text.get("categoryId"),
                # Deliberately preserve exact values; do not strip or normalize.
                "content": text.get("content"),
                "details": text.get("details"),
                "normalization_applied": False,
                "raw_bbox": raw_bbox,
                "bbox": bbox,
            }
        )
        record["issue_flags"] = sorted(flags)
        records.append(record)
    return records, skipped


def _derive_wire_record(
    common: dict[str, Any], edge_key: str, edge: dict[str, Any]
) -> dict[str, Any]:
    key_points = edge.get("keyPoints")
    key_points = key_points if isinstance(key_points, dict) else {}
    key_point_paths = edge.get("keyPointPaths")
    key_point_paths = key_point_paths if isinstance(key_point_paths, dict) else {}
    skeleton = key_points.get("skeleton")
    skeleton = skeleton if isinstance(skeleton, dict) else {}
    escaped_edge = _json_pointer_escape(edge_key)

    flags: list[str] = []
    if not key_point_paths:
        flags.append("missing_key_point_paths")
    if not skeleton:
        flags.append("missing_skeleton")

    connections: list[dict[str, Any]] = []
    raw_connections = edge.get("connect")
    if isinstance(raw_connections, dict):
        for pin_key, connection in sorted(raw_connections.items()):
            connection = connection if isinstance(connection, dict) else {}
            connections.append(
                {
                    "pin_key": pin_key,
                    "pin_id": connection.get("pinId"),
                    "component_id": connection.get("cpntId"),
                    "location": connection.get("location"),
                }
            )

    record = _base_record(
        common, "wire_paths", f"{common['sample_id']}:edge:{edge_key}"
    )
    record.update(
        {
            "edge_key": edge_key,
            "edge_id": edge.get("edgeId"),
            "state": edge.get("state"),
            "category": edge.get("category"),
            "netlabel": edge.get("netlabel"),
            "is_flywire": edge.get("isFlywire"),
            "connections": connections,
            "associated_text_ids": sorted(
                edge.get("associatedTexts", {})
                if isinstance(edge.get("associatedTexts"), dict)
                else []
            ),
            "original_coordinates_count": edge.get("originalCoordinatesCount"),
            "key_point_paths_ref": {
                "json_pointer": f"/edges/{escaped_edge}/keyPointPaths",
                "geometry": _path_geometry_summary(key_point_paths),
            },
            "skeleton_ref": {
                "json_pointer": f"/edges/{escaped_edge}/keyPoints/skeleton",
                "geometry": _point_geometry_summary(skeleton),
            },
            "dense_coordinates_included": False,
        }
    )
    record["issue_flags"] = sorted(flags)
    return record


def _derive_junctions(
    common: dict[str, Any],
    active_edges: dict[str, dict[str, Any]],
    canvas: dict[str, int],
    patch_size: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    candidates: dict[tuple[int | float, int | float], dict[str, Any]] = {}
    skeleton_owners: dict[tuple[int | float, int | float], set[str]] = defaultdict(set)
    skeleton_geometry: dict[tuple[int | float, int | float], set[str]] = defaultdict(set)
    skipped: Counter[str] = Counter()
    kind_map = {
        "endPoints": "endpoint",
        "divergePoints": "diverge",
        "turnPoints": "turn",
    }

    for edge_key, edge in sorted(active_edges.items()):
        escaped_edge = _json_pointer_escape(edge_key)
        key_points = edge.get("keyPoints")
        if not isinstance(key_points, dict):
            skipped["edges_without_keypoints"] += 1
            continue
        for source_kind, candidate_kind in kind_map.items():
            geometry_map = key_points.get(source_kind)
            if not isinstance(geometry_map, dict):
                continue
            for geometry, raw_points in sorted(geometry_map.items()):
                if not isinstance(raw_points, list):
                    continue
                for index, raw_point in enumerate(raw_points):
                    point = _point(raw_point)
                    if point is None:
                        skipped["invalid_junction_points"] += 1
                        continue
                    candidate = _candidate(candidates, point)
                    candidate["types"].add(candidate_kind)
                    candidate["edge_ids"].add(edge_key)
                    candidate["geometry_types"].add(geometry)
                    candidate["source_refs"].append(
                        {
                            "kind": candidate_kind,
                            "edge_key": edge_key,
                            "geometry": geometry,
                            "point_index": index,
                            "json_pointer": (
                                f"/edges/{escaped_edge}/keyPoints/"
                                f"{source_kind}/{_json_pointer_escape(geometry)}/{index}"
                            ),
                        }
                    )

        skeleton_map = key_points.get("skeleton")
        if not isinstance(skeleton_map, dict):
            continue
        for geometry, raw_points in sorted(skeleton_map.items()):
            if not isinstance(raw_points, list):
                continue
            for raw_point in raw_points:
                point = _point(raw_point)
                if point is None:
                    skipped["invalid_skeleton_points"] += 1
                    continue
                skeleton_owners[point].add(edge_key)
                skeleton_geometry[point].add(geometry)

    for point, edge_ids in sorted(skeleton_owners.items(), key=_point_sort_key_item):
        if len(edge_ids) < 2:
            continue
        candidate = _candidate(candidates, point)
        candidate["types"].add("cross")
        candidate["edge_ids"].update(edge_ids)
        candidate["geometry_types"].update(skeleton_geometry[point])
        for edge_key in sorted(edge_ids):
            candidate["source_refs"].append(
                {
                    "kind": "cross",
                    "edge_key": edge_key,
                    "json_pointer": (
                        f"/edges/{_json_pointer_escape(edge_key)}/keyPoints/skeleton"
                    ),
                }
            )

    records: list[dict[str, Any]] = []
    for point, candidate in sorted(candidates.items(), key=_point_sort_key_item):
        raw_patch_bbox, patch_bbox, padding, patch_flags = _patch_bbox(
            point, canvas, patch_size
        )
        flags = list(patch_flags)
        if not _point_in_canvas(point, canvas):
            flags.append("center_out_of_bounds")
        point_digest = hashlib.sha256(_canonical_bytes(list(point))).hexdigest()[:16]
        record = _base_record(
            common,
            "junction_patches",
            f"{common['sample_id']}:junction:{point_digest}",
        )
        skeleton_edges = sorted(skeleton_owners.get(point, set()))
        candidate_edge_keys = sorted(candidate["edge_ids"])
        candidate_edge_ids = [
            active_edges[edge_key].get("edgeId") for edge_key in candidate_edge_keys
        ]
        skeleton_edge_ids = [
            active_edges[edge_key].get("edgeId") for edge_key in skeleton_edges
        ]
        unique_refs = {
            _canonical_text(source_ref): source_ref
            for source_ref in candidate["source_refs"]
        }
        record.update(
            {
                "center": list(point),
                "candidate_types": sorted(candidate["types"]),
                "edge_keys": candidate_edge_keys,
                "edge_ids": candidate_edge_ids,
                "geometry_types": sorted(candidate["geometry_types"]),
                "patch_size": patch_size,
                "raw_patch_bbox": raw_patch_bbox,
                "patch_bbox": patch_bbox,
                "padding": padding,
                "cross": {
                    "is_cross": "cross" in candidate["types"],
                    "edge_count": len(skeleton_edges),
                    "edge_keys": skeleton_edges,
                    "edge_ids": skeleton_edge_ids,
                },
                "source_refs": [unique_refs[key] for key in sorted(unique_refs)],
                "image_crop_included": False,
            }
        )
        record["issue_flags"] = sorted(set(flags))
        records.append(record)
    return records, skipped


def _candidate(
    candidates: dict[tuple[int | float, int | float], dict[str, Any]],
    point: tuple[int | float, int | float],
) -> dict[str, Any]:
    return candidates.setdefault(
        point,
        {
            "types": set(),
            "edge_ids": set(),
            "geometry_types": set(),
            "source_refs": [],
        },
    )


def _clip_bbox(
    raw_value: Any, canvas: dict[str, int], flag_prefix: str
) -> tuple[Any, dict[str, int | float] | None, list[str]]:
    raw_bbox = _json_safe_copy(raw_value)
    if not isinstance(raw_value, dict):
        return raw_bbox, None, [f"{flag_prefix}_invalid"]
    keys = ("x1", "y1", "x2", "y2")
    if any(not _is_number(raw_value.get(key)) for key in keys):
        return raw_bbox, None, [f"{flag_prefix}_invalid"]

    clipped = {
        "x1": _clamp(raw_value["x1"], 0, canvas["width"]),
        "y1": _clamp(raw_value["y1"], 0, canvas["height"]),
        "x2": _clamp(raw_value["x2"], 0, canvas["width"]),
        "y2": _clamp(raw_value["y2"], 0, canvas["height"]),
    }
    flags: list[str] = []
    if any(clipped[key] != raw_value[key] for key in keys):
        flags.append(f"{flag_prefix}_clipped")
    if clipped["x2"] <= clipped["x1"] or clipped["y2"] <= clipped["y1"]:
        flags.append(f"{flag_prefix}_degenerate")
    return raw_bbox, clipped, flags


def _patch_bbox(
    center: tuple[int | float, int | float],
    canvas: dict[str, int],
    patch_size: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    half = patch_size // 2
    raw = {
        "x1": center[0] - half,
        "y1": center[1] - half,
        "x2": center[0] - half + patch_size,
        "y2": center[1] - half + patch_size,
    }
    clipped = {
        "x1": _clamp(raw["x1"], 0, canvas["width"]),
        "y1": _clamp(raw["y1"], 0, canvas["height"]),
        "x2": _clamp(raw["x2"], 0, canvas["width"]),
        "y2": _clamp(raw["y2"], 0, canvas["height"]),
    }
    padding = {
        "left": clipped["x1"] - raw["x1"],
        "top": clipped["y1"] - raw["y1"],
        "right": raw["x2"] - clipped["x2"],
        "bottom": raw["y2"] - clipped["y2"],
    }
    flags = ["patch_bbox_clipped"] if clipped != raw else []
    return raw, clipped, padding, flags


def _point(value: Any) -> tuple[int | float, int | float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    if not _is_number(value[0]) or not _is_number(value[1]):
        return None
    return value[0], value[1]


def _point_in_canvas(
    point: tuple[int | float, int | float], canvas: dict[str, int]
) -> bool:
    return 0 <= point[0] < canvas["width"] and 0 <= point[1] < canvas["height"]


def _pin_orientation(
    tip: tuple[int | float, int | float],
    base: tuple[int | float, int | float],
) -> dict[str, Any]:
    dx = tip[0] - base[0]
    dy = tip[1] - base[1]
    magnitude = math.hypot(dx, dy)
    if magnitude == 0:
        label = "overlap"
        unit = [0.0, 0.0]
        angle = None
    else:
        unit = [dx / magnitude, dy / magnitude]
        angle = math.degrees(math.atan2(dy, dx))
        if abs(dx) > abs(dy):
            label = "right" if dx > 0 else "left"
        elif abs(dy) > abs(dx):
            label = "down" if dy > 0 else "up"
        else:
            vertical = "down" if dy > 0 else "up"
            horizontal = "right" if dx > 0 else "left"
            label = f"{vertical}_{horizontal}"
    return {
        "label": label,
        "vector": [dx, dy],
        "unit_vector": unit,
        "angle_degrees": angle,
        "image_y_axis": "down",
    }


def _path_geometry_summary(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for geometry, raw_paths in sorted(value.items()):
        if not isinstance(raw_paths, list):
            result[geometry] = {"path_count": 0, "point_count": 0, "valid": False}
            continue
        point_count = sum(
            len(path) for path in raw_paths if isinstance(path, list)
        )
        result[geometry] = {
            "path_count": len(raw_paths),
            "point_count": point_count,
            "valid": all(isinstance(path, list) for path in raw_paths),
        }
    return result


def _point_geometry_summary(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for geometry, raw_points in sorted(value.items()):
        result[geometry] = {
            "point_count": len(raw_points) if isinstance(raw_points, list) else 0,
            "valid": isinstance(raw_points, list),
        }
    return result


def _point_sort_key_item(
    item: tuple[tuple[int | float, int | float], Any]
) -> tuple[float, float, str]:
    point = item[0]
    return float(point[1]), float(point[0]), _canonical_text(list(point))


def _json_pointer_escape(value: str) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _clamp(value: int | float, minimum: int, maximum: int) -> int | float:
    return min(maximum, max(minimum, value))


def _json_safe_copy(value: Any) -> Any:
    # Round-tripping avoids accidentally sharing mutable source annotation data
    # and proves that the preserved raw value remains strict JSON.
    return json.loads(_canonical_text(value))


def _canonical_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_bytes(value: Any) -> bytes:
    return _canonical_text(value).encode("utf-8")
