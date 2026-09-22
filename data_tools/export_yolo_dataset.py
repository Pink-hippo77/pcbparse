from __future__ import annotations

"""Export Dataset300 Component/Text manifests to deterministic YOLO datasets.

The converter deliberately operates from the already-derived, hash-bound JSONL
manifests.  Source-image split assignment is inherited before tiling, and every
tile records enough geometry to map predictions back to the original canvas.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw

from ._bootstrap import REPOSITORY_ROOT, default_dataset_root


SCHEMA_VERSION = 1
CONVERTER_VERSION = "pcbnet-yolo-d1-v1"
SUPPORTED_SPLITS = ("train", "val")


@dataclass(frozen=True)
class Tile:
    index: int
    x1: int
    y1: int
    x2: int
    y2: int
    output_width: int
    output_height: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def tile_id(self) -> str:
        return f"x{self.x1:05d}_y{self.y1:05d}"


@dataclass(frozen=True)
class BoxDecision:
    record_id: str
    class_id: int
    clipped_xyxy: tuple[float, float, float, float]
    visible_fraction: float
    truncated: bool
    center_in_tile: bool
    visibility_policy: str = "unclassified"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export hash-bound Dataset300 Component and Text annotations to "
            "YOLO tiles, plus a KiCad/long-tail diagnostic validation set."
        )
    )
    parser.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument(
        "--component-manifest",
        type=Path,
        default=Path("data/derived/dataset300-v2/component_detection.jsonl"),
    )
    parser.add_argument(
        "--text-manifest",
        type=Path,
        default=Path("data/derived/dataset300-v2/ocr_text.jsonl"),
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
        "--output",
        type=Path,
        default=Path("data/derived/pcbnet-upgrade-v1"),
    )
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--tile-overlap", type=int, default=96)
    parser.add_argument(
        "--component-min-visible-fraction",
        type=float,
        default=0.25,
        help=(
            "Preferred visible fraction for a clipped Component label. Lower-visibility "
            "fragments are retained with an explicit fallback policy because YOLO has no "
            "arbitrary ignore masks."
        ),
    )
    parser.add_argument(
        "--text-min-visible-fraction",
        type=float,
        default=0.50,
        help=(
            "Preferred visible fraction for a clipped single-class Text detection box. "
            "Lower-visibility fragments are retained and marked as safety labels."
        ),
    )
    parser.add_argument("--minimum-clipped-box-pixels", type=float, default=2.0)
    parser.add_argument(
        "--long-tail-max-train-instances",
        type=int,
        default=50,
        help="Classes at or below this train-object count are long-tail classes.",
    )
    parser.add_argument(
        "--special-validation-split",
        choices=("val", "test"),
        default="val",
        help=(
            "Frozen source split used for the diagnostic KiCad/long-tail subset. "
            "The default never consumes training rows."
        ),
    )
    parser.add_argument(
        "--skip-source-hash-check",
        action="store_true",
        help="Do not recompute source image SHA-256 values (not recommended).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the exact output directory after safety checks.",
    )
    return parser


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(
                    raw,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid {label} JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{label} row {line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{label} is empty: {path}")
    return rows


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_file(workspace: Path, raw: Path, label: str) -> Path:
    path = raw.expanduser()
    resolved = (path if path.is_absolute() else workspace / path).resolve(strict=False)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"{label} is missing or empty: {resolved}")
    return resolved


def _resolve_directory(workspace: Path, raw: Path, label: str) -> Path:
    path = raw.expanduser()
    resolved = (path if path.is_absolute() else workspace / path).resolve(strict=False)
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} directory is missing: {resolved}")
    return resolved


def _resolve_output(workspace: Path, raw: Path) -> Path:
    path = raw.expanduser()
    resolved = (path if path.is_absolute() else workspace / path).resolve(strict=False)
    if resolved == workspace:
        raise ValueError("output must not be the workspace root")
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("output must stay inside the workspace") from exc
    return resolved


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical_bytes(dict(row)).decode("utf-8"))
            handle.write("\n")


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _fraction(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number in [0,1]")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be finite and in [0,1]")
    return number


def _tile_origins(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    origins = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if origins[-1] != final:
        origins.append(final)
    return origins


def build_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[Tile]:
    _strict_int(width, "image width", minimum=1)
    _strict_int(height, "image height", minimum=1)
    _strict_int(tile_size, "tile size", minimum=1)
    _strict_int(overlap, "tile overlap")
    if overlap >= tile_size:
        raise ValueError("tile overlap must be smaller than tile size")
    tiles: list[Tile] = []
    for y1 in _tile_origins(height, tile_size, overlap):
        for x1 in _tile_origins(width, tile_size, overlap):
            tiles.append(
                Tile(
                    index=len(tiles),
                    x1=x1,
                    y1=y1,
                    x2=min(width, x1 + tile_size),
                    y2=min(height, y1 + tile_size),
                    output_width=tile_size,
                    output_height=tile_size,
                )
            )
    return tiles


def _box(record: Mapping[str, Any]) -> tuple[float, float, float, float]:
    raw = record.get("bbox")
    if not isinstance(raw, Mapping):
        raise ValueError(f"record {record.get('record_id')!r} has no bbox object")
    try:
        values = tuple(float(raw[key]) for key in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"record {record.get('record_id')!r} has an invalid bbox") from exc
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"record {record.get('record_id')!r} bbox is not finite")
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError(f"record {record.get('record_id')!r} bbox has non-positive area")
    return values


def _decision(
    record: Mapping[str, Any],
    tile: Tile,
    *,
    class_id: int,
) -> BoxDecision | None:
    x1, y1, x2, y2 = _box(record)
    ix1 = max(x1, float(tile.x1))
    iy1 = max(y1, float(tile.y1))
    ix2 = min(x2, float(tile.x2))
    iy2 = min(y2, float(tile.y2))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    area = (x2 - x1) * (y2 - y1)
    intersection = (ix2 - ix1) * (iy2 - iy1)
    fraction = intersection / area
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    center_in_tile = (
        tile.x1 <= center_x < tile.x2 and tile.y1 <= center_y < tile.y2
    )
    return BoxDecision(
        record_id=str(record["record_id"]),
        class_id=class_id,
        clipped_xyxy=(ix1, iy1, ix2, iy2),
        visible_fraction=fraction,
        truncated=fraction < 1.0 - 1e-12,
        center_in_tile=center_in_tile,
    )


def _format_yolo(decision: BoxDecision, tile: Tile) -> str:
    x1, y1, x2, y2 = decision.clipped_xyxy
    local_x1 = x1 - tile.x1
    local_y1 = y1 - tile.y1
    local_x2 = x2 - tile.x1
    local_y2 = y2 - tile.y1
    width = local_x2 - local_x1
    height = local_y2 - local_y1
    cx = (local_x1 + local_x2) / 2.0 / tile.output_width
    cy = (local_y1 + local_y2) / 2.0 / tile.output_height
    nw = width / tile.output_width
    nh = height / tile.output_height
    values = (cx, cy, nw, nh)
    if any(not 0.0 <= value <= 1.0 + 1e-12 for value in values):
        raise ValueError(f"normalized YOLO box is outside [0,1]: {values}")
    if nw <= 0.0 or nh <= 0.0:
        raise ValueError("normalized YOLO box has non-positive size")
    return f"{decision.class_id} " + " ".join(f"{value:.10f}" for value in values)


def _category_table(component_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    source: dict[tuple[int, str], set[str]] = defaultdict(set)
    names_by_id: dict[int, str] = {}
    ids_by_name: dict[str, int] = {}
    for row in component_rows:
        try:
            source_id = int(str(row["category_id"]))
            name = str(row["category"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid Component category in {row.get('record_id')!r}") from exc
        if source_id in names_by_id and names_by_id[source_id] != name:
            raise ValueError(f"Component category id {source_id} maps to multiple names")
        if name in ids_by_name and ids_by_name[name] != source_id:
            raise ValueError(f"Component category name {name!r} maps to multiple ids")
        names_by_id[source_id] = name
        ids_by_name[name] = source_id
        source[(source_id, name)].add(str(row.get("record_id", "")))
    table = []
    for yolo_id, (source_id, name) in enumerate(sorted(source)):
        table.append(
            {
                "yolo_class_id": yolo_id,
                "source_category_id": str(source_id),
                "source_category_name": name,
                "record_count": len(source[(source_id, name)]),
            }
        )
    if not table:
        raise ValueError("Component category table is empty")
    return table


def _class_maps(category_table: Sequence[Mapping[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    by_name = {
        str(item["source_category_name"]): int(item["yolo_class_id"])
        for item in category_table
    }
    by_source_id = {
        str(item["source_category_id"]): int(item["yolo_class_id"])
        for item in category_table
    }
    return by_name, by_source_id


def _validate_row_common(row: Mapping[str, Any], task: str) -> None:
    if row.get("task") != task:
        raise ValueError(f"record {row.get('record_id')!r} has task {row.get('task')!r}, expected {task!r}")
    for key in ("record_id", "sample_id", "source_image", "split", "source_domain"):
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"record lacks non-empty {key}: {row.get('record_id')!r}")
    if row["split"] not in {"train", "val", "test"}:
        raise ValueError(f"record has unsupported split {row['split']!r}")
    _box(row)


def _index_sources(
    source_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    split_by_sample: dict[str, str] = {}
    for row in split_rows:
        sample_id = str(row.get("sample_id", ""))
        split = str(row.get("split", ""))
        if not sample_id or split not in {"train", "val", "test"}:
            raise ValueError("split manifest contains an invalid sample/split")
        if sample_id in split_by_sample and split_by_sample[sample_id] != split:
            raise ValueError(f"sample {sample_id} occurs in multiple splits")
        split_by_sample[sample_id] = split

    sources: dict[str, dict[str, Any]] = {}
    for raw in source_rows:
        row = dict(raw)
        sample_id = str(row.get("sample_id", ""))
        files = row.get("files")
        image = files.get("image") if isinstance(files, Mapping) else None
        image_path = image.get("path") if isinstance(image, Mapping) else None
        if not sample_id or not isinstance(image_path, str) or not image_path:
            raise ValueError("source manifest row has no sample_id/image path")
        if sample_id not in split_by_sample:
            raise ValueError(f"source sample {sample_id} has no split assignment")
        row["split"] = split_by_sample[sample_id]
        if sample_id in sources:
            raise ValueError(f"duplicate source sample {sample_id}")
        sources[sample_id] = row
    if set(sources) != set(split_by_sample):
        missing = sorted(set(split_by_sample) - set(sources))
        extra = sorted(set(sources) - set(split_by_sample))
        raise ValueError(f"source/split sample mismatch: missing={missing}, extra={extra}")
    return sources


def _validate_record_sources(
    rows: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
) -> None:
    seen: set[str] = set()
    for row in rows:
        record_id = str(row["record_id"])
        if record_id in seen:
            raise ValueError(f"duplicate record_id {record_id!r}")
        seen.add(record_id)
        sample_id = str(row["sample_id"])
        source = sources.get(sample_id)
        if source is None:
            raise ValueError(f"record {record_id!r} references unknown sample {sample_id}")
        files = source.get("files")
        image = files.get("image") if isinstance(files, Mapping) else None
        expected_path = image.get("path") if isinstance(image, Mapping) else None
        if row["source_image"] != expected_path:
            raise ValueError(f"record {record_id!r} source_image differs from source manifest")
        if row["split"] != source["split"]:
            raise ValueError(f"record {record_id!r} split differs from source manifest")


def _yaml_names(names: Sequence[str]) -> str:
    rendered = ", ".join(json.dumps(name, ensure_ascii=False) for name in names)
    return f"[{rendered}]"


def _dataset_yaml(*, names: Sequence[str], train: str, val: str) -> str:
    return (
        "# Generated by data_tools.export_yolo_dataset; do not edit in place.\n"
        f"train: {train}\n"
        f"val: {val}\n"
        f"nc: {len(names)}\n"
        f"names: {_yaml_names(names)}\n"
    )


def _save_tile(source: Image.Image, tile: Tile, destination: Path) -> None:
    canvas = Image.new("RGB", (tile.output_width, tile.output_height), (255, 255, 255))
    crop = source.crop((tile.x1, tile.y1, tile.x2, tile.y2)).convert("RGB")
    canvas.paste(crop, (0, 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=False, compress_level=6)


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copyfile(source, destination)
        return "copy"


def _select_visual_audit_rows(
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    task_name: str,
    special_sample_ids: set[str],
    limit: int = 50,
) -> list[Mapping[str, Any]]:
    eligible = [
        row
        for row in manifest_rows
        if isinstance(row.get("tasks"), Mapping)
        and isinstance(row["tasks"].get(task_name), Mapping)
        and row["tasks"][task_name].get("written") is True
    ]
    selected: list[Mapping[str, Any]] = []
    seen_tiles: set[str] = set()

    def add(rows: Iterable[Mapping[str, Any]], maximum: int) -> None:
        added = 0
        for row in rows:
            tile_id = str(row["tile_id"])
            if tile_id in seen_tiles:
                continue
            selected.append(row)
            seen_tiles.add(tile_id)
            added += 1
            if len(selected) >= limit or added >= maximum:
                return

    special = sorted(
        (row for row in eligible if str(row["sample_id"]) in special_sample_ids),
        key=lambda row: (
            -int(row["tasks"][task_name]["label_count"]),
            str(row["tile_id"]),
        ),
    )
    add(special, min(20, limit))
    fragments = sorted(
        (
            row
            for row in eligible
            if any(
                label.get("visibility_policy") == "clipped_fragment_safety_label"
                for label in row["tasks"][task_name]["labels"]
            )
        ),
        key=lambda row: str(row["tile_id"]),
    )
    add(fragments, min(10, limit))
    negatives = sorted(
        (row for row in eligible if row["tasks"][task_name]["negative"] is True),
        key=lambda row: str(row["tile_id"]),
    )
    add(negatives, min(5, limit))
    remaining = sorted(
        eligible,
        key=lambda row: (
            -len(
                {
                    int(label["class_id"])
                    for label in row["tasks"][task_name]["labels"]
                }
            ),
            -int(row["tasks"][task_name]["label_count"]),
            str(row["tile_id"]),
        ),
    )
    add(remaining, limit)
    return selected[:limit]


def _render_visual_audit(
    *,
    staging: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    special_sample_ids: set[str],
    limit: int = 50,
) -> list[dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    colors = (
        "#e6194b",
        "#3cb44b",
        "#ffe119",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
        "#bcf60c",
        "#fabebe",
    )
    for task_name in ("component", "text"):
        selected = _select_visual_audit_rows(
            manifest_rows,
            task_name=task_name,
            special_sample_ids=special_sample_ids,
            limit=limit,
        )
        for row in selected:
            task = row["tasks"][task_name]
            image_path = staging / str(task["image"])
            with Image.open(image_path) as opened:
                canvas = opened.convert("RGB")
            draw = ImageDraw.Draw(canvas)
            crop_x1, crop_y1, _, _ = (float(value) for value in row["crop_xyxy"])
            for label in task["labels"]:
                sx1, sy1, sx2, sy2 = (
                    float(value) for value in label["clipped_xyxy_source"]
                )
                box = (sx1 - crop_x1, sy1 - crop_y1, sx2 - crop_x1, sy2 - crop_y1)
                class_id = int(label["class_id"])
                color = colors[class_id % len(colors)]
                width = 3 if label["visibility_policy"] == "clipped_fragment_safety_label" else 2
                draw.rectangle(box, outline=color, width=width)
                caption = f"{class_id}:{label['visible_fraction']:.2f}"
                draw.text((box[0] + 2, max(0.0, box[1] - 10)), caption, fill=color)
            output_path = staging / "visual_audit" / task_name / f"{row['tile_id']}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(output_path, format="PNG", optimize=False, compress_level=6)
            audit_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "task": task_name,
                    "tile_id": row["tile_id"],
                    "sample_id": row["sample_id"],
                    "source_domain": row["source_domain"],
                    "split": row["split"],
                    "label_count": task["label_count"],
                    "negative": task["negative"],
                    "special_validation_sample": str(row["sample_id"])
                    in special_sample_ids,
                    "overlay_path": _relative(output_path, staging),
                }
            )
    return audit_rows


def _task_decisions(
    rows: Sequence[Mapping[str, Any]],
    tile: Tile,
    *,
    class_for: Mapping[str, int] | None,
    min_visible_fraction: float,
    minimum_box_pixels: float,
    text_task: bool,
) -> tuple[list[BoxDecision], list[dict[str, Any]], list[str]]:
    included: list[BoxDecision] = []
    excluded_fragments: list[dict[str, Any]] = []
    touched: list[str] = []
    for row in rows:
        if text_task:
            class_id = 0
        else:
            assert class_for is not None
            class_id = class_for[str(row["category"])]
        decision = _decision(row, tile, class_id=class_id)
        if decision is None:
            continue
        touched.append(decision.record_id)
        x1, y1, x2, y2 = decision.clipped_xyxy
        width = x2 - x1
        height = y2 - y1
        if width < minimum_box_pixels or height < minimum_box_pixels:
            excluded_fragments.append(
                {
                    "record_id": decision.record_id,
                    "visible_fraction": decision.visible_fraction,
                    "clipped_width": width,
                    "clipped_height": height,
                    "reason": "below_minimum_clipped_box_pixels",
                }
            )
            continue
        if decision.visible_fraction >= min_visible_fraction:
            policy = "minimum_visible_fraction"
        elif decision.center_in_tile:
            policy = "center_inside_fallback"
        else:
            # YOLO has no arbitrary ignore-mask contract.  Label the visible
            # fragment instead of silently turning a known object into a
            # background target.  The manifest keeps this case auditable.
            policy = "clipped_fragment_safety_label"
        included.append(replace(decision, visibility_policy=policy))
    included.sort(key=lambda item: (item.class_id, item.record_id))
    return included, excluded_fragments, sorted(touched)


def _label_manifest_payload(
    decision: BoxDecision,
    source_record: Mapping[str, Any],
    *,
    text_task: bool,
) -> dict[str, Any]:
    payload = {
        "record_id": decision.record_id,
        "class_id": decision.class_id,
        "visible_fraction": decision.visible_fraction,
        "truncated": decision.truncated,
        "center_in_tile": decision.center_in_tile,
        "visibility_policy": decision.visibility_policy,
        "clipped_xyxy_source": list(decision.clipped_xyxy),
    }
    if text_task:
        payload["source_annotation"] = {
            "text_id": source_record.get("text_id"),
            "category": source_record.get("category"),
            "category_id": source_record.get("category_id"),
            "content": source_record.get("content"),
            "details": source_record.get("details"),
        }
    else:
        payload["source_annotation"] = {
            "component_id": source_record.get("component_id"),
            "component_key": source_record.get("component_key"),
            "category": source_record.get("category"),
            "category_id": source_record.get("category_id"),
        }
    return payload


def _prepare_output(destination: Path, overwrite: bool) -> tuple[Path, bool]:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists; pass --overwrite to replace it: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    return staging, destination.exists()


def _commit_output(staging: Path, destination: Path, replace_existing: bool) -> None:
    if replace_existing:
        # The caller has already checked that destination is an exact child of the workspace.
        shutil.rmtree(destination)
    staging.replace(destination)


def _group_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        grouped[str(raw["sample_id"])].append(dict(raw))
    for records in grouped.values():
        records.sort(key=lambda row: str(row["record_id"]))
    return dict(grouped)


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def export_yolo_datasets(
    *,
    workspace: Path,
    dataset_root: Path,
    component_manifest: Path,
    text_manifest: Path,
    source_manifest: Path,
    split_manifest: Path,
    output: Path,
    tile_size: int,
    tile_overlap: int,
    component_min_visible_fraction: float,
    text_min_visible_fraction: float,
    minimum_clipped_box_pixels: float,
    long_tail_max_train_instances: int,
    special_validation_split: str,
    verify_source_hashes: bool,
    overwrite: bool,
) -> dict[str, Any]:
    tile_size = _strict_int(tile_size, "tile size", minimum=1)
    tile_overlap = _strict_int(tile_overlap, "tile overlap")
    if tile_overlap >= tile_size:
        raise ValueError("tile overlap must be smaller than tile size")
    component_min_visible_fraction = _fraction(
        component_min_visible_fraction, "component minimum visible fraction"
    )
    text_min_visible_fraction = _fraction(
        text_min_visible_fraction, "text minimum visible fraction"
    )
    if (
        isinstance(minimum_clipped_box_pixels, bool)
        or not isinstance(minimum_clipped_box_pixels, (int, float))
        or not math.isfinite(float(minimum_clipped_box_pixels))
        or float(minimum_clipped_box_pixels) <= 0
    ):
        raise ValueError("minimum clipped box pixels must be a positive finite number")
    long_tail_max_train_instances = _strict_int(
        long_tail_max_train_instances, "long-tail maximum train instances", minimum=1
    )
    if special_validation_split not in {"val", "test"}:
        raise ValueError("special validation split must be val or test")

    component_rows = _read_jsonl(component_manifest, "Component manifest")
    text_rows = _read_jsonl(text_manifest, "Text manifest")
    source_rows = _read_jsonl(source_manifest, "source manifest")
    split_rows = _read_jsonl(split_manifest, "split manifest")
    for row in component_rows:
        _validate_row_common(row, "component_detection")
    for row in text_rows:
        _validate_row_common(row, "ocr_text")
    sources = _index_sources(source_rows, split_rows)
    _validate_record_sources(component_rows, sources)
    _validate_record_sources(text_rows, sources)

    category_table = _category_table(component_rows)
    class_by_name, class_by_source_id = _class_maps(category_table)
    component_by_sample = _group_rows(component_rows)
    text_by_sample = _group_rows(text_rows)

    train_counts = Counter(
        str(row["category"]) for row in component_rows if row["split"] == "train"
    )
    long_tail_classes = sorted(
        name for name, count in train_counts.items() if count <= long_tail_max_train_instances
    )
    special_samples: dict[str, set[str]] = defaultdict(set)
    for sample_id, source in sources.items():
        if source["split"] != special_validation_split:
            continue
        source_class = str(source.get("source_class", "")).lower()
        if source_class == "kicad":
            special_samples[sample_id].add("kicad")
        present = {
            str(row["category"])
            for row in component_by_sample.get(sample_id, [])
            if str(row["category"]) in long_tail_classes
        }
        if present:
            special_samples[sample_id].add("long_tail")

    config = {
        "converter_version": CONVERTER_VERSION,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "component_min_visible_fraction": component_min_visible_fraction,
        "text_min_visible_fraction": text_min_visible_fraction,
        "minimum_clipped_box_pixels": float(minimum_clipped_box_pixels),
        "included_splits": list(SUPPORTED_SPLITS),
        "special_validation_split": special_validation_split,
        "long_tail_max_train_instances": long_tail_max_train_instances,
    }
    config_sha256 = _sha256_bytes(_canonical_bytes(config))

    staging, replace_existing = _prepare_output(output, overwrite)
    manifest_rows: list[dict[str, Any]] = []
    special_sample_rows: list[dict[str, Any]] = []
    special_object_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    task_stats: dict[str, Counter[str]] = {
        "component": Counter(),
        "text": Counter(),
    }
    split_stats: dict[str, Counter[str]] = defaultdict(Counter)
    labeled_records: dict[str, set[str]] = {
        "component": set(),
        "text": set(),
    }
    storage_modes: Counter[str] = Counter()
    source_hashes_checked = 0

    try:
        for task in ("component_yolo", "text_yolo"):
            for split in SUPPORTED_SPLITS:
                (staging / task / "images" / split).mkdir(parents=True, exist_ok=True)
                (staging / task / "labels" / split).mkdir(parents=True, exist_ok=True)

        for sample_id in sorted(sources):
            source_meta = sources[sample_id]
            split = str(source_meta["split"])
            if split not in SUPPORTED_SPLITS:
                continue
            files = source_meta["files"]
            image_meta = files["image"]
            source_rel = str(image_meta["path"])
            source_path = (dataset_root / Path(source_rel)).resolve(strict=False)
            try:
                source_path.relative_to(dataset_root)
            except ValueError as exc:
                raise ValueError(f"source image escapes dataset root: {source_rel}") from exc
            if not source_path.is_file():
                raise FileNotFoundError(f"source image is missing: {source_path}")
            if verify_source_hashes:
                actual_sha = _sha256_file(source_path)
                expected_sha = str(image_meta.get("sha256", ""))
                if actual_sha != expected_sha:
                    raise ValueError(f"source image SHA-256 mismatch for sample {sample_id}")
                source_hashes_checked += 1
            else:
                actual_sha = str(image_meta.get("sha256", ""))

            with Image.open(source_path) as opened:
                source_image = opened.convert("RGB")
                width, height = source_image.size
                expected_width = int(image_meta.get("width", source_meta["canvas"]["width"]))
                expected_height = int(image_meta.get("height", source_meta["canvas"]["height"]))
                if (width, height) != (expected_width, expected_height):
                    raise ValueError(
                        f"source dimensions mismatch for sample {sample_id}: "
                        f"actual={(width, height)}, expected={(expected_width, expected_height)}"
                    )
                tiles = build_tiles(width, height, tile_size, tile_overlap)
                component_records = component_by_sample.get(sample_id, [])
                text_records = text_by_sample.get(sample_id, [])
                component_records_by_id = {
                    str(row["record_id"]): row for row in component_records
                }
                text_records_by_id = {str(row["record_id"]): row for row in text_records}
                special_tile_names: list[str] = []

                for tile in tiles:
                    tile_name = f"{sample_id}__{tile.tile_id}"
                    component_decisions, component_fragments, component_touched = _task_decisions(
                        component_records,
                        tile,
                        class_for=class_by_name,
                        min_visible_fraction=component_min_visible_fraction,
                        minimum_box_pixels=float(minimum_clipped_box_pixels),
                        text_task=False,
                    )
                    text_decisions, text_fragments, text_touched = _task_decisions(
                        text_records,
                        tile,
                        class_for=None,
                        min_visible_fraction=text_min_visible_fraction,
                        minimum_box_pixels=float(minimum_clipped_box_pixels),
                        text_task=True,
                    )
                    task_payload: dict[str, Any] = {}
                    written_image: Path | None = None

                    for task_name, decisions, fragments, touched in (
                        ("component", component_decisions, component_fragments, component_touched),
                        ("text", text_decisions, text_fragments, text_touched),
                    ):
                        directory = f"{task_name}_yolo"
                        task_stats[task_name]["ignored_subpixel_fragments"] += len(fragments)
                        image_path = staging / directory / "images" / split / f"{tile_name}.png"
                        label_path = staging / directory / "labels" / split / f"{tile_name}.txt"
                        if written_image is None:
                            _save_tile(source_image, tile, image_path)
                            written_image = image_path
                            storage_modes["png"] += 1
                        else:
                            storage_modes[_link_or_copy(written_image, image_path)] += 1
                        label_path.write_text(
                            "".join(_format_yolo(item, tile) + "\n" for item in decisions),
                            encoding="utf-8",
                            newline="\n",
                        )
                        for item in decisions:
                            labeled_records[task_name].add(item.record_id)
                        task_stats[task_name]["written_tiles"] += 1
                        task_stats[task_name]["labels"] += len(decisions)
                        task_stats[task_name]["negative_tiles"] += int(not decisions)
                        task_stats[task_name]["truncated_labels"] += sum(
                            int(item.truncated) for item in decisions
                        )
                        task_stats[task_name]["fragment_safety_labels"] += sum(
                            item.visibility_policy == "clipped_fragment_safety_label"
                            for item in decisions
                        )
                        split_stats[f"{task_name}:{split}"]["tiles"] += 1
                        split_stats[f"{task_name}:{split}"]["labels"] += len(decisions)
                        task_payload[task_name] = {
                            "written": True,
                            "image": _relative(image_path, staging),
                            "label": _relative(label_path, staging),
                            "label_count": len(decisions),
                            "negative": not decisions,
                            "touched_record_ids": touched,
                            "ignored_subpixel_fragments": fragments,
                            "labels": [
                                _label_manifest_payload(
                                    item,
                                    (
                                        text_records_by_id
                                        if task_name == "text"
                                        else component_records_by_id
                                    )[item.record_id],
                                    text_task=task_name == "text",
                                )
                                for item in decisions
                            ],
                        }

                    if sample_id in special_samples and task_payload.get("component", {}).get("written"):
                        source_component_image = staging / task_payload["component"]["image"]
                        source_component_label = staging / task_payload["component"]["label"]
                        special_image = (
                            staging
                            / "special_validation"
                            / "component_yolo"
                            / "images"
                            / "val"
                            / source_component_image.name
                        )
                        special_label = (
                            staging
                            / "special_validation"
                            / "component_yolo"
                            / "labels"
                            / "val"
                            / source_component_label.name
                        )
                        storage_modes[_link_or_copy(source_component_image, special_image)] += 1
                        storage_modes[_link_or_copy(source_component_label, special_label)] += 1
                        for reason in sorted(special_samples[sample_id]):
                            reason_image = (
                                staging
                                / "special_validation"
                                / reason
                                / "component_yolo"
                                / "images"
                                / "val"
                                / source_component_image.name
                            )
                            reason_label = (
                                staging
                                / "special_validation"
                                / reason
                                / "component_yolo"
                                / "labels"
                                / "val"
                                / source_component_label.name
                            )
                            storage_modes[_link_or_copy(source_component_image, reason_image)] += 1
                            storage_modes[_link_or_copy(source_component_label, reason_label)] += 1
                        special_tile_names.append(tile_name)

                    manifest_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "converter_version": CONVERTER_VERSION,
                            "config_sha256": config_sha256,
                            "tile_id": tile_name,
                            "tile_index": tile.index,
                            "sample_id": sample_id,
                            "split": split,
                            "duplicate_group": source_meta.get("duplicate_group"),
                            "source_domain": source_meta.get("source_class"),
                            "source_image": source_rel,
                            "source_image_sha256": actual_sha,
                            "source_canvas": {"width": width, "height": height},
                            "crop_xyxy": [tile.x1, tile.y1, tile.x2, tile.y2],
                            "output_size": {
                                "width": tile.output_width,
                                "height": tile.output_height,
                            },
                            "transform": {
                                "kind": "crop_then_bottom_right_white_pad",
                                "scale_x": 1.0,
                                "scale_y": 1.0,
                                "offset_x": -tile.x1,
                                "offset_y": -tile.y1,
                                "pad_left": 0,
                                "pad_top": 0,
                                "pad_right": tile.output_width - tile.width,
                                "pad_bottom": tile.output_height - tile.height,
                            },
                            "tasks": task_payload,
                        }
                    )

                if sample_id in special_samples:
                    selected_objects = component_by_sample.get(sample_id, [])
                    special_sample_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "sample_id": sample_id,
                            "source_image": source_rel,
                            "source_image_sha256": actual_sha,
                            "source_domain": source_meta.get("source_class"),
                            "original_split": split,
                            "reasons": sorted(special_samples[sample_id]),
                            "long_tail_categories": sorted(
                                {
                                    str(row["category"])
                                    for row in selected_objects
                                    if str(row["category"]) in long_tail_classes
                                }
                            ),
                            "component_count": len(selected_objects),
                            "tile_count": len(special_tile_names),
                            "tile_ids": special_tile_names,
                        }
                    )
                    for row in selected_objects:
                        special_object_rows.append(
                            {
                                "schema_version": SCHEMA_VERSION,
                                "record_id": row["record_id"],
                                "sample_id": sample_id,
                                "source_image": source_rel,
                                "original_split": split,
                                "source_domain": row["source_domain"],
                                "category": row["category"],
                                "category_id": row["category_id"],
                                "yolo_class_id": class_by_source_id[str(row["category_id"])],
                                "bbox": row["bbox"],
                                "selection_reasons": sorted(special_samples[sample_id]),
                                "is_long_tail": str(row["category"]) in long_tail_classes,
                            }
                        )

        component_expected = {
            str(row["record_id"])
            for row in component_rows
            if row["split"] in SUPPORTED_SPLITS
        }
        text_expected = {
            str(row["record_id"])
            for row in text_rows
            if row["split"] in SUPPORTED_SPLITS
        }
        missing_component = sorted(component_expected - labeled_records["component"])
        missing_text = sorted(text_expected - labeled_records["text"])
        if missing_component:
            raise ValueError(
                f"{len(missing_component)} Component records were never emitted; "
                f"first={missing_component[:5]}"
            )
        if missing_text:
            raise ValueError(
                f"{len(missing_text)} Text records were never emitted; first={missing_text[:5]}"
            )

        component_names = [str(item["source_category_name"]) for item in category_table]
        component_root = staging / "component_yolo"
        text_root = staging / "text_yolo"
        (component_root / "dataset.yaml").write_text(
            _dataset_yaml(names=component_names, train="images/train", val="images/val"),
            encoding="utf-8",
            newline="\n",
        )
        (text_root / "dataset.yaml").write_text(
            _dataset_yaml(names=["text"], train="images/train", val="images/val"),
            encoding="utf-8",
            newline="\n",
        )
        component_classes = {
            "schema_version": SCHEMA_VERSION,
            "label_contract": "YOLO foreground labels are contiguous 0..N-1",
            "classes": category_table,
            "by_yolo_class_id": {
                str(item["yolo_class_id"]): {
                    "source_category_id": item["source_category_id"],
                    "source_category_name": item["source_category_name"],
                }
                for item in category_table
            },
            "by_source_category_id": class_by_source_id,
            "by_source_category_name": class_by_name,
        }
        text_classes = {
            "schema_version": SCHEMA_VERSION,
            "label_contract": "single-class text detector; content/details remain in tile manifest",
            "classes": [{"yolo_class_id": 0, "name": "text"}],
        }
        _write_json(component_root / "classes.json", component_classes)
        _write_json(text_root / "classes.json", text_classes)
        _write_jsonl(staging / "tile_manifest.jsonl", manifest_rows)

        visual_audit_rows = _render_visual_audit(
            staging=staging,
            manifest_rows=manifest_rows,
            special_sample_ids=set(special_samples),
            limit=50,
        )
        _write_jsonl(staging / "visual_audit" / "manifest.jsonl", visual_audit_rows)

        special_root = staging / "special_validation"
        special_component = special_root / "component_yolo"
        special_component.mkdir(parents=True, exist_ok=True)
        (special_component / "dataset.yaml").write_text(
            _dataset_yaml(names=component_names, train="images/val", val="images/val"),
            encoding="utf-8",
            newline="\n",
        )
        _write_json(special_component / "classes.json", component_classes)
        for reason in ("kicad", "long_tail"):
            reason_component = special_root / reason / "component_yolo"
            reason_component.mkdir(parents=True, exist_ok=True)
            (reason_component / "dataset.yaml").write_text(
                _dataset_yaml(names=component_names, train="images/val", val="images/val"),
                encoding="utf-8",
                newline="\n",
            )
            _write_json(reason_component / "classes.json", component_classes)
        _write_jsonl(special_root / "sample_manifest.jsonl", special_sample_rows)
        _write_jsonl(special_root / "component_manifest.jsonl", special_object_rows)

        special_kicad = [row for row in special_sample_rows if "kicad" in row["reasons"]]
        special_long_tail = [row for row in special_sample_rows if "long_tail" in row["reasons"]]
        if len(special_kicad) < 3:
            warnings.append(
                "KiCad special validation has fewer than 3 source images; add an external "
                "project-grouped KiCad holdout before making release claims."
            )
        if not special_long_tail:
            warnings.append("No long-tail sample is present in the selected source split.")

        special_summary = {
            "schema_version": SCHEMA_VERSION,
            "kind": "component_kicad_long_tail_diagnostic_validation_v1",
            "status": "ready_for_diagnostic_evaluation",
            "source_split": special_validation_split,
            "selection_scope": "source-image before tiling",
            "training_rows_included": special_validation_split == "train",
            "independent_external_holdout": False,
            "limitations": [
                "This subset is drawn from the existing Dataset300 split and is not a new external holdout.",
                "It is suitable for stable domain/long-tail diagnostics, not final competition claims.",
            ],
            "long_tail_definition": {
                "maximum_train_instances": long_tail_max_train_instances,
                "classes": long_tail_classes,
                "train_counts": {name: train_counts[name] for name in long_tail_classes},
            },
            "sample_count": len(special_sample_rows),
            "kicad_sample_count": len(special_kicad),
            "long_tail_sample_count": len(special_long_tail),
            "component_count": len(special_object_rows),
            "tile_count": sum(int(row["tile_count"]) for row in special_sample_rows),
            "subsets": {
                "kicad": {
                    "dataset_yaml": "special_validation/kicad/component_yolo/dataset.yaml",
                    "sample_count": len(special_kicad),
                    "tile_count": sum(int(row["tile_count"]) for row in special_kicad),
                },
                "long_tail": {
                    "dataset_yaml": "special_validation/long_tail/component_yolo/dataset.yaml",
                    "sample_count": len(special_long_tail),
                    "tile_count": sum(int(row["tile_count"]) for row in special_long_tail),
                },
            },
            "sample_ids": [row["sample_id"] for row in special_sample_rows],
            "warnings": warnings,
        }
        _write_json(special_root / "summary.json", special_summary)

        source_inputs = {
            "component_manifest": {
                "path": _relative(component_manifest, workspace),
                "sha256": _sha256_file(component_manifest),
            },
            "text_manifest": {
                "path": _relative(text_manifest, workspace),
                "sha256": _sha256_file(text_manifest),
            },
            "source_manifest": {
                "path": _relative(source_manifest, workspace),
                "sha256": _sha256_file(source_manifest),
            },
            "split_manifest": {
                "path": _relative(split_manifest, workspace),
                "sha256": _sha256_file(split_manifest),
            },
        }
        duplicate_group_splits: dict[str, set[str]] = defaultdict(set)
        for source in sources.values():
            duplicate_group_splits[str(source.get("duplicate_group"))].add(str(source["split"]))
        leakage_groups = sorted(
            group for group, splits in duplicate_group_splits.items() if len(splits) > 1
        )
        if leakage_groups:
            raise ValueError(f"duplicate groups cross source splits: {leakage_groups[:10]}")

        audit = {
            "schema_version": SCHEMA_VERSION,
            "kind": "pcbnet_yolo_d1_conversion_audit",
            "status": "passed",
            "converter_version": CONVERTER_VERSION,
            "config": config,
            "config_sha256": config_sha256,
            "source_inputs": source_inputs,
            "source_hash_verification": {
                "enabled": verify_source_hashes,
                "checked_images": source_hashes_checked,
            },
            "split_contract": {
                "inherited_before_tiling": True,
                "included_splits": list(SUPPORTED_SPLITS),
                "duplicate_group_leakage_count": 0,
                "test_rows_materialized_for_training": 0,
            },
            "classes": {
                "component_count": len(category_table),
                "component_contiguous_zero_based": [
                    int(item["yolo_class_id"]) for item in category_table
                ]
                == list(range(len(category_table))),
                "text_count": 1,
            },
            "tasks": {
                name: dict(sorted(counter.items())) for name, counter in task_stats.items()
            },
            "by_task_split": {
                key: dict(sorted(counter.items())) for key, counter in sorted(split_stats.items())
            },
            "coverage": {
                "component_expected_records": len(component_expected),
                "component_emitted_records": len(labeled_records["component"]),
                "component_missing_records": 0,
                "text_expected_records": len(text_expected),
                "text_emitted_records": len(labeled_records["text"]),
                "text_missing_records": 0,
            },
            "tile_manifest_rows": len(manifest_rows),
            "visual_audit": {
                "selection": "deterministic special/fragment/negative/diverse-density strata",
                "component_overlays": sum(
                    row["task"] == "component" for row in visual_audit_rows
                ),
                "text_overlays": sum(row["task"] == "text" for row in visual_audit_rows),
                "manifest": "visual_audit/manifest.jsonl",
            },
            "image_storage_modes": dict(sorted(storage_modes.items())),
            "special_validation": special_summary,
            "warnings": warnings,
        }
        _write_json(staging / "conversion_audit.json", audit)

        (staging / "README.md").write_text(
            "# PCBnet YOLO D1 dataset\n\n"
            f"Generated by `{CONVERTER_VERSION}` with {tile_size}×{tile_size} tiles "
            f"and {tile_overlap}px overlap.\n\n"
            "## Training datasets\n\n"
            "- `component_yolo/dataset.yaml`: 34-class Component detector.\n"
            "- `text_yolo/dataset.yaml`: single-class Text detector.\n\n"
            "Only the frozen source `train` and `val` splits are materialized. Source split "
            "assignment is inherited before tiling; the original `test` split is absent. "
            "Images smaller than a tile are padded white on the right/bottom without scaling.\n\n"
            "Known boxes are clipped into every intersecting tile. Boxes below the preferred "
            "visible-fraction floor are marked `clipped_fragment_safety_label` in "
            "`tile_manifest.jsonl`; this avoids silently treating a known visible object as "
            "background because standard YOLO labels do not provide arbitrary ignore masks. "
            "Fragments smaller than the configured minimum pixel geometry are recorded but not "
            "labeled.\n\n"
            "## Diagnostic validation\n\n"
            "- `special_validation/kicad/component_yolo/dataset.yaml`\n"
            "- `special_validation/long_tail/component_yolo/dataset.yaml`\n"
            "- `special_validation/component_yolo/dataset.yaml` (combined)\n\n"
            "These subsets come only from the configured frozen source split and are useful for "
            "repeatable diagnostics. They are not a new external holdout. Read "
            "`special_validation/summary.json` before reporting results.\n\n"
            "## Audit evidence\n\n"
            "- `conversion_audit.json`: split, class, coverage and conversion checks.\n"
            "- `tile_manifest.jsonl`: reversible per-tile geometry and label lineage.\n"
            "- `visual_audit/`: 50 deterministic overlays per task.\n"
            "- `artifact_inventory.json`: hashes for generated files.\n",
            encoding="utf-8",
            newline="\n",
        )

        generated_files = sorted(
            path for path in staging.rglob("*") if path.is_file()
        )
        inventory = {
            "schema_version": SCHEMA_VERSION,
            "file_count": len(generated_files),
            "files": [
                {
                    "path": _relative(path, staging),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for path in generated_files
            ],
        }
        inventory["payload_sha256"] = _sha256_bytes(_canonical_bytes(inventory))
        _write_json(staging / "artifact_inventory.json", inventory)

        _commit_output(staging, output, replace_existing)
        staging = None
        return {
            "ok": True,
            "output": str(output),
            "config_sha256": config_sha256,
            "component_classes": len(category_table),
            "tile_manifest_rows": len(manifest_rows),
            "component_tiles": task_stats["component"]["written_tiles"],
            "text_tiles": task_stats["text"]["written_tiles"],
            "special_validation_samples": len(special_sample_rows),
            "special_validation_kicad_samples": len(special_kicad),
            "warnings": warnings,
        }
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = args.workspace_root.expanduser().resolve(strict=False)
    try:
        if not workspace.is_dir():
            raise FileNotFoundError(f"workspace root is missing: {workspace}")
        dataset_root = _resolve_directory(workspace, args.dataset_root, "dataset root")
        component_manifest = _resolve_file(
            workspace, args.component_manifest, "Component manifest"
        )
        text_manifest = _resolve_file(workspace, args.text_manifest, "Text manifest")
        source_manifest = _resolve_file(workspace, args.source_manifest, "source manifest")
        split_manifest = _resolve_file(workspace, args.split_manifest, "split manifest")
        output = _resolve_output(workspace, args.output)
        result = export_yolo_datasets(
            workspace=workspace,
            dataset_root=dataset_root,
            component_manifest=component_manifest,
            text_manifest=text_manifest,
            source_manifest=source_manifest,
            split_manifest=split_manifest,
            output=output,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            component_min_visible_fraction=args.component_min_visible_fraction,
            text_min_visible_fraction=args.text_min_visible_fraction,
            minimum_clipped_box_pixels=args.minimum_clipped_box_pixels,
            long_tail_max_train_instances=args.long_tail_max_train_instances,
            special_validation_split=args.special_validation_split,
            verify_source_hashes=not args.skip_source_hash_check,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
