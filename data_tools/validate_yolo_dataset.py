from __future__ import annotations

"""Independent integrity checks for a generated PCBnet YOLO D1 dataset."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from ._bootstrap import REPOSITORY_ROOT


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8-sig"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(
                raw,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(value)
    return rows


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _label_rows(path: Path, class_count: int) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = raw.split()
        if len(fields) != 5:
            raise ValueError(f"YOLO label must have 5 fields: {path}:{line_number}")
        try:
            class_id = int(fields[0])
            values = tuple(float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(f"invalid YOLO label number: {path}:{line_number}") from exc
        if not 0 <= class_id < class_count:
            raise ValueError(f"YOLO class id outside declared range: {path}:{line_number}")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"YOLO coordinate outside [0,1]: {path}:{line_number}")
        if values[2] <= 0.0 or values[3] <= 0.0:
            raise ValueError(f"YOLO width/height must be positive: {path}:{line_number}")
        rows.append((class_id, values[0], values[1], values[2], values[3]))
    return rows


def _validate_dataset_root(root: Path, *, tile_size: int) -> dict[str, int]:
    classes = _read_json(root / "classes.json")
    raw_classes = classes.get("classes")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise ValueError(f"classes.json has no classes array: {root}")
    class_ids = [int(item["yolo_class_id"]) for item in raw_classes]
    if class_ids != list(range(len(raw_classes))):
        raise ValueError(f"YOLO class ids are not contiguous 0..N-1: {root}")
    if not (root / "dataset.yaml").is_file():
        raise FileNotFoundError(f"dataset.yaml is missing: {root}")

    image_count = 0
    label_count = 0
    object_count = 0
    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.exists() and not label_dir.exists():
            continue
        images = {path.stem: path for path in image_dir.glob("*.png")}
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        if set(images) != set(labels):
            raise ValueError(
                f"image/label stem mismatch in {root}/{split}: "
                f"images_only={sorted(set(images)-set(labels))[:5]}, "
                f"labels_only={sorted(set(labels)-set(images))[:5]}"
            )
        for stem in sorted(images):
            with Image.open(images[stem]) as image:
                if image.size != (tile_size, tile_size):
                    raise ValueError(
                        f"tile has wrong size {image.size}, expected {(tile_size, tile_size)}: "
                        f"{images[stem]}"
                    )
            parsed = _label_rows(labels[stem], len(raw_classes))
            image_count += 1
            label_count += 1
            object_count += len(parsed)
    if image_count == 0:
        raise ValueError(f"YOLO dataset has no images: {root}")
    return {
        "images": image_count,
        "label_files": label_count,
        "objects": object_count,
        "classes": len(raw_classes),
    }


def validate_yolo_d1(root: Path, *, verify_inventory: bool = True) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=False)
    if not root.is_dir():
        raise FileNotFoundError(f"YOLO D1 root is missing: {root}")
    audit = _read_json(root / "conversion_audit.json")
    if audit.get("kind") != "pcbnet_yolo_d1_conversion_audit" or audit.get("status") != "passed":
        raise ValueError("conversion audit is not a passed YOLO D1 audit")
    config = audit.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("conversion audit has no config")
    tile_size = int(config["tile_size"])

    roots = {
        "component": root / "component_yolo",
        "text": root / "text_yolo",
        "special_combined": root / "special_validation" / "component_yolo",
        "special_kicad": root / "special_validation" / "kicad" / "component_yolo",
        "special_long_tail": root
        / "special_validation"
        / "long_tail"
        / "component_yolo",
    }
    datasets = {
        name: _validate_dataset_root(path, tile_size=tile_size)
        for name, path in roots.items()
    }
    if datasets["component"]["images"] != int(audit["tasks"]["component"]["written_tiles"]):
        raise ValueError("Component tile count differs from conversion audit")
    if datasets["text"]["images"] != int(audit["tasks"]["text"]["written_tiles"]):
        raise ValueError("Text tile count differs from conversion audit")

    tile_manifest = _read_jsonl(root / "tile_manifest.jsonl")
    if len(tile_manifest) != int(audit["tile_manifest_rows"]):
        raise ValueError("tile manifest row count differs from conversion audit")
    seen_tiles: set[str] = set()
    for row in tile_manifest:
        tile_id = str(row.get("tile_id", ""))
        if not tile_id or tile_id in seen_tiles:
            raise ValueError(f"empty or duplicate tile id {tile_id!r}")
        seen_tiles.add(tile_id)
        if row.get("split") not in {"train", "val"}:
            raise ValueError(f"tile manifest contains non-training split {row.get('split')!r}")
        crop = row.get("crop_xyxy")
        if not isinstance(crop, list) or len(crop) != 4:
            raise ValueError(f"tile {tile_id} has invalid crop")
        if not (crop[2] > crop[0] and crop[3] > crop[1]):
            raise ValueError(f"tile {tile_id} crop has non-positive area")
        for task_name in ("component", "text"):
            task = row["tasks"][task_name]
            if task.get("written") is not True:
                raise ValueError(f"tile {tile_id} task {task_name} was not written")
            image_path = root / str(task["image"])
            label_path = root / str(task["label"])
            if not image_path.is_file() or not label_path.is_file():
                raise FileNotFoundError(f"tile {tile_id} task {task_name} artifact is missing")
            class_count = datasets[task_name]["classes"]
            parsed = _label_rows(label_path, class_count)
            if len(parsed) != int(task["label_count"]) or len(parsed) != len(task["labels"]):
                raise ValueError(f"tile {tile_id} task {task_name} label count mismatch")

    special_summary = _read_json(root / "special_validation" / "summary.json")
    special_samples = _read_jsonl(root / "special_validation" / "sample_manifest.jsonl")
    if len(special_samples) != int(special_summary["sample_count"]):
        raise ValueError("special validation sample count mismatch")
    expected_split = special_summary.get("source_split")
    if any(row.get("original_split") != expected_split for row in special_samples):
        raise ValueError("special validation contains a sample from the wrong source split")
    if expected_split == "train" or special_summary.get("training_rows_included") is not False:
        raise ValueError("special validation improperly includes training rows")

    inventory_result: dict[str, Any] = {"verified": False, "file_count": 0}
    if verify_inventory:
        inventory_path = root / "artifact_inventory.json"
        inventory = _read_json(inventory_path)
        declared_payload = str(inventory.get("payload_sha256", ""))
        payload = dict(inventory)
        payload.pop("payload_sha256", None)
        actual_payload = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        if actual_payload != declared_payload:
            raise ValueError("artifact inventory payload SHA-256 mismatch")
        declared_files = inventory.get("files")
        if not isinstance(declared_files, list):
            raise ValueError("artifact inventory has no files array")
        actual_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != inventory_path
        }
        declared_paths = {str(item["path"]) for item in declared_files}
        if actual_paths != declared_paths:
            raise ValueError("artifact inventory file set differs from generated directory")
        for item in declared_files:
            path = root / str(item["path"])
            if path.stat().st_size != int(item["bytes"]):
                raise ValueError(f"artifact byte count mismatch: {path}")
            if _sha256_file(path) != item["sha256"]:
                raise ValueError(f"artifact SHA-256 mismatch: {path}")
        inventory_result = {"verified": True, "file_count": len(declared_files)}

    return {
        "ok": True,
        "root": str(root),
        "datasets": datasets,
        "tile_manifest_rows": len(tile_manifest),
        "special_validation_samples": len(special_samples),
        "inventory": inventory_result,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify a generated PCBnet YOLO D1 dataset")
    parser.add_argument(
        "--root",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "derived" / "pcbnet-upgrade-v1",
    )
    parser.add_argument("--skip-inventory", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = validate_yolo_d1(args.root, verify_inventory=not args.skip_inventory)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
