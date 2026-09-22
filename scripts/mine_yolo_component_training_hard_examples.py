from __future__ import annotations

"""Mine Component hard examples from the training split only.

The output is a deterministic curriculum manifest. It never reads validation
or test labels and does not modify the source YOLO dataset.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
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
    return intersection / (left_area + right_area - intersection)


def _load_yolo_labels(path: Path, *, width: int, height: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"invalid YOLO label at {path}:{line_number}")
        category = int(fields[0])
        cx, cy, box_width, box_height = (float(item) for item in fields[1:])
        values = (cx, cy, box_width, box_height)
        if not all(0.0 <= item <= 1.0 for item in values):
            raise ValueError(f"out-of-range YOLO label at {path}:{line_number}")
        rows.append(
            {
                "category": category,
                "box": [
                    (cx - box_width / 2.0) * width,
                    (cy - box_height / 2.0) * height,
                    (cx + box_width / 2.0) * width,
                    (cy + box_height / 2.0) * height,
                ],
            }
        )
    return rows


def _target_error_counts(
    truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    target_ids: set[int],
    match_iou: float,
) -> dict[int, dict[str, int]]:
    output: dict[int, dict[str, int]] = {}
    for category in sorted(target_ids):
        gt = [item for item in truth if int(item["category"]) == category]
        predicted = [
            item for item in predictions if int(item["category"]) == category
        ]
        matched_gt: set[int] = set()
        true_positives = 0
        false_positives = 0
        for prediction in sorted(
            predicted, key=lambda item: -float(item["score"])
        ):
            candidates = [
                (_iou(prediction["box"], item["box"]), index)
                for index, item in enumerate(gt)
                if index not in matched_gt
            ]
            overlap, index = max(candidates, default=(0.0, -1))
            if index >= 0 and overlap >= match_iou:
                matched_gt.add(index)
                true_positives += 1
            else:
                false_positives += 1
        output[category] = {
            "ground_truth": len(gt),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": len(gt) - true_positives,
        }
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("work/yolo/component-yolo11n-v2-precision/weights/best.pt"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/derived/pcbnet-upgrade-v1/component_yolo"),
    )
    parser.add_argument("--target-categories", nargs="+", default=["C", "V"])
    parser.add_argument("--score-threshold", type=float, default=0.20)
    parser.add_argument("--nms-iou", type=float, default=0.55)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-images-per-category", type=int, default=160)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/training/component_yolo11n_v2_train_hard_examples.json"
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    checkpoint = (ROOT / arguments.checkpoint).resolve(strict=True)
    dataset_root = (ROOT / arguments.dataset_root).resolve(strict=True)
    output_path = (ROOT / arguments.output).resolve(strict=False)
    if not 0.0 <= arguments.score_threshold <= 1.0:
        raise ValueError("score threshold must be in [0,1]")
    if not 0.0 < arguments.match_iou <= 1.0:
        raise ValueError("match IoU must be in (0,1]")
    if arguments.batch_size <= 0 or arguments.max_images_per_category <= 0:
        raise ValueError("batch size and selection cap must be positive")

    classes_path = dataset_root / "classes.json"
    classes = _read_object(classes_path).get("classes")
    if not isinstance(classes, list):
        raise ValueError("classes.json lacks classes array")
    names = {
        int(item["yolo_class_id"]): str(item["source_category_name"])
        for item in classes
    }
    ids_by_name = {name: category for category, name in names.items()}
    missing = sorted(set(arguments.target_categories).difference(ids_by_name))
    if missing:
        raise ValueError(f"unknown target categories: {missing}")
    target_ids = {ids_by_name[name] for name in arguments.target_categories}

    image_root = dataset_root / "images" / "train"
    label_root = dataset_root / "labels" / "train"
    images = sorted(
        path
        for path in image_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not images:
        raise ValueError("training image population is empty")

    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    observed_names = {
        int(key): str(value) for key, value in dict(model.names).items()
    }
    if observed_names != names:
        raise ValueError("checkpoint category names differ from dataset classes")

    rows: list[dict[str, Any]] = []
    aggregate = defaultdict(lambda: defaultdict(int))
    for start in range(0, len(images), arguments.batch_size):
        batch_paths = images[start : start + arguments.batch_size]
        results = model.predict(
            source=[str(path) for path in batch_paths],
            conf=float(arguments.score_threshold),
            iou=float(arguments.nms_iou),
            max_det=500,
            imgsz=int(arguments.imgsz),
            device=str(arguments.device),
            batch=len(batch_paths),
            verbose=False,
            save=False,
            stream=False,
        )
        if len(results) != len(batch_paths):
            raise RuntimeError("YOLO hard-example batch cardinality mismatch")
        for image_path, result in zip(batch_paths, results):
            label_path = label_root / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"training label missing: {label_path}")
            with Image.open(image_path) as image:
                width, height = image.size
            truth = _load_yolo_labels(label_path, width=width, height=height)
            boxes = getattr(result, "boxes", None)
            predictions: list[dict[str, Any]] = []
            if boxes is not None:
                for box, score, category in zip(
                    boxes.xyxy.detach().cpu().tolist(),
                    boxes.conf.detach().cpu().tolist(),
                    boxes.cls.detach().cpu().tolist(),
                ):
                    predictions.append(
                        {
                            "category": int(category),
                            "score": float(score),
                            "box": [float(value) for value in box],
                        }
                    )
            counts = _target_error_counts(
                truth,
                predictions,
                target_ids=target_ids,
                match_iou=float(arguments.match_iou),
            )
            errors = {
                names[category]: values
                for category, values in counts.items()
            }
            score = sum(
                2 * values["false_negatives"] + values["false_positives"]
                for values in counts.values()
            )
            for category, values in counts.items():
                for key, value in values.items():
                    aggregate[names[category]][key] += int(value)
            if score > 0:
                rows.append(
                    {
                        "stem": image_path.stem,
                        "image": image_path.relative_to(dataset_root).as_posix(),
                        "label": label_path.relative_to(dataset_root).as_posix(),
                        "hardness_score": score,
                        "target_errors": errors,
                    }
                )

    selected_by_category: dict[str, list[str]] = {}
    selected: set[str] = set()
    for category_name in arguments.target_categories:
        ranked = sorted(
            (
                row
                for row in rows
                if (
                    row["target_errors"][category_name]["false_negatives"] > 0
                    or row["target_errors"][category_name]["false_positives"] > 0
                )
            ),
            key=lambda row: (
                -(
                    2 * row["target_errors"][category_name]["false_negatives"]
                    + row["target_errors"][category_name]["false_positives"]
                ),
                -row["hardness_score"],
                row["stem"],
            ),
        )[: arguments.max_images_per_category]
        selected_by_category[category_name] = [row["stem"] for row in ranked]
        selected.update(selected_by_category[category_name])

    payload = {
        "schema_version": 1,
        "kind": "pcbparse_yolo_component_train_hard_example_manifest",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "selection_boundary": {
            "split": "train",
            "train_images_scored": len(images),
            "validation_images_scored": 0,
            "test_images_scored": 0,
            "label_dependent_selection_outside_train": False,
        },
        "protocol": {
            "target_categories": list(arguments.target_categories),
            "score_threshold": float(arguments.score_threshold),
            "nms_iou": float(arguments.nms_iou),
            "match_iou": float(arguments.match_iou),
            "imgsz": int(arguments.imgsz),
            "max_images_per_category": int(arguments.max_images_per_category),
            "hardness_score": "2 * false_negatives + false_positives",
        },
        "aggregate_errors": {
            name: dict(values) for name, values in sorted(aggregate.items())
        },
        "selected_by_category": selected_by_category,
        "selected_stems": sorted(selected),
        "selected_count": len(selected),
        "hard_images": sorted(
            (row for row in rows if row["stem"] in selected),
            key=lambda row: (-row["hardness_score"], row["stem"]),
        ),
        "lineage": {
            "checkpoint": {
                "path": checkpoint.relative_to(ROOT).as_posix(),
                "sha256": _sha256(checkpoint),
            },
            "classes": {
                "path": classes_path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(classes_path),
            },
            "dataset_yaml": {
                "path": (dataset_root / "dataset.yaml").relative_to(ROOT).as_posix(),
                "sha256": _sha256(dataset_root / "dataset.yaml"),
            },
        },
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["manifest_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(output_path)
    print(output_path)
    print(json.dumps({"selected_count": len(selected), "aggregate_errors": payload["aggregate_errors"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
