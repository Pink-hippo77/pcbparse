"""Analyze predicted Wire mask connected components against raw-v1 wire pixels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

from pcbparse.config import load_config
from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth
from pcbparse.io import load_image, tile_image_view
from pcbparse.ir.models import Point
from pcbparse.perception.production_provider import (
    composite_provider_config_from_pipeline,
    load_production_provider,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--models-lock", type=Path, required=True)
    parser.add_argument("--pipeline-config", type=Path, required=True)
    parser.add_argument("--case-ids", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.pipeline_config)
    provider = load_production_provider(
        args.models_lock,
        config=composite_provider_config_from_pipeline(config),
    )
    rows = [
        json.loads(line)
        for line in args.split_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    wanted = {item.strip() for item in args.case_ids.split(",") if item.strip()}
    rows = [
        row
        for row in rows
        if row.get("split") == args.split and str(row.get("sample_id")) in wanted
    ]
    if not rows:
        raise ValueError("no requested cases found")

    report = {"cases": []}
    for row in rows:
        case_id = str(row["sample_id"])
        image_path = args.dataset_root / str(row["files"]["image"]["path"])
        original = load_image(image_path)
        views = tile_image_view(
            original,
            tile_size=provider.config.resolved_wire_tile_size,
            overlap=provider.config.resolved_wire_tile_overlap,
        )
        masks = provider.wire_provider.segment(views)
        predicted = np.zeros((original.height, original.width), dtype=bool)
        for view, item in zip(views, masks):
            array = np.asarray(item.mask, dtype=bool)
            p0 = view.to_original(Point(x=0.0, y=0.0))
            p1 = view.to_original(Point(x=1.0, y=1.0))
            x0 = int(round(p0.x))
            y0 = int(round(p0.y))
            sx = int(round(p1.x - p0.x))
            sy = int(round(p1.y - p0.y))
            if (sx, sy) != (1, 1):
                raise ValueError(f"non-unit tile transform for {view.view_id}: {(sx, sy)}")
            h = min(array.shape[0], original.height - y0)
            w = min(array.shape[1], original.width - x0)
            if h > 0 and w > 0:
                predicted[y0:y0 + h, x0:x0 + w] |= array[:h, :w]

        annotation_path = next(args.dataset_root.joinpath(case_id).glob("*_CVJsonStd.json"))
        gt = build_raw_ground_truth(
            json.loads(annotation_path.read_text(encoding="utf-8")),
            case_id=case_id,
            annotation_path=annotation_path,
        )
        expected_image = Image.new("L", (gt.width, gt.height), 0)
        draw = ImageDraw.Draw(expected_image)
        for wire in gt.wires:
            if not wire.mask.eligible:
                continue
            for path in wire.paths:
                points = [(round(x), round(y)) for x, y in path]
                if len(points) == 1:
                    draw.point(points[0], fill=255)
                elif len(points) >= 2:
                    draw.line(points, fill=255, width=1)
        expected = np.asarray(expected_image) > 0
        expected_dilated = np.asarray(expected_image.filter(ImageFilter.MaxFilter(5))) > 0
        if args.preview_dir is not None:
            args.preview_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.where(predicted, 255, 0).astype(np.uint8)).save(
                args.preview_dir / f"{case_id}-predicted.png"
            )
            Image.fromarray(np.where(expected, 255, 0).astype(np.uint8)).save(
                args.preview_dir / f"{case_id}-expected.png"
            )
            overlay = np.zeros((original.height, original.width, 3), dtype=np.uint8)
            overlay[expected & predicted] = (255, 255, 255)
            overlay[expected & ~predicted] = (0, 180, 0)
            overlay[~expected & predicted] = (220, 0, 0)
            Image.fromarray(overlay).save(args.preview_dir / f"{case_id}-overlay.png")
        labels, count = ndimage.label(
            predicted,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        components = []
        for label in range(1, count + 1):
            ys, xs = np.nonzero(labels == label)
            if not len(xs):
                continue
            pixels = labels == label
            area = int(len(xs))
            exact_hits = int(np.count_nonzero(pixels & expected))
            tolerant_hits = int(np.count_nonzero(pixels & expected_dilated))
            bbox_area = (
                (int(xs.max()) - int(xs.min()) + 1)
                * (int(ys.max()) - int(ys.min()) + 1)
            )
            components.append(
                {
                    "label": int(label),
                    "area": area,
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "bbox_fill_ratio": round(area / max(1, bbox_area), 6),
                    "exact_precision": round(exact_hits / area, 6),
                    "tolerant_precision": round(tolerant_hits / area, 6),
                    "expected_pixels": int(np.count_nonzero(pixels & expected)),
                }
            )
        components.sort(key=lambda item: (-item["area"], item["label"]))
        report["cases"].append(
            {
                "case_id": case_id,
                "image_size": [original.width, original.height],
                "predicted_pixels": int(predicted.sum()),
                "expected_pixels": int(expected.sum()),
                "predicted_components": len(components),
                "components": components,
            }
        )
        print(
            f"{case_id}: pixels={int(predicted.sum())} components={len(components)} "
            f"largest={components[0] if components else None}",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
