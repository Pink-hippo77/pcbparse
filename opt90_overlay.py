#!/usr/bin/env python3
"""Render NEG/SIG overlays for visual evidence.

Source image in grayscale; GT wire centreline (±1) green; SIG (four-signature
strokes, pre-protection) blue; NEG (training negatives, protected) red.

Usage: python opt90_overlay.py 0001 0020 0200
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/data/yinjx/pcbparse-competition/project")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw

from pcbparse.training import wire_junction_materialize as wjm

DATASET_ROOT = ROOT / "Dataset300" / "Dataset300"
OUT = ROOT / "work" / "opt90-wire" / "overlays"


def main() -> int:
    rows_all = [json.loads(l) for l in open(ROOT / "data/derived/dataset300-v2/wire_paths.jsonl") if l.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    for sample in sys.argv[1:]:
        rows = [r for r in rows_all if r["sample_id"] == sample]
        annotation, _ = wjm._load_annotation_without_dense_coordinates(DATASET_ROOT / rows[0]["source_json"])
        image = Image.open(DATASET_ROOT / rows[0]["source_image"]).convert("RGB")
        width, height = image.size
        gt = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(gt)
        for row in rows:
            if row.get("category") != "wire":
                continue
            g = wjm._resolve_wire_geometry(row, annotation, ("wire",))
            if g.bbox is None:
                continue
            for path in g.paths:
                pts = [(round(x), round(y)) for x, y in path]
                if len(pts) == 1:
                    draw.point(pts[0], fill=255)
                else:
                    draw.line(pts, fill=255, width=1)
            for px, py in g.skeleton:
                draw.point((round(px), round(py)), fill=255)
        neg = np.asarray(Image.open(ROOT / "work/opt90-wire/neg-labels" / f"NEG_{sample}.png").convert("L")) > 0
        sig = np.asarray(Image.open(ROOT / "work/opt90-wire/neg-labels" / f"SIG_{sample}.png").convert("L")) > 0
        gt_arr = np.asarray(gt) > 0

        base = np.asarray(image.convert("L"), dtype=np.float32)
        rgb = np.stack([base, base, base], axis=2)
        rgb[sig] = np.stack([base[sig] * 0.4, base[sig] * 0.4 + 120, np.full(int(sig.sum()), 255.0)], axis=1)
        rgb[gt_arr] = np.stack([np.zeros(int(gt_arr.sum())), np.full(int(gt_arr.sum()), 255.0), np.zeros(int(gt_arr.sum()))], axis=1)
        rgb[neg] = np.stack([np.full(int(neg.sum()), 255.0), base[neg] * 0.3, base[neg] * 0.3], axis=1)
        out = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
        path = OUT / f"overlay_{sample}.png"
        out.save(path)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
