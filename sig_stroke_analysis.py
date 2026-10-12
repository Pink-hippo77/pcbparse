#!/usr/bin/env python3
"""Stroke-level analysis of the four-signature SIG annotation (误伤 semantics).

Pixel-level 误伤 (SIG pixels within GT wire +-3px) conflates two very different
things:
  * a stroke that IS a real wire but got marked "symbol/decoration"  -> true 误伤
  * a correct decoration/frame stroke whose dilated band crosses or runs
    alongside real wires                                           -> geometry overlap

This script splits SIG into connected strokes and reports both numbers:
  stroke_误伤率 = pixels of strokes whose majority lies on GT wire / total SIG px
  pixel_误伤率 = raw band overlap (the pre-registered number)

Usage: python sig_stroke_analysis.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/home/data/yinjx/pcbparse-competition/project")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402
from scipy import ndimage  # noqa: E402

from pcbparse.training import wire_junction_materialize as wjm  # noqa: E402

DATASET_ROOT = ROOT / "Dataset300" / "Dataset300"
NEG_DIR = ROOT / "work" / "opt90-wire" / "neg-labels"
OUT = ROOT / "work" / "opt90-wire" / "sig_stroke_analysis.json"
GT_TOL = 3


def gt_dilated(sample_id: str, rows: list[dict]) -> np.ndarray:
    annotation, _ = wjm._load_annotation_without_dense_coordinates(
        DATASET_ROOT / rows[0]["source_json"]
    )
    image = Image.open(DATASET_ROOT / rows[0]["source_image"])
    width, height = image.size
    target = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(target)
    for row in rows:
        if row.get("category") != "wire":
            continue
        geometry = wjm._resolve_wire_geometry(row, annotation, ("wire",))
        if geometry.bbox is None:
            continue
        for path in geometry.paths:
            pts = [(round(x), round(y)) for x, y in path]
            if len(pts) == 1:
                draw.point(pts[0], fill=255)
            else:
                draw.line(pts, fill=255, width=1)
        for px, py in geometry.skeleton:
            draw.point((round(px), round(py)), fill=255)
    return np.asarray(target.filter(ImageFilter.MaxFilter(2 * GT_TOL + 1))) > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows_by_sample: dict[str, list[dict]] = {}
    with open(ROOT / "data/derived/dataset300-v2/wire_paths.jsonl", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                continue
            rows_by_sample.setdefault(row["sample_id"], []).append(row)
    sample_ids = sorted(rows_by_sample)
    if args.limit:
        sample_ids = sample_ids[: args.limit]

    totals = {"sig_px": 0, "pixel_hit_px": 0, "stroke_hit_px": 0, "stroke_count": 0,
              "wirelike_strokes": 0, "crossing_only_px": 0}
    per_sample = []
    for sample_id in sample_ids:
        sig_path = NEG_DIR / f"SIG_{sample_id}.png"
        if not sig_path.is_file():
            continue
        sig = np.asarray(Image.open(sig_path).convert("L")) > 0
        hit = gt_dilated(sample_id, rows_by_sample[sample_id])
        labels, count = ndimage.label(sig, structure=np.ones((3, 3)))
        sig_px = int(sig.sum())
        pixel_hit = int((sig & hit).sum())
        stroke_hit = 0
        wirelike = 0
        stroke_count = 0
        for index in range(1, count + 1):
            component = labels == index
            size = int(component.sum())
            if size < 4:
                continue
            stroke_count += 1
            on_wire = int((component & hit).sum())
            if on_wire > 0.5 * size:      # the stroke itself is (mostly) a wire
                wirelike += 1
                stroke_hit += size
        totals["sig_px"] += sig_px
        totals["pixel_hit_px"] += pixel_hit
        totals["stroke_hit_px"] += stroke_hit
        totals["stroke_count"] += stroke_count
        totals["wirelike_strokes"] += wirelike
        totals["crossing_only_px"] += pixel_hit - stroke_hit
        per_sample.append({
            "sample_id": sample_id, "sig_px": sig_px,
            "pixel_hit_px": pixel_hit,
            "pixel_err_rate": round(pixel_hit / sig_px, 6) if sig_px else 0.0,
            "stroke_count": stroke_count, "wirelike_strokes": wirelike,
            "stroke_err_px": stroke_hit,
            "stroke_err_rate": round(stroke_hit / sig_px, 6) if sig_px else 0.0,
        })

    totals["pixel_err_rate"] = round(totals["pixel_hit_px"] / totals["sig_px"], 6) if totals["sig_px"] else 0.0
    totals["stroke_err_rate"] = round(totals["stroke_hit_px"] / totals["sig_px"], 6) if totals["sig_px"] else 0.0
    document = {
        "schema_version": 1,
        "kind": "opt90_sig_stroke_analysis",
        "gt_access": "diagnostic_only",
        "definition": {
            "pixel_err_rate": "SIG pixels within GT wire +-3px / SIG pixels (pre-registered band metric)",
            "stroke_err_rate": "pixels of SIG connected strokes whose >50% lies on GT wire / SIG pixels (true 误伤: marking a wire as symbol/decoration)",
            "crossing_only_px": "band overlap of otherwise-correct decoration strokes (crossings/parallel runs)",
        },
        "totals": totals,
        "samples": per_sample,
    }
    OUT.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
