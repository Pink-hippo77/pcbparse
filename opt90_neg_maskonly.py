#!/usr/bin/env python3
"""Mask-only NEG computation for straggler samples (unblocks materialization).

NEG = production wire model fired pixels farther than GT_TOL_PX from any GT wire
centreline.  NEG depends ONLY on the model mask + GT geometry — never on the
vectoriser/annotator — so heavy samples can produce their training negatives
while the SIG annotator (stratification only) is still grinding.

Usage: python opt90_neg_maskonly.py 0003 0016 ...
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path("/home/data/yinjx/pcbparse-competition/project")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "work/opt90-wire"))

import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

import opt90_neg_labeler as L  # noqa: E402


def main() -> int:
    samples = sys.argv[1:]
    rows_all = [json.loads(l) for l in open(L.WIRE_MANIFEST) if l.strip()]
    started = time.time()
    out_rows = []
    for sample_id in samples:
        t0 = time.time()
        rows = [r for r in rows_all if r["sample_id"] == sample_id and r.get("split") == "train"]
        if not rows:
            raise SystemExit(f"{sample_id}: no train rows")
        runtime = L._runtime()
        annotation, _ = L.wjm._load_annotation_without_dense_coordinates(
            L.DATASET_ROOT / rows[0]["source_json"])
        original = L.load_image(L.DATASET_ROOT / rows[0]["source_image"])
        width, height = original.width, original.height

        geometries = []
        for row in rows:
            if row.get("category") != "wire":
                continue
            geometry = L.wjm._resolve_wire_geometry(row, annotation, ("wire",))
            if geometry.bbox is not None:
                geometries.append(geometry)
        gt_mask = L._draw_geometry_target(geometries, width, height)
        gt_dil = np.asarray(gt_mask.filter(ImageFilter.MaxFilter(2 * L.GT_TOL_PX + 1))) > 0

        composite = runtime["composite"]
        views = L.tile_image_view(original, tile_size=composite.resolved_wire_tile_size,
                                  overlap=composite.resolved_wire_tile_overlap)
        masks = runtime["adapter"].segment(views)
        view_map = {view.view_id: view for view in views}
        canvas = np.zeros((height, width), dtype=bool)
        for wire_mask in masks:
            view = view_map[wire_mask.view_id]
            origin = view.to_original(L.Point(x=0.0, y=0.0))
            x0, y0 = int(round(origin.x)), int(round(origin.y))
            canvas[y0 : y0 + view.height, x0 : x0 + view.width] |= np.asarray(wire_mask.mask, dtype=bool)
        neg = canvas & ~gt_dil
        L.OUT_DIR.mkdir(parents=True, exist_ok=True)
        Image.fromarray((neg * 255).astype(np.uint8)).save(L.OUT_DIR / f"NEG_{sample_id}.png")
        row = {
            "sample_id": sample_id,
            "canvas": [width, height],
            "image": rows[0]["source_image"],
            "gt_px": int(np.count_nonzero(np.asarray(gt_mask))),
            "gt_dilated_px": int(gt_dil.sum()),
            "model_mask_px": int(canvas.sum()),
            "model_near_miss_px": int((canvas & gt_dil).sum()),
            "neg_px": int(neg.sum()),
            "tile_count": len(views),
            "stroke_path": "mask_only_straggler",
            "elapsed_s": round(time.time() - t0, 2),
        }
        out_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (L.OUT_DIR / "neg_maskonly_stragglers.json").write_text(
        json.dumps({"kind": "opt90_neg_maskonly_stragglers", "samples": out_rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(out_rows), "elapsed_s": round(time.time() - started, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
