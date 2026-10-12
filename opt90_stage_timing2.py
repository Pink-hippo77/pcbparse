#!/usr/bin/env python3
"""Stage timing of the opt90 negative labeler v2 path for one sample."""
from __future__ import annotations

import sys, time, json
from pathlib import Path

ROOT = Path("/home/data/yinjx/pcbparse-competition/project")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "work/opt90-wire"))

import numpy as np
import opt90_neg_labeler as L

sample = sys.argv[1] if len(sys.argv) > 1 else "0001"
rows = [json.loads(l) for l in open(L.WIRE_MANIFEST) if l.strip()]
rows = [r for r in rows if r["split"] == "train" and r["sample_id"] == sample]

t0 = time.time()
rt = L._runtime()
print(f"[{sample}] runtime load {time.time()-t0:.1f}s threshold={rt['threshold']}", flush=True)
annotation, _ = L.wjm._load_annotation_without_dense_coordinates(
    L.DATASET_ROOT / rows[0]["source_json"])
original = L.load_image(rows[0]["source_image"] and (L.DATASET_ROOT / rows[0]["source_image"]))
width, height = original.width, original.height
print(f"[{sample}] image {width}x{height} rows={len(rows)}", flush=True)

t1 = time.time()
geometries = []
for row in rows:
    g = L.wjm._resolve_wire_geometry(row, annotation, ("wire",))
    if g.bbox is not None:
        geometries.append(g)
gt_mask = L._draw_geometry_target(geometries, width, height)
gt_dil = np.asarray(gt_mask.filter(L.ImageFilter.MaxFilter(2 * L.GT_TOL_PX + 1))) > 0
print(f"[{sample}] gt raster {time.time()-t1:.1f}s geoms={len(geometries)}", flush=True)

composite = rt["composite"]
views = L.tile_image_view(original, tile_size=composite.resolved_wire_tile_size,
                          overlap=composite.resolved_wire_tile_overlap)
print(f"[{sample}] tiles={len(views)} size={composite.resolved_wire_tile_size}+{composite.resolved_wire_tile_overlap}", flush=True)
t2 = time.time()
masks = rt["adapter"].segment(views)
print(f"[{sample}] inference {time.time()-t2:.1f}s masks={len(masks)}", flush=True)

view_map = {v.view_id: v for v in views}
canvas = np.zeros((height, width), dtype=bool)
for m in masks:
    v = view_map[m.view_id]
    o = v.to_original(L.Point(x=0.0, y=0.0))
    x0, y0 = int(round(o.x)), int(round(o.y))
    canvas[y0:y0+v.height, x0:x0+v.width] |= np.asarray(m.mask, dtype=bool)
print(f"[{sample}] model_px={int(canvas.sum())}", flush=True)

t3 = time.time()
segments, _j, _l = L._vectorize_masks(masks, view_map, composite)
print(f"[{sample}] vectorize {time.time()-t3:.1f}s segments={len(segments)}", flush=True)

comps, labels = L._gt_regions_ir(annotation, (width, height))
ir = L.CircuitIR(image=L.ImageInfo(image_id=sample, path=rows[0]["source_image"],
                                  width=width, height=height),
                 components=comps, net_labels=labels, segments=list(segments))
t4 = time.time()
report = L.filter_decorative_wire_segments(ir)
print(f"[{sample}] filter {time.time()-t4:.1f}s suppressed={report.get('suppressed')} layers={report.get('layers')}", flush=True)
