#!/usr/bin/env python3
"""Per-stage timing of the opt90 negative labeler chain path for one sample."""
from __future__ import annotations

import sys, time, json
from pathlib import Path

ROOT = Path("/home/data/yinjx/pcbparse-competition/project")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "work/opt90-wire"))

import opt90_neg_labeler as L

sample = sys.argv[1] if len(sys.argv) > 1 else "0001"
rows = [json.loads(l) for l in open(L.WIRE_MANIFEST) if l.strip()]
rows = [r for r in rows if r["split"] == "train" and r["sample_id"] == sample]

t0 = time.time()
annotation, _ = L.wjm._load_annotation_without_dense_coordinates(
    L.DATASET_ROOT / rows[0]["source_json"])
image = L.Image.open(L.DATASET_ROOT / rows[0]["source_image"]).convert("RGB")
print(f"[{sample}] load {time.time()-t0:.1f}s", flush=True)

width, height = image.size
t1 = time.time()
geometries = []
for row in rows:
    g = L.wjm._resolve_wire_geometry(row, annotation, ("wire",))
    if g.bbox is not None:
        geometries.append(g)
gt_mask = L._draw_geometry_target(geometries, width, height)
print(f"[{sample}] gt raster {time.time()-t1:.1f}s geoms={len(geometries)}", flush=True)

t2 = time.time()
ink = L._ink_mask(image)
components, labels = L._gt_regions_ir(annotation, (width, height))
print(f"[{sample}] ink {time.time()-t2:.1f}s ink_px={int(ink.sum())} comps={len(components)} labels={len(labels)}", flush=True)

t3 = time.time()
strokes = L._StrokeSet()
poly_count = 0
from pcbparse.perception.wire_vectorizer import WireVectorizerConfig, vectorize_wire_mask
from skimage.morphology import thin as sk_thin
cfg = WireVectorizerConfig()
for y in range(0, height, L.TILE):
    for x in range(0, width, L.TILE):
        tile = ink[y:y+L.TILE, x:x+L.TILE]
        if not tile.any():
            continue
        tile = sk_thin(tile)
        res = vectorize_wire_mask(tile, origin=(float(x), float(y)), config=cfg,
                                  id_namespace=f"ink_{sample}_y{y}_x{x}")
        poly_count += len(res.polylines)
        for poly in res.polylines:
            strokes.add_polyline(poly)
print(f"[{sample}] stroke stage {time.time()-t3:.1f}s polys={poly_count} edges={len(strokes.segments)}", flush=True)

t4 = time.time()
strokes.aggregate_chains()
print(f"[{sample}] chain agg {time.time()-t4:.1f}s chains={len(strokes.segments)}", flush=True)

t5 = time.time()
from pcbparse.ir.models import CircuitIR, ImageInfo
ir = CircuitIR(image=ImageInfo(image_id=sample, path=rows[0]["source_image"], width=width, height=height),
               components=components, net_labels=labels, segments=strokes.segments)
report = L.filter_decorative_wire_segments(ir)
print(f"[{sample}] filter {time.time()-t5:.1f}s suppressed={report.get('suppressed')} layers={report.get('layers')}", flush=True)
