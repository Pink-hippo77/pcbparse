#!/usr/bin/env python3
"""Stage timing for the opt90 negative labeler on one sample/one tile."""
from __future__ import annotations

import sys, time, json
from pathlib import Path

ROOT = Path("/home/data/yinjx/pcbparse-competition/project")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image

from pcbparse.perception.wire_vectorizer import WireVectorizerConfig, vectorize_wire_mask
from pcbparse.training import wire_junction_materialize as wjm

sample = sys.argv[1] if len(sys.argv) > 1 else "0001"
rows = [json.loads(line) for line in open(ROOT / "data/derived/dataset300-v2/wire_paths.jsonl") if line.strip()]
rows = [r for r in rows if r["split"] == "train" and r["sample_id"] == sample]

t0 = time.time()
image_path = ROOT / "Dataset300" / "Dataset300" / rows[0]["source_image"]
json_path = ROOT / "Dataset300" / "Dataset300" / rows[0]["source_json"]
annotation, _ = wjm._load_annotation_without_dense_coordinates(json_path)
t1 = time.time()
image = Image.open(image_path).convert("RGB")
rgb = np.asarray(image, dtype=np.int16)
ink = (255 - rgb.min(axis=2)) > 45
t2 = time.time()
print(f"annotation {t1-t0:.1f}s  image+ink {t2-t1:.1f}s  size={image.size} ink_px={int(ink.sum())}")

geoms = []
for row in rows:
    g = wjm._resolve_wire_geometry(row, annotation, ("wire",))
    if g.bbox is not None:
        geoms.append(g)
t3 = time.time()
print(f"geometry {t3-t2:.1f}s  n={len(geoms)}")

cfg = WireVectorizerConfig()
height, width = ink.shape
best, tile = -1, None
for y in range(0, height, 512):
    for x in range(0, width, 512):
        chunk = ink[y:y+512, x:x+512]
        if int(chunk.sum()) > best:
            best, tile = int(chunk.sum()), chunk
print(f"tile ink_px={int(tile.sum())} (densest of {len(range(0,height,512))*len(range(0,width,512))} tiles)")
t4 = time.time()
result = vectorize_wire_mask(tile, origin=(0.0, 0.0), config=cfg, id_namespace="t0")
t5 = time.time()
print(f"vectorize 1 tile {t5-t4:.1f}s  polylines={len(result.polylines)} skeleton_px={result.skeleton_pixels}")

# run-length stroke primitive (candidate fast path) timing
t6 = time.time()
def runs(mask):
    out = []
    for axis in (1, 0):
        arr = mask if axis == 1 else mask.T
        padded = np.pad(arr, ((0, 0), (1, 1)))
        diff = np.diff(padded.astype(np.int8), axis=1)
        starts = np.argwhere(diff == 1)
        ends = np.argwhere(diff == -1)
        out.append((axis, len(starts)))
    return out
print("runlength", runs(tile), f"{time.time()-t6:.3f}s")

# thin-first variant: skimage thin -> production vectorizer
from skimage.morphology import thin as sk_thin
t7 = time.time()
skel = sk_thin(tile)
t8 = time.time()
result2 = vectorize_wire_mask(skel, origin=(0.0, 0.0), config=cfg, id_namespace="t1")
t9 = time.time()
print(f"skimage.thin {t8-t7:.1f}s + vectorize(thin) {t9-t8:.1f}s  polylines={len(result2.polylines)} skeleton_px={result2.skeleton_pixels}")
