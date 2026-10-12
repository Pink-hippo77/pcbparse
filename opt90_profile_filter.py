#!/usr/bin/env python3
"""Profile the four-signature annotator call for one sample."""
from __future__ import annotations

import sys, time, json
from pathlib import Path

ROOT = Path("/home/data/yinjx/pcbparse-competition/project")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image
from skimage.morphology import thin as sk_thin

from pcbparse.graph.physical import filter_decorative_wire_segments
from pcbparse.ir.models import BBox, CircuitIR, Component, ImageInfo, NetLabel, Point, WireSegment
from pcbparse.perception.wire_vectorizer import WireVectorizerConfig, vectorize_wire_mask
from pcbparse.training import wire_junction_materialize as wjm

sample = sys.argv[1] if len(sys.argv) > 1 else "0001"
rows = [json.loads(l) for l in open(ROOT / "data/derived/dataset300-v2/wire_paths.jsonl") if l.strip()]
rows = [r for r in rows if r["split"] == "train" and r["sample_id"] == sample]
image_path = ROOT / "Dataset300" / "Dataset300" / rows[0]["source_image"]
annotation, _ = wjm._load_annotation_without_dense_coordinates(ROOT / "Dataset300" / "Dataset300" / rows[0]["source_json"])
image = Image.open(image_path).convert("RGB")
width, height = image.size
ink = (255 - np.asarray(image, dtype=np.int16).min(axis=2)) > 45

cfg = WireVectorizerConfig()
segments = []
t0 = time.time()
for y in range(0, height, 512):
    for x in range(0, width, 512):
        tile = ink[y:y+512, x:x+512]
        if not tile.any():
            continue
        tile = sk_thin(tile)
        res = vectorize_wire_mask(tile, origin=(float(x), float(y)), config=cfg,
                                  id_namespace=f"p_{y}_{x}")
        for poly in res.polylines:
            pts = poly.points
            for i in range(len(pts) - 1):
                (ax, ay), (bx, by) = pts[i], pts[i+1]
                segments.append(WireSegment(internal_id=f"{poly.internal_id}#{i}",
                                            start=Point(x=float(ax), y=float(ay)),
                                            end=Point(x=float(bx), y=float(by))))
t1 = time.time()
axis = sum(1 for s in segments if abs(s.start.y - s.end.y) <= 2 or abs(s.start.x - s.end.x) <= 2)
print(f"segments={len(segments)} axis_aligned={axis} stroke_stage={t1-t0:.1f}s")

regions = wjm._collect_ignore_regions(annotation, {"width": width, "height": height}, wjm.WireJunctionMaterializeConfig())
components, labels = [], []
for r in regions:
    if r["kind"] == "component_bbox":
        x1, y1, x2, y2 = r["bbox"]
        components.append(Component(internal_id=f"C_{r['source_id']}", category="unknown",
                                    bbox=BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))))
    elif r["kind"] == "text_bbox":
        x1, y1, x2, y2 = r["bbox"]
        labels.append(NetLabel(internal_id=f"T_{r['source_id']}", instance_text="", resolved_text="",
                               point=Point(x=(float(x1)+float(x2))/2, y=(float(y1)+float(y2))/2)))
ir = CircuitIR(image=ImageInfo(image_id=sample, path=rows[0]["source_image"], width=width, height=height),
               components=components, net_labels=labels, segments=segments)
t2 = time.time()
rep = filter_decorative_wire_segments(ir)
t3 = time.time()
print(f"components={len(components)} labels={len(labels)}")
print(f"filter {t3-t2:.1f}s  suppressed={rep.get('suppressed')} layers={rep.get('layers')}")
