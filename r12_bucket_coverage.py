#!/usr/bin/env python3
"""G-w10: residual-FP bucket coverage (mechanism evidence for wire precision).

Bucket definition (r12b/r12e口径, val30, qna-v1, tol=3):
  FP pixels of the final wire segment set (after the pipeline's deterministic
  filters) that are
    * isolated   : farther than 8px from ANY GT wire (incl. qna-masked),
    * non-text   : outside OCR text bboxes (+2px),
    * non-edge   : outside component bbox edge bands (+/-7px),
    * colourless : near-gray ink (saturation <= 0.25), the r12h "colourless
                   symbol stroke" population.
This is the r12 residual-FP population (51 131 px total residual at the
r12-integration terminal) that the deterministic rules cannot catch and that
the retrained model must suppress.

Usage: r12_bucket_coverage.py <predictions.jsonl> <out.json>
GT-ACCESS: diagnostic_only.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path("/home/data/yinjx/pcbparse-competition/project")
sys.path.insert(0, str(ROOT / "src"))

from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth  # noqa: E402
from pcbparse.ir.models import CircuitIR  # noqa: E402

DATASET = ROOT / "Dataset300" / "Dataset300"
SOURCE_IMAGES: dict = {}
TOL = 3
ISO_DIST = 8.0
TEXT_MARGIN = 2.0
EDGE_TOL = 7.0
SAT_MAX = 0.25


def main() -> int:
    pred_manifest = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    rows = [json.loads(x) for x in pred_manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    source_images = {}
    with open(ROOT / "data/manifests/dataset300.jsonl", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source_images[row["sample_id"]] = row["files"]["image"]["path"]
    global SOURCE_IMAGES
    SOURCE_IMAGES = source_images

    totals: Counter = Counter()
    per_case = []
    for row in sorted(rows, key=lambda r: r["case_id"]):
        case = row["case_id"]
        ir_path = Path(row["ir_path"])
        if not ir_path.is_absolute():
            ir_path = (pred_manifest.parent / ir_path).resolve(strict=True)
        ir = CircuitIR.model_validate_json(ir_path.read_text(encoding="utf-8"))
        ann_path = sorted((DATASET / case).glob("*_CVJsonStd.json"))
        annotation = json.loads(ann_path[0].read_text(encoding="utf-8"))
        gt = build_raw_ground_truth(annotation, case_id=case, annotation_path=ann_path[0],
                                    mask_policy="qna-v1")
        elig_paths = [p for w in gt.wires if w.mask.eligible for p in w.paths]
        all_paths = [p for w in gt.wires for p in w.paths]

        expected = Image.new("L", (gt.width, gt.height), 0)
        draw = ImageDraw.Draw(expected)
        for path in elig_paths:
            pts = [(round(x), round(y)) for x, y in path]
            if len(pts) == 1:
                draw.point(pts[0], fill=255)
            elif len(pts) >= 2:
                draw.line(pts, fill=255, width=1)
        predicted = Image.new("L", (gt.width, gt.height), 0)
        draw = ImageDraw.Draw(predicted)
        for segment in ir.segments:
            draw.line(
                [(round(segment.start.x), round(segment.start.y)),
                 (round(segment.end.x), round(segment.end.y))],
                fill=255, width=1,
            )
        expected_all = Image.new("L", (gt.width, gt.height), 0)
        draw = ImageDraw.Draw(expected_all)
        for path in all_paths:
            pts = [(round(x), round(y)) for x, y in path]
            if len(pts) == 1:
                draw.point(pts[0], fill=255)
            elif len(pts) >= 2:
                draw.line(pts, fill=255, width=1)

        elig_mask = np.asarray(expected) > 0
        all_mask = np.asarray(expected_all) > 0
        pred_mask = np.asarray(predicted) > 0
        elig_dilated = ndimage.binary_dilation(elig_mask, structure=np.ones((3, 3)), iterations=TOL)
        fp_mask = pred_mask & ~elig_dilated
        ys, xs = np.nonzero(fp_mask)
        n = len(xs)
        if n:
            dist_all = ndimage.distance_transform_edt(~all_mask)[ys, xs]
        else:
            dist_all = np.zeros(0)

        in_text = np.zeros(n, dtype=bool)
        for token in ir.texts:
            in_text |= (
                (xs >= token.bbox.x1 - TEXT_MARGIN) & (xs <= token.bbox.x2 + TEXT_MARGIN)
                & (ys >= token.bbox.y1 - TEXT_MARGIN) & (ys <= token.bbox.y2 + TEXT_MARGIN)
            )
        in_edge = np.zeros(n, dtype=bool)
        for comp in ir.components:
            x1, y1, x2, y2 = comp.bbox.x1, comp.bbox.y1, comp.bbox.x2, comp.bbox.y2
            if x2 <= x1 or y2 <= y1:
                continue
            band = ((np.abs(xs - x1) <= EDGE_TOL) | (np.abs(xs - x2) <= EDGE_TOL)) & (
                (ys >= y1 - EDGE_TOL) & (ys <= y2 + EDGE_TOL))
            band |= ((np.abs(ys - y1) <= EDGE_TOL) | (np.abs(ys - y2) <= EDGE_TOL)) & (
                (xs >= x1 - EDGE_TOL) & (xs <= x2 + EDGE_TOL))
            in_edge |= band

        source_row = SOURCE_IMAGES[case]
        source = Image.open(DATASET / source_row).convert("RGB")
        rgb = np.asarray(source, dtype=np.float32) / 255.0
        saturation = rgb.max(axis=2) - rgb.min(axis=2)
        if n:
            sat_pts = saturation[ys, xs]
            colourless = sat_pts <= SAT_MAX
        else:
            colourless = np.zeros(0, dtype=bool)

        isolated = dist_all > ISO_DIST
        bucket = isolated & ~in_text & ~in_edge & colourless
        counts = {
            "fp_px": int(n),
            "isolated_px": int(isolated.sum()),
            "bucket_px": int(bucket.sum()),
            "bucket_non_text_edge_px": int((isolated & ~in_text & ~in_edge).sum()),
            "bucket_coloured_px": int((isolated & ~in_text & ~in_edge & ~colourless).sum()),
            "pred_px": int(pred_mask.sum()),
        }
        totals.update(counts)
        per_case.append({"case_id": case, **counts})

    summary = {
        "schema": "opt90-r12-bucket-coverage-v1",
        "gt_access": "diagnostic_only",
        "definition": {
            "tol_px": TOL,
            "isolated_distance_px": ISO_DIST,
            "text_margin_px": TEXT_MARGIN,
            "edge_band_px": EDGE_TOL,
            "saturation_max": SAT_MAX,
            "bucket": "FP pixels isolated >8px from ALL GT wires, outside text bboxes and comp edge bands, colourless ink (r12 residual symbol-stroke population)",
        },
        "cases": len(rows),
        "totals": dict(totals),
        "per_case": per_case,
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"totals": dict(totals)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
