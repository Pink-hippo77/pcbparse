"""Test tighter grouping parameters to reduce chaining.

Goal: each group should contain candidates for at most one physical text.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, r"E:\EDA挑战题目六\src")

WORKSPACE = Path(r"E:\EDA挑战题目六")
DATASET = WORKSPACE / "Dataset300" / "Dataset300"
CASE_ID = "0005"

from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth
from pcbparse.evaluation.text_box_eval import _bbox_iou, _center_distance, _gt_text_boxes
from pcbparse.evaluation.masked_metrics import _maximum_cardinality_matching

case_dir = DATASET / CASE_ID
ann_path = sorted(case_dir.glob("*_CVJsonStd.json"))[0]
annotation = json.loads(ann_path.read_text(encoding="utf-8"))
gt = build_raw_ground_truth(annotation, case_id=CASE_ID, annotation_path=ann_path)
gt_boxes = _gt_text_boxes(gt)

frozen_dirs = sorted((WORKSPACE / "work").glob("ocr-frozen-crops-*"))
frozen_data = json.loads((frozen_dirs[-1] / "frozen_crops.json").read_text(encoding="utf-8"))
candidates = frozen_data["candidates"]

cache_data = json.loads((WORKSPACE / "work" / "ocr-recognition-cache" / f"{CASE_ID}_paddle_recognition.json").read_text(encoding="utf-8"))
preds = {int(k): (v["pred"], v["score"]) for k, v in cache_data["preds"].items()}


def build_groups(cands, iou_thresh, dist_thresh, scale_limit=2.0):
    n = len(cands)
    parent = list(range(n))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        px, py = find(x), find(y)
        if px != py: parent[px] = py
    for i in range(n):
        bi = cands[i]["original_bbox"]; wi, hi = bi[2]-bi[0], bi[3]-bi[1]
        for j in range(i+1, n):
            bj = cands[j]["original_bbox"]; wj, hj = bj[2]-bj[0], bj[3]-bj[1]
            if wi>0 and wj>0 and max(wi/wj, wj/wi)>scale_limit: continue
            if hi>0 and hj>0 and max(hi/hj, hj/hi)>scale_limit: continue
            if _bbox_iou(tuple(bi), tuple(bj)) >= iou_thresh or _center_distance(tuple(bi), tuple(bj)) <= dist_thresh:
                union(i, j)
    groups = defaultdict(list)
    for i in range(n): groups[find(i)].append(i)
    return [members for members in sorted(groups.values(), key=lambda m: min(m))]


def sel_highest_score(members):
    best_ci, best_s = None, -1
    for ci in members:
        p, s = preds.get(ci, (None, 0))
        if p and s > best_s: best_s = s; best_ci = ci
    if best_ci is None: return None, None
    return preds[best_ci][0], candidates[best_ci]["original_bbox"]


def score(groups):
    selections = []
    for i, members in enumerate(groups):
        pred, bbox = sel_highest_score(members)
        if pred:
            selections.append({"gid": f"g{i}", "bbox": bbox, "pred": pred})
    match_pairs = []
    for pi, s in enumerate(selections):
        for gi, tb in enumerate(gt_boxes):
            iou = _bbox_iou(tuple(s["bbox"]), tuple(tb["bbox"]))
            dist = _center_distance(tuple(s["bbox"]), tuple(tb["bbox"]))
            if iou >= 0.3 or dist <= 12.0:
                match_pairs.append((pi, gi, max(iou, 0.01)))
    matches = _maximum_cardinality_matching(len(selections), len(gt_boxes), match_pairs,
        left_keys=[s["gid"] for s in selections], right_keys=[tb["object_id"] for tb in gt_boxes])
    p2g = {l: r for l, r, _ in matches}
    text_exact = sum(1 for pi, gi in p2g.items() if selections[pi]["pred"] == gt_boxes[gi]["gt_value"])
    return len(p2g), text_exact, len(selections)


configs = [
    ("baseline(0.3,8)", 0.3, 8.0, 2.0),
    ("tight(0.5,4)", 0.5, 4.0, 1.5),
    ("tighter(0.5,3)", 0.5, 3.0, 1.5),
    ("very_tight(0.7,2)", 0.7, 2.0, 1.3),
    ("ultra(0.8,1)", 0.8, 1.0, 1.2),
]

print(f"{'config':25s} {'groups':>7} {'sel':>5} {'match':>7} {'exact':>7} {'rate':>7}", flush=True)
for name, iou, dist, scale in configs:
    groups = build_groups(candidates, iou, dist, scale)
    sizes = [len(g) for g in groups]
    n_matched, text_exact, n_sel = score(groups)
    print(f"{name:25s} {len(groups):>7} {n_sel:>5} {n_matched:>7} {text_exact:>7} {text_exact/51:>7.4f}  (mean_size={np.mean(sizes):.0f} max={max(sizes)})", flush=True)

print(f"\noracle: 38/51 = 0.7451", flush=True)
