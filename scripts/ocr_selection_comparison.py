"""Compare selection strategies using cached recognition results.

Strategies: weighted_vote, highest_score, score_x_geometric, top3_multi.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
sys.path.insert(0, r"E:\EDA挑战题目六\src")

WORKSPACE = Path(r"E:\EDA挑战题目六")
DATASET = WORKSPACE / "Dataset300" / "Dataset300"
CASE_ID = "0005"
CACHE_DIR = WORKSPACE / "work" / "ocr-recognition-cache"

from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth
from pcbparse.evaluation.text_box_eval import _bbox_iou, _center_distance, _gt_text_boxes
from pcbparse.evaluation.masked_metrics import _maximum_cardinality_matching

case_dir = DATASET / CASE_ID
ann_path = sorted(case_dir.glob("*_CVJsonStd.json"))[0]
img_path = sorted(case_dir.glob("*.png"))[0]
img = Image.open(img_path).convert("RGB")
annotation = json.loads(ann_path.read_text(encoding="utf-8"))
gt = build_raw_ground_truth(annotation, case_id=CASE_ID, annotation_path=ann_path)
gt_boxes = _gt_text_boxes(gt)

frozen_dirs = sorted((WORKSPACE / "work").glob("ocr-frozen-crops-*"))
frozen_data = json.loads((frozen_dirs[-1] / "frozen_crops.json").read_text(encoding="utf-8"))
candidates = frozen_data["candidates"]

cache_file = CACHE_DIR / f"{CASE_ID}_paddle_recognition.json"
cache_data = json.loads(cache_file.read_text(encoding="utf-8"))
preds = {int(k): (v["pred"], v["score"]) for k, v in cache_data["preds"].items()}
print(f"loaded {len(gt_boxes)} GT, {len(candidates)} candidates, {len(preds)} preds", flush=True)


def build_groups(cands, iou_thresh=0.3, dist_thresh=8.0):
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
            if wi>0 and wj>0 and max(wi/wj, wj/wi)>2.0: continue
            if hi>0 and hj>0 and max(hi/hj, hj/hi)>2.0: continue
            if _bbox_iou(tuple(bi), tuple(bj)) >= iou_thresh or _center_distance(tuple(bi), tuple(bj)) <= dist_thresh:
                union(i, j)
    groups = defaultdict(list)
    for i in range(n): groups[find(i)].append(i)
    return [members for members in sorted(groups.values(), key=lambda m: min(m))]


def score_selection(selections, gt_boxes):
    pred_items = [(s["group_id"], tuple(s["actual_bbox"]), s["prediction"]) for s in selections if s["status"] == "accepted"]
    match_pairs = []
    for pi, (gid, bbox, pred) in enumerate(pred_items):
        for gi, tb in enumerate(gt_boxes):
            iou = _bbox_iou(bbox, tuple(tb["bbox"]))
            dist = _center_distance(bbox, tuple(tb["bbox"]))
            if iou >= 0.3 or dist <= 12.0:
                match_pairs.append((pi, gi, max(iou, 0.01)))
    matches = _maximum_cardinality_matching(len(pred_items), len(gt_boxes), match_pairs,
        left_keys=[p[0] for p in pred_items], right_keys=[tb["object_id"] for tb in gt_boxes])
    pred_to_gt = {l: r for l, r, _ in matches}
    text_exact = sum(1 for pi, gi in pred_to_gt.items() if pred_items[pi][2] == gt_boxes[gi]["gt_value"])
    return len(pred_to_gt), text_exact


groups = build_groups(candidates)
print(f"groups: {len(groups)}", flush=True)

# Strategy 1: weighted majority vote
def sel_weighted_vote(members):
    tp = [(ci, preds[ci][0], preds[ci][1]) for ci in members if preds.get(ci, (None,0))[0]]
    if not tp: return None, None, None
    tw = defaultdict(float); tb = {}
    for ci, p, s in tp: tw[p] += s; tb[p] = (ci, s, candidates[ci]["original_bbox"])
    best = max(tw, key=lambda t: (tw[t], tb[t][1]))
    return tb[best][0], best, tb[best][2]

# Strategy 2: highest recognition score
def sel_highest_score(members):
    best_ci, best_s = None, -1
    for ci in members:
        p, s = preds.get(ci, (None, 0))
        if p and s > best_s: best_s = s; best_ci = ci
    if best_ci is None: return None, None, None
    return best_ci, preds[best_ci][0], candidates[best_ci]["original_bbox"]

# Strategy 3: score * geometric quality
def sel_score_x_geom(members):
    best_ci, best_s = None, -1
    for ci in members:
        p, s = preds.get(ci, (None, 0))
        if not p: continue
        ob = candidates[ci]["original_bbox"]; w, h = ob[2]-ob[0], ob[3]-ob[1]
        gq = 1.0
        if w>0 and h>0: gq = max(0, 1.0 - (max(w/h, h/w)-1)/4)
        area = w*h
        if area < 15: gq *= 0.5
        combined = s * gq
        if combined > best_s: best_s = combined; best_ci = ci
    if best_ci is None: return None, None, None
    return best_ci, preds[best_ci][0], candidates[best_ci]["original_bbox"]

# Strategy 4: top-3 multi-candidate (pick 3 per group)
def sel_top3_multi(members):
    scored = []
    for ci in members:
        p, s = preds.get(ci, (None, 0))
        if p: scored.append((s, ci))
    scored.sort(key=lambda x: -x[0])
    results = []
    for s, ci in scored[:3]:
        results.append((ci, preds[ci][0], candidates[ci]["original_bbox"]))
    return results or [(None, None, None)]

strategies = {
    "weighted_vote": sel_weighted_vote,
    "highest_score": sel_highest_score,
    "score_x_geom": sel_score_x_geom,
}

for name, fn in strategies.items():
    selections = []
    for i, members in enumerate(groups):
        sel_ci, pred, bbox = fn(members)
        selections.append({"group_id": f"g{i:04d}", "selected_index": sel_ci,
            "prediction": pred, "actual_bbox": bbox,
            "status": "accepted" if pred else "rejected"})
    n_matched, text_exact = score_selection(selections, gt_boxes)
    print(f"  {name:20s}: matched={n_matched}/{len(gt_boxes)} text_exact={text_exact}/{len(gt_boxes)} = {text_exact/len(gt_boxes):.4f}", flush=True)

# Top-3 multi
selections = []
for i, members in enumerate(groups):
    results = sel_top3_multi(members)
    for j, (sel_ci, pred, bbox) in enumerate(results):
        if pred:
            selections.append({"group_id": f"g{i:04d}_{j}", "selected_index": sel_ci,
                "prediction": pred, "actual_bbox": bbox, "status": "accepted"})
n_matched, text_exact = score_selection(selections, gt_boxes)
print(f"  {'top3_multi':20s}: matched={n_matched}/{len(gt_boxes)} text_exact={text_exact}/{len(gt_boxes)} = {text_exact/len(gt_boxes):.4f}", flush=True)

print(f"\n  oracle: 38/51 = 0.7451", flush=True)
