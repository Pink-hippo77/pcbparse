"""Fix weighted_vote bug + test combined heuristic.

weighted_vote was returning candidate index instead of predicted text.
"""
from __future__ import annotations
import json, sys, re
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
print(f"loaded {len(gt_boxes)} GT, {len(candidates)} candidates", flush=True)


def build_groups(cands, iou_thresh=0.3, dist_thresh=8.0, scale_limit=2.0):
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


DESIGNATOR_RE = re.compile(r"^[A-Z]{1,4}\d{1,4}[A-Z]?$")
PIN_NUMBER_RE = re.compile(r"^\d{1,3}(,\d{1,3})*$")
VALUE_RE = re.compile(r"^\d+(\.\d+)?\s*[kKmMuUnNpP]?[Ωohm%FfHhVvAaWw]?$")
ALNUM_PIN_RE = re.compile(r"^(?:[A-Z]{1,2}\d{1,3}|\d{1,3}[A-Z]{1,2})$")


def text_quality_score(text):
    if not text or len(text) < 1: return -2.0
    score = 0.0
    if DESIGNATOR_RE.match(text): score += 3.0
    elif PIN_NUMBER_RE.match(text): score += 2.5
    elif VALUE_RE.match(text): score += 2.0
    elif ALNUM_PIN_RE.match(text): score += 2.0
    elif any(c.isdigit() for c in text) and any(c.isalpha() for c in text): score += 1.0
    elif any(c.isalpha() for c in text): score += 0.5
    if len(text) == 1: score -= 1.0
    if len(text) > 20: score -= 1.0
    if any(not c.isalnum() and c not in "._-/+, " for c in text): score -= 0.5
    return score


def sel_weighted_vote_fixed(members):
    """Fixed: return predicted text, not candidate index."""
    tp = [(ci, preds[ci][0], preds[ci][1]) for ci in members if preds.get(ci, (None,0))[0]]
    if not tp: return None, None
    tw = defaultdict(float); tb = {}
    for ci, p, s in tp: tw[p] += s; tb[p] = (ci, s, candidates[ci]["original_bbox"])
    best_text = max(tw, key=lambda t: (tw[t], tb[t][1]))
    return best_text, tb[best_text][2]  # return TEXT not index


def sel_highest_score(members):
    best_ci, best_s = None, -1
    for ci in members:
        p, s = preds.get(ci, (None, 0))
        if p and s > best_s: best_s = s; best_ci = ci
    if best_ci is None: return None, None
    return preds[best_ci][0], candidates[best_ci]["original_bbox"]


def sel_combined(members):
    """Combined: recognition score + text quality + cross-view bonus."""
    best_ci, best_s = None, -999
    # Count text frequencies for cross-view bonus
    text_count = defaultdict(int)
    for ci in members:
        p, _s = preds.get(ci, (None, 0))
        if p: text_count[p] += 1
    for ci in members:
        p, s = preds.get(ci, (None, 0))
        if not p: continue
        tq = text_quality_score(p)
        # Cross-view bonus: if text appears multiple times in group
        cv_bonus = min(2.0, text_count[p] * 0.5)
        combined = 2.0 * s + tq + cv_bonus
        if combined > best_s: best_s = combined; best_ci = ci
    if best_ci is None: return None, None
    return preds[best_ci][0], candidates[best_ci]["original_bbox"]


def sel_consensus(members):
    """Consensus: pick text with most votes, tiebreak by max score."""
    tp = [(ci, preds[ci][0], preds[ci][1]) for ci in members if preds.get(ci, (None,0))[0]]
    if not tp: return None, None
    text_count = defaultdict(int)
    text_best = {}
    for ci, p, s in tp:
        text_count[p] += 1
        if p not in text_best or s > text_best[p][1]:
            text_best[p] = (ci, s, candidates[ci]["original_bbox"])
    best_text = max(text_count, key=lambda t: (text_count[t], text_best[t][1]))
    return best_text, text_best[best_text][2]


def score_fn(sel_fn, groups):
    selections = []
    for i, members in enumerate(groups):
        pred, bbox = sel_fn(members)
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
    return len(p2g), text_exact


groups = build_groups(candidates)
print(f"groups: {len(groups)}", flush=True)

strategies = [
    ("highest_score", sel_highest_score),
    ("weighted_vote_fixed", sel_weighted_vote_fixed),
    ("consensus", sel_consensus),
    ("combined", sel_combined),
]

print(f"\n{'strategy':25s} {'match':>7} {'exact':>7} {'rate':>7}", flush=True)
for name, fn in strategies:
    n_matched, text_exact = score_fn(fn, groups)
    print(f"{name:25s} {n_matched:>7} {text_exact:>7} {text_exact/51:>7.4f}", flush=True)

print(f"\noracle: 38/51 = 0.7451", flush=True)
