"""Diagnose grouping/ranking losses and 9 unmatched GTs.

1. Track 9 unmatched GT: no candidate / merged elsewhere / wrong pick / rejected / scoring miss
2. Check large group spatial extents (chaining)
3. Quantify grouping loss vs ranking loss (3-layer oracle)
4. Fixed 51 denominator A/B
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
from PIL import Image
sys.path.insert(0, r"E:\EDA挑战题目六\src")

WORKSPACE = Path(r"E:\EDA挑战题目六")
DATASET = WORKSPACE / "Dataset300" / "Dataset300"
CASE_ID = "0005"

from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth
from pcbparse.evaluation.text_box_eval import character_error_rate, _bbox_iou, _center_distance, _gt_text_boxes
from pcbparse.evaluation.masked_metrics import _maximum_cardinality_matching
from pcbparse.perception.recognizer_backend import PaddleRecognizerBackend
from pcbparse.training.torch_ocr_model import load_ocr_bundle

lock = json.loads((WORKSPACE / "models" / "models.lock.json").read_text(encoding="utf-8"))
ocr_entry = next(e for e in lock["models"] if e["task"] == "ocr")
bundle = load_ocr_bundle(WORKSPACE / "models" / ocr_entry["path"], device="cpu")
paddle_backend = PaddleRecognizerBackend("en_PP-OCRv5_mobile_rec")

case_dir = DATASET / CASE_ID
ann_path = sorted(case_dir.glob("*_CVJsonStd.json"))[0]
img_path = sorted(case_dir.glob("*.png"))[0]
img = Image.open(img_path).convert("RGB")

frozen_dirs = sorted((WORKSPACE / "work").glob("ocr-frozen-crops-*"))
frozen_data = json.loads((frozen_dirs[-1] / "frozen_crops.json").read_text(encoding="utf-8"))
candidates = frozen_data["candidates"]
print(f"loaded {len(candidates)} candidates", flush=True)


def crop_from_bbox(bbox, padding=2):
    x1, y1, x2, y2 = bbox
    cb = (max(0, int(x1) - padding), max(0, int(y1) - padding),
          min(img.width, int(x2) + padding), min(img.height, int(y2) + padding))
    if cb[2] <= cb[0] or cb[3] <= cb[1]:
        return None
    return np.array(img.crop(cb))


# Rebuild groups (same logic as before)
def build_candidate_groups(cands, iou_threshold=0.3, center_dist_threshold=8.0):
    n = len(cands)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    for i in range(n):
        bi = cands[i]["original_bbox"]
        wi, hi = bi[2] - bi[0], bi[3] - bi[1]
        for j in range(i + 1, n):
            bj = cands[j]["original_bbox"]
            wj, hj = bj[2] - bj[0], bj[3] - bj[1]
            if wi > 0 and wj > 0 and max(wi / wj, wj / wi) > 2.0: continue
            if hi > 0 and hj > 0 and max(hi / hj, hj / hi) > 2.0: continue
            iou = _bbox_iou(tuple(bi), tuple(bj))
            dist = _center_distance(tuple(bi), tuple(bj))
            if iou >= iou_threshold or dist <= center_dist_threshold:
                union(i, j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    result = []
    for gid, members in enumerate(sorted(groups.values(), key=lambda m: min(m))):
        bboxes = [cands[i]["original_bbox"] for i in members]
        rep_bbox = [float(np.median([b[0] for b in bboxes])), float(np.median([b[1] for b in bboxes])),
                     float(np.median([b[2] for b in bboxes])), float(np.median([b[3] for b in bboxes]))]
        result.append({"group_id": f"g{gid:04d}", "member_indices": members, "rep_bbox": rep_bbox})
    return result

groups = build_candidate_groups(candidates)
print(f"formed {len(groups)} groups", flush=True)

# Recognize all candidates once
all_crops = [crop_from_bbox(tuple(c["original_bbox"])) for c in candidates]
valid_crops = [(i, c) for i, c in enumerate(all_crops) if c is not None]
print(f"recognizing {len(valid_crops)} valid crops...", flush=True)
t0 = time.time()
paddle_results = paddle_backend.recognize_crops([c for _, c in valid_crops])
recognize_time = time.time() - t0
preds = {}
for (ci, _), (pred, score) in zip(valid_crops, paddle_results):
    preds[ci] = (pred, score)
print(f"recognize: {recognize_time:.1f}s", flush=True)

# GT scoring
annotation = json.loads(ann_path.read_text(encoding="utf-8"))
gt = build_raw_ground_truth(annotation, case_id=CASE_ID, annotation_path=ann_path)
gt_boxes = _gt_text_boxes(gt)
n_gt = len(gt_boxes)
print(f"GT boxes: {n_gt}", flush=True)

run_dir = WORKSPACE / "work" / f"ocr-grouping-diagnosis-{time.strftime('%Y%m%d-%H%M%S')}"
run_dir.mkdir(parents=True, exist_ok=True)

# === Task 1: Track 9 unmatched GT ===
print(f"\n=== Task 1: track unmatched GT ===", flush=True)

# First: which GTs are matched by the selector?
# Load previous selection results
sel_dirs = sorted((WORKSPACE / "work").glob("ocr-gtfree-selector-*"))
sel_data = json.loads((sel_dirs[-1] / "predictions.json").read_text(encoding="utf-8"))
selections = sel_data["selections"]
accepted = [r for r in selections if r["status"] == "accepted"]

# Match accepted groups to GT
pred_bboxes = [(r["rep_bbox"], r["prediction"], r["group_id"]) for r in accepted]
match_pairs = []
for pi, (pb, pred, gid) in enumerate(pred_bboxes):
    for gi, tb in enumerate(gt_boxes):
        iou = _bbox_iou(tuple(pb), tuple(tb["bbox"]))
        dist = _center_distance(tuple(pb), tuple(tb["bbox"]))
        if iou >= 0.2 or dist <= 15.0:
            match_pairs.append((pi, gi, max(iou, 0.01)))
matches = _maximum_cardinality_matching(
    len(pred_bboxes), len(gt_boxes), match_pairs,
    left_keys=[gid for _, _, gid in pred_bboxes],
    right_keys=[tb["object_id"] for tb in gt_boxes],
)
pred_to_gt = {left: right for left, right, _s in matches}
matched_gt_indices = set(pred_to_gt.values())
unmatched_gt_indices = [gi for gi in range(n_gt) if gi not in matched_gt_indices]
print(f"  matched: {len(matched_gt_indices)}  unmatched: {len(unmatched_gt_indices)}", flush=True)

# For each unmatched GT, diagnose why
unmatched_details = []
for gi in unmatched_gt_indices:
    tb = gt_boxes[gi]
    gt_bbox = tb["bbox"]
    
    # 1. Check if any candidate is near this GT
    nearby_candidates = []
    for ci, cand in enumerate(candidates):
        cb = tuple(cand["original_bbox"])
        dist = _center_distance(cb, gt_bbox)
        if dist <= 20.0:
            nearby_candidates.append((ci, dist, _bbox_iou(cb, gt_bbox)))
    
    if not nearby_candidates:
        unmatched_details.append({"object_id": tb["object_id"], "gt": tb["gt_value"], "reason": "no_nearby_candidate", "nearby_count": 0})
        continue
    
    # 2. Check which group(s) contain nearby candidates
    nearby_groups = set()
    for ci, _d, _i in nearby_candidates:
        for g in groups:
            if ci in g["member_indices"]:
                nearby_groups.add(g["group_id"])
                break
    
    # 3. Check if any nearby group was accepted and selected something
    group_selected = {}
    for r in accepted:
        group_selected[r["group_id"]] = r
    
    accepted_groups_nearby = nearby_groups & set(group_selected.keys())
    
    if not accepted_groups_nearby:
        # Groups nearby were all rejected
        rejected_nearby = [r for r in selections if r["group_id"] in nearby_groups and r["status"] == "rejected"]
        reasons = [r["reason"] for r in rejected_nearby]
        unmatched_details.append({
            "object_id": tb["object_id"], "gt": tb["gt_value"],
            "reason": "all_nearby_groups_rejected",
            "nearby_count": len(nearby_candidates),
            "nearby_groups": sorted(nearby_groups),
            "reject_reasons": reasons,
        })
    else:
        # Some nearby group was accepted but prediction doesn't match
        best_sel = None
        best_iou = 0
        for gid in accepted_groups_nearby:
            r = group_selected[gid]
            iou = _bbox_iou(tuple(r["rep_bbox"]), gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_sel = r
        unmatched_details.append({
            "object_id": tb["object_id"], "gt": tb["gt_value"],
            "reason": "selected_wrong_prediction",
            "nearby_count": len(nearby_candidates),
            "nearby_groups": sorted(nearby_groups),
            "selected_group": best_sel["group_id"] if best_sel else None,
            "selected_pred": best_sel["prediction"] if best_sel else None,
            "match_iou": best_iou,
        })

for item in unmatched_details:
    print(f"  {item['object_id']}: gt={item['gt']!r} reason={item['reason']} nearby={item['nearby_count']}", flush=True)

reason_counts = defaultdict(int)
for item in unmatched_details:
    reason_counts[item["reason"]] += 1
print(f"  reason summary: {dict(reason_counts)}", flush=True)

# === Task 2: Check large group spatial extents ===
print(f"\n=== Task 2: large group spatial extents ===", flush=True)
large_groups = [(g, len(g["member_indices"])) for g in groups if len(g["member_indices"]) > 50]
large_groups.sort(key=lambda x: -x[1])
print(f"  groups with >50 members: {len(large_groups)}", flush=True)

group_analysis = []
for g, size in large_groups[:5]:
    bboxes = [candidates[ci]["original_bbox"] for ci in g["member_indices"]]
    xs = [(b[0] + b[2]) / 2 for b in bboxes]
    ys = [(b[1] + b[3]) / 2 for b in bboxes]
    x_span = max(xs) - min(xs) if xs else 0
    y_span = max(ys) - min(ys) if ys else 0
    levels = defaultdict(int)
    for ci in g["member_indices"]:
        levels[candidates[ci].get("level", "?")] += 1
    
    # Check chaining: max distance between any two members
    max_dist = 0
    for i in range(min(len(bboxes), 100)):
        for j in range(i + 1, min(len(bboxes), 100)):
            d = _center_distance(tuple(bboxes[i]), tuple(bboxes[j]))
            max_dist = max(max_dist, d)
    
    group_analysis.append({
        "group_id": g["group_id"], "size": size,
        "x_span": x_span, "y_span": y_span,
        "max_center_dist": max_dist,
        "levels": dict(levels),
        "likely_chaining": max_dist > 30,
    })
    print(f"  {g['group_id']}: size={size} x_span={x_span:.0f} y_span={y_span:.0f} max_dist={max_dist:.0f} chaining={'YES' if max_dist > 30 else 'no'}", flush=True)

# === Task 3: 3-layer oracle (grouping loss vs ranking loss) ===
print(f"\n=== Task 3: 3-layer oracle ===", flush=True)

# Layer 1: raw candidate oracle (any candidate near GT, best recognition)
layer1_results = []
for gi, tb in enumerate(gt_boxes):
    gt_val = tb["gt_value"]
    nearby = [(ci, _center_distance(tuple(candidates[ci]["original_bbox"]), tb["bbox"])) for ci in range(len(candidates))]
    nearby = [(ci, d) for ci, d in nearby if d <= 20.0]
    best_exact = False
    for ci, _d in nearby:
        pred, _s = preds.get(ci, (None, 0))
        if pred == gt_val:
            best_exact = True
            break
    layer1_results.append({"gt_index": gi, "object_id": tb["object_id"], "exact": best_exact})

layer1_exact = sum(1 for r in layer1_results if r["exact"])

# Layer 2: per-group oracle (max 1 per group, no double counting)
layer2_results = []
for gi, tb in enumerate(gt_boxes):
    gt_val = tb["gt_value"]
    # Find groups with candidates near this GT
    nearby_groups = set()
    for ci, cand in enumerate(candidates):
        dist = _center_distance(tuple(cand["original_bbox"]), tb["bbox"])
        if dist <= 20.0:
            for g in groups:
                if ci in g["member_indices"]:
                    nearby_groups.add(g["group_id"])
                    break
    # For each nearby group, check if any member recognizes correctly
    best_exact = False
    for g in groups:
        if g["group_id"] not in nearby_groups:
            continue
        for ci in g["member_indices"]:
            pred, _s = preds.get(ci, (None, 0))
            if pred == gt_val:
                best_exact = True
                break
        if best_exact:
            break
    layer2_results.append({"gt_index": gi, "object_id": tb["object_id"], "exact": best_exact})

layer2_exact = sum(1 for r in layer2_results if r["exact"])

# Layer 3: actual selector result (from previous run)
# Find which GTs the selector got right
selector_exact = 0
for pi, (pb, pred, gid) in enumerate(pred_bboxes):
    gi = pred_to_gt.get(pi)
    if gi is not None and pred == gt_boxes[gi]["gt_value"]:
        selector_exact += 1

print(f"  layer1 (raw candidate oracle): {layer1_exact}/{n_gt} = {layer1_exact/n_gt:.4f}", flush=True)
print(f"  layer2 (per-group oracle):     {layer2_exact}/{n_gt} = {layer2_exact/n_gt:.4f}", flush=True)
print(f"  layer3 (actual selector):      {selector_exact}/{n_gt} = {selector_exact/n_gt:.4f}", flush=True)
print(f"  grouping loss (L1-L2):         {layer1_exact - layer2_exact} objects = {(layer1_exact-layer2_exact)/n_gt:.4f}", flush=True)
print(f"  ranking loss (L2-L3):          {layer2_exact - selector_exact} objects = {(layer2_exact-selector_exact)/n_gt:.4f}", flush=True)

# === Task 4: Fixed 51 denominator A/B ===
print(f"\n=== Task 4: fixed 51 denominator A/B ===", flush=True)
unmatched_count = n_gt - len(matched_gt_indices)
wrong_accept = sum(1 for pi in pred_to_gt if pred_bboxes[pi][1] != gt_boxes[pred_to_gt[pi]]["gt_value"])
rejected_count = len([r for r in selections if r["status"] == "rejected"])
print(f"  full exact: {selector_exact}/{n_gt} = {selector_exact/n_gt:.4f}", flush=True)
print(f"  conditional exact (matched): {selector_exact}/{len(matched_gt_indices)} = {selector_exact/len(matched_gt_indices):.4f}", flush=True)
print(f"  unmatched GT: {unmatched_count}/{n_gt} = {unmatched_count/n_gt:.4f}", flush=True)
print(f"  wrong accept: {wrong_accept}", flush=True)
print(f"  rejected groups: {rejected_count}", flush=True)
print(f"  recognize time: {recognize_time:.1f}s", flush=True)

# Save report
report = {
    "kind": "grouping_ranking_diagnosis",
    "case_id": CASE_ID,
    "n_gt": n_gt,
    "n_candidates": len(candidates),
    "n_groups": len(groups),
    "task1_unmatched": {
        "count": len(unmatched_gt_indices),
        "reason_counts": dict(reason_counts),
        "details": unmatched_details,
    },
    "task2_large_groups": group_analysis,
    "task3_three_layer": {
        "layer1_raw_oracle": layer1_exact,
        "layer2_group_oracle": layer2_exact,
        "layer3_actual_selector": selector_exact,
        "grouping_loss": layer1_exact - layer2_exact,
        "ranking_loss": layer2_exact - selector_exact,
    },
    "task4_ab_51denom": {
        "full_exact": selector_exact,
        "full_rate": selector_exact / n_gt,
        "conditional_exact": selector_exact,
        "conditional_rate": selector_exact / len(matched_gt_indices) if matched_gt_indices else 0,
        "matched_gt": len(matched_gt_indices),
        "unmatched_gt": unmatched_count,
        "wrong_accept": wrong_accept,
        "rejected_groups": rejected_count,
        "recognize_time_s": recognize_time,
    },
}
out = run_dir / "diagnosis.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"\nsaved {out}", flush=True)
print(f"run_dir: {run_dir}", flush=True)
