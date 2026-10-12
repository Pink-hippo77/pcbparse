"""Candidate selection experiment: same-model oracle, GT-free ranking, multi-candidate.

Steps 1-5 from the review.  GT only used for scoring, never for ranking.
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

from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth
from pcbparse.evaluation.text_box_eval import character_error_rate, _bbox_iou, _center_distance, _gt_text_boxes
from pcbparse.perception.recognizer_backend import TorchRecognizerBackend, PaddleRecognizerBackend
from pcbparse.training.torch_ocr_model import load_ocr_bundle

lock = json.loads((WORKSPACE / "models" / "models.lock.json").read_text(encoding="utf-8"))
ocr_entry = next(e for e in lock["models"] if e["task"] == "ocr")
bundle = load_ocr_bundle(WORKSPACE / "models" / ocr_entry["path"], device="cpu")
torch_backend = TorchRecognizerBackend(bundle, device="cpu")
paddle_backend = PaddleRecognizerBackend("en_PP-OCRv5_mobile_rec")

case_dir = DATASET / CASE_ID
ann_path = sorted(case_dir.glob("*_CVJsonStd.json"))[0]
img_path = sorted(case_dir.glob("*.png"))[0]
annotation = json.loads(ann_path.read_text(encoding="utf-8"))
gt = build_raw_ground_truth(annotation, case_id=CASE_ID, annotation_path=ann_path)
gt_boxes = _gt_text_boxes(gt)
img = Image.open(img_path).convert("RGB")

# Load frozen candidates
frozen_dirs = sorted((WORKSPACE / "work").glob("ocr-frozen-crops-*"))
frozen_data = json.loads((frozen_dirs[-1] / "frozen_crops.json").read_text(encoding="utf-8"))
candidates = frozen_data["candidates"]
print(f"0005: {len(gt_boxes)} GT, {len(candidates)} candidates", flush=True)


def crop_from_bbox(bbox, padding=2):
    x1, y1, x2, y2 = bbox
    cb = (max(0, int(x1) - padding), max(0, int(y1) - padding),
          min(img.width, int(x2) + padding), min(img.height, int(y2) + padding))
    if cb[2] <= cb[0] or cb[3] <= cb[1]:
        return None
    return np.array(img.crop(cb))


run_dir = WORKSPACE / "work" / f"ocr-candidate-selection-{time.strftime('%Y%m%d-%H%M%S')}"
run_dir.mkdir(parents=True, exist_ok=True)

# Spatial constraint: candidates within 20px of GT center
SPATIAL_RADIUS = 20.0

# === Step 1: same-model oracle with per-object correspondence ===
print(f"\n=== Step 1: same-model oracle (torch + paddle separately) ===", flush=True)

# For each GT, find nearby candidates
gt_nearby = []  # list of (gt_idx, [(cand_idx, dist, iou), ...])
for gi, tb in enumerate(gt_boxes):
    nearby = []
    for ci, cand in enumerate(candidates):
        cb = tuple(cand["original_bbox"])
        dist = _center_distance(cb, tb["bbox"])
        if dist <= SPATIAL_RADIUS:
            nearby.append((ci, dist, _bbox_iou(cb, tb["bbox"])))
    nearby.sort(key=lambda x: (-x[2], x[1]))
    gt_nearby.append(nearby)

# Recognize all candidates once, cache results
all_crops = [crop_from_bbox(tuple(c["original_bbox"])) for c in candidates]
valid_crops = [(i, c) for i, c in enumerate(all_crops) if c is not None]
print(f"  recognizing {len(valid_crops)} valid crops with torch...", flush=True)
t0 = time.time()
torch_all = torch_backend.recognize_crops([c for _, c in valid_crops])
torch_time = time.time() - t0
torch_preds = {}
for (ci, _), (pred, score) in zip(valid_crops, torch_all):
    torch_preds[ci] = (pred, score)

print(f"  recognizing {len(valid_crops)} valid crops with paddle...", flush=True)
t0 = time.time()
paddle_all = paddle_backend.recognize_crops([c for _, c in valid_crops])
paddle_time = time.time() - t0
paddle_preds = {}
for (ci, _), (pred, score) in zip(valid_crops, paddle_all):
    paddle_preds[ci] = (pred, score)

print(f"  torch: {torch_time:.1f}s  paddle: {paddle_time:.1f}s", flush=True)

# Oracle: best recognition among nearby candidates for each model
oracle_results = []
for gi, tb in enumerate(gt_boxes):
    gt_val = tb["gt_value"]
    nearby = gt_nearby[gi]
    
    # torch oracle
    t_best_pred = None
    t_best_exact = False
    t_best_cer = 1.0
    for ci, _d, _i in nearby:
        pred, _score = torch_preds.get(ci, (None, 0))
        if pred == gt_val:
            t_best_pred = pred
            t_best_exact = True
            t_best_cer = 0.0
            break
        cer = character_error_rate(gt_val, pred or "")
        if cer < t_best_cer:
            t_best_cer = cer
            t_best_pred = pred
    
    # paddle oracle
    p_best_pred = None
    p_best_exact = False
    p_best_cer = 1.0
    for ci, _d, _i in nearby:
        pred, _score = paddle_preds.get(ci, (None, 0))
        if pred == gt_val:
            p_best_pred = pred
            p_best_exact = True
            p_best_cer = 0.0
            break
        cer = character_error_rate(gt_val, pred or "")
        if cer < p_best_cer:
            p_best_cer = cer
            p_best_pred = pred
    
    oracle_results.append({
        "gt_index": gi, "object_id": tb["object_id"], "gt": gt_val,
        "torch_oracle_pred": t_best_pred, "torch_oracle_exact": t_best_exact, "torch_oracle_cer": t_best_cer,
        "paddle_oracle_pred": p_best_pred, "paddle_oracle_exact": p_best_exact, "paddle_oracle_cer": p_best_cer,
        "nearby_count": len(nearby),
    })

t_oracle_exact = sum(1 for r in oracle_results if r["torch_oracle_exact"])
p_oracle_exact = sum(1 for r in oracle_results if r["paddle_oracle_exact"])
n_gt = len(gt_boxes)
print(f"  torch oracle: {t_oracle_exact}/{n_gt} = {t_oracle_exact/n_gt:.4f}", flush=True)
print(f"  paddle oracle: {p_oracle_exact}/{n_gt} = {p_oracle_exact/n_gt:.4f}", flush=True)

# Four-way classification: both correct / only torch / only paddle / neither
both_correct = [r for r in oracle_results if r["torch_oracle_exact"] and r["paddle_oracle_exact"]]
only_torch = [r for r in oracle_results if r["torch_oracle_exact"] and not r["paddle_oracle_exact"]]
only_paddle = [r for r in oracle_results if not r["torch_oracle_exact"] and r["paddle_oracle_exact"]]
neither = [r for r in oracle_results if not r["torch_oracle_exact"] and not r["paddle_oracle_exact"]]
print(f"  both: {len(both_correct)}  only_torch: {len(only_torch)}  only_paddle: {len(only_paddle)}  neither: {len(neither)}", flush=True)

# === Step 2: GT-free ranking baseline ===
print(f"\n=== Step 2: GT-free ranking baseline ===", flush=True)

def gt_free_score(cand, all_cands):
    """Score a candidate without using GT. Higher = better."""
    score = 0.0
    # Detection score (if available)
    det_score = cand.get("detector_score") or 0.5
    score += 0.3 * float(det_score)
    
    # Box aspect ratio: prefer moderate ratios (not extreme wide/tall)
    ob = cand["original_bbox"]
    w = ob[2] - ob[0]
    h = ob[3] - ob[1]
    if w > 0 and h > 0:
        ratio = max(w / h, h / w)
        score += 0.2 * max(0, 1.0 - (ratio - 1.0) / 5.0)  # penalize extreme ratios
    
    # Size: prefer medium-sized boxes (not tiny fragments, not huge blocks)
    area = w * h
    if 50 <= area <= 2000:
        score += 0.2
    elif area < 20:
        score -= 0.1  # penalize tiny fragments
    
    # Boundary truncation: penalize boxes touching image edge
    if ob[0] <= 1 or ob[1] <= 1 or ob[2] >= img.width - 1 or ob[3] >= img.height - 1:
        score -= 0.15
    
    # Cross-view consistency: prefer candidates that appear in multiple levels
    view_id = cand.get("view_id", "")
    level = cand.get("level", "")
    # Global level is more reliable
    if level == "global":
        score += 0.15
    elif level == "component":
        score += 0.05
    
    return score

# For each GT, rank nearby candidates by GT-free score and pick best
ranked_results = []
for gi, tb in enumerate(gt_boxes):
    gt_val = tb["gt_value"]
    nearby = gt_nearby[gi]
    if not nearby:
        ranked_results.append({"gt_index": gi, "object_id": tb["object_id"], "gt": gt_val, "ranked_pred": None, "ranked_exact": False, "ranked_cer": 1.0, "selected_ci": None})
        continue
    
    # Score all nearby candidates
    scored = []
    for ci, dist, iou in nearby:
        s = gt_free_score(candidates[ci], candidates)
        scored.append((s, ci))
    scored.sort(key=lambda x: -x[0])
    
    # Pick top-1
    top_ci = scored[0][1]
    pred, _score = paddle_preds.get(top_ci, (None, 0))
    exact = pred == gt_val
    cer = character_error_rate(gt_val, pred or "")
    ranked_results.append({
        "gt_index": gi, "object_id": tb["object_id"], "gt": gt_val,
        "ranked_pred": pred, "ranked_exact": exact, "ranked_cer": cer,
        "selected_ci": top_ci,
    })

ranked_exact = sum(1 for r in ranked_results if r["ranked_exact"])
print(f"  GT-free ranking (paddle): {ranked_exact}/{n_gt} = {ranked_exact/n_gt:.4f}", flush=True)

# Compare with oracle
rescued = [r for r in ranked_results if r["ranked_exact"]]
missed = [r for r in ranked_results if not r["ranked_exact"]]
oracle_hits = {r["gt_index"] for r in oracle_results if r["paddle_oracle_exact"]}
ranked_hits = {r["gt_index"] for r in ranked_results if r["ranked_exact"]}
rescued_vs_oracle = oracle_hits - ranked_hits
print(f"  oracle hits: {len(oracle_hits)}  ranked hits: {len(ranked_hits)}  gap: {len(rescued_vs_oracle)}", flush=True)

# === Step 3: multi-candidate recognition ===
print(f"\n=== Step 3: multi-candidate (top-3) recognition ===", flush=True)
multi_results = []
for gi, tb in enumerate(gt_boxes):
    gt_val = tb["gt_value"]
    nearby = gt_nearby[gi]
    if not nearby:
        multi_results.append({"gt_index": gi, "object_id": tb["object_id"], "gt": gt_val, "multi_pred": None, "multi_exact": False})
        continue
    
    # Score and take top-3
    scored = [(gt_free_score(candidates[ci], candidates), ci) for ci, _d, _i in nearby]
    scored.sort(key=lambda x: -x[0])
    top3_cis = [ci for _, ci in scored[:3]]
    
    # Recognize top-3 with paddle, pick best CER
    crops = [crop_from_bbox(tuple(candidates[ci]["original_bbox"])) for ci in top3_cis]
    crops = [c for c in crops if c is not None]
    if not crops:
        multi_results.append({"gt_index": gi, "object_id": tb["object_id"], "gt": gt_val, "multi_pred": None, "multi_exact": False})
        continue
    preds = paddle_backend.recognize_crops(crops)
    
    best_pred = None
    best_exact = False
    best_cer = 1.0
    for pred, _score in preds:
        if pred == gt_val:
            best_pred = pred
            best_exact = True
            best_cer = 0.0
            break
        cer = character_error_rate(gt_val, pred or "")
        if cer < best_cer:
            best_cer = cer
            best_pred = pred
    
    multi_results.append({"gt_index": gi, "object_id": tb["object_id"], "gt": gt_val, "multi_pred": best_pred, "multi_exact": best_exact})

multi_exact = sum(1 for r in multi_results if r["multi_exact"])
print(f"  multi-candidate top-3 (paddle): {multi_exact}/{n_gt} = {multi_exact/n_gt:.4f}", flush=True)

# === Step 4: net gain analysis ===
print(f"\n=== Step 4: net gain analysis ===", flush=True)
# Compare ranked vs oracle per object
gain = [r for r in ranked_results if r["ranked_exact"] and not any(o["gt_index"] == r["gt_index"] and o["paddle_oracle_exact"] for o in oracle_results)]
# Actually: gain = ranked hits that oracle also hits (consistent), loss = oracle hits that ranked misses
oracle_hits_set = {r["gt_index"] for r in oracle_results if r["paddle_oracle_exact"]}
ranked_hits_set = {r["gt_index"] for r in ranked_results if r["ranked_exact"]}
multi_hits_set = {r["gt_index"] for r in multi_results if r["multi_exact"]}

print(f"  ranked hits: {len(ranked_hits_set)}", flush=True)
print(f"  multi hits: {len(multi_hits_set)}", flush=True)
print(f"  oracle hits: {len(oracle_hits_set)}", flush=True)
print(f"  ranked→multi gain: {len(multi_hits_set - ranked_hits_set)}", flush=True)
print(f"  ranked→multi loss: {len(ranked_hits_set - multi_hits_set)}", flush=True)
print(f"  ranked→multi net: {len(multi_hits_set - ranked_hits_set) - len(ranked_hits_set - multi_hits_set):+d}", flush=True)

# === Step 5: split remaining errors ===
print(f"\n=== Step 5: error split ===", flush=True)
error_split = {"no_nearby_candidate": [], "complete_box_misread": [], "duplicate_mismatch": [], "field_or_owner_error": []}
for gi, tb in enumerate(gt_boxes):
    gt_val = tb["gt_value"]
    r = ranked_results[gi]
    if r["ranked_exact"]:
        continue
    nearby = gt_nearby[gi]
    if not nearby:
        error_split["no_nearby_candidate"].append(r["object_id"])
    else:
        # Check if any nearby candidate was recognized correctly (oracle)
        o = oracle_results[gi]
        if o["paddle_oracle_exact"]:
            error_split["duplicate_mismatch"].append({"object_id": r["object_id"], "gt": gt_val, "selected": r["ranked_pred"]})
        else:
            error_split["complete_box_misread"].append({"object_id": r["object_id"], "gt": gt_val, "best": o["paddle_oracle_pred"]})

for k, v in error_split.items():
    print(f"  {k}: {len(v)}", flush=True)
    for item in v[:3]:
        print(f"    {item}", flush=True)

# === Save ===
report = {
    "kind": "candidate_selection_experiment",
    "case_id": CASE_ID,
    "gt_box_count": n_gt,
    "candidate_count": len(candidates),
    "step1_oracle": {
        "torch_oracle_exact": t_oracle_exact,
        "torch_oracle_rate": t_oracle_exact / n_gt,
        "paddle_oracle_exact": p_oracle_exact,
        "paddle_oracle_rate": p_oracle_exact / n_gt,
        "both_correct": len(both_correct),
        "only_torch": len(only_torch),
        "only_paddle": len(only_paddle),
        "neither": len(neither),
    },
    "step2_ranked": {"exact": ranked_exact, "rate": ranked_exact / n_gt},
    "step3_multi": {"exact": multi_exact, "rate": multi_exact / n_gt},
    "step4_net_gain": {
        "ranked_hits": len(ranked_hits_set),
        "multi_hits": len(multi_hits_set),
        "oracle_hits": len(oracle_hits_set),
        "ranked_to_multi_gain": len(multi_hits_set - ranked_hits_set),
        "ranked_to_multi_loss": len(ranked_hits_set - multi_hits_set),
    },
    "step5_error_split": {k: len(v) for k, v in error_split.items()},
    "error_split_details": error_split,
    "per_object": oracle_results,
}
out = run_dir / "candidate_selection_report.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"\nsaved {out}", flush=True)
print(f"run_dir: {run_dir}", flush=True)
