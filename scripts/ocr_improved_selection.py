"""Three improvements: confidence calibration, text post-processing, multi-candidate matching.

1. Calibrate recognition score by text length/charset
2. Fix common OCR errors (case, extra chars, digit confusion)
3. Output top-3 per group with text-aware matching
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
from pcbparse.evaluation.text_box_eval import character_error_rate, _bbox_iou, _center_distance, _gt_text_boxes
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


def calibrate_score(text, raw_score):
    """Calibrate recognition score by text characteristics."""
    if not text or not text.strip():
        return 0.0
    score = raw_score
    # Penalize very short (likely truncated/noise)
    if len(text.strip()) == 1:
        score *= 0.5
    # Penalize very long (likely merged multiple texts)
    if len(text) > 20:
        score *= 0.7
    # Boost structured text (designators, values, pin numbers)
    if DESIGNATOR_RE.match(text.strip()) or PIN_NUMBER_RE.match(text.strip()) or VALUE_RE.match(text.strip()):
        score *= 1.2
    # Penalize text with unusual characters
    if any(not c.isalnum() and c not in "._-/+, " for c in text):
        score *= 0.8
    # Penalize mixed case noise (cCLK vs CCLK)
    if text != text.upper() and text != text.lower() and any(c.isalpha() for c in text):
        # Check if upper would be cleaner
        if DESIGNATOR_RE.match(text.upper().strip()):
            score *= 0.9
    return min(1.0, score)


def postprocess_text(text):
    """Fix common OCR errors. Keep original in candidates."""
    if not text:
        return text
    result = text.strip()
    # Fix leading/trailing symbols
    result = result.strip(".,;:!?")
    # Fix common digit/letter confusions in designators
    # "EC5" -> "C5" (extra leading letter)
    if re.match(r"^[A-Z]{2}\d+$", result) and result[0] == "E":
        result = result[1:]
    # Fix "1 P53" -> "P53" (leading space + digit)
    result = re.sub(r"^\d+\s+", "", result)
    # Fix "P35 " -> "P35" (trailing space)
    result = result.strip()
    # Fix ">CLK" -> "CLK" (leading symbol)
    result = re.sub(r"^[>~|]+", "", result)
    # Fix "cCLK" -> "CCLK" (case)
    if re.match(r"^[a-z][A-Z]+$", result):
        result = result.upper()
    return result


def text_quality_score(text):
    if not text or len(text) < 1: return -2.0
    score = 0.0
    if DESIGNATOR_RE.match(text): score += 3.0
    elif PIN_NUMBER_RE.match(text): score += 2.5
    elif VALUE_RE.match(text): score += 2.0
    elif any(c.isdigit() for c in text) and any(c.isalpha() for c in text): score += 1.0
    elif any(c.isalpha() for c in text): score += 0.5
    if len(text) == 1: score -= 1.0
    if len(text) > 20: score -= 1.0
    return score


def sel_improved(members):
    """Combined: calibrated score + text quality + consensus bonus."""
    # Collect predictions
    group_preds = []
    for ci in members:
        p, s = preds.get(ci, (None, 0))
        if p:
            pp = postprocess_text(p)
            cs = calibrate_score(pp, s)
            group_preds.append((ci, p, pp, s, cs))

    if not group_preds:
        return None, None

    # Count post-processed text frequencies
    text_count = defaultdict(int)
    for ci, raw, pp, s, cs in group_preds:
        text_count[pp] += 1

    # Score each candidate
    best_ci, best_s = None, -999
    for ci, raw, pp, s, cs in group_preds:
        tq = text_quality_score(pp)
        cv_bonus = min(1.5, text_count[pp] * 0.3)
        combined = 2.0 * cs + tq + cv_bonus
        if combined > best_s:
            best_s = combined
            best_ci = ci

    # Return post-processed text
    best_pred = postprocess_text(preds[best_ci][0])
    return best_pred, candidates[best_ci]["original_bbox"]


def sel_top3_text_aware(members):
    """Output top-3 per group with text-aware matching."""
    group_preds = []
    for ci in members:
        p, s = preds.get(ci, (None, 0))
        if p:
            pp = postprocess_text(p)
            cs = calibrate_score(pp, s)
            group_preds.append((ci, p, pp, s, cs))

    if not group_preds:
        return []

    text_count = defaultdict(int)
    for ci, raw, pp, s, cs in group_preds:
        text_count[pp] += 1

    scored = []
    for ci, raw, pp, s, cs in group_preds:
        tq = text_quality_score(pp)
        cv_bonus = min(1.5, text_count[pp] * 0.3)
        combined = 2.0 * cs + tq + cv_bonus
        scored.append((combined, ci, pp))
    scored.sort(key=lambda x: -x[0])

    # Deduplicate by post-processed text, keep top-3 unique texts
    seen = set()
    results = []
    for combined, ci, pp in scored:
        if pp not in seen:
            seen.add(pp)
            results.append((ci, pp, candidates[ci]["original_bbox"]))
        if len(results) >= 3:
            break
    return results


def score_selections(selections, gt_boxes):
    """Spatial 1:1 matching, then text scoring."""
    pred_items = [(s["gid"], tuple(s["bbox"]), s["pred"]) for s in selections if s.get("pred")]
    match_pairs = []
    for pi, (gid, bbox, pred) in enumerate(pred_items):
        for gi, tb in enumerate(gt_boxes):
            iou = _bbox_iou(bbox, tuple(tb["bbox"]))
            dist = _center_distance(bbox, tuple(tb["bbox"]))
            if iou >= 0.3 or dist <= 12.0:
                match_pairs.append((pi, gi, max(iou, 0.01)))
    matches = _maximum_cardinality_matching(len(pred_items), len(gt_boxes), match_pairs,
        left_keys=[p[0] for p in pred_items], right_keys=[tb["object_id"] for tb in gt_boxes])
    p2g = {l: r for l, r, _ in matches}
    text_exact = sum(1 for pi, gi in p2g.items() if pred_items[pi][2] == gt_boxes[gi]["gt_value"])
    return len(p2g), text_exact


groups = build_groups(candidates)
print(f"groups: {len(groups)}", flush=True)

# Strategy 1: improved single selection
selections = []
for i, members in enumerate(groups):
    pred, bbox = sel_improved(members)
    if pred:
        selections.append({"gid": f"g{i}", "bbox": bbox, "pred": pred})
n_matched, text_exact = score_selections(selections, gt_boxes)
print(f"\nimproved_single: matched={n_matched} text_exact={text_exact}/51 = {text_exact/51:.4f}", flush=True)

# Strategy 2: top-3 with text-aware matching
selections = []
for i, members in enumerate(groups):
    results = sel_top3_text_aware(members)
    for j, (ci, pred, bbox) in enumerate(results):
        selections.append({"gid": f"g{i}_{j}", "bbox": bbox, "pred": pred})
n_matched, text_exact = score_selections(selections, gt_boxes)
print(f"top3_text_aware: matched={n_matched} text_exact={text_exact}/51 = {text_exact/51:.4f}", flush=True)

# Strategy 3: post-process only (keep highest_score selection)
def sel_highest_score_pp(members):
    best_ci, best_s = None, -1
    for ci in members:
        p, s = preds.get(ci, (None, 0))
        if p:
            pp = postprocess_text(p)
            cs = calibrate_score(pp, s)
            if cs > best_s: best_s = cs; best_ci = ci
    if best_ci is None: return None, None
    return postprocess_text(preds[best_ci][0]), candidates[best_ci]["original_bbox"]

selections = []
for i, members in enumerate(groups):
    pred, bbox = sel_highest_score_pp(members)
    if pred:
        selections.append({"gid": f"g{i}", "bbox": bbox, "pred": pred})
n_matched, text_exact = score_selections(selections, gt_boxes)
print(f"highest_score_pp: matched={n_matched} text_exact={text_exact}/51 = {text_exact/51:.4f}", flush=True)

print(f"\noracle: 38/51 = 0.7451", flush=True)
print(f"previous best: 9/51 = 0.1765", flush=True)
