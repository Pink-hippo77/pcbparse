"""Four diagnostic experiments to locate the 7.8% bottleneck.

1. 0005 GT crop baseline (same-image recognition ability)
2. Strict geometric matching with max total IoU (not max cardinality)
3. Visual comparison: GT box / matched box / actual crop for each of 51 groups
4. Oracle: best recognition among ALL candidates near each GT (diagnostic only)
"""
from __future__ import annotations
import json, sys, time, hashlib
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, r"E:\EDA挑战题目六\src")

WORKSPACE = Path(r"E:\EDA挑战题目六")
DATASET = WORKSPACE / "Dataset300" / "Dataset300"
CASE_ID = "0005"

from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth
from pcbparse.evaluation.text_box_eval import character_error_rate, _bbox_iou, _center_distance, _gt_text_boxes
from pcbparse.evaluation.masked_metrics import _maximum_cardinality_matching
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
print(f"0005: {len(gt_boxes)} GT text boxes", flush=True)

# Load frozen candidates from previous run
frozen_dirs = sorted((WORKSPACE / "work").glob("ocr-frozen-crops-*"))
if not frozen_dirs:
    raise FileNotFoundError("no frozen crops dir; run ocr_frozen_comparison.py first")
frozen_data = json.loads((frozen_dirs[-1] / "frozen_crops.json").read_text(encoding="utf-8"))
candidates = frozen_data["candidates"]
print(f"frozen candidates: {len(candidates)}", flush=True)


def crop_from_bbox(bbox, padding=2):
    x1, y1, x2, y2 = bbox
    cb = (max(0, int(x1) - padding), max(0, int(y1) - padding),
          min(img.width, int(x2) + padding), min(img.height, int(y2) + padding))
    if cb[2] <= cb[0] or cb[3] <= cb[1]:
        return None
    return np.array(img.crop(cb))


run_dir = WORKSPACE / "work" / f"ocr-diagnostic-{time.strftime('%Y%m%d-%H%M%S')}"
run_dir.mkdir(parents=True, exist_ok=True)

# === Experiment 1: 0005 GT crop baseline ===
print(f"\n=== Exp 1: 0005 GT crop baseline ===", flush=True)
gt_crops = [crop_from_bbox(tb["bbox"]) for tb in gt_boxes]
gt_records = [{"gt_value": tb["gt_value"], "field": tb["field"] if "field" in tb else "?", "object_id": tb["object_id"]} for tb in gt_boxes]
# Rebuild field from raw records
from pcbparse.evaluation.raw_ground_truth import RawGroundTruth
def get_field(gt_box_id):
    for comp in gt.components:
        for field, target in (("designator", comp.designator), ("value", comp.value), ("search_name", comp.search_name)):
            if f"{comp.raw_id}.{field}" == gt_box_id:
                return field
    for pin in gt.pins:
        for field, target in (("number", pin.number), ("name", pin.name)):
            if f"{pin.raw_id}.{field}" == gt_box_id:
                return field
    return "?"

for tb, rec in zip(gt_boxes, gt_records):
    rec["field"] = get_field(tb["object_id"])

for backend, label in [(torch_backend, "torch"), (paddle_backend, "paddle")]:
    results = backend.recognize_crops(gt_crops)
    exact = sum(1 for (pred, _s), rec in zip(results, gt_records) if pred == rec["gt_value"])
    cer_sum = sum(character_error_rate(rec["gt_value"], (pred or "")) * len(rec["gt_value"]) for (pred, _s), rec in zip(results, gt_records))
    total_chars = sum(len(rec["gt_value"]) for rec in gt_records)
    n = len(gt_records)
    print(f"  {label}: exact={exact}/{n} = {exact/n:.4f}  corpus_cer={cer_sum/total_chars:.4f}", flush=True)

# === Experiment 2: strict matching with max total IoU ===
print(f"\n=== Exp 2: strict matching (max total IoU) ===", flush=True)
# Build all candidate-GT IoU pairs
all_pairs = []
for ci, cand in enumerate(candidates):
    cb = tuple(cand["original_bbox"])
    for gi, tb in enumerate(gt_boxes):
        iou = _bbox_iou(cb, tb["bbox"])
        if iou > 0:
            all_pairs.append((ci, gi, iou))

# Greedy max-total-IoU matching (sort by IoU desc, assign greedily)
all_pairs.sort(key=lambda x: -x[2])
used_c = set()
used_g = set()
strict_matches = []
for ci, gi, iou in all_pairs:
    if ci not in used_c and gi not in used_g:
        strict_matches.append((ci, gi, iou))
        used_c.add(ci)
        used_g.add(gi)

print(f"  strict matched: {len(strict_matches)} (vs cardinality: {len(_maximum_cardinality_matching(len(candidates), len(gt_boxes), all_pairs, left_keys=[c['stable_id'] for c in candidates], right_keys=[tb['object_id'] for tb in gt_boxes]))})", flush=True)
print(f"  mean IoU of strict matches: {np.mean([m[2] for m in strict_matches]):.3f}", flush=True)

# Score with strict matches
strict_crops = []
strict_gts = []
for ci, gi, iou in strict_matches:
    cb = tuple(candidates[ci]["original_bbox"])
    crop = crop_from_bbox(cb)
    if crop is not None:
        strict_crops.append(crop)
        strict_gts.append(gt_boxes[gi])

for backend, label in [(torch_backend, "torch"), (paddle_backend, "paddle")]:
    results = backend.recognize_crops(strict_crops)
    exact = sum(1 for (pred, _s), tb in zip(results, strict_gts) if pred == tb["gt_value"])
    cer_sum = sum(character_error_rate(tb["gt_value"], (pred or "")) * len(tb["gt_value"]) for (pred, _s), tb in zip(results, strict_gts))
    total_chars = sum(len(tb["gt_value"]) for tb in strict_gts)
    n = len(strict_gts)
    print(f"  {label}: exact={exact}/{n} = {exact/n:.4f}  corpus_cer={cer_sum/total_chars:.4f}", flush=True)

# === Experiment 3: visual comparison for 51 groups ===
print(f"\n=== Exp 3: visual comparison images ===", flush=True)
# Use cardinality matches for visual comparison (original matching)
all_pairs_card = [(ci, gi, _bbox_iou(tuple(c["original_bbox"]), tb["bbox"])) for ci, c in enumerate(candidates) for gi, tb in enumerate(gt_boxes)]
all_pairs_card = [(ci, gi, iou) for ci, gi, iou in all_pairs_card if iou > 0.1 or _center_distance(tuple(candidates[ci]["original_bbox"]), gt_boxes[gi]["bbox"]) <= 12.0]
card_matches = _maximum_cardinality_matching(len(candidates), len(gt_boxes), all_pairs_card, left_keys=[c["stable_id"] for c in candidates], right_keys=[tb["object_id"] for tb in gt_boxes])

vis_dir = run_dir / "crop_comparisons"
vis_dir.mkdir(exist_ok=True)
for rank, (ci, gi, score) in enumerate(sorted(card_matches, key=lambda x: -x[2])[:51]):
    cand_bbox = candidates[ci]["original_bbox"]
    gt_bbox = gt_boxes[gi]["bbox"]
    # Draw comparison image
    x_min = int(min(cand_bbox[0], gt_bbox[0])) - 10
    y_min = int(min(cand_bbox[1], gt_bbox[1])) - 10
    x_max = int(max(cand_bbox[2], gt_bbox[2])) + 10
    y_max = int(max(cand_bbox[3], gt_bbox[3])) + 10
    crop = img.crop((max(0, x_min), max(0, y_min), min(img.width, x_max), min(img.height, y_max)))
    draw = ImageDraw.Draw(crop)
    ox, oy = max(0, x_min), max(0, y_min)
    # GT box in green
    draw.rectangle([gt_bbox[0]-ox, gt_bbox[1]-oy, gt_bbox[2]-ox, gt_bbox[3]-oy], outline=(0, 255, 0), width=2)
    # Candidate box in red
    draw.rectangle([cand_bbox[0]-ox, cand_bbox[1]-oy, cand_bbox[2]-ox, cand_bbox[3]-oy], outline=(255, 0, 0), width=2)
    # Labels
    gt_val = gt_boxes[gi]["gt_value"]
    iou = _bbox_iou(tuple(cand_bbox), tuple(gt_bbox))
    dist = _center_distance(tuple(cand_bbox), tuple(gt_bbox))
    label = f"GT:{gt_val} IoU={iou:.2f} d={dist:.0f}"
    draw.text((5, 5), label, fill=(255, 255, 0))
    crop.save(vis_dir / f"{rank:03d}_{gt_boxes[gi]['object_id'].replace('.', '_')}.png")

print(f"  saved {len(card_matches)} comparison images to {vis_dir}", flush=True)

# === Experiment 4: oracle best recognition near each GT ===
print(f"\n=== Exp 4: oracle best recognition near each GT (diagnostic only) ===", flush=True)
oracle_results = []
for gi, tb in enumerate(gt_boxes):
    # Find ALL candidates within 20px of this GT
    nearby = []
    for ci, cand in enumerate(candidates):
        cb = tuple(cand["original_bbox"])
        dist = _center_distance(cb, tb["bbox"])
        if dist <= 20.0:
            nearby.append((ci, dist, _bbox_iou(cb, tb["bbox"])))
    nearby.sort(key=lambda x: (-x[2], x[1]))  # best IoU first

    # Recognize all nearby with both backends, find best
    crops = [crop_from_bbox(tuple(candidates[ci]["original_bbox"])) for ci, _d, _i in nearby]
    crops = [c for c in crops if c is not None]
    if not crops:
        oracle_results.append({"gt_index": gi, "object_id": tb["object_id"], "gt": tb["gt_value"], "best_pred": None, "best_exact": False, "nearby_count": 0})
        continue

    t_preds = torch_backend.recognize_crops(crops)
    p_preds = paddle_backend.recognize_crops(crops)

    best_pred = None
    best_exact = False
    for (tp, _ts), (pp, _ps) in zip(t_preds, p_preds):
        for pred in [tp, pp]:
            if pred == tb["gt_value"]:
                best_pred = pred
                best_exact = True
                break
        if best_exact:
            break
    if not best_exact:
        # Pick best CER
        all_preds = [(tp, character_error_rate(tb["gt_value"], tp or "")) for tp, _ in t_preds] + [(pp, character_error_rate(tb["gt_value"], pp or "")) for pp, _ in p_preds]
        all_preds.sort(key=lambda x: x[1])
        best_pred = all_preds[0][0] if all_preds else None

    oracle_results.append({
        "gt_index": gi, "object_id": tb["object_id"], "gt": tb["gt_value"],
        "best_pred": best_pred, "best_exact": best_exact,
        "nearby_count": len(nearby),
    })

oracle_exact = sum(1 for r in oracle_results if r["best_exact"])
n_gt = len(gt_boxes)
print(f"  oracle best recognition: {oracle_exact}/{n_gt} = {oracle_exact/n_gt:.4f}", flush=True)
print(f"  (this is an UPPER BOUND; includes all nearby candidates, not production selection)", flush=True)

# Breakdown: how many GTs have a perfect crop nearby vs not
have_perfect = [r for r in oracle_results if r["best_exact"]]
no_perfect = [r for r in oracle_results if not r["best_exact"]]
print(f"  GTs with perfect candidate nearby: {len(have_perfect)}", flush=True)
print(f"  GTs WITHOUT perfect candidate: {len(no_perfect)}", flush=True)
for r in no_perfect[:5]:
    print(f"    {r['object_id']}: gt={r['gt']!r} best={r['best_pred']!r} nearby={r['nearby_count']}", flush=True)

# === Summary ===
report = {
    "kind": "bottleneck_localization",
    "case_id": CASE_ID,
    "gt_box_count": n_gt,
    "candidate_count": len(candidates),
    "exp1_gt_baseline": {"note": "same-image GT crop recognition"},
    "exp2_strict_matching": {
        "matched": len(strict_matches),
        "mean_iou": float(np.mean([m[2] for m in strict_matches])) if strict_matches else 0,
    },
    "exp3_visual": {"saved_to": str(vis_dir)},
    "exp4_oracle": {
        "oracle_exact": oracle_exact,
        "oracle_rate": oracle_exact / n_gt if n_gt else 0,
        "have_perfect_candidate": len(have_perfect),
        "no_perfect_candidate": len(no_perfect),
        "note": "UPPER BOUND diagnostic only; not production score",
    },
    "oracle_details": oracle_results,
}
out = run_dir / "diagnostic_report.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"\nsaved {out}", flush=True)
print(f"run_dir: {run_dir}", flush=True)
