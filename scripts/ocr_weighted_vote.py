"""Weighted majority vote ranking + cached recognition for fast iteration.

Key insight: within a group, multiple candidates of the same physical text
will recognize to the same string.  Weighted vote by recognition score picks
the consensus, which is more robust than single-candidate geometric scoring.
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
from pcbparse.evaluation.text_box_eval import (
    character_error_rate, _bbox_iou, _center_distance, _gt_text_boxes,
)
from pcbparse.evaluation.masked_metrics import _maximum_cardinality_matching
from pcbparse.perception.recognizer_backend import PaddleRecognizerBackend
from pcbparse.training.torch_ocr_model import load_ocr_bundle

# Load models
lock = json.loads((WORKSPACE / "models" / "models.lock.json").read_text(encoding="utf-8"))
ocr_entry = next(e for e in lock["models"] if e["task"] == "ocr")
bundle = load_ocr_bundle(WORKSPACE / "models" / ocr_entry["path"], device="cpu")
paddle_backend = PaddleRecognizerBackend("en_PP-OCRv5_mobile_rec")

case_dir = DATASET / CASE_ID
ann_path = sorted(case_dir.glob("*_CVJsonStd.json"))[0]
img_path = sorted(case_dir.glob("*.png"))[0]
img = Image.open(img_path).convert("RGB")
annotation = json.loads(ann_path.read_text(encoding="utf-8"))
gt = build_raw_ground_truth(annotation, case_id=CASE_ID, annotation_path=ann_path)
gt_boxes = _gt_text_boxes(gt)

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


# === Cached recognition ===
CACHE_DIR.mkdir(parents=True, exist_ok=True)
cache_file = CACHE_DIR / f"{CASE_ID}_paddle_recognition.json"

if cache_file.exists():
    print(f"loading cached recognition from {cache_file}", flush=True)
    cache_data = json.loads(cache_file.read_text(encoding="utf-8"))
    preds = {int(k): (v["pred"], v["score"]) for k, v in cache_data["preds"].items()}
    print(f"  loaded {len(preds)} cached predictions", flush=True)
else:
    print(f"recognizing {len(candidates)} candidates (caching)...", flush=True)
    all_crops = [crop_from_bbox(tuple(c["original_bbox"])) for c in candidates]
    valid_crops = [(i, c) for i, c in enumerate(all_crops) if c is not None]
    t0 = time.time()
    rec_results = paddle_backend.recognize_crops([c for _, c in valid_crops])
    rec_time = time.time() - t0
    preds = {}
    for (ci, _), (pred, score) in zip(valid_crops, rec_results):
        preds[ci] = (pred, score)
    cache_data = {"case_id": CASE_ID, "model": "en_PP-OCRv5_mobile_rec",
                  "recognition_time_s": rec_time,
                  "preds": {str(k): {"pred": v[0], "score": v[1]} for k, v in preds.items()}}
    cache_file.write_text(json.dumps(cache_data, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  recognized in {rec_time:.1f}s, cached to {cache_file}", flush=True)


# === Build groups ===
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
        levels = {cands[i].get("level", "?") for i in members}
        view_ids = {cands[i].get("view_id", "?") for i in members}
        result.append({"group_id": f"g{gid:04d}", "member_indices": members,
                       "rep_bbox": rep_bbox, "levels": levels, "view_ids": view_ids})
    return result

groups = build_candidate_groups(candidates)
print(f"formed {len(groups)} groups", flush=True)


# === Weighted majority vote selection ===
def select_weighted_vote(group, all_cands, preds):
    """Select best candidate via weighted majority vote on recognition results."""
    group_preds = []
    for ci in group["member_indices"]:
        pred, score = preds.get(ci, (None, 0))
        if pred is not None and pred != "":
            group_preds.append((ci, pred, score))

    if not group_preds:
        return None, None, None, "rejected", "empty_recognition"

    # Weighted vote: sum recognition scores per text
    text_weight = defaultdict(float)
    text_best = {}  # text -> (ci, score, bbox)
    for ci, pred, score in group_preds:
        text_weight[pred] += score
        if pred not in text_best or score > text_best[pred][1]:
            text_best[pred] = (ci, score, all_cands[ci]["original_bbox"])

    # Pick text with highest total weight
    best_text = max(text_weight.keys(), key=lambda t: (text_weight[t], text_best[t][1]))
    best_ci, best_score, best_bbox = text_best[best_text]

    return best_ci, best_text, best_bbox, "accepted", None


# Run selection
selections = []
for g in groups:
    sel_ci, pred, actual_bbox, status, reason = select_weighted_vote(g, candidates, preds)
    selections.append({"group_id": g["group_id"], "selected_index": sel_ci,
        "prediction": pred, "actual_bbox": actual_bbox, "status": status, "reason": reason})

accepted = [s for s in selections if s["status"] == "accepted"]
rejected = [s for s in selections if s["status"] == "rejected"]
print(f"\naccepted: {len(accepted)}  rejected: {len(rejected)}", flush=True)

# === Field classification + binding ===
from pcbparse.ir.models import BBox, Component, Pin, Point, TextToken
from pcbparse.association.semantic_fields import infer_semantic_fields
from pcbparse.association.text_component import associate_component_texts
from pcbparse.association.text_pin import associate_pin_texts
from pcbparse.association.ocr_common import OCRAssociationConfig, canonical_details

components = []
pins = []
for comp in gt.components:
    if comp.mask.eligible and comp.bbox:
        components.append(Component(internal_id=comp.raw_id, key=None, category=comp.category,
            bbox=BBox(x1=comp.bbox[0], y1=comp.bbox[1], x2=comp.bbox[2], y2=comp.bbox[3])))
for pin in gt.pins:
    if pin.mask.eligible and pin.tip:
        base = Point(x=pin.base[0], y=pin.base[1]) if pin.base else None
        pins.append(Pin(internal_id=pin.raw_id, component_id=pin.owner_raw_id,
            point=Point(x=pin.tip[0], y=pin.tip[1]), base=base))

tokens = []
for s in accepted:
    ab = s["actual_bbox"]
    token = TextToken(internal_id=s["group_id"], raw_text=s["prediction"] or "",
        bbox=BBox(x1=ab[0], y1=ab[1], x2=ab[2], y2=ab[3]))
    tokens.append(token)

infer_semantic_fields(components, pins, tokens, [])
diagnostics = []
config = OCRAssociationConfig()
associate_component_texts(components, tokens, diagnostics, config)
associate_pin_texts(pins, tokens, diagnostics, config)

binding = {}
for s, token in zip(accepted, tokens):
    binding[s["group_id"]] = {
        "prediction": s["prediction"],
        "actual_bbox": s["actual_bbox"],
        "predicted_field": canonical_details(token.details) or "",
        "component_owner": str(token.component_id or "") or None,
        "pin_owner": str(token.pin_id or "") or None,
    }

# === Spatial matching + scoring ===
pred_items = [{"group_id": s["group_id"], "bbox": tuple(s["actual_bbox"]), "prediction": s["prediction"]} for s in accepted]
match_pairs = []
for pi, p in enumerate(pred_items):
    for gi, tb in enumerate(gt_boxes):
        iou = _bbox_iou(p["bbox"], tuple(tb["bbox"]))
        dist = _center_distance(p["bbox"], tuple(tb["bbox"]))
        if iou >= 0.3 or dist <= 12.0:
            match_pairs.append((pi, gi, max(iou, 0.01)))
matches = _maximum_cardinality_matching(
    len(pred_items), len(gt_boxes), match_pairs,
    left_keys=[p["group_id"] for p in pred_items],
    right_keys=[tb["object_id"] for tb in gt_boxes],
)
pred_to_gt = {left: right for left, right, _s in matches}

field_aliases = {
    "designator": {"designator"}, "value": {"value"},
    "search_name": {"searchname", "search_name", "name"},
    "number": {"pinnumber", "pin_number", "number"},
    "name": {"pinname", "pin_name", "name"},
}

text_exact = 0
field_correct = 0
owner_correct = 0
exact_and_owner = 0
for pi, gi in pred_to_gt.items():
    p = pred_items[pi]
    tb = gt_boxes[gi]
    b = binding.get(p["group_id"], {})
    is_text = p["prediction"] == tb["gt_value"]
    if is_text: text_exact += 1
    pred_field = (b.get("predicted_field") or "").lower()
    expected_field = tb.get("field", "?")
    is_field = pred_field in field_aliases.get(expected_field, {expected_field})
    if is_field: field_correct += 1
    owner_kind = tb.get("owner_kind", "component")
    if owner_kind == "component":
        is_owner = b.get("component_owner") == tb.get("owner_id")
    else:
        is_owner = b.get("pin_owner") == tb.get("owner_id")
    if is_owner: owner_correct += 1
    if is_text and is_owner: exact_and_owner += 1

n_gt = len(gt_boxes)
n_matched = len(pred_to_gt)
print(f"\n=== Weighted Majority Vote Results ===", flush=True)
print(f"  matched: {n_matched}/{n_gt}", flush=True)
print(f"  text_exact: {text_exact}/{n_gt} = {text_exact/n_gt:.4f}", flush=True)
print(f"  field_correct: {field_correct}/{n_gt} = {field_correct/n_gt:.4f}", flush=True)
print(f"  owner_correct: {owner_correct}/{n_gt} = {owner_correct/n_gt:.4f}", flush=True)
print(f"  exact+owner: {exact_and_owner}/{n_gt} = {exact_and_owner/n_gt:.4f}", flush=True)

print(f"\n=== Comparison ===", flush=True)
print(f"  previous (geometric): text=7 field=14 owner=5 e2e=3", flush=True)
print(f"  weighted vote:        text={text_exact} field={field_correct} owner={owner_correct} e2e={exact_and_owner}", flush=True)
print(f"  oracle (paddle):      text=38 field=- owner=- e2e=-", flush=True)

run_dir = WORKSPACE / "work" / f"ocr-weighted-vote-{time.strftime('%Y%m%d-%H%M%S')}"
run_dir.mkdir(parents=True, exist_ok=True)
report = {
    "kind": "weighted_majority_vote",
    "case_id": CASE_ID,
    "n_gt": n_gt, "n_matched": n_matched,
    "text_exact": text_exact, "field_correct": field_correct,
    "owner_correct": owner_correct, "exact_and_owner": exact_and_owner,
    "n_accepted": len(accepted), "n_rejected": len(rejected),
    "previous": {"text_exact": 7, "field_correct": 14, "owner_correct": 5, "exact_and_owner": 3},
    "oracle_text_exact": 38,
}
out = run_dir / "weighted_vote_report.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"\nsaved {out}", flush=True)
