"""P0+P1 optimization: improved ranking + field classification + binding.

Compares before/after on 0005 GT crops with spatial matching.
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
from pcbparse.evaluation.text_box_eval import (
    character_error_rate, _bbox_iou, _center_distance, _gt_text_boxes,
)
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


# Build groups (same as before)
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

# Recognize all candidates
all_crops = [crop_from_bbox(tuple(c["original_bbox"])) for c in candidates]
valid_crops = [(i, c) for i, c in enumerate(all_crops) if c is not None]
print(f"recognizing {len(valid_crops)} crops...", flush=True)
t0 = time.time()
rec_results = paddle_backend.recognize_crops([c for _, c in valid_crops])
rec_time = time.time() - t0
preds = {}
for (ci, _), (pred, score) in zip(valid_crops, rec_results):
    preds[ci] = (pred, score)
print(f"recognize: {rec_time:.1f}s", flush=True)


# === Improved GT-free ranking ===
def improved_quality_score(cand_idx, group, all_cands, preds):
    c = all_cands[cand_idx]
    ob = c["original_bbox"]
    w, h = ob[2] - ob[0], ob[3] - ob[1]
    score = 0.0

    # 1. Geometric quality
    if w > 0 and h > 0:
        ratio = max(w / h, h / w)
        score += 0.20 * max(0, 1.0 - (ratio - 1.0) / 4.0)
    area = w * h
    if 40 <= area <= 1500:
        score += 0.15
    elif area < 15:
        score -= 0.15
    if ob[0] > 2 and ob[1] > 2 and ob[2] < img.width - 2 and ob[3] < img.height - 2:
        score += 0.10

    # 2. Recognition confidence (calibrated: paddle scores mean ~0.6)
    pred, rec_score = preds.get(cand_idx, (None, 0))
    if pred is not None and pred != "":
        score += 0.25 * float(rec_score)
    else:
        score -= 0.20

    # 3. Cross-view text consistency
    if len(group.get("view_ids", set())) > 1:
        score += 0.15
    if len(group.get("levels", set())) > 1:
        score += 0.10

    # 4. Text shape quality (prefer alphanumeric over pure symbols)
    if pred is not None:
        if any(c.isalnum() for c in pred):
            score += 0.10
        if len(pred) >= 2:
            score += 0.05

    return score


def select_from_group_improved(group, all_cands, preds):
    scored = [(improved_quality_score(ci, group, all_cands, preds), ci) for ci in group["member_indices"]]
    scored.sort(key=lambda x: -x[0])
    top_ci = scored[0][1]
    pred, rec_score = preds.get(top_ci, (None, 0))
    actual_bbox = all_cands[top_ci]["original_bbox"]
    if pred is None or pred == "":
        return top_ci, None, actual_bbox, "rejected", "empty_recognition"
    if rec_score < 0.3:
        return top_ci, pred, actual_bbox, "rejected", "low_recognition_score"
    return top_ci, pred, actual_bbox, "accepted", None


# Run improved selection
selections = []
for g in groups:
    g["levels"] = {candidates[ci].get("level", "?") for ci in g["member_indices"]}
    g["view_ids"] = {candidates[ci].get("view_id", "?") for ci in g["member_indices"]}
    sel_ci, pred, actual_bbox, status, reason = select_from_group_improved(g, candidates, preds)
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

# Build geometry from GT (for binding evaluation)
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
    # Dual-ID owner
    owner_kind = tb.get("owner_kind", "component")
    if owner_kind == "component":
        is_owner = b.get("component_owner") == tb.get("owner_id")
    else:
        is_owner = b.get("pin_owner") == tb.get("owner_id")
    if is_owner: owner_correct += 1
    if is_text and is_owner: exact_and_owner += 1

n_gt = len(gt_boxes)
n_matched = len(pred_to_gt)
print(f"\n=== Results ===", flush=True)
print(f"  matched: {n_matched}/{n_gt}", flush=True)
print(f"  text_exact: {text_exact}/{n_gt} = {text_exact/n_gt:.4f}", flush=True)
print(f"  field_correct: {field_correct}/{n_gt} = {field_correct/n_gt:.4f}", flush=True)
print(f"  owner_correct: {owner_correct}/{n_gt} = {owner_correct/n_gt:.4f}", flush=True)
print(f"  exact+owner: {exact_and_owner}/{n_gt} = {exact_and_owner/n_gt:.4f}", flush=True)

# Compare with before
print(f"\n=== Comparison ===", flush=True)
print(f"  before: text=9 field=1 owner=0 e2e=0", flush=True)
print(f"  after:  text={text_exact} field={field_correct} owner={owner_correct} e2e={exact_and_owner}", flush=True)

run_dir = WORKSPACE / "work" / f"ocr-p0p1-opt-{time.strftime('%Y%m%d-%H%M%S')}"
run_dir.mkdir(parents=True, exist_ok=True)
report = {
    "kind": "p0p1_optimization",
    "case_id": CASE_ID,
    "n_gt": n_gt, "n_matched": n_matched,
    "text_exact": text_exact, "field_correct": field_correct,
    "owner_correct": owner_correct, "exact_and_owner": exact_and_owner,
    "before": {"text_exact": 9, "field_correct": 1, "owner_correct": 0, "exact_and_owner": 0},
    "n_accepted": len(accepted), "n_rejected": len(rejected),
}
out = run_dir / "optimization_report.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"\nsaved {out}", flush=True)
