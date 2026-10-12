"""P0+P1: greedy vs beam-1 vs conservative reranking, binding with real geometry,
0005 candidate-GT matching.  Saves per-object predictions for review."""
from __future__ import annotations
import json, sys, time, re
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, r"E:\EDA挑战题目六\src")

WORKSPACE = Path(r"E:\EDA挑战题目六")
DATASET = WORKSPACE / "Dataset300" / "Dataset300"

from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth
from pcbparse.evaluation.text_box_eval import character_error_rate

from pcbparse.training.torch_ocr_model import (
    load_ocr_bundle, greedy_ctc_decode, recognition_output_lengths, ctc_beam_decode,
)
import torch
from pcbparse.io.image_loader import load_image
from pcbparse.perception.torch_ocr_adapter import _recognition_tensor

lock = json.loads((WORKSPACE / "models" / "models.lock.json").read_text(encoding="utf-8"))
ocr_entry = next(e for e in lock["models"] if e["task"] == "ocr")
bundle = load_ocr_bundle(WORKSPACE / "models" / ocr_entry["path"], device="cpu")
height = bundle.preprocess["recognition_height"]
pad_value = bundle.preprocess["recognition_pad_value"]
print(f"bundle: {bundle.version}", flush=True)


def recognize_with_beam(image_path, bbox, padding=2, beam_k=5):
    view = load_image(image_path)
    x1, y1, x2, y2 = bbox
    cb = (max(0, int(x1) - padding), max(0, int(y1) - padding),
          min(view.width, int(x2) + padding), min(view.height, int(y2) + padding))
    if cb[2] <= cb[0] or cb[3] <= cb[1]:
        return None, []
    tensor, width, _ = _recognition_tensor(view, cb, torch, height=height)
    if width <= 0:
        return None, []
    padded = torch.full((1, 1, height, width), float(pad_value), dtype=torch.float32)
    padded[0, :, :, :width] = tensor
    widths = torch.tensor([width], dtype=torch.long)
    with torch.inference_mode():
        logits = bundle.recognizer(padded.to("cpu"), widths.to("cpu"))
    texts, _ = greedy_ctc_decode(logits, recognition_output_lengths(widths), bundle.charset)
    greedy_text = texts[0] if texts else None
    beam = ctc_beam_decode(logits, recognition_output_lengths(widths), bundle.charset, candidate_count=beam_k)
    candidates = list(beam[0]) if beam else []
    return greedy_text, candidates


def gt_text_records(gt):
    records = []
    for comp in gt.components:
        for field, target in (("designator", comp.designator), ("value", comp.value), ("search_name", comp.search_name)):
            if target.mask.eligible and target.value is not None and target.bbox is not None:
                records.append({"object_id": f"{comp.raw_id}.{field}", "field": field, "gt_value": str(target.value), "bbox": target.bbox, "owner_kind": "component", "owner_id": comp.raw_id})
    for pin in gt.pins:
        for field, target in (("number", pin.number), ("name", pin.name)):
            if target.mask.eligible and target.value is not None and target.bbox is not None:
                records.append({"object_id": f"{pin.raw_id}.{field}", "field": field, "gt_value": str(target.value), "bbox": target.bbox, "owner_kind": "pin", "owner_id": pin.raw_id})
    return records


DESIGNATOR_RE = re.compile(r"^[A-Z]{1,4}\d{1,4}$")
PIN_NUMBER_RE = re.compile(r"^\d{1,3}(,\d{1,3})*$")
VALUE_RE = re.compile(r"^\d+(\.\d+)?\s*[kKmMuUnNpP]?[Ωohm%FfHhVvAaWw]?$")


def field_constraint_score(text, field):
    if field == "designator":
        return 2.0 if DESIGNATOR_RE.match(text) else 0.0
    if field == "number":
        return 2.0 if PIN_NUMBER_RE.match(text) else 0.0
    if field == "value":
        return 1.5 if VALUE_RE.match(text) else 0.0
    return 0.5


def conservative_rerank(greedy_text, candidates, field, gt_value):
    pool = []
    if greedy_text:
        pool.append(greedy_text)
    for c in candidates:
        if c and c != greedy_text:
            pool.append(c)
    if not pool:
        return greedy_text
    scored = [(field_constraint_score(t, field), t) for t in pool]
    best = max(scored, key=lambda x: (x[0], -pool.index(x[1])))
    return best[1] if best[0] > 0 else greedy_text


# === P0: greedy vs beam-1 vs reranking on 0087 ===
print("\n=== P0: greedy vs beam-1 vs conservative reranking (0087) ===", flush=True)
case_id = "0087"
case_dir = DATASET / case_id
ann_path = sorted(case_dir.glob("*_CVJsonStd.json"))[0]
img_path = sorted(case_dir.glob("*.png"))[0]
annotation = json.loads(ann_path.read_text(encoding="utf-8"))
gt = build_raw_ground_truth(annotation, case_id=case_id, annotation_path=ann_path)
records = gt_text_records(gt)

results = []
for rec in records:
    greedy_text, candidates = recognize_with_beam(img_path, rec["bbox"], beam_k=5)
    beam1 = candidates[0] if candidates else None
    reranked = conservative_rerank(greedy_text, candidates, rec["field"], rec["gt_value"])
    gt_val = rec["gt_value"]
    greedy_ok = greedy_text == gt_val
    beam1_ok = beam1 == gt_val
    rerank_ok = reranked == gt_val
    in_candidates = gt_val in ([greedy_text] + [c for c in candidates if c]) if candidates else greedy_text == gt_val
    results.append({
        **rec,
        "greedy": greedy_text,
        "beam1": beam1,
        "beam_candidates": candidates[:5],
        "reranked": reranked,
        "greedy_ok": greedy_ok,
        "beam1_ok": beam1_ok,
        "rerank_ok": rerank_ok,
        "in_candidates": in_candidates,
    })

n = len(results)
greedy_hits = sum(1 for r in results if r["greedy_ok"])
beam1_hits = sum(1 for r in results if r["beam1_ok"])
rerank_hits = sum(1 for r in results if r["rerank_ok"])
in_cand = sum(1 for r in results if r["in_candidates"])
beam1_gain = sum(1 for r in results if r["beam1_ok"] and not r["greedy_ok"])
beam1_loss = sum(1 for r in results if not r["beam1_ok"] and r["greedy_ok"])
rerank_gain = sum(1 for r in results if r["rerank_ok"] and not r["greedy_ok"])
rerank_loss = sum(1 for r in results if not r["rerank_ok"] and r["greedy_ok"])

print(f"greedy: {greedy_hits}/{n} = {greedy_hits/n:.4f}", flush=True)
print(f"beam-1: {beam1_hits}/{n} = {beam1_hits/n:.4f} (gain={beam1_gain} loss={beam1_loss} net={beam1_gain-beam1_loss:+d})", flush=True)
print(f"rerank: {rerank_hits}/{n} = {rerank_hits/n:.4f} (gain={rerank_gain} loss={rerank_loss} net={rerank_gain-rerank_loss:+d})", flush=True)
print(f"beam-5 coverage: {in_cand}/{n} = {in_cand/n:.4f}", flush=True)
print(f"errors not in candidates: {n - in_cand}", flush=True)

by_field = defaultdict(lambda: {"n": 0, "greedy": 0, "beam1": 0, "rerank": 0, "gain": 0, "loss": 0})
for r in results:
    b = by_field[r["field"]]
    b["n"] += 1
    if r["greedy_ok"]: b["greedy"] += 1
    if r["beam1_ok"]: b["beam1"] += 1
    if r["rerank_ok"]: b["rerank"] += 1
    if r["rerank_ok"] and not r["greedy_ok"]: b["gain"] += 1
    if not r["rerank_ok"] and r["greedy_ok"]: b["loss"] += 1
print(f"\nBy field (rerank vs greedy):", flush=True)
for field in sorted(by_field):
    b = by_field[field]
    print(f"  {field}: n={b['n']} greedy={b['greedy']} beam1={b['beam1']} rerank={b['rerank']} gain={b['gain']} loss={b['loss']} net={b['gain']-b['loss']:+d}", flush=True)


# === P0: binding with real predicted geometry ===
print("\n=== P0: binding oracle with real geometry ===", flush=True)
from pcbparse.ir.models import BBox, Component, Pin, Point, TextToken
from pcbparse.association.semantic_fields import infer_semantic_fields
from pcbparse.association.text_component import associate_component_texts
from pcbparse.association.text_pin import associate_pin_texts
from pcbparse.association.ocr_common import OCRAssociationConfig, canonical_details

field_aliases = {
    "designator": {"designator"}, "value": {"value"},
    "search_name": {"searchname", "search_name", "name"},
    "number": {"pinnumber", "pin_number", "number"},
    "name": {"pinname", "pin_name", "name"},
}

def build_gt_geometry(gt):
    components = []
    pins = []
    for comp in gt.components:
        if comp.mask.eligible and comp.bbox:
            components.append(Component(internal_id=comp.raw_id, key=None, category=comp.category,
                bbox=BBox(x1=comp.bbox[0], y1=comp.bbox[1], x2=comp.bbox[2], y2=comp.bbox[3])))
    for pin in gt.pins:
        if pin.mask.eligible and pin.tip:
            base = None
            if pin.base:
                base = Point(x=pin.base[0], y=pin.base[1])
            orientation = "UNKNOWN"
            if pin.tip and pin.base:
                dx = pin.base[0] - pin.tip[0]
                dy = pin.base[1] - pin.tip[1]
                if abs(dx) > abs(dy):
                    orientation = "R" if dx > 0 else "L"
                else:
                    orientation = "B" if dy > 0 else "T"
            pins.append(Pin(internal_id=pin.raw_id, component_id=pin.owner_raw_id,
                point=Point(x=pin.tip[0], y=pin.tip[1]), base=base, orientation=orientation))
    return components, pins


def run_binding(gt, records, components, pins, label):
    tokens = []
    for rec in records:
        token = TextToken(internal_id=f"t-{rec['object_id']}", raw_text=rec["gt_value"],
            bbox=BBox(x1=rec["bbox"][0], y1=rec["bbox"][1], x2=rec["bbox"][2], y2=rec["bbox"][3]))
        tokens.append(token)
    infer_semantic_fields(components, pins, tokens, [])
    diagnostics = []
    config = OCRAssociationConfig()
    associate_component_texts(components, tokens, diagnostics, config)
    associate_pin_texts(pins, tokens, diagnostics, config)

    results = []
    for rec, token in zip(records, tokens):
        predicted_field = canonical_details(token.details) or ""
        field_ok = predicted_field.casefold() in field_aliases.get(rec["field"], {rec["field"]})
        if rec["owner_kind"] == "component":
            predicted_owner = str(token.component_id or "") or None
        else:
            predicted_owner = str(token.pin_id or "") or None
        owner_ok = predicted_owner == rec["owner_id"]
        if predicted_owner is None:
            unbound_reason = "no_candidate"
        elif not owner_ok:
            unbound_reason = "wrong_owner"
        else:
            unbound_reason = None
        if not field_ok and predicted_field == "":
            field_reason = "unclassified"
        elif not field_ok:
            field_reason = f"wrong_field:{predicted_field}"
        else:
            field_reason = None
        results.append({
            "object_id": rec["object_id"], "gt_field": rec["field"], "pred_field": predicted_field,
            "field_ok": field_ok, "gt_owner": rec["owner_id"], "pred_owner": predicted_owner,
            "owner_ok": owner_ok, "unbound_reason": unbound_reason, "field_reason": field_reason,
        })
    n = len(results)
    field_ok = sum(1 for r in results if r["field_ok"])
    owner_ok = sum(1 for r in results if r["owner_ok"])
    unbound = sum(1 for r in results if r["pred_owner"] is None)
    wrong = sum(1 for r in results if r["pred_owner"] is not None and not r["owner_ok"])
    unbound_reasons = defaultdict(int)
    for r in results:
        if r["unbound_reason"]:
            unbound_reasons[r["unbound_reason"]] += 1
    field_reasons = defaultdict(int)
    for r in results:
        if r["field_reason"]:
            field_reasons[r["field_reason"]] += 1
    print(f"\n--- {label} ---", flush=True)
    print(f"  field: {field_ok}/{n} = {field_ok/n:.3f}", flush=True)
    print(f"  owner: {owner_ok}/{n} = {owner_ok/n:.3f}", flush=True)
    print(f"  unbound: {unbound} = {unbound/n:.3f} reasons={dict(unbound_reasons)}", flush=True)
    print(f"  wrong_owner: {wrong} = {wrong/n:.3f}", flush=True)
    print(f"  field_reasons: {dict(field_reasons)}", flush=True)
    return {"label": label, "n": n, "field_ok": field_ok, "owner_ok": owner_ok, "unbound": unbound,
            "wrong_owner": wrong, "unbound_reasons": dict(unbound_reasons), "field_reasons": dict(field_reasons),
            "objects": results}

gt_components, gt_pins = build_gt_geometry(gt)
bind_gt = run_binding(gt, records, gt_components, gt_pins, "GT geometry (with base+orientation)")


# === P1: 0005 candidate-GT matching ===
print("\n=== P1: 0005 candidate-GT matching ===", flush=True)
from pcbparse.evaluation.text_box_eval import _bbox_iou, _center_distance, _gt_text_boxes

case_dir_05 = DATASET / "0005"
ann_path_05 = sorted(case_dir_05.glob("*_CVJsonStd.json"))[0]
annotation_05 = json.loads(ann_path_05.read_text(encoding="utf-8"))
gt_05 = build_raw_ground_truth(annotation_05, case_id="0005", annotation_path=ann_path_05)
gt_boxes_05 = _gt_text_boxes(gt_05)

exp_dir = WORKSPACE / "work" / "ocr-experiments-20260923-142356"
exp_data = json.loads((exp_dir / "experiments.json").read_text(encoding="utf-8"))
trace_05 = exp_data.get("task1_trace", [])
print(f"loaded {len(trace_05)} trace entries from {exp_dir.name}", flush=True)

def match_candidates(entries, gt_boxes, stage_filter):
    if stage_filter == "detected":
        filtered = [e for e in entries if e.get("original_bbox")]
    elif stage_filter == "geometry_accepted":
        filtered = [e for e in entries if e.get("geometry_accepted") and e.get("original_bbox")]
    elif stage_filter == "recognition_accepted":
        filtered = [e for e in entries if e.get("recognition_accepted") and e.get("original_bbox")]
    else:
        filtered = [e for e in entries if e.get("original_bbox")]

    candidates = []
    valid = []
    for entry in filtered:
        bbox = tuple(entry["original_bbox"])
        valid.append((entry, bbox))
    for pred_idx, (_e, bbox) in enumerate(valid):
        for gt_idx, target in enumerate(gt_boxes):
            iou = _bbox_iou(bbox, target["bbox"])
            dist = _center_distance(bbox, target["bbox"])
            if iou >= 0.3 or dist <= 12.0:
                candidates.append((pred_idx, gt_idx, max(iou, 0.01)))
    from pcbparse.evaluation.masked_metrics import _maximum_cardinality_matching
    matches = _maximum_cardinality_matching(
        len(valid), len(gt_boxes), candidates,
        left_keys=[f"{e.get('view_id')}:{e.get('detection_index')}" for e, _ in valid],
        right_keys=[item["object_id"] for item in gt_boxes],
    )
    matched_gt = {right for _l, right, _s in matches}
    return len(valid), len(matched_gt)

stages = ["detected", "geometry_accepted", "recognition_accepted"]
print(f"GT text boxes: {len(gt_boxes_05)}", flush=True)
for stage in stages:
    pred_count, tp = match_candidates(trace_05, gt_boxes_05, stage)
    recall = tp / len(gt_boxes_05) if gt_boxes_05 else 0
    precision = tp / pred_count if pred_count else 0
    print(f"  {stage}: candidates={pred_count} TP={tp} recall={recall:.4f} precision={precision:.4f}", flush=True)

by_level_tp = {}
for level in ("global", "component", "side_strip"):
    level_entries = [e for e in trace_05 if e.get("level") == level and e.get("original_bbox")]
    _pc, tp = match_candidates(level_entries, gt_boxes_05, "detected")
    by_level_tp[level] = {"entries": len(level_entries), "tp": tp}
    print(f"  {level}: entries={len(level_entries)} TP={tp}", flush=True)


# === Save ===
run_id = time.strftime("%Y%m%d-%H%M%S")
run_dir = WORKSPACE / "work" / f"ocr-p0p1-{run_id}"
run_dir.mkdir(parents=True, exist_ok=True)
report = {
    "kind": "p0p1_experiments",
    "run_id": run_id,
    "greedy_beam_rerank": {
        "n": n,
        "greedy_hits": greedy_hits, "greedy_rate": greedy_hits / n,
        "beam1_hits": beam1_hits, "beam1_rate": beam1_hits / n,
        "beam1_gain": beam1_gain, "beam1_loss": beam1_loss, "beam1_net": beam1_gain - beam1_loss,
        "rerank_hits": rerank_hits, "rerank_rate": rerank_hits / n,
        "rerank_gain": rerank_gain, "rerank_loss": rerank_loss, "rerank_net": rerank_gain - rerank_loss,
        "beam5_coverage": in_cand, "beam5_coverage_rate": in_cand / n,
        "errors_not_in_candidates": n - in_cand,
        "by_field": {k: dict(v) for k, v in by_field.items()},
    },
    "binding_oracle": {
        "gt_geometry": {k: v for k, v in bind_gt.items() if k != "objects"},
        "gt_geometry_objects": bind_gt["objects"],
    },
    "candidate_gt_matching_0005": {
        "gt_box_count": len(gt_boxes_05),
        "by_stage": {s: {"candidates": match_candidates(trace_05, gt_boxes_05, s)[0], "tp": match_candidates(trace_05, gt_boxes_05, s)[1]} for s in stages},
        "by_level_tp": by_level_tp,
    },
    "per_object_predictions": results,
}
out = run_dir / "p0p1_report.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"\nsaved {out}", flush=True)
print(f"run_dir: {run_dir}", flush=True)
