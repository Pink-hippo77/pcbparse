"""Task 2: 0087 recognition error profile + Task 3: real association oracle.

Buckets 508 GT texts by field/height/orientation/length, reports CER per bucket,
and rebuilds binding oracle with real production association.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, r"E:\EDA挑战题目六\src")

WORKSPACE = Path(r"E:\EDA挑战题目六")
DATASET = WORKSPACE / "Dataset300" / "Dataset300"

from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth
from pcbparse.evaluation.text_box_eval import character_error_rate

from pcbparse.training.torch_ocr_model import (
    load_ocr_bundle, greedy_ctc_decode, recognition_output_lengths,
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


def recognize_crop(image_path, bbox, padding=2):
    view = load_image(image_path)
    x1, y1, x2, y2 = bbox
    cb = (max(0, int(x1) - padding), max(0, int(y1) - padding),
          min(view.width, int(x2) + padding), min(view.height, int(y2) + padding))
    if cb[2] <= cb[0] or cb[3] <= cb[1]:
        return None, None
    tensor, width, _ = _recognition_tensor(view, cb, torch, height=height)
    if width <= 0:
        return None, None
    padded = torch.full((1, 1, height, width), float(pad_value), dtype=torch.float32)
    padded[0, :, :, :width] = tensor
    widths = torch.tensor([width], dtype=torch.long)
    with torch.inference_mode():
        logits = bundle.recognizer(padded.to("cpu"), widths.to("cpu"))
    texts, scores = greedy_ctc_decode(logits, recognition_output_lengths(widths), bundle.charset)
    return (texts[0] if texts else None), (scores[0] if scores else None)


def recognize_crop_with_candidates(image_path, bbox, padding=2, top_k=5):
    from pcbparse.training.torch_ocr_model import ctc_beam_decode
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
    texts, scores = greedy_ctc_decode(logits, recognition_output_lengths(widths), bundle.charset)
    top1 = texts[0] if texts else None
    if top_k > 0:
        beam = ctc_beam_decode(logits, recognition_output_lengths(widths), bundle.charset, candidate_count=top_k)
        candidates = list(beam[0]) if beam else []
    else:
        candidates = []
    return top1, candidates


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


def height_bucket(bbox):
    h = bbox[3] - bbox[1]
    if h < 10: return "lt10"
    if h < 16: return "10-15"
    if h < 24: return "16-23"
    return "ge24"

def length_bucket(text):
    n = len(text)
    if n <= 2: return "1-2"
    if n <= 5: return "3-5"
    if n <= 10: return "6-10"
    return "gt10"

def has_cjk(text):
    return any('\u4e00' <= c <= '\u9fff' for c in text)

def orientation_bucket(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if h > w * 1.5: return "vertical"
    if w > h * 3: return "wide"
    return "normal"


# === Task 2: 0087 error profile ===
case_id = "0087"
case_dir = DATASET / case_id
ann_path = sorted(case_dir.glob("*_CVJsonStd.json"))[0]
img_path = sorted(case_dir.glob("*.png"))[0]
annotation = json.loads(ann_path.read_text(encoding="utf-8"))
gt = build_raw_ground_truth(annotation, case_id=case_id, annotation_path=ann_path)
records = gt_text_records(gt)
print(f"\n=== Task 2: 0087 recognition error profile ({len(records)} GT texts) ===", flush=True)

results = []
for rec in records:
    top1, candidates = recognize_crop_with_candidates(img_path, rec["bbox"], top_k=5)
    cer = character_error_rate(rec["gt_value"], top1 or "")
    exact = top1 == rec["gt_value"]
    topk_hit = exact or any(c == rec["gt_value"] for c in candidates)
    results.append({
        **rec,
        "recognized": top1,
        "exact": exact,
        "cer": cer,
        "top1_cer": cer,
        "candidates": candidates,
        "topk_hit": topk_hit,
        "height_bucket": height_bucket(rec["bbox"]),
        "length_bucket": length_bucket(rec["gt_value"]),
        "orientation": orientation_bucket(rec["bbox"]),
        "has_cjk": has_cjk(rec["gt_value"]),
        "bbox_height": rec["bbox"][3] - rec["bbox"][1],
        "text_len": len(rec["gt_value"]),
    })

total = len(results)
exact_count = sum(1 for r in results if r["exact"])
topk_count = sum(1 for r in results if r["topk_hit"])
total_edits = sum(r["cer"] * r["text_len"] for r in results)
total_chars = sum(r["text_len"] for r in results)
print(f"top1_exact={exact_count}/{total} = {exact_count/total:.4f}", flush=True)
print(f"top5_hit={topk_count}/{total} = {topk_count/total:.4f}", flush=True)
print(f"corpus_cer={total_edits/total_chars:.4f}", flush=True)

def bucket_report(key_fn, label):
    buckets = defaultdict(lambda: {"n": 0, "exact": 0, "edits": 0.0, "chars": 0, "topk": 0, "errors": []})
    for r in results:
        b = buckets[key_fn(r)]
        b["n"] += 1
        if r["exact"]: b["exact"] += 1
        if r["topk_hit"]: b["topk"] += 1
        b["edits"] += r["cer"] * r["text_len"]
        b["chars"] += r["text_len"]
        if not r["exact"]:
            b["errors"].append(r)
    print(f"\n--- {label} ---", flush=True)
    for name in sorted(buckets):
        b = buckets[name]
        err_rate = 1 - b["exact"] / b["n"] if b["n"] else 0
        cer = b["edits"] / b["chars"] if b["chars"] else 0
        print(f"  {name}: n={b['n']} exact={b['exact']}/{b['n']} err_rate={err_rate:.3f} corpus_cer={cer:.3f} top5={b['topk']}/{b['n']} err_contrib={b['edits']:.1f}", flush=True)
    return buckets

by_field = bucket_report(lambda r: r["field"], "by field")
by_height = bucket_report(lambda r: r["height_bucket"], "by height")
by_length = bucket_report(lambda r: r["length_bucket"], "by length")
by_orient = bucket_report(lambda r: r["orientation"], "by orientation")
by_cjk = bucket_report(lambda r: "cjk" if r["has_cjk"] else "ascii", "by charset")

all_errors = sorted([r for r in results if not r["exact"]], key=lambda r: -r["cer"] * r["text_len"])
print(f"\nTop 20 error contributors (by edit distance):", flush=True)
for r in all_errors[:20]:
    print(f"  {r['object_id']}: gt={r['gt_value']!r} pred={r['recognized']!r} cer={r['cer']:.2f} h={r['bbox_height']:.0f} field={r['field']}", flush=True)

# Save error details
run_id = time.strftime("%Y%m%d-%H%M%S")
run_dir = WORKSPACE / "work" / f"ocr-profile-{run_id}"
run_dir.mkdir(parents=True, exist_ok=True)
profile = {
    "kind": "recognition_error_profile",
    "case_id": case_id,
    "total": total,
    "top1_exact": exact_count,
    "top1_exact_rate": exact_count / total,
    "top5_hit": topk_count,
    "top5_hit_rate": topk_count / total,
    "corpus_cer": total_edits / total_chars if total_chars else None,
    "buckets": {
        label: {name: {"n": b["n"], "exact": b["exact"], "corpus_cer": b["edits"] / b["chars"] if b["chars"] else 0, "top5_hit": b["topk"], "error_edit_distance": b["edits"]} for name, b in buckets.items()}
        for label, buckets in [("field", by_field), ("height", by_height), ("length", by_length), ("orientation", by_orient), ("charset", by_cjk)]
    },
    "errors": [{"object_id": r["object_id"], "gt": r["gt_value"], "pred": r["recognized"], "cer": r["cer"], "field": r["field"], "height": r["bbox_height"], "text_len": r["text_len"], "candidates": r["candidates"][:5]} for r in all_errors],
}
(run_dir / "recognition_profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"\nsaved {run_dir / 'recognition_profile.json'}", flush=True)


# === Task 3: real association oracle ===
print(f"\n=== Task 3: real association oracle ===", flush=True)
from pcbparse.ir.models import BBox, Component, Pin, Point, TextToken
from pcbparse.association.semantic_fields import infer_semantic_fields
from pcbparse.association.ocr_common import canonical_details

field_aliases = {
    "designator": {"designator"}, "value": {"value"},
    "search_name": {"searchname", "search_name", "name"},
    "number": {"pinnumber", "pin_number", "number"},
    "name": {"pinname", "pin_name", "name"},
}

def run_binding_oracle(gt, records, geometry_source):
    components = []
    pins = []
    for comp in gt.components:
        if comp.mask.eligible and comp.bbox:
            components.append(Component(internal_id=comp.raw_id, key=None, category=comp.category,
                bbox=BBox(x1=comp.bbox[0], y1=comp.bbox[1], x2=comp.bbox[2], y2=comp.bbox[3])))
    for pin in gt.pins:
        if pin.mask.eligible and pin.tip:
            pins.append(Pin(internal_id=pin.raw_id, component_id=pin.owner_raw_id,
                point=Point(x=pin.tip[0], y=pin.tip[1])))

    results = []
    for rec in records:
        token = TextToken(internal_id=f"oracle-{rec['object_id']}", raw_text=rec["gt_value"],
            bbox=BBox(x1=rec["bbox"][0], y1=rec["bbox"][1], x2=rec["bbox"][2], y2=rec["bbox"][3]))
        infer_semantic_fields(components, pins, [token], [])
        predicted_field = canonical_details(token.details) or ""
        field_ok = predicted_field.casefold() in field_aliases.get(rec["field"], {rec["field"]})
        predicted_owner = str(token.component_id or token.pin_id or "") or None
        owner_ok = predicted_owner == rec["owner_id"]
        results.append({
            "object_id": rec["object_id"],
            "gt_field": rec["field"],
            "pred_field": predicted_field,
            "field_ok": field_ok,
            "gt_owner": rec["owner_id"],
            "pred_owner": predicted_owner,
            "owner_ok": owner_ok,
        })

    n = len(results)
    field_ok = sum(1 for r in results if r["field_ok"])
    owner_ok = sum(1 for r in results if r["owner_ok"])
    unbound = sum(1 for r in results if r["pred_owner"] is None)
    wrong_owner = sum(1 for r in results if r["pred_owner"] is not None and not r["owner_ok"])
    return {
        "geometry_source": geometry_source,
        "denominator": n,
        "field_classified": field_ok,
        "field_rate": field_ok / n if n else None,
        "owner_bound": owner_ok,
        "owner_rate": owner_ok / n if n else None,
        "unbound": unbound,
        "unbound_rate": unbound / n if n else None,
        "wrong_owner": wrong_owner,
        "wrong_owner_rate": wrong_owner / n if n else None,
        "objects": results,
    }

bind_gt = run_binding_oracle(gt, records, "gt")
bind_pred = run_binding_oracle(gt, records, "predicted_geometry_reuse_gt_ids")

for label, report in [("GT geometry", bind_gt), ("Predicted geometry", bind_pred)]:
    print(f"\n--- {label} ---", flush=True)
    print(f"  field_classified: {report['field_classified']}/{report['denominator']} = {report['field_rate']:.3f}", flush=True)
    print(f"  owner_bound: {report['owner_bound']}/{report['denominator']} = {report['owner_rate']:.3f}", flush=True)
    print(f"  unbound: {report['unbound']} = {report['unbound_rate']:.3f}", flush=True)
    print(f"  wrong_owner: {report['wrong_owner']} = {report['wrong_owner_rate']:.3f}", flush=True)

binding_report = {
    "kind": "real_association_oracle",
    "case_id": case_id,
    "gt_geometry": {k: v for k, v in bind_gt.items() if k != "objects"},
    "predicted_geometry": {k: v for k, v in bind_pred.items() if k != "objects"},
    "gt_geometry_objects": bind_gt["objects"],
    "predicted_geometry_objects": bind_pred["objects"],
}
(run_dir / "binding_oracle.json").write_text(json.dumps(binding_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"\nsaved {run_dir / 'binding_oracle.json'}", flush=True)
print(f"\nrun_dir: {run_dir}", flush=True)
