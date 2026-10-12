"""Task 3 fix + Task 1 + Task 4: real association oracle, 0005 full trace, padding ablation."""
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


def recognize_crop(image_path, bbox, padding=2):
    view = load_image(image_path)
    x1, y1, x2, y2 = bbox
    cb = (max(0, int(x1) - padding), max(0, int(y1) - padding),
          min(view.width, int(x2) + padding), min(view.height, int(y2) + padding))
    if cb[2] <= cb[0] or cb[3] <= cb[1]:
        return None
    tensor, width, _ = _recognition_tensor(view, cb, torch, height=height)
    if width <= 0:
        return None
    padded = torch.full((1, 1, height, width), float(pad_value), dtype=torch.float32)
    padded[0, :, :, :width] = tensor
    widths = torch.tensor([width], dtype=torch.long)
    with torch.inference_mode():
        logits = bundle.recognizer(padded.to("cpu"), widths.to("cpu"))
    texts, _ = greedy_ctc_decode(logits, recognition_output_lengths(widths), bundle.charset)
    return texts[0] if texts else None


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


# === Task 3 fix: real association oracle with associate_* ===
print("\n=== Task 3: real association oracle (fixed) ===", flush=True)
from pcbparse.ir.models import BBox, Component, Pin, Point, TextToken, Diagnostic
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

def run_real_binding_oracle(gt, records, geometry_source):
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

    tokens = []
    for rec in records:
        token = TextToken(internal_id=f"oracle-{rec['object_id']}", raw_text=rec["gt_value"],
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
    wrong = sum(1 for r in results if r["pred_owner"] is not None and not r["owner_ok"])
    return {
        "geometry_source": geometry_source,
        "denominator": n,
        "field_classified": field_ok,
        "field_rate": field_ok / n if n else None,
        "owner_bound": owner_ok,
        "owner_rate": owner_ok / n if n else None,
        "unbound": unbound,
        "unbound_rate": unbound / n if n else None,
        "wrong_owner": wrong,
        "wrong_owner_rate": wrong / n if n else None,
        "objects": results,
    }

case_id = "0087"
case_dir = DATASET / case_id
ann_path = sorted(case_dir.glob("*_CVJsonStd.json"))[0]
img_path = sorted(case_dir.glob("*.png"))[0]
annotation = json.loads(ann_path.read_text(encoding="utf-8"))
gt = build_raw_ground_truth(annotation, case_id=case_id, annotation_path=ann_path)
records = gt_text_records(gt)

bind_gt = run_real_binding_oracle(gt, records, "gt")
bind_pred = run_real_binding_oracle(gt, records, "predicted_geometry")

for label, report in [("GT geometry", bind_gt), ("Predicted geometry", bind_pred)]:
    print(f"\n--- {label} ---", flush=True)
    print(f"  field: {report['field_classified']}/{report['denominator']} = {report['field_rate']:.3f}", flush=True)
    print(f"  owner: {report['owner_bound']}/{report['denominator']} = {report['owner_rate']:.3f}", flush=True)
    print(f"  unbound: {report['unbound']} = {report['unbound_rate']:.3f}", flush=True)
    print(f"  wrong_owner: {report['wrong_owner']} = {report['wrong_owner_rate']:.3f}", flush=True)


# === Task 1: 0005 full-chain diagnostic ===
print("\n=== Task 1: 0005 full-chain diagnostic ===", flush=True)
case_id_05 = "0005"
case_dir_05 = DATASET / case_id_05
ann_path_05 = sorted(case_dir_05.glob("*_CVJsonStd.json"))[0]
img_path_05 = sorted(case_dir_05.glob("*.png"))[0]
annotation_05 = json.loads(ann_path_05.read_text(encoding="utf-8"))
gt_05 = build_raw_ground_truth(annotation_05, case_id=case_id_05, annotation_path=ann_path_05)

from pcbparse.perception.production_provider import load_production_provider
from pcbparse.pipeline.orchestrator import infer_one

config = json.loads((WORKSPACE / "configs" / "pipeline.yaml").read_text(encoding="utf-8"))
config.setdefault("run", {})["submission_mode"] = "degraded"
provider = load_production_provider(WORKSPACE / "models" / "models.lock.json")

output_dir = WORKSPACE / "work" / "ocr-triage-full"
output_dir.mkdir(parents=True, exist_ok=True)
try:
    result = infer_one(
        img_path_05,
        output_dir / "0005.json",
        provider=provider,
        config=config,
        official_profile=WORKSPACE / "configs" / "official_schema.yaml",
    )
    infer_status = "ok"
except Exception as exc:
    infer_status = f"{type(exc).__name__}: {exc}"

ocr_provider = getattr(provider, "ocr_provider", None)
trace_log = getattr(ocr_provider, "_detection_trace_log", None) or []
print(f"infer status: {infer_status}", flush=True)
print(f"total trace entries: {len(trace_log)}", flush=True)

by_level = defaultdict(lambda: {"detected": 0, "geometry_accepted": 0, "recognition_attempted": 0, "recognition_accepted": 0, "rejected": 0, "reasons": defaultdict(int)})
for entry in trace_log:
    level = str(entry.get("level") or "unknown")
    b = by_level[level]
    b["detected"] += 1
    if entry.get("geometry_accepted"): b["geometry_accepted"] += 1
    if entry.get("recognition_attempted"): b["recognition_attempted"] += 1
    if entry.get("recognition_accepted"): b["recognition_accepted"] += 1
    if entry.get("outcome") == "rejected":
        b["rejected"] += 1
        reason = entry.get("reason") or "unknown"
        b["reasons"][reason] += 1

print(f"\nPer-level stage counts (0005):", flush=True)
total_detected = 0
for level in sorted(by_level):
    b = by_level[level]
    total_detected += b["detected"]
    reasons_str = ", ".join(f"{k}={v}" for k, v in sorted(b["reasons"].items()))
    print(f"  {level}: detected={b['detected']} geom_ok={b['geometry_accepted']} recog_attempted={b['recognition_attempted']} recog_ok={b['recognition_accepted']} rejected={b['rejected']} [{reasons_str}]", flush=True)
print(f"  TOTAL detected={total_detected}", flush=True)
batch_ids = sorted(set(e.get("batch_id", -1) for e in trace_log))
print(f"  batch_ids: {batch_ids}", flush=True)

trace_report = {
    "kind": "full_chain_trace",
    "case_id": case_id_05,
    "infer_status": infer_status,
    "total_entries": len(trace_log),
    "batch_count": len(batch_ids),
    "by_level": {k: {**{kk: vv for kk, vv in v.items() if kk != "reasons"}, "reasons": dict(v["reasons"])} for k, v in by_level.items()},
    "trace": trace_log,
}


# === Task 4: padding ablation on 0087 ===
print("\n=== Task 4: padding ablation on 0087 ===", flush=True)
records_0087 = gt_text_records(gt)
for padding in (0, 2, 4, 8):
    exact = 0
    total_edits = 0.0
    total_chars = 0
    for rec in records_0087:
        pred = recognize_crop(img_path, rec["bbox"], padding=padding)
        cer = character_error_rate(rec["gt_value"], pred or "")
        if pred == rec["gt_value"]:
            exact += 1
        total_edits += cer * len(rec["gt_value"])
        total_chars += len(rec["gt_value"])
    n = len(records_0087)
    print(f"  padding={padding}: exact={exact}/{n} = {exact/n:.4f} corpus_cer={total_edits/total_chars:.4f}", flush=True)

run_id = time.strftime("%Y%m%d-%H%M%S")
run_dir = WORKSPACE / "work" / f"ocr-experiments-{run_id}"
run_dir.mkdir(parents=True, exist_ok=True)

report = {
    "kind": "ocr_optimization_experiments",
    "run_id": run_id,
    "task1_full_trace": {k: v for k, v in trace_report.items() if k != "trace"},
    "task1_trace": trace_report["trace"],
    "task3_binding": {
        "gt_geometry": {k: v for k, v in bind_gt.items() if k != "objects"},
        "predicted_geometry": {k: v for k, v in bind_pred.items() if k != "objects"},
    },
}
(run_dir / "experiments.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"\nsaved {run_dir / 'experiments.json'}", flush=True)
print(f"run_dir: {run_dir}", flush=True)
