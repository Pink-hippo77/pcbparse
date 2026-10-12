"""Full 853-crop model comparison with fixed PaddleOCR parsing."""
import json, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
sys.path.insert(0, r"E:\EDA挑战题目六\src")

WORKSPACE = Path(r"E:\EDA挑战题目六")
DATASET = WORKSPACE / "Dataset300" / "Dataset300"
CASES = ("0001", "0005", "0016", "0087", "0088")

from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth
from pcbparse.evaluation.text_box_eval import character_error_rate

def gt_text_records(gt):
    records = []
    for comp in gt.components:
        for field, target in (("designator", comp.designator), ("value", comp.value), ("search_name", comp.search_name)):
            if target.mask.eligible and target.value is not None and target.bbox is not None:
                records.append({"object_id": f"{comp.raw_id}.{field}", "field": field, "gt_value": str(target.value), "bbox": target.bbox})
    for pin in gt.pins:
        for field, target in (("number", pin.number), ("name", pin.name)):
            if target.mask.eligible and target.value is not None and target.bbox is not None:
                records.append({"object_id": f"{pin.raw_id}.{field}", "field": field, "gt_value": str(target.value), "bbox": target.bbox})
    return records

all_records = []
for case_id in CASES:
    case_dir = DATASET / case_id
    ann_path = sorted(case_dir.glob("*_CVJsonStd.json"))[0]
    img_path = sorted(case_dir.glob("*.png"))[0]
    annotation = json.loads(ann_path.read_text(encoding="utf-8"))
    gt = build_raw_ground_truth(annotation, case_id=case_id, annotation_path=ann_path)
    recs = gt_text_records(gt)
    for r in recs:
        r["case_id"] = case_id
        r["image_path"] = str(img_path)
    all_records.extend(recs)

print(f"loaded {len(all_records)} crops", flush=True)

from paddleocr import TextRecognition

def make_recognizer(model_name):
    model = TextRecognition(model_name=model_name)
    def recognize(image_path, bbox, padding=2):
        img = Image.open(image_path).convert("RGB")
        x1, y1, x2, y2 = bbox
        cb = (max(0, int(x1) - padding), max(0, int(y1) - padding),
              min(img.width, int(x2) + padding), min(img.height, int(y2) + padding))
        if cb[2] <= cb[0] or cb[3] <= cb[1]:
            return None
        crop = np.array(img.crop(cb))
        try:
            result = model.predict(input=crop, batch_size=1)
            for res in result:
                raw = res.get("rec_text")
                if raw is None:
                    continue
                if not isinstance(raw, str):
                    raise TypeError(f"rec_text must be str, got {type(raw).__name__}")
                return raw
        except Exception as exc:
            return f"__ERROR__:{type(exc).__name__}"
        return None
    return recognize


def evaluate(recognize_fn, records, label):
    results = []
    t0 = time.time()
    for rec in records:
        t1 = time.time()
        pred = recognize_fn(rec["image_path"], rec["bbox"])
        elapsed = time.time() - t1
        gt_val = rec["gt_value"]
        cer = character_error_rate(gt_val, pred or "")
        is_exact = pred == gt_val
        is_error = isinstance(pred, str) and pred.startswith("__ERROR__")
        results.append({
            "object_id": rec["object_id"], "case_id": rec["case_id"],
            "field": rec["field"], "gt": gt_val, "pred": pred,
            "exact": is_exact, "cer": cer, "elapsed": elapsed, "is_error": is_error,
        })
    total = time.time() - t0
    n = len(results)
    exact = sum(1 for r in results if r["exact"])
    errors = sum(1 for r in results if r["is_error"])
    total_edits = sum(r["cer"] * len(r["gt"]) for r in results)
    total_chars = sum(len(r["gt"]) for r in results)
    corpus_cer = total_edits / total_chars if total_chars else 0

    by_field = defaultdict(lambda: {"n": 0, "exact": 0, "edits": 0.0, "chars": 0})
    for r in results:
        b = by_field[r["field"]]
        b["n"] += 1
        if r["exact"]: b["exact"] += 1
        b["edits"] += r["cer"] * len(r["gt"])
        b["chars"] += len(r["gt"])

    print(f"\n=== {label} ===", flush=True)
    print(f"  exact: {exact}/{n} = {exact/n:.4f}  errors: {errors}", flush=True)
    print(f"  corpus_cer: {corpus_cer:.4f}  time: {total:.1f}s  avg: {total/n*1000:.1f}ms", flush=True)
    for field in sorted(by_field):
        b = by_field[field]
        cer = b["edits"] / b["chars"] if b["chars"] else 0
        print(f"  {field}: n={b['n']} exact={b['exact']}/{b['n']} corpus_cer={cer:.4f}", flush=True)

    return {
        "label": label, "n": n, "exact": exact, "exact_rate": exact / n,
        "corpus_cer": corpus_cer, "errors": errors,
        "total_time_s": total, "avg_ms_per_crop": total / n * 1000,
        "by_field": {k: {"n": v["n"], "exact": v["exact"], "corpus_cer": v["edits"] / v["chars"] if v["chars"] else 0} for k, v in by_field.items()},
        "per_object": results,
    }


all_reports = {}

for model_name in ("en_PP-OCRv5_mobile_rec", "PP-OCRv5_server_rec"):
    try:
        recognize_fn = make_recognizer(model_name)
        all_reports[model_name] = evaluate(recognize_fn, all_records, model_name)
    except Exception as exc:
        print(f"\n=== {model_name} FAILED: {type(exc).__name__}: {exc} ===", flush=True)

# Also run torch baseline for comparison
from pcbparse.training.torch_ocr_model import (
    load_ocr_bundle, greedy_ctc_decode, recognition_output_lengths,
)
import torch
from pcbparse.io.image_loader import load_image
from pcbparse.perception.torch_ocr_adapter import _recognition_tensor

lock = json.loads((WORKSPACE / "models" / "models.lock.json").read_text(encoding="utf-8"))
ocr_entry = next(e for e in lock["models"] if e["task"] == "ocr")
bundle = load_ocr_bundle(WORKSPACE / "models" / ocr_entry["path"], device="cpu")
t_height = bundle.preprocess["recognition_height"]
t_pad = bundle.preprocess["recognition_pad_value"]

def torch_recognize(image_path, bbox, padding=2):
    view = load_image(image_path)
    x1, y1, x2, y2 = bbox
    cb = (max(0, int(x1) - padding), max(0, int(y1) - padding),
          min(view.width, int(x2) + padding), min(view.height, int(y2) + padding))
    if cb[2] <= cb[0] or cb[3] <= cb[1]:
        return None
    tensor, width, _ = _recognition_tensor(view, cb, torch, height=t_height)
    if width <= 0:
        return None
    padded = torch.full((1, 1, t_height, width), float(t_pad), dtype=torch.float32)
    padded[0, :, :, :width] = tensor
    widths = torch.tensor([width], dtype=torch.long)
    with torch.inference_mode():
        logits = bundle.recognizer(padded.to("cpu"), widths.to("cpu"))
    texts, _ = greedy_ctc_decode(logits, recognition_output_lengths(widths), bundle.charset)
    return texts[0] if texts else None

all_reports["torch"] = evaluate(torch_recognize, all_records, "torch (existing)")

print("\n=== Summary ===", flush=True)
for name, report in all_reports.items():
    print(f"  {name}: exact={report['exact_rate']:.4f} corpus_cer={report['corpus_cer']:.4f} errors={report['errors']} avg_ms={report['avg_ms_per_crop']:.1f}", flush=True)

run_id = time.strftime("%Y%m%d-%H%M%S")
run_dir = WORKSPACE / "work" / f"ocr-model-comparison-{run_id}"
run_dir.mkdir(parents=True, exist_ok=True)
summary = {
    "kind": "model_comparison_fixed_parsing",
    "run_id": run_id,
    "total_crops": len(all_records),
    "cases": list(CASES),
    "models": {name: {k: v for k, v in r.items() if k != "per_object"} for name, r in all_reports.items()},
    "per_object_by_model": {name: r["per_object"] for name, r in all_reports.items()},
}
out = run_dir / "comparison.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"\nsaved {out}", flush=True)
