"""Three-model recognition comparison on 853 GT text crops.

Compares: existing torch OCR bundle, PP-OCRv5_server_rec, en_PP-OCRv5_mobile_rec.
Fixed crop + evaluation protocol.  Reports exact, corpus CER, field buckets, latency.
"""
from __future__ import annotations
import json, sys, time, io
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, r"E:\EDA挑战题目六\src")

WORKSPACE = Path(r"E:\EDA挑战题目六")
DATASET = WORKSPACE / "Dataset300" / "Dataset300"
CASES = ("0001", "0005", "0016", "0087", "0088")

from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth
from pcbparse.evaluation.text_box_eval import character_error_rate

from pcbparse.training.torch_ocr_model import (
    load_ocr_bundle, greedy_ctc_decode, recognition_output_lengths,
)
import torch
from pcbparse.io.image_loader import load_image
from pcbparse.perception.torch_ocr_adapter import _recognition_tensor
from PIL import Image


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


def load_all_gt():
    all_records = []
    for case_id in CASES:
        case_dir = DATASET / case_id
        ann_path = sorted(case_dir.glob("*_CVJsonStd.json"))[0]
        img_path = sorted(case_dir.glob("*.png"))[0]
        annotation = json.loads(ann_path.read_text(encoding="utf-8"))
        gt = build_raw_ground_truth(annotation, case_id=case_id, annotation_path=ann_path)
        records = gt_text_records(gt)
        for r in records:
            r["case_id"] = case_id
            r["image_path"] = str(img_path)
        all_records.extend(records)
    return all_records


def crop_numpy(image_path, bbox, padding=2):
    import numpy as np
    img = Image.open(image_path).convert("RGB")
    x1, y1, x2, y2 = bbox
    cb = (max(0, int(x1) - padding), max(0, int(y1) - padding),
          min(img.width, int(x2) + padding), min(img.height, int(y2) + padding))
    if cb[2] <= cb[0] or cb[3] <= cb[1]:
        return None
    return np.array(img.crop(cb))


def make_torch_recognizer():
    lock = json.loads((WORKSPACE / "models" / "models.lock.json").read_text(encoding="utf-8"))
    ocr_entry = next(e for e in lock["models"] if e["task"] == "ocr")
    bundle = load_ocr_bundle(WORKSPACE / "models" / ocr_entry["path"], device="cpu")
    height = bundle.preprocess["recognition_height"]
    pad_value = bundle.preprocess["recognition_pad_value"]

    def recognize(image_path, bbox, padding=2):
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

    return recognize, bundle.version


def make_paddle_recognizer(model_name):
    from paddleocr import TextRecognition
    model = TextRecognition(model_name=model_name)

    def recognize(image_path, bbox, padding=2):
        crop = crop_numpy(image_path, bbox, padding)
        if crop is None:
            return None
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

    return recognize, model_name


def evaluate(recognize_fn, records, label):
    results = []
    t0 = time.time()
    for rec in records:
        t1 = time.time()
        pred = recognize_fn(rec["image_path"], rec["bbox"])
        elapsed = time.time() - t1
        gt_val = rec["gt_value"]
        cer = character_error_rate(gt_val, pred or "")
        results.append({
            "object_id": rec["object_id"], "case_id": rec["case_id"],
            "field": rec["field"], "gt": gt_val, "pred": pred,
            "exact": pred == gt_val, "cer": cer, "elapsed": elapsed,
        })
    total = time.time() - t0
    n = len(results)
    exact = sum(1 for r in results if r["exact"])
    total_edits = sum(r["cer"] * len(r["gt"]) for r in results)
    total_chars = sum(len(r["gt"]) for r in results)
    corpus_cer = total_edits / total_chars if total_chars else 0
    mean_cer = sum(r["cer"] for r in results) / n if n else 0

    by_field = defaultdict(lambda: {"n": 0, "exact": 0, "edits": 0.0, "chars": 0})
    for r in results:
        b = by_field[r["field"]]
        b["n"] += 1
        if r["exact"]: b["exact"] += 1
        b["edits"] += r["cer"] * len(r["gt"])
        b["chars"] += len(r["gt"])

    print(f"\n=== {label} ===", flush=True)
    print(f"  exact: {exact}/{n} = {exact/n:.4f}", flush=True)
    print(f"  corpus_cer: {corpus_cer:.4f}  mean_cer: {mean_cer:.4f}", flush=True)
    print(f"  total_time: {total:.1f}s  avg_per_crop: {total/n*1000:.1f}ms", flush=True)
    for field in sorted(by_field):
        b = by_field[field]
        cer = b["edits"] / b["chars"] if b["chars"] else 0
        print(f"  {field}: n={b['n']} exact={b['exact']}/{b['n']} corpus_cer={cer:.4f}", flush=True)

    return {
        "label": label, "n": n, "exact": exact, "exact_rate": exact / n,
        "corpus_cer": corpus_cer, "mean_cer": mean_cer,
        "total_time_s": total, "avg_ms_per_crop": total / n * 1000,
        "by_field": {k: {"n": v["n"], "exact": v["exact"], "corpus_cer": v["edits"] / v["chars"] if v["chars"] else 0} for k, v in by_field.items()},
        "per_object": results,
    }


def main():
    records = load_all_gt()
    print(f"loaded {len(records)} GT text crops from {len(CASES)} cases", flush=True)

    all_reports = {}

    torch_recognize, torch_version = make_torch_recognizer()
    all_reports["torch"] = evaluate(torch_recognize, records, f"torch ({torch_version})")

    for model_name in ("PP-OCRv5_server_rec", "en_PP-OCRv5_mobile_rec"):
        try:
            paddle_recognize, _ = make_paddle_recognizer(model_name)
            all_reports[model_name] = evaluate(paddle_recognize, records, model_name)
        except Exception as exc:
            print(f"\n=== {model_name} FAILED: {type(exc).__name__}: {exc} ===", flush=True)

    print("\n=== Summary ===", flush=True)
    for name, report in all_reports.items():
        print(f"  {name}: exact={report['exact_rate']:.4f} corpus_cer={report['corpus_cer']:.4f} avg_ms={report['avg_ms_per_crop']:.1f}", flush=True)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = WORKSPACE / "work" / f"ocr-model-comparison-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "kind": "model_comparison",
        "run_id": run_id,
        "total_crops": len(records),
        "cases": list(CASES),
        "models": {name: {k: v for k, v in r.items() if k != "per_object"} for name, r in all_reports.items()},
        "per_object_by_model": {name: r["per_object"] for name, r in all_reports.items()},
    }
    out = run_dir / "comparison.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    main()
