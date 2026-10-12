"""Quick test: en_PP-OCRv5_mobile_rec on 853 GT crops."""
import json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, r"E:\EDA挑战题目六\src")

from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth
from pcbparse.evaluation.text_box_eval import character_error_rate

DATASET = Path(r"E:\EDA挑战题目六\Dataset300\Dataset300")
CASES = ("0001", "0005", "0016", "0087", "0088")

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
print("loading en_PP-OCRv5_mobile_rec ...", flush=True)
t0 = time.time()
model = TextRecognition(model_name="en_PP-OCRv5_mobile_rec")
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

exact = 0
total_edits = 0.0
total_chars = 0
by_field = {}
errors = []
for rec in all_records:
    img = Image.open(rec["image_path"]).convert("RGB")
    x1, y1, x2, y2 = rec["bbox"]
    cb = (max(0, int(x1) - 2), max(0, int(y1) - 2), min(img.width, int(x2) + 2), min(img.height, int(y2) + 2))
    if cb[2] <= cb[0] or cb[3] <= cb[1]:
        pred = None
    else:
        crop = np.array(img.crop(cb))
        try:
            result = model.predict(input=crop, batch_size=1)
            pred = None
            for res in result:
                raw = res.get("rec_text")
                if raw is None:
                    continue
                if not isinstance(raw, str):
                    raise TypeError(f"rec_text must be str, got {type(raw).__name__}")
                pred = raw
                break
        except Exception as exc:
            pred = f"__ERROR__:{type(exc).__name__}"
    gt_val = rec["gt_value"]
    cer = character_error_rate(gt_val, pred or "")
    is_exact = pred == gt_val
    if is_exact:
        exact += 1
    else:
        errors.append({"object_id": rec["object_id"], "gt": gt_val, "pred": pred, "cer": cer})
    total_edits += cer * len(gt_val)
    total_chars += len(gt_val)
    f = rec["field"]
    if f not in by_field:
        by_field[f] = {"n": 0, "exact": 0}
    by_field[f]["n"] += 1
    if is_exact:
        by_field[f]["exact"] += 1

n = len(all_records)
print(f"en_PP-OCRv5_mobile_rec: exact={exact}/{n} = {exact/n:.4f} corpus_cer={total_edits/total_chars:.4f}", flush=True)
for f in sorted(by_field):
    b = by_field[f]
    print(f"  {f}: {b['exact']}/{b['n']}", flush=True)

# Save
run_id = time.strftime("%Y%m%d-%H%M%S")
out = Path(rf"E:\EDA挑战题目六\work\ocr-model-comparison-{run_id}")
out.mkdir(parents=True, exist_ok=True)
report = {
    "model": "en_PP-OCRv5_mobile_rec",
    "n": n, "exact": exact, "exact_rate": exact / n,
    "corpus_cer": total_edits / total_chars,
    "by_field": by_field,
    "errors_sample": errors[:50],
}
(out / "en_mobile_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"saved {out / 'en_mobile_result.json'}", flush=True)
