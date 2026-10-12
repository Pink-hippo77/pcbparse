"""Quick validation: PaddleOCR rec_text parsing fix on a few representative crops."""
import json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, r"E:\EDA挑战题目六\src")

from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth

DATASET = Path(r"E:\EDA挑战题目六\Dataset300\Dataset300")

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

# Pick a few representative crops from different cases
samples = []
for case_id in ("0001", "0087"):
    case_dir = DATASET / case_id
    ann_path = sorted(case_dir.glob("*_CVJsonStd.json"))[0]
    img_path = sorted(case_dir.glob("*.png"))[0]
    annotation = json.loads(ann_path.read_text(encoding="utf-8"))
    gt = build_raw_ground_truth(annotation, case_id=case_id, annotation_path=ann_path)
    recs = gt_text_records(gt)
    for r in recs:
        r["case_id"] = case_id
        r["image_path"] = str(img_path)
    # Pick diverse fields
    by_field = {}
    for r in recs:
        if r["field"] not in by_field:
            by_field[r["field"]] = r
    samples.extend(by_field.values())

print(f"validation samples: {len(samples)}", flush=True)
for s in samples:
    print(f"  {s['case_id']}:{s['object_id']} field={s['field']} gt={s['gt_value']!r}", flush=True)

from paddleocr import TextRecognition

for model_name in ("en_PP-OCRv5_mobile_rec", "PP-OCRv5_server_rec"):
    print(f"\n--- {model_name} ---", flush=True)
    try:
        model = TextRecognition(model_name=model_name)
    except Exception as exc:
        print(f"  LOAD FAILED: {type(exc).__name__}: {exc}", flush=True)
        continue
    for s in samples:
        img = Image.open(s["image_path"]).convert("RGB")
        x1, y1, x2, y2 = s["bbox"]
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
        match = "OK" if pred == s["gt_value"] else "MISS"
        print(f"  {s['object_id']}: gt={s['gt_value']!r} pred={pred!r} [{match}]", flush=True)
