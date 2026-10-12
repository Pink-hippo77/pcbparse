"""Fixed: evaluation key (case_id,object_id), error/empty separation, gain-loss identity.

Phase 2 real: freeze pre-recognition candidates from one image inference,
send to both backends, GT only for scoring.
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

from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth
from pcbparse.evaluation.text_box_eval import character_error_rate, _bbox_iou, _center_distance, _gt_text_boxes
from pcbparse.evaluation.masked_metrics import _maximum_cardinality_matching
from pcbparse.perception.recognizer_backend import (
    TorchRecognizerBackend,
    PaddleRecognizerBackend,
)
from pcbparse.training.torch_ocr_model import load_ocr_bundle

lock = json.loads((WORKSPACE / "models" / "models.lock.json").read_text(encoding="utf-8"))
ocr_entry = next(e for e in lock["models"] if e["task"] == "ocr")
bundle = load_ocr_bundle(WORKSPACE / "models" / ocr_entry["path"], device="cpu")
torch_backend = TorchRecognizerBackend(bundle, device="cpu")
paddle_backend = PaddleRecognizerBackend("en_PP-OCRv5_mobile_rec")
print(f"torch: {torch_backend.version}", flush=True)
print(f"paddle: {paddle_backend.version}", flush=True)


def gt_text_records(gt, case_id, image_path):
    records = []
    for comp in gt.components:
        for field, target in (("designator", comp.designator), ("value", comp.value), ("search_name", comp.search_name)):
            if target.mask.eligible and target.value is not None and target.bbox is not None:
                records.append({"key": (case_id, f"{comp.raw_id}.{field}"), "object_id": f"{comp.raw_id}.{field}", "case_id": case_id, "field": field, "gt_value": str(target.value), "bbox": target.bbox, "image_path": str(image_path)})
    for pin in gt.pins:
        for field, target in (("number", pin.number), ("name", pin.name)):
            if target.mask.eligible and target.value is not None and target.bbox is not None:
                records.append({"key": (case_id, f"{pin.raw_id}.{field}"), "object_id": f"{pin.raw_id}.{field}", "case_id": case_id, "field": field, "gt_value": str(target.value), "bbox": target.bbox, "image_path": str(image_path)})
    return records


def load_all_gt():
    all_records = []
    seen = set()
    for case_id in ("0001", "0005", "0016", "0087", "0088"):
        case_dir = DATASET / case_id
        ann_path = sorted(case_dir.glob("*_CVJsonStd.json"))[0]
        img_path = sorted(case_dir.glob("*.png"))[0]
        annotation = json.loads(ann_path.read_text(encoding="utf-8"))
        gt = build_raw_ground_truth(annotation, case_id=case_id, annotation_path=ann_path)
        recs = gt_text_records(gt, case_id, img_path)
        for r in recs:
            if r["key"] in seen:
                raise ValueError(f"duplicate key: {r['key']}")
            seen.add(r["key"])
        all_records.extend(recs)
    return all_records


def make_crops(records, padding=2):
    crops = []
    for rec in records:
        img = Image.open(rec["image_path"]).convert("RGB")
        x1, y1, x2, y2 = rec["bbox"]
        cb = (max(0, int(x1) - padding), max(0, int(y1) - padding),
              min(img.width, int(x2) + padding), min(img.height, int(y2) + padding))
        if cb[2] <= cb[0] or cb[3] <= cb[1]:
            crops.append(None)
        else:
            crops.append(np.array(img.crop(cb)))
    return crops


def evaluate_backend(backend, records, crops, label):
    t0 = time.time()
    results_raw = backend.recognize_crops(crops)
    elapsed = time.time() - t0

    if len(results_raw) != len(records):
        raise ValueError(f"output count mismatch: {len(results_raw)} != {len(records)}")

    results = []
    empty_count = 0
    error_count = 0
    for rec, (pred, score) in zip(records, results_raw):
        gt_val = rec["gt_value"]
        is_error = isinstance(pred, str) and pred.startswith("__ERROR__")
        is_empty = pred is None or (isinstance(pred, str) and pred == "")
        cer = character_error_rate(gt_val, pred or "")
        is_exact = pred == gt_val
        if is_empty:
            empty_count += 1
        if is_error:
            error_count += 1
        results.append({
            "key": list(rec["key"]), "case_id": rec["case_id"], "object_id": rec["object_id"],
            "field": rec["field"], "gt": gt_val, "pred": pred, "score": score,
            "exact": is_exact, "cer": cer, "is_empty": is_empty, "is_error": is_error,
        })

    n = len(results)
    exact = sum(1 for r in results if r["exact"])
    total_edits = sum(r["cer"] * len(r["gt"]) for r in results)
    total_chars = sum(len(r["gt"]) for r in results)
    corpus_cer = total_edits / total_chars if total_chars else 0

    by_field = defaultdict(lambda: {"n": 0, "exact": 0, "empty": 0, "error": 0})
    for r in results:
        b = by_field[r["field"]]
        b["n"] += 1
        if r["exact"]: b["exact"] += 1
        if r["is_empty"]: b["empty"] += 1
        if r["is_error"]: b["error"] += 1

    print(f"\n=== {label} ===", flush=True)
    print(f"  exact: {exact}/{n} = {exact/n:.4f}  empty: {empty_count}  errors: {error_count}", flush=True)
    print(f"  corpus_cer: {corpus_cer:.4f}  time: {elapsed:.1f}s  avg: {elapsed/n*1000:.1f}ms", flush=True)
    for field in sorted(by_field):
        b = by_field[field]
        print(f"  {field}: n={b['n']} exact={b['exact']}/{b['n']} empty={b['empty']} err={b['error']}", flush=True)

    return {
        "label": label, "n": n, "exact": exact, "exact_rate": exact / n,
        "empty_count": empty_count, "error_count": error_count,
        "corpus_cer": corpus_cer, "total_time_s": elapsed,
        "avg_ms_per_crop": elapsed / n * 1000,
        "by_field": {k: dict(v) for k, v in by_field.items()},
        "per_object": results,
    }


def main():
    records = load_all_gt()
    print(f"loaded {len(records)} GT crops (unique keys verified)", flush=True)
    crops = make_crops(records)

    torch_report = evaluate_backend(torch_backend, records, crops, "torch (existing)")
    paddle_report = evaluate_backend(paddle_backend, records, crops, "en_PP-OCRv5_mobile_rec")

    # Diff with (case_id, object_id) keys
    torch_by_key = {tuple(r["key"]): r for r in torch_report["per_object"]}
    paddle_by_key = {tuple(r["key"]): r for r in paddle_report["per_object"]}
    assert set(torch_by_key) == set(paddle_by_key), "key sets must match"

    gain = []
    loss = []
    for key in sorted(torch_by_key):
        t = torch_by_key[key]
        p = paddle_by_key[key]
        if p["exact"] and not t["exact"]:
            gain.append({"key": list(key), "field": t["field"], "gt": t["gt"], "torch": t["pred"], "paddle": p["pred"]})
        elif t["exact"] and not p["exact"]:
            loss.append({"key": list(key), "field": t["field"], "gt": t["gt"], "torch": t["pred"], "paddle": p["pred"]})

    # Identity check: gain - loss == paddle_exact - torch_exact
    identity_lhs = len(gain) - len(loss)
    identity_rhs = paddle_report["exact"] - torch_report["exact"]
    print(f"\n=== Diff (identity check) ===", flush=True)
    print(f"  gain: {len(gain)}  loss: {len(loss)}  net: {identity_lhs:+d}", flush=True)
    print(f"  identity: gain-loss={identity_lhs} == paddle_exact-torch_exact={identity_rhs} {'OK' if identity_lhs == identity_rhs else 'MISMATCH!'}", flush=True)
    assert identity_lhs == identity_rhs, f"identity violated: {identity_lhs} != {identity_rhs}"

    for item in loss[:10]:
        print(f"    LOSS {item['key']}: gt={item['gt']!r} torch={item['torch']!r} paddle={item['paddle']!r}", flush=True)

    # Real batch timing with varied crop sizes
    from collections import Counter
    sizes = [(c.shape[0], c.shape[1]) for c in crops if c is not None]
    size_dist = Counter(sizes).most_common(5)
    print(f"\n  crop size distribution (top 5): {size_dist}", flush=True)

    # Time with real crops in batches of 32
    batch_size = 32
    real_crops = [c for c in crops if c is not None]
    t0 = time.time()
    for i in range(0, min(len(real_crops), 320), batch_size):
        batch = real_crops[i:i+batch_size]
        paddle_backend.recognize_crops(batch)
    batch_elapsed = time.time() - t0
    n_batched = min(len(real_crops), 320)
    per_crop = batch_elapsed / n_batched * 1000
    print(f"  real batch timing: {n_batched} crops in {batch_elapsed:.2f}s = {per_crop:.1f}ms/crop (sequential predict)", flush=True)
    est_full = 11454 * per_crop / 1000
    print(f"  estimated full 11454 candidates: {est_full:.0f}s (sequential, not true batch)", flush=True)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = WORKSPACE / "work" / f"ocr-recognizer-comparison-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "kind": "recognizer_comparison_fixed_boxes_v2",
        "run_id": run_id,
        "total_crops": len(records),
        "key_format": "(case_id, object_id)",
        "torch": {k: v for k, v in torch_report.items() if k != "per_object"},
        "paddle": {k: v for k, v in paddle_report.items() if k != "per_object"},
        "diff": {
            "gain_count": len(gain), "loss_count": len(loss), "net": identity_lhs,
            "identity_check": identity_lhs == identity_rhs,
            "loss_samples": loss[:20], "gain_samples": gain[:20],
        },
        "batch_timing": {
            "method": "sequential_predict_batch_size_1",
            "real_crops_tested": n_batched,
            "per_crop_ms": per_crop,
            "est_full_candidates_s": est_full,
            "note": "not true batch processing; Paddle predict(batch_size=1) called in loop",
        },
    }
    out = run_dir / "comparison.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    main()
