"""Task 1: freeze 0005 predicted crops before recognition.

Runs the production detector, saves every pre-recognition candidate with
box, crop, stable ID, transform, and hash.  Both recognizers then receive
identical frozen crops.  GT only used for scoring.
"""
from __future__ import annotations
import json, sys, time, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
sys.path.insert(0, r"E:\EDA挑战题目六\src")

WORKSPACE = Path(r"E:\EDA挑战题目六")
DATASET = WORKSPACE / "Dataset300" / "Dataset300"
CASE_ID = "0005"

from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth
from pcbparse.evaluation.text_box_eval import character_error_rate, _bbox_iou, _center_distance, _gt_text_boxes
from pcbparse.evaluation.masked_metrics import _maximum_cardinality_matching
from pcbparse.perception.recognizer_backend import (
    TorchRecognizerBackend,
    PaddleRecognizerBackend,
)
from pcbparse.perception.production_provider import load_production_provider
from pcbparse.training.torch_ocr_model import load_ocr_bundle

# Load production provider to get detector + multi-level OCR scheduling
lock = json.loads((WORKSPACE / "models" / "models.lock.json").read_text(encoding="utf-8"))
ocr_entry = next(e for e in lock["models"] if e["task"] == "ocr")
bundle = load_ocr_bundle(WORKSPACE / "models" / ocr_entry["path"], device="cpu")
torch_backend = TorchRecognizerBackend(bundle, device="cpu")
paddle_backend = PaddleRecognizerBackend("en_PP-OCRv5_mobile_rec")
print(f"torch: {torch_backend.version}", flush=True)
print(f"paddle: {paddle_backend.version}", flush=True)

provider = load_production_provider(WORKSPACE / "models" / "models.lock.json")

# Run detection-only: capture pre-recognition candidates from trace
case_dir = DATASET / CASE_ID
ann_path = sorted(case_dir.glob("*_CVJsonStd.json"))[0]
img_path = sorted(case_dir.glob("*.png"))[0]
annotation = json.loads(ann_path.read_text(encoding="utf-8"))
gt = build_raw_ground_truth(annotation, case_id=CASE_ID, annotation_path=ann_path)
gt_boxes = _gt_text_boxes(gt)

from pcbparse.pipeline.orchestrator import infer_one
config = json.loads((WORKSPACE / "configs" / "pipeline.yaml").read_text(encoding="utf-8"))
config.setdefault("run", {})["submission_mode"] = "degraded"

output_dir = WORKSPACE / "work" / f"ocr-frozen-crops-{time.strftime('%Y%m%d-%H%M%S')}"
output_dir.mkdir(parents=True, exist_ok=True)

t0 = time.time()
try:
    result = infer_one(
        img_path,
        output_dir / f"{CASE_ID}.json",
        provider=provider,
        config=config,
        official_profile=WORKSPACE / "configs" / "official_schema.yaml",
    )
    infer_status = "ok"
except Exception as exc:
    infer_status = f"{type(exc).__name__}: {exc}"
infer_elapsed = time.time() - t0
print(f"infer: {infer_status} in {infer_elapsed:.1f}s", flush=True)

ocr_provider = getattr(provider, "ocr_provider", None)
trace_log = getattr(ocr_provider, "_detection_trace_log", None) or []
print(f"detection trace entries: {len(trace_log)}", flush=True)

# Freeze pre-recognition candidates (geometry_accepted only)
frozen = []
img = Image.open(img_path).convert("RGB")
for entry in trace_log:
    if not entry.get("geometry_accepted") or entry.get("original_bbox") is None:
        continue
    ob = entry["original_bbox"]
    vb = entry.get("view_bbox")
    cb = entry.get("crop_box")
    # Reconstruct crop from original image using original_bbox
    x1, y1, x2, y2 = ob
    crop_box = (max(0, int(x1) - 2), max(0, int(y1) - 2),
                min(img.width, int(x2) + 2), min(img.height, int(y2) + 2))
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        continue
    crop = img.crop(crop_box)
    crop_array = np.array(crop)
    crop_hash = hashlib.sha256(crop_array.tobytes()).hexdigest()[:16]
    stable_id = f"{entry.get('view_id', '?')}:{entry.get('detection_index', '?')}:{crop_hash}"
    frozen.append({
        "stable_id": stable_id,
        "case_id": CASE_ID,
        "view_id": entry.get("view_id"),
        "detection_index": entry.get("detection_index"),
        "level": entry.get("level"),
        "batch_id": entry.get("batch_id"),
        "original_bbox": ob,
        "view_bbox": vb,
        "crop_box": list(crop_box),
        "crop_hash": crop_hash,
        "detector_score": entry.get("detector_score"),
        "transform": entry.get("transform"),
        "crop_shape": list(crop_array.shape),
    })

print(f"frozen candidates: {len(frozen)}", flush=True)

# Save frozen crop metadata
meta_path = output_dir / "frozen_crops.json"
meta_path.write_text(json.dumps({
    "kind": "frozen_pre_recognition_crops",
    "case_id": CASE_ID,
    "image_path": str(img_path),
    "infer_status": infer_status,
    "infer_elapsed_s": infer_elapsed,
    "total_trace": len(trace_log),
    "frozen_count": len(frozen),
    "candidates": frozen,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"saved {meta_path}", flush=True)

# Reconstruct crops and send to both backends
frozen_crops = []
for fc in frozen:
    cb = tuple(fc["crop_box"])
    crop = img.crop(cb)
    frozen_crops.append(np.array(crop))

t0 = time.time()
torch_results = torch_backend.recognize_crops(frozen_crops)
torch_time = time.time() - t0

t0 = time.time()
paddle_results = paddle_backend.recognize_crops(frozen_crops)
paddle_time = time.time() - t0

print(f"\ntorch recognition: {torch_time:.1f}s  paddle: {paddle_time:.1f}s", flush=True)

# Match frozen candidates to GT for scoring
candidates_match = []
valid = []
for i, fc in enumerate(frozen):
    ob = fc["original_bbox"]
    valid.append((i, tuple(ob)))
for pred_idx, (_i, bbox) in enumerate(valid):
    for gt_idx, target in enumerate(gt_boxes):
        iou = _bbox_iou(bbox, target["bbox"])
        dist = _center_distance(bbox, target["bbox"])
        if iou >= 0.3 or dist <= 12.0:
            candidates_match.append((pred_idx, gt_idx, max(iou, 0.01)))
matches = _maximum_cardinality_matching(
    len(valid), len(gt_boxes), candidates_match,
    left_keys=[fc["stable_id"] for fc in frozen],
    right_keys=[item["object_id"] for item in gt_boxes],
)
pred_to_gt = {left: right for left, right, _s in matches}
print(f"matched {len(pred_to_gt)} frozen candidates to GT boxes", flush=True)

# Score both backends on matched candidates
torch_exact = 0
paddle_exact = 0
torch_cer_sum = 0.0
paddle_cer_sum = 0.0
total_chars = 0
for i, fc in enumerate(frozen):
    gt_idx = pred_to_gt.get(i)
    if gt_idx is None:
        continue
    gt_val = gt_boxes[gt_idx]["gt_value"]
    t_pred = torch_results[i][0] or ""
    p_pred = paddle_results[i][0] or ""
    t_cer = character_error_rate(gt_val, t_pred)
    p_cer = character_error_rate(gt_val, p_pred)
    if t_pred == gt_val: torch_exact += 1
    if p_pred == gt_val: paddle_exact += 1
    torch_cer_sum += t_cer * len(gt_val)
    paddle_cer_sum += p_cer * len(gt_val)
    total_chars += len(gt_val)

n_matched = len(pred_to_gt)
print(f"\n=== Frozen prediction-box scoring ({n_matched} matched) ===", flush=True)
print(f"  torch:  exact={torch_exact}/{n_matched} = {torch_exact/n_matched:.4f}  corpus_cer={torch_cer_sum/total_chars:.4f}", flush=True)
print(f"  paddle: exact={paddle_exact}/{n_matched} = {paddle_exact/n_matched:.4f}  corpus_cer={paddle_cer_sum/total_chars:.4f}", flush=True)

# Confidence analysis
t_scores = [r[1] for r in torch_results if r[0] is not None]
p_scores = [r[1] for r in paddle_results if r[0] is not None]
print(f"\n  torch score range: [{min(t_scores):.3f}, {max(t_scores):.3f}] mean={np.mean(t_scores):.3f}", flush=True)
print(f"  paddle score range: [{min(p_scores):.3f}, {max(p_scores):.3f}] mean={np.mean(p_scores):.3f}", flush=True)

report = {
    "kind": "frozen_prediction_box_comparison",
    "case_id": CASE_ID,
    "frozen_count": len(frozen),
    "matched_count": n_matched,
    "torch": {"exact": torch_exact, "exact_rate": torch_exact / n_matched if n_matched else 0, "corpus_cer": torch_cer_sum / total_chars if total_chars else 0, "time_s": torch_time},
    "paddle": {"exact": paddle_exact, "exact_rate": paddle_exact / n_matched if n_matched else 0, "corpus_cer": paddle_cer_sum / total_chars if total_chars else 0, "time_s": paddle_time},
    "confidence": {
        "torch_mean": float(np.mean(t_scores)) if t_scores else 0,
        "paddle_mean": float(np.mean(p_scores)) if p_scores else 0,
        "note": "scores not calibrated across backends",
    },
}
(output_dir / "frozen_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"\nsaved {output_dir / 'frozen_comparison.json'}", flush=True)
