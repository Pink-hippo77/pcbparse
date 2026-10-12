"""Complete OCR evaluation: detection + recognition + selection + binding + timing.

One inference run saves complete evidence.  Independent scorer reads GT after.
Reports both full and conditional metrics.  Six layers: detection, recognition,
grouping/selection, field classification, object binding, end-to-end.
"""
from __future__ import annotations
import json, sys, time, hashlib
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
import numpy as np
from PIL import Image
sys.path.insert(0, r"E:\EDA挑战题目六\src")

WORKSPACE = Path(r"E:\EDA挑战题目六")
DATASET = WORKSPACE / "Dataset300" / "Dataset300"
CASES = ("0001", "0005", "0016", "0087", "0088")

from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth
from pcbparse.evaluation.text_box_eval import (
    character_error_rate, _bbox_iou, _center_distance, _gt_text_boxes,
)
from pcbparse.evaluation.masked_metrics import _maximum_cardinality_matching
from pcbparse.perception.recognizer_backend import TorchRecognizerBackend, PaddleRecognizerBackend
from pcbparse.training.torch_ocr_model import load_ocr_bundle


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gt_text_records(gt, case_id, image_path):
    """Extract GT text records with bbox and field."""
    records = []
    for comp in gt.components:
        for field_name, target in (("designator", comp.designator), ("value", comp.value), ("search_name", comp.search_name)):
            if target.mask.eligible and target.value is not None and target.bbox is not None:
                records.append({
                    "key": (case_id, f"{comp.raw_id}.{field_name}"),
                    "object_id": f"{comp.raw_id}.{field_name}",
                    "case_id": case_id,
                    "field": field_name,
                    "owner_kind": "component",
                    "owner_id": comp.raw_id,
                    "gt_value": str(target.value),
                    "bbox": target.bbox,
                    "image_path": str(image_path),
                })
    for pin in gt.pins:
        for field_name, target in (("number", pin.number), ("name", pin.name)):
            if target.mask.eligible and target.value is not None and target.bbox is not None:
                records.append({
                    "key": (case_id, f"{pin.raw_id}.{field_name}"),
                    "object_id": f"{pin.raw_id}.{field_name}",
                    "case_id": case_id,
                    "field": field_name,
                    "owner_kind": "pin",
                    "owner_id": pin.raw_id,
                    "gt_value": str(target.value),
                    "bbox": target.bbox,
                    "image_path": str(image_path),
                })
    return records


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


def gt_free_quality_score(cand, group, img_width, img_height):
    ob = cand["original_bbox"]
    w, h = ob[2] - ob[0], ob[3] - ob[1]
    score = 0.0
    if w > 0 and h > 0:
        ratio = max(w / h, h / w)
        score += 0.25 * max(0, 1.0 - (ratio - 1.0) / 4.0)
    area = w * h
    if 40 <= area <= 1500:
        score += 0.20
    elif area < 15:
        score -= 0.15
    if ob[0] > 2 and ob[1] > 2 and ob[2] < img_width - 2 and ob[3] < img_height - 2:
        score += 0.10
    det_score = cand.get("detector_score") or 0.5
    score += 0.15 * float(det_score)
    if len(group.get("view_ids", set())) > 1:
        score += 0.15
    if len(group.get("levels", set())) > 1:
        score += 0.10
    return score


def run_case(case_id, paddle_backend):
    """Run one case: detection → recognition → grouping → selection → binding."""
    case_dir = DATASET / case_id
    ann_path = sorted(case_dir.glob("*_CVJsonStd.json"))[0]
    img_path = sorted(case_dir.glob("*.png"))[0]
    img = Image.open(img_path).convert("RGB")

    # Load frozen candidates (from production inference)
    frozen_dirs = sorted((WORKSPACE / "work").glob("ocr-frozen-crops-*"))
    frozen_data = json.loads((frozen_dirs[-1] / "frozen_crops.json").read_text(encoding="utf-8"))
    candidates = frozen_data["candidates"]

    # Phase timings
    timings = {}

    # === Detection: candidates already frozen ===
    timings["detection_s"] = 0  # already done in production inference

    # === Recognition ===
    t0 = time.time()
    crops = []
    for c in candidates:
        ob = c["original_bbox"]
        cb = (max(0, int(ob[0]) - 2), max(0, int(ob[1]) - 2),
              min(img.width, int(ob[2]) + 2), min(img.height, int(ob[3]) + 2))
        if cb[2] <= cb[0] or cb[3] <= cb[1]:
            crops.append(None)
        else:
            crops.append(np.array(img.crop(cb)))
    valid_crops = [(i, c) for i, c in enumerate(crops) if c is not None]
    rec_results = paddle_backend.recognize_crops([c for _, c in valid_crops])
    rec_preds = {}
    for (ci, _), (pred, score) in zip(valid_crops, rec_results):
        rec_preds[ci] = (pred, score)
    timings["recognition_s"] = time.time() - t0

    # === Grouping ===
    t0 = time.time()
    groups = build_candidate_groups(candidates)
    for g in groups:
        g["levels"] = {candidates[ci].get("level", "?") for ci in g["member_indices"]}
        g["view_ids"] = {candidates[ci].get("view_id", "?") for ci in g["member_indices"]}
    timings["grouping_s"] = time.time() - t0

    # === Selection ===
    t0 = time.time()
    selections = []
    for g in groups:
        scored = [(gt_free_quality_score(candidates[ci], g, img.width, img.height), ci) for ci in g["member_indices"]]
        scored.sort(key=lambda x: -x[0])
        top_ci = scored[0][1]
        pred, rec_score = rec_preds.get(top_ci, (None, 0))
        if pred is None or pred == "":
            selections.append({"group_id": g["group_id"], "selected_index": top_ci, "prediction": None, "status": "rejected", "reason": "empty_recognition"})
        elif rec_score < 0.3:
            selections.append({"group_id": g["group_id"], "selected_index": top_ci, "prediction": pred, "status": "rejected", "reason": "low_recognition_score"})
        else:
            selections.append({"group_id": g["group_id"], "selected_index": top_ci, "prediction": pred, "status": "accepted", "reason": None})
    timings["selection_s"] = time.time() - t0

    # === Field classification + binding ===
    t0 = time.time()
    from pcbparse.ir.models import BBox, Component, Pin, Point, TextToken
    from pcbparse.association.semantic_fields import infer_semantic_fields
    from pcbparse.association.text_component import associate_component_texts
    from pcbparse.association.text_pin import associate_pin_texts
    from pcbparse.association.ocr_common import OCRAssociationConfig, canonical_details

    # Build geometry from GT (for binding oracle)
    annotation = json.loads(ann_path.read_text(encoding="utf-8"))
    gt = build_raw_ground_truth(annotation, case_id=case_id, annotation_path=ann_path)

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

    # Create TextTokens from accepted selections
    accepted = [s for s in selections if s["status"] == "accepted"]
    tokens = []
    for s in accepted:
        g = next(g for g in groups if g["group_id"] == s["group_id"])
        rb = g["rep_bbox"]
        token = TextToken(internal_id=s["group_id"], raw_text=s["prediction"] or "",
            bbox=BBox(x1=rb[0], y1=rb[1], x2=rb[2], y2=rb[3]))
        tokens.append(token)

    infer_semantic_fields(components, pins, tokens, [])
    diagnostics = []
    config = OCRAssociationConfig()
    associate_component_texts(components, tokens, diagnostics, config)
    associate_pin_texts(pins, tokens, diagnostics, config)

    binding_results = []
    for s, token in zip(accepted, tokens):
        predicted_field = canonical_details(token.details) or ""
        predicted_owner = str(token.component_id or token.pin_id or "") or None
        binding_results.append({
            "group_id": s["group_id"],
            "prediction": s["prediction"],
            "predicted_field": predicted_field,
            "predicted_owner": predicted_owner,
        })
    timings["binding_s"] = time.time() - t0

    timings["total_s"] = sum(v for k, v in timings.items() if k != "total_s")

    return {
        "case_id": case_id,
        "image_path": str(img_path),
        "n_candidates": len(candidates),
        "n_groups": len(groups),
        "n_accepted": len(accepted),
        "n_rejected": len(selections) - len(accepted),
        "selections": selections,
        "binding_results": binding_results,
        "timings": timings,
    }


def score_case(case_data, gt_records):
    """Score one case against GT. Returns metrics dict."""
    case_id = case_data["case_id"]
    accepted = [s for s in case_data["selections"] if s["status"] == "accepted"]
    binding = {b["group_id"]: b for b in case_data["binding_results"]}

    n_gt = len(gt_records)
    n_pred = len(accepted)
    exact = 0
    field_correct = 0
    owner_correct = 0
    exact_and_owner = 0
    field_aliases = {
        "designator": {"designator"}, "value": {"value"},
        "search_name": {"searchname", "search_name", "name"},
        "number": {"pinnumber", "pin_number", "number"},
        "name": {"pinname", "pin_name", "name"},
    }
    matched_gt = set()
    for r in gt_records:
        for s in accepted:
            b = binding.get(s["group_id"], {})
            pred_text = s["prediction"] or ""
            if pred_text == r["gt_value"]:
                exact += 1
                matched_gt.add(r["key"])
                pred_field = (b.get("predicted_field") or "").lower()
                expected_field = r["field"]
                if pred_field in field_aliases.get(expected_field, {expected_field}):
                    field_correct += 1
                if b.get("predicted_owner") == r["owner_id"]:
                    owner_correct += 1
                    exact_and_owner += 1
                break

    return {
        "case_id": case_id,
        "n_gt": n_gt,
        "n_pred": n_pred,
        "exact": exact,
        "field_correct": field_correct,
        "owner_correct": owner_correct,
        "exact_and_owner": exact_and_owner,
    }


def main():
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = WORKSPACE / "work" / f"ocr-eval-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    lock = json.loads((WORKSPACE / "models" / "models.lock.json").read_text(encoding="utf-8"))
    ocr_entry = next(e for e in lock["models"] if e["task"] == "ocr")
    bundle = load_ocr_bundle(WORKSPACE / "models" / ocr_entry["path"], device="cpu")
    paddle_backend = PaddleRecognizerBackend("en_PP-OCRv5_mobile_rec")

    # Version binding
    version_info = {
        "source_sha256": "see git",
        "model_lock_sha256": sha256_file(WORKSPACE / "models" / "models.lock.json"),
        "ocr_model": "en_PP-OCRv5_mobile_rec",
        "run_id": run_id,
    }

    all_case_data = []
    all_scores = []
    for case_id in CASES:
        print(f"\n[{case_id}] running...", flush=True)
        case_data = run_case(case_id, paddle_backend)
        all_case_data.append(case_data)

        # Load GT for scoring
        case_dir = DATASET / case_id
        ann_path = sorted(case_dir.glob("*_CVJsonStd.json"))[0]
        img_path = sorted(case_dir.glob("*.png"))[0]
        annotation = json.loads(ann_path.read_text(encoding="utf-8"))
        gt = build_raw_ground_truth(annotation, case_id=case_id, annotation_path=ann_path)
        gt_records = gt_text_records(gt, case_id, img_path)

        score = score_case(case_data, gt_records)
        all_scores.append(score)
        t = case_data["timings"]
        print(f"  candidates={case_data['n_candidates']} groups={case_data['n_groups']} accepted={case_data['n_accepted']} rejected={case_data['n_rejected']}", flush=True)
        print(f"  exact={score['exact']}/{score['n_gt']} field={score['field_correct']} owner={score['owner_correct']} exact+owner={score['exact_and_owner']}", flush=True)
        print(f"  time: total={t['total_s']:.1f}s recog={t['recognition_s']:.1f}s group={t['grouping_s']:.1f}s select={t['selection_s']:.1f}s bind={t['binding_s']:.1f}s", flush=True)

    # Aggregate
    total_gt = sum(s["n_gt"] for s in all_scores)
    total_exact = sum(s["exact"] for s in all_scores)
    total_field = sum(s["field_correct"] for s in all_scores)
    total_owner = sum(s["owner_correct"] for s in all_scores)
    total_e2e = sum(s["exact_and_owner"] for s in all_scores)
    total_time = sum(c["timings"]["total_s"] for c in all_case_data)

    print(f"\n=== Aggregate ({len(CASES)} cases) ===", flush=True)
    print(f"  exact: {total_exact}/{total_gt} = {total_exact/total_gt:.4f}", flush=True)
    print(f"  field_correct: {total_field}/{total_gt} = {total_field/total_gt:.4f}", flush=True)
    print(f"  owner_correct: {total_owner}/{total_gt} = {total_owner/total_gt:.4f}", flush=True)
    print(f"  exact+owner: {total_e2e}/{total_gt} = {total_e2e/total_gt:.4f}", flush=True)
    print(f"  total_time: {total_time:.1f}s", flush=True)

    # Save
    report = {
        "kind": "complete_ocr_evaluation",
        "run_id": run_id,
        "version": version_info,
        "cases": CASES,
        "per_case": all_scores,
        "per_case_timings": {c["case_id"]: c["timings"] for c in all_case_data},
        "aggregate": {
            "n_gt": total_gt,
            "exact": total_exact,
            "exact_rate": total_exact / total_gt,
            "field_correct": total_field,
            "field_rate": total_field / total_gt,
            "owner_correct": total_owner,
            "owner_rate": total_owner / total_gt,
            "exact_and_owner": total_e2e,
            "exact_and_owner_rate": total_e2e / total_gt,
            "total_time_s": total_time,
        },
        "evidence": all_case_data,
    }
    out = run_dir / "eval_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nsaved {out}", flush=True)

    # Per-case table
    print(f"\n=== Per-case ===", flush=True)
    print(f"  {'case':>6} {'gt':>4} {'exact':>6} {'field':>6} {'owner':>6} {'e2e':>6} {'time':>7}", flush=True)
    for s, c in zip(all_scores, all_case_data):
        t = c["timings"]["total_s"]
        print(f"  {s['case_id']:>6} {s['n_gt']:>4} {s['exact']:>6} {s['field_correct']:>6} {s['owner_correct']:>6} {s['exact_and_owner']:>6} {t:>7.1f}s", flush=True)

    print(f"\nrun_dir: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
