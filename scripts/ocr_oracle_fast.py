"""OCR-02 oracle-only diagnostic (fast: no full pipeline, just recognizer/classifier/binder).

Reports macro + micro + corpus-level metrics.  Detection rate is null when
not measured.  Each run writes to its own timestamped directory.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, r"E:\EDA挑战题目六\src")

from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth
from pcbparse.evaluation.ocr_oracle import (
    recognition_oracle, binding_oracle, triage_priority, aggregate_recognition,
)

CASES = ("0001", "0005", "0016", "0087", "0088")
DATASET = Path(r"E:\EDA挑战题目六\Dataset300\Dataset300")
WORKSPACE = Path(r"E:\EDA挑战题目六")

from pcbparse.training.torch_ocr_model import (
    load_ocr_bundle, greedy_ctc_decode, recognition_output_lengths,
)
import torch
from pcbparse.io.image_loader import load_image
from pcbparse.perception.torch_ocr_adapter import _recognition_tensor

lock = json.loads((WORKSPACE / "models" / "models.lock.json").read_text(encoding="utf-8"))
ocr_entry = next(e for e in lock["models"] if e["task"] == "ocr")
bundle = load_ocr_bundle(WORKSPACE / "models" / ocr_entry["path"], device="cpu")
print(f"bundle loaded: {bundle.version}", flush=True)

height = bundle.preprocess["recognition_height"]
pad_value = bundle.preprocess["recognition_pad_value"]


def make_recognize_fn(image_path):
    def recognize_fn(bbox):
        view = load_image(image_path)
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        crop_box = (max(0, x1 - 2), max(0, y1 - 2), min(view.width, x2 + 2), min(view.height, y2 + 2))
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            return None
        tensor, width, _ = _recognition_tensor(view, crop_box, torch, height=height)
        if width <= 0:
            return None
        padded = torch.full((1, 1, height, width), float(pad_value), dtype=torch.float32)
        padded[0, :, :, :width] = tensor
        widths = torch.tensor([width], dtype=torch.long)
        with torch.inference_mode():
            logits = bundle.recognizer(padded.to("cpu"), widths.to("cpu"))
        texts, _ = greedy_ctc_decode(logits, recognition_output_lengths(widths), bundle.charset)
        return texts[0] if texts else None
    return recognize_fn


from pcbparse.ir.models import BBox, Component, Pin, Point, TextToken
from pcbparse.association.semantic_fields import infer_semantic_fields
from pcbparse.association.ocr_common import canonical_details


def make_classify_fn():
    def classify_fn(text, bbox):
        token = TextToken(internal_id="t", raw_text=text, bbox=BBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]))
        infer_semantic_fields([], [], [token], [])
        return canonical_details(token.details) or ""
    return classify_fn


def make_bind_fn_real(gt, geometry_source="gt"):
    """Real association-based binding: calls infer_semantic_fields with full context."""
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

    def bind_fn(text, bbox, hint):
        token = TextToken(internal_id="t", raw_text=text, bbox=BBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]))
        infer_semantic_fields(components, pins, [token], [])
        predicted = str(token.component_id or token.pin_id or "") or None
        if hint is not None:
            return predicted
        return predicted
    return bind_fn


all_rec = []
all_bind = []
for case_id in CASES:
    case_dir = DATASET / case_id
    ann_path = sorted(case_dir.glob("*_CVJsonStd.json"))[0]
    img_path = sorted(case_dir.glob("*.png"))[0]
    annotation = json.loads(ann_path.read_text(encoding="utf-8"))
    gt = build_raw_ground_truth(annotation, case_id=case_id, annotation_path=ann_path)

    rec_report = recognition_oracle(gt, make_recognize_fn(img_path))
    all_rec.append(rec_report)
    bind_report = binding_oracle(gt, make_classify_fn(), make_bind_fn_real(gt), geometry_source="gt")
    all_bind.append(bind_report)

    n = rec_report["denominator"]
    re_ = rec_report["exact_rate"]
    rc = rec_report["mean_cer"]
    corpus_rc = rec_report.get("corpus_cer")
    bf = bind_report["field_classified_rate"]
    bo = bind_report["owner_bound_free_rate"]
    print(f"[{case_id}] gt_texts={n} rec_exact={re_:.3f} rec_mean_cer={rc:.3f} rec_corpus_cer={corpus_rc:.3f} bind_field={bf:.3f} bind_owner_free={bo:.3f}", flush=True)

agg_rec = aggregate_recognition(all_rec)
macro_exact = agg_rec["macro_exact"]
micro_exact = agg_rec["micro_exact"]
macro_cer = agg_rec["macro_mean_cer"]
corpus_cer = agg_rec["corpus_cer"]
bind_field = sum(b["field_classified_count"] for b in all_bind) / sum(b["denominator"] for b in all_bind)
bind_owner_free = sum(b["owner_bound_free_count"] for b in all_bind) / sum(b["denominator"] for b in all_bind)

print(f"MACRO rec_exact={macro_exact:.4f} rec_mean_cer={macro_cer:.4f}", flush=True)
print(f"MICRO rec_exact={micro_exact:.4f} ({agg_rec['total_exact']}/{agg_rec['total_texts']}) corpus_cer={corpus_cer:.4f}", flush=True)
print(f"BIND field={bind_field:.4f} owner_free={bind_owner_free:.4f}", flush=True)

triage = triage_priority(
    image_only_report={"detection_rate": None},
    recognition_report={"exact_rate": micro_exact, "mean_cer": corpus_cer},
    binding_report={"field_classified_rate": bind_field, "owner_bound_free_rate": bind_owner_free, "owner_bound_trusted_rate": None},
)
print(f"TRIAGE priority: {triage['priority']}", flush=True)
for f in triage["findings"]:
    print(f"  - {f}", flush=True)

run_id = time.strftime("%Y%m%d-%H%M%S")
run_dir = WORKSPACE / "work" / f"ocr-triage-{run_id}"
run_dir.mkdir(parents=True, exist_ok=True)
report = {
    "kind": "ocr_triage_oracle",
    "run_id": run_id,
    "samples": list(CASES),
    "aggregate": {
        "macro_exact": macro_exact,
        "micro_exact": micro_exact,
        "macro_mean_cer": macro_cer,
        "corpus_cer": corpus_cer,
        "total_texts": agg_rec["total_texts"],
        "total_exact": agg_rec["total_exact"],
        "binding_field_rate": bind_field,
        "binding_owner_free_rate": bind_owner_free,
        "detection_rate": None,
        "detection_note": "not_measured; 0005 single-batch test showed 27/27 rejected by no_source_pixel_support but full multi-batch trace not yet collected",
    },
    "cases": [{"case_id": c, "recognition": r, "binding": b} for c, r, b in zip(CASES, all_rec, all_bind)],
    "triage": triage,
}
out = run_dir / "triage_report.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"saved {out}", flush=True)
