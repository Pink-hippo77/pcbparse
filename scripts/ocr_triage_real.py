"""OCR-02 real three-way diagnostic on fixed dev samples.

Runs image-only baseline, GT-crop recognition oracle, and GT-text binding
oracle on the five fixed development images.  Oracle paths are isolated from
production inference.  Produces a triage report recommending the optimization
priority (detection vs recognition vs binding).

Usage (requires .venv-train with torch):
    python scripts/ocr_triage_real.py --workspace-root . --output work/ocr-triage-real
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[1]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pcbparse.evaluation.detection_eval import evaluate_detection_candidates
from pcbparse.evaluation.ocr_oracle import (
    binding_oracle,
    recognition_oracle,
    triage_priority,
)
from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth
from pcbparse.evaluation.text_box_eval import evaluate_text_boxes

DEV_SAMPLES = ("0001", "0005", "0016", "0087", "0088")


def _load_annotation(dataset_root: Path, case_id: str) -> tuple[dict[str, Any], Path]:
    case_dir = dataset_root / case_id
    for candidate in sorted(case_dir.glob("*_CVJsonStd.json")):
        return json.loads(candidate.read_text(encoding="utf-8")), candidate
    raise FileNotFoundError(f"no CVJsonStd annotation in {case_dir}")


def _load_image_path(dataset_root: Path, case_id: str) -> Path:
    case_dir = dataset_root / case_id
    for candidate in sorted(case_dir.glob("*.png")):
        return candidate
    raise FileNotFoundError(f"no PNG image in {case_dir}")


def _gt_text_targets(gt: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for component in gt.components:
        for field, target in (
            ("designator", component.designator),
            ("value", component.value),
            ("search_name", component.search_name),
        ):
            if target.mask.eligible and target.value is not None and target.bbox is not None:
                records.append(
                    {
                        "object_id": f"{component.raw_id}.{field}",
                        "owner_kind": "component",
                        "owner_id": component.raw_id,
                        "field": field,
                        "gt_value": str(target.value),
                        "bbox": target.bbox,
                    }
                )
    for pin in gt.pins:
        for field, target in (("number", pin.number), ("name", pin.name)):
            if target.mask.eligible and target.value is not None and target.bbox is not None:
                records.append(
                    {
                        "object_id": f"{pin.raw_id}.{field}",
                        "owner_kind": "pin",
                        "owner_id": pin.raw_id,
                        "field": field,
                        "gt_value": str(target.value),
                        "bbox": target.bbox,
                    }
                )
    return records


def _make_recognize_fn(ocr_bundle: Any, image_path: Path):
    from PIL import Image

    from pcbparse.perception.torch_ocr_adapter import _recognition_tensor
    from pcbparse.training.torch_ocr_model import (
        greedy_ctc_decode,
        recognition_output_lengths,
    )

    image = Image.open(image_path).convert("RGB")
    torch = __import__("torch")
    height = ocr_bundle.preprocess["recognition_height"]
    pad_value = ocr_bundle.preprocess["recognition_pad_value"]
    charset = ocr_bundle.charset
    recognizer = ocr_bundle.recognizer
    device = "cpu"

    def recognize_fn(bbox: tuple[float, float, float, float]) -> str | None:
        from pcbparse.io.image_loader import load_image

        view = load_image(image_path)
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        crop_box = (max(0, x1 - 2), max(0, y1 - 2), min(view.width, x2 + 2), min(view.height, y2 + 2))
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            return None
        tensor, width, _sha = _recognition_tensor(view, crop_box, torch, height=height)
        if width <= 0:
            return None
        padded = torch.full(
            (1, 1, height, width), float(pad_value), dtype=torch.float32
        )
        padded[0, :, :, :width] = tensor
        widths = torch.tensor([width], dtype=torch.long)
        with torch.inference_mode():
            logits = recognizer(padded.to(device), widths.to(device))
        input_lengths = recognition_output_lengths(widths)
        texts, _scores = greedy_ctc_decode(logits, input_lengths, charset)
        return texts[0] if texts else None

    return recognize_fn


def _make_classify_fn() -> Any:
    from pcbparse.association.ocr_common import canonical_details
    from pcbparse.ir.models import BBox, TextToken

    def classify_fn(text: str, bbox: tuple[float, float, float, float]) -> str:
        token = TextToken(
            internal_id="oracle-token",
            raw_text=text,
            bbox=BBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
        )
        from pcbparse.association.semantic_fields import infer_semantic_fields

        infer_semantic_fields([], [], [token], [])
        return canonical_details(token.details) or ""

    return classify_fn


def _make_bind_fn(gt: Any) -> Any:
    from pcbparse.ir.models import BBox, Component, Pin, Point, TextToken

    components = []
    pins = []
    for comp in gt.components:
        if comp.mask.eligible and comp.bbox is not None:
            components.append(
                Component(
                    internal_id=comp.raw_id,
                    key=None,
                    category=comp.category,
                    bbox=BBox(x1=comp.bbox[0], y1=comp.bbox[1], x2=comp.bbox[2], y2=comp.bbox[3]),
                )
            )
    for pin in gt.pins:
        if pin.mask.eligible and pin.tip is not None:
            pins.append(
                Pin(
                    internal_id=pin.raw_id,
                    component_id=pin.owner_raw_id,
                    point=Point(x=pin.tip[0], y=pin.tip[1]),
                )
            )

    def bind_fn(text: str, bbox: tuple[float, float, float, float], hint: str | None) -> str | None:
        if hint is not None:
            return hint
        token = TextToken(
            internal_id="oracle-token",
            raw_text=text,
            details="",
            bbox=BBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
        )
        from pcbparse.association.semantic_fields import infer_semantic_fields

        infer_semantic_fields(components, pins, [token], [])
        if token.component_id:
            return str(token.component_id)
        if token.pin_id:
            return str(token.pin_id)
        return None

    return bind_fn


def run_case(
    *,
    dataset_root: Path,
    case_id: str,
    workspace_root: Path,
    provider: Any = None,
    ocr_bundle: Any = None,
) -> dict[str, Any]:
    annotation, annotation_path = _load_annotation(dataset_root, case_id)
    image_path = _load_image_path(dataset_root, case_id)
    gt = build_raw_ground_truth(annotation, case_id=case_id, annotation_path=annotation_path)

    from pcbparse.pipeline.orchestrator import infer_one

    if provider is None:
        from pcbparse.perception.production_provider import load_production_provider

        models_lock = workspace_root / "models" / "models.lock.json"
        provider = load_production_provider(models_lock)

    config_path = workspace_root / "configs" / "pipeline.yaml"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.setdefault("run", {})["submission_mode"] = "degraded"

    output_path = workspace_root / "work" / "ocr-triage-real" / f"{case_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_sink_dir = workspace_root / "work" / "ocr-triage-real" / f"{case_id}-evidence"

    def evidence_sink(stage: str, ir: Any) -> None:
        evidence_sink_dir.mkdir(parents=True, exist_ok=True)
        payload = ir.model_dump(mode="json")
        (evidence_sink_dir / f"{stage}.circuit-ir.json").write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1),
            encoding="utf-8",
        )

    t0 = time.time()
    try:
        result = infer_one(
            image_path,
            output_path,
            provider=provider,
            config=config,
            official_profile=workspace_root / "configs" / "official_schema.yaml",
            evidence_sink=evidence_sink,
        )
    except Exception as exc:
        elapsed = time.time() - t0
        ocr_provider_for_trace = getattr(provider, "ocr_provider", None)
        trace = getattr(ocr_provider_for_trace, "_last_detection_trace", None) or []
        detection_report = evaluate_detection_candidates(trace, gt)
        return {
            "case_id": case_id,
            "elapsed_seconds": round(elapsed, 3),
            "gt_box_count": len(_gt_text_targets(gt)),
            "image_only": {"detection_rate": detection_report.get("recall")},
            "detection": detection_report,
            "error": f"{type(exc).__name__}: {exc}",
        }
    elapsed = time.time() - t0

    ir = result.ir
    text_box_report = evaluate_text_boxes(ir, gt)

    ctx_trace = []
    ocr_provider = getattr(provider, "ocr_provider", None)
    if ocr_provider is not None:
        ctx_trace = getattr(ocr_provider, "_last_detection_trace", None) or []
    detection_report = evaluate_detection_candidates(ctx_trace, gt)

    if ocr_bundle is None:
        try:
            from pcbparse.training.torch_ocr_model import load_ocr_bundle

            models_lock_path = workspace_root / "models" / "models.lock.json"
            lock = json.loads(models_lock_path.read_text(encoding="utf-8"))
            ocr_entry = next(e for e in lock["models"] if e["task"] == "ocr")
            model_path = workspace_root / ocr_entry["path"]
            ocr_bundle = load_ocr_bundle(model_path, device="cpu")
        except Exception as exc:
            ocr_bundle = None
            ocr_bundle_error = f"{type(exc).__name__}: {exc}"
        else:
            ocr_bundle_error = None
    else:
        ocr_bundle_error = None

    if ocr_bundle is not None:
        recognize_fn = _make_recognize_fn(ocr_bundle, image_path)
        rec_report = recognition_oracle(gt, recognize_fn)
        classify_fn = _make_classify_fn()
        bind_fn = _make_bind_fn(gt)
        bind_report = binding_oracle(gt, classify_fn, bind_fn)
    else:
        rec_report = {"kind": "recognition_oracle", "status": "not_measured", "reason": ocr_bundle_error}
        bind_report = {"kind": "binding_oracle", "status": "not_measured", "reason": ocr_bundle_error}

    image_only_summary = {
        "detection_rate": detection_report.get("recall"),
        "detection_precision": detection_report.get("precision"),
        "text_box_report_kind": text_box_report.get("kind"),
    }

    return {
        "case_id": case_id,
        "elapsed_seconds": round(elapsed, 3),
        "gt_box_count": text_box_report.get("gt_box_count"),
        "image_only": image_only_summary,
        "detection": detection_report,
        "text_box_eval": text_box_report,
        "recognition_oracle": rec_report,
        "binding_oracle": bind_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR-02 real three-way diagnostic")
    parser.add_argument("--workspace-root", type=Path, default=WORKSPACE)
    parser.add_argument(
        "--dataset-root", type=Path, default=None,
        help="Dataset root containing numbered sample folders (default: workspace/Dataset300/Dataset300)",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    workspace_root = args.workspace_root.resolve()
    dataset_root = (args.dataset_root or workspace_root / "Dataset300" / "Dataset300").resolve()
    output_dir = (args.output or workspace_root / "work" / "ocr-triage-real").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    shared_provider = None
    shared_ocr_bundle = None
    try:
        from pcbparse.perception.production_provider import load_production_provider

        models_lock = workspace_root / "models" / "models.lock.json"
        shared_provider = load_production_provider(models_lock)
    except Exception as exc:
        print(f"provider load failed: {type(exc).__name__}: {exc}")
    try:
        from pcbparse.training.torch_ocr_model import load_ocr_bundle

        lock = json.loads((workspace_root / "models" / "models.lock.json").read_text(encoding="utf-8"))
        ocr_entry = next(e for e in lock["models"] if e["task"] == "ocr")
        shared_ocr_bundle = load_ocr_bundle(workspace_root / ocr_entry["path"], device="cpu")
    except Exception as exc:
        print(f"ocr bundle load failed: {type(exc).__name__}: {exc}")

    for case_id in DEV_SAMPLES:
        try:
            case = run_case(
                dataset_root=dataset_root,
                case_id=case_id,
                workspace_root=workspace_root,
                provider=shared_provider,
                ocr_bundle=shared_ocr_bundle,
            )
            cases.append(case)
            rec = case.get("recognition_oracle", {})
            det = case.get("detection", {})
            bind = case.get("binding_oracle", {})
            print(
                f"[{case_id}] det_recall={det.get('recall')} "
                f"rec_exact={rec.get('exact_rate')} rec_cer={rec.get('mean_cer')} "
                f"bind_field={bind.get('field_classified_rate')} "
                f"bind_owner={bind.get('owner_bound_free_rate')}"
            )
        except Exception as exc:
            print(f"[{case_id}] FAILED: {type(exc).__name__}: {exc}")
            cases.append({"case_id": case_id, "error": f"{type(exc).__name__}: {exc}"})

    measured = [c for c in cases if "error" not in c]
    agg_det_recall = _mean([c["detection"].get("recall") for c in measured])
    agg_rec_exact = _mean([c["recognition_oracle"].get("exact_rate") for c in measured if isinstance(c.get("recognition_oracle"), dict)])
    agg_rec_cer = _mean([c["recognition_oracle"].get("mean_cer") for c in measured if isinstance(c.get("recognition_oracle"), dict)])
    agg_bind_field = _mean([c["binding_oracle"].get("field_classified_rate") for c in measured if isinstance(c.get("binding_oracle"), dict)])
    agg_bind_owner = _mean([c["binding_oracle"].get("owner_bound_free_rate") for c in measured if isinstance(c.get("binding_oracle"), dict)])
    agg_bind_owner_trusted = _mean([c["binding_oracle"].get("owner_bound_trusted_rate") for c in measured if isinstance(c.get("binding_oracle"), dict)])

    triage = triage_priority(
        image_only_report={"detection_rate": agg_det_recall},
        recognition_report={"exact_rate": agg_rec_exact, "mean_cer": agg_rec_cer},
        binding_report={
            "field_classified_rate": agg_bind_field,
            "owner_bound_free_rate": agg_bind_owner,
            "owner_bound_trusted_rate": agg_bind_owner_trusted,
        },
    )

    report = {
        "kind": "ocr_triage_real",
        "task_id": "OCR-02",
        "samples": list(DEV_SAMPLES),
        "cases": cases,
        "aggregate": {
            "detection_recall": agg_det_recall,
            "recognition_exact_rate": agg_rec_exact,
            "recognition_mean_cer": agg_rec_cer,
            "binding_field_rate": agg_bind_field,
            "binding_owner_free_rate": agg_bind_owner,
            "binding_owner_trusted_rate": agg_bind_owner_trusted,
        },
        "triage": triage,
    }
    report_path = output_dir / "triage_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nreport: {report_path}")
    print(f"triage priority: {triage['priority']}")
    for finding in triage["findings"]:
        print(f"  - {finding}")
    return 0


def _mean(values: list[Any]) -> float | None:
    numbers = [v for v in values if isinstance(v, (int, float))]
    return sum(numbers) / len(numbers) if numbers else None


if __name__ == "__main__":
    raise SystemExit(main())
