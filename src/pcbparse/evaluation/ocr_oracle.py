"""OCR-02 three-way oracle diagnostics: detection vs recognition vs binding.

Three isolated experiments on fixed development samples:

1. **image-only** baseline — the normal pipeline output.
2. **recognition oracle** — GT text boxes cropped and sent to the recognizer
   only (no detector).  Measures raw recognition quality (exact / CER).
3. **binding oracle** — GT text + GT box fed to semantic classification and
   owner binding.  Measures classification and binding accuracy under both
   predicted-owner and trusted-owner conditions.

Oracle paths never enter production inference, official metrics or promotion
evidence.  Objects without trusted GT text boxes report ``not_measured``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pcbparse.evaluation.raw_ground_truth import RawGroundTruth
from pcbparse.evaluation.text_box_eval import character_error_rate
from pcbparse.ir.models import CircuitIR, TextToken


def _gt_text_records(gt: RawGroundTruth) -> list[dict[str, Any]]:
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
                        "text_id": target.text_id,
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
                        "text_id": target.text_id,
                    }
                )
    return records


def recognition_oracle(
    gt: RawGroundTruth,
    recognize_fn: Any,
) -> dict[str, Any]:
    """Feed GT text-box crops to the recognizer and measure exact / CER.

    ``recognize_fn(crop_image) -> str | None`` must be a pure recognizer
    (no detector).  This isolates recognition quality from detection quality.
    """
    records = _gt_text_records(gt)
    excluded: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for target in records:
        bbox = target["bbox"]
        try:
            recognized = recognize_fn(bbox)
        except Exception as exc:
            results.append(
                {
                    "object_id": target["object_id"],
                    "gt_value": target["gt_value"],
                    "recognized": None,
                    "exact": False,
                    "cer": None,
                    "status": "failed",
                    "reason": f"recognizer_exception:{type(exc).__name__}",
                }
            )
            continue
        if recognized is None or recognized == "":
            cer = character_error_rate(target["gt_value"], "")
            results.append(
                {
                    "object_id": target["object_id"],
                    "gt_value": target["gt_value"],
                    "recognized": recognized,
                    "exact": False,
                    "cer": cer,
                    "status": "failed",
                    "reason": "empty_recognition_as_deletion",
                }
            )
            continue
        exact = recognized == target["gt_value"]
        cer = character_error_rate(target["gt_value"], recognized)
        results.append(
            {
                "object_id": target["object_id"],
                "gt_value": target["gt_value"],
                "recognized": recognized,
                "exact": exact,
                "cer": cer,
                "status": "passed" if exact else "failed",
                "reason": None if exact else "text_mismatch",
            }
        )
    for component in gt.components:
        for field, target in (
            ("designator", component.designator),
            ("value", component.value),
            ("search_name", component.search_name),
        ):
            if target.mask.eligible and target.value is not None and target.bbox is None:
                excluded.append(
                    {"object_id": f"{component.raw_id}.{field}", "reason": "no_trusted_gt_text_bbox"}
                )
    exact_count = sum(1 for item in results if item["exact"])
    cer_values = [item["cer"] for item in results if item["cer"] is not None]
    exception_count = sum(
        1 for item in results if str(item.get("reason") or "").startswith("recognizer_exception")
    )
    measured = len(results)
    total_gt_chars = sum(len(item["gt_value"]) for item in results)
    total_edits = sum(
        item["cer"] * len(item["gt_value"]) for item in results if item["cer"] is not None
    )
    corpus_cer = total_edits / total_gt_chars if total_gt_chars else None
    return {
        "kind": "recognition_oracle",
        "task_id": "OCR-02",
        "case_id": gt.case_id,
        "denominator": measured,
        "denominator_excluded": excluded,
        "exact_count": exact_count,
        "exact_rate": exact_count / measured if measured else None,
        "micro_exact_rate": exact_count / measured if measured else None,
        "mean_cer": sum(cer_values) / len(cer_values) if cer_values else None,
        "corpus_cer": corpus_cer,
        "total_gt_chars": total_gt_chars,
        "total_edits": total_edits,
        "cer_count": len(cer_values),
        "exception_count": exception_count,
        "objects": results,
        "note": (
            "GT text-box crops sent to recognizer only; no detector involved. "
            "Empty/None recognition is scored as full deletion (CER against empty string). "
            "Exceptions are listed separately and excluded from CER. "
            "corpus_cer = total_edits / total_gt_chars (corpus-level, not mean of per-item CER)."
        ),
    }


def binding_oracle(
    gt: RawGroundTruth,
    classify_fn: Any,
    bind_fn: Any,
    *,
    geometry_source: str = "predicted",
) -> dict[str, Any]:
    """Feed GT text + box to classification and owner binding via real association.

    ``classify_fn(text, bbox) -> str`` returns the predicted field category.
    ``bind_fn(text, bbox, predicted_owner_hint) -> str | None`` returns the
    predicted owner ID.  ``predicted_owner_hint`` is ``None`` for the free
    condition and the GT owner ID for the trusted condition.  When the trusted
    condition is used, ``owner_source`` is marked ``oracle_replaced`` and owner
    accuracy is not evaluated for that condition.

    ``geometry_source`` labels whether the surrounding geometry is ``predicted``
    or ``gt`` (for controlled experiments).
    """
    records = _gt_text_records(gt)
    field_aliases: dict[str, set[str]] = {
        "designator": {"designator"},
        "value": {"value"},
        "search_name": {"searchname", "search_name", "name"},
        "number": {"pinnumber", "pin_number", "number"},
        "name": {"pinname", "pin_name", "name"},
    }
    results: list[dict[str, Any]] = []
    for target in records:
        text = target["gt_value"]
        bbox = target["bbox"]
        predicted_field = classify_fn(text, bbox)
        expected_field = target["field"]
        field_ok = predicted_field.casefold() in field_aliases.get(expected_field, {expected_field})
        predicted_owner_free = bind_fn(text, bbox, None)
        predicted_owner_trusted = bind_fn(text, bbox, target["owner_id"])
        results.append(
            {
                "object_id": target["object_id"],
                "gt_value": text,
                "expected_field": expected_field,
                "predicted_field": predicted_field or "",
                "field_classified": field_ok,
                "expected_owner": target["owner_id"],
                "predicted_owner_free": predicted_owner_free,
                "predicted_owner_trusted": predicted_owner_trusted,
                "owner_bound_free": predicted_owner_free == target["owner_id"],
                "owner_source_trusted": "oracle_replaced",
            }
        )
    measured = len(results)
    free_correct = sum(1 for item in results if item["owner_bound_free"])
    field_correct = sum(1 for item in results if item["field_classified"])
    return {
        "kind": "binding_oracle",
        "task_id": "OCR-02",
        "case_id": gt.case_id,
        "geometry_source": geometry_source,
        "denominator": measured,
        "field_classified_count": field_correct,
        "field_classified_rate": field_correct / measured if measured else None,
        "owner_bound_free_count": free_correct,
        "owner_bound_free_rate": free_correct / measured if measured else None,
        "owner_bound_trusted_count": None,
        "owner_bound_trusted_rate": None,
        "owner_source_trusted": "oracle_replaced",
        "objects": results,
        "note": (
            "GT text + box fed to real classifier/binder.  Trusted condition replaces "
            "owner via oracle and does not evaluate owner accuracy.  Free condition "
            "evaluates owner inference with no hint."
        ),
    }


def aggregate_recognition(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-case recognition reports with macro and micro averages."""
    measured = [r for r in reports if r.get("denominator")]
    if not measured:
        return {"macro_exact": None, "micro_exact": None, "macro_mean_cer": None, "corpus_cer": None}
    macro_exact = sum(r["exact_rate"] for r in measured if r.get("exact_rate") is not None) / len(measured)
    total_texts = sum(r["denominator"] for r in measured)
    total_exact = sum(r.get("exact_count", 0) for r in measured)
    micro_exact = total_exact / total_texts if total_texts else None
    cer_values = [r["mean_cer"] for r in measured if r.get("mean_cer") is not None]
    macro_mean_cer = sum(cer_values) / len(cer_values) if cer_values else None
    total_edits = sum(r.get("total_edits", 0) for r in measured)
    total_chars = sum(r.get("total_gt_chars", 0) for r in measured)
    corpus_cer = total_edits / total_chars if total_chars else None
    return {
        "macro_exact": macro_exact,
        "micro_exact": micro_exact,
        "total_texts": total_texts,
        "total_exact": total_exact,
        "macro_mean_cer": macro_mean_cer,
        "corpus_cer": corpus_cer,
        "total_edits": total_edits,
        "total_gt_chars": total_chars,
        "per_case": [
            {"case_id": r.get("case_id"), "exact_rate": r.get("exact_rate"), "mean_cer": r.get("mean_cer"), "denominator": r.get("denominator")}
            for r in measured
        ],
    }


def triage_priority(
    *,
    image_only_report: Mapping[str, Any] | None,
    recognition_report: Mapping[str, Any] | None,
    binding_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compare the three oracle paths and recommend the optimization priority.

    Logic: if recognition oracle exact_rate is low, recognition is the primary
    bottleneck.  If recognition is high but binding oracle field/owner rates are
    low, binding is the bottleneck.  If both oracles are high but image-only
    detection is low, detection is the bottleneck.
    """
    findings: list[str] = []
    priority: list[str] = []

    rec_exact = recognition_report.get("exact_rate") if recognition_report else None
    rec_cer = recognition_report.get("mean_cer") if recognition_report else None
    bind_field = binding_report.get("field_classified_rate") if binding_report else None
    bind_owner_free = binding_report.get("owner_bound_free_rate") if binding_report else None
    bind_owner_trusted = binding_report.get("owner_bound_trusted_rate") if binding_report else None

    if rec_exact is not None:
        if rec_exact < 0.6:
            findings.append(f"recognition oracle exact_rate={rec_exact:.3f} is low")
            priority.append("recognition")
        elif rec_exact < 0.85:
            findings.append(f"recognition oracle exact_rate={rec_exact:.3f} is moderate")
            priority.append("recognition")
        else:
            findings.append(f"recognition oracle exact_rate={rec_exact:.3f} is adequate")
    if rec_cer is not None:
        findings.append(f"recognition oracle mean_cer={rec_cer:.3f}")

    if bind_field is not None:
        if bind_field < 0.7:
            findings.append(f"binding oracle field_classified_rate={bind_field:.3f} is low")
            priority.append("binding_field")
        else:
            findings.append(f"binding oracle field_classified_rate={bind_field:.3f} is adequate")
    if bind_owner_free is not None and bind_owner_trusted is not None:
        gap = bind_owner_trusted - bind_owner_free
        if bind_owner_free < 0.7:
            findings.append(
                f"binding oracle owner_bound_free_rate={bind_owner_free:.3f} is low"
            )
            priority.append("binding_owner")
        if gap > 0.15:
            findings.append(
                f"owner binding gap (trusted={bind_owner_trusted:.3f} vs free={bind_owner_free:.3f}) "
                "suggests owner inference is the binding bottleneck"
            )
            priority.append("binding_owner")

    if image_only_report is not None:
        det = image_only_report.get("detection_rate")
        if det is not None and det < 0.5 and "recognition" not in priority:
            findings.append(f"image-only detection_rate={det:.3f} is low")
            priority.append("detection")
        elif det is None:
            findings.append("detection_rate is not_measured; detection priority cannot be assessed")

    if not priority:
        if rec_exact is None and bind_field is None:
            priority.append("not_measured")
        else:
            priority.append("detection")

    seen: set[str] = set()
    ordered: list[str] = []
    for item in priority:
        if item not in seen:
            seen.add(item)
            ordered.append(item)

    return {
        "kind": "ocr_triage",
        "task_id": "OCR-02",
        "priority": ordered,
        "findings": findings,
        "recognition_exact_rate": rec_exact,
        "recognition_mean_cer": rec_cer,
        "binding_field_rate": bind_field,
        "binding_owner_free_rate": bind_owner_free,
        "binding_owner_trusted_rate": bind_owner_trusted,
    }
