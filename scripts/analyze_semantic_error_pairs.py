"""Summarize image-only semantic-field errors without changing predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pcbparse.evaluation.masked_metrics import _component_metrics, _normal_text, _pin_metrics
from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth
from pcbparse.ir.models import CircuitIR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    field_pairs: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    field_missing: Counter[str] = Counter()
    domain_totals: dict[str, Counter[str]] = defaultdict(Counter)
    semantic_categories: Counter[str] = Counter()

    for ir_path in sorted(args.ir_dir.glob("*.circuit-ir.json"), key=lambda path: _natural_key(path.name)):
        case_id = ir_path.name.split(".", 1)[0]
        ir = CircuitIR.model_validate_json(ir_path.read_text(encoding="utf-8"))
        annotation_dir = args.dataset_root / case_id
        annotation_paths = sorted(annotation_dir.glob("*_CVJsonStd.json"))
        if len(annotation_paths) != 1:
            raise ValueError(f"{case_id}: expected one annotation, found {len(annotation_paths)}")
        gt = build_raw_ground_truth(
            json.loads(annotation_paths[0].read_text(encoding="utf-8")),
            case_id=case_id,
            annotation_path=annotation_paths[0],
        )
        component_metric, component_map = _component_metrics(ir, gt, threshold=0.5)
        _pin_metric, pin_map = _pin_metrics(ir, gt, component_map=component_map, tolerance=5.0)
        domain = str(gt.source_domain)
        for token in ir.texts:
            if token.details:
                semantic_categories[str(token.details)] += 1

        components_by_raw = {item.raw_id: item for item in gt.components}
        for component in ir.components:
            raw_id = component_map.get(component.internal_id)
            target = components_by_raw.get(raw_id) if raw_id else None
            if target is None:
                continue
            for field, prediction, expected in (
                ("component.designator", component.key, _target_value(target.designator)),
                ("component.value", component.value, _target_value(target.value)),
                ("component.search_name", component.name, _target_value(target.search_name)),
            ):
                _record(field, prediction, expected, domain, field_pairs, field_missing, domain_totals)

        pins_by_raw = {item.raw_id: item for item in gt.pins}
        for pin in ir.pins:
            raw_id = pin_map.get(pin.internal_id)
            target = pins_by_raw.get(raw_id) if raw_id else None
            if target is None:
                continue
            for field, prediction, expected in (
                ("pin.number", pin.number, _target_value(target.number)),
                ("pin.name", pin.name, _target_value(target.name)),
            ):
                _record(field, prediction, expected, domain, field_pairs, field_missing, domain_totals)

    payload: dict[str, Any] = {
        "cases": len(list(args.ir_dir.glob("*.circuit-ir.json"))),
        "field_pairs": {
            field: [
                {"prediction": prediction, "expected": expected, "count": count}
                for (prediction, expected), count in pairs.most_common(30)
            ]
            for field, pairs in sorted(field_pairs.items())
        },
        "field_missing_or_unmatched": dict(sorted(field_missing.items())),
        "domain_totals": {
            domain: dict(sorted(counts.items())) for domain, counts in sorted(domain_totals.items())
        },
        "semantic_categories": dict(sorted(semantic_categories.items())),
        "limitations": [
            "Pairs are extracted only for geometrically matched eligible objects; they do not prove that a production OCR correction is safe.",
            "Ground truth is used for analysis only and is never injected into Circuit-IR or inference.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _record(
    field: str,
    prediction: str | None,
    expected: str | None,
    domain: str,
    pairs: dict[str, Counter[tuple[str, str]]],
    missing: Counter[str],
    domain_totals: dict[str, Counter[str]],
) -> None:
    if not expected:
        return
    expected_normalized = _normal_text(expected)
    prediction_normalized = _normal_text(prediction or "")
    if not prediction_normalized:
        missing[field] += 1
        domain_totals[domain][f"{field}.missing"] += 1
    elif prediction_normalized == expected_normalized:
        domain_totals[domain][f"{field}.exact"] += 1
    else:
        pairs[field][(str(prediction), str(expected))] += 1
        domain_totals[domain][f"{field}.mismatch"] += 1


def _target_value(target: Any) -> str | None:
    if target is None:
        return None
    if isinstance(target, str):
        return target
    value = getattr(target, "value", None)
    return str(value) if value is not None else None

def _natural_key(value: str) -> tuple[tuple[int, object], ...]:
    import re

    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", value)
        if part
    )


if __name__ == "__main__":
    raise SystemExit(main())
