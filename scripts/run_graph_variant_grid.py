"""Evaluate deterministic graph variants on an existing image-only IR suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pcbparse.evaluation.masked_metrics import aggregate_masked_reports, evaluate_masked_case
from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth
from pcbparse.graph.physical import build_physical_graph
from pcbparse.ir.models import CircuitIR
from pcbparse.nets.hypergraph import build_electrical_nets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", action="append", help="evaluate only named variants")
    args = parser.parse_args()

    variants = _variants()
    if args.only:
        wanted = set(args.only)
        variants = [variant for variant in variants if variant["name"] in wanted]
    reports: list[dict[str, Any]] = []
    ir_paths = sorted(args.ir_dir.glob("*.circuit-ir.json"), key=lambda path: _natural_key(path.stem))
    if not ir_paths:
        raise FileNotFoundError(f"no Circuit-IR files found under {args.ir_dir}")

    for variant in variants:
        case_reports: list[dict[str, Any]] = []
        for ir_path in ir_paths:
            ir = CircuitIR.model_validate_json(ir_path.read_text(encoding="utf-8"))
            junctions = ir.junctions
            if not variant["junction_union"]:
                ir.junctions = []
            ir.physical_nets, _decisions = build_physical_graph(
                ir,
                endpoint_tolerance_px=variant["endpoint"],
                collinear_merge_tolerance_px=variant["collinear"],
                collinear_gap_merge_tolerance_px=variant["gap"],
                pin_snap_min_px=2.0,
                pin_snap_tolerance_px=variant["snap"],
                unique_margin=variant["unique_margin"],
            )
            ir.junctions = junctions
            ir.electrical_nets = build_electrical_nets(
                ir,
                merge_signal_labels=True,
                merge_power_labels=False,
                merge_ground_labels=False,
                min_text_score=0.75,
                merge_threshold=0.85,
            )
            case_id = ir_path.name.split(".", 1)[0]
            annotation_path = args.dataset_root / case_id
            annotations = sorted(annotation_path.glob("*_CVJsonStd.json"))
            if len(annotations) != 1:
                raise ValueError(f"{case_id}: expected one annotation, found {len(annotations)}")
            ground_truth = build_raw_ground_truth(
                json.loads(annotations[0].read_text(encoding="utf-8")),
                case_id=case_id,
                annotation_path=annotations[0],
            )
            case_reports.append(evaluate_masked_case(ir, ground_truth))

        aggregate = aggregate_masked_reports(case_reports)
        component = _f1(aggregate["component"]["detection"])
        pin = _f1(aggregate["pin"]["detection"])
        graph = _f1(aggregate["graph"]["pairwise_connectivity"])
        wire = float(aggregate["wire"]["f1"])
        reports.append(
            {
                "name": variant["name"],
                "parameters": variant,
                "cases": len(case_reports),
                "component_f1": component,
                "pin_f1": pin,
                "graph_f1": graph,
                "wire_f1": wire,
                "weighted_proxy": 0.30 * component + 0.25 * pin + 0.35 * graph + 0.10 * wire,
                "aggregate": aggregate,
            }
        )
        best = reports[-1]
        print(
            f"{best['name']}: weighted={best['weighted_proxy']:.6f} "
            f"graph={best['graph_f1']:.6f} pin={best['pin_f1']:.6f}",
            flush=True,
        )

    reports.sort(key=lambda item: (-float(item["weighted_proxy"]), -float(item["graph_f1"]), str(item["name"])))
    payload = {
        "suite": str(args.ir_dir.resolve()),
        "cases": len(ir_paths),
        "variants": reports,
        "ranking_policy": "weighted_proxy_then_graph_f1",
        "limitations": [
            "The model outputs and Pin/OCR/component associations are held fixed; only graph reconstruction is varied.",
            "weighted_proxy uses the local raw-v1 component/pin/graph/wire metrics and is not the official evaluator.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _variants() -> list[dict[str, Any]]:
    base = {
        "endpoint": 1.0,
        "collinear": 0.75,
        "gap": 6.0,
        "snap": 10.0,
        "unique_margin": 1.5,
        "junction_union": True,
    }
    variants: list[dict[str, Any]] = [{"name": "baseline", **base}]
    for endpoint in (0.5, 0.75, 1.25, 1.5, 2.0):
        variants.append({"name": f"endpoint{endpoint:g}", **{**base, "endpoint": endpoint}})
    variants.append({"name": "junctions-off", **{**base, "junction_union": False}})
    variants.append({
        "name": "endpoint1.5-margin0.5",
        **{**base, "endpoint": 1.5, "unique_margin": 0.5},
    })
    for unique_margin in (0.5, 2.0, 3.0, 4.0):
        variants.append(
            {
                "name": f"margin{unique_margin:g}",
                **{**base, "unique_margin": unique_margin},
            }
        )
    for collinear, gap, name in (
        (0.0, 6.0, "collinear0-gap6"),
        (0.75, 0.0, "collinear075-gap0"),
        (0.75, 2.0, "collinear075-gap2"),
        (0.75, 4.0, "collinear075-gap4"),
        (0.75, 8.0, "collinear075-gap8"),
        (1.0, 6.0, "collinear1-gap6"),
    ):
        variants.append({"name": name, **{**base, "collinear": collinear, "gap": gap}})
    for snap in (4.0, 6.0, 8.0, 12.0):
        variants.append({"name": f"snap{snap:g}", **{**base, "snap": snap}})
    return variants


def _f1(metric: dict[str, Any]) -> float:
    return float(metric["f1"])


def _natural_key(value: str) -> tuple[tuple[int, object], ...]:
    import re

    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", value)
        if part
    )


if __name__ == "__main__":
    raise SystemExit(main())
