"""Scan physical-graph parameters on already materialized candidate IRs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from pcbparse.evaluation.masked_metrics import evaluate_masked_case
from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth
from pcbparse.graph.physical import build_physical_graph
from pcbparse.ir.models import CircuitIR
from pcbparse.nets.hypergraph import build_electrical_nets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--endpoint", type=float, default=3.0)
    parser.add_argument("--pin-max", type=float, default=6.0)
    parser.add_argument("--pin-min", type=float, default=2.0)
    parser.add_argument("--unique-margin", type=float, default=1.5)
    parser.add_argument("--gap", type=float, default=6.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports: list[dict[str, object]] = []
    for ir_path in sorted(args.ir_dir.glob("*.circuit-ir.json")):
        case_id = ir_path.name.split(".", 1)[0]
        annotation_path = next(args.dataset_root.joinpath(case_id).glob("*_CVJsonStd.json"))
        ir = CircuitIR.model_validate_json(ir_path.read_text(encoding="utf-8"))
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        ground_truth = build_raw_ground_truth(
            annotation,
            case_id=case_id,
            annotation_path=annotation_path,
        )
        ir.physical_nets, _ = build_physical_graph(
            ir,
            endpoint_tolerance_px=args.endpoint,
            collinear_merge_tolerance_px=0.75,
            collinear_gap_merge_tolerance_px=args.gap,
            pin_snap_min_px=args.pin_min,
            pin_snap_tolerance_px=args.pin_max,
            unique_margin=args.unique_margin,
        )
        ir.electrical_nets = build_electrical_nets(ir)
        report = evaluate_masked_case(ir, ground_truth)
        report["case_id"] = case_id
        reports.append(report)

    graph = _aggregate_graph(reports)
    domains: dict[str, list[dict[str, object]]] = defaultdict(list)
    for report in reports:
        domains[str(report["source_domain"])].append(report)
    result = {
        "cases": len(reports),
        "parameters": {
            "endpoint": args.endpoint,
            "pin_min": args.pin_min,
            "pin_max": args.pin_max,
            "unique_margin": args.unique_margin,
            "gap": args.gap,
        },
        "graph": graph,
        "domains": {
            domain: _aggregate_graph(items) for domain, items in sorted(domains.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def _aggregate_graph(reports: list[dict[str, object]]) -> dict[str, float | int]:
    tp = sum(int(report["graph"]["pairwise_connectivity"]["tp"]) for report in reports)
    fp = sum(int(report["graph"]["pairwise_connectivity"]["fp"]) for report in reports)
    fn = sum(int(report["graph"]["pairwise_connectivity"]["fn"]) for report in reports)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if __name__ == "__main__":
    raise SystemExit(main())
