"""Read-only probes for the 2026-09-08 architecture review.

Run with .venv-test/Scripts/python.exe. Writes JSON to stdout only.
These probes describe observed behavior; they are not acceptance tests.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from pcbparse.association import associate_ocr
from pcbparse.graph.physical import build_physical_graph
from pcbparse.ir import CircuitIR, Component, ImageInfo, PhysicalNet, Point, TextToken, WireSegment
from pcbparse.validation.graph import validate_graph


def read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def read_rows(relative: str):
    with (ROOT / relative).open(encoding="utf-8-sig") as source:
        return [json.loads(line) for line in source if line.strip()]


def empty_ir(name: str) -> CircuitIR:
    return CircuitIR(image=ImageInfo(image_id=name, path="synthetic.png", width=300, height=100))


def connectivity_probes():
    output = []
    cases = [
        ("collinear_overlap", [(0, 20), (10, 30)], {"collinear_merge_tolerance_px": .75}),
        ("collinear_gap_5px", [(0, 10), (15, 25)], {"collinear_gap_merge_tolerance_px": 6}),
    ]
    for name, endpoints, options in cases:
        ir = empty_ir(name)
        ir.segments = [WireSegment(internal_id=f"s{i}", start=Point(x=a, y=10), end=Point(x=b, y=10), channel="horizontal") for i, (a, b) in enumerate(endpoints)]
        ir.physical_nets, _ = build_physical_graph(ir, endpoint_tolerance_px=1, **options)
        # Test with matching tolerances (the fix) and without (legacy behavior)
        validator_codes_fixed = [d.code for d in validate_graph(ir, **options)]
        validator_codes_legacy = [d.code for d in validate_graph(ir)]
        output.append({
            "case": name,
            "builder_net_count": len(ir.physical_nets),
            "validator_codes_with_matching_tolerances": validator_codes_fixed,
            "validator_codes_legacy_no_tolerance": validator_codes_legacy,
        })
    ir = empty_ir("distant_reused_polyline")
    ir.segments = [WireSegment(internal_id=f"s{i}", start=Point(x=a, y=10), end=Point(x=b, y=10), channel="horizontal", source_polyline_id="reused") for i, (a, b) in enumerate([(0, 10), (50, 60)])]
    built, _ = build_physical_graph(ir)
    ir.physical_nets = [PhysicalNet(internal_id="invalid_net", pins=[], segments=["s0", "s1"])]
    output.append({"case": "distant_reused_polyline", "builder_net_count": len(built), "validator_codes_for_invalid_net": [d.code for d in validate_graph(ir)]})
    return output


def naming_probes():
    output = []
    for name, category, values in [("R1", "R", ["R1"]), ("RUN", "vbus_in", ["RUN"]), ("duplicate_R1", "R", ["R1", "R1"])]:
        ir = empty_ir(name)
        for i, value in enumerate(values):
            cid, x = f"c{i}", 20 + i * 120
            ir.components.append(Component(internal_id=cid, category=category, bbox={"x1": x, "y1": 20, "x2": x + 20, "y2": 40}))
            ir.texts.append(TextToken(internal_id=f"t{i}", raw_text=value, details="designator", component_id=cid, bbox={"x1": x, "y1": 10, "x2": x + 20, "y2": 18}, score=.99))
        associate_ocr(ir, {"association": {"component_min_score": .3, "component_unique_margin": 0.0}})
        output.append({"case": name, "assigned_keys": [c.key for c in ir.components], "diagnostic_codes": [d.code for d in ir.diagnostics]})
    return output


def summarize_strict_run(relative: str):
    summary = read_json(relative + "/summary.json")
    rows = read_rows(relative + "/predictions.jsonl")
    diagnostics, stages = collections.Counter(), collections.Counter()
    for row in rows:
        for item in (row.get("error") or {}).get("diagnostics", []):
            diagnostics[(item["severity"], item["code"])] += 1
        for item in row.get("runtime_evidence", {}).get("trace", []):
            stages[item["stage"]] += item["elapsed_seconds"]
    return {
        "source": relative,
        **{key: summary.get(key) for key in ("status", "succeeded", "failed", "elapsed_seconds", "config_hash", "model_lock_sha256", "inference_run_id", "started_at_utc", "timing")},
        "diagnostics": [{"severity": severity, "code": code, "count": count} for (severity, code), count in diagnostics.most_common()],
        "stage_seconds": {key: round(value, 6) for key, value in stages.items()},
        "failed_rows_without_ir": sum(row.get("status") == "error" and not row.get("ir_path") for row in rows),
    }


def main():
    lock = read_json("models/models.lock.json")
    original = {row["sample_id"]: row["split"] for row in read_rows("data/splits/dataset300_split.jsonl")}
    release = read_rows("data/splits/dataset300_release50_v1.jsonl")
    report = read_json("work/release50-component-assoc-margin0-pinmargin16/reports.json")
    groups = collections.defaultdict(list)
    for row in report["cases"]:
        groups[row["source_domain"]].append(row)
    domains = {}
    for domain, rows in groups.items():
        metrics = {"case_count": len(rows)}
        for task, kind in [("component", "detection"), ("pin", "detection"), ("graph", "pairwise_connectivity")]:
            counts = {key: sum(row[task][kind][key] for row in rows) for key in ("tp", "fp", "fn")}
            tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
            metrics[task] = {**counts, "f1": 2 * tp / max(1, 2 * tp + fp + fn), "recall": tp / max(1, tp + fn)}
        domains[domain] = metrics
    observed_files = [
        "models/models.lock.json", "configs/pipeline.yaml", "configs/pipeline.release-candidate.yaml",
        "src/pcbparse/graph/physical.py", "src/pcbparse/validation/graph.py",
        "src/pcbparse/association/text_component.py", "src/pcbparse/perception/torch_pin_adapter.py",
        "src/pcbparse/output/official_serializer.py", "src/pcbparse/evaluation/official_metrics.py",
        "work/release50-component-assoc-margin0-pinmargin16/reports.json",
        "work/production-e2e-componentmargin0-pinmargin16/summary.json",
        "work/production-e2e-pin-point-projection2/summary.json",
        "artifacts/reference/2026_EDA_Challenge_Task6_AI_PCB_Schematic_Official_Guide.pdf",
    ]
    output = {
        "kind": "architecture_review_read_only_probe", "review_date": "2026-09-08",
        "limitations": ["Historical run metrics are not new inference results.", "Contract probes are synthetic diagnostics and are not accuracy measurements.", "raw-v1 is not the official competition scorer.", "No full release rerun or container validation was performed."],
        "file_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in observed_files},
        "model_entries": [{"task": e["task"], "path": e["path"], "sha256": e["sha256"], "blessed": e.get("blessed"), "inference": e.get("inference"), "evaluation": e.get("evaluation")} for e in lock["models"]],
        "release50_origin_splits": dict(collections.Counter(original[row["sample_id"]] for row in release)),
        "raw_baseline_source": "work/release50-component-assoc-margin0-pinmargin16/reports.json",
        "raw_baseline_aggregate": report["aggregate"], "raw_baseline_domain_micro": domains,
        "strict_runs": [summarize_strict_run(name) for name in ["work/production-e2e-componentmargin0-pinmargin16", "work/production-e2e-pin-point-projection2"]],
        "connectivity_probes": connectivity_probes(), "naming_probes": naming_probes(),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
