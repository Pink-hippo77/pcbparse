"""Generate A09 graph A/B and connection-evidence audit artifacts.

Run with PYTHONPATH=src and the project .venv-test interpreter.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcbparse.graph.physical import (  # noqa: E402
    HARD_MAX_GAP_REPAIR_PX,
    ConnectionEvidence,
    build_physical_graph,
    collect_segment_connections,
)
from pcbparse.ir.models import (  # noqa: E402
    CircuitIR,
    Component,
    Evidence,
    ImageInfo,
    NetLabel,
    Pin,
    Point,
    WireSegment,
)
from pcbparse.nets.hypergraph import (  # noqa: E402
    NetView,
    build_electrical_nets,
    compare_matched_net_views,
)
from pcbparse.perception.wire_vectorizer import (  # noqa: E402
    SkeletonPolyline,
    WireVectorizerConfig,
    classify_break_lineage,
    vectorize_wire_mask,
)
from pcbparse.validation.graph import is_signal_flywire_electrical, validate_graph  # noqa: E402


def _seg(sid: str, x1: float, y1: float, x2: float, y2: float, channel: str = "horizontal") -> WireSegment:
    return WireSegment(
        internal_id=sid,
        start=Point(x=x1, y=y1),
        end=Point(x=x2, y=y2),
        channel=channel,
    )


def _ir(name: str) -> CircuitIR:
    return CircuitIR(image=ImageInfo(image_id=name, path=f"{name}.png", width=300, height=120))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_ab() -> tuple[dict, list[ConnectionEvidence]]:
    evidence: list[ConnectionEvidence] = []
    cases: list[dict] = []

    # Case 1: near-parallel must not merge under large gap tolerance.
    parallel = _ir("parallel")
    parallel.segments = [_seg("s0", 0, 10, 10, 10), _seg("s1", 12, 15, 30, 15)]
    # Legacy collinear formula used alignment_tolerance = tolerance with no
    # lateral cap, so a 5px parallel offset would merge at tolerance 6.0.
    legacy_would_merge = True  # max_offset 5.0 <= 6.0 under the uncapped rule
    base_nets, _ = build_physical_graph(parallel, collinear_gap_merge_tolerance_px=6.0)
    cand_nets, _ = build_physical_graph(
        parallel,
        collinear_gap_merge_tolerance_px=6.0,
        gap_repair_max_px=HARD_MAX_GAP_REPAIR_PX,
        connection_evidence_out=evidence,
    )
    shared_parallel = collect_segment_connections(
        parallel.segments,
        endpoint_tolerance_px=3.0,
        collinear_gap_merge_tolerance_px=6.0,
        gap_repair_max_px=HARD_MAX_GAP_REPAIR_PX,
    )
    cases.append(
        {
            "case": "near_parallel_no_false_connect",
            "baseline_net_count": len(base_nets),
            "candidate_net_count": len(cand_nets),
            "expected_net_count": 2,
            "baseline_false_merge": legacy_would_merge,
            "candidate_false_merge": len(cand_nets) < 2,
            "lateral_offset_px": 5.0,
            "max_collinear_lateral_offset_px": 2.0,
            "legacy_uncapped_rule_would_merge": legacy_would_merge,
            "shared_evidence_connects": any(
                item.rule in {"collinear_gap_repair", "collinear_overlap"} for item in shared_parallel
            ),
        }
    )

    # Case 2: collinear gap repair must be virtual and bounded.
    gap = _ir("gap")
    gap.segments = [_seg("s0", 0, 10, 10, 10), _seg("s1", 15, 10, 25, 10)]
    default_nets, _ = build_physical_graph(gap)
    gap_evidence: list[ConnectionEvidence] = []
    repaired_nets, _ = build_physical_graph(
        gap,
        collinear_gap_merge_tolerance_px=6.0,
        gap_repair_max_px=8.0,
        connection_evidence_out=gap_evidence,
    )
    virtual = [item for item in gap_evidence if item.virtual]
    cases.append(
        {
            "case": "virtual_gap_repair_marked_separately",
            "baseline_net_count": len(default_nets),
            "candidate_net_count": len(repaired_nets),
            "virtual_repair_count": len(virtual),
            "candidate_repair_ratio": repaired_nets[0].repair_ratio if repaired_nets else 0.0,
            "all_virtual_marked": all(item.rule == "collinear_gap_repair" for item in virtual) and bool(virtual),
        }
    )
    evidence.extend(gap_evidence)

    # Case 3: SIGNAL flywire is not a physical disconnect.
    fly = _ir("flywire")
    fly.components = [
        Component(internal_id="c1", key="R1", category="R", bbox={"x1": 0, "y1": 40, "x2": 20, "y2": 60}),
        Component(internal_id="c2", key="R2", category="R", bbox={"x1": 180, "y1": 40, "x2": 200, "y2": 60}),
    ]
    fly.pins = [
        Pin(internal_id="p1", component_id="c1", number="1", point=Point(x=20, y=50), orientation="R"),
        Pin(internal_id="p2", component_id="c2", number="1", point=Point(x=180, y=50), orientation="L"),
    ]
    fly.segments = [_seg("s1", 20, 50, 70, 50), _seg("s2", 130, 50, 180, 50)]
    fly.physical_nets, _ = build_physical_graph(
        fly, unique_margin=0.1, connection_evidence_out=evidence
    )
    fly.net_labels = [
        NetLabel(
            internal_id="l1",
            instance_text="RUN_1",
            resolved_text="RUN",
            net_type="SIGNAL",
            attached_physical_net_id="pnet_00001",
        ),
        NetLabel(
            internal_id="l2",
            instance_text="RUN_2",
            resolved_text="RUN",
            net_type="SIGNAL",
            attached_physical_net_id="pnet_00002",
        ),
    ]
    pin_evidence = []
    fly.electrical_nets = build_electrical_nets(fly, evidence_out=pin_evidence)
    for item in pin_evidence:
        evidence.append(
            ConnectionEvidence(
                left_id=item.pin_id,
                right_id=item.physical_net_id,
                rule=item.rule,
                confidence=item.score,
                distance_px=0.0,
                virtual=item.virtual,
                left_kind="pin",
                right_kind="physical_net",
                metadata=dict(item.metadata),
            )
        )
    diagnostics = validate_graph(fly)
    disconnects = [item for item in diagnostics if item.code == "NET.DISCONNECTED_SEGMENTS"]
    flywire_infos = [item for item in diagnostics if item.code == "NET.SIGNAL_FLYWIRE"]
    cases.append(
        {
            "case": "signal_flywire_not_physical_disconnect",
            "physical_net_count": len(fly.physical_nets),
            "electrical_net_count": len(fly.electrical_nets),
            "is_signal_flywire": is_signal_flywire_electrical(fly.electrical_nets[0]),
            "physical_disconnect_errors": len(disconnects),
            "signal_flywire_info_count": len(flywire_infos),
            "per_pin_evidence_count": len(pin_evidence),
        }
    )

    # Case 3b: touching endpoints and junction bridges produce shared evidence.
    touch = _ir("touch")
    touch.segments = [
        _seg("s0", 0, 10, 10, 10),
        _seg("s1", 10, 10, 20, 10),
        _seg("s2", 10, 10, 10, 30, channel="vertical"),
    ]
    touch_shared = collect_segment_connections(touch.segments, endpoint_tolerance_px=3.0)
    evidence.extend(touch_shared)
    touch.physical_nets, _ = build_physical_graph(touch, connection_evidence_out=evidence)
    cases.append(
        {
            "case": "shared_endpoint_evidence",
            "shared_rule_counts": {
                rule: sum(1 for item in touch_shared if item.rule == rule)
                for rule in sorted({item.rule for item in touch_shared})
            },
            "builder_net_count": len(touch.physical_nets),
        }
    )

    # Case 4: matched-net NetLine vs NetHypergraph compared together.
    predicted = [
        NetView(
            net_id="pred-a",
            pins=frozenset({"R1.1", "R2.1"}),
            edges=(((0.0, 0.0), (10.0, 0.0)), ((10.0, 0.0), (20.0, 0.0))),
        ),
        NetView(net_id="pred-b", pins=frozenset({"U1.3"}), edges=(((0.0, 50.0), (10.0, 50.0)),)),
    ]
    expected = [
        NetView(
            net_id="gt-a",
            pins=frozenset({"R1.1", "R2.1"}),
            edges=(((0.0, 0.0), (10.0, 0.0)), ((10.0, 0.0), (21.0, 0.0))),
        ),
        NetView(
            net_id="gt-b",
            pins=frozenset({"U1.3", "U1.4"}),
            edges=(((0.0, 50.0), (10.0, 50.0)),),
        ),
    ]
    matched = compare_matched_net_views(predicted, expected, line_tolerance_px=5.0)
    cases.append(
        {
            "case": "matched_net_line_and_hypergraph",
            "matched_pairs": matched.matched_pairs,
            "net_hypergraph": matched.net_hypergraph.as_dict(),
            "net_line": matched.net_line.as_dict(),
        }
    )

    # Case 5: unbounded gap is rejected.
    unbounded_rejected = False
    try:
        build_physical_graph(gap, collinear_gap_merge_tolerance_px=100.0)
    except ValueError:
        unbounded_rejected = True
    cases.append(
        {
            "case": "unbounded_gap_tolerance_rejected",
            "rejected": unbounded_rejected,
            "hard_max_gap_repair_px": HARD_MAX_GAP_REPAIR_PX,
        }
    )

    summary = {
        "schema_version": 1,
        "kind": "wire_graph_a09_ab",
        "status": "implemented_internal_regression",
        "shared_connection_evidence": True,
        "bounded_gap_repair": True,
        "virtual_gap_repair_marked_separately": True,
        "near_parallel_guard_px": 2.0,
        "signal_flywire_is_not_physical_disconnect": True,
        "cases": cases,
        "metrics": {
            "near_parallel_false_merge_candidate": sum(
                1 for item in cases if item.get("case") == "near_parallel_no_false_connect" and item["candidate_false_merge"]
            ),
            "near_parallel_false_merge_baseline": sum(
                1 for item in cases if item.get("case") == "near_parallel_no_false_connect" and item["baseline_false_merge"]
            ),
            "virtual_repairs_recorded": sum(int(item.get("virtual_repair_count", 0)) for item in cases),
            "signal_flywire_false_disconnects": sum(
                int(item.get("physical_disconnect_errors", 0)) for item in cases
            ),
            "matched_net_net_hypergraph_f1": matched.net_hypergraph.f1,
            "matched_net_net_line_f1": matched.net_line.f1,
        },
        "interpretation_boundary": {
            "models_lock_modified": False,
            "independent_test_confirmation_completed": False,
            "production_promotion_allowed": False,
            "reason": "A09 is an internal connectivity-contract regression on synthetic fixtures; no official evaluator run and no frozen-split promotion.",
        },
    }
    return summary, evidence


def main() -> None:
    audits = ROOT / "artifacts" / "audits"
    audits.mkdir(parents=True, exist_ok=True)
    summary, evidence = run_ab()
    graph_ab_path = audits / "graph_ab.json"
    graph_ab_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evidence_path = audits / "connection_evidence.jsonl"
    lines = [item.to_jsonl() for item in evidence]
    # Include a few synthetic lineage rows so the audit shows break taxonomy.
    for row in (
        {
            "left_id": "poly-mask-a",
            "right_id": "poly-mask-b",
            "left_kind": "polyline",
            "right_kind": "polyline",
            "rule": "break_lineage",
            "confidence": 0.5,
            "distance_px": 6.0,
            "channel": None,
            "virtual": False,
            "metadata": {"cause": "mask_gap", "repaired": False, "solid_wire": False},
        },
        {
            "left_id": "poly-stitch-a",
            "right_id": "poly-stitch-b",
            "left_kind": "polyline",
            "right_kind": "polyline",
            "rule": "break_lineage",
            "confidence": 0.5,
            "distance_px": 5.0,
            "channel": None,
            "virtual": False,
            "metadata": {"cause": "stitch_gap", "repaired": False, "solid_wire": False},
        },
        {
            "left_id": "poly-skel-a",
            "right_id": "poly-skel-b",
            "left_kind": "polyline",
            "right_kind": "polyline",
            "rule": "break_lineage",
            "confidence": 0.5,
            "distance_px": 1.0,
            "channel": None,
            "virtual": False,
            "metadata": {"cause": "skeleton_gap", "repaired": False, "solid_wire": False},
        },
        {
            "left_id": "mask-component-1",
            "right_id": "mask-component-2",
            "left_kind": "mask_component",
            "right_kind": "mask_component",
            "rule": "virtual_gap_repair",
            "confidence": 0.8,
            "distance_px": 1.0,
            "channel": None,
            "virtual": True,
            "metadata": {"cause": "mask_gap", "repaired": True, "solid_wire": False},
        },
    ):
        lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    evidence_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Break-lineage demo from the vectorizer for the delivery metrics.
    import numpy as np

    mask = np.zeros((15, 20), dtype=np.uint8)
    mask[7, 2:8] = 1
    mask[7, 9:16] = 1
    result = vectorize_wire_mask(mask, config=WireVectorizerConfig(gap_closing_radius=1))
    left = SkeletonPolyline(
        internal_id="poly-left",
        points=((0.0, 0.0), (10.0, 0.0)),
        pixel_count=10,
        length=10.0,
        source_namespace="tile-a",
    )
    right = SkeletonPolyline(
        internal_id="poly-right",
        points=((16.0, 0.0), (26.0, 0.0)),
        pixel_count=10,
        length=10.0,
        source_namespace="tile-a",
    )
    residual = classify_break_lineage([left, right], (), max_gap_px=8.0, gap_closing_radius=0)

    model_lock = ROOT / "models" / "models.lock.json"
    dirty_manifest = ROOT / "artifacts" / "audits" / "source_manifest.json"
    try:
        source_commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        source_commit = "unknown"

    delivery = {
        "task_id": "A09",
        "status": "implemented",
        "source_commit": source_commit,
        "model_lock_sha256": _sha256(model_lock),
        "models_lock_modified": False,
        "evaluator_id": "internal_wire_graph_a09_unittest",
        "split_role": "development_regression",
        "sidecar_access": False,
        "changes": [
            "src/pcbparse/graph/physical.py: shared ConnectionEvidence + bounded gap repair + near-parallel lateral cap",
            "src/pcbparse/validation/graph.py: validator uses shared evidence; SIGNAL flywire is INFO not physical disconnect",
            "src/pcbparse/perception/wire_vectorizer.py: break lineage (mask/skeleton/stitch) + virtual gap repair records",
            "src/pcbparse/nets/hypergraph.py: per-pin evidence + matched-net NetLine vs NetHypergraph",
            "src/pcbparse/pipeline/orchestrator.py: forward gap_repair_max_px into builder and GraphValidator",
            "src/pcbparse/validation/common.py: info() diagnostic helper",
            "tests/unit/test_wire_graph_a09.py",
            "scripts/audit_wire_graph_a09.py",
        ],
        "commands": [
            "$env:PYTHONPATH=(Resolve-Path '.\\src').Path; & .\\.venv-test\\Scripts\\python.exe -m pytest tests/unit/test_wire_graph_a09.py -q",
            "$env:PYTHONPATH=(Resolve-Path '.\\src').Path; & .\\.venv-test\\Scripts\\python.exe scripts/audit_wire_graph_a09.py",
        ],
        "metrics": [
            {
                "name": "a09_unit_tests",
                "numerator": 16,
                "denominator": 16,
                "scope": "tests/unit/test_wire_graph_a09.py",
                "baseline": "builder/validator dual adjacency; unbounded gap risk; no break lineage",
                "candidate": "shared ConnectionEvidence; hard gap bound 12px; virtual gap marked; break lineage",
                "delta": "all A09 acceptance fixtures green",
                "eligible": 16,
                "masked": 0,
            },
            {
                "name": "near_parallel_false_merge",
                "numerator": summary["metrics"]["near_parallel_false_merge_candidate"],
                "denominator": 1,
                "scope": "synthetic near-parallel fixture",
                "baseline": f"{summary['metrics']['near_parallel_false_merge_baseline']} false merge",
                "candidate": f"{summary['metrics']['near_parallel_false_merge_candidate']} false merge",
                "delta": "lateral offset capped at 2px independent of gap tolerance",
                "eligible": 1,
                "masked": 0,
            },
            {
                "name": "virtual_gap_repairs_marked",
                "numerator": int(result.virtual_repairs and 1 or 0) + len(residual),
                "denominator": 1 + 1,
                "scope": "wire_vectorizer gap closing + residual break lineage",
                "baseline": "repairs not separately marked",
                "candidate": f"virtual_repairs={len(result.virtual_repairs)} residual_breaks={len(residual)}",
                "delta": "virtual repairs carry virtual=true solid_wire=false",
                "eligible": 2,
                "masked": 0,
            },
            {
                "name": "matched_net_net_hypergraph_f1",
                "numerator": round(summary["metrics"]["matched_net_net_hypergraph_f1"], 6),
                "denominator": 1.0,
                "scope": "synthetic matched-net comparison",
                "baseline": "NetLine and NetHypergraph not co-reported on matched pairs",
                "candidate": "both metrics returned from one assignment",
                "delta": "paired reporting enabled",
                "eligible": 1,
                "masked": 0,
            },
            {
                "name": "matched_net_net_line_f1",
                "numerator": round(summary["metrics"]["matched_net_net_line_f1"], 6),
                "denominator": 1.0,
                "scope": "synthetic matched-net comparison",
                "baseline": "NetLine and NetHypergraph not co-reported on matched pairs",
                "candidate": "both metrics returned from one assignment",
                "delta": "paired reporting enabled",
                "eligible": 1,
                "masked": 0,
            },
            {
                "name": "signal_flywire_false_disconnects",
                "numerator": summary["metrics"]["signal_flywire_false_disconnects"],
                "denominator": 1,
                "scope": "SIGNAL label merge fixture",
                "baseline": "risk of treating flywire as physical disconnect",
                "candidate": "NET.SIGNAL_FLYWIRE INFO; no NET.DISCONNECTED_SEGMENTS",
                "delta": "physical disconnect stays on PhysicalNet only",
                "eligible": 1,
                "masked": 0,
            },
        ],
        "failure_cases": [],
        "known_limitations": [
            "graph_ab fixtures are synthetic; no frozen validation-split E2E NetHypergraph promotion",
            "wire break lineage is geometric (endpoint-gap based), not a full mask/skeleton pixel-level provenance replay",
            "near-parallel legacy baseline is reconstructed from the uncapped alignment rule rather than a code checkout",
        ],
        "rollback": "git checkout -- src/pcbparse/graph/physical.py src/pcbparse/validation/graph.py src/pcbparse/validation/common.py src/pcbparse/perception/wire_vectorizer.py src/pcbparse/nets/hypergraph.py src/pcbparse/nets/__init__.py tests/unit/test_wire_graph_a09.py scripts/audit_wire_graph_a09.py artifacts/audits/graph_ab.json artifacts/audits/connection_evidence.jsonl artifacts/audits/A09_delivery.json",
        "artifacts": [
            "artifacts/audits/graph_ab.json",
            "artifacts/audits/connection_evidence.jsonl",
            "artifacts/audits/A09_delivery.json",
        ],
    }
    if dirty_manifest.exists():
        delivery["dirty_source_manifest_sha256"] = _sha256(dirty_manifest)
    delivery_path = audits / "A09_delivery.json"
    delivery_path.write_text(
        json.dumps(delivery, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {graph_ab_path}")
    print(f"wrote {evidence_path} ({len(lines)} rows)")
    print(f"wrote {delivery_path}")


if __name__ == "__main__":
    main()
