"""A09: wire and physical/electrical graph targeted fixes.

Covers:
- shared builder/validator ConnectionEvidence (no dual connectivity contracts)
- bounded gap repair / no false net merges from unbounded tolerances
- near-parallel traces must not falsely connect
- SIGNAL flywire is not a physical disconnect
- virtual gap repair is marked separately from solid wire
- wire_vectorizer break lineage (mask / skeleton / stitch)
- nets/hypergraph per-pin evidence
- matched-net NetLine vs NetHypergraph compared together
"""

from __future__ import annotations

import json
import unittest

from pcbparse.graph.physical import (
    HARD_MAX_GAP_REPAIR_PX,
    MAX_COLLINEAR_LATERAL_OFFSET_PX,
    ConnectionEvidence,
    ConnectionParams,
    build_physical_graph,
    collect_segment_connections,
    segments_are_connected,
    segments_have_collinear_overlap,
)
from pcbparse.ir.models import (
    CircuitIR,
    Component,
    ElectricalNet,
    Evidence,
    ImageInfo,
    NetLabel,
    PhysicalNet,
    Pin,
    Point,
    WireSegment,
)
from pcbparse.nets.hypergraph import (
    NetView,
    build_electrical_nets,
    compare_matched_net_views,
    electrical_nets_to_views,
)
from pcbparse.perception.wire_vectorizer import (
    SkeletonPolyline,
    VirtualGapRepair,
    WireJunctionCandidate,
    WireVectorizerConfig,
    classify_break_lineage,
    vectorize_wire_mask,
)
from pcbparse.validation.graph import is_signal_flywire_electrical, validate_graph


def _ir(name: str = "a09") -> CircuitIR:
    return CircuitIR(image=ImageInfo(image_id=name, path=f"{name}.png", width=300, height=120))


def _seg(
    internal_id: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    channel: str = "horizontal",
    **kwargs: object,
) -> WireSegment:
    return WireSegment(
        internal_id=internal_id,
        start=Point(x=x1, y=y1),
        end=Point(x=x2, y=y2),
        channel=channel,
        **kwargs,  # type: ignore[arg-type]
    )


class SharedConnectionEvidenceTests(unittest.TestCase):
    def test_builder_and_validator_share_connection_evidence(self) -> None:
        ir = _ir()
        ir.segments = [
            _seg("s0", 0, 10, 20, 10),
            _seg("s1", 10, 10, 30, 10),
        ]
        builder_evidence: list[ConnectionEvidence] = []
        ir.physical_nets, _ = build_physical_graph(
            ir,
            endpoint_tolerance_px=1.0,
            collinear_merge_tolerance_px=0.75,
            connection_evidence_out=builder_evidence,
        )
        self.assertEqual(1, len(ir.physical_nets))

        shared = collect_segment_connections(
            ir.segments,
            endpoint_tolerance_px=1.0,
            collinear_merge_tolerance_px=0.75,
        )
        self.assertTrue(any(item.rule == "collinear_overlap" for item in shared))
        self.assertTrue(segments_are_connected(["s0", "s1"], shared))
        self.assertTrue(segments_are_connected(["s0", "s1"], builder_evidence))

        errors = [
            item
            for item in validate_graph(
                ir,
                endpoint_tolerance_px=1.0,
                collinear_merge_tolerance_px=0.75,
            )
            if item.severity == "ERROR"
        ]
        self.assertEqual([], errors)

    def test_connection_evidence_records_rule_and_channel(self) -> None:
        evidence = collect_segment_connections(
            [
                _seg("s0", 0, 10, 10, 10),
                _seg("s1", 10, 10, 20, 10),
            ],
            endpoint_tolerance_px=3.0,
        )
        self.assertTrue(evidence)
        item = evidence[0]
        self.assertIn(item.rule, {"endpoint_proximity", "polyline_touch"})
        self.assertEqual("horizontal", item.channel)
        self.assertFalse(item.virtual)
        payload = json.loads(item.to_jsonl())
        self.assertEqual(item.left_id, payload["left_id"])
        self.assertEqual(item.right_id, payload["right_id"])


class BoundedGapAndParallelGuardsTests(unittest.TestCase):
    def test_gap_tolerance_cannot_exceed_hard_bound(self) -> None:
        with self.assertRaisesRegex(ValueError, "unbounded gap repair|hard bound"):
            ConnectionParams(collinear_gap_merge_tolerance_px=HARD_MAX_GAP_REPAIR_PX + 1.0)
        with self.assertRaisesRegex(ValueError, "unbounded gap repair|cannot exceed"):
            ConnectionParams(
                collinear_gap_merge_tolerance_px=10.0,
                gap_repair_max_px=4.0,
            )
        with self.assertRaises(ValueError):
            build_physical_graph(_ir(), collinear_gap_merge_tolerance_px=100.0)

    def test_near_parallel_lines_do_not_falsely_connect(self) -> None:
        # 5px lateral offset must stay two nets even with a large longitudinal
        # gap tolerance and large collinear-merge tolerance.
        left = _seg("s0", 0, 10, 30, 10)
        right = _seg("s1", 0, 15, 30, 15)
        self.assertFalse(segments_have_collinear_overlap(left, right, 6.0, allow_gap=False))
        self.assertFalse(segments_have_collinear_overlap(left, right, 6.0, allow_gap=True))
        self.assertLessEqual(MAX_COLLINEAR_LATERAL_OFFSET_PX, 2.0)

        ir = _ir()
        ir.segments = [
            _seg("s0", 0, 10, 10, 10),
            _seg("s1", 14, 15, 24, 15),
        ]
        nets, _ = build_physical_graph(
            ir,
            collinear_gap_merge_tolerance_px=6.0,
            gap_repair_max_px=HARD_MAX_GAP_REPAIR_PX,
        )
        self.assertEqual(2, len(nets))

    def test_collinear_gap_repair_is_virtual_not_solid(self) -> None:
        ir = _ir()
        ir.segments = [
            _seg("s0", 0, 10, 10, 10),
            _seg("s1", 15, 10, 25, 10),
        ]
        evidence: list[ConnectionEvidence] = []
        nets, _ = build_physical_graph(
            ir,
            collinear_gap_merge_tolerance_px=6.0,
            gap_repair_max_px=8.0,
            connection_evidence_out=evidence,
        )
        self.assertEqual(1, len(nets))
        virtual = [item for item in evidence if item.rule == "collinear_gap_repair"]
        self.assertEqual(1, len(virtual))
        self.assertTrue(virtual[0].virtual)
        self.assertGreater(nets[0].repair_ratio, 0.0)
        self.assertLessEqual(nets[0].repair_ratio, 1.0)
        # Virtual repair is marked on the segments and is not solid wire.
        repair_marks = [
            item
            for segment in ir.segments
            for item in segment.evidence
            if item.metadata.get("kind") == "virtual_gap_repair"
        ]
        self.assertTrue(repair_marks)
        self.assertTrue(all(item.metadata.get("virtual") for item in repair_marks))
        self.assertTrue(all(item.metadata.get("solid_wire") is False for item in repair_marks))

    def test_false_nets_do_not_merge_across_lateral_offset_with_gap_repair(self) -> None:
        ir = _ir()
        ir.segments = [
            _seg("s0", 0, 10, 10, 10),
            _seg("s1", 12, 14, 30, 14),
        ]
        nets, _ = build_physical_graph(
            ir,
            collinear_gap_merge_tolerance_px=8.0,
        )
        self.assertEqual(2, len(nets))


class SignalFlywireTests(unittest.TestCase):
    def test_signal_flywire_is_not_physical_disconnect(self) -> None:
        ir = _ir()
        ir.components = [
            Component(internal_id="c1", key="R1", category="R", bbox={"x1": 0, "y1": 40, "x2": 20, "y2": 60}),
            Component(internal_id="c2", key="R2", category="R", bbox={"x1": 180, "y1": 40, "x2": 200, "y2": 60}),
        ]
        ir.pins = [
            Pin(internal_id="p1", component_id="c1", number="1", point=Point(x=20, y=50), orientation="R"),
            Pin(internal_id="p2", component_id="c2", number="1", point=Point(x=180, y=50), orientation="L"),
        ]
        ir.segments = [
            _seg("s1", 20, 50, 70, 50),
            _seg("s2", 130, 50, 180, 50),
        ]
        ir.physical_nets, _ = build_physical_graph(ir, unique_margin=0.1)
        self.assertEqual(2, len(ir.physical_nets))
        ir.net_labels = [
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
        ir.electrical_nets = build_electrical_nets(ir)
        self.assertEqual(1, len(ir.electrical_nets))
        self.assertTrue(is_signal_flywire_electrical(ir.electrical_nets[0]))

        errors = [item for item in validate_graph(ir) if item.severity == "ERROR"]
        self.assertFalse(any(item.code == "NET.DISCONNECTED_SEGMENTS" for item in errors))
        infos = [item for item in validate_graph(ir) if item.code == "NET.SIGNAL_FLYWIRE"]
        self.assertEqual(1, len(infos))
        self.assertEqual("INFO", infos[0].severity)

    def test_physical_disconnected_segments_still_flagged_on_physical_net(self) -> None:
        ir = _ir()
        ir.segments = [
            _seg("s1", 10, 10, 40, 10),
            _seg("s2", 120, 80, 170, 80),
        ]
        ir.physical_nets = [
            PhysicalNet(internal_id="bad", pins=[], segments=["s1", "s2"])
        ]
        errors = [item for item in validate_graph(ir) if item.severity == "ERROR"]
        self.assertTrue(any(item.code == "NET.DISCONNECTED_SEGMENTS" for item in errors))


class WireBreakLineageTests(unittest.TestCase):
    def test_virtual_gap_repair_is_recorded_separately_from_solid_wire(self) -> None:
        import numpy as np

        mask = np.zeros((15, 20), dtype=np.uint8)
        mask[7, 2:8] = 1
        mask[7, 9:16] = 1
        result = vectorize_wire_mask(mask, config=WireVectorizerConfig(gap_closing_radius=1))
        self.assertEqual(1, len(result.polylines))
        self.assertEqual(1, len(result.virtual_repairs))
        repair = result.virtual_repairs[0]
        self.assertTrue(repair.virtual)
        self.assertFalse(repair.solid_wire)
        self.assertGreaterEqual(repair.bridge_gap_px, 0.0)

    def test_break_lineage_classifies_mask_skeleton_and_stitch(self) -> None:
        mask_left = SkeletonPolyline(
            internal_id="poly-left",
            points=((0.0, 0.0), (10.0, 0.0)),
            pixel_count=10,
            length=10.0,
            source_namespace="tile-a",
            source_tile="tile-a@0,0",
        )
        mask_right = SkeletonPolyline(
            internal_id="poly-right",
            points=((16.0, 0.0), (26.0, 0.0)),
            pixel_count=10,
            length=10.0,
            source_namespace="tile-a",
            source_tile="tile-a@0,0",
        )
        stitch_right = SkeletonPolyline(
            internal_id="poly-stitch",
            points=((11.0, 5.0), (20.0, 5.0)),
            pixel_count=10,
            length=9.0,
            source_namespace="tile-b",
            source_tile="tile-b@0,0",
        )
        skeleton_near = SkeletonPolyline(
            internal_id="poly-skeleton",
            points=((0.0, 80.0), (10.0, 80.0)),
            pixel_count=10,
            length=10.0,
            source_namespace="tile-a",
            source_tile="tile-a@0,0",
        )
        skeleton_partner = SkeletonPolyline(
            internal_id="poly-skeleton-2",
            points=((11.0, 80.0), (20.0, 80.0)),
            pixel_count=10,
            length=9.0,
            source_namespace="tile-a",
            source_tile="tile-a@0,0",
        )

        mask_breaks = classify_break_lineage(
            [mask_left, mask_right],
            (),
            max_gap_px=8.0,
            gap_closing_radius=0,
            rdp_epsilon=0.75,
        )
        self.assertEqual(1, len(mask_breaks))
        self.assertEqual("mask_gap", mask_breaks[0].cause)
        self.assertFalse(mask_breaks[0].repaired)

        stitch_breaks = classify_break_lineage(
            [mask_left, stitch_right],
            (),
            max_gap_px=12.0,
            gap_closing_radius=0,
            rdp_epsilon=0.75,
        )
        self.assertEqual(1, len(stitch_breaks))
        self.assertEqual("stitch_gap", stitch_breaks[0].cause)

        skeleton_breaks = classify_break_lineage(
            [skeleton_near, skeleton_partner],
            (),
            max_gap_px=8.0,
            gap_closing_radius=0,
            rdp_epsilon=0.75,
        )
        self.assertEqual(1, len(skeleton_breaks))
        self.assertEqual("skeleton_gap", skeleton_breaks[0].cause)

    def test_junction_split_is_not_a_residual_break(self) -> None:
        left = SkeletonPolyline(
            internal_id="poly-left",
            points=((0.0, 0.0), (10.0, 0.0)),
            pixel_count=10,
            length=10.0,
            source_namespace="tile-a",
        )
        right = SkeletonPolyline(
            internal_id="poly-right",
            points=((10.0, 0.0), (20.0, 0.0)),
            pixel_count=10,
            length=10.0,
            source_namespace="tile-a",
        )
        junction = WireJunctionCandidate(
            internal_id="j1",
            point=(10.0, 0.0),
            kind="endpoint",
            degree=2,
            polyline_ids=("poly-left", "poly-right"),
        )
        breaks = classify_break_lineage([left, right], [junction], max_gap_px=1.0)
        self.assertEqual(1, len(breaks))
        self.assertEqual("junction_split", breaks[0].cause)
        self.assertTrue(breaks[0].repaired)

    def test_virtual_repair_record_is_not_solid_wire(self) -> None:
        repair = VirtualGapRepair(
            repair_id="virtual-gap:1:2:r1",
            left_component_id=1,
            right_component_id=2,
            bridge_gap_px=1.0,
            closing_radius=1,
        )
        payload = repair.as_dict()
        self.assertTrue(payload["virtual"])
        self.assertFalse(payload["solid_wire"])


class HypergraphPerPinEvidenceTests(unittest.TestCase):
    def test_per_pin_evidence_recorded_for_physical_and_flywire(self) -> None:
        ir = _ir()
        ir.pins = [
            Pin(internal_id="p1", component_id="c1", number="1", point=Point(x=10, y=10)),
            Pin(internal_id="p2", component_id="c2", number="1", point=Point(x=20, y=10)),
            Pin(internal_id="p3", component_id="c3", number="1", point=Point(x=200, y=10)),
        ]
        ir.physical_nets = [
            PhysicalNet(internal_id="pnet_00001", pins=["p1", "p2"], segments=["s1"]),
            PhysicalNet(internal_id="pnet_00002", pins=["p3"], segments=["s2"]),
        ]
        ir.net_labels = [
            NetLabel(
                internal_id="l1",
                instance_text="CLK",
                resolved_text="CLK",
                net_type="SIGNAL",
                attached_physical_net_id="pnet_00001",
            ),
            NetLabel(
                internal_id="l2",
                instance_text="CLK",
                resolved_text="CLK",
                net_type="SIGNAL",
                attached_physical_net_id="pnet_00002",
            ),
        ]
        evidence = []
        electrical = build_electrical_nets(ir, evidence_out=evidence)
        self.assertEqual(1, len(electrical))
        self.assertEqual(3, len(evidence))
        by_pin = {item.pin_id: item for item in evidence}
        self.assertEqual("signal_flywire_merge", by_pin["p1"].rule)
        self.assertTrue(by_pin["p1"].virtual)
        self.assertTrue(by_pin["p3"].virtual)
        self.assertTrue(all(item.electrical_net_id == "enet_00001" for item in evidence))
        # Durable IR pin evidence is attached as well.
        pin_evidence = [
            item
            for pin in ir.pins
            for item in pin.evidence
            if item.metadata.get("kind") == "pin_net_evidence"
        ]
        self.assertEqual(3, len(pin_evidence))
        self.assertIn("pin_net_evidence", ir.metadata)


class MatchedNetLineVsHypergraphTests(unittest.TestCase):
    def test_matched_net_views_report_both_metrics(self) -> None:
        predicted = [
            NetView(
                net_id="pred-a",
                pins=frozenset({"R1.1", "R2.1"}),
                edges=(
                    ((0.0, 0.0), (10.0, 0.0)),
                    ((10.0, 0.0), (20.0, 0.0)),
                ),
            ),
            NetView(
                net_id="pred-b",
                pins=frozenset({"U1.3"}),
                edges=(((0.0, 50.0), (10.0, 50.0)),),
            ),
        ]
        expected = [
            NetView(
                net_id="gt-a",
                pins=frozenset({"R1.1", "R2.1"}),
                edges=(
                    ((0.0, 0.0), (10.0, 0.0)),
                    ((10.0, 0.0), (21.0, 0.0)),
                ),
            ),
            NetView(
                net_id="gt-b",
                pins=frozenset({"U1.3", "U1.4"}),
                edges=(((0.0, 50.0), (10.0, 50.0)),),
            ),
        ]
        comparison = compare_matched_net_views(predicted, expected, line_tolerance_px=5.0)
        self.assertEqual(2, comparison.matched_pairs)
        self.assertGreater(comparison.net_hypergraph.f1, 0.0)
        self.assertGreater(comparison.net_line.f1, 0.0)
        # Geometry-correct lines inside a wrong net still leave HyperGraph error.
        self.assertGreater(comparison.net_hypergraph.false_negative, 0)
        self.assertEqual(2, len(comparison.pair_details))
        payload = comparison.as_dict()
        self.assertIn("net_hypergraph", payload)
        self.assertIn("net_line", payload)

    def test_electrical_nets_project_to_views(self) -> None:
        nets = [
            ElectricalNet(
                internal_id="enet_00001",
                pins=["p1", "p2"],
                segments=["s1"],
                physical_net_ids=["pnet_00001"],
            )
        ]
        views = electrical_nets_to_views(
            nets,
            segment_edges={"s1": ((0.0, 0.0), (5.0, 0.0))},
        )
        self.assertEqual(1, len(views))
        self.assertEqual(frozenset({"p1", "p2"}), views[0].pins)
        self.assertEqual(1, len(views[0].edges))


class ConfigSwitchRollbackTests(unittest.TestCase):
    def test_new_behavior_is_parameter_gated(self) -> None:
        # Default (no gap repair) keeps the two fragments split.
        ir = _ir()
        ir.segments = [
            _seg("s0", 0, 10, 10, 10),
            _seg("s1", 15, 10, 25, 10),
        ]
        default_nets, _ = build_physical_graph(ir)
        self.assertEqual(2, len(default_nets))
        self.assertEqual(0.0, default_nets[0].repair_ratio)


if __name__ == "__main__":
    unittest.main()
