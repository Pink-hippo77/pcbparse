from __future__ import annotations

import unittest

from pcbparse.graph.physical import build_physical_graph
from pcbparse.association.text_component import resolve_component_keys
from pcbparse.ir import (
    AffineTransform,
    BBox,
    CircuitIR,
    Component,
    Evidence,
    ImageInfo,
    NetLabel,
    Pin,
    Point,
    TransformChain,
    WireSegment,
)
from pcbparse.nets.hypergraph import build_electrical_nets


class TransformTests(unittest.TestCase):
    def test_roundtrip_is_subpixel(self) -> None:
        chain = TransformChain(
            [
                AffineTransform.scale_translate(
                    scale_x=0.37,
                    scale_y=0.61,
                    offset_x=13.25,
                    offset_y=-7.5,
                    source_view="original",
                    target_view="tile",
                )
            ]
        )
        self.assertLessEqual(chain.max_roundtrip_error(1920, 1080, step=53), 0.5)


class GraphTests(unittest.TestCase):
    def test_designator_resolution_never_overwrites_existing_key(self) -> None:
        component = Component(
            internal_id="c1",
            key="DATASET_C_1",
            category="R",
            bbox=BBox(x1=0, y1=0, x2=20, y2=20),
        )
        from pcbparse.ir import TextToken

        token = TextToken(
            internal_id="t1",
            raw_text="R1",
            resolved_text="R1",
            details="designator",
            bbox=BBox(x1=0, y1=0, x2=10, y2=10),
        )
        self.assertEqual({}, resolve_component_keys([component], [token]))
        self.assertEqual("DATASET_C_1", component.key)

    def test_dataset_explicit_segment_membership_overrides_distance_snap(self) -> None:
        ir = CircuitIR(image=ImageInfo(image_id="fixture", path="fixture.png", width=200, height=100))
        ir.pins = [
            Pin(
                internal_id="p1",
                number="1",
                point=Point(x=1, y=1),
                edge_ids=["s1"],
                evidence=[Evidence(evidence_id="dataset:p1", source="dataset")],
            )
        ]
        ir.segments = [
            WireSegment(internal_id="s1", start=Point(x=100, y=50), end=Point(x=150, y=50))
        ]
        nets, decisions = build_physical_graph(ir)
        self.assertEqual(["p1"], nets[0].pins)
        self.assertEqual(["s1"], nets[0].segments)
        self.assertEqual("unmatched", decisions[0].status)

    def test_opt_in_collinear_overlap_merge_does_not_bridge_parallel_trace(self) -> None:
        ir = CircuitIR(image=ImageInfo(image_id="fixture", path="fixture.png", width=200, height=100))
        ir.segments = [
            WireSegment(
                internal_id="s1",
                start=Point(x=0, y=20),
                end=Point(x=10, y=20),
                channel="horizontal",
            ),
            WireSegment(
                internal_id="s2",
                start=Point(x=5, y=20),
                end=Point(x=15, y=20),
                channel="horizontal",
            ),
            WireSegment(
                internal_id="s3",
                start=Point(x=5, y=24),
                end=Point(x=15, y=24),
                channel="horizontal",
            ),
        ]

        default_nets, _ = build_physical_graph(ir)
        candidate_nets, _ = build_physical_graph(
            ir,
            collinear_merge_tolerance_px=0.75,
        )

        self.assertEqual(3, len(default_nets))
        self.assertEqual(2, len(candidate_nets))
        self.assertTrue(
            any(set(net.segments) == {"s1", "s2"} for net in candidate_nets)
        )
        self.assertTrue(any(net.segments == ["s3"] for net in candidate_nets))

    def test_opt_in_collinear_gap_merge_does_not_bridge_parallel_trace(self) -> None:
        ir = CircuitIR(image=ImageInfo(image_id="fixture", path="fixture.png", width=200, height=100))
        ir.segments = [
            WireSegment(
                internal_id="s1",
                start=Point(x=0, y=20),
                end=Point(x=10, y=20),
                channel="horizontal",
            ),
            WireSegment(
                internal_id="s2",
                start=Point(x=14, y=20),
                end=Point(x=24, y=20),
                channel="horizontal",
            ),
            WireSegment(
                internal_id="s3",
                start=Point(x=14, y=24),
                end=Point(x=24, y=24),
                channel="horizontal",
            ),
        ]

        default_nets, _ = build_physical_graph(ir)
        candidate_nets, _ = build_physical_graph(
            ir,
            collinear_gap_merge_tolerance_px=4.0,
        )

        self.assertEqual(3, len(default_nets))
        self.assertEqual(2, len(candidate_nets))
        self.assertTrue(
            any(set(net.segments) == {"s1", "s2"} for net in candidate_nets)
        )
        self.assertTrue(any(net.segments == ["s3"] for net in candidate_nets))

    def test_physical_and_signal_label_merge(self) -> None:
        ir = CircuitIR(image=ImageInfo(image_id="fixture", path="fixture.png", width=200, height=100))
        ir.components = [
            Component(internal_id="c1", key="R1", category="R", bbox=BBox(x1=0, y1=40, x2=20, y2=60)),
            Component(internal_id="c2", key="R2", category="R", bbox=BBox(x1=180, y1=40, x2=200, y2=60)),
        ]
        ir.pins = [
            Pin(internal_id="p1", component_id="c1", number="1", point=Point(x=20, y=50), orientation="R"),
            Pin(internal_id="p2", component_id="c2", number="1", point=Point(x=180, y=50), orientation="L"),
        ]
        ir.segments = [
            WireSegment(internal_id="s1", start=Point(x=20, y=50), end=Point(x=70, y=50)),
            WireSegment(internal_id="s2", start=Point(x=130, y=50), end=Point(x=180, y=50)),
        ]
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
        ir.physical_nets, decisions = build_physical_graph(ir, unique_margin=0.1)
        self.assertEqual(2, len(ir.physical_nets))
        self.assertTrue(all(decision.status == "accepted" for decision in decisions))
        # Builder regenerates IDs deterministically; fixture label IDs already target those IDs.
        ir.electrical_nets = build_electrical_nets(ir)
        self.assertEqual(1, len(ir.electrical_nets))
        self.assertEqual(["p1", "p2"], ir.electrical_nets[0].pins)

    def test_signal_labels_below_merge_threshold_stay_separate(self) -> None:
        ir = CircuitIR(image=ImageInfo(image_id="fixture", path="fixture.png", width=100, height=100))
        from pcbparse.ir import PhysicalNet

        ir.physical_nets = [
            PhysicalNet(internal_id="p1", pins=["pin1"]),
            PhysicalNet(internal_id="p2", pins=["pin2"]),
        ]
        ir.net_labels = [
            NetLabel(
                internal_id="l1",
                instance_text="CLK",
                resolved_text="CLK",
                net_type="SIGNAL",
                attached_physical_net_id="p1",
                score=0.84,
            ),
            NetLabel(
                internal_id="l2",
                instance_text="CLK",
                resolved_text="CLK",
                net_type="SIGNAL",
                attached_physical_net_id="p2",
                score=0.99,
            ),
        ]

        electrical = build_electrical_nets(ir, min_text_score=0.75, merge_threshold=0.85)

        self.assertEqual(2, len(electrical))


if __name__ == "__main__":
    unittest.main()
