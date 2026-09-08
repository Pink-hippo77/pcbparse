from __future__ import annotations

import unittest

from pcbparse.graph.physical import build_physical_graph
from pcbparse.ir import (
    CircuitIR,
    Component,
    Evidence,
    ImageInfo,
    Junction,
    NetLabel,
    Pin,
    Point,
    WireSegment,
)
from pcbparse.validation.graph import validate_graph


def _ir() -> CircuitIR:
    return CircuitIR(image=ImageInfo(image_id="graph-fixture", path="fixture.png", width=200, height=120))


class PhysicalGraphFixtureTests(unittest.TestCase):
    def test_direct_pin_to_pin_connection_without_wire_pixels(self) -> None:
        ir = _ir()
        ir.components = [
            Component(internal_id="c1", key="R1", category="R", bbox={"x1": 0, "y1": 40, "x2": 20, "y2": 60}),
            Component(internal_id="c2", key="R2", category="R", bbox={"x1": 20, "y1": 40, "x2": 40, "y2": 60}),
        ]
        ir.pins = [
            Pin(internal_id="p1", component_id="c1", number="1", point=Point(x=20, y=50)),
            Pin(internal_id="p2", component_id="c2", number="1", point=Point(x=20.4, y=50)),
        ]

        nets, _ = build_physical_graph(ir)

        self.assertEqual(1, len(nets))
        self.assertEqual(["p1", "p2"], nets[0].pins)
        self.assertEqual([], nets[0].segments)

    def test_t_junction_connects_channels_only_with_junction_evidence(self) -> None:
        ir = _ir()
        ir.segments = [
            WireSegment(
                internal_id="horizontal-left",
                start=Point(x=10, y=50),
                end=Point(x=50, y=50),
                channel="horizontal",
            ),
            WireSegment(
                internal_id="horizontal-right",
                start=Point(x=50, y=50),
                end=Point(x=90, y=50),
                channel="horizontal",
            ),
            WireSegment(
                internal_id="vertical",
                start=Point(x=50, y=50),
                end=Point(x=50, y=90),
                channel="vertical",
            ),
        ]
        ir.junctions = [
            Junction(
                internal_id="j1",
                point=Point(x=50, y=50),
                kind="t_junction",
                segment_ids=["horizontal-left", "horizontal-right", "vertical"],
            )
        ]

        nets, _ = build_physical_graph(ir)

        self.assertEqual(1, len(nets))
        self.assertEqual(["horizontal-left", "horizontal-right", "vertical"], nets[0].segments)
        ir.physical_nets = nets
        self.assertFalse(any(item.severity == "ERROR" for item in validate_graph(ir)))

    def test_non_connected_cross_remains_two_physical_nets(self) -> None:
        ir = _ir()
        ir.segments = [
            WireSegment(
                internal_id="horizontal",
                start=Point(x=10, y=50),
                end=Point(x=90, y=50),
                channel="horizontal",
            ),
            WireSegment(
                internal_id="vertical",
                start=Point(x=50, y=10),
                end=Point(x=50, y=90),
                channel="vertical",
            ),
        ]
        ir.junctions = [
            Junction(
                internal_id="j1",
                point=Point(x=50, y=50),
                kind="non_connected_cross",
                segment_ids=["horizontal", "vertical"],
            )
        ]

        nets, _ = build_physical_graph(ir)

        self.assertEqual(2, len(nets))
        self.assertEqual({("horizontal",), ("vertical",)}, {tuple(net.segments) for net in nets})

    def test_endpoint_classification_never_bridges_channel_separated_segments(self) -> None:
        ir = _ir()
        ir.segments = [
            WireSegment(
                internal_id="horizontal",
                start=Point(x=10, y=50),
                end=Point(x=50, y=50),
                channel="horizontal",
            ),
            WireSegment(
                internal_id="vertical",
                start=Point(x=50, y=50),
                end=Point(x=50, y=90),
                channel="vertical",
            ),
        ]
        ir.junctions = [
            Junction(
                internal_id="endpoint-candidate",
                point=Point(x=50, y=50),
                kind="endpoint",
                segment_ids=["horizontal", "vertical"],
            )
        ]

        nets, _ = build_physical_graph(ir)

        self.assertEqual(2, len(nets))
        self.assertEqual(
            {("horizontal",), ("vertical",)},
            {tuple(net.segments) for net in nets},
        )

    def test_non_connected_diagonal_cross_keeps_its_two_slope_channels(self) -> None:
        ir = _ir()
        ir.segments = [
            WireSegment(
                internal_id="down-right",
                start=Point(x=10, y=10),
                end=Point(x=50, y=50),
                channel="diagonal_positive",
            ),
            WireSegment(
                internal_id="up-right",
                start=Point(x=10, y=50),
                end=Point(x=50, y=10),
                channel="diagonal_negative",
            ),
        ]
        ir.junctions = [
            Junction(
                internal_id="diagonal-cross",
                point=Point(x=30, y=30),
                kind="non_connected_cross",
                segment_ids=["down-right", "up-right"],
            )
        ]

        nets, _ = build_physical_graph(ir)

        self.assertEqual(2, len(nets))

    def test_touching_bend_members_of_one_polyline_connect(self) -> None:
        ir = _ir()
        ir.segments = [
            WireSegment(
                internal_id="s1",
                start=Point(x=10, y=10),
                end=Point(x=50, y=10),
                source_polyline_id="polyline-1",
                channel="horizontal",
            ),
            WireSegment(
                internal_id="s2",
                start=Point(x=50, y=10),
                end=Point(x=50, y=40),
                source_polyline_id="polyline-1",
                channel="vertical",
            ),
        ]

        nets, _ = build_physical_graph(ir)

        self.assertEqual(1, len(nets))

    def test_distant_reused_polyline_id_does_not_create_giant_net(self) -> None:
        ir = _ir()
        ir.segments = [
            WireSegment(
                internal_id="s1",
                start=Point(x=10, y=10),
                end=Point(x=40, y=10),
                source_polyline_id="bad-reused-id",
            ),
            WireSegment(
                internal_id="s2",
                start=Point(x=120, y=80),
                end=Point(x=170, y=80),
                source_polyline_id="bad-reused-id",
            ),
        ]

        nets, _ = build_physical_graph(ir)

        self.assertEqual(2, len(nets))

    def test_untrusted_far_explicit_membership_is_ignored(self) -> None:
        ir = _ir()
        ir.pins = [
            Pin(
                internal_id="p1",
                component_id="c1",
                number="1",
                point=Point(x=1, y=1),
                edge_ids=["s1"],
                evidence=[Evidence(evidence_id="detector:p1", source="detector")],
            )
        ]
        ir.segments = [
            WireSegment(internal_id="s1", start=Point(x=100, y=50), end=Point(x=150, y=50))
        ]

        nets, decisions = build_physical_graph(ir)

        self.assertEqual(2, len(nets))
        self.assertFalse(any(net.pins and net.segments for net in nets))
        self.assertEqual("unmatched", decisions[0].status)

    def test_rejected_explicit_edge_does_not_fallback_snap_to_another_net(self) -> None:
        ir = _ir()
        ir.pins = [
            Pin(
                internal_id="p1",
                component_id="c1",
                number="1",
                point=Point(x=10, y=10),
                edge_ids=["declared"],
                evidence=[Evidence(evidence_id="detector:p1", source="detector")],
            )
        ]
        ir.segments = [
            WireSegment(internal_id="declared", start=Point(x=100, y=80), end=Point(x=140, y=80)),
            WireSegment(internal_id="near", start=Point(x=10, y=11), end=Point(x=50, y=11)),
        ]

        nets, decisions = build_physical_graph(ir, unique_margin=0.1)

        pin_net = next(net for net in nets if net.pins)
        self.assertEqual([], pin_net.segments)
        self.assertEqual("accepted", decisions[0].status)

    def test_label_point_attaches_only_to_unique_nearby_segment(self) -> None:
        ir = _ir()
        ir.segments = [
            WireSegment(internal_id="near", start=Point(x=10, y=20), end=Point(x=90, y=20)),
            WireSegment(internal_id="far", start=Point(x=10, y=80), end=Point(x=90, y=80)),
        ]
        ir.net_labels = [
            NetLabel(
                internal_id="label",
                instance_text="CLK",
                resolved_text="CLK",
                net_type="SIGNAL",
                point=Point(x=30, y=23),
            )
        ]

        nets, _ = build_physical_graph(ir, label_attach_tolerance_px=8.0)

        attached = next(net for net in nets if "label" in net.labels)
        self.assertEqual(["near"], attached.segments)
        self.assertEqual(attached.internal_id, ir.net_labels[0].attached_physical_net_id)

    def test_two_nearby_segments_of_same_net_do_not_make_pin_ambiguous(self) -> None:
        ir = _ir()
        ir.pins = [
            Pin(
                internal_id="pin",
                component_id="c1",
                number="1",
                point=Point(x=50, y=22),
                orientation="B",
            )
        ]
        ir.segments = [
            WireSegment(
                internal_id="left",
                start=Point(x=10, y=20),
                end=Point(x=50, y=20),
                source_polyline_id="one-net",
            ),
            WireSegment(
                internal_id="right",
                start=Point(x=50, y=20),
                end=Point(x=90, y=20),
                source_polyline_id="one-net",
            ),
        ]

        nets, decisions = build_physical_graph(ir, unique_margin=1.5)

        self.assertEqual("accepted", decisions[0].status)
        self.assertEqual(1, len(nets))
        self.assertEqual(["pin"], nets[0].pins)

    def test_pin_snap_minimum_cannot_silently_exceed_configured_maximum(self) -> None:
        ir = _ir()
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            build_physical_graph(
                ir,
                pin_snap_min_px=5.0,
                pin_snap_tolerance_px=4.0,
            )


if __name__ == "__main__":
    unittest.main()
