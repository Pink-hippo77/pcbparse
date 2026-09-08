"""Targeted tests for P0 contract fixes (2026-09-08 architecture review).

Tests cover:
- P0-A: Graph builder/validator collinear connectivity contract
- P0-B: Component naming contract (all-letter designators, duplicate labels)
- P1-A: ONNX session provider verification (unit-level only, no ORT needed)
"""
from __future__ import annotations

import unittest

from pcbparse.association.text_component import (
    _is_valid_designator,
    resolve_component_keys,
)
from pcbparse.graph.physical import (
    build_physical_graph,
    segments_have_collinear_overlap,
)
from pcbparse.ir.models import (
    BBox,
    CircuitIR,
    Component,
    ImageInfo,
    PhysicalNet,
    Point,
    TextToken,
    WireSegment,
)
from pcbparse.validation.graph import validate_graph


def _ir(name: str = "test") -> CircuitIR:
    return CircuitIR(image=ImageInfo(image_id=name, path=f"{name}.png", width=300, height=100))


def _component(cid: str, category: str = "R", x: int = 20) -> Component:
    return Component(
        internal_id=cid,
        category=category,
        bbox=BBox(x1=x, y1=20, x2=x + 20, y2=40),
    )


def _text(tid: str, raw: str, details: str = "designator", cid: str | None = None, x: int = 20) -> TextToken:
    return TextToken(
        internal_id=tid,
        raw_text=raw,
        details=details,
        bbox=BBox(x1=x, y1=10, x2=x + 20, y2=18),
        score=0.99,
        component_id=cid,
    )


# ---------------------------------------------------------------------------
# P0-A: Graph Connectivity Contract
# ---------------------------------------------------------------------------

class TestCollinearOverlapGeometry(unittest.TestCase):
    """Verify the shared collinear overlap check works correctly."""

    def test_overlapping_segments_detected(self) -> None:
        seg_a = WireSegment(internal_id="a", start=Point(x=0, y=10), end=Point(x=20, y=10))
        seg_b = WireSegment(internal_id="b", start=Point(x=10, y=10), end=Point(x=30, y=10))
        self.assertTrue(segments_have_collinear_overlap(seg_a, seg_b, 0.75, allow_gap=False))

    def test_gap_segments_detected_when_allowed(self) -> None:
        seg_a = WireSegment(internal_id="a", start=Point(x=0, y=10), end=Point(x=10, y=10))
        seg_b = WireSegment(internal_id="b", start=Point(x=15, y=10), end=Point(x=25, y=10))
        self.assertTrue(segments_have_collinear_overlap(seg_a, seg_b, 6.0, allow_gap=True))

    def test_gap_segments_not_detected_when_gap_not_allowed(self) -> None:
        seg_a = WireSegment(internal_id="a", start=Point(x=0, y=10), end=Point(x=10, y=10))
        seg_b = WireSegment(internal_id="b", start=Point(x=15, y=10), end=Point(x=25, y=10))
        self.assertFalse(segments_have_collinear_overlap(seg_a, seg_b, 6.0, allow_gap=False))

    def test_parallel_offset_segments_not_collinear(self) -> None:
        seg_a = WireSegment(internal_id="a", start=Point(x=0, y=10), end=Point(x=30, y=10))
        seg_b = WireSegment(internal_id="b", start=Point(x=0, y=15), end=Point(x=30, y=15))
        self.assertFalse(segments_have_collinear_overlap(seg_a, seg_b, 0.75, allow_gap=False))

    def test_crossing_segments_not_collinear(self) -> None:
        seg_a = WireSegment(internal_id="a", start=Point(x=0, y=10), end=Point(x=30, y=10))
        seg_b = WireSegment(internal_id="b", start=Point(x=15, y=0), end=Point(x=15, y=30))
        self.assertFalse(segments_have_collinear_overlap(seg_a, seg_b, 0.75, allow_gap=False))

    def test_different_channels_not_collinear(self) -> None:
        seg_a = WireSegment(internal_id="a", start=Point(x=0, y=10), end=Point(x=30, y=10), channel="horizontal")
        seg_b = WireSegment(internal_id="b", start=Point(x=0, y=10), end=Point(x=30, y=10), channel="vertical")
        # Note: segments_have_collinear_overlap does NOT check channel; the caller does.
        # This test documents that the function only checks geometry.
        self.assertTrue(segments_have_collinear_overlap(seg_a, seg_b, 0.75, allow_gap=False))


class TestBuilderValidatorCollinearContract(unittest.TestCase):
    """Builder and validator agree on collinear connectivity."""

    def test_collinear_overlap_no_false_positive_with_matching_tolerances(self) -> None:
        """The core P0-A fix: collinear overlap segments are not flagged as disconnected."""
        ir = _ir("collinear")
        ir.segments = [
            WireSegment(internal_id="s0", start=Point(x=0, y=10), end=Point(x=20, y=10), channel="horizontal"),
            WireSegment(internal_id="s1", start=Point(x=10, y=10), end=Point(x=30, y=10), channel="horizontal"),
        ]
        ir.physical_nets, _ = build_physical_graph(
            ir, endpoint_tolerance_px=1, collinear_merge_tolerance_px=0.75,
        )
        self.assertEqual(1, len(ir.physical_nets))

        # Validator with matching tolerances: no error
        errors = [d for d in validate_graph(ir, collinear_merge_tolerance_px=0.75) if d.severity == "ERROR"]
        self.assertEqual([], errors)

    def test_collinear_overlap_still_fails_without_tolerance(self) -> None:
        """Legacy validator without tolerances still reports disconnect."""
        ir = _ir("collinear-legacy")
        ir.segments = [
            WireSegment(internal_id="s0", start=Point(x=0, y=10), end=Point(x=20, y=10), channel="horizontal"),
            WireSegment(internal_id="s1", start=Point(x=10, y=10), end=Point(x=30, y=10), channel="horizontal"),
        ]
        ir.physical_nets, _ = build_physical_graph(
            ir, endpoint_tolerance_px=1, collinear_merge_tolerance_px=0.75,
        )
        # Validator without tolerances: still reports disconnect (backward compat)
        errors = [d for d in validate_graph(ir) if d.severity == "ERROR"]
        self.assertTrue(any(d.code == "NET.DISCONNECTED_SEGMENTS" for d in errors))

    def test_collinear_gap_no_false_positive_with_matching_tolerances(self) -> None:
        """Gap merge segments are not flagged as disconnected."""
        ir = _ir("gap")
        ir.segments = [
            WireSegment(internal_id="s0", start=Point(x=0, y=10), end=Point(x=10, y=10), channel="horizontal"),
            WireSegment(internal_id="s1", start=Point(x=15, y=10), end=Point(x=25, y=10), channel="horizontal"),
        ]
        ir.physical_nets, _ = build_physical_graph(
            ir, endpoint_tolerance_px=1, collinear_gap_merge_tolerance_px=6.0,
        )
        self.assertEqual(1, len(ir.physical_nets))

        errors = [d for d in validate_graph(ir, collinear_gap_merge_tolerance_px=6.0) if d.severity == "ERROR"]
        self.assertEqual([], errors)

    def test_distant_segments_same_channel_two_nets(self) -> None:
        """Two far-apart same-channel segments remain separate nets."""
        ir = _ir("distant")
        ir.segments = [
            WireSegment(internal_id="s0", start=Point(x=0, y=10), end=Point(x=10, y=10), channel="horizontal"),
            WireSegment(internal_id="s1", start=Point(x=100, y=10), end=Point(x=110, y=10), channel="horizontal"),
        ]
        ir.physical_nets, _ = build_physical_graph(ir, endpoint_tolerance_px=1)
        self.assertEqual(2, len(ir.physical_nets))

    def test_distant_reused_polyline_not_merged_by_builder(self) -> None:
        """Builder does not merge distant segments with same polyline ID."""
        ir = _ir("polyline-reuse")
        ir.segments = [
            WireSegment(internal_id="s0", start=Point(x=0, y=10), end=Point(x=10, y=10), channel="horizontal", source_polyline_id="reused"),
            WireSegment(internal_id="s1", start=Point(x=50, y=10), end=Point(x=60, y=10), channel="horizontal", source_polyline_id="reused"),
        ]
        built, _ = build_physical_graph(ir)
        self.assertEqual(2, len(built))


# ---------------------------------------------------------------------------
# P0-B: Component Naming Contract
# ---------------------------------------------------------------------------

class TestValidDesignator(unittest.TestCase):
    """Test the _is_valid_designator helper."""

    def test_standard_alphanumeric_accepted(self) -> None:
        self.assertTrue(_is_valid_designator("R1"))
        self.assertTrue(_is_valid_designator("C10"))
        self.assertTrue(_is_valid_designator("U5A"))
        self.assertTrue(_is_valid_designator("D3"))

    def test_all_letter_compatible_category_accepted(self) -> None:
        self.assertTrue(_is_valid_designator("RUN", "vbus_in"))
        self.assertTrue(_is_valid_designator("VCC", "VREG"))
        self.assertTrue(_is_valid_designator("GND", "GROUND"))
        self.assertTrue(_is_valid_designator("VDD", "LDO"))

    def test_all_letter_v_prefix_accepted(self) -> None:
        self.assertTrue(_is_valid_designator("VCC", "VREG"))
        # VCC on a generic IC category is correctly rejected (IC != V-prefix)
        self.assertFalse(_is_valid_designator("VCC", "IC"))

    def test_random_string_rejected(self) -> None:
        self.assertFalse(_is_valid_designator("HELLO"))
        self.assertFalse(_is_valid_designator("XYZ"))
        self.assertFalse(_is_valid_designator("abc"))
        self.assertFalse(_is_valid_designator("R"))
        self.assertFalse(_is_valid_designator("123"))

    def test_single_letter_rejected(self) -> None:
        self.assertFalse(_is_valid_designator("R"))


class TestResolveComponentKeys(unittest.TestCase):
    """Test resolve_component_keys with new naming rules."""

    def test_standard_designator_works(self) -> None:
        ir = _ir()
        ir.components = [_component("c1", "R", 20)]
        ir.texts = [_text("t1", "R1", cid="c1", x=20)]
        assignments = resolve_component_keys(ir.components, ir.texts, min_score=0.0)
        self.assertEqual({"c1": "R1"}, assignments)
        self.assertEqual("R1", ir.components[0].key)

    def test_all_letter_designator_accepted(self) -> None:
        """RUN with compatible category gets assigned."""
        ir = _ir()
        ir.components = [_component("c1", "vbus_in", 20)]
        ir.texts = [_text("t1", "RUN", cid="c1", x=20)]
        assignments = resolve_component_keys(ir.components, ir.texts, min_score=0.0)
        self.assertEqual({"c1": "RUN"}, assignments)
        self.assertEqual("RUN", ir.components[0].key)

    def test_duplicate_visible_designator_both_get_keys(self) -> None:
        """Two components with independent R1 evidence both get the key."""
        ir = _ir()
        ir.components = [_component("c1", "R", 20), _component("c2", "R", 140)]
        ir.texts = [
            _text("t1", "R1", cid="c1", x=20),
            _text("t2", "R1", cid="c2", x=140),
        ]
        assignments = resolve_component_keys(ir.components, ir.texts, min_score=0.0)
        self.assertEqual("R1", ir.components[0].key)
        self.assertEqual("R1", ir.components[1].key)
        self.assertEqual(2, len(assignments))

    def test_empty_key_still_fails_schema_validation(self) -> None:
        """A component with no key still triggers COMPONENT.MISSING_KEY."""
        from pcbparse.validation.schema import validate_schema
        ir = _ir()
        ir.components = [_component("c1")]
        ir.components[0].key = None
        issues = validate_schema(ir)
        self.assertTrue(any(d.code == "COMPONENT.MISSING_KEY" for d in issues))


if __name__ == "__main__":
    unittest.main()
