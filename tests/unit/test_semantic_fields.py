from __future__ import annotations

import unittest

from pcbparse.association.semantic_fields import infer_semantic_fields
from pcbparse.ir.models import (
    BBox,
    Component,
    Evidence,
    ImageInfo,
    Pin,
    Point,
    TextToken,
    WireSegment,
)


def _component(
    internal_id: str,
    category: str,
    bbox: BBox,
) -> Component:
    return Component(internal_id=internal_id, category=category, bbox=bbox)


def _pin(internal_id: str, component_id: str, point: Point) -> Pin:
    return Pin(internal_id=internal_id, component_id=component_id, point=point)


def _token(
    internal_id: str,
    text: str,
    bbox: BBox,
    *,
    score: float = 0.9,
    level: str = "global",
    side: str | None = None,
    component_id: str | None = None,
    details: str = "",
) -> TextToken:
    metadata: dict[str, object] = {"level": level}
    if side is not None:
        metadata["side"] = side
    if component_id is not None:
        metadata["component_id"] = component_id
    evidence = Evidence(
        evidence_id=f"ocr:{internal_id}",
        source="ocr",
        score=score,
        metadata=metadata,
    )
    return TextToken(
        internal_id=internal_id,
        raw_text=text,
        bbox=bbox,
        score=score,
        details=details,
        evidence=[evidence],
    )


class SemanticFieldInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.components = [
            _component("component:A", "Capacitor", BBox(x1=100, y1=100, x2=140, y2=130)),
            _component("component:B", "IC", BBox(x1=300, y1=100, x2=360, y2=140)),
        ]
        self.pins = [
            _pin("pin:A1", "component:A", Point(x=105, y=105)),
            _pin("pin:A2", "component:A", Point(x=105, y=125)),
        ]
        self.segments = [
            WireSegment(internal_id="segment:S1", start=Point(x=50, y=50), end=Point(x=90, y=50))
        ]

    def test_side_strip_digit_is_pin_number(self) -> None:
        token = _token(
            "text:t1",
            "1",
            BBox(x1=102, y1=96, x2=106, y2=100),
            level="side_strip",
            side="T",
            component_id="component:A",
        )
        infer_semantic_fields(self.components, self.pins, [token], self.segments)
        self.assertEqual("pinnumber", token.details)
        self.assertEqual("component:A", token.component_id)

    def test_component_roi_designator_matches_category(self) -> None:
        token = _token(
            "text:t2",
            "C12",
            BBox(x1=115, y1=110, x2=130, y2=118),
            level="component",
            component_id="component:A",
        )
        infer_semantic_fields(self.components, self.pins, [token], self.segments)
        self.assertEqual("designator", token.details)
        self.assertEqual("component:A", token.component_id)

    def test_generic_box_accepts_any_conservative_designator_prefix(self) -> None:
        component = _component(
            "component:box",
            "Box",
            BBox(x1=10, y1=10, x2=50, y2=50),
        )
        token = _token(
            "text:box-designator",
            "SW2",
            BBox(x1=16, y1=14, x2=34, y2=22),
            level="component",
            component_id=component.internal_id,
        )

        summary = infer_semantic_fields([component], [], [token], [])

        self.assertEqual("designator", token.details)
        self.assertEqual(component.internal_id, token.component_id)
        self.assertEqual({"designator": 1}, summary["by_field"])

    def test_component_roi_digit_is_pin_number_not_value(self) -> None:
        token = _token(
            "text:t3",
            "2",
            BBox(x1=118, y1=120, x2=122, y2=124),
            level="component",
            component_id="component:A",
        )
        infer_semantic_fields(self.components, self.pins, [token], self.segments)
        self.assertEqual("pinnumber", token.details)

    def test_box_global_title_is_search_name(self) -> None:
        component = _component(
            "component:box", "Box", BBox(x1=100, y1=100, x2=300, y2=300)
        )
        token = _token(
            "text:title", "Sensor Fusion / 6 DOF IMU",
            BBox(x1=120, y1=70, x2=260, y2=90),
        )
        summary = infer_semantic_fields([component], [], [token], [])
        self.assertEqual("searchname", token.details)
        self.assertIsNone(token.component_id)
        self.assertEqual({"searchname": 1}, summary["by_field"])

    def test_short_box_title_with_space_is_search_name(self) -> None:
        component = _component(
            "component:box", "Box", BBox(x1=100, y1=100, x2=300, y2=300)
        )
        token = _token(
            "text:short-title", "USB C",
            BBox(x1=120, y1=70, x2=180, y2=90),
        )
        summary = infer_semantic_fields([component], [], [token], [])
        self.assertEqual("searchname", token.details)
        self.assertEqual({"searchname": 1}, summary["by_field"])
    def test_global_value_near_component(self) -> None:
        token = _token("text:t4", "10uF", BBox(x1=120, y1=135, x2=145, y2=142))
        infer_semantic_fields(self.components, self.pins, [token], self.segments)
        self.assertEqual("value", token.details)
        self.assertIsNone(token.component_id)

    def test_global_netlabel_near_segment(self) -> None:
        token = _token("text:t5", "+5V_OUT", BBox(x1=40, y1=44, x2=75, y2=52))
        infer_semantic_fields(self.components, self.pins, [token], self.segments)
        self.assertEqual("netlabel", token.details)

    def test_single_character_garbage_not_classified(self) -> None:
        token = _token(
            "text:t6",
            ".",
            BBox(x1=102, y1=96, x2=103, y2=97),
            level="side_strip",
            side="R",
            component_id="component:A",
        )
        infer_semantic_fields(self.components, self.pins, [token], self.segments)
        self.assertEqual("", token.details)

    def test_existing_details_are_preserved(self) -> None:
        token = _token(
            "text:t7",
            "C12",
            BBox(x1=115, y1=110, x2=130, y2=118),
            details="pinnumber",
        )
        infer_semantic_fields(self.components, self.pins, [token], self.segments)
        self.assertEqual("pinnumber", token.details)

    def test_global_pin_number_near_pin(self) -> None:
        token = _token("text:t8", "2", BBox(x1=98, y1=103, x2=102, y2=107))
        infer_semantic_fields(self.components, self.pins, [token], self.segments)
        self.assertEqual("pinnumber", token.details)
        self.assertIsNone(token.component_id)


if __name__ == "__main__":
    unittest.main()
