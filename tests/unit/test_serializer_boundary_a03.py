"""A03 serializer boundary: no IR mutation, explicit quality counts."""

from __future__ import annotations

import copy
import unittest

from pcbparse.ir.models import (
    BBox,
    CircuitIR,
    Component,
    ElectricalNet,
    ImageInfo,
    IRState,
    PhysicalNet,
    Pin,
    Point,
    WireSegment,
)
from pcbparse.output import OfficialSchemaError
from pcbparse.output.official_serializer import (
    serialize_official,
    serialize_official_with_quality,
)


def make_ir(*, key: str | None = "R1", pin_number: str | None = "1") -> CircuitIR:
    return CircuitIR(
        state=IRState.VALIDATED,
        image=ImageInfo(
            image_id="board-a03",
            path="board.png",
            width=100,
            height=80,
            sha256="b" * 64,
        ),
        components=[
            Component(
                internal_id="c1",
                key=key,
                category="resistor",
                value="10k",
                bbox=BBox(x1=10.0, y1=10.0, x2=20.0, y2=20.0),
            )
        ],
        pins=[
            Pin(
                internal_id="p1",
                component_id="c1",
                number=pin_number,
                name="A",
                point=Point(x=10.0, y=15.0),
            )
        ],
        segments=[
            WireSegment(
                internal_id="s1",
                start=Point(x=10.0, y=15.0),
                end=Point(x=30.0, y=15.0),
            )
        ],
        physical_nets=[
            PhysicalNet(internal_id="pn1", pins=["p1"], segments=["s1"])
        ],
        electrical_nets=[
            ElectricalNet(internal_id="en1", pins=["p1"], segments=["s1"])
        ],
        diagnostics=[],
    )


class SerializerBoundaryTests(unittest.TestCase):
    def test_serialize_does_not_mutate_ir(self) -> None:
        ir = make_ir()
        before = copy.deepcopy(ir.model_dump(mode="json"))
        output = serialize_official(ir)
        after = ir.model_dump(mode="json")
        self.assertEqual(before, after)
        self.assertIn("R1", output["components"])

    def test_missing_key_rejected_without_ir_mutation(self) -> None:
        ir = make_ir(key=None)
        before_key = ir.components[0].key
        before = copy.deepcopy(ir.model_dump(mode="json"))
        with self.assertRaises(OfficialSchemaError):
            serialize_official(ir)
        self.assertIsNone(before_key)
        self.assertEqual(before, ir.model_dump(mode="json"))

    def test_strict_rejects_missing_key(self) -> None:
        ir = make_ir(key=None)
        with self.assertRaises(OfficialSchemaError):
            serialize_official_with_quality(ir, mode="strict")

    def test_degraded_also_rejects_missing_key(self) -> None:
        ir = make_ir(key=None)
        before = copy.deepcopy(ir.model_dump(mode="json"))
        with self.assertRaises(OfficialSchemaError):
            serialize_official_with_quality(ir, mode="degraded")
        self.assertEqual(ir.model_dump(mode="json"), before)

    def test_empty_pin_number_rejected_without_ir_mutation(self) -> None:
        ir = make_ir(pin_number="")
        before = copy.deepcopy(ir.model_dump(mode="json"))
        with self.assertRaises(OfficialSchemaError):
            serialize_official_with_quality(ir, mode="strict")
        self.assertEqual(ir.model_dump(mode="json"), before)

    def test_placeholder_pin_number_rejected(self) -> None:
        ir = make_ir(pin_number="∅")
        with self.assertRaises(OfficialSchemaError):
            serialize_official_with_quality(ir, mode="strict")

    def test_explicit_pin_number_counted(self) -> None:
        ir = make_ir(pin_number="7")
        _, quality = serialize_official_with_quality(ir, mode="strict")
        self.assertEqual(quality["pin_number_missing"], 0)
        self.assertEqual(quality["pin_explicit_numbers"], 1)
        self.assertTrue(quality["strict_success"])


if __name__ == "__main__":
    unittest.main()
