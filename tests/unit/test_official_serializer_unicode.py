"""Model-free regressions for Unicode identifiers in official JSON output."""
import unittest

from pcbparse.ir.models import (
    BBox,
    CircuitIR,
    Component,
    ElectricalNet,
    ImageInfo,
    IRState,
    Pin,
    Point,
)
from pcbparse.output.official_serializer import (
    natural_sort_key,
    official_json_bytes,
    serialize_official,
)


class UnicodeIdentifierTests(unittest.TestCase):
    def circuit(self, *, component_key="U1", pin_numbers=("1", "2")):
        return CircuitIR(
            state=IRState.VALIDATED,
            image=ImageInfo(image_id="unicode", path="unused", width=100, height=100),
            components=[Component(
                internal_id="component", key=component_key, category="Box",
                bbox=BBox(x1=20, y1=20, x2=80, y2=80),
            )],
            pins=[Pin(
                internal_id=f"pin:{index}", component_id="component", number=number,
                point=Point(x=10, y=30 + index * 10),
            ) for index, number in enumerate(pin_numbers)],
            electrical_nets=[ElectricalNet(
                internal_id="net", pins=[f"pin:{index}" for index in range(len(pin_numbers))],
            )],
        )

    def test_natural_order_keeps_ascii_and_unicode_decimal_digits_numeric(self):
        values = ["R１０", "R10", "R٢", "R2", "R01", "R1"]
        expected = ["R01", "R1", "R2", "R٢", "R10", "R１０"]
        self.assertEqual(sorted(values, key=natural_sort_key), expected)
        self.assertEqual(sorted(reversed(values), key=natural_sort_key), expected)

    def test_nondecimal_digits_are_sortable_without_normalizing_the_text(self):
        values = ["①", "²", "₂", "R1①", "R1²", "R1₂"]
        ordered = sorted(values, key=natural_sort_key)
        self.assertCountEqual(ordered, values)
        self.assertEqual(sorted(reversed(values), key=natural_sort_key), ordered)

    def test_component_designator_with_digit_suffix_serializes_unchanged(self):
        for key in ("R1²", "R1①", "R1₂"):
            with self.subTest(key=key):
                output = serialize_official(self.circuit(component_key=key))
                self.assertEqual(list(output["components"]), [key])
                self.assertEqual(output["nets"]["net_1"]["hyperGraph"], f"({key}.1,{key}.2)")
                self.assertIn(key.encode("utf-8"), official_json_bytes(output))

    def test_nondecimal_pin_numbers_serialize_unchanged(self):
        for number in ("²", "①", "₂"):
            with self.subTest(number=number):
                output = serialize_official(self.circuit(pin_numbers=(number,)))
                self.assertEqual(list(output["pins"]["U1"]), [f"pin_{number}"])
                self.assertEqual(output["nets"]["net_1"]["hyperGraph"], f"(U1.{number})")
                self.assertIn(number.encode("utf-8"), official_json_bytes(output))

    def test_nondecimal_internal_ids_do_not_break_serialization(self):
        ir = self.circuit()
        ir.components[0].internal_id = "①"
        ir.pins[0].internal_id = "²"
        for pin in ir.pins:
            pin.component_id = "①"
        ir.electrical_nets[0].pins = [p.internal_id for p in ir.pins]
        self.assertEqual(list(serialize_official(ir)["pins"]["U1"]), ["pin_1", "pin_2"])

    def test_canonical_bytes_ignore_pin_and_net_member_insertion_order(self):
        ir = self.circuit(component_key="U1²", pin_numbers=("10", "2", "²", "①"))
        before = official_json_bytes(serialize_official(ir))
        ir.pins.reverse()
        ir.electrical_nets[0].pins.reverse()
        self.assertEqual(official_json_bytes(serialize_official(ir)), before)


if __name__ == "__main__":
    unittest.main()
