from __future__ import annotations

import unittest

from pcbparse.association.ocr_common import OCRAssociationConfig
from pcbparse.association.pin_component import associate_pin_components
from pcbparse.ir.models import BBox, CircuitIR, Component, ImageInfo, Pin, Point


class PinComponentTests(unittest.TestCase):
    def make_ir(self):
        return CircuitIR(image=ImageInfo(image_id="case", path="image.png", width=200, height=100),
                         components=[Component(internal_id="left", category="R", bbox=BBox(x1=10, y1=10, x2=30, y2=30)),
                                     Component(internal_id="right", category="R", bbox=BBox(x1=50, y1=10, x2=70, y2=30))],
                         pins=[Pin(internal_id="unique", point=Point(x=9, y=20)),
                               Pin(internal_id="tie", point=Point(x=40, y=20)),
                               Pin(internal_id="remote", point=Point(x=190, y=90)),
                               Pin(internal_id="owned", point=Point(x=69, y=20), component_id="left", number="7")])

    def test_unique_boundary_only_existing_owners_and_numbers_preserved(self):
        ir = self.make_ir()
        config = OCRAssociationConfig(pin_component_assignment_enabled=True, pin_component_max_distance_px=20)
        report = associate_pin_components(ir, config)
        self.assertEqual([pin.component_id for pin in ir.pins], ["left", None, None, "left"])
        self.assertEqual([pin.number for pin in ir.pins], [None, None, None, "7"])
        self.assertTrue(all(pin.base is None and pin.orientation == "UNKNOWN" for pin in ir.pins))
        self.assertEqual((report["assigned"], report["ambiguous"], report["unmatched"], report["preserved"]), (1, 1, 1, 1))
        before = [len(pin.evidence) for pin in ir.pins]
        associate_pin_components(ir, config)
        self.assertEqual([len(pin.evidence) for pin in ir.pins], before)

    def test_disabled_by_default_keeps_baseline_pins(self):
        ir = self.make_ir()
        before = ir.model_dump()
        associate_pin_components(ir, OCRAssociationConfig())
        self.assertEqual(ir.model_dump(), before)

    def test_tied_owner_does_not_depend_on_component_list_order(self):
        ir = self.make_ir()
        ir.components.reverse()
        associate_pin_components(ir, OCRAssociationConfig(pin_component_assignment_enabled=True))
        self.assertIsNone(ir.pins[1].component_id)


if __name__ == "__main__":
    unittest.main()
