from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from PIL import Image

from pcbparse.ir.models import Evidence, Pin, Point
from pcbparse.perception.composite_provider import CompositePerceptionProvider, CompositeProviderConfig
from pcbparse.pipeline.orchestrator import PipelineError, infer_one


class EmptyAdapter:
    version = "empty-fixture"
    def predict(self, image, ctx):
        return []
    def recognize(self, views, ctx):
        return []
    def segment(self, views, ctx):
        return []


class WholeImagePin:
    version = "whole-image-fixture"
    def __init__(self):
        self.calls = 0
    def predict(self, component_views, ctx):
        raise AssertionError("full-image provider must not be invoked for each ROI")
    def predict_image(self, image, ctx):
        self.calls += 1
        return [Pin(internal_id="local", point=Point(x=130, y=20),
                    evidence=[Evidence(evidence_id="pin", source="detector", view_id=image.view_id)])]


class FullImagePinIntegrationTests(unittest.TestCase):
    def provider(self, pin):
        empty = EmptyAdapter()
        return CompositePerceptionProvider(component_provider=empty, ocr_provider=empty, pin_provider=pin,
                                           wire_provider=empty, config=CompositeProviderConfig(tile_size=64, tile_overlap=8))

    def test_full_image_pin_detected_once_when_component_detector_is_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "input.png"
            Image.new("RGB", (160, 80), "white").save(image)
            pin = WholeImagePin()
            ir = self.provider(pin).build_ir(image)
            self.assertEqual(pin.calls, 1)
            self.assertEqual(ir.metadata["pin_inference_scope"], "full_image")
            self.assertEqual((ir.pins[0].point.x, ir.pins[0].point.y), (130, 20))
            self.assertIsNone(ir.pins[0].component_id)
            self.assertIsNone(ir.pins[0].number)
            self.assertEqual(ir.pins[0].evidence[0].view_id, "original")

    def test_missing_owner_and_number_cannot_bypass_strict_serializer(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "input.png"
            output = Path(directory) / "output.json"
            Image.new("RGB", (160, 80), "white").save(image)
            with self.assertRaises(PipelineError):
                infer_one(image, output, provider=self.provider(WholeImagePin()),
                          config={"run": {"profile": "pin_package_experiment", "submission_mode": "strict"},
                                  "validation": {"require_zero_errors": True}})
            self.assertFalse(output.exists())

    def test_incorrect_full_image_view_evidence_is_rejected(self):
        class InvalidView(WholeImagePin):
            def predict_image(self, image, ctx):
                pins = super().predict_image(image, ctx)
                pins[0].evidence[0].view_id = "component_roi"
                return pins
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "input.png"
            Image.new("RGB", (160, 80), "white").save(image)
            with self.assertRaisesRegex(ValueError, "unknown source view"):
                self.provider(InvalidView()).build_ir(image)


if __name__ == "__main__":
    unittest.main()
