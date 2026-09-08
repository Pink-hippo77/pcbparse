from __future__ import annotations

import unittest
from pathlib import Path

from pcbparse.io import detect_image_format, load_image, make_component_view
from pcbparse.ir.models import IRState, Point
from pcbparse.output.official_serializer import serialize_official
from pcbparse.perception import DatasetGoldenProvider, find_sidecar
from pcbparse.perception.golden_provider import GoldenPolicy


WORKSPACE = Path(__file__).resolve().parents[2]
DATASET = WORKSPACE / "Dataset300" / "Dataset300"


class DatasetGoldenProviderIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.provider = DatasetGoldenProvider()

    def assert_ir_coordinates_in_bounds(self, ir) -> None:
        width = float(ir.image.width)
        height = float(ir.image.height)

        def assert_point(point, label: str) -> None:
            self.assertGreaterEqual(point.x, 0.0, label)
            self.assertLessEqual(point.x, width, label)
            self.assertGreaterEqual(point.y, 0.0, label)
            self.assertLessEqual(point.y, height, label)

        def assert_bbox(bbox, label: str) -> None:
            self.assertGreaterEqual(bbox.x1, 0.0, label)
            self.assertLessEqual(bbox.x2, width, label)
            self.assertGreaterEqual(bbox.y1, 0.0, label)
            self.assertLessEqual(bbox.y2, height, label)
            self.assertLessEqual(bbox.x1, bbox.x2, label)
            self.assertLessEqual(bbox.y1, bbox.y2, label)

        for component in ir.components:
            assert_bbox(component.bbox, component.internal_id)
        for pin in ir.pins:
            assert_point(pin.point, pin.internal_id)
            if pin.base is not None:
                assert_point(pin.base, pin.internal_id)
        for text in ir.texts:
            assert_bbox(text.bbox, text.internal_id)
        for segment in ir.segments:
            assert_point(segment.start, segment.internal_id)
            assert_point(segment.end, segment.internal_id)
            self.assertNotEqual(segment.start, segment.end, segment.internal_id)
        for label in ir.net_labels:
            if label.point is not None:
                assert_point(label.point, label.internal_id)

    def test_real_sample_0001_resolves_references_and_preserves_provenance(self) -> None:
        sample = DATASET / "0001"
        image = sample / "0001-KiCad.png"
        sidecar = sample / "0001-KiCad_CVJsonStd.json"
        self.assertEqual(find_sidecar(sample), sidecar.resolve())
        self.assertEqual(find_sidecar(image), sidecar.resolve())

        ir = self.provider.build_ir(image)
        self.assertEqual(ir.state, IRState.PERCEIVED)
        self.assertEqual(ir.image.actual_format, "PNG")
        self.assertEqual(ir.image.source_domain, "kicad")
        self.assertEqual((len(ir.components), len(ir.pins), len(ir.texts)), (91, 224, 486))
        self.assertEqual(ir.physical_nets, [])
        self.assertEqual(ir.electrical_nets, [])

        components = {item.source_object_id: item for item in ir.components}
        pins = {item.source_object_id: item for item in ir.pins}
        texts = {item.source_object_id: item for item in ir.texts}
        labels = {item.evidence[0].metadata["source_edge_id"]: item for item in ir.net_labels}

        self.assertEqual(components["C_1"].key, "U1")
        self.assertEqual(components["C_1"].name, "scenario_1_1_2-KiCAD_original")
        self.assertEqual(components["C_1"].source_object_id, "C_1")
        self.assertEqual(components["C_3"].key, "C6")
        # Two official value tokens (capacitance and voltage rating) are
        # ambiguous.  The conservative provider keeps both as evidence and
        # does not guess which one belongs in Component.value.
        self.assertEqual(components["C_3"].value, "")
        self.assertEqual(
            components["C_3"].evidence[0].metadata["resolution"]["value_candidates"],
            ["4.7uF", "6.3V"],
        )
        self.assertIsNone(components["C_27"].key)

        pin = pins["P_102"]
        self.assertEqual(pin.component_id, "component:C_1")
        self.assertEqual((pin.number, pin.name, pin.orientation), ("7", "GPI09", "R"))
        self.assertEqual(texts["T_129"].pin_id, "pin:P_102")
        self.assertEqual(texts["T_128"].pin_id, "pin:P_102")
        self.assertIsNone(pins["P_1"].number)
        self.assertEqual(labels["E_1"].resolved_text, "A1")
        self.assertIsNone(labels["E_1"].attached_physical_net_id)
        segment_ids = {segment.internal_id for segment in ir.segments}
        self.assertTrue(pin.edge_ids)
        self.assertTrue(set(pin.edge_ids).issubset(segment_ids))
        self.assertTrue(
            all(set(candidate.edge_ids).issubset(segment_ids) for candidate in ir.pins)
        )

        missing_reference = [
            issue for issue in ir.diagnostics if issue.code == "DATASET_EDGE_PIN_REFERENCE_MISSING"
        ]
        self.assertTrue(any("P_193" in issue.object_ids for issue in missing_reference))
        hints = ir.metadata["dataset_graph_hints"]
        self.assertFalse(hints["authoritative"])
        self.assertTrue(hints["rebuild_required"])
        self.assertTrue(hints["physical_nets"])
        self.assertTrue(hints["electrical_nets"])
        self.assertFalse(any(item.code.startswith("DEBUG.") for item in ir.diagnostics))
        self.assert_ir_coordinates_in_bounds(ir)

    def test_safe_unmarked_designator_recovery_is_dataset_backed(self) -> None:
        ir = self.provider.build_ir(DATASET / "0016")
        component = next(item for item in ir.components if item.source_object_id == "C_20")
        self.assertEqual(component.key, "LED1")
        resolution = component.evidence[0].metadata["resolution"]
        self.assertEqual(resolution["designator_source"], "associated-unmarked-compatible")
        self.assertIn("LED1", resolution["designator_candidates"])
        self.assertTrue(
            any(
                item.code == "DATASET_DESIGNATOR_RECOVERED"
                and component.internal_id in item.object_ids
                for item in ir.diagnostics
            )
        )

    def test_dataset_debug_fills_required_ids_without_claiming_ocr(self) -> None:
        strict = DatasetGoldenProvider(GoldenPolicy.STRICT).build_ir(DATASET / "0001")
        self.assertTrue(any(component.key is None for component in strict.components))
        self.assertTrue(any(pin.number is None for pin in strict.pins))
        active_only = DatasetGoldenProvider(GoldenPolicy.ACTIVE_ONLY).build_ir(DATASET / "0001")
        self.assertEqual(active_only.metadata["provider"]["policy"], "active_only")
        self.assertEqual(
            [component.key for component in active_only.components],
            [component.key for component in strict.components],
        )
        self.assertEqual(
            [pin.number for pin in active_only.pins],
            [pin.number for pin in strict.pins],
        )

        debug = DatasetGoldenProvider("dataset_debug").build_ir(DATASET / "0001")
        component_keys = [component.key for component in debug.components]
        self.assertTrue(all(component_keys))
        self.assertEqual(len(component_keys), len(set(component_keys)))
        self.assertTrue(all(pin.number for pin in debug.pins))
        by_component: dict[str | None, set[str]] = {}
        for pin in debug.pins:
            numbers = by_component.setdefault(pin.component_id, set())
            self.assertNotIn(pin.number, numbers)
            numbers.add(pin.number)

        components = {item.source_object_id: item for item in debug.components}
        pins = {item.source_object_id: item for item in debug.pins}
        self.assertEqual(components["C_27"].key, "DATASET_C_27")
        self.assertEqual(pins["P_1"].number, "DATASET_P_1")
        component_surrogate = components["C_27"].evidence[0].metadata["debug_surrogate"]
        pin_surrogate = pins["P_1"].evidence[0].metadata["debug_surrogate"]
        self.assertFalse(component_surrogate["is_ocr_evidence"])
        self.assertFalse(pin_surrogate["is_ocr_evidence"])
        self.assertIsNone(component_surrogate["original_value"])
        self.assertIsNone(pin_surrogate["original_value"])
        codes = {item.code for item in debug.diagnostics}
        self.assertIn("DEBUG.SURROGATE_COMPONENT_KEY", codes)
        self.assertIn("DEBUG.SURROGATE_PIN_NUMBER", codes)
        self.assertFalse(debug.metadata["provider"]["submission_safe"])

        # Prove the debug-only identifiers satisfy the official serializer's
        # key and pin-number container constraints for this real sample.
        debug.state = IRState.VALIDATED
        output = serialize_official(debug, WORKSPACE / "configs" / "official_schema.yaml")
        self.assertEqual(len(output["components"]), len(debug.components))
        self.assertEqual(sum(len(group) for group in output["pins"].values()), len(debug.pins))

    def test_dataset_debug_replaces_real_collisions_transparently(self) -> None:
        strict_components = DatasetGoldenProvider().build_ir(DATASET / "0088")
        strict_keys = [component.key for component in strict_components.components]
        self.assertLess(len(set(strict_keys)), len(strict_keys))

        debug_components = DatasetGoldenProvider(GoldenPolicy.DATASET_DEBUG).build_ir(
            DATASET / "0088"
        )
        debug_keys = [component.key for component in debug_components.components]
        self.assertEqual(len(debug_keys), len(set(debug_keys)))
        replaced_components = [
            component
            for component in debug_components.components
            if component.evidence[0].metadata.get("debug_surrogate", {}).get("reason")
            == "collision"
        ]
        self.assertTrue(replaced_components)
        self.assertTrue(all(component.key.startswith("DATASET_C_") for component in replaced_components))
        self.assertTrue(
            any(
                item.code == "DEBUG.SURROGATE_COMPONENT_COLLISION"
                for item in debug_components.diagnostics
            )
        )

        strict_pins = DatasetGoldenProvider().build_ir(DATASET / "0006")
        strict_groups: dict[tuple[str | None, str | None], int] = {}
        for pin in strict_pins.pins:
            key = (pin.component_id, pin.number)
            if pin.number:
                strict_groups[key] = strict_groups.get(key, 0) + 1
        self.assertTrue(any(count > 1 for count in strict_groups.values()))

        debug_pins = DatasetGoldenProvider(GoldenPolicy.DATASET_DEBUG).build_ir(DATASET / "0006")
        debug_groups: set[tuple[str | None, str | None]] = set()
        for pin in debug_pins.pins:
            key = (pin.component_id, pin.number)
            self.assertNotIn(key, debug_groups)
            debug_groups.add(key)
        collided_pin = next(
            pin
            for pin in debug_pins.pins
            if pin.evidence[0].metadata.get("debug_surrogate", {}).get("reason") == "collision"
        )
        surrogate = collided_pin.evidence[0].metadata["debug_surrogate"]
        self.assertTrue(collided_pin.number.startswith("DATASET_P_"))
        self.assertIsNotNone(surrogate["original_value"])
        self.assertFalse(surrogate["is_ocr_evidence"])
        self.assertTrue(
            any(item.code == "DEBUG.SURROGATE_PIN_COLLISION" for item in debug_pins.diagnostics)
        )

    def test_dataset_debug_excludes_raw_orphans_without_changing_strict_policy(self) -> None:
        strict = DatasetGoldenProvider(GoldenPolicy.STRICT).build_ir(DATASET / "0004")
        orphan_ids = {
            pin.internal_id for pin in strict.pins if pin.orphan or pin.component_id is None
        }
        self.assertEqual(2, len(orphan_ids))

        debug = DatasetGoldenProvider(GoldenPolicy.DATASET_DEBUG).build_ir(DATASET / "0004")
        self.assertTrue(orphan_ids.isdisjoint({pin.internal_id for pin in debug.pins}))
        excluded = {
            object_id
            for item in debug.diagnostics
            if item.code == "DEBUG.EXCLUDED_ORPHAN_PIN"
            for object_id in item.object_ids
        }
        self.assertEqual(orphan_ids, excluded)

    def test_dataset_debug_escapes_hypergraph_delimiter_values(self) -> None:
        strict = DatasetGoldenProvider(GoldenPolicy.STRICT).build_ir(DATASET / "0074")
        self.assertTrue(
            any(
                any(character in (component.key or "") for character in ".,()")
                for component in strict.components
            )
            or any(
                any(character in (pin.number or "") for character in ".,()")
                for pin in strict.pins
            )
        )

        debug = DatasetGoldenProvider(GoldenPolicy.DATASET_DEBUG).build_ir(DATASET / "0074")
        identifiers = [component.key or "" for component in debug.components]
        identifiers.extend(pin.number or "" for pin in debug.pins)
        self.assertFalse(any(any(character in value for character in ".,()") for value in identifiers))
        self.assertTrue(
            any(
                item.code in {
                    "DEBUG.SURROGATE_COMPONENT_FORMAT",
                    "DEBUG.SURROGATE_PIN_FORMAT",
                }
                for item in debug.diagnostics
            )
        )

    def test_deceptive_png_suffixes_use_magic_byte_decoders(self) -> None:
        cases = {
            "0016": ("0016-Datasheet.png", "JPEG"),
            "0202": ("001_01_01.png", "WEBP"),
        }
        for sample_id, (filename, actual_format) in cases.items():
            with self.subTest(sample_id=sample_id):
                image = DATASET / sample_id / filename
                self.assertEqual(image.suffix.lower(), ".png")
                self.assertEqual(detect_image_format(image), actual_format)
                view = load_image(image)
                self.assertEqual(view.actual_format, actual_format)
                self.assertEqual(view.transform.max_roundtrip_error(view.width, view.height), 0.0)

                ir = self.provider.build_ir(image.parent)
                self.assertEqual(ir.image.actual_format, actual_format)
                self.assertEqual((ir.image.width, ir.image.height), view.size)
                self.assert_ir_coordinates_in_bounds(ir)

    def test_component_views_keep_an_explicit_invertible_transform_chain(self) -> None:
        image = DATASET / "0001" / "0001-KiCad.png"
        image_view = load_image(image)
        ir = self.provider.build_ir(image)
        component = next(item for item in ir.components if item.source_object_id == "C_3")
        component_view = make_component_view(image_view, component, padding=8.0)

        self.assertGreaterEqual(len(component_view.transform.transforms), 2)
        center_in_roi = component_view.full_roi.to_view(component.bbox.center)
        restored = component_view.full_roi.to_original(center_in_roi)
        self.assertAlmostEqual(restored.x, component.bbox.center.x)
        self.assertAlmostEqual(restored.y, component.bbox.center.y)
        self.assertEqual(set(component_view.side_strips), {"L", "R", "T", "B"})
        for strip in component_view.side_strips.values():
            original = strip.to_original(Point(x=0.0, y=0.0))
            roundtrip = strip.to_view(original)
            self.assertAlmostEqual(roundtrip.x, 0.0)
            self.assertAlmostEqual(roundtrip.y, 0.0)


if __name__ == "__main__":
    unittest.main()
