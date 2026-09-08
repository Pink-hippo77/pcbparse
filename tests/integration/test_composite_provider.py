from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pcbparse.ir.models import BBox, Component, Evidence, Junction, Pin, Point, TextToken
from pcbparse.perception.composite_provider import CompositePerceptionProvider, CompositeProviderConfig
from pcbparse.perception.protocols import WireMask
from pcbparse.graph.physical import build_physical_graph
from pcbparse.validation.graph import validate_graph


class ComponentStub:
    version = "component-stub"

    def predict(self, image, ctx):
        del ctx
        if image.view_id != "tile_000_001":
            return []
        return [Component(internal_id="c1", category="R", bbox=BBox(x1=10, y1=10, x2=30, y2=30))]


class OCRStub:
    version = "ocr-stub"

    def recognize(self, views, ctx):
        del ctx
        view = next((item for item in views if item.view_id == "tile_000_001"), None)
        # Multi-level OCR invokes the same provider for global tiles, component
        # ROIs, and side strips.  A targeted fixture has no detections in a
        # batch that does not contain its target view.
        if view is None:
            return []
        return [
            TextToken(
                internal_id="t1",
                raw_text="R1",
                resolved_text="R1",
                details="designator",
                bbox=BBox(x1=10, y1=0, x2=30, y2=8),
                evidence=[Evidence(evidence_id="t1", source="ocr", view_id=view.view_id)],
            )
        ]


class PinStub:
    version = "pin-stub"

    def predict(self, component_views, ctx):
        del ctx
        view = component_views[0].full_roi
        return [
            Pin(
                internal_id="p1",
                component_id=component_views[0].component_id,
                number="1",
                point=Point(x=0, y=10),
                evidence=[Evidence(evidence_id="p1", source="detector", view_id=view.view_id)],
            )
        ]


class WireStub:
    version = "wire-stub"

    def segment(self, views, ctx):
        del ctx
        view = next(item for item in views if item.view_id == "tile_000_001")
        mask = np.zeros((view.height, view.width), dtype=bool)
        mask[50, 5:25] = True
        return [WireMask(view_id=view.view_id, mask=mask, score=0.9)]


class TopologyWireStub:
    version = "wire-topology-stub"

    def segment(self, views, ctx):
        del ctx
        view = views[0]
        mask = np.zeros((view.height, view.width), dtype=bool)
        mask[40, 10:71] = True
        mask[15:41, 40] = True
        return [WireMask(view_id=view.view_id, mask=mask, score=0.95)]


class DuplicateComponentStub:
    version = "component-duplicate-stub"

    def predict(self, image, ctx):
        del ctx
        # Same local ID from every tile, but distinct original locations.
        return [Component(internal_id="0", category="R", bbox=BBox(x1=5, y1=10, x2=20, y2=25))]


class DuplicateOCRStub:
    version = "ocr-duplicate-stub"

    def recognize(self, views, ctx):
        del ctx
        return [
            TextToken(
                internal_id="0",
                raw_text="R1",
                resolved_text="R1",
                details="designator",
                bbox=BBox(x1=5, y1=2, x2=20, y2=9),
                evidence=[Evidence(evidence_id=f"ocr:{view.view_id}", source="ocr", view_id=view.view_id)],
            )
            for view in views
        ]


class JunctionStub:
    version = "junction-stub"

    def classify(self, views, segments, ctx):
        del segments, ctx
        view = next(item for item in views if item.view_id == "tile_000_001")
        return [
            Junction(
                internal_id="0",
                point=Point(x=10, y=20),
                kind="endpoint",
                evidence=[Evidence(evidence_id="j:0", source="detector", view_id=view.view_id)],
            )
        ]


class CompositeProviderTests(unittest.TestCase):
    def test_default_component_wire_tiles_remain_backward_compatible(self) -> None:
        config = CompositeProviderConfig()

        self.assertEqual(1024, config.tile_size)
        self.assertEqual(160, config.tile_overlap)
        self.assertEqual(512, config.ocr_tile_size)
        self.assertEqual(1024, config.resolved_component_tile_size)
        self.assertEqual(160, config.resolved_component_tile_overlap)
        self.assertEqual(1024, config.resolved_wire_tile_size)
        self.assertEqual(160, config.resolved_wire_tile_overlap)
        self.assertEqual(0, config.component_roi_padding_px)
        self.assertEqual(0, config.resolved_ocr_component_roi_padding_px)
        self.assertEqual(0, config.resolved_pin_component_roi_padding_px)
        self.assertFalse(config.learned_junctions_enabled)

    def test_adaptive_component_fallback_is_image_only_and_traced(self) -> None:
        class AdaptiveComponentStub:
            version = "component-adaptive-stub"

            def __init__(self) -> None:
                self.widths = []

            def predict(self, image, ctx):
                del ctx
                self.widths.append(image.width)
                return [
                    Component(
                        internal_id=f"candidate-{len(self.widths)}",
                        category="R",
                        bbox=BBox(x1=10, y1=10, x2=30, y2=30),
                        score=0.9,
                    )
                ]

        class EmptyOCRStub:
            version = "ocr-empty-stub"

            def recognize(self, views, ctx):
                del views, ctx
                return []

        class EmptyPinStub:
            version = "pin-empty-stub"

            def predict(self, component_views, ctx):
                del component_views, ctx
                return []

        class EmptyWireStub:
            version = "wire-empty-stub"

            def segment(self, views, ctx):
                del views, ctx
                return []

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.png"
            Image.new("RGB", (100, 80), "white").save(path)
            component = AdaptiveComponentStub()
            provider = CompositePerceptionProvider(
                component_provider=component,
                ocr_provider=EmptyOCRStub(),
                pin_provider=EmptyPinStub(),
                wire_provider=EmptyWireStub(),
                config=CompositeProviderConfig(
                    tile_size=100,
                    tile_overlap=0,
                    component_tile_size=100,
                    component_tile_overlap=0,
                    component_adaptive_fallback_enabled=True,
                    component_fallback_tile_size=50,
                    component_fallback_tile_overlap=0,
                    component_fallback_trigger_max_count=1,
                    component_fallback_trigger_max_box_fraction=0.0,
                     component_fallback_trigger_max_bbox_area_fraction=1.0,
                    ocr_tile_size=100,
                    ocr_tile_overlap=0,
                ),
            )

            ir = provider.build_ir(path)
            gate = ir.metadata["component_candidate_gate"]
            fallback = gate["adaptive_fallback"]
            self.assertTrue(fallback["enabled"])
            self.assertTrue(fallback["triggered"])
            self.assertEqual(50, fallback["tile_size"])
            self.assertGreater(fallback["raw_count"], 0)
            self.assertIn(50, component.widths)
            self.assertEqual(1, gate["primary_count"])

    def test_component_candidate_gate_limits_downstream_fanout(self) -> None:
        class ScoredComponentStub:
            version = "component-scored-stub"

            def predict(self, image, ctx):
                del image, ctx
                return [
                    Component(
                        internal_id="high",
                        category="R",
                        bbox=BBox(x1=10, y1=10, x2=30, y2=30),
                        score=0.9,
                    ),
                    Component(
                        internal_id="low",
                        category="R",
                        bbox=BBox(x1=40, y1=10, x2=60, y2=30),
                        score=0.44,
                    ),
                    Component(
                        internal_id="second",
                        category="C",
                        bbox=BBox(x1=70, y1=10, x2=90, y2=30),
                        score=0.8,
                    ),
                ]

        class RecordingOCRStub:
            version = "ocr-recording-stub"

            def __init__(self) -> None:
                self.component_view_count = 0

            def recognize(self, views, ctx):
                del ctx
                self.component_view_count += sum(
                    1
                    for view in views
                    if view.view_id.startswith("component-ocr:")
                    and view.view_id.endswith(":full")
                )
                return []

        class RecordingPinStub:
            version = "pin-recording-stub"

            def __init__(self) -> None:
                self.component_count = 0

            def predict(self, component_views, ctx):
                del ctx
                self.component_count += len(component_views)
                return []

        class EmptyWireStub:
            version = "wire-empty-stub"

            def segment(self, views, ctx):
                del views, ctx
                return []

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.png"
            Image.new("RGB", (100, 80), "white").save(path)
            ocr = RecordingOCRStub()
            pin = RecordingPinStub()
            provider = CompositePerceptionProvider(
                component_provider=ScoredComponentStub(),
                ocr_provider=ocr,
                pin_provider=pin,
                wire_provider=EmptyWireStub(),
                config=CompositeProviderConfig(
                    tile_size=100,
                    tile_overlap=0,
                    ocr_tile_size=100,
                    ocr_tile_overlap=0,
                    component_candidate_min_score=0.5,
                    component_max_candidates=1,
                ),
            )

            ir = provider.build_ir(path)

            gate = ir.metadata["component_candidate_gate"]
            self.assertEqual(3, gate["raw_count"])
            self.assertEqual(1, gate["score_filtered_count"])
            self.assertEqual(2, gate["after_nms_count"])
            self.assertEqual(1, gate["cap_filtered_count"])
            self.assertEqual(1, gate["final_count"])
            self.assertEqual(1, len(ir.components))
            self.assertEqual(1, ocr.component_view_count)
            self.assertEqual(1, pin.component_count)

    def test_failed_learned_junction_is_quarantined_by_default(self) -> None:
        class MustNotRunJunction:
            version = "junction-quarantined-fixture"

            def classify(self, views, segments, ctx):
                del views, segments, ctx
                raise AssertionError("quarantined Junction provider was invoked")

        class EmptyPinStub:
            version = "pin-empty-stub"

            def predict(self, component_views, ctx):
                del component_views, ctx
                return []

        class EmptyWireStub:
            version = "wire-empty-stub"

            def segment(self, views, ctx):
                del views, ctx
                return []

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.png"
            Image.new("RGB", (100, 80), "white").save(path)
            provider = CompositePerceptionProvider(
                component_provider=ComponentStub(),
                ocr_provider=OCRStub(),
                pin_provider=EmptyPinStub(),
                wire_provider=EmptyWireStub(),
                junction_provider=MustNotRunJunction(),
                config=CompositeProviderConfig(
                    tile_size=100,
                    tile_overlap=20,
                    ocr_tile_size=100,
                    ocr_tile_overlap=20,
                ),
            )

            ir = provider.build_ir(path)

            self.assertNotIn("junction", ir.metadata["model_versions"])
            self.assertEqual(
                "junction-quarantined-fixture",
                ir.metadata["quarantined_model_versions"]["junction"],
            )
            self.assertFalse(
                ir.metadata["perception_config"]["learned_junctions_enabled"]
            )

    def test_component_and_wire_tiles_can_be_resolved_independently(self) -> None:
        class RecordingComponentStub:
            version = "component-recording-stub"

            def __init__(self) -> None:
                self.widths = []

            def predict(self, image, ctx):
                del ctx
                self.widths.append(image.width)
                return []

        class EmptyOCRStub:
            version = "ocr-empty-stub"

            def recognize(self, views, ctx):
                del views, ctx
                return []

        class EmptyPinStub:
            version = "pin-empty-stub"

            def predict(self, component_views, ctx):
                del component_views, ctx
                return []

        class RecordingWireStub:
            version = "wire-recording-stub"

            def __init__(self) -> None:
                self.widths = []

            def segment(self, views, ctx):
                del ctx
                self.widths = [view.width for view in views]
                return []

        class RecordingJunctionStub:
            version = "junction-recording-stub"

            def __init__(self) -> None:
                self.widths = []

            def classify(self, views, segments, ctx):
                del segments, ctx
                self.widths = [view.width for view in views]
                return []

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.png"
            Image.new("RGB", (180, 100), "white").save(path)
            component_provider = RecordingComponentStub()
            wire_provider = RecordingWireStub()
            junction_provider = RecordingJunctionStub()
            provider = CompositePerceptionProvider(
                component_provider=component_provider,
                ocr_provider=EmptyOCRStub(),
                pin_provider=EmptyPinStub(),
                wire_provider=wire_provider,
                junction_provider=junction_provider,
                config=CompositeProviderConfig(
                    component_tile_size=80,
                    component_tile_overlap=0,
                    wire_tile_size=200,
                    wire_tile_overlap=0,
                    learned_junctions_enabled=True,
                ),
            )

            ir = provider.build_ir(path)

            self.assertTrue(component_provider.widths)
            self.assertLessEqual(max(component_provider.widths), 80)
            self.assertEqual([180], wire_provider.widths)
            self.assertEqual(wire_provider.widths, junction_provider.widths)
            self.assertEqual(80, ir.metadata["perception_config"]["component_tile_size"])
            self.assertEqual(200, ir.metadata["perception_config"]["wire_tile_size"])

    def test_absolute_component_roi_padding_overrides_smaller_relative_padding(self) -> None:
        class RecordingPinStub(PinStub):
            def __init__(self) -> None:
                self.crop_box = None

            def predict(self, component_views, ctx):
                self.crop_box = component_views[0].full_roi.metadata["crop_box"]
                return super().predict(component_views, ctx)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.png"
            Image.new("RGB", (150, 80), "white").save(path)
            pin_provider = RecordingPinStub()
            provider = CompositePerceptionProvider(
                component_provider=ComponentStub(),
                ocr_provider=OCRStub(),
                pin_provider=pin_provider,
                wire_provider=WireStub(),
                config=CompositeProviderConfig(
                    tile_size=100,
                    tile_overlap=20,
                    ocr_tile_size=100,
                    ocr_tile_overlap=20,
                    component_roi_padding_factor=0.15,
                    component_roi_padding_px=8,
                    wire_minimum_run_px=4,
                ),
            )

            ir = provider.build_ir(path)

            # Restored component bbox is [60, 10, 80, 30].  Relative padding
            # is 3px, so the configured absolute 8px must win.
            self.assertEqual([52, 2, 88, 38], pin_provider.crop_box)
            self.assertEqual(8, ir.metadata["perception_config"]["component_roi_padding_px"])

    def test_ocr_and_pin_component_contexts_are_independent_and_auditable(self) -> None:
        class RecordingOCRStub:
            version = "ocr-recording-stub"

            def __init__(self) -> None:
                self.full_rois = []

            def recognize(self, views, ctx):
                del ctx
                self.full_rois.extend(
                    view for view in views if view.view_id.startswith("component-ocr:")
                    and view.view_id.endswith(":full")
                )
                return []

        class RecordingPinStub(PinStub):
            def __init__(self) -> None:
                self.full_rois = []

            def predict(self, component_views, ctx):
                self.full_rois = [item.full_roi for item in component_views]
                return super().predict(component_views, ctx)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.png"
            Image.new("RGB", (150, 80), "white").save(path)
            ocr_provider = RecordingOCRStub()
            pin_provider = RecordingPinStub()
            provider = CompositePerceptionProvider(
                component_provider=ComponentStub(),
                ocr_provider=ocr_provider,
                pin_provider=pin_provider,
                wire_provider=WireStub(),
                config=CompositeProviderConfig(
                    tile_size=100,
                    tile_overlap=20,
                    ocr_tile_size=100,
                    ocr_tile_overlap=20,
                    component_roi_padding_factor=0.0,
                    ocr_component_roi_padding_px=2,
                    pin_component_roi_padding_px=8,
                ),
            )

            ir = provider.build_ir(path)

            self.assertEqual([[58, 8, 82, 32]], [view.metadata["crop_box"] for view in ocr_provider.full_rois])
            self.assertEqual([[52, 2, 88, 38]], [view.metadata["crop_box"] for view in pin_provider.full_rois])
            self.assertTrue(ocr_provider.full_rois[0].view_id.startswith("component-ocr:"))
            self.assertTrue(pin_provider.full_rois[0].view_id.startswith("component-pin:"))
            self.assertNotEqual(ocr_provider.full_rois[0].view_id, pin_provider.full_rois[0].view_id)
            self.assertEqual(2, ir.metadata["perception_config"]["ocr_component_roi_padding_px"])
            self.assertEqual(8, ir.metadata["perception_config"]["pin_component_roi_padding_px"])
            self.assertEqual(
                [ocr_provider.full_rois[0].view_id],
                ir.metadata["ocr_component_views"],
            )
            self.assertEqual(
                [pin_provider.full_rois[0].view_id],
                ir.metadata["pin_component_views"],
            )

    def test_cross_tile_and_roi_coordinates_restore_to_original(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.png"
            Image.new("RGB", (150, 80), "white").save(path)
            provider = CompositePerceptionProvider(
                component_provider=ComponentStub(),
                ocr_provider=OCRStub(),
                pin_provider=PinStub(),
                wire_provider=WireStub(),
                config=CompositeProviderConfig(
                    tile_size=100,
                    tile_overlap=20,
                    ocr_tile_size=100,
                    ocr_tile_overlap=20,
                    wire_minimum_run_px=4,
                ),
            )
            ir = provider.build_ir(path)
            self.assertEqual(1, len(ir.components))
            # Second tile starts at x=50 to cover the final 100 pixels.
            self.assertEqual(60.0, ir.components[0].bbox.x1)
            self.assertEqual(60.0, ir.texts[0].bbox.x1)
            self.assertGreaterEqual(ir.pins[0].point.x, 50.0)
            self.assertTrue(any(segment.start.x >= 55.0 for segment in ir.segments))
            self.assertEqual("wire-stub", ir.metadata["model_versions"]["wire"])
            self.assertEqual(100, ir.metadata["perception_config"]["tile_size"])
            self.assertEqual(20, ir.metadata["perception_config"]["tile_overlap"])
            self.assertEqual(100, ir.metadata["perception_config"]["ocr_tile_size"])
            for collection, kind in (
                (ir.components, "component"),
                (ir.texts, "text"),
                (ir.pins, "pin"),
                (ir.segments, "segment"),
            ):
                self.assertTrue(collection, kind)
                binding = collection[0].evidence[0].metadata[
                    "local_to_global_binding"
                ]
                self.assertEqual("image_view_local_to_original_v1", binding["contract"])
                self.assertEqual(kind, binding["object_kind"])
                matrix = binding["local_to_original_matrix"]
                self.assertEqual([0.0, 0.0, 1.0], matrix[2])
                self.assertEqual(1.0, matrix[0][0])
                self.assertEqual(1.0, matrix[1][1])
                self.assertEqual(binding["view_id"], collection[0].evidence[0].view_id)

    def test_disconnected_mask_runs_remain_separate_physical_nets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.png"
            Image.new("RGB", (150, 80), "white").save(path)
            provider = CompositePerceptionProvider(
                component_provider=ComponentStub(),
                ocr_provider=OCRStub(),
                pin_provider=PinStub(),
                wire_provider=WireStub(),
                config=CompositeProviderConfig(
                    tile_size=100,
                    tile_overlap=20,
                    ocr_tile_size=100,
                    ocr_tile_overlap=20,
                    wire_minimum_run_px=4,
                ),
            )
            ir = provider.build_ir(path)
            horizontal = [segment for segment in ir.segments if segment.channel == "horizontal"]
            self.assertEqual(len(horizontal), len({segment.source_polyline_id for segment in horizontal}))
            nets, _ = build_physical_graph(ir)
            self.assertGreaterEqual(len(nets), 2)

    def test_tile_local_ids_are_namespaced_and_junction_is_restored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.png"
            Image.new("RGB", (150, 80), "white").save(path)
            provider = CompositePerceptionProvider(
                component_provider=DuplicateComponentStub(),
                ocr_provider=DuplicateOCRStub(),
                pin_provider=PinStub(),
                wire_provider=WireStub(),
                junction_provider=JunctionStub(),
                config=CompositeProviderConfig(
                    tile_size=100,
                    tile_overlap=20,
                    ocr_tile_size=100,
                    ocr_tile_overlap=20,
                    wire_minimum_run_px=4,
                    learned_junctions_enabled=True,
                ),
            )

            ir = provider.build_ir(path)

            self.assertEqual(len(ir.components), len({item.internal_id for item in ir.components}))
            self.assertEqual(len(ir.texts), len({item.internal_id for item in ir.texts}))
            self.assertTrue(all(item.internal_id.startswith("component:") for item in ir.components))
            self.assertTrue(all(item.internal_id.startswith("text:") for item in ir.texts))
            self.assertEqual(60.0, ir.junctions[0].point.x)
            self.assertEqual("0", ir.junctions[0].evidence[0].metadata["provider_local_id"])

    def test_missing_source_view_evidence_fails_closed(self) -> None:
        class MissingViewOCR:
            version = "missing-view"

            def recognize(self, views, ctx):
                del views, ctx
                return [
                    TextToken(
                        internal_id="t",
                        raw_text="R1",
                        bbox=BBox(x1=0, y1=0, x2=10, y2=8),
                        evidence=[],
                    )
                ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.png"
            Image.new("RGB", (150, 80), "white").save(path)
            provider = CompositePerceptionProvider(
                component_provider=ComponentStub(),
                ocr_provider=MissingViewOCR(),
                pin_provider=PinStub(),
                wire_provider=WireStub(),
                config=CompositeProviderConfig(tile_size=100, tile_overlap=20),
            )
            with self.assertRaisesRegex(ValueError, "no source view evidence"):
                provider.build_ir(path)

    def test_pin_component_owner_must_match_its_source_component_view(self) -> None:
        class WrongOwnerPin:
            version = "wrong-owner"

            def predict(self, component_views, ctx):
                view = component_views[0].full_roi
                return [
                    Pin(
                        internal_id="wrong-owner-pin",
                        component_id="component:not-the-view-owner",
                        number="1",
                        point=Point(x=1, y=1),
                        evidence=[
                            Evidence(
                                evidence_id="wrong-owner",
                                source="detector",
                                view_id=view.view_id,
                            )
                        ],
                    )
                ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.png"
            Image.new("RGB", (150, 80), "white").save(path)
            provider = CompositePerceptionProvider(
                component_provider=ComponentStub(),
                ocr_provider=OCRStub(),
                pin_provider=WrongOwnerPin(),
                wire_provider=WireStub(),
                config=CompositeProviderConfig(
                    tile_size=100,
                    tile_overlap=20,
                    ocr_tile_size=100,
                    ocr_tile_overlap=20,
                ),
            )

            with self.assertRaisesRegex(ValueError, "component owner.*conflicts"):
                provider.build_ir(path)

    def test_skeleton_t_junction_reaches_graph_validator_with_three_branches(self) -> None:
        class NoComponents:
            version = "none"

            def predict(self, image, ctx):
                return []

        class NoOCR:
            version = "none"

            def recognize(self, views, ctx):
                return []

        class NoPins:
            version = "none"

            def predict(self, component_views, ctx):
                return []

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.png"
            Image.new("RGB", (80, 80), "white").save(path)
            provider = CompositePerceptionProvider(
                component_provider=NoComponents(),
                ocr_provider=NoOCR(),
                pin_provider=NoPins(),
                wire_provider=TopologyWireStub(),
                config=CompositeProviderConfig(
                    tile_size=80,
                    tile_overlap=0,
                    wire_minimum_run_px=2,
                ),
            )

            ir = provider.build_ir(path)
            junction = next(item for item in ir.junctions if item.kind == "t_junction")
            self.assertGreaterEqual(len(set(junction.segment_ids)), 3)
            ir.physical_nets, _ = build_physical_graph(ir)
            errors = [item for item in validate_graph(ir) if item.severity == "ERROR"]
            self.assertEqual([], errors)

    def test_vectorized_diagonal_slopes_receive_distinct_connectivity_channels(self) -> None:
        class NoComponents:
            version = "none"

            def predict(self, image, ctx):
                return []

        class NoOCR:
            version = "none"

            def recognize(self, views, ctx):
                return []

        class NoPins:
            version = "none"

            def predict(self, component_views, ctx):
                return []

        class DiagonalWireStub:
            version = "diagonal-wire-stub"

            def segment(self, views, ctx):
                view = views[0]
                mask = np.zeros((view.height, view.width), dtype=bool)
                for offset in range(12):
                    mask[8 + offset, 8 + offset] = True
                    mask[36 - offset, 8 + offset] = True
                return [WireMask(view_id=view.view_id, mask=mask, score=0.95)]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.png"
            Image.new("RGB", (48, 48), "white").save(path)
            provider = CompositePerceptionProvider(
                component_provider=NoComponents(),
                ocr_provider=NoOCR(),
                pin_provider=NoPins(),
                wire_provider=DiagonalWireStub(),
                config=CompositeProviderConfig(
                    tile_size=48,
                    tile_overlap=0,
                    wire_minimum_run_px=2,
                ),
            )

            ir = provider.build_ir(path)

            self.assertTrue(
                {"diagonal_positive", "diagonal_negative"}.issubset(
                    {segment.channel for segment in ir.segments}
                )
            )


if __name__ == "__main__":
    unittest.main()