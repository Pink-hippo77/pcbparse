from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pcbparse.ir.models import BBox, Component, Evidence, Junction, Pin, Point, TextToken
from pcbparse.perception.composite_provider import (
    CompositePerceptionProvider,
    CompositeProviderConfig,
    _class_aware_nms,
    _filter_unconfirmed_internal_border_components,
    _recover_targeted_components,
    _refine_internal_border_components,
    _restore_components,
)
from pcbparse.io.image_loader import load_image, tile_image_view
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
    def test_targeted_small_object_recovery_batches_and_filters_categories(self) -> None:
        class RecoveryStub:
            version = "component-targeted-recovery-stub"

            def __init__(self) -> None:
                self.batch_sizes: list[int] = []

            def predict(self, image, ctx):
                del image, ctx
                raise AssertionError("batch recovery should use predict_many")

            def predict_many(self, images, ctx):
                del ctx
                self.batch_sizes.append(len(images))
                return [
                    [
                        Component(
                            internal_id=f"switch-{index}",
                            category="Switch",
                            bbox=BBox(x1=40, y1=40, x2=70, y2=55),
                            score=0.8,
                            evidence=[
                                Evidence(
                                    evidence_id=f"switch-{index}",
                                    source="detector",
                                    view_id=image.view_id,
                                )
                            ],
                        ),
                        Component(
                            internal_id=f"ignored-c-{index}",
                            category="C",
                            bbox=BBox(x1=90, y1=40, x2=120, y2=60),
                            score=0.9,
                            evidence=[
                                Evidence(
                                    evidence_id=f"ignored-c-{index}",
                                    source="detector",
                                    view_id=image.view_id,
                                )
                            ],
                        ),
                    ]
                    for index, image in enumerate(images)
                ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.png"
            Image.new("RGB", (800, 256), "white").save(path)
            original = load_image(path)
            provider = RecoveryStub()
            recovered, stats = _recover_targeted_components(
                provider,
                [],
                original,
                None,
                enabled=True,
                categories=("Switch",),
                tile_size=256,
                tile_overlap=32,
                min_dimension_px=768,
                batch_size=3,
                min_score=0.2,
                internal_border_margin_px=32.0,
                nms_iou=0.55,
                prefer_non_border=False,
                center_preference_score_bonus=0.05,
            )

        self.assertEqual([3, 1], provider.batch_sizes)
        self.assertEqual(4, stats["tile_count"])
        self.assertEqual(2, stats["inference_call_count"])
        self.assertEqual(8, stats["raw_candidate_count"])
        self.assertEqual(4, stats["targeted_candidate_count"])
        self.assertTrue(stats["triggered"])
        self.assertTrue(recovered)
        self.assertEqual({"Switch"}, {item.category for item in recovered})

    def test_internal_border_refinement_batches_centred_crops_when_supported(self) -> None:
        class BatchRefinementStub:
            version = "component-batch-refinement-stub"

            def __init__(self) -> None:
                self.batch_view_ids: list[list[str]] = []

            def predict(self, image, ctx):
                del image, ctx
                raise AssertionError("sequential prediction should not be used")

            def predict_many(self, images, ctx):
                del ctx
                self.batch_view_ids.append([image.view_id for image in images])
                return [[] for _ in images]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.png"
            Image.new("RGB", (1600, 512), "white").save(path)
            original = load_image(path)
            candidates = [
                Component(
                    internal_id=f"candidate-{index}",
                    category="Box",
                    bbox=box,
                    score=score,
                    evidence=[
                        Evidence(
                            evidence_id=f"candidate-{index}",
                            source="detector",
                            metadata={"touches_internal_tile_border": True},
                        )
                    ],
                )
                for index, (box, score) in enumerate(
                    [
                        (BBox(x1=480, y1=80, x2=512, y2=150), 0.9),
                        (BBox(x1=930, y1=200, x2=960, y2=280), 0.8),
                    ]
                )
            ]
            provider = BatchRefinementStub()

            refined, stats = _refine_internal_border_components(
                provider,
                candidates,
                original,
                None,
                enabled=True,
                categories=("Box",),
                tile_size=512,
                match_iou=0.5,
                max_candidates=8,
                min_score=0.2,
                internal_border_margin_px=32.0,
                batch_size=8,
            )

        self.assertEqual(
            [["component-refine:0000", "component-refine:0001"]],
            provider.batch_view_ids,
        )
        self.assertEqual(2, stats["attempted_count"])
        self.assertEqual(candidates, refined)

    def test_internal_border_refinement_replaces_truncated_box_from_centred_crop(self) -> None:
        class RefinementComponentStub:
            version = "component-refinement-stub"

            def __init__(self) -> None:
                self.views = []

            def predict(self, image, ctx):
                del ctx
                self.views.append(image)
                return [
                    Component(
                        internal_id="refined",
                        category="Box",
                        bbox=BBox(x1=230, y1=90, x2=310, y2=170),
                        score=0.91,
                        evidence=[
                            Evidence(
                                evidence_id="refined",
                                source="detector",
                                view_id=image.view_id,
                            )
                        ],
                    )
                ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.png"
            Image.new("RGB", (960, 512), "white").save(path)
            original = load_image(path)
            provider = RefinementComponentStub()
            truncated = Component(
                internal_id="truncated",
                category="Box",
                bbox=BBox(x1=480, y1=100, x2=512, y2=160),
                score=0.88,
                evidence=[
                    Evidence(
                        evidence_id="truncated",
                        source="detector",
                        view_id="tile_000_000",
                        metadata={"touches_internal_tile_border": True},
                    )
                ],
            )

            refined, stats = _refine_internal_border_components(
                provider,
                [truncated],
                original,
                None,
                enabled=True,
                categories=("Box",),
                tile_size=512,
                match_iou=0.5,
                max_candidates=8,
                min_score=0.2,
                internal_border_margin_px=32.0,
            )

            self.assertEqual(1, len(provider.views))
            self.assertEqual([240, 0, 752, 512], provider.views[0].metadata["crop_box"])
            self.assertEqual(BBox(x1=470, y1=90, x2=550, y2=170), refined[0].bbox)
            self.assertTrue(
                refined[0].evidence[0].metadata["internal_border_refinement"]
            )
            self.assertEqual(
                {
                    "eligible_count": 1,
                    "attempted_count": 1,
                    "raw_refinement_count": 1,
                    "matched_count": 1,
                    "replaced_count": 1,
                },
                stats,
            )

    def test_internal_border_refinement_is_recall_safe_when_no_match_exists(self) -> None:
        class EmptyRefinementStub:
            version = "component-refinement-empty-stub"

            def predict(self, image, ctx):
                del image, ctx
                return []

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.png"
            Image.new("RGB", (960, 512), "white").save(path)
            original = load_image(path)
            candidate = Component(
                internal_id="candidate",
                category="Box",
                bbox=BBox(x1=480, y1=100, x2=512, y2=160),
                score=0.88,
                evidence=[
                    Evidence(
                        evidence_id="candidate",
                        source="detector",
                        metadata={"touches_internal_tile_border": True},
                    )
                ],
            )

            refined, stats = _refine_internal_border_components(
                EmptyRefinementStub(),
                [candidate],
                original,
                None,
                enabled=True,
                categories=("Box",),
                tile_size=512,
                match_iou=0.5,
                max_candidates=8,
                min_score=0.2,
                internal_border_margin_px=32.0,
            )

            self.assertIs(candidate, refined[0])
            self.assertEqual(0, stats["replaced_count"])

    def test_component_center_preferred_nms_only_changes_overlapping_tie_break(self) -> None:
        border = Component(
            internal_id="border",
            category="R",
            bbox=BBox(x1=100, y1=100, x2=150, y2=150),
            score=0.90,
            evidence=[
                Evidence(
                    evidence_id="border",
                    source="detector",
                    metadata={"touches_internal_tile_border": True},
                )
            ],
        )
        centered = Component(
            internal_id="centered",
            category="R",
            bbox=BBox(x1=102, y1=102, x2=152, y2=152),
            score=0.80,
            evidence=[
                Evidence(
                    evidence_id="centered",
                    source="detector",
                    metadata={"touches_internal_tile_border": False},
                )
            ],
        )
        separate = Component(
            internal_id="separate",
            category="R",
            bbox=BBox(x1=300, y1=100, x2=350, y2=150),
            score=0.70,
        )

        default = _class_aware_nms([border, centered, separate], 0.55)
        preferred = _class_aware_nms(
            [border, centered, separate],
            0.55,
            prefer_non_border=True,
        )
        conservative = _class_aware_nms(
            [border, centered, separate],
            0.55,
            prefer_non_border=True,
            center_preference_score_bonus=0.05,
        )

        self.assertEqual({"border", "separate"}, {item.internal_id for item in default})
        self.assertEqual(
            {"centered", "separate"},
            {item.internal_id for item in preferred},
        )
        self.assertEqual(
            {"border", "separate"},
            {item.internal_id for item in conservative},
        )

    def test_component_restore_marks_only_internal_tile_edges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.png"
            Image.new("RGB", (960, 512), "white").save(path)
            original = load_image(path)
            left_tile, right_tile = tile_image_view(
                original,
                tile_size=512,
                overlap=64,
            )
            self.assertEqual([0, 0, 512, 512], left_tile.metadata["crop_box"])
            self.assertEqual([448, 0, 960, 512], right_tile.metadata["crop_box"])

            internal_edge = Component(
                internal_id="internal",
                category="R",
                bbox=BBox(x1=470, y1=100, x2=505, y2=150),
                evidence=[
                    Evidence(
                        evidence_id="internal",
                        source="detector",
                        view_id=left_tile.view_id,
                    )
                ],
            )
            external_edge = Component(
                internal_id="external",
                category="R",
                bbox=BBox(x1=470, y1=100, x2=505, y2=150),
                evidence=[
                    Evidence(
                        evidence_id="external",
                        source="detector",
                        view_id=right_tile.view_id,
                    )
                ],
            )

            restored_internal = _restore_components(
                [internal_edge],
                left_tile,
                original_size=original.size,
                internal_border_margin_px=32.0,
            )[0]
            restored_external = _restore_components(
                [external_edge],
                right_tile,
                original_size=original.size,
                internal_border_margin_px=32.0,
            )[0]

            self.assertTrue(
                restored_internal.evidence[0].metadata[
                    "touches_internal_tile_border"
                ]
            )
            self.assertEqual(
                7.0,
                restored_internal.evidence[0].metadata[
                    "internal_tile_border_distance_px"
                ],
            )
            self.assertFalse(
                restored_external.evidence[0].metadata[
                    "touches_internal_tile_border"
                ]
            )

    def test_internal_border_confirmation_targets_only_configured_categories(self) -> None:
        def candidate(
            internal_id: str,
            category: str,
            bbox: BBox,
            view_id: str,
            *,
            border: bool,
        ) -> Component:
            return Component(
                internal_id=internal_id,
                category=category,
                bbox=bbox,
                score=0.8,
                evidence=[
                    Evidence(
                        evidence_id=internal_id,
                        source="detector",
                        view_id=view_id,
                        metadata={"touches_internal_tile_border": border},
                    )
                ],
            )

        supported_a = candidate(
            "supported-a", "Box", BBox(x1=10, y1=10, x2=50, y2=50), "tile-a", border=True
        )
        supported_b = candidate(
            "supported-b", "Box", BBox(x1=12, y1=12, x2=52, y2=52), "tile-b", border=False
        )
        unsupported = candidate(
            "unsupported", "Box", BBox(x1=100, y1=10, x2=140, y2=50), "tile-a", border=True
        )
        untargeted = candidate(
            "untargeted", "R", BBox(x1=200, y1=10, x2=240, y2=50), "tile-a", border=True
        )

        kept = _filter_unconfirmed_internal_border_components(
            [supported_a, supported_b, unsupported, untargeted],
            enabled=True,
            categories=("Box",),
            confirmation_iou=0.30,
        )

        self.assertEqual(
            {"supported-a", "supported-b", "untargeted"},
            {item.internal_id for item in kept},
        )

        truncated_a = candidate(
            "truncated-a", "Box", BBox(x1=0, y1=0, x2=40, y2=100), "tile-a", border=True
        )
        truncated_b = candidate(
            "truncated-b", "Box", BBox(x1=20, y1=0, x2=120, y2=100), "tile-b", border=True
        )
        self.assertEqual(
            [],
            _filter_unconfirmed_internal_border_components(
                [truncated_a, truncated_b],
                enabled=True,
                categories=("Box",),
                confirmation_iou=0.3,
                confirmation_metric="iou",
            ),
        )
        self.assertEqual(
            {"truncated-a", "truncated-b"},
            {
                item.internal_id
                for item in _filter_unconfirmed_internal_border_components(
                    [truncated_a, truncated_b],
                    enabled=True,
                    categories=("Box",),
                    confirmation_iou=0.3,
                    confirmation_metric="intersection_over_smaller",
                )
            },
        )

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
