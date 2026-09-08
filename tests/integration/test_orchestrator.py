from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from pcbparse.config import load_config
from pcbparse.graph.physical import build_physical_graph
from pcbparse.pipeline.orchestrator import infer_batch, infer_one
from pcbparse.pipeline.providers import EmptyImageProvider
from pcbparse.ir.models import (
    BBox,
    Component,
    Evidence,
    IRState,
    Junction,
    Pin,
    Point,
    WireSegment,
)


class VersionedEmptyProvider(EmptyImageProvider):
    version = "fixture-provider-v1"

    def build_ir(self, input_path, ctx=None):
        ir = super().build_ir(input_path, ctx)
        ir.metadata["model_versions"] = {"component": "fixture-component-v2"}
        return ir


class AmbiguousComponentProvider(EmptyImageProvider):
    version = "ambiguous-fixture"

    def build_ir(self, input_path, ctx=None):
        ir = super().build_ir(input_path, ctx)
        ir.state = IRState.PERCEIVED
        ir.components = [
            Component(
                internal_id="component:1",
                category="R",
                bbox=BBox(x1=2, y1=2, x2=12, y2=12),
                evidence=[
                    Evidence(
                        evidence_id="ambiguous:value",
                        source="rule",
                        score=0.9,
                        metadata={
                            "kind": "component-field-resolution",
                            "field": "value",
                            "decision": "ambiguous",
                            "candidates": [
                                {"value": "10k", "score": 0.9},
                                {"value": "100k", "score": 0.5},
                            ],
                        },
                    )
                ],
            )
        ]
        return ir


class AmbiguousCrossingProvider(EmptyImageProvider):
    version = "ambiguous-crossing-fixture"

    def build_ir(self, input_path, ctx=None):
        ir = super().build_ir(input_path, ctx)
        ir.state = IRState.PERCEIVED
        ir.segments = [
            WireSegment(
                internal_id="left",
                start=Point(x=4, y=16),
                end=Point(x=16, y=16),
                channel="horizontal",
            ),
            WireSegment(
                internal_id="right",
                start=Point(x=16, y=16),
                end=Point(x=28, y=16),
                channel="horizontal",
            ),
            WireSegment(
                internal_id="top",
                start=Point(x=16, y=4),
                end=Point(x=16, y=16),
                channel="vertical",
            ),
            WireSegment(
                internal_id="bottom",
                start=Point(x=16, y=16),
                end=Point(x=16, y=28),
                channel="vertical",
            ),
        ]
        ir.junctions = [
            Junction(
                internal_id="crossing",
                point=Point(x=16, y=16),
                kind="ambiguous",
                segment_ids=["left", "right", "top", "bottom"],
                score=0.8,
            )
        ]
        return ir


class OwnerlessPinProvider(EmptyImageProvider):
    version = "ownerless-pin-fixture"

    def build_ir(self, input_path, ctx=None):
        ir = super().build_ir(input_path, ctx)
        ir.state = IRState.PERCEIVED
        ir.pins = [
            Pin(
                internal_id="ownerless-pin",
                component_id=None,
                number="1",
                point=Point(x=8, y=8),
            )
        ]
        return ir


class MissingNumberPinProvider(EmptyImageProvider):
    version = "missing-number-pin-fixture"

    def build_ir(self, input_path, ctx=None):
        ir = super().build_ir(input_path, ctx)
        ir.state = IRState.PERCEIVED
        ir.components = [
            Component(
                internal_id="component:run",
                key="RUN",
                category="NET_input",
                name="RUN",
                bbox=BBox(x1=2, y1=2, x2=12, y2=12),
            )
        ]
        ir.pins = [
            Pin(
                internal_id="pin:run",
                component_id="component:run",
                number="∅",
                point=Point(x=8, y=8),
            )
        ]
        return ir


class OrchestratorTests(unittest.TestCase):
    def test_batch_rejects_duplicate_stems_before_writing(self) -> None:
        config = load_config("configs/pipeline.yaml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left"
            right = root / "right"
            left.mkdir()
            right.mkdir()
            Image.new("RGB", (16, 16), "white").save(left / "board.png")
            Image.new("RGB", (16, 16), "white").save(right / "board.png")
            output = root / "output"
            with self.assertRaisesRegex(ValueError, "duplicate stems"):
                infer_batch(
                    [left / "board.png", right / "board.png"],
                    output,
                    provider=EmptyImageProvider(),
                    config=config,
                    official_profile="configs/official_schema.yaml",
                )
            self.assertFalse(list(output.glob("*.json")))

    def test_provider_model_versions_reach_run_manifest(self) -> None:
        config = load_config("configs/pipeline.yaml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "fixture.png"
            Image.new("RGB", (32, 24), "white").save(image_path)
            result = infer_one(
                image_path,
                root / "versioned.json",
                provider=VersionedEmptyProvider(),
                config=config,
                official_profile="configs/official_schema.yaml",
            )
            self.assertEqual("fixture-component-v2", result.context.model_versions["component"])
            self.assertEqual("fixture-provider-v1", result.context.model_versions["provider"])

    def test_declared_pin_snap_bounds_reach_physical_graph(self) -> None:
        config = load_config("configs/pipeline.yaml")
        config["association"]["pin_wire_snap_min_px"] = 1.25
        config["association"]["pin_wire_snap_max_px"] = 4.5
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "fixture.png"
            Image.new("RGB", (16, 16), "white").save(image_path)
            with mock.patch(
                "pcbparse.pipeline.orchestrator.build_physical_graph",
                wraps=build_physical_graph,
            ) as graph_builder:
                infer_one(
                    image_path,
                    root / "snap-config.json",
                    provider=EmptyImageProvider(),
                    config=config,
                    official_profile="configs/official_schema.yaml",
                )

            self.assertEqual(1.25, graph_builder.call_args.kwargs["pin_snap_min_px"])
            self.assertEqual(
                4.5, graph_builder.call_args.kwargs["pin_snap_tolerance_px"]
            )

    def test_empty_provider_produces_valid_deterministic_json(self) -> None:
        config = load_config("configs/pipeline.yaml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "fixture.png"
            Image.new("RGB", (32, 24), "white").save(image_path)
            first = infer_one(
                image_path,
                root / "first.json",
                provider=EmptyImageProvider(),
                config=config,
                official_profile="configs/official_schema.yaml",
            )
            second = infer_one(
                image_path,
                root / "second.json",
                provider=EmptyImageProvider(),
                config=config,
                official_profile="configs/official_schema.yaml",
            )
            self.assertEqual({"components": {}, "pins": {}, "nets": {}}, first.output)
            self.assertEqual(first.output_hash, second.output_hash)
            self.assertEqual((root / "first.json").read_bytes(), (root / "second.json").read_bytes())

    def test_degraded_projection_drops_ownerless_pin_before_reference_validation(self) -> None:
        config = load_config("configs/pipeline.yaml")
        config["run"]["submission_mode"] = "degraded"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "fixture.png"
            Image.new("RGB", (16, 16), "white").save(image_path)

            result = infer_one(
                image_path,
                root / "degraded.json",
                provider=OwnerlessPinProvider(),
                config=config,
                official_profile="configs/official_schema.yaml",
            )

            self.assertEqual({"components": {}, "pins": {}, "nets": {}}, result.output)
            self.assertEqual(1, result.ir.metadata["degraded_submission_prune"]["pins_dropped"])
            self.assertNotIn(
                "PIN.UNKNOWN_COMPONENT", {item.code for item in result.diagnostics}
            )

    def test_degraded_projection_retains_and_auto_numbers_owned_empty_pin(self) -> None:
        config = load_config("configs/pipeline.yaml")
        config["run"]["submission_mode"] = "degraded"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "fixture.png"
            Image.new("RGB", (16, 16), "white").save(image_path)

            result = infer_one(
                image_path,
                root / "degraded-auto-pin.json",
                provider=MissingNumberPinProvider(),
                config=config,
                official_profile="configs/official_schema.yaml",
            )

            self.assertEqual(
                0,
                result.ir.metadata["degraded_submission_prune"]["pins_dropped"],
            )
            self.assertEqual(["pin_1"], list(result.output["pins"]["RUN"]))
            self.assertEqual("vbus_in", result.output["components"]["RUN"]["type"])
            self.assertEqual(
                {"x": 8.0, "y": 8.0},
                result.output["pins"]["RUN"]["pin_1"]["point"],
            )

    def test_selective_reasoning_stage_is_wired_before_validation(self) -> None:
        config = load_config("configs/pipeline.yaml")
        config["qwen"].update(
            {
                "enabled": True,
                "backend": "candidate_only",
                "route_threshold": 0.0,
                "min_time_reserve_seconds": 0.0,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "fixture.png"
            Image.new("RGB", (32, 24), "white").save(image_path)
            # Missing official key remains a validator error, but reasoning must
            # have filled the supplied value and recorded its stage first.
            with self.assertRaises(Exception) as captured:
                infer_one(
                    image_path,
                    root / "blocked.json",
                    provider=AmbiguousComponentProvider(),
                    config=config,
                    official_profile="configs/official_schema.yaml",
                )
            diagnostics = getattr(captured.exception, "diagnostics", [])
            self.assertIn("REASONING.APPLIED", {item.code for item in diagnostics})

    def test_junction_reasoning_precedes_graphs_and_changes_final_connectivity(self) -> None:
        config = load_config("configs/pipeline.yaml")
        config["qwen"].update(
            {
                "enabled": True,
                "backend": "candidate_only",
                "route_threshold": 0.0,
                "min_time_reserve_seconds": 0.0,
                "candidate_min_score": 0.5,
                "candidate_min_margin": 0.0,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "crossing.png"
            Image.new("RGB", (32, 32), "white").save(image_path)
            provider = AmbiguousCrossingProvider()

            # The unresolved crossing has two channel-separated physical nets.
            unresolved = provider.build_ir(image_path)
            unresolved_nets, _ = build_physical_graph(unresolved)
            self.assertEqual(2, len(unresolved_nets))

            result = infer_one(
                image_path,
                root / "resolved.json",
                provider=provider,
                config=config,
                official_profile="configs/official_schema.yaml",
            )

            self.assertEqual("connected_cross", result.ir.junctions[0].kind)
            self.assertEqual(1, len(result.ir.physical_nets))
            self.assertEqual(1, len(result.ir.electrical_nets))
            self.assertEqual(
                {"left", "right", "top", "bottom"},
                set(result.ir.physical_nets[0].segments),
            )
            self.assertFalse([item for item in result.diagnostics if item.severity == "ERROR"])
            self.assertEqual(1, len(result.output["nets"]))
            stages = [item.stage for item in result.context.trace]
            self.assertLess(stages.index("selective_reasoning"), stages.index("physical_graph"))
            self.assertLess(stages.index("selective_reasoning"), stages.index("electrical_graph"))


if __name__ == "__main__":
    unittest.main()
