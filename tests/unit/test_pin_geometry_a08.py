"""A08 pin geometry/identity: tip vs base, cap audit, owner reject, pin-pin contact.

Replaces the earlier example assertions with real adapter materialization, real
decoder calls, real assignment and real graph construction.  The submission
config's pin budget is verified to propagate all the way to
``decode_pin_heatmaps`` and synthetic 16/32/64-pin responses document the
top-k truncation behaviour at the historical cap of 12 versus the
visible-edge-length budget.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from pcbparse.association.ocr_common import OCRAssociationConfig
from pcbparse.association.text_pin import associate_pin_texts
from pcbparse.config import load_config
from pcbparse.graph.physical import build_physical_graph
from pcbparse.io import ComponentView, ImageView
from pcbparse.ir.models import (
    BBox,
    CircuitIR,
    Component,
    ImageInfo,
    IRState,
    Pin,
    Point,
    TextToken,
)
from pcbparse.ir.transforms import AffineTransform, TransformChain
from pcbparse.perception.production_provider import (
    _apply_pin_inference_overrides,
    _parse_pin_inference_overrides,
)
from pcbparse.perception.torch_pin_adapter import (
    PinAdapterConfig as TorchPinAdapterConfig,
    TorchPinProvider,
)
from pcbparse.perception import torch_pin_adapter as pin_adapter

try:
    import torch
except ImportError:  # core runtime intentionally keeps torch optional
    torch = None  # type: ignore[assignment]


WORKSPACE = Path(__file__).resolve().parents[2]


def _identity_view(view_id: str, width: int, height: int) -> ImageView:
    return ImageView(
        view_id=view_id,
        image=Image.fromarray(np.full((height, width, 3), 255, dtype=np.uint8)),
        transform=TransformChain([AffineTransform.identity(view_id)]),
    )


def _component_view(
    component_id: str, *, view_px: int, bbox: tuple[float, float, float, float]
) -> ComponentView:
    return ComponentView(
        component_id=component_id,
        component_bbox=BBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
        full_roi=_identity_view(f"component:{component_id}:full", view_px, view_px),
    )


def _materialized_geometry(view_px: int) -> object:
    return pin_adapter._LetterboxGeometry(
        input_size=(view_px, view_px),
        resized_size=(view_px, view_px),
        padding=(0, 0, 0, 0),
        source_size=(view_px, view_px),
    )


def _match_within(
    predicted: list[tuple[float, float]],
    expected: list[tuple[float, float]],
    tolerance: float,
) -> tuple[int, int]:
    """Greedy nearest matching at a fixed tolerance; returns (matched, false positives)."""

    remaining = list(expected)
    matched = 0
    for point in sorted(predicted):
        best_index = None
        best_distance = None
        for index, target in enumerate(remaining):
            distance = math.hypot(point[0] - target[0], point[1] - target[1])
            if distance <= tolerance and (best_distance is None or distance < best_distance):
                best_index = index
                best_distance = distance
        if best_index is None:
            continue
        matched += 1
        remaining.pop(best_index)
    return matched, len(predicted) - matched


class PinBudgetConfigTests(unittest.TestCase):
    def test_config_default_cap_is_higher_than_twelve(self) -> None:
        cfg = TorchPinAdapterConfig()
        self.assertGreaterEqual(cfg.max_detections_per_component, 64)
        self.assertGreater(cfg.base_length_pixels, 0)

    def test_legacy_cap_twelve_propagates_as_fixed_budget(self) -> None:
        overrides = _parse_pin_inference_overrides({"max_detections_per_component": 12})
        provider = TorchPinProvider(
            backend=object(), version="a08-fixed", config=TorchPinAdapterConfig()
        )
        _apply_pin_inference_overrides(provider, overrides)
        self.assertEqual(12, provider.config.max_detections_per_component)
        for rect in ((0.0, 0.0, 4.0, 4.0), (0.0, 0.0, 128.0, 128.0)):
            self.assertEqual(12, provider.config.pin_budget_for(rect))

    def test_submission_pin_config_reaches_adapter_config(self) -> None:
        config = load_config(WORKSPACE / "configs" / "pipeline.edarace.yaml")
        overrides = _parse_pin_inference_overrides(config["pin"])
        provider = TorchPinProvider(
            backend=object(), version="a08-submission", config=TorchPinAdapterConfig()
        )
        _apply_pin_inference_overrides(provider, overrides)
        self.assertEqual("edge_length_v1", provider.config.pin_budget_mode)
        self.assertEqual(12, provider.config.pin_budget_floor)
        self.assertEqual(128, provider.config.max_detections_per_component)
        self.assertEqual(
            float(config["pin"]["pin_budget_pitch_px"]),
            provider.config.pin_budget_pitch_px,
        )
        self.assertEqual(float(config["pin"]["score_threshold"]), provider.config.score_threshold)
        self.assertEqual(float(config["pin"]["box_margin_px"]), provider.config.box_margin_px)

    def test_edge_length_budget_keeps_floor_scales_and_caps(self) -> None:
        config = load_config(WORKSPACE / "configs" / "pipeline.edarace.yaml")
        provider = TorchPinProvider(
            backend=object(), version="a08-budget", config=TorchPinAdapterConfig()
        )
        _apply_pin_inference_overrides(provider, _parse_pin_inference_overrides(config["pin"]))
        budget = provider.config.pin_budget_for
        self.assertEqual(12, budget((100.0, 100.0, 124.0, 112.0)), "tiny part keeps cap-12 floor")
        self.assertEqual(64, budget((0.0, 0.0, 128.0, 128.0)), "dense IC scales with edge length")
        self.assertEqual(128, budget((0.0, 0.0, 512.0, 512.0)), "global cap retained")
        self.assertEqual(12, budget(None), "unknown geometry falls back to the floor")

    def test_dense_ic_budget_matches_pin_count_so_no_truncation(self) -> None:
        provider = TorchPinProvider(
            backend=object(),
            version="a08-dense",
            config=TorchPinAdapterConfig(
                pin_budget_mode="edge_length_v1",
                pin_budget_floor=12,
                pin_budget_pitch_px=8.0,
                max_detections_per_component=128,
            ),
        )
        dense_budget = provider.config.pin_budget_for((0.0, 0.0, 128.0, 128.0))
        self.assertGreaterEqual(dense_budget, 64)
        self.assertEqual(
            0,
            audit_pin_cap_truncation({"ic_dense": 64}, cap=dense_budget)["truncated_components"],
        )
        self.assertEqual(
            1,
            audit_pin_cap_truncation({"ic_dense": 64}, cap=12)["truncated_components"],
        )


class PinMaterializationA08Tests(unittest.TestCase):
    def test_tip_base_distance_matches_base_length(self) -> None:
        provider = TorchPinProvider(
            backend=object(),
            version="a08-geometry",
            config=TorchPinAdapterConfig(input_size=(64, 64), base_length_pixels=8.0),
        )
        view = _component_view("C1", view_px=64, bbox=(0.0, 0.0, 64.0, 64.0))
        detections = [
            {
                "x": 3.25,
                "y": 2.5,
                "score": 0.9,
                "orientation": (1.0, 0.0),
                "anchor": (3, 2),
            }
        ]
        pins = provider._materialize_component_pins(
            view, detections, geometry=_materialized_geometry(64)
        )
        self.assertEqual(1, len(pins))
        pin = pins[0]
        assert pin.base is not None
        dist = math.hypot(pin.point.x - pin.base.x, pin.point.y - pin.base.y)
        self.assertAlmostEqual(8.0, dist, places=5)
        self.assertAlmostEqual(13.0, pin.point.x, places=5)
        self.assertAlmostEqual(10.0, pin.point.y, places=5)
        self.assertAlmostEqual(5.0, pin.base.x, places=5)

    def test_orientation_labels_are_cardinal_and_owner_is_view_component(self) -> None:
        provider = TorchPinProvider(
            backend=object(),
            version="a08-geometry",
            config=TorchPinAdapterConfig(input_size=(64, 64), base_length_pixels=8.0),
        )
        view = _component_view("C9", view_px=64, bbox=(0.0, 0.0, 64.0, 64.0))
        units = {"L": (-1.0, 0.0), "R": (1.0, 0.0), "T": (0.0, -1.0), "B": (0.0, 1.0)}
        detections = [
            {
                "x": 10.0 + index,
                "y": 10.0,
                "score": 0.9,
                "orientation": unit,
                "anchor": (10 + index, 10),
            }
            for index, unit in enumerate(units.values())
        ]
        pins = provider._materialize_component_pins(
            view, detections, geometry=_materialized_geometry(64)
        )
        self.assertEqual(sorted(units), sorted(pin.orientation for pin in pins))
        self.assertEqual({"C9"}, {pin.component_id for pin in pins})

    def test_body_bbox_is_not_tip(self) -> None:
        body = BBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
        tip = Point(x=10.0, y=5.0)
        on_border = tip.x in (body.x1, body.x2) or tip.y in (body.y1, body.y2)
        self.assertTrue(on_border)
        self.assertNotEqual((body.x1 + body.x2) / 2, tip.x)


class PinOwnershipAndContactTests(unittest.TestCase):
    def _ir(self, pins: list[Pin]) -> CircuitIR:
        return CircuitIR(
            state=IRState.SEMANTIC,
            image=ImageInfo(image_id="a08", path="a08.png", width=120, height=80),
            components=[
                Component(
                    internal_id="c1",
                    key="R1",
                    category="resistor",
                    bbox=BBox(x1=0, y1=0, x2=10, y2=10),
                ),
                Component(
                    internal_id="c2",
                    key="R2",
                    category="resistor",
                    bbox=BBox(x1=20, y1=0, x2=30, y2=10),
                ),
            ],
            pins=pins,
        )

    def test_owner_unknown_pin_is_never_force_bound(self) -> None:
        ir = self._ir(
            [
                Pin(
                    internal_id="p_ownerless",
                    component_id=None,
                    number="1",
                    point=Point(x=50.0, y=50.0),
                ),
                Pin(
                    internal_id="p_known",
                    component_id="c1",
                    number="1",
                    point=Point(x=50.0, y=50.0),
                ),
            ]
        )
        evidence: list[object] = []
        nets, _ = build_physical_graph(ir, connection_evidence_out=evidence)
        self.assertFalse(
            [item for item in evidence if getattr(item, "rule", "") == "direct_pin_contact"],
            "missing owner must not be force bound by direct-pin-contact",
        )
        for net in nets:
            self.assertNotEqual(
                {"p_ownerless", "p_known"}, set(net.pins), "no forced bind without owner"
            )

    def test_direct_pin_contact_is_legal_without_wire(self) -> None:
        ir = self._ir(
            [
                Pin(
                    internal_id="p_left",
                    component_id="c1",
                    number="1",
                    point=Point(x=50.0, y=50.0),
                ),
                Pin(
                    internal_id="p_right",
                    component_id="c2",
                    number="1",
                    point=Point(x=50.0, y=50.0),
                ),
            ]
        )
        evidence: list[object] = []
        nets, _ = build_physical_graph(ir, connection_evidence_out=evidence)
        joined = [net for net in nets if set(net.pins) == {"p_left", "p_right"}]
        self.assertEqual(1, len(joined), "contacting pins must share one net without wires")
        contacts = [item for item in evidence if getattr(item, "rule", "") == "direct_pin_contact"]
        self.assertEqual(1, len(contacts))
        self.assertFalse(contacts[0].virtual)
        self.assertEqual(0.0, contacts[0].distance_px)

    def test_same_owner_pins_are_not_direct_contact(self) -> None:
        ir = self._ir(
            [
                Pin(
                    internal_id="p_a",
                    component_id="c1",
                    number="1",
                    point=Point(x=50.0, y=50.0),
                ),
                Pin(
                    internal_id="p_b",
                    component_id="c1",
                    number="2",
                    point=Point(x=50.0, y=50.0),
                ),
            ]
        )
        evidence: list[object] = []
        build_physical_graph(ir, connection_evidence_out=evidence)
        self.assertFalse(
            [item for item in evidence if getattr(item, "rule", "") == "direct_pin_contact"],
            "same-owner pins never join through direct pin contact",
        )

    def test_wrong_owner_text_is_rejected_by_assignment(self) -> None:
        pin = Pin(
            internal_id="pin:1",
            component_id="C1",
            number=None,
            name="",
            point=Point(x=50.0, y=50.0),
        )
        wrong = TextToken(
            internal_id="text:wrong",
            raw_text="5",
            details="pinNumber",
            bbox=BBox(x1=44.0, y1=44.0, x2=56.0, y2=56.0),
            score=0.99,
            component_id="C_OTHER",
        )
        summary = associate_pin_texts([pin], [wrong], [], OCRAssociationConfig())
        self.assertEqual(0, summary["assigned_fields"])
        self.assertIsNone(pin.number)

        right = TextToken(
            internal_id="text:right",
            raw_text="5",
            details="pinNumber",
            bbox=BBox(x1=44.0, y1=44.0, x2=56.0, y2=56.0),
            score=0.99,
            component_id="C1",
        )
        summary = associate_pin_texts([pin], [right], [], OCRAssociationConfig())
        self.assertEqual(1, summary["assigned_fields"])
        self.assertEqual("5", pin.number)


def audit_pin_cap_truncation(pin_counts: dict[str, int], cap: int) -> dict[str, object]:
    """Count components that would be truncated by a hard per-component cap."""
    truncated = {cid: n for cid, n in pin_counts.items() if n > cap}
    return {
        "cap": cap,
        "components": len(pin_counts),
        "truncated_components": len(truncated),
        "truncated_pin_total": sum(n - cap for n in truncated.values()),
        "examples": dict(list(truncated.items())[:5]),
    }


class PinCapAuditTests(unittest.TestCase):
    def test_dense_ic_would_be_flagged_not_silently_cut(self) -> None:
        report = audit_pin_cap_truncation({"ic_dsp": 64, "r1": 2}, cap=12)
        self.assertEqual(report["truncated_components"], 1)
        self.assertEqual(report["truncated_pin_total"], 52)


class _SyntheticBackend:
    """Duck-typed torch-pin backend returning one fixed synthetic response."""

    def __init__(self, peaks: list[tuple[int, int, float]], grid: int) -> None:
        self.peaks = peaks
        self.grid = grid

    def run(self, batch: object) -> dict[str, object]:
        batch_size = int(batch.shape[0])
        grid = self.grid
        logits = torch.full((batch_size, 1, grid, grid), -20.0)
        offset = torch.zeros((batch_size, 2, grid, grid))
        orientation = torch.zeros((batch_size, 2, grid, grid))
        for x, y, score in self.peaks:
            logits[0, 0, y, x] = math.log(score / (1.0 - score))
            orientation[0, 0, y, x] = 1.0
        return {"heatmap_logits": logits, "offset": offset, "orientation": orientation}


@unittest.skipUnless(torch is not None, "optional PyTorch runtime is not installed")
class PinDecodeBudgetTests(unittest.TestCase):
    VIEW_PX = 128
    GRID = 32

    @staticmethod
    def _peaks(count: int) -> list[tuple[int, int, float]]:
        peaks = []
        for index in range(count):
            row, column = divmod(index, 8)
            peaks.append((1 + 2 * column, 1 + 2 * row, 0.99 - 0.001 * index))
        return peaks

    def _provider(self, *, fixed_cap: int | None = None) -> TorchPinProvider:
        if fixed_cap is not None:
            config = TorchPinAdapterConfig(
                input_size=(self.VIEW_PX, self.VIEW_PX),
                score_threshold=0.5,
                box_margin_px=0.0,
                max_detections_per_component=fixed_cap,
                pin_budget_mode="fixed",
            )
        else:
            config = TorchPinAdapterConfig(
                input_size=(self.VIEW_PX, self.VIEW_PX),
                score_threshold=0.5,
                box_margin_px=0.0,
                max_detections_per_component=128,
                pin_budget_mode="edge_length_v1",
                pin_budget_floor=12,
                pin_budget_pitch_px=8.0,
            )
        return TorchPinProvider(backend=None, version="a08-synthetic", config=config)

    def _run(self, provider: TorchPinProvider, peaks: list[tuple[int, int, float]]):
        provider._backend = _SyntheticBackend(peaks, self.GRID)
        view = _component_view(
            "DENSE_IC", view_px=self.VIEW_PX, bbox=(0.0, 0.0, float(self.VIEW_PX), float(self.VIEW_PX))
        )
        return provider.predict([view], None)

    def test_decoder_receives_config_budget_end_to_end(self) -> None:
        config = load_config(WORKSPACE / "configs" / "pipeline.edarace.yaml")
        provider = TorchPinProvider(
            backend=None, version="a08-e2e", config=TorchPinAdapterConfig()
        )
        _apply_pin_inference_overrides(provider, _parse_pin_inference_overrides(config["pin"]))
        input_px = provider.config.input_size[0]
        provider._backend = _SyntheticBackend(
            [(26, 26, 0.9), (28, 26, 0.9)], input_px // provider.config.output_stride
        )
        tiny = _component_view(
            "TINY", view_px=input_px, bbox=(100.0, 100.0, 124.0, 112.0)
        )
        dense = _component_view(
            "DENSE", view_px=input_px, bbox=(192.0, 192.0, 320.0, 320.0)
        )
        from pcbparse.training.torch_pin_model import decode_pin_heatmaps

        with mock.patch(
            "pcbparse.perception.torch_pin_adapter.decode_pin_heatmaps",
            wraps=decode_pin_heatmaps,
        ) as spy:
            provider.predict([tiny], None)
            self.assertEqual([12], list(spy.call_args.kwargs["max_detections"]))
            provider.predict([dense], None)
            self.assertEqual(
                [64],
                list(spy.call_args.kwargs["max_detections"]),
                "512px perimeter budgets 64 pin slots at the configured pitch",
            )
        huge = _component_view(
            "HUGE", view_px=input_px, bbox=(0.0, 0.0, float(input_px), float(input_px))
        )
        with mock.patch(
            "pcbparse.perception.torch_pin_adapter.decode_pin_heatmaps",
            wraps=decode_pin_heatmaps,
        ) as spy:
            provider.predict([huge], None)
            self.assertEqual(
                [128],
                list(spy.call_args.kwargs["max_detections"]),
                "global cap still bounds the budget",
            )

    def test_synthetic_dense_responses_truncate_at_cap_twelve(self) -> None:
        for count in (16, 32, 64):
            with self.subTest(pins=count):
                peaks = self._peaks(count)
                pins = self._run(self._provider(fixed_cap=12), peaks)
                self.assertEqual(12, len(pins), f"cap-12 must truncate {count} candidates")
                self.assertEqual({"DENSE_IC"}, {pin.component_id for pin in pins})
                kept = {(peak[0] * 4, peak[1] * 4) for peak in peaks[:12]}
                produced = [(pin.point.x, pin.point.y) for pin in pins]
                self.assertEqual(kept, set(produced), "top-scored candidates survive truncation")
                matched_4, fp_4 = _match_within(produced, [(x * 4, y * 4) for x, y, _ in peaks], 4.0)
                matched_5, fp_5 = _match_within(produced, [(x * 4, y * 4) for x, y, _ in peaks], 5.0)
                self.assertEqual(12, matched_4)
                self.assertEqual(0, fp_4)
                self.assertEqual(12, matched_5)
                self.assertEqual(0, fp_5)
                self.assertEqual(count - 12, count - matched_5, "missed pins stay missed at 5px")

    def test_synthetic_dense_responses_survive_edge_length_budget(self) -> None:
        for count in (16, 32, 64):
            with self.subTest(pins=count):
                peaks = self._peaks(count)
                pins = self._run(self._provider(), peaks)
                self.assertEqual(count, len(pins), "edge-length budget must not truncate")
                self.assertEqual({"DENSE_IC"}, {pin.component_id for pin in pins})
                produced = [(pin.point.x, pin.point.y) for pin in pins]
                expected = [(x * 4, y * 4) for x, y, _ in peaks]
                matched_4, fp_4 = _match_within(produced, expected, 4.0)
                matched_5, fp_5 = _match_within(produced, expected, 5.0)
                self.assertEqual(count, matched_4)
                self.assertEqual(0, fp_4)
                self.assertEqual(count, matched_5)
                self.assertEqual(0, fp_5)


if __name__ == "__main__":
    unittest.main()
