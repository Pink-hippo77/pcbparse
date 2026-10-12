"""Virtual-repair provenance must survive from mask closing to IR and final JSON.

Acceptance fixtures use the real CompositePerceptionProvider with small wire
masks: (1) a broken wire whose gap is closed by morphology must keep its
"not solid wire" markers in IR records, segment evidence and the dumped JSON;
(2) neighbouring parallel wires, a broken crossing and text-noise blobs must
not be mass-merged into false nets by that same closing.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pcbparse.graph.physical import build_physical_graph
from pcbparse.ir.models import IRState
from pcbparse.nets.hypergraph import build_electrical_nets
from pcbparse.output.official_serializer import (
    serialize_official,
    serialize_official_with_quality,
)
from pcbparse.perception.composite_provider import (
    CompositePerceptionProvider,
    CompositeProviderConfig,
)
from pcbparse.perception.protocols import WireMask


class EmptyComponentStub:
    version = "component-empty"

    def predict(self, image, ctx):
        del image, ctx
        return []


class EmptyOCRStub:
    version = "ocr-empty"

    def recognize(self, views, ctx):
        del views, ctx
        return []


class EmptyPinStub:
    version = "pin-empty"

    def predict(self, component_views, ctx):
        del component_views, ctx
        return []


class MaskWireStub:
    version = "wire-mask-stub"

    def __init__(self, builder) -> None:
        self._builder = builder

    def segment(self, views, ctx):
        del ctx
        view = views[0]
        mask = np.zeros((view.height, view.width), dtype=bool)
        self._builder(mask)
        return [WireMask(view_id=view.view_id, mask=mask, score=0.9)]


def _provider(builder, *, gap_closing_radius_px: int) -> CompositePerceptionProvider:
    return CompositePerceptionProvider(
        component_provider=EmptyComponentStub(),
        ocr_provider=EmptyOCRStub(),
        pin_provider=EmptyPinStub(),
        wire_provider=MaskWireStub(builder),
        config=CompositeProviderConfig(
            wire_gap_closing_radius_px=gap_closing_radius_px,
            wire_minimum_run_px=4,
        ),
    )


def _build(builder, *, gap_closing_radius_px: int):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fixture.png"
        Image.new("RGB", (64, 64), "white").save(path)
        return _provider(
            builder, gap_closing_radius_px=gap_closing_radius_px
        ).build_ir(path)


class VirtualRepairLineageTests(unittest.TestCase):
    def test_repair_marker_survives_to_ir_and_final_json(self) -> None:
        def builder(mask: np.ndarray) -> None:
            mask[32, 5:30] = True
            mask[32, 32:60] = True

        ir = _build(builder, gap_closing_radius_px=1)

        self.assertEqual(1, len(ir.repairs))
        record = ir.repairs[0]
        self.assertEqual("mask_closing_virtual_gap", record.rule_id)
        self.assertEqual(1, len(record.object_ids))
        self.assertEqual(
            record.object_ids,
            [segment.internal_id for segment in ir.segments if segment.synthetic],
        )
        self.assertEqual(
            sorted(record.evidence_ids),
            sorted(
                item.evidence_id
                for segment in ir.segments
                for item in segment.evidence
                if item.source == "repair"
            ),
        )
        self.assertEqual(2, record.after["added_pixels"])
        self.assertEqual(1, record.after["closing_radius"])
        self.assertEqual([29.0, 32.0], record.before["left_point"])
        self.assertEqual([32.0, 32.0], record.before["right_point"])
        self.assertEqual([30.0, 32.0], record.after["bridge_point"])
        self.assertIn("wire_breaks", ir.metadata)

        final_json = json.dumps(ir.model_dump(mode="json"), ensure_ascii=False)
        self.assertIn("mask_closing_virtual_gap", final_json)
        self.assertIn(record.repair_id, final_json)
        self.assertIn("wire-virtual-repair:", final_json)

        ir.state = IRState.VALIDATED
        _output, quality = serialize_official_with_quality(ir, mode="strict")
        provenance = quality["wire_provenance"]
        self.assertEqual(1, provenance["virtual_repaired_segment_count"])
        self.assertEqual(record.object_ids, provenance["virtual_repaired_segments"])
        self.assertEqual(1, provenance["synthetic_segment_count"])
        self.assertEqual(1, provenance["mask_closing_repairs"])
        self.assertFalse(provenance["relation_only_repairs_become_edges"])

        ir.physical_nets, _ = build_physical_graph(ir)
        ir.electrical_nets = build_electrical_nets(ir)
        output = serialize_official(ir)
        edge_points = [
            point
            for net_key in output["nets"]
            for edge in output["nets"][net_key]["edges"].values()
            for point in edge
        ]
        self.assertIn({"x": 5.0, "y": 32.0}, edge_points)
        self.assertIn({"x": 59.0, "y": 32.0}, edge_points)
        self.assertTrue(
            any(point["x"] == 30.0 and point["y"] == 32.0 for point in edge_points)
            or all(
                point["x"] in (5.0, 59.0) for point in edge_points
            ),
            "the repaired span must be part of the final official JSON edges",
        )

    def test_parallel_wires_stay_separate_under_closing(self) -> None:
        def builder(mask: np.ndarray) -> None:
            mask[20, 5:30] = True
            mask[20, 32:60] = True
            mask[28, 5:30] = True
            mask[28, 32:60] = True

        ir = _build(builder, gap_closing_radius_px=1)
        rows = sorted({round(segment.start.y) for segment in ir.segments})
        self.assertEqual([20.0, 28.0], rows, "closing must not fuse parallel wires")
        self.assertEqual(2, len(ir.repairs), "each wire gap repairs independently")
        nets, _ = build_physical_graph(ir)
        self.assertEqual(2, len(nets), "no false net merge from closing")

    def test_broken_crossing_stays_split_under_closing(self) -> None:
        def builder(mask: np.ndarray) -> None:
            mask[32, 5:60] = True
            mask[5:26, 32] = True
            mask[39:60, 32] = True

        ir = _build(builder, gap_closing_radius_px=1)
        self.assertEqual(
            3, len(ir.segments), "a multi-pixel crossing gap must survive radius-1 closing"
        )
        self.assertEqual([], ir.repairs, "closing must not fabricate the cross join")
        nets, _ = build_physical_graph(ir)
        self.assertGreaterEqual(len(nets), 2, "crossing halves stay separate nets")

    def test_text_noise_becomes_no_wires_under_closing(self) -> None:
        def builder(mask: np.ndarray) -> None:
            mask[8:10, 8:10] = True
            mask[20:22, 30:32] = True
            mask[40:42, 12:14] = True
            mask[50:52, 48:50] = True

        ir = _build(builder, gap_closing_radius_px=1)
        self.assertEqual([], ir.segments, "noise blobs must not become wires")
        self.assertEqual([], ir.repairs)
        nets, _ = build_physical_graph(ir)
        self.assertEqual([], nets)


if __name__ == "__main__":
    unittest.main()
