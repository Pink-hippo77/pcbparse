from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from pcbparse.ir import CircuitIR, Evidence, ImageInfo, Junction, Pin, Point, WireSegment


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "audit_val_graph_oracle_decomposition.py"
)
_SPEC = importlib.util.spec_from_file_location("val_graph_oracle_decomposition", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def _fixture_ir() -> CircuitIR:
    ir = CircuitIR(
        image=ImageInfo(image_id="fixture", path="fixture.png", width=100, height=100)
    )
    ir.pins = [
        Pin(
            internal_id="p1",
            point=Point(x=10, y=50),
            edge_ids=["s1"],
            evidence=[Evidence(evidence_id="dataset:p1", source="dataset")],
        ),
        Pin(
            internal_id="p2",
            point=Point(x=50, y=90),
            edge_ids=["s2"],
            evidence=[Evidence(evidence_id="dataset:p2", source="dataset")],
        ),
    ]
    ir.segments = [
        WireSegment(
            internal_id="s1",
            start=Point(x=10, y=50),
            end=Point(x=50, y=50),
            source_edge_id="E_1",
            source_polyline_id="poly-1",
            channel="horizontal",
            evidence=[Evidence(evidence_id="dataset:s1", source="dataset")],
        ),
        WireSegment(
            internal_id="s2",
            start=Point(x=50, y=50),
            end=Point(x=50, y=90),
            source_edge_id="E_2",
            source_polyline_id="poly-2",
            channel="vertical",
            evidence=[Evidence(evidence_id="dataset:s2", source="dataset")],
        ),
    ]
    return ir


class PairMetricTests(unittest.TestCase):
    def test_pair_metric_reports_count_weighted_tp_fp_fn(self) -> None:
        reference = audit._pin_pairs([["a", "b", "c"]])
        predicted = audit._pin_pairs([["a", "b"], ["c", "d"]])

        result = audit._score_pair_partition(reference, predicted)

        self.assertEqual(1, result["tp"])
        self.assertEqual(1, result["fp"])
        self.assertEqual(2, result["fn"])
        self.assertAlmostEqual(0.5, result["precision"])
        self.assertAlmostEqual(1.0 / 3.0, result["recall"])
        self.assertAlmostEqual(0.4, result["f1"])
        self.assertFalse(result["exact_pair_relation"])

    def test_empty_reference_and_prediction_are_exact(self) -> None:
        result = audit._score_pair_partition(set(), set())
        self.assertEqual(1.0, result["precision"])
        self.assertEqual(1.0, result["recall"])
        self.assertEqual(1.0, result["f1"])
        self.assertTrue(result["exact_pair_relation"])


class OracleContractTests(unittest.TestCase):
    def test_strip_membership_is_deep_and_preserves_geometry(self) -> None:
        original = _fixture_ir()
        original.segments[0].source_polyline_ids = ["hidden-oracle-path"]

        stripped = audit._strip_oracle_membership(original)

        self.assertEqual(["s1"], original.pins[0].edge_ids)
        self.assertEqual("E_1", original.segments[0].source_edge_id)
        self.assertEqual([], stripped.pins[0].edge_ids)
        self.assertEqual([], stripped.pins[0].evidence)
        self.assertIsNone(stripped.segments[0].source_edge_id)
        self.assertIsNone(stripped.segments[0].source_polyline_id)
        self.assertEqual([], stripped.segments[0].source_polyline_ids)
        self.assertEqual(["hidden-oracle-path"], original.segments[0].source_polyline_ids)
        self.assertEqual([], stripped.segments[0].evidence)
        self.assertEqual(original.segments[0].start, stripped.segments[0].start)

    def test_junction_variant_changes_only_constructed_pair_relation(self) -> None:
        geometry = audit._strip_oracle_membership(_fixture_ir())
        reference = [["p1", "p2"]]

        without = audit._evaluate_ir(geometry, reference)
        geometry.junctions = [
            Junction(
                internal_id="j1",
                point=Point(x=50, y=50),
                kind="t_junction",
                segment_ids=["s1", "s2"],
            )
        ]
        with_junction = audit._evaluate_ir(geometry, reference)

        self.assertEqual(0.0, without["f1"])
        self.assertEqual(1, without["fn"])
        self.assertEqual(1.0, with_junction["f1"])
        self.assertEqual(1, with_junction["tp"])

    def test_constructability_fails_closed_for_incomplete_saved_evidence(self) -> None:
        matrix = audit._constructability_matrix(
            val_samples=30,
            wire_covered_samples=18,
            eligible_wire_tiles=154,
            ineligible_wire_tiles=70,
            trusted_junction_rows=3498,
            unresolved_junction_rows=408,
        )
        by_name = {row["combination"]: row for row in matrix}
        self.assertFalse(
            by_name["saved_formal_wire_masks + current_graph_builder on complete frozen val"][
                "constructable"
            ]
        )
        self.assertIn(
            "18/30",
            by_name["saved_formal_wire_masks + current_graph_builder on complete frozen val"][
                "reason"
            ],
        )
        self.assertFalse(
            by_name["full_gt_wire + full_gt_junction + current_graph_builder"][
                "constructable"
            ]
        )

    def test_payload_hash_excludes_only_its_own_field(self) -> None:
        report = {"kind": "fixture", "selection_boundary": {"test_rows_used": 0}}
        first = audit._payload_sha256(report)
        report["audit_payload_sha256"] = first
        self.assertEqual(first, audit._payload_sha256(report))
        report["selection_boundary"]["test_rows_used"] = 1
        self.assertNotEqual(first, audit._payload_sha256(report))


if __name__ == "__main__":
    unittest.main()
