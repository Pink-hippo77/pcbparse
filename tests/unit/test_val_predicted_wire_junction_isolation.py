from __future__ import annotations

import base64
import importlib.util
import unittest
from pathlib import Path

import numpy as np

from pcbparse.ir import CircuitIR, Evidence, ImageInfo, Junction, Pin, Point, WireSegment


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "audit_val_predicted_wire_junction_isolation.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "val_predicted_wire_junction_isolation", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def _packed(mask: np.ndarray, *, record_id: str = "v1") -> dict[str, object]:
    values = np.asarray(mask, dtype=np.uint8)
    payload = np.packbits(values.reshape(-1), bitorder="little").tobytes()
    return {
        "record_id": record_id,
        "sample_id": "s1",
        "shape": list(values.shape),
        "encoding": "base64-packbits-little",
        "mask": base64.b64encode(payload).decode("ascii"),
    }


def _golden_fixture() -> CircuitIR:
    ir = CircuitIR(
        image=ImageInfo(image_id="s1", path="s1.png", width=100, height=100)
    )
    ir.pins = [
        Pin(
            internal_id="p1",
            component_id="c1",
            point=Point(x=10, y=50),
            edge_ids=["hidden-1"],
            evidence=[Evidence(evidence_id="dataset:p1", source="dataset")],
        ),
        Pin(
            internal_id="p2",
            component_id="c2",
            point=Point(x=50, y=90),
            edge_ids=["hidden-2"],
            evidence=[Evidence(evidence_id="dataset:p2", source="dataset")],
        ),
    ]
    return ir


def _segments() -> list[WireSegment]:
    return [
        WireSegment(
            internal_id="s1",
            start=Point(x=10, y=50),
            end=Point(x=50, y=50),
            source_polyline_id="poly-1",
            channel="horizontal",
        ),
        WireSegment(
            internal_id="s2",
            start=Point(x=50, y=50),
            end=Point(x=50, y=90),
            source_polyline_id="poly-2",
            channel="vertical",
        ),
    ]


class PredictionDecoderTests(unittest.TestCase):
    def test_strict_decoder_round_trips_and_rejects_padding_bits(self) -> None:
        mask = np.asarray([[1, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
        row = _packed(mask)
        np.testing.assert_array_equal(mask.astype(bool), audit.decode_prediction_mask(row))

        packed = bytearray(base64.b64decode(row["mask"]))
        packed[-1] |= 0b10000000
        row["mask"] = base64.b64encode(packed).decode("ascii")
        with self.assertRaisesRegex(ValueError, "padding bits"):
            audit.decode_prediction_mask(row)


class PopulationContractTests(unittest.TestCase):
    def test_complete_saved_sample_is_accepted(self) -> None:
        tiles = [
            {
                "record_id": "a",
                "sample_id": "s1",
                "supervision": {"eligible": True},
            },
            {
                "record_id": "b",
                "sample_id": "s1",
                "supervision": {"eligible": True},
            },
            {
                "record_id": "c",
                "sample_id": "s2",
                "supervision": {"eligible": False},
            },
        ]
        predictions = {
            "a": {"record_id": "a", "sample_id": "s1"},
            "b": {"record_id": "b", "sample_id": "s1"},
        }
        result = audit._strict_complete_population(tiles, predictions)
        self.assertEqual(["s1"], result["covered_sample_ids"])
        self.assertEqual(["s2"], result["excluded_sample_ids"])
        self.assertTrue(result["all_covered_samples_have_every_val_tile"])

    def test_partial_saved_sample_fails_closed(self) -> None:
        tiles = [
            {
                "record_id": "a",
                "sample_id": "s1",
                "supervision": {"eligible": True},
            },
            {
                "record_id": "b",
                "sample_id": "s1",
                "supervision": {"eligible": False},
            },
        ]
        predictions = {"a": {"record_id": "a", "sample_id": "s1"}}
        with self.assertRaisesRegex(ValueError, "refuses partial"):
            audit._strict_complete_population(tiles, predictions)


class IsolationContractTests(unittest.TestCase):
    def test_model_semantics_are_the_only_changed_ir_field(self) -> None:
        segments = _segments()
        vectorized = [
            Junction(
                internal_id="vectorized",
                point=Point(x=50, y=50),
                kind="t_junction",
                segment_ids=["s1", "s2"],
                score=1.0,
            )
        ]
        learned = [
            Junction(
                internal_id="learned",
                point=Point(x=50, y=50),
                kind="endpoint",
                segment_ids=["s1", "s2"],
                score=0.9,
            )
        ]
        without, with_learned = audit._construct_variants(
            _golden_fixture(), segments, vectorized, learned, tolerance=1.0
        )

        self.assertEqual(
            audit._nonjunction_ir_sha256(without),
            audit._nonjunction_ir_sha256(with_learned),
        )
        self.assertEqual("t_junction", without.junctions[0].kind)
        self.assertEqual("endpoint", with_learned.junctions[0].kind)
        self.assertEqual([], without.pins[0].edge_ids)
        self.assertEqual([], with_learned.pins[0].edge_ids)

        reference = [["p1", "p2"]]
        vectorized_metric = audit._evaluate_ir(without, reference)
        learned_metric = audit._evaluate_ir(with_learned, reference)
        self.assertEqual(1.0, vectorized_metric["f1"])
        self.assertEqual(0.0, learned_metric["f1"])
        delta = audit._delta(vectorized_metric, learned_metric)
        self.assertEqual("degraded", delta["outcome_by_f1"])
        self.assertTrue(delta["pair_relation_changed"])

    def test_determinism_verifier_requires_two_identical_builds(self) -> None:
        calls = []

        def stable():
            calls.append(1)
            return {"kind": "fixture", "selection_boundary": {"test_rows_used": 0}}

        report, hashes = audit.verify_deterministic_replays(stable, 2)
        self.assertEqual(2, len(calls))
        self.assertEqual(1, len(set(hashes)))
        self.assertEqual("fixture", report["kind"])

        values = iter(({"value": 1}, {"value": 2}))
        with self.assertRaisesRegex(RuntimeError, "replays differ"):
            audit.verify_deterministic_replays(lambda: next(values), 2)
        with self.assertRaisesRegex(ValueError, ">= 2"):
            audit.verify_deterministic_replays(stable, 1)

    def test_macro_and_junction_population_preserve_weighting_evidence(self) -> None:
        rows = [
            {
                "variants": {
                    "predicted_wire_vectorized_only": {"f1": 0.2},
                    "predicted_wire_vectorized_plus_learned": {"f1": 0.8},
                },
                "junctions": {
                    "learned_raw": {
                        "count": 3,
                        "kind_counts": {"endpoint": 2, "t_junction": 1},
                        "connective_count": 1,
                        "nonconnective_or_ambiguous_count": 2,
                    }
                },
            },
            {
                "variants": {
                    "predicted_wire_vectorized_only": {"f1": 0.8},
                    "predicted_wire_vectorized_plus_learned": {"f1": 0.4},
                },
                "junctions": {
                    "learned_raw": {
                        "count": 2,
                        "kind_counts": {"bend": 1, "noise": 1},
                        "connective_count": 1,
                        "nonconnective_or_ambiguous_count": 1,
                    }
                },
            },
        ]

        macro = audit._sample_macro_summary(rows)
        self.assertAlmostEqual(0.5, macro["vectorized_only_mean_f1"])
        self.assertAlmostEqual(0.6, macro["vectorized_plus_learned_mean_f1"])
        self.assertAlmostEqual(0.1, macro["mean_paired_f1_delta"])
        junctions = audit._aggregate_junction_population(rows, "learned_raw")
        self.assertEqual(5, junctions["count"])
        self.assertEqual(2, junctions["connective_count"])
        self.assertEqual(
            {"bend": 1, "endpoint": 2, "noise": 1, "t_junction": 1},
            junctions["kind_counts"],
        )


if __name__ == "__main__":
    unittest.main()
