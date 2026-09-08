from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "audit_pin_validation_strata.py"
)
_SPEC = importlib.util.spec_from_file_location("pin_validation_strata_audit", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


class PinValidationStrataMetricTests(unittest.TestCase):
    def test_retained_assignment_matches_maximum_cardinality_not_greedy(self) -> None:
        # p0 initially takes t0 (stable distance tie); p1 can only take t0.  The
        # augmenting path must move p0 to t1 so both predictions are matched.
        predicted = [{"x": 1.0, "y": 0.0}, {"x": 0.0, "y": 0.0}]
        truth = [(0.0, 0.0), (2.0, 0.0)]
        pairs = audit._match_pairs(predicted, truth, threshold=1.0)
        self.assertEqual(((0, 1), (1, 0)), pairs)

    def test_target_group_attribution_reconciles_micro_totals(self) -> None:
        counts = audit._attributed_counts(
            truth_groups=["left", "right", "right"],
            prediction_groups=["up", "left", "down"],
            pairs=[(0, 0), (1, 1)],
        )
        self.assertEqual((1, 1, 1), counts["left"])
        self.assertEqual((1, 1, 2), counts["right"])
        self.assertEqual((0, 1, 0), counts["down"])
        self.assertEqual(2, sum(row[0] for row in counts.values()))
        self.assertEqual(3, sum(row[1] for row in counts.values()))
        self.assertEqual(3, sum(row[2] for row in counts.values()))

    def test_boundary_corner_and_fixed_size_count_bins_are_explicit(self) -> None:
        box = (10.0, 20.0, 30.0, 40.0)
        self.assertEqual(
            ("corner", "corner_TL"), audit._boundary_groups((10.2, 20.2), box)
        )
        self.assertEqual(("edge", "edge_R"), audit._boundary_groups((30.2, 31.0), box))
        self.assertEqual(
            ("off_boundary", "off_boundary"),
            audit._boundary_groups((20.0, 30.0), box),
        )
        self.assertEqual("0", audit._gt_count_bin(0))
        self.assertEqual("17+", audit._gt_count_bin(17))
        self.assertEqual("small_sqrt_area_lt_32px", audit._size_bin(1023.9))
        self.assertEqual("medium_sqrt_area_32_to_96px", audit._size_bin(1024.0))
        self.assertEqual("large_sqrt_area_ge_96px", audit._size_bin(96.0**2))

    def test_micro_summary_reproduces_detection_prf(self) -> None:
        result = audit._summary(3, 5, 4)
        self.assertAlmostEqual(0.6, result["precision_4px"])
        self.assertAlmostEqual(0.75, result["recall_4px"])
        self.assertAlmostEqual(2.0 / 3.0, result["f1_4px"])
        self.assertAlmostEqual(1.25, result["pred_to_gt_ratio"])
        with self.assertRaisesRegex(ValueError, "invalid Pin detection totals"):
            audit._summary(5, 4, 6)


class PinValidationStrataBoundaryTests(unittest.TestCase):
    @staticmethod
    def _validation() -> dict[str, dict[str, object]]:
        return {
            "val-record": {
                "record_id": "val-record",
                "view_id": "val-view",
                "sample_id": "v",
                "component_box_heatmap": (10.0, 10.0, 20.0, 20.0),
            }
        }

    def test_prediction_reader_accepts_exact_val_population(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "record_id": "val-record",
                        "view_id": "val-view",
                        "sample_id": "v",
                        "output_stride": 4,
                        "detections_heatmap": [
                            {
                                "x": 10.0,
                                "y": 15.0,
                                "score": 0.9,
                                "orientation": [-1.0, 0.0],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows, inventory = audit._read_predictions(
                path,
                validation=self._validation(),
                score_threshold=0.75,
                max_detections=128,
                input_size=(384, 384),
            )
        self.assertEqual(1, inventory["prediction_rows"])
        self.assertEqual(1, inventory["decoded_detections"])
        self.assertEqual("left", rows["val-record"][0]["orientation"])

    def test_prediction_from_non_val_population_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "record_id": "test-record",
                        "view_id": "test-view",
                        "sample_id": "t",
                        "output_stride": 4,
                        "detections_heatmap": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exact frozen-val ROI population"):
                audit._read_predictions(
                    path,
                    validation=self._validation(),
                    score_threshold=0.75,
                    max_detections=128,
                    input_size=(384, 384),
                )


if __name__ == "__main__":
    unittest.main()
