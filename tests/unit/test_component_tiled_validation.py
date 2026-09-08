from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "evaluate_component_tiled_validation.py"
)
_SPEC = importlib.util.spec_from_file_location("component_tiled_validation", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
_CUDA_ENVIRONMENT_BEFORE_IMPORT = os.environ.get("CUDA_VISIBLE_DEVICES")
_SPEC.loader.exec_module(audit)
_CUDA_ENVIRONMENT_AFTER_IMPORT = os.environ.get("CUDA_VISIBLE_DEVICES")


class ComponentTiledProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = {
            "image": {"component_tile_size": 512, "component_tile_overlap": 64}
        }
        self.thresholds = {"component": {"min_score": 0.20, "nms_iou": 0.55}}
        self.summary = {"config": {"tile_size": 512, "tile_overlap": 64}}

    def test_accepts_only_predeclared_fixed_protocol(self) -> None:
        protocol = audit._fixed_protocol(
            self.pipeline, self.thresholds, self.summary
        )
        self.assertEqual(512, protocol["tile_size"])
        self.assertEqual(64, protocol["tile_overlap"])
        self.assertEqual(448, protocol["tile_stride"])
        self.assertEqual(0.20, protocol["score_threshold"])
        self.assertEqual(0.55, protocol["component_nms_iou"])
        self.assertFalse(protocol["thresholds_searched"])

    def test_import_does_not_modify_cuda_visibility(self) -> None:
        self.assertEqual(
            _CUDA_ENVIRONMENT_BEFORE_IMPORT, _CUDA_ENVIRONMENT_AFTER_IMPORT
        )

    def test_rejects_any_geometry_or_threshold_drift(self) -> None:
        cases = [
            ({"image": {"component_tile_size": 1024, "component_tile_overlap": 64}}, self.thresholds, self.summary),
            (self.pipeline, {"component": {"min_score": 0.19, "nms_iou": 0.55}}, self.summary),
            (self.pipeline, {"component": {"min_score": 0.20, "nms_iou": 0.50}}, self.summary),
            (self.pipeline, self.thresholds, {"config": {"tile_size": 512, "tile_overlap": 63}}),
        ]
        for pipeline, thresholds, summary in cases:
            with self.subTest(
                pipeline=pipeline, thresholds=thresholds, summary=summary
            ):
                with self.assertRaises(ValueError):
                    audit._fixed_protocol(pipeline, thresholds, summary)


class ComponentTiledMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.categories = [
            {"official_id": 3, "name": "R"},
            {"official_id": 4, "name": "C"},
            {"official_id": 5, "name": "LED"},
        ]
        self.truth = {
            "v1": [
                {"category": "R", "box": [0.0, 0.0, 10.0, 10.0]},
                {"category": "C", "box": [20.0, 20.0, 30.0, 30.0]},
            ],
            "v2": [{"category": "R", "box": [0.0, 0.0, 10.0, 10.0]}],
        }

    def test_matches_by_class_and_global_score_order(self) -> None:
        predictions = [
            {
                "sample_id": "v1",
                "predictions": [
                    {"category": "R", "score": 0.9, "box": [0, 0, 10, 10]},
                    {"category": "C", "score": 0.8, "box": [20, 20, 30, 30]},
                    {"category": "R", "score": 0.7, "box": [50, 50, 60, 60]},
                ],
            },
            {
                "sample_id": "v2",
                "predictions": [
                    {"category": "R", "score": 0.6, "box": [0, 0, 10, 10]}
                ],
            },
        ]
        metrics = audit._evaluate_predictions(
            predictions, self.truth, self.categories
        )
        self.assertEqual(3, metrics["true_positives"])
        self.assertEqual(1, metrics["false_positives"])
        self.assertEqual(0, metrics["false_negatives"])
        self.assertAlmostEqual(0.75, metrics["precision50"])
        self.assertAlmostEqual(1.0, metrics["recall50"])
        self.assertAlmostEqual(6.0 / 7.0, metrics["f1_50"])
        self.assertEqual(["LED"], metrics["classes_without_ground_truth"])
        self.assertIsNone(metrics["per_class"]["LED"]["ap50"])

    def test_rejects_out_of_contract_prediction_category(self) -> None:
        predictions = [
            {
                "sample_id": "v1",
                "predictions": [
                    {"category": "UNKNOWN", "score": 0.9, "box": [0, 0, 10, 10]}
                ],
            }
        ]
        with self.assertRaises(ValueError):
            audit._evaluate_predictions(predictions, self.truth, self.categories)

    def test_iou_uses_half_open_box_geometry(self) -> None:
        self.assertEqual(1.0, audit._box_iou([0, 0, 10, 10], [0, 0, 10, 10]))
        self.assertEqual(0.0, audit._box_iou([0, 0, 10, 10], [10, 0, 20, 10]))
        self.assertAlmostEqual(
            25.0 / 175.0,
            audit._box_iou([0, 0, 10, 10], [5, 5, 15, 15]),
        )


if __name__ == "__main__":
    unittest.main()
