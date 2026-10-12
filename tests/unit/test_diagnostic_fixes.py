"""Tests for diagnostic correctness fixes: multi-batch trace, coordinate restore,
independent stage states, macro/micro averaging, binding oracle without hint shortcut."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "src"))

from pcbparse.evaluation.detection_eval import evaluate_detection_candidates
from pcbparse.evaluation.ocr_oracle import (
    aggregate_recognition,
    binding_oracle,
    recognition_oracle,
)
from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth


def _text(content: str, details: str, x1: int = 1) -> dict:
    return {
        "tid": "unused",
        "state": "active",
        "category": "params",
        "content": content,
        "details": details,
        "location": {"x1": x1, "y1": 1, "x2": x1 + 8, "y2": 4},
    }


def annotation() -> dict:
    texts = {"T_D1": _text("R1", "designator", x1=1), "T_V1": _text("10k", "value", x1=12)}
    p1 = {"pinId": "P_1", "edgeId": "E_1", "pinNumberByOCR": None, "pinNumberByVLM": None,
          "pinNameByOCR": None, "pinNameByVLM": None, "tip": [5, 10], "base": [2, 10]}
    return {
        "canvas": {"width": 40, "height": 30},
        "nowCls": "unit",
        "components": {"C_1": {"cpntId": "C_1", "state": "active", "category": "resistor",
            "location": {"x1": 0, "y1": 5, "x2": 10, "y2": 15}, "pins": {"P_1": p1},
            "associatedTexts": {"T_D1": texts["T_D1"], "T_V1": texts["T_V1"]},
            "searchNameByOCR": None, "searchNameByVLM": None, "searchNameByLib": None}},
        "pins": {"P_1": p1},
        "texts": texts,
        "edges": {"E_1": {"edgeId": "E_1", "state": "active", "netlabel": None,
            "connect": {"P_1": {"pinId": "P_1", "cpntId": "C_1", "location": [5, 10]}},
            "keyPointPaths": {"wire": [[[5, 10], [8, 10]]]}, "keyPoints": {}}},
    }


class MultiBatchTraceTests(unittest.TestCase):
    def test_trace_accumulates_across_batches(self) -> None:
        batch1 = [{"view_id": "v1", "detection_index": 0, "original_bbox": [1, 1, 9, 4], "geometry_accepted": True, "recognition_attempted": True, "recognition_accepted": True, "outcome": "kept", "batch_id": 0, "level": "global"}]
        batch2 = [{"view_id": "v2", "detection_index": 0, "original_bbox": [12, 1, 20, 4], "geometry_accepted": True, "recognition_attempted": True, "recognition_accepted": True, "outcome": "kept", "batch_id": 1, "level": "component"}]
        combined = batch1 + batch2
        gt = build_raw_ground_truth(annotation(), case_id="0001")
        report = evaluate_detection_candidates(combined, gt)
        self.assertEqual(2, report["total_candidates"])
        self.assertEqual(2, report["stages"]["geometry_accepted"]["candidate_count"])

    def test_trace_from_second_image_not_mixed(self) -> None:
        batch1 = [{"view_id": "v1", "detection_index": 0, "original_bbox": [1, 1, 9, 4], "geometry_accepted": True, "recognition_attempted": True, "recognition_accepted": True, "outcome": "kept", "batch_id": 0}]
        gt = build_raw_ground_truth(annotation(), case_id="0001")
        report1 = evaluate_detection_candidates(batch1, gt)
        report2 = evaluate_detection_candidates([], gt)
        self.assertEqual(1, report1["total_candidates"])
        self.assertEqual(0, report2["total_candidates"])


class CoordinateRestoreTests(unittest.TestCase):
    def test_original_bbox_preferred_over_view_bbox(self) -> None:
        entry = {
            "view_id": "v1", "detection_index": 0,
            "view_bbox": [0, 0, 8, 3],
            "original_bbox": [1, 1, 9, 4],
            "geometry_accepted": True, "recognition_attempted": True,
            "recognition_accepted": True, "outcome": "kept", "batch_id": 0,
        }
        gt = build_raw_ground_truth(annotation(), case_id="0001")
        report = evaluate_detection_candidates([entry], gt)
        self.assertEqual(1, report["stages"]["geometry_accepted"]["true_positive"])

    def test_fallback_to_view_bbox_when_no_original(self) -> None:
        entry = {
            "view_id": "v1", "detection_index": 0,
            "view_bbox": [1, 1, 9, 4],
            "geometry_accepted": True, "recognition_attempted": True,
            "recognition_accepted": True, "outcome": "kept", "batch_id": 0,
        }
        gt = build_raw_ground_truth(annotation(), case_id="0001")
        report = evaluate_detection_candidates([entry], gt)
        self.assertEqual(1, report["stages"]["geometry_accepted"]["true_positive"])


class IndependentStageTests(unittest.TestCase):
    def test_empty_recognition_does_not_change_detection_tp(self) -> None:
        entry = {
            "view_id": "v1", "detection_index": 0,
            "original_bbox": [1, 1, 9, 4],
            "geometry_accepted": True,
            "recognition_attempted": True,
            "recognition_accepted": False,
            "outcome": "rejected",
            "reason": "empty_recognition",
            "batch_id": 0,
        }
        gt = build_raw_ground_truth(annotation(), case_id="0001")
        report = evaluate_detection_candidates([entry], gt)
        self.assertEqual(1, report["stages"]["geometry_accepted"]["true_positive"])
        self.assertEqual(0, report["stages"]["recognition_accepted"]["true_positive"])

    def test_geometry_rejected_excluded_from_detection(self) -> None:
        entry = {
            "view_id": "v1", "detection_index": 0,
            "original_bbox": [1, 1, 9, 4],
            "geometry_accepted": False,
            "recognition_attempted": False,
            "recognition_accepted": False,
            "outcome": "rejected",
            "reason": "no_source_pixel_support",
            "batch_id": 0,
        }
        gt = build_raw_ground_truth(annotation(), case_id="0001")
        report = evaluate_detection_candidates([entry], gt)
        self.assertEqual(0, report["stages"]["geometry_accepted"]["candidate_count"])
        self.assertEqual(1, report["stages"]["detected"]["candidate_count"])


class MacroMicroAverageTests(unittest.TestCase):
    def test_macro_and_micro_differ_with_unequal_denominators(self) -> None:
        reports = [
            {"case_id": "a", "denominator": 10, "exact_count": 9, "exact_rate": 0.9, "mean_cer": 0.1, "total_edits": 5, "total_gt_chars": 50},
            {"case_id": "b", "denominator": 100, "exact_count": 40, "exact_rate": 0.4, "mean_cer": 0.5, "total_edits": 200, "total_gt_chars": 400},
        ]
        agg = aggregate_recognition(reports)
        self.assertAlmostEqual(0.65, agg["macro_exact"])
        self.assertAlmostEqual(49 / 110, agg["micro_exact"])
        self.assertAlmostEqual(0.3, agg["macro_mean_cer"])
        self.assertAlmostEqual(205 / 450, agg["corpus_cer"])

    def test_empty_reports_give_none(self) -> None:
        agg = aggregate_recognition([])
        self.assertIsNone(agg["macro_exact"])
        self.assertIsNone(agg["micro_exact"])


class BindingOracleNoHintShortcutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gt = build_raw_ground_truth(annotation(), case_id="0001")

    def test_trusted_owner_not_directly_returned(self) -> None:
        def classify_fn(text, bbox):
            return "designator" if text == "R1" else "value"

        def bind_fn(text, bbox, hint):
            return "C_WRONG"

        report = binding_oracle(self.gt, classify_fn, bind_fn)
        for obj in report["objects"]:
            self.assertFalse(obj["owner_bound_free"])
            self.assertEqual("oracle_replaced", obj["owner_source_trusted"])
        self.assertIsNone(report["owner_bound_trusted_rate"])

    def test_free_condition_uses_actual_prediction(self) -> None:
        def classify_fn(text, bbox):
            return "designator" if text == "R1" else "value"

        def bind_fn(text, bbox, hint):
            return "C_1"

        report = binding_oracle(self.gt, classify_fn, bind_fn)
        self.assertEqual(2, report["owner_bound_free_count"])

    def test_geometry_source_labeled(self) -> None:
        report = binding_oracle(
            self.gt,
            lambda t, b: "designator",
            lambda t, b, h: "C_1",
            geometry_source="predicted",
        )
        self.assertEqual("predicted", report["geometry_source"])


if __name__ == "__main__":
    unittest.main()
