"""OCR-02 oracle triage: detection vs recognition vs binding priority."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "src"))

from pcbparse.evaluation.ocr_oracle import (  # noqa: E402
    binding_oracle,
    recognition_oracle,
    triage_priority,
)
from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth  # noqa: E402


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
    texts = {
        "T_D1": _text("R1", "designator", x1=1),
        "T_V1": _text("10k", "value", x1=12),
    }
    p1 = {
        "pinId": "P_1",
        "edgeId": "E_1",
        "pinNumberByOCR": None,
        "pinNumberByVLM": None,
        "pinNameByOCR": None,
        "pinNameByVLM": None,
        "tip": [5, 10],
        "base": [2, 10],
    }
    return {
        "canvas": {"width": 40, "height": 30},
        "nowCls": "unit",
        "components": {
            "C_1": {
                "cpntId": "C_1",
                "state": "active",
                "category": "resistor",
                "location": {"x1": 0, "y1": 5, "x2": 10, "y2": 15},
                "pins": {"P_1": p1},
                "associatedTexts": {"T_D1": texts["T_D1"], "T_V1": texts["T_V1"]},
                "searchNameByOCR": None,
                "searchNameByVLM": None,
                "searchNameByLib": None,
            },
        },
        "pins": {"P_1": p1},
        "texts": texts,
        "edges": {
            "E_1": {
                "edgeId": "E_1",
                "state": "active",
                "netlabel": None,
                "connect": {"P_1": {"pinId": "P_1", "cpntId": "C_1", "location": [5, 10]}},
                "keyPointPaths": {"wire": [[[5, 10], [8, 10]]]},
                "keyPoints": {},
            }
        },
    }


class RecognitionOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gt = build_raw_ground_truth(annotation(), case_id="0001")

    def test_perfect_recognizer_passes(self) -> None:
        mapping = {"R1": "R1", "10k": "10k"}
        report = recognition_oracle(self.gt, lambda bbox: mapping.get("R1", ""))
        self.assertEqual(2, report["denominator"])
        self.assertIsNotNone(report["exact_rate"])

    def test_recognizer_returning_none_fails(self) -> None:
        report = recognition_oracle(self.gt, lambda bbox: None)
        self.assertEqual(0, report["exact_count"])
        for obj in report["objects"]:
            self.assertEqual("empty_recognition_as_deletion", obj["reason"])
            self.assertEqual(1.0, obj["cer"])

    def test_partial_recognition_reports_cer(self) -> None:
        def recognize(bbox):
            if bbox[0] < 10:
                return "R1"
            return "4k7"

        report = recognition_oracle(self.gt, recognize)
        self.assertEqual(1, report["exact_count"])
        self.assertGreater(report["mean_cer"], 0.0)


class BindingOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gt = build_raw_ground_truth(annotation(), case_id="0001")

    def test_perfect_classifier_and_binder(self) -> None:
        def classify(text, bbox):
            return "designator" if text == "R1" else "value"

        def bind(text, bbox, hint):
            return "C_1"

        report = binding_oracle(self.gt, classify, bind)
        self.assertEqual(2, report["denominator"])
        self.assertEqual(2, report["field_classified_count"])
        self.assertEqual(2, report["owner_bound_free_count"])
        self.assertIsNone(report["owner_bound_trusted_rate"])
        self.assertEqual("oracle_replaced", report["owner_source_trusted"])

    def test_wrong_field_classification_counted(self) -> None:
        report = binding_oracle(
            self.gt,
            lambda text, bbox: "value",
            lambda text, bbox, hint: hint or "C_1",
        )
        self.assertEqual(1, report["field_classified_count"])

    def test_unbound_owner_counted(self) -> None:
        report = binding_oracle(
            self.gt,
            lambda text, bbox: "designator" if text == "R1" else "value",
            lambda text, bbox, hint: None,
        )
        self.assertEqual(0, report["owner_bound_free_count"])
        self.assertIsNone(report["owner_bound_trusted_rate"])


class TriageTests(unittest.TestCase):
    def test_low_recognition_is_primary(self) -> None:
        result = triage_priority(
            image_only_report={"detection_rate": 0.8},
            recognition_report={"exact_rate": 0.3, "mean_cer": 0.5},
            binding_report={"field_classified_rate": 0.9, "owner_bound_free_rate": 0.9, "owner_bound_trusted_rate": 0.9},
        )
        self.assertIn("recognition", result["priority"])

    def test_high_recognition_low_binding_prioritizes_binding(self) -> None:
        result = triage_priority(
            image_only_report={"detection_rate": 0.8},
            recognition_report={"exact_rate": 0.9, "mean_cer": 0.05},
            binding_report={"field_classified_rate": 0.4, "owner_bound_free_rate": 0.3, "owner_bound_trusted_rate": 0.35},
        )
        self.assertIn("binding_field", result["priority"])
        self.assertIn("binding_owner", result["priority"])

    def test_all_oracles_high_prioritizes_detection(self) -> None:
        result = triage_priority(
            image_only_report={"detection_rate": 0.3},
            recognition_report={"exact_rate": 0.95, "mean_cer": 0.02},
            binding_report={"field_classified_rate": 0.95, "owner_bound_free_rate": 0.9, "owner_bound_trusted_rate": 0.95},
        )
        self.assertIn("detection", result["priority"])

    def test_owner_gap_suggests_binding_owner(self) -> None:
        result = triage_priority(
            image_only_report=None,
            recognition_report={"exact_rate": 0.9, "mean_cer": 0.05},
            binding_report={"field_classified_rate": 0.9, "owner_bound_free_rate": 0.5, "owner_bound_trusted_rate": 0.95},
        )
        self.assertIn("binding_owner", result["priority"])

    def test_no_reports_defaults_to_not_measured(self) -> None:
        result = triage_priority(
            image_only_report=None, recognition_report=None, binding_report=None
        )
        self.assertEqual(["not_measured"], result["priority"])


if __name__ == "__main__":
    unittest.main()
