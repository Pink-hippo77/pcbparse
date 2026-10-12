"""Targeted tests for OCR-01/02 evaluation fixes: dual ID, empty recognition,
all-rejected detection, raw vs post-processed text."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "src"))

from pcbparse.evaluation.detection_eval import evaluate_detection_candidates  # noqa: E402
from pcbparse.evaluation.ocr_oracle import recognition_oracle  # noqa: E402
from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth  # noqa: E402
from pcbparse.evaluation.text_box_eval import evaluate_text_boxes  # noqa: E402
from pcbparse.ir.models import (  # noqa: E402
    BBox,
    CircuitIR,
    Component,
    ElectricalNet,
    ImageInfo,
    IRState,
    Pin,
    Point,
    TextToken,
    WireSegment,
)


def _text(content: str, details: str, x1: int = 1) -> dict:
    return {
        "tid": "unused",
        "state": "active",
        "category": "params",
        "content": content,
        "details": details,
        "location": {"x1": x1, "y1": 1, "x2": x1 + 8, "y2": 4},
    }


def annotation_with_pin_text() -> dict:
    texts = {
        "T_D1": _text("R1", "designator", x1=1),
        "T_V1": _text("10k", "value", x1=12),
        "T_NUM1": _text("3", "pinNumber", x1=24),
    }
    p1 = {
        "pinId": "P_1",
        "edgeId": "E_1",
        "pinNumberByOCR": "T_NUM1",
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


def _token(tid: str, text: str, *, details: str = "designator", x1: float = 1) -> TextToken:
    return TextToken(
        internal_id=tid,
        raw_text=text,
        details=details,
        bbox=BBox(x1=x1, y1=1, x2=x1 + 8, y2=4),
    )


def _ir_with_tokens(tokens: list[TextToken]) -> CircuitIR:
    return CircuitIR(
        state=IRState.SEMANTIC,
        image=ImageInfo(image_id="0001", path="i.png", width=40, height=30, sha256="a"),
        components=[
            Component(
                internal_id="C_1",
                key="R1",
                category="resistor",
                value="10k",
                bbox=BBox(x1=0, y1=5, x2=10, y2=15),
            )
        ],
        pins=[
            Pin(internal_id="P_1", component_id="C_1", number="3", point=Point(x=5, y=10))
        ],
        texts=tokens,
        segments=[WireSegment(internal_id="s1", start=Point(x=5, y=10), end=Point(x=8, y=10))],
        electrical_nets=[ElectricalNet(internal_id="n1", pins=["P_1"], segments=["s1"])],
    )


class PinOwnerComparisonTests(unittest.TestCase):
    """P0: pin field owner comparison must use pin ID, not component ID."""

    def setUp(self) -> None:
        self.gt = build_raw_ground_truth(annotation_with_pin_text(), case_id="0001")

    def test_pin_field_with_pin_id_passes(self) -> None:
        t = _token("t1", "3", details="pinNumber", x1=24)
        t.pin_id = "P_1"
        t.component_id = "C_1"
        ir = _ir_with_tokens([t])
        report = evaluate_text_boxes(ir, self.gt)
        by_id = {item["object_id"]: item for item in report["objects"]}
        row = by_id["P_1.number"]
        self.assertEqual("passed", row["stages"]["owner_bound"]["status"], row["stages"])

    def test_pin_field_with_only_component_id_fails(self) -> None:
        t = _token("t1", "3", details="pinNumber", x1=24)
        t.component_id = "C_1"
        ir = _ir_with_tokens([t])
        report = evaluate_text_boxes(ir, self.gt)
        by_id = {item["object_id"]: item for item in report["objects"]}
        row = by_id["P_1.number"]
        self.assertEqual("failed", row["stages"]["owner_bound"]["status"])
        self.assertIn("unbound", row["stages"]["owner_bound"]["reason"])

    def test_pred_to_gt_pin_mapping_used(self) -> None:
        t = _token("t1", "3", details="pinNumber", x1=24)
        t.pin_id = "internal-pin-99"
        ir = _ir_with_tokens([t])
        report = evaluate_text_boxes(
            ir, self.gt, pred_to_gt_pin={"internal-pin-99": "P_1"}
        )
        by_id = {item["object_id"]: item for item in report["objects"]}
        row = by_id["P_1.number"]
        self.assertEqual("passed", row["stages"]["owner_bound"]["status"], row["stages"])


class EmptyRecognitionCERTests(unittest.TestCase):
    """P0: empty recognition must score as deletion CER, not be excluded."""

    def setUp(self) -> None:
        self.gt = build_raw_ground_truth(annotation_with_pin_text(), case_id="0001")

    def test_none_recognition_scores_full_deletion_cer(self) -> None:
        report = recognition_oracle(self.gt, lambda bbox: None)
        self.assertEqual(report["cer_count"], report["denominator"])
        for obj in report["objects"]:
            self.assertIsNotNone(obj["cer"])
            self.assertEqual(1.0, obj["cer"])
            self.assertEqual("empty_recognition_as_deletion", obj["reason"])

    def test_empty_string_recognition_scores_full_deletion_cer(self) -> None:
        report = recognition_oracle(self.gt, lambda bbox: "")
        for obj in report["objects"]:
            self.assertEqual(1.0, obj["cer"])

    def test_mixed_none_and_correct_gives_nonzero_mean_cer(self) -> None:
        calls = iter(["R1", None, "3"])

        def recognize(bbox):
            return next(calls)

        report = recognition_oracle(self.gt, recognize)
        self.assertEqual(2, report["exact_count"])
        self.assertGreater(report["mean_cer"], 0.0)
        self.assertEqual(report["cer_count"], report["denominator"])

    def test_exception_is_separate_and_excluded_from_cer(self) -> None:
        def recognize(bbox):
            if bbox[0] < 10:
                raise RuntimeError("boom")
            return "10k"

        report = recognition_oracle(self.gt, recognize)
        self.assertEqual(1, report["exception_count"])
        self.assertEqual(report["cer_count"], report["denominator"] - report["exception_count"])
        for obj in report["objects"]:
            if obj["reason"] and obj["reason"].startswith("recognizer_exception"):
                self.assertIsNone(obj["cer"])


class RawVsPostProcessedTests(unittest.TestCase):
    """raw_text and resolved_text must be reported separately."""

    def setUp(self) -> None:
        self.gt = build_raw_ground_truth(annotation_with_pin_text(), case_id="0001")

    def test_post_processed_correction_flagged(self) -> None:
        t = _token("t1", "Rl", details="designator", x1=1)
        t.component_id = "C_1"
        t.resolved_text = "R1"
        ir = _ir_with_tokens([t])
        report = evaluate_text_boxes(ir, self.gt)
        by_id = {item["object_id"]: item for item in report["objects"]}
        row = by_id["C_1.designator"]
        stage = row["stages"]["text_recognized"]
        self.assertTrue(stage["post_processed"])
        self.assertFalse(stage["raw_exact"])
        self.assertGreater(stage["raw_cer"], 0.0)

    def test_raw_text_correct_not_flagged(self) -> None:
        t = _token("t1", "R1", details="designator", x1=1)
        t.component_id = "C_1"
        ir = _ir_with_tokens([t])
        report = evaluate_text_boxes(ir, self.gt)
        by_id = {item["object_id"]: item for item in report["objects"]}
        stage = by_id["C_1.designator"]["stages"]["text_recognized"]
        self.assertFalse(stage["post_processed"])
        self.assertTrue(stage["raw_exact"])
        self.assertEqual(0.0, stage["raw_cer"])


class DetectionOnlyEvalTests(unittest.TestCase):
    """P1: detection P/R from pre-recognition candidates, independent of recognition."""

    def setUp(self) -> None:
        self.gt = build_raw_ground_truth(annotation_with_pin_text(), case_id="0001")

    def test_detection_hit_even_if_recognition_would_fail(self) -> None:
        trace = [
            {
                "view_id": "v1", "detection_index": 0, "original_bbox": [1, 1, 9, 4],
                "geometry_accepted": True, "recognition_attempted": True,
                "recognition_accepted": True, "outcome": "kept", "batch_id": 0,
            },
            {
                "view_id": "v1", "detection_index": 1, "original_bbox": [12, 1, 20, 4],
                "geometry_accepted": True, "recognition_attempted": True,
                "recognition_accepted": True, "outcome": "kept", "batch_id": 0,
            },
            {
                "view_id": "v1", "detection_index": 2, "original_bbox": [24, 1, 32, 4],
                "geometry_accepted": True, "recognition_attempted": True,
                "recognition_accepted": True, "outcome": "kept", "batch_id": 0,
            },
        ]
        report = evaluate_detection_candidates(trace, self.gt)
        self.assertEqual(3, report["gt_box_count"])
        self.assertEqual(3, report["stages"]["geometry_accepted"]["true_positive"])
        self.assertEqual(1.0, report["stages"]["geometry_accepted"]["precision"])
        self.assertEqual(1.0, report["stages"]["geometry_accepted"]["recall"])

    def test_all_rejected_gives_zero_recall_with_reasons(self) -> None:
        trace = [
            {
                "view_id": "v1", "detection_index": 0,
                "geometry_accepted": False, "recognition_attempted": False,
                "recognition_accepted": False, "outcome": "rejected",
                "reason": "no_source_pixel_support", "batch_id": 0,
            },
            {
                "view_id": "v1", "detection_index": 1,
                "geometry_accepted": False, "recognition_attempted": False,
                "recognition_accepted": False, "outcome": "rejected",
                "reason": "empty_recognition", "batch_id": 0,
            },
        ]
        report = evaluate_detection_candidates(trace, self.gt)
        self.assertEqual(0, report["stages"]["geometry_accepted"]["candidate_count"])
        self.assertEqual(2, len(report["rejected"]))
        self.assertEqual(0.0, report["stages"]["geometry_accepted"]["recall"])
        self.assertIsNone(report["stages"]["geometry_accepted"]["precision"])
        self.assertEqual(3, len(report["stages"]["geometry_accepted"]["missed_object_ids"]))
        self.assertIn("no_source_pixel_support", report["reject_reasons"])
        self.assertIn("empty_recognition", report["reject_reasons"])

    def test_empty_trace_gives_not_measured_precision(self) -> None:
        report = evaluate_detection_candidates([], self.gt)
        self.assertIsNone(report["stages"]["geometry_accepted"]["precision"])
        self.assertEqual(0.0, report["stages"]["geometry_accepted"]["recall"])
        self.assertEqual(3, report["stages"]["geometry_accepted"]["false_negative"])

    def test_false_positives_counted(self) -> None:
        trace = [
            {
                "view_id": "v1", "detection_index": 0, "original_bbox": [1, 1, 9, 4],
                "geometry_accepted": True, "recognition_attempted": True,
                "recognition_accepted": True, "outcome": "kept", "batch_id": 0,
            },
            {
                "view_id": "v1", "detection_index": 1, "original_bbox": [35, 20, 39, 28],
                "geometry_accepted": True, "recognition_attempted": True,
                "recognition_accepted": True, "outcome": "kept", "batch_id": 0,
            },
        ]
        report = evaluate_detection_candidates(trace, self.gt)
        self.assertEqual(1, report["stages"]["geometry_accepted"]["true_positive"])
        self.assertEqual(1, report["stages"]["geometry_accepted"]["false_positive"])
        self.assertEqual(2, report["stages"]["geometry_accepted"]["false_negative"])


if __name__ == "__main__":
    unittest.main()
