"""OCR-01 real text-box evaluation: 1:1 matching, exact/CER/field/owner."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "src"))

from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth  # noqa: E402
from pcbparse.evaluation.text_box_eval import (  # noqa: E402
    TEXT_BOX_STAGES,
    character_error_rate,
    evaluate_text_boxes,
)
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


def annotation_with_bboxes() -> dict:
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


def prediction_tokens(tokens: list[TextToken]) -> CircuitIR:
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
            Pin(internal_id="P_1", component_id="C_1", number="1", point=Point(x=5, y=10))
        ],
        texts=tokens,
        segments=[WireSegment(internal_id="s1", start=Point(x=5, y=10), end=Point(x=8, y=10))],
        electrical_nets=[ElectricalNet(internal_id="n1", pins=["P_1"], segments=["s1"])],
    )


def _token(tid: str, text: str, *, details: str = "designator", x1: float = 1) -> TextToken:
    return TextToken(
        internal_id=tid,
        raw_text=text,
        details=details,
        bbox=BBox(x1=x1, y1=1, x2=x1 + 8, y2=4),
    )


class CERTests(unittest.TestCase):
    def test_exact_match_is_zero(self) -> None:
        self.assertEqual(0.0, character_error_rate("R1", "R1"))

    def test_one_substitution_in_two_chars(self) -> None:
        self.assertAlmostEqual(0.5, character_error_rate("R1", "R2"))

    def test_empty_reference_is_one(self) -> None:
        self.assertEqual(1.0, character_error_rate("", "X"))

    def test_empty_hypothesis_is_one(self) -> None:
        self.assertEqual(1.0, character_error_rate("R1", ""))


class TextBoxEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gt = build_raw_ground_truth(annotation_with_bboxes(), case_id="0001")

    def test_exact_match_passes_all_stages(self) -> None:
        t1 = _token("t1", "R1", details="designator", x1=1)
        t1.component_id = "C_1"
        t2 = _token("t2", "10k", details="value", x1=12)
        t2.component_id = "C_1"
        ir = prediction_tokens([t1, t2])
        report = evaluate_text_boxes(ir, self.gt)
        self.assertEqual(2, report["gt_box_count"])
        self.assertEqual(2, report["matched_count"])
        for obj in report["objects"]:
            self.assertIsNone(obj["first_failed_stage"], obj["object_id"])
            for stage in TEXT_BOX_STAGES:
                self.assertEqual("passed", obj["stages"][stage]["status"], f"{obj['object_id']}.{stage}")

    def test_undetected_box_fails_detection(self) -> None:
        ir = prediction_tokens([_token("t1", "R1", details="designator", x1=1)])
        report = evaluate_text_boxes(ir, self.gt)
        by_id = {item["object_id"]: item for item in report["objects"]}
        self.assertEqual("box_detected", by_id["C_1.value"]["first_failed_stage"])
        self.assertEqual("not_measured", by_id["C_1.value"]["stages"]["text_recognized"]["status"])

    def test_wrong_text_reports_cer(self) -> None:
        ir = prediction_tokens(
            [_token("t1", "R1", details="designator", x1=1), _token("t2", "4k7", details="value", x1=12)]
        )
        report = evaluate_text_boxes(ir, self.gt)
        by_id = {item["object_id"]: item for item in report["objects"]}
        row = by_id["C_1.value"]
        self.assertEqual("text_recognized", row["first_failed_stage"])
        cer = row["stages"]["text_recognized"]["cer"]
        self.assertGreater(cer, 0.0)

    def test_wrong_field_classification_fails(self) -> None:
        t1 = _token("t1", "R1", details="value", x1=1)
        t1.component_id = "C_1"
        t2 = _token("t2", "10k", details="value", x1=12)
        t2.component_id = "C_1"
        ir = prediction_tokens([t1, t2])
        report = evaluate_text_boxes(ir, self.gt)
        by_id = {item["object_id"]: item for item in report["objects"]}
        self.assertEqual("field_classified", by_id["C_1.designator"]["first_failed_stage"])

    def test_wrong_owner_binding_fails(self) -> None:
        t1 = _token("t1", "R1", details="designator", x1=1)
        t1.component_id = "C_OTHER"
        t2 = _token("t2", "10k", details="value", x1=12)
        t2.component_id = "C_1"
        ir = prediction_tokens([t1, t2])
        report = evaluate_text_boxes(ir, self.gt)
        by_id = {item["object_id"]: item for item in report["objects"]}
        self.assertEqual("owner_bound", by_id["C_1.designator"]["first_failed_stage"])

    def test_unmatched_predicted_tokens_are_false_positives(self) -> None:
        ir = prediction_tokens(
            [
                _token("t1", "R1", details="designator", x1=1),
                _token("t2", "10k", details="value", x1=12),
                _token("t3", "GHOST", details="value", x1=25),
            ]
        )
        report = evaluate_text_boxes(ir, self.gt)
        self.assertEqual(1, report["false_positive_count"])
        self.assertEqual("t3", report["false_positives"][0]["token_id"])

    def test_not_measured_when_no_trusted_bbox(self) -> None:
        annotation = annotation_with_bboxes()
        annotation["texts"]["T_D1"].pop("location")
        gt = build_raw_ground_truth(annotation, case_id="0001")
        ir = prediction_tokens([_token("t2", "10k", details="value", x1=12)])
        report = evaluate_text_boxes(ir, gt)
        reasons = {item["object_id"]: item["reason"] for item in report["denominator_excluded"]}
        self.assertIn("C_1.designator", reasons)
        self.assertEqual("no_trusted_gt_text_bbox", reasons["C_1.designator"])
        self.assertEqual(1, report["gt_box_count"])


if __name__ == "__main__":
    unittest.main()
