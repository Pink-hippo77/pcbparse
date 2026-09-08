from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pcbparse.evaluation.masked_metrics import bbox_iou, evaluate_masked_case
from pcbparse.evaluation.raw_cli import (
    AccuracyManifestError,
    PredictionRecord,
    assert_accuracy_artifact,
    evaluate_prediction_manifest,
    read_prediction_manifest,
)
from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth
from pcbparse.ir.models import (
    BBox,
    CircuitIR,
    Component,
    ElectricalNet,
    Evidence,
    IRState,
    ImageInfo,
    Pin,
    Point,
    WireSegment,
)


def raw_annotation() -> dict:
    texts = {
        "T_D1": _text("R1", "designator"),
        "T_D2": _text("R2", "designator"),
        "T_V1": _text("10k", "value"),
        "T_V2": _text("1%", "value"),
        "T_V3": _text("22k", "value"),
        "T_NUM1": _text("1", "pinNumber"),
        "T_NAME1": _text("A", "pinName"),
    }
    p1 = {
        "pinId": "P_1",
        "edgeId": "E_1",
        "pinNumberByOCR": "T_NUM1",
        "pinNumberByVLM": None,
        "pinNameByOCR": "T_NAME1",
        "pinNameByVLM": None,
        "tip": [5, 10],
        "base": [2, 10],
    }
    p2 = {
        "pinId": "P_2",
        "edgeId": "E_1",
        "pinNumberByOCR": None,
        "pinNumberByVLM": None,
        "pinNameByOCR": None,
        "pinNameByVLM": None,
        "tip": [35, 10],
        "base": [38, 10],
    }
    orphan = {
        "pinId": "P_3",
        "edgeId": None,
        "pinNumberByOCR": None,
        "pinNumberByVLM": None,
        "pinNameByOCR": None,
        "pinNameByVLM": None,
        "tip": [20, 20],
        "base": [20, 23],
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
                "associatedTexts": {
                    "T_D1": texts["T_D1"],
                    "T_V1": texts["T_V1"],
                    "T_V2": texts["T_V2"],
                },
                "searchNameByOCR": None,
                "searchNameByVLM": None,
                "searchNameByLib": None,
            },
            "C_2": {
                "cpntId": "C_2",
                "state": "active",
                "category": "resistor",
                "location": {"x1": 30, "y1": 5, "x2": 40, "y2": 15},
                "pins": {"P_2": p2},
                "associatedTexts": {
                    "T_D2": texts["T_D2"],
                    "T_V3": texts["T_V3"],
                },
                "searchNameByOCR": None,
                "searchNameByVLM": None,
                "searchNameByLib": None,
            },
            "C_3": {
                "cpntId": "C_3",
                "state": "deleted",
                "category": "resistor",
                "location": {"x1": 15, "y1": 1, "x2": 25, "y2": 8},
                "pins": {},
                "associatedTexts": {},
                "searchNameByOCR": None,
                "searchNameByVLM": None,
                "searchNameByLib": None,
            },
        },
        "pins": {"P_1": p1, "P_2": p2, "P_3": orphan},
        "texts": texts,
        "edges": {
            "E_1": {
                "edgeId": "E_1",
                "state": "active",
                "netlabel": None,
                "connect": {
                    "P_1": {"pinId": "P_1", "cpntId": "C_1", "location": [5, 10]},
                    "P_2": {"pinId": "P_2", "cpntId": "C_2", "location": [35, 10]},
                },
                "keyPointPaths": {"wire": [[[5, 10], [35, 10]]]},
                "keyPoints": {},
            },
            "E_2": {
                "edgeId": "E_2",
                "state": "deleted",
                "netlabel": None,
                "connect": {},
                "keyPointPaths": {},
                "keyPoints": {},
            },
        },
    }


def _text(content: str, details: str) -> dict:
    return {
        "tid": "unused",
        "state": "active",
        "category": "params",
        "content": content,
        "details": details,
        "location": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
    }


def prediction_ir(*, include_bad_owner_pin: bool = False) -> CircuitIR:
    pins = [
        Pin(
            internal_id="pred-pin-1",
            component_id="pred-component-1",
            number="1",
            name="A",
            point=Point(x=5, y=10),
        ),
        Pin(
            internal_id="pred-pin-2",
            component_id="pred-component-2",
            number="2",
            name="",
            point=Point(x=35, y=10),
        ),
    ]
    if include_bad_owner_pin:
        pins.append(
            Pin(
                internal_id="pred-pin-bad-owner",
                component_id="pred-component-2",
                number="9",
                point=Point(x=5, y=10),
            )
        )
    return CircuitIR(
        state=IRState.SEMANTIC,
        image=ImageInfo(
            image_id="0001",
            path="image.png",
            width=40,
            height=30,
            sha256="abc123",
        ),
        components=[
            Component(
                internal_id="pred-component-1",
                key="R1",
                category="resistor",
                value="10k",
                bbox=BBox(x1=0, y1=5, x2=10, y2=15),
            ),
            Component(
                internal_id="pred-component-2",
                key="R2",
                category="resistor",
                value="22k",
                bbox=BBox(x1=30, y1=5, x2=40, y2=15),
            ),
        ],
        pins=pins,
        segments=[
            WireSegment(
                internal_id="pred-segment-1",
                start=Point(x=5, y=11),
                end=Point(x=35, y=11),
            )
        ],
        electrical_nets=[
            ElectricalNet(
                internal_id="pred-net-1",
                pins=["pred-pin-1", "pred-pin-2"],
                segments=["pred-segment-1"],
            )
        ],
        metadata={"provider": {"name": "HeuristicVisionProvider", "submission_safe": True}},
    )


class RawGroundTruthTests(unittest.TestCase):
    def test_masks_are_explicit_and_never_filled_with_surrogates(self) -> None:
        gt = build_raw_ground_truth(raw_annotation(), case_id="0001")
        coverage = gt.coverage()
        self.assertEqual(3, coverage["objects"]["component"]["total"])
        self.assertEqual(2, coverage["objects"]["component"]["eligible"])
        self.assertEqual(
            1,
            coverage["objects"]["component"]["mask_reasons"]["inactive_component"],
        )
        self.assertEqual(1, coverage["objects"]["pin"]["mask_reasons"]["no_active_owner"])
        self.assertEqual(
            1,
            coverage["fields"]["component.value"]["mask_reasons"][
                "ambiguous_component_value"
            ],
        )
        c1 = next(item for item in gt.components if item.raw_id == "C_1")
        self.assertFalse(c1.value.mask.eligible)
        self.assertIsNone(c1.value.value)
        values = [
            target.value
            for component in gt.components
            for target in (component.designator, component.value, component.search_name)
            if target.value is not None
        ]
        self.assertFalse(any(value.startswith("DATASET_") for value in values))
        self.assertIn(frozenset({"P_1", "P_2"}), gt.connectivity_nets)


class MaskedMetricTests(unittest.TestCase):
    def test_component_pin_wire_and_graph_are_scored_separately(self) -> None:
        gt = build_raw_ground_truth(raw_annotation(), case_id="0001")
        report = evaluate_masked_case(
            prediction_ir(include_bad_owner_pin=True),
            gt,
            component_iou_threshold=0.5,
            pin_distance_tolerance_px=2,
            wire_tolerance_px=1,
        )
        self.assertEqual((2, 0, 0), _prf_tuple(report["component"]["detection"]))
        self.assertEqual(1.0, report["component"]["category"]["accuracy"])
        self.assertEqual((2, 1, 0), _prf_tuple(report["pin"]["detection"]))
        self.assertEqual(1, report["pin"]["fields"]["number"]["eligible"])
        self.assertEqual(1.0, report["pin"]["fields"]["number"]["exact_accuracy_end_to_end"])
        self.assertEqual(1.0, report["wire"]["precision"])
        self.assertEqual(1.0, report["wire"]["recall"])
        self.assertEqual((1, 0, 0), _prf_tuple(report["graph"]["pairwise_connectivity"]))

    def test_iou_does_not_accept_disjoint_boxes_at_coordinate_tolerance(self) -> None:
        self.assertEqual(0.0, bbox_iou((0, 0, 1, 1), (20, 20, 21, 21)))


class AccuracyManifestTests(unittest.TestCase):
    def test_manifest_and_ir_both_reject_annotation_backed_providers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ir_path = root / "ir.json"
            ir_path.write_text(prediction_ir().model_dump_json(), encoding="utf-8")
            manifest = root / "predictions.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "case_id": "0001",
                        "ir_path": str(ir_path),
                        "provider_kind": "dataset_debug",
                        "sidecar_access": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(AccuracyManifestError):
                read_prediction_manifest(manifest)

            leaked = prediction_ir()
            leaked.components[0].evidence.append(
                Evidence(evidence_id="leak", source="dataset")
            )
            record = PredictionRecord("0001", ir_path, "production", False, None)
            with self.assertRaises(AccuracyManifestError):
                assert_accuracy_artifact(record, leaked)

    def test_cli_core_reads_ir_path_and_emits_mask_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            sample = dataset / "0001"
            sample.mkdir(parents=True)
            (sample / "sample_CVJsonStd.json").write_text(
                json.dumps(raw_annotation()), encoding="utf-8"
            )
            ir_path = root / "ir.json"
            ir_path.write_text(prediction_ir().model_dump_json(), encoding="utf-8")
            manifest = root / "predictions.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "case_id": "0001",
                        "ir_path": str(ir_path),
                        "provider_kind": "heuristic",
                        "sidecar_access": False,
                        "image_sha256": "abc123",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = evaluate_prediction_manifest(
                manifest,
                dataset_root=dataset,
                wire_tolerance_px=1,
            )
            self.assertEqual("image-only", result["accuracy_mode"])
            self.assertEqual(1, result["aggregate"]["cases"])
            self.assertEqual(
                1,
                result["aggregate"]["coverage"]["objects"]["pin"]["masked"],
            )


def _prf_tuple(value: dict) -> tuple[int, int, int]:
    return value["tp"], value["fp"], value["fn"]


if __name__ == "__main__":
    unittest.main()
