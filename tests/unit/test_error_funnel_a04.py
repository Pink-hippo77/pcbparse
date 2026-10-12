"""A04 per-object error funnels: three object-consistent funnels with real IDs.

Covers text-field / pin / net funnels, wrong-owner and wrong-field failure
attribution, unmeasured-not-zero semantics, and oracle crossing refusal.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "src"))
sys.path.insert(0, str(WORKSPACE / "scripts"))

import error_funnel_a04 as a04  # noqa: E402
from pcbparse.evaluation.error_funnel import (  # noqa: E402
    NET_STAGES,
    PIN_STAGES,
    TEXT_FIELD_STAGES,
    build_error_funnel,
    not_measured_funnel,
)
from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth  # noqa: E402
from pcbparse.ir.models import (  # noqa: E402
    BBox,
    CircuitIR,
    Component,
    ElectricalNet,
    ImageInfo,
    IRState,
    Pin,
    Point,
    WireSegment,
)


def _text(content: str, details: str) -> dict:
    return {
        "tid": "unused",
        "state": "active",
        "category": "params",
        "content": content,
        "details": details,
        "location": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
    }


def raw_annotation() -> dict:
    texts = {
        "T_D1": _text("R1", "designator"),
        "T_D2": _text("R2", "designator"),
        "T_V1": _text("10k", "value"),
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
            "C_2": {
                "cpntId": "C_2",
                "state": "active",
                "category": "resistor",
                "location": {"x1": 30, "y1": 5, "x2": 40, "y2": 15},
                "pins": {"P_2": p2},
                "associatedTexts": {"T_D2": texts["T_D2"], "T_V3": texts["T_V3"]},
                "searchNameByOCR": None,
                "searchNameByVLM": None,
                "searchNameByLib": None,
            },
        },
        "pins": {"P_1": p1, "P_2": p2},
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
        },
    }


def prediction_ir(*, swapped_values: bool = False) -> CircuitIR:
    value_1, value_2 = ("22k", "10k") if swapped_values else ("10k", "22k")
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
    return CircuitIR(
        state=IRState.SEMANTIC,
        image=ImageInfo(
            image_id="0001", path="image.png", width=40, height=30, sha256="abc123"
        ),
        components=[
            Component(
                internal_id="pred-component-1",
                key="R1",
                category="resistor",
                value=value_1,
                bbox=BBox(x1=0, y1=5, x2=10, y2=15),
            ),
            Component(
                internal_id="pred-component-2",
                key="R2",
                category="resistor",
                value=value_2,
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


def official_output() -> dict:
    return {
        "components": {
            "R1": {"type": "resistor", "Name": None, "value": "10k", "bbox": [0, 15, 10, 25]},
            "R2": {"type": "resistor", "Name": None, "value": "22k", "bbox": [30, 15, 40, 25]},
        },
        "pins": {
            "R1": {"pin_1": {"pinname": "A", "point": {"x": 5.0, "y": 20.0}}},
            "R2": {"pin_2": {"pinname": "", "point": {"x": 35.0, "y": 20.0}}},
        },
        "nets": {},
    }


def _stage(funnel: dict, name: str) -> dict:
    return next(row for row in funnel["stages"] if row["stage"] == name)


class ErrorFunnelObjectConsistencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gt = build_raw_ground_truth(raw_annotation(), case_id="0001")

    def test_three_object_consistent_funnels_carry_ids_and_reasons(self) -> None:
        report = build_error_funnel(
            prediction_ir(),
            self.gt,
            official_output=official_output(),
            wire_tolerance_px=3,
        )
        self.assertEqual({"text_field", "pin", "net"}, set(report["funnels"]))
        text_funnel = report["funnels"]["text_field"]
        self.assertEqual([row["stage"] for row in text_funnel["stages"]], list(TEXT_FIELD_STAGES))
        for row in text_funnel["stages"]:
            self.assertIsInstance(row["included_object_ids"], list)
            self.assertIsInstance(row["excluded"], list)
            for item in row["excluded"]:
                self.assertIn("object_id", item)
                self.assertTrue(item["reason"])
        self.assertGreater(text_funnel["denominator"], 0)
        stage_denominators = [row["denominator"] for row in text_funnel["stages"]]
        self.assertTrue(all(value is not None for value in stage_denominators))
        self.assertEqual("C_1.designator", text_funnel["objects"][0]["object_id"])
        pin_funnel = report["funnels"]["pin"]
        self.assertEqual([row["stage"] for row in pin_funnel["stages"]], list(PIN_STAGES))
        net_funnel = report["funnels"]["net"]
        self.assertEqual([row["stage"] for row in net_funnel["stages"]], list(NET_STAGES))

    def test_masked_fields_land_in_denominator_excluded_with_reasons(self) -> None:
        report = build_error_funnel(prediction_ir(), self.gt)
        excluded = {
            item["object_id"]: item["reason"]
            for item in report["funnels"]["text_field"]["denominator_excluded"]
        }
        self.assertIn("C_1.search_name", excluded)
        self.assertIn("missing_component_search_name", excluded["C_1.search_name"])
        self.assertIn("P_2.number", excluded)

    def test_good_prediction_passes_all_three_funnels(self) -> None:
        report = build_error_funnel(
            prediction_ir(),
            self.gt,
            official_output=official_output(),
            wire_tolerance_px=3,
        )
        for kind, final_stage in (
            ("text_field", "output_correct"),
            ("pin", "complete_pin_hit"),
            ("net", "segments_correct"),
        ):
            row = _stage(report["funnels"][kind], final_stage)
            self.assertEqual("measured", row["status"], kind)
            self.assertEqual([], row["excluded"], kind)
            self.assertEqual(row["count"], row["evaluated"], kind)
        pin_final = _stage(report["funnels"]["pin"], "complete_pin_hit")
        self.assertEqual(1, pin_final["count"])
        self.assertEqual(2, pin_final["denominator"])
        unmeasured = {item["object_id"]: item["reason"] for item in pin_final["not_measured"]}
        self.assertEqual({"pin:P_2": "no_visible_pin_fields"}, unmeasured)


class TextFieldFunnelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gt = build_raw_ground_truth(raw_annotation(), case_id="0001")

    def test_wrong_owner_text_fails_owner_stage_with_reason(self) -> None:
        report = build_error_funnel(prediction_ir(swapped_values=True), self.gt)
        by_id = {item["object_id"]: item for item in report["funnels"]["text_field"]["objects"]}
        value_row = by_id["C_2.value"]
        self.assertEqual("owner_correct", value_row["first_failed_stage"])
        self.assertEqual("passed", value_row["stages"]["char_correct"]["status"])
        self.assertEqual("passed", value_row["stages"]["field_correct"]["status"])
        self.assertEqual(
            "exact_text_on_wrong_owner", value_row["stages"]["owner_correct"]["reason"]
        )
        owner_row = _stage(report["funnels"]["text_field"], "owner_correct")
        self.assertIn("C_2.value", [item["object_id"] for item in owner_row["excluded"]])

    def test_unrecognized_text_fails_char_stage(self) -> None:
        ir = prediction_ir()
        ir.components[0].value = "100k"
        ir.components[1].value = "47k"
        report = build_error_funnel(ir, self.gt)
        by_id = {item["object_id"]: item for item in report["funnels"]["text_field"]["objects"]}
        row = by_id["C_1.value"]
        self.assertEqual("char_correct", row["first_failed_stage"])
        self.assertEqual(
            "normalized_text_not_recognized", row["stages"]["char_correct"]["reason"]
        )


class PinFunnelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gt = build_raw_ground_truth(raw_annotation(), case_id="0001")

    def test_localized_wrong_owner_pin_fails_owner_stage(self) -> None:
        ir = prediction_ir()
        ir.pins[0].component_id = "pred-component-2"
        report = build_error_funnel(ir, self.gt)
        by_id = {item["object_id"]: item for item in report["funnels"]["pin"]["objects"]}
        row = by_id["pin:P_1"]
        self.assertEqual("passed", row["stages"]["localized"]["status"])
        self.assertEqual("owner_correct", row["first_failed_stage"])
        self.assertEqual("owner_binding_wrong", row["stages"]["owner_correct"]["reason"])
        self.assertEqual(
            "not_measured", row["stages"]["number_name_correct"]["status"]
        )

    def test_number_mismatch_fails_fields_stage(self) -> None:
        ir = prediction_ir()
        ir.pins[0].number = "7"
        report = build_error_funnel(ir, self.gt)
        by_id = {item["object_id"]: item for item in report["funnels"]["pin"]["objects"]}
        row = by_id["pin:P_1"]
        self.assertEqual("number_name_correct", row["first_failed_stage"])
        self.assertEqual("number_mismatch", row["stages"]["number_name_correct"]["reason"])


def net_annotation() -> dict:
    def pin(pin_id: str, edge_id: str, tip: list, base: list) -> dict:
        return {
            "pinId": pin_id,
            "edgeId": edge_id,
            "pinNumberByOCR": None,
            "pinNumberByVLM": None,
            "pinNameByOCR": None,
            "pinNameByVLM": None,
            "tip": tip,
            "base": base,
        }

    def component(comp_id: str, bbox: dict, pins: dict) -> dict:
        return {
            "cpntId": comp_id,
            "state": "active",
            "category": "resistor",
            "location": bbox,
            "pins": pins,
            "associatedTexts": {},
            "searchNameByOCR": None,
            "searchNameByVLM": None,
            "searchNameByLib": None,
        }

    p1 = pin("P_1", "E_1", [5, 10], [2, 10])
    p2 = pin("P_2", "E_1", [35, 10], [38, 10])
    p5 = pin("P_5", "E_1", [20, 10], [20, 7])
    p3 = pin("P_3", "E_2", [5, 25], [2, 25])
    p4 = pin("P_4", "E_2", [35, 25], [38, 25])

    def edge(edge_id: str, pins: dict, path: list) -> dict:
        return {
            "edgeId": edge_id,
            "state": "active",
            "netlabel": None,
            "connect": {
                pin_id: {"pinId": pin_id, "cpntId": value["owner"], "location": value["tip"]}
                for pin_id, value in pins.items()
            },
            "keyPointPaths": {"wire": [path]},
            "keyPoints": {},
        }

    owners_a = {"P_1": {"owner": "C_1", "tip": [5, 10]}, "P_2": {"owner": "C_2", "tip": [35, 10]}, "P_5": {"owner": "C_5", "tip": [20, 10]}}
    owners_b = {"P_3": {"owner": "C_3", "tip": [5, 25]}, "P_4": {"owner": "C_4", "tip": [35, 25]}}
    return {
        "canvas": {"width": 40, "height": 35},
        "nowCls": "unit",
        "components": {
            "C_1": component("C_1", {"x1": 0, "y1": 5, "x2": 10, "y2": 15}, {"P_1": p1}),
            "C_2": component("C_2", {"x1": 30, "y1": 5, "x2": 40, "y2": 15}, {"P_2": p2}),
            "C_5": component("C_5", {"x1": 15, "y1": 5, "x2": 25, "y2": 15}, {"P_5": p5}),
            "C_3": component("C_3", {"x1": 0, "y1": 20, "x2": 10, "y2": 30}, {"P_3": p3}),
            "C_4": component("C_4", {"x1": 30, "y1": 20, "x2": 40, "y2": 30}, {"P_4": p4}),
        },
        "pins": {"P_1": p1, "P_2": p2, "P_5": p5, "P_3": p3, "P_4": p4},
        "texts": {},
        "edges": {
            "E_1": edge("E_1", owners_a, [[5, 10], [35, 10]]),
            "E_2": edge("E_2", owners_b, [[5, 25], [35, 25]]),
        },
    }


def two_net_prediction(*, mode: str = "good", offset_second_wire: bool = False) -> CircuitIR:
    specs = [
        ("pred-c1", "R1", (0, 5, 10, 15), "pred-p1", (5, 10)),
        ("pred-c2", "R2", (30, 5, 40, 15), "pred-p2", (35, 10)),
        ("pred-c5", "R5", (15, 5, 25, 15), "pred-p5", (20, 10)),
        ("pred-c3", "R3", (0, 20, 10, 30), "pred-p3", (5, 25)),
        ("pred-c4", "R4", (30, 20, 40, 30), "pred-p4", (35, 25)),
    ]
    components = [
        Component(
            internal_id=cid,
            key=key,
            category="resistor",
            bbox=BBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
        )
        for cid, key, bbox, _pid, _point in specs
    ]
    pins = [
        Pin(internal_id=pid, component_id=cid, number="1", name="", point=Point(x=point[0], y=point[1]))
        for cid, _key, _bbox, pid, point in specs
    ]
    second_y = 30 if offset_second_wire else 26
    segments = [
        WireSegment(internal_id="seg-1", start=Point(x=5, y=11), end=Point(x=35, y=11)),
        WireSegment(internal_id="seg-2", start=Point(x=5, y=second_y), end=Point(x=35, y=second_y)),
    ]
    net_a = ["pred-p1", "pred-p2", "pred-p5"]
    net_b = ["pred-p3", "pred-p4"]
    if mode == "missing_p5":
        net_a.remove("pred-p5")
        net_b.append("pred-p5")
    elif mode == "foreign_p4":
        net_a.append("pred-p4")
        net_b.remove("pred-p4")
    return CircuitIR(
        state=IRState.SEMANTIC,
        image=ImageInfo(
            image_id="0002", path="nets.png", width=40, height=35, sha256="def456"
        ),
        components=components,
        pins=pins,
        segments=segments,
        electrical_nets=[
            ElectricalNet(internal_id="pred-net-a", pins=net_a, segments=["seg-1"]),
            ElectricalNet(internal_id="pred-net-b", pins=net_b, segments=["seg-2"]),
        ],
        metadata={"provider": {"name": "HeuristicVisionProvider", "submission_safe": True}},
    )


class NetFunnelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gt = build_raw_ground_truth(net_annotation(), case_id="0002")

    def _object(self, report: dict, pin_ids: list[str]) -> dict:
        return next(
            item
            for item in report["funnels"]["net"]["objects"]
            if item["pin_ids"] == pin_ids
        )

    def test_good_two_net_prediction_passes_membership_and_segments(self) -> None:
        report = build_error_funnel(two_net_prediction(), self.gt, wire_tolerance_px=3)
        row = self._object(report, ["P_1", "P_2", "P_5"])
        self.assertIsNone(row["first_failed_stage"])
        self.assertEqual("passed", row["stages"]["segments_correct"]["status"])

    def test_missing_pin_fails_membership_with_pin_ids(self) -> None:
        report = build_error_funnel(
            two_net_prediction(mode="missing_p5"), self.gt, wire_tolerance_px=3
        )
        row = self._object(report, ["P_1", "P_2", "P_5"])
        self.assertEqual("membership_correct", row["first_failed_stage"])
        reason = row["stages"]["membership_correct"]["reason"]
        self.assertIn("missing_from_pred_net:P_5", reason)
        self.assertEqual("passed", row["stages"]["matched"]["status"])
        self.assertEqual("not_measured", row["stages"]["segments_correct"]["status"])

    def test_foreign_mapped_pin_fails_membership_with_pin_ids(self) -> None:
        report = build_error_funnel(
            two_net_prediction(mode="foreign_p4"), self.gt, wire_tolerance_px=3
        )
        row = self._object(report, ["P_1", "P_2", "P_5"])
        self.assertEqual("membership_correct", row["first_failed_stage"])
        reason = row["stages"]["membership_correct"]["reason"]
        self.assertIn("foreign_mapped_pins:P_4", reason)
        matched_row = _stage(report["funnels"]["net"], "matched")
        self.assertEqual(1, matched_row["count"])

    def test_segment_geometry_completeness_is_checked(self) -> None:
        report = build_error_funnel(
            two_net_prediction(offset_second_wire=True), self.gt, wire_tolerance_px=1
        )
        row = self._object(report, ["P_3", "P_4"])
        self.assertEqual("segments_correct", row["first_failed_stage"])
        reason = row["stages"]["segments_correct"]["reason"]
        self.assertIn("incomplete_centerline_coverage", reason)


class UnmeasuredSemanticsTests(unittest.TestCase):
    def test_output_stage_is_null_not_zero_without_official_output(self) -> None:
        gt = build_raw_ground_truth(raw_annotation(), case_id="0001")
        report = build_error_funnel(prediction_ir(), gt)
        row = _stage(report["funnels"]["text_field"], "output_correct")
        self.assertEqual("not_measured", row["status"])
        self.assertIsNone(row["count"])
        self.assertTrue(row["not_measured"])
        self.assertFalse(row["excluded"])

    def test_scaffold_never_fabricates_zeros(self) -> None:
        scaffold = not_measured_funnel("prediction_manifest_not_supplied")
        self.assertEqual("not_measured", scaffold["status"])
        for funnel in scaffold["funnels"].values():
            for row in funnel["stages"]:
                self.assertIsNone(row["count"])
                self.assertIsNone(row["denominator"])
                self.assertEqual("not_measured", row["status"])

    def test_baseline_e2e_nulls_without_summary(self) -> None:
        base = a04.baseline_e2e({})
        self.assertEqual("not_measured", base["status"])
        self.assertIsNone(base["image_count"])
        self.assertIsNone(base["strict_success"])
        self.assertIsNone(base["metrics"])

    def test_oracle_crossing_without_trusted_gt_is_uncomputable(self) -> None:
        oracle = a04.oracle_decomposition([
            {
                "name": "crossing_wire_oracle",
                "crossing_dependent": True,
                "trusted_crossing_gt": False,
                "metric": "wire_f1",
                "value": 0.9,
            }
        ])
        entry = oracle["experiments"][0]
        self.assertFalse(entry["computable"])
        self.assertIsNone(entry["value"])
        self.assertIn("uncomputable", entry["note"])


class ManifestBindingTests(unittest.TestCase):
    def test_manifest_requires_hash_bindings_and_image_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = {
                "case_id": "0001",
                "ir_path": "x.json",
                "image_sha256": "a" * 64,
                "source_sha256": "b" * 64,
                "config_sha256": "c" * 64,
                "model_lock_sha256": "d" * 64,
                "sidecar_access": False,
            }
            path = root / "manifest.jsonl"
            path.write_text(json.dumps(good) + "\n", encoding="utf-8")
            records = a04.read_prediction_manifest(path)
            self.assertEqual(1, len(records))
            self.assertEqual("bound", a04._case_binding(records[0])["status"])

            bad = dict(good)
            bad.pop("config_sha256")
            path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "config_sha256"):
                a04.read_prediction_manifest(path)

            leaky = dict(good)
            leaky["sidecar_access"] = True
            path.write_text(json.dumps(leaky) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sidecar_access"):
                a04.read_prediction_manifest(path)


if __name__ == "__main__":
    unittest.main()
