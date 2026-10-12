"""A01 contract audit fixtures: coordinates flip, duplicate designators,
multi pin numbers, zero-length edges. Also proves GT bytes are never rewritten.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pcbparse.evaluation.contract_audit import (
    ContractAuditor,
    classify_edge,
    discover_targets,
    expand_official_pin_numbers,
    flip_y,
    flip_y_twice,
    inspect_target,
)
from pcbparse.ir.models import (
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
from pcbparse.output.official_serializer import (
    natural_sort_key,
    serialize_official,
    serialize_official_with_quality,
)
from pcbparse.output.official_schema import validate_official_output


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class CoordinateFlipFixtures(unittest.TestCase):
    def test_flip_y_once_and_twice_roundtrip(self) -> None:
        height = 80.0
        source_y = 10.5
        flipped = flip_y(source_y, height)
        self.assertEqual(69.5, flipped)
        self.assertEqual(source_y, flip_y_twice(source_y, height))

    def test_dataset_style_flip_only_at_official_boundary(self) -> None:
        """IR stays image top-left; serialize flips y once to official bottom-left."""
        height = 80.0
        ir = CircuitIR(
            state=IRState.VALIDATED,
            image=ImageInfo(image_id="flip", path="flip.png", width=100, height=int(height)),
            components=[
                Component(
                    internal_id="c1",
                    key="R1",
                    category="resistor",
                    value="10k",
                    bbox=BBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0),
                )
            ],
            pins=[
                Pin(
                    internal_id="p1",
                    component_id="c1",
                    number="1",
                    name="A",
                    point=Point(x=10.0, y=25.0),
                )
            ],
            segments=[
                WireSegment(
                    internal_id="s1",
                    start=Point(x=10.0, y=25.0),
                    end=Point(x=30.0, y=25.0),
                )
            ],
            electrical_nets=[ElectricalNet(internal_id="n1", pins=["p1"], segments=["s1"])],
        )
        output = serialize_official(ir)
        pin_point = output["pins"]["R1"]["pin_1"]["point"]
        self.assertEqual(25.0, ir.pins[0].point.y, "IR must stay in source frame")
        self.assertEqual(height - 25.0, pin_point["y"])
        # bbox is encoded in the official bottom-left frame as well
        bbox = output["components"]["R1"]["bbox"]
        self.assertEqual([10.0, height - 40.0, 30.0, height - 20.0], bbox)
        validate_official_output(output)


class DuplicateDesignatorFixtures(unittest.TestCase):
    def test_duplicate_designators_get_stable_suffixes(self) -> None:
        ir = CircuitIR(
            state=IRState.VALIDATED,
            image=ImageInfo(image_id="dup", path="dup.png", width=100, height=80),
            components=[
                Component(
                    internal_id="c10",
                    key="R1",
                    category="resistor",
                    value="1k",
                    bbox=BBox(x1=1.0, y1=1.0, x2=10.0, y2=10.0),
                ),
                Component(
                    internal_id="c2",
                    key="R1",
                    category="resistor",
                    value="2k",
                    bbox=BBox(x1=20.0, y1=1.0, x2=30.0, y2=10.0),
                ),
            ],
            pins=[
                Pin(internal_id="p1", component_id="c10", number="1", name="A", point=Point(x=1.0, y=1.0)),
                Pin(internal_id="p2", component_id="c2", number="1", name="A", point=Point(x=20.0, y=1.0)),
            ],
            electrical_nets=[
                ElectricalNet(internal_id="n1", pins=["p1"], segments=[]),
                ElectricalNet(internal_id="n2", pins=["p2"], segments=[]),
            ],
        )
        output = serialize_official(ir)
        self.assertIn("R1_1", output["components"])
        self.assertIn("R1_2", output["components"])
        self.assertNotIn("R1", output["components"])
        # natural internal-id order: c2 before c10
        self.assertEqual("1k", output["components"]["R1_2"]["value"])
        self.assertEqual("2k", output["components"]["R1_1"]["value"])
        self.assertTrue(("R1",) < ("R1_1",) or natural_sort_key("R1") < natural_sort_key("R1_1"))
        validate_official_output(output)

    def test_inspector_counts_duplicate_designator_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "a_target_named.json"
            _write_json(
                path,
                {
                    "components": {
                        "c_a": {"Name": "R1", "type": "r", "value": "1k", "bbox": [0, 0, 1, 1]},
                        "c_b": {"Name": "R1", "type": "r", "value": "2k", "bbox": [2, 0, 3, 1]},
                    },
                    "pins": {},
                    "nets": {},
                },
            )
            before = _sha256(path)
            report = inspect_target(path)
            self.assertEqual(2, report.placeholder_keys.get("duplicate_designator_names"))
            self.assertEqual(before, _sha256(path), "GT bytes must not change")


class MultiPinNumberFixtures(unittest.TestCase):
    def test_comma_multi_pin_numbers_expand(self) -> None:
        self.assertEqual(("5", "6", "7"), expand_official_pin_numbers("pin_5,6,7"))
        self.assertEqual(("5", "6", "7"), expand_official_pin_numbers("5,6,7"))
        self.assertEqual(("1",), expand_official_pin_numbers("pin_1"))
        self.assertEqual((), expand_official_pin_numbers("pin_"))
        self.assertEqual((), expand_official_pin_numbers("pin_∅"))

    def test_validation_expands_multi_numbers_without_mutating_source(self) -> None:
        from pcbparse.validation.schema import validate_schema

        ir = CircuitIR(
            state=IRState.VALIDATED,
            image=ImageInfo(image_id="multi", path="multi.png", width=10, height=10),
            components=[
                Component(
                    internal_id="c1",
                    key="U1",
                    category="ic",
                    bbox=BBox(x1=0.0, y1=0.0, x2=5.0, y2=5.0),
                )
            ],
            pins=[
                Pin(
                    internal_id="p1",
                    component_id="c1",
                    number="5,6,7",
                    name="IO",
                    point=Point(x=1.0, y=1.0),
                ),
                Pin(
                    internal_id="p2",
                    component_id="c1",
                    number="7",
                    name="IO7",
                    point=Point(x=2.0, y=1.0),
                ),
            ],
        )
        codes = [item.code for item in validate_schema(ir)]
        self.assertIn("PIN.DUPLICATE_NUMBER", codes)
        self.assertEqual("5,6,7", ir.pins[0].number, "source pin number preserved")

    def test_inspector_flags_multi_pin_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "b_target.json"
            _write_json(
                path,
                {
                    "components": {},
                    "pins": {
                        "U1": {
                            "pin_5,6,7": {"pinname": "IO", "point": {"x": 1.0, "y": 2.0}},
                        }
                    },
                    "nets": {},
                },
            )
            before = _sha256(path)
            report = inspect_target(path)
            self.assertEqual(1, report.pin_number_shapes.get("multi_number"))
            self.assertEqual(before, _sha256(path))


class ZeroLengthEdgeFixtures(unittest.TestCase):
    def test_classify_zero_length_edges(self) -> None:
        same = [{"x": 1.0, "y": 2.0}, {"x": 1.0, "y": 2.0}]
        different = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 2.0}]
        self.assertEqual("zero_length", classify_edge(same))
        self.assertEqual("two_point_array", classify_edge(different))

    def test_serializer_keeps_zero_length_edges_and_validator_accepts(self) -> None:
        ir = CircuitIR(
            state=IRState.VALIDATED,
            image=ImageInfo(image_id="z", path="z.png", width=50, height=50),
            components=[
                Component(
                    internal_id="c1",
                    key="R1",
                    category="resistor",
                    bbox=BBox(x1=0.0, y1=0.0, x2=5.0, y2=5.0),
                )
            ],
            pins=[
                Pin(internal_id="p1", component_id="c1", number="1", name="A", point=Point(x=1.0, y=1.0))
            ],
            segments=[
                WireSegment(
                    internal_id="s0",
                    start=Point(x=1.0, y=1.0),
                    end=Point(x=1.0, y=1.0),
                )
            ],
            electrical_nets=[ElectricalNet(internal_id="n1", pins=["p1"], segments=["s0"])],
        )
        output = serialize_official(ir)
        edges = output["nets"]["net_1"]["edges"]
        self.assertEqual(1, len(edges))
        edge = next(iter(edges.values()))
        self.assertEqual(2, len(edge))
        self.assertEqual(edge[0], edge[1])
        validate_official_output(output)

    def test_inspector_counts_zero_length_without_rewriting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "c_target.json"
            _write_json(
                path,
                {
                    "components": {},
                    "pins": {},
                    "nets": {
                        "net_1": {
                            "hyperGraph": "(R1.1)",
                            "edges": {
                                "edge_1": [
                                    {"x": 4.0, "y": 5.0},
                                    {"x": 4.0, "y": 5.0},
                                ]
                            },
                        }
                    },
                },
            )
            before = _sha256(path)
            report = inspect_target(path)
            self.assertEqual(1, report.placeholder_keys.get("zero_length_edges"))
            self.assertEqual(before, _sha256(path))


class ContractAuditorInventoryTests(unittest.TestCase):
    def test_discovers_both_target_patterns_and_never_rewrites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "200_train_cases" / "0001"
            golden = root / "10_GTcase" / "0025-Altium Designer"
            train_path = train / "0001-KiCad_target_named.json"
            golden_path = golden / "0025-Altium Designer_target.json"
            payload = {
                "components": {
                    "∅75": {"Name": "∅75", "type": "gnd", "value": None, "bbox": [1, 2, 3, 4]},
                    "R5": {"Name": "R5", "type": "r", "value": None, "bbox": [1, 2, 3, 4]},
                },
                "pins": {
                    "U1": {"": {"pinname": "", "point": {"x": 1.0, "y": 2.0}}},
                    "J1": {},
                },
                "nets": {
                    "net_1": {
                        "hyperGraph": "(U1.1,R5.1)",
                        "edges": {"edge_1": [{"x": 0, "y": 0}, {"x": 0, "y": 0}]},
                    }
                },
            }
            _write_json(train_path, payload)
            _write_json(golden_path, payload)
            before = {path: _sha256(path) for path in (train_path, golden_path)}

            paths = discover_targets(root)
            self.assertEqual(2, len(paths))
            kinds = {inspect_target(path).kind for path in paths}
            self.assertEqual({"train200_target_named", "golden10_target"}, kinds)

            out = root / "audits"
            auditor = ContractAuditor()
            summary = auditor.run(root, out)
            self.assertEqual(2, summary["file_count"])
            diff = json.loads((out / "contract_diff.json").read_text(encoding="utf-8"))
            profile = json.loads((out / "contract_profile.json").read_text(encoding="utf-8"))
            self.assertFalse(diff["policy"]["rewrite_original_gt"])
            self.assertFalse(diff["policy"]["guess_unobservable_internal_ids"])
            self.assertTrue(profile["acceptance"]["no_guessing_unobservable_ids"])
            items = {row["item"]: row for row in diff["conflicts"]}
            self.assertIn("component.key.placeholder", items)
            self.assertEqual("conflicting", items["component.key.placeholder"]["status"])
            self.assertIn("component.type:r", items)
            self.assertEqual("resistor", items["component.type:r"]["official_alias_target"])
            self.assertIn("component.value.null", items)
            self.assertGreaterEqual(items["component.value.null"]["count"], 2)
            self.assertIn("pins.empty_number", items)
            self.assertIn("net.zero_length_edges", items)
            # emptyset keys must remain placeholders in the profile policy
            self.assertFalse(profile["placeholder_key_policy"]["prediction_may_invent_gt_placeholder"])
            for path, digest in before.items():
                self.assertEqual(digest, _sha256(path), f"GT rewritten: {path}")

    def test_missing_type_mapping_is_not_guessed(self) -> None:
        auditor = ContractAuditor()
        report = type(
            "R",
            (),
            {
                "type_counts": {"m3螺丝": 3, "r": 1},
                "top_keys": ["components", "pins", "nets"],
                "field_presence": {"component.Name": 4},
                "missing_markers": {},
                "placeholder_keys": {},
                "pin_number_shapes": {},
                "edge_shapes": {},
                "examples": {},
                "kind": "train200_target_named",
                "path": "x.json",
            },
        )()
        diff = auditor.compare([report])
        rows = {row["item"]: row for row in diff["conflicts"]}
        self.assertEqual("conflicting", rows["component.type:m3螺丝"]["status"])
        self.assertEqual("m3螺丝", rows["component.type:m3螺丝"]["official_alias_target"])
        self.assertEqual("mapped", rows["component.type:r"]["status"])
        self.assertEqual("resistor", rows["component.type:r"]["official_alias_target"])


class DegradedVsStrictCounts(unittest.TestCase):
    def test_missing_key_rejected_in_both_modes(self) -> None:
        ir = CircuitIR(
            state=IRState.VALIDATED,
            image=ImageInfo(image_id="q", path="q.png", width=10, height=10),
            components=[
                Component(
                    internal_id="c1",
                    key=None,
                    category="resistor",
                    bbox=BBox(x1=0.0, y1=0.0, x2=2.0, y2=2.0),
                )
            ],
            pins=[],
            electrical_nets=[],
        )
        with self.assertRaises(Exception):
            serialize_official_with_quality(ir, mode="degraded")
        with self.assertRaises(Exception):
            serialize_official_with_quality(ir, mode="strict")


if __name__ == "__main__":
    unittest.main()
