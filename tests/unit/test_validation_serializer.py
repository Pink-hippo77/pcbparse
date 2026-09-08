from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pcbparse.ir.models import (
    BBox,
    CircuitIR,
    Component,
    Diagnostic,
    ElectricalNet,
    ImageInfo,
    IRState,
    PhysicalNet,
    Pin,
    Point,
    WireSegment,
)
from pcbparse.output import (
    OfficialSchemaError,
    atomic_write_official_json,
    load_official_schema,
    natural_sort_key,
    official_json_bytes,
    official_json_hash,
    round_half_away_from_zero,
    serialize_official,
    validate_official_output,
)
from pcbparse.validation import run_validators, validate_and_repair_once
from pcbparse.validation.api import validate_ir


def make_ir(*, state: IRState = IRState.SEMANTIC) -> CircuitIR:
    return CircuitIR(
        state=state,
        image=ImageInfo(
            image_id="board-1",
            path="board.png",
            width=100,
            height=80,
            sha256="a" * 64,
        ),
        components=[
            Component(
                internal_id="cmp_r10",
                key="R10",
                category="resistor",
                value="10k",
                bbox=BBox(x1=20.5, y1=8.5, x2=30.49, y2=14.5),
            ),
            Component(
                internal_id="cmp_r2",
                key="R2",
                category="resistor",
                value="2k",
                bbox=BBox(x1=48.5, y1=8.5, x2=58.5, y2=14.5),
            ),
        ],
        pins=[
            Pin(
                internal_id="pin_r2_10",
                component_id="cmp_r2",
                number="10",
                name="B",
                point=Point(x=50.4, y=12.6),
            ),
            Pin(
                internal_id="pin_r10_1",
                component_id="cmp_r10",
                number="1",
                name="A",
                point=Point(x=20.5, y=10.5),
            ),
            Pin(
                internal_id="pin_r2_2",
                component_id="cmp_r2",
                number="2",
                point=Point(x=49.5, y=13.5),
            ),
        ],
        segments=[
            WireSegment(
                internal_id="seg_20",
                start=Point(x=50.4, y=12.6),
                end=Point(x=20.5, y=10.5),
            ),
            WireSegment(
                internal_id="seg_3",
                start=Point(x=49.5, y=13.5),
                end=Point(x=70.5, y=13.5),
            ),
            WireSegment(
                internal_id="seg_2",
                start=Point(x=80.5, y=13.5),
                end=Point(x=70.5, y=13.5),
            ),
        ],
        electrical_nets=[
            ElectricalNet(
                internal_id="enet_z",
                pins=["pin_r2_10", "pin_r10_1"],
                segments=["seg_20"],
            ),
            ElectricalNet(
                internal_id="enet_a",
                pins=["pin_r2_2"],
                segments=["seg_2", "seg_3"],
            ),
        ],
    )


class ValidationTests(unittest.TestCase):
    def test_upstream_diagnostics_survive_and_errors_block_validation(self) -> None:
        upstream_warning = Diagnostic(
            severity="WARNING",
            code="PERCEPTION.LOW_CONFIDENCE",
            message="provider retained an ambiguous observation",
            object_ids=["cmp_r10"],
            stage="perception.component",
        )
        upstream_error = Diagnostic(
            severity="ERROR",
            code="ASSOCIATION.CONTRADICTION",
            message="association evidence is contradictory",
            object_ids=["pin_r10_1"],
            stage="association.pin_text",
        )
        for repair in (False, True):
            with self.subTest(repair=repair):
                ir = make_ir()
                ir.diagnostics = [upstream_warning, upstream_error]

                checked, issues = validate_ir(ir, repair=repair)

                self.assertEqual(IRState.SEMANTIC, checked.state)
                self.assertEqual(
                    {"PERCEPTION.LOW_CONFIDENCE", "ASSOCIATION.CONTRADICTION"},
                    {issue.code for issue in issues if issue.code in {
                        "PERCEPTION.LOW_CONFIDENCE", "ASSOCIATION.CONTRADICTION"
                    }},
                )
                self.assertEqual(issues, checked.diagnostics)

    def test_revalidation_deduplicates_existing_validator_diagnostic(self) -> None:
        ir = make_ir()
        ir.components[0].key = None
        generated = next(
            issue for issue in run_validators(ir) if issue.code == "COMPONENT.MISSING_KEY"
        )
        ir.diagnostics = [generated.model_copy(deep=True)]

        checked, issues = validate_ir(ir)

        self.assertEqual(1, sum(issue.code == "COMPONENT.MISSING_KEY" for issue in issues))
        self.assertEqual(issues, checked.diagnostics)

    def test_graph_accepts_source_membership_and_nearby_canonical_endpoints(self) -> None:
        ir = make_ir()
        ir.segments = [
            WireSegment(
                internal_id="s1",
                start=Point(x=0, y=0),
                end=Point(x=10, y=0),
                source_edge_id="E_1",
            ),
            WireSegment(
                internal_id="s2",
                start=Point(x=12, y=0),
                end=Point(x=20, y=0),
                source_edge_id="E_1",
            ),
        ]
        ir.physical_nets = [
            PhysicalNet(internal_id="pn1", pins=[], segments=["s1", "s2"])
        ]
        codes = {item.code for item in run_validators(ir)}
        self.assertNotIn("NET.DISCONNECTED_SEGMENTS", codes)

    def test_valid_ir_advances_to_validated_without_mutating_input(self) -> None:
        original = make_ir()
        validated, issues = validate_ir(original)

        self.assertEqual([], [issue for issue in issues if issue.severity == "ERROR"])
        self.assertEqual(IRState.VALIDATED, validated.state)
        self.assertEqual(IRState.SEMANTIC, original.state)
        self.assertIsNot(validated, original)

    def test_canvas_edge_coordinates_follow_continuous_half_open_contract(self) -> None:
        ir = make_ir()
        ir.components[0].bbox = BBox(x1=90, y1=70, x2=100, y2=80)
        ir.pins[0].point = Point(x=100, y=80)
        ir.segments[0].end = Point(x=100, y=80)

        issues = run_validators(ir)

        border_ids = {"cmp_r10", "pin_r2_10", "seg_20"}
        self.assertFalse(
            any(
                issue.code == "GEOM.OUT_OF_BOUNDS"
                and border_ids.intersection(issue.object_ids)
                for issue in issues
            )
        )

        ir.pins[0].point = Point(x=100.01, y=80)
        self.assertTrue(
            any(
                issue.code == "GEOM.OUT_OF_BOUNDS"
                and "pin_r2_10" in issue.object_ids
                for issue in run_validators(ir)
            )
        )

    def test_schema_reference_geometry_and_graph_diagnostics_are_stable(self) -> None:
        ir = make_ir()
        ir.components[1].key = "R10"
        ir.pins[0].component_id = "missing-component"
        ir.pins[1].point.x = 100.25
        ir.segments.append(
            WireSegment(
                internal_id="seg_zero",
                start=Point(x=1, y=1),
                end=Point(x=1, y=1),
            )
        )
        ir.electrical_nets[0].pins.append("pin_r10_1")

        first = run_validators(ir)
        second = run_validators(ir)
        self.assertEqual(
            [issue.model_dump() for issue in first],
            [issue.model_dump() for issue in second],
        )
        codes = {issue.code for issue in first}
        self.assertTrue(
            {
                "PIN.UNKNOWN_COMPONENT",
                "GEOM.OUT_OF_BOUNDS",
                "EDGE.ZERO_LENGTH",
                "NET.DUPLICATE_PIN",
            }.issubset(codes)
        )
        self.assertNotIn("COMPONENT.DUPLICATE_KEY", codes)
        self.assertTrue(all("." in issue.code for issue in first))

    def test_one_pass_repairs_only_low_risk_cases_and_rewrites_references(self) -> None:
        ir = make_ir()
        ir.pins[1].point.x = -0.25
        ir.segments.extend(
            [
                WireSegment(
                    internal_id="seg_duplicate",
                    start=Point(x=20.5, y=10.5),
                    end=Point(x=50.4, y=12.6),
                    score=0.5,
                ),
                WireSegment(
                    internal_id="seg_zero",
                    start=Point(x=5, y=5),
                    end=Point(x=5, y=5),
                ),
            ]
        )
        ir.pins[0].edge_ids = ["seg_duplicate", "seg_zero"]
        ir.electrical_nets[0].segments.extend(["seg_duplicate", "seg_zero"])
        ir.electrical_nets[0].pins.append("pin_r10_1")

        repaired, issues = validate_and_repair_once(ir, config_hash="cfg-1")

        self.assertFalse(any(issue.severity == "ERROR" for issue in issues))
        self.assertEqual(IRState.VALIDATED, repaired.state)
        self.assertEqual(0.0, repaired.pin_index()["pin_r10_1"].point.x)
        self.assertEqual({"seg_20", "seg_2", "seg_3"}, set(repaired.segment_index()))
        self.assertEqual(["seg_20"], repaired.pin_index()["pin_r2_10"].edge_ids)
        self.assertEqual(
            ["seg_20"],
            next(net for net in repaired.electrical_nets if net.internal_id == "enet_z").segments,
        )
        self.assertEqual(
            {"pin_r2_10", "pin_r10_1"},
            set(next(net for net in repaired.electrical_nets if net.internal_id == "enet_z").pins),
        )
        self.assertEqual(-0.25, ir.pin_index()["pin_r10_1"].point.x)
        rules = {repair.rule_id for repair in repaired.repairs}
        self.assertEqual(
            {
                "REPAIR.CLAMP_COORDINATE",
                "REPAIR.REMOVE_DUPLICATE_SEGMENT",
                "REPAIR.REMOVE_ZERO_LENGTH_SEGMENT",
                "REPAIR.DEDUPLICATE_NET_PINS",
            },
            rules,
        )
        self.assertTrue(all(repair.config_hash == "cfg-1" for repair in repaired.repairs))

    def test_repair_does_not_guess_component_key(self) -> None:
        ir = make_ir()
        ir.components[0].key = None
        repaired, issues = validate_ir(ir, repair=True)

        self.assertIsNone(repaired.components[0].key)
        self.assertIn("COMPONENT.MISSING_KEY", {issue.code for issue in issues})
        self.assertEqual(IRState.SEMANTIC, repaired.state)

    def test_multi_pin_numbers_are_expanded_for_duplicate_validation(self) -> None:
        ir = make_ir()
        ir.pins[0].number = "5,6,7"
        ir.pins[2].number = "6"

        issues = run_validators(ir)

        duplicate = [issue for issue in issues if issue.code == "PIN.DUPLICATE_NUMBER"]
        self.assertEqual(1, len(duplicate))
        self.assertEqual({"pin_r2_10", "pin_r2_2"}, set(duplicate[0].object_ids))

    def test_failed_revalidation_demotes_stale_validated_state(self) -> None:
        ir = make_ir(state=IRState.VALIDATED)
        ir.pins[0].component_id = "missing"
        checked, issues = validate_ir(ir)

        self.assertTrue(any(issue.severity == "ERROR" for issue in issues))
        self.assertEqual(IRState.SEMANTIC, checked.state)
        self.assertEqual(IRState.VALIDATED, ir.state)


class SerializerTests(unittest.TestCase):
    def test_subpixel_geometry_is_preserved_in_bottom_left_coordinates(self) -> None:
        ir = make_ir(state=IRState.VALIDATED)
        ir.components[0].bbox = BBox(x1=1.1, y1=1.1, x2=1.4, y2=1.4)
        ir.segments[0].start = Point(x=1.1, y=1.1)
        ir.segments[0].end = Point(x=1.4, y=1.4)
        output = serialize_official(ir)

        self.assertEqual([1.1, 78.6, 1.4, 78.9], output["components"]["R10"]["bbox"])
        geometries = [
            edge
            for net in output["nets"].values()
            for edge in net["edges"].values()
        ]
        self.assertIn(
            [{"x": 1.1, "y": 78.9}, {"x": 1.4, "y": 78.6}],
            geometries,
        )

    def test_custom_profile_flows_through_hash_and_atomic_write(self) -> None:
        ir = make_ir(state=IRState.VALIDATED)
        profile = load_official_schema("configs/official_schema.yaml")
        profile["pin"]["key_template"] = "P{number}"
        profile["net"]["key_template"] = "N{index}"
        profile["net"]["edge_key_template"] = "E{index}"
        output = serialize_official(ir, profile)
        self.assertIn("N1", output["nets"])
        self.assertEqual(64, len(official_json_hash(output, profile)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.json"
            atomic_write_official_json(output, path, profile)
            self.assertTrue(path.is_file())

    def validated_ir(self) -> CircuitIR:
        result, issues = validate_ir(make_ir())
        self.assertFalse(any(issue.severity == "ERROR" for issue in issues))
        return result

    def test_requires_validated_state(self) -> None:
        with self.assertRaisesRegex(OfficialSchemaError, "requires VALIDATED"):
            serialize_official(make_ir())

    def test_natural_sort_continuous_coordinates_and_exact_shape(self) -> None:
        output = serialize_official(self.validated_ir())

        self.assertEqual(["components", "pins", "nets"], list(output))
        self.assertEqual(["R2", "R10"], list(output["components"]))
        self.assertEqual(["pin_2", "pin_10"], list(output["pins"]["R2"]))
        self.assertEqual(
            ["type", "Name", "value", "bbox"],
            list(output["components"]["R10"]),
        )
        self.assertEqual(
            [20.5, 65.5, 30.49, 71.5],
            output["components"]["R10"]["bbox"],
        )
        self.assertEqual(
            {"x": 20.5, "y": 69.5},
            output["pins"]["R10"]["pin_1"]["point"],
        )
        self.assertEqual("(R2.2)", output["nets"]["net_1"]["hyperGraph"])
        self.assertEqual("(R2.10,R10.1)", output["nets"]["net_2"]["hyperGraph"])
        self.assertEqual(["edge_1", "edge_2"], list(output["nets"]["net_1"]["edges"]))
        self.assertEqual(-2, round_half_away_from_zero(-1.5))
        self.assertEqual(2, round_half_away_from_zero(1.5))
        self.assertEqual(["R2", "R10"], sorted(["R10", "R2"], key=natural_sort_key))
        self.assertNotEqual(natural_sort_key("R1"), natural_sort_key("R01"))
        self.assertEqual(
            sorted(["R1", "R01"], key=natural_sort_key),
            sorted(["R01", "R1"], key=natural_sort_key),
        )
        validate_official_output(output)

    def test_workspace_official_profile_is_supported(self) -> None:
        profile = Path(__file__).resolve().parents[2] / "configs" / "official_schema.yaml"
        output = serialize_official(self.validated_ir(), profile)
        self.assertEqual({"components", "pins", "nets"}, set(output))

    def test_name_spelling_is_an_explicit_strict_profile_choice(self) -> None:
        ir = self.validated_ir()
        default_output = serialize_official(ir)
        self.assertIn("Name", default_output["components"]["R2"])
        self.assertNotIn("name", default_output["components"]["R2"])

        lowercase_profile = load_official_schema()
        lowercase_profile["version"] = "official-guide-2026-table-spelling-v1"
        lowercase_profile["component"]["fields"][1] = "name"
        lowercase_output = serialize_official(ir, lowercase_profile)
        self.assertIn("name", lowercase_output["components"]["R2"])
        self.assertNotIn("Name", lowercase_output["components"]["R2"])
        validate_official_output(lowercase_output, lowercase_profile)
        with self.assertRaises(OfficialSchemaError):
            validate_official_output(lowercase_output)

    def test_malformed_component_profile_raises_schema_error(self) -> None:
        profile = load_official_schema()
        profile["component"] = []
        with self.assertRaisesRegex(OfficialSchemaError, "component must be an object"):
            load_official_schema(profile)

    def test_official_guide_duplicate_multi_auto_and_flywire_fixture(self) -> None:
        ir = CircuitIR(
            state=IRState.VALIDATED,
            image=ImageInfo(
                image_id="official-contract",
                path="official.png",
                width=120,
                height=100,
                sha256="b" * 64,
            ),
            components=[
                Component(
                    internal_id="cmp_10",
                    key="U1",
                    category="Box",
                    name="MCU",
                    bbox=BBox(x1=25.25, y1=30.5, x2=40.75, y2=50.5),
                ),
                Component(
                    internal_id="cmp_2",
                    key="U1",
                    category="Box",
                    bbox=BBox(x1=5.25, y1=10.5, x2=20.75, y2=25.5),
                ),
            ],
            pins=[
                Pin(
                    internal_id="pin_2",
                    component_id="cmp_2",
                    number="∅",
                    name="AUTO",
                    point=Point(x=10.25, y=20.5),
                ),
                Pin(
                    internal_id="pin_10",
                    component_id="cmp_10",
                    number="5,6,7",
                    name="BUS",
                    point=Point(x=30.25, y=40.5),
                ),
            ],
            physical_nets=[
                PhysicalNet(internal_id="pn_1", pins=["pin_2"]),
                PhysicalNet(internal_id="pn_2", pins=["pin_10"]),
            ],
            electrical_nets=[
                ElectricalNet(
                    internal_id="enet_run",
                    pins=["pin_2", "pin_10"],
                    label="RUN",
                    net_type="SIGNAL",
                    physical_net_ids=["pn_1", "pn_2"],
                )
            ],
        )

        output = serialize_official(ir)

        self.assertEqual(["U1_1", "U1_2"], list(output["components"]))
        self.assertEqual("ic", output["components"]["U1_1"]["type"])
        self.assertIsNone(output["components"]["U1_1"]["Name"])
        self.assertEqual(
            [5.25, 74.5, 20.75, 89.5],
            output["components"]["U1_1"]["bbox"],
        )
        self.assertEqual(["pin_1"], list(output["pins"]["U1_1"]))
        self.assertEqual(
            ["pin_5", "pin_6", "pin_7"], list(output["pins"]["U1_2"])
        )
        net = output["nets"]["net_1"]
        self.assertEqual("(U1_1.1,U1_2.5,U1_2.6,U1_2.7)", net["hyperGraph"])
        self.assertEqual(
            {
                "edge_1": [
                    {"x": 10.25, "y": 79.5},
                    {"x": 10.25, "y": 79.5},
                ],
                "edge_2": [
                    {"x": 30.25, "y": 59.5},
                    {"x": 30.25, "y": 59.5},
                ],
            },
            net["edges"],
        )
        validate_official_output(output)

    def test_official_validator_accepts_arbitrary_keys_and_zero_length_edges(self) -> None:
        output = {
            "components": {
                "U1": {
                    "type": "ic",
                    "Name": "MCU",
                    "value": None,
                    "bbox": [1.25, 2.5, 8.75, 9.5],
                }
            },
            "pins": {
                "U1": {
                    "pin_A5": {
                        "pinname": "RUN",
                        "point": {"x": 1.25, "y": 5.5},
                    }
                }
            },
            "nets": {
                "$1N695": {
                    "hyperGraph": "(U1.A5)",
                    "edges": {
                        "label_anchor": [
                            {"x": 1.25, "y": 5.5},
                            {"x": 1.25, "y": 5.5},
                        ]
                    },
                }
            },
        }

        validate_official_output(output)
        invalid = json.loads(json.dumps(output))
        invalid["nets"]["$1N695"]["edges"]["label_anchor"].append(
            {"x": 2.0, "y": 5.5}
        )
        with self.assertRaisesRegex(OfficialSchemaError, "exactly two points"):
            validate_official_output(invalid)

    def test_dataset300_0001_coordinates_flip_only_at_official_boundary(self) -> None:
        """Freeze real 0001 raster coordinates against the guide coordinate frame."""

        ir = CircuitIR(
            state=IRState.VALIDATED,
            image=ImageInfo(
                image_id="0001",
                path="0001-KiCad.png",
                width=1442,
                height=1090,
                sha256="c" * 64,
            ),
            components=[
                Component(
                    internal_id="C_1",
                    key="U1",
                    category="Box",
                    bbox=BBox(x1=360, y1=175, x2=527, y2=842),
                ),
                Component(
                    internal_id="C_2",
                    key="U2",
                    category="Box",
                    bbox=BBox(x1=1028, y1=119, x2=1111, y2=258),
                ),
            ],
            pins=[
                Pin(
                    internal_id="P_1",
                    component_id="C_1",
                    number="1",
                    point=Point(x=540, y=765),
                )
            ],
            segments=[
                WireSegment(
                    internal_id="E_1",
                    start=Point(x=540, y=765),
                    end=Point(x=611, y=765),
                )
            ],
            electrical_nets=[
                ElectricalNet(
                    internal_id="N_1", pins=["P_1"], segments=["E_1"]
                )
            ],
        )

        output = serialize_official(ir)

        self.assertEqual(
            [1028.0, 832.0, 1111.0, 971.0],
            output["components"]["U2"]["bbox"],
        )
        self.assertEqual(
            {"x": 540.0, "y": 325.0},
            output["pins"]["U1"]["pin_1"]["point"],
        )
        self.assertEqual(
            [{"x": 540.0, "y": 325.0}, {"x": 611.0, "y": 325.0}],
            output["nets"]["net_1"]["edges"]["edge_1"],
        )

    def test_input_list_order_does_not_change_bytes_or_hash(self) -> None:
        first = self.validated_ir()
        second = first.model_copy(deep=True)
        second.components.reverse()
        second.pins.reverse()
        second.segments.reverse()
        second.electrical_nets.reverse()
        for net in second.electrical_nets:
            net.pins.reverse()
            net.segments.reverse()

        output_one = serialize_official(first)
        output_two = serialize_official(second)
        self.assertEqual(official_json_bytes(output_one), official_json_bytes(output_two))
        hashes = [official_json_hash(output_one) for _ in range(3)]
        self.assertEqual(1, len(set(hashes)))
        self.assertEqual(hashes[0], official_json_hash(output_two))

    def test_canonical_bytes_ignore_caller_mapping_insertion_order(self) -> None:
        output = serialize_official(self.validated_ir())
        reordered = {
            "nets": {
                key: {
                    "edges": {
                        edge_key: [
                            dict(reversed(list(point.items()))) for point in edge
                        ]
                        for edge_key, edge in reversed(list(value["edges"].items()))
                    },
                    "hyperGraph": value["hyperGraph"],
                }
                for key, value in reversed(list(output["nets"].items()))
            },
            "pins": {
                key: {
                    pin_key: {
                        "point": dict(reversed(list(pin["point"].items()))),
                        "pinname": pin["pinname"],
                    }
                    for pin_key, pin in reversed(list(value.items()))
                }
                for key, value in reversed(list(output["pins"].items()))
            },
            "components": {
                key: dict(reversed(list(value.items())))
                for key, value in reversed(list(output["components"].items()))
            },
        }

        self.assertEqual(official_json_bytes(output), official_json_bytes(reordered))
        self.assertEqual(official_json_hash(output), official_json_hash(reordered))

    def test_json_object_keys_must_be_strings_before_encoding(self) -> None:
        invalid = {
            "components": {
                1: {"type": "R", "Name": None, "value": None, "bbox": [1, 1, 2, 2]}
            },
            "pins": {1: {}},
            "nets": {},
        }
        with self.assertRaisesRegex(OfficialSchemaError, "keys must be non-empty strings"):
            validate_official_output(invalid)

    def test_extra_field_guard_rejects_root_and_nested_extras(self) -> None:
        output = serialize_official(self.validated_ir())
        with_root_extra = dict(output)
        with_root_extra["diagnostics"] = []
        with self.assertRaises(OfficialSchemaError):
            validate_official_output(with_root_extra)

        nested = json.loads(json.dumps(output))
        nested["components"]["R2"]["score"] = 0.9
        with self.assertRaises(OfficialSchemaError):
            validate_official_output(nested)

    def test_atomic_write_replaces_target_with_canonical_bytes(self) -> None:
        output = serialize_official(self.validated_ir())
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "submission.json"
            target.write_text("old", encoding="utf-8")
            result = atomic_write_official_json(output, target)

            self.assertEqual(target, result)
            self.assertEqual(official_json_bytes(output), target.read_bytes())
            self.assertEqual([], list(target.parent.glob("*.tmp")))

    def test_unknown_references_fail_before_output(self) -> None:
        ir = self.validated_ir()
        ir.electrical_nets[0].pins.append("missing")
        with self.assertRaisesRegex(OfficialSchemaError, "unknown pin"):
            serialize_official(ir)


if __name__ == "__main__":
    unittest.main()
