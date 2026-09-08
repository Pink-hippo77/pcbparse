from __future__ import annotations

import copy
import math
import unittest

from pcbparse.evaluation.official_metrics import (
    _maximum_weight_assignment,
    aggregate_reports,
    evaluate_official_json,
)
from pcbparse.output.official_schema import OfficialSchemaError


def fixture() -> dict:
    return {
        "components": {
            "R1": {
                "type": "resistor",
                "Name": None,
                "value": "10k",
                "bbox": [10, 10, 20, 20],
            }
        },
        "pins": {"R1": {"pin_1": {"pinname": "A", "point": {"x": 10, "y": 15}}}},
        "nets": {
            "net_1": {
                "hyperGraph": "(R1.1)",
                "edges": {"edge_1": [{"x": 10, "y": 15}, {"x": 50, "y": 15}]},
            }
        },
    }


def network_fixture(nets: dict, component_keys: tuple[str, ...]) -> dict:
    return {
        "components": {
            key: {
                "type": "vbus_in",
                "Name": key,
                "value": None,
                "bbox": [index * 20, 0, index * 20 + 10, 10],
            }
            for index, key in enumerate(component_keys)
        },
        "pins": {
            key: {
                "pin_1": {
                    "pinname": "1",
                    "point": {"x": index * 20, "y": 5},
                }
            }
            for index, key in enumerate(component_keys)
        },
        "nets": copy.deepcopy(nets),
    }


class OfficialMetricTests(unittest.TestCase):
    def test_identical_is_one(self) -> None:
        report = evaluate_official_json(fixture(), fixture())
        self.assertEqual(1.0, report.weighted_score)

    def test_strict_text_and_tolerant_geometry(self) -> None:
        predicted = copy.deepcopy(fixture())
        predicted["components"]["R1"]["bbox"][0] += 20
        predicted["components"]["R1"]["bbox"][2] += 20
        predicted["pins"]["R1"]["pin_1"]["point"]["x"] += 5
        predicted["nets"]["net_1"]["edges"]["edge_1"].reverse()
        report = evaluate_official_json(predicted, fixture())
        self.assertEqual(1.0, report.weighted_score)
        predicted["components"]["R1"]["value"] = "10K"
        report = evaluate_official_json(predicted, fixture())
        self.assertEqual(0.0, report.component.f1)

    def test_component_text_matching_follows_type_specific_guide_rule(self) -> None:
        expected = fixture()

        resistor = copy.deepcopy(expected)
        resistor["components"]["R1"]["Name"] = "ignored-for-resistor"
        self.assertEqual(
            1.0, evaluate_official_json(resistor, expected).component.f1
        )
        wrong_type = copy.deepcopy(expected)
        wrong_type["components"]["R1"]["type"] = "capacitor"
        wrong_type_report = evaluate_official_json(wrong_type, expected).component
        self.assertEqual(
            (0, 1, 1),
            (
                wrong_type_report.true_positive,
                wrong_type_report.false_positive,
                wrong_type_report.false_negative,
            ),
        )

        capacitor_expected = copy.deepcopy(expected)
        capacitor_expected["components"]["R1"].update(
            {"type": "capacitor", "Name": "C", "value": "10pF"}
        )
        capacitor_name_diff = copy.deepcopy(capacitor_expected)
        capacitor_name_diff["components"]["R1"]["Name"] = "ignored"
        self.assertEqual(
            1.0,
            evaluate_official_json(
                capacitor_name_diff, capacitor_expected
            ).component.f1,
        )
        capacitor_value_diff = copy.deepcopy(capacitor_expected)
        capacitor_value_diff["components"]["R1"]["value"] = "10PF"
        self.assertEqual(
            0.0,
            evaluate_official_json(
                capacitor_value_diff, capacitor_expected
            ).component.f1,
        )

        ic_expected = copy.deepcopy(expected)
        ic_expected["components"]["R1"].update(
            {"type": "ic", "Name": "RP2350A", "value": None}
        )
        ic_value_diff = copy.deepcopy(ic_expected)
        ic_value_diff["components"]["R1"]["value"] = "ignored-for-ic"
        self.assertEqual(
            1.0,
            evaluate_official_json(ic_value_diff, ic_expected).component.f1,
        )
        ic_name_diff = copy.deepcopy(ic_expected)
        ic_name_diff["components"]["R1"]["Name"] = "RP2350B"
        self.assertEqual(
            0.0,
            evaluate_official_json(ic_name_diff, ic_expected).component.f1,
        )

        other_expected = copy.deepcopy(expected)
        other_expected["components"]["R1"].update(
            {"type": "vbus_in", "Name": "RUN", "value": None}
        )
        other_text_diff = copy.deepcopy(other_expected)
        other_text_diff["components"]["R1"].update(
            {"Name": "ignored", "value": "ignored"}
        )
        self.assertEqual(
            1.0,
            evaluate_official_json(other_text_diff, other_expected).component.f1,
        )

    def test_aggregate_is_micro(self) -> None:
        good = evaluate_official_json(fixture(), fixture())
        bad_prediction = {"components": {}, "pins": {}, "nets": {}}
        bad = evaluate_official_json(bad_prediction, fixture())
        aggregate = aggregate_reports([good, bad])
        self.assertAlmostEqual(2 / 3, aggregate.component.f1)

    def test_hypergraph_assignment_scores_pin_membership_after_net_matching(self) -> None:
        expected = fixture()
        expected["components"] = {
            key: {
                "type": "vbus_in",
                "Name": key,
                "value": None,
                "bbox": [index * 20, 0, index * 20 + 10, 10],
            }
            for index, key in enumerate(("A", "B", "C", "D"))
        }
        expected["pins"] = {
            key: {
                "pin_1": {
                    "pinname": "1",
                    "point": {"x": index * 20, "y": 5},
                }
            }
            for index, key in enumerate(("A", "B", "C", "D"))
        }
        expected["nets"] = {
            "net_1": {
                "hyperGraph": "(A.1,B.1)",
                "edges": {
                    "edge_1": [{"x": 0, "y": 5}, {"x": 20, "y": 5}]
                },
            },
            "net_2": {
                "hyperGraph": "(C.1)",
                "edges": {
                    "edge_1": [{"x": 40, "y": 5}, {"x": 50, "y": 5}]
                },
            },
        }
        predicted = copy.deepcopy(expected)
        predicted["nets"] = {
            "net_1": copy.deepcopy(expected["nets"]["net_2"]),
            "net_2": {
                "hyperGraph": "(A.1,D.1)",
                "edges": copy.deepcopy(expected["nets"]["net_1"]["edges"]),
            },
        }

        report = evaluate_official_json(predicted, expected)

        self.assertEqual(2, report.net_hypergraph.true_positive)
        self.assertEqual(1, report.net_hypergraph.false_positive)
        self.assertEqual(1, report.net_hypergraph.false_negative)
        self.assertAlmostEqual(2 / 3, report.net_hypergraph.f1)
        self.assertEqual(2, report.net_line.true_positive)

    def test_rectangular_maximum_weight_assignment(self) -> None:
        self.assertEqual(
            [(0, 0), (1, 1)],
            _maximum_weight_assignment([[10.0, 1.0, 2.0], [1.0, 9.0, 3.0]]),
        )
        self.assertEqual(
            [(0, 0), (2, 1)],
            _maximum_weight_assignment([[10.0, 1.0], [2.0, 3.0], [1.0, 9.0]]),
        )

    def test_assignment_empty_ragged_nonfinite_and_ties_are_explicit(self) -> None:
        self.assertEqual([], _maximum_weight_assignment([]))
        self.assertEqual([], _maximum_weight_assignment([[]]))
        self.assertEqual([], _maximum_weight_assignment([[], []]))
        with self.assertRaisesRegex(ValueError, "rectangular matrix"):
            _maximum_weight_assignment([[], [1.0]])
        for bad in (math.nan, math.inf, -math.inf, True):
            with self.subTest(weight=bad):
                with self.assertRaisesRegex(ValueError, "finite numbers"):
                    _maximum_weight_assignment([[bad]])
        tied = [[0.0, 0.0], [2.0 / 3.0, 2.0 / 3.0]]
        first = _maximum_weight_assignment(tied)
        self.assertEqual(first, _maximum_weight_assignment(tied))
        self.assertEqual([(0, 0), (1, 1)], first)

    def test_tolerances_must_be_finite_nonnegative_numbers(self) -> None:
        invalid = (-1.0, math.nan, math.inf, -math.inf, True, "5")
        for name in (
            "bbox_tolerance_px",
            "point_tolerance_px",
            "line_tolerance_px",
        ):
            for value in invalid:
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(
                        ValueError, "finite non-negative number"
                    ):
                        evaluate_official_json(
                            fixture(), fixture(), **{name: value}
                        )

    def test_empty_outputs_have_the_explicit_prf_empty_set_convention(self) -> None:
        empty = {"components": {}, "pins": {}, "nets": {}}
        report = evaluate_official_json(empty, empty)
        for metric in (
            report.component,
            report.pin,
            report.net_hypergraph,
            report.net_line,
        ):
            self.assertEqual((0, 0, 0), (
                metric.true_positive,
                metric.false_positive,
                metric.false_negative,
            ))
            self.assertEqual(1.0, metric.f1)
        self.assertEqual(1.0, report.weighted_score)

    def test_zero_overlap_nets_are_unmatched_and_members_are_fp_fn(self) -> None:
        edge = {"edge_1": [{"x": 0, "y": 0}, {"x": 10, "y": 0}]}
        predicted = network_fixture(
            {"pred-any-key": {"hyperGraph": "(A.1)", "edges": edge}},
            ("A", "B"),
        )
        expected = network_fixture(
            {"gt-any-key": {"hyperGraph": "(B.1)", "edges": edge}},
            ("A", "B"),
        )
        report = evaluate_official_json(predicted, expected)
        self.assertEqual(
            (0, 1, 1),
            (
                report.net_hypergraph.true_positive,
                report.net_hypergraph.false_positive,
                report.net_hypergraph.false_negative,
            ),
        )
        # The guide says NetLine is compared within matched net pairs but does
        # not publish the unmatched-pair boundary.  This local evaluator's
        # documented conservative assumption charges both unmatched edges.
        self.assertEqual(
            (0, 1, 1),
            (
                report.net_line.true_positive,
                report.net_line.false_positive,
                report.net_line.false_negative,
            ),
        )
        self.assertTrue(any(
            "unmatched-net edges" in item for item in report.assumptions
        ))

    def test_rectangular_net_assignment_counts_unmatched_members_once(self) -> None:
        predicted = network_fixture(
            {
                "pred_1": {"hyperGraph": "(A.1)", "edges": {}},
                "pred_2": {"hyperGraph": "(B.1)", "edges": {}},
            },
            ("A", "B"),
        )
        expected = network_fixture(
            {"gt_1": {"hyperGraph": "(A.1)", "edges": {}}},
            ("A", "B"),
        )
        extra_prediction = evaluate_official_json(
            predicted, expected
        ).net_hypergraph
        missing_prediction = evaluate_official_json(
            expected, predicted
        ).net_hypergraph
        self.assertEqual(
            (1, 1, 0),
            (
                extra_prediction.true_positive,
                extra_prediction.false_positive,
                extra_prediction.false_negative,
            ),
        )
        self.assertEqual(
            (1, 0, 1),
            (
                missing_prediction.true_positive,
                missing_prediction.false_positive,
                missing_prediction.false_negative,
            ),
        )

    def test_duplicate_hypergraph_members_are_rejected_before_scoring(self) -> None:
        duplicated = network_fixture(
            {
                "net_1": {
                    "hyperGraph": "(A.1,A.1)",
                    "edges": {},
                }
            },
            ("A",),
        )
        with self.assertRaisesRegex(OfficialSchemaError, "duplicate pin refs"):
            evaluate_official_json(duplicated, duplicated)

    def test_netline_is_one_to_one_and_endpoint_order_independent(self) -> None:
        segment = [{"x": 0, "y": 0}, {"x": 10, "y": 0}]
        expected = network_fixture(
            {
                "net_1": {
                    "hyperGraph": "(A.1)",
                    "edges": {"edge_1": copy.deepcopy(segment)},
                }
            },
            ("A",),
        )
        predicted = copy.deepcopy(expected)
        predicted["nets"]["net_1"]["edges"] = {
            "renamed_1": list(reversed(copy.deepcopy(segment))),
            "renamed_2": copy.deepcopy(segment),
        }
        report = evaluate_official_json(predicted, expected)
        self.assertEqual(
            (1, 1, 0),
            (
                report.net_line.true_positive,
                report.net_line.false_positive,
                report.net_line.false_negative,
            ),
        )

    def test_equal_weight_net_ties_do_not_depend_on_arbitrary_net_keys(self) -> None:
        empty_edges: dict = {}
        predicted_nets = {
            "pred_a": {"hyperGraph": "(A.1)", "edges": empty_edges},
            "pred_z": {"hyperGraph": "(B.1,C.1)", "edges": empty_edges},
        }
        expected_semantics = (
            {"hyperGraph": "(B.1)", "edges": empty_edges},
            {"hyperGraph": "(B.1,C.1,D.1,E.1)", "edges": empty_edges},
        )
        predicted = network_fixture(
            predicted_nets, ("A", "B", "C", "D", "E")
        )
        expected_first = network_fixture(
            {"gt_a": expected_semantics[0], "gt_z": expected_semantics[1]},
            ("A", "B", "C", "D", "E"),
        )
        expected_renamed = network_fixture(
            {"gt_z": expected_semantics[0], "gt_a": expected_semantics[1]},
            ("A", "B", "C", "D", "E"),
        )
        first = evaluate_official_json(predicted, expected_first)
        renamed = evaluate_official_json(predicted, expected_renamed)
        self.assertEqual(first.net_hypergraph, renamed.net_hypergraph)
        self.assertEqual(
            (2, 1, 3),
            (
                first.net_hypergraph.true_positive,
                first.net_hypergraph.false_positive,
                first.net_hypergraph.false_negative,
            ),
        )


if __name__ == "__main__":
    unittest.main()
