from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from pcbparse.evaluation.metrics import PRF1
from pcbparse.output.official_schema import validate_official_output


@dataclass(frozen=True, slots=True)
class OfficialMetricReport:
    component: PRF1
    pin: PRF1
    net_hypergraph: PRF1
    net_line: PRF1
    assumptions: tuple[str, ...]

    @property
    def weighted_score(self) -> float:
        return (
            0.30 * self.component.f1
            + 0.25 * self.pin.f1
            + 0.35 * self.net_hypergraph.f1
            + 0.10 * self.net_line.f1
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluator_kind": "guide_v1_assumption_regression",
            "official": False,
            "component": self.component.as_dict(),
            "pin": self.pin.as_dict(),
            "net_hypergraph": self.net_hypergraph.as_dict(),
            "net_line": self.net_line.as_dict(),
            "weighted_score": self.weighted_score,
            "assumptions": list(self.assumptions),
        }


def evaluate_official_json(
    predicted: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    bbox_tolerance_px: float = 20.0,
    point_tolerance_px: float = 5.0,
    line_tolerance_px: float = 5.0,
) -> OfficialMetricReport:
    """Regression-check guide-v1 outputs with explicitly documented assumptions.

    This local evaluator is intentionally replaceable. The competition's latest
    evaluator remains authoritative when its matching order becomes available.
    Never use ``weighted_score`` as a claimed competition score.
    """

    bbox_tolerance_px = _nonnegative_finite_tolerance(
        bbox_tolerance_px, "bbox_tolerance_px"
    )
    point_tolerance_px = _nonnegative_finite_tolerance(
        point_tolerance_px, "point_tolerance_px"
    )
    line_tolerance_px = _nonnegative_finite_tolerance(
        line_tolerance_px, "line_tolerance_px"
    )
    validate_official_output(predicted)
    validate_official_output(expected)
    component = _keyed_metric(
        predicted["components"],
        expected["components"],
        lambda left, right: _component_matches(left, right, bbox_tolerance_px),
    )
    pin = _keyed_metric(
        _flatten_pins(predicted["pins"]),
        _flatten_pins(expected["pins"]),
        lambda left, right: _pin_matches(left, right, point_tolerance_px),
    )
    hypergraph, line = _net_metrics(
        predicted["nets"],
        expected["nets"],
        line_tolerance_px=line_tolerance_px,
    )
    return OfficialMetricReport(
        component=component,
        pin=pin,
        net_hypergraph=hypergraph,
        net_line=line,
        assumptions=(
            "GUIDE RULE: components match by exact key/type and type-selective strict text.",
            "GUIDE RULE: IC compares Name; resistor/capacitor compare value; other types ignore component text.",
            "ASSUMPTION: bbox tolerance is inclusive and applied independently to all four coordinates.",
            "ASSUMPTION: pins match by exact component/pin key and pinname plus Euclidean point tolerance.",
            "GUIDE RULE: HyperGraph nets are assignment-matched before pin membership is scored.",
            "ASSUMPTION: local assignment maximizes pairwise pin-set F1 and discards zero-overlap pairs.",
            "ASSUMPTION: equal-weight assignment ties use canonical pin/edge content, never arbitrary net keys.",
            "ASSUMPTION: NetLine uses assignment-matched net pairs, ignores endpoint order, and conservatively penalizes unmatched-net edges.",
            "Replace this evaluator when the official evaluation script is released.",
        ),
    )


def aggregate_reports(reports: Iterable[OfficialMetricReport]) -> OfficialMetricReport:
    items = list(reports)
    if not items:
        zero = PRF1(0, 0, 0)
        return OfficialMetricReport(zero, zero, zero, zero, ("No cases evaluated.",))
    return OfficialMetricReport(
        component=_sum_prf(item.component for item in items),
        pin=_sum_prf(item.pin for item in items),
        net_hypergraph=_sum_prf(item.net_hypergraph for item in items),
        net_line=_sum_prf(item.net_line for item in items),
        assumptions=items[0].assumptions,
    )


def _sum_prf(values: Iterable[PRF1]) -> PRF1:
    items = list(values)
    return PRF1(
        sum(item.true_positive for item in items),
        sum(item.false_positive for item in items),
        sum(item.false_negative for item in items),
    )


def _nonnegative_finite_tolerance(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must be a finite non-negative number")
    return float(value)


def _keyed_metric(
    predicted: Mapping[str, Any],
    expected: Mapping[str, Any],
    matches: Any,
) -> PRF1:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for key in sorted(set(predicted) | set(expected)):
        if key not in expected:
            false_positive += 1
        elif key not in predicted:
            false_negative += 1
        elif matches(predicted[key], expected[key]):
            true_positive += 1
        else:
            false_positive += 1
            false_negative += 1
    return PRF1(true_positive, false_positive, false_negative)


def _component_matches(predicted: Mapping[str, Any], expected: Mapping[str, Any], tolerance: float) -> bool:
    component_type = expected["type"]
    if predicted["type"] != component_type:
        return False
    if component_type == "ic" and predicted["Name"] != expected["Name"]:
        return False
    if component_type in {"resistor", "capacitor"} and predicted["value"] != expected["value"]:
        return False
    return all(abs(float(left) - float(right)) <= tolerance for left, right in zip(predicted["bbox"], expected["bbox"]))


def _flatten_pins(groups: Mapping[str, Any]) -> dict[str, Any]:
    return {
        f"{component_key}.{pin_key}": pin
        for component_key, pins in groups.items()
        for pin_key, pin in pins.items()
    }


def _pin_matches(predicted: Mapping[str, Any], expected: Mapping[str, Any], tolerance: float) -> bool:
    if predicted["pinname"] != expected["pinname"]:
        return False
    left = predicted["point"]
    right = expected["point"]
    return math.hypot(float(left["x"]) - float(right["x"]), float(left["y"]) - float(right["y"])) <= tolerance


def _pin_set(net: Mapping[str, Any]) -> frozenset[str]:
    raw = str(net["hyperGraph"])
    if not (raw.startswith("(") and raw.endswith(")")):
        return frozenset()
    body = raw[1:-1]
    return frozenset(item for item in body.split(",") if item)


def _net_metrics(
    predicted: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    line_tolerance_px: float,
) -> tuple[PRF1, PRF1]:
    # Page 8 explicitly says net keys need not agree with GT.  Sorting by those
    # arbitrary keys would still leak them into equal-weight Hungarian ties and
    # could change the member TP/FP/FN after a harmless key rename.  Canonical
    # content order makes the local tie policy deterministic and key-invariant.
    predicted_nets = sorted(predicted.values(), key=_net_content_sort_key)
    expected_nets = sorted(expected.values(), key=_net_content_sort_key)
    predicted_pin_sets = [_pin_set(net) for net in predicted_nets]
    expected_pin_sets = [_pin_set(net) for net in expected_nets]
    weights = [
        [_pin_set_f1(left, right) for right in expected_pin_sets]
        for left in predicted_pin_sets
    ]
    assignments = [
        (left_index, right_index)
        for left_index, right_index in _maximum_weight_assignment(weights)
        if weights[left_index][right_index] > 0.0
    ]
    matched_predicted = {left for left, _right in assignments}
    matched_expected = {right for _left, right in assignments}
    hyper_tp = 0
    hyper_fp = 0
    hyper_fn = 0
    line_tp = 0
    line_fp = 0
    line_fn = 0

    for left_index, right_index in assignments:
        left_pins = predicted_pin_sets[left_index]
        right_pins = expected_pin_sets[right_index]
        hyper_tp += len(left_pins & right_pins)
        hyper_fp += len(left_pins - right_pins)
        hyper_fn += len(right_pins - left_pins)

        left_edges = list(predicted_nets[left_index]["edges"].values())
        right_edges = list(expected_nets[right_index]["edges"].values())
        matches = _maximum_edge_matching(
            left_edges, right_edges, line_tolerance_px
        )
        line_tp += matches
        line_fp += len(left_edges) - matches
        line_fn += len(right_edges) - matches

    for index, (net, pins) in enumerate(zip(predicted_nets, predicted_pin_sets)):
        if index not in matched_predicted:
            hyper_fp += len(pins)
            line_fp += len(net["edges"])
    for index, (net, pins) in enumerate(zip(expected_nets, expected_pin_sets)):
        if index not in matched_expected:
            hyper_fn += len(pins)
            line_fn += len(net["edges"])
    return PRF1(hyper_tp, hyper_fp, hyper_fn), PRF1(line_tp, line_fp, line_fn)


def _net_content_sort_key(net: Mapping[str, Any]) -> tuple[Any, ...]:
    pin_signature = tuple(sorted(_pin_set(net)))
    edge_signature = tuple(
        sorted(_edge_content_sort_key(edge) for edge in net["edges"].values())
    )
    return pin_signature, edge_signature


def _edge_content_sort_key(
    edge: Sequence[Mapping[str, Any]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    points = sorted(
        (
            (float(edge[0]["x"]), float(edge[0]["y"])),
            (float(edge[1]["x"]), float(edge[1]["y"])),
        )
    )
    return points[0], points[1]


def _pin_set_f1(left: frozenset[str], right: frozenset[str]) -> float:
    denominator = len(left) + len(right)
    return 2.0 * len(left & right) / denominator if denominator else 0.0


def _maximum_weight_assignment(weights: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Return a deterministic maximum-weight rectangular assignment.

    This is the O(n^3) Hungarian shortest-augmenting-path algorithm.  It keeps
    the guide-v1 regression evaluator dependency-free and stable on large net
    sets.  Zero-weight pairs are returned here and filtered by the caller.
    """

    if not weights:
        return []
    row_count = len(weights)
    column_count = len(weights[0])
    if any(len(row) != column_count for row in weights):
        raise ValueError("assignment weights must form a rectangular matrix")
    if column_count == 0:
        return []
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for row in weights
        for value in row
    ):
        raise ValueError("assignment weights must be finite numbers")
    if row_count > column_count:
        transposed = [
            [float(weights[row][column]) for row in range(row_count)]
            for column in range(column_count)
        ]
        return [
            (column, row)
            for row, column in _maximum_weight_assignment(transposed)
        ]

    # The implementation minimizes costs; negating weights maximizes them.
    costs = [[-float(value) for value in row] for row in weights]
    row_potential = [0.0] * (row_count + 1)
    column_potential = [0.0] * (column_count + 1)
    column_owner = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)

    for row in range(1, row_count + 1):
        column_owner[0] = row
        minimum = [math.inf] * (column_count + 1)
        used = [False] * (column_count + 1)
        column = 0
        while True:
            used[column] = True
            active_row = column_owner[column]
            delta = math.inf
            next_column = 0
            for candidate in range(1, column_count + 1):
                if used[candidate]:
                    continue
                reduced = (
                    costs[active_row - 1][candidate - 1]
                    - row_potential[active_row]
                    - column_potential[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    predecessor[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(column_count + 1):
                if used[candidate]:
                    row_potential[column_owner[candidate]] += delta
                    column_potential[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if column_owner[column] == 0:
                break
        while True:
            previous = predecessor[column]
            column_owner[column] = column_owner[previous]
            column = previous
            if column == 0:
                break

    return sorted(
        (owner - 1, column - 1)
        for column, owner in enumerate(column_owner[1:], start=1)
        if owner != 0
    )


def _maximum_edge_matching(predicted: Sequence[Any], expected: Sequence[Any], tolerance: float) -> int:
    adjacency = [
        [index for index, target in enumerate(expected) if _edge_matches(edge, target, tolerance)]
        for edge in predicted
    ]
    target_owner: dict[int, int] = {}

    def augment(source: int, visited: set[int]) -> bool:
        for target in adjacency[source]:
            if target in visited:
                continue
            visited.add(target)
            if target not in target_owner or augment(target_owner[target], visited):
                target_owner[target] = source
                return True
        return False

    return sum(1 for source in range(len(predicted)) if augment(source, set()))


def _edge_matches(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], tolerance: float) -> bool:
    direct = _point_distance(left[0], right[0]) <= tolerance and _point_distance(left[1], right[1]) <= tolerance
    reverse = _point_distance(left[0], right[1]) <= tolerance and _point_distance(left[1], right[0]) <= tolerance
    return direct or reverse


def _point_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return math.hypot(float(left["x"]) - float(right["x"]), float(left["y"]) - float(right["y"]))
