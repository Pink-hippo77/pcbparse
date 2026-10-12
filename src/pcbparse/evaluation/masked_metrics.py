from __future__ import annotations

import math
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import combinations
from statistics import median
from typing import Any, TypeVar

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from pcbparse.evaluation.metrics import PRF1
from pcbparse.ir.models import BBox, CircuitIR

from .raw_ground_truth import RawGroundTruth, TextTarget


def evaluate_masked_case(
    ir: CircuitIR,
    ground_truth: RawGroundTruth,
    *,
    component_iou_threshold: float = 0.5,
    pin_distance_tolerance_px: float = 5.0,
    wire_tolerance_px: int = 3,
) -> dict[str, Any]:
    if not 0.0 < component_iou_threshold <= 1.0:
        raise ValueError("component_iou_threshold must be in (0, 1]")
    if pin_distance_tolerance_px < 0:
        raise ValueError("pin_distance_tolerance_px must be non-negative")
    if type(wire_tolerance_px) is not int or wire_tolerance_px < 0:
        raise ValueError("wire_tolerance_px must be a non-negative integer")

    component_report, component_map = _component_metrics(
        ir,
        ground_truth,
        threshold=component_iou_threshold,
    )
    pin_report, pin_map = _pin_metrics(
        ir,
        ground_truth,
        component_map=component_map,
        tolerance=pin_distance_tolerance_px,
    )
    wire_report = _wire_metrics(ir, ground_truth, tolerance=wire_tolerance_px)
    graph_report = _graph_metrics(ir, ground_truth, pin_map=pin_map)
    return {
        "policy_version": ground_truth.policy_version,
        "case_id": ground_truth.case_id,
        "source_domain": ground_truth.source_domain,
        "thresholds": {
            "component_iou": component_iou_threshold,
            "pin_distance_px": pin_distance_tolerance_px,
            "wire_centerline_tolerance_px": wire_tolerance_px,
        },
        "coverage": ground_truth.coverage(),
        "component": component_report,
        "pin": pin_report,
        "wire": wire_report,
        "graph": graph_report,
    }


def aggregate_masked_reports(reports: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(reports)
    if not items:
        return {"cases": 0, "policy_version": "raw-v1", "source_domains": {}}

    component_detection = _aggregate_prf(item["component"]["detection"] for item in items)
    pin_detection = _aggregate_prf(item["pin"]["detection"] for item in items)
    graph_pairwise = _aggregate_prf(item["graph"]["pairwise_connectivity"] for item in items)
    category_correct = sum(int(item["component"]["category"]["correct"]) for item in items)
    category_normalized = sum(
        int(item["component"]["category"]["normalized_correct"]) for item in items
    )
    category_evaluated = sum(int(item["component"]["category"]["evaluated"]) for item in items)
    point_errors = [
        float(error)
        for item in items
        for error in item["pin"]["point_error_px"]["values"]
    ]
    wire_predicted = sum(int(item["wire"]["predicted_pixels"]) for item in items)
    wire_expected = sum(int(item["wire"]["expected_pixels"]) for item in items)
    wire_predicted_hits = sum(int(item["wire"]["predicted_hit_pixels"]) for item in items)
    wire_expected_hits = sum(int(item["wire"]["expected_hit_pixels"]) for item in items)
    wire_precision = _ratio(wire_predicted_hits, wire_predicted, empty=1.0 if not wire_expected else 0.0)
    wire_recall = _ratio(wire_expected_hits, wire_expected, empty=1.0 if not wire_predicted else 0.0)

    fields: dict[str, Any] = {}
    for group, field_names in (
        ("component", ("designator", "value", "search_name")),
        ("pin", ("number", "name")),
    ):
        for field_name in field_names:
            key = f"{group}.{field_name}"
            fields[key] = _aggregate_field(
                item[group]["fields"][field_name] for item in items
            )

    coverage = _aggregate_coverage(item["coverage"] for item in items)
    return {
        "cases": len(items),
        "policy_version": str(items[0]["policy_version"]),
        "source_domains": dict(sorted(Counter(str(item["source_domain"]) for item in items).items())),
        "coverage": coverage,
        "component": {
            "detection": component_detection,
            "category": {
                "correct": category_correct,
                "normalized_correct": category_normalized,
                "evaluated": category_evaluated,
                "accuracy": _optional_ratio(category_correct, category_evaluated),
                "normalized_accuracy": _optional_ratio(category_normalized, category_evaluated),
            },
        },
        "pin": {
            "detection": pin_detection,
            "point_error_px": _error_summary(point_errors, include_values=False),
        },
        "fields": fields,
        "wire": {
            "predicted_pixels": wire_predicted,
            "expected_pixels": wire_expected,
            "predicted_hit_pixels": wire_predicted_hits,
            "expected_hit_pixels": wire_expected_hits,
            "precision": wire_precision,
            "recall": wire_recall,
            "f1": _f1(wire_precision, wire_recall),
        },
        "graph": {"pairwise_connectivity": graph_pairwise},
    }


def _component_metrics(
    ir: CircuitIR,
    gt: RawGroundTruth,
    *,
    threshold: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    predicted = sorted(ir.components, key=lambda item: _natural_key(item.internal_id))
    expected = sorted(
        (item for item in gt.components if item.mask.eligible and item.bbox is not None),
        key=lambda item: _natural_key(item.raw_id),
    )
    candidates: list[tuple[int, int, float]] = []
    for pred_index, component in enumerate(predicted):
        predicted_bbox = _bbox_tuple(component.bbox)
        for gt_index, target in enumerate(expected):
            assert target.bbox is not None
            score = bbox_iou(predicted_bbox, target.bbox)
            if score >= threshold:
                candidates.append((pred_index, gt_index, score))
    matches = _maximum_cardinality_matching(
        len(predicted),
        len(expected),
        candidates,
        left_keys=[item.internal_id for item in predicted],
        right_keys=[item.raw_id for item in expected],
    )
    matched_pred = {left for left, _right, _score in matches}
    ignored_pred = {
        pred_index
        for pred_index, component in enumerate(predicted)
        if pred_index not in matched_pred
        and any(
            target.bbox is not None
            and bbox_iou(_bbox_tuple(component.bbox), target.bbox) >= threshold
            for target in gt.components
            if not target.mask.eligible
        )
    }
    metric = PRF1(
        true_positive=len(matches),
        false_positive=len(predicted) - len(matched_pred) - len(ignored_pred),
        false_negative=len(expected) - len(matches),
    )
    by_pred = {left: (right, score) for left, right, score in matches}
    component_map = {
        predicted[left].internal_id: expected[right].raw_id
        for left, (right, _score) in by_pred.items()
    }
    category_exact = sum(
        predicted[left].category == expected[right].category
        for left, right, _score in matches
    )
    category_normalized = sum(
        _normal_text(predicted[left].category) == _normal_text(expected[right].category)
        for left, right, _score in matches
    )
    fields = {
        "designator": _component_field_metrics(
            predicted,
            expected,
            by_pred,
            target=lambda item: item.designator,
            prediction=lambda item: item.key,
        ),
        "value": _component_field_metrics(
            predicted,
            expected,
            by_pred,
            target=lambda item: item.value,
            prediction=lambda item: item.value,
        ),
        "search_name": _component_field_metrics(
            predicted,
            expected,
            by_pred,
            target=lambda item: item.search_name,
            prediction=lambda item: item.name,
        ),
    }
    return (
        {
            "detection": {**metric.as_dict(), "ignored_predictions": len(ignored_pred)},
            "matched_iou": _score_summary([score for _left, _right, score in matches]),
            "category": {
                "correct": category_exact,
                "normalized_correct": category_normalized,
                "evaluated": len(matches),
                "accuracy": _optional_ratio(category_exact, len(matches)),
                "normalized_accuracy": _optional_ratio(category_normalized, len(matches)),
            },
            "fields": fields,
        },
        component_map,
    )


def _pin_metrics(
    ir: CircuitIR,
    gt: RawGroundTruth,
    *,
    component_map: Mapping[str, str],
    tolerance: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    predicted = sorted(ir.pins, key=lambda item: _natural_key(item.internal_id))
    expected = sorted(
        (item for item in gt.pins if item.mask.eligible and item.tip is not None),
        key=lambda item: _natural_key(item.raw_id),
    )
    candidates: list[tuple[int, int, float]] = []
    for pred_index, pin in enumerate(predicted):
        owner = component_map.get(str(pin.component_id)) if pin.component_id else None
        if owner is None:
            continue
        for gt_index, target in enumerate(expected):
            if target.owner_raw_id != owner or target.tip is None:
                continue
            distance = math.hypot(pin.point.x - target.tip[0], pin.point.y - target.tip[1])
            if distance <= tolerance:
                # The generic matcher maximizes score; negate distance through
                # a monotone positive score while retaining the exact distance.
                candidates.append((pred_index, gt_index, tolerance - distance + 1.0))
    matched = _maximum_cardinality_matching(
        len(predicted),
        len(expected),
        candidates,
        left_keys=[item.internal_id for item in predicted],
        right_keys=[item.raw_id for item in expected],
    )
    by_pred: dict[int, tuple[int, float]] = {}
    distances: list[float] = []
    for left, right, _score in matched:
        assert expected[right].tip is not None
        distance = math.hypot(
            predicted[left].point.x - expected[right].tip[0],
            predicted[left].point.y - expected[right].tip[1],
        )
        by_pred[left] = (right, distance)
        distances.append(distance)
    matched_pred = set(by_pred)
    ignored_pred: set[int] = set()
    for pred_index, pin in enumerate(predicted):
        if pred_index in matched_pred:
            continue
        owner = component_map.get(str(pin.component_id)) if pin.component_id else None
        if owner is None:
            continue
        if any(
            target.owner_raw_id == owner
            and target.tip is not None
            and math.hypot(pin.point.x - target.tip[0], pin.point.y - target.tip[1]) <= tolerance
            for target in gt.pins
            if not target.mask.eligible
        ):
            ignored_pred.add(pred_index)
    metric = PRF1(
        true_positive=len(matched),
        false_positive=len(predicted) - len(matched_pred) - len(ignored_pred),
        false_negative=len(expected) - len(matched),
    )
    pin_map = {
        predicted[left].internal_id: expected[right].raw_id
        for left, (right, _distance) in by_pred.items()
    }
    fields = {
        "number": _pin_field_metrics(
            predicted,
            expected,
            by_pred,
            target=lambda item: item.number,
            prediction=lambda item: item.number,
        ),
        "name": _pin_field_metrics(
            predicted,
            expected,
            by_pred,
            target=lambda item: item.name,
            prediction=lambda item: item.name,
        ),
    }
    return (
        {
            "detection": {**metric.as_dict(), "ignored_predictions": len(ignored_pred)},
            "owner_constrained": True,
            "point_error_px": _error_summary(distances, include_values=True),
            "fields": fields,
        },
        pin_map,
    )


def _wire_metrics(ir: CircuitIR, gt: RawGroundTruth, *, tolerance: int) -> dict[str, Any]:
    expected = [item for item in gt.wires if item.mask.eligible]
    expected_paths = [path for wire in expected for path in wire.paths]
    predicted_segments = [
        (
            (segment.start.x, segment.start.y),
            (segment.end.x, segment.end.y),
        )
        for segment in ir.segments
    ]
    return {
        "eligible_raw_wires": len(expected),
        "predicted_segments": len(ir.segments),
        **raster_centerline_coverage(
            expected_paths,
            predicted_segments,
            width=gt.width,
            height=gt.height,
            tolerance=tolerance,
        ),
        "tolerance_distance": "Chebyshev (square dilation)",
    }


def raster_centerline_coverage(
    expected_paths: Sequence[Sequence[tuple[float, float]]],
    predicted_segments: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    *,
    width: int,
    height: int,
    tolerance: int,
) -> dict[str, Any]:
    expected_image = Image.new("L", (width, height), 0)
    expected_draw = ImageDraw.Draw(expected_image)
    for path in expected_paths:
        integer_path = [(round(x), round(y)) for x, y in path]
        if len(integer_path) == 1:
            expected_draw.point(integer_path[0], fill=255)
        elif len(integer_path) >= 2:
            expected_draw.line(integer_path, fill=255, width=1)

    predicted_image = Image.new("L", (width, height), 0)
    predicted_draw = ImageDraw.Draw(predicted_image)
    for (start, end) in predicted_segments:
        predicted_draw.line(
            [
                (round(start[0]), round(start[1])),
                (round(end[0]), round(end[1])),
            ],
            fill=255,
            width=1,
        )
    expected_mask = np.asarray(expected_image) > 0
    predicted_mask = np.asarray(predicted_image) > 0
    if tolerance:
        kernel = 2 * tolerance + 1
        expected_dilated = np.asarray(expected_image.filter(ImageFilter.MaxFilter(kernel))) > 0
        predicted_dilated = np.asarray(predicted_image.filter(ImageFilter.MaxFilter(kernel))) > 0
    else:
        expected_dilated = expected_mask
        predicted_dilated = predicted_mask
    predicted_pixels = int(predicted_mask.sum())
    expected_pixels = int(expected_mask.sum())
    predicted_hits = int((predicted_mask & expected_dilated).sum())
    expected_hits = int((expected_mask & predicted_dilated).sum())
    precision = _ratio(
        predicted_hits,
        predicted_pixels,
        empty=1.0 if expected_pixels == 0 else 0.0,
    )
    recall = _ratio(
        expected_hits,
        expected_pixels,
        empty=1.0 if predicted_pixels == 0 else 0.0,
    )
    return {
        "predicted_pixels": predicted_pixels,
        "expected_pixels": expected_pixels,
        "predicted_hit_pixels": predicted_hits,
        "expected_hit_pixels": expected_hits,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def _graph_metrics(
    ir: CircuitIR,
    gt: RawGroundTruth,
    *,
    pin_map: Mapping[str, str],
) -> dict[str, Any]:
    expected_pairs = _connectivity_pairs(gt.connectivity_nets)
    predicted_nets: list[set[str]] = []
    for net in ir.electrical_nets:
        mapped = {pin_map[pin_id] for pin_id in net.pins if pin_id in pin_map}
        if mapped:
            predicted_nets.append(mapped)
    predicted_pairs = _connectivity_pairs(predicted_nets)
    metric = PRF1(
        true_positive=len(predicted_pairs & expected_pairs),
        false_positive=len(predicted_pairs - expected_pairs),
        false_negative=len(expected_pairs - predicted_pairs),
    )
    return {
        "pairwise_connectivity": metric.as_dict(),
        "expected_nets": len(gt.connectivity_nets),
        "predicted_nets_with_matched_pins": len(predicted_nets),
        "eligible_gt_pins": sum(item.mask.eligible for item in gt.pins),
        "matched_gt_pins": len(set(pin_map.values())),
        "conditioning": "matched eligible raw pins only",
    }


T = TypeVar("T")


def _component_field_metrics(
    predicted: Sequence[Any],
    expected: Sequence[T],
    by_pred: Mapping[int, tuple[int, float]],
    *,
    target: Callable[[T], TextTarget],
    prediction: Callable[[Any], Any],
) -> dict[str, Any]:
    by_expected = {right: left for left, (right, _score) in by_pred.items()}
    return _field_metrics(predicted, expected, by_expected, target=target, prediction=prediction)


def _pin_field_metrics(
    predicted: Sequence[Any],
    expected: Sequence[T],
    by_pred: Mapping[int, tuple[int, float]],
    *,
    target: Callable[[T], TextTarget],
    prediction: Callable[[Any], Any],
) -> dict[str, Any]:
    by_expected = {right: left for left, (right, _distance) in by_pred.items()}
    return _field_metrics(predicted, expected, by_expected, target=target, prediction=prediction)


def _field_metrics(
    predicted: Sequence[Any],
    expected: Sequence[T],
    by_expected: Mapping[int, int],
    *,
    target: Callable[[T], TextTarget],
    prediction: Callable[[Any], Any],
) -> dict[str, Any]:
    eligible = [index for index, item in enumerate(expected) if target(item).mask.eligible]
    matched = [index for index in eligible if index in by_expected]
    exact = 0
    normalized = 0
    for index in matched:
        expected_value = target(expected[index]).value
        predicted_value = prediction(predicted[by_expected[index]])
        if isinstance(predicted_value, str) and expected_value is not None:
            exact += predicted_value == expected_value
            normalized += _normal_text(predicted_value) == _normal_text(expected_value)
    return {
        "eligible": len(eligible),
        "matched_eligible": len(matched),
        "exact_correct": exact,
        "normalized_correct": normalized,
        "exact_accuracy_end_to_end": _optional_ratio(exact, len(eligible)),
        "normalized_accuracy_end_to_end": _optional_ratio(normalized, len(eligible)),
        "exact_accuracy_on_matched": _optional_ratio(exact, len(matched)),
        "normalized_accuracy_on_matched": _optional_ratio(normalized, len(matched)),
    }


def bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _maximum_cardinality_matching(
    left_count: int,
    right_count: int,
    candidates: Sequence[tuple[int, int, float]],
    *,
    left_keys: Sequence[str],
    right_keys: Sequence[str],
) -> list[tuple[int, int, float]]:
    adjacency: dict[int, list[tuple[int, float]]] = defaultdict(list)
    scores: dict[tuple[int, int], float] = {}
    for left, right, score in candidates:
        if not (0 <= left < left_count and 0 <= right < right_count):
            raise ValueError("matching candidate index is out of bounds")
        adjacency[left].append((right, float(score)))
        scores[(left, right)] = float(score)
    for left in adjacency:
        adjacency[left].sort(key=lambda item: (-item[1], _natural_key(right_keys[item[0]])))
    owner: dict[int, int] = {}

    def augment(left: int, visited: set[int]) -> bool:
        for right, _score in adjacency.get(left, ()):
            if right in visited:
                continue
            visited.add(right)
            if right not in owner or augment(owner[right], visited):
                owner[right] = left
                return True
        return False

    for left in sorted(range(left_count), key=lambda index: _natural_key(left_keys[index])):
        augment(left, set())
    return sorted(
        ((left, right, scores[(left, right)]) for right, left in owner.items()),
        key=lambda item: (_natural_key(left_keys[item[0]]), _natural_key(right_keys[item[1]])),
    )


def _connectivity_pairs(nets: Iterable[Iterable[str]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for net in nets:
        pins = sorted(set(map(str, net)), key=_natural_key)
        pairs.update(tuple(sorted(pair, key=_natural_key)) for pair in combinations(pins, 2))
    return pairs


def _aggregate_prf(values: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(values)
    metric = PRF1(
        true_positive=sum(int(item["tp"]) for item in items),
        false_positive=sum(int(item["fp"]) for item in items),
        false_negative=sum(int(item["fn"]) for item in items),
    )
    result = metric.as_dict()
    if any("ignored_predictions" in item for item in items):
        result["ignored_predictions"] = sum(int(item.get("ignored_predictions", 0)) for item in items)
    return result


def _aggregate_field(values: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(values)
    eligible = sum(int(item["eligible"]) for item in items)
    matched = sum(int(item["matched_eligible"]) for item in items)
    exact = sum(int(item["exact_correct"]) for item in items)
    normalized = sum(int(item["normalized_correct"]) for item in items)
    return {
        "eligible": eligible,
        "matched_eligible": matched,
        "exact_correct": exact,
        "normalized_correct": normalized,
        "exact_accuracy_end_to_end": _optional_ratio(exact, eligible),
        "normalized_accuracy_end_to_end": _optional_ratio(normalized, eligible),
        "exact_accuracy_on_matched": _optional_ratio(exact, matched),
        "normalized_accuracy_on_matched": _optional_ratio(normalized, matched),
    }


def _aggregate_coverage(values: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(values)
    result: dict[str, Any] = {"objects": {}, "fields": {}}
    for section in ("objects", "fields"):
        keys = sorted({key for item in items for key in item[section]})
        for key in keys:
            entries = [item[section][key] for item in items if key in item[section]]
            reasons: Counter[str] = Counter()
            for entry in entries:
                reasons.update({str(reason): int(count) for reason, count in entry["mask_reasons"].items()})
            result[section][key] = {
                "total": sum(int(entry["total"]) for entry in entries),
                "eligible": sum(int(entry["eligible"]) for entry in entries),
                "masked": sum(int(entry["masked"]) for entry in entries),
                "mask_reasons": dict(sorted(reasons.items())),
            }
    graph_entries = [item["graph_connections"] for item in items]
    graph_reasons: Counter[str] = Counter()
    for entry in graph_entries:
        graph_reasons.update({str(reason): int(count) for reason, count in entry["mask_reasons"].items()})
    result["graph_connections"] = {
        "total": sum(int(entry["total"]) for entry in graph_entries),
        "eligible": sum(int(entry["eligible"]) for entry in graph_entries),
        "masked": sum(int(entry["masked"]) for entry in graph_entries),
        "mask_reasons": dict(sorted(graph_reasons.items())),
    }
    return result


def _bbox_tuple(value: BBox) -> tuple[float, float, float, float]:
    return value.x1, value.y1, value.x2, value.y2


def _normal_text(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _score_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _error_summary(values: Sequence[float], *, include_values: bool) -> dict[str, Any]:
    ordered = sorted(map(float, values))
    result: dict[str, Any] = {
        "count": len(ordered),
        "median": median(ordered) if ordered else None,
        "p95": _percentile(ordered, 0.95) if ordered else None,
        "max": max(ordered) if ordered else None,
    }
    if include_values:
        result["values"] = ordered
    return result


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _ratio(numerator: int, denominator: int, *, empty: float) -> float:
    return numerator / denominator if denominator else empty


def _optional_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _natural_key(value: object) -> tuple[tuple[int, object], ...]:
    import re

    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold(), part)
        for part in re.split(r"(\d+)", str(value))
        if part
    )


__all__ = [
    "aggregate_masked_reports",
    "bbox_iou",
    "evaluate_masked_case",
    "raster_centerline_coverage",
]
