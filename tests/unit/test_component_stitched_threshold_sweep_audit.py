from __future__ import annotations

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "audit_component_stitched_threshold_sweep.py"
SPEC = importlib.util.spec_from_file_location(
    "audit_component_stitched_threshold_sweep", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _raises(exception: type[BaseException], match: str):
    return unittest.TestCase().assertRaisesRegex(exception, match)


def _truth(
    annotation_id: int,
    sample_id: str,
    category: str,
    box: list[float],
    domain: str = "domain-a",
) -> dict[str, object]:
    return {
        "annotation_id": annotation_id,
        "sample_id": sample_id,
        "category": category,
        "box": box,
        "source_domain": domain,
    }


def _detection(
    sample_id: str,
    category: str,
    score: float,
    box: list[float],
    prediction_index: int,
    domain: str = "domain-a",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "category": category,
        "score": score,
        "box": box,
        "prediction_index": prediction_index,
        "source_view_id": "tile_000_000",
        "source_class": domain,
        "source_domain": domain,
    }


def _fixture_population() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    truths = [
        _truth(1, "a", "R", [0.0, 0.0, 10.0, 10.0]),
        _truth(2, "a", "R", [10.0, 0.0, 20.0, 10.0]),
        _truth(3, "a", "C", [0.0, 0.0, 10.0, 10.0]),
        _truth(4, "b", "R", [0.0, 0.0, 10.0, 10.0], "domain-b"),
    ]
    detections = [
        _detection("a", "R", 0.9, [0.0, 0.0, 10.0, 10.0], 0),
        _detection("a", "C", 0.9, [0.0, 0.0, 10.0, 10.0], 1),
        _detection("a", "R", 0.7, [10.0, 0.0, 20.0, 10.0], 2),
        _detection("a", "R", 0.7, [30.0, 30.0, 40.0, 40.0], 3),
        _detection(
            "b", "R", 0.4, [0.0, 0.0, 5.0, 10.0], 0, "domain-b"
        ),
    ]
    return truths, detections


def test_module_has_no_torch_import() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert "torch" not in imports
    assert "torch" not in AUDIT.__dict__


def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with _raises(ValueError, "duplicate JSON key"):
        AUDIT._read_object(duplicate, "fixture")

    nonfinite = tmp_path / "nonfinite.jsonl"
    nonfinite.write_text('{"score": NaN}\n', encoding="utf-8")
    with _raises(ValueError, "non-standard JSON constant"):
        AUDIT._read_jsonl(nonfinite, "fixture")


def test_incremental_boundaries_equal_brute_force_refiltering() -> None:
    truths, detections = _fixture_population()
    annotated = AUDIT._annotate_matches(detections, truths, iou_threshold=0.5)
    prefix = AUDIT._build_prefix(annotated, len(truths))

    for threshold in sorted({float(row["score"]) for row in detections}):
        retained = [
            row for row in detections if float(row["score"]) >= threshold
        ]
        brute = AUDIT._annotate_matches(retained, truths, iou_threshold=0.5)
        brute_tp = sum(bool(row["is_true_positive"]) for row in brute)
        expected = AUDIT._metric_counts(brute_tp, len(brute) - brute_tp, len(truths))
        assert AUDIT._state_at(prefix, threshold) == expected


def test_same_score_group_is_recorded_only_after_entire_group() -> None:
    truths, detections = _fixture_population()
    annotated = AUDIT._annotate_matches(detections, truths, iou_threshold=0.5)
    prefix = AUDIT._build_prefix(annotated, len(truths))
    state = AUDIT._state_at(prefix, 0.7)
    assert state == AUDIT._metric_counts(3, 1, 4)
    assert prefix["scores_desc"] == [0.9, 0.7, 0.4]


def test_iou_exactly_point_five_is_true_positive_and_classes_are_isolated() -> None:
    truths = [
        _truth(1, "a", "R", [0.0, 0.0, 10.0, 10.0]),
        _truth(2, "a", "C", [0.0, 0.0, 10.0, 10.0]),
    ]
    detections = [
        _detection("a", "R", 0.8, [0.0, 0.0, 5.0, 10.0], 0),
        _detection("a", "X", 0.7, [0.0, 0.0, 10.0, 10.0], 1),
    ]
    annotated = AUDIT._annotate_matches(detections, truths, iou_threshold=0.5)
    flags = {row["category"]: row["is_true_positive"] for row in annotated}
    assert flags == {"R": True, "X": False}


def test_boundary_grid_is_exact_union_and_does_not_invent_floor() -> None:
    candidates = {
        "v4": {"detections": [{"score": 0.2001}, {"score": 0.7}]},
        "v5": {"detections": [{"score": 0.3}, {"score": 0.7}]},
    }
    thresholds, origins = AUDIT._boundary_thresholds(candidates, 0.2)
    assert thresholds == [0.2001, 0.3, 0.7]
    assert 0.2 not in thresholds
    assert origins == {0.2001: ["v4"], 0.3: ["v5"], 0.7: ["v4", "v5"]}

    candidates["v4"]["detections"].append({"score": 0.1999})
    with _raises(ValueError, "below population floor"):
        AUDIT._boundary_thresholds(candidates, 0.2)


def test_best_threshold_uses_exact_f1_precision_recall_then_higher_score() -> None:
    rows = [
        {
            "score_threshold": 0.4,
            "score_boundary_origins": ["model"],
            "metrics": {"model": AUDIT._metric_counts(2, 2, 2)},
        },
        {
            "score_threshold": 0.6,
            "score_boundary_origins": ["model"],
            "metrics": {"model": AUDIT._metric_counts(1, 0, 2)},
        },
        {
            "score_threshold": 0.7,
            "score_boundary_origins": ["other"],
            "metrics": {"model": AUDIT._metric_counts(1, 0, 2)},
        },
    ]
    # The first two rows have exactly the same F1 (2/3); precision selects 0.6.
    assert AUDIT._best_row(rows, "model")["score_threshold"] == 0.6

    tied = [
        {
            "score_threshold": threshold,
            "score_boundary_origins": ["model"],
            "metrics": {"model": AUDIT._metric_counts(1, 0, 2)},
        }
        for threshold in (0.6, 0.8)
    ]
    assert AUDIT._best_row(tied, "model")["score_threshold"] == 0.8


def test_domain_and_class_slices_reuse_valid_independent_match_flags() -> None:
    truths, detections = _fixture_population()
    annotated = AUDIT._annotate_matches(detections, truths, iou_threshold=0.5)
    domain_b = AUDIT._slice_metric(
        annotated, truths, threshold=0.2, sample_ids={"b"}
    )
    assert domain_b == AUDIT._metric_counts(1, 0, 1)
    class_c = AUDIT._slice_metric(
        annotated, truths, threshold=0.2, category="C"
    )
    assert class_c == AUDIT._metric_counts(1, 0, 1)


def test_atomic_writer_refuses_to_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    AUDIT._atomic_write_new(output, {"value": 1})
    original = output.read_bytes()
    assert json.loads(original) == {"value": 1}
    with _raises(FileExistsError, "refusing to overwrite"):
        AUDIT._atomic_write_new(output, {"value": 2})
    assert output.read_bytes() == original


def load_tests(loader, tests, pattern):
    del loader, tests, pattern
    suite = unittest.TestSuite()
    direct = (
        test_module_has_no_torch_import,
        test_incremental_boundaries_equal_brute_force_refiltering,
        test_same_score_group_is_recorded_only_after_entire_group,
        test_iou_exactly_point_five_is_true_positive_and_classes_are_isolated,
        test_boundary_grid_is_exact_union_and_does_not_invent_floor,
        test_best_threshold_uses_exact_f1_precision_recall_then_higher_score,
        test_domain_and_class_slices_reuse_valid_independent_match_flags,
    )
    for function in direct:
        suite.addTest(unittest.FunctionTestCase(function))

    def with_temporary_path(function):
        def run() -> None:
            with tempfile.TemporaryDirectory() as temporary:
                function(Path(temporary))

        return run

    for function in (
        test_strict_json_rejects_duplicate_keys_and_nonfinite_constants,
        test_atomic_writer_refuses_to_overwrite,
    ):
        suite.addTest(unittest.FunctionTestCase(with_temporary_path(function)))
    return suite
