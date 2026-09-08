from __future__ import annotations

"""Audit frozen Pin-v5 validation predictions without running a model.

The audit fails closed unless the completed formal status, candidate bundle,
model card, model checkpoint, ONNX artifact, prediction JSONL, current
resume-compatible training config, and Dataset300-v2 materialization all agree.
Only already-decoded validation predictions are scored.  No image is decoded,
no model forward is executed, and no score/NMS parameter is searched.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


FINAL_FORMAL_STATUSES = frozenset({"candidate_evidence_ready", "gate_failed"})
SPLITS = ("train", "val", "test")
MATCH_RADIUS_INPUT_PX = 4.0
OUTPUT_STRIDE = 4
MATCH_RADIUS_HEATMAP = MATCH_RADIUS_INPUT_PX / OUTPUT_STRIDE
BOUNDARY_TOLERANCE_HEATMAP = MATCH_RADIUS_HEATMAP
GT_COUNT_BINS: tuple[tuple[str, int, int | None], ...] = (
    ("0", 0, 1),
    ("1-2", 1, 3),
    ("3-4", 3, 5),
    ("5-8", 5, 9),
    ("9-16", 9, 17),
    ("17+", 17, None),
)
SIZE_BINS: tuple[tuple[str, float, float | None], ...] = (
    ("small_sqrt_area_lt_32px", 0.0, 32.0**2),
    ("medium_sqrt_area_32_to_96px", 32.0**2, 96.0**2),
    ("large_sqrt_area_ge_96px", 96.0**2, None),
)
WEAK_RECALL_CUTOFF = 0.30
WEAK_MIN_GT = 5


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Build a hash-bound, frozen-validation Pin-v5 stratification audit "
            "from the formal prediction artifact; never run inference or tune."
        )
    )
    parser.add_argument("--workspace-root", type=Path, default=root)
    parser.add_argument(
        "--run-root", type=Path, default=Path("work/train-pin-v5-boundary")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train/dataset300_v2_pin_v5_boundary.json"),
    )
    parser.add_argument(
        "--materialization-summary",
        type=Path,
        default=Path(
            "artifacts/training/materialized/dataset300-v2/component-pin/summary.json"
        ),
    )
    parser.add_argument(
        "--pin-index",
        type=Path,
        default=Path(
            "artifacts/training/materialized/dataset300-v2/component-pin/pin_index.jsonl"
        ),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/manifests/dataset300.jsonl"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/splits/dataset300_split.jsonl"),
    )
    parser.add_argument(
        "--task-manifest",
        type=Path,
        default=Path("data/derived/dataset300-v2/pin_keypoints.jsonl"),
    )
    parser.add_argument(
        "--derived-summary",
        type=Path,
        default=Path("data/derived/dataset300-v2/summary.json"),
    )
    parser.add_argument(
        "--component-manifest",
        type=Path,
        default=Path("data/derived/dataset300-v2/component_detection.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/pin_v5_val_stratified.json"),
    )
    return parser


def _resolve_within(
    root: Path, raw: Path, label: str, *, kind: str = "file", must_exist: bool = True
) -> Path:
    path = raw.expanduser()
    resolved = (path if path.is_absolute() else root / path).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside workspace: {resolved}") from exc
    if must_exist:
        if kind == "directory":
            valid = resolved.is_dir()
        else:
            valid = resolved.is_file() and resolved.stat().st_size > 0
        if not valid:
            raise FileNotFoundError(f"{label} is missing or empty: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _loads(raw: str, label: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = _loads(path.read_text(encoding="utf-8"), f"{label} JSON at {path}")
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _iter_jsonl(path: Path, label: str) -> Iterable[dict[str, Any]]:
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"blank {label} row at {path}:{line_number}")
            value = _loads(raw, f"{label} JSONL at {path}:{line_number}")
            if not isinstance(value, dict):
                raise ValueError(f"{label} row is not an object at {path}:{line_number}")
            rows += 1
            yield value
    if rows == 0:
        raise ValueError(f"{label} is empty: {path}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an array")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _pair(value: Any, label: str) -> tuple[float, float]:
    row = _sequence(value, label)
    if len(row) != 2:
        raise ValueError(f"{label} must contain two values")
    return _finite(row[0], label), _finite(row[1], label)


def _box(value: Any, label: str) -> tuple[float, float, float, float]:
    row = _sequence(value, label)
    if len(row) != 4:
        raise ValueError(f"{label} must contain four values")
    x1, y1, x2, y2 = (_finite(item, label) for item in row)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{label} must have positive width and height")
    return x1, y1, x2, y2


def _close(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _expect_sha(actual: str, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} expected SHA-256 is missing or invalid")
    if actual != expected.lower():
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected.lower()}, actual={actual}"
        )


def _relative(path: Path, workspace: Path) -> str:
    return path.relative_to(workspace).as_posix()


def _unique_index(
    rows: Iterable[Mapping[str, Any]], key: str, label: str
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        identity = _text(row.get(key), f"{label}.{key}")
        if identity in output:
            raise ValueError(f"duplicate {label} {key}: {identity}")
        output[identity] = row
    return output


def _match_pairs(
    predicted: Sequence[Mapping[str, Any]],
    expected: Sequence[tuple[float, float]],
    threshold: float = MATCH_RADIUS_HEATMAP,
) -> tuple[tuple[int, int], ...]:
    """Mirror the production maximum-cardinality matcher and retain owners."""

    radius = _finite(threshold, "matching threshold")
    if radius <= 0:
        raise ValueError("matching threshold must be positive")
    adjacency: list[list[int]] = []
    for prediction_number, item in enumerate(predicted):
        x = _finite(item.get("x"), f"prediction[{prediction_number}].x")
        y = _finite(item.get("y"), f"prediction[{prediction_number}].y")
        choices = [
            index
            for index, point in enumerate(expected)
            if math.hypot(x - point[0], y - point[1]) <= radius
        ]
        choices.sort(key=lambda index: math.hypot(x - expected[index][0], y - expected[index][1]))
        adjacency.append(choices)
    owner: dict[int, int] = {}

    def augment(prediction_index: int, seen: set[int]) -> bool:
        for expected_index in adjacency[prediction_index]:
            if expected_index in seen:
                continue
            seen.add(expected_index)
            if expected_index not in owner or augment(owner[expected_index], seen):
                owner[expected_index] = prediction_index
                return True
        return False

    for index in range(len(predicted)):
        augment(index, set())
    return tuple(sorted(((prediction, truth) for truth, prediction in owner.items())))


def _summary(matched: int, predicted: int, truth: int) -> dict[str, Any]:
    if min(matched, predicted, truth) < 0 or matched > min(predicted, truth):
        raise ValueError("invalid Pin detection totals")
    precision = matched / predicted if predicted else 0.0
    recall = matched / truth if truth else 0.0
    denominator = precision + recall
    f1 = 2.0 * precision * recall / denominator if denominator else 0.0
    return {
        "matched": matched,
        "pred_count": predicted,
        "gt_count": truth,
        "false_positive_count": predicted - matched,
        "false_negative_count": truth - matched,
        "precision_4px": precision,
        "recall_4px": recall,
        "f1_4px": f1,
        "pred_to_gt_ratio": predicted / truth if truth else None,
    }


def _new_accumulator() -> dict[str, Any]:
    return {
        "matched": 0,
        "predicted": 0,
        "truth": 0,
        "views": set(),
        "samples": set(),
    }


def _accumulate(
    table: dict[str, dict[str, Any]],
    key: str,
    *,
    matched: int,
    predicted: int,
    truth: int,
    view_id: str,
    sample_id: str,
) -> None:
    row = table.setdefault(key, _new_accumulator())
    row["matched"] += matched
    row["predicted"] += predicted
    row["truth"] += truth
    row["views"].add(view_id)
    row["samples"].add(sample_id)


def _finalize_table(table: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in sorted(table):
        row = table[key]
        output[key] = {
            **_summary(int(row["matched"]), int(row["predicted"]), int(row["truth"])),
            "view_count": len(row["views"]),
            "sample_count": len(row["samples"]),
        }
    return output


def _attributed_counts(
    *,
    truth_groups: Sequence[str],
    prediction_groups: Sequence[str],
    pairs: Sequence[tuple[int, int]],
) -> dict[str, tuple[int, int, int]]:
    """Attribute a TP to truth; unmatched predictions retain their own group."""

    matched_predictions = {prediction for prediction, _truth in pairs}
    matched_truths = {truth for _prediction, truth in pairs}
    if len(matched_predictions) != len(pairs) or len(matched_truths) != len(pairs):
        raise ValueError("matching pairs are not one-to-one")
    if any(index >= len(prediction_groups) for index in matched_predictions) or any(
        index >= len(truth_groups) for index in matched_truths
    ):
        raise ValueError("matching pair index is out of range")
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for prediction_index, truth_index in pairs:
        key = truth_groups[truth_index]
        totals[key][0] += 1
        totals[key][1] += 1
        totals[key][2] += 1
    for truth_index, key in enumerate(truth_groups):
        if truth_index not in matched_truths:
            totals[key][2] += 1
    for prediction_index, key in enumerate(prediction_groups):
        if prediction_index not in matched_predictions:
            totals[key][1] += 1
    return {key: tuple(values) for key, values in totals.items()}


def _gt_count_bin(count: int) -> str:
    if type(count) is not int or count < 0:
        raise ValueError("GT count must be a non-negative integer")
    for name, minimum, maximum in GT_COUNT_BINS:
        if count >= minimum and (maximum is None or count < maximum):
            return name
    raise AssertionError("unreachable GT-count bin")


def _size_bin(area_input_px: float) -> str:
    area = _finite(area_input_px, "component area")
    if area <= 0:
        raise ValueError("component area must be positive")
    for name, minimum, maximum in SIZE_BINS:
        if area >= minimum and (maximum is None or area < maximum):
            return name
    raise AssertionError("unreachable component-size bin")


def _orientation_group(value: Any, label: str) -> str:
    x, y = _pair(value, label)
    magnitude = math.hypot(x, y)
    if magnitude <= 1e-12:
        return "unknown_zero_vector"
    x /= magnitude
    y /= magnitude
    if abs(x) >= abs(y):
        return "right" if x >= 0 else "left"
    return "down" if y >= 0 else "up"


def _boundary_groups(
    point: tuple[float, float],
    component_box: tuple[float, float, float, float],
    *,
    tolerance: float = BOUNDARY_TOLERANCE_HEATMAP,
) -> tuple[str, str]:
    x, y = point
    x1, y1, x2, y2 = component_box
    near_left = abs(x - x1) <= tolerance and y1 - tolerance <= y <= y2 + tolerance
    near_right = abs(x - x2) <= tolerance and y1 - tolerance <= y <= y2 + tolerance
    near_top = abs(y - y1) <= tolerance and x1 - tolerance <= x <= x2 + tolerance
    near_bottom = abs(y - y2) <= tolerance and x1 - tolerance <= x <= x2 + tolerance
    vertical = min(
        ((abs(x - x1), "L", near_left), (abs(x - x2), "R", near_right)),
        key=lambda item: (item[0], item[1]),
    )
    horizontal = min(
        ((abs(y - y1), "T", near_top), (abs(y - y2), "B", near_bottom)),
        key=lambda item: (item[0], item[1]),
    )
    if vertical[2] and horizontal[2]:
        return "corner", f"corner_{horizontal[1]}{vertical[1]}"
    if vertical[2]:
        return "edge", f"edge_{vertical[1]}"
    if horizontal[2]:
        return "edge", f"edge_{horizontal[1]}"
    return "off_boundary", "off_boundary"


def _transform_point(
    point: tuple[float, float], matrix_value: Any, label: str
) -> tuple[float, float]:
    matrix = _sequence(matrix_value, label)
    if len(matrix) != 3:
        raise ValueError(f"{label} must have three rows")
    rows = [_sequence(row, label) for row in matrix]
    if any(len(row) != 3 for row in rows):
        raise ValueError(f"{label} must be 3x3")
    values = [[_finite(item, label) for item in row] for row in rows]
    x, y = point
    denominator = values[2][0] * x + values[2][1] * y + values[2][2]
    if abs(denominator) <= 1e-12:
        raise ValueError(f"{label} maps point to infinity")
    return (
        (values[0][0] * x + values[0][1] * y + values[0][2]) / denominator,
        (values[1][0] * x + values[1][1] * y + values[1][2]) / denominator,
    )


def _inspect_config(workspace: Path, path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(workspace / "src"))
    from pcbparse.training.config import (
        load_training_config,
        training_config_resume_sha256s,
        training_config_sha256,
    )

    config = load_training_config(path)
    task = config.tasks["pin"]
    return {
        "canonical_sha256": training_config_sha256(config),
        "resume_compatible_sha256s": list(training_config_resume_sha256s(config)),
        "dataset": {
            "config_hash": config.dataset.config_hash,
            "source_manifest_sha256": config.dataset.source_manifest_sha256,
            "split_manifest_sha256": config.dataset.split_manifest_sha256,
            "task_manifest_sha256": config.dataset.task_manifest_sha256["pin_keypoints"],
            "summary_sha256": config.dataset.summary_sha256,
            "expected_sample_count": config.dataset.expected_sample_count,
        },
        "protocol": {
            "architecture": task.architecture,
            "input_size": list(task.input_size),
            "score_threshold": float(task.pin_score_threshold),
            "nms_radius_heatmap_cells": float(task.pin_nms_radius),
            "max_detections_per_view": int(task.pin_max_detections),
            "box_margin_input_pixels": float(task.pin_box_margin_px),
            "box_prior_mode": str(task.pin_box_prior_mode),
            "output_stride": OUTPUT_STRIDE,
            "evaluation_gates": [dict(row) for row in task.evaluation_gates],
        },
    }


def _inspect_checkpoint(workspace: Path, path: Path) -> dict[str, Any]:
    # This is intentionally set before importing torch.  map_location="cpu"
    # below is the operative guarantee; no forward pass is performed.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.path.insert(0, str(workspace / "src"))
    from pcbparse.training.torch_pin_model import load_pin_checkpoint

    model, payload = load_pin_checkpoint(path, map_location="cpu", strict=True)
    del model
    return {
        "schema": payload.get("schema"),
        "task": payload.get("task"),
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "lineage": payload.get("lineage"),
        "lineage_sha256": payload.get("lineage_sha256"),
        "metrics": payload.get("metrics"),
        "model_config": payload.get("model_config"),
    }


def _verify_status_and_bundle(
    *,
    workspace: Path,
    run_root: Path,
    paths: Mapping[str, Path],
    hashes: Mapping[str, str],
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    status = _read_object(paths["run_status"], "Pin run status")
    bundle = _read_object(paths["bundle_manifest"], "Pin bundle manifest")
    card = _read_object(paths["model_card"], "Pin model card")
    metrics_file = _read_object(paths["metrics"], "Pin metrics")
    if (
        status.get("task") != "pin"
        or status.get("status") not in FINAL_FORMAL_STATUSES
        or status.get("formal_evidence") is not True
        or status.get("smoke") is not False
        or not status.get("finished_at_utc")
    ):
        raise ValueError("Pin status is not a completed formal non-smoke run")
    if status.get("status") == "candidate_evidence_ready" and (
        status.get("registerable") is not True or status.get("gates_passed") is not True
    ):
        raise ValueError("candidate-ready Pin status is not registerable/gate-passing")
    planned = status.get("planned_epochs")
    completed = status.get("completed_epochs")
    best_epoch = status.get("best_epoch")
    if (
        type(planned) is not int
        or type(completed) is not int
        or type(best_epoch) is not int
        or not 0 < best_epoch <= completed <= planned
        or (completed < planned and status.get("early_stopped") is not True)
    ):
        raise ValueError("Pin status does not prove a valid formal termination")
    gates = _sequence(status.get("gates"), "status.gates")
    if status.get("gates_passed") is True and any(
        not isinstance(gate, Mapping) or gate.get("passed") is not True for gate in gates
    ):
        raise ValueError("status.gates_passed contradicts individual gates")

    for key, expected_path in (("artifact", paths["model_onnx"]), ("predictions", paths["predictions"])):
        reference = _mapping(status.get(key), f"status.{key}")
        actual_path = _resolve_within(
            workspace,
            Path(_text(reference.get("path"), f"status.{key}.path")),
            f"status {key}",
        )
        if actual_path != expected_path:
            raise ValueError(f"status.{key}.path differs from supplied run artifact")
        _expect_sha(hashes["model_onnx" if key == "artifact" else "predictions"], reference.get("sha256"), f"status {key}")

    if (
        bundle.get("task") != "pin"
        or bundle.get("status") != "candidate"
        or bundle.get("blessed") is not False
        or card.get("task") != "pin"
    ):
        raise ValueError("Pin bundle/model card candidate contract is invalid")
    bundle_artifact = _resolve_within(
        workspace,
        run_root / Path(_text(bundle.get("artifact_path"), "bundle artifact_path")),
        "bundle artifact",
    )
    bundle_card = _resolve_within(
        workspace,
        run_root / Path(_text(bundle.get("model_card_path"), "bundle model_card_path")),
        "bundle model card",
    )
    if bundle_artifact != paths["model_onnx"] or bundle_card != paths["model_card"]:
        raise ValueError("bundle paths do not resolve to the supplied ONNX/model card")
    _expect_sha(hashes["model_onnx"], bundle.get("artifact_sha256"), "ONNX/bundle")
    _expect_sha(hashes["model_card"], bundle.get("model_card_sha256"), "card/bundle")
    if bundle.get("artifact_bytes") != paths["model_onnx"].stat().st_size:
        raise ValueError("bundle artifact_bytes differs from ONNX bytes")
    if (
        card.get("artifact_sha256") != bundle.get("artifact_sha256")
        or card.get("artifact_bytes") != bundle.get("artifact_bytes")
        or card.get("version") != bundle.get("version")
        or card.get("evaluation") != bundle.get("evaluation")
        or card.get("gate_policy") != bundle.get("gate_policy")
        or card.get("export_consistency") != bundle.get("export_consistency")
    ):
        raise ValueError("model card evidence differs from bundle manifest")

    evaluation = _mapping(bundle.get("evaluation"), "bundle.evaluation")
    _expect_sha(hashes["predictions"], evaluation.get("predictions_sha256"), "predictions/bundle evaluation")
    if evaluation.get("split") != "val" or evaluation.get("gates") != status.get("gates"):
        raise ValueError("bundle evaluation split/gates differ from formal status")
    status_metrics = _mapping(status.get("metrics"), "status.metrics")
    if (
        evaluation.get("metrics") != status_metrics
        or card.get("evaluation", {}).get("metrics") != status_metrics
        or metrics_file.get("metrics") != status_metrics
        or checkpoint.get("metrics") != status_metrics
    ):
        raise ValueError("formal metrics differ across status/bundle/card/file/checkpoint")

    lineage = _mapping(status.get("lineage"), "status.lineage")
    if checkpoint.get("lineage") != lineage or checkpoint.get("epoch") != best_epoch:
        raise ValueError("model.pt lineage/epoch differs from formal run status")
    if checkpoint.get("task") != "pin":
        raise ValueError("model.pt task is not Pin")
    model_config = _mapping(checkpoint.get("model_config"), "checkpoint.model_config")
    if model_config.get("output_stride") != OUTPUT_STRIDE:
        raise ValueError("model.pt output_stride differs from frozen protocol")

    compatible = set(_sequence(config.get("resume_compatible_sha256s"), "config compatible hashes"))
    run_config_hash = lineage.get("training_config_sha256")
    if run_config_hash not in compatible:
        raise ValueError("formal run config hash is not resume-compatible with current config")
    if any(
        value not in compatible
        for value in _sequence(
            lineage.get("resume_compatible_training_config_sha256s", []),
            "status resume-compatible hashes",
        )
    ):
        raise ValueError("status resume-compatible config hash is no longer compatible")
    for document, label in ((bundle, "bundle"), (card, "card")):
        if document.get("training_config_sha256") != run_config_hash:
            raise ValueError(f"{label} training config hash differs from status")
    protocol = _mapping(config.get("protocol"), "config.protocol")
    if (
        protocol.get("architecture") != lineage.get("architecture")
        or protocol.get("input_size") != lineage.get("input_size")
        or protocol.get("output_stride") != model_config.get("output_stride")
    ):
        raise ValueError("config/checkpoint/run architecture contract differs")
    if bundle.get("gate_policy", {}).get("evaluation_gates") != protocol.get("evaluation_gates"):
        raise ValueError("config evaluation gates differ from candidate bundle")

    summary_lineage = _mapping(summary.get("lineage"), "materialization summary.lineage")
    if lineage.get("materialization_lineage") != summary_lineage:
        raise ValueError("run/materialization lineage payload differs")
    expected_lineage = {
        "binding_sha256": summary.get("binding_sha256"),
        "materialization_commit_id": summary.get("binding_sha256"),
        "materialization_commit_sha256": hashes["materialization_summary"],
        "source_manifest_sha256": hashes["source_manifest"],
        "split_manifest_sha256": hashes["split_manifest"],
        "task_manifest_sha256": hashes["task_manifest"],
        "dataset_config_hash": summary_lineage.get("derivation_config_sha256"),
    }
    for key, expected in expected_lineage.items():
        if lineage.get(key) != expected:
            raise ValueError(f"status/materialization lineage mismatch for {key}")
    config_dataset = _mapping(config.get("dataset"), "config.dataset")
    for key, expected in (
        ("config_hash", expected_lineage["dataset_config_hash"]),
        ("source_manifest_sha256", hashes["source_manifest"]),
        ("split_manifest_sha256", hashes["split_manifest"]),
        ("task_manifest_sha256", hashes["task_manifest"]),
        ("summary_sha256", hashes["derived_summary"]),
    ):
        if config_dataset.get(key) != expected:
            raise ValueError(f"current config Dataset300 lineage mismatch for {key}")

    inputs = _mapping(summary_lineage.get("inputs"), "summary.lineage.inputs")
    for summary_key, input_key in (
        ("source_manifest", "source_manifest"),
        ("split_manifest", "split_manifest"),
        ("pin_manifest", "task_manifest"),
        ("component_manifest", "component_manifest"),
        ("summary", "derived_summary"),
    ):
        reference = _mapping(inputs.get(summary_key), f"summary input {summary_key}")
        _expect_sha(hashes[input_key], reference.get("sha256"), f"{summary_key}/summary")
    summary_outputs = _mapping(summary.get("outputs"), "summary.outputs")
    _expect_sha(
        hashes["pin_index"],
        _mapping(summary_outputs.get("pin_index.jsonl"), "summary pin index").get("sha256"),
        "pin index/materialization",
    )

    initialization = _mapping(lineage.get("initialization"), "status initialization")
    init_path = _resolve_within(
        workspace,
        Path(_text(initialization.get("path"), "initialization.path")),
        "Pin initialization checkpoint",
    )
    _expect_sha(_sha256(init_path), initialization.get("sha256"), "Pin initialization checkpoint")
    resume = _mapping(lineage.get("resume"), "status resume")
    if resume.get("resumed") is True:
        resume_path = _resolve_within(
            workspace,
            Path(_text(resume.get("checkpoint_path"), "resume.checkpoint_path")),
            "Pin resume checkpoint",
        )
        _expect_sha(_sha256(resume_path), resume.get("checkpoint_sha256"), "Pin resume checkpoint")

    evidence = {
        "artifact_path": status["artifact"]["path"],
        "version": bundle["version"],
        "license": bundle["license"],
        "precision": bundle["precision"],
        "adapter_entrypoint": bundle["adapter_smoke"]["entrypoint"],
        "export_consistency": bundle["export_consistency"],
        "evaluation": bundle["evaluation"],
        "intended_use": card["intended_use"],
        "limitations": status["limitations"],
    }
    if _canonical_sha256(evidence) != status.get("candidate_result_sha256"):
        raise ValueError("formal candidate_result_sha256 does not reproduce")
    return status, bundle, card, metrics_file


def _prepare_validation_population(
    *,
    pin_index_path: Path,
    source_manifest_path: Path,
    split_manifest_path: Path,
    binding: str,
    input_size: Sequence[int],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    source_by_sample = _unique_index(
        _iter_jsonl(source_manifest_path, "source manifest"),
        "sample_id",
        "source manifest",
    )
    split_by_sample = _unique_index(
        _iter_jsonl(split_manifest_path, "split manifest"),
        "sample_id",
        "split manifest",
    )
    if set(source_by_sample) != set(split_by_sample):
        raise ValueError("source and split manifest sample populations differ")
    for sample_id in source_by_sample:
        split = _text(split_by_sample[sample_id].get("split"), "split manifest split")
        if split not in SPLITS or source_by_sample[sample_id].get("sample_id") != sample_id:
            raise ValueError(f"invalid frozen split/source row for sample {sample_id}")

    desired_height, desired_width = (int(value) for value in input_size)
    if desired_height <= 0 or desired_width <= 0:
        raise ValueError("Pin model input_size must be positive")
    records_by_split = {split: 0 for split in SPLITS}
    gt_inventory_by_split = {split: 0 for split in SPLITS}
    negative_records_by_split = {split: 0 for split in SPLITS}
    unique_records: set[str] = set()
    validation: dict[str, dict[str, Any]] = {}
    for record in _iter_jsonl(pin_index_path, "Pin index"):
        record_id = _text(record.get("record_id"), "pin_index.record_id")
        if record_id in unique_records:
            raise ValueError(f"duplicate Pin index record_id: {record_id}")
        unique_records.add(record_id)
        if record.get("binding_sha256") != binding:
            raise ValueError(f"Pin index binding mismatch at {record_id}")
        sample_id = _text(record.get("sample_id"), "pin_index.sample_id")
        split = _text(record.get("split"), "pin_index.split")
        if split not in SPLITS or sample_id not in split_by_sample:
            raise ValueError(f"Pin index split/sample is invalid at {record_id}")
        if split_by_sample[sample_id].get("split") != split:
            raise ValueError(f"Pin index/frozen split mismatch at {record_id}")
        source = source_by_sample[sample_id]
        source_files = _mapping(source.get("files"), "source.files")
        source_image = _mapping(source_files.get("image"), "source.files.image")
        if (
            record.get("source_image") != source_image.get("path")
            or record.get("source_image_sha256") != source_image.get("sha256")
        ):
            raise ValueError(f"Pin index/source image lineage mismatch at {record_id}")
        views = _sequence(record.get("views"), "pin_index.views")
        roi_views = [view for view in views if isinstance(view, Mapping) and view.get("view_kind") == "roi"]
        if len(roi_views) != 1:
            raise ValueError(f"Pin index record must contain exactly one ROI view: {record_id}")
        view = roi_views[0]
        if view.get("sample_id") != sample_id or view.get("split") != split:
            raise ValueError(f"ROI view identity differs from Pin index at {record_id}")
        records_by_split[split] += 1
        if record.get("negative_component") is True:
            negative_records_by_split[split] += 1
        targets = _sequence(view.get("targets"), "ROI targets")
        visible_targets = [target for target in targets if isinstance(target, Mapping) and target.get("visibility") == 2]
        if len(visible_targets) != sum(
            1 for target in targets if isinstance(target, Mapping) and target.get("visibility") == 2
        ):
            raise AssertionError("unreachable target count")
        if any(not isinstance(target, Mapping) or target.get("visibility") not in (0, 2) for target in targets):
            raise ValueError(f"ROI target visibility/schema is invalid at {record_id}")
        gt_inventory_by_split[split] += len(visible_targets)
        if split != "val":
            continue

        output_width, output_height = _pair(view.get("output_size"), "ROI output_size")
        if output_width <= 0 or output_height <= 0:
            raise ValueError(f"ROI output_size is invalid at {record_id}")
        component_original = _box(record.get("component_bbox"), "component_bbox")
        transform = _mapping(view.get("transform"), "ROI transform")
        matrix = transform.get("forward")
        transformed = [
            _transform_point(point, matrix, "ROI transform.forward")
            for point in (
                (component_original[0], component_original[1]),
                (component_original[2], component_original[1]),
                (component_original[2], component_original[3]),
                (component_original[0], component_original[3]),
            )
        ]
        component_input = (
            min(point[0] for point in transformed) * desired_width / output_width,
            min(point[1] for point in transformed) * desired_height / output_height,
            max(point[0] for point in transformed) * desired_width / output_width,
            max(point[1] for point in transformed) * desired_height / output_height,
        )
        if component_input[2] <= component_input[0] or component_input[3] <= component_input[1]:
            raise ValueError(f"transformed component box is invalid at {record_id}")
        component_heatmap = tuple(value / OUTPUT_STRIDE for value in component_input)
        gt: list[dict[str, Any]] = []
        for target_number, target in enumerate(visible_targets):
            descriptor = _mapping(target.get("heatmap"), "target.heatmap")
            old_hm_width, old_hm_height = _pair(descriptor.get("output_size"), "heatmap.output_size")
            point = _pair(descriptor.get("point"), "heatmap.point")
            scaled = (
                point[0] * math.ceil(desired_width / OUTPUT_STRIDE) / old_hm_width,
                point[1] * math.ceil(desired_height / OUTPUT_STRIDE) / old_hm_height,
            )
            orientation_view = _mapping(target.get("orientation_view"), "target.orientation_view")
            orientation = _orientation_group(
                orientation_view.get("unit_vector"),
                f"target[{target_number}].orientation_view.unit_vector",
            )
            original_orientation = _mapping(target.get("orientation_original"), "target.orientation_original")
            declared_orientation = _text(original_orientation.get("label"), "orientation_original.label")
            if declared_orientation in {"left", "right", "up", "down"} and orientation != declared_orientation:
                raise ValueError(f"view/original orientation label mismatch at {record_id}")
            role, detail = _boundary_groups(scaled, component_heatmap)
            gt.append(
                {
                    "point": scaled,
                    "orientation": orientation,
                    "boundary_role": role,
                    "boundary_detail": detail,
                    "pin_id": str(target.get("pin_id", "")),
                }
            )
        area_input = (component_input[2] - component_input[0]) * (component_input[3] - component_input[1])
        validation[record_id] = {
            "record_id": record_id,
            "view_id": _text(view.get("view_id"), "ROI view_id"),
            "sample_id": sample_id,
            "source_domain": _text(source.get("source_class"), "source.source_class"),
            "component_category": _text(record.get("component_category"), "component_category"),
            "component_area_input_px": area_input,
            "component_size_bin": _size_bin(area_input),
            "gt_count_bin": _gt_count_bin(len(gt)),
            "component_box_heatmap": component_heatmap,
            "gt": gt,
            "negative_component": record.get("negative_component") is True,
        }

    expected_val_samples = {
        sample_id for sample_id, row in split_by_sample.items() if row.get("split") == "val"
    }
    observed_val_samples = {row["sample_id"] for row in validation.values()}
    if observed_val_samples != expected_val_samples:
        raise ValueError("Pin validation ROI population does not cover the frozen val samples")
    return validation, {
        "source_sample_count": len(source_by_sample),
        "split_sample_count": len(split_by_sample),
        "index_records_by_split": records_by_split,
        "visible_gt_inventory_by_split": gt_inventory_by_split,
        "negative_component_records_by_split": negative_records_by_split,
        "frozen_val_sample_count": len(expected_val_samples),
    }


def _read_predictions(
    path: Path,
    *,
    validation: Mapping[str, Mapping[str, Any]],
    score_threshold: float,
    max_detections: int,
    input_size: Sequence[int],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    predictions: dict[str, list[dict[str, Any]]] = {}
    seen_views: set[str] = set()
    detections = 0
    hm_height = math.ceil(int(input_size[0]) / OUTPUT_STRIDE)
    hm_width = math.ceil(int(input_size[1]) / OUTPUT_STRIDE)
    for row in _iter_jsonl(path, "Pin predictions"):
        record_id = _text(row.get("record_id"), "prediction.record_id")
        if record_id in predictions:
            raise ValueError(f"duplicate prediction record_id: {record_id}")
        if record_id not in validation:
            raise ValueError(
                "prediction rows must equal the exact frozen-val ROI population; "
                f"unknown record={record_id}"
            )
        expected = validation[record_id]
        view_id = _text(row.get("view_id"), "prediction.view_id")
        if view_id in seen_views:
            raise ValueError(f"duplicate prediction view_id: {view_id}")
        seen_views.add(view_id)
        if (
            view_id != expected["view_id"]
            or row.get("sample_id") != expected["sample_id"]
            or row.get("output_stride") != OUTPUT_STRIDE
        ):
            raise ValueError(f"prediction/index identity mismatch at {record_id}")
        raw_detections = _sequence(row.get("detections_heatmap"), "detections_heatmap")
        if len(raw_detections) > max_detections:
            raise ValueError(f"prediction exceeds max_detections at {record_id}")
        current: list[dict[str, Any]] = []
        last_score = float("inf")
        for detection_number, raw in enumerate(raw_detections):
            detection = _mapping(raw, f"detection[{detection_number}]")
            score = _finite(detection.get("score"), "detection.score")
            x = _finite(detection.get("x"), "detection.x")
            y = _finite(detection.get("y"), "detection.y")
            if not score_threshold <= score <= 1.0 or score > last_score + 1e-12:
                raise ValueError(f"prediction score/order violates frozen decoder at {record_id}")
            if not 0.0 <= x < hm_width or not 0.0 <= y < hm_height:
                raise ValueError(f"prediction lies outside model heatmap at {record_id}")
            last_score = score
            role, detail = _boundary_groups((x, y), expected["component_box_heatmap"])
            current.append(
                {
                    "x": x,
                    "y": y,
                    "score": score,
                    "orientation": _orientation_group(
                        detection.get("orientation"), "detection.orientation"
                    ),
                    "boundary_role": role,
                    "boundary_detail": detail,
                }
            )
        predictions[record_id] = current
        detections += len(current)
    if set(predictions) != set(validation):
        missing = sorted(set(validation) - set(predictions))
        extra = sorted(set(predictions) - set(validation))
        raise ValueError(
            "prediction rows must equal the exact frozen-val ROI population; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return predictions, {
        "prediction_rows": len(predictions),
        "decoded_detections": detections,
        "unique_prediction_views": len(seen_views),
    }


def _confusion_increment(
    table: dict[str, dict[str, int]], truth: str, prediction: str
) -> None:
    table.setdefault(truth, {})[prediction] = table.setdefault(truth, {}).get(prediction, 0) + 1


def _agreement(table: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    total = sum(sum(row.values()) for row in table.values())
    exact = sum(row.get(label, 0) for label, row in table.items())
    return {
        "matched_pair_count": total,
        "exact_group_agreement_count": exact,
        "exact_group_agreement": exact / total if total else None,
        "confusion": {key: dict(sorted(table[key].items())) for key in sorted(table)},
    }


def _score_strata(
    *,
    validation: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    production_matcher: Callable[[Sequence[Mapping[str, Any]], Sequence[tuple[float, float]], float], int],
) -> dict[str, Any]:
    dimension_tables: dict[str, dict[str, dict[str, Any]]] = {
        name: {}
        for name in (
            "source_domain",
            "component_category",
            "component_size_input_px",
            "ground_truth_count_per_view",
            "pin_orientation",
            "boundary_role",
            "boundary_detail",
        )
    }
    sample_totals: dict[str, dict[str, Any]] = {}
    view_rows: list[dict[str, Any]] = []
    overall_matched = overall_pred = overall_gt = 0
    orientation_confusion: dict[str, dict[str, int]] = {}
    boundary_role_confusion: dict[str, dict[str, int]] = {}
    boundary_detail_confusion: dict[str, dict[str, int]] = {}

    for record_id, record in validation.items():
        current_predictions = predictions[record_id]
        gt = record["gt"]
        points = [item["point"] for item in gt]
        pairs = _match_pairs(current_predictions, points)
        production_count = production_matcher(
            current_predictions, points, MATCH_RADIUS_HEATMAP
        )
        if len(pairs) != production_count:
            raise ValueError(f"retained assignment differs from production matcher at {record_id}")
        matched = production_count
        pred_count = len(current_predictions)
        gt_count = len(gt)
        overall_matched += matched
        overall_pred += pred_count
        overall_gt += gt_count
        for dimension, key in (
            ("source_domain", record["source_domain"]),
            ("component_category", record["component_category"]),
            ("component_size_input_px", record["component_size_bin"]),
            ("ground_truth_count_per_view", record["gt_count_bin"]),
        ):
            _accumulate(
                dimension_tables[dimension],
                str(key),
                matched=matched,
                predicted=pred_count,
                truth=gt_count,
                view_id=record["view_id"],
                sample_id=record["sample_id"],
            )
        sample_key = record["sample_id"]
        _accumulate(
            sample_totals,
            sample_key,
            matched=matched,
            predicted=pred_count,
            truth=gt_count,
            view_id=record["view_id"],
            sample_id=sample_key,
        )
        for dimension, truth_key, prediction_key in (
            ("pin_orientation", "orientation", "orientation"),
            ("boundary_role", "boundary_role", "boundary_role"),
            ("boundary_detail", "boundary_detail", "boundary_detail"),
        ):
            attributed = _attributed_counts(
                truth_groups=[str(item[truth_key]) for item in gt],
                prediction_groups=[str(item[prediction_key]) for item in current_predictions],
                pairs=pairs,
            )
            for key, (group_matched, group_pred, group_gt) in attributed.items():
                _accumulate(
                    dimension_tables[dimension],
                    key,
                    matched=group_matched,
                    predicted=group_pred,
                    truth=group_gt,
                    view_id=record["view_id"],
                    sample_id=record["sample_id"],
                )
        for prediction_index, truth_index in pairs:
            prediction = current_predictions[prediction_index]
            truth = gt[truth_index]
            _confusion_increment(
                orientation_confusion, truth["orientation"], prediction["orientation"]
            )
            _confusion_increment(
                boundary_role_confusion,
                truth["boundary_role"],
                prediction["boundary_role"],
            )
            _confusion_increment(
                boundary_detail_confusion,
                truth["boundary_detail"],
                prediction["boundary_detail"],
            )
        view_rows.append(
            {
                "record_id": record_id,
                "view_id": record["view_id"],
                "sample_id": record["sample_id"],
                "source_domain": record["source_domain"],
                "component_category": record["component_category"],
                "component_size_bin": record["component_size_bin"],
                "component_area_input_px": record["component_area_input_px"],
                "gt_count_bin": record["gt_count_bin"],
                **_summary(matched, pred_count, gt_count),
            }
        )

    overall = _summary(overall_matched, overall_pred, overall_gt)
    finalized_dimensions = {
        name: _finalize_table(table) for name, table in dimension_tables.items()
    }
    finalized_samples = _finalize_table(sample_totals)
    domain_by_sample = {
        record["sample_id"]: record["source_domain"] for record in validation.values()
    }
    sample_rows = [
        {"sample_id": sample_id, "source_domain": domain_by_sample[sample_id], **metrics}
        for sample_id, metrics in finalized_samples.items()
    ]
    sample_rows.sort(
        key=lambda row: (
            row["recall_4px"],
            row["f1_4px"],
            row["precision_4px"],
            -row["gt_count"],
            row["sample_id"],
        )
    )
    positive_view_rows = [row for row in view_rows if row["gt_count"] > 0]
    positive_view_rows.sort(
        key=lambda row: (
            row["recall_4px"],
            row["f1_4px"],
            row["precision_4px"],
            -row["gt_count"],
            row["record_id"],
        )
    )
    weak_groups: list[dict[str, Any]] = []
    zero_recall_groups: list[dict[str, Any]] = []
    for dimension, groups in finalized_dimensions.items():
        for group, metrics in groups.items():
            row = {"dimension": dimension, "group": group, **metrics}
            if metrics["gt_count"] > 0 and metrics["matched"] == 0:
                zero_recall_groups.append(row)
            if metrics["gt_count"] >= WEAK_MIN_GT and metrics["recall_4px"] < WEAK_RECALL_CUTOFF:
                weak_groups.append(row)
    ordering = lambda row: (
        row["recall_4px"],
        row["f1_4px"],
        -row["gt_count"],
        row["dimension"],
        row["group"],
    )
    weak_groups.sort(key=ordering)
    zero_recall_groups.sort(key=ordering)
    zero_recall_views = [row for row in positive_view_rows if row["matched"] == 0]
    zero_recall_samples = [
        row for row in sample_rows if row["gt_count"] > 0 and row["matched"] == 0
    ]
    return {
        "overall": overall,
        "strata": finalized_dimensions,
        "matched_pair_diagnostics": {
            "pin_orientation": _agreement(orientation_confusion),
            "boundary_role": _agreement(boundary_role_confusion),
            "boundary_detail": _agreement(boundary_detail_confusion),
        },
        "weakness_audit": {
            "weak_group_definition": {
                "recall_strictly_below": WEAK_RECALL_CUTOFF,
                "minimum_gt_count": WEAK_MIN_GT,
                "selection_role": "fixed diagnostic definition; not a tuned model threshold",
            },
            "zero_recall_groups": zero_recall_groups,
            "weak_groups": weak_groups,
            "validation_samples_ranked_weakest_first": sample_rows,
            "zero_recall_sample_count": len(zero_recall_samples),
            "zero_recall_samples": zero_recall_samples,
            "positive_gt_view_count": len(positive_view_rows),
            "zero_recall_view_count": len(zero_recall_views),
            "zero_recall_views_preview_limit": 100,
            "zero_recall_views_preview": zero_recall_views[:100],
            "weakest_positive_gt_views_preview_limit": 100,
            "weakest_positive_gt_views_preview": positive_view_rows[:100],
        },
    }


def build_audit(
    *,
    workspace: Path,
    run_root: Path,
    config_path: Path,
    materialization_summary_path: Path,
    pin_index_path: Path,
    source_manifest_path: Path,
    split_manifest_path: Path,
    task_manifest_path: Path,
    derived_summary_path: Path,
    component_manifest_path: Path,
    checkpoint_inspector: Callable[[Path, Path], Mapping[str, Any]] = _inspect_checkpoint,
    config_inspector: Callable[[Path, Path], Mapping[str, Any]] = _inspect_config,
) -> dict[str, Any]:
    paths = {
        "run_status": run_root / "torch_run_status.json",
        "bundle_manifest": run_root / "bundle.manifest.json",
        "model_card": run_root / "model_card.json",
        "metrics": run_root / "metrics.json",
        "model_pt": run_root / "model.pt",
        "model_onnx": run_root / "model.onnx",
        "predictions": run_root / "predictions.jsonl",
        "training_config": config_path,
        "materialization_summary": materialization_summary_path,
        "pin_index": pin_index_path,
        "source_manifest": source_manifest_path,
        "split_manifest": split_manifest_path,
        "task_manifest": task_manifest_path,
        "derived_summary": derived_summary_path,
        "component_manifest": component_manifest_path,
    }
    for key, path in paths.items():
        _resolve_within(workspace, path, key)
    hashes = {key: _sha256(path) for key, path in paths.items()}
    summary = _read_object(materialization_summary_path, "materialization summary")
    config = dict(config_inspector(workspace, config_path))
    checkpoint = dict(checkpoint_inspector(workspace, paths["model_pt"]))
    status, bundle, _card, _metrics_file = _verify_status_and_bundle(
        workspace=workspace,
        run_root=run_root,
        paths=paths,
        hashes=hashes,
        config=config,
        checkpoint=checkpoint,
        summary=summary,
    )
    protocol = _mapping(config.get("protocol"), "config.protocol")
    validation, inventory = _prepare_validation_population(
        pin_index_path=pin_index_path,
        source_manifest_path=source_manifest_path,
        split_manifest_path=split_manifest_path,
        binding=_text(summary.get("binding_sha256"), "summary.binding_sha256"),
        input_size=_sequence(protocol.get("input_size"), "protocol.input_size"),
    )
    predictions, prediction_inventory = _read_predictions(
        paths["predictions"],
        validation=validation,
        score_threshold=_finite(protocol.get("score_threshold"), "score threshold"),
        max_detections=int(protocol.get("max_detections_per_view")),
        input_size=_sequence(protocol.get("input_size"), "protocol.input_size"),
    )
    sys.path.insert(0, str(workspace / "src"))
    from pcbparse.training.torch_pin_model import _maximum_threshold_matches

    scored = _score_strata(
        validation=validation,
        predictions=predictions,
        production_matcher=_maximum_threshold_matches,
    )
    status_metrics = _mapping(status.get("metrics"), "status.metrics")
    overall = scored["overall"]
    reconciliation = {
        "gt_count": overall["gt_count"] == int(round(float(status_metrics.get("gt_count", -1)))),
        "pred_count": overall["pred_count"] == int(round(float(status_metrics.get("pred_count", -1)))),
        "pin_precision_4px": _close(overall["precision_4px"], status_metrics.get("pin_precision_4px")),
        "pin_recall_4px": _close(overall["recall_4px"], status_metrics.get("pin_recall_4px")),
        "pin_f1_4px": _close(overall["f1_4px"], status_metrics.get("pin_f1_4px")),
        "pred_to_gt_ratio": _close(overall["pred_to_gt_ratio"], status_metrics.get("pred_to_gt_ratio")),
    }
    if not all(reconciliation.values()):
        raise ValueError(f"frozen prediction replay differs from formal metrics: {reconciliation}")
    if overall["gt_count"] != inventory["visible_gt_inventory_by_split"]["val"]:
        raise ValueError("scored validation GT count differs from Pin-index inventory")

    script_path = Path(__file__).resolve()
    evaluator_path = workspace / "src/pcbparse/training/torch_pin_model.py"
    source_split_counts = {
        split: sum(
            1
            for row in _iter_jsonl(split_manifest_path, "split manifest recount")
            if row.get("split") == split
        )
        for split in SPLITS
    }
    if sum(source_split_counts.values()) != inventory["split_sample_count"]:
        raise ValueError("frozen split recount differs from split-manifest population")
    inputs = {
        key: {
            "path": _relative(path, workspace),
            "bytes": path.stat().st_size,
            "sha256": hashes[key],
        }
        for key, path in paths.items()
    }
    lineage = _mapping(status.get("lineage"), "status.lineage")
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pcbparse_pin_frozen_val_stratified_audit",
        "task": "pin",
        "evaluation_provenance": {
            "kind": "completed_formal_run_frozen_prediction_replay",
            "source_status": status.get("status"),
            "source_finished_at_utc": status.get("finished_at_utc"),
            "source_best_epoch": status.get("best_epoch"),
            "source_completed_epochs": status.get("completed_epochs"),
            "source_planned_epochs": status.get("planned_epochs"),
            "source_early_stopped": status.get("early_stopped"),
            "predictions_are_current_candidate_artifact": True,
            "model_forward_executed_by_audit": False,
            "image_decode_executed_by_audit": False,
            "checkpoint_strict_loaded_on_cpu": True,
            "gpu_used_by_audit": False,
            "threshold_search_executed": False,
            "nms_search_executed": False,
            "configuration_tuned": False,
            "training_process_interaction": False,
            "models_lock_read_by_audit": False,
            "models_lock_modified_by_audit": False,
        },
        "protocol": {
            **dict(protocol),
            "matching_radius_input_pixels": MATCH_RADIUS_INPUT_PX,
            "matching_radius_heatmap_cells": MATCH_RADIUS_HEATMAP,
            "matching": (
                "production maximum-cardinality one-to-one Euclidean matcher; "
                "retained deterministic assignments assert equal cardinality per view"
            ),
            "boundary_tolerance_heatmap_cells": BOUNDARY_TOLERANCE_HEATMAP,
            "component_size_definition": (
                "area of the status/config bound component rectangle after ROI-to-"
                "384x384 model-input scaling; COCO-style 32^2/96^2 bins"
            ),
            "gt_count_bins": [name for name, _minimum, _maximum in GT_COUNT_BINS],
            "target_level_attribution": (
                "matched prediction and TP use the matched GT group; unmatched GT "
                "uses its GT group; unmatched prediction uses its predicted group"
            ),
        },
        "selection_boundary": {
            "split": "val",
            "manifest_only_index_records_inspected_by_split": inventory["index_records_by_split"],
            "source_samples_by_split": source_split_counts,
            "prediction_rows_scored_by_split": {
                "train": 0,
                "val": prediction_inventory["prediction_rows"],
                "test": 0,
            },
            "decoded_predictions_by_split": {
                "train": 0,
                "val": prediction_inventory["decoded_detections"],
                "test": 0,
            },
            "scored_gt_pins_by_split": {
                "train": 0,
                "val": overall["gt_count"],
                "test": 0,
            },
            "decoded_images_by_split": {"train": 0, "val": 0, "test": 0},
            "model_forwards_by_split": {"train": 0, "val": 0, "test": 0},
            "test_split_used_for_tuning": False,
            "train_split_used_for_tuning": False,
            "prediction_population_exactly_equals_val_roi_index": True,
        },
        "coverage": {
            **inventory,
            **prediction_inventory,
            "scored_validation_records": len(validation),
            "scored_validation_unique_samples": len({row["sample_id"] for row in validation.values()}),
        },
        "overall": overall,
        "formal_metric_reconciliation": {
            "all_equal": all(reconciliation.values()),
            "checks": reconciliation,
            "formal_status_metrics": dict(status_metrics),
        },
        "strata": scored["strata"],
        "matched_pair_diagnostics": scored["matched_pair_diagnostics"],
        "weakness_audit": scored["weakness_audit"],
        "hash_chain": {
            "all_checked_links_passed": True,
            "formal_candidate_result_sha256_reproduced": status.get("candidate_result_sha256"),
            "materialization_binding_sha256": summary.get("binding_sha256"),
            "materialization_commit_sha256": hashes["materialization_summary"],
            "checkpoint_lineage_sha256": checkpoint.get("lineage_sha256"),
            "run_training_config_sha256": lineage.get("training_config_sha256"),
            "current_training_config_canonical_sha256": config.get("canonical_sha256"),
            "current_training_config_resume_compatible_sha256s": config.get("resume_compatible_sha256s"),
            "checkpoint_binding_note": (
                "The original formal status directly binds ONNX and predictions, "
                "not model.pt. This audit newly binds model.pt bytes and strictly "
                "verifies its payload lineage/epoch/metrics against that status."
            ),
            "inputs": inputs,
            "audit_script": {
                "path": _relative(script_path, workspace),
                "sha256": _sha256(script_path),
            },
            "production_evaluator_source": {
                "path": _relative(evaluator_path, workspace),
                "sha256": _sha256(evaluator_path),
                "matcher": "_maximum_threshold_matches",
            },
        },
        "metric_authority": {
            "reported_metric": "repository-local anonymous Pin detection P/R/F1 at 4 input pixels",
            "official_dataset300_pin_f1_verified": False,
            "official_evaluator_invoked": False,
            "official_evaluator_artifact_present": False,
            "boundary": (
                "This reproduces the repository formal validator on a frozen, "
                "hash-bound validation population. It is not the competition's "
                "official PinF1 and cannot close that release subgate."
            ),
        },
        "release_subgates": {
            "formal_artifact_hash_chain": "pass",
            "frozen_val_population_boundary": "pass",
            "repository_metric_reproduction": "pass",
            "stratified_diagnostics": "pass",
            "official_dataset300_pin_f1": "open_official_evaluator_required",
            "models_lock_blessing": "not_attempted_not_authorized",
        },
        "runtime": {
            "implementation": platform.python_implementation(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device_contract": "CPU checkpoint load plus JSON replay only",
            "cuda_visible_devices_for_checkpoint_load": "-1",
        },
    }
    report["audit_payload_sha256"] = _canonical_sha256(report)
    return report


def _atomic_write(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    workspace = arguments.workspace_root.expanduser().resolve(strict=True)
    run_root = _resolve_within(
        workspace, arguments.run_root, "Pin run root", kind="directory"
    )
    resolved = {
        name: _resolve_within(workspace, getattr(arguments, name), name)
        for name in (
            "config",
            "materialization_summary",
            "pin_index",
            "source_manifest",
            "split_manifest",
            "task_manifest",
            "derived_summary",
            "component_manifest",
        )
    }
    output = _resolve_within(
        workspace, arguments.output, "audit output", must_exist=False
    )
    report = build_audit(
        workspace=workspace,
        run_root=run_root,
        config_path=resolved["config"],
        materialization_summary_path=resolved["materialization_summary"],
        pin_index_path=resolved["pin_index"],
        source_manifest_path=resolved["source_manifest"],
        split_manifest_path=resolved["split_manifest"],
        task_manifest_path=resolved["task_manifest"],
        derived_summary_path=resolved["derived_summary"],
        component_manifest_path=resolved["component_manifest"],
    )
    _atomic_write(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "file_sha256": _sha256(output),
                "audit_payload_sha256": report["audit_payload_sha256"],
                "overall": report["overall"],
                "train_decoded_predictions": report["selection_boundary"]["decoded_predictions_by_split"]["train"],
                "test_decoded_predictions": report["selection_boundary"]["decoded_predictions_by_split"]["test"],
                "official_dataset300_pin_f1": report["release_subgates"]["official_dataset300_pin_f1"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
