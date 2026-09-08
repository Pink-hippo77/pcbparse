from __future__ import annotations

"""Decompose frozen-val Wire/Junction/physical-graph topology errors.

The audit is deliberately diagnostic-only.  It evaluates pairwise pin
co-connectivity on the frozen validation split, never reads test labels for a
metric, never searches a threshold, and never changes a model or training
configuration.  Saved formal Wire masks and formal Junction patch predictions
are used only where their populations are genuinely comparable.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pcbparse.graph.physical import build_physical_graph  # noqa: E402
from pcbparse.io import load_image  # noqa: E402
from pcbparse.ir.models import CircuitIR  # noqa: E402
from pcbparse.perception.golden_provider import DatasetGoldenProvider  # noqa: E402
from pcbparse.perception.torch_junction_adapter import TorchJunctionProvider  # noqa: E402
from pcbparse.training.torch_wire_junction import JUNCTION_CLASSES  # noqa: E402


AUDIT_VERSION = "val-graph-oracle-decomposition-v1"
MATERIALIZATION_RUN_ID = (
    "5233aeb838db659970f0ce73480753a2f4a4e79366612878986d3bf18bca610e"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--source-root", type=Path, default=Path("Dataset300/Dataset300")
    )
    parser.add_argument(
        "--source-manifest", type=Path, default=Path("data/manifests/dataset300.jsonl")
    )
    parser.add_argument(
        "--split-manifest", type=Path, default=Path("data/splits/dataset300_split.jsonl")
    )
    parser.add_argument(
        "--wire-manifest",
        type=Path,
        default=Path("data/derived/dataset300-v2/wire_paths.jsonl"),
    )
    parser.add_argument(
        "--junction-manifest",
        type=Path,
        default=Path("data/derived/dataset300-v2/junction_patches.jsonl"),
    )
    parser.add_argument(
        "--materialized-run",
        type=Path,
        default=Path(
            "artifacts/training/materialized/dataset300-v2/wire-junction/runs/"
            + MATERIALIZATION_RUN_ID
        ),
    )
    parser.add_argument(
        "--wire-predictions",
        type=Path,
        default=Path("work/train-wire-formal4/predictions.jsonl"),
    )
    parser.add_argument(
        "--wire-run-status",
        type=Path,
        default=Path("work/train-wire-formal4/torch_run_status.json"),
    )
    parser.add_argument(
        "--wire-model", type=Path, default=Path("work/train-wire-formal4/model.pt")
    )
    parser.add_argument(
        "--wire-audit",
        type=Path,
        default=Path("artifacts/audits/wire_formal4_val_stratified.json"),
    )
    parser.add_argument(
        "--junction-predictions",
        type=Path,
        default=Path("work/train-junction-formal/predictions.jsonl"),
    )
    parser.add_argument(
        "--junction-run-status",
        type=Path,
        default=Path("work/train-junction-formal/torch_run_status.json"),
    )
    parser.add_argument(
        "--junction-formal-model",
        type=Path,
        default=Path("work/train-junction-formal/model.onnx"),
    )
    parser.add_argument(
        "--junction-work-checkpoint",
        type=Path,
        default=Path("work/train-junction-formal/model.pt"),
    )
    parser.add_argument(
        "--junction-truth-audit",
        type=Path,
        default=Path("artifacts/audits/junction_truth_audit.json"),
    )
    parser.add_argument("--models-lock", type=Path, default=Path("models/models.lock.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/val_graph_oracle_decomposition.json"),
    )
    return parser


def _resolve(root: Path, raw: Path, label: str, *, directory: bool = False) -> Path:
    candidate = raw.expanduser()
    path = (candidate if candidate.is_absolute() else root / candidate).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside workspace: {path}") from exc
    if directory:
        if not path.is_dir():
            raise FileNotFoundError(f"{label} directory is missing: {path}")
    elif not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"{label} is missing or empty: {path}")
    return path


def _resolve_output(root: Path, raw: Path) -> Path:
    candidate = raw.expanduser()
    path = (candidate if candidate.is_absolute() else root / candidate).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"output must stay inside workspace: {path}") from exc
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _loads(raw: str, label: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {label} JSON: {exc}") from exc


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = _loads(path.read_text(encoding="utf-8"), label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _iter_jsonl(path: Path, label: str) -> Iterable[dict[str, Any]]:
    found = False
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            found = True
            value = _loads(raw, f"{label}:{line_number}")
            if not isinstance(value, dict):
                raise ValueError(f"{label}:{line_number} row must be an object")
            yield value
    if not found:
        raise ValueError(f"{label} is empty: {path}")


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an array")
    return value


def _expect(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected={expected!r}, actual={actual!r}")


def _expect_close(actual: Any, expected: Any, label: str, tolerance: float = 1e-12) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        raise ValueError(f"{label} expected value must be numeric")
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{label} mismatch: expected={expected!r}, actual={actual!r}")


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _index(rows: Iterable[Mapping[str, Any]], key: str, label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        identity = _string(row.get(key), f"{label}.{key}")
        if identity in result:
            raise ValueError(f"duplicate {label} {key}: {identity}")
        result[identity] = row
    return result


def _payload_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("audit_payload_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _verify_payload_hash(value: Mapping[str, Any], label: str) -> None:
    expected = _string(value.get("audit_payload_sha256"), f"{label}.audit_payload_sha256")
    _expect(_payload_sha256(value), expected, f"{label} payload SHA-256")


def _source(path: Path, root: Path, *, expected_sha256: Any = None) -> dict[str, Any]:
    actual = _sha256(path)
    if expected_sha256 is not None:
        _expect(actual, expected_sha256, f"{path.name} SHA-256")
    return {
        "path": _relative(path, root),
        "sha256": actual,
        "verified": True,
    }


def _ratio(numerator: int, denominator: int, *, opposite_total: int) -> float:
    if denominator:
        return float(numerator / denominator)
    return 1.0 if opposite_total == 0 else 0.0


def _pin_pairs(groups: Iterable[Iterable[str]]) -> set[tuple[str, str]]:
    """Return the unordered pairwise co-connectivity relation."""

    pairs: set[tuple[str, str]] = set()
    for raw_group in groups:
        group = sorted(set(raw_group))
        for left, right in combinations(group, 2):
            pairs.add((left, right))
    return pairs


def _score_pair_partition(
    reference: set[tuple[str, str]], predicted: set[tuple[str, str]]
) -> dict[str, Any]:
    tp = len(reference & predicted)
    fp = len(predicted - reference)
    fn = len(reference - predicted)
    precision = _ratio(tp, tp + fp, opposite_total=tp + fn)
    recall = _ratio(tp, tp + fn, opposite_total=tp + fp)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "reference_pairs": len(reference),
        "predicted_pairs": len(predicted),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_pair_relation": reference == predicted,
    }


def _strip_oracle_membership(ir: CircuitIR) -> CircuitIR:
    """Keep GT geometry/points while removing hidden source membership hints."""

    result = ir.model_copy(deep=True)
    result.junctions = []
    result.physical_nets = []
    result.electrical_nets = []
    for segment in result.segments:
        segment.source_edge_id = None
        segment.source_polyline_id = None
        segment.evidence = []
    for pin in result.pins:
        pin.edge_ids = []
        pin.evidence = []
    return result


def _evaluate_ir(
    ir: CircuitIR,
    reference_groups: Iterable[Iterable[str]],
    *,
    builder_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidate = ir.model_copy(deep=True)
    nets, decisions = build_physical_graph(candidate, **dict(builder_kwargs or {}))
    reference_pairs = _pin_pairs(reference_groups)
    predicted_pairs = _pin_pairs(net.pins for net in nets)
    result = _score_pair_partition(reference_pairs, predicted_pairs)
    result.update(
        {
            "physical_net_count": len(nets),
            "pin_bearing_net_count": sum(bool(net.pins) for net in nets),
            "segment_only_net_count": sum(bool(net.segments) and not net.pins for net in nets),
            "snap_status_counts": dict(sorted(Counter(item.status for item in decisions).items())),
        }
    )
    return result


def _aggregate_variant(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    reference = sum(int(row["reference_pairs"]) for row in rows)
    predicted = sum(int(row["predicted_pairs"]) for row in rows)
    precision = _ratio(tp, tp + fp, opposite_total=tp + fn)
    recall = _ratio(tp, tp + fn, opposite_total=tp + fp)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    snap_counts: Counter[str] = Counter()
    for row in rows:
        snap_counts.update(_mapping(row["snap_status_counts"], "snap_status_counts"))
    return {
        "samples": len(rows),
        "reference_pairs": reference,
        "predicted_pairs": predicted,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_pair_relation_samples": sum(bool(row["exact_pair_relation"]) for row in rows),
        "physical_net_count": sum(int(row["physical_net_count"]) for row in rows),
        "pin_bearing_net_count": sum(int(row["pin_bearing_net_count"]) for row in rows),
        "segment_only_net_count": sum(int(row["segment_only_net_count"]) for row in rows),
        "snap_status_counts": dict(sorted(snap_counts.items())),
    }


def _classification_metrics(
    predictions: Sequence[Mapping[str, Any]], classes: Sequence[str]
) -> dict[str, Any]:
    confusion: dict[str, Counter[str]] = {name: Counter() for name in classes}
    correct = 0
    for row in predictions:
        truth = _string(row.get("truth"), "junction prediction truth")
        prediction = _string(row.get("prediction"), "junction prediction class")
        if truth not in confusion or prediction not in confusion:
            raise ValueError(f"unexpected junction class truth={truth!r}, prediction={prediction!r}")
        confusion[truth][prediction] += 1
        correct += truth == prediction
    metrics: dict[str, float] = {}
    observed_f1: list[float] = []
    for name in classes:
        tp = confusion[name][name]
        fp = sum(confusion[truth][name] for truth in classes if truth != name)
        fn = sum(confusion[name][prediction] for prediction in classes if prediction != name)
        support = sum(confusion[name].values())
        precision = float(tp / (tp + fp)) if tp + fp else 0.0
        recall = float(tp / (tp + fn)) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[f"support/{name}"] = float(support)
        metrics[f"precision/{name}"] = precision
        metrics[f"recall/{name}"] = recall
        metrics[f"f1/{name}"] = f1
        if support:
            observed_f1.append(f1)
    metrics["accuracy"] = float(correct / len(predictions)) if predictions else 0.0
    metrics["macro_f1"] = sum(metrics[f"f1/{name}"] for name in classes) / len(classes)
    metrics["macro_f1_observed"] = sum(observed_f1) / len(observed_f1) if observed_f1 else 0.0
    metrics["observed_class_count"] = float(len(observed_f1))
    return {
        "metrics": metrics,
        "confusion": {
            truth: {prediction: int(confusion[truth][prediction]) for prediction in classes}
            for truth in classes
        },
    }


def _model_entry(models_lock: Mapping[str, Any], task: str) -> Mapping[str, Any]:
    entries = _sequence(models_lock.get("models"), "models.lock models")
    matched = [item for item in entries if isinstance(item, Mapping) and item.get("task") == task]
    if len(matched) != 1:
        raise ValueError(f"models.lock must contain exactly one {task!r} entry")
    return matched[0]


def _resolve_locked_artifact(
    workspace: Path,
    models_lock_path: Path,
    entry: Mapping[str, Any],
    label: str,
) -> Path:
    raw = Path(_string(entry.get("path"), f"{label}.path"))
    path = (models_lock_path.parent / raw).resolve(strict=False)
    try:
        path.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"{label} artifact escapes workspace: {path}") from exc
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"{label} artifact missing: {path}")
    _expect(_sha256(path), entry.get("sha256"), f"{label} locked artifact SHA-256")
    return path


def _validate_source_file(
    source_root: Path,
    metadata: Mapping[str, Any],
    label: str,
) -> tuple[Path, str]:
    raw = Path(_string(metadata.get("path"), f"{label}.path"))
    path = (source_root / raw).resolve(strict=False)
    try:
        path.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes source root: {path}") from exc
    actual = _sha256(path)
    _expect(actual, metadata.get("sha256"), f"{label} SHA-256")
    return path, actual


def _constructability_matrix(
    *,
    val_samples: int,
    wire_covered_samples: int,
    eligible_wire_tiles: int,
    ineligible_wire_tiles: int,
    trusted_junction_rows: int,
    unresolved_junction_rows: int,
) -> list[dict[str, Any]]:
    return [
        {
            "combination": "gt_source_membership + gt_pin_points + current_graph_builder",
            "constructable": True,
            "role": "builder upper-bound diagnostic",
        },
        {
            "combination": "gt_wire_geometry + gt_pin_points + no_junction + current_graph_builder",
            "constructable": True,
            "role": "geometry/association contract diagnostic",
        },
        {
            "combination": "gt_wire_geometry + gt_pin_points + locked_junction_checkpoint_on_production_candidates + current_graph_builder",
            "constructable": True,
            "role": "conditional Junction integration replay; not predicted-Wire E2E",
        },
        {
            "combination": "saved_formal_wire_masks + current_graph_builder on complete frozen val",
            "constructable": False,
            "reason": (
                f"saved masks cover {wire_covered_samples}/{val_samples} val samples and "
                f"{eligible_wire_tiles} eligible tiles; {ineligible_wire_tiles} val tiles have no "
                "comparable prediction, so a complete graph would require new inference"
            ),
        },
        {
            "combination": "saved_formal_junction_predictions + predicted-Wire graph",
            "constructable": False,
            "reason": (
                f"the {trusted_junction_rows} saved predictions are GT-derived patch record IDs "
                "without production predicted-Wire segment membership"
            ),
        },
        {
            "combination": "full_gt_wire + full_gt_junction + current_graph_builder",
            "constructable": False,
            "reason": (
                f"{unresolved_junction_rows} frozen-val junction candidates remain UNKNOWN/needs_review; "
                "trusted crossing-class support is incomplete"
            ),
        },
    ]


def build_audit(
    *,
    workspace: Path,
    source_root: Path,
    source_manifest_path: Path,
    split_manifest_path: Path,
    wire_manifest_path: Path,
    junction_manifest_path: Path,
    materialized_run: Path,
    wire_predictions_path: Path,
    wire_status_path: Path,
    wire_model_path: Path,
    wire_audit_path: Path,
    junction_predictions_path: Path,
    junction_status_path: Path,
    junction_formal_model_path: Path,
    junction_work_checkpoint_path: Path,
    junction_truth_audit_path: Path,
    models_lock_path: Path,
) -> dict[str, Any]:
    summary_path = materialized_run / "summary.json"
    wire_tiles_path = materialized_run / "wire_tiles.jsonl"
    junction_annotations_path = materialized_run / "junction_annotations.jsonl"
    for path, label in (
        (summary_path, "materialization summary"),
        (wire_tiles_path, "wire tiles"),
        (junction_annotations_path, "junction annotations"),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"{label} missing: {path}")

    hashes = {
        "source_manifest": _sha256(source_manifest_path),
        "split_manifest": _sha256(split_manifest_path),
        "wire_manifest": _sha256(wire_manifest_path),
        "junction_manifest": _sha256(junction_manifest_path),
        "summary": _sha256(summary_path),
        "wire_tiles": _sha256(wire_tiles_path),
        "junction_annotations": _sha256(junction_annotations_path),
        "wire_predictions": _sha256(wire_predictions_path),
        "wire_status": _sha256(wire_status_path),
        "wire_model": _sha256(wire_model_path),
        "wire_audit": _sha256(wire_audit_path),
        "junction_predictions": _sha256(junction_predictions_path),
        "junction_status": _sha256(junction_status_path),
        "junction_formal_model": _sha256(junction_formal_model_path),
        "junction_work_checkpoint": _sha256(junction_work_checkpoint_path),
        "junction_truth_audit": _sha256(junction_truth_audit_path),
        "models_lock": _sha256(models_lock_path),
    }

    summary = _read_object(summary_path, "materialization summary")
    _expect(summary.get("kind"), "wire_junction_materialization", "materialization kind")
    _expect(summary.get("complete"), True, "materialization complete")
    _expect(summary.get("run_id"), materialized_run.name, "materialization run_id")
    lineage_inputs = _mapping(
        _mapping(summary.get("lineage"), "materialization lineage").get("inputs"),
        "materialization lineage inputs",
    )
    for key, hash_key in (
        ("source_manifest_sha256", "source_manifest"),
        ("split_manifest_sha256", "split_manifest"),
        ("wire_manifest_sha256", "wire_manifest"),
        ("junction_manifest_sha256", "junction_manifest"),
    ):
        _expect(lineage_inputs.get(key), hashes[hash_key], f"materialization {key}")
    outputs = _mapping(summary.get("outputs"), "materialization outputs")
    _expect(
        _mapping(outputs.get("wire_tiles"), "summary wire_tiles").get("sha256"),
        hashes["wire_tiles"],
        "summary wire_tiles SHA-256",
    )
    _expect(
        _mapping(outputs.get("junction_annotations"), "summary junction_annotations").get(
            "sha256"
        ),
        hashes["junction_annotations"],
        "summary junction_annotations SHA-256",
    )

    split_rows = list(_iter_jsonl(split_manifest_path, "split manifest"))
    split_index = _index(split_rows, "sample_id", "split manifest")
    split_by_sample = {
        sample_id: _string(row.get("split"), f"split {sample_id}")
        for sample_id, row in split_index.items()
    }
    val_sample_ids = sorted(
        sample_id for sample_id, split in split_by_sample.items() if split == "val"
    )
    if not val_sample_ids:
        raise ValueError("frozen validation split is empty")
    val_sample_set = set(val_sample_ids)

    source_rows = list(_iter_jsonl(source_manifest_path, "source manifest"))
    source_index = _index(source_rows, "sample_id", "source manifest")
    missing_sources = sorted(val_sample_set - set(source_index))
    if missing_sources:
        raise ValueError(f"val samples missing from source manifest: {missing_sources}")

    models_lock = _read_object(models_lock_path, "models.lock")
    wire_lock = _model_entry(models_lock, "wire")
    junction_lock = _model_entry(models_lock, "junction")
    locked_wire_path = _resolve_locked_artifact(workspace, models_lock_path, wire_lock, "wire")
    locked_junction_path = _resolve_locked_artifact(
        workspace, models_lock_path, junction_lock, "junction"
    )
    _expect(hashes["wire_model"], wire_lock.get("sha256"), "formal Wire/model.lock SHA-256")
    _expect(
        hashes["junction_work_checkpoint"],
        junction_lock.get("sha256"),
        "formal Junction checkpoint/model.lock SHA-256",
    )

    wire_status = _read_object(wire_status_path, "Wire run status")
    _expect(wire_status.get("task"), "wire", "Wire run task")
    _expect(wire_status.get("formal_evidence"), True, "Wire formal_evidence")
    _expect(wire_status.get("smoke"), False, "Wire smoke")
    _expect(
        _mapping(wire_status.get("artifact"), "Wire artifact").get("sha256"),
        hashes["wire_model"],
        "Wire artifact SHA-256",
    )
    _expect(
        _mapping(wire_status.get("predictions"), "Wire predictions").get("sha256"),
        hashes["wire_predictions"],
        "Wire predictions SHA-256",
    )
    wire_lineage = _mapping(wire_status.get("lineage"), "Wire lineage")
    for key, actual in (
        ("binding_sha256", materialized_run.name),
        ("materialization_commit_id", materialized_run.name),
        ("materialization_commit_sha256", hashes["summary"]),
        ("source_manifest_sha256", hashes["source_manifest"]),
        ("split_manifest_sha256", hashes["split_manifest"]),
        ("task_manifest_sha256", hashes["wire_manifest"]),
    ):
        _expect(wire_lineage.get(key), actual, f"Wire lineage {key}")

    wire_audit = _read_object(wire_audit_path, "Wire stratified audit")
    _verify_payload_hash(wire_audit, "Wire stratified audit")
    _expect(
        _mapping(_mapping(wire_audit.get("sources"), "Wire audit sources").get("predictions"),
                 "Wire audit predictions source").get("sha256"),
        hashes["wire_predictions"],
        "Wire audit prediction SHA-256",
    )

    tile_by_id: dict[str, Mapping[str, Any]] = {}
    eligible_val_ids: set[str] = set()
    val_tile_ids: set[str] = set()
    eligible_val_samples: set[str] = set()
    for row in _iter_jsonl(wire_tiles_path, "wire tiles"):
        record_id = _string(row.get("record_id"), "wire tile record_id")
        if record_id in tile_by_id:
            raise ValueError(f"duplicate wire tile record_id: {record_id}")
        tile_by_id[record_id] = row
        sample_id = _string(row.get("sample_id"), f"wire tile {record_id} sample_id")
        split = _string(row.get("split"), f"wire tile {record_id} split")
        _expect(split_by_sample.get(sample_id), split, f"wire tile {record_id} split binding")
        if split == "val":
            val_tile_ids.add(record_id)
            supervision = _mapping(row.get("supervision"), f"wire tile {record_id} supervision")
            if supervision.get("eligible") is True:
                eligible_val_ids.add(record_id)
                eligible_val_samples.add(sample_id)
    wire_prediction_rows = list(_iter_jsonl(wire_predictions_path, "Wire predictions"))
    wire_prediction_index = _index(wire_prediction_rows, "record_id", "Wire predictions")
    _expect(set(wire_prediction_index), eligible_val_ids, "Wire prediction frozen-val population")
    for record_id, row in wire_prediction_index.items():
        expected_sample = tile_by_id[record_id].get("sample_id")
        _expect(row.get("sample_id"), expected_sample, f"Wire prediction {record_id} sample_id")
    wire_audit_boundary = _mapping(
        wire_audit.get("selection_boundary"), "Wire audit selection_boundary"
    )
    _expect(
        wire_audit_boundary.get("val_eligible_tile_rows_used"),
        len(eligible_val_ids),
        "Wire audit eligible val count",
    )
    _expect(wire_audit_boundary.get("test_rows_used"), 0, "Wire audit test rows")
    _expect(
        wire_audit_boundary.get("train_rows_used_for_metrics"),
        0,
        "Wire audit train metric rows",
    )

    junction_status = _read_object(junction_status_path, "Junction run status")
    _expect(junction_status.get("task"), "junction", "Junction run task")
    _expect(junction_status.get("formal_evidence"), True, "Junction formal_evidence")
    _expect(junction_status.get("smoke"), False, "Junction smoke")
    _expect(
        _mapping(junction_status.get("artifact"), "Junction artifact").get("sha256"),
        hashes["junction_formal_model"],
        "Junction formal artifact SHA-256",
    )
    _expect(
        _mapping(junction_status.get("predictions"), "Junction predictions").get("sha256"),
        hashes["junction_predictions"],
        "Junction predictions SHA-256",
    )
    junction_lineage = _mapping(junction_status.get("lineage"), "Junction lineage")
    for key, actual in (
        ("binding_sha256", materialized_run.name),
        ("materialization_commit_id", materialized_run.name),
        ("materialization_commit_sha256", hashes["summary"]),
        ("source_manifest_sha256", hashes["source_manifest"]),
        ("split_manifest_sha256", hashes["split_manifest"]),
        ("task_manifest_sha256", hashes["junction_manifest"]),
    ):
        _expect(junction_lineage.get(key), actual, f"Junction lineage {key}")

    annotation_by_id: dict[str, Mapping[str, Any]] = {}
    val_annotations: list[Mapping[str, Any]] = []
    for row in _iter_jsonl(junction_annotations_path, "junction annotations"):
        record_id = _string(row.get("record_id"), "junction annotation record_id")
        if record_id in annotation_by_id:
            raise ValueError(f"duplicate junction annotation record_id: {record_id}")
        annotation_by_id[record_id] = row
        sample_id = _string(row.get("sample_id"), f"junction annotation {record_id} sample_id")
        split = _string(row.get("split"), f"junction annotation {record_id} split")
        _expect(split_by_sample.get(sample_id), split, f"junction annotation {record_id} split")
        if split == "val":
            val_annotations.append(row)
    trusted_val_annotations = {
        _string(row.get("record_id"), "trusted junction record_id"): row
        for row in val_annotations
        if row.get("label_status") == "trusted_gt"
    }
    unresolved_val_annotations = [
        row for row in val_annotations if row.get("label_status") != "trusted_gt"
    ]
    junction_prediction_rows = list(
        _iter_jsonl(junction_predictions_path, "Junction predictions")
    )
    junction_prediction_index = _index(
        junction_prediction_rows, "record_id", "Junction predictions"
    )
    _expect(
        set(junction_prediction_index),
        set(trusted_val_annotations),
        "Junction prediction trusted frozen-val population",
    )
    for record_id, prediction in junction_prediction_index.items():
        annotation = trusted_val_annotations[record_id]
        _expect(prediction.get("truth"), annotation.get("label"), f"Junction {record_id} truth")
    classification = _classification_metrics(junction_prediction_rows, JUNCTION_CLASSES)
    declared_junction_metrics = _mapping(junction_status.get("metrics"), "Junction metrics")
    for key, value in _mapping(classification["metrics"], "computed Junction metrics").items():
        _expect_close(declared_junction_metrics.get(key), value, f"Junction metric {key}")

    junction_truth_audit = _read_object(junction_truth_audit_path, "junction truth audit")
    truth_lineage = _mapping(junction_truth_audit.get("lineage"), "junction truth lineage")
    _expect(
        truth_lineage.get("junction_annotations_sha256"),
        hashes["junction_annotations"],
        "junction truth audit annotations SHA-256",
    )
    _expect(
        truth_lineage.get("source_manifest_sha256"),
        hashes["source_manifest"],
        "junction truth audit source manifest SHA-256",
    )
    _expect(
        truth_lineage.get("split_manifest_sha256"),
        hashes["split_manifest"],
        "junction truth audit split manifest SHA-256",
    )

    replay_entry = dict(junction_lock)
    replay_entry["device"] = "cpu"
    junction_provider = TorchJunctionProvider(locked_junction_path, replay_entry)
    if junction_provider.device != "cpu":
        raise RuntimeError(f"Junction replay must remain on CPU, got {junction_provider.device!r}")

    variant_rows: dict[str, list[dict[str, Any]]] = {
        "oracle_source_membership_current_builder": [],
        "oracle_source_membership_no_augmentation_ablation": [],
        "gt_geometry_no_junction_current_builder": [],
        "gt_geometry_locked_junction_current_builder": [],
    }
    per_sample: list[dict[str, Any]] = []
    source_binding_rows: list[dict[str, str]] = []
    replay_kind_counts: Counter[str] = Counter()
    source_domains: Counter[str] = Counter()
    reference_group_total = 0
    provider = DatasetGoldenProvider()
    for sample_id in val_sample_ids:
        source_row = source_index[sample_id]
        files = _mapping(source_row.get("files"), f"source {sample_id} files")
        image_path, image_sha = _validate_source_file(
            source_root,
            _mapping(files.get("image"), f"source {sample_id} image"),
            f"source {sample_id} image",
        )
        _json_path, json_sha = _validate_source_file(
            source_root,
            _mapping(files.get("json"), f"source {sample_id} json"),
            f"source {sample_id} json",
        )
        source_binding_rows.append(
            {"sample_id": sample_id, "image_sha256": image_sha, "json_sha256": json_sha}
        )
        ir = provider.build_ir(image_path)
        _expect(ir.image.image_id, sample_id, f"Golden IR {sample_id} image_id")
        hints = _mapping(ir.metadata.get("dataset_graph_hints"), f"{sample_id} graph hints")
        _expect(hints.get("authoritative"), False, f"{sample_id} graph hint authority")
        _expect(hints.get("rebuild_required"), True, f"{sample_id} graph hint rebuild flag")
        raw_reference = _sequence(hints.get("physical_nets"), f"{sample_id} physical hints")
        reference_groups: list[list[str]] = []
        pin_ids = {pin.internal_id for pin in ir.pins}
        for position, raw_hint in enumerate(raw_reference):
            hint = _mapping(raw_hint, f"{sample_id} physical hint {position}")
            group = [str(item) for item in _sequence(hint.get("pin_ids"), "hint pin_ids")]
            unknown_pins = sorted(set(group) - pin_ids)
            if unknown_pins:
                raise ValueError(f"{sample_id} reference hint contains unknown pins: {unknown_pins}")
            reference_groups.append(group)
        reference_group_total += len(reference_groups)
        domain = _string(source_row.get("source_class"), f"source {sample_id} domain")
        source_domains[domain] += 1

        oracle_default = _evaluate_ir(ir, reference_groups)
        oracle_no_aug = _evaluate_ir(
            ir,
            reference_groups,
            builder_kwargs={
                "endpoint_tolerance_px": 0.0,
                "pin_snap_min_px": 0.0,
                "pin_snap_tolerance_px": 0.0,
                "direct_pin_tolerance_px": -1.0,
            },
        )
        geometry_ir = _strip_oracle_membership(ir)
        geometry_no_junction = _evaluate_ir(geometry_ir, reference_groups)
        predicted_junctions = junction_provider.classify(
            [load_image(image_path)], geometry_ir.segments
        )
        replay_kind_counts.update(item.kind for item in predicted_junctions)
        geometry_with_junction = geometry_ir.model_copy(deep=True)
        geometry_with_junction.junctions = predicted_junctions
        geometry_junction = _evaluate_ir(geometry_with_junction, reference_groups)

        metrics_by_variant = {
            "oracle_source_membership_current_builder": oracle_default,
            "oracle_source_membership_no_augmentation_ablation": oracle_no_aug,
            "gt_geometry_no_junction_current_builder": geometry_no_junction,
            "gt_geometry_locked_junction_current_builder": geometry_junction,
        }
        for name, metrics in metrics_by_variant.items():
            variant_rows[name].append(metrics)
        per_sample.append(
            {
                "sample_id": sample_id,
                "source_domain": domain,
                "reference_group_count": len(reference_groups),
                "reference_pair_count": len(_pin_pairs(reference_groups)),
                "junction_replay_kind_counts": dict(
                    sorted(Counter(item.kind for item in predicted_junctions).items())
                ),
                "variants": metrics_by_variant,
            }
        )

    aggregates = {
        name: _aggregate_variant(rows) for name, rows in variant_rows.items()
    }
    oracle = aggregates["oracle_source_membership_current_builder"]
    no_aug = aggregates["oracle_source_membership_no_augmentation_ablation"]
    no_junction = aggregates["gt_geometry_no_junction_current_builder"]
    with_junction = aggregates["gt_geometry_locked_junction_current_builder"]

    trusted_label_support = Counter(
        _string(row.get("label"), "trusted Junction label")
        for row in trusted_val_annotations.values()
    )
    unresolved_status = Counter(
        _string(row.get("label_status"), "unresolved Junction status")
        for row in unresolved_val_annotations
    )
    unresolved_labels = Counter(str(row.get("label")) for row in unresolved_val_annotations)
    unresolved_owner_edges = Counter()
    for row in unresolved_val_annotations:
        source = _mapping(row.get("source"), "unresolved Junction source")
        refs = _sequence(source.get("source_refs"), "unresolved Junction source_refs")
        unresolved_owner_edges[len({str(_mapping(ref, "source ref").get("edge_key")) for ref in refs})] += 1

    wire_overall = _mapping(wire_audit.get("overall_frozen_val"), "Wire overall_frozen_val")
    formal_wire_proxy = {
        "role": "upstream tile proxy only; not used to construct an E2E graph",
        "prediction_rows": len(wire_prediction_rows),
        "covered_val_samples": len(eligible_val_samples),
        "total_val_samples": len(val_sample_ids),
        "total_val_tiles": len(val_tile_ids),
        "eligible_val_tiles": len(eligible_val_ids),
        "ineligible_val_tiles_without_comparable_predictions": len(val_tile_ids - eligible_val_ids),
        "covered_sample_ids": sorted(eligible_val_samples),
        "uncovered_sample_ids": sorted(val_sample_set - eligible_val_samples),
        "corrected_micro": wire_overall.get("micro"),
        "tile_macro": wire_overall.get("tile_macro"),
        "nonempty_gt_only_diagnostic": wire_audit.get("nonempty_gt_only_diagnostic"),
    }

    constructability = _constructability_matrix(
        val_samples=len(val_sample_ids),
        wire_covered_samples=len(eligible_val_samples),
        eligible_wire_tiles=len(eligible_val_ids),
        ineligible_wire_tiles=len(val_tile_ids - eligible_val_ids),
        trusted_junction_rows=len(trusted_val_annotations),
        unresolved_junction_rows=len(unresolved_val_annotations),
    )

    code_paths = {
        "audit_script": Path(__file__).resolve(),
        "physical_graph_builder": workspace / "src/pcbparse/graph/physical.py",
        "golden_provider": workspace / "src/pcbparse/perception/golden_provider.py",
        "junction_adapter": workspace / "src/pcbparse/perception/torch_junction_adapter.py",
    }
    sources: dict[str, Any] = {
        "source_manifest": _source(source_manifest_path, workspace),
        "split_manifest": _source(split_manifest_path, workspace),
        "wire_manifest": _source(wire_manifest_path, workspace),
        "junction_manifest": _source(junction_manifest_path, workspace),
        "materialization_summary": _source(summary_path, workspace),
        "wire_tiles": _source(wire_tiles_path, workspace),
        "junction_annotations": _source(junction_annotations_path, workspace),
        "wire_predictions": _source(wire_predictions_path, workspace),
        "wire_run_status": _source(wire_status_path, workspace),
        "wire_formal_model": _source(wire_model_path, workspace),
        "wire_locked_candidate": _source(locked_wire_path, workspace),
        "wire_stratified_audit": _source(wire_audit_path, workspace),
        "junction_predictions": _source(junction_predictions_path, workspace),
        "junction_run_status": _source(junction_status_path, workspace),
        "junction_formal_onnx": _source(junction_formal_model_path, workspace),
        "junction_work_checkpoint": _source(junction_work_checkpoint_path, workspace),
        "junction_locked_replay_checkpoint": _source(locked_junction_path, workspace),
        "junction_truth_audit": _source(junction_truth_audit_path, workspace),
        "models_lock": _source(models_lock_path, workspace),
    }
    sources.update({name: _source(path, workspace) for name, path in code_paths.items()})

    report: dict[str, Any] = {
        "kind": "pcbparse_frozen_val_graph_oracle_decomposition_audit",
        "schema_version": 1,
        "audit_version": AUDIT_VERSION,
        "selection_boundary": {
            "evaluation_split": "val",
            "val_source_samples_used": len(val_sample_ids),
            "train_rows_used": 0,
            "test_rows_used": 0,
            "test_labels_used": 0,
            "wire_val_prediction_rows_used_as_proxy": len(wire_prediction_rows),
            "wire_prediction_rows_used_for_graph_construction": 0,
            "junction_val_patch_predictions_used_as_context": len(junction_prediction_rows),
            "junction_patch_predictions_used_for_graph_construction": 0,
            "junction_checkpoint_replay_val_samples": len(val_sample_ids),
            "thresholds_tuned": False,
            "test_used_for_tuning": False,
            "junction_score_threshold_source": "JunctionAdapterConfig default/current locked adapter contract",
            "junction_score_threshold": junction_provider.config.score_threshold,
        },
        "sources": sources,
        "source_file_binding": {
            "val_files_verified_against_source_manifest": len(source_binding_rows) * 2,
            "val_sample_count": len(source_binding_rows),
            "binding_sha256": hashlib.sha256(_canonical_bytes(source_binding_rows)).hexdigest(),
            "source_domain_counts": dict(sorted(source_domains.items())),
        },
        "reference_contract": {
            "provider": "DatasetGoldenProvider(dataset300-golden-v2, strict)",
            "reference": "metadata.dataset_graph_hints.physical_nets pin_ids",
            "authoritative": False,
            "use": "dataset-derived oracle/debug pair relation only; not official competition truth",
            "metric": "unordered pairwise pin co-connectivity, micro-aggregated across samples",
            "metric_limit": (
                "pairwise relations do not distinguish singleton-only partitions or segment-only nets"
            ),
            "reference_group_count": reference_group_total,
            "reference_pair_count": oracle["reference_pairs"],
        },
        "constructability_matrix": constructability,
        "formal_wire_upstream_proxy": formal_wire_proxy,
        "formal_junction_patch_evidence": {
            "status": junction_status.get("status"),
            "gates_passed": junction_status.get("gates_passed"),
            "prediction_rows": len(junction_prediction_rows),
            "trusted_val_support": dict(sorted(trusted_label_support.items())),
            "computed_metrics": classification["metrics"],
            "confusion": classification["confusion"],
            "candidate_population_warning": (
                "these predictions are on GT-derived materialized patches, not candidates generated "
                "from predicted Wire segments"
            ),
            "unresolved_val_rows": len(unresolved_val_annotations),
            "unresolved_status_counts": dict(sorted(unresolved_status.items())),
            "unresolved_label_counts": dict(sorted(unresolved_labels.items())),
            "unresolved_owner_edge_count": {
                str(key): value for key, value in sorted(unresolved_owner_edges.items())
            },
            "full_six_class_oracle_available": False,
        },
        "oracle_variants": {
            "contracts": {
                "oracle_source_membership_current_builder": (
                    "GT source-edge/polyline and explicit pin membership retained; current defaults"
                ),
                "oracle_source_membership_no_augmentation_ablation": (
                    "fixed diagnostic ablation: endpoint=0, pin snap=0, direct-pin disabled; "
                    "not a parameter search"
                ),
                "gt_geometry_no_junction_current_builder": (
                    "GT Wire geometry and GT Pin coordinates retained; source memberships/evidence stripped"
                ),
                "gt_geometry_locked_junction_current_builder": (
                    "same GT geometry plus locked candidate Junction checkpoint on actual production "
                    "candidate generator, forced CPU"
                ),
            },
            "aggregate": aggregates,
            "per_sample": per_sample,
        },
        "junction_cpu_replay": {
            "device_requested": "cpu",
            "device_actual": junction_provider.device,
            "model_version": junction_provider.version,
            "model_sha256": junction_provider.artifact_sha256,
            "models_lock_matches": junction_provider.artifact_sha256 == junction_lock.get("sha256"),
            "models_lock_status": junction_lock.get("status"),
            "models_lock_blessed": junction_lock.get("blessed"),
            "adapter_config": asdict(junction_provider.config),
            "candidate_kind_counts": dict(sorted(replay_kind_counts.items())),
        },
        "driver_attribution": {
            "builder_oracle_residual": {
                "evidence": "directly_measured",
                "f1": oracle["f1"],
                "fp": oracle["fp"],
                "fn": oracle["fn"],
                "default_augmentation_fp_delta_vs_fixed_ablation": oracle["fp"] - no_aug["fp"],
                "interpretation": (
                    "the current graph builder is close but not exact even with hidden oracle memberships; "
                    "remaining error is primarily over-connection"
                ),
            },
            "geometry_association_contract": {
                "evidence": "directly_measured_on_gt_geometry_without_hidden_membership",
                "f1": no_junction["f1"],
                "fp": no_junction["fp"],
                "fn": no_junction["fn"],
                "interpretation": (
                    "geometry/pin association loses reference co-connectivity before learned Wire error is introduced"
                ),
            },
            "junction_integration": {
                "evidence": "directly_measured_conditional_replay_on_gt_wire_geometry",
                "f1_without_junction": no_junction["f1"],
                "f1_with_locked_junction": with_junction["f1"],
                "f1_delta": with_junction["f1"] - no_junction["f1"],
                "tp_delta": with_junction["tp"] - no_junction["tp"],
                "fp_delta": with_junction["fp"] - no_junction["fp"],
                "fn_delta": with_junction["fn"] - no_junction["fn"],
                "interpretation": (
                    "the locked Junction checkpoint/candidate integration is the strongest directly "
                    "measured conditional over-merge driver in this audit"
                ),
                "caveat": (
                    "GT Golden geometry/channel distribution is not identical to predicted-Wire vectorization, "
                    "so this is a conditional stress replay rather than full production E2E"
                ),
            },
            "predicted_wire": {
                "evidence": "upstream_proxy_only",
                "tile_connectivity_score": _mapping(
                    wire_overall.get("tile_macro"), "Wire tile macro"
                ).get("connectivity_score"),
                "coverage_samples": len(eligible_val_samples),
                "total_val_samples": len(val_sample_ids),
                "interpretation": (
                    "saved formal Wire evidence indicates a material topology risk, but its exact full-graph "
                    "contribution is not identifiable from the incomplete saved mask population"
                ),
            },
            "exact_wire_vs_junction_percentage_attribution_available": False,
        },
        "hard_blockers": [
            "Saved formal Wire predictions do not cover every frozen-val tile/sample; new inference is required for a complete predicted-Wire graph.",
            "Frozen-val Junction crossing candidates include unresolved needs_review rows and trusted connected/non-connected crossing support is absent.",
            "Formal Junction predictions are GT-patch-distribution records and cannot be attached to predicted-Wire segment IDs.",
            "Dataset graph hints explicitly declare authoritative=false and are not official evaluator truth.",
            "The Junction candidate is unblessed and its six-class macro-F1 gate failed.",
        ],
        "promotion_policy": {
            "diagnostic_only": True,
            "promotion_eligible": False,
            "production_blessing": False,
            "models_lock_modified": False,
            "training_config_modified": False,
            "threshold_search_performed": False,
            "test_used": False,
        },
    }
    report["audit_payload_sha256"] = _payload_sha256(report)
    return report


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workspace = args.workspace_root.expanduser().resolve()
    if not workspace.is_dir():
        raise FileNotFoundError(f"workspace root is missing: {workspace}")
    source_root = _resolve(workspace, args.source_root, "source root", directory=True)
    materialized_run = _resolve(
        workspace, args.materialized_run, "materialized run", directory=True
    )
    report = build_audit(
        workspace=workspace,
        source_root=source_root,
        source_manifest_path=_resolve(workspace, args.source_manifest, "source manifest"),
        split_manifest_path=_resolve(workspace, args.split_manifest, "split manifest"),
        wire_manifest_path=_resolve(workspace, args.wire_manifest, "wire manifest"),
        junction_manifest_path=_resolve(
            workspace, args.junction_manifest, "junction manifest"
        ),
        materialized_run=materialized_run,
        wire_predictions_path=_resolve(
            workspace, args.wire_predictions, "Wire predictions"
        ),
        wire_status_path=_resolve(workspace, args.wire_run_status, "Wire run status"),
        wire_model_path=_resolve(workspace, args.wire_model, "Wire model"),
        wire_audit_path=_resolve(workspace, args.wire_audit, "Wire audit"),
        junction_predictions_path=_resolve(
            workspace, args.junction_predictions, "Junction predictions"
        ),
        junction_status_path=_resolve(
            workspace, args.junction_run_status, "Junction run status"
        ),
        junction_formal_model_path=_resolve(
            workspace, args.junction_formal_model, "Junction formal model"
        ),
        junction_work_checkpoint_path=_resolve(
            workspace, args.junction_work_checkpoint, "Junction work checkpoint"
        ),
        junction_truth_audit_path=_resolve(
            workspace, args.junction_truth_audit, "Junction truth audit"
        ),
        models_lock_path=_resolve(workspace, args.models_lock, "models.lock"),
    )
    output = _resolve_output(workspace, args.output)
    _atomic_write(output, report)
    print(
        json.dumps(
            {
                "output": _relative(output, workspace),
                "audit_payload_sha256": report["audit_payload_sha256"],
                "test_rows_used": report["selection_boundary"]["test_rows_used"],
                "junction_device": report["junction_cpu_replay"]["device_actual"],
                "variant_f1": {
                    key: value["f1"]
                    for key, value in report["oracle_variants"]["aggregate"].items()
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
