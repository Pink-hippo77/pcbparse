from __future__ import annotations

"""Audit a completed Wire candidate on the exact frozen validation tiles.

This script consumes already-emitted binary predictions.  It never runs model
inference, searches thresholds, or computes a metric on train/test rows.  The
historical trainer report is a tile-macro average; this audit reproduces that
number for lineage reconciliation and additionally reports count-aware micro
precision/recall/F1 across all validation skeleton pixels.
"""

import argparse
import base64
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


FINAL_FORMAL_STATUSES = frozenset({"candidate_evidence_ready", "gate_failed"})
DENSITY_BINS: tuple[tuple[str, float, float | None], ...] = (
    ("empty", 0.0, 0.0),
    ("very_sparse", 0.0, 0.001),
    ("sparse", 0.001, 0.01),
    ("dense", 0.01, None),
)
SOURCE_SIZE_BINS: tuple[tuple[str, int, int | None], ...] = (
    ("small_lt_0_5mp", 0, 500_000),
    ("medium_0_5_to_1_5mp", 500_000, 1_500_000),
    ("large_ge_1_5mp", 1_500_000, None),
)


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=root)
    parser.add_argument(
        "--predictions", type=Path, default=Path("work/train-wire-formal4/predictions.jsonl")
    )
    parser.add_argument(
        "--run-status", type=Path, default=Path("work/train-wire-formal4/torch_run_status.json")
    )
    parser.add_argument(
        "--run-metrics", type=Path, default=Path("work/train-wire-formal4/metrics.json")
    )
    parser.add_argument(
        "--model-artifact", type=Path, default=Path("work/train-wire-formal4/model.pt")
    )
    parser.add_argument(
        "--candidate-artifact", type=Path, default=Path("models/candidates/wire/model.pt")
    )
    parser.add_argument("--models-lock", type=Path, default=Path("models/models.lock.json"))
    parser.add_argument(
        "--training-config", type=Path, default=Path("configs/train/dataset300_v2_torch.json")
    )
    parser.add_argument(
        "--materialized-run",
        type=Path,
        default=Path(
            "artifacts/training/materialized/dataset300-v2/wire-junction/runs/"
            "5233aeb838db659970f0ce73480753a2f4a4e79366612878986d3bf18bca610e"
        ),
    )
    parser.add_argument(
        "--wire-manifest", type=Path, default=Path("data/derived/dataset300-v2/wire_paths.jsonl")
    )
    parser.add_argument(
        "--source-manifest", type=Path, default=Path("data/manifests/dataset300.jsonl")
    )
    parser.add_argument(
        "--split-manifest", type=Path, default=Path("data/splits/dataset300_split.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/wire_formal4_val_stratified.json"),
    )
    return parser


def _resolve_within(root: Path, raw: Path, label: str, *, must_exist: bool = True) -> Path:
    path = raw.expanduser()
    resolved = (path if path.is_absolute() else root / path).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside workspace: {resolved}") from exc
    if must_exist and (not resolved.is_file() or resolved.stat().st_size <= 0):
        raise FileNotFoundError(f"{label} is missing or empty: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise ValueError(f"invalid {label} JSON: {exc}") from exc


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = _loads(path.read_text(encoding="utf-8"), label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = _loads(raw, f"{label}:{line_number}")
            if not isinstance(value, dict):
                raise ValueError(f"{label}:{line_number}: row must be an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{label} is empty: {path}")
    return rows


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
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


def _unique_index(
    rows: Iterable[Mapping[str, Any]], key: str, label: str
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        identity = _string(row.get(key), f"{label}.{key}")
        if identity in output:
            raise ValueError(f"duplicate {label} {key}: {identity}")
        output[identity] = row
    return output


def _expect_sha(actual: str, expected: Any, label: str) -> None:
    if expected != actual:
        raise ValueError(f"{label} SHA-256 mismatch: expected={expected!r}, actual={actual}")


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def decode_prediction_mask(row: Mapping[str, Any]) -> np.ndarray:
    """Strictly decode one little-bit-order packbits mask."""

    if row.get("encoding") != "base64-packbits-little":
        raise ValueError("prediction encoding must be base64-packbits-little")
    shape = _sequence(row.get("shape"), "prediction shape")
    if len(shape) != 2 or any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError("prediction shape must contain two positive integers")
    height, width = int(shape[0]), int(shape[1])
    raw_text = _string(row.get("mask"), "prediction mask")
    try:
        packed = base64.b64decode(raw_text, validate=True)
    except Exception as exc:
        raise ValueError("prediction mask is not strict base64") from exc
    expected_bytes = (height * width + 7) // 8
    if len(packed) != expected_bytes:
        raise ValueError(
            f"prediction packed byte count mismatch: expected={expected_bytes}, actual={len(packed)}"
        )
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")
    if bool(bits[height * width :].any()):
        raise ValueError("prediction mask has non-zero padding bits")
    return bits[: height * width].reshape(height, width).astype(bool)


def _component_count(mask: np.ndarray) -> int:
    remaining = np.asarray(mask, dtype=bool).copy()
    height, width = remaining.shape
    count = 0
    for raw_y, raw_x in np.argwhere(remaining):
        y, x = int(raw_y), int(raw_x)
        if not remaining[y, x]:
            continue
        count += 1
        remaining[y, x] = False
        pending = [(y, x)]
        while pending:
            current_y, current_x = pending.pop()
            for next_y in range(max(0, current_y - 1), min(height, current_y + 2)):
                for next_x in range(max(0, current_x - 1), min(width, current_x + 2)):
                    if remaining[next_y, next_x]:
                        remaining[next_y, next_x] = False
                        pending.append((next_y, next_x))
    return count


def _ratio(numerator: float, denominator: float, *, opposite_total: float) -> float:
    if denominator:
        return float(numerator / denominator)
    return 1.0 if opposite_total == 0.0 else 0.0


def _minimum_pool_cross(image: np.ndarray) -> np.ndarray:
    """NumPy equivalent of the trainer's vertical/horizontal soft erosion."""

    vertical = np.pad(
        image, ((1, 1), (0, 0)), mode="constant", constant_values=np.inf
    )
    vertical_min = np.minimum.reduce(
        (vertical[:-2, :], vertical[1:-1, :], vertical[2:, :])
    )
    horizontal = np.pad(
        image, ((0, 0), (1, 1)), mode="constant", constant_values=np.inf
    )
    horizontal_min = np.minimum.reduce(
        (horizontal[:, :-2], horizontal[:, 1:-1], horizontal[:, 2:])
    )
    return np.minimum(vertical_min, horizontal_min)


def _maximum_pool_square(image: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius < 0:
        raise ValueError("pool radius must be non-negative")
    if radius == 0:
        return image.copy()
    padded = np.pad(
        image,
        ((radius, radius), (radius, radius)),
        mode="constant",
        constant_values=-np.inf,
    )
    height, width = image.shape
    windows = [
        padded[y : y + height, x : x + width]
        for y in range(radius * 2 + 1)
        for x in range(radius * 2 + 1)
    ]
    return np.maximum.reduce(windows)


def _soft_skeletonize_numpy(
    inputs: np.ndarray, *, iterations: int, valid: np.ndarray
) -> np.ndarray:
    """Exact CPU replay of ``torch_wire_model.soft_skeletonize`` for 2-D masks."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    image = np.clip(np.asarray(inputs, dtype=np.float32), 0.0, 1.0)
    valid_float = np.asarray(valid, dtype=np.float32)
    image = image * valid_float

    def opened(values: np.ndarray) -> np.ndarray:
        eroded = _minimum_pool_cross(values) * valid_float
        return _maximum_pool_square(eroded, radius=1) * valid_float

    skeleton = np.maximum(image - opened(image), 0.0) * valid_float
    for _ in range(iterations - 1):
        image = _minimum_pool_cross(image) * valid_float
        delta = np.maximum(image - opened(image), 0.0) * valid_float
        skeleton = skeleton + np.maximum(delta - skeleton * delta, 0.0)
    return skeleton * valid_float


def evaluate_masks(
    prediction: np.ndarray,
    target: np.ndarray,
    ignore: np.ndarray,
    *,
    skeleton_iterations: int = 8,
    tolerance_px: int = 1,
) -> dict[str, float]:
    """Independently evaluate one binary tile under the formal metric contract."""

    if prediction.shape != target.shape or target.shape != ignore.shape:
        raise ValueError("prediction, target, and ignore masks must have identical shapes")
    if prediction.ndim != 2:
        raise ValueError("wire masks must be two-dimensional")
    if skeleton_iterations <= 0 or tolerance_px < 0:
        raise ValueError("invalid skeleton/tolerance policy")
    valid_np = ~np.asarray(ignore, dtype=bool)
    predicted_np = np.asarray(prediction, dtype=bool) & valid_np
    target_np = np.asarray(target, dtype=bool) & valid_np
    predicted_skeleton = _soft_skeletonize_numpy(
        predicted_np, iterations=skeleton_iterations, valid=valid_np
    ) > 0.5
    target_skeleton = _soft_skeletonize_numpy(
        target_np, iterations=skeleton_iterations, valid=valid_np
    ) > 0.5
    predicted_neighbourhood = (
        _maximum_pool_square(predicted_skeleton.astype(np.float32), tolerance_px)
        > 0.5
    ) & valid_np
    target_neighbourhood = (
        _maximum_pool_square(target_skeleton.astype(np.float32), tolerance_px)
        > 0.5
    ) & valid_np
    predicted_count = float(predicted_skeleton.sum())
    target_count = float(target_skeleton.sum())
    precision_hits = float((predicted_skeleton & target_neighbourhood).sum())
    recall_hits = float((target_skeleton & predicted_neighbourhood).sum())
    # Preserve the historical per-tile implementation exactly for lineage
    # reconciliation: a zero denominator returned 1 for that individual ratio,
    # even when the opposite side was non-empty.  A one-sided-empty tile still
    # has F1=0 because the other ratio is zero.  ``summarize`` deliberately does
    # not carry this convention into the corrected population-level micro
    # metric, which uses the raw counts below.
    precision = precision_hits / predicted_count if predicted_count else 1.0
    recall = recall_hits / target_count if target_count else 1.0
    denominator = precision + recall
    f1 = 2.0 * precision * recall / denominator if denominator else 0.0
    predicted_components = _component_count(predicted_skeleton & valid_np)
    target_components = _component_count(target_skeleton & valid_np)
    if predicted_components == target_components == 0:
        connectivity = 1.0
    else:
        connectivity = max(
            0.0,
            1.0
            - abs(predicted_components - target_components)
            / max(predicted_components, target_components, 1),
        )
    return {
        "precision_hits": precision_hits,
        "recall_hits": recall_hits,
        "predicted_skeleton_pixels": predicted_count,
        "target_skeleton_pixels": target_count,
        "centerline_precision": precision,
        "centerline_recall": recall,
        "centerline_f1": f1,
        "skeleton_f1": f1,
        "predicted_components": float(predicted_components),
        "target_components": float(target_components),
        "component_count_error": float(abs(predicted_components - target_components)),
        "connectivity_score": float(connectivity),
        "valid_pixels": float(valid_np.sum()),
        "raw_predicted_positive_pixels": float(predicted_np.sum()),
        "raw_target_positive_pixels": float(target_np.sum()),
    }


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "tiles": 0,
            "samples": 0,
            "target_skeleton_pixels": 0,
            "predicted_skeleton_pixels": 0,
            "micro": None,
            "tile_macro": None,
        }
    sums = {
        name: sum(float(row[name]) for row in records)
        for name in (
            "precision_hits",
            "recall_hits",
            "predicted_skeleton_pixels",
            "target_skeleton_pixels",
            "valid_pixels",
            "raw_predicted_positive_pixels",
            "raw_target_positive_pixels",
        )
    }
    precision = _ratio(
        sums["precision_hits"],
        sums["predicted_skeleton_pixels"],
        opposite_total=sums["target_skeleton_pixels"],
    )
    recall = _ratio(
        sums["recall_hits"],
        sums["target_skeleton_pixels"],
        opposite_total=sums["predicted_skeleton_pixels"],
    )
    denominator = precision + recall
    f1 = 2.0 * precision * recall / denominator if denominator else 0.0
    count = len(records)
    macro_names = (
        "centerline_precision",
        "centerline_recall",
        "centerline_f1",
        "connectivity_score",
        "component_count_error",
    )
    return {
        "tiles": count,
        "samples": len({str(row["sample_id"]) for row in records}),
        "empty_target_tiles": sum(float(row["target_skeleton_pixels"]) == 0.0 for row in records),
        "empty_prediction_tiles": sum(
            float(row["predicted_skeleton_pixels"]) == 0.0 for row in records
        ),
        "empty_target_false_positive_tiles": sum(
            float(row["target_skeleton_pixels"]) == 0.0
            and float(row["predicted_skeleton_pixels"]) > 0.0
            for row in records
        ),
        "valid_pixels": int(sums["valid_pixels"]),
        "target_skeleton_pixels": int(sums["target_skeleton_pixels"]),
        "predicted_skeleton_pixels": int(sums["predicted_skeleton_pixels"]),
        "target_density": (
            sums["target_skeleton_pixels"] / sums["valid_pixels"]
            if sums["valid_pixels"]
            else 0.0
        ),
        "micro": {
            "precision_hits": int(sums["precision_hits"]),
            "recall_hits": int(sums["recall_hits"]),
            "centerline_precision": precision,
            "centerline_recall": recall,
            "centerline_f1": f1,
            "skeleton_f1": f1,
        },
        "tile_macro": {
            name: sum(float(row[name]) for row in records) / count
            for name in macro_names
        },
    }


def density_stratum(density: float, target_pixels: float) -> str:
    if target_pixels == 0.0:
        return "empty"
    if density <= 0.001:
        return "very_sparse"
    if density <= 0.01:
        return "sparse"
    return "dense"


def source_size_stratum(area: int) -> str:
    if area < 500_000:
        return "small_lt_0_5mp"
    if area < 1_500_000:
        return "medium_0_5_to_1_5mp"
    return "large_ge_1_5mp"


def _stratify(
    records: Sequence[Mapping[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row[field])].append(row)
    return {name: summarize(rows) for name, rows in sorted(groups.items())}


def _current_training_config_sha256(path: Path, workspace: Path) -> str | None:
    source_root = workspace / "src"
    if not source_root.is_dir():
        return None
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    try:
        from pcbparse.training.config import load_training_config, training_config_sha256

        return training_config_sha256(load_training_config(path))
    except Exception:
        return None


def _lock_wire_entry(document: Mapping[str, Any]) -> Mapping[str, Any]:
    models = _sequence(document.get("models"), "models.lock models")
    matches = [row for row in models if isinstance(row, Mapping) and row.get("task") == "wire"]
    if len(matches) != 1:
        raise ValueError("models.lock must contain exactly one Wire entry")
    return matches[0]


def build_audit(
    *,
    workspace: Path,
    predictions_path: Path,
    status_path: Path,
    metrics_path: Path,
    model_path: Path,
    materialized_run: Path,
    wire_manifest_path: Path,
    source_manifest_path: Path,
    split_manifest_path: Path,
    candidate_artifact_path: Path | None = None,
    models_lock_path: Path | None = None,
    training_config_path: Path | None = None,
) -> dict[str, Any]:
    summary_path = materialized_run / "summary.json"
    wire_tiles_path = materialized_run / "wire_tiles.jsonl"
    required = {
        "predictions": predictions_path,
        "run_status": status_path,
        "run_metrics": metrics_path,
        "model_artifact": model_path,
        "materialization_summary": summary_path,
        "wire_tiles": wire_tiles_path,
        "wire_manifest": wire_manifest_path,
        "source_manifest": source_manifest_path,
        "split_manifest": split_manifest_path,
    }
    if candidate_artifact_path is not None:
        required["candidate_artifact"] = candidate_artifact_path
    if models_lock_path is not None:
        required["models_lock"] = models_lock_path
    if training_config_path is not None:
        required["current_training_config"] = training_config_path
    for label, path in required.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"{label} is missing or empty: {path}")
    hashes = {name: _sha256(path) for name, path in required.items()}

    status = _read_object(status_path, "Wire run status")
    if status.get("task") != "wire" or status.get("status") not in FINAL_FORMAL_STATUSES:
        raise ValueError("Wire run is not completed formal evidence")
    if status.get("formal_evidence") is not True or status.get("smoke") is not False:
        raise ValueError("Wire run is not formal non-smoke evidence")
    if not status.get("finished_at_utc"):
        raise ValueError("Wire run status lacks finished_at_utc")
    _expect_sha(
        hashes["predictions"],
        _mapping(status.get("predictions"), "status.predictions").get("sha256"),
        "predictions/run status",
    )
    _expect_sha(
        hashes["model_artifact"],
        _mapping(status.get("artifact"), "status.artifact").get("sha256"),
        "model/run status",
    )

    metrics_document = _read_object(metrics_path, "Wire metrics")
    if metrics_document.get("split") != "val" or metrics_document.get("formal_evidence") is not True:
        raise ValueError("run metrics are not frozen validation evidence")
    historical_metrics = _mapping(metrics_document.get("metrics"), "run metrics.metrics")
    if dict(historical_metrics) != dict(_mapping(status.get("metrics"), "status.metrics")):
        raise ValueError("run status and metrics.json disagree")

    summary = _read_object(summary_path, "wire-junction summary")
    if summary.get("kind") != "wire_junction_materialization" or summary.get("complete") is not True:
        raise ValueError("wire-junction materialization is incomplete")
    run_id = _string(summary.get("run_id"), "materialization run_id")
    if materialized_run.name != run_id:
        raise ValueError("materialized directory name differs from run_id")
    outputs = _mapping(summary.get("outputs"), "summary.outputs")
    _expect_sha(
        hashes["wire_tiles"],
        _mapping(outputs.get("wire_tiles"), "summary wire_tiles output").get("sha256"),
        "wire_tiles/summary",
    )
    summary_lineage = _mapping(summary.get("lineage"), "summary.lineage")
    summary_inputs = _mapping(summary_lineage.get("inputs"), "summary.lineage.inputs")
    for source_name, lineage_name in (
        ("wire_manifest", "wire_manifest_sha256"),
        ("source_manifest", "source_manifest_sha256"),
        ("split_manifest", "split_manifest_sha256"),
    ):
        _expect_sha(hashes[source_name], summary_inputs.get(lineage_name), f"{source_name}/summary")

    run_lineage = _mapping(status.get("lineage"), "status.lineage")
    expected_lineage = {
        "task": "wire",
        "formal_evidence": True,
        "smoke": False,
        "binding_sha256": run_id,
        "materialization_commit_id": run_id,
        "materialization_commit_sha256": hashes["materialization_summary"],
        "source_manifest_sha256": hashes["source_manifest"],
        "split_manifest_sha256": hashes["split_manifest"],
        "task_manifest_sha256": hashes["wire_manifest"],
    }
    for key, expected in expected_lineage.items():
        if run_lineage.get(key) != expected:
            raise ValueError(
                f"run lineage mismatch for {key}: expected={expected!r}, actual={run_lineage.get(key)!r}"
            )

    split_rows = _read_jsonl(split_manifest_path, "split manifest")
    split_by_sample = {
        sample_id: _string(row.get("split"), "split")
        for sample_id, row in _unique_index(split_rows, "sample_id", "split manifest").items()
    }
    if any(value not in {"train", "val", "test"} for value in split_by_sample.values()):
        raise ValueError("split manifest contains unsupported split")
    source_by_sample = _unique_index(
        _read_jsonl(source_manifest_path, "source manifest"), "sample_id", "source manifest"
    )
    if set(source_by_sample) != set(split_by_sample):
        raise ValueError("source and split sample populations differ")

    tile_rows = _read_jsonl(wire_tiles_path, "wire tiles")
    tile_by_id = _unique_index(tile_rows, "record_id", "wire tiles")
    eligible_by_split: dict[str, set[str]] = defaultdict(set)
    tile_counts: Counter[str] = Counter()
    for record_id, row in tile_by_id.items():
        sample_id = _string(row.get("sample_id"), "tile sample_id")
        split = _string(row.get("split"), "tile split")
        if split_by_sample.get(sample_id) != split:
            raise ValueError(f"wire tile/split mismatch at {record_id}")
        supervision = _mapping(row.get("supervision"), "tile supervision")
        eligible = supervision.get("eligible")
        if type(eligible) is not bool:
            raise ValueError(f"tile eligibility is not boolean: {record_id}")
        tile_counts[f"{split}_{'eligible' if eligible else 'ineligible'}"] += 1
        if eligible:
            eligible_by_split[split].add(record_id)

    prediction_rows = _read_jsonl(predictions_path, "Wire predictions")
    prediction_by_id = _unique_index(prediction_rows, "record_id", "Wire predictions")
    expected_val_ids = eligible_by_split["val"]
    if set(prediction_by_id) != expected_val_ids:
        unexpected = sorted(set(prediction_by_id) - expected_val_ids)
        missing = sorted(expected_val_ids - set(prediction_by_id))
        raise ValueError(
            "predictions are not the exact frozen eligible-val population: "
            f"unexpected={unexpected[:4]}, missing={missing[:4]}"
        )

    evaluated: list[dict[str, Any]] = []
    for record_id in sorted(prediction_by_id):
        prediction_row = prediction_by_id[record_id]
        tile = tile_by_id[record_id]
        sample_id = _string(tile.get("sample_id"), "tile sample_id")
        if prediction_row.get("sample_id") != sample_id:
            raise ValueError(f"prediction sample mismatch at {record_id}")
        prediction = decode_prediction_mask(prediction_row)
        tile_size = _sequence(tile.get("tile_size"), "tile_size")
        if len(tile_size) != 2 or prediction.shape != (int(tile_size[1]), int(tile_size[0])):
            raise ValueError(f"prediction/tile shape mismatch at {record_id}")
        target_info = _mapping(tile.get("target_mask"), "target mask")
        ignore_info = _mapping(tile.get("ignore_mask"), "ignore mask")
        target_path = _resolve_within(materialized_run, Path(_string(target_info.get("path"), "target path")), "target mask")
        ignore_path = _resolve_within(materialized_run, Path(_string(ignore_info.get("path"), "ignore path")), "ignore mask")
        _expect_sha(_sha256(target_path), target_info.get("sha256"), f"target mask {record_id}")
        _expect_sha(_sha256(ignore_path), ignore_info.get("sha256"), f"ignore mask {record_id}")
        with Image.open(target_path) as image:
            target = np.asarray(image, dtype=np.uint8) > 0
        with Image.open(ignore_path) as image:
            ignore = np.asarray(image, dtype=np.uint8) > 0
        if target.shape != prediction.shape or ignore.shape != prediction.shape:
            raise ValueError(f"materialized mask shape mismatch at {record_id}")
        if int(target.sum()) != int(target_info.get("foreground_pixels", -1)):
            raise ValueError(f"target foreground count mismatch at {record_id}")
        if int(ignore.sum()) != int(ignore_info.get("foreground_pixels", -1)):
            raise ValueError(f"ignore foreground count mismatch at {record_id}")
        if bool((target & ignore).any()):
            raise ValueError(f"target overlaps ignore at {record_id}")
        metric = evaluate_masks(prediction, target, ignore)
        source = source_by_sample[sample_id]
        source_class = _string(source.get("source_class"), "source_class")
        canvas = _mapping(source.get("canvas"), "source canvas")
        width = int(canvas.get("width", 0))
        height = int(canvas.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid source canvas for {sample_id}")
        density = (
            metric["target_skeleton_pixels"] / metric["valid_pixels"]
            if metric["valid_pixels"]
            else 0.0
        )
        evaluated.append(
            {
                "record_id": record_id,
                "sample_id": sample_id,
                "source_domain": source_class,
                "source_image_area": width * height,
                "source_size_stratum": source_size_stratum(width * height),
                "density_stratum": density_stratum(density, metric["target_skeleton_pixels"]),
                "tile_coverage_stratum": (
                    "full_512" if prediction.shape == (512, 512) else "boundary_clipped"
                ),
                "target_density": density,
                **metric,
            }
        )

    overall = summarize(evaluated)
    nonempty = summarize([row for row in evaluated if row["target_skeleton_pixels"] > 0])
    reproduced = {
        "centerline_precision": math.isclose(
            overall["tile_macro"]["centerline_precision"],
            _finite(historical_metrics.get("centerline_precision"), "historical precision"),
            rel_tol=0.0,
            abs_tol=2e-7,
        ),
        "centerline_recall": math.isclose(
            overall["tile_macro"]["centerline_recall"],
            _finite(historical_metrics.get("centerline_recall"), "historical recall"),
            rel_tol=0.0,
            abs_tol=2e-7,
        ),
        "skeleton_f1": math.isclose(
            overall["tile_macro"]["centerline_f1"],
            _finite(historical_metrics.get("skeleton_f1"), "historical F1"),
            rel_tol=0.0,
            abs_tol=2e-7,
        ),
        "connectivity_score": math.isclose(
            overall["tile_macro"]["connectivity_score"],
            _finite(historical_metrics.get("connectivity_score"), "historical connectivity"),
            rel_tol=0.0,
            abs_tol=2e-7,
        ),
        "component_count_error": math.isclose(
            overall["tile_macro"]["component_count_error"],
            _finite(historical_metrics.get("component_count_error"), "historical component error"),
            rel_tol=0.0,
            abs_tol=2e-7,
        ),
    }
    if not all(reproduced.values()):
        raise ValueError(
            "independent tile-macro calculation does not reproduce the formal run: "
            f"checks={reproduced}, computed={overall['tile_macro']}, source={dict(historical_metrics)}"
        )

    by_domain = _stratify(evaluated, "source_domain")
    by_density = _stratify(evaluated, "density_stratum")
    by_source_size = _stratify(evaluated, "source_size_stratum")
    by_tile_coverage = _stratify(evaluated, "tile_coverage_stratum")
    sample_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in evaluated:
        sample_groups[str(row["sample_id"])].append(row)
    sample_summaries = {
        sample_id: summarize(rows) for sample_id, rows in sorted(sample_groups.items())
    }
    worst_samples = sorted(
        (
            {
                "sample_id": sample_id,
                "tiles": summary["tiles"],
                "target_skeleton_pixels": summary["target_skeleton_pixels"],
                **summary["micro"],
            }
            for sample_id, summary in sample_summaries.items()
            if summary["target_skeleton_pixels"] > 0
        ),
        key=lambda row: (float(row["skeleton_f1"]), str(row["sample_id"])),
    )[:10]
    weak_candidates: list[dict[str, Any]] = []
    for family, groups in (
        ("source_domain", by_domain),
        ("target_density", by_density),
        ("source_image_size", by_source_size),
        ("tile_coverage", by_tile_coverage),
    ):
        for name, result in groups.items():
            if result["target_skeleton_pixels"] <= 0:
                continue
            weak_candidates.append(
                {
                    "family": family,
                    "stratum": name,
                    "tiles": result["tiles"],
                    "samples": result["samples"],
                    "target_skeleton_pixels": result["target_skeleton_pixels"],
                    **result["micro"],
                }
            )
    weakest_strata = sorted(
        weak_candidates,
        key=lambda row: (float(row["skeleton_f1"]), str(row["family"]), str(row["stratum"])),
    )[:8]

    candidate_copy_check: dict[str, Any] | None = None
    lock_check: dict[str, Any] | None = None
    if candidate_artifact_path is not None:
        candidate_copy_check = {
            "path": _relative(candidate_artifact_path, workspace),
            "sha256": hashes["candidate_artifact"],
            "matches_formal_artifact": hashes["candidate_artifact"] == hashes["model_artifact"],
        }
        if not candidate_copy_check["matches_formal_artifact"]:
            raise ValueError("candidate Wire artifact differs from formal4 model.pt")
    if models_lock_path is not None:
        lock = _read_object(models_lock_path, "models.lock")
        wire_entry = _lock_wire_entry(lock)
        lock_check = {
            "path": _relative(models_lock_path, workspace),
            "sha256": hashes["models_lock"],
            "wire_status": wire_entry.get("status"),
            "wire_blessed": wire_entry.get("blessed"),
            "wire_artifact_sha256": wire_entry.get("sha256"),
            "matches_formal_artifact": wire_entry.get("sha256") == hashes["model_artifact"],
        }
        if not lock_check["matches_formal_artifact"]:
            raise ValueError("models.lock Wire hash differs from formal4 artifact")

    config_check: dict[str, Any] | None = None
    if training_config_path is not None:
        current_semantic_hash = _current_training_config_sha256(training_config_path, workspace)
        run_config_hash = _string(run_lineage.get("training_config_sha256"), "run config hash")
        config_check = {
            "path": _relative(training_config_path, workspace),
            "file_sha256": hashes["current_training_config"],
            "current_semantic_sha256": current_semantic_hash,
            "run_declared_semantic_sha256": run_config_hash,
            "exact_semantic_match": current_semantic_hash == run_config_hash,
            "exact_historical_config_snapshot_archived": False,
            "risk": (
                "The current baseline config has evolved since this run; the run binds its old "
                "semantic hash and gate/task subset in the model card, but not a full config snapshot."
            ),
        }

    runner_source = workspace / "src/pcbparse/training/torch_backend.py"
    runner_text = runner_source.read_text(encoding="utf-8")
    split_contract = {
        "train_dataset_explicit": 'WireTorchDataset(run_dir, source_root=source_root, split="train")' in runner_text,
        "val_dataset_explicit": 'WireTorchDataset(run_dir, source_root=source_root, split="val")' in runner_text,
        "test_dataset_constructed_in_wire_trainer": 'WireTorchDataset(run_dir, source_root=source_root, split="test")' in runner_text,
    }
    if not split_contract["train_dataset_explicit"] or not split_contract["val_dataset_explicit"]:
        raise ValueError("current Wire runner lacks explicit train/val split construction")
    if split_contract["test_dataset_constructed_in_wire_trainer"]:
        raise ValueError("current Wire trainer constructs a test dataset")

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pcbparse_wire_frozen_val_stratified_audit",
        "candidate": {
            "source_run": _relative(status_path.parent, workspace),
            "status": status["status"],
            "formal_evidence": True,
            "best_epoch": status.get("best_epoch"),
            "completed_epochs": status.get("completed_epochs"),
            "artifact_sha256": hashes["model_artifact"],
            "predictions_sha256": hashes["predictions"],
            "candidate_copy": candidate_copy_check,
            "models_lock_read_only_check": lock_check,
            "training_config_reproducibility": config_check,
        },
        "selection_boundary": {
            "fit_split_contract": "train",
            "evaluation_split": "val",
            "train_eligible_tile_rows_available_to_fit": len(eligible_by_split["train"]),
            "val_eligible_tile_rows_used": len(expected_val_ids),
            "test_eligible_tile_rows_available_but_excluded": len(eligible_by_split["test"]),
            "train_rows_used_for_metrics": 0,
            "test_rows_used": 0,
            "test_manifest_rows_inspected_only_for_leakage_audit": len(eligible_by_split["test"]),
            "prediction_rows_overlapping_train": len(set(prediction_by_id) & eligible_by_split["train"]),
            "prediction_rows_overlapping_test": len(set(prediction_by_id) & eligible_by_split["test"]),
            "threshold": 0.5,
            "tolerance_px": 1,
            "skeleton_iterations": 8,
            "thresholds_tuned": False,
            "test_used_for_tuning": False,
            "runner_split_contract": split_contract,
        },
        "metric_contract": {
            "historical": (
                "tile-macro: compute P/R/F1 independently per eligible validation tile, "
                "then average tiles equally"
            ),
            "corrected_primary": (
                "micro: sum tolerance-matched predicted/target skeleton hits and pixel "
                "denominators over the entire frozen validation population before P/R/F1"
            ),
            "connectivity": "tile-macro auxiliary; eight-connected component-count agreement",
            "empty_tile_convention": (
                "historical per-tile zero-denominator P or R is 1, so both-empty scores 1 "
                "and one-sided-empty F1 is 0; corrected micro uses global counts and treats "
                "a zero denominator as 1 only when the opposite global total is also zero"
            ),
            "density_bins": {
                "empty": "target skeleton pixels = 0",
                "very_sparse": "0 < target/valid <= 0.001",
                "sparse": "0.001 < target/valid <= 0.01",
                "dense": "target/valid > 0.01",
            },
            "source_size_bins": {
                "small_lt_0_5mp": "canvas pixels < 500000",
                "medium_0_5_to_1_5mp": "500000 <= canvas pixels < 1500000",
                "large_ge_1_5mp": "canvas pixels >= 1500000",
            },
        },
        "sources": {
            name: {"path": _relative(path, workspace), "sha256": hashes[name], "verified": True}
            for name, path in required.items()
        }
        | {
            "metric_implementation": {
                "path": _relative(workspace / "src/pcbparse/training/torch_wire_model.py", workspace),
                "sha256": _sha256(workspace / "src/pcbparse/training/torch_wire_model.py"),
                "verified": True,
            },
            "runner_implementation": {
                "path": _relative(runner_source, workspace),
                "sha256": _sha256(runner_source),
                "verified": True,
            },
        },
        "lineage_checks": {
            "materialization_run_id": run_id,
            "wire_tile_rows_by_split_eligibility": dict(sorted(tile_counts.items())),
            "prediction_population_exactly_equals_eligible_val": True,
            "prediction_duplicate_ids": 0,
            "missing_val_predictions": 0,
            "unexpected_predictions": 0,
            "historical_tile_macro_reproduction": reproduced,
        },
        "population": {
            "validation_tiles": len(evaluated),
            "validation_samples": len({row["sample_id"] for row in evaluated}),
            "empty_gt_tiles": overall["empty_target_tiles"],
            "nonempty_gt_tiles": len(evaluated) - overall["empty_target_tiles"],
            "empty_gt_tile_share": overall["empty_target_tiles"] / len(evaluated),
        },
        "historical_reported_tile_macro": {
            key: float(value)
            for key, value in historical_metrics.items()
            if isinstance(value, (int, float))
        },
        "overall_frozen_val": overall,
        "nonempty_gt_only_diagnostic": nonempty,
        "historical_vs_corrected": {
            "historical_tile_macro_f1": overall["tile_macro"]["centerline_f1"],
            "corrected_micro_f1": overall["micro"]["centerline_f1"],
            "absolute_f1_difference": (
                overall["micro"]["centerline_f1"]
                - overall["tile_macro"]["centerline_f1"]
            ),
        },
        "by_source_domain": by_domain,
        "by_target_density": by_density,
        "by_source_image_size": by_source_size,
        "by_tile_coverage": by_tile_coverage,
        "weakest_supported_strata": weakest_strata,
        "worst_nonempty_samples": worst_samples,
        "implementation_findings": [
            {
                "severity": "high",
                "finding": "formal Wire P/R/F1 used an unlabelled tile-macro aggregation",
                "evidence": {
                    "empty_gt_tiles": overall["empty_target_tiles"],
                    "validation_tiles": overall["tiles"],
                    "historical_tile_macro_f1": overall["tile_macro"]["centerline_f1"],
                    "corrected_micro_f1": overall["micro"]["centerline_f1"],
                },
                "remediation": (
                    "metric implementation now emits raw hit/denominator counters and the "
                    "trainer aggregates global micro P/R/F1; no existing weight or gate was changed"
                ),
            },
            {
                "severity": "medium",
                "finding": "the exact historical full training config was not archived beside the run",
                "evidence": config_check,
                "remediation": "archive a byte-for-byte config snapshot in every future formal bundle",
            },
        ],
        "competition_readiness": {
            "assessment": "not_proven_highly_usable",
            "pixel_centerline_gate_passed_historically": bool(status.get("gates_passed")),
            "reasons": [
                "Corrected micro and strata are validation-only; no official test score exists.",
                "Connectivity remains a tile-macro auxiliary and does not prove end-to-end graph correctness.",
                "Sparse/empty tile composition materially affects the historical headline metric.",
                "The weakest supported source/density strata remain below the aggregate result.",
            ],
        },
        "tile_diagnostics": sorted(evaluated, key=lambda row: str(row["record_id"])),
    }
    canonical = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    report["audit_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workspace = args.workspace_root.expanduser().resolve(strict=True)
    materialized_run = (workspace / args.materialized_run).resolve()
    try:
        materialized_run.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("materialized run must stay inside workspace") from exc
    if not materialized_run.is_dir():
        raise FileNotFoundError(f"materialized run is missing: {materialized_run}")
    paths = {
        "predictions_path": _resolve_within(workspace, args.predictions, "predictions"),
        "status_path": _resolve_within(workspace, args.run_status, "run status"),
        "metrics_path": _resolve_within(workspace, args.run_metrics, "run metrics"),
        "model_path": _resolve_within(workspace, args.model_artifact, "model artifact"),
        "wire_manifest_path": _resolve_within(workspace, args.wire_manifest, "wire manifest"),
        "source_manifest_path": _resolve_within(workspace, args.source_manifest, "source manifest"),
        "split_manifest_path": _resolve_within(workspace, args.split_manifest, "split manifest"),
        "candidate_artifact_path": _resolve_within(
            workspace, args.candidate_artifact, "candidate artifact"
        ),
        "models_lock_path": _resolve_within(workspace, args.models_lock, "models lock"),
        "training_config_path": _resolve_within(
            workspace, args.training_config, "training config"
        ),
    }
    report = build_audit(
        workspace=workspace,
        materialized_run=materialized_run,
        **paths,
    )
    output = _resolve_within(workspace, args.output, "audit output", must_exist=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(output)
    print(f"audit_payload_sha256={report['audit_payload_sha256']}")
    print(json.dumps(report["historical_vs_corrected"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
