from __future__ import annotations

"""Strictly isolate learned-Junction impact on saved predicted-Wire val graphs.

This audit is diagnostic-only.  It consumes only hash-bound frozen-validation
Wire masks, replays the locked Junction checkpoint on CPU, and compares two
otherwise identical graph inputs:

* deterministic Composite vectorizer junctions only; and
* those same junctions merged with learned Junction decisions.

No missing Wire tile is filled from ground truth, no threshold is searched,
and no train/test row contributes to a metric.
"""

import argparse
import base64
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


# Prevent an audit invocation from acquiring a GPU even when CUDA is present.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPT_ROOT = REPOSITORY_ROOT / "scripts"
for search_root in (SOURCE_ROOT, SCRIPT_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from audit_val_graph_oracle_decomposition import (  # noqa: E402
    MATERIALIZATION_RUN_ID,
    _aggregate_variant,
    _atomic_write,
    _canonical_bytes,
    _expect,
    _index,
    _iter_jsonl,
    _mapping,
    _model_entry,
    _payload_sha256,
    _pin_pairs,
    _read_object,
    _relative,
    _resolve,
    _resolve_locked_artifact,
    _resolve_output,
    _score_pair_partition,
    _sequence,
    _sha256,
    _source,
    _string,
    _strip_oracle_membership,
    _validate_source_file,
)
from pcbparse.graph.physical import build_physical_graph  # noqa: E402
from pcbparse.io import crop_image_view, load_image  # noqa: E402
from pcbparse.ir.models import BBox, CircuitIR, Evidence, Junction, Point, WireSegment  # noqa: E402
from pcbparse.perception.composite_provider import (  # noqa: E402
    CompositeProviderConfig,
    _assign_global_ids,
    _merge_junctions,
    _restore_junctions,
    _vectorize_masks,
)
from pcbparse.perception.golden_provider import DatasetGoldenProvider  # noqa: E402
from pcbparse.perception.protocols import WireMask  # noqa: E402
from pcbparse.perception.torch_junction_adapter import TorchJunctionProvider  # noqa: E402


AUDIT_VERSION = "val-predicted-wire-junction-isolation-v1"
CONNECTIVE_JUNCTION_KINDS = frozenset({"t_junction", "connected_cross", "bend"})


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
        "--junction-replay-checkpoint",
        type=Path,
        default=Path("work/train-junction-formal/model.pt"),
    )
    parser.add_argument("--models-lock", type=Path, default=Path("models/models.lock.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/val_predicted_wire_junction_isolation.json"),
    )
    parser.add_argument(
        "--determinism-replays",
        type=int,
        default=2,
        help="Independent CPU audit builds that must be byte-identical before attestation.",
    )
    return parser


def decode_prediction_mask(row: Mapping[str, Any]) -> np.ndarray:
    """Strictly decode one little-bit-order packed binary mask."""

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
            f"prediction packed byte count mismatch: expected={expected_bytes}, "
            f"actual={len(packed)}"
        )
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")
    if bool(bits[height * width :].any()):
        raise ValueError("prediction mask has non-zero padding bits")
    return bits[: height * width].reshape(height, width).astype(bool)


def _tile_bbox(row: Mapping[str, Any], label: str) -> tuple[int, int, int, int]:
    raw = _mapping(row.get("tile_bbox"), f"{label} tile_bbox")
    values: list[int] = []
    for name in ("x1", "y1", "x2", "y2"):
        value = raw.get(name)
        if type(value) is not int:
            raise ValueError(f"{label} tile_bbox.{name} must be an integer")
        values.append(value)
    x1, y1, x2, y2 = values
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError(f"{label} tile_bbox is invalid")
    return x1, y1, x2, y2


def _strict_complete_population(
    val_tiles: Sequence[Mapping[str, Any]],
    prediction_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Require saved predictions to cover whole samples, never partial boards."""

    all_by_sample: dict[str, set[str]] = defaultdict(set)
    eligible_by_sample: dict[str, set[str]] = defaultdict(set)
    for row in val_tiles:
        record_id = _string(row.get("record_id"), "val tile record_id")
        sample_id = _string(row.get("sample_id"), f"val tile {record_id} sample_id")
        all_by_sample[sample_id].add(record_id)
        supervision = _mapping(row.get("supervision"), f"val tile {record_id} supervision")
        if supervision.get("eligible") is True:
            eligible_by_sample[sample_id].add(record_id)

    predicted_ids = set(prediction_index)
    eligible_ids = set().union(*eligible_by_sample.values()) if eligible_by_sample else set()
    _expect(predicted_ids, eligible_ids, "saved Wire prediction/eligible-val population")
    predicted_by_sample: dict[str, set[str]] = defaultdict(set)
    for record_id, prediction in prediction_index.items():
        sample_id = _string(prediction.get("sample_id"), f"prediction {record_id} sample_id")
        predicted_by_sample[sample_id].add(record_id)

    partial = sorted(
        sample_id
        for sample_id, record_ids in predicted_by_sample.items()
        if record_ids != all_by_sample.get(sample_id, set())
    )
    if partial:
        raise ValueError(
            "strict Junction isolation refuses partial predicted-Wire samples: "
            + ", ".join(partial)
        )
    covered = sorted(predicted_by_sample)
    excluded = sorted(set(all_by_sample) - set(covered))
    return {
        "covered_sample_ids": covered,
        "excluded_sample_ids": excluded,
        "covered_samples": len(covered),
        "excluded_samples": len(excluded),
        "total_val_samples": len(all_by_sample),
        "covered_tiles": len(predicted_ids),
        "excluded_tiles": sum(len(all_by_sample[item]) for item in excluded),
        "total_val_tiles": sum(len(item) for item in all_by_sample.values()),
        "all_covered_samples_have_every_val_tile": True,
        "per_sample_tile_counts": {
            sample_id: {
                "total": len(all_by_sample[sample_id]),
                "eligible": len(eligible_by_sample.get(sample_id, set())),
                "predicted": len(predicted_by_sample.get(sample_id, set())),
            }
            for sample_id in sorted(all_by_sample)
        },
    }


def _pair_relation_sha256(pairs: Iterable[tuple[str, str]]) -> str:
    return hashlib.sha256(_canonical_bytes([list(item) for item in sorted(pairs)])).hexdigest()


def _evaluate_ir(
    ir: CircuitIR,
    reference_groups: Iterable[Iterable[str]],
) -> dict[str, Any]:
    candidate = ir.model_copy(deep=True)
    nets, decisions = build_physical_graph(candidate)
    reference_pairs = _pin_pairs(reference_groups)
    predicted_pairs = _pin_pairs(net.pins for net in nets)
    result = _score_pair_partition(reference_pairs, predicted_pairs)
    result.update(
        {
            "predicted_pair_relation_sha256": _pair_relation_sha256(predicted_pairs),
            "physical_net_count": len(nets),
            "pin_bearing_net_count": sum(bool(net.pins) for net in nets),
            "segment_only_net_count": sum(
                bool(net.segments) and not net.pins for net in nets
            ),
            "snap_status_counts": dict(
                sorted(Counter(item.status for item in decisions).items())
            ),
        }
    )
    return result


def _nonjunction_ir_sha256(ir: CircuitIR) -> str:
    payload = ir.model_dump(mode="json")
    for field in ("junctions", "physical_nets", "electrical_nets"):
        payload.pop(field, None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _segment_geometry_sha256(segments: Sequence[WireSegment]) -> str:
    rows = [
        {
            "internal_id": item.internal_id,
            "start": [item.start.x, item.start.y],
            "end": [item.end.x, item.end.y],
            "channel": item.channel,
            "source_polyline_id": item.source_polyline_id,
        }
        for item in sorted(segments, key=lambda value: value.internal_id)
    ]
    return hashlib.sha256(_canonical_bytes(rows)).hexdigest()


def _construct_variants(
    golden_ir: CircuitIR,
    segments: Sequence[WireSegment],
    vectorized_junctions: Sequence[Junction],
    learned_junctions: Sequence[Junction],
    *,
    tolerance: float,
) -> tuple[CircuitIR, CircuitIR]:
    """Build two deep, non-Junction-identical IRs for the isolated comparison."""

    base = _strip_oracle_membership(golden_ir)
    base.segments = [item.model_copy(deep=True) for item in segments]
    base.junctions = []
    vectorized_only = base.model_copy(deep=True)
    vectorized_only.junctions = [
        item.model_copy(deep=True) for item in vectorized_junctions
    ]
    _assign_global_ids(vectorized_only.junctions, "junction")

    vectorized_plus_learned = base.model_copy(deep=True)
    vectorized_plus_learned.junctions = _merge_junctions(
        [item.model_copy(deep=True) for item in vectorized_junctions],
        [item.model_copy(deep=True) for item in learned_junctions],
        tolerance=tolerance,
    )
    _assign_global_ids(vectorized_plus_learned.junctions, "junction")
    without_hash = _nonjunction_ir_sha256(vectorized_only)
    with_hash = _nonjunction_ir_sha256(vectorized_plus_learned)
    _expect(with_hash, without_hash, "non-Junction comparison input SHA-256")
    return vectorized_only, vectorized_plus_learned


def _kind_counts(junctions: Sequence[Junction]) -> dict[str, int]:
    return dict(sorted(Counter(item.kind for item in junctions).items()))


def _junction_summary(junctions: Sequence[Junction]) -> dict[str, Any]:
    kinds = Counter(item.kind for item in junctions)
    return {
        "count": len(junctions),
        "kind_counts": dict(sorted(kinds.items())),
        "connective_count": sum(
            count for kind, count in kinds.items() if kind in CONNECTIVE_JUNCTION_KINDS
        ),
        "nonconnective_or_ambiguous_count": sum(
            count for kind, count in kinds.items() if kind not in CONNECTIVE_JUNCTION_KINDS
        ),
    }


def _delta(without: Mapping[str, Any], with_learned: Mapping[str, Any]) -> dict[str, Any]:
    f1_delta = float(with_learned["f1"]) - float(without["f1"])
    return {
        "tp": int(with_learned["tp"]) - int(without["tp"]),
        "fp": int(with_learned["fp"]) - int(without["fp"]),
        "fn": int(with_learned["fn"]) - int(without["fn"]),
        "predicted_pairs": int(with_learned["predicted_pairs"])
        - int(without["predicted_pairs"]),
        "precision": float(with_learned["precision"]) - float(without["precision"]),
        "recall": float(with_learned["recall"]) - float(without["recall"]),
        "f1": f1_delta,
        "pair_relation_changed": (
            without["predicted_pair_relation_sha256"]
            != with_learned["predicted_pair_relation_sha256"]
        ),
        "outcome_by_f1": (
            "improved" if f1_delta > 1e-12 else "degraded" if f1_delta < -1e-12 else "unchanged"
        ),
    }


def _aggregate_by_domain(
    rows: Sequence[Mapping[str, Any]], variant: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_string(row.get("source_domain"), "per-sample source_domain")].append(
            _mapping(_mapping(row.get("variants"), "per-sample variants").get(variant), variant)
        )
    return {
        domain: _aggregate_variant(values) for domain, values in sorted(grouped.items())
    }


def _aggregate_junction_population(
    rows: Sequence[Mapping[str, Any]], variant: str
) -> dict[str, Any]:
    kinds: Counter[str] = Counter()
    count = 0
    connective = 0
    nonconnective = 0
    for row in rows:
        summary = _mapping(
            _mapping(row.get("junctions"), "per-sample junctions").get(variant),
            f"junctions.{variant}",
        )
        count += int(summary["count"])
        connective += int(summary["connective_count"])
        nonconnective += int(summary["nonconnective_or_ambiguous_count"])
        kinds.update(_mapping(summary.get("kind_counts"), f"junctions.{variant}.kinds"))
    return {
        "count": count,
        "kind_counts": {name: int(value) for name, value in sorted(kinds.items())},
        "connective_count": connective,
        "nonconnective_or_ambiguous_count": nonconnective,
    }


def _sample_macro_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("sample macro summary requires at least one row")
    base = np.asarray(
        [
            float(row["variants"]["predicted_wire_vectorized_only"]["f1"])
            for row in rows
        ],
        dtype=np.float64,
    )
    learned = np.asarray(
        [
            float(row["variants"]["predicted_wire_vectorized_plus_learned"]["f1"])
            for row in rows
        ],
        dtype=np.float64,
    )
    delta = learned - base
    return {
        "samples": len(rows),
        "vectorized_only_mean_f1": float(np.mean(base)),
        "vectorized_plus_learned_mean_f1": float(np.mean(learned)),
        "mean_paired_f1_delta": float(np.mean(delta)),
        "median_paired_f1_delta": float(np.median(delta)),
        "minimum_paired_f1_delta": float(np.min(delta)),
        "maximum_paired_f1_delta": float(np.max(delta)),
        "weighting": "each sample has equal weight; exact duplicate boards remain separate rows",
    }


def _validate_materialization(
    *,
    summary: Mapping[str, Any],
    materialized_run: Path,
    hashes: Mapping[str, str],
) -> None:
    _expect(summary.get("kind"), "wire_junction_materialization", "materialization kind")
    _expect(summary.get("complete"), True, "materialization complete")
    _expect(summary.get("run_id"), materialized_run.name, "materialization run_id")
    inputs = _mapping(
        _mapping(summary.get("lineage"), "materialization lineage").get("inputs"),
        "materialization lineage inputs",
    )
    for key, hash_key in (
        ("source_manifest_sha256", "source_manifest"),
        ("split_manifest_sha256", "split_manifest"),
        ("wire_manifest_sha256", "wire_manifest"),
        ("junction_manifest_sha256", "junction_manifest"),
    ):
        _expect(inputs.get(key), hashes[hash_key], f"materialization {key}")
    outputs = _mapping(summary.get("outputs"), "materialization outputs")
    _expect(
        _mapping(outputs.get("wire_tiles"), "summary wire_tiles").get("sha256"),
        hashes["wire_tiles"],
        "summary wire_tiles SHA-256",
    )


def _validate_run_lineage(
    status: Mapping[str, Any],
    *,
    task: str,
    materialized_run: Path,
    hashes: Mapping[str, str],
    manifest_hash_key: str,
) -> None:
    _expect(status.get("task"), task, f"{task} run task")
    _expect(status.get("formal_evidence"), True, f"{task} formal_evidence")
    _expect(status.get("smoke"), False, f"{task} smoke")
    lineage = _mapping(status.get("lineage"), f"{task} lineage")
    for key, expected in (
        ("binding_sha256", materialized_run.name),
        ("materialization_commit_id", materialized_run.name),
        ("materialization_commit_sha256", hashes["summary"]),
        ("source_manifest_sha256", hashes["source_manifest"]),
        ("split_manifest_sha256", hashes["split_manifest"]),
        ("task_manifest_sha256", hashes[manifest_hash_key]),
    ):
        _expect(lineage.get(key), expected, f"{task} lineage {key}")


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
    junction_status_path: Path,
    junction_formal_model_path: Path,
    junction_replay_checkpoint_path: Path,
    models_lock_path: Path,
) -> dict[str, Any]:
    summary_path = materialized_run / "summary.json"
    wire_tiles_path = materialized_run / "wire_tiles.jsonl"
    for path, label in (
        (summary_path, "materialization summary"),
        (wire_tiles_path, "materialized Wire tiles"),
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
        "wire_predictions": _sha256(wire_predictions_path),
        "wire_status": _sha256(wire_status_path),
        "wire_model": _sha256(wire_model_path),
        "junction_status": _sha256(junction_status_path),
        "junction_formal_model": _sha256(junction_formal_model_path),
        "junction_replay_checkpoint": _sha256(junction_replay_checkpoint_path),
        "models_lock": _sha256(models_lock_path),
    }
    summary = _read_object(summary_path, "materialization summary")
    _validate_materialization(summary=summary, materialized_run=materialized_run, hashes=hashes)

    split_index = _index(
        _iter_jsonl(split_manifest_path, "split manifest"),
        "sample_id",
        "split manifest",
    )
    split_by_sample = {
        sample_id: _string(row.get("split"), f"split {sample_id}")
        for sample_id, row in split_index.items()
    }
    val_sample_ids = sorted(
        sample_id for sample_id, split in split_by_sample.items() if split == "val"
    )
    if not val_sample_ids:
        raise ValueError("frozen validation split is empty")
    source_index = _index(
        _iter_jsonl(source_manifest_path, "source manifest"),
        "sample_id",
        "source manifest",
    )
    missing_sources = sorted(set(val_sample_ids) - set(source_index))
    if missing_sources:
        raise ValueError(f"val samples missing from source manifest: {missing_sources}")

    tile_index = _index(
        _iter_jsonl(wire_tiles_path, "materialized Wire tiles"),
        "record_id",
        "materialized Wire tiles",
    )
    val_tiles = [
        row
        for row in tile_index.values()
        if _string(row.get("split"), "Wire tile split") == "val"
    ]
    for record_id, row in tile_index.items():
        sample_id = _string(row.get("sample_id"), f"Wire tile {record_id} sample_id")
        _expect(
            row.get("split"),
            split_by_sample.get(sample_id),
            f"Wire tile {record_id} split binding",
        )
    prediction_index = _index(
        _iter_jsonl(wire_predictions_path, "saved Wire predictions"),
        "record_id",
        "saved Wire predictions",
    )
    for record_id, prediction in prediction_index.items():
        tile = tile_index.get(record_id)
        if tile is None:
            raise ValueError(f"Wire prediction references unknown tile {record_id}")
        _expect(
            prediction.get("sample_id"),
            tile.get("sample_id"),
            f"Wire prediction {record_id} sample binding",
        )
    population = _strict_complete_population(val_tiles, prediction_index)

    models_lock = _read_object(models_lock_path, "models.lock")
    wire_lock = _model_entry(models_lock, "wire")
    junction_lock = _model_entry(models_lock, "junction")
    locked_wire_path = _resolve_locked_artifact(
        workspace, models_lock_path, wire_lock, "locked Wire"
    )
    locked_junction_path = _resolve_locked_artifact(
        workspace, models_lock_path, junction_lock, "locked Junction"
    )
    _expect(hashes["wire_model"], wire_lock.get("sha256"), "formal Wire/model.lock SHA-256")
    _expect(
        hashes["junction_replay_checkpoint"],
        junction_lock.get("sha256"),
        "Junction replay/model.lock SHA-256",
    )

    wire_status = _read_object(wire_status_path, "Wire run status")
    _validate_run_lineage(
        wire_status,
        task="wire",
        materialized_run=materialized_run,
        hashes=hashes,
        manifest_hash_key="wire_manifest",
    )
    _expect(
        _mapping(wire_status.get("artifact"), "Wire artifact").get("sha256"),
        hashes["wire_model"],
        "Wire run artifact SHA-256",
    )
    _expect(
        _mapping(wire_status.get("predictions"), "Wire predictions").get("sha256"),
        hashes["wire_predictions"],
        "Wire run prediction SHA-256",
    )
    junction_status = _read_object(junction_status_path, "Junction run status")
    _validate_run_lineage(
        junction_status,
        task="junction",
        materialized_run=materialized_run,
        hashes=hashes,
        manifest_hash_key="junction_manifest",
    )
    _expect(
        _mapping(junction_status.get("artifact"), "Junction artifact").get("sha256"),
        hashes["junction_formal_model"],
        "Junction formal artifact SHA-256",
    )

    replay_entry = deepcopy(dict(junction_lock))
    replay_entry["device"] = "cpu"
    inference = dict(_mapping(replay_entry.get("inference", {}), "Junction inference"))
    inference["device"] = "cpu"
    replay_entry["inference"] = inference
    junction_provider = TorchJunctionProvider(locked_junction_path, replay_entry)
    if junction_provider.device != "cpu":
        raise RuntimeError(f"Junction replay must remain on CPU, got {junction_provider.device!r}")

    config = CompositeProviderConfig()
    vectorized_rows: list[dict[str, Any]] = []
    learned_rows: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    source_bindings: list[dict[str, str]] = []
    input_bindings: list[dict[str, Any]] = []
    duplicate_groups: dict[str, list[str]] = defaultdict(list)
    provider = DatasetGoldenProvider()
    tiles_by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record_id in prediction_index:
        tiles_by_sample[_string(prediction_index[record_id].get("sample_id"), "prediction sample")].append(
            tile_index[record_id]
        )

    for sample_id in population["covered_sample_ids"]:
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
        source_bindings.append(
            {"sample_id": sample_id, "image_sha256": image_sha, "json_sha256": json_sha}
        )
        original_view = load_image(image_path)
        golden_ir = provider.build_ir(image_path)
        _expect(golden_ir.image.image_id, sample_id, f"Golden IR {sample_id} image_id")
        hints = _mapping(
            golden_ir.metadata.get("dataset_graph_hints"), f"{sample_id} graph hints"
        )
        _expect(hints.get("authoritative"), False, f"{sample_id} graph hint authority")
        _expect(hints.get("rebuild_required"), True, f"{sample_id} graph rebuild flag")
        reference_groups: list[list[str]] = []
        pin_ids = {pin.internal_id for pin in golden_ir.pins}
        for position, raw_hint in enumerate(
            _sequence(hints.get("physical_nets"), f"{sample_id} physical hints")
        ):
            hint = _mapping(raw_hint, f"{sample_id} physical hint {position}")
            group = [
                str(item) for item in _sequence(hint.get("pin_ids"), "hint pin_ids")
            ]
            unknown = sorted(set(group) - pin_ids)
            if unknown:
                raise ValueError(f"{sample_id} reference hint has unknown pins: {unknown}")
            reference_groups.append(group)

        views = []
        masks = []
        prediction_binding_rows = []
        for tile in sorted(
            tiles_by_sample[sample_id], key=lambda item: _string(item.get("record_id"), "tile id")
        ):
            record_id = _string(tile.get("record_id"), "tile record_id")
            prediction = prediction_index[record_id]
            x1, y1, x2, y2 = _tile_bbox(tile, record_id)
            if x2 > original_view.width or y2 > original_view.height:
                raise ValueError(f"tile {record_id} escapes source image")
            view = crop_image_view(
                original_view,
                BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                view_id=record_id,
            )
            mask = decode_prediction_mask(prediction)
            if mask.shape != (view.height, view.width):
                raise ValueError(
                    f"prediction {record_id} shape {mask.shape} differs from source crop "
                    f"{(view.height, view.width)}"
                )
            evidence = Evidence(
                evidence_id=f"saved-wire-prediction:{record_id}",
                source="detector",
                view_id=view.view_id,
                score=1.0,
                payload_ref=_relative(wire_predictions_path, workspace),
                metadata={
                    "task": "wire",
                    "artifact_sha256": hashes["wire_model"],
                    "prediction_file_sha256": hashes["wire_predictions"],
                    "saved_binary_mask": True,
                    "confidence_not_persisted": True,
                },
            )
            views.append(view)
            masks.append(WireMask(view_id=view.view_id, mask=mask, score=1.0, evidence=[evidence]))
            prediction_binding_rows.append(
                {
                    "record_id": record_id,
                    "tile_bbox": [x1, y1, x2, y2],
                    "foreground_pixels": int(mask.sum()),
                }
            )

        view_index = {item.view_id: item for item in views}
        segments, vectorized_junctions = _vectorize_masks(masks, view_index, config)
        learned_local = junction_provider.classify(views, segments)
        learned_junctions = _restore_junctions(learned_local, view_index)
        vectorized_only_ir, vectorized_plus_learned_ir = _construct_variants(
            golden_ir,
            segments,
            vectorized_junctions,
            learned_junctions,
            tolerance=config.wire_tile_dedup_tolerance_px,
        )
        vectorized_metric = _evaluate_ir(vectorized_only_ir, reference_groups)
        learned_metric = _evaluate_ir(vectorized_plus_learned_ir, reference_groups)
        vectorized_rows.append(vectorized_metric)
        learned_rows.append(learned_metric)
        domain = _string(source_row.get("source_class"), f"source {sample_id} domain")
        duplicate_group = _string(
            source_row.get("duplicate_group"), f"source {sample_id} duplicate_group"
        )
        duplicate_groups[duplicate_group].append(sample_id)
        sample_delta = _delta(vectorized_metric, learned_metric)
        nonjunction_hash = _nonjunction_ir_sha256(vectorized_only_ir)
        segment_hash = _segment_geometry_sha256(segments)
        input_bindings.append(
            {
                "sample_id": sample_id,
                "prediction_tiles": prediction_binding_rows,
                "nonjunction_ir_sha256": nonjunction_hash,
                "segment_geometry_sha256": segment_hash,
            }
        )
        per_sample.append(
            {
                "sample_id": sample_id,
                "source_domain": domain,
                "duplicate_group": duplicate_group,
                "tile_count": len(views),
                "predicted_wire_foreground_pixels": sum(
                    item["foreground_pixels"] for item in prediction_binding_rows
                ),
                "predicted_segment_count": len(segments),
                "segment_geometry_sha256": segment_hash,
                "nonjunction_ir_sha256_both_variants": nonjunction_hash,
                "junctions": {
                    "vectorized_only": _junction_summary(vectorized_only_ir.junctions),
                    "learned_raw": _junction_summary(learned_junctions),
                    "vectorized_plus_learned_merged": _junction_summary(
                        vectorized_plus_learned_ir.junctions
                    ),
                },
                "reference_group_count": len(reference_groups),
                "reference_pair_count": len(_pin_pairs(reference_groups)),
                "variants": {
                    "predicted_wire_vectorized_only": vectorized_metric,
                    "predicted_wire_vectorized_plus_learned": learned_metric,
                },
                "learned_junction_delta": sample_delta,
            }
        )

    aggregates = {
        "predicted_wire_vectorized_only": _aggregate_variant(vectorized_rows),
        "predicted_wire_vectorized_plus_learned": _aggregate_variant(learned_rows),
    }
    aggregate_delta = _delta(
        {
            **aggregates["predicted_wire_vectorized_only"],
            "predicted_pair_relation_sha256": _pair_relation_sha256(
                (row["sample_id"], row["variants"]["predicted_wire_vectorized_only"]["predicted_pair_relation_sha256"])
                for row in per_sample
            ),
        },
        {
            **aggregates["predicted_wire_vectorized_plus_learned"],
            "predicted_pair_relation_sha256": _pair_relation_sha256(
                (row["sample_id"], row["variants"]["predicted_wire_vectorized_plus_learned"]["predicted_pair_relation_sha256"])
                for row in per_sample
            ),
        },
    )
    outcomes = Counter(
        _mapping(row.get("learned_junction_delta"), "sample delta").get("outcome_by_f1")
        for row in per_sample
    )
    changed_relations = sum(
        bool(_mapping(row.get("learned_junction_delta"), "sample delta").get("pair_relation_changed"))
        for row in per_sample
    )
    source_domains = Counter(row["source_domain"] for row in per_sample)
    sample_macro = _sample_macro_summary(per_sample)
    ranked_reference_support = sorted(
        (
            {
                "sample_id": row["sample_id"],
                "reference_pairs": int(row["reference_pair_count"]),
            }
            for row in per_sample
        ),
        key=lambda item: (-item["reference_pairs"], item["sample_id"]),
    )
    reference_pair_total = sum(item["reference_pairs"] for item in ranked_reference_support)
    largest_tile_sample = max(
        per_sample, key=lambda item: (int(item["tile_count"]), item["sample_id"])
    )

    source_paths = {
        "source_manifest": source_manifest_path,
        "split_manifest": split_manifest_path,
        "wire_manifest": wire_manifest_path,
        "junction_manifest": junction_manifest_path,
        "materialization_summary": summary_path,
        "wire_tiles": wire_tiles_path,
        "wire_predictions": wire_predictions_path,
        "wire_run_status": wire_status_path,
        "wire_formal_model": wire_model_path,
        "wire_locked_candidate": locked_wire_path,
        "junction_run_status": junction_status_path,
        "junction_formal_model": junction_formal_model_path,
        "junction_replay_checkpoint": junction_replay_checkpoint_path,
        "junction_locked_candidate": locked_junction_path,
        "models_lock": models_lock_path,
        "audit_script": Path(__file__).resolve(),
        "base_graph_audit_helpers": SCRIPT_ROOT / "audit_val_graph_oracle_decomposition.py",
        "composite_provider": SOURCE_ROOT / "pcbparse/perception/composite_provider.py",
        "wire_vectorizer": SOURCE_ROOT / "pcbparse/perception/wire_vectorizer.py",
        "junction_adapter": SOURCE_ROOT / "pcbparse/perception/torch_junction_adapter.py",
        "physical_graph_builder": SOURCE_ROOT / "pcbparse/graph/physical.py",
        "golden_provider": SOURCE_ROOT / "pcbparse/perception/golden_provider.py",
    }
    sources = {name: _source(path, workspace) for name, path in source_paths.items()}

    report: dict[str, Any] = {
        "kind": "pcbparse_frozen_val_predicted_wire_junction_isolation_audit",
        "schema_version": 1,
        "audit_version": AUDIT_VERSION,
        "selection_boundary": {
            "evaluation_split": "val",
            "total_frozen_val_samples": population["total_val_samples"],
            "covered_val_samples_used": population["covered_samples"],
            "excluded_val_samples": population["excluded_samples"],
            "val_prediction_tiles_used": population["covered_tiles"],
            "train_rows_used": 0,
            "test_rows_used": 0,
            "test_labels_used": 0,
            "thresholds_tuned": False,
            "test_used_for_tuning": False,
            "wire_inference_rerun": False,
            "junction_inference_device": junction_provider.device,
            "junction_score_threshold_source": "locked adapter default/contract; not searched",
            "junction_score_threshold": junction_provider.config.score_threshold,
        },
        "sources": sources,
        "population": population,
        "population_concentration": {
            "largest_tile_sample_id": largest_tile_sample["sample_id"],
            "largest_tile_sample_tiles": largest_tile_sample["tile_count"],
            "largest_tile_sample_share": float(
                largest_tile_sample["tile_count"] / population["covered_tiles"]
            ),
            "largest_reference_pair_sample": ranked_reference_support[0],
            "largest_reference_pair_sample_share": float(
                ranked_reference_support[0]["reference_pairs"] / reference_pair_total
            ),
            "top_three_reference_pair_share": float(
                sum(item["reference_pairs"] for item in ranked_reference_support[:3])
                / reference_pair_total
            ),
            "covered_duplicate_groups": len(duplicate_groups),
            "duplicate_groups_with_multiple_covered_samples": {
                group: sorted(samples)
                for group, samples in sorted(duplicate_groups.items())
                if len(samples) > 1
            },
        },
        "source_file_binding": {
            "covered_source_files_verified": len(source_bindings) * 2,
            "binding_sha256": hashlib.sha256(_canonical_bytes(source_bindings)).hexdigest(),
            "source_domain_counts": dict(sorted(source_domains.items())),
        },
        "comparison_input_binding": {
            "sha256": hashlib.sha256(_canonical_bytes(input_bindings)).hexdigest(),
            "samples": input_bindings,
        },
        "comparison_contract": {
            "strict_isolation": True,
            "only_changed_field": "CircuitIR.junctions",
            "shared_inputs": [
                "same saved binary predicted-Wire masks",
                "same Composite _vectorize_masks output segments",
                "same strict DatasetGoldenProvider components/pins/text",
                "same hidden-membership stripping",
                "same build_physical_graph defaults",
            ],
            "vectorized_only": (
                "Composite deterministic skeleton endpoint/T candidates; ambiguous four-way "
                "crosses remain non-connective"
            ),
            "vectorized_plus_learned": (
                "same deterministic candidates merged with locked Junction checkpoint output "
                "using Composite model-first semantic ownership"
            ),
            "nonjunction_ir_hash_checked_equal_per_sample": True,
            "component_pin_conditioning": (
                "GT Dataset300 components and pin points are used identically in both arms; this "
                "isolates Junction but is not production predicted-Component/Pin E2E"
            ),
            "wire_population": (
                "formal4 saved 0.5-threshold binary masks on materialized 512px non-overlap views"
            ),
            "production_tiling_gap": (
                "saved masks are not available on current Composite 1024px/160px-overlap views; "
                "therefore this is exact for the saved formal population, not exact production tiling"
            ),
            "saved_wire_confidence_gap": (
                "predictions.jsonl stores binary masks but not mask confidence; audit assigns the "
                "same neutral score=1.0 in both arms, which does not change graph connectivity"
            ),
            "composite_vectorizer_config": config.to_dict(),
            "graph_builder_kwargs": {},
        },
        "reference_contract": {
            "provider": "DatasetGoldenProvider(dataset300-golden-v2, strict)",
            "reference": "metadata.dataset_graph_hints.physical_nets pin_ids",
            "authoritative": False,
            "metric": "unordered pairwise pin co-connectivity, micro-aggregated",
            "metric_limit": (
                "pairwise relations do not score singleton-only partitions or segment-only nets"
            ),
            "official_evaluator_truth": False,
        },
        "variants": {
            "aggregate": aggregates,
            "sample_macro": sample_macro,
            "junction_population": {
                name: _aggregate_junction_population(per_sample, name)
                for name in (
                    "vectorized_only",
                    "learned_raw",
                    "vectorized_plus_learned_merged",
                )
            },
            "by_source_domain": {
                "predicted_wire_vectorized_only": _aggregate_by_domain(
                    per_sample, "predicted_wire_vectorized_only"
                ),
                "predicted_wire_vectorized_plus_learned": _aggregate_by_domain(
                    per_sample, "predicted_wire_vectorized_plus_learned"
                ),
            },
            "per_sample": per_sample,
        },
        "isolated_learned_junction_effect": {
            "aggregate_delta_with_learned_minus_vectorized_only": aggregate_delta,
            "sample_macro_f1": sample_macro,
            "sample_outcome_counts_by_f1": dict(sorted(outcomes.items())),
            "samples_with_changed_pair_relation": changed_relations,
            "samples_total": len(per_sample),
            "causal_scope": (
                "Within this fixed 18-sample saved-Wire/GT-Pin/current-builder replay, any arm "
                "difference is caused by learned Junction integration. Remaining baseline error "
                "cannot be uniquely divided between predicted Wire, GT-Pin geometry association, "
                "and graph builder without additional oracles."
            ),
        },
        "junction_cpu_replay": {
            "device_requested": "cpu",
            "device_actual": junction_provider.device,
            "model_version": junction_provider.version,
            "model_sha256": junction_provider.artifact_sha256,
            "models_lock_matches": junction_provider.artifact_sha256
            == junction_lock.get("sha256"),
            "models_lock_status": junction_lock.get("status"),
            "models_lock_blessed": junction_lock.get("blessed"),
            "formal_run_status": junction_status.get("status"),
            "formal_gates_passed": junction_status.get("gates_passed"),
            "adapter_config": asdict(junction_provider.config),
        },
        "inference_boundaries": [
            "No metric is reported for the 12 frozen-val samples whose every Wire tile is supervision-ineligible and has no saved prediction.",
            "The 18-sample estimate must not be extrapolated to all 30 validation samples or official hidden test data.",
            "Micro pair-F1 and equal-sample macro-F1 answer different weighting questions; both are reported because support is highly uneven.",
            "Covered samples 0089 and 0283 are an exact duplicate-image/annotation group and remain separate rows, matching the frozen manifest population.",
            "Dataset graph hints explicitly declare authoritative=false; metrics are debug/oracle evidence, not official score evidence.",
            "This comparison isolates Junction integration but does not identify the absolute predicted-Wire or graph-builder share of remaining error.",
            "The locked Junction candidate is unblessed and its formal six-class gate failed.",
        ],
        "promotion_policy": {
            "diagnostic_only": True,
            "promotion_eligible": False,
            "production_blessing": False,
            "models_lock_modified": False,
            "training_config_modified": False,
            "active_training_output_modified": False,
            "threshold_search_performed": False,
            "test_used": False,
        },
    }
    report["audit_payload_sha256"] = _payload_sha256(report)
    return report


def verify_deterministic_replays(
    builder: Callable[[], Mapping[str, Any]], count: int
) -> tuple[dict[str, Any], list[str]]:
    if type(count) is not int or count < 2:
        raise ValueError("determinism_replays must be an integer >= 2")
    reports = [dict(builder()) for _ in range(count)]
    payloads = [_canonical_bytes(item) for item in reports]
    if any(payload != payloads[0] for payload in payloads[1:]):
        hashes = [hashlib.sha256(payload).hexdigest() for payload in payloads]
        raise RuntimeError(f"independent CPU audit replays differ: {hashes}")
    hashes = [hashlib.sha256(payload).hexdigest() for payload in payloads]
    return reports[0], hashes


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workspace = args.workspace_root.expanduser().resolve()
    if not workspace.is_dir():
        raise FileNotFoundError(f"workspace root is missing: {workspace}")
    source_root = _resolve(workspace, args.source_root, "source root", directory=True)
    materialized_run = _resolve(
        workspace, args.materialized_run, "materialized run", directory=True
    )
    kwargs = {
        "workspace": workspace,
        "source_root": source_root,
        "source_manifest_path": _resolve(
            workspace, args.source_manifest, "source manifest"
        ),
        "split_manifest_path": _resolve(
            workspace, args.split_manifest, "split manifest"
        ),
        "wire_manifest_path": _resolve(workspace, args.wire_manifest, "Wire manifest"),
        "junction_manifest_path": _resolve(
            workspace, args.junction_manifest, "Junction manifest"
        ),
        "materialized_run": materialized_run,
        "wire_predictions_path": _resolve(
            workspace, args.wire_predictions, "Wire predictions"
        ),
        "wire_status_path": _resolve(
            workspace, args.wire_run_status, "Wire run status"
        ),
        "wire_model_path": _resolve(workspace, args.wire_model, "Wire model"),
        "junction_status_path": _resolve(
            workspace, args.junction_run_status, "Junction run status"
        ),
        "junction_formal_model_path": _resolve(
            workspace, args.junction_formal_model, "Junction formal model"
        ),
        "junction_replay_checkpoint_path": _resolve(
            workspace, args.junction_replay_checkpoint, "Junction replay checkpoint"
        ),
        "models_lock_path": _resolve(workspace, args.models_lock, "models.lock"),
    }
    report, replay_hashes = verify_deterministic_replays(
        lambda: build_audit(**kwargs), args.determinism_replays
    )
    report["determinism_verification"] = {
        "independent_cpu_replays": args.determinism_replays,
        "byte_identical_before_attestation": True,
        "replay_sha256": replay_hashes,
        "all_replay_hashes_equal": len(set(replay_hashes)) == 1,
    }
    report["audit_payload_sha256"] = _payload_sha256(report)
    output = _resolve_output(workspace, args.output)
    _atomic_write(output, report)
    effect = report["isolated_learned_junction_effect"][
        "aggregate_delta_with_learned_minus_vectorized_only"
    ]
    print(
        json.dumps(
            {
                "output": _relative(output, workspace),
                "audit_payload_sha256": report["audit_payload_sha256"],
                "covered_val_samples": report["population"]["covered_samples"],
                "test_rows_used": report["selection_boundary"]["test_rows_used"],
                "junction_device": report["selection_boundary"][
                    "junction_inference_device"
                ],
                "f1_delta_with_learned": effect["f1"],
                "deterministic_replays": args.determinism_replays,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
