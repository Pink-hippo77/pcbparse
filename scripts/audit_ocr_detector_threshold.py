from __future__ import annotations

"""Calibrate the OCR detector threshold without touching the test split.

The workflow is deliberately split into four commands:

1. ``cache --split val`` runs the completed detector once on frozen validation
   tiles and stores only sigmoid probability maps.
2. ``select`` searches a predeclared threshold grid on validation labels.
3. ``cache --split test --selection ...`` runs the same checkpoint once on the
   frozen test tiles, after the validation selection is immutable.
4. ``confirm`` evaluates exactly the selected threshold on test labels once.

All model forwards are CPU-only and probability caches are hash-bound to the
checkpoint, training configuration, frozen materialization, tile order, and
this evaluator.  No command changes a checkpoint, training configuration,
watcher, run status, or models.lock.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


CACHE_KIND = "pcbparse_ocr_detector_probability_cache"
SELECTION_KIND = "pcbparse_ocr_detector_validation_threshold_selection"
CONFIRMATION_KIND = "pcbparse_ocr_detector_frozen_test_confirmation"
PAYLOAD_FIELD = "audit_payload_sha256"
FINAL_FORMAL_STATUSES = frozenset({"candidate_evidence_ready", "gate_failed"})
HMEAN_GATE = 0.45
BASELINE_THRESHOLD = 0.50
# Coarse search is fixed before any labels are read.  A deterministic 0.005
# refinement is then made around the validation-only coarse winner.
COARSE_THRESHOLDS = tuple(round(value / 1000.0, 3) for value in range(150, 751, 25))
REFINE_RADIUS = 0.025
REFINE_STEP = 0.005


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only OCR detector threshold calibration with validation/test "
            "anti-leakage boundaries."
        )
    )
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("cache", help="cache one split's model probabilities")
    cache.add_argument("--split", choices=("val", "test"), required=True)
    cache.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train/dataset300_v2_ocr_v5_allgate.json"),
    )
    cache.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("work/train-ocr-v5-allgate/model.pt"),
    )
    cache.add_argument(
        "--run-status",
        type=Path,
        default=Path("work/train-ocr-v5-allgate/torch_run_status.json"),
    )
    cache.add_argument(
        "--materialization-root",
        type=Path,
        default=Path("artifacts/training/materialized/dataset300-v2/ocr"),
    )
    cache.add_argument("--output-dir", type=Path, required=True)
    cache.add_argument("--selection", type=Path)
    cache.add_argument("--batch-size", type=int, default=1)
    cache.add_argument("--cpu-threads", type=int, default=1)
    cache.add_argument(
        "--interim-epoch-diagnostic",
        action="store_true",
        help=(
            "allow checkpoint_epoch_NNN.pt from a still-running formal run; "
            "the resulting evidence is diagnostic and never promotion eligible"
        ),
    )

    select = subparsers.add_parser(
        "select", help="select one detector threshold on a validation cache"
    )
    select.add_argument("--cache-dir", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)

    confirm = subparsers.add_parser(
        "confirm", help="confirm the selected threshold once on frozen test"
    )
    confirm.add_argument("--selection", type=Path, required=True)
    confirm.add_argument("--cache-dir", type=Path, required=True)
    confirm.add_argument("--output", type=Path, required=True)
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r} in {label}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key {key!r} in {label}")
            output[key] = value
        return output

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _read_list(path: Path, label: str) -> list[Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{label} root must be an array")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(encoded, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _with_payload_hash(document: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(document))
    output.pop(PAYLOAD_FIELD, None)
    output[PAYLOAD_FIELD] = _canonical_sha256(output)
    return output


def _verify_payload_hash(document: Mapping[str, Any], label: str) -> str:
    declared = document.get(PAYLOAD_FIELD)
    if not isinstance(declared, str) or len(declared) != 64:
        raise ValueError(f"{label} lacks a valid canonical payload SHA-256")
    payload = copy.deepcopy(dict(document))
    payload.pop(PAYLOAD_FIELD, None)
    if _canonical_sha256(payload) != declared:
        raise ValueError(f"{label} canonical payload SHA-256 mismatch")
    return declared


def _resolve_within(
    root: Path,
    raw: str | Path,
    label: str,
    *,
    require_file: bool = False,
    require_dir: bool = False,
) -> Path:
    candidate = Path(raw).expanduser()
    path = (candidate if candidate.is_absolute() else root / candidate).resolve(
        strict=False
    )
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside workspace: {path}") from exc
    if require_file and not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if require_dir and not path.is_dir():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def _relative(path: Path, root: Path) -> str:
    return path.resolve(strict=False).relative_to(root).as_posix()


def _positive_integer(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_probability(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{label} must be finite and in [0, 1]")
    return float(value)


def _lineage_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "task",
        "dataset_config_hash",
        "source_manifest_sha256",
        "split_manifest_sha256",
        "task_manifest_sha256",
        "binding_sha256",
        "materialization_commit_id",
        "materialization_commit_sha256",
        "architecture",
        "input_size",
        "smoke",
        "formal_evidence",
        "training_config_sha256",
    )
    return {key: value.get(key) for key in keys}


def _checkpoint_role(
    status: Mapping[str, Any], checkpoint: Path, *, interim: bool
) -> str:
    lineage = status.get("lineage")
    if (
        status.get("task") != "ocr"
        or status.get("smoke") is not False
        or not isinstance(lineage, Mapping)
        or lineage.get("formal_evidence") is not True
    ):
        raise RuntimeError("run status is not a formal OCR lineage")
    if interim:
        prefix = "checkpoint_epoch_"
        suffix = checkpoint.stem[len(prefix) :] if checkpoint.stem.startswith(prefix) else ""
        if len(suffix) != 3 or not suffix.isdigit():
            raise RuntimeError("interim audit requires checkpoint_epoch_NNN.pt")
        return "interim_formal_epoch_diagnostic"
    if status.get("status") not in FINAL_FORMAL_STATUSES:
        raise RuntimeError("OCR run is not a completed formal run")
    if status.get("formal_evidence") is not True or not status.get("finished_at_utc"):
        raise RuntimeError("completed OCR status lacks formal termination evidence")
    planned = status.get("planned_epochs")
    completed = status.get("completed_epochs")
    if (
        type(planned) is not int
        or type(completed) is not int
        or completed <= 0
        or completed > planned
        or (completed != planned and status.get("early_stopped") is not True)
    ):
        raise RuntimeError("completed OCR status has invalid epoch termination")
    if checkpoint.name != "model.pt":
        raise RuntimeError("completed OCR calibration requires model.pt")
    artifact = status.get("artifact")
    if not isinstance(artifact, Mapping) or artifact.get("sha256") != _sha256_file(checkpoint):
        raise RuntimeError("completed OCR status is not bound to the supplied model.pt")
    return "completed_formal_best_model"


def _source_entry(path: Path, workspace: Path) -> dict[str, Any]:
    return {"path": _relative(path, workspace), "sha256": _sha256_file(path)}


def _cache_sources(
    *,
    workspace: Path,
    config_path: Path,
    status_path: Path,
    checkpoint_path: Path,
    materialization_root: Path,
) -> dict[str, Any]:
    return {
        "training_config": _source_entry(config_path, workspace),
        "run_status": _source_entry(status_path, workspace),
        "checkpoint": _source_entry(checkpoint_path, workspace),
        "materialization_manifest": _source_entry(
            materialization_root / "manifest.json", workspace
        ),
        "detection_index": _source_entry(
            materialization_root / "detection_index.jsonl", workspace
        ),
        "audit_script": _source_entry(Path(__file__).resolve(), workspace),
        "detector_evaluator": _source_entry(
            workspace / "src/pcbparse/training/torch_ocr_model.py", workspace
        ),
        "detection_dataset": _source_entry(
            workspace / "src/pcbparse/training/torch_ocr_detection.py", workspace
        ),
    }


def _verify_source_entries(
    workspace: Path, raw: Any, label: str
) -> dict[str, dict[str, str]]:
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"{label} is missing")
    output: dict[str, dict[str, str]] = {}
    for name, entry in raw.items():
        if not isinstance(name, str) or not isinstance(entry, Mapping):
            raise ValueError(f"{label} entry is invalid")
        relative = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError(f"{label}.{name} path/hash is invalid")
        path = _resolve_within(workspace, relative, f"{label}.{name}", require_file=True)
        if _sha256_file(path) != expected:
            raise ValueError(f"{label}.{name} SHA-256 mismatch")
        output[name] = {"path": relative, "sha256": expected}
    return output


def _thresholds_around(coarse_winner: float) -> tuple[float, ...]:
    winner = _finite_probability(coarse_winner, "coarse winner")
    start = max(0.0, winner - REFINE_RADIUS)
    end = min(1.0, winner + REFINE_RADIUS)
    count = int(round((end - start) / REFINE_STEP))
    values = [round(start + index * REFINE_STEP, 3) for index in range(count + 1)]
    values.extend(COARSE_THRESHOLDS)
    values.append(BASELINE_THRESHOLD)
    return tuple(sorted(set(values)))


def _selection_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    threshold = _finite_probability(row.get("threshold"), "threshold")
    return (
        float(row["hmean"]),
        float(row["recall"]),
        float(row["precision"]),
        -abs(threshold - BASELINE_THRESHOLD),
        threshold,
    )


def _select_best(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("threshold result grid is empty")
    thresholds = [_finite_probability(row.get("threshold"), "threshold") for row in rows]
    if len(thresholds) != len(set(thresholds)):
        raise ValueError("threshold result grid contains duplicates")
    for row in rows:
        for metric in ("hmean", "recall", "precision"):
            _finite_probability(row.get(metric), metric)
    return dict(max(rows, key=_selection_key))


def _aggregate_counts(metrics: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "true_positives",
        "false_positives",
        "false_negatives",
        "predictions",
        "targets",
        "ignored_predictions",
    )
    counts = {key: int(metrics[key]) for key in keys}
    if any(value < 0 for value in counts.values()):
        raise ValueError("detector metrics contain negative counts")
    tp = counts["true_positives"]
    fp = counts["false_positives"]
    fn = counts["false_negatives"]
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    hmean = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "hmean": hmean,
        "mean_matched_iou": float(metrics["mean_matched_iou"]),
    }


def _evaluate_threshold(
    *,
    probabilities: np.ndarray,
    boxes: Sequence[np.ndarray],
    ignore_masks: np.ndarray,
    threshold: float,
    min_area: int,
    iou_threshold: float,
) -> dict[str, Any]:
    from pcbparse.training.torch_ocr_model import detection_box_metrics

    metrics = detection_box_metrics(
        probabilities[:, None, :, :],
        ground_truth_boxes=boxes,
        ignore_masks=ignore_masks,
        threshold=threshold,
        min_area=min_area,
        iou_threshold=iou_threshold,
    )
    return {"threshold": threshold, **_aggregate_counts(metrics)}


def _load_dataset_inputs(
    *, workspace: Path, cache: Mapping[str, Any], tile_ids: Sequence[str]
) -> tuple[Any, list[np.ndarray], np.ndarray, dict[str, Any]]:
    sources = cache["sources"]
    config_path = _resolve_within(
        workspace,
        sources["training_config"]["path"],
        "cached training config",
        require_file=True,
    )
    from pcbparse.training.config import load_training_config
    from pcbparse.training.torch_ocr_detection import TorchOCRDetectionDataset

    training_config = load_training_config(config_path)
    task_config = training_config.tasks["ocr"]
    split = str(cache["split"])
    materialization_manifest = _resolve_within(
        workspace,
        sources["materialization_manifest"]["path"],
        "cached materialization manifest",
        require_file=True,
    )
    materialization_root = materialization_manifest.parent
    source_root = _resolve_within(
        workspace,
        training_config.dataset.source_root,
        "Dataset300 source root",
        require_dir=True,
    )
    dataset = TorchOCRDetectionDataset(
        materialization_root,
        source_root=source_root,
        split=split,
        resize=int(task_config.ocr_detector_size),
        target_shrink_ratio=float(task_config.ocr_target_shrink_ratio),
    )
    if len(dataset) != len(tile_ids):
        raise ValueError("probability cache does not cover the complete frozen split")
    boxes: list[np.ndarray] = []
    ignore_masks: list[np.ndarray] = []
    sample_ids: set[str] = set()
    ground_truth = 0
    for index, expected_tile_id in enumerate(tile_ids):
        item = dataset[index]
        if item["tile_id"] != expected_tile_id:
            raise ValueError("probability cache tile order differs from frozen dataset")
        metadata = item["metadata"]
        if metadata.get("split") != split:
            raise ValueError("a non-selected split reached threshold evaluation")
        local_boxes = item["boxes"].detach().cpu().numpy().astype(np.float32, copy=False)
        local_ignore = item["ignore"].detach().cpu().numpy().astype(np.uint8, copy=False)
        boxes.append(local_boxes)
        ignore_masks.append(local_ignore)
        ground_truth += int(local_boxes.shape[0])
        sample_ids.add(str(item["sample_id"]))
    return (
        task_config,
        boxes,
        np.stack(ignore_masks, axis=0),
        {
            "tiles": len(tile_ids),
            "unique_sample_ids": len(sample_ids),
            "ground_truth_boxes": ground_truth,
            "source_images_verified_while_loading": len(dataset._verified_sources),
        },
    )


def _verify_cache(
    workspace: Path, cache_dir: Path, *, expected_split: str
) -> tuple[dict[str, Any], np.ndarray, list[str], str]:
    manifest_path = cache_dir / "cache.manifest.json"
    manifest = _read_object(manifest_path, "OCR detector probability cache")
    payload_hash = _verify_payload_hash(manifest, "OCR detector probability cache")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != CACHE_KIND
        or manifest.get("status") != "complete"
        or manifest.get("split") != expected_split
    ):
        raise ValueError("OCR detector probability cache contract is invalid")
    if (
        manifest.get("model_forward_device") != "cpu"
        or manifest.get("labels_scored_while_caching") is not False
        or manifest.get("threshold_search_while_caching") is not False
    ):
        raise ValueError("probability cache execution boundary is invalid")
    sources = _verify_source_entries(workspace, manifest.get("sources"), "cache sources")
    manifest["sources"] = sources
    probability = manifest.get("probability_file")
    tile_file = manifest.get("tile_id_file")
    if not isinstance(probability, Mapping) or not isinstance(tile_file, Mapping):
        raise ValueError("cache data-file metadata is missing")
    if probability.get("path") != "probabilities.npy" or tile_file.get("path") != "tile_ids.json":
        raise ValueError("cache data-file names are not canonical")
    probability_path = cache_dir / "probabilities.npy"
    tile_path = cache_dir / "tile_ids.json"
    if _sha256_file(probability_path) != probability.get("sha256"):
        raise ValueError("cached probabilities SHA-256 mismatch")
    if _sha256_file(tile_path) != tile_file.get("sha256"):
        raise ValueError("cached tile IDs SHA-256 mismatch")
    probabilities = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    if probabilities.dtype != np.float32 or probabilities.ndim != 3:
        raise ValueError("cached probabilities must be float32 [N,H,W]")
    declared_shape = probability.get("shape")
    if declared_shape != list(probabilities.shape):
        raise ValueError("cached probability shape differs from manifest")
    if not np.isfinite(probabilities).all() or float(probabilities.min()) < 0.0 or float(probabilities.max()) > 1.0:
        raise ValueError("cached probabilities are non-finite or outside [0,1]")
    tile_ids = _read_list(tile_path, "cached tile IDs")
    if (
        any(not isinstance(item, str) or not item for item in tile_ids)
        or len(tile_ids) != len(set(tile_ids))
        or len(tile_ids) != probabilities.shape[0]
        or _canonical_sha256(tile_ids) != tile_file.get("canonical_sha256")
    ):
        raise ValueError("cached tile IDs are invalid or misaligned")
    coverage = manifest.get("coverage")
    if not isinstance(coverage, Mapping) or coverage.get("tiles") != len(tile_ids):
        raise ValueError("cache coverage differs from data files")
    return manifest, probabilities, tile_ids, payload_hash


def _verify_selection(
    workspace: Path, selection_path: Path
) -> tuple[dict[str, Any], str]:
    selection = _read_object(selection_path, "OCR validation threshold selection")
    payload_hash = _verify_payload_hash(selection, "OCR validation threshold selection")
    if (
        selection.get("schema_version") != 1
        or selection.get("kind") != SELECTION_KIND
        or selection.get("status") != "validation_threshold_selected"
    ):
        raise ValueError("OCR validation threshold selection contract is invalid")
    boundary = selection.get("selection_boundary")
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("split") != "val"
        or boundary.get("test_tiles_loaded") != 0
        or boundary.get("test_labels_scored") != 0
        or boundary.get("test_thresholds_searched") != 0
    ):
        raise ValueError("threshold selection is not validation-only")
    _finite_probability(selection.get("selected_threshold"), "selected threshold")
    checkpoint = selection.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("threshold selection lacks checkpoint binding")
    checkpoint_path = _resolve_within(
        workspace, checkpoint.get("path", ""), "selected checkpoint", require_file=True
    )
    if _sha256_file(checkpoint_path) != checkpoint.get("sha256"):
        raise ValueError("selected checkpoint SHA-256 mismatch")
    return selection, payload_hash


def _cache_command(arguments: argparse.Namespace, workspace: Path) -> dict[str, Any]:
    # This assignment occurs before importing torch.  It makes accidental CUDA
    # use impossible even when the training process owns cuda:0 concurrently.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    config_path = _resolve_within(workspace, arguments.config, "training config", require_file=True)
    checkpoint_path = _resolve_within(workspace, arguments.checkpoint, "OCR checkpoint", require_file=True)
    status_path = _resolve_within(workspace, arguments.run_status, "OCR run status", require_file=True)
    materialization_root = _resolve_within(
        workspace, arguments.materialization_root, "OCR materialization", require_dir=True
    )
    output_dir = _resolve_within(workspace, arguments.output_dir, "cache output directory")
    if output_dir.exists():
        raise FileExistsError(f"cache output directory already exists: {output_dir}")
    status = _read_object(status_path, "OCR run status")
    checkpoint_role = _checkpoint_role(
        status, checkpoint_path, interim=bool(arguments.interim_epoch_diagnostic)
    )
    if arguments.split == "val" and arguments.selection is not None:
        raise ValueError("validation probability caching must not receive a selection")
    selection_source: dict[str, Any] | None = None
    selection: dict[str, Any] | None = None
    if arguments.split == "test":
        if arguments.selection is None:
            raise ValueError("test probability caching requires immutable validation selection")
        selection_path = _resolve_within(
            workspace, arguments.selection, "validation selection", require_file=True
        )
        selection, selection_payload = _verify_selection(workspace, selection_path)
        selection_source = {
            **_source_entry(selection_path, workspace),
            "payload_sha256": selection_payload,
            "selected_threshold": selection["selected_threshold"],
        }

    import torch

    from pcbparse.training.config import (
        load_training_config,
        training_config_resume_sha256s,
    )
    from pcbparse.training.torch_ocr_detection import (
        TorchOCRDetectionDataset,
        detection_collate,
    )
    from pcbparse.training.torch_ocr_model import load_ocr_bundle

    threads = _positive_integer(arguments.cpu_threads, "cpu-threads")
    batch_size = _positive_integer(arguments.batch_size, "batch-size")
    torch.set_num_threads(threads)
    training_config = load_training_config(config_path)
    task_config = training_config.tasks["ocr"]
    lineage = status["lineage"]
    if lineage.get("training_config_sha256") not in training_config_resume_sha256s(training_config):
        raise RuntimeError("OCR status/training configuration lineage mismatch")
    source_root = _resolve_within(
        workspace,
        training_config.dataset.source_root,
        "Dataset300 source root",
        require_dir=True,
    )
    dataset = TorchOCRDetectionDataset(
        materialization_root,
        source_root=source_root,
        split=arguments.split,
        resize=int(task_config.ocr_detector_size),
        target_shrink_ratio=float(task_config.ocr_target_shrink_ratio),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=detection_collate,
        drop_last=False,
    )
    checkpoint_sha_before = _sha256_file(checkpoint_path)
    bundle = load_ocr_bundle(
        checkpoint_path, device="cpu", expected_sha256=checkpoint_sha_before
    )
    if _lineage_identity(bundle.lineage) != _lineage_identity(lineage):
        raise RuntimeError("OCR checkpoint identity differs from run status")
    if checkpoint_role == "completed_formal_best_model":
        if bundle.lineage != lineage or bundle.epoch != status.get("best_epoch"):
            raise RuntimeError("completed OCR model.pt differs from final run status")
    else:
        expected_epoch = int(checkpoint_path.stem.rsplit("_", 1)[1])
        if bundle.epoch != expected_epoch:
            raise RuntimeError("interim checkpoint filename/payload epoch mismatch")
    if selection is not None:
        selected_checkpoint = selection["checkpoint"]
        if selected_checkpoint.get("sha256") != checkpoint_sha_before:
            raise RuntimeError("test cache checkpoint differs from validation selection")
    detector = bundle.detector.to("cpu")
    detector.eval()
    detector_size = int(task_config.ocr_detector_size)
    temp_dir = output_dir.with_name(f".{output_dir.name}.building")
    if temp_dir.exists():
        raise FileExistsError(f"temporary cache directory already exists: {temp_dir}")
    temp_dir.mkdir(parents=True)
    probability_path = temp_dir / "probabilities.npy"
    probabilities = np.lib.format.open_memmap(
        probability_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(dataset), detector_size, detector_size),
    )
    tile_ids: list[str] = []
    sample_ids: set[str] = set()
    cursor = 0
    try:
        with torch.inference_mode():
            for batch in loader:
                metadata = batch.get("metadata")
                if not isinstance(metadata, list) or any(
                    row.get("split") != arguments.split for row in metadata
                ):
                    raise RuntimeError("non-selected split reached probability cache")
                images = batch["images"].to(device="cpu", dtype=torch.float32)
                local = torch.sigmoid(detector(images).float()).squeeze(1).cpu().numpy()
                if local.shape[1:] != (detector_size, detector_size):
                    raise RuntimeError("detector output shape differs from frozen cache contract")
                if not np.isfinite(local).all() or float(local.min()) < 0.0 or float(local.max()) > 1.0:
                    raise RuntimeError("detector probabilities are invalid")
                next_cursor = cursor + local.shape[0]
                probabilities[cursor:next_cursor] = local.astype(np.float32, copy=False)
                tile_ids.extend(str(item) for item in batch["tile_ids"])
                sample_ids.update(str(item) for item in batch["sample_ids"])
                cursor = next_cursor
        probabilities.flush()
        del probabilities
        if cursor != len(dataset) or len(tile_ids) != len(set(tile_ids)):
            raise RuntimeError("probability cache coverage is incomplete or duplicated")
        checkpoint_sha_after = _sha256_file(checkpoint_path)
        if checkpoint_sha_after != checkpoint_sha_before:
            raise RuntimeError("OCR checkpoint changed while probabilities were cached")
        tile_path = temp_dir / "tile_ids.json"
        _write_json_atomic(tile_path, tile_ids)
        sources = _cache_sources(
            workspace=workspace,
            config_path=config_path,
            status_path=status_path,
            checkpoint_path=checkpoint_path,
            materialization_root=materialization_root,
        )
        manifest = _with_payload_hash(
            {
                "schema_version": 1,
                "kind": CACHE_KIND,
                "status": "complete",
                "created_at_utc": _utc_now(),
                "split": arguments.split,
                "checkpoint_role": checkpoint_role,
                "formal_parent_completed": checkpoint_role == "completed_formal_best_model",
                "promotion_eligible": False,
                "model_forward_device": "cpu",
                "labels_scored_while_caching": False,
                "threshold_search_while_caching": False,
                "test_cache_created_after_validation_selection": arguments.split == "test",
                "sources": sources,
                "selection_source": selection_source,
                "inference": {
                    "torch_version": str(torch.__version__),
                    "cpu_threads": threads,
                    "batch_size": batch_size,
                    "detector_size": [detector_size, detector_size],
                    "sigmoid_applied": True,
                    "storage_dtype": "float32",
                    "checkpoint_epoch": bundle.epoch,
                    "checkpoint_sha256_before_and_after_equal": True,
                },
                "coverage": {
                    "tiles": len(tile_ids),
                    "unique_sample_ids": len(sample_ids),
                    "tile_ids_sha256": _canonical_sha256(tile_ids),
                    "source_images_hash_verified": len(dataset._verified_sources),
                },
                "probability_file": {
                    "path": "probabilities.npy",
                    "sha256": _sha256_file(probability_path),
                    "bytes": probability_path.stat().st_size,
                    "shape": [len(tile_ids), detector_size, detector_size],
                    "dtype": "float32",
                },
                "tile_id_file": {
                    "path": "tile_ids.json",
                    "sha256": _sha256_file(tile_path),
                    "bytes": tile_path.stat().st_size,
                    "canonical_sha256": _canonical_sha256(tile_ids),
                },
            }
        )
        _write_json_atomic(temp_dir / "cache.manifest.json", manifest)
        os.replace(temp_dir, output_dir)
        return manifest
    except Exception:
        try:
            del probabilities
        except UnboundLocalError:
            pass
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _select_command(arguments: argparse.Namespace, workspace: Path) -> dict[str, Any]:
    cache_dir = _resolve_within(workspace, arguments.cache_dir, "validation cache", require_dir=True)
    output_path = _resolve_within(workspace, arguments.output, "validation selection output")
    cache, probabilities, tile_ids, cache_payload = _verify_cache(
        workspace, cache_dir, expected_split="val"
    )
    task_config, boxes, ignore_masks, coverage = _load_dataset_inputs(
        workspace=workspace, cache=cache, tile_ids=tile_ids
    )
    coarse_rows = [
        _evaluate_threshold(
            probabilities=probabilities,
            boxes=boxes,
            ignore_masks=ignore_masks,
            threshold=threshold,
            min_area=int(task_config.ocr_min_area),
            iou_threshold=float(task_config.ocr_iou_threshold),
        )
        for threshold in COARSE_THRESHOLDS
    ]
    coarse_best = _select_best(coarse_rows)
    thresholds = _thresholds_around(float(coarse_best["threshold"]))
    rows_by_threshold = {float(row["threshold"]): row for row in coarse_rows}
    for threshold in thresholds:
        if threshold not in rows_by_threshold:
            rows_by_threshold[threshold] = _evaluate_threshold(
                probabilities=probabilities,
                boxes=boxes,
                ignore_masks=ignore_masks,
                threshold=threshold,
                min_area=int(task_config.ocr_min_area),
                iou_threshold=float(task_config.ocr_iou_threshold),
            )
    grid = [rows_by_threshold[threshold] for threshold in sorted(rows_by_threshold)]
    selected = _select_best(grid)
    baseline = rows_by_threshold[BASELINE_THRESHOLD]
    checkpoint = cache["sources"]["checkpoint"]
    document = _with_payload_hash(
        {
            "schema_version": 1,
            "kind": SELECTION_KIND,
            "status": "validation_threshold_selected",
            "created_at_utc": _utc_now(),
            "formal_parent_completed": cache.get("formal_parent_completed") is True,
            "promotion_eligible": False,
            "selection_boundary": {
                "split": "val",
                "validation_tiles_loaded": len(tile_ids),
                "test_tiles_loaded": 0,
                "test_labels_scored": 0,
                "test_thresholds_searched": 0,
            },
            "protocol": {
                "coarse_thresholds": list(COARSE_THRESHOLDS),
                "coarse_winner": coarse_best["threshold"],
                "refine_radius": REFINE_RADIUS,
                "refine_step": REFINE_STEP,
                "evaluated_thresholds": [row["threshold"] for row in grid],
                "selection_order": [
                    "highest_hmean",
                    "highest_recall",
                    "highest_precision",
                    "closest_to_configured_0.5",
                    "higher_threshold",
                ],
                "iou_threshold": float(task_config.ocr_iou_threshold),
                "minimum_component_area": int(task_config.ocr_min_area),
                "probabilities_cached_before_label_scoring": True,
            },
            "source_cache": {
                "path": _relative(cache_dir / "cache.manifest.json", workspace),
                "sha256": _sha256_file(cache_dir / "cache.manifest.json"),
                "payload_sha256": cache_payload,
            },
            "checkpoint": dict(checkpoint),
            "coverage": coverage,
            "configured_threshold_baseline": baseline,
            "grid": grid,
            "selected_threshold": selected["threshold"],
            "selected_validation_metrics": selected,
            "validation_gate": {
                "metric": "detector_hmean",
                "operator": "gte",
                "threshold": HMEAN_GATE,
                "value": selected["hmean"],
                "passed": selected["hmean"] >= HMEAN_GATE,
            },
            "calibration_delta": {
                "hmean": selected["hmean"] - baseline["hmean"],
                "recall": selected["recall"] - baseline["recall"],
                "precision": selected["precision"] - baseline["precision"],
            },
            "models_lock_modified": False,
            "blessed": False,
        }
    )
    _write_json_atomic(output_path, document)
    return document


def _confirm_command(arguments: argparse.Namespace, workspace: Path) -> dict[str, Any]:
    selection_path = _resolve_within(
        workspace, arguments.selection, "validation selection", require_file=True
    )
    cache_dir = _resolve_within(workspace, arguments.cache_dir, "test cache", require_dir=True)
    output_path = _resolve_within(workspace, arguments.output, "test confirmation output")
    selection, selection_payload = _verify_selection(workspace, selection_path)
    cache, probabilities, tile_ids, cache_payload = _verify_cache(
        workspace, cache_dir, expected_split="test"
    )
    source = cache.get("selection_source")
    if (
        not isinstance(source, Mapping)
        or source.get("sha256") != _sha256_file(selection_path)
        or source.get("payload_sha256") != selection_payload
        or source.get("selected_threshold") != selection.get("selected_threshold")
    ):
        raise ValueError("test probability cache is not bound to this validation selection")
    if cache["sources"]["checkpoint"] != selection.get("checkpoint"):
        raise ValueError("validation selection and test cache checkpoints differ")
    task_config, boxes, ignore_masks, coverage = _load_dataset_inputs(
        workspace=workspace, cache=cache, tile_ids=tile_ids
    )
    threshold = _finite_probability(selection["selected_threshold"], "selected threshold")
    metrics = _evaluate_threshold(
        probabilities=probabilities,
        boxes=boxes,
        ignore_masks=ignore_masks,
        threshold=threshold,
        min_area=int(task_config.ocr_min_area),
        iou_threshold=float(task_config.ocr_iou_threshold),
    )
    document = _with_payload_hash(
        {
            "schema_version": 1,
            "kind": CONFIRMATION_KIND,
            "status": (
                "fixed_threshold_test_confirmation_passed"
                if metrics["hmean"] >= HMEAN_GATE
                else "fixed_threshold_test_confirmation_gate_failed"
            ),
            "created_at_utc": _utc_now(),
            "formal_parent_completed": (
                selection.get("formal_parent_completed") is True
                and cache.get("formal_parent_completed") is True
            ),
            "promotion_eligible": False,
            "anti_leakage_boundary": {
                "selection_split": "val",
                "confirmation_split": "test",
                "test_probability_cache_created_after_selection": True,
                "test_threshold_search_performed": False,
                "test_thresholds_evaluated": [threshold],
                "test_labels_scored_once": True,
            },
            "validation_selection": {
                "path": _relative(selection_path, workspace),
                "sha256": _sha256_file(selection_path),
                "payload_sha256": selection_payload,
                "selected_threshold": threshold,
                "selected_validation_metrics": selection["selected_validation_metrics"],
            },
            "test_cache": {
                "path": _relative(cache_dir / "cache.manifest.json", workspace),
                "sha256": _sha256_file(cache_dir / "cache.manifest.json"),
                "payload_sha256": cache_payload,
            },
            "checkpoint": dict(cache["sources"]["checkpoint"]),
            "coverage": coverage,
            "fixed_threshold": threshold,
            "test_metrics": metrics,
            "test_gate": {
                "metric": "detector_hmean",
                "operator": "gte",
                "threshold": HMEAN_GATE,
                "value": metrics["hmean"],
                "passed": metrics["hmean"] >= HMEAN_GATE,
            },
            "models_lock_modified": False,
            "blessed": False,
        }
    )
    _write_json_atomic(output_path, document)
    return document


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    workspace = Path(arguments.workspace_root).expanduser().resolve(strict=True)
    if arguments.command == "cache":
        result = _cache_command(arguments, workspace)
        output = _resolve_within(workspace, arguments.output_dir, "cache output") / "cache.manifest.json"
    elif arguments.command == "select":
        result = _select_command(arguments, workspace)
        output = _resolve_within(workspace, arguments.output, "selection output")
    elif arguments.command == "confirm":
        result = _confirm_command(arguments, workspace)
        output = _resolve_within(workspace, arguments.output, "confirmation output")
    else:  # pragma: no cover - argparse owns this invariant.
        raise AssertionError(f"unsupported command {arguments.command!r}")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _sha256_file(output),
                "kind": result["kind"],
                "status": result["status"],
                "selected_threshold": result.get("selected_threshold", result.get("fixed_threshold")),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
