from __future__ import annotations

"""Audit Pin score thresholds on the frozen validation split only.

The script deliberately has no test-split option.  It reuses the exact Pin
resize, box-margin, decoding, and one-to-one 4 px matching implementation used
by formal training, while running one model forward pass per validation batch
for the complete threshold grid.
"""

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


REQUIRED_THRESHOLDS = (0.15, 0.30, 0.45, 0.60, 0.75)
AUDIT_NMS_RADIUS = 2.0
AUDIT_MAX_DETECTIONS = 128
AUDIT_MATCH_RADIUS_PX = 4.0
AUDIT_BOX_PRIOR_MODE = "boundary_band_v2"
FINAL_FORMAL_STATUSES = frozenset({"candidate_evidence_ready", "gate_failed"})


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a completed Pin checkpoint on the frozen validation split "
            "at a fixed, auditable score-threshold grid."
        )
    )
    parser.add_argument("--workspace-root", type=Path, default=root)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train/dataset300_v2_pin_v4.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("work/train-pin-v4/model.pt"),
        help="completed run's best-model checkpoint (normally model.pt)",
    )
    parser.add_argument(
        "--run-status",
        type=Path,
        default=Path("work/train-pin-v4/torch_run_status.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/pin_v4_validation_threshold_audit.json"),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or cuda:N (default: auto)",
    )
    parser.add_argument(
        "--interim-warm-start-diagnostic",
        action="store_true",
        help=(
            "explicitly allow checkpoint_epoch_NNN.pt from an interrupted formal "
            "run; the report remains non-registerable validation-only evidence"
        ),
    )
    parser.add_argument(
        "--extra-threshold",
        action="append",
        default=[],
        type=float,
        help=(
            "add a validation threshold without removing the frozen required "
            "grid; may be repeated"
        ),
    )
    return parser


def _resolve_within(root: Path, raw: Path, label: str, *, must_exist: bool) -> Path:
    path = raw.expanduser()
    resolved = (path if path.is_absolute() else root / path).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside workspace: {resolved}") from exc
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def _device(torch: Any, requested: str) -> Any:
    value = str(requested).strip().lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value != "cpu" and value != "cuda" and not value.startswith("cuda:"):
        raise ValueError("--device must be auto, cpu, cuda, or cuda:N")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def _thresholds(extra: Iterable[float]) -> tuple[float, ...]:
    values = [*REQUIRED_THRESHOLDS, *extra]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
        for value in values
    ):
        raise ValueError("all score thresholds must be finite and in [0, 1]")
    return tuple(sorted({float(value) for value in values}))


def _micro_summary(*, matched: int, predicted: int, truth: int) -> dict[str, Any]:
    if min(matched, predicted, truth) < 0 or matched > min(predicted, truth):
        raise ValueError("invalid Pin detection totals")
    precision = matched / predicted if predicted else (1.0 if truth == 0 else 0.0)
    recall = matched / truth if truth else (1.0 if predicted == 0 else 0.0)
    denominator = precision + recall
    f1 = 2.0 * precision * recall / denominator if denominator else 0.0
    return {
        "matched": matched,
        "pred_count": predicted,
        "gt_count": truth,
        "micro_precision_4px": precision,
        "micro_recall_4px": recall,
        "micro_f1_4px": f1,
        "pred_to_gt_ratio": (
            predicted / truth if truth else (1.0 if predicted == 0 else 1.0e9)
        ),
    }


def _require_checkpoint_role(
    status: Mapping[str, Any],
    checkpoint: Path,
    *,
    interim_warm_start_diagnostic: bool,
) -> str:
    if interim_warm_start_diagnostic:
        stem = checkpoint.stem
        prefix = "checkpoint_epoch_"
        suffix = stem[len(prefix) :] if stem.startswith(prefix) else ""
        if len(suffix) != 3 or not suffix.isdigit():
            raise RuntimeError(
                "interim warm-start audit requires checkpoint_epoch_NNN.pt"
            )
        lineage = status.get("lineage")
        if not isinstance(lineage, Mapping) or lineage.get("formal_evidence") is not True:
            raise RuntimeError("interim checkpoint lacks formal-run lineage")
        if status.get("task") != "pin" or status.get("smoke") is not False:
            raise RuntimeError("interim checkpoint did not come from a formal Pin run")
        return "interim_formal_epoch_warm_start_diagnostic"

    state = status.get("status")
    if state not in FINAL_FORMAL_STATUSES:
        raise RuntimeError(
            "Pin run is not a completed formal run: "
            f"status={state!r}; expected one of {sorted(FINAL_FORMAL_STATUSES)}"
        )
    if status.get("formal_evidence") is not True or not status.get("finished_at_utc"):
        raise RuntimeError("Pin run status lacks completed formal-evidence markers")
    planned = status.get("planned_epochs")
    completed = status.get("completed_epochs")
    early_stopped = status.get("early_stopped")
    if (
        type(planned) is not int
        or type(completed) is not int
        or completed <= 0
        or completed > planned
        or (completed != planned and early_stopped is not True)
    ):
        raise RuntimeError("Pin run status does not prove a valid formal termination")
    if checkpoint.name != "model.pt":
        raise RuntimeError(
            "completed best-model audit requires model.pt; interim epoch checkpoints "
            "are intentionally rejected"
        )
    return "completed_formal_best_model"


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
    )
    return {key: value.get(key) for key in keys}


def _close(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    workspace = arguments.workspace_root.expanduser().resolve(strict=True)
    config_path = _resolve_within(
        workspace, arguments.config, "training config", must_exist=True
    )
    checkpoint_path = _resolve_within(
        workspace, arguments.checkpoint, "Pin checkpoint", must_exist=True
    )
    status_path = _resolve_within(
        workspace, arguments.run_status, "Pin run status", must_exist=True
    )
    output_path = _resolve_within(
        workspace, arguments.output, "audit output", must_exist=False
    )
    status = _read_object(status_path, "Pin run status")
    checkpoint_role = _require_checkpoint_role(
        status,
        checkpoint_path,
        interim_warm_start_diagnostic=arguments.interim_warm_start_diagnostic,
    )

    sys.path.insert(0, str(workspace / "src"))
    import torch

    from pcbparse.training.config import (
        load_training_config,
        training_config_resume_sha256s,
        training_config_sha256,
    )
    from pcbparse.training.preflight import preflight_training_data
    from pcbparse.training.torch_backend import _prepare_pin_batch
    from pcbparse.training.torch_component_pin import (
        PinKeypointDataset,
        pin_keypoint_collate,
    )
    from pcbparse.training.torch_pin_model import (
        compute_pin_metrics,
        load_pin_checkpoint,
        split_pin_outputs,
    )

    training_config = load_training_config(config_path)
    task_config = training_config.tasks["pin"]
    config_sha256 = training_config_sha256(training_config)
    compatible_config_sha256s = training_config_resume_sha256s(training_config)
    preflight = preflight_training_data(config_path, workspace_root=workspace)
    if preflight.get("ok") is not True:
        codes = sorted(
            {
                str(issue.get("code", "UNKNOWN"))
                for issue in preflight.get("issues", [])
            }
        )
        raise RuntimeError(f"training-data preflight failed: {', '.join(codes)}")
    status_lineage = status.get("lineage")
    if not isinstance(status_lineage, dict):
        raise RuntimeError("Pin run status lacks lineage")
    if status_lineage.get("training_config_sha256") not in compatible_config_sha256s:
        raise RuntimeError("Pin run/config training_config_sha256 mismatch")
    if status_lineage.get("split_manifest_sha256") != (
        training_config.dataset.split_manifest_sha256
    ):
        raise RuntimeError("Pin run/config split-manifest mismatch")
    if status_lineage.get("formal_evidence") is not True:
        raise RuntimeError("Pin checkpoint lineage is not formal evidence")

    if not _close(task_config.pin_nms_radius, AUDIT_NMS_RADIUS):
        raise RuntimeError(
            f"audit requires pin_nms_radius={AUDIT_NMS_RADIUS:g}; "
            f"config has {task_config.pin_nms_radius!r}"
        )
    if int(task_config.pin_max_detections) != AUDIT_MAX_DETECTIONS:
        raise RuntimeError(
            f"audit requires pin_max_detections={AUDIT_MAX_DETECTIONS}; "
            f"config has {task_config.pin_max_detections!r}"
        )

    materialization_root = workspace / "artifacts/training/materialized/dataset300-v2"
    task_root = materialization_root / "component-pin"
    summary_path = task_root / "summary.json"
    pin_index_path = task_root / "pin_index.jsonl"
    materialization_summary = _read_object(summary_path, "Pin materialization summary")
    expected_binding = status_lineage.get("binding_sha256")
    if materialization_summary.get("binding_sha256") != expected_binding:
        raise RuntimeError("Pin run/materialization binding mismatch")

    source_root = Path(preflight["paths"]["source_root"]).resolve(strict=True)
    dataset = PinKeypointDataset(
        pin_index_path,
        source_root,
        view_mode="roi",
        splits="val",
        expected_binding_sha256=str(expected_binding),
    )
    if not dataset:
        raise RuntimeError("frozen Pin validation split is empty")

    selected_device = _device(torch, arguments.device)
    model, checkpoint = load_pin_checkpoint(checkpoint_path, map_location="cpu")
    checkpoint_lineage = checkpoint.get("lineage")
    if not isinstance(checkpoint_lineage, dict):
        raise RuntimeError("Pin checkpoint lacks lineage")
    if checkpoint_lineage.get("training_config_sha256") not in compatible_config_sha256s:
        raise RuntimeError("Pin checkpoint/config training_config_sha256 mismatch")
    if arguments.interim_warm_start_diagnostic:
        if _lineage_identity(checkpoint_lineage) != _lineage_identity(status_lineage):
            raise RuntimeError("interim checkpoint identity differs from run status")
        expected_epoch = int(checkpoint_path.stem.rsplit("_", 1)[1])
        if checkpoint.get("epoch") != expected_epoch:
            raise RuntimeError("interim checkpoint filename/payload epoch mismatch")
    else:
        if checkpoint_lineage != status_lineage:
            raise RuntimeError("Pin checkpoint lineage differs from completed run status")
        if checkpoint.get("epoch") != status.get("best_epoch"):
            raise RuntimeError(
                "model.pt epoch does not match the completed run's best_epoch"
            )
    model.to(selected_device)
    model.eval()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(task_config.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=selected_device.type == "cuda",
        collate_fn=pin_keypoint_collate,
        drop_last=False,
    )
    thresholds = _thresholds(arguments.extra_threshold)
    totals = {
        threshold: {"matched": 0, "predicted": 0, "truth": 0}
        for threshold in thresholds
    }
    evaluated_views = 0
    validation_sample_ids: set[str] = set()
    with torch.inference_mode():
        for batch in loader:
            images, targets = _prepare_pin_batch(
                torch, batch, task_config, selected_device
            )
            metadata = targets.get("metadata")
            if not isinstance(metadata, list) or any(
                row.get("split") != "val" for row in metadata
            ):
                raise RuntimeError("non-validation record reached Pin threshold audit")
            evaluated_views += len(metadata)
            validation_sample_ids.update(str(row["sample_id"]) for row in metadata)
            outputs = split_pin_outputs(model(images))
            for threshold in thresholds:
                metrics = compute_pin_metrics(
                    outputs,
                    targets,
                    heatmap_stride=4,
                    score_threshold=threshold,
                    pck_thresholds=(AUDIT_MATCH_RADIUS_PX,),
                    nms_radius=AUDIT_NMS_RADIUS,
                    max_detections=AUDIT_MAX_DETECTIONS,
                    box_rects=targets.get("box_rects"),
                    box_margin_px=float(task_config.pin_box_margin_px),
                    box_prior_mode=AUDIT_BOX_PRIOR_MODE,
                )
                truth = int(round(float(metrics["gt_count"])))
                predicted = int(round(float(metrics["pred_count"])))
                matched = int(round(float(metrics["pck@4"]) * truth))
                totals[threshold]["matched"] += matched
                totals[threshold]["predicted"] += predicted
                totals[threshold]["truth"] += truth

    if evaluated_views != len(dataset):
        raise RuntimeError(
            f"validation coverage mismatch: evaluated={evaluated_views}, "
            f"dataset={len(dataset)}"
        )
    grid: list[dict[str, Any]] = []
    for threshold in thresholds:
        row = {
            "score_threshold": threshold,
            **_micro_summary(
                matched=totals[threshold]["matched"],
                predicted=totals[threshold]["predicted"],
                truth=totals[threshold]["truth"],
            ),
        }
        grid.append(row)
    best = max(
        grid,
        key=lambda row: (
            row["micro_f1_4px"],
            row["micro_recall_4px"],
            row["micro_precision_4px"],
            row["score_threshold"],
        ),
    )

    configured_row = next(
        (
            row
            for row in grid
            if _close(row["score_threshold"], task_config.pin_score_threshold)
        ),
        None,
    )
    status_metrics = status.get("metrics")
    configured_reconciliation: dict[str, Any] = {
        "configured_score_threshold": float(task_config.pin_score_threshold),
        "grid_contains_configured_threshold": configured_row is not None,
        "status_metrics_available": isinstance(status_metrics, dict),
        "matched": None,
    }
    if arguments.interim_warm_start_diagnostic:
        configured_reconciliation.update(
            {
                "comparable": False,
                "reason": (
                    "run-status metrics used filled_rect_v1; this audit deliberately "
                    "uses boundary_band_v2"
                ),
            }
        )
    elif configured_row is not None and isinstance(status_metrics, dict):
        comparisons = {
            "gt_count": (configured_row["gt_count"], status_metrics.get("gt_count")),
            "pred_count": (
                configured_row["pred_count"],
                status_metrics.get("pred_count"),
            ),
            "micro_precision_4px": (
                configured_row["micro_precision_4px"],
                status_metrics.get("pin_precision_4px"),
            ),
            "micro_recall_4px": (
                configured_row["micro_recall_4px"],
                status_metrics.get("pin_recall_4px", status_metrics.get("pck_4px")),
            ),
            "micro_f1_4px": (
                configured_row["micro_f1_4px"],
                status_metrics.get("pin_f1_4px"),
            ),
            "pred_to_gt_ratio": (
                configured_row["pred_to_gt_ratio"],
                status_metrics.get("pred_to_gt_ratio"),
            ),
        }
        configured_reconciliation["comparisons"] = {
            name: {
                "audit": left,
                "run_status": right,
                "equal": _close(left, right),
            }
            for name, (left, right) in comparisons.items()
        }
        configured_reconciliation["matched"] = all(
            row["equal"]
            for row in configured_reconciliation["comparisons"].values()
        )

    split_manifest_path = Path(preflight["paths"]["split_manifest"]).resolve(
        strict=True
    )
    evaluator_paths = (
        workspace / "src/pcbparse/training/torch_pin_model.py",
        workspace / "src/pcbparse/training/torch_backend.py",
    )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "pin",
        "assessment_scope": {
            "split": "val",
            "selection_role": "validation-only score-threshold selection",
            "checkpoint_role": checkpoint_role,
            "test_records_evaluated": 0,
            "test_split_used_for_tuning": False,
            "coverage": "all ROI views in the frozen validation split",
        },
        "protocol": {
            "required_score_thresholds": list(REQUIRED_THRESHOLDS),
            "evaluated_score_thresholds": list(thresholds),
            "matching": (
                "anonymous maximum-cardinality one-to-one Euclidean matching "
                "within 4 input-image pixels"
            ),
            "threshold_selection": (
                "maximize validation micro F1; tie-break by recall, precision, "
                "then higher threshold"
            ),
            "nms_radius_heatmap_cells": AUDIT_NMS_RADIUS,
            "max_detections_per_view": AUDIT_MAX_DETECTIONS,
            "component_box_prior_mode": AUDIT_BOX_PRIOR_MODE,
            "component_box_prior": "distance to rectangle perimeter",
            "box_prior_filter_order": "before NMS acceptance and max-detections truncation",
            "box_margin_input_pixels": float(task_config.pin_box_margin_px),
            "model_input_size": list(task_config.input_size),
            "output_stride": 4,
        },
        "lineage": {
            "training_config": {
                "path": config_path.relative_to(workspace).as_posix(),
                "file_sha256": _sha256_file(config_path),
                "canonical_sha256": config_sha256,
                "resume_compatible_sha256s": list(compatible_config_sha256s),
            },
            "split_manifest": {
                "path": split_manifest_path.relative_to(workspace).as_posix(),
                "sha256": _sha256_file(split_manifest_path),
            },
            "materialization_summary": {
                "path": summary_path.relative_to(workspace).as_posix(),
                "sha256": _sha256_file(summary_path),
                "binding_sha256": expected_binding,
            },
            "pin_index": {
                "path": pin_index_path.relative_to(workspace).as_posix(),
                "sha256": _sha256_file(pin_index_path),
            },
            "checkpoint": {
                "path": checkpoint_path.relative_to(workspace).as_posix(),
                "sha256": _sha256_file(checkpoint_path),
                "epoch": checkpoint.get("epoch"),
                "lineage_sha256": checkpoint.get("lineage_sha256"),
            },
            "run_status": {
                "path": status_path.relative_to(workspace).as_posix(),
                "sha256": _sha256_file(status_path),
                "status": status.get("status"),
                "best_epoch": status.get("best_epoch"),
                "completed_epochs": status.get("completed_epochs"),
                "planned_epochs": status.get("planned_epochs"),
                "finished_at_utc": status.get("finished_at_utc"),
            },
            "audit_script": {
                "path": Path(__file__).resolve().relative_to(workspace).as_posix(),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "evaluator_sources": [
                {
                    "path": path.relative_to(workspace).as_posix(),
                    "sha256": _sha256_file(path),
                }
                for path in evaluator_paths
            ],
        },
        "coverage": {
            "validation_views": evaluated_views,
            "validation_unique_sample_ids": len(validation_sample_ids),
            "ground_truth_pins": grid[0]["gt_count"],
        },
        "grid": grid,
        "best_validation_threshold": best,
        "configured_threshold_reconciliation": configured_reconciliation,
        "promotion": {
            "models_lock_modified": False,
            "blessed": False,
            "note": (
                "This is validation-only threshold evidence. The evaluated epoch "
                "checkpoint is not itself promoted or blessed; a v5 formal run "
                "must independently complete its readiness and end-to-end gates."
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    output_path.write_text(encoded, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "sha256": _sha256_file(output_path),
                "checkpoint_role": checkpoint_role,
                "box_prior_mode": AUDIT_BOX_PRIOR_MODE,
                "best_validation_threshold": best,
                "test_records_evaluated": 0,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
