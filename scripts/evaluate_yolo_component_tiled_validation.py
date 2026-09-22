from __future__ import annotations

"""Evaluate a YOLO Component checkpoint with the fixed production-style replay.

The run is diagnostic-only.  It reuses the established 512/64 source-image
tiling, adapter threshold 0.20, and class-aware cross-tile NMS 0.55.  It never
edits ``models.lock.json`` and never reads train or test annotations.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import evaluate_component_tiled_validation as base  # noqa: E402
from pcbparse.perception.yolo_component_adapter import factory  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("work/yolo/component-yolo11n-v1/weights/best.pt"),
    )
    parser.add_argument(
        "--category-table",
        type=Path,
        default=Path("data/derived/pcbnet-upgrade-v1/component_yolo/classes.json"),
    )
    parser.add_argument(
        "--component-coco",
        type=Path,
        default=Path(
            "artifacts/training/materialized/dataset300-v2/"
            "component-pin/component_coco.json"
        ),
    )
    parser.add_argument(
        "--materialization-summary",
        type=Path,
        default=Path(
            "artifacts/training/materialized/dataset300-v2/"
            "component-pin/summary.json"
        ),
    )
    parser.add_argument(
        "--source-root", type=Path, default=Path("Dataset300/Dataset300")
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=Path("configs/pipeline.component512-experiment.yaml"),
    )
    parser.add_argument(
        "--thresholds-config",
        type=Path,
        default=Path("configs/thresholds.yaml"),
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/train/yolo_component_y11n_v1.json"),
    )
    parser.add_argument(
        "--training-results",
        type=Path,
        default=Path("work/yolo/component-yolo11n-v1/results.csv"),
    )
    parser.add_argument(
        "--training-log",
        type=Path,
        default=Path("work/yolo/component-yolo11n-v1.stdout.log"),
    )
    parser.add_argument(
        "--training-status",
        type=Path,
        default=None,
        help=(
            "Completed launcher_status.json. When supplied, this signed-by-hash "
            "record replaces terminal-log completion markers."
        ),
    )
    parser.add_argument(
        "--models-lock", type=Path, default=Path("models/models.lock.json")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work/eval-component-yolo11n-v1-tiled512-val"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--per-class-thresholds-audit",
        type=Path,
        default=None,
        help="Validation-only per-class threshold calibration to replay.",
    )
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser


def _category_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = base._sequence(payload.get("classes"), "YOLO category classes")
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        row = base._mapping(raw, "YOLO category row")
        normalized.append(
            {
                "model_label": int(row["yolo_class_id"]),
                "official_id": int(str(row["source_category_id"])),
                "name": str(row["source_category_name"]),
            }
        )
    normalized.sort(key=lambda item: item["model_label"])
    if [row["model_label"] for row in normalized] != list(range(len(normalized))):
        raise ValueError("YOLO category labels are not contiguous")
    return normalized


def _verify_category_binding(
    yolo_rows: Sequence[Mapping[str, Any]],
    index_rows: Sequence[Mapping[str, Any]],
) -> None:
    yolo = sorted((int(row["official_id"]), str(row["name"])) for row in yolo_rows)
    index = sorted((int(row["official_id"]), str(row["name"])) for row in index_rows)
    if yolo != index:
        raise ValueError("YOLO and frozen Component category tables differ")


def _training_completion(
    results_path: Path,
    log_path: Path | None,
    *,
    status_path: Path | None = None,
    checkpoint_path: Path | None = None,
    training_config_path: Path | None = None,
) -> dict[str, Any]:
    import csv

    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("YOLO results.csv is empty")
    parsed = [
        {
            "epoch": int(row["epoch"]),
            "map50": float(row["metrics/mAP50(B)"]),
            "map50_95": float(row["metrics/mAP50-95(B)"]),
        }
        for row in rows
    ]
    best = max(parsed, key=lambda row: (row["map50_95"], -row["epoch"]))
    if status_path is not None:
        status = base._read_object(status_path, "YOLO launcher status")
        if status.get("kind") != "pcbparse_yolo_component_training_launcher_status":
            raise ValueError("unexpected YOLO launcher status kind")
        if status.get("status") != "training_complete" or status.get("ok") is not True:
            raise ValueError("YOLO launcher status does not confirm successful completion")
        if status.get("mode") != "train":
            raise ValueError("YOLO launcher status is not a full training run")
        if status.get("models_lock_modified") is not False:
            raise ValueError("YOLO launcher status does not preserve models.lock.json")

        completed_epochs = parsed[-1]["epoch"]
        epoch_indexing = status.get("epoch_indexing")
        legacy_epoch_fields = epoch_indexing is None and status.get("best_epoch") is None
        recorded_completed_epochs = int(status.get("completed_epochs", -1))
        expected_recorded_epochs = (
            completed_epochs + 1 if legacy_epoch_fields else completed_epochs
        )
        if recorded_completed_epochs != expected_recorded_epochs:
            raise ValueError("YOLO launcher status completed epoch count differs from results.csv")
        status_best_epoch = status.get("best_epoch")
        if status_best_epoch is None and status.get("best_epoch_zero_based") is not None:
            status_best_epoch = int(status["best_epoch_zero_based"])
            if not legacy_epoch_fields:
                status_best_epoch += 1
        if int(status_best_epoch if status_best_epoch is not None else -1) != best["epoch"]:
            raise ValueError("YOLO launcher status best epoch differs from results.csv")
        if abs(float(status.get("best_map50", -1.0)) - best["map50"]) > 1e-9:
            raise ValueError("YOLO launcher status best mAP50 differs from results.csv")
        if abs(float(status.get("best_map50_95", -1.0)) - best["map50_95"]) > 1e-9:
            raise ValueError("YOLO launcher status best mAP50-95 differs from results.csv")

        artifacts = base._mapping(status.get("artifacts"), "launcher status artifacts")
        results_artifact = base._mapping(
            artifacts.get("results.csv"), "launcher results.csv artifact"
        )
        if str(results_artifact.get("sha256", "")).lower() != base._sha256(results_path):
            raise ValueError("YOLO launcher status results.csv hash mismatch")
        if checkpoint_path is not None:
            checkpoint_artifact = base._mapping(
                artifacts.get("weights/best.pt"), "launcher best checkpoint artifact"
            )
            if str(checkpoint_artifact.get("sha256", "")).lower() != base._sha256(
                checkpoint_path
            ):
                raise ValueError("YOLO launcher status best checkpoint hash mismatch")
        if training_config_path is not None:
            config_artifact = base._mapping(
                status.get("training_config"), "launcher training config"
            )
            if str(config_artifact.get("sha256", "")).lower() != base._sha256(
                training_config_path
            ):
                raise ValueError("YOLO launcher status training config hash mismatch")

        configured_epochs = int(status.get("configured_epochs", -1))
        stopped_before_limit = bool(status.get("stopped_before_epoch_limit"))
        recorded_limit_comparison = recorded_completed_epochs < configured_epochs
        if stopped_before_limit != recorded_limit_comparison:
            raise ValueError("YOLO launcher status early-stop fields are inconsistent")
        return {
            "completed_epochs": completed_epochs,
            "best_epoch": best["epoch"],
            "best_map50": best["map50"],
            "best_map50_95": best["map50_95"],
            "early_stopping_confirmed": stopped_before_limit,
            "success_marker_confirmed": True,
            "completion_evidence": "launcher_status",
            "launcher_epoch_fields_normalized": legacy_epoch_fields,
        }

    if log_path is None:
        raise ValueError("training log is required when launcher status is absent")
    # The training log contains ANSI progress bytes and escaped non-ASCII
    # paths.  Completion markers are ASCII, so verify them on raw bytes rather
    # than making the contract depend on terminal text decoding.
    log_bytes = log_path.read_bytes()
    # Ultralytics may insert an ANSI reset sequence between the label and the
    # message, so check the two stable phrases independently.
    completion_marker = (
        b"EarlyStopping:" in log_bytes and b"Training stopped early" in log_bytes
    )
    success_marker = b'"ok": true' in log_bytes
    if not completion_marker or not success_marker:
        raise ValueError("YOLO training log lacks successful early-stop completion markers")
    return {
        "completed_epochs": parsed[-1]["epoch"],
        "best_epoch": best["epoch"],
        "best_map50": best["map50"],
        "best_map50_95": best["map50_95"],
        "early_stopping_confirmed": True,
        "success_marker_confirmed": True,
        "completion_evidence": "terminal_log",
    }


def _per_class_thresholds_from_audit(
    path: Path,
    *,
    workspace: Path,
    checkpoint_sha256: str,
    population_floor: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    audit = base._read_object(path, "per-class threshold calibration")
    payload = dict(audit)
    claimed = payload.pop("calibration_payload_sha256", None)
    if claimed != base._canonical_sha256(payload):
        raise ValueError("per-class threshold calibration payload hash mismatch")
    if (
        audit.get("kind")
        != "pcbparse_yolo_component_validation_per_class_threshold_calibration"
        or audit.get("status") != "validation_calibration_complete"
        or audit.get("diagnostic_only") is not True
        or audit.get("promotion_eligible") is not False
    ):
        raise ValueError("per-class threshold calibration identity mismatch")
    raw_thresholds = base._mapping(
        audit.get("selected_thresholds"), "selected per-class thresholds"
    )
    thresholds = {
        str(name): base._finite(value, f"threshold {name}")
        for name, value in raw_thresholds.items()
    }
    if any(value < population_floor or value > 1.0 for value in thresholds.values()):
        raise ValueError("per-class threshold falls outside the saved population")

    sources = base._mapping(audit.get("sources"), "calibration sources")
    source_status_entry = base._mapping(
        sources.get("tiled_status"), "calibration source tiled status"
    )
    source_status_path = base._resolve_within(
        workspace,
        base._nonempty_string(source_status_entry.get("path"), "source status path"),
        "calibration source tiled status",
    )
    if base._sha256(source_status_path) != source_status_entry.get("sha256"):
        raise ValueError("calibration source tiled status hash mismatch")
    source_status = base._read_object(source_status_path, "calibration source status")
    source_checkpoint = base._mapping(
        base._mapping(source_status.get("lineage"), "source status lineage").get(
            "checkpoint"
        ),
        "source checkpoint",
    )
    if source_checkpoint.get("sha256") != checkpoint_sha256:
        raise ValueError("calibration was selected for a different checkpoint")
    return thresholds, audit


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    workspace = Path(arguments.workspace_root).expanduser().resolve(strict=True)
    resolve = base._resolve_within
    checkpoint_path = resolve(workspace, arguments.checkpoint, "YOLO checkpoint")
    category_path = resolve(workspace, arguments.category_table, "YOLO category table")
    component_coco_path = resolve(workspace, arguments.component_coco, "Component COCO")
    summary_path = resolve(
        workspace, arguments.materialization_summary, "materialization summary"
    )
    source_root = resolve(
        workspace, arguments.source_root, "Dataset300 source root", kind="directory"
    )
    pipeline_path = resolve(workspace, arguments.pipeline_config, "pipeline config")
    thresholds_path = resolve(workspace, arguments.thresholds_config, "thresholds config")
    training_config_path = resolve(
        workspace, arguments.training_config, "YOLO training config"
    )
    training_results_path = resolve(
        workspace, arguments.training_results, "YOLO training results"
    )
    training_status_path = (
        resolve(workspace, arguments.training_status, "YOLO launcher status")
        if arguments.training_status is not None
        else None
    )
    training_log_path = (
        None
        if training_status_path is not None
        else resolve(workspace, arguments.training_log, "YOLO training log")
    )
    models_lock_path = resolve(workspace, arguments.models_lock, "models lock")
    output_dir = resolve(
        workspace,
        arguments.output_dir,
        "output directory",
        kind="directory",
        must_exist=False,
    )
    base._prepare_output(output_dir, workspace)

    lock_sha256_before = base._sha256(models_lock_path)
    summary = base._read_object(summary_path, "materialization summary")
    binding_sha256 = base._nonempty_string(
        summary.get("binding_sha256"), "materialization binding_sha256"
    )
    protocol = base._fixed_protocol(
        base._read_object(pipeline_path, "pipeline config"),
        base._read_object(thresholds_path, "thresholds config"),
        summary,
    )
    component_coco = base._read_object(component_coco_path, "Component COCO")
    samples, index_categories, selection_boundary = base._frozen_validation_samples(
        component_coco=component_coco,
        source_root=source_root,
        binding_sha256=binding_sha256,
        max_val_samples=arguments.max_val_samples,
    )
    category_payload = base._read_object(category_path, "YOLO category table")
    yolo_categories = _category_rows(category_payload)
    _verify_category_binding(yolo_categories, index_categories)
    completion = _training_completion(
        training_results_path,
        training_log_path,
        status_path=training_status_path,
        checkpoint_path=checkpoint_path,
        training_config_path=training_config_path,
    )

    checkpoint_sha256 = base._sha256(checkpoint_path)
    per_class_audit_path = (
        resolve(
            workspace,
            arguments.per_class_thresholds_audit,
            "per-class threshold calibration",
        )
        if arguments.per_class_thresholds_audit is not None
        else None
    )
    per_class_thresholds: dict[str, float] = {}
    per_class_audit: dict[str, Any] | None = None
    if per_class_audit_path is not None:
        per_class_thresholds, per_class_audit = _per_class_thresholds_from_audit(
            per_class_audit_path,
            workspace=workspace,
            checkpoint_sha256=checkpoint_sha256,
            population_floor=float(protocol["score_threshold"]),
        )
    adapter = factory(
        checkpoint_path,
        {
            "task": "component",
            "sha256": checkpoint_sha256,
            "category_table_path": str(category_path),
            "category_table_sha256": base._sha256(category_path),
            "inference": {
                "device": str(arguments.device),
                "score_threshold": float(protocol["score_threshold"]),
                "nms_iou": float(protocol["component_nms_iou"]),
                "max_detections": int(protocol["max_detections_per_tile"]),
                "imgsz": int(arguments.imgsz),
                "per_class_score_thresholds": per_class_thresholds,
            },
        },
    )
    predictions_path = output_dir / "predictions.jsonl"
    prediction_rows, performance = base._run_inference(
        samples=samples,
        adapter=adapter,
        protocol=protocol,
        predictions_path=predictions_path,
    )
    ground_truth_by_sample = {
        str(item["sample_id"]): list(item["_ground_truth"]) for item in samples
    }
    metrics = base._evaluate_predictions(
        prediction_rows,
        ground_truth_by_sample,
        index_categories,
        iou_threshold=float(protocol["match_iou"]),
    )
    complete_validation = len(samples) == selection_boundary["available_val_samples"]
    lock_sha256_after = base._sha256(models_lock_path)
    if lock_sha256_before != lock_sha256_after:
        raise RuntimeError("models.lock.json changed during diagnostic evaluation")
    status: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pcbparse_yolo_component_tiled_frozen_val_diagnostic",
        "task": "component",
        "status": (
            "diagnostic_evaluation_complete"
            if complete_validation
            else "diagnostic_smoke_complete"
        ),
        "registerable": False,
        "formal_evidence": False,
        "promotion_eligible": False,
        "blessing_authorized": False,
        "smoke": not complete_validation,
        "finished_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "device": str(arguments.device),
        "adapter_version": adapter.version,
        "protocol": {
            **protocol,
            "model_imgsz": int(arguments.imgsz),
            "per_class_score_thresholds": per_class_thresholds,
            "per_class_thresholds_validation_tuned": per_class_audit is not None,
        },
        "selection_boundary": selection_boundary,
        "metrics": metrics,
        "performance": performance,
        "training_completion": completion,
        "artifacts": {
            "predictions": {
                "path": base._relative(predictions_path, workspace),
                "sha256": base._sha256(predictions_path),
                "row_count": len(prediction_rows),
            }
        },
        "lineage": {
            "checkpoint": {
                "path": base._relative(checkpoint_path, workspace),
                "sha256": checkpoint_sha256,
            },
            "category_table": {
                "path": base._relative(category_path, workspace),
                "sha256": base._sha256(category_path),
            },
            "component_coco": {
                "path": base._relative(component_coco_path, workspace),
                "sha256": base._sha256(component_coco_path),
                "binding_sha256": binding_sha256,
            },
            "materialization_summary": {
                "path": base._relative(summary_path, workspace),
                "sha256": base._sha256(summary_path),
            },
            "pipeline_config": {
                "path": base._relative(pipeline_path, workspace),
                "sha256": base._sha256(pipeline_path),
            },
            "thresholds_config": {
                "path": base._relative(thresholds_path, workspace),
                "sha256": base._sha256(thresholds_path),
            },
            "training_config": {
                "path": base._relative(training_config_path, workspace),
                "sha256": base._sha256(training_config_path),
            },
            "training_results": {
                "path": base._relative(training_results_path, workspace),
                "sha256": base._sha256(training_results_path),
            },
            **(
                {
                    "per_class_threshold_calibration": {
                        "path": base._relative(per_class_audit_path, workspace),
                        "sha256": base._sha256(per_class_audit_path),
                        "payload_sha256": per_class_audit[
                            "calibration_payload_sha256"
                        ],
                    }
                }
                if per_class_audit_path is not None and per_class_audit is not None
                else {}
            ),
            **(
                {
                    "training_status": {
                        "path": base._relative(training_status_path, workspace),
                        "sha256": base._sha256(training_status_path),
                    }
                }
                if training_status_path is not None
                else {
                    "training_log": {
                        "path": base._relative(training_log_path, workspace),
                        "sha256": base._sha256(training_log_path),
                    }
                }
            ),
            "models_lock": {
                "path": base._relative(models_lock_path, workspace),
                "sha256_before": lock_sha256_before,
                "sha256_after": lock_sha256_after,
                "modified": False,
            },
        },
        "interpretation_boundary": {
            "validation_configuration_tuned": per_class_audit is not None,
            "test_split_used": False,
            "models_lock_modified": False,
            "full_validation_conclusion_allowed": complete_validation,
            "direct_promotion_allowed": False,
        },
        "limitations": [
            "This replay is diagnostic-only and cannot register, bless, or promote a model.",
            *(
                [
                    "The shared floor, tiling, and NMS remain fixed; only hash-bound validation-selected per-class thresholds are replayed."
                ]
                if per_class_audit is not None
                else [
                    "The fixed threshold, tiling, and NMS were declared before this YOLO replay; no search is performed."
                ]
            ),
            *(
                [
                    "Per-class thresholds were selected on this validation split; replay metrics are selection-biased."
                ]
                if per_class_audit is not None
                else []
            ),
            "Only frozen validation images and annotations are scored; train and test usage is zero.",
            *(
                []
                if complete_validation
                else [
                    "This smoke uses a lexicographic validation prefix and cannot estimate full-validation performance."
                ]
            ),
        ],
    }
    status["diagnostic_payload_sha256"] = base._canonical_sha256(status)
    status_path = output_dir / "diagnostic_status.json"
    base._atomic_write_json(status_path, status)
    print(status_path)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    print(f"diagnostic_payload_sha256={status['diagnostic_payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
