from __future__ import annotations

"""Re-evaluate a completed Component checkpoint with one explicit score floor.

This is a diagnostic-only, frozen-validation replay.  It exists because
TorchVision applies an internal RoIHeads score threshold before the project
metric applies its configured threshold.  The script verifies the parent run,
training configuration, checkpoint bytes, and materialization binding; uses
only full views from the validation split; and emits new predictions without
changing the parent run or making a model registerable.
"""

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from pcbparse.training.config import (
    load_training_config,
    training_config_resume_sha256s,
    training_config_sha256,
)
from pcbparse.training.torch_backend import _component_predictions
from pcbparse.training.torch_component_model import evaluate_map50, load_checkpoint
from pcbparse.training.torch_component_pin import (
    ComponentDetectionDataset,
    component_detection_collate,
)


FINAL_FORMAL_STATUSES = frozenset({"candidate_evidence_ready", "gate_failed"})


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Replay a completed formal Component checkpoint on frozen validation "
            "with one explicitly declared TorchVision/application score floor."
        )
    )
    parser.add_argument("--workspace-root", type=Path, default=root)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train/dataset300_v2_component_v4.json"),
    )
    parser.add_argument(
        "--parent-run-status",
        type=Path,
        default=Path("work/train-component-v4/torch_run_status.json"),
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
        "--output-dir",
        type=Path,
        default=Path("work/eval-component-v4-scorefloor0"),
    )
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    return parser


def _resolve_within(
    root: Path, raw: str | Path, label: str, *, must_exist: bool
) -> Path:
    path = Path(raw).expanduser()
    resolved = (path if path.is_absolute() else root / path).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside workspace: {resolved}") from exc
    if must_exist and (not resolved.is_file() or resolved.stat().st_size <= 0):
        raise FileNotFoundError(f"{label} is missing or empty: {resolved}")
    return resolved


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _gate_results(task_config: Any, metrics: Mapping[str, float]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for gate in task_config.evaluation_gates:
        if gate.metric not in metrics:
            raise ValueError(f"diagnostic metric lacks configured gate {gate.metric!r}")
        value = float(metrics[gate.metric])
        threshold = float(gate.threshold)
        passed = value >= threshold if gate.operator == "gte" else value <= threshold
        output.append(
            {
                "metric": gate.metric,
                "operator": gate.operator,
                "threshold": threshold,
                "value": value,
                "passed": passed,
            }
        )
    return output


def main() -> int:
    arguments = _parser().parse_args()
    workspace = arguments.workspace_root.expanduser().resolve(strict=True)
    config_path = _resolve_within(workspace, arguments.config, "training config", must_exist=True)
    parent_status_path = _resolve_within(
        workspace, arguments.parent_run_status, "parent run status", must_exist=True
    )
    component_coco_path = _resolve_within(
        workspace, arguments.component_coco, "Component COCO", must_exist=True
    )
    summary_path = _resolve_within(
        workspace,
        arguments.materialization_summary,
        "materialization summary",
        must_exist=True,
    )
    output = _resolve_within(
        workspace, arguments.output_dir, "output directory", must_exist=False
    )
    if output.exists():
        entries = list(output.iterdir()) if output.is_dir() else [output]
        if entries:
            raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    score_threshold = float(arguments.score_threshold)
    if not math.isfinite(score_threshold) or not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score-threshold must be finite and in [0, 1]")

    config = load_training_config(config_path)
    task_config = config.tasks["component"]
    config_sha256 = training_config_sha256(config)
    parent = _read_object(parent_status_path, "parent Component run status")
    if parent.get("task") != "component":
        raise ValueError("parent run status task is not component")
    if parent.get("status") not in FINAL_FORMAL_STATUSES:
        raise ValueError("parent Component run is not in a completed formal state")
    if parent.get("formal_evidence") is not True or parent.get("smoke") is not False:
        raise ValueError("parent Component run is not formal non-smoke evidence")
    if not parent.get("finished_at_utc"):
        raise ValueError("parent Component run lacks a completion timestamp")
    parent_lineage = _mapping(parent.get("lineage"), "parent.lineage")
    parent_config_sha256 = parent_lineage.get("training_config_sha256")
    compatible_config_sha256s = training_config_resume_sha256s(config)
    if parent_config_sha256 not in compatible_config_sha256s:
        raise ValueError(
            "training config semantic hash is not in the explicitly supported "
            "no-op schema history for the parent run"
        )

    artifact = _mapping(parent.get("artifact"), "parent.artifact")
    checkpoint_path = _resolve_within(
        workspace, str(artifact.get("path", "")), "parent Component checkpoint", must_exist=True
    )
    checkpoint_sha256 = _sha256(checkpoint_path)
    if checkpoint_sha256 != artifact.get("sha256"):
        raise ValueError("parent Component checkpoint SHA-256 mismatch")

    summary = _read_object(summary_path, "materialization summary")
    binding_sha256 = summary.get("binding_sha256")
    if not isinstance(binding_sha256, str) or len(binding_sha256) != 64:
        raise ValueError("materialization summary lacks binding_sha256")
    if parent_lineage.get("binding_sha256") != binding_sha256:
        raise ValueError("parent run/materialization binding mismatch")

    source_root = _resolve_within(
        workspace, config.dataset.source_root, "Dataset300 source root", must_exist=False
    )
    if not source_root.is_dir():
        raise FileNotFoundError(f"Dataset300 source root is not a directory: {source_root}")

    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError("PyTorch is required for Component re-evaluation") from exc
    device = torch.device(str(arguments.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    dataset = ComponentDetectionDataset(
        component_coco_path,
        source_root,
        splits="val",
        view_kinds="full",
        expected_binding_sha256=binding_sha256,
    )
    if len(dataset) <= 0:
        raise ValueError("frozen Component validation dataset is empty")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(task_config.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=component_detection_collate,
    )

    loaded = load_checkpoint(
        checkpoint_path,
        device=device,
        expected_sha256=checkpoint_sha256,
        strict=True,
    )
    if loaded.lineage.get("binding_sha256") != binding_sha256:
        raise ValueError("checkpoint/materialization binding mismatch")
    roi_heads = getattr(loaded.model, "roi_heads", None)
    if roi_heads is None or not hasattr(roi_heads, "score_thresh"):
        raise ValueError("loaded Component model lacks RoIHeads.score_thresh")
    internal_score_threshold_before = float(roi_heads.score_thresh)
    roi_heads.score_thresh = score_threshold

    result = evaluate_map50(
        loaded.model,
        loader,
        device,
        score_threshold=score_threshold,
        category_table=loaded.category_table,
    )
    metrics = {
        "bbox_map50": float(result["map50"]),
        "precision50": float(result["precision50"]),
        "recall50": float(result["recall50"]),
    }
    predictions_path = output / "predictions.jsonl"
    _component_predictions(
        torch,
        loaded.model,
        loader,
        device,
        predictions_path,
        max_batches=None,
    )
    predictions_sha256 = _sha256(predictions_path)
    gates = _gate_results(task_config, metrics)
    finished_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    parent_sha256 = _sha256(parent_status_path)
    status: dict[str, Any] = {
        "schema_version": 1,
        "task": "component",
        "status": "diagnostic_evaluation_complete",
        "registerable": False,
        "formal_evidence": False,
        "smoke": False,
        "finished_at_utc": finished_at,
        "device": str(device),
        "metrics": metrics,
        "gates": gates,
        "gates_observed_passed": all(bool(item["passed"]) for item in gates),
        "score_contract": {
            "configured_component_score_threshold": float(
                task_config.component_score_threshold
            ),
            "torchvision_internal_before": internal_score_threshold_before,
            "torchvision_internal_applied": score_threshold,
            "metric_filter_applied": score_threshold,
            "thresholds_searched": False,
        },
        "selection_boundary": {
            "split": "val",
            "view_kind": "full",
            "validation_view_rows_used": len(dataset),
            "train_rows_used": 0,
            "test_rows_used": 0,
        },
        "predictions": {
            "path": predictions_path.relative_to(workspace).as_posix(),
            "sha256": predictions_sha256,
        },
        "lineage": dict(parent_lineage),
        "diagnostic_parent": {
            "run_status": {
                "path": parent_status_path.relative_to(workspace).as_posix(),
                "sha256": parent_sha256,
                "status": parent["status"],
            },
            "checkpoint": {
                "path": checkpoint_path.relative_to(workspace).as_posix(),
                "sha256": checkpoint_sha256,
            },
            "training_config": {
                "path": config_path.relative_to(workspace).as_posix(),
                "current_semantic_sha256": config_sha256,
                "parent_semantic_sha256": parent_config_sha256,
                "compatible_semantic_sha256s": list(compatible_config_sha256s),
            },
            "component_coco": {
                "path": component_coco_path.relative_to(workspace).as_posix(),
                "sha256": _sha256(component_coco_path),
            },
            "materialization_summary": {
                "path": summary_path.relative_to(workspace).as_posix(),
                "sha256": _sha256(summary_path),
                "binding_sha256": binding_sha256,
            },
        },
        "limitations": [
            "This replay is diagnostic-only and cannot register or bless a model.",
            "It changes no weights and performs no threshold search.",
            "Only the frozen validation split was scored; train and test rows were not used.",
        ],
    }
    status["diagnostic_payload_sha256"] = _canonical_sha256(status)
    status_path = output / "diagnostic_status.json"
    _write_json(status_path, status)
    print(status_path)
    print(json.dumps(metrics, sort_keys=True, separators=(",", ":")))
    print(f"diagnostic_payload_sha256={status['diagnostic_payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
