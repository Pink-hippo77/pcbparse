from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .artifacts import (
    EvaluationEvidence,
    ExportConsistency,
    build_inference_contract,
    finalize_candidate_bundle,
)
from .config import ModelTask, TrainingConfig, load_training_config
from .preflight import preflight_training_data


class TrainingRunnerError(RuntimeError):
    """Training did not produce a gateable, provenance-bound candidate."""


class TrainingBackendResult(BaseModel):
    """Framework-neutral result contract implemented by an optional backend."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    artifact_path: str = Field(min_length=1)
    predictions_path: str = Field(min_length=1)
    version: str = Field(min_length=1)
    license: str = Field(min_length=1)
    precision: str = Field(min_length=1)
    adapter_entrypoint: str = Field(min_length=1)
    export_consistency: ExportConsistency
    evaluation: EvaluationEvidence
    intended_use: str = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)


def run_configured_training(
    config_path: str | Path,
    *,
    task: ModelTask,
    output_directory: str | Path,
    workspace_root: str | Path | None = None,
) -> Path:
    """Invoke an explicitly configured backend and finalize a candidate bundle.

    The shipped tiny configuration intentionally has no backend entrypoint until
    a real framework/model implementation is installed.  This function fails
    closed before claiming a run, and an external backend cannot bypass data,
    export, metric, artifact-hash, or adapter-import gates.
    """

    root = Path(workspace_root).expanduser().resolve() if workspace_root else Path.cwd().resolve()
    config_file = Path(config_path).expanduser()
    if not config_file.is_absolute():
        config_file = root / config_file
    config: TrainingConfig = load_training_config(config_file)
    if task not in config.tasks:
        raise TrainingRunnerError(f"task {task!r} is absent from training configuration")
    preflight = preflight_training_data(config_file, workspace_root=root)
    if not preflight["ok"]:
        codes = sorted({issue["code"] for issue in preflight["issues"]})
        raise TrainingRunnerError(
            f"training data preflight failed ({', '.join(codes[:8])}); no training was started"
        )
    entrypoint = config.tasks[task].runner_entrypoint
    if entrypoint is None:
        raise TrainingRunnerError(
            f"task {task!r} has no runner_entrypoint; install/configure a real optional "
            "training backend before starting—no model has been trained"
        )
    module_name, attribute_name = entrypoint.rsplit(":", 1)
    try:
        runner = getattr(importlib.import_module(module_name), attribute_name)
    except (ImportError, AttributeError, ModuleNotFoundError) as exc:
        raise TrainingRunnerError(
            f"optional training backend {entrypoint!r} is unavailable: "
            f"{type(exc).__name__}: {exc}; no model has been trained"
        ) from exc
    if not callable(runner):
        raise TrainingRunnerError(f"training backend {entrypoint!r} is not callable")
    output = Path(output_directory).expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    raw_result: Any = runner(
        task=task,
        task_config=config.tasks[task],
        training_config=config,
        preflight_report=preflight,
        output_directory=output,
    )
    try:
        result = TrainingBackendResult.model_validate(raw_result)
    except ValueError as exc:
        raise TrainingRunnerError(
            f"training backend returned an invalid evidence contract: {exc}"
        ) from exc
    return finalize_candidate_bundle(
        output,
        artifact_path=result.artifact_path,
        evaluation_predictions_path=result.predictions_path,
        task=task,
        version=result.version,
        license_name=result.license,
        precision=result.precision,
        adapter_entrypoint=result.adapter_entrypoint,
        inference=build_inference_contract(task, config.tasks[task]),
        training_config_sha256=preflight["training_config_sha256"],
        dataset_config_hash=config.dataset.config_hash,
        source_manifest_sha256=config.dataset.source_manifest_sha256,
        split_manifest_sha256=config.dataset.split_manifest_sha256,
        gate_policy={
            "task_manifest_sha256": config.dataset.task_manifest_sha256[
                config.tasks[task].dataset_task
            ],
            "export_min_samples": config.tasks[task].export_min_samples,
            "export_max_abs_error": config.tasks[task].export_max_abs_error,
            "evaluation_gates": [
                gate.model_dump(mode="json")
                for gate in config.tasks[task].evaluation_gates
            ],
        },
        export_consistency=result.export_consistency,
        evaluation=result.evaluation,
        intended_use=result.intended_use,
        limitations=result.limitations,
    )


__all__ = ["TrainingBackendResult", "TrainingRunnerError", "run_configured_training"]
