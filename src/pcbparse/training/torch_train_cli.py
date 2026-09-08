from __future__ import annotations

"""Command-line entry point for the fail-closed PyTorch backend.

Invoke with ``python -m pcbparse.training.torch_train_cli`` from the optional
training environment.  Formal runs produce a candidate bundle only; this
module never writes ``models.lock.json`` and never marks a model blessed.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from pcbparse.training.artifacts import (
    CandidateBundleError,
    build_inference_contract,
    finalize_candidate_bundle,
)
from pcbparse.training.config import MODEL_TASKS, load_training_config
from pcbparse.training.preflight import preflight_training_data
from pcbparse.training.runner import TrainingBackendResult
from pcbparse.training.torch_backend import (
    TorchSmokeRunComplete,
    TorchTrainingError,
    run_torch_task,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one frozen Dataset300 task with real PyTorch and create an "
            "unblessed candidate bundle only after all formal gates pass."
        )
    )
    parser.add_argument("--config", required=True, help="strict training JSON")
    parser.add_argument("--task", required=True, choices=MODEL_TASKS)
    parser.add_argument("--output", required=True, help="task candidate directory")
    parser.add_argument(
        "--workspace-root",
        default=".",
        help="workspace containing Dataset300 and fixed materializations",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or cuda:N (default: auto)",
    )
    parser.add_argument(
        "--precision",
        choices=("fp16", "fp32", "bf16"),
        help="override autocast precision; CPU is always effective fp32",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="bounded diagnostic; cannot create candidate evidence",
    )
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--smoke-train-batches", type=int, default=1)
    parser.add_argument("--smoke-val-batches", type=int, default=1)
    parser.add_argument("--smoke-export-samples", type=int, default=1)
    parser.add_argument(
        "--raw-result-only",
        action="store_true",
        help="do not finalize the already-gated raw result into a candidate bundle",
    )
    parser.add_argument(
        "--resume-checkpoint",
        help=(
            "resume a formal run from its output/resume_checkpoint.pt; the "
            "checkpoint must match the task, config, data, architecture and optimizer"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    for name in (
        "smoke_epochs",
        "smoke_train_batches",
        "smoke_val_batches",
        "smoke_export_samples",
    ):
        if getattr(arguments, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if arguments.raw_result_only and arguments.smoke:
        parser.error("--raw-result-only has no effect on a non-registerable smoke run")
    if arguments.resume_checkpoint and arguments.smoke:
        parser.error("--resume-checkpoint is available only for formal runs")

    workspace = Path(arguments.workspace_root).expanduser().resolve(strict=False)
    config_path = Path(arguments.config).expanduser()
    if not config_path.is_absolute():
        config_path = workspace / config_path
    output = Path(arguments.output).expanduser()
    if not output.is_absolute():
        output = workspace / output
    output = output.resolve(strict=False)
    resume_checkpoint_arg = arguments.resume_checkpoint
    if not resume_checkpoint_arg and not arguments.smoke:
        # 自动断点续训：输出目录存在 resume_checkpoint.pt 时自动继续，
        # 中断后重跑同一命令即可从上次进度继续，无需手动指定。
        auto_resume = output / "resume_checkpoint.pt"
        if auto_resume.is_file():
            resume_checkpoint_arg = auto_resume
            print(
                f"[resume] auto-resuming from {auto_resume}",
                file=sys.stderr,
            )
    if resume_checkpoint_arg:
        resume = Path(resume_checkpoint_arg).expanduser()
        if not resume.is_absolute():
            resume = workspace / resume
        os.environ["PCBPARSE_TRAIN_RESUME_CHECKPOINT"] = str(
            resume.resolve(strict=False)
        )
    else:
        os.environ.pop("PCBPARSE_TRAIN_RESUME_CHECKPOINT", None)
    os.environ["PCBPARSE_TRAIN_DEVICE"] = arguments.device
    if arguments.precision:
        os.environ["PCBPARSE_TRAIN_PRECISION"] = arguments.precision
    else:
        os.environ.pop("PCBPARSE_TRAIN_PRECISION", None)
    if arguments.smoke:
        os.environ["PCBPARSE_TRAIN_SMOKE"] = "1"
        os.environ["PCBPARSE_TRAIN_SMOKE_EPOCHS"] = str(arguments.smoke_epochs)
        os.environ["PCBPARSE_TRAIN_SMOKE_TRAIN_BATCHES"] = str(
            arguments.smoke_train_batches
        )
        os.environ["PCBPARSE_TRAIN_SMOKE_VAL_BATCHES"] = str(
            arguments.smoke_val_batches
        )
        os.environ["PCBPARSE_TRAIN_SMOKE_EXPORT_SAMPLES"] = str(
            arguments.smoke_export_samples
        )
    else:
        for name in (
            "PCBPARSE_TRAIN_SMOKE",
            "PCBPARSE_TRAIN_SMOKE_EPOCHS",
            "PCBPARSE_TRAIN_SMOKE_TRAIN_BATCHES",
            "PCBPARSE_TRAIN_SMOKE_VAL_BATCHES",
            "PCBPARSE_TRAIN_SMOKE_EXPORT_SAMPLES",
        ):
            os.environ.pop(name, None)

    try:
        config = load_training_config(config_path)
        preflight = preflight_training_data(config_path, workspace_root=workspace)
        if not preflight.get("ok"):
            codes = sorted(
                {
                    str(issue.get("code", "UNKNOWN"))
                    for issue in preflight.get("issues", [])
                }
            )
            raise TorchTrainingError(
                "training preflight failed: " + ", ".join(codes[:12])
            )
        task_config = config.tasks[arguments.task]
        raw = run_torch_task(
            task=arguments.task,
            task_config=task_config,
            training_config=config,
            preflight_report=preflight,
            output_directory=output,
        )
        result = TrainingBackendResult.model_validate(raw)
        if arguments.raw_result_only:
            print(
                json.dumps(
                    result.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
            )
            return 0
        manifest = finalize_candidate_bundle(
            output,
            artifact_path=result.artifact_path,
            evaluation_predictions_path=result.predictions_path,
            task=arguments.task,
            version=result.version,
            license_name=result.license,
            precision=result.precision,
            adapter_entrypoint=result.adapter_entrypoint,
            inference=build_inference_contract(arguments.task, task_config),
            training_config_sha256=preflight["training_config_sha256"],
            dataset_config_hash=config.dataset.config_hash,
            source_manifest_sha256=config.dataset.source_manifest_sha256,
            split_manifest_sha256=config.dataset.split_manifest_sha256,
            gate_policy={
                "task_manifest_sha256": config.dataset.task_manifest_sha256[
                    task_config.dataset_task
                ],
                "export_min_samples": task_config.export_min_samples,
                "export_max_abs_error": task_config.export_max_abs_error,
                "evaluation_gates": [
                    gate.model_dump(mode="json")
                    for gate in task_config.evaluation_gates
                ],
            },
            export_consistency=result.export_consistency,
            evaluation=result.evaluation,
            intended_use=result.intended_use,
            limitations=result.limitations,
        )
        print(
            json.dumps(
                {
                    "status": "candidate",
                    "blessed": False,
                    "bundle_manifest": str(manifest),
                    "models_lock_modified": False,
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    except TorchSmokeRunComplete as exc:
        print(
            json.dumps(
                {
                    "status": "smoke_complete",
                    "formal_evidence": False,
                    "registerable": False,
                    "report": str(exc.report_path),
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    except (CandidateBundleError, TorchTrainingError, ValueError, OSError) as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "registerable": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
