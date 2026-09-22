from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from pcbparse import __version__
from pcbparse.cli import _apply_profile
from pcbparse.config import config_hash, load_config
from pcbparse.output.official_serializer import (
    atomic_write_official_json,
    official_json_hash,
)
from pcbparse.perception.production_provider import (
    composite_provider_config_from_pipeline,
    load_production_provider,
    required_model_tasks_for_config,
)
from pcbparse.pipeline.orchestrator import infer_one
from pcbparse.service.readiness import readiness_report


_EMPTY_OFFICIAL_OUTPUT: dict[str, dict[str, Any]] = {
    "components": {},
    "pins": {},
    "nets": {},
}


class SubmissionAgentError(RuntimeError):
    """Raised when the EDARace batch contract cannot be satisfied."""


@dataclass(frozen=True, slots=True)
class RuntimeBundle:
    provider: Any
    config: dict[str, Any]
    readiness: dict[str, Any]
    config_sha256: str
    model_lock_sha256: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="/submission/agent_system",
        description="Run PCBParse over one EDARace image batch",
    )
    parser.add_argument("-input", "--input", dest="input_dir", type=Path, required=True)
    parser.add_argument("-output", "--output", dest="output_dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_submission(args.input_dir, args.output_dir)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "type": type(exc).__name__,
                    "message": _safe_error_message(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    return 0


def run_submission(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    app_root: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the official one-shot batch contract.

    The evaluator supplies one read-only directory of PNG files.  This function
    never reads annotation sidecars and always attempts to publish exactly one
    ``<case_name>_Pred.json`` for every discovered image.  A case-local pipeline
    failure produces a schema-valid empty prediction so the remaining batch can
    still be evaluated; the failure is preserved in ``decision_log.json``.
    """

    started_monotonic = time.monotonic()
    started_at = _utc_now()
    source = Path(input_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not source.is_dir():
        raise SubmissionAgentError(f"input directory does not exist: {source}")
    if not destination.is_dir():
        raise SubmissionAgentError(f"output directory does not exist: {destination}")

    inputs = _discover_images(source)
    final_results = destination / "final_results"
    final_results.mkdir(parents=True, exist_ok=True)
    logger = _build_logger(final_results / "run.log")
    logger.info("EDARace batch started: input=%s cases=%d", source, len(inputs))

    root = _resolve_app_root(app_root)
    runtime: RuntimeBundle | None = None
    initialization_error: Exception | None = None
    try:
        runtime = _load_runtime(root)
        logger.info(
            "Production runtime ready: config=%s model_lock=%s",
            runtime.config_sha256,
            runtime.model_lock_sha256,
        )
    except Exception as exc:
        initialization_error = exc
        logger.exception("Production runtime initialization failed")

    total_limit = _total_time_limit(runtime.config if runtime is not None else None)
    cases: list[dict[str, Any]] = []
    token_totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    for image_path in inputs:
        case_started = time.monotonic()
        output_path = final_results / f"{image_path.stem}_Pred.json"
        if initialization_error is not None:
            case = _write_fallback_case(
                image_path,
                output_path,
                initialization_error,
                elapsed_seconds=0.0,
                reason="runtime_initialization_failed",
            )
        elif time.monotonic() - started_monotonic >= total_limit:
            case = _write_fallback_case(
                image_path,
                output_path,
                SubmissionAgentError(
                    f"whole-program time budget of {total_limit:.3f}s was reached"
                ),
                elapsed_seconds=0.0,
                reason="total_time_budget_exhausted",
            )
        else:
            assert runtime is not None
            try:
                result = infer_one(
                    image_path,
                    output_path,
                    provider=runtime.provider,
                    config=runtime.config,
                )
                manifest = result.context.manifest()
                _merge_token_totals(token_totals, manifest.get("tokens"))
                diagnostics = _diagnostic_summary(result.diagnostics)
                case = {
                    "case_name": image_path.stem,
                    "input_file": image_path.name,
                    "output_file": output_path.relative_to(destination).as_posix(),
                    "status": "ok",
                    "degraded": diagnostics["warning"] > 0,
                    "fallback": False,
                    "elapsed_seconds": _rounded(time.monotonic() - case_started),
                    "input_sha256": _sha256_file(image_path),
                    "output_sha256": result.output_hash,
                    "diagnostics": diagnostics,
                    "model_versions": manifest.get("model_versions", {}),
                    "tokens": _token_mapping(manifest.get("tokens")),
                    "error": None,
                }
            except Exception as exc:
                manifest = getattr(exc, "context_manifest", None)
                if isinstance(manifest, Mapping):
                    _merge_token_totals(token_totals, manifest.get("tokens"))
                case = _write_fallback_case(
                    image_path,
                    output_path,
                    exc,
                    elapsed_seconds=time.monotonic() - case_started,
                    reason="case_inference_failed",
                )
                if isinstance(manifest, Mapping):
                    case["tokens"] = _token_mapping(manifest.get("tokens"))
                    case["model_versions"] = manifest.get("model_versions", {})
                logger.exception("Case %s fell back to an empty prediction", image_path.stem)
        cases.append(case)
        logger.info(
            "Case %s complete: status=%s elapsed=%.3fs",
            image_path.stem,
            case["status"],
            case["elapsed_seconds"],
        )

    _assert_exact_output_coverage(inputs, final_results)
    fallback_count = sum(bool(case["fallback"]) for case in cases)
    degraded_count = sum(bool(case["degraded"]) for case in cases)
    report = {
        "schema_version": 1,
        "kind": "edarace-agent-decision-log",
        "agent": {
            "name": "pcb-schematic-parser",
            "version": __version__,
        },
        "status": "completed" if fallback_count == 0 else "completed_with_fallbacks",
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "elapsed_seconds": _rounded(time.monotonic() - started_monotonic),
        "input_dir": str(source),
        "output_dir": str(destination),
        "final_results_dir": "final_results",
        "submission_mode": "degraded",
        "runtime": _runtime_record(runtime, initialization_error),
        "token_usage": token_totals,
        "summary": {
            "status": "ok" if fallback_count == 0 else "degraded",
            "total_cases": len(cases),
            "successful_cases": len(cases) - fallback_count,
            "fallback_cases": fallback_count,
            "degraded_cases": degraded_count,
            "output_files": len(cases),
        },
        "cases": cases,
    }
    _atomic_write_json(destination / "decision_log.json", report)
    logger.info(
        "EDARace batch completed: cases=%d fallback=%d elapsed=%.3fs",
        len(cases),
        fallback_count,
        report["elapsed_seconds"],
    )
    _close_logger(logger)
    return report


def _resolve_app_root(value: str | Path | None) -> Path:
    if value is not None:
        return Path(value).expanduser().resolve()
    environment = os.environ.get("PCBPARSE_APP_ROOT")
    if environment:
        return Path(environment).expanduser().resolve()
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "configs").is_dir() and (source_root / "models").is_dir():
        return source_root
    return Path("/submission/app")


def _load_runtime(app_root: Path) -> RuntimeBundle:
    config_path = app_root / "configs" / "pipeline.edarace.yaml"
    profile_path = app_root / "configs" / "profiles" / "competition.yaml"
    lock_path = app_root / "models" / "models.lock.json"
    for path, label in (
        (config_path, "EDARace pipeline config"),
        (profile_path, "competition profile"),
        (lock_path, "model lock"),
    ):
        if not path.is_file():
            raise SubmissionAgentError(f"{label} is missing: {path}")

    config = _apply_profile(load_config(config_path), load_config(profile_path))
    run_config = config.get("run", {})
    if not isinstance(run_config, Mapping) or run_config.get("submission_mode") != "degraded":
        raise SubmissionAgentError(
            "EDARace config must explicitly set run.submission_mode='degraded'"
        )
    perception_config = composite_provider_config_from_pipeline(config)
    required_tasks = required_model_tasks_for_config(perception_config)
    readiness = readiness_report(
        provider_name="production",
        models_lock=lock_path,
        required_tasks=required_tasks,
        require_lineage=True,
    )
    if readiness.get("ready") is not True or readiness.get("status") != "ready":
        raise SubmissionAgentError(
            "production readiness failed: "
            + json.dumps(readiness.get("problems", []), ensure_ascii=False)
        )
    provider = load_production_provider(lock_path, config=perception_config)
    return RuntimeBundle(
        provider=provider,
        config=config,
        readiness=readiness,
        config_sha256=config_hash(config),
        model_lock_sha256=_sha256_file(lock_path),
    )


def _discover_images(input_dir: Path) -> list[Path]:
    images = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if not images:
        raise SubmissionAgentError(f"no PNG images found in input directory: {input_dir}")
    collisions = [
        stem
        for stem, count in Counter(path.stem.casefold() for path in images).items()
        if count > 1
    ]
    if collisions:
        raise SubmissionAgentError(
            f"input image stems collide after case folding: {sorted(collisions)}"
        )
    return images


def _write_fallback_case(
    image_path: Path,
    output_path: Path,
    error: Exception,
    *,
    elapsed_seconds: float,
    reason: str,
) -> dict[str, Any]:
    atomic_write_official_json(_EMPTY_OFFICIAL_OUTPUT, output_path)
    diagnostics = getattr(error, "diagnostics", [])
    return {
        "case_name": image_path.stem,
        "input_file": image_path.name,
        "output_file": f"final_results/{output_path.name}",
        "status": "fallback_empty",
        "degraded": True,
        "fallback": True,
        "fallback_reason": reason,
        "elapsed_seconds": _rounded(elapsed_seconds),
        "input_sha256": _sha256_file(image_path),
        "output_sha256": official_json_hash(_EMPTY_OFFICIAL_OUTPUT),
        "diagnostics": _diagnostic_summary(diagnostics),
        "model_versions": {},
        "tokens": {"calls": 0, "input_tokens": 0, "output_tokens": 0},
        "error": {
            "type": type(error).__name__,
            "message": _safe_error_message(error),
        },
    }


def _assert_exact_output_coverage(inputs: Iterable[Path], final_results: Path) -> None:
    expected = {f"{path.stem}_Pred.json" for path in inputs}
    actual = {
        path.name
        for path in final_results.iterdir()
        if path.is_file() and path.name.endswith("_Pred.json")
    }
    if actual != expected:
        raise SubmissionAgentError(
            "prediction output coverage mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _runtime_record(
    runtime: RuntimeBundle | None, initialization_error: Exception | None
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": platform.python_version(),
        "provider": "production",
        "llm": {
            "enabled": False,
            "model": os.environ.get("LLM_MODEL"),
        },
    }
    if runtime is not None:
        record.update(
            {
                "ready": True,
                "config_sha256": runtime.config_sha256,
                "model_lock_sha256": runtime.model_lock_sha256,
                "required_model_tasks": runtime.readiness.get(
                    "required_learned_tasks", []
                ),
                "verified_model_tasks": runtime.readiness.get(
                    "verified_learned_tasks", []
                ),
            }
        )
    else:
        record.update(
            {
                "ready": False,
                "initialization_error": (
                    {
                        "type": type(initialization_error).__name__,
                        "message": _safe_error_message(initialization_error),
                    }
                    if initialization_error is not None
                    else None
                ),
            }
        )
    return record


def _diagnostic_summary(diagnostics: Iterable[Any]) -> dict[str, Any]:
    severities: Counter[str] = Counter()
    codes: Counter[str] = Counter()
    for diagnostic in diagnostics:
        severity = str(getattr(diagnostic, "severity", "UNKNOWN")).upper()
        code = str(getattr(diagnostic, "code", "UNKNOWN"))
        severities[severity] += 1
        codes[code] += 1
    return {
        "total": sum(severities.values()),
        "error": severities["ERROR"],
        "warning": severities["WARNING"],
        "info": severities["INFO"],
        "codes": dict(sorted(codes.items())),
    }


def _total_time_limit(config: Mapping[str, Any] | None) -> float:
    if config is None:
        return 7000.0
    run = config.get("run", {})
    if not isinstance(run, Mapping):
        return 7000.0
    value = run.get("max_total_seconds", 7000.0)
    if type(value) not in {int, float} or float(value) <= 0:
        raise SubmissionAgentError("run.max_total_seconds must be positive")
    return float(value)


def _token_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0}
    return {
        key: int(value.get(key, 0))
        for key in ("calls", "input_tokens", "output_tokens")
    }


def _merge_token_totals(total: dict[str, int], value: Any) -> None:
    for key, item in _token_mapping(value).items():
        total[key] += item


def _build_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"pcbparse.submission.{os.getpid()}.{time.monotonic_ns()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_error_message(error: Exception | None) -> str:
    if error is None:
        return ""
    return str(error).replace("\x00", "")[:2000]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _rounded(value: float) -> float:
    return round(max(0.0, float(value)), 6)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RuntimeBundle",
    "SubmissionAgentError",
    "build_parser",
    "main",
    "run_submission",
]
