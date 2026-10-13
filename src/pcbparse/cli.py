from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from pcbparse import __version__
from pcbparse.config import load_config
from pcbparse.pipeline.orchestrator import infer_batch, infer_one
from pcbparse.pipeline.providers import EmptyImageProvider
from pcbparse.service.http_api import APIServerConfig, serve
from pcbparse.service.readiness import ReadinessGateError, readiness_report


_PRODUCTION_PROFILES = frozenset({"baseline", "competition"})
_PROVIDER_CHOICES = ("production", "dataset_golden", "dataset_debug", "heuristic", "empty")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pcbparse", description="PCB schematic parsing baseline")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    infer = commands.add_parser("infer", help="Run deterministic inference")
    infer.add_argument("--input", required=True, type=Path)
    infer.add_argument("--output", required=True, type=Path)
    infer.add_argument("--config", type=Path, default=_resource_path("pipeline.json"))
    infer.add_argument("--profile", default="baseline")
    infer.add_argument("--provider", choices=_PROVIDER_CHOICES)
    infer.add_argument("--models-lock", type=Path, help="Model lock used by the production readiness gate")
    infer.add_argument(
        "--allow-unready-debug",
        action="store_true",
        help="Explicitly permit degraded heuristic inference when a production profile is unready",
    )
    infer.add_argument("--fail-fast", action="store_true")
    infer.add_argument(
        "--evaluation-ir-dir",
        type=Path,
        help=(
            "Persist final Circuit-IR plus a prediction manifest for raw-v1 evaluation; "
            "annotation providers are marked sidecar_access=true and will be rejected by accuracy mode"
        ),
    )
    infer.add_argument(
        "--evidence-dir",
        type=Path,
        help=(
            "Persist four-stage IR snapshots (perception/association/pre_prune/final) "
            "for offline OCR diagnostics; never pollutes official JSON output"
        ),
    )

    health = commands.add_parser("healthcheck", help="Run import/config/serializer smoke checks")
    health.add_argument("--config", type=Path, default=_resource_path("pipeline.json"))
    health.add_argument("--provider", choices=_PROVIDER_CHOICES, default="production")
    health.add_argument("--models-lock", type=Path, help="Model lock to verify for readiness")
    health.add_argument("--require-ready", action="store_true")

    api = commands.add_parser("serve", help="Serve filesystem-scoped JSON inference API")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8080)
    api.add_argument("--input-root", required=True, type=Path)
    api.add_argument("--output-root", required=True, type=Path)
    api.add_argument("--config", type=Path, default=_resource_path("pipeline.json"))
    api.add_argument("--profile", default="baseline")
    api.add_argument("--provider", choices=_PROVIDER_CHOICES)
    api.add_argument("--models-lock", type=Path, help="Model lock used by the service readiness gate")
    api.add_argument(
        "--allow-unready-debug",
        action="store_true",
        help="Permit requests to use an unready provider; local debugging only",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "healthcheck":
            return _healthcheck(args)
        if args.command == "infer":
            return _infer(args)
        if args.command == "serve":
            return _serve(args)
    except KeyboardInterrupt:
        return 130
    except ReadinessGateError as exc:
        print(
            json.dumps(
                {
                    "status": "unready",
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "readiness": exc.report,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(json.dumps({"status": "error", "type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 2


def _healthcheck(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    runtime_smoke: dict[str, Any] = {"status": "not_requested"}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "smoke.png"
        Image.new("RGB", (16, 16), "white").save(source)
        result = infer_one(
            source,
            root / "smoke.json",
            provider=EmptyImageProvider(),
            config=config,
            official_profile=_resource_path("official_schema.json"),
        )
        readiness = readiness_report(
            provider_name=args.provider,
            models_lock=args.models_lock,
            required_tasks=_required_model_tasks(config),
            require_lineage=args.require_ready and args.provider == "production",
        )
        if args.require_ready and readiness["ready"]:
            try:
                production_provider = _provider(
                    args.provider,
                    models_lock=args.models_lock,
                    pipeline_config=config,
                )
                smoke_ir = production_provider.build_ir(source)
                if smoke_ir.state.value != "PERCEIVED":
                    raise RuntimeError(
                        f"production provider smoke returned state {smoke_ir.state.value!r}"
                    )
                runtime_smoke = {
                    "status": "ok",
                    "provider": type(production_provider).__name__,
                    "components": len(smoke_ir.components),
                    "pins": len(smoke_ir.pins),
                    "texts": len(smoke_ir.texts),
                    "segments": len(smoke_ir.segments),
                    "junctions": len(smoke_ir.junctions),
                    "perception_config": _provider_perception_config(production_provider),
                }
            except Exception as exc:
                issue = {
                    "code": "provider_load_or_smoke_failed",
                    "message": (
                        "verified model metadata could not construct and execute the production "
                        f"provider: {type(exc).__name__}: {exc}"
                    ),
                }
                readiness = copy.deepcopy(readiness)
                readiness["ready"] = False
                readiness["status"] = "unready"
                readiness.setdefault("issues", []).append(issue)
                readiness.setdefault("problems", []).append(issue["message"])
                runtime_smoke = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
    status = "unready" if args.require_ready and not readiness["ready"] else "ok"
    print(
        json.dumps(
            {
                "status": status,
                "liveness": "ok",
                "readiness": readiness,
                "runtime_smoke": runtime_smoke,
                "version": __version__,
                "serializer_sha256": result.output_hash,
                "stages": [item.stage for item in result.context.trace],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 1 if args.require_ready and not readiness["ready"] else 0


def _infer(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    profile = _load_profile(args.profile)
    config = _apply_profile(config, profile)
    provider_name = args.provider or str(profile.get("provider", "dataset_golden"))
    readiness = readiness_report(
        provider_name=provider_name,
        models_lock=args.models_lock,
        required_tasks=_required_model_tasks(config),
        require_lineage=provider_name == "production",
    )
    execution_mode = _infer_execution_mode(
        profile,
        provider_name=provider_name,
        readiness=readiness,
        allow_unready_debug=args.allow_unready_debug,
    )
    provider = _provider(
        provider_name,
        models_lock=args.models_lock,
        pipeline_config=config,
    )
    inputs = _discover_images(args.input)
    if not inputs:
        raise FileNotFoundError(f"no decodable images found under {args.input}")
    evidence_sink = (
        _make_evidence_sink(args.evidence_dir) if args.evidence_dir is not None else None
    )
    results, errors = infer_batch(
        inputs,
        args.output,
        provider=provider,
        config=config,
        official_profile=_resource_path("official_schema.json"),
        continue_on_error=not args.fail_fast,
        evidence_sink=evidence_sink,
    )
    pipeline_status = "ok" if not errors else "partial"
    evaluation_manifest = (
        _write_evaluation_artifacts(
            results,
            args.evaluation_ir_dir,
            provider_name=provider_name,
            models_lock=args.models_lock,
        )
        if args.evaluation_ir_dir is not None
        else None
    )
    status = {
        "debug": "debug",
        "production": pipeline_status,
        "unready_debug": "degraded",
    }[execution_mode]
    print(
        json.dumps(
            {
                "status": status,
                "pipeline_status": pipeline_status,
                "execution_mode": execution_mode,
                "degraded": execution_mode == "unready_debug",
                "debug": execution_mode != "production",
                "provider": provider_name,
                "perception_config": _provider_perception_config(provider),
                "readiness": readiness,
                "inputs": len(inputs),
                "succeeded": len(results),
                "failed": len(errors),
                "outputs": [str(result.output_path) for result in results],
                "error_manifest": (
                    str(args.output / "error_manifest.json") if errors else None
                ),
                "evaluation_manifest": (
                    str(evaluation_manifest) if evaluation_manifest is not None else None
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not errors else 1


def _serve(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    profile = _load_profile(args.profile)
    config = _apply_profile(config, profile)
    provider_name = args.provider or str(profile.get("provider", "dataset_golden"))
    provider = _provider(
        provider_name,
        models_lock=args.models_lock,
        pipeline_config=config,
    )
    api_config = APIServerConfig(
        input_root=args.input_root,
        output_root=args.output_root,
        pipeline_config=config,
        provider=provider,
        official_profile=_resource_path("official_schema.json"),
        models_lock=args.models_lock,
        allow_unready_debug=args.allow_unready_debug,
    )
    print(
        json.dumps(
            {
                "status": "serving",
                "host": args.host,
                "port": args.port,
                "provider": provider_name,
                "perception_config": _provider_perception_config(provider),
            },
            sort_keys=True,
        )
    )
    serve(args.host, args.port, api_config)
    return 0


def _load_profile(name: str) -> dict[str, Any]:
    path = Path(name)
    if not path.exists():
        workspace_path = Path("configs/profiles") / f"{name}.yaml"
        path = workspace_path if workspace_path.exists() else _resource_path(f"profiles/{name}.json")
    return load_config(path)


def _resource_path(relative: str) -> Path:
    path = Path(__file__).resolve().parent / "resources" / relative
    if not path.is_file():
        raise FileNotFoundError(f"packaged resource is missing: {relative}")
    return path


def _apply_profile(config: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    required = {"name", "provider", "qwen_enabled", "artifact_policy", "strict_validation"}
    missing = sorted(required - set(profile))
    if missing:
        raise ValueError(f"profile is missing required fields: {missing}")
    effective = copy.deepcopy(config)
    run = effective.setdefault("run", {})
    qwen = effective.setdefault("qwen", {})
    validation = effective.setdefault("validation", {})
    if not all(isinstance(item, dict) for item in (run, qwen, validation)):
        raise ValueError("pipeline run/qwen/validation sections must be objects")
    run["profile"] = str(profile["name"])
    run["artifact_policy"] = str(profile["artifact_policy"])
    qwen_env = os.environ.get("PCBPARSE_QWEN_ENABLED")
    if qwen_env is not None:
        # Deployment-time override: operators can force Qwen on/off without
        # editing committed profiles.  Credentials still come from env vars.
        qwen["enabled"] = qwen_env.strip().lower() in {"1", "true", "yes", "on"}
    else:
        qwen["enabled"] = bool(profile["qwen_enabled"])
    validation["require_zero_errors"] = bool(profile["strict_validation"])
    return effective


def _infer_execution_mode(
    profile: dict[str, Any],
    *,
    provider_name: str,
    readiness: dict[str, Any],
    allow_unready_debug: bool,
) -> str:
    profile_name = str(profile.get("name", "")).strip().lower()
    if not profile_name:
        raise ValueError("profile name must be a non-empty string")
    if provider_name == "heuristic":
        if allow_unready_debug:
            return "unready_debug"
        raise ReadinessGateError(readiness, allow_hint=True)
    if profile_name in _PRODUCTION_PROFILES:
        if readiness.get("ready") is True:
            return "production"
        raise ReadinessGateError(readiness)
    return "debug"


def _provider(
    name: str,
    *,
    models_lock: str | Path | None = None,
    pipeline_config: dict[str, Any] | None = None,
):
    if name == "production":
        if pipeline_config is not None and "pin_package" in pipeline_config:
            raise ValueError(
                "Pin package config is experimental; use python -m data_tools.pin_package_infer. "
                "The production entry point uses models/models.lock.json."
            )
        from pcbparse.perception.production_provider import (
            composite_provider_config_from_pipeline,
            load_production_provider,
        )

        lock_path = models_lock or Path("models/models.lock.json")
        return load_production_provider(
            lock_path,
            config=composite_provider_config_from_pipeline(pipeline_config),
        )
    if name == "empty":
        return EmptyImageProvider()
    if name in {"dataset_golden", "dataset_debug"}:
        from pcbparse.perception.golden_provider import DatasetGoldenProvider

        try:
            return DatasetGoldenProvider(policy="dataset_debug" if name == "dataset_debug" else "strict")
        except TypeError:
            if name == "dataset_debug":
                raise RuntimeError("dataset_debug policy is not available in this build")
            return DatasetGoldenProvider()
    if name == "heuristic":
        try:
            from pcbparse.perception.heuristic_provider import HeuristicVisionProvider
        except ImportError:
            return EmptyImageProvider()
        return HeuristicVisionProvider()
    raise ValueError(f"unknown provider {name!r}")


def _provider_perception_config(provider: Any) -> dict[str, Any] | None:
    from pcbparse.perception.production_provider import resolved_perception_config

    return resolved_perception_config(provider)


def _required_model_tasks(pipeline_config: dict[str, Any]) -> frozenset[str]:
    from pcbparse.perception.production_provider import (
        composite_provider_config_from_pipeline,
        required_model_tasks_for_config,
    )

    return required_model_tasks_for_config(
        composite_provider_config_from_pipeline(pipeline_config)
    )


def _discover_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    images: list[Path] = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file() or "__MACOSX" in candidate.parts or candidate.name.startswith("._"):
            continue
        if candidate.suffix.lower() in {".json", ".eds", ".net"} or candidate.name == ".DS_Store":
            continue
        try:
            with Image.open(candidate) as image:
                image.verify()
            images.append(candidate)
        except OSError:
            continue
    return images


def _make_evidence_sink(directory: Path):
    destination = directory.expanduser().resolve(strict=False)
    destination.mkdir(parents=True, exist_ok=True)

    def sink(stage: str, ir: Any) -> None:
        payload = json.dumps(
            ir.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        name = f"{stage}.circuit-ir.json"
        _atomic_bytes(destination / name, payload)

    return sink


def _write_evaluation_artifacts(
    results: list[Any],
    output_directory: Path,
    *,
    provider_name: str,
    models_lock: Path | None,
) -> Path:
    destination = output_directory.expanduser().resolve(strict=False)
    destination.mkdir(parents=True, exist_ok=True)
    annotation_backed = provider_name in {"dataset_golden", "dataset_debug"}
    lock_digest = _sha256_path(models_lock) if models_lock is not None else None
    rows: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for result in sorted(results, key=lambda item: item.input_path.as_posix()):
        parent_name = result.input_path.parent.name
        case_id = parent_name if re.fullmatch(r"\d{4}", parent_name) else result.input_path.stem
        if case_id in seen_case_ids:
            raise ValueError(
                f"evaluation artifact case_id collision: {case_id!r}; use unique numbered sample folders"
            )
        seen_case_ids.add(case_id)
        ir_name = f"{case_id}.circuit-ir.json"
        ir_path = destination / ir_name
        payload = json.dumps(
            result.ir.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        _atomic_bytes(ir_path, payload)
        rows.append(
            {
                "case_id": case_id,
                "ir_path": ir_name,
                "provider_kind": provider_name,
                "sidecar_access": annotation_backed,
                "image_sha256": result.ir.image.sha256,
                "config_hash": result.context.config_hash,
                "model_lock_sha256": lock_digest,
                "input_paths_read": (
                    [result.ir.image.path, "adjacent *_CVJsonStd.json"]
                    if annotation_backed
                    else [result.ir.image.path]
                ),
            }
        )
    manifest = destination / "predictions.jsonl"
    rendered = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode("utf-8")
    _atomic_bytes(manifest, rendered)
    return manifest


def _sha256_path(path: Path) -> str:
    target = path.expanduser().resolve(strict=False)
    if not target.is_file():
        raise FileNotFoundError(f"model lock does not exist: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
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


if __name__ == "__main__":
    raise SystemExit(main())
