from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from pcbparse import __version__
from pcbparse.config import config_hash
from pcbparse.pipeline.orchestrator import IRProvider, infer_one
from pcbparse.service.readiness import readiness_report


@dataclass(slots=True)
class APIServerConfig:
    input_root: Path
    output_root: Path
    pipeline_config: dict[str, Any]
    provider: IRProvider
    official_profile: str | Path | dict[str, Any] | None = None
    models_lock: str | Path | None = None
    allow_unready_debug: bool = False
    max_request_bytes: int = 64 * 1024
    max_concurrent_inferences: int = 1


def create_server(host: str, port: int, config: APIServerConfig) -> ThreadingHTTPServer:
    config.input_root = config.input_root.resolve()
    config.output_root = config.output_root.resolve()
    config.output_root.mkdir(parents=True, exist_ok=True)
    if config.max_concurrent_inferences < 1:
        raise ValueError("max_concurrent_inferences must be at least one")
    readiness = readiness_report(
        provider_name=type(config.provider).__name__,
        models_lock=config.models_lock,
        required_tasks=_required_model_tasks(config.pipeline_config),
    )
    output_locks: dict[Path, threading.Lock] = {}
    output_locks_guard = threading.Lock()
    inference_slots = threading.BoundedSemaphore(config.max_concurrent_inferences)

    class Handler(BaseHTTPRequestHandler):
        server_version = "pcbparse/0.1"

        def do_GET(self) -> None:  # noqa: N802
            request_path = urlparse(self.path).path
            if request_path == "/health":
                self._json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "version": __version__,
                        "config_hash": config_hash(config.pipeline_config),
                        "provider": type(config.provider).__name__,
                        "ready": readiness["ready"],
                    },
                )
                return
            if request_path == "/ready":
                self._json(HTTPStatus.OK if readiness["ready"] else HTTPStatus.SERVICE_UNAVAILABLE, readiness)
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/v1/infer":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            try:
                payload = self._read_payload()
                input_path = _safe_child(config.input_root, str(payload["input"]))
                if not readiness["ready"] and not config.allow_unready_debug:
                    self._json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {"error": "not_ready", "message": "provider/model readiness gate failed", "readiness": readiness},
                    )
                    return
                output_name = str(payload.get("output", f"{input_path.stem}.json"))
                if Path(output_name).name != output_name or not output_name.endswith(".json"):
                    raise ValueError("output must be a simple .json filename")
                output_path = _safe_child(config.output_root, output_name)
                if not inference_slots.acquire(blocking=False):
                    self._json(
                        HTTPStatus.TOO_MANY_REQUESTS,
                        {
                            "error": "inference_busy",
                            "message": "all bounded inference slots are busy; retry later",
                        },
                        headers={"Retry-After": "1"},
                    )
                    return
                try:
                    with output_locks_guard:
                        output_lock = output_locks.setdefault(output_path, threading.Lock())
                    if not output_lock.acquire(blocking=False):
                        self._json(
                            HTTPStatus.CONFLICT,
                            {"error": "output_busy", "message": "another request is writing this output"},
                        )
                        return
                    try:
                        result = infer_one(
                            input_path,
                            output_path,
                            provider=config.provider,
                            config=config.pipeline_config,
                            official_profile=config.official_profile,
                        )
                    finally:
                        output_lock.release()
                finally:
                    inference_slots.release()
                self._json(
                    HTTPStatus.OK,
                    {
                        "status": "ok" if readiness["ready"] else "debug",
                        "execution_mode": "production" if readiness["ready"] else "unready_debug",
                        "input": str(input_path.relative_to(config.input_root)),
                        "output": output_name,
                        "sha256": result.output_hash,
                        "counts": {
                            "components": len(result.output["components"]),
                            "pins": sum(len(group) for group in result.output["pins"].values()),
                            "nets": len(result.output["nets"]),
                        },
                        "trace": result.context.manifest()["trace"],
                    },
                )
            except KeyError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "missing_field", "message": str(exc)})
            except (ValueError, FileNotFoundError) as exc:
                self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": type(exc).__name__, "message": str(exc)})
            except Exception:
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "internal_error", "message": "inference failed; inspect server diagnostics"},
                )

        def _read_payload(self) -> dict[str, Any]:
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "application/json":
                raise ValueError("Content-Type must be application/json")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("invalid Content-Length") from exc
            if length <= 0 or length > config.max_request_bytes:
                raise ValueError("request body size is invalid")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def _json(
            self,
            status: HTTPStatus,
            payload: dict[str, Any],
            *,
            headers: dict[str, str] | None = None,
        ) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)


def serve(host: str, port: int, config: APIServerConfig) -> None:
    server = create_server(host, port, config)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("path escapes configured root") from exc
    if root == candidate:
        raise ValueError("path must identify a file under the configured root")
    return candidate


def _required_model_tasks(pipeline_config: dict[str, Any]) -> frozenset[str]:
    from pcbparse.perception.production_provider import (
        composite_provider_config_from_pipeline,
        required_model_tasks_for_config,
    )

    return required_model_tasks_for_config(
        composite_provider_config_from_pipeline(pipeline_config)
    )
