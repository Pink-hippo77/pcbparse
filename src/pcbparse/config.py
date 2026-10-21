from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON-compatible YAML.

    Project configuration files deliberately use the JSON subset of YAML so the
    baseline stays dependency-light while remaining valid YAML 1.2.
    """

    config_path = Path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{config_path} must use the JSON-compatible YAML subset: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a top-level object")
    _apply_qwen_overlay(data, config_path)
    return data


def _apply_qwen_overlay(config: dict[str, Any], config_path: str | Path) -> None:
    """Enable the Qwen reasoner from explicit environment variables.

    Credentials stay out of the committed pipeline config.  Environment
    variables (`QWEN_ENDPOINT` / `QWEN_MODEL` / `QWEN_API_KEY`) supply the
    endpoint and model.  A local credentials file is accepted only when
    `PCBPARSE_ALLOW_LOCAL_QWEN_CONFIG=1` is explicitly set for local development.
    """

    qwen = config.setdefault("qwen", {})
    if not isinstance(qwen, dict):
        qwen = {}
        config["qwen"] = qwen
    candidates: list[Path] = []
    official_config = any(os.environ.get(name) for name in ('LLM_BASE_URL', 'LLM_MODEL', 'LLM_API_KEY'))
    legacy_config = any(os.environ.get(name) for name in ('QWEN_ENDPOINT', 'QWEN_MODEL', 'QWEN_API_KEY'))
    if os.environ.get("PCBPARSE_ALLOW_LOCAL_QWEN_CONFIG") == "1" and not (official_config or legacy_config):
        config_dir = Path(config_path).expanduser().resolve().parent
        candidates = [config_dir / "qwen_local.json"]
        for parent in (config_dir, config_dir.parent, config_dir.parent.parent):
            candidates.append(parent / "configs" / "qwen_local.json")
        candidates.append(Path.cwd() / "configs" / "qwen_local.json")
    overlay: dict[str, Any] = {}
    for local_path in candidates:
        if local_path.is_file():
            try:
                candidate = json.loads(local_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                candidate = {}
            if isinstance(candidate, dict):
                overlay = {**candidate, **overlay}
    if overlay:
        try:
            for key in (
                "endpoint",
                "model",
                "api_key",
                "timeout_seconds",
                "route_threshold",
                "max_calls_per_image",
                "max_input_tokens_per_image",
                "max_output_tokens_per_call",
                "min_time_reserve_seconds",
                "send_images",
            ):
                value = overlay.get(key)
                if value not in (None, ""):
                    qwen[key] = value
        except (TypeError, ValueError):
            pass
    # Treat an injected credential family atomically. A partial official
    # configuration must not borrow a stale personal key or model.
    env_fields = (("LLM_BASE_URL", "endpoint"), ("LLM_MODEL", "model"), ("LLM_API_KEY", "api_key")) if official_config else (
        ("QWEN_ENDPOINT", "endpoint"), ("QWEN_MODEL", "model"), ("QWEN_API_KEY", "api_key"))
    for env_name, key in env_fields:
        value = os.environ.get(env_name)
        if value:
            qwen[key] = value
    if qwen.get("endpoint") and qwen.get("model"):
        try:
            qwen['endpoint'] = qwen_chat_endpoint(str(qwen['endpoint']))
        except ValueError:
            # Optional reasoning must not prevent visual inference from
            # starting. Record a fixed reason, never the supplied URL/key.
            qwen["enabled"] = False
            qwen["backend"] = "disabled"
            qwen["configuration_error"] = "invalid_endpoint"
            return
        qwen["enabled"] = True
        qwen["backend"] = "http"
        # An explicit runtime endpoint/model must not silently retain the
        # disabled profile's zero call/token budgets. Keep positive limits.
        for key, default in (("max_calls_per_image", 4), ("max_input_tokens_per_image", 8192),
                             ("max_output_tokens_per_call", 192)):
            if int(qwen.get(key, 0)) <= 0:
                qwen[key] = default


def qwen_chat_endpoint(value: str) -> str:
    """Accept either an OpenAI-compatible base URL or a completion endpoint."""
    parts = urlsplit(value.strip())
    if parts.scheme not in {'http', 'https'} or not parts.netloc or parts.username or parts.password:
        raise ValueError('Qwen endpoint must be an HTTP(S) URL without embedded credentials')
    path = parts.path.rstrip('/')
    if not path.endswith('/chat/completions'):
        path += '/chat/completions' if path.endswith('/v1') else '/v1/chat/completions' if not path else '/chat/completions'
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ''))


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

