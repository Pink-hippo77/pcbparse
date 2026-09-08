from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


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
    if os.environ.get("PCBPARSE_ALLOW_LOCAL_QWEN_CONFIG") == "1":
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
    for env_name, key in (
        ("QWEN_ENDPOINT", "endpoint"),
        ("QWEN_MODEL", "model"),
        ("QWEN_API_KEY", "api_key"),
    ):
        value = os.environ.get(env_name)
        if value:
            qwen[key] = value
    if qwen.get("endpoint") and qwen.get("model"):
        qwen["enabled"] = True
        qwen["backend"] = "http"


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

