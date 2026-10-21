"""Explicit, hash-bound pin candidate, separate from the baseline model lock."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pcbparse.perception.submission_ocr import _asset
from pcbparse.perception.torch_pin_adapter import create_pin_provider


def apply_submission_pin(base: Any, config: dict[str, Any], root: Path) -> dict[str, Any] | None:
    options = config.get("submission_pin")
    if options is None:
        return None
    if not isinstance(options, dict) or options.get("enabled") is not True:
        raise ValueError("submission_pin must be explicitly enabled")
    if set(options) - {"enabled", "model_path", "model_sha256", "split_sha256", "inference"}:
        raise ValueError("unsupported submission_pin options")
    path = _asset(root, options["model_path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != options.get("model_sha256"):
        raise ValueError("submission pin artifact SHA256 mismatch")
    inference = options.get("inference")
    if not isinstance(inference, dict) or inference.get("component_mask_channel") is not True:
        raise ValueError("submission pin candidate requires its trained ownership-mask input")
    entry = {"task": "pin", "sha256": digest, "artifact_format": "torchscript",
             "version": "clean-pin-v11-"+digest[:12], "inference": inference}
    candidate = create_pin_provider(path, entry)
    if not hasattr(base, "pin_provider"):
        raise TypeError("submission pin requires a composite provider")
    base.pin_provider = candidate
    return {"model_sha256": digest, "version": candidate.version,
            "split_sha256": options.get("split_sha256"),
            "baseline_lock_modified": False, "model_promotion": False}
