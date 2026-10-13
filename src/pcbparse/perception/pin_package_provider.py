"""Explicit experimental Pin replacement without changing the production lock."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

from pcbparse.perception.composite_provider import CompositePerceptionProvider
from pcbparse.perception.pin_package_adapter import PinPackageAdapter
from pcbparse.perception.production_provider import (
    _apply_ocr_inference_overrides, _apply_wire_inference_overrides,
    _resolve_model_path, _verify_provider_contract, composite_provider_config_from_pipeline,
    required_model_tasks_for_config,
)
from pcbparse.pipeline.context import stable_file_hash
from pcbparse.service.readiness import ReadinessGateError, readiness_report


class PinPackageExperimentalProvider(CompositePerceptionProvider):
    version = "pin-package-experimental-v1"
    provider_kind = "pin_package_experiment"
    execution_mode = "experiment"

    def build_ir(self, input_path, ctx=None):
        ir = super().build_ir(input_path, ctx)
        ir.metadata["pin_package_experiment"] = dict(self.experiment_metadata)
        ir.metadata["execution_mode"] = self.execution_mode
        ir.metadata["provider_kind"] = self.provider_kind
        return ir


def load_pin_package_experiment(models_lock: str | Path, pipeline_config: dict, *, device: str = "cpu") -> PinPackageExperimentalProvider:
    run = pipeline_config.get("run", {})
    if run.get("profile") != "pin_package_experiment" or run.get("submission_mode", "strict") != "strict":
        raise ValueError("Pin package requires pin_package_experiment profile and strict submission mode")
    if pipeline_config.get("validation", {}).get("require_zero_errors") is not True:
        raise ValueError("Pin package experiment requires validation.require_zero_errors=true")
    package = pipeline_config.get("pin_package", {})
    if package.get("experimental") is not True:
        raise ValueError("Pin package must be explicitly marked experimental")
    if pipeline_config.get("pin"):
        raise ValueError("legacy Pin decoder overrides cannot be applied to the package adapter")
    config = composite_provider_config_from_pipeline(pipeline_config)
    lock_path = Path(models_lock).resolve()
    report = readiness_report(provider_name="production", models_lock=lock_path,
                              required_tasks=required_model_tasks_for_config(config), require_lineage=True)
    if not report["ready"]:
        raise ReadinessGateError(report)
    entries = {entry["task"]: entry for entry in json.loads(lock_path.read_text(encoding="utf-8"))["models"]}
    loaded = {}
    for task in ("component", "ocr", "wire", "junction"):
        if task not in required_model_tasks_for_config(config):
            continue
        entry = dict(entries[task])
        entry["inference"] = {**entry.get("inference", {}), "device": device}
        module, name = entry["adapter"].rsplit(":", 1)
        provider = getattr(importlib.import_module(module), name)(_resolve_model_path(lock_path, entry["path"]), entry)
        if task == "ocr":
            provider = _apply_ocr_inference_overrides(provider, config.ocr_inference_overrides)
        if task == "wire":
            provider = _apply_wire_inference_overrides(provider, config.wire_inference_overrides)
        _verify_provider_contract(provider, task)
        loaded[task] = provider
    descriptor = Path(package["descriptor_path"]).resolve()
    pin = PinPackageAdapter(descriptor, sha256=package["descriptor_sha256"], device=device)
    provider = PinPackageExperimentalProvider(component_provider=loaded["component"], ocr_provider=loaded["ocr"],
                                             pin_provider=pin, wire_provider=loaded["wire"],
                                             junction_provider=loaded.get("junction"), config=config)
    provider.experiment_metadata = {"production_ready": False, "training_lineage_verified": False,
        "baseline_models_lock_sha256": stable_file_hash(lock_path),
        "replaced_baseline_pin_sha256": entries["pin"]["sha256"],
        "descriptor_sha256": pin.descriptor_sha256, "weights": pin.descriptor["weights"],
        "runtime_device": str(pin.device), "changed_tasks": ["pin"]}
    return provider
