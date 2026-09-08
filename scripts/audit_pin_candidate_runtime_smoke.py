from __future__ import annotations

"""Run a CPU-only production Pin-adapter smoke on one frozen Dataset300 val ROI.

This audit deliberately does not edit ``models.lock.json`` or bless the
candidate.  It builds the candidate lock entry in memory, imports the exact
production adapter entrypoint, and executes two independently loaded ONNX
providers.  CUDA provider discovery is masked to model a CPU-only deployment
host while the candidate's hash-bound ``device=auto`` contract remains
unchanged.
"""

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Execute the runtime-bound Pin-v5 candidate twice through the "
            "production adapter on one deterministic Dataset300 val ROI."
        )
    )
    parser.add_argument("--workspace-root", type=Path, default=root)
    parser.add_argument(
        "--candidate-bundle",
        type=Path,
        default=Path(
            "work/candidates-runtime-bound/pin-v5-boundary/bundle.manifest.json"
        ),
    )
    parser.add_argument(
        "--source-bundle",
        type=Path,
        default=Path("work/train-pin-v5-boundary/bundle.manifest.json"),
    )
    parser.add_argument(
        "--pin-index",
        type=Path,
        default=Path(
            "artifacts/training/materialized/dataset300-v2/component-pin/"
            "pin_index.jsonl"
        ),
    )
    parser.add_argument(
        "--materialization-summary",
        type=Path,
        default=Path(
            "artifacts/training/materialized/dataset300-v2/component-pin/summary.json"
        ),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/splits/dataset300_split.jsonl"),
    )
    parser.add_argument(
        "--source-root", type=Path, default=Path("Dataset300/Dataset300")
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=Path("configs/pipeline.component512-experiment.yaml"),
    )
    parser.add_argument(
        "--models-lock", type=Path, default=Path("models/models.lock.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/pin_v5_candidate_runtime_provider_smoke.json"),
    )
    return parser


def _resolve_within(
    root: Path,
    raw: Path,
    label: str,
    *,
    kind: str = "file",
    must_exist: bool = True,
) -> Path:
    path = raw.expanduser()
    resolved = (path if path.is_absolute() else root / path).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside workspace: {resolved}") from exc
    if must_exist:
        valid = resolved.is_dir() if kind == "directory" else resolved.is_file()
        if not valid or (kind == "file" and resolved.stat().st_size <= 0):
            raise FileNotFoundError(f"{label} is missing or empty: {resolved}")
    return resolved


def _relative(path: Path, root: Path) -> str:
    return path.resolve(strict=False).relative_to(root).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _loads(raw: str, label: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = _loads(path.read_text(encoding="utf-8"), f"{label} at {path}")
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _read_jsonl_record(
    path: Path, *, predicate: Any, label: str
) -> tuple[dict[str, Any], int, str]:
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                raise ValueError(f"{label} contains a blank line at {line_number}")
            try:
                decoded = raw_line.decode("utf-8")
            except UnicodeError as exc:
                raise ValueError(
                    f"{label} contains invalid UTF-8 at line {line_number}"
                ) from exc
            value = _loads(decoded, f"{label} line {line_number}")
            if not isinstance(value, dict):
                raise ValueError(f"{label} line {line_number} is not an object")
            if predicate(value):
                return value, line_number, hashlib.sha256(raw_line).hexdigest()
    raise ValueError(f"{label} has no row satisfying the deterministic fixture policy")


def _select_fixture(
    predictions_path: Path, pin_index_path: Path
) -> dict[str, Any]:
    prediction, prediction_line, prediction_line_sha = _read_jsonl_record(
        predictions_path,
        predicate=lambda row: isinstance(row.get("detections_heatmap"), list)
        and bool(row["detections_heatmap"]),
        label="candidate evaluation predictions",
    )
    record_id = prediction.get("record_id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("selected prediction row lacks record_id")
    record, index_line, index_line_sha = _read_jsonl_record(
        pin_index_path,
        predicate=lambda row: row.get("record_id") == record_id,
        label="materialized Pin index",
    )
    if record.get("split") != "val":
        raise ValueError("selected prediction does not resolve to a frozen val record")
    if record.get("sample_id") != prediction.get("sample_id"):
        raise ValueError("selected prediction sample_id differs from Pin index")
    if not isinstance(record.get("pins"), list) or not record["pins"]:
        raise ValueError("selected validation fixture has no ground-truth pins")
    roi_views = [
        item
        for item in record.get("views", [])
        if isinstance(item, dict) and item.get("view_kind") == "roi"
    ]
    if len(roi_views) != 1:
        raise ValueError("selected validation fixture must have exactly one ROI view")
    if roi_views[0].get("view_id") != prediction.get("view_id"):
        raise ValueError("selected prediction view_id differs from Pin index ROI")
    return {
        "prediction": prediction,
        "prediction_line_number": prediction_line,
        "prediction_line_sha256": prediction_line_sha,
        "record": record,
        "index_line_number": index_line,
        "index_line_sha256": index_line_sha,
        "roi": roi_views[0],
    }


def _bbox_values(value: Any, label: str) -> tuple[float, float, float, float]:
    if isinstance(value, Mapping):
        raw = [value.get(name) for name in ("x1", "y1", "x2", "y2")]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = list(value)
    else:
        raise ValueError(f"{label} must be a bbox")
    if len(raw) != 4 or any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in raw
    ):
        raise ValueError(f"{label} must contain four finite numbers")
    values = tuple(float(item) for item in raw)
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError(f"{label} must have positive area")
    return values


def _expected_letterbox(
    source_size: tuple[int, int], input_size: tuple[int, int]
) -> dict[str, Any]:
    source_width, source_height = source_size
    input_width, input_height = input_size
    if min(source_width, source_height, input_width, input_height) <= 0:
        raise ValueError("letterbox dimensions must be positive")
    scale = min(input_width / source_width, input_height / source_height)
    resized_width = max(1, min(input_width, int(round(source_width * scale))))
    resized_height = max(1, min(input_height, int(round(source_height * scale))))
    pad_left = (input_width - resized_width) // 2
    pad_top = (input_height - resized_height) // 2
    return {
        "input_size_width_height": [input_width, input_height],
        "tensor_shape_nchw": [1, 3, input_height, input_width],
        "source_size_width_height": [source_width, source_height],
        "resized_size_width_height": [resized_width, resized_height],
        "padding_left_top_right_bottom": [
            pad_left,
            pad_top,
            input_width - resized_width - pad_left,
            input_height - resized_height - pad_top,
        ],
    }


def _array_fingerprint(tensor: Any) -> dict[str, Any]:
    import numpy as np

    array = tensor.detach().to("cpu").contiguous().numpy()
    raw = array.tobytes(order="C")
    finite = bool(np.isfinite(array).all())
    return {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "finite": finite,
        "minimum": float(array.min()) if array.size else None,
        "maximum": float(array.max()) if array.size else None,
    }


class _CapturingBackend:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    def run(self, image: Any) -> dict[str, Any]:
        raw_outputs = self.delegate.run(image)
        if not isinstance(raw_outputs, Mapping):
            raise ValueError("Pin backend output must be a mapping")
        call = {
            "input": _array_fingerprint(image),
            "outputs": {
                str(name): _array_fingerprint(value)
                for name, value in sorted(raw_outputs.items())
            },
        }
        call["outputs_sha256"] = _canonical_sha256(call["outputs"])
        self.calls.append(call)
        return dict(raw_outputs)


def _pin_coordinate_contract(
    pins: Sequence[Any], component_view: Any, original: Any, expected: Mapping[str, Any]
) -> dict[str, Any]:
    from pcbparse.ir.models import Pin

    view = component_view.full_roi
    crop = [float(value) for value in view.metadata["crop_box"]]
    serialized: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    all_local = True
    all_original = True
    all_crop = True
    all_roundtrip = True
    all_base = True
    all_metadata = True
    all_schema_roundtrip = True
    maximum_roundtrip_error = 0.0
    for pin in pins:
        payload = pin.model_dump(mode="json")
        # Strict finite JSON serialization is part of the provider contract.
        _canonical_bytes(payload)
        reconstructed = Pin.model_validate(payload)
        schema_equal = reconstructed.model_dump(mode="json") == payload
        all_schema_roundtrip = all_schema_roundtrip and schema_equal
        serialized.append(payload)

        point = pin.point
        local_ok = 0.0 <= point.x < view.width and 0.0 <= point.y < view.height
        restored = view.to_original(point)
        original_ok = (
            0.0 <= restored.x < original.width and 0.0 <= restored.y < original.height
        )
        crop_ok = (
            crop[0] <= restored.x < crop[2] and crop[1] <= restored.y < crop[3]
        )
        returned = view.to_view(restored)
        roundtrip_error = math.hypot(returned.x - point.x, returned.y - point.y)
        roundtrip_ok = roundtrip_error <= 1e-9
        maximum_roundtrip_error = max(maximum_roundtrip_error, roundtrip_error)

        if pin.base is None:
            base_ok = pin.orientation == "UNKNOWN"
            base_distance = None
        else:
            base_distance = math.hypot(
                pin.point.x - pin.base.x, pin.point.y - pin.base.y
            )
            base_ok = (
                math.isfinite(pin.base.x)
                and math.isfinite(pin.base.y)
                and abs(base_distance - 8.0) <= 1e-6
            )
        evidence_ok = bool(pin.evidence)
        for evidence in pin.evidence:
            metadata = evidence.metadata
            evidence_ok = evidence_ok and (
                metadata.get("inference_size")
                == expected["input_size_width_height"]
                and metadata.get("letterbox_resized_size")
                == expected["resized_size_width_height"]
                and metadata.get("letterbox_padding")
                == expected["padding_left_top_right_bottom"]
                and metadata.get("semantic_fields_generated") is False
                and evidence.view_id == view.view_id
            )
        all_local = all_local and local_ok
        all_original = all_original and original_ok
        all_crop = all_crop and crop_ok
        all_roundtrip = all_roundtrip and roundtrip_ok
        all_base = all_base and base_ok
        all_metadata = all_metadata and evidence_ok
        rows.append(
            {
                "internal_id": pin.internal_id,
                "local_point": [point.x, point.y],
                "original_point": [restored.x, restored.y],
                "local_point_in_bounds": local_ok,
                "original_point_in_canvas": original_ok,
                "original_point_in_roi_crop": crop_ok,
                "roundtrip_error_px": roundtrip_error,
                "base_length_px": base_distance,
                "base_contract": base_ok,
                "evidence_geometry_contract": evidence_ok,
                "strict_schema_roundtrip": schema_equal,
            }
        )
    encoded = _canonical_bytes(serialized)
    return {
        "pin_count": len(serialized),
        "serialized_pins": serialized,
        "serialized_bytes": len(encoded),
        "serialized_sha256": hashlib.sha256(encoded).hexdigest(),
        "coordinates": rows,
        "all_local_points_in_bounds": all_local,
        "all_original_points_in_canvas": all_original,
        "all_original_points_in_roi_crop": all_crop,
        "all_coordinate_roundtrips_within_1e-9_px": all_roundtrip,
        "maximum_coordinate_roundtrip_error_px": maximum_roundtrip_error,
        "all_base_vectors_finite_and_bound_length": all_base,
        "all_evidence_geometry_matches_input_contract": all_metadata,
        "all_strict_schema_roundtrips": all_schema_roundtrip,
    }


def _snapshot(paths: Mapping[str, Path], workspace: Path) -> dict[str, Any]:
    return {
        name: {
            "path": _relative(path, workspace),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for name, path in sorted(paths.items())
    }


def _run_once(
    *,
    factory: Any,
    entry: Mapping[str, Any],
    model_path: Path,
    component_view: Any,
    original: Any,
    expected_geometry: Mapping[str, Any],
    production_contract_verifier: Any,
    ort: Any,
) -> dict[str, Any]:
    original_available = ort.get_available_providers
    try:
        # The host has a CUDA-enabled ORT wheel, but this audit is explicitly
        # CPU-only.  Masking capability discovery is equivalent to executing
        # the unmodified device=auto entry on a CPU-only deployment host.
        ort.get_available_providers = lambda: ["CPUExecutionProvider"]
        provider = factory(model_path, entry)
    finally:
        ort.get_available_providers = original_available

    production_contract_verifier(provider, "pin")
    if provider.__class__.__name__ != "TorchPinProvider":
        raise ValueError("candidate adapter did not construct TorchPinProvider")
    delegate = provider._backend
    session = getattr(delegate, "session", None)
    if session is None or not callable(getattr(session, "get_providers", None)):
        raise ValueError("runtime-bound Pin candidate did not construct ONNX backend")
    session_providers = list(session.get_providers())
    capture = _CapturingBackend(delegate)
    provider._backend = capture
    context = SimpleNamespace(model_versions={})
    pins = provider.predict([component_view], context)
    if len(capture.calls) != 1:
        raise ValueError("one ComponentView must produce exactly one backend call")
    contract = _pin_coordinate_contract(
        pins, component_view, original, expected_geometry
    )
    return {
        "provider_class": (
            f"{provider.__class__.__module__}.{provider.__class__.__qualname__}"
        ),
        "provider_version": provider.version,
        "adapter_config": {
            "input_size_width_height": list(provider.config.input_size),
            "device": provider.config.device,
            "output_stride": provider.config.output_stride,
            "score_threshold": provider.config.score_threshold,
            "nms_kernel": provider.config.nms_kernel,
            "nms_radius_heatmap": provider.config.nms_radius_heatmap,
            "max_detections_per_component": (
                provider.config.max_detections_per_component
            ),
            "base_length_pixels": provider.config.base_length_pixels,
            "box_margin_px": provider.config.box_margin_px,
            "box_prior_mode": provider.config.box_prior_mode,
        },
        "backend_class": (
            f"{delegate.__class__.__module__}.{delegate.__class__.__qualname__}"
        ),
        "onnx_session_execution_providers": session_providers,
        "backend_call": capture.calls[0],
        "context_model_versions": dict(sorted(context.model_versions.items())),
        "decoded": contract,
    }


def _atomic_write(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_audit(
    *,
    workspace: Path,
    candidate_bundle_path: Path,
    source_bundle_path: Path,
    pin_index_path: Path,
    materialization_summary_path: Path,
    split_manifest_path: Path,
    source_root: Path,
    pipeline_config_path: Path,
    models_lock_path: Path,
) -> dict[str, Any]:
    # This must happen before importing torch/onnxruntime or the adapter.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("PYTHONHASHSEED", "0")

    from pcbparse.io import load_image, make_component_view
    from pcbparse.ir.models import BBox, Component
    from pcbparse.perception.production_provider import (
        _resolve_model_path,
        _verify_provider_contract,
        composite_provider_config_from_pipeline,
    )
    from pcbparse.training.artifacts import build_candidate_lock_entry

    import onnxruntime as ort
    import torch

    candidate_root = candidate_bundle_path.parent
    source_root_bundle = source_bundle_path.parent
    candidate_manifest = _read_object(candidate_bundle_path, "candidate bundle")
    source_manifest = _read_object(source_bundle_path, "source candidate bundle")
    predictions_path = _resolve_within(
        candidate_root,
        Path(str(candidate_manifest.get("evaluation_predictions_path"))),
        "candidate predictions",
    )
    artifact_path = _resolve_within(
        candidate_root,
        Path(str(candidate_manifest.get("artifact_path"))),
        "candidate model artifact",
    )
    candidate_card_path = _resolve_within(
        candidate_root,
        Path(str(candidate_manifest.get("model_card_path"))),
        "candidate model card",
    )
    source_artifact_path = _resolve_within(
        source_root_bundle,
        Path(str(source_manifest.get("artifact_path"))),
        "source model artifact",
    )
    source_predictions_path = source_root_bundle / "predictions.jsonl"
    source_card_path = source_root_bundle / "model_card.json"
    rebind_audit_path = candidate_root / "rebind.audit.json"
    protected_paths = {
        "models_lock": models_lock_path,
        "source_bundle_manifest": source_bundle_path,
        "source_model_card": source_card_path,
        "source_model_artifact": source_artifact_path,
        "source_predictions": source_predictions_path,
        "candidate_bundle_manifest": candidate_bundle_path,
        "candidate_model_card": candidate_card_path,
        "candidate_model_artifact": artifact_path,
        "candidate_predictions": predictions_path,
        "candidate_rebind_audit": rebind_audit_path,
    }
    before = _snapshot(protected_paths, workspace)

    # Reproduce the candidate's existing rebind evidence: its in-memory lock
    # entry is bundle-directory-relative (``model.onnx``), then resolved by the
    # same helper used by the production multi-model loader.  No lock file is
    # created and the primary lock is never edited.
    entry = build_candidate_lock_entry(
        candidate_bundle_path, lock_directory=candidate_root
    )
    entry_before_sha = _canonical_sha256(entry)
    rebind_audit = _read_object(rebind_audit_path, "candidate rebind audit")
    rebind_destination = rebind_audit.get("destination")
    if not isinstance(rebind_destination, Mapping):
        raise ValueError("candidate rebind audit lacks destination evidence")
    if (
        rebind_destination.get("candidate_lock_entry_sha256") != entry_before_sha
        or rebind_destination.get("bundle_manifest_sha256")
        != _sha256(candidate_bundle_path)
        or rebind_destination.get("artifact_sha256") != _sha256(artifact_path)
        or rebind_destination.get("evaluation_predictions_sha256")
        != _sha256(predictions_path)
    ):
        raise ValueError("candidate lock entry differs from rebind audit hash chain")
    if entry.get("status") != "candidate" or entry.get("blessed") is not False:
        raise ValueError("runtime smoke requires an unblessed candidate lock entry")
    if entry.get("inference") != candidate_manifest.get("inference"):
        raise ValueError("built lock entry inference differs from candidate bundle")
    candidate_lock_anchor = candidate_root / "models.lock.json"
    resolved_entry_artifact = _resolve_model_path(
        candidate_lock_anchor, str(entry.get("path"))
    )
    if resolved_entry_artifact != artifact_path:
        raise ValueError("built lock entry does not resolve to candidate artifact")

    materialization = _read_object(
        materialization_summary_path, "component-pin materialization summary"
    )
    expected_index_sha = (
        materialization.get("outputs", {})
        .get("pin_index.jsonl", {})
        .get("sha256")
    )
    actual_index_sha = _sha256(pin_index_path)
    if expected_index_sha != actual_index_sha:
        raise ValueError("Pin index differs from materialization summary")
    split_sha = _sha256(split_manifest_path)
    if (
        materialization.get("lineage", {})
        .get("inputs", {})
        .get("split_manifest", {})
        .get("sha256")
        != split_sha
        or candidate_manifest.get("split_manifest_sha256") != split_sha
    ):
        raise ValueError("split manifest lineage differs across candidate/materialization")

    fixture = _select_fixture(predictions_path, pin_index_path)
    record = fixture["record"]
    roi = fixture["roi"]
    if record.get("binding_sha256") != materialization.get("binding_sha256"):
        raise ValueError("selected fixture binding differs from materialization")
    source_image = _resolve_within(
        source_root,
        Path(str(record.get("source_image"))),
        "Dataset300 source image",
    )
    if _sha256(source_image) != record.get("source_image_sha256"):
        raise ValueError("selected Dataset300 source image hash changed")

    pipeline_raw = _read_object(pipeline_config_path, "pipeline config")
    perception_config = composite_provider_config_from_pipeline(pipeline_raw)
    summary_margin = materialization.get("config", {}).get("pin_roi_margin_px")
    if perception_config.resolved_pin_component_roi_padding_px != summary_margin:
        raise ValueError("production Pin ROI padding differs from materialization")
    component_box = _bbox_values(record.get("component_bbox"), "component_bbox")
    component = Component(
        internal_id=str(record.get("component_id")),
        key=str(record.get("component_key")),
        category=str(record.get("component_category")),
        bbox=BBox(
            x1=component_box[0],
            y1=component_box[1],
            x2=component_box[2],
            y2=component_box[3],
        ),
    )
    original = load_image(source_image, view_id="original")
    padding = max(
        max(component.bbox.width, component.bbox.height)
        * perception_config.component_roi_padding_factor,
        perception_config.resolved_pin_component_roi_padding_px,
    )
    component_view = make_component_view(
        original,
        component,
        padding=padding,
        view_id_prefix="component-pin",
    )
    production_crop = [int(value) for value in component_view.full_roi.metadata["crop_box"]]
    indexed_crop_float = _bbox_values(roi.get("crop_bbox"), "indexed ROI crop_bbox")
    indexed_crop = [int(value) for value in indexed_crop_float]
    if production_crop != indexed_crop:
        raise ValueError("production ComponentView crop differs from frozen val ROI")
    indexed_source_size = roi.get("source_size")
    if indexed_source_size != [component_view.full_roi.width, component_view.full_roi.height]:
        raise ValueError("production ComponentView size differs from frozen val ROI")
    if component_view.full_roi.width == component_view.full_roi.height:
        raise ValueError("fixture must be non-square to expose width/height order")

    inference_size = tuple(int(value) for value in entry["inference"]["input_size"])
    expected_geometry = _expected_letterbox(component_view.full_roi.size, inference_size)

    module_name, attribute_name = str(entry["adapter"]).rsplit(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute_name)
    host_available_providers = list(ort.get_available_providers())
    runs = [
        _run_once(
            factory=factory,
            entry=entry,
            model_path=resolved_entry_artifact,
            component_view=component_view,
            original=original,
            expected_geometry=expected_geometry,
            production_contract_verifier=_verify_provider_contract,
            ort=ort,
        )
        for _ in range(2)
    ]
    entry_after_sha = _canonical_sha256(entry)
    after = _snapshot(protected_paths, workspace)

    decoded_hashes = [item["decoded"]["serialized_sha256"] for item in runs]
    input_hashes = [item["backend_call"]["input"]["sha256"] for item in runs]
    output_hashes = [item["backend_call"]["outputs_sha256"] for item in runs]
    coordinate_checks = [
        item["decoded"][name]
        for item in runs
        for name in (
            "all_local_points_in_bounds",
            "all_original_points_in_canvas",
            "all_original_points_in_roi_crop",
            "all_coordinate_roundtrips_within_1e-9_px",
            "all_base_vectors_finite_and_bound_length",
            "all_evidence_geometry_matches_input_contract",
            "all_strict_schema_roundtrips",
        )
    ]
    checks = {
        "candidate_bundle_reverified_and_lock_entry_buildable": True,
        "candidate_rebind_audit_lock_entry_hash_chain_matched": True,
        "candidate_entry_unmodified_during_factory_load": (
            entry_before_sha == entry_after_sha
        ),
        "production_adapter_contract_verified_both_runs": True,
        "actual_torch_pin_provider_loaded_both_runs": all(
            item["provider_class"].endswith(".TorchPinProvider") for item in runs
        ),
        "actual_onnx_backend_loaded_both_runs": all(
            item["backend_class"].endswith("._ONNXBackend") for item in runs
        ),
        "cpu_execution_provider_only_both_runs": all(
            item["onnx_session_execution_providers"] == ["CPUExecutionProvider"]
            for item in runs
        ),
        "candidate_device_auto_preserved_both_runs": all(
            item["adapter_config"]["device"] == "auto" for item in runs
        ),
        "non_square_frozen_val_roi_matches_production_crop": (
            production_crop == indexed_crop
            and component_view.full_roi.width != component_view.full_roi.height
        ),
        "input_width_height_order_matches_nchw_tensor": all(
            item["adapter_config"]["input_size_width_height"]
            == expected_geometry["input_size_width_height"]
            and item["backend_call"]["input"]["shape"]
            == expected_geometry["tensor_shape_nchw"]
            for item in runs
        ),
        "runtime_emitted_nonempty_results_both_runs": all(
            item["decoded"]["pin_count"] > 0 for item in runs
        ),
        "coordinate_and_strict_serialization_contracts_both_runs": all(
            coordinate_checks
        ),
        "input_tensor_bytes_identical_across_runs": len(set(input_hashes)) == 1,
        "raw_model_output_bytes_identical_across_runs": len(set(output_hashes)) == 1,
        "decoded_serialized_results_identical_across_runs": (
            len(set(decoded_hashes)) == 1
        ),
        "source_target_bundles_and_models_lock_unchanged": before == after,
    }
    overall = "pass" if all(checks.values()) else "fail"

    script_path = Path(__file__).resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pin_candidate_production_runtime_provider_smoke",
        "status": overall,
        "scope": {
            "candidate_direct_runtime_inference": True,
            "current_models_lock_inference": False,
            "models_lock_blessing_attempted": False,
            "model_or_predictions_recomputed": False,
            "threshold_or_nms_tuned": False,
            "boundary": (
                "This is real CPU inference through the runtime-bound candidate's "
                "production TorchPinProvider factory. The candidate remains unblessed "
                "and was not inserted into models.lock, so this is not current-lock "
                "five-model production inference."
            ),
        },
        "candidate": {
            "bundle_manifest": {
                "path": _relative(candidate_bundle_path, workspace),
                "sha256": _sha256(candidate_bundle_path),
            },
            "source_bundle_manifest": {
                "path": _relative(source_bundle_path, workspace),
                "sha256": _sha256(source_bundle_path),
            },
            "artifact": {
                "path": _relative(artifact_path, workspace),
                "bytes": artifact_path.stat().st_size,
                "sha256": _sha256(artifact_path),
            },
            "evaluation_predictions": {
                "path": _relative(predictions_path, workspace),
                "sha256": _sha256(predictions_path),
            },
            "adapter_entrypoint": entry["adapter"],
            "runtime_inference": entry["inference"],
            "candidate_lock_entry_sha256": entry_before_sha,
            "rebind_audit": {
                "path": _relative(rebind_audit_path, workspace),
                "sha256": _sha256(rebind_audit_path),
            },
            "candidate_lock_entry_resolution": {
                "synthetic_lock_anchor_not_created": _relative(
                    candidate_lock_anchor, workspace
                ),
                "resolved_artifact_path": _relative(
                    resolved_entry_artifact, workspace
                ),
            },
            "candidate_lock_entry": entry,
        },
        "fixture": {
            "selection_policy": (
                "first file-order candidate evaluation prediction row with at "
                "least one detection; resolve its exact record_id in the hash-bound "
                "Pin index and require split=val plus non-empty GT"
            ),
            "record_id": record["record_id"],
            "record_number": record.get("record_number"),
            "sample_id": record["sample_id"],
            "split": record["split"],
            "component_id": record["component_id"],
            "component_category": record["component_category"],
            "component_bbox_original": list(component_box),
            "ground_truth_pin_count": len(record["pins"]),
            "formal_prediction_detection_count": len(
                fixture["prediction"]["detections_heatmap"]
            ),
            "prediction_line_number": fixture["prediction_line_number"],
            "prediction_line_sha256": fixture["prediction_line_sha256"],
            "pin_index_line_number": fixture["index_line_number"],
            "pin_index_line_sha256": fixture["index_line_sha256"],
            "pin_index": {
                "path": _relative(pin_index_path, workspace),
                "sha256": actual_index_sha,
            },
            "materialization_summary": {
                "path": _relative(materialization_summary_path, workspace),
                "sha256": _sha256(materialization_summary_path),
                "binding_sha256": materialization["binding_sha256"],
            },
            "split_manifest": {
                "path": _relative(split_manifest_path, workspace),
                "sha256": split_sha,
            },
            "source_image": {
                "path": _relative(source_image, workspace),
                "sha256": _sha256(source_image),
                "size_width_height": [original.width, original.height],
            },
            "pipeline_config": {
                "path": _relative(pipeline_config_path, workspace),
                "sha256": _sha256(pipeline_config_path),
                "resolved_pin_component_roi_padding_px": (
                    perception_config.resolved_pin_component_roi_padding_px
                ),
                "resolved_component_roi_padding_factor": (
                    perception_config.component_roi_padding_factor
                ),
                "effective_padding_px": padding,
            },
            "indexed_roi_crop_left_top_right_bottom": indexed_crop,
            "production_roi_crop_left_top_right_bottom": production_crop,
            "production_roi_size_width_height": [
                component_view.full_roi.width,
                component_view.full_roi.height,
            ],
            "expected_adapter_letterbox": expected_geometry,
        },
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "host_onnxruntime_available_providers_before_cpu_mask": (
                host_available_providers
            ),
            "cpu_only_mask": (
                "onnxruntime.get_available_providers was scoped to "
                "[CPUExecutionProvider] only while each unmodified device=auto "
                "provider was constructed; the created sessions independently "
                "report CPUExecutionProvider only"
            ),
            "runs": runs,
        },
        "protected_files_before": before,
        "protected_files_after": after,
        "checks": checks,
        "release_subgates": {
            "candidate_pin_runtime_provider_smoke": overall,
            "candidate_pin_cpu_repeatability": (
                "pass"
                if checks["raw_model_output_bytes_identical_across_runs"]
                and checks["decoded_serialized_results_identical_across_runs"]
                else "fail"
            ),
            "current_models_lock_pin_inference": (
                "open_candidate_unblessed_not_in_primary_lock"
            ),
            "five_model_current_lock_production_inference": "open_not_in_scope",
            "official_dataset300_pin_f1": "open_official_evaluator_required",
        },
        "audit_implementation": {
            "path": _relative(script_path, workspace),
            "sha256": _sha256(script_path),
        },
    }
    report["audit_payload_sha256"] = _canonical_sha256(report)
    return report


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    workspace = arguments.workspace_root.expanduser().resolve(strict=True)
    if str(workspace / "src") not in sys.path:
        sys.path.insert(0, str(workspace / "src"))
    resolved = {
        name: _resolve_within(workspace, getattr(arguments, name), name)
        for name in (
            "candidate_bundle",
            "source_bundle",
            "pin_index",
            "materialization_summary",
            "split_manifest",
            "pipeline_config",
            "models_lock",
        )
    }
    source_root = _resolve_within(
        workspace, arguments.source_root, "source_root", kind="directory"
    )
    output = _resolve_within(
        workspace, arguments.output, "audit output", must_exist=False
    )
    report = build_audit(
        workspace=workspace,
        candidate_bundle_path=resolved["candidate_bundle"],
        source_bundle_path=resolved["source_bundle"],
        pin_index_path=resolved["pin_index"],
        materialization_summary_path=resolved["materialization_summary"],
        split_manifest_path=resolved["split_manifest"],
        source_root=source_root,
        pipeline_config_path=resolved["pipeline_config"],
        models_lock_path=resolved["models_lock"],
    )
    _atomic_write(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "file_sha256": _sha256(output),
                "audit_payload_sha256": report["audit_payload_sha256"],
                "status": report["status"],
                "record_id": report["fixture"]["record_id"],
                "pin_counts": [
                    item["decoded"]["pin_count"]
                    for item in report["runtime"]["runs"]
                ],
                "decoded_sha256s": [
                    item["decoded"]["serialized_sha256"]
                    for item in report["runtime"]["runs"]
                ],
                "models_lock_sha256": report["protected_files_after"][
                    "models_lock"
                ]["sha256"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
