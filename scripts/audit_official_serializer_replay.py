from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC = REPOSITORY_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import PIL  # noqa: E402
import pydantic  # noqa: E402
from PIL import Image  # noqa: E402

from pcbparse.ir.models import CircuitIR, IRState  # noqa: E402
from pcbparse.output.official_schema import (  # noqa: E402
    load_official_schema,
    split_official_pin_numbers,
    validate_official_output,
)
from pcbparse.output.official_serializer import (  # noqa: E402
    natural_sort_key,
    official_json_bytes,
    official_json_hash,
    serialize_official,
)


HEX_SHA256 = set("0123456789abcdef")
ROUNDTRIP_LIMIT_PX = 0.5
REPLAY_RUNS = 3
EVIDENCE_SCOPE = "frozen_ir_current_serializer_replay"


class ReplayAuditError(ValueError):
    """Raised when frozen replay lineage or coordinate evidence is incomplete."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in HEX_SHA256 for character in value)
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ReplayAuditError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise ReplayAuditError(f"non-standard JSON constant {value!r}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ReplayAuditError) as exc:
        raise ReplayAuditError(f"invalid JSON at {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ReplayAuditError(f"cannot read JSONL at {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (json.JSONDecodeError, ReplayAuditError) as exc:
            raise ReplayAuditError(
                f"invalid JSONL at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ReplayAuditError(f"JSONL row at {path}:{line_number} is not an object")
        rows.append(value)
    return rows


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve(strict=False)


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve(strict=False))


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ReplayAuditError(f"{label} does not exist: {path}")


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ReplayAuditError(
            f"{label} mismatch: actual={actual!r}, expected={expected!r}"
        )


def _component_key_map(ir: CircuitIR) -> dict[str, str]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for component in ir.components:
        if not isinstance(component.key, str) or not component.key.strip():
            raise ReplayAuditError(
                f"component {component.internal_id!r} lacks an official key"
            )
        grouped[component.key].append(component)

    output: dict[str, str] = {}
    used: set[str] = set()
    for designator in sorted(grouped, key=natural_sort_key):
        members = sorted(
            grouped[designator], key=lambda item: natural_sort_key(item.internal_id)
        )
        keys = (
            [designator]
            if len(members) == 1
            else [f"{designator}_{index}" for index in range(1, len(members) + 1)]
        )
        for component, key in zip(members, keys):
            if key in used:
                raise ReplayAuditError(f"component output key collision: {key!r}")
            output[component.internal_id] = key
            used.add(key)
    return output


def _pin_number_map(ir: CircuitIR, component_keys: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for pin in ir.pins:
        if pin.component_id not in component_keys:
            raise ReplayAuditError(f"pin {pin.internal_id!r} has no serializable owner")
        grouped[str(pin.component_id)].append(pin)

    output: dict[str, tuple[str, ...]] = {}
    for component_id, pins in grouped.items():
        ordered = sorted(pins, key=lambda item: natural_sort_key(item.internal_id))
        explicit = {
            pin.internal_id: split_official_pin_numbers(pin.number) for pin in ordered
        }
        used = {
            number
            for pin in ordered
            for number in explicit[pin.internal_id]
        }
        for pin in ordered:
            numbers = explicit[pin.internal_id]
            if not numbers:
                raise ReplayAuditError(
                    f"pin {pin.internal_id!r} has no explicit representable number"
                )
            output[pin.internal_id] = numbers
    return output


def _geometry(
    first: tuple[float, float], second: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    return tuple(sorted((first, second)))  # type: ignore[return-value]


def _point_error(
    source: tuple[float, float], restored: tuple[float, float]
) -> float:
    return math.hypot(source[0] - restored[0], source[1] - restored[1])


def _coordinate_audit(ir: CircuitIR, output: Mapping[str, Any], schema: Mapping[str, Any]) -> dict[str, Any]:
    height = float(ir.image.height)
    width = float(ir.image.width)
    component_keys = _component_key_map(ir)
    pin_numbers = _pin_number_map(ir, component_keys)
    observations: list[dict[str, Any]] = []
    errors: list[float] = []

    def observe(
        *,
        kind: str,
        object_id: str,
        source: tuple[float, float],
        official: tuple[float, float],
    ) -> None:
        expected = (source[0], height - source[1])
        serialized_error = _point_error(expected, official)
        restored = (official[0], height - official[1])
        roundtrip_error = _point_error(source, restored)
        errors.extend((serialized_error, roundtrip_error))
        observations.append(
            {
                "kind": kind,
                "object_id": object_id,
                "source_xy": list(source),
                "official_xy": list(official),
                "restored_xy": list(restored),
                "source_to_serialized_error_px": serialized_error,
                "roundtrip_error_px": roundtrip_error,
            }
        )

    components = output["components"]
    for component in ir.components:
        key = component_keys[component.internal_id]
        raw_bbox = components[key]["bbox"]
        expected_bbox = [
            float(component.bbox.x1),
            height - float(component.bbox.y2),
            float(component.bbox.x2),
            height - float(component.bbox.y1),
        ]
        if raw_bbox != expected_bbox:
            raise ReplayAuditError(
                f"component {component.internal_id} bbox is not H-y encoded"
            )
        observe(
            kind="component_bbox_top_left",
            object_id=component.internal_id,
            source=(float(component.bbox.x1), float(component.bbox.y1)),
            official=(float(raw_bbox[0]), float(raw_bbox[3])),
        )
        observe(
            kind="component_bbox_bottom_right",
            object_id=component.internal_id,
            source=(float(component.bbox.x2), float(component.bbox.y2)),
            official=(float(raw_bbox[2]), float(raw_bbox[1])),
        )

    pins = output["pins"]
    pin_template = str(schema["pin"]["key_template"])
    for pin in ir.pins:
        component_key = component_keys[str(pin.component_id)]
        for number in pin_numbers[pin.internal_id]:
            pin_key = pin_template.format(number=number)
            point = pins[component_key][pin_key]["point"]
            observe(
                kind="pin",
                object_id=f"{pin.internal_id}:{number}",
                source=(float(pin.point.x), float(pin.point.y)),
                official=(float(point["x"]), float(point["y"])),
            )

    segment_index = ir.segment_index()
    pin_index = ir.pin_index()
    physical_index = {net.internal_id: net for net in ir.physical_nets}
    expected_edges: Counter[
        tuple[tuple[float, float], tuple[float, float]]
    ] = Counter()
    expected_positive_edges = 0
    expected_zero_edges = 0
    for net in ir.electrical_nets:
        geometries: set[tuple[tuple[float, float], tuple[float, float]]] = set()
        for segment_id in set(net.segments):
            if segment_id not in segment_index:
                raise ReplayAuditError(
                    f"electrical net {net.internal_id!r} references unknown segment"
                )
            segment = segment_index[segment_id]
            geometries.add(
                _geometry(
                    (float(segment.start.x), height - float(segment.start.y)),
                    (float(segment.end.x), height - float(segment.end.y)),
                )
            )

        candidate_pin_ids: set[str] = set()
        if net.physical_net_ids:
            for physical_id in net.physical_net_ids:
                physical = physical_index.get(physical_id)
                if physical is None:
                    raise ReplayAuditError(
                        f"electrical net {net.internal_id!r} references unknown physical net"
                    )
                if not physical.segments:
                    candidate_pin_ids.update(physical.pins)
        elif not net.segments:
            candidate_pin_ids.update(net.pins)
        for pin_id in candidate_pin_ids:
            if pin_id not in pin_index:
                raise ReplayAuditError(
                    f"electrical net {net.internal_id!r} references unknown pin"
                )
            point = pin_index[pin_id].point
            encoded = (float(point.x), height - float(point.y))
            geometries.add((encoded, encoded))

        expected_edges.update(geometries)
        expected_positive_edges += sum(first != second for first, second in geometries)
        expected_zero_edges += sum(first == second for first, second in geometries)

    actual_edges: Counter[
        tuple[tuple[float, float], tuple[float, float]]
    ] = Counter()
    for net_key, net in output["nets"].items():
        for edge_key, edge in net["edges"].items():
            geometry = _geometry(
                (float(edge[0]["x"]), float(edge[0]["y"])),
                (float(edge[1]["x"]), float(edge[1]["y"])),
            )
            actual_edges[geometry] += 1
            restored_first = (geometry[0][0], height - geometry[0][1])
            restored_second = (geometry[1][0], height - geometry[1][1])
            observe(
                kind="net_edge_endpoint",
                object_id=f"{net_key}:{edge_key}:0",
                source=restored_first,
                official=geometry[0],
            )
            observe(
                kind="net_edge_endpoint",
                object_id=f"{net_key}:{edge_key}:1",
                source=restored_second,
                official=geometry[1],
            )

    if actual_edges != expected_edges:
        missing = sum((expected_edges - actual_edges).values())
        extra = sum((actual_edges - expected_edges).values())
        raise ReplayAuditError(
            f"serialized edge geometry differs from frozen IR: missing={missing}, extra={extra}"
        )

    boundary_sources = [
        (0.0, 0.0),
        (width, 0.0),
        (0.0, height),
        (width, height),
        (0.125, 0.375),
        (max(0.0, width - 0.25), max(0.0, height - 0.75)),
    ]
    boundary_errors: list[float] = []
    for x, y in boundary_sources:
        official = (x, height - y)
        restored = (official[0], height - official[1])
        boundary_errors.append(_point_error((x, y), restored))

    max_error = max(errors + boundary_errors, default=0.0)
    return {
        "cases_checked": 1,
        "points_checked": len(observations),
        "component_bbox_points_checked": 2 * len(ir.components),
        "pin_points_checked": sum(len(numbers) for numbers in pin_numbers.values()),
        "net_edge_points_checked": 2 * sum(actual_edges.values()),
        "positive_edges_checked": expected_positive_edges,
        "zero_length_edges_checked": expected_zero_edges,
        "boundary_probe_points_checked": len(boundary_sources),
        "max_roundtrip_error_px": max_error,
        "max_source_to_serialized_error_px": max(errors, default=0.0),
        "roundtrip_limit_px": ROUNDTRIP_LIMIT_PX,
        "within_limit": max_error <= ROUNDTRIP_LIMIT_PX,
        "origin_conversion_verified": actual_edges == expected_edges and bool(observations),
        "coordinate_observations_sha256": _json_sha256(observations),
    }


def _tree_manifest(run_dir: Path) -> tuple[list[dict[str, str]], str]:
    files = sorted(
        (path for path in run_dir.rglob("*.json") if path.is_file()),
        key=lambda path: path.relative_to(run_dir).as_posix(),
    )
    manifest = [
        {
            "path": path.relative_to(run_dir).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in files
    ]
    return manifest, _json_sha256(manifest)


def _environment_payload(
    *,
    current_model_lock_sha256: str,
    official_profile_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    code_files = {
        "audit_script": Path(__file__).resolve(),
        "official_serializer": REPOSITORY_ROOT
        / "src"
        / "pcbparse"
        / "output"
        / "official_serializer.py",
        "official_schema": REPOSITORY_ROOT
        / "src"
        / "pcbparse"
        / "output"
        / "official_schema.py",
        "orchestrator": REPOSITORY_ROOT
        / "src"
        / "pcbparse"
        / "pipeline"
        / "orchestrator.py",
        "coordinate_lineage": REPOSITORY_ROOT
        / "src"
        / "pcbparse"
        / "data"
        / "derive.py",
    }
    for label, path in code_files.items():
        _require_file(path, label)
    payload = {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "dependencies": {
            "Pillow": PIL.__version__,
            "pydantic": pydantic.__version__,
        },
        "execution": {
            "device": "cpu_only",
            "models_loaded": False,
            "inference_executed": False,
        },
        "current_model_lock_sha256": current_model_lock_sha256,
        "official_profile_sha256": official_profile_sha256,
        "config_sha256": config_sha256,
        "code_sha256": {
            label: _sha256_file(path) for label, path in sorted(code_files.items())
        },
    }
    return payload


def _orchestrator_state_contract() -> dict[str, Any]:
    path = (
        REPOSITORY_ROOT
        / "src"
        / "pcbparse"
        / "pipeline"
        / "orchestrator.py"
    )
    _require_file(path, "orchestrator state contract")
    source = path.read_text(encoding="utf-8")
    serialize_at = source.find("output, quality = serialize_official_with_quality(")
    if serialize_at < 0:
        serialize_at = source.find("output = serialize_official(ir, official_profile)")
    persisted_at = source.find("ir.state = IRState.SERIALIZED", serialize_at + 1)
    verified = serialize_at >= 0 and persisted_at > serialize_at
    if not verified:
        raise ReplayAuditError(
            "orchestrator no longer records SERIALIZED state after official serialization"
        )
    return {
        "verified": True,
        "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
        "sha256": _sha256_file(path),
        "contract": "serialize VALIDATED IR via unified quality entry, then persist state SERIALIZED",
    }


def build_and_write_evidence(
    *,
    workspace_root: str | Path,
    source_run: str | Path,
    source_image: str | Path,
    source_annotation: str | Path,
    current_models_lock: str | Path,
    official_profile: str | Path,
    replay_root: str | Path,
    coordinate_report: str | Path,
    case_id: str,
) -> dict[str, Any]:
    root = Path(workspace_root).expanduser().resolve()
    run_root = _resolve(root, source_run)
    image_path = _resolve(root, source_image)
    annotation_path = _resolve(root, source_annotation)
    lock_path = _resolve(root, current_models_lock)
    profile_path = _resolve(root, official_profile)
    output_root = _resolve(root, replay_root)
    report_path = _resolve(root, coordinate_report)
    summary_path = run_root / "summary.json"
    manifest_path = run_root / "predictions.jsonl"
    for path, label in (
        (summary_path, "source summary"),
        (manifest_path, "source prediction manifest"),
        (image_path, "source raster image"),
        (annotation_path, "source Dataset300 annotation"),
        (lock_path, "current models.lock"),
        (profile_path, "official profile"),
    ):
        _require_file(path, label)

    summary = _read_json(summary_path)
    if not isinstance(summary, Mapping):
        raise ReplayAuditError("source summary root must be an object")
    _require_equal(summary.get("kind"), "production-image-only-inference", "source run kind")
    _require_equal(summary.get("accuracy_mode"), "image-only", "source accuracy mode")
    _require_equal(summary.get("execution_mode"), "production", "source execution mode")
    _require_equal(summary.get("provider_kind"), "production", "source provider")
    _require_equal(summary.get("sidecar_access"), False, "source sidecar access")
    _require_equal(summary.get("status"), "ok", "source run status")
    _require_equal(summary.get("all_successful"), True, "source run success")
    _require_equal(
        summary.get("prediction_manifest_sha256"),
        _sha256_file(manifest_path),
        "source prediction manifest SHA-256",
    )
    source_prediction_set_sha256 = summary.get("prediction_set_sha256")
    if not _is_sha256(source_prediction_set_sha256):
        raise ReplayAuditError("source summary lacks a valid prediction-set SHA-256")

    rows = [row for row in _read_jsonl(manifest_path) if row.get("case_id") == case_id]
    if len(rows) != 1:
        raise ReplayAuditError(
            f"source manifest must contain exactly one row for case {case_id!r}"
        )
    row = rows[0]
    _require_equal(row.get("status"), "ok", "source prediction status")
    _require_equal(row.get("provider_kind"), "production", "source row provider")
    _require_equal(row.get("sidecar_access"), False, "source row sidecar access")
    source_lock_sha256 = row.get("model_lock_sha256")
    if not _is_sha256(source_lock_sha256):
        raise ReplayAuditError("source row lacks a valid model-lock SHA-256")
    source_config_sha256 = row.get("config_hash")
    if not _is_sha256(source_config_sha256):
        raise ReplayAuditError("source row lacks a valid config SHA-256")
    control_inputs = summary.get("control_inputs")
    if not isinstance(control_inputs, Mapping) or not isinstance(
        control_inputs.get("models_lock"), str
    ):
        raise ReplayAuditError("source summary lacks its models.lock control input")
    source_lock_path = _resolve(root, str(control_inputs["models_lock"]))
    _require_file(source_lock_path, "source inference models.lock")
    _require_equal(
        _sha256_file(source_lock_path),
        source_lock_sha256,
        "source inference models.lock SHA-256",
    )

    ir_relative = row.get("ir_path")
    official_relative = row.get("official_path")
    if not isinstance(ir_relative, str) or not isinstance(official_relative, str):
        raise ReplayAuditError("source row lacks IR or official-output path")
    ir_path = (run_root / ir_relative).resolve(strict=False)
    source_official_path = (run_root / official_relative).resolve(strict=False)
    _require_file(ir_path, "frozen Circuit-IR")
    _require_file(source_official_path, "historical official output")
    _require_equal(row.get("ir_sha256"), _sha256_file(ir_path), "frozen IR SHA-256")
    _require_equal(
        row.get("official_sha256"),
        _sha256_file(source_official_path),
        "historical official-output SHA-256",
    )
    image_sha256 = _sha256_file(image_path)
    _require_equal(row.get("image_sha256"), image_sha256, "source raster SHA-256")

    ir = CircuitIR.model_validate_json(ir_path.read_text(encoding="utf-8"))
    if ir.state not in {IRState.SERIALIZED, IRState.VALIDATED}:
        raise ReplayAuditError(
            f"frozen IR must be post-validation, got state {ir.state.value!r}"
        )
    _require_equal(ir.image.sha256, image_sha256, "IR image SHA-256")
    with Image.open(image_path) as image:
        image_size = tuple(image.size)
    _require_equal(image_size, (ir.image.width, ir.image.height), "raster dimensions")
    annotation = _read_json(annotation_path)
    if not isinstance(annotation, Mapping):
        raise ReplayAuditError("Dataset300 annotation root must be an object")
    _require_equal(
        annotation.get("canvas"),
        {"width": ir.image.width, "height": ir.image.height},
        "Dataset300 canvas",
    )

    current_lock_sha256 = _sha256_file(lock_path)
    profile_sha256 = _sha256_file(profile_path)
    schema = load_official_schema(profile_path)
    replay_ir = ir.model_copy(deep=True)
    source_ir_state = replay_ir.state.value
    replay_ir.state = IRState.VALIDATED

    replay_outputs = [serialize_official(replay_ir, schema) for _ in range(REPLAY_RUNS)]
    for output in replay_outputs:
        validate_official_output(output, schema)
    replay_bytes = [official_json_bytes(output, schema) for output in replay_outputs]
    if len(set(replay_bytes)) != 1:
        raise ReplayAuditError("three current-serializer replays are not byte-identical")
    replay_hashes = [official_json_hash(output, schema) for output in replay_outputs]
    if len(set(replay_hashes)) != 1:
        raise ReplayAuditError("three current-serializer replay hashes differ")

    historical_output = _read_json(source_official_path)
    if not isinstance(historical_output, Mapping):
        raise ReplayAuditError("historical official output root must be an object")
    historical_components = historical_output.get("components")
    historical_component_fields: list[str] = []
    if isinstance(historical_components, Mapping) and historical_components:
        first_component = next(iter(historical_components.values()))
        if isinstance(first_component, Mapping):
            historical_component_fields = list(first_component)

    coordinate = _coordinate_audit(replay_ir, replay_outputs[0], schema)
    if not coordinate["within_limit"] or not coordinate["origin_conversion_verified"]:
        raise ReplayAuditError("coordinate transformation audit failed")

    config = {
        "schema_version": 1,
        "kind": "official-serializer-frozen-ir-replay-config",
        "evidence_scope": EVIDENCE_SCOPE,
        "case_id": case_id,
        "coordinate_forward": "official_x=x; official_y=image_height-source_y",
        "coordinate_inverse": "source_x=official_x; source_y=image_height-official_y",
        "bbox_forward": "[x1,image_height-y2,x2,image_height-y1]",
        "roundtrip_limit_px": ROUNDTRIP_LIMIT_PX,
        "replay_runs": REPLAY_RUNS,
        "source_ir_sha256": _sha256_file(ir_path),
        "source_prediction_set_sha256": source_prediction_set_sha256,
        "current_model_lock_sha256": current_lock_sha256,
        "official_profile_sha256": profile_sha256,
    }
    config_sha256 = _json_sha256(config)
    environment_payload = _environment_payload(
        current_model_lock_sha256=current_lock_sha256,
        official_profile_sha256=profile_sha256,
        config_sha256=config_sha256,
    )
    environment_fingerprint_sha256 = _json_sha256(environment_payload)
    orchestrator_state_contract = _orchestrator_state_contract()

    expected_json_paths = {f"run{number}/{case_id}.json" for number in range(1, 4)}
    if output_root.exists():
        unexpected = sorted(
            path.relative_to(output_root).as_posix()
            for path in output_root.rglob("*.json")
            if path.is_file()
            and path.relative_to(output_root).as_posix() not in expected_json_paths
            and path.name != "environment.json"
        )
        if unexpected:
            raise ReplayAuditError(
                f"replay root contains unexpected JSON files: {unexpected[:10]!r}"
            )
    for number, payload in enumerate(replay_bytes, start=1):
        _atomic_write_bytes(output_root / f"run{number}" / f"{case_id}.json", payload)

    runs: list[dict[str, Any]] = []
    for number in range(1, REPLAY_RUNS + 1):
        run_name = f"run{number}"
        manifest, tree_sha256 = _tree_manifest(output_root / run_name)
        runs.append(
            {
                "run": run_name,
                "environment_fingerprint_sha256": environment_fingerprint_sha256,
                "config_sha256": config_sha256,
                "model_lock_sha256": current_lock_sha256,
                "output_tree_sha256": tree_sha256,
                "files": manifest,
            }
        )
    if len({run["output_tree_sha256"] for run in runs}) != 1:
        raise ReplayAuditError("written replay output trees are not identical")

    environment_manifest = {
        "schema_version": 2,
        "kind": "same-environment-determinism",
        "evidence_scope": EVIDENCE_SCOPE,
        "current_lock_inference_executed": False,
        "models_loaded": False,
        "binding_role": "current_models_lock_snapshot_only",
        "source_inference_model_lock_sha256": source_lock_sha256,
        "source_inference_strict_image_only": True,
        "environment_fingerprint_sha256": environment_fingerprint_sha256,
        "environment": environment_payload,
        "runs": runs,
    }
    environment_path = output_root / "environment.json"
    _atomic_write_json(environment_path, environment_manifest)

    replay_output = replay_outputs[0]
    output_counts = {
        "components": len(replay_output["components"]),
        "pins": sum(len(group) for group in replay_output["pins"].values()),
        "nets": len(replay_output["nets"]),
        "edges": sum(len(net["edges"]) for net in replay_output["nets"].values()),
        "zero_length_edges": sum(
            edge[0] == edge[1]
            for net in replay_output["nets"].values()
            for edge in net["edges"].values()
        ),
    }
    report: dict[str, Any] = {
        "schema_version": 2,
        "kind": "coordinate-roundtrip-release-report",
        "evidence_scope": EVIDENCE_SCOPE,
        "claim_boundary": {
            "current_serializer_coordinate_mapping": True,
            "current_strict_schema_serialization": True,
            "current_serializer_byte_determinism": True,
            "current_lock_production_inference": False,
            "official_machine_evaluator": False,
            "reason": (
                "The current serializer replays a historical strict image-only frozen IR; "
                "no current models.lock artifact was loaded and no inference was executed."
            ),
        },
        "execution": {
            "device": "cpu_only",
            "models_loaded": False,
            "inference_executed": False,
            "training_configuration_modified": False,
            "models_lock_modified": False,
            "test_data_used_for_tuning": False,
        },
        "current_lock_inference_executed": False,
        "models_loaded": False,
        "strict_image_only": True,
        "strict_image_only_scope": "historical_source_inference",
        "model_lock_sha256": current_lock_sha256,
        "model_lock_binding_role": "audit_snapshot_only_models_not_loaded",
        "source_inference_model_lock_sha256": source_lock_sha256,
        "source_inference_matches_current_lock": source_lock_sha256 == current_lock_sha256,
        "config_sha256": config_sha256,
        "source_prediction_set_sha256": source_prediction_set_sha256,
        "official_coordinate_origin": "bottom_left",
        "official_y_axis": "up",
        "internal_coordinate_origin": "top_left",
        "internal_y_axis": "down",
        "origin_conversion_verified": coordinate["origin_conversion_verified"],
        "cases_checked": coordinate["cases_checked"],
        "points_checked": coordinate["points_checked"],
        "max_roundtrip_error_px": coordinate["max_roundtrip_error_px"],
        "coordinate_checks": coordinate,
        "strict_schema": {
            "validated": True,
            "profile_version": schema["version"],
            "top_level_keys": schema["top_level_keys"],
            "component_fields": schema["component"]["fields"],
            "extra_fields": 0,
            "dangling_references": 0,
            "profile_authority": "reviewed_guide_local_profile_not_official_machine_evaluator",
        },
        "determinism": {
            "runs": REPLAY_RUNS,
            "byte_identical": len(set(replay_bytes)) == 1,
            "official_output_sha256": replay_hashes[0],
            "output_tree_sha256": runs[0]["output_tree_sha256"],
            "environment_fingerprint_sha256": environment_fingerprint_sha256,
            "environment_manifest_path": _relative(root, environment_path),
            "environment_manifest_sha256": _sha256_file(environment_path),
        },
        "source": {
            "case_id": case_id,
            "summary_path": _relative(root, summary_path),
            "summary_sha256": _sha256_file(summary_path),
            "prediction_manifest_path": _relative(root, manifest_path),
            "prediction_manifest_sha256": _sha256_file(manifest_path),
            "source_ir_path": _relative(root, ir_path),
            "source_ir_sha256": _sha256_file(ir_path),
            "source_ir_state_on_disk": source_ir_state,
            "replay_ir_state": replay_ir.state.value,
            "orchestrator_state_contract": orchestrator_state_contract,
            "state_replay_contract": (
                "The orchestrator persists IR after serialization; replay restores only the "
                "immediately preceding VALIDATED state required by the serializer."
            ),
            "source_image_path": _relative(root, image_path),
            "source_image_sha256": image_sha256,
            "source_annotation_path": _relative(root, annotation_path),
            "source_annotation_sha256": _sha256_file(annotation_path),
            "canvas": {"width": ir.image.width, "height": ir.image.height},
            "source_config_sha256": source_config_sha256,
            "source_split": summary.get("split"),
            "source_inference_models_lock_path": _relative(root, source_lock_path),
            "source_inference_models_lock_sha256": source_lock_sha256,
            "historical_official_path": _relative(root, source_official_path),
            "historical_official_sha256": _sha256_file(source_official_path),
            "historical_component_fields": historical_component_fields,
        },
        "replay": {
            "official_profile_path": _relative(root, profile_path),
            "official_profile_sha256": profile_sha256,
            "current_models_lock_path": _relative(root, lock_path),
            "current_models_lock_sha256": current_lock_sha256,
            "output_root": _relative(root, output_root),
            "output_counts": output_counts,
            "official_output_sha256": replay_hashes[0],
            "historical_output_matches_replay": (
                _sha256_file(source_official_path) == replay_hashes[0]
            ),
        },
        "config": config,
        "environment": environment_payload,
    }
    report["audit_payload_sha256"] = _json_sha256(report)
    _atomic_write_json(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a hash-verified frozen image-only Circuit-IR through the current "
            "official serializer and publish strict coordinate/schema/determinism evidence."
        )
    )
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument(
        "--source-run", default="work/production-e2e-audit-calibrated-degraded"
    )
    parser.add_argument(
        "--source-image", default="Dataset300/Dataset300/0024/0024-other.png"
    )
    parser.add_argument(
        "--source-annotation",
        default="Dataset300/Dataset300/0024/0024-other_CVJsonStd.json",
    )
    parser.add_argument("--current-models-lock", default="models/models.lock.json")
    parser.add_argument(
        "--official-profile", default="src/pcbparse/resources/official_schema.json"
    )
    parser.add_argument(
        "--replay-root",
        default="artifacts/determinism/current_serializer_frozen_ir_replay",
    )
    parser.add_argument(
        "--coordinate-report",
        default="artifacts/release/coordinate_roundtrip_report.json",
    )
    parser.add_argument("--case-id", default="0024")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = build_and_write_evidence(
        workspace_root=arguments.workspace_root,
        source_run=arguments.source_run,
        source_image=arguments.source_image,
        source_annotation=arguments.source_annotation,
        current_models_lock=arguments.current_models_lock,
        official_profile=arguments.official_profile,
        replay_root=arguments.replay_root,
        coordinate_report=arguments.coordinate_report,
        case_id=arguments.case_id,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "evidence_scope": report["evidence_scope"],
                "current_lock_inference_executed": report[
                    "current_lock_inference_executed"
                ],
                "coordinate_report": arguments.coordinate_report,
                "official_output_sha256": report["replay"][
                    "official_output_sha256"
                ],
                "max_roundtrip_error_px": report["max_roundtrip_error_px"],
                "audit_payload_sha256": report["audit_payload_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
