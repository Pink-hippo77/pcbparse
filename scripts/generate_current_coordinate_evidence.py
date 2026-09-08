from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC = REPOSITORY_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PIL import Image  # noqa: E402

from pcbparse.config import config_hash, load_config  # noqa: E402
from pcbparse.evaluation.infer_manifest import (  # noqa: E402
    _apply_accuracy_profile,
    production_runtime_source_manifest,
)
from pcbparse.ir.models import CircuitIR, IRState  # noqa: E402
from pcbparse.output.official_schema import (  # noqa: E402
    load_official_schema,
    validate_official_output,
)
from pcbparse.output.official_serializer import (  # noqa: E402
    natural_sort_key,
    official_json_bytes,
    serialize_official,
)
from pcbparse.output import official_serializer as serializer_module  # noqa: E402
from pcbparse.perception.production_provider import (  # noqa: E402
    composite_provider_config_from_pipeline,
    required_model_tasks_for_config,
)


HEX_SHA256 = set("0123456789abcdef")
ROUNDTRIP_LIMIT_PX = 0.5
EVIDENCE_SCOPE = "current_lock_production_image_only_inference"
CANONICAL_RELEASE_MANIFEST = Path("data/splits/dataset300_release50_v1.jsonl")
CANONICAL_RELEASE_SPLIT = "release50"
FORMAL_RELEASE_CASES = 50
RELEASE_SUITE_CONTRACT = "canonical_dataset300_release50_run_binding_v1"
REQUIRED_MODEL_TASKS = {"component", "junction", "ocr", "pin", "wire"}
RUNTIME_COORDINATE_SOURCES = (
    "scripts/generate_current_coordinate_evidence.py",
    "src/pcbparse/evaluation/infer_manifest.py",
    "src/pcbparse/io/image_loader.py",
    "src/pcbparse/ir/transforms.py",
    "src/pcbparse/perception/production_provider.py",
    "src/pcbparse/perception/composite_provider.py",
    "src/pcbparse/perception/coordinate_binding.py",
    "src/pcbparse/perception/ocr_multilevel.py",
    "src/pcbparse/perception/wire_vectorizer.py",
    "src/pcbparse/perception/torch_component_adapter.py",
    "src/pcbparse/perception/torch_ocr_adapter.py",
    "src/pcbparse/perception/torch_pin_adapter.py",
    "src/pcbparse/perception/torch_wire_adapter.py",
    "src/pcbparse/perception/torch_junction_adapter.py",
    "src/pcbparse/pipeline/orchestrator.py",
    "src/pcbparse/output/official_serializer.py",
    "src/pcbparse/output/official_schema.py",
    # Retained because the release gate's historical field explicitly binds it.
    "src/pcbparse/data/derive.py",
)


class CoordinateEvidenceError(ValueError):
    """Raised when current-run coordinate evidence is incomplete or inconsistent."""


def _reject_constant(value: str) -> None:
    raise CoordinateEvidenceError(f"non-standard JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise CoordinateEvidenceError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, CoordinateEvidenceError) as exc:
        raise CoordinateEvidenceError(f"invalid JSON at {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CoordinateEvidenceError(f"cannot read JSONL at {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (json.JSONDecodeError, CoordinateEvidenceError) as exc:
            raise CoordinateEvidenceError(
                f"invalid JSONL at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise CoordinateEvidenceError(
                f"JSONL row at {path}:{line_number} is not an object"
            )
        rows.append(value)
    return rows


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


def _compact_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in HEX_SHA256 for character in value)
    )


def _resolve_inside(root: Path, value: str | Path, label: str) -> Path:
    candidate = Path(value).expanduser()
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve(
        strict=False
    )
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CoordinateEvidenceError(f"{label} escapes workspace: {resolved}") from exc
    return resolved


def _relative(root: Path, path: Path) -> str:
    return path.resolve(strict=False).relative_to(root).as_posix()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise CoordinateEvidenceError(f"{label} does not exist: {path}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CoordinateEvidenceError(message)


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise CoordinateEvidenceError(f"{label} is missing")
    rendered = value.strip()
    try:
        parsed = datetime.fromisoformat(
            rendered[:-1] + "+00:00" if rendered.endswith("Z") else rendered
        )
    except ValueError as exc:
        raise CoordinateEvidenceError(f"{label} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise CoordinateEvidenceError(f"{label} lacks timezone")
    return parsed.astimezone(timezone.utc)


def _atomic_write(path: Path, payload: bytes) -> None:
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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n",
    )


def _stable_prediction_set_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "case_id",
        "duplicate_group",
        "image_sha256",
        "ir_sha256",
        "official_sha256",
        "official_output_sha256",
        "provider_kind",
        "model_lock_sha256",
        "config_hash",
        "sidecar_access",
        "status",
        "error",
    )
    if any(any(field not in row for field in fields) for row in rows):
        raise CoordinateEvidenceError("prediction rows lack stable-set fields")
    return _compact_sha256(
        [{field: row[field] for field in fields} for row in rows]
    )


def _path_manifest(root: Path, paths: Sequence[Path]) -> list[dict[str, str]]:
    unique: dict[str, Path] = {}
    for path in paths:
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise CoordinateEvidenceError(
                f"artifact path escapes workspace: {resolved}"
            ) from exc
        _require_file(resolved, "bound artifact")
        unique[str(resolved).casefold()] = resolved
    return [
        {"path": _relative(root, path), "sha256": _sha256_file(path)}
        for path in sorted(unique.values(), key=lambda item: _relative(root, item))
    ]


def _finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CoordinateEvidenceError(f"{label} is not numeric")
    rendered = float(value)
    if not math.isfinite(rendered):
        raise CoordinateEvidenceError(f"{label} is not finite")
    return rendered


def _point(value: object, label: str, width: float, height: float) -> tuple[float, float]:
    if not isinstance(value, Mapping):
        raise CoordinateEvidenceError(f"{label} is not a point")
    x = _finite(value.get("x"), f"{label}.x")
    y = _finite(value.get("y"), f"{label}.y")
    if not (0.0 <= x <= width and 0.0 <= y <= height):
        raise CoordinateEvidenceError(f"{label} is outside the image canvas")
    return x, y


def _bbox(value: object, label: str, width: float, height: float) -> tuple[float, float, float, float]:
    if not isinstance(value, Mapping):
        raise CoordinateEvidenceError(f"{label} is not a bbox")
    x1 = _finite(value.get("x1"), f"{label}.x1")
    y1 = _finite(value.get("y1"), f"{label}.y1")
    x2 = _finite(value.get("x2"), f"{label}.x2")
    y2 = _finite(value.get("y2"), f"{label}.y2")
    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        raise CoordinateEvidenceError(f"{label} is invalid or outside the image canvas")
    return x1, y1, x2, y2


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _matrix_roundtrip(ir: CircuitIR) -> tuple[int, float, str]:
    observations: list[dict[str, Any]] = []
    previous_target: str | None = None
    for index, raw in enumerate(ir.transforms):
        if not isinstance(raw, Mapping):
            raise CoordinateEvidenceError(f"IR transform {index} is not an object")
        matrix = raw.get("matrix")
        if (
            not isinstance(matrix, list)
            or len(matrix) != 3
            or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
        ):
            raise CoordinateEvidenceError(f"IR transform {index} is not 3x3")
        values = [
            [_finite(value, f"IR transform {index} matrix") for value in row]
            for row in matrix
        ]
        if values[2] != [0.0, 0.0, 1.0]:
            raise CoordinateEvidenceError(
                f"IR transform {index} is not an affine xy transform"
            )
        source_view = raw.get("source_view")
        target_view = raw.get("target_view")
        if not isinstance(source_view, str) or not source_view:
            raise CoordinateEvidenceError(f"IR transform {index} lacks source_view")
        if not isinstance(target_view, str) or not target_view:
            raise CoordinateEvidenceError(f"IR transform {index} lacks target_view")
        if previous_target is not None and source_view != previous_target:
            raise CoordinateEvidenceError("IR transform chain is not contiguous")
        previous_target = target_view
        a, b, c = values[0]
        d, e, f = values[1]
        determinant = a * e - b * d
        if abs(determinant) < 1e-12:
            raise CoordinateEvidenceError(f"IR transform {index} is singular")
        probes = ((0.0, 0.0), (1.0, 1.0), (17.25, 9.75))
        for probe_index, (x, y) in enumerate(probes):
            tx, ty = a * x + b * y + c, d * x + e * y + f
            rx = (e * (tx - c) - b * (ty - f)) / determinant
            ry = (-d * (tx - c) + a * (ty - f)) / determinant
            error = _distance((x, y), (rx, ry))
            observations.append(
                {
                    "transform_index": index,
                    "probe_index": probe_index,
                    "source_xy": [x, y],
                    "view_xy": [tx, ty],
                    "restored_xy": [rx, ry],
                    "roundtrip_error_px": error,
                }
            )
    maximum = max(
        (float(item["roundtrip_error_px"]) for item in observations), default=0.0
    )
    return len(observations), maximum, _compact_sha256(observations)


def _axis_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise CoordinateEvidenceError("resolved tile geometry is invalid")
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    values = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if values[-1] != last:
        values.append(last)
    return values


def _local_view_roundtrip(
    ir: CircuitIR,
    perception: Mapping[str, Any],
) -> tuple[int, float, str, int, dict[str, int]]:
    """Independently bind persisted local evidence to final Circuit-IR geometry."""

    width, height = int(ir.image.width), int(ir.image.height)
    view_records: list[tuple[str, str, int, int, int, int]] = []
    view_maps: dict[str, dict[str, tuple[int, int, int, int]]] = {
        "component": {},
        "wire": {},
        "ocr": {},
        "component_ocr": {},
        "component_pin": {},
    }

    def add_view(
        family: str,
        view_id: str,
        left: int,
        top: int,
        view_width: int,
        view_height: int,
    ) -> None:
        record = (left, top, view_width, view_height)
        previous = view_maps[family].get(view_id)
        if previous is not None and previous != record:
            raise CoordinateEvidenceError(
                f"reconstructed {family} view {view_id!r} is ambiguous"
            )
        view_maps[family][view_id] = record
        view_records.append(
            (family, view_id, left, top, view_width, view_height)
        )

    def add_tiles(family: str, tile_size_field: str, overlap_field: str) -> None:
        tile_size = perception.get(tile_size_field)
        overlap = perception.get(overlap_field)
        if type(tile_size) is not int or type(overlap) is not int:
            raise CoordinateEvidenceError(
                f"resolved perception config lacks {family} tile geometry"
            )
        if width <= tile_size and height <= tile_size:
            add_view(family, "original", 0, 0, width, height)
            return
        for row, top in enumerate(_axis_starts(height, tile_size, overlap)):
            for column, left in enumerate(_axis_starts(width, tile_size, overlap)):
                identifier = f"tile_{row:03d}_{column:03d}"
                add_view(
                    family,
                    identifier,
                    left,
                    top,
                    min(width, left + tile_size) - left,
                    min(height, top + tile_size) - top,
                )

    add_tiles("component", "component_tile_size", "component_tile_overlap")
    add_tiles("wire", "wire_tile_size", "wire_tile_overlap")
    add_tiles("ocr", "ocr_tile_size", "ocr_tile_overlap")

    padding_factor = _finite(
        perception.get("component_roi_padding_factor"),
        "resolved component_roi_padding_factor",
    )
    for family, prefix, absolute_field in (
        ("component_ocr", "component-ocr", "ocr_component_roi_padding_px"),
        ("component_pin", "component-pin", "pin_component_roi_padding_px"),
    ):
        absolute_padding = _finite(
            perception.get(absolute_field), f"resolved {absolute_field}"
        )
        for component in ir.components:
            padding = max(
                max(float(component.bbox.width), float(component.bbox.height))
                * padding_factor,
                absolute_padding,
            )
            left = max(0, min(width, math.floor(component.bbox.x1 - padding)))
            top = max(0, min(height, math.floor(component.bbox.y1 - padding)))
            right = max(left, min(width, math.ceil(component.bbox.x2 + padding)))
            bottom = max(top, min(height, math.ceil(component.bbox.y2 + padding)))
            if right <= left or bottom <= top:
                raise CoordinateEvidenceError(
                    f"component {component.internal_id} has an invalid reconstructed ROI"
                )
            full_id = f"{prefix}:{component.internal_id}:full"
            full_width, full_height = right - left, bottom - top
            add_view(family, full_id, left, top, full_width, full_height)
            strip_width = max(1.0, full_width * 0.2)
            strip_height = max(1.0, full_height * 0.2)
            side_origins = {
                "L": (0, 0),
                "R": (math.floor(full_width - strip_width), 0),
                "T": (0, 0),
                "B": (0, math.floor(full_height - strip_height)),
            }
            for side, (offset_x, offset_y) in side_origins.items():
                side_id = f"{full_id}:{side}"
                side_width = (
                    math.ceil(strip_width) if side in {"L", "R"} else full_width
                )
                side_height = (
                    math.ceil(strip_height) if side in {"T", "B"} else full_height
                )
                add_view(
                    family,
                    side_id,
                    left + offset_x,
                    top + offset_y,
                    min(side_width, width - left - offset_x),
                    min(side_height, height - top - offset_y),
                )

    observations: list[dict[str, Any]] = []
    for family, view_id, left, top, view_width, view_height in view_records:
        if view_width <= 0 or view_height <= 0:
            raise CoordinateEvidenceError(f"reconstructed view {view_id} is empty")
        for probe_index, (local_x, local_y) in enumerate(
            (
                (0.0, 0.0),
                (float(view_width), 0.0),
                (0.0, float(view_height)),
                (float(view_width), float(view_height)),
                (float(view_width) / 2.0, float(view_height) / 2.0),
            )
        ):
            global_xy = (local_x + left, local_y + top)
            restored = (global_xy[0] - left, global_xy[1] - top)
            error = _distance((local_x, local_y), restored)
            observations.append(
                {
                    "family": family,
                    "view_id": view_id,
                    "probe_index": probe_index,
                    "local_xy": [local_x, local_y],
                    "global_xy": [global_xy[0], global_xy[1]],
                    "restored_local_xy": [restored[0], restored[1]],
                    "roundtrip_error_px": error,
                }
            )

    binding_observations: list[dict[str, Any]] = []
    local_evidence_checked = 0
    module_objects_checked: Counter[str] = Counter()

    def finite_vector(value: Any, length: int, label: str) -> list[float]:
        if not isinstance(value, list) or len(value) != length:
            raise CoordinateEvidenceError(f"{label} must contain {length} numbers")
        return [_finite(item, label) for item in value]

    def apply_matrix(matrix: list[list[float]], point: list[float]) -> list[float]:
        return [
            matrix[0][0] * point[0]
            + matrix[0][1] * point[1]
            + matrix[0][2],
            matrix[1][0] * point[0]
            + matrix[1][1] * point[1]
            + matrix[1][2],
        ]

    def geometry_points(
        geometry: Mapping[str, Any],
        *,
        expected_kind: str,
        label: str,
    ) -> list[tuple[str, list[float]]]:
        kind = geometry.get("kind")
        if kind != expected_kind:
            raise CoordinateEvidenceError(
                f"{label} geometry kind {kind!r} is not {expected_kind!r}"
            )
        if expected_kind == "bbox_xyxy":
            values = finite_vector(geometry.get("values"), 4, f"{label}.values")
            if not values[0] < values[2] or not values[1] < values[3]:
                raise CoordinateEvidenceError(f"{label} bbox is invalid")
            return [("top_left", values[:2]), ("bottom_right", values[2:])]
        if expected_kind == "point":
            output = [
                ("point", finite_vector(geometry.get("point_xy"), 2, f"{label}.point_xy"))
            ]
            if "base_xy" in geometry:
                output.append(
                    ("base", finite_vector(geometry.get("base_xy"), 2, f"{label}.base_xy"))
                )
            return output
        if expected_kind == "segment":
            return [
                ("start", finite_vector(geometry.get("start_xy"), 2, f"{label}.start_xy")),
                ("end", finite_vector(geometry.get("end_xy"), 2, f"{label}.end_xy")),
            ]
        raise CoordinateEvidenceError(f"unsupported coordinate binding kind {expected_kind}")

    def final_points(item: Any, expected_kind: str) -> list[tuple[str, list[float]]]:
        if expected_kind == "bbox_xyxy":
            return [
                ("top_left", [float(item.bbox.x1), float(item.bbox.y1)]),
                ("bottom_right", [float(item.bbox.x2), float(item.bbox.y2)]),
            ]
        if expected_kind == "point":
            output = [("point", [float(item.point.x), float(item.point.y)])]
            if getattr(item, "base", None) is not None:
                output.append(("base", [float(item.base.x), float(item.base.y)]))
            return output
        return [
            ("start", [float(item.start.x), float(item.start.y)]),
            ("end", [float(item.end.x), float(item.end.y)]),
        ]

    def check_evidence(
        items: Sequence[Any],
        *,
        family_names: tuple[str, ...],
        label: str,
        expected_kind: str,
    ) -> None:
        nonlocal local_evidence_checked
        expected_object_kind = label
        allowed_views = {
            view_id: record
            for family in family_names
            for view_id, record in view_maps[family].items()
        }
        for item in items:
            item_binding_count = 0
            item_matches_final = False
            for evidence in item.evidence:
                metadata = evidence.metadata
                declares_local = bool(
                    str(metadata.get("coordinate_system", "")).startswith(
                        "image_view_local"
                    )
                    or metadata.get("coordinate_space") == "image_view_local"
                    or "heatmap_point" in metadata
                    or (
                        evidence.source == "geometry"
                        and metadata.get("vectorizer")
                    )
                )
                binding = metadata.get("local_to_global_binding")
                if binding is None:
                    if declares_local:
                        raise CoordinateEvidenceError(
                            f"{label} {item.internal_id} local evidence lacks a persisted binding"
                        )
                    continue
                if not isinstance(binding, Mapping):
                    raise CoordinateEvidenceError(
                        f"{label} {item.internal_id} coordinate binding is not an object"
                    )
                if binding.get("contract") != "image_view_local_to_original_v1":
                    raise CoordinateEvidenceError(
                        f"{label} {item.internal_id} coordinate binding contract is invalid"
                    )
                if binding.get("object_kind") != expected_object_kind:
                    raise CoordinateEvidenceError(
                        f"{label} {item.internal_id} coordinate binding object kind differs"
                    )
                view_id = binding.get("view_id")
                if view_id != evidence.view_id or view_id not in allowed_views:
                    raise CoordinateEvidenceError(
                        f"{label} {item.internal_id} references unknown local view {view_id!r}"
                    )
                left, top, view_width, view_height = allowed_views[str(view_id)]
                if binding.get("view_size") != [view_width, view_height]:
                    raise CoordinateEvidenceError(
                        f"{label} {item.internal_id} persisted view size differs"
                    )
                raw_matrix = binding.get("local_to_original_matrix")
                if (
                    not isinstance(raw_matrix, list)
                    or len(raw_matrix) != 3
                    or any(not isinstance(row, list) or len(row) != 3 for row in raw_matrix)
                ):
                    raise CoordinateEvidenceError(
                        f"{label} {item.internal_id} coordinate binding matrix is invalid"
                    )
                matrix = [
                    [_finite(value, f"{label} {item.internal_id} binding matrix") for value in row]
                    for row in raw_matrix
                ]
                expected_matrix = [
                    [1.0, 0.0, float(left)],
                    [0.0, 1.0, float(top)],
                    [0.0, 0.0, 1.0],
                ]
                matrix_error = max(
                    abs(matrix[row][column] - expected_matrix[row][column])
                    for row in range(3)
                    for column in range(3)
                )
                if matrix_error > 1e-9:
                    raise CoordinateEvidenceError(
                        f"{label} {item.internal_id} binding matrix differs from reconstructed view"
                    )
                local_geometry = binding.get("local_geometry")
                original_geometry = binding.get("original_geometry")
                if not isinstance(local_geometry, Mapping) or not isinstance(
                    original_geometry, Mapping
                ):
                    raise CoordinateEvidenceError(
                        f"{label} {item.internal_id} binding geometry is missing"
                    )
                local_points = geometry_points(
                    local_geometry,
                    expected_kind=expected_kind,
                    label=f"{label} {item.internal_id} local",
                )
                original_points = geometry_points(
                    original_geometry,
                    expected_kind=expected_kind,
                    label=f"{label} {item.internal_id} original",
                )
                if [name for name, _ in local_points] != [
                    name for name, _ in original_points
                ]:
                    raise CoordinateEvidenceError(
                        f"{label} {item.internal_id} local/original geometry differs"
                    )
                reconstructed = [
                    (name, apply_matrix(matrix, point))
                    for name, point in local_points
                ]
                transform_error = max(
                    _distance(tuple(point), tuple(dict(original_points)[name]))
                    for name, point in reconstructed
                )
                if transform_error > ROUNDTRIP_LIMIT_PX:
                    raise CoordinateEvidenceError(
                        f"{label} {item.internal_id} persisted local-to-global transform differs"
                    )
                final_geometry = final_points(item, expected_kind)
                declared_by_name = dict(original_points)
                final_names_match = [name for name, _ in final_geometry] == [
                    name for name, _ in original_points
                ]
                final_error: float | None = (
                    max(
                        _distance(tuple(point), tuple(declared_by_name[name]))
                        for name, point in final_geometry
                    )
                    if final_names_match
                    else None
                )
                matches_final = (
                    final_error is not None
                    and final_error <= ROUNDTRIP_LIMIT_PX
                )
                item_matches_final = item_matches_final or matches_final
                item_binding_count += 1
                local_evidence_checked += 1
                binding_observations.append(
                    {
                        "object_kind": label,
                        "object_id": item.internal_id,
                        "evidence_id": evidence.evidence_id,
                        "view_id": view_id,
                        "local_geometry": dict(local_geometry),
                        "declared_original_geometry": dict(original_geometry),
                        "matrix_error_px": matrix_error,
                        "transform_error_px": transform_error,
                        "final_geometry_error_px": final_error,
                        "matches_final_object": matches_final,
                    }
                )
            exempt_synthetic = label == "segment" and bool(
                getattr(item, "synthetic", False)
            )
            if not exempt_synthetic and item_binding_count == 0:
                raise CoordinateEvidenceError(
                    f"{label} {item.internal_id} has no persisted local-to-global binding"
                )
            if item_binding_count and not item_matches_final:
                raise CoordinateEvidenceError(
                    f"{label} {item.internal_id} has no binding matching final IR geometry"
                )
            if item_binding_count:
                module_objects_checked[label] += 1

    check_evidence(
        ir.components,
        family_names=("component",),
        label="component",
        expected_kind="bbox_xyxy",
    )
    check_evidence(
        ir.texts,
        family_names=("ocr", "component_ocr"),
        label="text",
        expected_kind="bbox_xyxy",
    )
    check_evidence(
        ir.pins,
        family_names=("component_pin",),
        label="pin",
        expected_kind="point",
    )
    check_evidence(
        ir.segments,
        family_names=("wire",),
        label="segment",
        expected_kind="segment",
    )
    check_evidence(
        ir.junctions,
        family_names=("wire",),
        label="junction",
        expected_kind="point",
    )
    maximum = max(
        [
            *(float(item["roundtrip_error_px"]) for item in observations),
            *(
                float(item["transform_error_px"])
                for item in binding_observations
            ),
            *(
                float(item["final_geometry_error_px"])
                for item in binding_observations
                if isinstance(item["final_geometry_error_px"], (int, float))
                and not isinstance(item["final_geometry_error_px"], bool)
                and math.isfinite(float(item["final_geometry_error_px"]))
            ),
        ],
        default=0.0,
    )
    return (
        len(observations),
        maximum,
        _compact_sha256(
            {"view_probes": observations, "coordinate_bindings": binding_observations}
        ),
        local_evidence_checked,
        dict(sorted(module_objects_checked.items())),
    )


def _range_check_ir(ir: CircuitIR) -> int:
    width, height = float(ir.image.width), float(ir.image.height)
    points = 0
    for item in ir.components:
        _bbox(item.bbox.model_dump(mode="json"), f"component {item.internal_id}", width, height)
        points += 2
    for item in ir.texts:
        _bbox(item.bbox.model_dump(mode="json"), f"text {item.internal_id}", width, height)
        points += 2
    for item in ir.pins:
        _point(item.point.model_dump(mode="json"), f"pin {item.internal_id}", width, height)
        points += 1
        if item.base is not None:
            _point(item.base.model_dump(mode="json"), f"pin base {item.internal_id}", width, height)
            points += 1
    for item in ir.segments:
        _point(item.start.model_dump(mode="json"), f"segment start {item.internal_id}", width, height)
        _point(item.end.model_dump(mode="json"), f"segment end {item.internal_id}", width, height)
        points += 2
    for item in ir.junctions:
        _point(item.point.model_dump(mode="json"), f"junction {item.internal_id}", width, height)
        points += 1
    for item in ir.net_labels:
        if item.point is not None:
            _point(item.point.model_dump(mode="json"), f"net label {item.internal_id}", width, height)
            points += 1
    return points


def _independent_net_geometry_records(
    ir: CircuitIR,
    net: Any,
    image_height: float,
) -> dict[
    tuple[tuple[float, float], tuple[float, float]],
    tuple[tuple[float, float], tuple[float, float]],
]:
    """Build expected official edges directly from IR using literal ``H-y``.

    This deliberately does not call any serializer geometry helper.  It is an
    independent oracle for the release audit, including the pin-only
    zero-length edge rule required at the official serialization boundary.
    Values map official endpoint geometry to its corresponding IR source
    endpoints, both in the same canonical endpoint order.
    """

    def encoded_source_pair(
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> tuple[
        tuple[tuple[float, float], tuple[float, float]],
        tuple[tuple[float, float], tuple[float, float]],
    ]:
        paired = sorted(
            (
                ((float(first[0]), float(image_height) - float(first[1])), first),
                ((float(second[0]), float(image_height) - float(second[1])), second),
            ),
            key=lambda item: (item[0], item[1]),
        )
        return (
            (paired[0][0], paired[1][0]),
            (
                (float(paired[0][1][0]), float(paired[0][1][1])),
                (float(paired[1][1][0]), float(paired[1][1][1])),
            ),
        )

    records: dict[
        tuple[tuple[float, float], tuple[float, float]],
        tuple[tuple[float, float], tuple[float, float]],
    ] = {}
    segment_index = ir.segment_index()
    for segment_id in sorted(set(net.segments)):
        segment = segment_index.get(segment_id)
        if segment is None:
            raise CoordinateEvidenceError(
                f"net {net.internal_id} references unknown segment {segment_id}"
            )
        geometry, sources = encoded_source_pair(
            (float(segment.start.x), float(segment.start.y)),
            (float(segment.end.x), float(segment.end.y)),
        )
        records.setdefault(geometry, sources)

    candidate_pin_ids: set[str] = set()
    physical_index = {item.internal_id: item for item in ir.physical_nets}
    if net.physical_net_ids:
        for physical_id in set(net.physical_net_ids):
            physical = physical_index.get(physical_id)
            if physical is None:
                raise CoordinateEvidenceError(
                    f"net {net.internal_id} references unknown physical net {physical_id}"
                )
            if not physical.segments:
                candidate_pin_ids.update(physical.pins)
    elif not net.segments:
        candidate_pin_ids.update(net.pins)

    pin_index = ir.pin_index()
    for pin_id in sorted(candidate_pin_ids):
        pin = pin_index.get(pin_id)
        if pin is None:
            raise CoordinateEvidenceError(
                f"net {net.internal_id} references unknown pin {pin_id}"
            )
        source = (float(pin.point.x), float(pin.point.y))
        geometry, sources = encoded_source_pair(source, source)
        records.setdefault(geometry, sources)
    return records


def _coordinate_case(
    *,
    case_id: str,
    image_path: Path,
    ir_path: Path,
    official_path: Path,
    row: Mapping[str, Any],
    schema: Mapping[str, Any],
    resolved_perception: Mapping[str, Any],
) -> dict[str, Any]:
    with Image.open(image_path) as opened:
        opened.load()
        width, height = int(opened.width), int(opened.height)
    try:
        ir = CircuitIR.model_validate_json(ir_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CoordinateEvidenceError(f"case {case_id}: invalid Circuit-IR: {exc}") from exc
    _require(ir.state in {IRState.SERIALIZED, IRState.VALIDATED}, f"case {case_id}: IR is not post-validation")
    _require((ir.image.width, ir.image.height) == (width, height), f"case {case_id}: IR canvas differs from raster")
    _require(ir.image.sha256 == row.get("image_sha256"), f"case {case_id}: IR image hash mismatch")
    _require(Path(ir.image.path).expanduser().resolve() == image_path, f"case {case_id}: IR image path mismatch")
    official = _read_json(official_path)
    if not isinstance(official, Mapping):
        raise CoordinateEvidenceError(f"case {case_id}: official output is not an object")
    try:
        validate_official_output(official, schema)
    except Exception as exc:
        raise CoordinateEvidenceError(f"case {case_id}: official schema validation failed: {exc}") from exc
    replay_ir = ir.model_copy(deep=True)
    replay_ir.state = IRState.VALIDATED
    try:
        replay = serialize_official(replay_ir, schema)
        replay_bytes = official_json_bytes(replay, schema)
    except Exception as exc:
        raise CoordinateEvidenceError(f"case {case_id}: current serializer replay failed: {exc}") from exc
    if replay_bytes != official_path.read_bytes():
        raise CoordinateEvidenceError(
            f"case {case_id}: current serializer bytes differ from formal output"
        )

    ir_geometry_points = _range_check_ir(ir)
    transform_points, transform_maximum, transform_hash = _matrix_roundtrip(ir)
    (
        local_view_points,
        local_view_maximum,
        local_view_hash,
        local_evidence_checked,
        local_module_objects_checked,
    ) = _local_view_roundtrip(ir, resolved_perception)
    component_keys = serializer_module._component_key_map(replay_ir)
    pin_numbers = serializer_module._pin_number_map(replay_ir, component_keys)
    observations: list[dict[str, Any]] = []
    forward_errors: list[float] = []
    roundtrip_errors: list[float] = []

    def observe(kind: str, object_id: str, source: tuple[float, float], actual: tuple[float, float]) -> None:
        expected = (source[0], float(height) - source[1])
        restored = (actual[0], float(height) - actual[1])
        forward = _distance(expected, actual)
        roundtrip = _distance(source, restored)
        if not math.isfinite(forward) or not math.isfinite(roundtrip):
            raise CoordinateEvidenceError(f"case {case_id}: non-finite coordinate error")
        forward_errors.append(forward)
        roundtrip_errors.append(roundtrip)
        observations.append(
            {
                "kind": kind,
                "object_id": object_id,
                "source_xy": [source[0], source[1]],
                "official_xy": [actual[0], actual[1]],
                "expected_official_xy": [expected[0], expected[1]],
                "restored_xy": [restored[0], restored[1]],
                "source_to_serialized_error_px": forward,
                "roundtrip_error_px": roundtrip,
            }
        )

    official_components = official.get("components")
    official_pins = official.get("pins")
    official_nets = official.get("nets")
    if not all(isinstance(value, Mapping) for value in (official_components, official_pins, official_nets)):
        raise CoordinateEvidenceError(f"case {case_id}: official sections are invalid")
    if set(official_components) != set(component_keys.values()):
        raise CoordinateEvidenceError(f"case {case_id}: component output coverage differs from IR")
    component_points = 0
    for component in replay_ir.components:
        key = component_keys[component.internal_id]
        raw_bbox = official_components[key].get("bbox")
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            raise CoordinateEvidenceError(f"case {case_id}: component {key} bbox is invalid")
        values = [_finite(value, f"case {case_id} component {key} bbox") for value in raw_bbox]
        observe(
            "component_bbox_top_left",
            component.internal_id,
            (float(component.bbox.x1), float(component.bbox.y1)),
            (values[0], values[3]),
        )
        observe(
            "component_bbox_bottom_right",
            component.internal_id,
            (float(component.bbox.x2), float(component.bbox.y2)),
            (values[2], values[1]),
        )
        component_points += 2

    pin_template = str(schema["pin"]["key_template"])
    expanded_pin_points = 0
    for pin in replay_ir.pins:
        component_key = component_keys[str(pin.component_id)]
        group = official_pins.get(component_key)
        if not isinstance(group, Mapping):
            raise CoordinateEvidenceError(f"case {case_id}: pin group {component_key} is missing")
        for number in pin_numbers[pin.internal_id]:
            pin_key = pin_template.format(number=number)
            encoded = group.get(pin_key)
            point_payload = encoded.get("point") if isinstance(encoded, Mapping) else None
            actual = _point(
                point_payload,
                f"case {case_id} pin {component_key}.{pin_key}",
                float(width),
                float(height),
            )
            observe(
                "pin",
                f"{pin.internal_id}:{number}",
                (float(pin.point.x), float(pin.point.y)),
                actual,
            )
            expanded_pin_points += 1

    pin_index = replay_ir.pin_index()
    segment_index = replay_ir.segment_index()
    pin_ref_template = str(schema["net"]["hypergraph_pin_template"])
    ordered_nets = sorted(
        replay_ir.electrical_nets,
        key=lambda net: serializer_module._net_signature(
            net,
            pin_index,
            component_keys,
            pin_numbers,
            segment_index,
            pin_ref_template,
            float(height),
        ),
    )
    expected_net_keys = [
        str(schema["net"]["key_template"]).format(index=index)
        for index in range(1, len(ordered_nets) + 1)
    ]
    if list(official_nets) != expected_net_keys:
        raise CoordinateEvidenceError(f"case {case_id}: net output order/coverage differs from IR")
    edge_points = 0
    zero_length_edges = 0
    for net_index, (net, net_key) in enumerate(zip(ordered_nets, expected_net_keys), start=1):
        expected_geometry_records = _independent_net_geometry_records(
            replay_ir,
            net,
            float(height),
        )
        expected_geometries = set(expected_geometry_records)
        raw_net = official_nets[net_key]
        raw_edges = raw_net.get("edges") if isinstance(raw_net, Mapping) else None
        if not isinstance(raw_edges, Mapping):
            raise CoordinateEvidenceError(f"case {case_id}: net {net_key} edges are missing")
        actual_geometries: Counter[tuple[tuple[float, float], tuple[float, float]]] = Counter()
        for edge_key, edge in raw_edges.items():
            if not isinstance(edge, list) or len(edge) != 2:
                raise CoordinateEvidenceError(f"case {case_id}: edge {net_key}.{edge_key} is invalid")
            first = _point(edge[0], f"case {case_id} edge {net_key}.{edge_key}[0]", float(width), float(height))
            second = _point(edge[1], f"case {case_id} edge {net_key}.{edge_key}[1]", float(width), float(height))
            geometry = tuple(sorted((first, second)))
            actual_geometries[geometry] += 1
        if actual_geometries != Counter(expected_geometries):
            raise CoordinateEvidenceError(
                f"case {case_id}: per-net edge geometry mismatch for {net_key}"
            )
        for edge_number, geometry in enumerate(sorted(expected_geometries), start=1):
            source_geometry = expected_geometry_records[geometry]
            for endpoint_index, (actual, source) in enumerate(
                zip(geometry, source_geometry)
            ):
                observe(
                    "net_edge_endpoint",
                    f"{net.internal_id}:{edge_number}:{endpoint_index}",
                    source,
                    actual,
                )
                edge_points += 1
            if geometry[0] == geometry[1]:
                zero_length_edges += 1

    points_checked = component_points + expanded_pin_points + edge_points
    if points_checked != len(observations):
        raise CoordinateEvidenceError(f"case {case_id}: coordinate point recount mismatch")
    max_forward = max(forward_errors, default=0.0)
    max_roundtrip = max(
        roundtrip_errors + [transform_maximum, local_view_maximum], default=0.0
    )
    if max_forward > ROUNDTRIP_LIMIT_PX or max_roundtrip > ROUNDTRIP_LIMIT_PX:
        raise CoordinateEvidenceError(f"case {case_id}: coordinate error exceeds 0.5 px")
    return {
        "case_id": case_id,
        "image_sha256": _sha256_file(image_path),
        "ir_sha256": _sha256_file(ir_path),
        "official_sha256": _sha256_file(official_path),
        "width": width,
        "height": height,
        "component_bbox_points_checked": component_points,
        "pin_points_checked": expanded_pin_points,
        "net_edge_points_checked": edge_points,
        "zero_length_edges_checked": zero_length_edges,
        "points_checked": points_checked,
        "ir_geometry_points_range_checked": ir_geometry_points,
        "persisted_transform_points_checked": transform_points,
        "persisted_transform_observations_sha256": transform_hash,
        "local_to_global_transform_points_checked": local_view_points,
        "local_to_global_coordinate_evidence_checked": local_evidence_checked,
        "local_to_global_module_objects_checked": local_module_objects_checked,
        "local_to_global_max_roundtrip_error_px": max(
            transform_maximum, local_view_maximum
        ),
        "local_to_global_observations_sha256": local_view_hash,
        "max_source_to_serialized_error_px": max_forward,
        "max_roundtrip_error_px": max_roundtrip,
        "coordinate_observations_sha256": _compact_sha256(observations),
    }


def recompute_coordinate_evidence(
    *,
    workspace_root: str | Path,
    source_summary: str | Path,
    release_manifest: str | Path,
    current_models_lock: str | Path,
    pipeline_config: str | Path,
    profile: str | Path,
) -> dict[str, Any]:
    """Recompute coordinate evidence from one already-completed formal run.

    This function never loads a model or calls inference.  It verifies and
    replays only the IR/output bytes already bound by the production summary.
    """

    root = Path(workspace_root).expanduser().resolve()
    summary_path = _resolve_inside(root, source_summary, "source summary")
    release_path = _resolve_inside(root, release_manifest, "release manifest")
    canonical_release_path = (root / CANONICAL_RELEASE_MANIFEST).resolve(
        strict=False
    )
    _require(
        release_path == canonical_release_path,
        "formal coordinate evidence requires canonical "
        f"{CANONICAL_RELEASE_MANIFEST.as_posix()}",
    )
    lock_path = _resolve_inside(root, current_models_lock, "models.lock")
    pipeline_path = _resolve_inside(root, pipeline_config, "pipeline config")
    profile_path = _resolve_inside(root, profile, "production profile")
    for path, label in (
        (summary_path, "source summary"),
        (release_path, "release manifest"),
        (lock_path, "models.lock"),
        (pipeline_path, "pipeline config"),
        (profile_path, "production profile"),
    ):
        _require_file(path, label)

    runtime_paths = [root / relative for relative in RUNTIME_COORDINATE_SOURCES]
    for path in runtime_paths:
        _require_file(path, "coordinate runtime source")
    runtime_manifest = _path_manifest(root, runtime_paths)

    summary = _read_json(summary_path)
    if not isinstance(summary, Mapping):
        raise CoordinateEvidenceError("source summary root is not an object")
    source_runtime_manifest = summary.get("runtime_source_manifest")
    if not isinstance(source_runtime_manifest, Mapping):
        raise CoordinateEvidenceError(
            "source summary lacks inference-time runtime_source_manifest"
        )
    try:
        current_runtime_manifest = production_runtime_source_manifest(root)
    except Exception as exc:
        raise CoordinateEvidenceError(
            f"cannot recompute current production runtime source manifest: {exc}"
        ) from exc
    _require(
        dict(source_runtime_manifest) == current_runtime_manifest,
        "source summary inference-time runtime source manifest differs from current bytes",
    )
    required_summary = {
        "kind": "production-image-only-inference",
        "accuracy_mode": "image-only",
        "execution_mode": "production",
        "provider_kind": "production",
        "sidecar_access": False,
        "status": "ok",
        "all_successful": True,
        "failed": 0,
        "not_run": 0,
    }
    for field, expected in required_summary.items():
        _require(summary.get(field) == expected, f"source summary {field} is not {expected!r}")
    _require(
        summary.get("split") == CANONICAL_RELEASE_SPLIT,
        f"source summary split is not {CANONICAL_RELEASE_SPLIT!r}",
    )
    for field in ("selected_cases", "attempted", "succeeded"):
        _require(
            summary.get(field) == FORMAL_RELEASE_CASES,
            f"source summary {field} is not exactly {FORMAL_RELEASE_CASES}",
        )
    _require(
        type(summary.get("schema_version")) is int
        and int(summary["schema_version"]) >= 2,
        "source summary schema_version is not formal",
    )
    timing = summary.get("timing")
    if not isinstance(timing, Mapping):
        raise CoordinateEvidenceError("source summary lacks formal timing contract")
    timing_contract = {
        "contract": "production_wall_clock_scope_v1",
        "scope": "function_entry_through_prediction_manifest_fsync",
        "cold_start_included": True,
        "config_and_profile_load_included": True,
        "readiness_included": True,
        "provider_and_model_load_included": True,
        "case_loop_included": True,
        "prediction_manifest_write_included": True,
        "summary_write_and_cli_teardown_included": False,
    }
    for field, expected in timing_contract.items():
        _require(
            timing.get(field) == expected,
            f"source summary timing.{field} is not {expected!r}",
        )
    timing_values = {
        field: _finite(timing.get(field), f"source timing.{field}")
        for field in (
            "initialization_seconds",
            "case_loop_seconds",
            "evidence_finalize_seconds",
            "total_elapsed_seconds",
        )
    }
    _require(
        all(value >= 0.0 for value in timing_values.values()),
        "source timing contains a negative duration",
    )
    elapsed = _finite(summary.get("elapsed_seconds"), "source elapsed_seconds")
    _require(elapsed >= 0.0, "source elapsed_seconds is negative")
    _require(
        math.isclose(
            elapsed,
            timing_values["total_elapsed_seconds"],
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "source elapsed_seconds differs from timing total",
    )
    _require(
        math.isclose(
            timing_values["initialization_seconds"]
            + timing_values["case_loop_seconds"]
            + timing_values["evidence_finalize_seconds"],
            timing_values["total_elapsed_seconds"],
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "source timing partitions do not reproduce total",
    )
    run_id = summary.get("inference_run_id")
    _require(_is_sha256(run_id), "source summary lacks a valid inference_run_id")
    started_at = _parse_utc(summary.get("started_at_utc"), "source started_at_utc")
    completed_at = _parse_utc(summary.get("completed_at_utc"), "source completed_at_utc")
    generated_at = _parse_utc(summary.get("generated_at_utc"), "source generated_at_utc")
    _require(started_at <= completed_at == generated_at, "source summary timestamps are inconsistent")
    _require(completed_at <= datetime.now(timezone.utc), "source summary is future-dated")

    controls = summary.get("control_inputs")
    if not isinstance(controls, Mapping):
        raise CoordinateEvidenceError("source summary lacks control_inputs")
    expected_controls = {
        "split_manifest": release_path,
        "models_lock": lock_path,
        "pipeline_config": pipeline_path,
        "profile": profile_path,
    }
    for field, expected_path in expected_controls.items():
        raw = controls.get(field)
        if not isinstance(raw, str):
            raise CoordinateEvidenceError(f"source control_inputs.{field} is missing")
        actual = _resolve_inside(root, raw, f"source control_inputs.{field}")
        _require(actual == expected_path, f"source control_inputs.{field} differs from requested current input")
    raw_dataset = controls.get("dataset_root")
    raw_official_profile = controls.get("official_profile")
    if not isinstance(raw_dataset, str) or not isinstance(raw_official_profile, str):
        raise CoordinateEvidenceError("source summary lacks dataset/official-profile controls")
    dataset_root = _resolve_inside(root, raw_dataset, "source dataset_root")
    official_profile_path = _resolve_inside(root, raw_official_profile, "official profile")
    _require(dataset_root.is_dir(), "source dataset_root is missing")
    _require_file(official_profile_path, "official profile")
    _require(summary.get("selection_control_paths_read") == [str(release_path)], "selection control path is not exactly the release manifest")
    _require(summary.get("split_manifest_sha256") == _sha256_file(release_path), "release manifest SHA-256 differs from source summary")

    effective = _apply_accuracy_profile(load_config(pipeline_path), load_config(profile_path))
    effective_hash = config_hash(effective)
    resolved_config = composite_provider_config_from_pipeline(effective)
    resolved_perception = resolved_config.to_dict()
    required_model_tasks = set(required_model_tasks_for_config(resolved_config))
    expected_model_task_plan = {
        "contract": "production_model_task_plan_v1",
        "junction_mode": (
            "learned_model_plus_deterministic_vectorizer"
            if resolved_config.learned_junctions_enabled
            else "deterministic_vectorizer"
        ),
        "required_model_tasks": sorted(required_model_tasks),
        "loaded_model_tasks": sorted(required_model_tasks),
        "learned_junction_model_required": resolved_config.learned_junctions_enabled,
        "learned_junction_model_loaded": resolved_config.learned_junctions_enabled,
        "learned_junction_model_invoked": resolved_config.learned_junctions_enabled,
    }

    lock_sha256 = _sha256_file(lock_path)
    _require(summary.get("model_lock_sha256") == lock_sha256, "source summary is not bound to current models.lock")
    lock_payload = _read_json(lock_path)
    raw_models = lock_payload.get("models") if isinstance(lock_payload, Mapping) else None
    if not isinstance(raw_models, list):
        raise CoordinateEvidenceError("models.lock has no models list")
    model_paths: list[Path] = []
    tasks: list[str] = []
    for index, entry in enumerate(raw_models):
        if not isinstance(entry, Mapping):
            raise CoordinateEvidenceError(f"models.lock entry {index} is not an object")
        task = str(entry.get("task", "")).strip().lower()
        raw_path = entry.get("path")
        if task not in REQUIRED_MODEL_TASKS or not isinstance(raw_path, str):
            raise CoordinateEvidenceError(f"models.lock entry {index} has invalid task/path")
        model_path = (lock_path.parent / raw_path).resolve(strict=False)
        try:
            model_path.relative_to(lock_path.parent)
        except ValueError as exc:
            raise CoordinateEvidenceError(f"models.lock entry {index} escapes models root") from exc
        _require_file(model_path, f"model artifact {task}")
        _require(entry.get("sha256") == _sha256_file(model_path), f"model artifact {task} hash mismatch")
        _require(entry.get("status") == "blessed" and entry.get("blessed") is True, f"model artifact {task} is not blessed")
        tasks.append(task)
        model_paths.append(model_path)
    _require(
        set(tasks) == required_model_tasks
        and len(tasks) == len(required_model_tasks),
        "models.lock task coverage is not exact for the resolved Junction mode",
    )

    _require(summary.get("config_hash") == effective_hash, "source summary is not bound to current production config")
    _require(summary.get("resolved_perception_config") == resolved_perception, "source summary resolved perception config differs")
    _require(summary.get("resolved_perception_config_sha256") == _compact_sha256(resolved_perception), "resolved perception config hash differs")
    _require(
        summary.get("model_task_plan") == expected_model_task_plan,
        "source summary model task plan differs from resolved Junction mode",
    )
    readiness = summary.get("readiness")
    _require(isinstance(readiness, Mapping) and readiness.get("status") == "ready" and readiness.get("ready") is True, "source readiness is not ready")
    for field in (
        "required_learned_tasks",
        "registered_learned_tasks",
        "verified_learned_tasks",
    ):
        _require(
            readiness.get(field) == sorted(required_model_tasks),
            f"source readiness {field} coverage differs",
        )

    raw_manifest = summary.get("prediction_manifest")
    if not isinstance(raw_manifest, str) or not raw_manifest:
        raise CoordinateEvidenceError("source summary lacks prediction_manifest")
    prediction_path = (summary_path.parent / raw_manifest).resolve(strict=False)
    try:
        prediction_path.relative_to(summary_path.parent)
    except ValueError as exc:
        raise CoordinateEvidenceError("prediction manifest escapes source run") from exc
    _require_file(prediction_path, "prediction manifest")
    _require(summary.get("prediction_manifest_sha256") == _sha256_file(prediction_path), "prediction manifest hash differs")
    rows = _read_jsonl(prediction_path)
    _require(summary.get("selected_cases") == len(rows) == summary.get("succeeded"), "source case counts are not exact")
    _require(summary.get("prediction_set_sha256") == _stable_prediction_set_sha256(rows), "prediction-set hash differs")

    release_rows = _read_jsonl(release_path)
    _require(
        len(release_rows) == FORMAL_RELEASE_CASES,
        "canonical release manifest does not contain exactly "
        f"{FORMAL_RELEASE_CASES} rows",
    )
    selected_release = [row for row in release_rows if row.get("split") == summary.get("split")]
    selected_release.sort(key=lambda item: natural_sort_key(item.get("sample_id")))
    expected_case_ids = [str(item.get("sample_id")) for item in selected_release]
    _require(
        len(selected_release) == FORMAL_RELEASE_CASES,
        f"canonical release manifest does not contain exactly {FORMAL_RELEASE_CASES} "
        f"{CANONICAL_RELEASE_SPLIT!r} rows",
    )
    _require(
        all(
            isinstance(item.get("sample_id"), str)
            and bool(str(item.get("sample_id")).strip())
            for item in selected_release
        ),
        "canonical release manifest contains a missing/invalid sample_id",
    )
    _require(
        len(set(expected_case_ids)) == FORMAL_RELEASE_CASES,
        "canonical release manifest does not contain exactly 50 unique case IDs",
    )
    expected_membership: dict[str, list[str]] = defaultdict(list)
    for item in selected_release:
        raw_group = item.get("duplicate_group")
        _require(
            isinstance(raw_group, str) and bool(raw_group.strip()),
            f"release case {item.get('sample_id')!r} lacks duplicate_group",
        )
        expected_membership[raw_group].append(str(item["sample_id"]))
    expected_membership = dict(
        sorted(expected_membership.items(), key=lambda item: natural_sort_key(item[0]))
    )
    for item in selected_release:
        group = str(item["duplicate_group"])
        _require(
            type(item.get("duplicate_group_size")) is int
            and int(item["duplicate_group_size"])
            == len(expected_membership[group]),
            f"release case {item['sample_id']}: duplicate_group_size differs from "
            "canonical group membership",
        )
    _require(
        summary.get("selected_duplicate_groups") == len(expected_membership),
        "source summary selected_duplicate_groups differs from canonical release suite",
    )
    actual_case_ids = [str(row.get("case_id")) for row in rows]
    _require(
        actual_case_ids == expected_case_ids
        and len(set(actual_case_ids)) == FORMAL_RELEASE_CASES,
        "prediction cases do not exactly cover the canonical release50 suite",
    )
    coverage = summary.get("coverage")
    if not isinstance(coverage, Mapping):
        raise CoordinateEvidenceError("source summary lacks coverage")
    membership: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        membership[str(row.get("duplicate_group"))].append(str(row.get("case_id")))
    membership = dict(sorted(membership.items(), key=lambda item: natural_sort_key(item[0])))
    _require(
        membership == expected_membership,
        "prediction duplicate-group membership differs from canonical release suite",
    )
    _require(
        coverage.get("selected_duplicate_groups") == len(expected_membership),
        "coverage selected_duplicate_groups differs from canonical release suite",
    )
    _require(coverage.get("manifest_case_ids_sha256") == _compact_sha256(actual_case_ids), "manifest case-id hash differs")
    _require(coverage.get("expected_case_ids_sha256") == _compact_sha256(actual_case_ids), "expected case-id hash differs")
    _require(coverage.get("expected_duplicate_group_membership_sha256") == _compact_sha256(membership), "duplicate-group membership hash differs")
    _require(coverage.get("exact_case_coverage") is True and coverage.get("exact_duplicate_group_coverage") is True, "source coverage flags are not exact")

    schema = load_official_schema(official_profile_path)
    release_by_id = {str(item.get("sample_id")): item for item in selected_release}
    shared_inputs = {
        str(path.resolve()).casefold()
        for path in [pipeline_path, profile_path, lock_path, official_profile_path, *model_paths]
    }
    input_paths: list[Path] = [
        summary_path,
        prediction_path,
        release_path,
        pipeline_path,
        profile_path,
        lock_path,
        official_profile_path,
        *model_paths,
        *runtime_paths,
    ]
    output_paths: list[Path] = []
    case_manifest: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    seen_outputs: set[str] = set()
    for index, (row, release_row) in enumerate(zip(rows, selected_release)):
        case_id = actual_case_ids[index]
        _require(row.get("status") == "ok" and row.get("error") is None, f"case {case_id}: prediction is not successful")
        _require(row.get("provider_kind") == "production" and row.get("sidecar_access") is False, f"case {case_id}: prediction is not formal image-only production")
        _require(row.get("model_lock_sha256") == lock_sha256 and row.get("config_hash") == effective_hash, f"case {case_id}: lock/config binding differs")
        _require(row.get("inference_run_id") == run_id, f"case {case_id}: inference_run_id differs")
        _require(row.get("duplicate_group") == release_row.get("duplicate_group"), f"case {case_id}: duplicate group differs")
        files = release_row.get("files")
        image_record = files.get("image") if isinstance(files, Mapping) else None
        image_relative = image_record.get("path") if isinstance(image_record, Mapping) else None
        if not isinstance(image_relative, str):
            raise CoordinateEvidenceError(f"case {case_id}: release image path is missing")
        image_path = (dataset_root / image_relative).resolve(strict=False)
        try:
            image_path.relative_to(dataset_root)
        except ValueError as exc:
            raise CoordinateEvidenceError(f"case {case_id}: image escapes dataset_root") from exc
        _require_file(image_path, f"case {case_id} image")
        image_hash = _sha256_file(image_path)
        _require(row.get("image_sha256") == image_record.get("sha256") == image_hash, f"case {case_id}: image hash differs")
        image_key = str(image_path).casefold()
        _require(image_key not in seen_images, f"case {case_id}: image path is reused")
        seen_images.add(image_key)
        raw_inputs = row.get("input_paths_read")
        if not isinstance(raw_inputs, list) or not raw_inputs:
            raise CoordinateEvidenceError(f"case {case_id}: input_paths_read is missing")
        resolved_inputs = [
            _resolve_inside(root, raw, f"case {case_id} input")
            for raw in raw_inputs
            if isinstance(raw, str)
        ]
        _require(len(resolved_inputs) == len(raw_inputs), f"case {case_id}: input path type is invalid")
        _require(resolved_inputs[0] == image_path, f"case {case_id}: first input is not selected image")
        _require({str(path).casefold() for path in resolved_inputs[1:]} == shared_inputs, f"case {case_id}: input allowlist differs")
        _require(len({str(path).casefold() for path in resolved_inputs}) == len(resolved_inputs), f"case {case_id}: input paths contain aliases/duplicates")
        input_paths.extend(resolved_inputs)

        row_outputs: dict[str, Path] = {}
        for field in ("ir_path", "official_path"):
            raw_output = row.get(field)
            if not isinstance(raw_output, str) or not raw_output:
                raise CoordinateEvidenceError(f"case {case_id}: {field} is missing")
            output = (summary_path.parent / raw_output).resolve(strict=False)
            try:
                output.relative_to(summary_path.parent)
            except ValueError as exc:
                raise CoordinateEvidenceError(f"case {case_id}: {field} escapes source run") from exc
            _require_file(output, f"case {case_id} {field}")
            output_key = str(output).casefold()
            _require(output_key not in seen_outputs, f"case {case_id}: output path is reused")
            seen_outputs.add(output_key)
            row_outputs[field] = output
            output_paths.append(output)
        _require(row.get("ir_sha256") == _sha256_file(row_outputs["ir_path"]), f"case {case_id}: IR hash differs")
        persisted_ir = _read_json(row_outputs["ir_path"])
        persisted_metadata = (
            persisted_ir.get("metadata") if isinstance(persisted_ir, Mapping) else None
        )
        _require(
            isinstance(persisted_metadata, Mapping),
            f"case {case_id}: IR metadata is missing",
        )
        _require(
            persisted_metadata.get("model_task_plan") == expected_model_task_plan,
            f"case {case_id}: IR model task plan differs",
        )
        model_versions = persisted_metadata.get("model_versions")
        _require(
            isinstance(model_versions, Mapping)
            and sorted(model_versions) == sorted(required_model_tasks),
            f"case {case_id}: IR model_versions task set differs",
        )
        if not resolved_config.learned_junctions_enabled:
            _require(
                persisted_metadata.get("quarantined_model_versions", {}) in ({}, None),
                f"case {case_id}: deterministic Junction mode retains a learned model",
            )
        official_hash = _sha256_file(row_outputs["official_path"])
        _require(row.get("official_sha256") == row.get("official_output_sha256") == official_hash, f"case {case_id}: official output hash differs")
        case_manifest.append(
            _coordinate_case(
                case_id=case_id,
                image_path=image_path,
                ir_path=row_outputs["ir_path"],
                official_path=row_outputs["official_path"],
                row=row,
                schema=schema,
                resolved_perception=resolved_perception,
            )
        )

    source_input_manifest = _path_manifest(root, input_paths)
    source_output_manifest = _path_manifest(root, output_paths)
    totals = Counter()
    local_module_totals: Counter[str] = Counter()
    for case in case_manifest:
        for field in (
            "component_bbox_points_checked",
            "pin_points_checked",
            "net_edge_points_checked",
            "zero_length_edges_checked",
            "points_checked",
            "ir_geometry_points_range_checked",
            "local_to_global_transform_points_checked",
            "local_to_global_coordinate_evidence_checked",
            "persisted_transform_points_checked",
        ):
            totals[field] += int(case[field])
        raw_module_counts = case.get("local_to_global_module_objects_checked")
        if not isinstance(raw_module_counts, Mapping):
            raise CoordinateEvidenceError(
                f"case {case.get('case_id')} lacks local-to-global module counts"
            )
        for module, value in raw_module_counts.items():
            if (
                module not in {"component", "text", "pin", "segment", "junction"}
                or type(value) is not int
                or value < 0
            ):
                raise CoordinateEvidenceError(
                    f"case {case.get('case_id')} has invalid local-to-global module count"
                )
            local_module_totals[module] += value
    maximum_forward = max(
        (float(case["max_source_to_serialized_error_px"]) for case in case_manifest),
        default=0.0,
    )
    maximum_roundtrip = max(
        (float(case["max_roundtrip_error_px"]) for case in case_manifest), default=0.0
    )
    _require(case_manifest and totals["points_checked"] > 0, "formal coordinate evidence contains no serialized points")
    required_local_modules = ("component", "text", "pin", "segment", "junction")
    missing_local_modules = [
        module for module in required_local_modules if local_module_totals[module] <= 0
    ]
    _require(
        not missing_local_modules,
        "formal coordinate evidence has no bound local-to-global objects for modules: "
        + ", ".join(missing_local_modules),
    )
    source_summary_sha256 = _sha256_file(summary_path)
    prediction_manifest_sha256 = _sha256_file(prediction_path)
    release_manifest_sha256 = _sha256_file(release_path)
    source_run_binding = {
        "contract": RELEASE_SUITE_CONTRACT,
        "release_manifest_path": CANONICAL_RELEASE_MANIFEST.as_posix(),
        "release_manifest_sha256": release_manifest_sha256,
        "split": CANONICAL_RELEASE_SPLIT,
        "cases": FORMAL_RELEASE_CASES,
        "duplicate_groups": len(expected_membership),
        "model_lock_sha256": lock_sha256,
        "config_sha256": effective_hash,
        "case_ids_sha256": _compact_sha256(actual_case_ids),
        "duplicate_group_membership_sha256": _compact_sha256(
            expected_membership
        ),
        "source_summary_path": _relative(root, summary_path),
        "source_summary_sha256": source_summary_sha256,
        "source_inference_run_id": run_id,
        "source_completed_at_utc": summary.get("completed_at_utc"),
        "source_prediction_manifest_sha256": prediction_manifest_sha256,
        "source_prediction_set_sha256": summary.get("prediction_set_sha256"),
    }
    return {
        "model_lock_sha256": lock_sha256,
        "config_sha256": effective_hash,
        "source_inference_run_id": run_id,
        "source_started_at_utc": summary.get("started_at_utc"),
        "source_completed_at_utc": summary.get("completed_at_utc"),
        "source_summary_path": _relative(root, summary_path),
        "source_summary_sha256": source_summary_sha256,
        "source_prediction_manifest_path": _relative(root, prediction_path),
        "source_prediction_manifest_sha256": prediction_manifest_sha256,
        "source_prediction_set_sha256": summary.get("prediction_set_sha256"),
        "source_run_binding": source_run_binding,
        "source_run_binding_sha256": _compact_sha256(source_run_binding),
        "source_inference_runtime_manifest": dict(source_runtime_manifest),
        "source_inference_runtime_manifest_sha256": _compact_sha256(
            source_runtime_manifest
        ),
        "release_manifest_path": _relative(root, release_path),
        "release_manifest_sha256": release_manifest_sha256,
        "release_suite_contract": RELEASE_SUITE_CONTRACT,
        "release_suite_split": CANONICAL_RELEASE_SPLIT,
        "release_suite_cases": FORMAL_RELEASE_CASES,
        "release_suite_duplicate_groups": len(expected_membership),
        "release_suite_duplicate_group_membership_sha256": _compact_sha256(
            expected_membership
        ),
        "pipeline_config_path": _relative(root, pipeline_path),
        "profile_path": _relative(root, profile_path),
        "official_profile_path": _relative(root, official_profile_path),
        "official_profile_sha256": _sha256_file(official_profile_path),
        "cases_checked": len(case_manifest),
        "case_ids_sha256": _compact_sha256(actual_case_ids),
        "points_checked": totals["points_checked"],
        "component_bbox_points_checked": totals["component_bbox_points_checked"],
        "pin_points_checked": totals["pin_points_checked"],
        "net_edge_points_checked": totals["net_edge_points_checked"],
        "zero_length_edges_checked": totals["zero_length_edges_checked"],
        "ir_geometry_points_range_checked": totals["ir_geometry_points_range_checked"],
        "local_to_global_transform_points_checked": totals[
            "local_to_global_transform_points_checked"
        ],
        "local_to_global_coordinate_evidence_checked": totals[
            "local_to_global_coordinate_evidence_checked"
        ],
        "local_to_global_module_objects_checked": {
            module: local_module_totals[module]
            for module in required_local_modules
        },
        "persisted_transform_points_checked": totals[
            "persisted_transform_points_checked"
        ],
        "max_source_to_serialized_error_px": maximum_forward,
        "max_roundtrip_error_px": maximum_roundtrip,
        "coordinate_case_manifest": case_manifest,
        "coordinate_case_manifest_semantic_sha256": _compact_sha256(case_manifest),
        "source_input_manifest": source_input_manifest,
        "source_input_manifest_sha256": _compact_sha256(source_input_manifest),
        "source_output_manifest": source_output_manifest,
        "source_output_manifest_sha256": _compact_sha256(source_output_manifest),
        "runtime_coordinate_sources": runtime_manifest,
        "runtime_coordinate_sources_sha256": _compact_sha256(runtime_manifest),
        "serializer_sha256": _sha256_file(
            root / "src/pcbparse/output/official_serializer.py"
        ),
        "coordinate_lineage_sha256": _sha256_file(root / "src/pcbparse/data/derive.py"),
        "coordinate_auditor_sha256": _sha256_file(
            root / "scripts/generate_current_coordinate_evidence.py"
        ),
    }


def build_and_write_coordinate_evidence(
    *,
    workspace_root: str | Path,
    source_summary: str | Path,
    release_manifest: str | Path,
    current_models_lock: str | Path,
    pipeline_config: str | Path,
    profile: str | Path,
    coordinate_case_manifest: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    root = Path(workspace_root).expanduser().resolve()
    report_path = _resolve_inside(root, output, "coordinate report")
    case_manifest_path = _resolve_inside(
        root, coordinate_case_manifest, "coordinate case manifest"
    )
    if report_path == case_manifest_path:
        raise CoordinateEvidenceError("coordinate report and case manifest paths collide")
    try:
        evidence = recompute_coordinate_evidence(
            workspace_root=root,
            source_summary=source_summary,
            release_manifest=release_manifest,
            current_models_lock=current_models_lock,
            pipeline_config=pipeline_config,
            profile=profile,
        )
        case_payload = b"".join(
            _canonical_bytes(item) for item in evidence["coordinate_case_manifest"]
        )
        _atomic_write(case_manifest_path, case_payload)
        generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        report: dict[str, Any] = {
            "schema_version": 3,
            "kind": "coordinate-roundtrip-release-report",
            "status": "ok",
            "formal_release_evidence": True,
            "evidence_scope": EVIDENCE_SCOPE,
            "generated_at_utc": generated_at,
            "current_lock_inference_executed": True,
            "models_loaded": True,
            "strict_image_only": True,
            "official_coordinate_origin": "bottom_left",
            "official_y_axis": "up",
            "internal_coordinate_origin": "top_left",
            "internal_y_axis": "down",
            "origin_conversion_verified": True,
            **{
                key: value
                for key, value in evidence.items()
                if key != "coordinate_case_manifest"
            },
            "coordinate_case_manifest_path": _relative(root, case_manifest_path),
            "coordinate_case_manifest_sha256": hashlib.sha256(case_payload).hexdigest(),
            "execution": {
                "inference_executed_by_generator": False,
                "models_loaded_by_generator": False,
                "source_run_inference_already_completed": True,
                "models_lock_modified": False,
                "training_started_or_stopped": False,
            },
        }
        report["audit_payload_sha256"] = _compact_sha256(report)
        _atomic_json(report_path, report)
        return report
    except Exception as exc:
        failure = {
            "schema_version": 3,
            "kind": "coordinate-roundtrip-release-report",
            "status": "failed",
            "formal_release_evidence": False,
            "evidence_scope": EVIDENCE_SCOPE,
            "generated_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "current_lock_inference_executed": False,
            "models_loaded": False,
            "strict_image_only": False,
            "origin_conversion_verified": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        _atomic_json(report_path, failure)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fail-closed current-lock coordinate evidence from an already "
            "completed formal production inference run; no inference is executed."
        )
    )
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--source-summary", required=True)
    parser.add_argument("--release-manifest", required=True)
    parser.add_argument("--models-lock", default="models/models.lock.json")
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    parser.add_argument("--profile", default="configs/profiles/competition.yaml")
    parser.add_argument(
        "--coordinate-case-manifest",
        default="artifacts/release/coordinate_current_cases.jsonl",
    )
    parser.add_argument(
        "--output", default="artifacts/release/coordinate_current_report.json"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = build_and_write_coordinate_evidence(
            workspace_root=arguments.workspace_root,
            source_summary=arguments.source_summary,
            release_manifest=arguments.release_manifest,
            current_models_lock=arguments.models_lock,
            pipeline_config=arguments.pipeline_config,
            profile=arguments.profile,
            coordinate_case_manifest=arguments.coordinate_case_manifest,
            output=arguments.output,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "type": type(exc).__name__, "message": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "output": arguments.output,
                "coordinate_case_manifest": arguments.coordinate_case_manifest,
                "cases_checked": report["cases_checked"],
                "points_checked": report["points_checked"],
                "max_source_to_serialized_error_px": report[
                    "max_source_to_serialized_error_px"
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


__all__ = [
    "CANONICAL_RELEASE_MANIFEST",
    "CANONICAL_RELEASE_SPLIT",
    "CoordinateEvidenceError",
    "FORMAL_RELEASE_CASES",
    "RELEASE_SUITE_CONTRACT",
    "RUNTIME_COORDINATE_SOURCES",
    "build_and_write_coordinate_evidence",
    "recompute_coordinate_evidence",
]
