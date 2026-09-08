from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from pcbparse.io import ImageView, load_image
from pcbparse.ir.models import (
    BBox,
    CircuitIR,
    Component,
    Diagnostic,
    Evidence,
    IRState,
    NetLabel,
    Pin,
    Point,
    TextToken,
    WireSegment,
)


_SIDECAR_SUFFIX = "_CVJsonStd.json"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_TEXT_ID = re.compile(r"^T_\d+$")
_STAGE = "perception.dataset_golden"
_UNMARKED_DESIGNATOR = re.compile(r"^([A-Za-z]{1,4})([0-9]{1,3})$")
_UNMARKED_DESIGNATOR_PREFIXES: dict[str, frozenset[str]] = {
    # This fallback is intentionally narrow.  Unmarked strings that merely
    # resemble a part number (for example MAX232) or a value (R10K/C1J) are not
    # accepted as designators.
    "ANT": frozenset({"ANT"}),
    "BATTARY": frozenset({"BT"}),
    "BJT": frozenset({"Q"}),
    "C": frozenset({"C"}),
    "CRYSTAL": frozenset({"X", "Y"}),
    "D": frozenset({"D"}),
    "FUSE": frozenset({"F"}),
    "JUMPER": frozenset({"JP"}),
    "L": frozenset({"L"}),
    "LED": frozenset({"D", "LED"}),
    "M": frozenset({"M"}),
    "MOSFET": frozenset({"Q"}),
    "MOTOR": frozenset({"M"}),
    "R": frozenset({"R"}),
    "SWITCH": frozenset({"SW"}),
    "TESTPOINT": frozenset({"TP"}),
}


class GoldenPolicy(StrEnum):
    """Resolution policy for Dataset300 annotations.

    ``ACTIVE_ONLY`` is a reserved compatibility name and currently has the
    same no-invention behavior as ``STRICT``.  ``DATASET_DEBUG`` is explicitly
    non-submission behavior: it fills serializer-required identifiers with
    visibly prefixed, deterministic surrogates and emits diagnostics.
    """

    STRICT = "strict"
    DATASET_DEBUG = "dataset_debug"
    ACTIVE_ONLY = "active_only"


# A descriptive alias for callers that prefer the longer API name.
GoldenResolutionPolicy = GoldenPolicy


class GoldenProviderError(RuntimeError):
    """Base class for deterministic Dataset300 conversion failures."""


class SidecarNotFoundError(GoldenProviderError):
    """No unique Dataset300 annotation sidecar could be found."""


class GoldenAnnotationError(GoldenProviderError):
    """A Dataset300 sidecar does not satisfy the minimum conversion schema."""


def find_sidecar(input_path: str | Path) -> Path:
    """Locate the unique ``*_CVJsonStd.json`` for an image or sample directory."""

    path = Path(input_path)
    if path.is_dir():
        candidates = sorted(
            candidate.resolve()
            for candidate in path.glob(f"*{_SIDECAR_SUFFIX}")
            if candidate.is_file() and not candidate.name.startswith("._")
        )
    elif path.is_file() and path.name.endswith(_SIDECAR_SUFFIX):
        candidates = [path.resolve()]
    elif path.is_file():
        candidate = path.with_name(f"{path.stem}{_SIDECAR_SUFFIX}")
        candidates = [candidate.resolve()] if candidate.is_file() else []
    else:
        raise SidecarNotFoundError(f"input does not exist: {path}")

    if not candidates:
        raise SidecarNotFoundError(f"no {_SIDECAR_SUFFIX} sidecar found for {path}")
    if len(candidates) != 1:
        rendered = ", ".join(str(candidate) for candidate in candidates)
        raise SidecarNotFoundError(f"multiple Dataset300 sidecars found for {path}: {rendered}")
    return candidates[0]


def _find_image(input_path: Path, sidecar: Path) -> Path:
    if input_path.is_file() and input_path.resolve() != sidecar:
        return input_path.resolve()

    image_stem = sidecar.name[: -len(_SIDECAR_SUFFIX)]
    candidates = sorted(
        candidate.resolve()
        for candidate in sidecar.parent.iterdir()
        if candidate.is_file()
        and candidate.stem == image_stem
        and candidate.suffix.lower() in _IMAGE_SUFFIXES
        and not candidate.name.startswith("._")
    )
    if not candidates:
        raise SidecarNotFoundError(f"no image paired with sidecar {sidecar}")
    if len(candidates) != 1:
        rendered = ", ".join(str(candidate) for candidate in candidates)
        raise SidecarNotFoundError(f"multiple images paired with sidecar {sidecar}: {rendered}")
    return candidates[0]


def _id_key(value: str) -> tuple[str, int, str]:
    prefix, _, suffix = value.rpartition("_")
    if suffix.isdigit():
        return prefix, int(suffix), value
    return value, -1, value


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GoldenAnnotationError(f"{name} must be a JSON object")
    return value


def _finite_number(value: Any, *, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise GoldenAnnotationError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise GoldenAnnotationError(f"{field} must be finite")
    return number


def _evidence(
    evidence_id: str,
    sidecar: Path,
    fragment: str,
    **metadata: Any,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source="dataset",
        view_id="original",
        score=1.0,
        payload_ref=f"{sidecar}#{fragment}",
        metadata=metadata,
    )


def _diagnostic(
    diagnostics: list[Diagnostic],
    *,
    code: str,
    message: str,
    object_ids: Sequence[str] = (),
    severity: str = "WARNING",
) -> None:
    diagnostics.append(
        Diagnostic(
            severity=severity,  # type: ignore[arg-type]
            code=code,
            message=message,
            object_ids=list(object_ids),
            stage=_STAGE,
        )
    )


def _clip_bbox(
    raw: Any,
    *,
    width: int,
    height: int,
    diagnostics: list[Diagnostic],
    object_id: str,
    field: str,
) -> BBox:
    mapping = _require_mapping(raw, field)
    x1 = _finite_number(mapping.get("x1"), field=f"{field}.x1")
    y1 = _finite_number(mapping.get("y1"), field=f"{field}.y1")
    x2 = _finite_number(mapping.get("x2"), field=f"{field}.x2")
    y2 = _finite_number(mapping.get("y2"), field=f"{field}.y2")
    if x2 < x1 or y2 < y1:
        raise GoldenAnnotationError(f"{field} has reversed coordinates for {object_id}")
    clipped = (
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    )
    if clipped != (x1, y1, x2, y2):
        _diagnostic(
            diagnostics,
            code="DATASET_BBOX_CLIPPED",
            message=(
                f"clipped {field} from {(x1, y1, x2, y2)} to {clipped} "
                f"for canvas {width}x{height}"
            ),
            object_ids=[object_id],
        )
    return BBox(x1=clipped[0], y1=clipped[1], x2=clipped[2], y2=clipped[3])


def _clip_point(
    raw: Any,
    *,
    width: int,
    height: int,
    diagnostics: list[Diagnostic],
    object_id: str,
    field: str,
) -> Point:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
        raise GoldenAnnotationError(f"{field} must be a two-number point for {object_id}")
    x = _finite_number(raw[0], field=f"{field}[0]")
    y = _finite_number(raw[1], field=f"{field}[1]")
    clipped = (min(max(x, 0.0), float(width)), min(max(y, 0.0), float(height)))
    if clipped != (x, y):
        _diagnostic(
            diagnostics,
            code="DATASET_POINT_CLIPPED",
            message=f"clipped {field} from {(x, y)} to {clipped} for canvas {width}x{height}",
            object_ids=[object_id],
        )
    return Point(x=clipped[0], y=clipped[1])


def _text_value(reference: Any, texts: Mapping[str, Any]) -> tuple[str | None, str | None]:
    if reference is None:
        return None, None
    if not isinstance(reference, str):
        return None, None
    if reference in texts:
        text = texts[reference]
        if isinstance(text, Mapping) and isinstance(text.get("content"), str):
            value = text["content"].strip()
            return (value or None), reference
        return None, reference
    if _TEXT_ID.fullmatch(reference):
        return None, reference
    value = reference.strip()
    return (value or None), None


def _unique(values: Iterable[str | None]) -> tuple[str | None, list[str]]:
    candidates: list[str] = []
    for value in values:
        if value and value not in candidates:
            candidates.append(value)
    return (candidates[0] if len(candidates) == 1 else None), candidates


def _associated_values(
    component: Mapping[str, Any],
    texts: Mapping[str, Any],
    detail: str,
) -> tuple[str | None, list[str]]:
    associated = component.get("associatedTexts", {})
    if not isinstance(associated, Mapping):
        return None, []
    values: list[str | None] = []
    for text_id in associated:
        text = texts.get(str(text_id))
        if not isinstance(text, Mapping):
            continue
        text_detail = str(text.get("details", "")).casefold()
        if text_detail == detail.casefold():
            content = text.get("content")
            values.append(content.strip() if isinstance(content, str) else None)
    return _unique(values)


def _resolve_component_fields(
    component: Mapping[str, Any],
    texts: Mapping[str, Any],
) -> tuple[str | None, str, str, dict[str, Any]]:
    key, key_candidates = _associated_values(component, texts, "designator")
    value, value_candidates = _associated_values(component, texts, "value")

    search_values: list[str | None] = []
    search_refs: list[str] = []
    for field in ("searchNameByVLM", "searchNameByOCR", "searchNameByLib"):
        resolved, text_id = _text_value(component.get(field), texts)
        search_values.append(resolved)
        if text_id:
            search_refs.append(text_id)
    name, name_candidates = _unique(search_values)
    if not name_candidates:
        name, name_candidates = _associated_values(component, texts, "searchName")

    return (
        key,
        name or "",
        value or "",
        {
            "designator_candidates": key_candidates,
            "name_candidates": name_candidates,
            "value_candidates": value_candidates,
            "search_name_text_ids": search_refs,
            "ambiguous_fields": [
                field
                for field, candidates in (
                    ("designator", key_candidates),
                    ("name", name_candidates),
                    ("value", value_candidates),
                )
                if len(candidates) > 1
            ],
        },
    )


def _unmarked_designator_candidate(
    component: Mapping[str, Any],
    texts: Mapping[str, Any],
) -> str | None:
    """Return one strongly typed, unmarked associated designator candidate.

    Some official annotations attach a real reference designator to a
    component but leave ``TextToken.details`` empty.  Recovery is safe only
    when exactly one distinct associated token has a short ``PREFIX+digits``
    form and that prefix is a conventional designator for the annotated
    component category.  Marked search names/net labels and alphanumeric value
    suffixes are deliberately excluded.
    """

    allowed = _UNMARKED_DESIGNATOR_PREFIXES.get(
        str(component.get("category", "")).upper()
    )
    if not allowed:
        return None
    associated = component.get("associatedTexts", {})
    if not isinstance(associated, Mapping):
        return None
    candidates: set[str] = set()
    for text_id in associated:
        text = texts.get(str(text_id))
        if not isinstance(text, Mapping) or str(text.get("details", "")) != "":
            continue
        content = text.get("content")
        if not isinstance(content, str):
            continue
        normalized = content.strip().upper()
        match = _UNMARKED_DESIGNATOR.fullmatch(normalized)
        if match and match.group(1).upper() in allowed:
            candidates.add(normalized)
    return next(iter(candidates)) if len(candidates) == 1 else None


def _surrogate_value(prefix: str, source_id: str, reserved: set[str]) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", source_id).strip("_") or "UNKNOWN"
    base = f"{prefix}_{normalized}"
    candidate = base
    suffix = 2
    while candidate in reserved:
        candidate = f"{base}_{suffix}"
        suffix += 1
    reserved.add(candidate)
    return candidate


def _record_debug_surrogate(
    evidence: Evidence,
    *,
    field: str,
    reason: str,
    original_value: str | None,
    surrogate_value: str,
) -> None:
    evidence.metadata["debug_surrogate"] = {
        "field": field,
        "reason": reason,
        "original_value": original_value,
        "surrogate_value": surrogate_value,
        "is_ocr_evidence": False,
        "submission_safe": False,
    }


def _apply_debug_surrogates(
    components: list[Component],
    pins: list[Pin],
    diagnostics: list[Diagnostic],
) -> None:
    """Make debug IR identifier fields structurally serializable.

    Surrogates never claim to be OCR/model output.  Genuine values and their
    candidates remain in dataset evidence, and every replacement is surfaced
    by a ``DEBUG.SURROGATE_*`` diagnostic.
    """

    key_groups: dict[str, list[Component]] = defaultdict(list)
    for component in components:
        if component.key:
            key_groups[component.key].append(component)
    collision_keys = {key for key, group in key_groups.items() if len(group) > 1}
    reserved_component_keys = {
        component.key
        for component in components
        if component.key and component.key not in collision_keys
    }
    for component in sorted(components, key=lambda item: _id_key(item.source_object_id or item.internal_id)):
        original = component.key
        incompatible = bool(original and ("," in original or "." in original or "(" in original or ")" in original))
        if original and original not in collision_keys and not incompatible:
            continue
        reason = "format" if incompatible else "collision" if original else "missing"
        source_id = component.source_object_id or component.internal_id
        surrogate = _surrogate_value("DATASET", source_id, reserved_component_keys)
        component.key = surrogate
        component.key_score = 0.0
        if component.evidence:
            _record_debug_surrogate(
                component.evidence[0],
                field="component.key",
                reason=reason,
                original_value=original,
                surrogate_value=surrogate,
            )
        code = (
            "DEBUG.SURROGATE_COMPONENT_COLLISION"
            if reason == "collision"
            else "DEBUG.SURROGATE_COMPONENT_FORMAT"
            if reason == "format"
            else "DEBUG.SURROGATE_COMPONENT_KEY"
        )
        _diagnostic(
            diagnostics,
            code=code,
            message=(
                f"dataset-debug replaced {reason} component key {original!r} with "
                f"non-submission surrogate {surrogate!r}; this is not OCR evidence"
            ),
            object_ids=[component.internal_id],
            severity="INFO",
        )

    pin_number_groups: dict[tuple[str | None, str], list[Pin]] = defaultdict(list)
    for pin in pins:
        if pin.number:
            pin_number_groups[(pin.component_id, pin.number)].append(pin)
    colliding_pins = {
        pin.internal_id
        for group in pin_number_groups.values()
        if len(group) > 1
        for pin in group
    }
    reserved_by_component: dict[str | None, set[str]] = defaultdict(set)
    for pin in pins:
        if pin.number and pin.internal_id not in colliding_pins and not _debug_identifier_incompatible(pin.number):
            reserved_by_component[pin.component_id].add(pin.number)

    for pin in sorted(pins, key=lambda item: _id_key(item.source_object_id or item.internal_id)):
        original = pin.number
        incompatible = bool(original and _debug_identifier_incompatible(original))
        if original and pin.internal_id not in colliding_pins and not incompatible:
            continue
        reason = "format" if incompatible else "collision" if pin.internal_id in colliding_pins else "missing"
        source_id = pin.source_object_id or pin.internal_id
        surrogate = _surrogate_value(
            "DATASET", source_id, reserved_by_component[pin.component_id]
        )
        pin.number = surrogate
        if pin.evidence:
            _record_debug_surrogate(
                pin.evidence[0],
                field="pin.number",
                reason=reason,
                original_value=original,
                surrogate_value=surrogate,
            )
        code = (
            "DEBUG.SURROGATE_PIN_COLLISION"
            if reason == "collision"
            else "DEBUG.SURROGATE_PIN_FORMAT"
            if reason == "format"
            else "DEBUG.SURROGATE_PIN_NUMBER"
        )
        _diagnostic(
            diagnostics,
            code=code,
            message=(
                f"dataset-debug replaced {reason} pin number {original!r} with "
                f"non-submission surrogate {surrogate!r}; this is not OCR evidence"
            ),
            object_ids=[pin.internal_id],
            severity="INFO",
        )


def _debug_identifier_incompatible(value: str) -> bool:
    return any(character in value for character in (".", ",", "(", ")"))


def _resolve_pin_field(
    pin: Mapping[str, Any],
    texts: Mapping[str, Any],
    fields: Sequence[str],
) -> tuple[str | None, list[str], list[str]]:
    values: list[str | None] = []
    text_ids: list[str] = []
    for field in fields:
        value, text_id = _text_value(pin.get(field), texts)
        values.append(value)
        if text_id:
            text_ids.append(text_id)
    resolved, candidates = _unique(values)
    return resolved, candidates, text_ids


def _orientation(tip: Point, base: Point) -> str:
    dx = tip.x - base.x
    dy = tip.y - base.y
    if dx == 0 and dy == 0:
        return "UNKNOWN"
    if abs(dx) >= abs(dy):
        return "R" if dx > 0 else "L"
    return "B" if dy > 0 else "T"


def _raw_point(raw: Any) -> tuple[float, float] | None:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
        return None
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in raw):
        return None
    x, y = float(raw[0]), float(raw[1])
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _compressed_coordinate_pairs(points: Sequence[Any]) -> Iterator[tuple[Any, Any, int]]:
    """Yield direction-run segments while retaining only O(1) dense points."""

    valid = ((index, point) for index, raw in enumerate(points) if (point := _raw_point(raw)) is not None)
    try:
        first_index, first = next(valid)
    except StopIteration:
        return
    start = first
    previous = first
    previous_index = first_index
    direction: tuple[int, int] | None = None
    for index, current in valid:
        if current == previous:
            continue
        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        new_direction = (
            0 if dx == 0 else (1 if dx > 0 else -1),
            0 if dy == 0 else (1 if dy > 0 else -1),
        )
        if direction is None:
            direction = new_direction
        elif new_direction != direction:
            if start != previous:
                yield start, previous, previous_index
            start = previous
            direction = new_direction
        previous = current
        previous_index = index
    if start != previous:
        yield start, previous, previous_index


def _edge_pairs(edge: Mapping[str, Any]) -> Iterator[tuple[str, int, int, Any, Any, str]]:
    """Prefer compact key-point paths and only scan dense coordinates as fallback."""

    paths = edge.get("keyPointPaths", {})
    emitted = False
    if isinstance(paths, Mapping):
        for channel in sorted(paths):
            polylines = paths[channel]
            if not isinstance(polylines, Sequence) or isinstance(polylines, (str, bytes)):
                continue
            for polyline_index, polyline in enumerate(polylines):
                if not isinstance(polyline, Sequence) or isinstance(polyline, (str, bytes)):
                    continue
                for pair_index, (start, end) in enumerate(zip(polyline, polyline[1:])):
                    if _raw_point(start) is None or _raw_point(end) is None:
                        continue
                    emitted = True
                    yield str(channel), polyline_index, pair_index, start, end, "keyPointPaths"
    if emitted:
        return

    coordinates = edge.get("coordinates", [])
    if not isinstance(coordinates, Sequence) or isinstance(coordinates, (str, bytes)):
        return
    for pair_index, (start, end, source_index) in enumerate(
        _compressed_coordinate_pairs(coordinates)
    ):
        yield "wire", 0, pair_index, start, end, f"coordinates:{source_index}"


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if _id_key(left_root) <= _id_key(right_root):
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def _normal_net_name(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _net_type(value: str) -> str:
    normalized = _normal_net_name(value).upper()
    if normalized in {"0", "GND", "GROUND", "AGND", "DGND", "PGND", "VSS"} or normalized.endswith("_GND"):
        return "GROUND"
    if (
        normalized in {"VCC", "VDD", "VEE", "VBAT", "VIN", "VOUT", "AVCC", "DVCC"}
        or re.fullmatch(r"[+-]?\d+(?:\.\d+)?V", normalized)
        or normalized.startswith(("VCC_", "VDD_", "+", "-"))
    ):
        return "POWER"
    return "SIGNAL" if normalized else "UNKNOWN"


class DatasetGoldenProvider:
    """Convert Dataset300 annotations into serializer-independent Circuit-IR."""

    version = "dataset300-golden-v2"

    def __init__(
        self,
        policy: GoldenPolicy | str = GoldenPolicy.STRICT,
    ) -> None:
        try:
            self.policy = GoldenPolicy(policy)
        except ValueError as exc:
            choices = ", ".join(item.value for item in GoldenPolicy)
            raise ValueError(f"unknown golden policy {policy!r}; expected one of {choices}") from exc

    def build_ir(self, input_path: str | Path, ctx: Any = None) -> CircuitIR:
        del ctx  # The golden conversion is context-free and deterministic.
        source_path = Path(input_path)
        sidecar = find_sidecar(source_path)
        image_path = _find_image(source_path, sidecar)
        image_view = load_image(image_path)

        try:
            raw = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GoldenAnnotationError(f"cannot parse Dataset300 sidecar {sidecar}: {exc}") from exc
        annotation = _require_mapping(raw, "annotation")
        canvas = _require_mapping(annotation.get("canvas"), "canvas")
        canvas_width = int(_finite_number(canvas.get("width"), field="canvas.width"))
        canvas_height = int(_finite_number(canvas.get("height"), field="canvas.height"))
        components_raw = _require_mapping(annotation.get("components"), "components")
        pins_raw = _require_mapping(annotation.get("pins"), "pins")
        texts_raw = _require_mapping(annotation.get("texts"), "texts")
        edges_raw = _require_mapping(annotation.get("edges"), "edges")

        diagnostics: list[Diagnostic] = []
        if (canvas_width, canvas_height) != image_view.size:
            _diagnostic(
                diagnostics,
                code="DATASET_CANVAS_MISMATCH",
                message=(
                    f"annotation canvas {canvas_width}x{canvas_height} differs from "
                    f"decoded image {image_view.width}x{image_view.height}; canonical "
                    "coordinates were clipped to the decoded image"
                ),
            )
        width, height = image_view.size

        active_components = {
            str(component_id): _require_mapping(component, f"components.{component_id}")
            for component_id, component in components_raw.items()
            if isinstance(component, Mapping) and component.get("state") == "active"
        }
        active_texts = {
            str(text_id): _require_mapping(text, f"texts.{text_id}")
            for text_id, text in texts_raw.items()
            if isinstance(text, Mapping) and text.get("state") == "active"
        }
        active_edges = {
            str(edge_id): _require_mapping(edge, f"edges.{edge_id}")
            for edge_id, edge in edges_raw.items()
            if isinstance(edge, Mapping) and edge.get("state") == "active"
        }

        component_internal = {
            component_id: f"component:{component_id}" for component_id in active_components
        }
        pin_internal = {str(pin_id): f"pin:{pin_id}" for pin_id in pins_raw}
        text_internal = {text_id: f"text:{text_id}" for text_id in active_texts}

        pin_owners: dict[str, set[str]] = defaultdict(set)
        component_text_owners: dict[str, set[str]] = defaultdict(set)
        for component_id, component in active_components.items():
            nested_pins = component.get("pins", {})
            if isinstance(nested_pins, Mapping):
                for pin_id in nested_pins:
                    pin_owners[str(pin_id)].add(component_id)
            associated = component.get("associatedTexts", {})
            if isinstance(associated, Mapping):
                for text_id in associated:
                    if str(text_id) in active_texts:
                        component_text_owners[str(text_id)].add(component_id)
            for field in ("searchNameByOCR", "searchNameByVLM", "searchNameByLib"):
                reference = component.get(field)
                if isinstance(reference, str) and reference in active_texts:
                    component_text_owners[reference].add(component_id)

        pin_text_owners: dict[str, set[str]] = defaultdict(set)
        for pin_id, value in pins_raw.items():
            pin = _require_mapping(value, f"pins.{pin_id}")
            for field in (
                "pinNumberByOCR",
                "pinNumberByVLM",
                "pinNameByOCR",
                "pinNameByVLM",
            ):
                reference = pin.get(field)
                if isinstance(reference, str) and reference in active_texts:
                    pin_text_owners[reference].add(str(pin_id))

        # A pin can be represented by pin.edgeId, edge.connect, or both.  Keep
        # all active memberships and make conflicts visible rather than choosing
        # one source silently.
        pin_edge_memberships: dict[str, set[str]] = defaultdict(set)
        for pin_id, value in pins_raw.items():
            pin = _require_mapping(value, f"pins.{pin_id}")
            edge_id = pin.get("edgeId")
            if isinstance(edge_id, str) and edge_id in active_edges:
                pin_edge_memberships[str(pin_id)].add(edge_id)
        for edge_id, edge in active_edges.items():
            connections = edge.get("connect", {})
            if not isinstance(connections, Mapping):
                _diagnostic(
                    diagnostics,
                    code="DATASET_EDGE_CONNECT_INVALID",
                    message=f"edge {edge_id} connect is not an object",
                    object_ids=[edge_id],
                )
                continue
            for pin_id in connections:
                raw_pin_id = str(pin_id)
                pin_edge_memberships[raw_pin_id].add(edge_id)
                if raw_pin_id not in pins_raw:
                    _diagnostic(
                        diagnostics,
                        code="DATASET_EDGE_PIN_REFERENCE_MISSING",
                        message=f"active edge {edge_id} references absent top-level pin {raw_pin_id}",
                        object_ids=[edge_id, raw_pin_id],
                    )

        union_find = _UnionFind(active_edges)
        for edge_ids in pin_edge_memberships.values():
            ordered = sorted((edge for edge in edge_ids if edge in active_edges), key=_id_key)
            for edge_id in ordered[1:]:
                union_find.union(ordered[0], edge_id)
        physical_edge_groups: dict[str, list[str]] = defaultdict(list)
        for edge_id in active_edges:
            physical_edge_groups[union_find.find(edge_id)].append(edge_id)
        physical_groups = [
            sorted(edge_ids, key=_id_key) for edge_ids in physical_edge_groups.values()
        ]
        physical_groups.sort(key=lambda group: _id_key(group[0]))
        edge_to_physical: dict[str, str] = {}
        for edge_ids in physical_groups:
            hint_id = f"dataset-physical:{edge_ids[0]}"
            for edge_id in edge_ids:
                edge_to_physical[edge_id] = hint_id

        edge_label_text: dict[str, str] = {}
        edge_label_ref: dict[str, str] = {}
        for edge_id, edge in active_edges.items():
            value, text_id = _text_value(edge.get("netlabel"), active_texts)
            if value:
                edge_label_text[edge_id] = value
            if text_id:
                edge_label_ref[edge_id] = text_id

        physical_label_sets: dict[str, set[str]] = defaultdict(set)
        label_spelling: dict[str, str] = {}
        for edge_id, value in edge_label_text.items():
            normalized = _normal_net_name(value)
            physical_label_sets[edge_to_physical[edge_id]].add(normalized)
            label_spelling.setdefault(normalized, value)
        electrical_key_for_physical: dict[str, str] = {}
        for edge_ids in physical_groups:
            physical_id = edge_to_physical[edge_ids[0]]
            labels = physical_label_sets.get(physical_id, set())
            if len(labels) == 1:
                electrical_key_for_physical[physical_id] = f"label:{next(iter(labels))}"
            else:
                electrical_key_for_physical[physical_id] = f"physical:{physical_id}"

        electrical_groups: dict[str, list[str]] = defaultdict(list)
        for physical_id, key in electrical_key_for_physical.items():
            electrical_groups[key].append(physical_id)
        physical_to_electrical: dict[str, str] = {}
        for key, physical_ids in sorted(electrical_groups.items()):
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
            hint_id = f"dataset-electrical:{digest}"
            for physical_id in physical_ids:
                physical_to_electrical[physical_id] = hint_id

        component_resolutions: dict[
            str, tuple[str | None, str, str, dict[str, Any]]
        ] = {
            component_id: _resolve_component_fields(component, active_texts)
            for component_id, component in active_components.items()
        }
        genuine_keys = {
            key.casefold()
            for key, _name, _value, _resolution in component_resolutions.values()
            if key
        }
        unmarked_candidates = {
            component_id: candidate
            for component_id, component in active_components.items()
            if component_resolutions[component_id][0] is None
            and (candidate := _unmarked_designator_candidate(component, active_texts))
            is not None
        }
        candidate_counts: dict[str, int] = defaultdict(int)
        for candidate in unmarked_candidates.values():
            candidate_counts[candidate.casefold()] += 1
        for component_id, candidate in unmarked_candidates.items():
            key, name, value, resolution = component_resolutions[component_id]
            normalized = candidate.casefold()
            resolution["unmarked_designator_candidate"] = candidate
            if candidate_counts[normalized] == 1 and normalized not in genuine_keys:
                resolution["designator_candidates"].append(candidate)
                resolution["designator_source"] = "associated-unmarked-compatible"
                component_resolutions[component_id] = (candidate, name, value, resolution)
                _diagnostic(
                    diagnostics,
                    code="DATASET_DESIGNATOR_RECOVERED",
                    message=(
                        f"recovered unique category-compatible designator {candidate!r} "
                        "from an active associated text with empty details"
                    ),
                    object_ids=[component_internal[component_id]],
                    severity="INFO",
                )
            else:
                resolution["unmarked_designator_rejected"] = (
                    "candidate-collision"
                    if candidate_counts[normalized] > 1
                    else "marked-designator-collision"
                )

        for component_id, (key, _name, _value, resolution) in component_resolutions.items():
            if key and "designator_source" not in resolution:
                resolution["designator_source"] = "details:designator"

        components: list[Component] = []
        for component_id in sorted(active_components, key=_id_key):
            raw_component = active_components[component_id]
            internal_id = component_internal[component_id]
            bbox_source = "realBBox" if raw_component.get("realBBox") is not None else "location"
            bbox = _clip_bbox(
                raw_component.get(bbox_source),
                width=width,
                height=height,
                diagnostics=diagnostics,
                object_id=internal_id,
                field=f"components.{component_id}.{bbox_source}",
            )
            key, name, value, resolution = component_resolutions[component_id]
            rotation_raw = raw_component.get("rotation")
            rotation = int(rotation_raw) if isinstance(rotation_raw, (int, float)) else None
            evidence = _evidence(
                f"dataset:component:{component_id}",
                sidecar,
                f"/components/{component_id}",
                source_object_id=component_id,
                category_id=raw_component.get("categoryId"),
                bbox_source=bbox_source,
                raw_location=raw_component.get("location"),
                raw_real_bbox=raw_component.get("realBBox"),
                resolution=resolution,
            )
            components.append(
                Component(
                    internal_id=internal_id,
                    key=key,
                    category=str(raw_component.get("category", "Other")),
                    name=name,
                    value=value,
                    bbox=bbox,
                    score=1.0,
                    key_score=1.0 if key else 0.0,
                    rotation=rotation,
                    source_object_id=component_id,
                    evidence=[evidence],
                )
            )

        texts: list[TextToken] = []
        text_bbox: dict[str, BBox] = {}
        for text_id in sorted(active_texts, key=_id_key):
            raw_text = active_texts[text_id]
            internal_id = text_internal[text_id]
            bbox = _clip_bbox(
                raw_text.get("location"),
                width=width,
                height=height,
                diagnostics=diagnostics,
                object_id=internal_id,
                field=f"texts.{text_id}.location",
            )
            text_bbox[text_id] = bbox
            component_owners = sorted(component_text_owners.get(text_id, ()), key=_id_key)
            pin_owners_for_text = sorted(pin_text_owners.get(text_id, ()), key=_id_key)
            content = str(raw_text.get("content", ""))
            texts.append(
                TextToken(
                    internal_id=internal_id,
                    raw_text=content,
                    candidates=[content] if content else [],
                    resolved_text=content or None,
                    category=str(raw_text.get("category", "unknown")),
                    details=str(raw_text.get("details", "")),
                    bbox=bbox,
                    score=1.0,
                    source_object_id=text_id,
                    component_id=(
                        component_internal[component_owners[0]]
                        if len(component_owners) == 1
                        else None
                    ),
                    pin_id=(pin_internal[pin_owners_for_text[0]] if len(pin_owners_for_text) == 1 else None),
                    evidence=[
                        _evidence(
                            f"dataset:text:{text_id}",
                            sidecar,
                            f"/texts/{text_id}",
                            source_object_id=text_id,
                            category_id=raw_text.get("categoryId"),
                            component_owner_candidates=[
                                component_internal[owner] for owner in component_owners
                            ],
                            pin_owner_candidates=[pin_internal[owner] for owner in pin_owners_for_text],
                        )
                    ],
                )
            )

        pins: list[Pin] = []
        for pin_id in sorted((str(value) for value in pins_raw), key=_id_key):
            raw_pin = _require_mapping(pins_raw[pin_id], f"pins.{pin_id}")
            internal_id = pin_internal[pin_id]
            tip = _clip_point(
                raw_pin.get("tip"),
                width=width,
                height=height,
                diagnostics=diagnostics,
                object_id=internal_id,
                field=f"pins.{pin_id}.tip",
            )
            base = _clip_point(
                raw_pin.get("base"),
                width=width,
                height=height,
                diagnostics=diagnostics,
                object_id=internal_id,
                field=f"pins.{pin_id}.base",
            )
            owners = sorted(pin_owners.get(pin_id, ()), key=_id_key)
            component_id = component_internal[owners[0]] if len(owners) == 1 else None
            orphan = len(owners) != 1
            if orphan:
                _diagnostic(
                    diagnostics,
                    code="DATASET_ORPHAN_PIN",
                    message=(
                        f"top-level pin {pin_id} has "
                        f"{len(owners)} active component owners; it was preserved as orphan"
                    ),
                    object_ids=[internal_id],
                    severity="INFO",
                )
            number, number_candidates, number_text_ids = _resolve_pin_field(
                raw_pin,
                active_texts,
                ("pinNumberByVLM", "pinNumberByOCR"),
            )
            name, name_candidates, name_text_ids = _resolve_pin_field(
                raw_pin,
                active_texts,
                ("pinNameByVLM", "pinNameByOCR"),
            )
            edge_ids = sorted(pin_edge_memberships.get(pin_id, ()), key=_id_key)
            physical_hints = sorted(
                {edge_to_physical[edge_id] for edge_id in edge_ids if edge_id in edge_to_physical}
            )
            electrical_hints = sorted(
                {physical_to_electrical[hint] for hint in physical_hints if hint in physical_to_electrical}
            )
            if len(edge_ids) > 1:
                direct_edge = raw_pin.get("edgeId")
                _diagnostic(
                    diagnostics,
                    code="DATASET_PIN_EDGE_MULTIPLE",
                    message=(
                        f"pin {pin_id} participates in active edges {edge_ids}; "
                        f"pin.edgeId={direct_edge!r} was retained as provenance only"
                    ),
                    object_ids=[internal_id, *edge_ids],
                    severity="INFO",
                )
            pins.append(
                Pin(
                    internal_id=internal_id,
                    component_id=component_id,
                    number=number,
                    name=name or "",
                    point=tip,
                    base=base,
                    orientation=_orientation(tip, base),  # type: ignore[arg-type]
                    # Circuit-IR edge_ids are references to WireSegment IDs,
                    # not Dataset300's source-edge IDs.  They are populated
                    # after source edges have been vectorized below.
                    edge_ids=[],
                    score=1.0,
                    source_object_id=pin_id,
                    orphan=orphan,
                    evidence=[
                        _evidence(
                            f"dataset:pin:{pin_id}",
                            sidecar,
                            f"/pins/{pin_id}",
                            source_object_id=pin_id,
                            raw_component_ids=owners,
                            raw_edge_id=raw_pin.get("edgeId"),
                            active_edge_ids=edge_ids,
                            number_candidates=number_candidates,
                            number_text_ids=number_text_ids,
                            name_candidates=name_candidates,
                            name_text_ids=name_text_ids,
                            physical_net_hints=physical_hints,
                            electrical_net_hints=electrical_hints,
                        )
                    ],
                )
            )

        segments: list[WireSegment] = []
        segment_ids_by_edge: dict[str, list[str]] = defaultdict(list)
        for edge_id in sorted(active_edges, key=_id_key):
            raw_edge = active_edges[edge_id]
            produced = 0
            for channel, polyline_index, pair_index, raw_start, raw_end, geometry_source in _edge_pairs(raw_edge):
                segment_id = f"segment:{edge_id}:{channel}:{polyline_index}:{pair_index}"
                start = _clip_point(
                    raw_start,
                    width=width,
                    height=height,
                    diagnostics=diagnostics,
                    object_id=segment_id,
                    field=f"edges.{edge_id}.{geometry_source}.start",
                )
                end = _clip_point(
                    raw_end,
                    width=width,
                    height=height,
                    diagnostics=diagnostics,
                    object_id=segment_id,
                    field=f"edges.{edge_id}.{geometry_source}.end",
                )
                if start == end:
                    continue
                physical_hint = edge_to_physical[edge_id]
                electrical_hint = physical_to_electrical[physical_hint]
                segment_ids_by_edge[edge_id].append(segment_id)
                segments.append(
                    WireSegment(
                        internal_id=segment_id,
                        start=start,
                        end=end,
                        score=1.0,
                        source_edge_id=edge_id,
                        source_polyline_id=f"{edge_id}:{channel}:{polyline_index}",
                        synthetic=False,
                        channel=channel,
                        evidence=[
                            _evidence(
                                f"dataset:{segment_id}",
                                sidecar,
                                f"/edges/{edge_id}",
                                source_edge_id=edge_id,
                                geometry_source=geometry_source.split(":", 1)[0],
                                physical_net_hint=physical_hint,
                                electrical_net_hint=electrical_hint,
                            )
                        ],
                    )
                )
                produced += 1
            if produced == 0 and raw_edge.get("coordinates"):
                _diagnostic(
                    diagnostics,
                    code="DATASET_EDGE_GEOMETRY_EMPTY",
                    message=f"active edge {edge_id} had coordinates but yielded no non-zero segment",
                    object_ids=[edge_id],
                )

        # Restore source edge membership as valid Circuit-IR segment
        # references.  Raw E_* IDs remain available in evidence metadata.
        for pin in pins:
            raw_pin_id = pin.source_object_id
            if raw_pin_id is None:
                continue
            pin.edge_ids = [
                segment_id
                for edge_id in sorted(pin_edge_memberships.get(raw_pin_id, ()), key=_id_key)
                for segment_id in segment_ids_by_edge[edge_id]
            ]

        net_labels: list[NetLabel] = []
        label_ids_by_edge: dict[str, list[str]] = defaultdict(list)
        for edge_id in sorted(edge_label_text, key=_id_key):
            value = edge_label_text[edge_id]
            text_id = edge_label_ref.get(edge_id)
            point: Point | None = text_bbox[text_id].center if text_id in text_bbox else None
            if point is None:
                connections = active_edges[edge_id].get("connect", {})
                if isinstance(connections, Mapping):
                    for connection in connections.values():
                        if isinstance(connection, Mapping) and _raw_point(connection.get("location")):
                            point = _clip_point(
                                connection["location"],
                                width=width,
                                height=height,
                                diagnostics=diagnostics,
                                object_id=f"net-label:{edge_id}",
                                field=f"edges.{edge_id}.connect.location",
                            )
                            break
            label_id = f"net-label:{edge_id}"
            physical_hint = edge_to_physical[edge_id]
            electrical_hint = physical_to_electrical[physical_hint]
            label_ids_by_edge[edge_id].append(label_id)
            net_labels.append(
                NetLabel(
                    internal_id=label_id,
                    instance_text=value,
                    resolved_text=value,
                    net_type=_net_type(value),  # type: ignore[arg-type]
                    point=point,
                    attached_physical_net_id=None,
                    score=1.0,
                    evidence=[
                        _evidence(
                            f"dataset:net-label:{edge_id}",
                            sidecar,
                            f"/edges/{edge_id}/netlabel",
                            source_edge_id=edge_id,
                            source_text_id=text_id,
                            physical_net_hint=physical_hint,
                            electrical_net_hint=electrical_hint,
                        )
                    ],
                )
            )

        physical_hints: list[dict[str, Any]] = []
        for edge_ids in physical_groups:
            hint_id = edge_to_physical[edge_ids[0]]
            source_pin_ids = sorted(
                {
                    pin_id
                    for pin_id, memberships in pin_edge_memberships.items()
                    if any(edge_id in memberships for edge_id in edge_ids)
                },
                key=_id_key,
            )
            labels = sorted(
                {edge_label_text[edge_id] for edge_id in edge_ids if edge_id in edge_label_text}
            )
            physical_hints.append(
                {
                    "hint_id": hint_id,
                    "source_edge_ids": edge_ids,
                    "source_pin_ids": source_pin_ids,
                    "pin_ids": [pin_internal[pin_id] for pin_id in source_pin_ids if pin_id in pin_internal],
                    "missing_pin_ids": [pin_id for pin_id in source_pin_ids if pin_id not in pin_internal],
                    "segment_ids": [
                        segment_id for edge_id in edge_ids for segment_id in segment_ids_by_edge[edge_id]
                    ],
                    "net_label_ids": [
                        label_id for edge_id in edge_ids for label_id in label_ids_by_edge[edge_id]
                    ],
                    "label_candidates": labels,
                }
            )

        electrical_hints: list[dict[str, Any]] = []
        for key, physical_ids in sorted(electrical_groups.items()):
            hint_id = physical_to_electrical[physical_ids[0]]
            label = label_spelling.get(key.removeprefix("label:")) if key.startswith("label:") else None
            relevant = [hint for hint in physical_hints if hint["hint_id"] in physical_ids]
            electrical_hints.append(
                {
                    "hint_id": hint_id,
                    "physical_hint_ids": sorted(physical_ids),
                    "pin_ids": sorted({pin_id for hint in relevant for pin_id in hint["pin_ids"]}),
                    "segment_ids": sorted(
                        {segment_id for hint in relevant for segment_id in hint["segment_ids"]}
                    ),
                    "label": label,
                    "net_type": _net_type(label) if label else "UNKNOWN",
                    "inference": "same-resolved-label" if label else "physical-membership-only",
                }
            )

        if self.policy == GoldenPolicy.DATASET_DEBUG:
            _apply_debug_surrogates(components, pins, diagnostics)
            orphan_pin_ids = {pin.internal_id for pin in pins if pin.orphan or pin.component_id is None}
            if orphan_pin_ids:
                pins = [pin for pin in pins if pin.internal_id not in orphan_pin_ids]
                for count, orphan_id in enumerate(sorted(orphan_pin_ids), start=1):
                    _diagnostic(
                        diagnostics,
                        code="DEBUG.EXCLUDED_ORPHAN_PIN",
                        message=(
                            f"dataset-debug excluded orphan pin {orphan_id!r} from official "
                            "serialization; raw annotation remains in provider diagnostics"
                        ),
                        object_ids=[orphan_id],
                        severity="INFO",
                    )
                for hint in physical_hints:
                    hint["pin_ids"] = [
                        pin_id for pin_id in hint.get("pin_ids", []) if pin_id not in orphan_pin_ids
                    ]
                for hint in electrical_hints:
                    hint["pin_ids"] = [
                        pin_id for pin_id in hint.get("pin_ids", []) if pin_id not in orphan_pin_ids
                    ]

        sample_id = sidecar.parent.name if sidecar.parent.name else image_path.stem
        image_info = image_view.image_info(
            image_id=sample_id,
            source_domain=str(annotation.get("nowCls", "unknown")),
        )
        return CircuitIR(
            state=IRState.PERCEIVED,
            image=image_info,
            transforms=image_view.transform.to_records(),
            components=components,
            pins=pins,
            texts=texts,
            segments=segments,
            junctions=[],
            net_labels=net_labels,
            physical_nets=[],
            electrical_nets=[],
            diagnostics=diagnostics,
            repairs=[],
            metadata={
                "provider": {
                    "name": type(self).__name__,
                    "version": self.version,
                    "policy": self.policy.value,
                    "submission_safe": self.policy != GoldenPolicy.DATASET_DEBUG,
                },
                "dataset": {
                    "sample_id": sample_id,
                    "annotation_path": str(sidecar),
                    "source_class": annotation.get("nowCls", "unknown"),
                    "canvas": {"width": canvas_width, "height": canvas_height},
                    "active_counts": {
                        "components": len(active_components),
                        "texts": len(active_texts),
                        "edges": len(active_edges),
                        "pins": len(pins_raw),
                    },
                },
                "dataset_graph_hints": {
                    "physical_nets": physical_hints,
                    "electrical_nets": electrical_hints,
                    "authoritative": False,
                    "rebuild_required": True,
                },
            },
        )


__all__ = [
    "DatasetGoldenProvider",
    "GoldenPolicy",
    "GoldenResolutionPolicy",
    "GoldenAnnotationError",
    "GoldenProviderError",
    "SidecarNotFoundError",
    "find_sidecar",
]
