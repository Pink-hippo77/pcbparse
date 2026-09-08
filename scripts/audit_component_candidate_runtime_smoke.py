from __future__ import annotations

"""Strict CPU production-adapter smoke/replay for the formal Component-v5 candidate.

The audit is deliberately read-only with respect to training outputs and
``models.lock.json``.  It rebuilds the candidate lock entry in memory from the
cryptographically bound bundle, imports the exact production factory, hides
CUDA before torch is imported, and performs two independent checkpoint loads
on one deterministic frozen validation view.

The candidate's runtime options are never changed.  A separate, fixed
high-confidence replay subset compares the formal CUDA prediction evidence to
CPU inference; this is a cross-backend stability check, not a score-threshold
search and not new evaluation evidence.
"""

import argparse
import gc
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


FINAL_STATUS = "candidate_evidence_ready"
REPLAY_SCORE_FLOOR = 0.80
REPLAY_SCORE_BOUNDARY_MARGIN = 0.05
REPLAY_MINIMUM_DETECTIONS = 3
REPLAY_MINIMUM_IOU = 0.99
REPLAY_MAXIMUM_BOX_ABS_ERROR_PX = 0.25
REPLAY_MAXIMUM_SCORE_ABS_ERROR = 0.02
COORDINATE_ROUNDTRIP_TOLERANCE_PX = 1e-8


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Execute the formal Component-v5 candidate twice through the exact "
            "production adapter on CPU and replay one hash-bound val view."
        )
    )
    parser.add_argument("--workspace-root", type=Path, default=root)
    parser.add_argument(
        "--candidate-bundle",
        type=Path,
        default=Path("work/train-component-v5-fullmix/bundle.manifest.json"),
    )
    parser.add_argument(
        "--run-status",
        type=Path,
        default=Path("work/train-component-v5-fullmix/torch_run_status.json"),
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path(
            "configs/train/dataset300_v2_component_v5_fullmix.json"
        ),
    )
    parser.add_argument(
        "--component-index",
        type=Path,
        default=Path(
            "artifacts/training/materialized/dataset300-v2/component-pin/"
            "component_coco.json"
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
        "--models-lock", type=Path, default=Path("models/models.lock.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/audits/"
            "component_v5_candidate_runtime_provider_smoke_replay.json"
        ),
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


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _bbox(value: Any, label: str) -> tuple[float, float, float, float]:
    if isinstance(value, Mapping):
        raw = [value.get(name) for name in ("x1", "y1", "x2", "y2")]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = list(value)
    else:
        raise ValueError(f"{label} must be a bbox")
    if len(raw) != 4:
        raise ValueError(f"{label} must contain four coordinates")
    values = tuple(_finite_number(item, f"{label} coordinate") for item in raw)
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError(f"{label} must have positive area")
    return values


def _validate_prediction_row(row: Mapping[str, Any], label: str) -> None:
    for name in ("sample_id", "view_id"):
        if not isinstance(row.get(name), str) or not row[name]:
            raise ValueError(f"{label} lacks non-empty {name}")
    boxes = row.get("boxes")
    scores = row.get("scores")
    labels = row.get("labels")
    if not all(isinstance(value, list) for value in (boxes, scores, labels)):
        raise ValueError(f"{label} boxes/scores/labels must be arrays")
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError(f"{label} boxes/scores/labels cardinality mismatch")
    for index, box in enumerate(boxes):
        _bbox(box, f"{label}.boxes[{index}]")
    for index, score in enumerate(scores):
        number = _finite_number(score, f"{label}.scores[{index}]")
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"{label}.scores[{index}] is outside [0,1]")
    if any(type(value) is not int or value <= 0 for value in labels):
        raise ValueError(f"{label} labels must be positive integers")


def _eligible_replay_scores(scores: Sequence[Any]) -> bool:
    numeric = [_finite_number(value, "prediction score") for value in scores]
    high = [value for value in numeric if value >= REPLAY_SCORE_FLOOR]
    margin_ok = all(
        abs(value - REPLAY_SCORE_FLOOR) >= REPLAY_SCORE_BOUNDARY_MARGIN
        for value in numeric
    )
    return len(high) >= REPLAY_MINIMUM_DETECTIONS and margin_ok


def _select_fixture(
    predictions_path: Path, component_coco: Mapping[str, Any]
) -> dict[str, Any]:
    images = component_coco.get("images")
    annotations = component_coco.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("Component COCO images/annotations must be arrays")
    image_by_view: dict[str, dict[str, Any]] = {}
    image_positions: dict[str, int] = {}
    for position, value in enumerate(images):
        if not isinstance(value, dict):
            raise ValueError("Component COCO image rows must be objects")
        view_id = value.get("view_id")
        if not isinstance(view_id, str) or not view_id or view_id in image_by_view:
            raise ValueError("Component COCO has missing/duplicate view_id")
        image_by_view[view_id] = value
        image_positions[view_id] = position
    usable_annotations: dict[int, int] = {}
    for value in annotations:
        if not isinstance(value, dict) or type(value.get("image_id")) is not int:
            raise ValueError("Component COCO annotation lacks integer image_id")
        if value.get("ignore") in (1, True) or value.get("iscrowd") in (1, True):
            continue
        usable_annotations[value["image_id"]] = (
            usable_annotations.get(value["image_id"], 0) + 1
        )

    with predictions_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                raise ValueError(
                    f"candidate predictions contain a blank line at {line_number}"
                )
            try:
                row = _loads(
                    raw_line.decode("utf-8"),
                    f"candidate predictions line {line_number}",
                )
            except UnicodeError as exc:
                raise ValueError(
                    f"candidate predictions contain invalid UTF-8 at line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"candidate predictions line {line_number} is not an object"
                )
            _validate_prediction_row(row, f"candidate predictions line {line_number}")
            image = image_by_view.get(str(row["view_id"]))
            if image is None:
                raise ValueError(
                    f"prediction view_id {row['view_id']!r} is absent from Component COCO"
                )
            image_id = image.get("id")
            exact_val_full = (
                image.get("sample_id") == row["sample_id"]
                and image.get("split") == "val"
                and image.get("view_kind") == "full"
                and type(image_id) is int
                and usable_annotations.get(image_id, 0) > 0
            )
            if exact_val_full and _eligible_replay_scores(row["scores"]):
                return {
                    "prediction": row,
                    "prediction_line_number": line_number,
                    "prediction_line_sha256": hashlib.sha256(raw_line).hexdigest(),
                    "image": image,
                    "image_position": image_positions[str(row["view_id"])],
                    "image_canonical_sha256": _canonical_sha256(image),
                    "usable_ground_truth_count": usable_annotations[image_id],
                }
    raise ValueError(
        "candidate predictions have no deterministic val/full replay fixture with "
        f">={REPLAY_MINIMUM_DETECTIONS} detections at score "
        f">={REPLAY_SCORE_FLOOR:.2f} and the required boundary margin"
    )


def _array_fingerprint(tensor: Any) -> dict[str, Any]:
    import numpy as np

    array = tensor.detach().to("cpu").contiguous().numpy()
    raw = array.tobytes(order="C")
    return {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "finite": bool(np.isfinite(array).all()),
        "minimum": float(array.min()) if array.size else None,
        "maximum": float(array.max()) if array.size else None,
    }


class _CapturingDetector:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    def __call__(self, images: Any) -> Any:
        if not isinstance(images, (list, tuple)) or len(images) != 1:
            raise ValueError("Component provider must call detector with one-image list")
        outputs = self.delegate(images)
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
            raise ValueError("Component detector returned non-unit output")
        prediction = outputs[0]
        if not isinstance(prediction, Mapping):
            raise ValueError("Component detector output must be a mapping")
        if set(prediction) != {"boxes", "scores", "labels"}:
            raise ValueError(
                "Component detector output keys differ from boxes/scores/labels"
            )
        output_fingerprints = {
            name: _array_fingerprint(prediction[name])
            for name in ("boxes", "scores", "labels")
        }
        self.calls.append(
            {
                "input": _array_fingerprint(images[0]),
                "outputs": output_fingerprints,
                "outputs_sha256": _canonical_sha256(output_fingerprints),
            }
        )
        return outputs


def _box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (float(left[2]) - float(left[0])) * (
        float(left[3]) - float(left[1])
    )
    right_area = (float(right[2]) - float(right[0])) * (
        float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _compare_replay_rows(
    formal: Sequence[Mapping[str, Any]], runtime: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    formal_high = [
        dict(row) for row in formal if float(row["score"]) >= REPLAY_SCORE_FLOOR
    ]
    runtime_high = [
        dict(row) for row in runtime if float(row["score"]) >= REPLAY_SCORE_FLOOR
    ]
    unmatched = set(range(len(runtime_high)))
    matches: list[dict[str, Any]] = []
    for formal_index, reference in enumerate(formal_high):
        candidates = [
            runtime_index
            for runtime_index in unmatched
            if runtime_high[runtime_index]["category"] == reference["category"]
        ]
        if not candidates:
            matches.append(
                {
                    "formal_index": formal_index,
                    "runtime_index": None,
                    "category": reference["category"],
                    "matched": False,
                    "reason": "no runtime detection of the same category",
                }
            )
            continue
        runtime_index = max(
            candidates,
            key=lambda index: (
                _box_iou(reference["box"], runtime_high[index]["box"]),
                -index,
            ),
        )
        unmatched.remove(runtime_index)
        observed = runtime_high[runtime_index]
        iou = _box_iou(reference["box"], observed["box"])
        box_error = max(
            abs(float(left) - float(right))
            for left, right in zip(reference["box"], observed["box"])
        )
        score_error = abs(float(reference["score"]) - float(observed["score"]))
        passed = (
            iou >= REPLAY_MINIMUM_IOU
            and box_error <= REPLAY_MAXIMUM_BOX_ABS_ERROR_PX
            and score_error <= REPLAY_MAXIMUM_SCORE_ABS_ERROR
        )
        matches.append(
            {
                "formal_index": formal_index,
                "runtime_index": runtime_index,
                "category": reference["category"],
                "formal_score": float(reference["score"]),
                "runtime_score": float(observed["score"]),
                "iou": iou,
                "maximum_box_absolute_error_px": box_error,
                "score_absolute_error": score_error,
                "matched": passed,
            }
        )
    matched_rows = [row for row in matches if row.get("runtime_index") is not None]
    maximum_box_error = max(
        (float(row["maximum_box_absolute_error_px"]) for row in matched_rows),
        default=math.inf if formal_high else 0.0,
    )
    maximum_score_error = max(
        (float(row["score_absolute_error"]) for row in matched_rows),
        default=math.inf if formal_high else 0.0,
    )
    minimum_iou = min(
        (float(row["iou"]) for row in matched_rows),
        default=0.0 if formal_high else 1.0,
    )
    passed = (
        len(formal_high) >= REPLAY_MINIMUM_DETECTIONS
        and len(formal_high) == len(runtime_high)
        and not unmatched
        and len(matched_rows) == len(formal_high)
        and all(bool(row.get("matched")) for row in matches)
    )
    return {
        "passed": passed,
        "formal_detection_count": len(formal_high),
        "runtime_detection_count": len(runtime_high),
        "unmatched_runtime_indices": sorted(unmatched),
        "minimum_iou": minimum_iou,
        "maximum_box_absolute_error_px": maximum_box_error,
        "maximum_score_absolute_error": maximum_score_error,
        "matches": matches,
    }


def _formal_rows(
    prediction: Mapping[str, Any], model_label_to_category: Mapping[int, str]
) -> list[dict[str, Any]]:
    _validate_prediction_row(prediction, "selected formal prediction")
    rows: list[dict[str, Any]] = []
    for index, (box, score, label) in enumerate(
        zip(prediction["boxes"], prediction["scores"], prediction["labels"])
    ):
        if label not in model_label_to_category:
            raise ValueError(
                f"formal prediction label {label} is absent from checkpoint categories"
            )
        rows.append(
            {
                "formal_index": index,
                "category": model_label_to_category[label],
                "score": float(score),
                "box": list(_bbox(box, f"formal prediction box {index}")),
            }
        )
    return rows


def _runtime_rows(components: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "runtime_index": index,
            "category": component.category,
            "score": float(component.score),
            "box": [
                float(component.bbox.x1),
                float(component.bbox.y1),
                float(component.bbox.x2),
                float(component.bbox.y2),
            ],
        }
        for index, component in enumerate(components)
    ]


def _component_coordinate_contract(
    components: Sequence[Any],
    *,
    view: Any,
    image_record: Mapping[str, Any],
    category_by_name: Mapping[str, Mapping[str, Any]],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    from pcbparse.ir.models import Component
    from pcbparse.training.component_pin_materialize import transform_point

    transform = image_record.get("transform")
    if not isinstance(transform, Mapping):
        raise ValueError("Component image record lacks transform")
    forward = transform.get("forward")
    inverse = transform.get("inverse")
    source_size = image_record.get("source_size")
    if (
        not isinstance(source_size, list)
        or len(source_size) != 2
        or any(type(value) is not int or value <= 0 for value in source_size)
    ):
        raise ValueError("Component image record source_size is invalid")
    serialized: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    evidence_ids: set[str] = set()
    all_local = True
    all_roundtrip = True
    all_schema = True
    all_semantic_empty = True
    all_evidence = True
    maximum_roundtrip_error = 0.0
    for component in components:
        payload = component.model_dump(mode="json")
        _canonical_bytes(payload)
        reconstructed = Component.model_validate(payload)
        schema_ok = reconstructed.model_dump(mode="json") == payload
        serialized.append(payload)
        unique_id = component.internal_id not in ids
        ids.add(component.internal_id)
        box = (
            component.bbox.x1,
            component.bbox.y1,
            component.bbox.x2,
            component.bbox.y2,
        )
        local_ok = (
            0.0 <= box[0] < box[2] <= float(view.width)
            and 0.0 <= box[1] < box[3] <= float(view.height)
        )
        source_corners: list[list[float]] = []
        roundtrip_error = 0.0
        for point in ((box[0], box[1]), (box[2], box[3])):
            source_point = transform_point(point, inverse)
            returned = transform_point(source_point, forward)
            error = math.hypot(returned[0] - point[0], returned[1] - point[1])
            roundtrip_error = max(roundtrip_error, error)
            source_corners.append([source_point[0], source_point[1]])
        roundtrip_ok = roundtrip_error <= COORDINATE_ROUNDTRIP_TOLERANCE_PX
        maximum_roundtrip_error = max(maximum_roundtrip_error, roundtrip_error)
        semantic_ok = (
            component.key is None
            and component.name == ""
            and component.value == ""
            and component.key_score == 0.0
        )
        category = category_by_name.get(component.category)
        evidence_ok = category is not None and len(component.evidence) == 1
        for evidence in component.evidence:
            evidence_unique = evidence.evidence_id not in evidence_ids
            evidence_ids.add(evidence.evidence_id)
            evidence_ok = evidence_ok and evidence_unique and (
                evidence.source == "detector"
                and evidence.view_id == view.view_id
                and evidence.payload_ref == f"sha256:{checkpoint_sha256}"
                and evidence.metadata.get("coordinate_system")
                == "image_view_local_xyxy"
                and evidence.metadata.get("official_category_id")
                == category.get("official_id")
                and evidence.metadata.get("model_label")
                == category.get("model_label")
            )
        all_local = all_local and local_ok
        all_roundtrip = all_roundtrip and roundtrip_ok
        all_schema = all_schema and schema_ok
        all_semantic_empty = all_semantic_empty and semantic_ok
        all_evidence = all_evidence and evidence_ok and unique_id
        rows.append(
            {
                "internal_id": component.internal_id,
                "category": component.category,
                "score": float(component.score),
                "local_bbox": list(box),
                "source_bbox_corners": source_corners,
                "source_bbox_intersects_source_canvas": (
                    source_corners[1][0] > 0.0
                    and source_corners[1][1] > 0.0
                    and source_corners[0][0] < source_size[0]
                    and source_corners[0][1] < source_size[1]
                ),
                "local_bbox_in_bounds": local_ok,
                "coordinate_roundtrip_error_px": roundtrip_error,
                "semantic_fields_unassigned": semantic_ok,
                "evidence_contract": evidence_ok,
                "strict_schema_roundtrip": schema_ok,
            }
        )
    encoded = _canonical_bytes(serialized)
    return {
        "component_count": len(serialized),
        "serialized_component_samples": (
            serialized[:3] + serialized[-3:] if len(serialized) > 6 else serialized
        ),
        "serialized_bytes": len(encoded),
        "serialized_sha256": hashlib.sha256(encoded).hexdigest(),
        "coordinate_samples": rows[:3] + rows[-3:] if len(rows) > 6 else rows,
        "all_local_bboxes_in_bounds": all_local,
        "all_coordinate_roundtrips_within_1e_8_px": all_roundtrip,
        "maximum_coordinate_roundtrip_error_px": maximum_roundtrip_error,
        "all_semantic_fields_unassigned": all_semantic_empty,
        "all_evidence_and_ids_match_checkpoint_contract": all_evidence,
        "all_strict_schema_roundtrips": all_schema,
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


def _terminal_status_contract(
    status: Mapping[str, Any], *, expected_training_config_sha256: str
) -> None:
    if (
        status.get("status") != FINAL_STATUS
        or status.get("task") != "component"
        or status.get("registerable") is not True
        or status.get("formal_evidence") is not True
        or status.get("smoke") is not False
        or not isinstance(status.get("finished_at_utc"), str)
        or not status["finished_at_utc"]
        or status.get("gates_passed") is not True
    ):
        raise ValueError("Component-v5 status is not a formal candidate terminal state")
    lineage = status.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("Component-v5 terminal status lacks lineage")
    if (
        lineage.get("task") != "component"
        or lineage.get("formal_evidence") is not True
        or lineage.get("smoke") is not False
        or lineage.get("training_config_sha256")
        != expected_training_config_sha256
    ):
        raise ValueError("Component-v5 terminal lineage differs from training config")


def _run_once(
    *,
    factory: Any,
    entry: Mapping[str, Any],
    model_path: Path,
    view: Any,
    image_record: Mapping[str, Any],
    category_by_name: Mapping[str, Mapping[str, Any]],
    expected_category_table: Sequence[Mapping[str, Any]],
    expected_binding_sha256: str,
    formal_rows: Sequence[Mapping[str, Any]],
    production_contract_verifier: Any,
) -> dict[str, Any]:
    import torch

    provider = factory(model_path, entry)
    production_contract_verifier(provider, "component")
    if provider.__class__.__name__ != "TorchComponentAdapter":
        raise ValueError("candidate factory did not construct TorchComponentAdapter")
    actual_category_table = tuple(provider.category_table)
    if _canonical_bytes(actual_category_table) != _canonical_bytes(
        tuple(expected_category_table)
    ):
        raise ValueError(
            "Component checkpoint category table differs from frozen Component COCO"
        )
    if (
        provider._checkpoint.lineage.get("binding_sha256")
        != expected_binding_sha256
    ):
        raise ValueError(
            "Component checkpoint binding differs from materialization summary"
        )
    delegate = provider._model
    devices = sorted({str(parameter.device) for parameter in delegate.parameters()})
    if devices != ["cpu"] or provider._device_name != "cpu":
        raise ValueError(f"Component provider is not CPU-only: {devices!r}")
    capture = _CapturingDetector(delegate)
    provider._model = capture
    context = SimpleNamespace(model_versions={})
    components = provider.predict(view, context)
    if len(capture.calls) != 1:
        raise ValueError("one Component ImageView must produce exactly one model call")
    coordinate = _component_coordinate_contract(
        components,
        view=view,
        image_record=image_record,
        category_by_name=category_by_name,
        checkpoint_sha256=provider.checkpoint_sha256,
    )
    replay = _compare_replay_rows(formal_rows, _runtime_rows(components))
    result = {
        "provider_class": (
            f"{provider.__class__.__module__}.{provider.__class__.__qualname__}"
        ),
        "provider_version": provider.version,
        "checkpoint_sha256": provider.checkpoint_sha256,
        "checkpoint_epoch": provider._checkpoint.epoch,
        "checkpoint_framework": provider._checkpoint.framework,
        "checkpoint_preprocessing": provider._checkpoint.preprocessing,
        "checkpoint_lineage": provider._checkpoint.lineage,
        "checkpoint_category_table_count": len(actual_category_table),
        "checkpoint_category_table_sha256": _canonical_sha256(
            actual_category_table
        ),
        "parameter_devices": devices,
        "resolved_device": provider._device_name,
        "adapter_config": {
            "score_threshold": provider._score_threshold,
            "max_detections": provider._max_detections,
        },
        "model_call": capture.calls[0],
        "context_model_versions": dict(sorted(context.model_versions.items())),
        "decoded": coordinate,
        "formal_high_confidence_replay": replay,
    }
    del components, capture, delegate, provider
    gc.collect()
    return result


def _atomic_write_new(path: Path, report: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing audit output: {path}")
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
    run_status_path: Path,
    training_config_path: Path,
    component_index_path: Path,
    materialization_summary_path: Path,
    split_manifest_path: Path,
    source_root: Path,
    models_lock_path: Path,
) -> dict[str, Any]:
    # This must precede torch/torchvision and adapter imports.  It affects only
    # this audit process and cannot affect an independently running trainer.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("PYTHONHASHSEED", "0")
    source_path = workspace / "src"
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

    import torch

    from pcbparse.io import ImageView
    from pcbparse.ir.transforms import AffineTransform, TransformChain
    from pcbparse.perception.production_provider import (
        _resolve_model_path,
        _verify_provider_contract,
    )
    from pcbparse.training.artifacts import (
        build_candidate_lock_entry,
        build_inference_contract,
    )
    from pcbparse.training.component_pin_materialize import decode_indexed_view
    from pcbparse.training.config import (
        load_training_config,
        training_config_sha256,
    )
    from pcbparse.training.torch_component_model import normalize_category_table

    if torch.cuda.is_available():
        raise RuntimeError(
            "CUDA remains visible after the CPU-only audit mask; refusing runtime smoke"
        )
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(20260812)
    torch.use_deterministic_algorithms(True)

    config = load_training_config(training_config_path)
    config_sha = training_config_sha256(config)
    task_config = config.tasks["component"]
    expected_inference = build_inference_contract("component", task_config)
    status = _read_object(run_status_path, "Component-v5 run status")
    _terminal_status_contract(
        status, expected_training_config_sha256=config_sha
    )
    candidate_manifest = _read_object(candidate_bundle_path, "candidate bundle")
    candidate_root = candidate_bundle_path.parent
    attachment_paths: dict[str, Path] = {}
    raw_attachments = candidate_manifest.get("evidence_attachments", [])
    if not isinstance(raw_attachments, list):
        raise ValueError("candidate evidence_attachments must be an array")
    for raw_attachment in raw_attachments:
        if not isinstance(raw_attachment, Mapping):
            raise ValueError("candidate evidence attachment must be an object")
        kind = raw_attachment.get("kind")
        if not isinstance(kind, str) or not kind or kind in attachment_paths:
            raise ValueError("candidate evidence attachment kind is missing/duplicated")
        attachment_path = _resolve_within(
            candidate_root,
            Path(str(raw_attachment.get("path", ""))),
            f"candidate evidence attachment {kind!r}",
        )
        if raw_attachment.get("sha256") != _sha256(attachment_path):
            raise ValueError(f"candidate evidence attachment {kind!r} SHA-256 mismatch")
        attachment_paths[kind] = attachment_path
    calibrated_mode = bool(attachment_paths)
    calibrated_contract: dict[str, Any] | None = None
    if calibrated_mode:
        required_attachments = {
            "validation_score_calibration",
            "independent_frozen_test_confirmation",
            "test_capture_population_scorefloor020",
            "formal_parent_status",
            "source_candidate_bundle",
        }
        if set(attachment_paths) != required_attachments:
            raise ValueError(
                "calibrated Component candidate evidence attachment set differs: "
                f"observed={sorted(attachment_paths)!r}"
            )
        if _sha256(attachment_paths["formal_parent_status"]) != _sha256(run_status_path):
            raise ValueError("calibrated candidate formal-parent attachment differs from run status")
        confirmation = _read_object(
            attachment_paths["independent_frozen_test_confirmation"],
            "calibrated Component test confirmation",
        )
        declared_confirmation_payload = confirmation.get("confirmation_payload_sha256")
        confirmation_payload = dict(confirmation)
        confirmation_payload.pop("confirmation_payload_sha256", None)
        if (
            not isinstance(declared_confirmation_payload, str)
            or _canonical_sha256(confirmation_payload) != declared_confirmation_payload
            or confirmation.get("kind")
            != "pcbparse_component_calibrated_frozen_test_confirmation"
            or confirmation.get("status") != "calibrated_test_confirmation_passed"
            or confirmation.get("formal_evidence") is not True
            or confirmation.get("gates_passed") is not True
            or confirmation.get("protected_sources_unchanged") is not True
        ):
            raise ValueError("calibrated Component test confirmation is not final/hash-valid")
        selection = confirmation.get("selection_boundary")
        protocol = confirmation.get("protocol")
        calibration = confirmation.get("calibration")
        confirmation_artifacts = confirmation.get("artifacts")
        if not all(
            isinstance(value, Mapping)
            for value in (selection, protocol, calibration, confirmation_artifacts)
        ):
            raise ValueError("calibrated Component confirmation lacks structured contract")
        if (
            selection.get("split") != "test"
            or selection.get("available_test_samples") != 30
            or selection.get("selected_test_samples") != 30
            or selection.get("train_samples_loaded") != 0
            or selection.get("val_samples_loaded") != 0
            or protocol.get("test_thresholds_searched") is not False
        ):
            raise ValueError("calibrated Component confirmation is not exact independent test evidence")
        selected_threshold = float(protocol.get("selected_score_threshold"))
        if not 0.0 <= selected_threshold <= 1.0:
            raise ValueError("calibrated Component selected threshold is invalid")
        if float(calibration.get("selected_score_threshold")) != selected_threshold:
            raise ValueError("calibration and test-confirmation thresholds differ")
        calibration_audit = calibration.get("audit")
        if (
            not isinstance(calibration_audit, Mapping)
            or calibration_audit.get("sha256")
            != _sha256(attachment_paths["validation_score_calibration"])
        ):
            raise ValueError("calibrated candidate selection attachment differs from confirmation")
        source_candidate = _read_object(
            attachment_paths["source_candidate_bundle"],
            "calibrated source candidate bundle",
        )
        if (
            source_candidate.get("artifact_sha256")
            != candidate_manifest.get("artifact_sha256")
            or source_candidate.get("inference") != expected_inference
        ):
            raise ValueError("calibrated candidate source bundle is not the formal v5/config contract")
        runtime_inference = dict(expected_inference)
        runtime_inference["score_threshold"] = selected_threshold
        confirmation_predictions = confirmation_artifacts.get("predictions")
        if not isinstance(confirmation_predictions, Mapping):
            raise ValueError("calibrated confirmation lacks prediction evidence")
        if (
            candidate_manifest.get("evaluation_predictions_path") is None
            or candidate_manifest.get("evaluation", {}).get("split") != "test"
            or candidate_manifest.get("evaluation", {}).get("predictions_sha256")
            != confirmation_predictions.get("sha256")
        ):
            raise ValueError("calibrated candidate evaluation differs from test confirmation")
        confirmation_metrics = confirmation.get("metrics")
        candidate_metrics = candidate_manifest.get("evaluation", {}).get("metrics")
        if not isinstance(confirmation_metrics, Mapping) or not isinstance(candidate_metrics, Mapping):
            raise ValueError("calibrated candidate/confirmation metrics are missing")
        if any(float(confirmation_metrics.get(name)) != float(value) for name, value in candidate_metrics.items()):
            raise ValueError("calibrated candidate metrics differ from test confirmation")
        calibrated_contract = {
            "selected_score_threshold": selected_threshold,
            "runtime_inference": runtime_inference,
            "confirmation_payload_sha256": declared_confirmation_payload,
            "test_sample_count": 30,
            "attachment_kinds": sorted(attachment_paths),
        }
    predictions_path = _resolve_within(
        candidate_root,
        Path(str(candidate_manifest.get("evaluation_predictions_path"))),
        "candidate evaluation predictions",
    )
    artifact_path = _resolve_within(
        candidate_root,
        Path(str(candidate_manifest.get("artifact_path"))),
        "candidate artifact",
    )
    card_path = _resolve_within(
        candidate_root,
        Path(str(candidate_manifest.get("model_card_path"))),
        "candidate model card",
    )
    status_artifact = status.get("artifact")
    status_predictions = status.get("predictions")
    status_export = status.get("onnx_export")
    if not all(
        isinstance(value, Mapping)
        for value in (status_artifact, status_predictions, status_export)
    ):
        raise ValueError("Component-v5 terminal status lacks artifact evidence")
    export_path = _resolve_within(
        workspace,
        Path(str(status_export.get("path"))),
        "Component-v5 verified export",
    )
    status_predictions_path = _resolve_within(
        workspace,
        Path(str(status_predictions.get("path"))),
        "Component-v5 formal predictions",
    )
    status_paths_match = (
        (
            calibrated_mode
            or Path(str(status_artifact.get("path"))).resolve(strict=False)
            == artifact_path
        )
        and status_artifact.get("sha256") == _sha256(artifact_path)
        and status_predictions.get("sha256") == _sha256(status_predictions_path)
        and status_export.get("sha256") == _sha256(export_path)
    )
    if not calibrated_mode:
        status_paths_match = status_paths_match and status_predictions_path == predictions_path
    if not status_paths_match:
        raise ValueError("Component-v5 status paths/hashes differ from candidate files")

    component_coco = _read_object(component_index_path, "Component COCO")
    materialization = _read_object(
        materialization_summary_path, "component-pin materialization summary"
    )
    index_sha = _sha256(component_index_path)
    expected_index_sha = (
        materialization.get("outputs", {})
        .get("component_coco.json", {})
        .get("sha256")
    )
    if expected_index_sha != index_sha:
        raise ValueError("Component COCO differs from materialization summary")
    split_sha = _sha256(split_manifest_path)
    materialization_split_sha = (
        materialization.get("lineage", {})
        .get("inputs", {})
        .get("split_manifest", {})
        .get("sha256")
    )
    if (
        materialization_split_sha != split_sha
        or candidate_manifest.get("split_manifest_sha256") != split_sha
    ):
        raise ValueError("split-manifest lineage differs across candidate/materialization")

    entry = build_candidate_lock_entry(
        candidate_bundle_path, lock_directory=candidate_root
    )
    entry_before_sha = _canonical_sha256(entry)
    expected_runtime_inference = (
        calibrated_contract["runtime_inference"]
        if calibrated_contract is not None
        else expected_inference
    )
    if (
        entry.get("task") != "component"
        or entry.get("status") != "candidate"
        or entry.get("blessed") is not False
        or entry.get("training_config_sha256") != config_sha
        or entry.get("inference") != expected_runtime_inference
        or candidate_manifest.get("inference") != expected_runtime_inference
    ):
        raise ValueError("candidate lock entry differs from formal v5/config contract")
    candidate_lock_anchor = candidate_root / "models.lock.json"
    resolved_artifact = _resolve_model_path(
        candidate_lock_anchor, str(entry.get("path"))
    )
    if resolved_artifact != artifact_path:
        raise ValueError("in-memory candidate lock entry resolves to wrong artifact")

    categories = normalize_category_table(component_coco.get("categories", []))
    category_by_name = {str(row["name"]): dict(row) for row in categories}
    model_label_to_category = {
        int(row["model_label"]): str(row["name"]) for row in categories
    }
    replay_predictions_path = (
        status_predictions_path if calibrated_mode else predictions_path
    )
    fixture = _select_fixture(replay_predictions_path, component_coco)
    image_record = fixture["image"]
    if image_record.get("binding_sha256") != materialization.get("binding_sha256"):
        raise ValueError("selected Component fixture binding differs from materialization")
    source_image = _resolve_within(
        source_root,
        Path(str(image_record.get("source_image"))),
        "Dataset300 source image",
    )
    if _sha256(source_image) != image_record.get("source_image_sha256"):
        raise ValueError("selected Dataset300 source image hash changed")
    decoded = decode_indexed_view(image_record, source_root)
    output_size = image_record.get("output_size")
    if output_size != [decoded.width, decoded.height]:
        decoded.close()
        raise ValueError("decoded Component fixture size differs from index")
    view = ImageView(
        view_id=str(image_record["view_id"]),
        image=decoded,
        transform=TransformChain(
            [AffineTransform.identity(str(image_record["view_id"]))]
        ),
        source_path=source_image,
        sha256=str(image_record["source_image_sha256"]),
        metadata={"frozen_component_view": True},
    )
    decoded_rgb = decoded.convert("RGB").tobytes()
    decoded_fingerprint = {
        "mode": "RGB",
        "size_width_height": [decoded.width, decoded.height],
        "bytes": len(decoded_rgb),
        "sha256": hashlib.sha256(decoded_rgb).hexdigest(),
    }
    formal = _formal_rows(fixture["prediction"], model_label_to_category)

    protected_paths = {
        "models_lock": models_lock_path,
        "candidate_bundle_manifest": candidate_bundle_path,
        "candidate_model_card": card_path,
        "candidate_model_artifact": artifact_path,
        "candidate_predictions": predictions_path,
        "formal_parent_predictions": status_predictions_path,
        "candidate_verified_export": export_path,
        "candidate_run_status": run_status_path,
        "training_config": training_config_path,
        "component_index": component_index_path,
        "materialization_summary": materialization_summary_path,
        "split_manifest": split_manifest_path,
        "selected_source_image": source_image,
        **{
            f"candidate_attachment_{kind}": path
            for kind, path in attachment_paths.items()
        },
    }
    before = _snapshot(protected_paths, workspace)
    module_name, attribute_name = str(entry["adapter"]).rsplit(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute_name)
    runs = [
        _run_once(
            factory=factory,
            entry=entry,
            model_path=resolved_artifact,
            view=view,
            image_record=image_record,
            category_by_name=category_by_name,
            expected_category_table=categories,
            expected_binding_sha256=str(materialization["binding_sha256"]),
            formal_rows=formal,
            production_contract_verifier=_verify_provider_contract,
        )
        for _ in range(2)
    ]
    view.image.close()
    entry_after_sha = _canonical_sha256(entry)
    after = _snapshot(protected_paths, workspace)

    decoded_hashes = [row["decoded"]["serialized_sha256"] for row in runs]
    input_hashes = [row["model_call"]["input"]["sha256"] for row in runs]
    raw_output_hashes = [row["model_call"]["outputs_sha256"] for row in runs]
    checks = {
        "formal_candidate_terminal_status_verified": True,
        "candidate_bundle_reverified_and_lock_entry_buildable": True,
        "candidate_entry_unblessed_and_not_inserted_into_primary_lock": True,
        "candidate_entry_unmodified_during_factory_load": (
            entry_before_sha == entry_after_sha
        ),
        "hash_bound_runtime_inference_matches_candidate_contract": (
            entry["inference"] == expected_runtime_inference
        ),
        "calibrated_evidence_contract_verified_when_present": (
            calibrated_contract is not None if calibrated_mode else True
        ),
        "actual_production_component_adapter_loaded_both_runs": all(
            row["provider_class"].endswith(".TorchComponentAdapter")
            for row in runs
        ),
        "cpu_only_checkpoint_and_inference_both_runs": all(
            row["resolved_device"] == "cpu"
            and row["parameter_devices"] == ["cpu"]
            for row in runs
        ),
        "checkpoint_full_category_table_and_materialization_binding_both_runs": all(
            row["checkpoint_category_table_count"] == len(categories)
            and row["checkpoint_category_table_sha256"]
            == _canonical_sha256(categories)
            and row["checkpoint_lineage"].get("binding_sha256")
            == materialization["binding_sha256"]
            for row in runs
        ),
        "candidate_device_auto_preserved_both_runs": (
            entry["inference"]["device"] == "auto"
        ),
        "runtime_emitted_nonempty_results_both_runs": all(
            row["decoded"]["component_count"] > 0 for row in runs
        ),
        "runtime_respected_bound_output_cap_both_runs": all(
            row["decoded"]["component_count"]
            <= int(entry["inference"]["max_detections"])
            for row in runs
        ),
        "coordinate_semantic_evidence_and_schema_contracts_both_runs": all(
            bool(row["decoded"][name])
            for row in runs
            for name in (
                "all_local_bboxes_in_bounds",
                "all_coordinate_roundtrips_within_1e_8_px",
                "all_semantic_fields_unassigned",
                "all_evidence_and_ids_match_checkpoint_contract",
                "all_strict_schema_roundtrips",
            )
        ),
        "formal_high_confidence_replay_within_fixed_cross_backend_tolerance_both_runs": all(
            row["formal_high_confidence_replay"]["passed"] for row in runs
        ),
        "input_tensor_bytes_identical_across_independent_loads": (
            len(set(input_hashes)) == 1
        ),
        "raw_model_output_bytes_identical_across_independent_cpu_loads": (
            len(set(raw_output_hashes)) == 1
        ),
        "decoded_component_bytes_identical_across_independent_cpu_loads": (
            len(set(decoded_hashes)) == 1
        ),
        "candidate_training_data_and_models_lock_unchanged": before == after,
    }
    overall = "pass" if all(checks.values()) else "fail"
    script_path = Path(__file__).resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "component_candidate_production_runtime_provider_smoke_replay",
        "status": overall,
        "scope": {
            "candidate_direct_runtime_inference": True,
            "calibrated_runtime_candidate": calibrated_mode,
            "production_factory_entrypoint": True,
            "current_models_lock_inference": False,
            "models_lock_blessing_attempted": False,
            "weights_changed": False,
            "threshold_or_nms_tuned": False,
            "test_split_used": False,
            "boundary": (
                "Real CPU inference through the hash-bound v5 candidate's production "
                "factory. The candidate remains unblessed and absent from the primary "
                "production lock, so this is not five-model current-lock inference."
            ),
        },
        "candidate": {
            "bundle_manifest": {
                "path": _relative(candidate_bundle_path, workspace),
                "sha256": _sha256(candidate_bundle_path),
            },
            "run_status": {
                "path": _relative(run_status_path, workspace),
                "sha256": _sha256(run_status_path),
                "status": status["status"],
                "best_epoch": status.get("best_epoch"),
                "completed_epochs": status.get("completed_epochs"),
                "metrics": status.get("metrics"),
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
            "verified_export": {
                "path": _relative(export_path, workspace),
                "sha256": _sha256(export_path),
                "formal_max_abs_error": status_export.get("max_abs_error"),
                "formal_sample_count": status_export.get("sample_count"),
            },
            "adapter_entrypoint": entry["adapter"],
            "runtime_inference": entry["inference"],
            "calibrated_contract": calibrated_contract,
            "candidate_lock_entry_sha256": entry_before_sha,
            "candidate_lock_entry": entry,
            "candidate_lock_entry_resolution": {
                "synthetic_lock_anchor_not_created": _relative(
                    candidate_lock_anchor, workspace
                ),
                "resolved_artifact_path": _relative(
                    resolved_artifact, workspace
                ),
            },
        },
        "fixture": {
            "selection_policy": (
                "first file-order formal prediction resolving to exact split=val, "
                "view_kind=full, non-empty GT, with at least three detections at "
                "fixed score>=0.80 and no prediction within 0.05 of that replay floor"
            ),
            "replay_constants": {
                "score_floor": REPLAY_SCORE_FLOOR,
                "score_boundary_margin": REPLAY_SCORE_BOUNDARY_MARGIN,
                "minimum_detection_count": REPLAY_MINIMUM_DETECTIONS,
                "minimum_iou": REPLAY_MINIMUM_IOU,
                "maximum_box_absolute_error_px": (
                    REPLAY_MAXIMUM_BOX_ABS_ERROR_PX
                ),
                "maximum_score_absolute_error": REPLAY_MAXIMUM_SCORE_ABS_ERROR,
            },
            "sample_id": image_record["sample_id"],
            "view_id": image_record["view_id"],
            "split": image_record["split"],
            "view_kind": image_record["view_kind"],
            "component_index_image_position_zero_based": fixture["image_position"],
            "component_index_image_canonical_sha256": fixture[
                "image_canonical_sha256"
            ],
            "usable_ground_truth_count": fixture["usable_ground_truth_count"],
            "formal_prediction_count": len(fixture["prediction"]["boxes"]),
            "formal_high_confidence_replay_count": sum(
                float(value) >= REPLAY_SCORE_FLOOR
                for value in fixture["prediction"]["scores"]
            ),
            "prediction_line_number": fixture["prediction_line_number"],
            "prediction_line_sha256": fixture["prediction_line_sha256"],
            "source_image": {
                "path": _relative(source_image, workspace),
                "sha256": _sha256(source_image),
                "source_size_width_height": image_record["source_size"],
            },
            "materialized_view": {
                "output_size_width_height": image_record["output_size"],
                "crop_bbox": image_record["crop_bbox"],
                "resized_size_width_height": image_record["resized_size"],
                "padding_left_top_right_bottom": image_record["padding"],
                "transform": image_record["transform"],
                "decoded_rgb": decoded_fingerprint,
            },
            "component_index": {
                "path": _relative(component_index_path, workspace),
                "sha256": index_sha,
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
        },
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": importlib.import_module("torchvision").__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "independent_checkpoint_loads": len(runs),
            "runs": runs,
        },
        "protected_files_before": before,
        "protected_files_after": after,
        "checks": checks,
        "release_subgates": {
            "candidate_component_runtime_provider_smoke": overall,
            "candidate_component_cpu_repeatability": (
                "pass"
                if checks[
                    "raw_model_output_bytes_identical_across_independent_cpu_loads"
                ]
                and checks[
                    "decoded_component_bytes_identical_across_independent_cpu_loads"
                ]
                else "fail"
            ),
            "candidate_component_formal_to_cpu_replay": (
                "pass"
                if checks[
                    "formal_high_confidence_replay_within_fixed_cross_backend_tolerance_both_runs"
                ]
                else "fail"
            ),
            "current_models_lock_component_inference": (
                "open_candidate_unblessed_not_in_primary_lock"
            ),
            "five_model_current_lock_production_inference": "open_not_in_scope",
            "official_dataset300_component_metric": "open_official_evaluator_required",
        },
        "audit_implementation": {
            "path": _relative(script_path, workspace),
            "sha256": _sha256(script_path),
        },
        "limitations": [
            "This is runtime/replay evidence only; it cannot register, bless, or promote the candidate.",
            "The fixed high-confidence replay subset checks CUDA-to-CPU stability and is not a replacement for full frozen-validation metrics.",
            "The official competition evaluator is unavailable; no official score is claimed.",
            "Only one frozen validation view is decoded; train and test rows are not loaded or scored.",
        ],
    }
    report["audit_payload_sha256"] = _canonical_sha256(report)
    return report


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    workspace = arguments.workspace_root.expanduser().resolve(strict=True)
    resolved = {
        name: _resolve_within(workspace, getattr(arguments, name), name)
        for name in (
            "candidate_bundle",
            "run_status",
            "training_config",
            "component_index",
            "materialization_summary",
            "split_manifest",
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
        run_status_path=resolved["run_status"],
        training_config_path=resolved["training_config"],
        component_index_path=resolved["component_index"],
        materialization_summary_path=resolved["materialization_summary"],
        split_manifest_path=resolved["split_manifest"],
        source_root=source_root,
        models_lock_path=resolved["models_lock"],
    )
    _atomic_write_new(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "file_sha256": _sha256(output),
                "audit_payload_sha256": report["audit_payload_sha256"],
                "status": report["status"],
                "view_id": report["fixture"]["view_id"],
                "component_counts": [
                    row["decoded"]["component_count"]
                    for row in report["runtime"]["runs"]
                ],
                "decoded_sha256s": [
                    row["decoded"]["serialized_sha256"]
                    for row in report["runtime"]["runs"]
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
