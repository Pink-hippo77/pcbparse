from __future__ import annotations

"""Ultralytics YOLO ``ComponentProvider`` adapter.

Ultralytics remains a delayed dependency so the core PCBParse runtime can be
imported without the training environment.  A category sidecar is mandatory:
YOLO labels are contiguous while the official Dataset300 category ids are not.
"""

import hashlib
import json
import math
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

from pcbparse.io import ImageView
from pcbparse.ir.models import BBox, Component, Evidence


class YoloComponentArtifactError(RuntimeError):
    """The YOLO checkpoint or its category binding violates the contract."""


class YoloComponentAdapter:
    """Run a frozen YOLO detector in ``ImageView`` local coordinates."""

    def __init__(
        self,
        model: Any,
        *,
        checkpoint_sha256: str,
        category_table: Sequence[Mapping[str, Any]],
        device: str,
        score_threshold: float,
        nms_iou: float,
        max_detections: int,
        imgsz: int,
        per_class_score_thresholds: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
            raise ValueError("checkpoint_sha256 must be a 64-character digest")
        self._model = model
        self._checkpoint_sha256 = checkpoint_sha256.lower()
        self._device_name = _device(device)
        self._score_threshold = _unit_interval(score_threshold, "score_threshold")
        self._nms_iou = _unit_interval(nms_iou, "nms_iou")
        self._max_detections = _positive_integer(max_detections, "max_detections")
        self._imgsz = _positive_integer(imgsz, "imgsz")
        self._categories = _normalize_category_table(category_table)
        self._per_class_score_thresholds = _normalize_per_class_thresholds(
            per_class_score_thresholds,
            categories=self._categories,
            population_floor=self._score_threshold,
        )
        _verify_model_names(model, self._categories)
        task = getattr(model, "task", "detect")
        if task != "detect":
            raise YoloComponentArtifactError(
                f"YOLO Component checkpoint task must be 'detect', observed {task!r}"
            )
        self._lock = threading.RLock()
        self.version = f"yolo-component/{self._checkpoint_sha256[:12]}"

    @property
    def category_table(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(self._categories[index]) for index in sorted(self._categories))

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256

    @property
    def per_class_score_thresholds(self) -> dict[str, float]:
        return dict(self._per_class_score_thresholds)

    def predict(self, image: ImageView, ctx: Any) -> list[Component]:
        """Return deterministic local ``xyxy`` detections for one image view."""

        if not isinstance(image, ImageView):
            raise TypeError("YoloComponentAdapter.predict expects an ImageView")
        _check_budget(ctx, "component_detector:before")
        with self._lock:
            raw_results = self._model.predict(
                source=image.image.convert("RGB"),
                conf=self._score_threshold,
                iou=self._nms_iou,
                max_det=self._max_detections,
                imgsz=self._imgsz,
                device=self._device_name,
                verbose=False,
                save=False,
                stream=False,
            )
        _check_budget(ctx, "component_detector:after")
        if not isinstance(raw_results, (list, tuple)) or len(raw_results) != 1:
            raise RuntimeError("YOLO Component detector returned invalid result cardinality")
        output = self._decode_result(image, raw_results[0])
        self._record_version(ctx)
        return output

    def predict_many(
        self,
        images: list[ImageView],
        ctx: Any,
    ) -> list[list[Component]]:
        """Run one Ultralytics call for a bounded batch of image views."""

        if not isinstance(images, list) or any(
            not isinstance(image, ImageView) for image in images
        ):
            raise TypeError("YoloComponentAdapter.predict_many expects ImageView list")
        if not images:
            return []
        _check_budget(ctx, "component_detector:before")
        with self._lock:
            raw_results = self._model.predict(
                source=[image.image.convert("RGB") for image in images],
                conf=self._score_threshold,
                iou=self._nms_iou,
                max_det=self._max_detections,
                imgsz=self._imgsz,
                device=self._device_name,
                batch=len(images),
                verbose=False,
                save=False,
                stream=False,
            )
        _check_budget(ctx, "component_detector:after")
        if not isinstance(raw_results, (list, tuple)) or len(raw_results) != len(images):
            raise RuntimeError("YOLO Component batch returned invalid result cardinality")
        output = [
            self._decode_result(image, result)
            for image, result in zip(images, raw_results)
        ]
        self._record_version(ctx)
        return output

    def _decode_result(self, image: ImageView, raw_result: Any) -> list[Component]:
        boxes = getattr(raw_result, "boxes", None)
        if boxes is None:
            return []
        try:
            xyxy = boxes.xyxy.detach().cpu().tolist()
            scores = boxes.conf.detach().cpu().tolist()
            labels = boxes.cls.detach().cpu().tolist()
        except (AttributeError, TypeError) as exc:
            raise RuntimeError("YOLO result lacks xyxy/conf/cls tensors") from exc
        if not (len(xyxy) == len(scores) == len(labels)):
            raise RuntimeError("YOLO xyxy/conf/cls cardinality mismatch")

        candidates: list[tuple[float, int, tuple[float, float, float, float], int]] = []
        for source_index, (raw_box, raw_score, raw_label) in enumerate(
            zip(xyxy, scores, labels)
        ):
            score = float(raw_score)
            label_value = float(raw_label)
            if not math.isfinite(score) or score < self._score_threshold:
                continue
            if not math.isfinite(label_value) or not label_value.is_integer():
                continue
            label = int(label_value)
            if label not in self._categories:
                continue
            category = self._categories[label]
            effective_threshold = self._per_class_score_thresholds.get(
                str(category["name"]), self._score_threshold
            )
            if score < effective_threshold:
                continue
            if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
                continue
            values = tuple(float(value) for value in raw_box)
            if any(not math.isfinite(value) for value in values):
                continue
            x1 = min(float(image.width), max(0.0, values[0]))
            y1 = min(float(image.height), max(0.0, values[1]))
            x2 = min(float(image.width), max(0.0, values[2]))
            y2 = min(float(image.height), max(0.0, values[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            candidates.append((score, label, (x1, y1, x2, y2), source_index))
        candidates.sort(
            key=lambda item: (
                -item[0],
                item[1],
                item[2][1],
                item[2][0],
                item[2][3],
                item[2][2],
                item[3],
            )
        )
        candidates = candidates[: self._max_detections]

        image_identity = image.sha256 or hashlib.sha256(
            image.image.convert("RGB").tobytes()
        ).hexdigest()
        occurrences: dict[str, int] = {}
        output: list[Component] = []
        for score, label, box, source_index in candidates:
            category = self._categories[label]
            effective_threshold = self._per_class_score_thresholds.get(
                str(category["name"]), self._score_threshold
            )
            signature = (
                f"{image_identity}|{image.view_id}|{label}|"
                + "|".join(f"{coordinate:.4f}" for coordinate in box)
            )
            digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:20]
            duplicate_index = occurrences.get(digest, 0)
            occurrences[digest] = duplicate_index + 1
            local_id = (
                f"component-local:{digest}"
                if duplicate_index == 0
                else f"component-local:{digest}:{duplicate_index}"
            )
            evidence_id = (
                f"detector:{self._checkpoint_sha256[:16]}:{digest}:{duplicate_index}"
            )
            bounded_score = max(0.0, min(1.0, score))
            output.append(
                Component(
                    internal_id=local_id,
                    key=None,
                    category=str(category["name"]),
                    name="",
                    value="",
                    bbox=BBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                    score=bounded_score,
                    key_score=0.0,
                    evidence=[
                        Evidence(
                            evidence_id=evidence_id,
                            source="detector",
                            view_id=image.view_id,
                            score=bounded_score,
                            payload_ref=f"sha256:{self._checkpoint_sha256}",
                            metadata={
                                "architecture": "yolo-detect",
                                "official_category_id": category["official_id"],
                                "model_label": label,
                                "coordinate_system": "image_view_local_xyxy",
                                "source_detection_index": source_index,
                                "imgsz": self._imgsz,
                                "per_tile_nms_iou": self._nms_iou,
                                "effective_score_threshold": effective_threshold,
                            },
                        )
                    ],
                )
            )
        return output

    def _record_version(self, ctx: Any) -> None:
        if ctx is not None:
            versions = getattr(ctx, "model_versions", None)
            if isinstance(versions, dict):
                versions["component"] = self.version


def factory(path: str | Path, entry: Mapping[str, Any]) -> YoloComponentAdapter:
    """Production-provider factory with artifact and category-binding gates."""

    if not isinstance(entry, Mapping):
        raise TypeError("YOLO Component model-lock entry must be a mapping")
    task = entry.get("task")
    if task is not None and task != "component":
        raise ValueError(f"YOLO Component adapter cannot load task {task!r}")
    checkpoint_path = Path(path).expanduser().resolve(strict=True)
    checkpoint_sha256 = _sha256(checkpoint_path)
    expected_sha256 = entry.get("sha256")
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or expected_sha256.lower() != checkpoint_sha256:
            raise YoloComponentArtifactError("YOLO Component checkpoint SHA-256 mismatch")

    runtime_contract = _runtime_contract_from_candidate_bundle(
        checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
    )
    ultralytics = _require_ultralytics()
    model = ultralytics.YOLO(str(checkpoint_path))
    category_table = _category_table_from_entry(
        entry,
        checkpoint_path=checkpoint_path,
        runtime_contract=runtime_contract,
    )
    return YoloComponentAdapter(
        model,
        checkpoint_sha256=checkpoint_sha256,
        category_table=category_table,
        device=_resolve_auto_device(_entry_value(entry, "device", default="cpu")),
        score_threshold=_entry_value(entry, "score_threshold", default=0.25),
        nms_iou=_entry_value(
            entry,
            "nms_iou",
            default=(
                runtime_contract.get("nms_iou", 0.55)
                if runtime_contract is not None
                else 0.55
            ),
        ),
        max_detections=_entry_value(entry, "max_detections", default=300),
        imgsz=_entry_value(
            entry,
            "imgsz",
            default=(
                runtime_contract.get("imgsz", 640)
                if runtime_contract is not None
                else 640
            ),
        ),
        per_class_score_thresholds=_entry_value(
            entry,
            "per_class_score_thresholds",
            default=(
                runtime_contract.get("per_class_score_thresholds", {})
                if runtime_contract is not None
                else {}
            ),
        ),
    )


def _category_table_from_entry(
    entry: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    runtime_contract: Mapping[str, Any] | None = None,
) -> Sequence[Mapping[str, Any]]:
    inline = entry.get("category_table")
    raw_path = _entry_value(entry, "category_table_path", default=None)
    if inline is not None and raw_path is not None:
        raise YoloComponentArtifactError(
            "declare either category_table or category_table_path, not both"
        )
    if inline is not None:
        if not isinstance(inline, Sequence) or isinstance(inline, (str, bytes)):
            raise YoloComponentArtifactError("category_table must be an array")
        return inline
    if raw_path is None and runtime_contract is not None:
        rows = runtime_contract.get("classes")
        if not isinstance(rows, list):
            raise YoloComponentArtifactError(
                "YOLO runtime contract lacks classes array"
            )
        return rows
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise YoloComponentArtifactError(
            "YOLO Component entry must bind category_table or category_table_path"
        )
    sidecar_path = Path(raw_path).expanduser()
    if not sidecar_path.is_absolute():
        sidecar_path = checkpoint_path.parent / sidecar_path
    sidecar_path = sidecar_path.resolve(strict=True)
    expected_sidecar_sha = _entry_value(entry, "category_table_sha256", default=None)
    if expected_sidecar_sha is not None:
        if not isinstance(expected_sidecar_sha, str) or _sha256(sidecar_path) != expected_sidecar_sha.lower():
            raise YoloComponentArtifactError("YOLO category sidecar SHA-256 mismatch")
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise YoloComponentArtifactError("invalid YOLO category sidecar JSON") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("classes"), list):
        raise YoloComponentArtifactError("YOLO category sidecar lacks classes array")
    return payload["classes"]


def _runtime_contract_from_candidate_bundle(
    checkpoint_path: Path,
    *,
    checkpoint_sha256: str,
) -> Mapping[str, Any] | None:
    """Load the hash-bound YOLO sidecar carried by a candidate bundle.

    Standard candidate manifests already bind evidence attachments.  Keeping
    the YOLO-only category map and internal inference geometry in one such
    attachment avoids expanding the shared Component inference schema while
    still making every outcome-affecting value tamper evident.
    """

    manifest_path = checkpoint_path.parent / "bundle.manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise YoloComponentArtifactError("invalid YOLO candidate bundle manifest") from exc
    if not isinstance(manifest, Mapping) or manifest.get("task") != "component":
        raise YoloComponentArtifactError("YOLO candidate bundle task is not component")
    artifact_name = manifest.get("artifact_path")
    if not isinstance(artifact_name, str) or not artifact_name.strip():
        raise YoloComponentArtifactError("YOLO candidate bundle lacks artifact_path")
    artifact_path = _bundle_file(
        checkpoint_path.parent, artifact_name, "YOLO candidate artifact"
    )
    if artifact_path != checkpoint_path:
        raise YoloComponentArtifactError(
            "YOLO candidate manifest does not bind the loaded checkpoint"
        )
    if manifest.get("artifact_sha256") != checkpoint_sha256:
        raise YoloComponentArtifactError("YOLO candidate artifact digest mismatch")
    attachments = manifest.get("evidence_attachments")
    if not isinstance(attachments, list):
        raise YoloComponentArtifactError("YOLO candidate bundle lacks attachments")
    matches = [
        item
        for item in attachments
        if isinstance(item, Mapping)
        and item.get("kind") == "yolo_component_runtime_contract_v1"
    ]
    if len(matches) != 1:
        raise YoloComponentArtifactError(
            "YOLO candidate bundle must bind exactly one runtime contract"
        )
    attachment = matches[0]
    raw_attachment_path = attachment.get("path")
    if not isinstance(raw_attachment_path, str):
        raise YoloComponentArtifactError("YOLO runtime contract path is invalid")
    contract_path = _bundle_file(
        checkpoint_path.parent,
        raw_attachment_path,
        "YOLO runtime contract",
    )
    expected_sha256 = attachment.get("sha256")
    if not isinstance(expected_sha256, str) or _sha256(contract_path) != expected_sha256:
        raise YoloComponentArtifactError("YOLO runtime contract SHA-256 mismatch")
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise YoloComponentArtifactError("invalid YOLO runtime contract JSON") from exc
    if not isinstance(contract, Mapping):
        raise YoloComponentArtifactError("YOLO runtime contract must be an object")
    if (
        contract.get("schema_version") != 1
        or contract.get("kind") != "pcbparse_yolo_component_runtime_contract"
        or contract.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise YoloComponentArtifactError("YOLO runtime contract identity mismatch")
    _unit_interval(contract.get("nms_iou"), "runtime contract nms_iou")
    _positive_integer(contract.get("imgsz"), "runtime contract imgsz")
    categories = _normalize_category_table(contract.get("classes"))
    _normalize_per_class_thresholds(
        contract.get("per_class_score_thresholds"),
        categories=categories,
        population_floor=0.0,
    )
    return contract


def _bundle_file(root: Path, raw: str, label: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise YoloComponentArtifactError(f"{label} path must be bundle-relative")
    resolved = (root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise YoloComponentArtifactError(f"{label} escapes candidate bundle") from exc
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise YoloComponentArtifactError(f"{label} is missing or empty")
    return resolved


def _normalize_category_table(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise YoloComponentArtifactError("YOLO category table must be a non-empty array")
    categories: dict[int, dict[str, Any]] = {}
    official_ids: set[int] = set()
    names: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise YoloComponentArtifactError("YOLO category table rows must be objects")
        try:
            label = int(raw.get("yolo_class_id", raw.get("model_label")))
            official_id = int(raw.get("source_category_id", raw.get("official_id")))
            name = str(raw.get("source_category_name", raw.get("name", ""))).strip()
        except (TypeError, ValueError) as exc:
            raise YoloComponentArtifactError("invalid YOLO category row") from exc
        if label < 0 or official_id < 0 or not name:
            raise YoloComponentArtifactError("invalid YOLO category id or name")
        if label in categories or official_id in official_ids or name in names:
            raise YoloComponentArtifactError("duplicate YOLO category id or name")
        categories[label] = {
            "model_label": label,
            "official_id": official_id,
            "name": name,
        }
        official_ids.add(official_id)
        names.add(name)
    if sorted(categories) != list(range(len(categories))):
        raise YoloComponentArtifactError("YOLO labels must be contiguous 0..N-1")
    return categories


def _verify_model_names(model: Any, categories: Mapping[int, Mapping[str, Any]]) -> None:
    raw_names = getattr(model, "names", None)
    if isinstance(raw_names, Mapping):
        names = {int(key): str(value) for key, value in raw_names.items()}
    elif isinstance(raw_names, Sequence) and not isinstance(raw_names, (str, bytes)):
        names = {index: str(value) for index, value in enumerate(raw_names)}
    else:
        raise YoloComponentArtifactError("YOLO checkpoint lacks model names")
    expected = {label: str(row["name"]) for label, row in categories.items()}
    if names != expected:
        raise YoloComponentArtifactError(
            "YOLO checkpoint names do not match the bound category table"
        )


def _normalize_per_class_thresholds(
    raw: Mapping[str, Any] | None,
    *,
    categories: Mapping[int, Mapping[str, Any]],
    population_floor: float,
) -> dict[str, float]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("per_class_score_thresholds must be an object")
    known_names = {str(row["name"]) for row in categories.values()}
    normalized: dict[str, float] = {}
    for raw_name, raw_threshold in raw.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("per-class threshold names must be non-empty strings")
        name = raw_name.strip()
        if name not in known_names:
            raise ValueError(f"unknown per-class threshold category {name!r}")
        threshold = _unit_interval(
            raw_threshold, f"per_class_score_thresholds.{name}"
        )
        if threshold < population_floor:
            raise ValueError(
                f"per_class_score_thresholds.{name} cannot be below score_threshold"
            )
        normalized[name] = threshold
    return dict(sorted(normalized.items()))


def _entry_value(entry: Mapping[str, Any], name: str, *, default: Any) -> Any:
    inference = entry.get("inference")
    if inference is not None and not isinstance(inference, Mapping):
        raise ValueError("YOLO Component inference must be an object")
    nested = inference if isinstance(inference, Mapping) else {}
    if name in entry and name in nested and entry[name] != nested[name]:
        raise ValueError(f"YOLO Component inference.{name} conflicts with top-level {name}")
    if name in entry:
        return entry[name]
    return nested.get(name, default)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_integer(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _unit_interval(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number in [0,1]")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be a finite number in [0,1]")
    return number


def _device(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("device must be a non-empty string")
    return value.strip().lower()


def _resolve_auto_device(value: Any) -> Any:
    """Translate the shared runtime's ``auto`` token for Ultralytics."""

    if not isinstance(value, str) or value.strip().lower() != "auto":
        return value
    try:
        import torch
    except (ImportError, OSError):
        return "cpu"
    return "0" if torch.cuda.is_available() else "cpu"


def _check_budget(ctx: Any, stage: str) -> None:
    if ctx is None:
        return
    budget = getattr(ctx, "time_budget", None)
    check = getattr(budget, "check", None)
    if callable(check):
        check(stage=stage)


def _require_ultralytics() -> Any:
    try:
        import importlib

        return importlib.import_module("ultralytics")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Ultralytics is required for the YOLO Component adapter; "
            "use the project .venv-yolo runtime"
        ) from exc


__all__ = ["YoloComponentAdapter", "YoloComponentArtifactError", "factory"]
