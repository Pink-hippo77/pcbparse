from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pcbparse.ir.models import CircuitIR

from .masked_metrics import aggregate_masked_reports, evaluate_masked_case
from .raw_ground_truth import build_raw_ground_truth


class AccuracyManifestError(ValueError):
    """Raised when a prediction artifact cannot prove image-only inference."""


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    case_id: str
    ir_path: Path
    provider_kind: str
    sidecar_access: bool
    image_sha256: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate image-only Circuit-IR artifacts against masked raw CVJsonStd targets"
    )
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split")
    parser.add_argument("--mask-policy", choices=("raw-v1",), default="raw-v1")
    parser.add_argument("--component-iou", type=float, default=0.5)
    parser.add_argument("--pin-distance-px", type=float, default=5.0)
    parser.add_argument("--wire-tolerance-px", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = evaluate_prediction_manifest(
        args.prediction_manifest,
        dataset_root=args.dataset_root,
        split_manifest=args.split_manifest,
        split=args.split,
        component_iou_threshold=args.component_iou,
        pin_distance_tolerance_px=args.pin_distance_px,
        wire_tolerance_px=args.wire_tolerance_px,
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write(args.output, rendered.encode("utf-8"))
    print(rendered, end="")
    return 0


def evaluate_prediction_manifest(
    manifest_path: str | Path,
    *,
    dataset_root: str | Path,
    split_manifest: str | Path | None = None,
    split: str | None = None,
    component_iou_threshold: float = 0.5,
    pin_distance_tolerance_px: float = 5.0,
    wire_tolerance_px: int = 3,
) -> dict[str, Any]:
    manifest = Path(manifest_path).expanduser().resolve()
    dataset = Path(dataset_root).expanduser().resolve()
    records = read_prediction_manifest(manifest)
    if split is not None and split_manifest is None:
        raise AccuracyManifestError("--split requires --split-manifest")
    if split_manifest is not None:
        allowed = _split_case_ids(Path(split_manifest), split=split)
        actual = {record.case_id for record in records}
        missing = sorted(allowed - actual)
        extra = sorted(actual - allowed)
        if missing or extra:
            raise AccuracyManifestError(
                "prediction manifest must exactly cover the selected split; "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )

    case_reports: list[dict[str, Any]] = []
    for record in records:
        ir = CircuitIR.model_validate_json(record.ir_path.read_text(encoding="utf-8"))
        assert_accuracy_artifact(record, ir)
        annotation_path = _annotation_path(dataset, record.case_id)
        annotation = _strict_json_object(annotation_path)
        ground_truth = build_raw_ground_truth(
            annotation,
            case_id=record.case_id,
            annotation_path=annotation_path,
        )
        report = evaluate_masked_case(
            ir,
            ground_truth,
            component_iou_threshold=component_iou_threshold,
            pin_distance_tolerance_px=pin_distance_tolerance_px,
            wire_tolerance_px=wire_tolerance_px,
        )
        report["prediction"] = {
            "ir_path": str(record.ir_path),
            "provider_kind": record.provider_kind,
            "sidecar_access": record.sidecar_access,
            "image_sha256": ir.image.sha256,
        }
        case_reports.append(report)
    return {
        "schema_version": 1,
        "policy_version": "raw-v1",
        "accuracy_mode": "image-only",
        "prediction_manifest": str(manifest),
        "dataset_root": str(dataset),
        "cases": case_reports,
        "aggregate": aggregate_masked_reports(case_reports),
        "limitations": [
            "raw-v1 is a local masked evaluator, not the unreleased official competition evaluator.",
            "Component matching maximizes deterministic cardinality above an IoU threshold, not AP over confidences.",
            "Wire scoring is symmetric centerline pixel overlap; raw edges without usable geometry are masked.",
            "Graph scoring is pairwise connectivity conditioned on geometrically matched eligible pins.",
            "Ambiguous connected-vs-non-connected crossings are not inferred from skeleton overlap.",
            "sidecar_access=false is a fail-closed manifest contract, not an operating-system access audit; run inference in an image-only sandbox.",
        ],
    }


def read_prediction_manifest(path: str | Path) -> list[PredictionRecord]:
    manifest = Path(path).expanduser().resolve()
    records: list[PredictionRecord] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            raw = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise AccuracyManifestError(f"invalid JSON on manifest line {line_number}: {exc}") from exc
        if not isinstance(raw, dict):
            raise AccuracyManifestError(f"manifest line {line_number} must be an object")
        case_id = raw.get("case_id")
        ir_path = raw.get("ir_path")
        provider = raw.get("provider_kind", raw.get("provider"))
        sidecar_access = raw.get("sidecar_access")
        if not isinstance(case_id, str) or not case_id:
            raise AccuracyManifestError(f"manifest line {line_number} has no case_id")
        if case_id in seen:
            raise AccuracyManifestError(f"duplicate case_id in prediction manifest: {case_id}")
        seen.add(case_id)
        if not isinstance(ir_path, str) or not ir_path:
            raise AccuracyManifestError(f"manifest line {line_number} has no ir_path")
        if not isinstance(provider, str) or not provider:
            raise AccuracyManifestError(f"manifest line {line_number} has no provider_kind")
        if sidecar_access is not False:
            raise AccuracyManifestError(
                f"case {case_id}: accuracy manifest must explicitly declare sidecar_access=false"
            )
        resolved_ir = Path(ir_path).expanduser()
        if not resolved_ir.is_absolute():
            resolved_ir = manifest.parent / resolved_ir
        resolved_ir = resolved_ir.resolve()
        if not resolved_ir.is_file():
            raise AccuracyManifestError(f"case {case_id}: ir_path does not exist: {resolved_ir}")
        image_sha = raw.get("image_sha256", raw.get("input_image_sha256"))
        if image_sha is not None and (not isinstance(image_sha, str) or not image_sha):
            raise AccuracyManifestError(f"case {case_id}: image_sha256 must be a non-empty string")
        record = PredictionRecord(
            case_id=case_id,
            ir_path=resolved_ir,
            provider_kind=provider,
            sidecar_access=False,
            image_sha256=image_sha,
        )
        _reject_provider_name(record.provider_kind, case_id=case_id)
        records.append(record)
    if not records:
        raise AccuracyManifestError("prediction manifest is empty")
    records.sort(key=lambda item: _natural_key(item.case_id))
    return records


def assert_accuracy_artifact(record: PredictionRecord, ir: CircuitIR) -> None:
    _reject_provider_name(record.provider_kind, case_id=record.case_id)
    if record.image_sha256 is not None and record.image_sha256 != ir.image.sha256:
        raise AccuracyManifestError(
            f"case {record.case_id}: manifest/image IR SHA-256 mismatch"
        )
    metadata_provider = ir.metadata.get("provider")
    provider_text = json.dumps(metadata_provider, ensure_ascii=False, sort_keys=True).casefold()
    _reject_provider_name(provider_text, case_id=record.case_id)
    if isinstance(metadata_provider, dict) and metadata_provider.get("submission_safe") is False:
        raise AccuracyManifestError(
            f"case {record.case_id}: provider metadata declares submission_safe=false"
        )
    if "dataset" in ir.metadata or "dataset_graph_hints" in ir.metadata:
        raise AccuracyManifestError(
            f"case {record.case_id}: Circuit-IR contains annotation-derived dataset metadata"
        )
    for collection in (
        ir.components,
        ir.pins,
        ir.texts,
        ir.segments,
        ir.junctions,
        ir.net_labels,
    ):
        for item in collection:
            if any(evidence.source == "dataset" for evidence in item.evidence):
                raise AccuracyManifestError(
                    f"case {record.case_id}: Circuit-IR contains dataset evidence"
                )


def _reject_provider_name(value: str, *, case_id: str) -> None:
    normalized = value.replace("-", "_").replace(" ", "").casefold()
    forbidden = (
        "dataset_golden",
        "dataset_debug",
        "datasetgoldenprovider",
        "goldenpolicy",
    )
    if any(marker in normalized for marker in forbidden):
        raise AccuracyManifestError(
            f"case {case_id}: annotation-backed provider is forbidden in accuracy mode"
        )


def _annotation_path(dataset_root: Path, case_id: str) -> Path:
    case_root = (dataset_root / case_id).resolve()
    try:
        case_root.relative_to(dataset_root)
    except ValueError as exc:
        raise AccuracyManifestError(f"case_id escapes dataset root: {case_id}") from exc
    candidates = sorted(case_root.glob("*_CVJsonStd.json"))
    if len(candidates) != 1:
        raise AccuracyManifestError(
            f"case {case_id}: expected one *_CVJsonStd.json, found {len(candidates)}"
        )
    return candidates[0]


def _split_case_ids(path: Path, *, split: str | None) -> set[str]:
    result: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(value.get("sample_id"), str):
            raise AccuracyManifestError(f"invalid split manifest line {line_number}")
        if split is None or value.get("split") == split:
            result.add(value["sample_id"])
    if not result:
        raise AccuracyManifestError(f"selected split has no cases: {split!r}")
    return result


def _strict_json_object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise AccuracyManifestError(f"annotation root must be an object: {path}")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _natural_key(value: object) -> tuple[tuple[int, object], ...]:
    import re

    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold(), part)
        for part in re.split(r"(\d+)", str(value))
        if part
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AccuracyManifestError",
    "PredictionRecord",
    "assert_accuracy_artifact",
    "evaluate_prediction_manifest",
    "main",
    "read_prediction_manifest",
]
