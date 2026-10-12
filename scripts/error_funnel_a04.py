"""A04 error funnel + oracle decomposition (real per-object replay, no GT into preds).

Three object-consistent funnels replace the retired mixed-stage funnel:

1. text fields over one batch of visible GT text fields,
2. pins over one batch of evaluable GT pins,
3. nets over one batch of evaluable GT nets.

Stage rows carry object IDs with inclusion/exclusion reasons.  Un-run or
un-annotated measurements stay ``not_measured`` (null counts), never zeros.
Ground truth is read only here, in the independent evaluator; the prediction
manifest must bind image/source/config/model hashes and stay image-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from pcbparse.evaluation.error_funnel import (
    aggregate_error_funnels,
    build_error_funnel,
    not_measured_funnel,
)
from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth
from pcbparse.ir.models import CircuitIR


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json_object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def read_prediction_manifest(path: Path) -> list[dict[str, Any]]:
    """Read the image-only prediction manifest (one JSON object per line)."""

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        for key in ("case_id", "ir_path", "image_sha256", "source_sha256", "config_sha256"):
            if not isinstance(record.get(key), str) or not record.get(key):
                raise ValueError(f"{path}:{line_number} missing binding field {key!r}")
        if record.get("sidecar_access") is not False:
            raise ValueError(
                f"{path}:{line_number} must declare sidecar_access=false (image-only)"
            )
        records.append(record)
    return records


def _case_binding(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "image_sha256": record["image_sha256"],
        "source_sha256": record["source_sha256"],
        "config_sha256": record["config_sha256"],
        "model_lock_sha256": record.get("model_lock_sha256"),
        "status": "bound",
    }


def _verify_image_binding(ir: CircuitIR, record: Mapping[str, Any], case_id: str) -> None:
    actual = ir.image.sha256
    if actual and actual != record["image_sha256"]:
        raise ValueError(
            f"case {case_id}: IR image sha {actual} does not match manifest binding"
        )


def replay_prediction_manifest(
    manifest_path: Path,
    *,
    dataset_root: Path,
    official_output_dir: Path | None = None,
    component_iou: float = 0.5,
    pin_distance_px: float = 5.0,
    wire_tolerance_px: int = 3,
) -> dict[str, Any]:
    records = read_prediction_manifest(manifest_path)
    case_reports: list[dict[str, Any]] = []
    for record in records:
        case_id = str(record["case_id"])
        ir_path = Path(str(record["ir_path"])).expanduser()
        ir = CircuitIR.model_validate_json(ir_path.read_text(encoding="utf-8"))
        _verify_image_binding(ir, record, case_id)
        annotation_path = _annotation_path(dataset_root, case_id)
        ground_truth = build_raw_ground_truth(
            _strict_json_object(annotation_path),
            case_id=case_id,
            annotation_path=annotation_path,
        )
        official_output: dict[str, Any] | None = None
        if official_output_dir is not None:
            output_path = official_output_dir / f"{case_id}.json"
            if output_path.is_file():
                official_output = _strict_json_object(output_path)
        report = build_error_funnel(
            ir,
            ground_truth,
            official_output=official_output,
            component_iou_threshold=component_iou,
            pin_distance_tolerance_px=pin_distance_px,
            wire_tolerance_px=wire_tolerance_px,
            binding=_case_binding(record),
        )
        case_reports.append(report)
    return {
        "schema_version": "2.0",
        "kind": "error_funnel_replay",
        "task_id": "A04",
        "status": "measured" if case_reports else "not_measured",
        "reason": None if case_reports else "empty_prediction_manifest",
        "prediction_manifest": str(manifest_path),
        "dataset_root": str(dataset_root),
        "cases": case_reports,
        "aggregate": aggregate_error_funnels(case_reports),
        "limitations": [
            "raw-v1 is a local masked evaluator, not the unreleased official evaluator",
            "text detection stage is proxied by owner-object matching at IR level",
            "net segments are complete-coverage checked per matched net, crossing GT stays untrusted",
        ],
    }


def _annotation_path(dataset_root: Path, case_id: str) -> Path:
    direct = dataset_root / f"{case_id}.json"
    if direct.is_file():
        return direct
    matches = sorted(dataset_root.rglob(f"{case_id}*.json"))
    annotation_matches = [
        item
        for item in matches
        if "target" not in item.stem and "Pred" not in item.stem
    ]
    if len(annotation_matches) == 1:
        return annotation_matches[0]
    raise FileNotFoundError(
        f"cannot resolve single annotation for case {case_id!r} under {dataset_root}"
    )


def oracle_decomposition(experiments: list[dict[str, Any]]) -> dict[str, Any]:
    """Record non-additive oracle replacements. Never sum deltas as attribution %."""

    cleaned = []
    for exp in experiments:
        computable = exp.get("computable", True)
        note = exp.get("note")
        if exp.get("crossing_dependent") and not exp.get("trusted_crossing_gt"):
            computable = False
            note = "uncomputable: no trusted crossing GT"
        cleaned.append({
            "name": exp.get("name"),
            "retained_predictions": exp.get("retained_predictions"),
            "gt_coverage_complete": exp.get("gt_coverage_complete"),
            "metric": exp.get("metric"),
            "value": exp.get("value") if computable else None,
            "computable": computable,
            "note": note,
        })
    return {
        "schema_version": "2.0",
        "kind": "oracle_decomposition",
        "task_id": "A04",
        "status": "measured" if cleaned else "not_measured",
        "experiments": cleaned,
        "non_additive": True,
        "policy": {
            "oracle_evidence_excluded_from_promotion": True,
            "missing_crossing_support": "mark_uncomputable",
        },
    }


def baseline_e2e(summary: dict[str, Any]) -> dict[str, Any]:
    measured = bool(summary)
    return {
        "schema_version": "2.0",
        "kind": "baseline_e2e",
        "task_id": "A04",
        "status": "measured" if measured else "not_measured",
        "evaluator_id": summary.get("evaluator_id", "raw-v1-masked") if measured else None,
        "split_role": summary.get("split_role", "development_regression") if measured else None,
        "sidecar_access": False,
        "image_count": summary.get("image_count") if measured else None,
        "strict_success": summary.get("strict_success") if measured else None,
        "degraded_nonempty": summary.get("degraded_nonempty") if measured else None,
        "empty_result": summary.get("empty_result") if measured else None,
        "fallback_empty": summary.get("fallback_empty") if measured else None,
        "failed_to_write": summary.get("failed_to_write") if measured else None,
        "metrics": summary.get("metrics") if measured else None,
        "limitations": summary.get("limitations", [
            "raw-v1 is not the official evaluator",
            "public 210 images are not a new holdout",
        ]) if measured else ["no summary supplied: all counts are null, not zero"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A04 error funnel builder")
    parser.add_argument("--prediction-manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--official-output-dir", type=Path)
    parser.add_argument("--oracle-json", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/audits"))
    parser.add_argument("--component-iou", type=float, default=0.5)
    parser.add_argument("--pin-distance-px", type=float, default=5.0)
    parser.add_argument("--wire-tolerance-px", type=int, default=3)
    args = parser.parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.prediction_manifest is not None and args.dataset_root is not None:
        funnel = replay_prediction_manifest(
            args.prediction_manifest,
            dataset_root=args.dataset_root,
            official_output_dir=args.official_output_dir,
            component_iou=args.component_iou,
            pin_distance_px=args.pin_distance_px,
            wire_tolerance_px=args.wire_tolerance_px,
        )
    else:
        funnel = not_measured_funnel("prediction_manifest_or_dataset_root_not_supplied")
    (out / "error_funnel.json").write_text(
        json.dumps(funnel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    experiments = (
        json.loads(args.oracle_json.read_text(encoding="utf-8"))
        if args.oracle_json
        else []
    )
    if isinstance(experiments, dict):
        experiments = experiments.get("experiments", [])
    oracle = oracle_decomposition(experiments)
    (out / "oracle_decomposition.json").write_text(
        json.dumps(oracle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = (
        json.loads(args.summary_json.read_text(encoding="utf-8"))
        if args.summary_json
        else {}
    )
    base = baseline_e2e(summary)
    (out / "baseline_e2e.json").write_text(
        json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "error_funnel": str(out / "error_funnel.json"),
        "oracle_decomposition": str(out / "oracle_decomposition.json"),
        "baseline_e2e": str(out / "baseline_e2e.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
