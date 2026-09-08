from __future__ import annotations

"""Confirm a validation-selected Component score threshold on frozen test data.

The threshold sweep is deliberately kept selection-only.  This program binds
that immutable audit to the same v5 checkpoint, captures the predeclared
score>=0.20 tiled population on the untouched Dataset300 test split, and then
evaluates the already-selected threshold without searching test labels.

No model lock is read or modified.  A passing result is independent evidence
for a later candidate repackage; it is not a blessing operation.
"""

import argparse
import copy
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import evaluate_component_tiled_validation as tiled  # noqa: E402

from pcbparse.perception.composite_provider import (  # noqa: E402
    _class_aware_nms,
    _restore_components,
)
from pcbparse.perception.torch_component_adapter import (  # noqa: E402
    TorchComponentAdapter,
)
from pcbparse.training.torch_component_model import load_checkpoint  # noqa: E402


CALIBRATION_KIND = "pcbparse_component_stitched_validation_score_threshold_sweep"
CALIBRATION_STATUS = "validation_calibration_audit_complete"
CONFIRMATION_KIND = "pcbparse_component_calibrated_frozen_test_confirmation"
TEST_SPLIT = "test"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Confirm one validation-selected Component score threshold on the "
            "independent frozen Dataset300 test split."
        )
    )
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/train/dataset300_v2_component_v5_fullmix.json"),
    )
    parser.add_argument(
        "--parent-status",
        type=Path,
        default=Path("work/train-component-v5-fullmix/torch_run_status.json"),
    )
    parser.add_argument(
        "--component-coco",
        type=Path,
        default=Path(
            "artifacts/training/materialized/dataset300-v2/"
            "component-pin/component_coco.json"
        ),
    )
    parser.add_argument(
        "--materialization-summary",
        type=Path,
        default=Path(
            "artifacts/training/materialized/dataset300-v2/"
            "component-pin/summary.json"
        ),
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=Path("configs/pipeline.component512-experiment.yaml"),
    )
    parser.add_argument(
        "--thresholds-config",
        type=Path,
        default=Path("configs/thresholds.yaml"),
    )
    parser.add_argument(
        "--calibration-audit",
        type=Path,
        default=Path(
            "artifacts/audits/component_v4_v5_stitched_threshold_sweep.json"
        ),
    )
    parser.add_argument("--candidate-label", default="v5")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work/confirm-component-v5-calibrated-test"),
    )
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _source_entry(path: Path, workspace: Path) -> dict[str, Any]:
    return {
        "path": tiled._relative(path, workspace),
        "sha256": tiled._sha256(path),
    }


def _verify_source_entry(
    workspace: Path, raw: Any, label: str
) -> tuple[Path, str]:
    entry = tiled._mapping(raw, label)
    path = tiled._resolve_within(
        workspace,
        tiled._nonempty_string(entry.get("path"), f"{label}.path"),
        label,
    )
    expected = tiled._nonempty_string(entry.get("sha256"), f"{label}.sha256")
    actual = tiled._sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path, actual


def _calibration_contract(
    *,
    workspace: Path,
    audit_path: Path,
    candidate_label: str,
    parent_status_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    audit = tiled._read_object(audit_path, "Component calibration audit")
    if audit.get("kind") != CALIBRATION_KIND or audit.get("status") != CALIBRATION_STATUS:
        raise ValueError("calibration audit kind/status is not final")
    if (
        audit.get("validation_calibration_only") is not True
        or audit.get("formal_evidence") is not False
        or audit.get("promotion_eligible") is not False
        or audit.get("protected_files_unchanged") is not True
    ):
        raise ValueError("calibration audit boundary is not selection-only and immutable")
    declared_payload = tiled._nonempty_string(
        audit.get("audit_payload_sha256"), "calibration audit payload SHA-256"
    )
    payload = copy.deepcopy(audit)
    payload.pop("audit_payload_sha256", None)
    if tiled._canonical_sha256(payload) != declared_payload:
        raise ValueError("calibration audit canonical payload SHA-256 mismatch")
    boundary = tiled._mapping(audit.get("selection_boundary"), "selection_boundary")
    if (
        boundary.get("split") != "val"
        or boundary.get("threshold_search_performed_on_validation") is not True
        or boundary.get("threshold_search_performed_on_test") is not False
    ):
        raise ValueError("calibration audit does not prove validation-only selection")
    execution = tiled._mapping(audit.get("execution_boundary"), "execution_boundary")
    if execution.get("test_rows_used") != 0 or execution.get("test_split_used") is not False:
        raise ValueError("calibration audit consumed the independent test split")

    candidates = tiled._mapping(audit.get("candidate_evidence"), "candidate_evidence")
    candidate = tiled._mapping(candidates.get(candidate_label), f"candidate {candidate_label}")
    audit_parent_path, audit_parent_sha = _verify_source_entry(
        workspace, candidate.get("formal_parent_status"), "calibration parent status"
    )
    if audit_parent_path != parent_status_path or audit_parent_sha != tiled._sha256(parent_status_path):
        raise ValueError("calibration audit is not bound to the supplied formal parent")
    audit_checkpoint_path, audit_checkpoint_sha = _verify_source_entry(
        workspace,
        candidate.get("checkpoint_bytes_verified_not_loaded"),
        "calibration checkpoint",
    )
    if audit_checkpoint_path != checkpoint_path or audit_checkpoint_sha != tiled._sha256(checkpoint_path):
        raise ValueError("calibration audit is not bound to the supplied checkpoint")
    _verify_source_entry(
        workspace, candidate.get("diagnostic_status"), "calibration tiled status"
    )
    _verify_source_entry(
        workspace, candidate.get("stitched_predictions"), "calibration predictions"
    )

    selected = tiled._mapping(
        candidate.get("best_validation_calibration"), "best validation calibration"
    )
    threshold = tiled._finite(selected.get("score_threshold"), "selected score threshold")
    if not tiled.EXPECTED_SCORE_THRESHOLD <= threshold <= 1.0:
        raise ValueError("selected threshold is outside the captured prediction population")
    if selected.get("score_is_owned_prediction_boundary") is not True:
        raise ValueError("selected threshold is not an observed candidate-owned score")
    comparison = tiled._mapping(
        audit.get("candidate_specific_best_comparison"),
        "candidate_specific_best_comparison",
    )
    thresholds = tiled._mapping(comparison.get("thresholds"), "comparison thresholds")
    if tiled._finite(thresholds.get(candidate_label), "comparison selected threshold") != threshold:
        raise ValueError("candidate threshold differs across calibration evidence")
    return {
        "audit": _source_entry(audit_path, workspace),
        "audit_payload_sha256": declared_payload,
        "candidate_label": candidate_label,
        "selected_score_threshold": threshold,
        "selection_split": "val",
        "selection_metrics": dict(
            tiled._mapping(selected.get("metrics"), "selection metrics")
        ),
        "captured_population_floor": tiled.EXPECTED_SCORE_THRESHOLD,
    }


def _frozen_test_samples(
    *,
    component_coco: Mapping[str, Any],
    source_root: Path,
    binding_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Read only full views and labels owned by the frozen test split."""

    info = tiled._mapping(component_coco.get("info"), "Component COCO info")
    if info.get("binding_sha256") != binding_sha256:
        raise ValueError("Component COCO/materialization binding mismatch")
    categories: list[dict[str, Any]] = []
    category_by_id: dict[int, dict[str, Any]] = {}
    for raw in tiled._sequence(component_coco.get("categories"), "COCO categories"):
        category = tiled._mapping(raw, "COCO category")
        category_id = tiled._positive_integer(category.get("id"), "COCO category id")
        if category_id in category_by_id:
            raise ValueError(f"duplicate COCO category id {category_id}")
        row = {
            "index_id": category_id,
            "official_id": int(str(category.get("source_category_id"))),
            "name": tiled._nonempty_string(category.get("name"), "COCO category name"),
        }
        category_by_id[category_id] = row
        categories.append(row)
    categories.sort(key=lambda row: (row["official_id"], row["name"]))

    split_view_counts: dict[str, int] = {}
    selected: list[dict[str, Any]] = []
    for raw in tiled._sequence(component_coco.get("images"), "COCO images"):
        image = tiled._mapping(raw, "COCO image")
        split = tiled._nonempty_string(image.get("split"), "COCO image split")
        view_kind = tiled._nonempty_string(image.get("view_kind"), "COCO view kind")
        key = f"{split}:{view_kind}"
        split_view_counts[key] = split_view_counts.get(key, 0) + 1
        if split != TEST_SPLIT or view_kind != "full":
            continue
        if image.get("binding_sha256") != binding_sha256:
            raise ValueError("test image binding differs from materialization")
        selected.append(dict(image))
    selected.sort(key=lambda item: str(item.get("sample_id")))
    sample_ids = [
        tiled._nonempty_string(item.get("sample_id"), "test sample_id")
        for item in selected
    ]
    if not selected or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("frozen test full-view population is empty or duplicated")

    selected_by_image_id: dict[int, dict[str, Any]] = {}
    for image in selected:
        image_id = tiled._positive_integer(image.get("id"), "test image id")
        selected_by_image_id[image_id] = image
        source_size = tiled._sequence(image.get("source_size"), "test source_size")
        if len(source_size) != 2:
            raise ValueError("test source_size must contain width and height")
        width = tiled._positive_integer(source_size[0], "test source width")
        height = tiled._positive_integer(source_size[1], "test source height")
        source_image = tiled._nonempty_string(image.get("source_image"), "test source image")
        image_path = (source_root / source_image).resolve(strict=False)
        try:
            image_path.relative_to(source_root)
        except ValueError as exc:
            raise ValueError("test source image escapes Dataset300 root") from exc
        if not image_path.is_file() or image_path.stat().st_size <= 0:
            raise FileNotFoundError(f"test source image is missing: {image_path}")
        if tiled._sha256(image_path) != image.get("source_image_sha256"):
            raise ValueError(f"test source image SHA-256 mismatch: {source_image}")
        image["_resolved_path"] = image_path
        image["_source_width"] = width
        image["_source_height"] = height
        image["_ground_truth"] = []

    annotation_ids: set[int] = set()
    for raw in tiled._sequence(component_coco.get("annotations"), "COCO annotations"):
        annotation = tiled._mapping(raw, "COCO annotation")
        image_id = annotation.get("image_id")
        if image_id not in selected_by_image_id:
            continue
        annotation_id = tiled._positive_integer(annotation.get("id"), "test annotation id")
        if annotation_id in annotation_ids:
            raise ValueError(f"duplicate selected test annotation id {annotation_id}")
        annotation_ids.add(annotation_id)
        if annotation.get("ignore") not in (0, False, None):
            raise ValueError("full-view test ground truth unexpectedly ignored")
        category_id = tiled._positive_integer(
            annotation.get("category_id"), "test annotation category id"
        )
        category = category_by_id.get(category_id)
        if category is None:
            raise ValueError("test annotation references unknown category")
        box = tiled._bbox(annotation.get("source_bbox"), "test source_bbox")
        image = selected_by_image_id[int(image_id)]
        width = int(image["_source_width"])
        height = int(image["_source_height"])
        if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise ValueError("test source_bbox lies outside source image")
        image["_ground_truth"].append(
            {
                "annotation_id": annotation_id,
                "category": category["name"],
                "official_id": category["official_id"],
                "box": list(box),
            }
        )
    for image in selected:
        image["_ground_truth"].sort(key=lambda item: item["annotation_id"])

    return selected, categories, {
        "split": TEST_SPLIT,
        "available_test_samples": len(selected),
        "selected_test_samples": len(selected),
        "selection_policy": "all frozen test samples; no label-dependent selection",
        "selected_sample_ids": sample_ids,
        "selected_sample_ids_sha256": tiled._canonical_sha256(sample_ids),
        "selected_test_annotations": sum(len(item["_ground_truth"]) for item in selected),
        "train_samples_loaded": 0,
        "train_images_decoded": 0,
        "train_annotations_scored": 0,
        "val_samples_loaded": 0,
        "val_images_decoded": 0,
        "val_annotations_scored": 0,
        "index_split_view_counts_metadata_only": dict(sorted(split_view_counts.items())),
    }


def _filter_rows(
    rows: Sequence[Mapping[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        predictions = [
            dict(item)
            for item in tiled._sequence(raw.get("predictions"), "captured predictions")
            if tiled._finite(
                tiled._mapping(item, "captured prediction").get("score"),
                "captured prediction score",
            )
            >= threshold
        ]
        row["predictions"] = predictions
        row["calibrated_score_threshold"] = threshold
        output.append(row)
    return output


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(tiled._prediction_line(row) for row in rows)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _gate_results(task_config: Any, metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for gate in task_config.evaluation_gates:
        value = tiled._finite(metrics.get(gate.metric), f"metric {gate.metric}")
        threshold = float(gate.threshold)
        passed = value >= threshold if gate.operator == "gte" else value <= threshold
        output.append(
            {
                "metric": gate.metric,
                "operator": gate.operator,
                "threshold": threshold,
                "value": value,
                "passed": passed,
            }
        )
    return output


def main(argv: Iterable[str] | None = None) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    arguments = _parser().parse_args(argv)
    workspace = Path(arguments.workspace_root).expanduser().resolve(strict=True)
    paths = {
        "training_config": tiled._resolve_within(workspace, arguments.training_config, "training config"),
        "parent_status": tiled._resolve_within(workspace, arguments.parent_status, "parent status"),
        "component_coco": tiled._resolve_within(workspace, arguments.component_coco, "Component COCO"),
        "materialization_summary": tiled._resolve_within(workspace, arguments.materialization_summary, "materialization summary"),
        "pipeline_config": tiled._resolve_within(workspace, arguments.pipeline_config, "pipeline config"),
        "thresholds_config": tiled._resolve_within(workspace, arguments.thresholds_config, "thresholds config"),
        "calibration_audit": tiled._resolve_within(workspace, arguments.calibration_audit, "calibration audit"),
    }
    output = tiled._resolve_within(
        workspace, arguments.output_dir, "output directory", kind="directory", must_exist=False
    )
    tiled._prepare_output(output, workspace)
    cpu_threads = tiled._positive_integer(arguments.cpu_threads, "cpu-threads")

    parent_contract = tiled._parent_contract(
        workspace=workspace,
        config_path=paths["training_config"],
        parent_status_path=paths["parent_status"],
        component_coco_path=paths["component_coco"],
        materialization_summary_path=paths["materialization_summary"],
    )
    calibration = _calibration_contract(
        workspace=workspace,
        audit_path=paths["calibration_audit"],
        candidate_label=tiled._nonempty_string(arguments.candidate_label, "candidate label"),
        parent_status_path=paths["parent_status"],
        checkpoint_path=parent_contract["checkpoint_path"],
    )
    pipeline = tiled._read_object(paths["pipeline_config"], "pipeline config")
    thresholds = tiled._read_object(paths["thresholds_config"], "thresholds config")
    protocol = tiled._fixed_protocol(
        pipeline, thresholds, parent_contract["materialization_summary"]
    )
    component_coco = tiled._read_object(paths["component_coco"], "Component COCO")
    samples, categories, selection = _frozen_test_samples(
        component_coco=component_coco,
        source_root=parent_contract["source_root"],
        binding_sha256=parent_contract["binding_sha256"],
    )
    ground_truth = {
        str(item["sample_id"]): list(item["_ground_truth"]) for item in samples
    }

    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError("PyTorch is required for Component test confirmation") from exc
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(20260812)
    torch.use_deterministic_algorithms(True)
    loaded = load_checkpoint(
        parent_contract["checkpoint_path"],
        device="cpu",
        expected_sha256=parent_contract["checkpoint_sha256"],
        strict=True,
    )
    if loaded.lineage.get("binding_sha256") != parent_contract["binding_sha256"]:
        raise ValueError("checkpoint/materialization binding mismatch")
    adapter = TorchComponentAdapter(
        loaded,
        device="cpu",
        score_threshold=float(protocol["score_threshold"]),
        max_detections=int(protocol["max_detections_per_tile"]),
    )
    capture_path = output / "capture_population_scorefloor020.jsonl"
    captured_rows, performance = tiled._run_inference(
        samples=samples,
        adapter=adapter,
        protocol=protocol,
        predictions_path=capture_path,
    )
    selected_threshold = float(calibration["selected_score_threshold"])
    calibrated_rows = _filter_rows(captured_rows, selected_threshold)
    predictions_path = output / "predictions.jsonl"
    _write_jsonl(predictions_path, calibrated_rows)
    floor_metrics = tiled._evaluate_predictions(
        captured_rows, ground_truth, categories, iou_threshold=tiled.MATCH_IOU
    )
    calibrated_metrics = tiled._evaluate_predictions(
        calibrated_rows, ground_truth, categories, iou_threshold=tiled.MATCH_IOU
    )
    gates = _gate_results(parent_contract["task_config"], calibrated_metrics)
    gates_passed = all(bool(item["passed"]) for item in gates)
    if not gates_passed:
        status_label = "calibrated_test_confirmation_gate_failed"
    else:
        status_label = "calibrated_test_confirmation_passed"

    protected = {name: _source_entry(path, workspace) for name, path in paths.items()}
    status: dict[str, Any] = {
        "schema_version": 1,
        "kind": CONFIRMATION_KIND,
        "status": status_label,
        "task": "component",
        "formal_evidence": True,
        "registerable_repackage_evidence": gates_passed,
        "promotion_eligible": gates_passed,
        "blessing_authorized": False,
        "models_lock_read": False,
        "models_lock_modified": False,
        "finished_at_utc": _utc_now(),
        "device": "cpu",
        "deterministic_algorithms": True,
        "cpu_threads": torch.get_num_threads(),
        "calibration": calibration,
        "protocol": {
            **protocol,
            "captured_score_floor": float(protocol["score_threshold"]),
            "selected_score_threshold": selected_threshold,
            "selected_threshold_applied_after_class_aware_nms": True,
            "post_nms_filter_equivalent_to_pre_nms_filter_for_retained_higher_scores": True,
            "test_thresholds_searched": False,
        },
        "selection_boundary": selection,
        "metrics": calibrated_metrics,
        "population_floor_metrics": floor_metrics,
        "delta_calibrated_minus_floor": {
            key: float(calibrated_metrics[key]) - float(floor_metrics[key])
            for key in ("bbox_map50", "precision50", "recall50", "f1_50")
        },
        "gates": gates,
        "gates_passed": gates_passed,
        "performance": performance,
        "artifacts": {
            "predictions": {
                **_source_entry(predictions_path, workspace),
                "row_count": len(calibrated_rows),
                "prediction_count": sum(len(row["predictions"]) for row in calibrated_rows),
            },
            "captured_population": {
                **_source_entry(capture_path, workspace),
                "row_count": len(captured_rows),
                "prediction_count": sum(len(row["predictions"]) for row in captured_rows),
            },
        },
        "lineage": {
            **parent_contract["evidence"],
            "protected_sources": protected,
        },
        "checks": {
            "validation_selection_audit_bound": True,
            "selection_audit_used_zero_test_rows": True,
            "checkpoint_and_parent_match_calibration": True,
            "exact_frozen_test_population": len(samples) == 30,
            "test_labels_not_used_for_threshold_selection": True,
            "test_threshold_search_count": 0,
            "configured_gate_recomputed_on_test": True,
            "all_configured_gates_passed": gates_passed,
        },
        "limitations": [
            "The 30-image test split is an internal frozen Dataset300 holdout, not the organizer's hidden leaderboard set.",
            "This evidence authorizes only an unblessed candidate repackage with the exact selected score threshold.",
            "No official competition score is claimed.",
        ],
    }
    if len(samples) != 30:
        raise ValueError(f"formal confirmation requires exactly 30 frozen test samples; got {len(samples)}")
    after = {name: _source_entry(path, workspace) for name, path in paths.items()}
    if after != protected:
        raise RuntimeError("protected calibration/test source bytes changed during inference")
    status["protected_sources_unchanged"] = True
    status["confirmation_payload_sha256"] = tiled._canonical_sha256(status)
    status_path = output / "calibrated_test_status.json"
    tiled._atomic_write_json(status_path, status)
    print(status_path)
    print(json.dumps(calibrated_metrics, sort_keys=True, separators=(",", ":")))
    print(f"confirmation_payload_sha256={status['confirmation_payload_sha256']}")
    return 0 if gates_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
