"""Run an isolated end-to-end probe with one historical model swapped in.

The formal model lock is only used to load the current baseline.  The swapped
artifact is loaded directly with its own byte hash and is never promoted or
written into ``models/models.lock.json``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib
import json
import time
from pathlib import Path
from typing import Any

from pcbparse.association import associate_ocr
from pcbparse.config import load_config
from pcbparse.evaluation.raw_cli import evaluate_prediction_manifest
from pcbparse.graph.physical import build_physical_graph
from pcbparse.ir.models import IRState
from pcbparse.nets.hypergraph import build_electrical_nets
from pcbparse.perception.production_provider import (
    composite_provider_config_from_pipeline,
    load_production_provider,
)
from pcbparse.pipeline.context import RunContext


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--models-lock", type=Path, required=True)
    parser.add_argument("--pipeline-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--swap-task", choices=("component", "ocr", "pin", "wire"), required=True
    )
    parser.add_argument("--swap-path", type=Path, required=True)
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--pin-score-threshold", type=float, default=0.15)
    parser.add_argument("--pin-max-detections", type=int, default=128)
    parser.add_argument("--runtime-pin-score-threshold", type=float, default=None)
    parser.add_argument("--runtime-pin-max-detections", type=int, default=None)
    parser.add_argument("--runtime-pin-box-prior-mode", choices=("filled_rect_v1", "boundary_band_v2"), default=None)
    parser.add_argument("--runtime-pin-box-margin", type=float, default=None)
    parser.add_argument("--runtime-pin-roi-padding", type=float, default=None)
    parser.add_argument("--runtime-component-min-score", type=float, default=None)
    parser.add_argument("--runtime-component-max-candidates", type=int, default=None)
    parser.add_argument("--runtime-component-nms-iou", type=float, default=None)
    parser.add_argument("--ocr-detector-threshold", type=float, default=None)
    parser.add_argument("--ocr-detector-min-area", type=int, default=None)
    parser.add_argument("--ocr-recognition-candidate-count", type=int, default=None)
    parser.add_argument("--wire-threshold", type=float, default=None)
    parser.add_argument("--association-max-distance-factor", type=float, default=None)
    parser.add_argument("--component-min-score", type=float, default=None)
    parser.add_argument("--component-unique-margin", type=float, default=None)
    parser.add_argument("--leave-global-component-targets-ownerless", action="store_true")
    parser.add_argument("--pin-min-score", type=float, default=None)
    parser.add_argument("--pin-unique-margin", type=float, default=None)
    parser.add_argument("--rdp-epsilon", type=float, default=None)
    parser.add_argument("--endpoint-merge", type=float, default=None)
    parser.add_argument("--collinear-merge", type=float, default=None)
    parser.add_argument("--collinear-gap", type=float, default=None)
    parser.add_argument("--pin-snap-min", type=float, default=None)
    parser.add_argument("--pin-snap-max", type=float, default=None)
    parser.add_argument("--graph-unique-margin", type=float, default=None)
    parser.add_argument("--disable-deterministic-junction-union", action="store_true")
    parser.add_argument("--wire-minimum-run", type=int, default=None)
    parser.add_argument("--wire-gap-closing-radius", type=int, default=None)
    parser.add_argument("--wire-hole-area", type=int, default=None)
    parser.add_argument("--wire-max-component-pixels", type=int, default=None)
    args = parser.parse_args()
    run_started = time.monotonic()

    config = load_config(args.pipeline_config)
    if (
        args.ocr_detector_threshold is not None
        or args.ocr_detector_min_area is not None
        or args.ocr_recognition_candidate_count is not None
    ):
        ocr_config = dict(config.get("ocr", {}))
        if args.ocr_detector_threshold is not None:
            ocr_config["detector_threshold"] = args.ocr_detector_threshold
        if args.ocr_detector_min_area is not None:
            ocr_config["detector_min_area"] = args.ocr_detector_min_area
        if args.ocr_recognition_candidate_count is not None:
            ocr_config["recognition_candidate_count"] = args.ocr_recognition_candidate_count
        config["ocr"] = ocr_config
    if args.wire_threshold is not None:
        wire_config = dict(config.get("wire", {}))
        wire_config["threshold"] = args.wire_threshold
        config["wire"] = wire_config
    if any(value is not None for value in (args.runtime_pin_score_threshold, args.runtime_pin_max_detections, args.runtime_pin_box_prior_mode, args.runtime_pin_box_margin)):
        pin_config = dict(config.get("pin", {}))
        if args.runtime_pin_score_threshold is not None:
            pin_config["score_threshold"] = args.runtime_pin_score_threshold
        if args.runtime_pin_max_detections is not None:
            pin_config["max_detections_per_component"] = args.runtime_pin_max_detections
        if args.runtime_pin_box_prior_mode is not None:
            pin_config["box_prior_mode"] = args.runtime_pin_box_prior_mode
        if args.runtime_pin_box_margin is not None:
            pin_config["box_margin_px"] = args.runtime_pin_box_margin
        config["pin"] = pin_config
    if args.runtime_pin_roi_padding is not None:
        image_config = dict(config.get("image", {}))
        image_config["pin_component_roi_padding_px"] = args.runtime_pin_roi_padding
        config["image"] = image_config
    if any(
        value is not None
        for value in (
            args.runtime_component_min_score,
            args.runtime_component_max_candidates,
            args.runtime_component_nms_iou,
        )
    ):
        image_config = dict(config.get("image", {}))
        if args.runtime_component_min_score is not None:
            image_config["component_candidate_min_score"] = args.runtime_component_min_score
        if args.runtime_component_max_candidates is not None:
            image_config["component_max_candidates"] = args.runtime_component_max_candidates
        config["image"] = image_config
        if args.runtime_component_nms_iou is not None:
            image_config["component_nms_iou"] = args.runtime_component_nms_iou
            config["image"] = image_config
    graph_config = dict(config.get("graph", {}))
    if args.wire_minimum_run is not None:
        graph_config["wire_minimum_run_px"] = args.wire_minimum_run

    if args.rdp_epsilon is not None:
        graph_config["rdp_epsilon_px"] = args.rdp_epsilon
    if args.endpoint_merge is not None:
        graph_config["endpoint_merge_tolerance_px"] = args.endpoint_merge
    if args.collinear_merge is not None:
        graph_config["collinear_merge_tolerance_px"] = args.collinear_merge
    if args.collinear_gap is not None:
        graph_config["collinear_gap_merge_tolerance_px"] = args.collinear_gap
    if args.graph_unique_margin is not None:
        graph_config["unique_margin"] = args.graph_unique_margin
    if graph_config != config.get("graph", {}):
        config["graph"] = graph_config
    if args.leave_global_component_targets_ownerless:
        association_config = dict(config.get("association", {}))
        association_config["bind_global_component_targets"] = False
        config["association"] = association_config
    if any(value is not None for value in (args.association_max_distance_factor, args.component_min_score, args.component_unique_margin, args.pin_min_score, args.pin_unique_margin, args.pin_snap_min, args.pin_snap_max, args.graph_unique_margin)):
        association_config = dict(config.get("association", {}))
        if args.association_max_distance_factor is not None:
            association_config["max_component_text_distance_factor"] = args.association_max_distance_factor
        if args.component_min_score is not None:
            association_config["component_min_score"] = args.component_min_score
        if args.component_unique_margin is not None:
            association_config["component_unique_margin"] = args.component_unique_margin
        if args.pin_min_score is not None:
            association_config["pin_min_score"] = args.pin_min_score
        if args.pin_unique_margin is not None:
            association_config["pin_unique_margin"] = args.pin_unique_margin
        if args.pin_snap_min is not None:
            association_config["pin_wire_snap_min_px"] = args.pin_snap_min
        if args.pin_snap_max is not None:
            association_config["pin_wire_snap_max_px"] = args.pin_snap_max
        if args.graph_unique_margin is not None:
            association_config["unique_margin"] = args.graph_unique_margin
        config["association"] = association_config
    provider_config = composite_provider_config_from_pipeline(config)
    provider = load_production_provider(args.models_lock, config=provider_config)
    inherited_wire_threshold = getattr(provider.wire_provider, "threshold", None)
    if args.runtime_component_nms_iou is not None:
        provider.config.component_nms_iou = args.runtime_component_nms_iou
    if args.wire_gap_closing_radius is not None:
        provider.config.wire_gap_closing_radius_px = args.wire_gap_closing_radius
    if args.wire_hole_area is not None:
        provider.config.wire_maximum_hole_area_px = args.wire_hole_area
    if args.wire_max_component_pixels is not None:
        provider.config.wire_maximum_component_pixels = args.wire_max_component_pixels
    current_lock = json.loads(args.models_lock.read_text(encoding="utf-8"))
    entries = {str(item["task"]): dict(item) for item in current_lock["models"]}
    provider_attr = f"{args.swap_task}_provider"
    setattr(
        provider,
        provider_attr,
        _load_swap(
            args.swap_task,
            args.swap_path,
            entries[args.swap_task],
            pin_score_threshold=args.pin_score_threshold,
            pin_max_detections=args.pin_max_detections,
        ),
    )
    if args.swap_task == "wire":
        # _load_swap builds a standalone adapter with its artifact default
        # threshold. Restore the active pipeline threshold so a wire model swap
        # remains a one-variable comparison.
        threshold = args.wire_threshold
        if threshold is None:
            threshold = inherited_wire_threshold
        if threshold is not None:
            provider.wire_provider.threshold = float(threshold)
    if args.swap_task == "pin":
        swapped_pin = provider.pin_provider
        pin_overrides: dict[str, Any] = {}
        if args.runtime_pin_score_threshold is not None:
            pin_overrides["score_threshold"] = args.runtime_pin_score_threshold
        if args.runtime_pin_max_detections is not None:
            pin_overrides["max_detections_per_component"] = args.runtime_pin_max_detections
        if args.runtime_pin_box_prior_mode is not None:
            pin_overrides["box_prior_mode"] = args.runtime_pin_box_prior_mode
        if args.runtime_pin_box_margin is not None:
            pin_overrides["box_margin_px"] = args.runtime_pin_box_margin
        if pin_overrides:
            swapped_pin.config = replace(swapped_pin.config, **pin_overrides)
    if args.swap_task == "ocr":
        if args.ocr_detector_threshold is not None:
            provider.ocr_provider.detector_threshold = args.ocr_detector_threshold
        if args.ocr_detector_min_area is not None:
            provider.ocr_provider.detector_min_area = args.ocr_detector_min_area
        if args.ocr_recognition_candidate_count is not None:
            provider.ocr_provider.recognition_candidate_count = args.ocr_recognition_candidate_count

    selected = _selected_rows(args.split_manifest, args.split)
    wanted = {item.strip() for item in args.case_ids.split(",") if item.strip()}
    if wanted:
        selected = [row for row in selected if str(row["sample_id"]) in wanted]
    if not selected:
        raise ValueError("no selected cases remain after --case-ids filtering")

    output_dir = args.output_dir.resolve()
    ir_dir = output_dir / "ir"
    ir_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    graph = config.get("graph", {})
    association = config.get("association", {})
    labels = config.get("labels", {})
    for row in selected:
        case_id = str(row["sample_id"])
        image_path = args.dataset_root / str(row["files"]["image"]["path"])
        try:
            ctx = RunContext.start(image_path, config)
            ir = provider.build_ir(image_path, ctx)
            associate_ocr(ir, config)
            original_junctions = ir.junctions
            if args.disable_deterministic_junction_union:
                ir.junctions = []
            ir.physical_nets, decisions = build_physical_graph(
                ir,
                endpoint_tolerance_px=float(graph.get("endpoint_merge_tolerance_px", 3.0)),
                collinear_merge_tolerance_px=float(graph.get("collinear_merge_tolerance_px", 0.0)),
                collinear_gap_merge_tolerance_px=float(graph.get("collinear_gap_merge_tolerance_px", 0.0)),
                pin_snap_min_px=float(association.get("pin_wire_snap_min_px", 2.0)),
                pin_snap_tolerance_px=float(
                    association.get("pin_wire_snap_max_px", graph.get("pin_snap_tolerance_px", 6.0))
                ),
                pin_point_projection_max_px=float(graph.get("pin_point_projection_max_px", 0.0)),
                unique_margin=float(association.get("unique_margin", 1.5)),
            )
            ir.metadata["snap_decisions"] = [
                {
                    "pin_id": item.pin_id,
                    "status": item.status,
                    "segment_id": item.segment_id,
                    "cost": item.cost,
                    "margin": item.margin,
                }
                for item in decisions
            ]
            if args.disable_deterministic_junction_union:
                ir.junctions = original_junctions
            ir.electrical_nets = build_electrical_nets(
                ir,
                merge_signal_labels=bool(labels.get("merge_signal_labels", True)),
                merge_power_labels=bool(labels.get("merge_power_labels", False)),
                merge_ground_labels=bool(labels.get("merge_ground_labels", False)),
                min_text_score=float(labels.get("min_text_score", 0.75)),
                merge_threshold=float(labels.get("merge_threshold", 0.85)),
            )
            ir.state = IRState.SEMANTIC
            ir_path = ir_dir / f"{case_id}.circuit-ir.json"
            ir_path.write_text(ir.model_dump_json(), encoding="utf-8")
            manifest_rows.append(
                {
                    "case_id": case_id,
                    "ir_path": str(ir_path),
                    "provider_kind": "production-model-swap-probe",
                    "sidecar_access": False,
                    "image_sha256": ir.image.sha256,
                }
            )
            print(
                f"{case_id}: components={len(ir.components)} pins={len(ir.pins)} "
                f"segments={len(ir.segments)} physical_nets={len(ir.physical_nets)}",
                flush=True,
            )
        except Exception as exc:
            errors.append({"case_id": case_id, "type": type(exc).__name__, "message": str(exc)})
            print(f"{case_id}: ERROR {type(exc).__name__}: {exc}", flush=True)

    manifest_path = output_dir / "predictions.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    summary: dict[str, object] = {
        "cases": len(selected),
        "succeeded": len(manifest_rows),
        "failed": len(errors),
        "errors": errors,
        "swap_task": args.swap_task,
        "swap_path": str(args.swap_path.resolve()),
        "swap_sha256": _sha256(args.swap_path),
        "manifest": str(manifest_path),
        "elapsed_seconds": round(time.monotonic() - run_started, 3),
    }
    if not errors:
        report = evaluate_prediction_manifest(manifest_path, dataset_root=args.dataset_root)
        (output_dir / "reports.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary["aggregate"] = report["aggregate"]
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if not errors else 1


def _load_swap(
    task: str,
    path: Path,
    current_entry: dict[str, Any],
    *,
    pin_score_threshold: float,
    pin_max_detections: int,
) -> Any:
    path = path.resolve(strict=True)
    entry = dict(current_entry)
    entry["sha256"] = _sha256(path)
    entry["version"] = f"probe-{task}-{entry['sha256'][:12]}"
    if task == "component":
        entry["inference"] = {"device": "auto", "score_threshold": 0.0, "max_detections": 300}
        module, attribute = "pcbparse.perception.torch_component_adapter", "factory"
    elif task == "ocr":
        entry.pop("lineage", None)
        module, attribute = "pcbparse.perception.torch_ocr_adapter", "factory"
    elif task == "pin":
        entry["inference"] = {
            "device": "auto",
            "input_size": [384, 384],
            "output_stride": 4,
            "score_threshold": pin_score_threshold,
            "nms_kernel": 3,
            "nms_radius_heatmap": 2.0,
            "max_detections_per_component": pin_max_detections,
            "base_length_pixels": 8.0,
            "box_margin_px": 4.0,
            "box_prior_mode": "filled_rect_v1",
        }
        entry["artifact_format"] = "onnx"
        module, attribute = "pcbparse.perception.torch_pin_adapter", "create_pin_provider"
    else:
        entry["inference"] = {"device": "auto", "threshold": 0.5}
        module, attribute = "pcbparse.perception.torch_wire_adapter", "factory"
    factory = getattr(importlib.import_module(module), attribute)
    return factory(path, entry)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_rows(path: Path, split: str) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in rows if row.get("split") == split]
    if not selected:
        raise ValueError(f"split has no rows: {split}")
    return selected


if __name__ == "__main__":
    raise SystemExit(main())
