"""Run the production perception/association/graph chain for a local raw-v1 probe.

This helper deliberately passes raster image paths only to the provider.  It
does not run the strict official serializer, so its output is evidence for
candidate comparison rather than a submission artifact.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

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
from pcbparse.training.artifacts import build_candidate_lock_entry, load_candidate_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--models-lock", type=Path, required=True)
    parser.add_argument("--pipeline-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    shadow_group = parser.add_mutually_exclusive_group()
    shadow_group.add_argument(
        "--component-candidate-bundle",
        type=Path,
        help=(
            "Optional unblessed Component bundle to inject in memory for an "
            "image-only shadow run; models.lock.json is never edited."
        ),
    )
    shadow_group.add_argument(
        "--component-shadow-checkpoint",
        type=Path,
        help=(
            "Optional unpackaged YOLO checkpoint for a validation-only shadow run. "
            "This path is never written to models.lock.json."
        ),
    )
    parser.add_argument(
        "--component-shadow-category-table",
        type=Path,
        default=Path("data/derived/pcbnet-upgrade-v1/component_yolo/classes.json"),
    )
    parser.add_argument("--component-shadow-score-threshold", type=float, default=0.20)
    parser.add_argument("--component-shadow-nms-iou", type=float, default=0.55)
    parser.add_argument("--component-shadow-imgsz", type=int, default=640)
    parser.add_argument("--component-shadow-max-detections", type=int, default=500)
    parser.add_argument(
        "--component-shadow-per-class-thresholds-audit",
        type=Path,
        help="Hash-verified validation calibration used only with a shadow checkpoint.",
    )
    args = parser.parse_args()
    if (
        args.component_shadow_per_class_thresholds_audit is not None
        and args.component_shadow_checkpoint is None
    ):
        raise ValueError(
            "per-class threshold audit requires --component-shadow-checkpoint"
        )

    config = load_config(args.pipeline_config)
    provider = load_production_provider(
        args.models_lock,
        config=composite_provider_config_from_pipeline(config),
    )
    component_candidate: dict[str, str] | None = None
    if args.component_candidate_bundle is not None:
        bundle_path = args.component_candidate_bundle.expanduser().resolve(strict=True)
        bundle = load_candidate_bundle(bundle_path)
        if bundle.task != "component" or bundle.status != "candidate" or bundle.blessed is not False:
            raise ValueError("shadow Component bundle must be an unblessed component candidate")
        entry = build_candidate_lock_entry(
            bundle_path, lock_directory=bundle_path.parent
        )
        module_name, attribute_name = str(entry["adapter"]).rsplit(":", 1)
        adapter_factory = getattr(importlib.import_module(module_name), attribute_name)
        model_path = (bundle_path.parent / str(entry["path"])).resolve(strict=True)
        provider.component_provider = adapter_factory(model_path, entry)
        component_candidate = {
            "bundle": str(bundle_path),
            "bundle_sha256": _sha256(bundle_path),
            "model_sha256": str(entry["sha256"]),
            "adapter": str(entry["adapter"]),
            "version": str(entry["version"]),
        }
    elif args.component_shadow_checkpoint is not None:
        checkpoint_path = args.component_shadow_checkpoint.expanduser().resolve(
            strict=True
        )
        category_path = args.component_shadow_category_table.expanduser().resolve(
            strict=True
        )
        adapter_factory = getattr(
            importlib.import_module("pcbparse.perception.yolo_component_adapter"),
            "factory",
        )
        checkpoint_sha256 = _sha256(checkpoint_path)
        per_class_thresholds: dict[str, float] = {}
        threshold_audit: dict[str, object] | None = None
        threshold_audit_path: Path | None = None
        if args.component_shadow_per_class_thresholds_audit is not None:
            threshold_audit_path = (
                args.component_shadow_per_class_thresholds_audit.expanduser().resolve(
                    strict=True
                )
            )
            threshold_module = importlib.import_module(
                "evaluate_yolo_component_tiled_validation"
            )
            per_class_thresholds, threshold_audit = (
                threshold_module._per_class_thresholds_from_audit(
                    threshold_audit_path,
                    workspace=ROOT.resolve(strict=True),
                    checkpoint_sha256=checkpoint_sha256,
                    population_floor=args.component_shadow_score_threshold,
                )
            )
        entry = {
            "task": "component",
            "sha256": checkpoint_sha256,
            "category_table_path": str(category_path),
            "category_table_sha256": _sha256(category_path),
            "inference": {
                "device": "auto",
                "score_threshold": args.component_shadow_score_threshold,
                "nms_iou": args.component_shadow_nms_iou,
                "max_detections": args.component_shadow_max_detections,
                "imgsz": args.component_shadow_imgsz,
                "per_class_score_thresholds": per_class_thresholds,
            },
        }
        provider.component_provider = adapter_factory(checkpoint_path, entry)
        component_candidate = {
            "checkpoint": str(checkpoint_path),
            "model_sha256": checkpoint_sha256,
            "category_table": str(category_path),
            "category_table_sha256": _sha256(category_path),
            "adapter": "pcbparse.perception.yolo_component_adapter:factory",
            "version": f"validation-shadow-{checkpoint_sha256[:16]}",
            "evidence_boundary": "unpackaged_validation_only",
            "per_class_score_thresholds": json.dumps(
                per_class_thresholds, sort_keys=True, separators=(",", ":")
            ),
            **(
                {
                    "threshold_audit": str(threshold_audit_path),
                    "threshold_audit_sha256": _sha256(threshold_audit_path),
                    "threshold_audit_payload_sha256": str(
                        threshold_audit["calibration_payload_sha256"]
                    ),
                }
                if threshold_audit_path is not None and threshold_audit is not None
                else {}
            ),
        }
    output_dir = args.output_dir.resolve()
    ir_dir = output_dir / "ir"
    ir_dir.mkdir(parents=True, exist_ok=True)
    selected = _selected_rows(args.split_manifest, args.split)
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
            ir.physical_nets, snap_decisions = build_physical_graph(
                ir,
                endpoint_tolerance_px=float(graph.get("endpoint_merge_tolerance_px", 3.0)),
                collinear_merge_tolerance_px=float(
                    graph.get("collinear_merge_tolerance_px", 0.0)
                ),
                collinear_gap_merge_tolerance_px=float(
                    graph.get("collinear_gap_merge_tolerance_px", 0.0)
                ),
                pin_snap_min_px=float(
                    association.get("pin_wire_snap_min_px", 2.0)
                ),
                pin_snap_tolerance_px=float(
                    association.get(
                        "pin_wire_snap_max_px",
                        graph.get("pin_snap_tolerance_px", 6.0),
                    )
                ),
                pin_point_projection_max_px=float(graph.get("pin_point_projection_max_px", 0.0)),
                unique_margin=float(association.get("unique_margin", 1.5)),
            )
            ir.metadata["snap_decisions"] = [
                {
                    "pin_id": decision.pin_id,
                    "status": decision.status,
                    "segment_id": decision.segment_id,
                    "cost": decision.cost,
                    "margin": decision.margin,
                }
                for decision in snap_decisions
            ]
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
                    "provider_kind": (
                        "candidate_shadow" if component_candidate is not None else "production"
                    ),
                    "sidecar_access": False,
                    "image_sha256": ir.image.sha256,
                }
            )
            print(
                f"{case_id}: components={len(ir.components)} pins={len(ir.pins)} "
                f"segments={len(ir.segments)} physical_nets={len(ir.physical_nets)}",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - probe failure is data-dependent.
            errors.append(
                {"case_id": case_id, "type": type(exc).__name__, "message": str(exc)}
            )
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
        "manifest": str(manifest_path),
        "component_candidate": component_candidate,
    }
    if not errors:
        report = evaluate_prediction_manifest(
            manifest_path,
            dataset_root=args.dataset_root,
        )
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


def _selected_rows(path: Path, split: str) -> list[dict[str, object]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if row.get("split") == split]
    if not selected:
        raise ValueError(f"split has no rows: {split}")
    return selected


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
