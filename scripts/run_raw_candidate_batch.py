"""Run the production perception/association/graph chain for a local raw-v1 probe.

This helper deliberately passes raster image paths only to the provider.  It
does not run the strict official serializer, so its output is evidence for
candidate comparison rather than a submission artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    args = parser.parse_args()

    config = load_config(args.pipeline_config)
    provider = load_production_provider(
        args.models_lock,
        config=composite_provider_config_from_pipeline(config),
    )
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
                    "provider_kind": "production",
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


if __name__ == "__main__":
    raise SystemExit(main())
