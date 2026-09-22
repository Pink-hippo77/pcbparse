from __future__ import annotations

"""Run one unblessed YOLO Component candidate with the locked downstream stack."""

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import evaluate_component_tiled_validation as base  # noqa: E402
from pcbparse.config import load_config  # noqa: E402
from pcbparse.ir.models import IRState  # noqa: E402
from pcbparse.perception.production_provider import (  # noqa: E402
    composite_provider_config_from_pipeline,
    load_production_provider,
)
from pcbparse.perception.yolo_component_adapter import factory  # noqa: E402
from pcbparse.training.artifacts import build_candidate_lock_entry  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument(
        "--candidate-bundle",
        type=Path,
        default=Path(
            "work/candidates-runtime-bound/component-yolo11n-v1-fixed020/"
            "bundle.manifest.json"
        ),
    )
    parser.add_argument(
        "--models-lock", type=Path, default=Path("models/models.lock.json")
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=Path("configs/pipeline.component512-experiment.yaml"),
    )
    parser.add_argument(
        "--sample-image",
        type=Path,
        default=Path("Dataset300/Dataset300/0030/0030-Datasheet.png"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/audits/component_yolo11n_v1_composite_shadow.json"
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    workspace = Path(arguments.workspace_root).expanduser().resolve(strict=True)
    bundle_path = base._resolve_within(
        workspace, arguments.candidate_bundle, "candidate bundle"
    )
    lock_path = base._resolve_within(workspace, arguments.models_lock, "models lock")
    pipeline_path = base._resolve_within(
        workspace, arguments.pipeline_config, "pipeline config"
    )
    sample_path = base._resolve_within(
        workspace, arguments.sample_image, "shadow sample image"
    )
    output_path = base._resolve_within(
        workspace, arguments.output, "shadow audit output", must_exist=False
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite shadow audit: {output_path}")

    lock_sha_before = base._sha256(lock_path)
    pipeline = load_config(pipeline_path)
    provider = load_production_provider(
        lock_path,
        config=composite_provider_config_from_pipeline(pipeline),
    )
    locked_component_version = str(provider.component_provider.version)
    entry = build_candidate_lock_entry(
        bundle_path, lock_directory=bundle_path.parent
    )
    model_path = (bundle_path.parent / str(entry["path"])).resolve(strict=True)
    candidate_adapter = factory(model_path, entry)
    provider.component_provider = candidate_adapter
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, OSError):
        pass

    started = time.perf_counter()
    ir = provider.build_ir(sample_path)
    elapsed = time.perf_counter() - started
    if ir.state != IRState.PERCEIVED:
        raise RuntimeError(f"shadow provider returned unexpected state {ir.state!r}")
    versions = base._mapping(ir.metadata.get("model_versions"), "IR model versions")
    if versions.get("component") != candidate_adapter.version:
        raise RuntimeError("IR metadata does not identify the YOLO candidate")
    component_ids = {item.internal_id for item in ir.components}
    dangling_pins = [item.internal_id for item in ir.pins if item.component_id not in component_ids]
    if dangling_pins:
        raise RuntimeError("shadow IR contains pins with missing Component owners")
    width, height = int(ir.image.width), int(ir.image.height)
    if any(
        not (
            0 <= item.bbox.x1 < item.bbox.x2 <= width
            and 0 <= item.bbox.y1 < item.bbox.y2 <= height
        )
        for item in ir.components
    ):
        raise RuntimeError("shadow Component box lies outside the original image")
    lock_sha_after = base._sha256(lock_path)
    if lock_sha_before != lock_sha_after:
        raise RuntimeError("models.lock.json changed during composite shadow run")

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pcbparse_yolo_component_composite_shadow",
        "status": "composite_shadow_passed",
        "candidate_only": True,
        "formal_metric_evidence": False,
        "production_blessing": False,
        "models_lock_modified": False,
        "finished_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "checks": {
            "locked_downstream_provider_loaded": True,
            "component_provider_swapped_in_memory_only": True,
            "component_version_propagated": True,
            "component_boxes_within_source": True,
            "all_pin_owners_resolve": True,
            "perceived_ir_completed": True,
            "models_lock_unchanged": True,
        },
        "result": {
            "sample_id": sample_path.parent.name,
            "sample_image_sha256": base._sha256(sample_path),
            "elapsed_seconds": elapsed,
            "components": len(ir.components),
            "pins": len(ir.pins),
            "texts": len(ir.texts),
            "segments": len(ir.segments),
            "junctions": len(ir.junctions),
            "locked_component_version_replaced": locked_component_version,
            "candidate_component_version": candidate_adapter.version,
            "model_versions": dict(versions),
            "component_candidate_gate": ir.metadata.get("component_candidate_gate"),
            "phase_timings_seconds": ir.metadata.get("phase_timings_seconds"),
        },
        "lineage": {
            "candidate_bundle": {
                "path": base._relative(bundle_path, workspace),
                "sha256": base._sha256(bundle_path),
            },
            "candidate_model": {
                "path": base._relative(model_path, workspace),
                "sha256": base._sha256(model_path),
            },
            "pipeline_config": {
                "path": base._relative(pipeline_path, workspace),
                "sha256": base._sha256(pipeline_path),
            },
            "models_lock": {
                "path": base._relative(lock_path, workspace),
                "sha256_before": lock_sha_before,
                "sha256_after": lock_sha_after,
            },
        },
        "limitations": [
            "This one-image shadow run checks integration, not aggregate system accuracy.",
            "The current locked Component model is replaced only in memory and remains production authority.",
        ],
    }
    report["audit_payload_sha256"] = base._canonical_sha256(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base._atomic_write_json(output_path, report)
    print(output_path)
    print(json.dumps(report["result"], ensure_ascii=False, sort_keys=True))
    print(f"audit_payload_sha256={report['audit_payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
