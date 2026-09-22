from __future__ import annotations

"""Load the packaged YOLO Component candidate through its real adapter."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import evaluate_component_tiled_validation as base  # noqa: E402
from pcbparse.io import load_image, tile_image_view  # noqa: E402
from pcbparse.perception.yolo_component_adapter import factory  # noqa: E402
from pcbparse.training.artifacts import (  # noqa: E402
    build_candidate_lock_entry,
    load_candidate_bundle,
)


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
        "--sample-image",
        type=Path,
        default=Path("Dataset300/Dataset300/0030/0030-Datasheet.png"),
    )
    parser.add_argument(
        "--models-lock", type=Path, default=Path("models/models.lock.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/audits/component_yolo11n_v1_candidate_runtime_smoke.json"
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    workspace = Path(arguments.workspace_root).expanduser().resolve(strict=True)
    bundle_path = base._resolve_within(
        workspace, arguments.candidate_bundle, "candidate bundle"
    )
    sample_path = base._resolve_within(
        workspace, arguments.sample_image, "runtime smoke sample"
    )
    lock_path = base._resolve_within(workspace, arguments.models_lock, "models lock")
    output_path = base._resolve_within(
        workspace, arguments.output, "runtime smoke output", must_exist=False
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite runtime smoke: {output_path}")
    lock_sha_before = base._sha256(lock_path)
    bundle = load_candidate_bundle(bundle_path)
    if bundle.status != "candidate" or bundle.blessed is not False:
        raise ValueError("runtime smoke requires an unblessed candidate bundle")
    entry = build_candidate_lock_entry(
        bundle_path, lock_directory=bundle_path.parent
    )
    model_path = (bundle_path.parent / str(entry["path"])).resolve(strict=True)
    adapter = factory(model_path, entry)
    original = load_image(sample_path)
    views = tile_image_view(original, tile_size=512, overlap=64)
    if not views:
        raise ValueError("runtime smoke sample produced no component tiles")
    first = adapter.predict(views[0], None)
    second = adapter.predict(views[0], None)
    first_json = [item.model_dump(mode="json") for item in first]
    second_json = [item.model_dump(mode="json") for item in second]
    if first_json != second_json:
        raise RuntimeError("packaged YOLO adapter is not deterministic on repeated inference")
    if not first:
        raise RuntimeError("packaged YOLO adapter emitted no detections on the smoke tile")
    if any(
        not item.evidence
        or item.evidence[0].metadata.get("coordinate_system")
        != "image_view_local_xyxy"
        or "official_category_id" not in item.evidence[0].metadata
        for item in first
    ):
        raise RuntimeError("packaged YOLO detections violate evidence/category contract")
    lock_sha_after = base._sha256(lock_path)
    if lock_sha_before != lock_sha_after:
        raise RuntimeError("models.lock.json changed during candidate runtime smoke")
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pcbparse_yolo_component_candidate_runtime_smoke",
        "status": "runtime_smoke_passed",
        "candidate_only": True,
        "production_blessing": False,
        "models_lock_modified": False,
        "finished_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "checks": {
            "bundle_reverified": True,
            "production_factory_imported": True,
            "hash_bound_runtime_contract_loaded": True,
            "real_checkpoint_loaded": True,
            "repeated_inference_deterministic": True,
            "local_coordinates_and_official_ids_present": True,
            "models_lock_unchanged": True,
        },
        "result": {
            "adapter_version": adapter.version,
            "sample_id": sample_path.parent.name,
            "sample_image_sha256": base._sha256(sample_path),
            "tile_view_id": views[0].view_id,
            "tile_size": [views[0].width, views[0].height],
            "detection_count": len(first),
            "categories": sorted({item.category for item in first}),
        },
        "lineage": {
            "candidate_bundle": {
                "path": base._relative(bundle_path, workspace),
                "sha256": base._sha256(bundle_path),
            },
            "model": {
                "path": base._relative(model_path, workspace),
                "sha256": base._sha256(model_path),
            },
            "models_lock": {
                "path": base._relative(lock_path, workspace),
                "sha256_before": lock_sha_before,
                "sha256_after": lock_sha_after,
            },
        },
        "limitations": [
            "This is a one-tile runtime integration smoke, not a metric evaluation.",
            "The candidate remains unblessed and is not loaded by production models.lock.json.",
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
