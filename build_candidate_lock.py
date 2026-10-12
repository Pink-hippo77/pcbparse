#!/usr/bin/env python3
"""Build the opt90 wire evaluation candidate lock.

Starts from the r07 candidate lock (component/ocr/pin entries byte-copied,
so the ONLY variable versus the r12-integration terminal run is the wire
model) and replaces the wire entry from the opt90 training bundle with honest
candidate-evaluation semantics (blessed=false, status='candidate').

Usage: build_candidate_lock.py <bundle_dir> <out_lock>
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path("/home/data/yinjx/pcbparse-competition/project")
TEMPLATE = ROOT / "work/module-opt80-20261007/r07/candidate-lock-ppocr.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    bundle_dir = Path(sys.argv[1]).expanduser().resolve()
    out_lock = Path(sys.argv[2]).expanduser().resolve()
    manifest_path = bundle_dir / "bundle.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task") != "wire":
        raise SystemExit("bundle is not a wire bundle")
    artifact = bundle_dir / manifest["artifact_path"]
    model_card = bundle_dir / manifest["model_card_path"]
    if sha(artifact) != manifest["artifact_sha256"]:
        raise SystemExit("bundle artifact hash mismatch")
    if sha(model_card) != manifest["model_card_sha256"]:
        raise SystemExit("model card hash mismatch")

    lock = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    entries = [e for e in lock["models"] if e.get("task") != "wire"]
    rel = lambda p: Path(__import__("os").path.relpath(p, out_lock.parent)).as_posix()
    adapter_smoke = manifest["adapter_smoke"]
    entry = {
        "adapter": f"{adapter_smoke['module']}:{adapter_smoke['attribute']}",
        "blessed": False,
        "bundle_manifest_sha256": sha(manifest_path),
        "candidate_manifest_path": rel(manifest_path),
        "dataset_config_hash": manifest["dataset_config_hash"],
        "evaluation": manifest["evaluation"],
        "export_consistency": manifest["export_consistency"],
        "gate_policy": manifest["gate_policy"],
        "inference": manifest["inference"],
        "license": manifest["license"],
        "model_card_path": rel(model_card),
        "model_card_sha256": manifest["model_card_sha256"],
        "path": rel(artifact),
        "precision": manifest["precision"],
        "sha256": manifest["artifact_sha256"],
        "status": "candidate",
        "task": "wire",
        "training_config_sha256": manifest["training_config_sha256"],
        "version": manifest["version"],
        "candidate_evaluation_only": True,
        "opt90_negative_patch_training": True,
    }
    entries.append(entry)
    entries.sort(key=lambda e: e["task"])
    lock["models"] = entries
    lock["note"] = (
        "opt90 wire candidate-evaluation lock: component/ocr/pin byte-copied "
        "from work/module-opt80-20261007/r07/candidate-lock-ppocr.json; wire = "
        "opt90 explicit-negative-patch retrain candidate. Production lock untouched."
    )
    out_lock.parent.mkdir(parents=True, exist_ok=True)
    out_lock.write_text(json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(json.dumps({"out": str(out_lock), "lock_sha256": sha(out_lock),
                      "wire_version": entry["version"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
