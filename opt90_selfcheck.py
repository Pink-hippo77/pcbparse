#!/usr/bin/env python3
"""opt90 materialization self-check.

Validates the opt90 negative-patch materialization the way training sees it,
plus the injection invariants:

1. ``WireTorchDataset`` (the training loader) constructs: SHA-256 bindings,
   foreground counts, row schema, lineage run ids.
2. Per tile (eligible rows only):
   * ``target ∩ ignore = ∅`` (loader contract);
   * NEG pixels are never left ignored (the injection flipped them to explicit
     negative supervision): ``ignore ∩ NEG_tile == ∅``;
   * declared ``opt90.ignore_pixels_flipped_to_negative <= |NEG_tile|``.
3. Ineligible rows (unusable sparse geometry) stay fully ignored and keep the
   NEG pixels ignored (no supervision may leak into those samples).

Usage: python opt90_selfcheck.py [--root work/opt90-wire/materialized] [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/home/data/yinjx/pcbparse-competition/project")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from pcbparse.training.torch_wire_junction import WireTorchDataset  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="work/opt90-wire/materialized")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    root = (ROOT / args.root).resolve()
    latest = json.loads((root / "latest.json").read_text(encoding="utf-8"))
    run_dir = (root / latest["run_dir"]).resolve()
    summary_path = run_dir / "summary.json"
    if _sha256_file(summary_path) != latest["summary_sha256"]:
        raise SystemExit("latest.json summary sha mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cfg = summary["materializer_config"]
    print("run_id", summary["run_id"])
    print("materializer_config_sha256", summary["materializer_config_sha256"])
    print("negative_label_tree_sha256", cfg.get("negative_label_tree_sha256"))
    print("neg_patch_per_sample", cfg.get("neg_patch_per_sample"))
    print("wire_tile_count", summary["materialization"]["wire_tile_count"])

    dataset = WireTorchDataset(run_dir, source_root=ROOT / "Dataset300" / "Dataset300", split=None)
    print("loader records", len(dataset.records))

    neg_cache: dict[str, object] = {}
    _MISSING = object()
    flipped_total = 0
    negp_tiles = grid_tiles = 0
    by_sample: dict[str, dict] = {}
    checked = 0
    problems: list[str] = []
    for row in dataset.records:
        if args.limit and checked >= args.limit:
            break
        checked += 1
        sample_id = row["sample_id"]
        opt90 = row.get("opt90", {})
        if opt90.get("negative_patch"):
            negp_tiles += 1
        else:
            grid_tiles += 1
        stats = by_sample.setdefault(sample_id, {"grid": 0, "negp": 0, "flipped": 0,
                                                 "injection": bool(opt90.get("injection_active"))})
        stats["negp" if opt90.get("negative_patch") else "grid"] += 1
        declared = int(opt90.get("ignore_pixels_flipped_to_negative", 0))
        flipped_total += declared
        stats["flipped"] += declared
        if opt90.get("injection_active") is False and declared:
            problems.append(f"{row['record_id']}: flips declared without active injection")

        neg = neg_cache.get(sample_id, _MISSING)
        if neg is _MISSING:
            neg_path = run_dir / "wire" / "neg_masks" / f"{sample_id}.png"
            if neg_path.is_file():
                neg = np.asarray(Image.open(neg_path).convert("L")) > 0
            else:
                neg = None  # val/test sample: no injection is derived from evaluation data
            neg_cache[sample_id] = neg

        box = row["tile_bbox"]
        neg_tile = (
            neg[box["y1"] : box["y2"], box["x1"] : box["x2"]]
            if neg is not None
            else np.zeros((box["y2"] - box["y1"], box["x2"] - box["x1"]), dtype=bool)
        )
        ignore = np.asarray(Image.open(run_dir / row["ignore_mask"]["path"]).convert("L")) > 0
        target = np.asarray(Image.open(run_dir / row["target_mask"]["path"]).convert("L")) > 0

        if bool((ignore & target).any()):
            problems.append(f"{row['record_id']}: target∩ignore non-empty")
        if declared > int(neg_tile.sum()):
            problems.append(f"{row['record_id']}: declared flips {declared} > NEG pixels")
        if row["supervision"]["eligible"]:
            overlap = int((ignore & neg_tile).sum())
            if overlap:
                problems.append(f"{row['record_id']}: {overlap} NEG pixels still ignored")
        else:
            if not bool(ignore.all()):
                problems.append(f"{row['record_id']}: ineligible tile not fully ignored")

    print("checked", checked, "grid_tiles", grid_tiles, "negp_tiles", negp_tiles)
    print("flipped_pixels_total", flipped_total)
    for name in sorted(by_sample):
        print(" ", name, by_sample[name])
    if problems:
        print("PROBLEMS:", len(problems))
        for item in problems[:10]:
            print(" ", item)
        return 1
    print("self-check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
