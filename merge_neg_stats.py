#!/usr/bin/env python3
"""Assemble the final neg_stats.json from run artifacts.

Sources (in precedence order):
  1. neg-labels/neg_stats.json          - the labeler's own final document if
                                          the full pool run completed (authoritative);
  2. neg_full.log done-lines            - per-sample rows printed by workers
                                          (neg_px / sig_px / sig_err_rate / elapsed);
  3. neg-labels/neg_maskonly_stragglers.json - mask-only NEG rows for the dense
                                          stragglers (NEG口径 identical).

Missing fields for log-sourced rows are recorded explicitly as null; the
document is honest about which samples used which path.

Usage: python merge_neg_stats.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/data/yinjx/pcbparse-competition/project")
WORK = ROOT / "work/opt90-wire"
NEG_DIR = WORK / "neg-labels"
OUT = NEG_DIR / "neg_stats_final.json"


def main() -> int:
    authoritative = NEG_DIR / "neg_stats.json"
    rows: dict[str, dict] = {}
    source = "logs"
    if authoritative.is_file():
        document = json.loads(authoritative.read_text(encoding="utf-8"))
        for row in document.get("samples", []):
            row = dict(row)
            row["row_source"] = "labeler_pool_final"
            rows[row["sample_id"]] = row
        source = "labeler_pool_final"

    log_path = WORK / "neg_full.log"
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith('{"done"'):
                continue
            item = json.loads(line)
            sample_id = item["done"]
            if sample_id in rows:
                continue
            sig_px = item.get("sig_px", 0)
            rows[sample_id] = {
                "sample_id": sample_id,
                "neg_px": item.get("neg_px"),
                "sig_px": sig_px,
                "sig_err_rate": item.get("sig_err_rate"),
                "sig_hit_gt_px": round(sig_px * item.get("sig_err_rate", 0.0)),
                "elapsed_s": item.get("elapsed_s"),
                "model_mask_px": None,
                "model_near_miss_px": None,
                "layers": None,
                "row_source": "pool_done_log",
            }

    straggler_path = NEG_DIR / "neg_maskonly_stragglers.json"
    if straggler_path.is_file():
        document = json.loads(straggler_path.read_text(encoding="utf-8"))
        for row in document.get("samples", []):
            sample_id = row["sample_id"]
            existing = rows.get(sample_id, {})
            merged = {**existing, **{k: v for k, v in row.items()}}
            merged["row_source"] = (existing.get("row_source", "none") + "+mask_only_neg")
            merged.setdefault("sig_px", None)
            merged.setdefault("sig_err_rate", None)
            rows[sample_id] = merged

    done = [row for row in rows.values() if row.get("neg_px") is not None]
    neg_total = sum(row["neg_px"] for row in done)
    sig_rows = [row for row in done if row.get("sig_px") is not None]
    sig_total = sum(row["sig_px"] for row in sig_rows)
    sig_hit = sum(row.get("sig_hit_gt_px") or round(row["sig_px"] * (row.get("sig_err_rate") or 0.0))
                  for row in sig_rows)
    document = {
        "schema_version": 2,
        "kind": "opt90_wire_negative_labeler_assembled",
        "authoritative_source": source,
        "note": (
            "assembled from run artifacts; rows sourced from pool_done_log lack "
            "model_mask_px / model_near_miss_px / per-layer counts (the pool's "
            "full rows were only written on complete pool exit)"
        ),
        "totals": {
            "samples": len(rows),
            "rows_with_neg": len(done),
            "rows_with_sig": len(sig_rows),
            "neg_px": neg_total,
            "sig_px": sig_total,
            "sig_hit_gt_px": sig_hit,
            "sig_err_rate": round(sig_hit / sig_total, 6) if sig_total else 0.0,
        },
        "samples": [rows[key] for key in sorted(rows)],
    }
    OUT.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(document["totals"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
