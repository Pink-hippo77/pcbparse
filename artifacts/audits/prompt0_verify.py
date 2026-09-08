"""Prompt 0 verification: model lock hash, split exposure, status checks."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(r"E:\EDA挑战题目六")

def main():
    # 1. Model lock hash
    lock_path = ROOT / "models/models.lock.json"
    lock_bytes = lock_path.read_bytes()
    actual_hash = hashlib.sha256(lock_bytes).hexdigest()
    expected = "af35e1299b755d03ff37889f9bb26e33601373f0f7f19db3a84468e896461413"
    print(f"Model lock SHA-256: {actual_hash}")
    print(f"Expected:           {expected}")
    print(f"Match: {actual_hash == expected}")
    print()

    # 2. Split exposure
    with open(ROOT / "data/splits/dataset300_split.jsonl") as f:
        splits = [json.loads(line) for line in f if line.strip()]
    from collections import Counter
    split_counts = Counter(row["split"] for row in splits)
    print(f"Total split entries: {len(splits)}")
    print(f"Split distribution: {dict(split_counts)}")

    with open(ROOT / "data/splits/dataset300_release50_v1.jsonl") as f:
        release50 = [json.loads(line) for line in f if line.strip()]
    print(f"Release50 entries: {len(release50)}")

    # Cross-reference: which release50 IDs came from test vs val
    release_ids = {row["sample_id"] for row in release50}
    origin_map = {row["sample_id"]: row["split"] for row in splits}
    release_origins = Counter(origin_map.get(sid, "missing") for sid in release_ids)
    print(f"Release50 origin splits: {dict(release_origins)}")
    print()

    # 3. Model lock entries
    lock = json.loads(lock_bytes)
    print(f"Lock blessed: {lock.get('blessed')}")
    print(f"Lock junction_mode: {lock.get('junction_mode')}")
    for entry in lock["models"]:
        task = entry["task"]
        path = entry["path"]
        blessed = entry.get("blessed", False)
        gate_info = entry.get("evaluation", {}).get("gates", [])
        gate_statuses = [(g["metric"], g.get("passed", False), g.get("value")) for g in gate_info]
        print(f"  {task}: path={path}, blessed={blessed}")
        for metric, passed, value in gate_statuses:
            print(f"    gate: {metric}={'PASS' if passed else 'FAIL'} (value={value})")
    print()

    # 4. Config comparison
    default_cfg = json.loads((ROOT / "configs/pipeline.yaml").read_text())
    rc_cfg = json.loads((ROOT / "configs/pipeline.release-candidate.yaml").read_text())
    
    # Check collinear tolerances
    default_graph = default_cfg.get("graph", {})
    rc_graph = rc_cfg.get("graph", {})
    print("Graph config comparison:")
    print(f"  default collinear_merge_tolerance_px: {default_graph.get('collinear_merge_tolerance_px', 'MISSING')}")
    print(f"  rc collinear_merge_tolerance_px: {rc_graph.get('collinear_merge_tolerance_px', 'MISSING')}")
    print(f"  default collinear_gap_merge_tolerance_px: {default_graph.get('collinear_gap_merge_tolerance_px', 'MISSING')}")
    print(f"  rc collinear_gap_merge_tolerance_px: {rc_graph.get('collinear_gap_merge_tolerance_px', 'MISSING')}")
    print(f"  default endpoint_merge_tolerance_px: {default_graph.get('endpoint_merge_tolerance_px')}")
    print(f"  rc endpoint_merge_tolerance_px: {rc_graph.get('endpoint_merge_tolerance_px')}")
    print()

    # 5. Check work directories
    work_dirs = sorted((ROOT / "work").iterdir()) if (ROOT / "work").exists() else []
    print(f"Work directories ({len(work_dirs)}):")
    for d in work_dirs[:10]:
        if d.is_dir():
            summary = d / "summary.json"
            if summary.exists():
                s = json.loads(summary.read_text())
                print(f"  {d.name}: status={s.get('status')}, succeeded={s.get('succeeded')}, failed={s.get('failed')}")
            else:
                print(f"  {d.name}: (no summary.json)")
    if len(work_dirs) > 10:
        print(f"  ... and {len(work_dirs) - 10} more")

if __name__ == "__main__":
    main()
