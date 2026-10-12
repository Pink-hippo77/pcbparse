"""Quantify degraded-submission pruning per case (opt90-comp). GT-ACCESS: diagnostic_only.

Reads ir.metadata['degraded_submission_prune'] from the run-s6fix evaluation
IRs (the pipeline records it itself) to separate DETECTION from KEY-BINDING
losses: components detected by the perception phase but dropped for lacking a
non-empty key.

Usage: prune_stats.py <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path('/home/data/yinjx/pcbparse-competition/project')
MANIFEST = ROOT / 'work/opt90-lead/eval-s6fix/s6fix-eval-manifest.jsonl'


def main() -> int:
    out_path = Path(sys.argv[1])
    rows = [json.loads(x) for x in MANIFEST.read_text(encoding='utf-8').splitlines() if x.strip()]
    records = []
    tot_drop = tot_kept = tot_pin_drop = 0
    for row in sorted(rows, key=lambda r: r['case_id']):
        case = row['case_id']
        ir = json.loads(Path(row['ir_path']).read_text(encoding='utf-8'))
        meta = ir.get('metadata', {})
        prune = meta.get('degraded_submission_prune', {})
        kept = len(ir.get('components', []))
        dropped = int(prune.get('components_dropped', 0))
        tot_kept += kept
        tot_drop += dropped
        tot_pin_drop += int(prune.get('pins_dropped', 0))
        records.append({
            'case': case,
            'components_kept': kept,
            'components_dropped_no_key': dropped,
            'detected_total': kept + dropped,
            'pins_dropped': int(prune.get('pins_dropped', 0)),
            'pins_dropped_unowned': int(prune.get('pins_dropped_unowned', 0)),
            'dropped_ids': prune.get('component_ids_dropped', []),
        })
    out = {
        'schema': 'opt90-comp-prune-stats-v1',
        'gt_access': 'diagnostic_only',
        'totals': {'kept': tot_kept, 'dropped_no_key': tot_drop,
                   'detected_total': tot_kept + tot_drop,
                   'pins_dropped': tot_pin_drop},
        'records': records,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    for r in records:
        print(f"{r['case']} kept={r['components_kept']:3d} dropped_no_key={r['components_dropped_no_key']:3d} "
              f"detected={r['detected_total']:3d}")
    print('TOTAL', out['totals'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
