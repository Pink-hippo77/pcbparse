"""Val30 case profile join (opt90-comp, diagnostic): domain / canvas / grayscale.

Usage: case_profile.py <out.json>
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path('/home/data/yinjx/pcbparse-competition/project')
SPLIT = ROOT / 'data/splits/dataset300_split.jsonl'
MANIFEST = ROOT / 'work/opt90-lead/eval-s6fix/s6fix-eval-manifest.jsonl'


def main() -> int:
    import sys
    out_path = Path(sys.argv[1])
    rows = [json.loads(x) for x in SPLIT.read_text(encoding='utf-8').splitlines() if x.strip()]
    cases = {r['case_id'] for r in MANIFEST.read_text(encoding='utf-8').splitlines() if r.strip()
             for r in [json.loads(r)]}
    out = []
    for r in rows:
        if r['sample_id'] not in cases:
            continue
        img = r['files']['image']
        out.append({
            'case': r['sample_id'], 'split': r['split'], 'domain': r['source_class'],
            'base_name': r['base_name'], 'dup_group_size': r['duplicate_group_size'],
            'w': img['width'], 'h': img['height'],
            'gray_stddev': img['grayscale_stddev'],
            'components': r['counts']['components'], 'pins': r['counts']['pins'],
            'texts': r['counts']['texts'],
        })
    out.sort(key=lambda r: r['case'])
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    for r in out:
        print(f"{r['case']} {r['domain']:9s} {r['w']}x{r['h']} gray={r['gray_stddev']:5.1f} "
              f"comp={r['components']:3d} pins={r['pins']:3d} texts={r['texts']:3d} {r['base_name']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
