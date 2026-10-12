"""Dump IR metadata keys and any component funnel stats (opt90-comp). Usage: dump_meta2.py <case>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path('/home/data/yinjx/pcbparse-competition/project')
MANIFEST = ROOT / 'work/opt90-lead/eval-s6fix/s6fix-eval-manifest.jsonl'


def walk(obj, prefix=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f'{prefix}.{k}' if prefix else k
            if isinstance(v, (dict, list)):
                if any(t in k.lower() for t in ('component', 'prune', 'unique', 'count', 'funnel', 'drop', 'filter', 'assoc', 'dedup')):
                    print(p, '=', json.dumps(v, ensure_ascii=False)[:800])
                walk(v, p)
            else:
                if any(t in k.lower() for t in ('component', 'prune', 'unique', 'count', 'funnel', 'drop', 'filter', 'assoc', 'dedup')):
                    print(p, '=', v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:50]):
            walk(v, f'{prefix}[{i}]')


def main() -> int:
    case = sys.argv[1]
    rows = [json.loads(x) for x in MANIFEST.read_text(encoding='utf-8').splitlines() if x.strip()]
    row = next(r for r in rows if r['case_id'] == case)
    ir = json.loads(Path(row['ir_path']).read_text(encoding='utf-8'))
    walk(ir.get('metadata', {}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
