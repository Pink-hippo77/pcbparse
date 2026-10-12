"""Dump IR metadata for one case (opt90-comp diagnostic). Usage: dump_meta.py <case>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path('/home/data/yinjx/pcbparse-competition/project')
MANIFEST = ROOT / 'work/opt90-lead/eval-s6fix/s6fix-eval-manifest.jsonl'


def main() -> int:
    case = sys.argv[1]
    rows = [json.loads(x) for x in MANIFEST.read_text(encoding='utf-8').splitlines() if x.strip()]
    row = next(r for r in rows if r['case_id'] == case)
    ir = json.loads(Path(row['ir_path']).read_text(encoding='utf-8'))
    print(json.dumps(ir.get('metadata', {}), ensure_ascii=False, indent=1)[:6000])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
