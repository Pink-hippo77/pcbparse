"""Dump IR components for one case (opt90-comp diagnostic). Usage: dump_ir.py <case>
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path('/home/data/yinjx/pcbparse-competition/project')
sys.path.insert(0, str(ROOT / 'src'))

from pcbparse.ir.models import CircuitIR  # noqa: E402

MANIFEST = ROOT / 'work/opt90-lead/eval-s6fix/s6fix-eval-manifest.jsonl'


def main() -> int:
    import json
    case = sys.argv[1]
    rows = [json.loads(x) for x in MANIFEST.read_text(encoding='utf-8').splitlines() if x.strip()]
    row = next(r for r in rows if r['case_id'] == case)
    ir = CircuitIR.model_validate_json(Path(row['ir_path']).read_text(encoding='utf-8'))
    print('components:', len(ir.components))
    for c in ir.components:
        b = [round(float(v), 1) for v in (c.bbox.x1, c.bbox.y1, c.bbox.x2, c.bbox.y2)]
        ev = [(e.source, e.view_id) for e in c.evidence]
        print(c.internal_id, c.category, round(c.score, 4), repr(c.key), b, ev)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
