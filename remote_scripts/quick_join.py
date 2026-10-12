"""Quick-probe join (opt90-comp). GT-ACCESS: diagnostic_only.

For the quick-probe cases, join raw candidates against every GT component of
the case (from fn-attribution.json's FN records plus re-derived TPs are NOT
needed: we only classify the FN set), reporting:
  * score band distribution of raw primary candidates;
  * per FN: best raw overlap (any class) at each scale;
  * for zero-recall classes: predicted raw class.

Usage: quick_join.py <case> [<case> ...]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path('/home/data/yinjx/pcbparse-competition/project')
FN = ROOT / 'work/opt90-comp/fn-attribution.json'


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    fn_all = json.loads(FN.read_text(encoding='utf-8'))
    for case in sys.argv[1:]:
        raws = [json.loads(x) for x in
                (ROOT / 'work/opt90-comp' / f'quick-raw-{case}.jsonl').read_text(encoding='utf-8').splitlines()
                if x.strip()]
        prim = [r for r in raws if r['scale'] == 'primary']
        bands = Counter('ge_0.5' if r['score'] >= 0.5 else '0.3-0.5' if r['score'] >= 0.3
                        else '0.1-0.3' if r['score'] >= 0.1 else '0.02-0.1' for r in prim)
        print(f'== {case}: raw primary={len(prim)} fallback={sum(1 for r in raws if r["scale"]=="fallback")} '
              f'bands={dict(bands)}', flush=True)
        fns = [r for r in fn_all['fn_records'] if r['case'] == case]
        stats = Counter()
        detail = []
        for r in fns:
            gb = tuple(r['gt_bbox'])
            best = {'primary': None, 'fallback': None}
            for raw in raws:
                s = iou(gb, tuple(raw['bbox']))
                if s >= 0.1 and (best[raw['scale']] is None or s > best[raw['scale']][0]):
                    best[raw['scale']] = (s, raw)
            p, f = best['primary'], best['fallback']
            top = p or f
            if top is None:
                stats['no_raw'] += 1
            elif top[1]['score'] >= 0.3:
                stats['raw_ge_0.3'] += 1
            else:
                stats['raw_lt_0.3'] += 1
            detail.append({
                'gt_id': r['gt_id'], 'gt_category': r['gt_category'],
                'gt_designator': r['gt_designator'], 'gt_bbox': r['gt_bbox'],
                'raw512': None if p is None else {'iou': round(p[0], 3), 'score': p[1]['score'],
                                                  'category': p[1]['category']},
                'raw256': None if f is None else {'iou': round(f[0], 3), 'score': f[1]['score'],
                                                  'category': f[1]['category']},
            })
        print('   fn_raw_summary:', dict(stats), flush=True)
        out = ROOT / 'work/opt90-comp' / f'quick-join-{case}.json'
        out.write_text(json.dumps({'case': case, 'fn_total': len(fns), 'raw_bands': dict(bands),
                                   'fn_raw_summary': dict(stats), 'detail': detail},
                                  ensure_ascii=False, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
