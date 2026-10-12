"""Category x case-group recall strata (opt90-comp, diagnostic). GT-ACCESS: diagnostic_only.

Splits val30 into the six detector-collapse sheets and the rest, and reports
per-category GT/TP/FN/recall for each group.

Usage: group_strata.py <eval-manifest.jsonl> <out.json>
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/home/data/yinjx/pcbparse-competition/project')
sys.path.insert(0, str(ROOT / 'src'))

from pcbparse.evaluation.masked_metrics import _component_metrics  # noqa: E402
from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth  # noqa: E402
from pcbparse.ir.models import CircuitIR  # noqa: E402

DATASET = ROOT / 'Dataset300' / 'Dataset300'
DISASTER = ('0089', '0091', '0092', '0234', '0280', '0283')


def main() -> int:
    manifest = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    rows = [json.loads(x) for x in manifest.read_text(encoding='utf-8').splitlines() if x.strip()]
    gt_g = defaultdict(Counter)
    tp_g = defaultdict(Counter)
    for row in sorted(rows, key=lambda r: r['case_id']):
        case = row['case_id']
        group = 'disaster' if case in DISASTER else 'normal'
        ir_path = Path(row['ir_path'])
        if not ir_path.is_absolute():
            ir_path = (manifest.parent / ir_path).resolve(strict=True)
        ir = CircuitIR.model_validate_json(ir_path.read_text(encoding='utf-8'))
        ann = sorted((DATASET / case).glob('*_CVJsonStd.json'))
        annotation = json.loads(ann[0].read_text(encoding='utf-8'))
        gt = build_raw_ground_truth(annotation, case_id=case, annotation_path=ann[0],
                                    mask_policy='qna-v1')
        _rep, component_map = _component_metrics(ir, gt, threshold=0.5)
        matched = set(component_map.values())
        for t in gt.components:
            if not (t.mask.eligible and t.bbox is not None):
                continue
            gt_g[group][t.category] += 1
            if t.raw_id in matched:
                tp_g[group][t.category] += 1
    out = {}
    for group in ('disaster', 'normal'):
        tab = {}
        for cat in sorted(gt_g[group]):
            g = gt_g[group][cat]
            t = tp_g[group][cat]
            tab[cat] = {'gt': g, 'tp': t, 'fn': g - t, 'recall': round(t / g, 4)}
        out[group] = {
            'total_gt': sum(gt_g[group].values()),
            'total_tp': sum(tp_g[group].values()),
            'recall': round(sum(tp_g[group].values()) / max(1, sum(gt_g[group].values())), 4),
            'by_category': tab,
        }
    out_path.write_text(json.dumps({'schema': 'opt90-comp-group-strata-v1',
                                    'gt_access': 'diagnostic_only', **out},
                                   ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
