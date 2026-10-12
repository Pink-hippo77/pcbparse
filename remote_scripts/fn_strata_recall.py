"""FN strata with base rates: recall by GT category / size bucket (opt90-comp).

GT-ACCESS: diagnostic_only.

Adds TP-side strata to fn-attribution.json so every FN count can be read as a
recall (FN share alone is confounded by the population distribution).

Usage: fn_strata_recall.py <eval-manifest.jsonl> <fn-attribution.json> <out.json>
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/home/data/yinjx/pcbparse-competition/project')
sys.path.insert(0, str(ROOT / 'src'))

from pcbparse.evaluation.masked_metrics import _component_metrics  # noqa: E402
from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth  # noqa: E402
from pcbparse.ir.models import CircuitIR  # noqa: E402

DATASET = ROOT / 'Dataset300' / 'Dataset300'


def size_bucket(bbox):
    side = math.sqrt(max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
    for edge, name in ((20, 'a_lt20'), (40, 'b_20-40'), (80, 'c_40-80'),
                       (160, 'd_80-160')):
        if side < edge:
            return name
    return 'e_ge160'


def main() -> int:
    manifest = Path(sys.argv[1])
    fn_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])
    fn_all = json.loads(fn_path.read_text(encoding='utf-8'))
    rows = [json.loads(x) for x in manifest.read_text(encoding='utf-8').splitlines() if x.strip()]

    gt_cat, tp_cat = Counter(), Counter()
    gt_size, tp_size = Counter(), Counter()
    gt_case = Counter()
    designator_text = {}
    for row in sorted(rows, key=lambda r: r['case_id']):
        case = row['case_id']
        ir_path = Path(row['ir_path'])
        if not ir_path.is_absolute():
            ir_path = (manifest.parent / ir_path).resolve(strict=True)
        ir = CircuitIR.model_validate_json(ir_path.read_text(encoding='utf-8'))
        ann = sorted((DATASET / case).glob('*_CVJsonStd.json'))
        annotation = json.loads(ann[0].read_text(encoding='utf-8'))
        gt = build_raw_ground_truth(annotation, case_id=case, annotation_path=ann[0],
                                    mask_policy='qna-v1')
        _rep, component_map = _component_metrics(ir, gt, threshold=0.5)
        matched_gt = set(component_map.values())
        for target in gt.components:
            if not (target.mask.eligible and target.bbox is not None):
                continue
            gt_cat[target.category] += 1
            gt_size[size_bucket(tuple(float(v) for v in target.bbox))] += 1
            gt_case[case] += 1
            designator_text[(case, target.raw_id)] = target.designator.value
            if target.raw_id in matched_gt:
                tp_cat[target.category] += 1
                tp_size[size_bucket(tuple(float(v) for v in target.bbox))] += 1

    def table(counter_gt, counter_tp):
        out = {}
        for k in sorted(counter_gt):
            g = counter_gt[k]
            t = counter_tp[k]
            out[k] = {'gt': g, 'tp': t, 'fn': g - t,
                      'recall': round(t / g, 4) if g else None}
        return out

    # S7 40: what text does each missed owner box carry?
    s7_rows = fn_all['s7_owner_40']['records']
    s7_texts = [
        {'case': r['case'], 'gt_id': r['gt_id'], 'cls': r['cls'],
         'gt_designator': r['gt_designator'],
         'gt_category': r['gt_category'], 'gt_side_px': r['gt_side_px'],
         'best_pred': r['best_pred'], 'best_iou': r['best_iou']}
        for r in s7_rows
    ]
    out = {
        'schema': 'opt90-comp-fn-strata-v1',
        'gt_access': 'diagnostic_only',
        'by_category': table(gt_cat, tp_cat),
        'by_size': table(gt_size, tp_size),
        'gt_per_case': dict(sorted(gt_case.items())),
        's7_owner_40_detail': s7_texts,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({'by_category': out['by_category'], 'by_size': out['by_size']},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
