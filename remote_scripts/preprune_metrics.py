"""Pre-prune detection metrics + S7 owner reclassification (opt90-comp).

GT-ACCESS: diagnostic_only.

Rebuilds the perception-phase component set (score gate + class-aware NMS +
cap, exactly the production component phase, primary 512/64 scale) from the
raw detector dumps and computes detection P/R/F1 BEFORE the degraded-submission
key prune, then reclassifies the 40 S7 detection-bound owner boxes.

Sources: work/opt90-comp/quick-raw-<case>.jsonl (preferred) and
work/opt90-comp/raw-candidates.jsonl (full sweep, primary scale rows).

Usage: preprune_metrics.py <fn-attribution.json> <s7-attribution.json> <out.json>
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path('/home/data/yinjx/pcbparse-competition/project')
sys.path.insert(0, str(ROOT / 'src'))

from pcbparse.evaluation.masked_metrics import _component_metrics  # noqa: E402
from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth  # noqa: E402
from pcbparse.ir.models import BBox, CircuitIR, Component, Evidence  # noqa: E402
from pcbparse.perception.composite_provider import _class_aware_nms  # noqa: E402

DATASET = ROOT / 'Dataset300' / 'Dataset300'
CONFIG = ROOT / 'work/module-opt80-20261007/r12-integration/pipeline-integration.yaml'
MANIFEST = ROOT / 'work/opt90-lead/eval-s6fix/s6fix-eval-manifest.jsonl'


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_raw(case):
    rows = []
    quick = ROOT / 'work/opt90-comp' / f'quick-raw-{case}.jsonl'
    if quick.exists():
        rows = [json.loads(x) for x in quick.read_text(encoding='utf-8').splitlines() if x.strip()]
    else:
        full = ROOT / 'work/opt90-comp' / 'raw-candidates.jsonl'
        if full.exists():
            rows = [json.loads(x) for x in full.read_text(encoding='utf-8').splitlines()
                    if x.strip() and json.loads(x)['case'] == case]
    return [r for r in rows if r['scale'] == 'primary']


def make_component(raw, idx):
    return Component(
        internal_id=f'pre:{raw["case"]}:{idx}',
        key=None,
        category=str(raw['category']),
        name='', value='',
        bbox=BBox(x1=raw['bbox'][0], y1=raw['bbox'][1], x2=raw['bbox'][2], y2=raw['bbox'][3]),
        score=float(raw['score']),
        key_score=0.0,
        evidence=[Evidence(evidence_id='preprune', source='detector', view_id='tile')],
    )


def main() -> int:
    fn_path = Path(sys.argv[1])
    s7_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])
    fn_all = json.loads(fn_path.read_text(encoding='utf-8'))
    s7 = json.loads(s7_path.read_text(encoding='utf-8'))
    s7_wanted = {(r['case'], r['true_owner']) for r in s7['records'] if not r['true_owner_detected']}

    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))['image']
    gate = float(cfg['component_candidate_min_score'])
    nms_iou = float(cfg['component_nms_iou'])
    cap = int(cfg['component_max_candidates'])

    rows = [json.loads(x) for x in MANIFEST.read_text(encoding='utf-8').splitlines() if x.strip()]
    total = Counter()
    per_case = []
    s7_rows = []
    fn_reclass = Counter()
    covered_cases = set()
    for row in sorted(rows, key=lambda r: r['case_id']):
        case = row['case_id']
        raws = load_raw(case)
        if not raws:
            per_case.append({'case': case, 'raw_available': False})
            continue
        covered_cases.add(case)
        kept = [make_component(r, i) for i, r in enumerate(raws) if r['score'] >= gate]
        kept = _class_aware_nms(kept, nms_iou)
        if len(kept) > cap:
            kept = sorted(kept, key=lambda c: -c.score)[:cap]
        ir_path = Path(row['ir_path'])
        clone = CircuitIR.model_validate_json(ir_path.read_text(encoding='utf-8'))
        clone.components = kept
        ann = sorted((DATASET / case).glob('*_CVJsonStd.json'))
        annotation = json.loads(ann[0].read_text(encoding='utf-8'))
        gt = build_raw_ground_truth(annotation, case_id=case, annotation_path=ann[0],
                                    mask_policy='qna-v1')
        rep, component_map = _component_metrics(clone, gt, threshold=0.5)
        d = rep['detection']
        total['tp'] += d['tp']; total['fp'] += d['fp']; total['fn'] += d['fn']
        per_case.append({'case': case, 'raw_available': True, 'kept': len(kept),
                         'tp': d['tp'], 'fp': d['fp'], 'fn': d['fn'],
                         'recall': round(d['recall'], 4), 'precision': round(d['precision'], 4)})
        matched = set(component_map.values())
        boxes = {c.internal_id: (c.bbox.x1, c.bbox.y1, c.bbox.x2, c.bbox.y2) for c in kept}
        for (c, owner) in sorted(s7_wanted):
            if c != case:
                continue
            gt_box = None
            for t in gt.components:
                if t.raw_id == owner and t.bbox is not None:
                    gt_box = tuple(float(v) for v in t.bbox)
            if gt_box is None:
                continue
            best = max((iou(gt_box, b) for b in boxes.values()), default=0.0)
            s7_rows.append({'case': case, 'owner': owner,
                            'preprune_matched': owner in matched,
                            'preprune_best_iou': round(best, 3)})
        for r in fn_all['fn_records']:
            if r['case'] != case:
                continue
            gb = tuple(r['gt_bbox'])
            best = max((iou(gb, b) for b in boxes.values()), default=0.0)
            cls = 'preprune_tp_window' if best >= 0.5 else (
                'preprune_near' if best >= 0.1 else 'preprune_blind')
            fn_reclass[cls] += 1

    tp, fp, fn = total['tp'], total['fp'], total['fn']
    s7_cov = sum(1 for r in s7_rows if r['preprune_matched'])
    out = {
        'schema': 'opt90-comp-preprune-metrics-v1',
        'gt_access': 'diagnostic_only',
        'cases_with_raw': sorted(covered_cases),
        'preprune_detection': {
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(tp / max(1, tp + fp), 4),
            'recall': round(tp / max(1, tp + fn), 4),
            'f1': round(2 * tp / max(1, 2 * tp + fp + fn), 4),
        },
        'postprune_baseline': {'tp': 616, 'fp': 99, 'fn': 496,
                               'precision': 0.8615, 'recall': 0.5540},
        'fn_reclassification': dict(fn_reclass),
        's7_owners_preprune': {
            'total': len(s7_rows),
            'covered_iou50': s7_cov,
            'records': s7_rows,
        },
        'per_case': per_case,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    brief = {k: v for k, v in out.items() if k not in ('per_case', 's7_owners_preprune')}
    brief['s7_owners_preprune'] = {k: v for k, v in out['s7_owners_preprune'].items() if k != 'records'}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
