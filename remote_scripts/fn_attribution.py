"""Component detection FN stratified attribution (goal line D, opt90-comp).

GT-ACCESS: diagnostic_only. GT is read only inside this offline diagnostic
script; nothing here feeds inference or training input.

For every eligible GT component box of the val30 stack (run-s6fix eval IRs):
  * reproduce the production matching (_component_metrics, IoU>=0.5,
    category-agnostic maximum-cardinality);
  * for each false negative, measure the best geometric relation to ANY
    predicted component box and classify:
      - competing_match : >=1 pred with IoU>=0.5 exists but that pred was
                          matched to a different GT (duplicate/merged box);
      - near_miss       : best IoU in [0.1, 0.5)   (geometric misalignment);
      - weak_overlap    : best IoU in (0, 0.1);
      - no_overlap      : no pred box overlaps the GT at all.
  * stratify by GT category, GT size bucket, case / source domain;
  * join the S7 owner-binding 40 detection-bound records;
  * pair near-miss FNs with their best pred and test pred.key vs GT designator
    (evidence that the detector found the same physical object).

Usage: fn_attribution.py <eval-manifest.jsonl> <s7-attribution.json> <out.json>
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


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def size_bucket(bbox):
    side = math.sqrt(max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
    for edge, name in ((20, 'a_lt20'), (40, 'b_20-40'), (80, 'c_40-80'),
                       (160, 'd_80-160')):
        if side < edge:
            return name
    return 'e_ge160'


def main() -> int:
    manifest = Path(sys.argv[1])
    s7_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])
    rows = [json.loads(x) for x in manifest.read_text(encoding='utf-8').splitlines() if x.strip()]
    s7 = json.loads(s7_path.read_text(encoding='utf-8'))
    s7_index = {(r['case'], r['true_owner']): r for r in s7['records']}

    fn_records = []
    pair_records = []
    per_case = []
    totals = Counter()

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
        report, component_map = _component_metrics(ir, gt, threshold=0.5)
        matched_gt = set(component_map.values())
        pred_to_gt = dict(component_map)

        expected = [c for c in gt.components if c.mask.eligible and c.bbox is not None]
        predicted = list(ir.components)
        pred_boxes = [
            (float(p.bbox.x1), float(p.bbox.y1), float(p.bbox.x2), float(p.bbox.y2))
            for p in predicted
        ]

        case_fn = 0
        for target in expected:
            gt_box = tuple(float(v) for v in target.bbox)
            if target.raw_id in matched_gt:
                continue
            case_fn += 1
            best_iou, best_idx = 0.0, -1
            comp_iou, comp_idx = 0.0, -1
            for idx, pb in enumerate(pred_boxes):
                s = iou(gt_box, pb)
                if s > best_iou:
                    best_iou, best_idx = s, idx
                if s >= 0.5 and s > comp_iou:
                    comp_iou, comp_idx = s, idx
            if comp_idx >= 0:
                cls = 'competing_match'
                best_iou, best_idx = comp_iou, comp_idx
            elif best_iou >= 0.1:
                cls = 'near_miss'
            elif best_iou > 0.0:
                cls = 'weak_overlap'
            else:
                cls = 'no_overlap'
            best_pred = predicted[best_idx] if best_idx >= 0 else None
            gcx = (gt_box[0] + gt_box[2]) / 2.0
            gcy = (gt_box[1] + gt_box[3]) / 2.0
            if best_pred is not None:
                pb = pred_boxes[best_idx]
                pcx = (pb[0] + pb[2]) / 2.0
                pcy = (pb[1] + pb[3]) / 2.0
                center_dist = math.hypot(pcx - gcx, pcy - gcy)
            else:
                center_dist = None
            designator = target.designator.value
            s7_key = (case, target.raw_id)
            rec = {
                'case': case,
                'gt_id': target.raw_id,
                'gt_category': target.category,
                'gt_designator': designator,
                'gt_bbox': [round(v, 1) for v in gt_box],
                'gt_side_px': round(math.sqrt(max(1.0, (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]))), 1),
                'size_bucket': size_bucket(gt_box),
                'cls': cls,
                'best_iou': round(best_iou, 4),
                'best_center_dist_px': None if center_dist is None else round(center_dist, 1),
                'best_pred': None if best_pred is None else {
                    'id': best_pred.internal_id,
                    'score': round(float(best_pred.score), 4),
                    'category': best_pred.category,
                    'key': best_pred.key,
                    'key_score': round(float(best_pred.key_score), 4),
                },
                'best_pred_matched_to': (
                    pred_to_gt.get(best_pred.internal_id) if best_pred is not None else None
                ),
                's7_owner_case': s7_key in s7_index,
            }
            if best_pred is not None and best_pred.key and designator:
                rec['key_matches_gt_designator'] = (
                    str(best_pred.key).strip().casefold() == str(designator).strip().casefold()
                )
            fn_records.append(rec)
            totals[cls] += 1

        # FP <-> FN pairing: predicted box not matched to any GT
        for idx, pred in enumerate(predicted):
            if pred.internal_id in pred_to_gt:
                continue
            pb = pred_boxes[idx]
            best_iou, best_j = 0.0, -1
            for j, target in enumerate(expected):
                s = iou(pb, tuple(float(v) for v in target.bbox))
                if s > best_iou:
                    best_iou, best_j = s, j
            pair_records.append({
                'case': case,
                'pred_id': pred.internal_id,
                'score': round(float(pred.score), 4),
                'category': pred.category,
                'key': pred.key,
                'best_gt': None if best_j < 0 else expected[best_j].raw_id,
                'best_gt_iou': round(best_iou, 4),
                'best_gt_designator': None if best_j < 0 else expected[best_j].designator.value,
            })

        det = report['detection']
        per_case.append({
            'case': case,
            'source_domain': gt.source_domain,
            'gt_total': len(expected),
            'tp': det['tp'],
            'fp': det['fp'],
            'fn': det['fn'],
            'fn_computed': case_fn,
            'recall': round(det['recall'], 4),
        })

    # ---- aggregates ----
    by_category = defaultdict(Counter)
    by_size = defaultdict(Counter)
    by_case = defaultdict(Counter)
    for r in fn_records:
        by_category[r['gt_category']][r['cls']] += 1
        by_size[r['size_bucket']][r['cls']] += 1
        by_case[r['case']][r['cls']] += 1

    # coverage at looser IoU (category-agnostic best-overlap lens)
    def cover(edge):
        n = 0
        for r in fn_records:
            if r['best_iou'] >= edge or r['cls'] == 'competing_match':
                n += 1
        return n

    s7_fn = [r for r in fn_records if r['s7_owner_case']]
    key_matched = [r for r in fn_records if r.get('key_matches_gt_designator')]
    fp_key_hits = [
        p for p in pair_records
        if p['key'] and p['best_gt_designator']
        and str(p['key']).strip().casefold() == str(p['best_gt_designator']).strip().casefold()
    ]
    tp0 = per_case and sum(c['tp'] for c in per_case)
    fn0 = sum(c['fn'] for c in per_case)
    fp0 = sum(c['fp'] for c in per_case)

    near = totals['near_miss'] + totals['competing_match']
    out = {
        'schema': 'opt90-comp-fn-attribution-v1',
        'gt_access': 'diagnostic_only',
        'manifest': str(manifest),
        'totals': {
            'tp': tp0, 'fp': fp0, 'fn': fn0,
            'fn_class': dict(totals),
            'recall_iou50': round(tp0 / max(1, tp0 + fn0), 4),
        },
        'fn_recoverable_lens': {
            'fn_with_any_pred_overlap': cover(0.000001),
            'fn_best_iou_ge_0_1': cover(0.1),
            'fn_best_iou_ge_0_3': cover(0.3),
            'fn_competing_or_near_miss': near,
            'recall_ceiling_if_all_ge_0_1_recovered': round((tp0 + cover(0.1)) / max(1, tp0 + fn0), 4),
            'recall_ceiling_if_all_near_recovered': round((tp0 + near) / max(1, tp0 + fn0), 4),
        },
        'fn_by_category': {k: dict(v) for k, v in sorted(by_category.items())},
        'fn_by_size': {k: dict(v) for k, v in sorted(by_size.items())},
        'fn_by_case': {k: dict(v) for k, v in sorted(by_case.items())},
        'key_pairing': {
            'fn_best_pred_key_equals_gt_designator': len(key_matched),
            'fp_key_equals_best_gt_designator': len(fp_key_hits),
        },
        's7_owner_40': {
            'joined': len(s7_fn),
            'fn_class': dict(Counter(r['cls'] for r in s7_fn)),
            'by_category': dict(Counter(r['gt_category'] for r in s7_fn)),
            'by_size': dict(Counter(r['size_bucket'] for r in s7_fn)),
            'records': s7_fn,
        },
        'per_case': per_case,
        'fn_records': fn_records,
        'fp_pair_records': pair_records,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    brief = {k: v for k, v in out.items()
             if k in ('totals', 'fn_recoverable_lens', 'fn_by_category', 'fn_by_size',
                      'key_pairing', 's7_owner_40')}
    brief['s7_owner_40'] = {k: v for k, v in brief['s7_owner_40'].items() if k != 'records'}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
