"""Handover: per-component list of the 438 key-pruned components (opt90-comp).

GT-ACCESS: diagnostic_only.

Joins the pipeline's own degraded_submission_prune ledger (dropped internal
ids) with:
  * the reconstructed pre-prune component set (score gate + class-aware NMS +
    cap on the raw detector dumps; deterministic, same ordering rule as
    _assign_global_ids) to recover each dropped box where a raw dump exists;
  * the matched GT component (IoU>=0.5) and whether that GT carries an
    ANNOTATED designator (the GT governance column the Lead asked for).

Usage: handover_pruned.py <out.json> <out.md>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path('/home/data/yinjx/pcbparse-competition/project')
sys.path.insert(0, str(ROOT / 'src'))

from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth  # noqa: E402
from pcbparse.ir.models import BBox, Component, Evidence  # noqa: E402
from pcbparse.perception.composite_provider import _assign_global_ids, _class_aware_nms  # noqa: E402

DATASET = ROOT / 'Dataset300' / 'Dataset300'
CONFIG = ROOT / 'work/module-opt80-20261007/r12-integration/pipeline-integration.yaml'
MANIFEST = ROOT / 'work/opt90-lead/eval-s6fix/s6fix-eval-manifest.jsonl'
PRUNE = ROOT / 'work/opt90-comp/prune-stats.json'


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
            for line in full.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r['case'] == case:
                        rows.append(r)
    return [r for r in rows if r['scale'] == 'primary']


def main() -> int:
    out_json = Path(sys.argv[1])
    out_md = Path(sys.argv[2])
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))['image']
    gate = float(cfg['component_candidate_min_score'])
    nms_iou = float(cfg['component_nms_iou'])
    cap = int(cfg['component_max_candidates'])
    prune = json.loads(PRUNE.read_text(encoding='utf-8'))
    prune_by_case = {r['case']: r for r in prune['records']}
    rows = [json.loads(x) for x in MANIFEST.read_text(encoding='utf-8').splitlines() if x.strip()]

    records = []
    md = ['# HANDOVER — 438 个被 key 裁剪的组件（逐例清单）', '',
          'GT-ACCESS: diagnostic_only。`gt_designator_annotated` 列 = 与该框 IoU≥0.5 的 GT 是否在标注中带位号',
          '（False = 标注无位号，属 GT_GOVERNANCE 口径治理项；None = 无 GT 匹配/无框几何）。', '']
    md.append('| case | internal_id | score | category | bbox | best_gt | gt_category | gt_designator | gt_designator_annotated |')
    md.append('|---|---|---|---|---|---|---|---|---|')
    stat = {'gt_match': 0, 'gt_designator_annotated': 0, 'gt_designator_missing': 0,
            'no_gt_match': 0, 'box_reconstructed': 0, 'box_unavailable': 0}
    for row in sorted(rows, key=lambda r: r['case_id']):
        case = row['case_id']
        info = prune_by_case.get(case)
        if not info or not info.get('dropped_ids'):
            continue
        dropped_ids = list(info['dropped_ids'])
        ann = sorted((DATASET / case).glob('*_CVJsonStd.json'))
        annotation = json.loads(ann[0].read_text(encoding='utf-8'))
        gt = build_raw_ground_truth(annotation, case_id=case, annotation_path=ann[0],
                                    mask_policy='qna-v1')
        gt_boxes = [(t, tuple(float(v) for v in t.bbox)) for t in gt.components
                    if t.mask.eligible and t.bbox is not None]
        raws = load_raw(case)
        id_to_box = {}
        if raws:
            comps = []
            for i, r in enumerate(raws):
                if r['score'] < gate:
                    continue
                comps.append(Component(
                    internal_id=f'x{i}', key=None, category=str(r['category']),
                    name='', value='',
                    bbox=BBox(x1=r['bbox'][0], y1=r['bbox'][1], x2=r['bbox'][2], y2=r['bbox'][3]),
                    score=float(r['score']), key_score=0.0,
                    evidence=[Evidence(evidence_id='h', source='detector', view_id='tile')]))
            kept = _class_aware_nms(comps, nms_iou)
            if len(kept) > cap:
                kept = sorted(kept, key=lambda c: -c.score)[:cap]
            _assign_global_ids(kept, 'component')
            for c in kept:
                id_to_box[c.internal_id] = (c, raws)
        for did in dropped_ids:
            hit = id_to_box.get(did)
            if hit is None:
                stat['box_unavailable'] += 1
                records.append({'case': case, 'internal_id': did, 'box': None,
                                'best_gt': None, 'gt_category': None,
                                'gt_designator': None, 'gt_designator_annotated': None})
                md.append(f'| {case} | {did} | — | — | — | — | — | — | box_unavailable |')
                continue
            stat['box_reconstructed'] += 1
            comp = hit[0]
            box = (float(comp.bbox.x1), float(comp.bbox.y1), float(comp.bbox.x2), float(comp.bbox.y2))
            best, best_i = None, 0.0
            for t, gb in gt_boxes:
                s = iou(box, gb)
                if s > best_i:
                    best, best_i = t, s
            matched = best if best_i >= 0.5 else None
            if matched is None:
                stat['no_gt_match'] += 1
            else:
                stat['gt_match'] += 1
                des = matched.designator.value
                annotated = bool(des and str(des).strip())
                stat['gt_designator_annotated' if annotated else 'gt_designator_missing'] += 1
            des_v = matched.designator.value if matched is not None else None
            annotated_v = (bool(des_v and str(des_v).strip())) if matched is not None else None
            records.append({
                'case': case, 'internal_id': did,
                'score': round(float(comp.score), 4), 'category': comp.category,
                'bbox': [round(v, 1) for v in box],
                'best_gt': matched.raw_id if matched is not None else None,
                'best_gt_iou': round(best_i, 3),
                'gt_category': matched.category if matched is not None else (best.category if best is not None else None),
                'gt_designator': des_v,
                'gt_designator_annotated': annotated_v,
            })
            md.append(f"| {case} | {did} | {comp.score:.3f} | {comp.category} | "
                      f"{[round(v,1) for v in box]} | {matched.raw_id if matched is not None else '—'} | "
                      f"{matched.category if matched is not None else '—'} | "
                      f"{des_v if des_v else '—'} | {annotated_v} |")
    out = {'schema': 'opt90-comp-handover-pruned-v1', 'gt_access': 'diagnostic_only',
           'statistics': stat, 'records': records}
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    out_md.write_text('\n'.join(md), encoding='utf-8')
    print(json.dumps(stat, ensure_ascii=False))
    print('records:', len(records))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
