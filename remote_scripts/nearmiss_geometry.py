"""Near-miss box geometry: systematic convention bias vs random error (opt90-comp).

GT-ACCESS: diagnostic_only.

For every FN with a near-miss pred box (best IoU in [0.1, 0.5)) and for every
TP match, compute edge deltas (pred - GT), size ratios and centre offset.
A consistent sign across cases => systematic box-convention mismatch
(addressable by label convention / box regression); scattered deltas =>
genuine localization noise.

Usage: nearmiss_geometry.py <eval-manifest.jsonl> <out.json>
"""
from __future__ import annotations

import json
import math
import sys
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


def stats(vals):
    if not vals:
        return None
    vals = sorted(vals)
    def q(p):
        return round(vals[min(len(vals) - 1, int(p * len(vals)))], 2)
    return {'n': len(vals), 'p10': q(0.10), 'p25': q(0.25), 'p50': q(0.50),
            'p75': q(0.75), 'p90': q(0.90),
            'mean': round(sum(vals) / len(vals), 2),
            'frac_neg': round(sum(1 for v in vals if v < 0) / len(vals), 3)}


def main() -> int:
    manifest = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    rows = [json.loads(x) for x in manifest.read_text(encoding='utf-8').splitlines() if x.strip()]
    near_rows = []
    tp_rows = []
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
        expected = [c for c in gt.components if c.mask.eligible and c.bbox is not None]
        gt_by_id = {c.raw_id: c for c in expected}
        pred_boxes = {
            p.internal_id: (float(p.bbox.x1), float(p.bbox.y1), float(p.bbox.x2), float(p.bbox.y2))
            for p in ir.components
        }
        # TP pairs
        for pid, gid in component_map.items():
            pb = pred_boxes[pid]
            gb = tuple(float(v) for v in gt_by_id[gid].bbox)
            tp_rows.append(make_row(case, gid, gb, pb, iou(gb, pb)))
        matched_gt = set(component_map.values())
        for target in expected:
            if target.raw_id in matched_gt:
                continue
            gb = tuple(float(v) for v in target.bbox)
            best, best_i = None, 0.0
            for pid, pb in pred_boxes.items():
                s = iou(gb, pb)
                if s > best_i:
                    best, best_i = pb, s
            if best is not None and best_i >= 0.1:
                near_rows.append(make_row(case, target.raw_id, gb, best, best_i))

    def agg(rows):
        out = {}
        for key in ('dx1', 'dy1', 'dx2', 'dy2', 'dw', 'dh', 'w_ratio', 'h_ratio',
                    'center_dx', 'center_dy', 'iou'):
            out[key] = stats([r[key] for r in rows])
        out['signed_area_ratio'] = stats([
            (r['w_ratio'] * r['h_ratio']) for r in rows])
        return out

    near_rows.sort(key=lambda r: -r['iou'])
    out = {
        'schema': 'opt90-comp-nearmiss-geometry-v1',
        'gt_access': 'diagnostic_only',
        'near_miss_count': len(near_rows),
        'tp_count': len(tp_rows),
        'near_miss_agg': agg(near_rows),
        'tp_agg': agg(tp_rows),
        'near_miss_records': near_rows,
        'tp_iou_histogram': {
            '0.5-0.55': sum(1 for r in tp_rows if r['iou'] < 0.55),
            '0.55-0.65': sum(1 for r in tp_rows if 0.55 <= r['iou'] < 0.65),
            '0.65-0.8': sum(1 for r in tp_rows if 0.65 <= r['iou'] < 0.8),
            '0.8-1.0': sum(1 for r in tp_rows if r['iou'] >= 0.8),
        },
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    brief = {k: v for k, v in out.items() if k not in ('near_miss_records', 'tp_agg')}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    return 0


def make_row(case, gid, gb, pb, i):
    dw = pb[2] - pb[0]
    dh = pb[3] - pb[1]
    gw = gb[2] - gb[0]
    gh = gb[3] - gb[1]
    return {
        'case': case, 'gt_id': gid, 'iou': round(i, 4),
        'gt_bbox': [round(v, 1) for v in gb],
        'pred_bbox': [round(v, 1) for v in pb],
        'dx1': round(pb[0] - gb[0], 2), 'dy1': round(pb[1] - gb[1], 2),
        'dx2': round(pb[2] - gb[2], 2), 'dy2': round(pb[3] - gb[3], 2),
        'dw': round(dw - gw, 2), 'dh': round(dh - gh, 2),
        'w_ratio': round(dw / gw, 3) if gw else None,
        'h_ratio': round(dh / gh, 3) if gh else None,
        'center_dx': round((pb[0] + pb[2]) / 2 - (gb[0] + gb[2]) / 2, 2),
        'center_dy': round((pb[1] + pb[3]) / 2 - (gb[1] + gb[3]) / 2, 2),
    }


if __name__ == '__main__':
    raise SystemExit(main())
