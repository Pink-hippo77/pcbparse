"""Deterministic box-geometry replay (opt90-comp, diagnostic). GT-ACCESS: diagnostic_only.

Applies fixed, nameless transforms (centre scale / margin dilation) to the
predicted component boxes of the run-s6fix stack and recomputes the production
matching (_component_metrics, IoU>=0.5). This is an offline pre-test of the
box-convention hypothesis: the TP population shows a systematic inset
(pred box ~= 4-6% smaller than GT), so a calibrated expansion should flip the
near-threshold cluster. The full curve is reported as a diagnostic lens; no
value is selected from val30 for production without train-side evidence.

Usage: box_replay.py <eval-manifest.jsonl> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path('/home/data/yinjx/pcbparse-competition/project')
sys.path.insert(0, str(ROOT / 'src'))

from pcbparse.evaluation.masked_metrics import _component_metrics  # noqa: E402
from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth  # noqa: E402
from pcbparse.ir.models import CircuitIR  # noqa: E402

DATASET = ROOT / 'Dataset300' / 'Dataset300'


def transform(box, *, scale, margin):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    nx1, ny1 = cx - w / 2.0 - margin, cy - h / 2.0 - margin
    nx2, ny2 = cx + w / 2.0 + margin, cy + h / 2.0 + margin
    return (nx1, ny1, nx2, ny2)


def main() -> int:
    manifest = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    rows = [json.loads(x) for x in manifest.read_text(encoding='utf-8').splitlines() if x.strip()]

    cases = []
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
        expected = [c for c in gt.components if c.mask.eligible and c.bbox is not None]
        cases.append((case, ir, gt, expected))

    variants = [('baseline', 1.0, 0.0)]
    for s in (1.03, 1.06, 1.10):
        variants.append((f'scale_{s}', s, 0.0))
    for m in (1.0, 2.0, 3.0):
        variants.append((f'margin_{m}', 1.0, m))

    out = {'schema': 'opt90-comp-box-replay-v1', 'gt_access': 'diagnostic_only', 'variants': {}}
    for name, s, m in variants:
        tp = fp = fn = 0
        for case, ir, gt, expected in cases:
            clone = ir.model_copy(deep=True)
            for comp in clone.components:
                b = (float(comp.bbox.x1), float(comp.bbox.y1),
                     float(comp.bbox.x2), float(comp.bbox.y2))
                nb = transform(b, scale=s, margin=m)
                comp.bbox.x1, comp.bbox.y1, comp.bbox.x2, comp.bbox.y2 = nb
            rep, _ = _component_metrics(clone, gt, threshold=0.5)
            d = rep['detection']
            tp += d['tp']; fp += d['fp']; fn += d['fn']
        out['variants'][name] = {
            'scale': s, 'margin': m, 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(tp / max(1, tp + fp), 4),
            'recall': round(tp / max(1, tp + fn), 4),
            'f1': round(2 * tp / max(1, 2 * tp + fp + fn), 4),
        }
        print(name, out['variants'][name], flush=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
