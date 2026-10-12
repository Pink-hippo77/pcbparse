"""Offline score-gate replay (opt90-comp, diagnostic). GT-ACCESS: diagnostic_only.

Replays the production component candidate phase from the raw detector dump
(work/opt90-comp/raw-candidates.jsonl, primary 512/64 scale): score gate ->
class-aware NMS (component_nms_iou from the pipeline config) -> per-image cap ->
production matching (IoU>=0.5, category-agnostic). Reports the P/R/F1 curve
over the gate value so the score-threshold lever can be quantified offline
before any validation run.

The 256/32 fallback scale is intentionally NOT merged (production only runs it
on sparse sheets); its potential is reported separately as `fallback_only`
rescues of the primary-set misses.

Usage: gate_replay.py <fn-attribution.json> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path('/home/data/yinjx/pcbparse-competition/project')
sys.path.insert(0, str(ROOT / 'src'))

from pcbparse.evaluation.masked_metrics import _component_metrics  # noqa: E402
from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth  # noqa: E402
from pcbparse.ir.models import BBox, CircuitIR, Component, Evidence  # noqa: E402
from pcbparse.perception.composite_provider import _class_aware_nms  # noqa: E402

DATASET = ROOT / 'Dataset300' / 'Dataset300'
RAW = ROOT / 'work/opt90-comp/raw-candidates.jsonl'
CONFIG = ROOT / 'work/module-opt80-20261007/r12-integration/pipeline-integration.yaml'
MANIFEST = ROOT / 'work/opt90-lead/eval-s6fix/s6fix-eval-manifest.jsonl'
GATES = (0.30, 0.25, 0.20, 0.15, 0.10, 0.05)


def make_component(raw):
    return Component(
        internal_id=f"raw:{raw['case']}:{raw['scale']}:{abs(hash((tuple(raw['bbox']), raw['score'], raw['category'])))}",
        key=None,
        category=str(raw['category']),
        name="",
        value="",
        bbox=BBox(x1=raw['bbox'][0], y1=raw['bbox'][1], x2=raw['bbox'][2], y2=raw['bbox'][3]),
        score=float(raw['score']),
        key_score=0.0,
        evidence=[Evidence(evidence_id='raw-replay', source='detector', view_id='raw')],
    )


def main() -> int:
    out_path = Path(sys.argv[1])
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))['image']
    nms_iou = float(cfg['component_nms_iou'])
    cap = int(cfg['component_max_candidates'])

    raw_by_case = {}
    for line in RAW.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r['scale'] != 'primary':
            continue
        raw_by_case.setdefault(r['case'], []).append(r)

    rows = [json.loads(x) for x in MANIFEST.read_text(encoding='utf-8').splitlines() if x.strip()]
    cases = []
    for row in sorted(rows, key=lambda r: r['case_id']):
        case = row['case_id']
        if case not in raw_by_case:
            continue
        ir_path = Path(row['ir_path'])
        if not ir_path.is_absolute():
            ir_path = (MANIFEST.parent / ir_path).resolve(strict=True)
        baseline_ir = CircuitIR.model_validate_json(ir_path.read_text(encoding='utf-8'))
        ann = sorted((DATASET / case).glob('*_CVJsonStd.json'))
        annotation = json.loads(ann[0].read_text(encoding='utf-8'))
        gt = build_raw_ground_truth(annotation, case_id=case, annotation_path=ann[0],
                                    mask_policy='qna-v1')
        cases.append((case, raw_by_case[case], gt, baseline_ir))

    out = {'schema': 'opt90-comp-gate-replay-v1', 'gt_access': 'diagnostic_only',
           'cases': [c for c, _r, _g, _b in cases], 'gates': {}}
    for gate in GATES:
        tp = fp = fn = 0
        for case, raws, gt, baseline_ir in cases:
            kept = [make_component(r) for r in raws if r['score'] >= gate]
            kept = _class_aware_nms(kept, nms_iou)
            if len(kept) > cap:
                kept = sorted(kept, key=lambda c: -c.score)[:cap]
            clone = baseline_ir.model_copy(deep=True)
            clone.components = kept
            rep, _ = _component_metrics(clone, gt, threshold=0.5)
            d = rep['detection']
            tp += d['tp']; fp += d['fp']; fn += d['fn']
        out['gates'][f'{gate:.2f}'] = {
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(tp / max(1, tp + fp), 4),
            'recall': round(tp / max(1, tp + fn), 4),
            'f1': round(2 * tp / max(1, 2 * tp + fp + fn), 4),
        }
        print(gate, out['gates'][f'{gate:.2f}'], flush=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
