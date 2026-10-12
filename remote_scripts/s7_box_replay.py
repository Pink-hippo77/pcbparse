"""S7 owner-box recovery under deterministic box transforms (opt90-comp).

GT-ACCESS: diagnostic_only. For each fixed transform of the predicted component
boxes, recompute the production matching and count how many of the 40
detection-bound S7 owner GT boxes become matched (component_map values).

Usage: s7_box_replay.py <eval-manifest.jsonl> <s7-attribution.json> <out.json>
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
    return (cx - w / 2.0 - margin, cy - h / 2.0 - margin,
            cx + w / 2.0 + margin, cy + h / 2.0 + margin)


def main() -> int:
    manifest = Path(sys.argv[1])
    s7_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])
    s7 = json.loads(s7_path.read_text(encoding='utf-8'))
    wanted = {(r['case'], r['true_owner']) for r in s7['records']
              if not r['true_owner_detected']}
    rows = [json.loads(x) for x in manifest.read_text(encoding='utf-8').splitlines() if x.strip()]
    cases = []
    for row in sorted(rows, key=lambda r: r['case_id']):
        case = row['case_id']
        if not any(c == case for c, _o in wanted):
            continue
        ir_path = Path(row['ir_path'])
        if not ir_path.is_absolute():
            ir_path = (manifest.parent / ir_path).resolve(strict=True)
        ir = CircuitIR.model_validate_json(ir_path.read_text(encoding='utf-8'))
        ann = sorted((DATASET / case).glob('*_CVJsonStd.json'))
        annotation = json.loads(ann[0].read_text(encoding='utf-8'))
        gt = build_raw_ground_truth(annotation, case_id=case, annotation_path=ann[0],
                                    mask_policy='qna-v1')
        cases.append((case, ir, gt))

    variants = [('baseline', 1.0, 0.0), ('scale_1.03', 1.03, 0.0),
                ('scale_1.10', 1.10, 0.0), ('margin_1', 1.0, 1.0),
                ('margin_2', 1.0, 2.0), ('margin_1_scale_1.05', 1.05, 1.0)]
    out = {'schema': 'opt90-comp-s7-box-replay-v1', 'gt_access': 'diagnostic_only',
           'owner_total': len(wanted), 'variants': {}}
    for name, s, m in variants:
        recovered = []
        for case, ir, gt in cases:
            clone = ir.model_copy(deep=True)
            for comp in clone.components:
                b = (float(comp.bbox.x1), float(comp.bbox.y1),
                     float(comp.bbox.x2), float(comp.bbox.y2))
                nb = transform(b, scale=s, margin=m)
                comp.bbox.x1, comp.bbox.y1, comp.bbox.x2, comp.bbox.y2 = nb
            _rep, component_map = _component_metrics(clone, gt, threshold=0.5)
            matched = set(component_map.values())
            for (c, owner) in sorted(wanted):
                if c == case and owner in matched:
                    recovered.append(f'{c}:{owner}')
        out['variants'][name] = {'scale': s, 'margin': m,
                                 'owners_recovered': len(recovered),
                                 'recovered': recovered}
        print(name, len(recovered), recovered, flush=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
