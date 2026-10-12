"""Schema arithmetic ceiling for component detection recall (opt90-comp).

GT-ACCESS: diagnostic_only.

The official output schema keys components by designator; the Q&A convention
adds one rule (V/Gnd use the empty-set key '∅', assigned by
_assign_qna_special_coordinate_keys). Everything else needs a bound designator
text. This script counts, per val30 GT component, whether it is representable
at all (V/Gnd category OR has a designator in the annotation) and therefore
what recall ceiling the 1112-box denominator imposes.

Usage: schema_ceiling.py <out.json>
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path('/home/data/yinjx/pcbparse-competition/project')
sys.path.insert(0, str(ROOT / 'src'))

from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth  # noqa: E402

DATASET = ROOT / 'Dataset300' / 'Dataset300'
MANIFEST = ROOT / 'work/opt90-lead/eval-s6fix/s6fix-eval-manifest.jsonl'


def main() -> int:
    out_path = Path(sys.argv[1])
    rows = [json.loads(x) for x in MANIFEST.read_text(encoding='utf-8').splitlines() if x.strip()]
    total = Counter()
    by_cat = {}
    per_case = []
    for row in sorted(rows, key=lambda r: r['case_id']):
        case = row['case_id']
        ann = sorted((DATASET / case).glob('*_CVJsonStd.json'))
        annotation = json.loads(ann[0].read_text(encoding='utf-8'))
        gt = build_raw_ground_truth(annotation, case_id=case, annotation_path=ann[0],
                                    mask_policy='qna-v1')
        c = Counter()
        for t in gt.components:
            if not (t.mask.eligible and t.bbox is not None):
                continue
            has_des = bool(t.designator.value and str(t.designator.value).strip()) and t.designator.mask.eligible
            is_power = t.category in ('V', 'Gnd')
            key = ('power_' if is_power else '') + ('desig' if has_des else 'nodesig')
            c[key] += 1
            c['total'] += 1
            slot = by_cat.setdefault(t.category, Counter())
            slot['total'] += 1
            slot['representable' if (is_power or has_des) else 'unrepresentable'] += 1
        total.update(c)
        per_case.append({'case': case, **dict(c)})
    rep = total['power_desig'] + total['power_nodesig'] + total['desig']
    out = {
        'schema': 'opt90-comp-schema-ceiling-v1',
        'gt_access': 'diagnostic_only',
        'gt_total': total['total'],
        'power_with_designator': total['power_desig'],
        'power_without_designator': total['power_nodesig'],
        'nonpower_with_designator': total['desig'],
        'nonpower_without_designator': total['nodesig'],
        'representable_total': rep,
        'schema_recall_ceiling': round(rep / max(1, total['total']), 4),
        'by_category': {k: dict(v) for k, v in sorted(by_cat.items())},
        'per_case': per_case,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in out.items() if k not in ('by_category', 'per_case')},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
