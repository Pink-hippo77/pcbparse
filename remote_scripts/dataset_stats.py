"""Training-data category census + model category table (opt90-comp, diagnostic).

GT-ACCESS: training-side derived labels only (dataset300-v2 component_detection
manifest), split by the frozen split manifest (train240 trainable / val30 / test30).

Usage: dataset_stats.py <out.json>
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/home/data/yinjx/pcbparse-competition/project')
sys.path.insert(0, str(ROOT / 'src'))

DERIVED = ROOT / 'data/derived/dataset300-v2/component_detection.jsonl'
SPLIT = ROOT / 'data/splits/dataset300_split.jsonl'
LOCK = ROOT / 'models/models.lock.json'


def main() -> int:
    out_path = Path(sys.argv[1])
    split = {}
    for line in SPLIT.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        split[str(row.get('case_id'))] = str(row.get('split') or row.get('group') or row.get('subset'))

    per_split = defaultdict(Counter)
    per_split_images = Counter()
    per_domain = defaultdict(Counter)
    box_sizes = defaultdict(list)
    seen_case = set()
    for line in DERIVED.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        case = str(row.get('sample_id'))
        sp = str(row.get('split') or split.get(case, 'unknown'))
        domain = str(row.get('source_domain'))
        cat = str(row.get('category'))
        per_split[sp][cat] += 1
        per_domain[domain][cat] += 1
        bbox = row.get('bbox') or {}
        try:
            side = max(1.0, (float(bbox['x2']) - float(bbox['x1'])) * (float(bbox['y2']) - float(bbox['y1']))) ** 0.5
            box_sizes[sp].append(side)
        except Exception:
            pass
        if (sp, case) not in seen_case:
            seen_case.add((sp, case))
            per_split_images[sp] += 1

    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    entry = next(m for m in lock['models'] if m.get('task') == 'component')

    # category table of the shipped checkpoint (read via training helper)
    cat_table = None
    try:
        import torch
        ckpt = torch.load(ROOT / 'models' / entry['path'], map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            cat_table = ckpt.get('category_table')
    except Exception as exc:  # pragma: no cover
        cat_table = f'ERROR: {exc}'

    def pct(vals):
        vals = sorted(vals)
        if not vals:
            return None
        def q(p):
            return round(vals[min(len(vals) - 1, int(p * len(vals)))], 1)
        return {'n': len(vals), 'p10': q(0.10), 'p50': q(0.50), 'p90': q(0.90)}

    out = {
        'schema': 'opt90-comp-dataset-stats-v1',
        'gt_access': 'training_derived_manifest_only',
        'split_images': dict(per_split_images),
        'boxes_per_category': {k: dict(v.most_common()) for k, v in per_split.items()},
        'boxes_per_domain': {k: dict(v.most_common()) for k, v in per_domain.items()},
        'box_side_px': {k: pct(v) for k, v in box_sizes.items()},
        'model_category_table': cat_table,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1)[:6000])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
