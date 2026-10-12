"""Quick raw probe for selected cases (opt90-comp). GT-ACCESS: diagnostic_only.

Same raw-candidate dump as raw_detector_probe.py but restricted to the named
cases and without the FN join, so the decisive disaster-sheet question
(model fires below gate vs never fires) can be answered before the full sweep
finishes.

Usage: quick_probe.py <case> [<case> ...]   -> work/opt90-comp/quick-raw-<case>.jsonl
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path('/home/data/yinjx/pcbparse-competition/project')
sys.path.insert(0, str(ROOT / 'src'))

from pcbparse.io import load_image, tile_image_view  # noqa: E402
from pcbparse.perception.composite_provider import _restore_components  # noqa: E402
from pcbparse.perception.torch_component_adapter import factory  # noqa: E402

CONFIG = ROOT / 'work/module-opt80-20261007/r12-integration/pipeline-integration.yaml'
LOCK = ROOT / 'models/models.lock.json'
MANIFEST = ROOT / 'work/module-relative10-20261005-v1/production-lock-val30-20261006/input_manifest.json'


def main() -> int:
    wanted = set(sys.argv[1:])
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))['image']
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    entry = next(m for m in lock['models'] if m.get('task') == 'component')
    entry = dict(entry)
    inference = dict(entry.get('inference') or {})
    inference['device'] = 'cpu'
    entry['inference'] = inference
    adapter = factory(ROOT / 'models' / entry['path'], entry)
    rows = json.loads(MANIFEST.read_text(encoding='utf-8'))
    scales = [
        ('primary', int(cfg['component_tile_size']), int(cfg['component_tile_overlap'])),
        ('fallback', int(cfg['component_fallback_tile_size']), int(cfg['component_fallback_tile_overlap'])),
    ]
    for row in sorted(rows, key=lambda r: r['case_id']):
        case = row['case_id']
        if case not in wanted:
            continue
        t0 = time.time()
        original = load_image(row['image_path'])
        out = (ROOT / 'work/opt90-comp' / f'quick-raw-{case}.jsonl').open('w', encoding='utf-8')
        for scale_name, tile_size, overlap in scales:
            views = tile_image_view(original, tile_size=tile_size, overlap=overlap)
            for view in views:
                for item in _restore_components(adapter.predict(view, None), view,
                                                original_size=original.size,
                                                internal_border_margin_px=32.0):
                    if item.score < 0.02:
                        continue
                    out.write(json.dumps({
                        'case': case, 'scale': scale_name,
                        'score': round(float(item.score), 4), 'category': item.category,
                        'bbox': [round(float(v), 1) for v in
                                 (item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2)],
                    }) + '\n')
        out.close()
        print(f'{case} done {time.time() - t0:.0f}s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
