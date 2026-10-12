"""Raw component-detector probe (goal line D, opt90-comp).

GT-ACCESS: diagnostic_only. This script only READS GT to score offline raw
detector candidates; raw candidates never feed inference or training.

Replicates the production component primary pass (512/64 tiles, adapter score
gate 0.0 = everything the model emits, restored to original coordinates) and
the adaptive-fallback scale (256/32 tiles) of the best stack config
(work/module-opt80-20261007/r12-integration/pipeline-integration.yaml), then
joins every false negative from fn-attribution.json against the raw candidates:

  fired_but_dropped : raw primary candidate overlaps FN GT (IoU>=RAW_EDGE) but
                      no IR-level pred overlaps it even weakly -> the model
                      fired and the pipeline dropped/suppressed it
                      (score gate / NMS / candidate cap / match competition).
  scale_sensitive   : only the 256-scale raw pass fires -> small-object scale
                      route (data/crop/tiling), not a filter problem.
  model_blind       : neither scale fires -> model capacity / training data
                      (small symbol / low contrast) route.

Usage: raw_detector_probe.py <fn-attribution.json> <out.json> <raw-jsonl>
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/home/data/yinjx/pcbparse-competition/project')
sys.path.insert(0, str(ROOT / 'src'))

RAW_EDGE = 0.3

sys.path.insert(0, str(ROOT))
from pcbparse.io import load_image, tile_image_view  # noqa: E402
from pcbparse.perception.composite_provider import _restore_components  # noqa: E402
from pcbparse.perception.torch_component_adapter import factory  # noqa: E402

CONFIG = ROOT / 'work/module-opt80-20261007/r12-integration/pipeline-integration.yaml'
LOCK = ROOT / 'models/models.lock.json'
MANIFEST = ROOT / 'work/module-relative10-20261005-v1/production-lock-val30-20261006/input_manifest.json'


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    fn_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    raw_path = Path(sys.argv[3])
    fn_all = json.loads(fn_path.read_text(encoding='utf-8'))
    fn_records = fn_all['fn_records']

    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))['image']
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    entry = next(m for m in lock['models'] if m.get('task') == 'component')
    entry = dict(entry)
    inference = dict(entry.get('inference') or {})
    inference['device'] = 'cpu'
    entry['inference'] = inference
    adapter = factory(ROOT / 'models' / entry['path'], entry)
    print(f'[probe] component model loaded: {entry.get("version")} score_gate=0.0', flush=True)

    rows = json.loads(MANIFEST.read_text(encoding='utf-8'))
    scales = [
        ('primary', int(cfg['component_tile_size']), int(cfg['component_tile_overlap'])),
        ('fallback', int(cfg['component_fallback_tile_size']), int(cfg['component_fallback_tile_overlap'])),
    ]

    raw_handle = raw_path.open('w', encoding='utf-8')
    fn_best = {}  # (case, gt_id) -> {scale: (best_iou, best_score)}
    raw_counts = Counter()
    t_start = time.time()
    for row in sorted(rows, key=lambda r: r['case_id']):
        case = row['case_id']
        case_fns = [r for r in fn_records if r['case'] == case]
        original = load_image(row['image_path'])
        for scale_name, tile_size, overlap in scales:
            views = tile_image_view(original, tile_size=tile_size, overlap=overlap)
            for view in views:
                preds = adapter.predict(view, None)
                restored = _restore_components(
                    preds, view, original_size=original.size,
                    internal_border_margin_px=float(cfg.get('component_internal_border_margin_px', 32.0)),
                )
                for item in restored:
                    if item.score < 0.02:
                        continue
                    raw_counts[scale_name] += 1
                    raw_handle.write(json.dumps({
                        'case': case, 'scale': scale_name,
                        'score': round(float(item.score), 4),
                        'category': item.category,
                        'bbox': [round(float(v), 1) for v in
                                 (item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2)],
                    }) + '\n')
                    pb = (float(item.bbox.x1), float(item.bbox.y1),
                          float(item.bbox.x2), float(item.bbox.y2))
                    for r in case_fns:
                        gb = tuple(r['gt_bbox'])
                        s = iou(gb, pb)
                        if s >= 0.05:
                            key = (case, r['gt_id'])
                            slot = fn_best.setdefault(key, {})
                            prev = slot.get(scale_name)
                            if prev is None or s > prev[0]:
                                slot[scale_name] = (s, float(item.score), item.category)
        print(f'[probe] {case} done {time.time() - t_start:.0f}s raw={dict(raw_counts)}', flush=True)
    raw_handle.close()

    joined = []
    class_counts = Counter()
    by_category = defaultdict(Counter)
    by_size = defaultdict(Counter)
    for r in fn_records:
        key = (r['case'], r['gt_id'])
        slot = fn_best.get(key, {})
        p = slot.get('primary')
        f = slot.get('fallback')
        ir_hit = r['best_iou'] >= 0.1 or r['cls'] == 'competing_match'
        if ir_hit:
            verdict = 'ir_geometry'
        elif p is not None and p[0] >= RAW_EDGE:
            verdict = 'fired_but_dropped'
        elif f is not None and f[0] >= RAW_EDGE:
            verdict = 'scale_sensitive'
        else:
            verdict = 'model_blind'
        class_counts[verdict] += 1
        by_category[r['gt_category']][verdict] += 1
        by_size[r['size_bucket']][verdict] += 1
        joined.append({
            'case': r['case'], 'gt_id': r['gt_id'],
            'gt_category': r['gt_category'], 'size_bucket': r['size_bucket'],
            'gt_designator': r['gt_designator'], 'gt_bbox': r['gt_bbox'],
            'ir_cls': r['cls'], 'ir_best_iou': r['best_iou'],
            'raw_primary_iou': None if p is None else round(p[0], 4),
            'raw_primary_score': None if p is None else round(p[1], 4),
            'raw_primary_category': None if p is None else p[2],
            'raw_fallback_iou': None if f is None else round(f[0], 4),
            'raw_fallback_score': None if f is None else round(f[1], 4),
            'verdict': verdict,
            's7_owner_case': r['s7_owner_case'],
        })

    s7_join = Counter(j['verdict'] for j in joined if j['s7_owner_case'])
    out = {
        'schema': 'opt90-comp-raw-probe-v1',
        'gt_access': 'diagnostic_only',
        'config': str(CONFIG),
        'model': entry.get('version'),
        'model_sha256': entry.get('sha256'),
        'raw_candidate_counts': dict(raw_counts),
        'fn_verdict': dict(class_counts),
        'fn_verdict_by_category': {k: dict(v) for k, v in sorted(by_category.items())},
        'fn_verdict_by_size': {k: dict(v) for k, v in sorted(by_size.items())},
        's7_owner_40_verdict': dict(s7_join),
        'joined': joined,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    brief = {k: v for k, v in out.items() if k not in ('joined',)}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
