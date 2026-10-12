"""Replicate the production component phase step by step (opt90-comp diagnostic).

Runs the exact build_ir component-phase call sequence (tiles -> predict ->
restore -> score gate -> border refinement (off) -> border confirmation (off)
-> class-aware NMS -> candidate cap) for one case with the production config
and production component model, printing counts after every step. Decides
whether the run-s6fix IR loss happens inside the component phase or later.

GT-ACCESS: none (this script reads no GT at all).

Usage: replicate_component_phase.py <case>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path('/home/data/yinjx/pcbparse-competition/project')
sys.path.insert(0, str(ROOT / 'src'))

from pcbparse.io import load_image, tile_image_view  # noqa: E402
from pcbparse.perception.composite_provider import (  # noqa: E402
    _class_aware_nms,
    _filter_component_candidates,
    _filter_unconfirmed_internal_border_components,
    _predict_component_views,
    _refine_internal_border_components,
    _restore_components,
)
from pcbparse.perception.torch_component_adapter import factory  # noqa: E402

CONFIG = ROOT / 'work/module-opt80-20261007/r12-integration/pipeline-integration.yaml'
LOCK = ROOT / 'models/models.lock.json'
MANIFEST = ROOT / 'work/module-relative10-20261005-v1/production-lock-val30-20261006/input_manifest.json'


def main() -> int:
    case = sys.argv[1]
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))['image']
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    entry = next(m for m in lock['models'] if m.get('task') == 'component')
    entry = dict(entry)
    inference = dict(entry.get('inference') or {})
    inference['device'] = 'cpu'
    entry['inference'] = inference
    adapter = factory(ROOT / 'models' / entry['path'], entry)

    rows = json.loads(MANIFEST.read_text(encoding='utf-8'))
    row = next(r for r in rows if r['case_id'] == case)
    original = load_image(row['image_path'])
    print('image', row['image_path'], original.size, flush=True)

    tiles = tile_image_view(original,
                            tile_size=int(cfg['component_tile_size']),
                            overlap=int(cfg['component_tile_overlap']))
    print('tiles', len(tiles), [t.view_id for t in tiles], flush=True)
    predictions, _ = _predict_component_views(adapter, tiles, None, batch_size=1)
    print('per-tile raw counts', [len(p) for p in predictions], flush=True)

    candidates = []
    raw_total = gated_total = 0
    for view, predicted in zip(tiles, predictions):
        restored = _restore_components(predicted, view, original_size=original.size,
                                       internal_border_margin_px=32.0)
        raw_total += len(restored)
        gated = _filter_component_candidates(
            restored, min_score=float(cfg['component_candidate_min_score']))
        gated_total += len(gated)
        candidates.extend(gated)
    print('after restore:', raw_total, 'after score gate:', gated_total, flush=True)

    refined, _stats = _refine_internal_border_components(
        adapter, candidates, original, None,
        enabled=False, categories=(), tile_size=int(cfg['component_tile_size']),
        match_iou=0.5, max_candidates=16, min_score=float(cfg['component_candidate_min_score']),
        internal_border_margin_px=32.0, batch_size=1)
    print('after border refinement:', len(refined), flush=True)
    confirmed = _filter_unconfirmed_internal_border_components(
        refined, enabled=False, categories=(), confirmation_iou=0.3, confirmation_metric='iou')
    print('after border confirmation:', len(confirmed), flush=True)
    kept = _class_aware_nms(confirmed, float(cfg['component_nms_iou']),
                            prefer_non_border=False, center_preference_score_bonus=1.0)
    print('after class-aware NMS:', len(kept), flush=True)
    if len(kept) > int(cfg['component_max_candidates']):
        kept = sorted(kept, key=lambda c: -c.score)[: int(cfg['component_max_candidates'])]
    print('after cap:', len(kept), flush=True)
    from collections import Counter
    print('kept categories:', Counter(c.category for c in kept), flush=True)
    for c in sorted(kept, key=lambda c: (c.bbox.y1, c.bbox.x1))[:20]:
        print('  ', c.category, round(c.score, 4),
              [round(float(v), 1) for v in (c.bbox.x1, c.bbox.y1, c.bbox.x2, c.bbox.y2)],
              [e.view_id for e in c.evidence][:1])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
