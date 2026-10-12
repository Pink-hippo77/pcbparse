"""Raw-candidate overlay (opt90-comp). GT-ACCESS: diagnostic_only.

Draws raw detector candidates (colour ramp by score) plus GT boxes (red) for a
case, so fire sparsity / spatial clustering / score depression can be seen.

Usage: make_raw_overlay.py <case> <out.png> [max_side]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path('/home/data/yinjx/pcbparse-competition/project')
MANIFEST = ROOT / 'work/module-relative10-20261005-v1/production-lock-val30-20261006/input_manifest.json'
FN = ROOT / 'work/opt90-comp/fn-attribution.json'


def main() -> int:
    case = sys.argv[1]
    out_path = Path(sys.argv[2])
    max_side = int(sys.argv[3]) if len(sys.argv) > 3 else 1400
    raws = [json.loads(x) for x in
            (ROOT / 'work/opt90-comp' / f'quick-raw-{case}.jsonl').read_text(encoding='utf-8').splitlines()
            if x.strip()]
    fn_all = json.loads(FN.read_text(encoding='utf-8'))
    rows = json.loads(MANIFEST.read_text(encoding='utf-8'))
    row = next(r for r in rows if r['case_id'] == case)
    image = Image.open(row['image_path']).convert('RGB')
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(image)

    def color(score):
        if score >= 0.5:
            return (0, 160, 0)
        if score >= 0.3:
            return (0, 120, 255)
        return (255, 140, 0) if score >= 0.1 else (160, 160, 160)

    for r in fn_all['fn_records']:
        if r['case'] != case:
            continue
        b = [v * scale for v in r['gt_bbox']]
        draw.rectangle(b, outline=(255, 0, 0), width=2)
    for raw in raws:
        if raw['scale'] != 'primary':
            continue
        b = [v * scale for v in raw['bbox']]
        draw.rectangle(b, outline=color(raw['score']), width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(out_path, image.size, 'raw_primary=', sum(1 for r in raws if r['scale'] == 'primary'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
