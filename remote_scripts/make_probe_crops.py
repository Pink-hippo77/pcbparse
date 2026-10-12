"""Diagnostic overlay crops for FN clusters (opt90-comp). GT-ACCESS: diagnostic_only.

Draws GT boxes (red) and predicted component boxes (green) on a case crop so
the failure mode can be inspected visually. Never feeds inference.

Usage: make_probe_crops.py <fn-attribution.json> <case> <x1> <y1> <x2> <y2> <out.png>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path('/home/data/yinjx/pcbparse-competition/project')
sys.path.insert(0, str(ROOT / 'src'))

from PIL import Image, ImageDraw  # noqa: E402

from pcbparse.ir.models import CircuitIR  # noqa: E402

MANIFEST = ROOT / 'work/opt90-lead/eval-s6fix/s6fix-eval-manifest.jsonl'
DATASET = ROOT / 'Dataset300' / 'Dataset300'


def main() -> int:
    fn_path = Path(sys.argv[1])
    case = sys.argv[2]
    x1, y1, x2, y2 = (float(v) for v in sys.argv[3:7])
    out_path = Path(sys.argv[7])
    fn_all = json.loads(fn_path.read_text(encoding='utf-8'))
    rows = [json.loads(x) for x in MANIFEST.read_text(encoding='utf-8').splitlines() if x.strip()]
    row = next(r for r in rows if r['case_id'] == case)
    ir_path = Path(row['ir_path'])
    if not ir_path.is_absolute():
        ir_path = (MANIFEST.parent / ir_path).resolve(strict=True)
    ir = CircuitIR.model_validate_json(ir_path.read_text(encoding='utf-8'))
    img_path = Path(row.get('image_path') or (DATASET / case / f'{case}-Datasheet.png'))
    if not img_path.exists():
        cand = sorted((DATASET / case).glob('*.png'))
        img_path = cand[0]
    image = Image.open(img_path).convert('RGB')
    crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
    scale = max(1, min(4, int(1400 / max(1, crop.width))))
    crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
    draw = ImageDraw.Draw(crop)

    def rect(box, color, width=2):
        px = [(box[0] - x1) * scale, (box[1] - y1) * scale,
              (box[2] - x1) * scale, (box[3] - y1) * scale]
        draw.rectangle(px, outline=color, width=width)

    for r in fn_all['fn_records']:
        if r['case'] != case:
            continue
        rect(r['gt_bbox'], (255, 0, 0), 3)
        draw.text(((r['gt_bbox'][0] - x1) * scale + 2, (r['gt_bbox'][1] - y1) * scale + 2),
                  f"{r['gt_id']}:{r['gt_designator']}", fill=(255, 0, 0))
    for comp in ir.components:
        b = (float(comp.bbox.x1), float(comp.bbox.y1), float(comp.bbox.x2), float(comp.bbox.y2))
        if b[2] < x1 or b[0] > x2 or b[3] < y1 or b[1] > y2:
            continue
        rect(b, (0, 200, 0), 2)
        draw.text(((b[0] - x1) * scale + 2, (b[3] - y1) * scale - 12),
                  f"{comp.category}:{comp.score:.2f}", fill=(0, 160, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    print(out_path, crop.size)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
