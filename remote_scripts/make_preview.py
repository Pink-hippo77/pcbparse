"""Full-image preview for visual domain diagnosis (opt90-comp). Usage:
make_preview.py <case> <out.png> [max_side]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path('/home/data/yinjx/pcbparse-competition/project')
MANIFEST = ROOT / 'work/module-relative10-20261005-v1/production-lock-val30-20261006/input_manifest.json'


def main() -> int:
    case = sys.argv[1]
    out_path = Path(sys.argv[2])
    max_side = int(sys.argv[3]) if len(sys.argv) > 3 else 1400
    rows = json.loads(MANIFEST.read_text(encoding='utf-8'))
    row = next(r for r in rows if r['case_id'] == case)
    image = Image.open(row['image_path']).convert('RGB')
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(out_path, image.size)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
