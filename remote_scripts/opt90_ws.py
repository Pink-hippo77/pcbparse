"""opt90-comp: build an isolated hardlink training workspace.

Creates <ws> with hardlinked Dataset300 / data / models trees (read-only
shared inputs, no copy) and places the component-pin materialization at
<ws>/artifacts/training/materialized/dataset300-v2/component-pin. The shared
artifacts/training/materialized tree of the project is never written.

Usage: opt90_ws.py <ws_dir> <materialized_component_pin_dir>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path('/home/data/yinjx/pcbparse-competition/project')


def hardlink_tree(src: Path, dst: Path) -> tuple[int, int]:
    files = links = 0
    for path in src.rglob('*'):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        files += 1
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, target)
        links += 1
    return files, links


def main() -> int:
    ws = Path(sys.argv[1])
    mat = Path(sys.argv[2])
    ws.mkdir(parents=True, exist_ok=True)
    for name in ('Dataset300', 'data', 'models'):
        src = ROOT / name
        if not src.exists():
            print(f'skip missing {src}')
            continue
        files, links = hardlink_tree(src, ws / name)
        print(f'{name}: files={files} linked={links}', flush=True)
    dest = ws / 'artifacts/training/materialized/dataset300-v2/component-pin'
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise SystemExit(f'dest already exists: {dest}')
    os.symlink(Path(mat).resolve(), dest, target_is_directory=True)
    print('materialization linked at', dest, '->', Path(mat).resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
