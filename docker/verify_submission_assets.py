"""Verify additional offline assets using only the Python standard library."""
import hashlib
import json
from pathlib import Path
import sys


def verify(root: Path) -> int:
    root = root.resolve()
    manifest = json.loads((root/'models/submission-assets.manifest.json').read_text('utf8'))
    entries = manifest['files']
    if not entries or len({r['path'] for r in entries}) != len(entries):
        raise ValueError('missing or duplicate submission asset bindings')
    for row in entries:
        path = (root/row['path']).resolve(strict=True)
        path.relative_to(root)
        if not path.is_file() or path.stat().st_size != row['bytes']:
            raise ValueError('submission asset size mismatch: '+row['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('submission asset SHA256 mismatch: '+row['path'])
    return len(entries)


if __name__ == '__main__':
    print('Verified submission assets:', verify(Path(sys.argv[1])))
