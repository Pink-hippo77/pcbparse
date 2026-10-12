"""opt90-comp: build an evaluation-only candidate lock with the opt90 component model.

Modeled on work/opt90-pin/make_pin_candidate_lock.py. The base lock is the r07
PP-OCR candidate lock (the exact lock the run-s6fix baseline stack used); ONLY
the component task entry is replaced by the hash-bound opt90 component
candidate bundle. All other entries (ocr=PP-OCR ONNX, pin, wire) are copied
verbatim with paths re-relativized.

Honesty: the lock declares blessed=false / status='candidate-evaluation' at
lock and entry level, so readiness stays UNREADY (fail-closed preserved).
models/models.lock.json is never touched.

Usage: make_comp_candidate_lock.py <base-candidate-lock> <component-bundle-dir> <out-lock>
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    base_lock_path = Path(sys.argv[1]).resolve()
    bundle_dir = Path(sys.argv[2]).resolve()
    out_path = Path(sys.argv[3]).resolve()

    lock = json.loads(base_lock_path.read_text(encoding='utf-8'))
    keep = {m['task']: m for m in lock['models']}
    manifest_path = bundle_dir / 'bundle.manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('task') != 'component':
        raise SystemExit('bundle manifest task is not component')
    artifact_path = bundle_dir / str(manifest['artifact_path'])
    artifact_sha = sha(artifact_path)
    if artifact_sha != manifest['artifact_sha256']:
        raise SystemExit('bundle artifact sha256 mismatch')

    def rel(target: Path) -> str:
        return Path(os.path.relpath(target, out_path.parent)).as_posix()

    export_format = 'torchscript' if artifact_path.suffix.lower() in ('.pt', '.ts', '.torchscript') else 'onnx'
    comp_entry = {
        'task': 'component',
        'adapter': 'pcbparse.perception.torch_component_adapter:factory',
        'artifact_format': export_format,
        'path': rel(artifact_path),
        'sha256': artifact_sha,
        'version': str(manifest.get('version') or f'opt90-comp-{artifact_sha[:12]}'),
        'status': 'candidate-evaluation',
        'blessed': False,
        'inference': manifest.get('inference', {}),
        'bundle_manifest_sha256': sha(manifest_path),
        'candidate_manifest_path': rel(manifest_path),
        'dataset_config_hash': manifest.get('dataset_config_hash'),
        'training_config_sha256': manifest.get('training_config_sha256'),
        'evaluation': manifest.get('evaluation', {}),
        'gate_policy': manifest.get('gate_policy', {}),
        'model_card_sha256': manifest.get('model_card_sha256'),
        'license': manifest.get('license'),
        'note': (
            'opt90 component candidate: warm-started from the blessed '
            'component-v5-fullmix checkpoint (34d79a6d...), single preregistered '
            'factor per work/opt90-comp/DESIGN.md. Evaluation only - NOT '
            'promoted. Production lock untouched.'
        ),
    }

    models = []
    for task in ('component', 'ocr', 'pin', 'wire'):
        if task == 'component':
            models.append(comp_entry)
            continue
        entry = dict(keep[task])
        entry['path'] = rel((base_lock_path.parent / entry['path']).resolve())
        entry['candidate_manifest_path'] = rel(
            (base_lock_path.parent / keep[task]['candidate_manifest_path']).resolve()
        ) if keep[task].get('candidate_manifest_path') else entry.get('candidate_manifest_path')
        models.append(entry)

    candidate = {
        'schema_version': lock.get('schema_version', '1.0'),
        'required_tasks': lock.get('required_tasks'),
        'junction_mode': lock.get('junction_mode'),
        'models': models,
        'status': 'candidate-evaluation',
        'blessed': False,
        'note': (
            'opt90-comp evaluation lock: r07 PP-OCR candidate lock with ONLY the '
            'component entry replaced. Not blessed; production lock untouched.'
        ),
    }
    out_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding='utf-8')
    print('wrote', out_path)
    print('component artifact', artifact_path.name, artifact_sha)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
