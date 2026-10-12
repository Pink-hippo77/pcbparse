"""GT-free downstream A/B replay from hash-verified perception snapshots."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pcbparse.config import load_config
from pcbparse.evaluation.infer_manifest import _apply_accuracy_profile, production_runtime_source_manifest
from pcbparse.ir.models import CircuitIR
from pcbparse.pipeline.orchestrator import infer_one


def replay(manifest: Path, config_path: Path, destination: Path, profile_path: Path | None = None) -> dict:
    source = manifest.resolve().parent
    records = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not records or len({row['case_id'] for row in records}) != len(records):
        raise ValueError('replay requires nonempty unique case IDs')
    destination.mkdir(parents=True, exist_ok=False)
    config = load_config(config_path)
    if profile_path is not None:
        config = _apply_accuracy_profile(config, load_config(profile_path))
    runtime_before = production_runtime_source_manifest(Path(__file__).resolve().parents[1])
    rows = []
    for row in records:
        if row.get('sidecar_access') is not False:
            raise ValueError('source must declare image-only predictions')
        stages = {}
        result_row = {
            'case_id': row['case_id'], 'image_sha256': row['image_sha256'],
            'provider_kind': 'frozen_perception_replay', 'sidecar_access': False,
            'fresh_model_inference': False, 'stage_evidence': stages,
            'source_official_sha256': row.get('official_sha256'),
        }
        try:
            evidence = row['stage_evidence']['perception']
            path = (source / evidence['path']).resolve()
            path.relative_to(source)
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != evidence['sha256']:
                raise ValueError('source perception hash mismatch')
            frozen = CircuitIR.model_validate_json(payload)
            if frozen.image.sha256 != row['image_sha256'] or frozen.state.value != 'PERCEIVED':
                raise ValueError('source image binding/state mismatch')

            class ReplayProvider:
                version = 'frozen-perception-diagnostic-v1'
                def build_ir(self, input_path, ctx):
                    return CircuitIR.model_validate_json(payload)

            def save_stage(stage, ir):
                relative = Path('stage_evidence') / row['case_id'] / f'{stage}.circuit-ir.json'
                snapshot = json.dumps(ir.model_dump(mode='json'), ensure_ascii=False,
                                      sort_keys=True, separators=(',', ':')).encode('utf-8')
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(snapshot)
                stages[stage] = {'path': relative.as_posix(), 'sha256': hashlib.sha256(snapshot).hexdigest(),
                                 'state': ir.state.value}

            result = infer_one(frozen.image.path, destination / 'official' / f"{row['case_id']}.json",
                               provider=ReplayProvider(), config=config, evidence_sink=save_stage)
            result_row.update(status='ok', quality=result.quality,
                              official_sha256=hashlib.sha256(result.output_path.read_bytes()).hexdigest(),
                              official_path=result.output_path.relative_to(destination).as_posix(),
                              config_hash=result.context.config_hash,
                              source_perception_sha256=evidence['sha256'])
        except Exception as exc:
            result_row.update(status='error', error={'type': type(exc).__name__, 'message': str(exc)})
        rows.append(result_row)
    (destination / 'predictions.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n'
                                                           for row in rows), encoding='utf-8')
    runtime_after = production_runtime_source_manifest(Path(__file__).resolve().parents[1])
    report = {'kind': 'frozen_perception_diagnostic_replay', 'gt_access': False,
              'fresh_model_inference': False, 'cases': len(rows),
              'ok': sum(row['status'] == 'ok' for row in rows),
              'same_source_official_bytes': sum(row.get('official_sha256') == row.get('source_official_sha256')
                                                 for row in rows if row['status'] == 'ok'),
              'source_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'config_file_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
              'profile_file_sha256': (hashlib.sha256(profile_path.read_bytes()).hexdigest()
                                      if profile_path is not None else None),
              'runtime_sources': runtime_before,
              'runtime_sources_unchanged': runtime_before == runtime_after,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (destination / 'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report = replay(args.manifest, args.config, args.output_dir, args.profile)
    print(json.dumps({key: value for key, value in report.items() if key != 'runtime_sources'}))
