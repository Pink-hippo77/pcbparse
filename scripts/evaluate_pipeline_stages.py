"""Offline scoring of image-only pipeline snapshots. Never used in prediction."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from pcbparse.evaluation.masked_metrics import (
    _component_metrics, _pin_metrics, aggregate_masked_reports, evaluate_masked_case,
)
from pcbparse.evaluation.raw_cli import _annotation_path, _strict_json_object
from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth
from pcbparse.evaluation.text_box_eval import evaluate_text_boxes
from pcbparse.ir.models import CircuitIR


def evaluate_stages(manifest: Path, dataset_root: Path) -> dict:
    root = manifest.resolve().parent
    dataset_root = dataset_root.resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = [row['case_id'] for row in rows]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError('manifest must have nonempty unique case IDs')
    case_reports = []
    by_stage = defaultdict(list)
    text_stage_counts = defaultdict(Counter)
    for row in rows:
        if row.get('sidecar_access') is not False or row.get('provider_kind') not in {
            'production', 'frozen_perception_replay'
        }:
            raise ValueError(f"{row['case_id']}: not an image-only or frozen-perception artifact")
        case_id = row['case_id']
        annotation = _annotation_path(dataset_root, case_id)
        gt = build_raw_ground_truth(_strict_json_object(annotation), case_id=case_id, annotation_path=annotation)
        stages = {}
        for stage, evidence in row.get('stage_evidence', {}).items():
            path = (root / evidence['path']).resolve()
            path.relative_to(root)
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != evidence['sha256']:
                raise ValueError(f'{case_id}: snapshot hash mismatch for {stage}')
            ir = CircuitIR.model_validate_json(payload)
            if ir.image.sha256 != row['image_sha256']:
                raise ValueError(f'{case_id}: snapshot image hash mismatch')
            raw = evaluate_masked_case(ir, gt)
            _, component_map = _component_metrics(ir, gt, threshold=0.5)
            _, pin_map = _pin_metrics(ir, gt, component_map=component_map, tolerance=5.0)
            text = evaluate_text_boxes(ir, gt, iou_threshold=0.3, center_tolerance_px=0.0,
                                       pred_to_gt_component=component_map, pred_to_gt_pin=pin_map)
            # GT covers field texts, not every glyph in the image. Unmatched
            # outputs are not proven detector false positives.
            text['unmatched_field_outputs'] = text.pop('false_positives')
            text['unmatched_field_output_count'] = text.pop('false_positive_count')
            counts = {name: len(getattr(ir, name)) for name in
                      ('components', 'pins', 'texts', 'segments', 'physical_nets', 'electrical_nets')}
            stages[stage] = {'counts': counts, 'raw': raw, 'text': text,
                             'pruning': ir.metadata.get('degraded_submission_prune', {})}
            by_stage[stage].append(raw)
            text_stage_counts[stage]['gt_box_count'] += text['gt_box_count']
            text_stage_counts[stage]['unmatched_field_output_count'] += text['unmatched_field_output_count']
            for item in text['stages']:
                text_stage_counts[stage][item['stage']] += item['count']
            for obj in text['objects']:
                verdicts = obj['stages']
                if all(verdicts[name]['status'] == 'passed' for name in
                       ('text_recognized', 'field_classified', 'owner_bound')):
                    text_stage_counts[stage]['exact_field_and_owner'] += 1
        case_reports.append({'case_id': case_id, 'execution_status': row['status'],
                             'quality': row.get('quality'), 'error': row.get('error'), 'stages': stages})
    return {
        'kind': 'offline_pipeline_stage_evaluation', 'split_role': 'development_regression',
        'official_score': None, 'selected_cases': len(rows),
        'execution_status_counts': dict(Counter(row['status'] for row in rows)),
        'provider_kind_counts': dict(Counter(row['provider_kind'] for row in rows)),
        'stage_case_coverage': {stage: len(items) for stage, items in by_stage.items()},
        'aggregate_by_stage': {stage: aggregate_masked_reports(items) for stage, items in by_stage.items()},
        'text_by_stage': {stage: dict(counts) for stage, counts in text_stage_counts.items()},
        'cases': case_reports,
        'lineage': {'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'limitations': [
            'raw-v1 and text-box metrics are local diagnostics, not official scores.',
            'Stage aggregates cover available snapshots only; stage_case_coverage exposes missing cases.',
            'All selected cases, including errors and not_run, remain in execution_status_counts.',
            'Text matching uses one-to-one IoU >= 0.3; GT inventories do not cover all image text.',
            'Pre-graph connectivity is unavailable; zero graph metrics before graph construction are not final quality.',
            'Ground truth is opened only in this offline scorer, never supplied to inference.',
            'Frozen-perception replays evaluate downstream rules only, not fresh model inference or latency.',
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite {args.output}')
    report = evaluate_stages(args.manifest, args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('selected_cases', 'execution_status_counts',
                                                  'stage_case_coverage', 'text_by_stage')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
