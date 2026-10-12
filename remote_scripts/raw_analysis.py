"""Raw-probe deep analysis (opt90-comp, diagnostic). GT-ACCESS: diagnostic_only.

Answers the Lead's three pre-DESIGN questions from the raw detector dump
(work/opt90-comp/raw-candidates.jsonl, all candidates score>=0.02 at the
production 512/64 primary scale and the 256/32 fallback scale):

  A. zero-recall classes (NET_input/PIN/Buzzer/M/AMP/Net_*/Dc): for every FN of
     those classes, does ANY raw box overlap it (any predicted class, any
     score)?  -> class confusion at decode time (box exists, wrong class,
     dropped below gate) vs representation gap (no box at all).
  B. per-case raw score distribution: is the disaster-case population globally
     score-depressed (calibration) or simply sparse (representation)?
  C. S7 23 completely-undetected owner boxes: does the 256/32 fallback scale
     cover them (Lead gate: >=15/23 recoverable)?

Usage: raw_analysis.py <fn-attribution.json> <raw-probe.json> <out.json>
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/home/data/yinjx/pcbparse-competition/project')
RAW = ROOT / 'work/opt90-comp/raw-candidates.jsonl'

DISASTER = ('0089', '0091', '0092', '0234', '0280', '0283')
ZERO_RECALL_CLASSES = ('NET_input', 'PIN', 'Buzzer', 'M', 'AMP',
                       'Net_bidirection', 'Net_output', 'Dc')
RAW_EDGE = 0.3


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    fn_all = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    probe = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
    out_path = Path(sys.argv[3])

    raw_by_case = defaultdict(list)
    for line in RAW.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        raw_by_case[r['case']].append(r)

    joined = {(j['case'], j['gt_id']): j for j in probe['joined']}

    # ---- A. zero-recall classes: raw overlap of FN boxes ----
    a_rows = []
    for r in fn_all['fn_records']:
        if r['gt_category'] not in ZERO_RECALL_CLASSES:
            continue
        gb = tuple(r['gt_bbox'])
        best = None
        for raw in raw_by_case.get(r['case'], []):
            s = iou(gb, tuple(raw['bbox']))
            if s >= 0.1 and (best is None or s > best[0]):
                best = (s, raw)
        j = joined.get((r['case'], r['gt_id']), {})
        a_rows.append({
            'case': r['case'], 'gt_id': r['gt_id'], 'gt_category': r['gt_category'],
            'gt_designator': r['gt_designator'], 'gt_bbox': r['gt_bbox'],
            'raw_overlap_iou': None if best is None else round(best[0], 3),
            'raw_category': None if best is None else best[1]['category'],
            'raw_score': None if best is None else best[1]['score'],
            'raw_scale': None if best is None else best[1]['scale'],
            'raw_class_matches_gt': None if best is None else
                best[1]['category'] == r['gt_category'],
            'verdict': j.get('verdict'),
        })
    a_summary = Counter(
        'no_raw_box' if row['raw_overlap_iou'] is None else
        ('raw_class_confused' if not row['raw_class_matches_gt'] else 'raw_class_correct')
        for row in a_rows)
    a_by_class = defaultdict(Counter)
    for row in a_rows:
        key = 'no_raw_box' if row['raw_overlap_iou'] is None else (
            'raw_class_confused' if not row['raw_class_matches_gt'] else 'raw_class_correct')
        a_by_class[row['gt_category']][key] += 1
    # raw predicted-class confusion tally (what class does the model think it is)
    a_confusion = Counter(
        (row['gt_category'], row['raw_category']) for row in a_rows
        if row['raw_overlap_iou'] is not None and not row['raw_class_matches_gt'])

    # ---- B. per-case raw score distribution ----
    b_rows = {}
    for case, rows in sorted(raw_by_case.items()):
        prim = [r for r in rows if r['scale'] == 'primary']
        bands = Counter()
        for r in prim:
            s = r['score']
            band = ('ge_0.5' if s >= 0.5 else '0.3-0.5' if s >= 0.3 else
                    '0.1-0.3' if s >= 0.1 else '0.02-0.1')
            bands[band] += 1
        b_rows[case] = {
            'raw_primary': len(prim),
            'ge_0.3': bands['ge_0.5'] + bands['0.3-0.5'],
            'bands': dict(bands),
            'mean_score': round(sum(r['score'] for r in prim) / max(1, len(prim)), 3),
            'top_score': max((r['score'] for r in prim), default=0.0),
            'disaster': case in DISASTER,
        }

    def agg(cases):
        sel = [b_rows[c] for c in cases if c in b_rows]
        return {
            'cases': len(sel),
            'raw_primary_total': sum(r['raw_primary'] for r in sel),
            'ge_0.3_total': sum(r['ge_0.3'] for r in sel),
            'mean_score_avg': round(sum(r['mean_score'] for r in sel) / max(1, len(sel)), 3),
            'top_score_avg': round(sum(r['top_score'] for r in sel) / max(1, len(sel)), 3),
        }

    # ---- C. S7 23 no_overlap owners: fallback-scale coverage ----
    s7_no = [r for r in fn_all['s7_owner_40']['records'] if r['cls'] == 'no_overlap']
    c_rows = []
    for r in s7_no:
        gb = tuple(r['gt_bbox'])
        best512 = best256 = None
        for raw in raw_by_case.get(r['case'], []):
            s = iou(gb, tuple(raw['bbox']))
            if raw['scale'] == 'primary':
                if best512 is None or s > best512[0]:
                    best512 = (s, raw)
            else:
                if best256 is None or s > best256[0]:
                    best256 = (s, raw)
        c_rows.append({
            'case': r['case'], 'gt_id': r['gt_id'], 'gt_category': r['gt_category'],
            'gt_designator': r['gt_designator'], 'gt_bbox': r['gt_bbox'],
            'raw512_iou': None if best512 is None else round(best512[0], 3),
            'raw512_score': None if best512 is None else best512[1]['score'],
            'raw256_iou': None if best256 is None else round(best256[0], 3),
            'raw256_score': None if best256 is None else best256[1]['score'],
            'covered_256_ge_0.3': best256 is not None and best256[0] >= RAW_EDGE,
        })
    c_covered = sum(1 for r in c_rows if r['covered_256_ge_0.3'])

    out = {
        'schema': 'opt90-comp-raw-analysis-v1',
        'gt_access': 'diagnostic_only',
        'A_zero_recall_classes': {
            'fn_total': len(a_rows),
            'summary': dict(a_summary),
            'by_class': {k: dict(v) for k, v in a_by_class.items()},
            'confusion_pairs': {f'{k[0]}->{k[1]}': v for k, v in
                                a_confusion.most_common(20)},
            'records': a_rows,
        },
        'B_score_distribution': {
            'per_case': b_rows,
            'disaster_agg': agg(DISASTER),
            'normal_agg': agg([c for c in b_rows if c not in DISASTER]),
        },
        'C_s7_no_overlap_fallback': {
            'total': len(c_rows),
            'covered_256_ge_0.3': c_covered,
            'lead_gate_ge_15_of_23': c_covered >= 15,
            'records': c_rows,
        },
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    brief = {
        'A_summary': out['A_zero_recall_classes']['summary'],
        'A_by_class': out['A_zero_recall_classes']['by_class'],
        'A_confusion': out['A_zero_recall_classes']['confusion_pairs'],
        'B_disaster': out['B_score_distribution']['disaster_agg'],
        'B_normal': out['B_score_distribution']['normal_agg'],
        'C': {k: v for k, v in out['C_s7_no_overlap_fallback'].items() if k != 'records'},
    }
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
