"""Offline development audit; GT is used only for scoring frozen predictions."""
from __future__ import annotations
import argparse, ast, hashlib, json, time, unittest
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from pcbparse.evaluation.masked_metrics import _maximum_cardinality_matching
from pcbparse.evaluation.raw_ground_truth import build_raw_ground_truth
from pcbparse.evaluation.text_box_eval import _gt_text_boxes

def match(n, m, edges):
    return _maximum_cardinality_matching(n, m, edges,
        left_keys=[str(i) for i in range(n)], right_keys=[str(i) for i in range(m)])

def oracle(edges, values, targets, groups):
    raw, grouped, witnesses = [], {}, {}
    for ci, gi, score in edges:
        if values[ci] != targets[gi]:
            continue
        raw.append((ci, gi, score))
        key = (groups[ci], gi)
        if score > grouped.get(key, -1):
            grouped[key] = score
            witnesses[key] = ci
    l1 = match(len(values), len(targets), raw)
    l2 = match(max(groups) + 1, len(targets), [(g,t,s) for (g,t),s in grouped.items()])
    return l1, l2, [{"group":g,"gt":t,"candidate":witnesses[g,t]} for g,t,s in l2]

class CapacityTests(unittest.TestCase):
    def test_one_group_cannot_explain_two_texts(self):
        a,b,_ = oracle([(0,0,1),(1,1,1)], ["A","B"], ["A","B"], [0,0])
        self.assertEqual((len(a),len(b)), (2,1))
    def test_far_member_cannot_supply_answer(self):
        a,b,_ = oracle([(0,0,1)], ["X","A"], ["A"], [0,0])
        self.assertEqual((len(a),len(b)), (0,0))
    def test_candidate_not_reused(self):
        a,b,_ = oracle([(0,0,1),(0,1,1)], ["A"], ["A","A"], [0])
        self.assertEqual((len(a),len(b)), (1,1))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    root = Path.cwd()
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(CapacityTests))
    if not result.wasSuccessful():
        raise RuntimeError('capacity regression failed')
    fp = root/'work/ocr-frozen-crops-20260923-202215/frozen_crops.json'
    sp = root/'work/ocr-gtfree-selector-20260923-212840/predictions.json'
    frozen = json.loads(fp.read_text(encoding='utf-8')); cands = frozen['candidates']
    selections = json.loads(sp.read_text(encoding='utf-8'))['selections']
    # Run only the historical pure grouping function, never its top-level model/GT code.
    source = root/'scripts/ocr_grouping_diagnosis.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='build_candidate_groups')
    from pcbparse.evaluation.text_box_eval import _bbox_iou, _center_distance
    ns = dict(defaultdict=defaultdict, np=np, _bbox_iou=_bbox_iou, _center_distance=_center_distance)
    # Vectorized exact equivalent of historical union rules, preserving i,j order.
    boxes=np.array([c['original_bbox'] for c in cands],float)
    centers=(boxes[:,:2]+boxes[:,2:])/2; wh=boxes[:,2:]-boxes[:,:2]
    parent=list(range(len(cands)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    for i in range(len(cands)-1):
        ids=np.arange(i+1,len(cands)); w=wh[ids]
        valid=(np.maximum(wh[i,0]/w[:,0],w[:,0]/wh[i,0])<=2)&(np.maximum(wh[i,1]/w[:,1],w[:,1]/wh[i,1])<=2)
        inter=np.maximum(0,np.minimum(boxes[i,2:],boxes[ids,2:])-np.maximum(boxes[i,:2],boxes[ids,:2])).prod(axis=1)
        iou=inter/(wh[i].prod()+w.prod(axis=1)-inter)
        valid &= (iou>=.3)|(np.linalg.norm(centers[ids]-centers[i],axis=1)<=8)
        for j in ids[valid]:
            a,b=find(i),find(int(j))
            if a!=b: parent[a]=b
    buckets=defaultdict(list)
    for i in range(len(cands)): buckets[find(i)].append(i)
    members=sorted(buckets.values(),key=min)
    assert len(members)==len(selections)
    group_ids=[0]*len(cands)
    for g,items in enumerate(members):
        assert selections[g]['member_count']==len(items)
        assert selections[g]['selected_index'] in items
        for ci in items: group_ids[ci]=g
    (out/'groups.json').write_text(json.dumps(members),encoding='utf-8')
    print(f'groups verified: {len(members)}; candidates: {len(cands)}',flush=True)
    image_path=Path(frozen['image_path'])
    manifest={"kind":"offline_development_audit", "image_sha256":digest(image_path),
        "frozen_sha256":digest(fp),"selection_sha256":digest(sp),"script_sha256":digest(__file__),
        "group_source_sha256":digest(source),"model_lock_sha256":digest(root/'models/models.lock.json'),
        "production_ab":"not_measured_backend_not_wired", "capacity_tests":result.testsRun}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    from pcbparse.perception.recognizer_backend import PaddleRecognizerBackend
    backend=PaddleRecognizerBackend('en_PP-OCRv5_mobile_rec')
    img=Image.open(image_path).convert('RGB'); values=[]; scores=[]; cache={}
    t0=time.perf_counter()
    with (out/'recognition.jsonl').open('w',encoding='utf-8') as f:
        for i,c in enumerate(cands):
            crop=np.array(img.crop(tuple(c['crop_box'])))
            h=hashlib.sha256(crop.tobytes()).hexdigest()
            assert h.startswith(c['crop_hash'])
            key=(tuple(crop.shape),h)
            if key not in cache: cache[key]=backend.recognize_crops([crop])[0]
            text,score=cache[key]; values.append(text); scores.append(score)
            f.write(json.dumps(dict(index=i,id=c['stable_id'],text=text,score=score,sha256=h),ensure_ascii=False)+'\n')
            if i%500==0:
                f.flush(); print(f'recognized {i}/{len(cands)}, unique={len(cache)}, elapsed={time.perf_counter()-t0:.1f}s',flush=True)
    elapsed=time.perf_counter()-t0
    # GT is opened only after frozen image predictions are complete.
    ann=next((root/'Dataset300/Dataset300/0005').glob('*_CVJsonStd.json'))
    gt=_gt_text_boxes(build_raw_ground_truth(json.loads(ann.read_text(encoding='utf-8')),case_id='0005',annotation_path=ann))
    targets=[g['gt_value'] for g in gt]; accepted=[r for r in selections if r['status']=='accepted']
    reports={}
    for policy in ['center20','iou03','iou05']:
        edges=[]
        for ci,c in enumerate(cands):
            for gi,g in enumerate(gt):
                iou=_bbox_iou(tuple(c['original_bbox']),g['bbox'])
                ok=_center_distance(tuple(c['original_bbox']),g['bbox'])<=20 if policy=='center20' else iou>=float(policy[-2:])/10
                if ok: edges.append((ci,gi,iou))
        a,b,witness=oracle(edges,values,targets,group_ids)
        rows={}
        for mode in ['representative','selected']:
            e=[]
            for pi,r in enumerate(accepted):
                box=r['rep_bbox'] if mode=='representative' else cands[r['selected_index']]['original_bbox']
                for gi,g in enumerate(gt):
                    iou=_bbox_iou(tuple(box),g['bbox'])
                    ok=_center_distance(tuple(box),g['bbox'])<=20 if policy=='center20' else iou>=float(policy[-2:])/10
                    if ok:e.append((pi,gi,iou))
            matches=match(len(accepted),len(gt),e)
            exact=sum(accepted[p]['prediction']==targets[t] for p,t,s in matches)
            rows[mode]=dict(matched=len(matches),exact=exact,denominator=len(gt),unmatched_outputs=len(accepted)-len(matches),matched_wrong=len(matches)-exact)
        reports[policy]=dict(raw_oracle=len(a),group_oracle=len(b),grouping_loss=len(a)-len(b),witnesses=witness,selector=rows)
        assert len(b)<=len(a)
    fresh_changed=sum(values[r['selected_index']]!=r['prediction'] for r in selections)
    report=dict(policies=reports,accepted=len(accepted),rejected=len(selections)-len(accepted),recognition_seconds=elapsed,
        unique_crops=len(cache),selected_text_changes_vs_historical=fresh_changed,
        timing_scope='offline recognizer with identical pixel cache; not production latency',
        limitation='backend hides runtime exceptions as empty; unmatched outputs are not all proven false positives; GT fields are partial text inventory')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:{q:v for q,v in r.items() if q!='witnesses'} for k,r in reports.items()},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
