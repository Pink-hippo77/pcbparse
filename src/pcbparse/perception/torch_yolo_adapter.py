"""Bounded inference for a hash-bound YOLO TorchScript component detector.

Ultralytics is needed for training/export only. Deployed inference uses the
existing Torch, TorchVision and OpenCV dependencies and a frozen class table.
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path
import numpy as np

from pcbparse.io import ImageView
from pcbparse.ir.models import BBox, Component, Evidence


class TorchYoloComponentAdapter:
    def __init__(self, path, *, sha256, category_table, device='auto',
                 score_threshold=.3, nms_iou=.55, max_detections=512, input_size=640):
        import torch
        path=Path(path)
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != sha256:
            raise ValueError('TorchScript component checkpoint SHA256 mismatch')
        table={int(r['model_label']):dict(r) for r in category_table}
        if sorted(table)!=list(range(len(table))) or not table:
            raise ValueError('Component class IDs must be contiguous')
        if not 0 <= score_threshold <= 1 or not 0 < nms_iou <= 1:
            raise ValueError('Invalid component score/NMS threshold')
        if not isinstance(input_size,int) or input_size <= 0 or input_size % 32:
            raise ValueError('Component input size must be a positive multiple of 32')
        if not isinstance(max_detections,int) or max_detections<=0:
            raise ValueError('Invalid component output cap')
        self.device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device)
        with path.open('rb') as stream:
            self.model=torch.jit.load(stream,map_location=self.device).eval()
        self.categories=table;self.size=input_size;self.threshold=score_threshold
        self.nms_iou=nms_iou;self.max_detections=max_detections
        self.checkpoint_sha256=sha256;self.version='torchscript-yolo-component/'+sha256[:12]
        self._lock=threading.Lock()

    def _tensor(self, image):
        import cv2
        import torch
        rgb=np.asarray(image.image.convert('RGB'))
        w,h=image.width,image.height
        ratio=min(self.size/w,self.size/h)
        rw,rh=round(w*ratio),round(h*ratio)
        resized=cv2.resize(rgb,(rw,rh),interpolation=cv2.INTER_LINEAR) if (rw,rh)!=(w,h) else rgb
        left,top=round((self.size-rw)/2-.1),round((self.size-rh)/2-.1)
        canvas=np.full((self.size,self.size,3),114,dtype=np.uint8)
        canvas[top:top+rh,left:left+rw]=resized
        # Match the source backend's scale_boxes rounding, including odd pads.
        geometry=(ratio,round((self.size-w*ratio)/2-.1),round((self.size-h*ratio)/2-.1))
        return torch.from_numpy(canvas.transpose(2,0,1).copy()).float().div_(255),geometry

    def predict(self, image, ctx):
        return self.predict_many([image],ctx)[0]

    def predict_many(self, images, ctx):
        import torch
        from torchvision.ops import batched_nms
        if any(not isinstance(v,ImageView) for v in images):
            raise TypeError('Component inference expects ImageView objects')
        result=[]
        for start in range(0,len(images),8):
            if ctx is not None:ctx.time_budget.check(stage='component_detector:before')
            views=images[start:start+8]
            prepared=[self._tensor(v) for v in views]
            batch=torch.stack([v[0] for v in prepared]).to(self.device)
            with self._lock,torch.inference_mode():
                output=self.model(batch)
            if isinstance(output,(tuple,list)):output=output[0]
            if output.ndim!=3 or output.shape[:2]!=(len(views),4+len(self.categories)):
                raise ValueError('Unexpected component export output dimensions')
            for view,raw,(_,geometry) in zip(views,output,prepared):
                pred=raw.transpose(0,1)
                scores,labels=pred[:,4:].max(1)
                keep=(scores>=self.threshold)&torch.isfinite(pred).all(1)
                centers=pred[keep,:4];scores=scores[keep];labels=labels[keep]
                boxes=torch.cat((centers[:,:2]-centers[:,2:]/2,centers[:,:2]+centers[:,2:]/2),1)
                selected=batched_nms(boxes,scores,labels,self.nms_iou)[:self.max_detections]
                boxes,scores,labels=boxes[selected].cpu().numpy(),scores[selected].cpu().numpy(),labels[selected].cpu().numpy()
                ratio,px,py=geometry
                if len(boxes):
                    boxes[:,[0,2]]=(boxes[:,[0,2]]-px)/ratio
                    boxes[:,[1,3]]=(boxes[:,[1,3]]-py)/ratio
                    boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,view.width)
                    boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,view.height)
                candidates=[]
                for index,(b,score,label) in enumerate(zip(boxes,scores,labels)):
                    if b[2]>b[0] and b[3]>b[1]:
                        candidates.append((float(score),int(label),tuple(float(v) for v in b),index))
                candidates.sort(key=lambda c:(-c[0],c[1],c[2][1],c[2][0],c[2][3],c[2][2],c[3]))
                rows=[]
                for index,(score,label,b,_) in enumerate(candidates):
                    identity=f'{view.view_id}:torch-yolo:{index:06d}'
                    rows.append(Component(internal_id=identity,category=self.categories[label]['name'],score=score,
                        bbox=BBox(x1=b[0],y1=b[1],x2=b[2],y2=b[3]),
                        evidence=[Evidence(evidence_id=identity,source='detector',view_id=view.view_id,score=score,
                            metadata={'model_sha256':self.checkpoint_sha256,'model_label':label,
                                      'source_category_id':self.categories[label]['official_id']})]))
                result.append(rows)
            if ctx is not None:ctx.time_budget.check(stage='component_detector:after')
        if ctx is not None and isinstance(getattr(ctx,'model_versions',None),dict):
            ctx.model_versions['component']=self.version
        return result
