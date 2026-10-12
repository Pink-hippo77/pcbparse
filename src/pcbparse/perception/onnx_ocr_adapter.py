"""Hash-bound, offline DB/CTC OCRProvider using optional ONNX Runtime.

Core imports require only NumPy/Pillow/Pydantic. Model files and the charset
are explicit descriptor assets; Paddle caches, OpenCV and downloads are absent.
This provider is a candidate until complete model-promotion evidence is supplied.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pcbparse.io import ImageView
from pcbparse.ir.models import BBox, Evidence, TextToken
from pcbparse.perception.protocols import ProviderError
from pcbparse.perception.onnx_ocr_processing import (
    crop_quad, db_polygons, decode_ctc, detection_tensor, recognition_tensor,
)


class ONNXOCRAdapterError(ProviderError):
    pass


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(1024*1024): digest.update(block)
    return digest.hexdigest()


def _object(pairs):
    result = {}
    for key,value in pairs:
        if key in result: raise ONNXOCRAdapterError(f'duplicate descriptor key {key!r}')
        result[key] = value
    return result


def _sha(value, name):
    if not isinstance(value,str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ONNXOCRAdapterError(f'{name} must be a lowercase SHA256')
    return value


def _bound_asset(root: Path, binding: Any, name: str) -> Path:
    if not isinstance(binding,dict) or set(binding) != {'path','sha256'}:
        raise ONNXOCRAdapterError(f'{name} requires exactly path and sha256')
    raw = binding['path']
    if not isinstance(raw,str) or not raw.strip() or Path(raw).is_absolute():
        raise ONNXOCRAdapterError(f'{name} path must be relative')
    path = (root/raw).resolve(strict=True)
    try: path.relative_to(root)
    except ValueError as exc: raise ONNXOCRAdapterError(f'{name} escapes descriptor directory') from exc
    if not path.is_file() or _hash(path) != _sha(binding['sha256'],f'{name}.sha256'):
        raise ONNXOCRAdapterError(f'{name} artifact hash mismatch')
    return path


def _probability(value, name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ONNXOCRAdapterError(f'{name} must be finite in [0,1]')
    return float(value)


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ONNXOCRAdapterError(f'{name} must be an integer in [{low},{high}]')
    return value


class ONNXOCRAdapter:
    def __init__(self, descriptor: Path, entry: Mapping[str,Any]):
        descriptor = Path(descriptor).resolve(strict=True)
        expected = _sha(entry.get('sha256'),'OCR descriptor sha256')
        if not descriptor.is_file() or descriptor.stat().st_size > 1024*1024 or _hash(descriptor) != expected:
            raise ONNXOCRAdapterError('OCR descriptor hash/size mismatch')
        payload = json.loads(descriptor.read_text(encoding='utf-8'),object_pairs_hook=_object)
        if not isinstance(payload,dict) or set(payload) != {'schema_version','architecture','detector','recognizer','characters','preprocess','provenance'}:
            raise ONNXOCRAdapterError('unsupported OCR descriptor fields')
        if type(payload['schema_version']) is not int or payload['schema_version'] != 1 or payload['architecture'] != 'db_ctc_onnx':
            raise ONNXOCRAdapterError('unsupported OCR descriptor version/architecture')
        self.detector_path = _bound_asset(descriptor.parent,payload['detector'],'detector')
        self.recognizer_path = _bound_asset(descriptor.parent,payload['recognizer'],'recognizer')
        characters = payload['characters']
        if not isinstance(characters,list) or not 2 <= len(characters) <= 32768 or characters[0] != '' or any(
            not isinstance(c,str) or len(c) != 1 for c in characters[1:]
        ) or len(set(characters)) != len(characters):
            raise ONNXOCRAdapterError('invalid CTC charset; blank must be index zero')
        self.charset = tuple(characters)
        preprocess = payload['preprocess']
        if not isinstance(preprocess,dict) or set(preprocess) != {'color_order','detector_limit_side_len','detector_max_side_limit','recognition_height','recognition_minimum_width','recognition_maximum_width'}:
            raise ONNXOCRAdapterError('unsupported OCR preprocessing descriptor')
        if preprocess['color_order'] != 'BGR' or preprocess['recognition_height'] != 48:
            raise ONNXOCRAdapterError('unsupported OCR color order/recognition height')
        self.limit_side_len = _integer(preprocess['detector_limit_side_len'],'detector_limit_side_len',1,4000)
        self.max_side_limit = _integer(preprocess['detector_max_side_limit'],'detector_max_side_limit',self.limit_side_len,4000)
        self.minimum_width = _integer(preprocess['recognition_minimum_width'],'recognition_minimum_width',1,3200)
        self.maximum_width = _integer(preprocess['recognition_maximum_width'],'recognition_maximum_width',self.minimum_width,3200)
        parameters = entry.get('inference',{})
        if not isinstance(parameters,Mapping) or set(parameters)-{
            'device','cpu_threads','detector_threshold','box_threshold','unclip_ratio',
            'recognition_threshold','recognition_batch_size','max_candidates'}:
            raise ONNXOCRAdapterError('unsupported OCR inference parameters')
        self.detector_threshold = parameters.get('detector_threshold',.3)
        self.box_threshold = _probability(parameters.get('box_threshold',.6),'box_threshold')
        self.recognition_threshold = _probability(parameters.get('recognition_threshold',0.),'recognition_threshold')
        self.unclip_ratio = parameters.get('unclip_ratio',1.5)
        if isinstance(self.unclip_ratio,bool) or not isinstance(self.unclip_ratio,(int,float)) or not math.isfinite(self.unclip_ratio) or not 0 < self.unclip_ratio <= 4:
            raise ONNXOCRAdapterError('unclip_ratio must be finite in (0,4]')
        self.batch_size = _integer(parameters.get('recognition_batch_size',16),'recognition_batch_size',1,64)
        self.max_candidates = _integer(parameters.get('max_candidates',1000),'max_candidates',1,2000)
        threads = _integer(parameters.get('cpu_threads',4),'cpu_threads',1,16)
        requested = parameters.get('device','cpu')
        if requested not in {'cpu','auto','cuda','gpu'}: raise ONNXOCRAdapterError('unsupported OCR device')
        try: import onnxruntime as ort
        except ImportError as exc: raise ONNXOCRAdapterError('ONNX OCR requires optional onnxruntime') from exc
        available = set(ort.get_available_providers())
        use_cuda = requested in {'cuda','gpu'} or requested == 'auto' and 'CUDAExecutionProvider' in available
        if requested in {'cuda','gpu'} and 'CUDAExecutionProvider' not in available:
            raise ONNXOCRAdapterError('CUDA OCR was requested but is unavailable')
        providers = ['CUDAExecutionProvider','CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        try:
            self.detector = ort.InferenceSession(str(self.detector_path),sess_options=options,providers=providers)
            self.recognizer = ort.InferenceSession(str(self.recognizer_path),sess_options=options,providers=providers)
        except Exception as exc: raise ONNXOCRAdapterError(f'cannot load ONNX OCR models: {type(exc).__name__}: {exc}') from exc
        self.active_providers = {'detector': self.detector.get_providers(), 'recognizer': self.recognizer.get_providers()}
        if use_cuda and any('CUDAExecutionProvider' not in value for value in self.active_providers.values()):
            if requested in {'cuda','gpu'}: raise ONNXOCRAdapterError('requested CUDA OCR session fell back to CPU')
            logging.getLogger('pcbparse.ocr').warning('ONNX OCR auto device fell back: %s',self.active_providers)
        self.detector_input = self._input(self.detector,'detector',height=None)
        self.recognizer_input = self._input(self.recognizer,'recognizer',height=48)
        douts, routs = self.detector.get_outputs(), self.recognizer.get_outputs()
        if len(douts) != 1 or len(douts[0].shape) != 4 or douts[0].shape[1] != 1:
            raise ONNXOCRAdapterError('detector must output N1HW probabilities')
        if len(routs) != 1 or len(routs[0].shape) != 3 or routs[0].shape[-1] != len(self.charset):
            raise ONNXOCRAdapterError('recognizer output/charset dimensions differ')
        self.version = f'onnx-db-ctc/{expected[:16]}'
        self.descriptor_sha256 = expected
        self.runtime_version = ort.__version__
        self._lock = threading.RLock()
        self._last_detection_trace = []
        self._detection_trace_log = []

    @property
    def detector_threshold(self): return self._detector_threshold

    @detector_threshold.setter
    def detector_threshold(self,value):
        number = _probability(value,'detector_threshold')
        if not 0 < number < 1: raise ONNXOCRAdapterError('detector_threshold must be in (0,1)')
        self._detector_threshold = number

    @staticmethod
    def _input(session,name,*,height):
        inputs = session.get_inputs()
        if len(inputs) != 1 or len(inputs[0].shape) != 4 or inputs[0].shape[1] != 3 or inputs[0].type != 'tensor(float)':
            raise ONNXOCRAdapterError(f'{name} must consume NCHW float32 images')
        if height is not None and inputs[0].shape[2] != height:
            raise ONNXOCRAdapterError('recognizer height differs from descriptor')
        return inputs[0].name

    @staticmethod
    def _check(ctx,stage):
        budget = getattr(ctx,'time_budget',None)
        if budget is not None: budget.check(stage=stage)

    def infer_bgr(self,bgr,ctx=None):
        """Return raw text, confidence and view-local quadrilaterals."""
        self._check(ctx,'ocr_onnx_detection')
        tensor = detection_tensor(bgr,limit_side_len=self.limit_side_len,max_side_limit=self.max_side_limit)
        probabilities = self.detector.run(None,{self.detector_input:tensor})
        self._check(ctx,'ocr_onnx_detection')
        if len(probabilities) != 1 or probabilities[0].ndim != 4 or probabilities[0].shape[:2] != (1,1):
            raise ONNXOCRAdapterError('invalid detector batch output')
        polygons = db_polygons(probabilities[0][0,0],width=bgr.shape[1],height=bgr.shape[0],
            threshold=self.detector_threshold,box_threshold=self.box_threshold,
            unclip_ratio=self.unclip_ratio,max_candidates=self.max_candidates,
            check=lambda:self._check(ctx,'ocr_onnx_postprocess'))
        crops = []
        for index,(polygon,score) in enumerate(polygons):
            self._check(ctx,'ocr_onnx_crop')
            crop = crop_quad(bgr,polygon,check=lambda:self._check(ctx,'ocr_onnx_crop'))
            crops.append((index,crop))
        order = sorted(crops,key=lambda item:(item[1].shape[1]/item[1].shape[0],item[0]))
        results = {}
        for start in range(0,len(order),self.batch_size):
            self._check(ctx,'ocr_onnx_recognition')
            batch = order[start:start+self.batch_size]
            tensors = [recognition_tensor(crop,minimum_width=self.minimum_width,maximum_width=self.maximum_width) for _,crop in batch]
            width = max(array.shape[2] for array in tensors)
            inputs = np.stack([np.pad(array,((0,0),(0,0),(0,width-array.shape[2]))) for array in tensors])
            outputs = self.recognizer.run(None,{self.recognizer_input:inputs})
            self._check(ctx,'ocr_onnx_recognition')
            if len(outputs) != 1 or outputs[0].shape[0] != len(batch):
                raise ONNXOCRAdapterError('recognizer output cardinality mismatch')
            texts = decode_ctc(outputs[0],self.charset)
            for (index,_),result in zip(batch,texts): results[index] = result
        return [(text,score,polygon,det_score) for index,(polygon,det_score) in enumerate(polygons)
                for text,score in [results[index]] if text.strip() and score >= self.recognition_threshold]

    def recognize(self,views: list[ImageView],ctx: Any) -> list[TextToken]:
        if len({view.view_id for view in views}) != len(views): raise ONNXOCRAdapterError('duplicate OCR view IDs')
        with self._lock:
            result,trace = [],[]
            for view in views:
                bgr = np.asarray(view.image.convert('RGB'))[:,:,::-1].copy()
                pixels_hash = hashlib.sha256(bgr.tobytes()).hexdigest()
                predictions = self.infer_bgr(bgr,ctx)
                trace.append({'view_id':view.view_id,'detections':len(predictions),'runtime':'onnxruntime',
                              'active_providers':self.active_providers,'descriptor_sha256':self.descriptor_sha256})
                for index,(text,score,polygon,det_score) in enumerate(predictions):
                    low,high = polygon.min(0),polygon.max(0)
                    x1,y1 = max(0,float(low[0])),max(0,float(low[1]))
                    x2,y2 = min(view.width,float(high[0])),min(view.height,float(high[1]))
                    if x2 <= x1 or y2 <= y1: continue
                    token_hash = hashlib.sha256(f'{view.view_id}|{pixels_hash}|{index}|{text}'.encode()).hexdigest()[:24]
                    result.append(TextToken(internal_id=f'ocr-onnx:{token_hash}',raw_text=text,
                        bbox=BBox(x1=x1,y1=y1,x2=x2,y2=y2),score=score,category='unknown',details='',
                        evidence=[Evidence(evidence_id=f'ocr-onnx-evidence:{token_hash}',source='ocr',
                            view_id=view.view_id,score=score,payload_ref=f'sha256:{self.descriptor_sha256}',
                            metadata={'provider':'onnx_db_ctc','model_version':self.version,
                                      'descriptor_sha256':self.descriptor_sha256,'input_pixels_sha256':pixels_hash,
                                      'coordinate_space':'image_view_local','polygon':polygon.tolist(),
                                      'detection_score':det_score,'recognition_score':score,
                                      'semantic_fields_generated':False,'runtime_version':self.runtime_version,
                                      'active_providers':self.active_providers})]))
            self._last_detection_trace = trace
            return result


def factory(model_path: str | Path,entry: Mapping[str,Any]) -> ONNXOCRAdapter:
    if not isinstance(entry,Mapping) or entry.get('task') not in (None,'ocr'):
        raise ONNXOCRAdapterError('model entry must describe OCR')
    return ONNXOCRAdapter(Path(model_path),entry)


__all__ = ['ONNXOCRAdapter','ONNXOCRAdapterError','factory']
