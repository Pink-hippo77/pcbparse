import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from pcbparse.io import ImageView
from pcbparse.ir.transforms import TransformChain
from pcbparse.perception.onnx_ocr_adapter import ONNXOCRAdapterError, factory
from pcbparse.perception.ocr_multilevel import run_multilevel_ocr
from pcbparse.perception.production_provider import _apply_ocr_inference_overrides, ProductionProviderError


class _Session:
    def __init__(self,path,**kwargs):
        self.det = Path(path).name == 'det.onnx'
        self.calls = 0
    def get_providers(self): return ['CPUExecutionProvider']
    def get_inputs(self):
        return [SimpleNamespace(name='x',type='tensor(float)',shape=[None,3,None if self.det else 48,None])]
    def get_outputs(self):
        return [SimpleNamespace(name='y',shape=[None,1,None,None] if self.det else [None,None,3])]
    def run(self,outputs,inputs):
        self.calls += 1
        array = inputs['x']
        if self.det:
            result = np.zeros((1,1,*array.shape[-2:]),dtype=np.float32)
            result[0,0,5:14,6:30] = .99
        else:
            result = np.zeros((len(array),5,3),dtype=np.float32)
            for timestep,index in enumerate((1,1,0,2,2)): result[:,timestep,index] = .99
        return [result]


ORT = SimpleNamespace(__version__='test',SessionOptions=SimpleNamespace,
    InferenceSession=_Session,get_available_providers=lambda:['CPUExecutionProvider'])


class ONNXOCRAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('det.onnx','rec.onnx'): (self.root/name).write_bytes(name.encode())
        self.descriptor = {'schema_version':1,'architecture':'db_ctc_onnx',
            'detector':self.binding('det.onnx'),'recognizer':self.binding('rec.onnx'),
            'characters':['','A','B'], 'preprocess':{'color_order':'BGR','detector_limit_side_len':64,
                'detector_max_side_limit':4000,'recognition_height':48,'recognition_minimum_width':320,
                'recognition_maximum_width':3200},'provenance':{'promoted':False}}
        self.path = self.root/'bundle.json'

    def binding(self,name): return {'path':name,'sha256':hashlib.sha256((self.root/name).read_bytes()).hexdigest()}
    def write(self):
        self.path.write_text(json.dumps(self.descriptor),encoding='utf-8')
        return {'task':'ocr','sha256':hashlib.sha256(self.path.read_bytes()).hexdigest(),'inference':{'device':'cpu'}}
    def load(self):
        with patch.dict(sys.modules,{'onnxruntime':ORT}): return factory(self.path,self.write())
    def view(self,name='original'):
        return ImageView(view_id=name,image=Image.new('RGB',(80,40),'white'),transform=TransformChain())

    def test_asset_tamper_and_path_escape_fail_before_inference(self):
        entry = self.write()
        (self.root/'rec.onnx').write_bytes(b'changed')
        with self.assertRaisesRegex(ONNXOCRAdapterError,'hash mismatch'): factory(self.path,entry)
        self.descriptor['recognizer'] = self.binding('rec.onnx')
        self.descriptor['detector']['path'] = '../det.onnx'
        # A missing escaped target is also a hard failure; create it to prove
        # containment validation independently of file existence.
        outside = self.root.parent/'det.onnx'
        if outside.exists(): self.skipTest('temporary parent already has det.onnx')
        outside.write_bytes(b'outside'); self.addCleanup(outside.unlink)
        with self.assertRaisesRegex(ONNXOCRAdapterError,'escapes'): factory(self.path,self.write())

    def test_descriptor_hash_and_charset_dimension_fail_closed(self):
        entry = self.write(); self.path.write_text('{}',encoding='utf-8')
        with self.assertRaisesRegex(ONNXOCRAdapterError,'descriptor hash'): factory(self.path,entry)
        self.descriptor['characters'].append('C')
        with patch.dict(sys.modules,{'onnxruntime':ORT}):
            with self.assertRaisesRegex(ONNXOCRAdapterError,'charset dimensions'): factory(self.path,self.write())

    def test_large_multilingual_charset_decodes_last_class_and_rejects_dimension_mismatch(self):
        count = 18710
        self.descriptor['characters'] = ['', *(chr(0x3400 + i) for i in range(count - 1))]
        class LargeSession(_Session):
            def get_outputs(self):
                return [SimpleNamespace(name='y', shape=[None, 1, None, None] if self.det else [None, None, count])]
            def run(self, outputs, inputs):
                if self.det:
                    return super().run(outputs, inputs)
                result = np.zeros((len(inputs['x']), 3, count), dtype=np.float32)
                result[:, 0, count - 1] = .99
                result[:, 1, 0] = .99
                result[:, 2, count - 1] = .99
                return [result]
        large_ort = SimpleNamespace(**{**ORT.__dict__, 'InferenceSession': LargeSession})
        with patch.dict(sys.modules, {'onnxruntime': large_ort}):
            provider = factory(self.path, self.write())
            tokens = provider.recognize([self.view()], None)
            self.assertEqual(tokens[0].raw_text, self.descriptor['characters'][-1] * 2)
            self.descriptor['characters'].pop()
            with self.assertRaisesRegex(ONNXOCRAdapterError, 'charset dimensions'):
                factory(self.path, self.write())

    def test_large_charset_remains_bounded_and_rejects_duplicates(self):
        self.descriptor['characters'] = ['', *(chr(0x3400 + i) for i in range(32768))]
        with self.assertRaisesRegex(ONNXOCRAdapterError, 'invalid CTC charset'):
            factory(self.path, self.write())
        self.descriptor['characters'] = ['', *(chr(0x3400 + i) for i in range(18708)), chr(0x3400)]
        with self.assertRaisesRegex(ONNXOCRAdapterError, 'invalid CTC charset'):
            factory(self.path, self.write())

    def test_unknown_configuration_and_cuda_fallback_fail(self):
        entry = self.write(); entry['inference']['silent_download'] = True
        with self.assertRaisesRegex(ONNXOCRAdapterError,'unsupported'): factory(self.path,entry)
        entry['inference'] = {'device':'cuda'}
        with patch.dict(sys.modules,{'onnxruntime':ORT}):
            with self.assertRaisesRegex(ONNXOCRAdapterError,'CUDA'): factory(self.path,entry)

    def test_tokens_are_deterministic_local_and_untyped(self):
        provider = self.load(); views = [self.view('one'),self.view('two')]
        first = provider.recognize(views,None); second = provider.recognize(views,None)
        self.assertEqual([t.model_dump() for t in first],[t.model_dump() for t in second])
        self.assertEqual([t.raw_text for t in first],['AB','AB'])
        self.assertEqual(len({t.internal_id for t in first}),2)
        self.assertTrue(all(t.category == 'unknown' and not t.details and t.component_id is None and t.pin_id is None for t in first))
        self.assertTrue(all(t.bbox.x2 <= 80 and t.bbox.y2 <= 40 for t in first))
        with self.assertRaisesRegex(ONNXOCRAdapterError,'duplicate'): provider.recognize([views[0],views[0]],None)

    def test_budget_after_detector_prevents_recognition(self):
        provider = self.load()
        class Budget:
            calls = 0
            def check(self,*,stage):
                self.calls += 1
                if self.calls >= 2: raise TimeoutError('expired after detector')
        with self.assertRaisesRegex(TimeoutError,'expired'): provider.recognize([self.view()],SimpleNamespace(time_budget=Budget()))
        self.assertEqual(provider.detector.calls,1)
        self.assertEqual(provider.recognizer.calls,0)

    def test_multilevel_contract_and_legacy_backend_override_rejection(self):
        provider = self.load()
        tokens = run_multilevel_ocr(global_views=[self.view()],component_views=[],provider=provider,ctx=None)
        self.assertEqual(len(tokens),1)
        self.assertEqual(tokens[0].raw_text,'AB')
        with self.assertRaisesRegex(ProductionProviderError,'backend-capable'):
            _apply_ocr_inference_overrides(provider,{'recognizer_backend':'en_PP-OCRv5_mobile_rec'})

    def test_import_requires_no_optional_framework(self):
        source = str(Path(__file__).resolve().parents[2]/'src')
        code = "import sys;sys.path.insert(0,sys.argv[1]);import pcbparse.perception.onnx_ocr_adapter;assert not any(n.split('.')[0] in {'paddle','paddlex','paddleocr','cv2','torch','onnxruntime'} for n in sys.modules)"
        result = subprocess.run([sys.executable,'-I','-c',code,source],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__ == '__main__': unittest.main()
