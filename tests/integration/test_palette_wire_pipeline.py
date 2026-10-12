import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pcbparse.config import load_config
from pcbparse.pipeline.orchestrator import infer_one
from tests.unit.test_palette_wire import color_fixture


class PaletteWirePipelineTests(unittest.TestCase):
    def test_strict_validation_determinism_and_image_only_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'fixture.png'
            ir=color_fixture(source)
            (root/'fixture.json').write_text('invalid sidecar trap',encoding='utf-8')
            class Provider:
                def build_ir(self,path,ctx):return ir.model_copy(deep=True)
            config=load_config('configs/pipeline.edarace.yaml')
            config['graph']['palette_wire_filter_enabled']=True
            a=infer_one(source,root/'a.json',provider=Provider(),config=config)
            b=infer_one(source,root/'b.json',provider=Provider(),config=config)
            self.assertEqual((root/'a.json').read_bytes(),(root/'b.json').read_bytes())
            self.assertEqual(set(a.output),{'components','pins','nets'})
            self.assertFalse([d for d in a.diagnostics if d.severity=='ERROR'])
            self.assertEqual(a.ir.metadata['palette_wire_filter']['removed_segment_ids'],['s0','s1'])
            self.assertEqual(a.ir.metadata['palette_wire_filter'],b.ir.metadata['palette_wire_filter'])

    def test_default_off_skips_pixel_filter_and_gray_output_is_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'fixture.png'
            ir=color_fixture(source,'black','black')
            class Provider:
                def build_ir(self,path,ctx):return ir.model_copy(deep=True)
            config=load_config('configs/pipeline.yaml')
            config['graph'].pop('palette_wire_filter_enabled',None)
            with mock.patch('pcbparse.pipeline.orchestrator.prepare_palette_wire') as prepare:
                infer_one(source,root/'off.json',provider=Provider(),config=config)
                prepare.assert_not_called()
            config['graph']['palette_wire_filter_enabled']=True
            result=infer_one(source,root/'on.json',provider=Provider(),config=config)
            self.assertEqual((root/'off.json').read_bytes(),(root/'on.json').read_bytes())
            self.assertFalse(result.ir.metadata['palette_wire_filter']['palette']['enabled'])

    def test_invalid_flag_fails_before_provider_and_serialization(self):
        for value in ('true',1,None):
            with self.subTest(value=value),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);source=root/'fixture.png';color_fixture(source)
                config=load_config('configs/pipeline.yaml');config['graph']['palette_wire_filter_enabled']=value
                provider=mock.Mock()
                with self.assertRaisesRegex(ValueError,'must be a boolean'):
                    infer_one(source,root/'output.json',provider=provider,config=config)
                provider.build_ir.assert_not_called()
                self.assertFalse((root/'output.json').exists())


if __name__=='__main__':unittest.main()
