"""Preserve detector/plate consensus and complete printed label stacks."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from pcbparse.ir.models import BBox,CircuitIR,Component,Evidence,ImageInfo,TextToken
from pcbparse.perception.capacitor_symbol_extent import restore_capacitor_symbol_extents
from pcbparse.perception.passive_visual_readback import _capacitor_label_stack,recover_visual_capacitors
from pcbparse.perception.symbol_body import refine_component_body_boxes
from pcbparse.output.official_serializer import serialize_official
from pcbparse.ir.models import IRState


class CapacitorTests(unittest.TestCase):
    def fixture(self,horizontal=False):
        rgb=np.full((220,220,3),255,np.uint8);red=(200,0,0);green=(0,130,0)
        rgb[100:103,80:121]=red;rgb[110:113,80:121]=red
        rgb[40:101,99:102]=red;rgb[112:171,99:102]=red
        rgb[5:40,99:102]=green;rgb[171:215,99:102]=green
        plate=BBox(x1=80,y1=101,x2=120,y2=111)
        model=BBox(x1=76,y1=90,x2=124,y2=122)
        if horizontal:
            rgb=rgb.transpose(1,0,2)
            plate=BBox(x1=101,y1=80,x2=111,y2=120)
            model=BBox(x1=90,y1=76,x2=122,y2=124)
        c=Component(internal_id='cap',category='C',bbox=model,body_bbox=plate,score=.99,key='C8',
            evidence=[Evidence(evidence_id='plate',source='geometry',metadata={'method':'visible-capacitor-plates-and-reference'}),
                Evidence(evidence_id='detector',source='detector',metadata={'local_to_global_binding':{
                    'original_geometry':{'values':[model.x1,model.y1,model.x2,model.y2]}}})])
        ir=CircuitIR(image=ImageInfo(image_id='synthetic',path='unused',width=220,height=220),components=[c])
        return ir,rgb

    def model(self,ir,values):
        ir.components[0].evidence[-1].metadata['local_to_global_binding']['original_geometry']['values']=values

    def test_vertical_extent_keeps_visible_plates_and_detector_normal_extent(self):
        ir,rgb=self.fixture();self.assertEqual(len(restore_capacitor_symbol_extents(ir,rgb)['updates']),1)
        self.assertEqual(ir.components[0].body_bbox,BBox(x1=80,y1=90,x2=120,y2=122))

    def test_horizontal_extent_has_the_same_evidence_requirements(self):
        ir,rgb=self.fixture(True);self.assertEqual(len(restore_capacitor_symbol_extents(ir,rgb)['updates']),1)
        self.assertEqual(ir.components[0].body_bbox,BBox(x1=90,y1=80,x2=122,y2=120))

    def test_generated_capacitor_has_no_independent_model_extent(self):
        ir,rgb=self.fixture();ir.components[0].evidence.pop()
        self.assertFalse(restore_capacitor_symbol_extents(ir,rgb)['updates'])

    def test_color_transition_prevents_extent_from_covering_the_wire(self):
        ir,rgb=self.fixture();self.model(ir,[76,30,124,122])
        self.assertFalse(restore_capacitor_symbol_extents(ir,rgb)['updates'])

    def test_broken_stem_prevents_extension(self):
        ir,rgb=self.fixture();rgb[93:98,99:102]=255
        self.assertFalse(restore_capacitor_symbol_extents(ir,rgb)['updates'])

    def test_model_must_enclose_both_plates(self):
        ir,rgb=self.fixture();self.model(ir,[76,105,124,122])
        self.assertFalse(restore_capacitor_symbol_extents(ir,rgb)['updates'])

    def test_model_must_be_coaxial(self):
        ir,rgb=self.fixture();self.model(ir,[90,90,138,122])
        self.assertFalse(restore_capacitor_symbol_extents(ir,rgb)['updates'])

    def test_large_unrelated_model_box_is_not_accepted(self):
        ir,rgb=self.fixture();self.model(ir,[20,20,180,190])
        self.assertFalse(restore_capacitor_symbol_extents(ir,rgb)['updates'])

    def test_low_confidence_detection_does_not_supply_extent(self):
        ir,rgb=self.fixture();ir.components[0].score=.5
        self.assertFalse(restore_capacitor_symbol_extents(ir,rgb)['updates'])

    def test_later_body_measurement_and_serializer_keep_the_validated_extent(self):
        ir,rgb=self.fixture();restore_capacitor_symbol_extents(ir,rgb)
        with patch('pcbparse.perception.symbol_body.readback_rgb',return_value=rgb):
            refine_component_body_boxes(ir,'unused')
        ir.state=IRState.VALIDATED
        output=serialize_official(ir)
        self.assertEqual(output['components']['C8']['bbox'],[80.,98.,120.,130.])

    def test_new_measurement_without_consensus_still_returns_the_plate_body(self):
        ir,rgb=self.fixture()
        with patch('pcbparse.perception.symbol_body.readback_rgb',return_value=rgb):
            refine_component_body_boxes(ir,'unused')
        self.assertEqual(ir.components[0].body_bbox,BBox(x1=80,y1=101,x2=120,y2=111))

    def test_stale_extent_does_not_override_different_observed_plates(self):
        ir,rgb=self.fixture();restore_capacitor_symbol_extents(ir,rgb)
        rgb=np.roll(rgb,10,axis=0)
        c=ir.components[0];c.bbox=BBox(x1=76,y1=100,x2=124,y2=132)
        with patch('pcbparse.perception.symbol_body.readback_rgb',return_value=rgb):
            refine_component_body_boxes(ir,'unused')
        self.assertEqual(c.body_bbox,BBox(x1=80,y1=111,x2=120,y2=121))

    def text(self,key,text,box,score=.99):
        return TextToken(internal_id=key,raw_text=text,bbox=BBox(**dict(zip(('x1','y1','x2','y2'),box))),score=score)

    def test_label_stack_recovers_symbol_cut_off_by_first_line_roi(self):
        rgb=np.full((200,200,3),255,np.uint8);red=(200,0,0)
        rgb[100:137,94:97]=red;rgb[100:137,104:107]=red
        rgb[117:120,65:95]=red;rgb[117:120,106:136]=red
        ref=self.text('ref','C8',[90,30,110,50]);value=self.text('value','100nF',[80,57,120,77])
        def ir():
            return CircuitIR(image=ImageInfo(image_id='synthetic',path='unused',width=200,height=200),texts=[ref.model_copy(deep=True),value.model_copy(deep=True)])
        old=ir();recover_visual_capacitors(old,'unused',rgb=rgb,occluded_plates=False)
        new=ir();recover_visual_capacitors(new,'unused',rgb=rgb,occluded_plates=True)
        self.assertFalse(old.components)
        self.assertEqual([c.key for c in new.components],['C8'])
        self.assertEqual(len(new.pins),2)

    def test_ambiguous_second_lines_do_not_expand_the_search(self):
        ref=self.text('ref','C8',[90,30,110,50])
        a=self.text('a','100nF',[80,57,120,77]);b=self.text('b','10uF',[80,57,120,77])
        self.assertEqual(_capacitor_label_stack(ref,[ref,a,b]),ref.bbox)

    def test_neighbor_value_is_not_a_label_stack(self):
        ref=self.text('ref','C8',[90,30,110,50]);v=self.text('v','100nF',[130,57,170,77])
        self.assertEqual(_capacitor_label_stack(ref,[ref,v]),ref.bbox)

    def test_low_confidence_value_is_not_a_label_stack(self):
        ref=self.text('ref','C8',[90,30,110,50]);v=self.text('v','100nF',[80,57,120,77],.8)
        self.assertEqual(_capacitor_label_stack(ref,[ref,v]),ref.bbox)


if __name__=='__main__':
    unittest.main()
