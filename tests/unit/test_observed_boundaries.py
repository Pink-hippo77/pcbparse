"""Boundary evidence must survive model fusion without inventing terminals."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from pcbparse.graph.palette_wire import PalettePolicy,palette_cleanup_stability
from pcbparse.ir.models import BBox,CircuitIR,Component,Evidence,ImageInfo,Pin,Point
from pcbparse.perception.observed_terminal_boundaries import refine_observed_terminal_boundaries


class BoundaryTests(unittest.TestCase):
    def ic(self,wire=(0,130,0)):
        rgb=np.full((220,220,3),255,np.uint8)
        red=(200,0,0)
        rgb[60:63,60:161]=red;rgb[158:161,60:161]=red
        rgb[60:161,60:63]=red;rgb[60:161,158:161]=red
        rgb[89:92,30:61]=red;rgb[89:92,4:30]=wire
        rgb[129:132,160:191]=red;rgb[129:132,191:216]=wire
        rgb[30:61,89:92]=red;rgb[4:30,89:92]=wire
        rgb[160:191,129:132]=red;rgb[191:216,129:132]=wire
        body=BBox(x1=60,y1=60,x2=160,y2=160)
        c=Component(internal_id='chip',category='Box',bbox=body,body_bbox=body,score=.99,key='U8',
            evidence=[Evidence(evidence_id='visual-ic-evidence:chip',source='geometry')])
        pins=[]
        for i,(side,x,y,axis) in enumerate([('L',45,90,90),('R',175,130,130),('T',90,45,90),('B',130,175,130)]):
            pins.append(Pin(internal_id=side,component_id='chip',orientation=side,point=Point(x=x,y=y),
                number=str(i+1),name='SIG'+str(i),score=.99,
                evidence=[Evidence(evidence_id='row:'+side,source='geometry',
                    metadata={'kind':'visible-stroke-and-text-lanes','side':side,'position':axis})]))
        ir=CircuitIR(image=ImageInfo(image_id='synthetic',path='unused',width=220,height=220),components=[c],pins=pins)
        return ir,rgb

    def test_all_four_orientations_use_the_visible_transition(self):
        ir,rgb=self.ic();result=refine_observed_terminal_boundaries(ir,rgb)
        self.assertEqual(len(result['updates']),4)
        for side,target in [('L',(30,90)),('R',(190,130)),('T',(90,30)),('B',(130,190))]:
            p=next(p for p in ir.pins if p.orientation==side)
            self.assertLessEqual(abs(p.point.x-target[0])+abs(p.point.y-target[1]),2)

    def test_identity_and_count_are_preserved(self):
        ir,rgb=self.ic();before=[(p.internal_id,p.component_id,p.number,p.name)for p in ir.pins]
        refine_observed_terminal_boundaries(ir,rgb)
        self.assertEqual(before,[(p.internal_id,p.component_id,p.number,p.name)for p in ir.pins])

    def test_same_color_continuation_does_not_authorize_a_boundary(self):
        ir,rgb=self.ic((200,0,0));self.assertFalse(refine_observed_terminal_boundaries(ir,rgb)['updates'])

    def test_blank_space_does_not_authorize_a_boundary(self):
        ir,rgb=self.ic((255,255,255));self.assertFalse(refine_observed_terminal_boundaries(ir,rgb)['updates'])

    def test_monochrome_is_unchanged(self):
        ir,rgb=self.ic();rgb=np.repeat(rgb.min(2)[:,:,None],3,axis=2)
        self.assertFalse(refine_observed_terminal_boundaries(ir,rgb)['updates'])

    def test_unverified_rectangle_is_not_sufficient(self):
        ir,rgb=self.ic();ir.components[0].evidence=[]
        self.assertFalse(refine_observed_terminal_boundaries(ir,rgb)['updates'])

    def test_a_decorated_pin_keeps_its_separate_geometry(self):
        ir,rgb=self.ic();ir.pins[0].evidence[0].metadata['decorated_lead']=True
        old=ir.pins[0].point.model_copy(deep=True)
        self.assertEqual(len(refine_observed_terminal_boundaries(ir,rgb)['updates']),3)
        self.assertEqual(ir.pins[0].point,old)

    def test_wrong_row_does_not_take_a_neighboring_stroke(self):
        ir,rgb=self.ic();ir.pins[0].evidence[0].metadata['position']=100
        self.assertEqual(len(refine_observed_terminal_boundaries(ir,rgb)['updates']),3)

    def test_other_component_body_vetoes_a_terminal(self):
        ir,rgb=self.ic();b=BBox(x1=28,y1=85,x2=35,y2=95)
        ir.components.append(Component(internal_id='other',category='R',bbox=b,body_bbox=b,score=.99))
        self.assertEqual(len(refine_observed_terminal_boundaries(ir,rgb)['updates']),3)

    def test_large_capacitor_stems_are_measured_beyond_24px(self):
        rgb=np.full((220,220,3),255,np.uint8);red=(200,0,0);green=(0,130,0)
        rgb[100:103,80:121]=red;rgb[110:113,80:121]=red
        rgb[40:101,99:102]=red;rgb[112:171,99:102]=red
        rgb[5:40,99:102]=green;rgb[171:215,99:102]=green
        b=BBox(x1=80,y1=101,x2=120,y2=111)
        c=Component(internal_id='cap',category='C',bbox=b,body_bbox=b,score=.99,
            evidence=[Evidence(evidence_id='cap',source='geometry',metadata={'method':'visible-capacitor-plates-and-reference'})])
        ir=CircuitIR(image=ImageInfo(image_id='synthetic',path='unused',width=220,height=220),components=[c],
            pins=[Pin(internal_id='top',component_id='cap',orientation='T',point=Point(x=100,y=82)),
                  Pin(internal_id='bottom',component_id='cap',orientation='B',point=Point(x=100,y=130))])
        self.assertEqual(len(refine_observed_terminal_boundaries(ir,rgb)['updates']),2)
        self.assertLessEqual(abs(ir.pins[0].point.y-40),2)
        self.assertLessEqual(abs(ir.pins[1].point.y-170),2)

    def test_endpoint_repair_does_not_newly_enable_destructive_palette_filter(self):
        ir,rgb=self.ic();refine_observed_terminal_boundaries(ir,rgb)
        with patch('pcbparse.graph.palette_wire.estimate_palette',return_value={'enabled':False,'reason':'insufficient_anchor_agreement'}):
            result=palette_cleanup_stability(ir,None,None,{'enabled':True,'wire_hue_bin':4},PalettePolicy())
        self.assertFalse(result['stable'])
        self.assertEqual(result['reason'],'terminal_refinement_changes_palette')

    def test_agreeing_independent_palette_remains_enabled(self):
        ir,rgb=self.ic();refine_observed_terminal_boundaries(ir,rgb)
        with patch('pcbparse.graph.palette_wire.estimate_palette',return_value={'enabled':True,'wire_hue_bin':4}):
            result=palette_cleanup_stability(ir,None,None,{'enabled':True,'wire_hue_bin':4},PalettePolicy())
        self.assertTrue(result['stable'])

    def test_conflicting_independent_color_vetoes_filter(self):
        ir,rgb=self.ic();refine_observed_terminal_boundaries(ir,rgb)
        with patch('pcbparse.graph.palette_wire.estimate_palette',return_value={'enabled':True,'wire_hue_bin':0}):
            result=palette_cleanup_stability(ir,None,None,{'enabled':True,'wire_hue_bin':4},PalettePolicy())
        self.assertFalse(result['stable'])

    def test_terminal_and_duplicate_cleanup_must_pass_their_combined_prior_view(self):
        ir,rgb=self.ic();refine_observed_terminal_boundaries(ir,rgb)
        duplicate=ir.components[0].model_copy(update={'internal_id':'discarded'})
        ir.metadata['tiled_ic_fragments']={'palette_observations':[
            {'component':duplicate.model_dump(),'pins':[]}]}
        seen=[]
        def estimate(view,*args):
            has_duplicate='discarded' in view.component_index()
            original_left=view.pin_index()['L'].point.x==45
            seen.append((has_duplicate,original_left))
            return {'enabled':not(has_duplicate and original_left),'wire_hue_bin':4}
        with patch('pcbparse.graph.palette_wire.estimate_palette',side_effect=estimate):
            result=palette_cleanup_stability(ir,None,None,{'enabled':True,'wire_hue_bin':4},PalettePolicy())
        self.assertFalse(result['stable'])
        self.assertIn((True,True),seen)


if __name__=='__main__':
    unittest.main()
