"""Only independently visible two-terminal symbol/wire transitions may move tips."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from pcbparse.ir.models import BBox,CircuitIR,Component,Evidence,ImageInfo,Pin,Point
from pcbparse.perception.observed_terminal_boundaries import refine_observed_terminal_boundaries


class TwoTerminalBoundaryTests(unittest.TestCase):
    def sample(self, horizontal=False, wire=(0,130,0), verified=True):
        rgb=np.full((180,180,3),255,np.uint8)
        red=(200,0,0)
        rgb[60:63,80:101]=red;rgb[108:111,80:101]=red
        rgb[60:111,80:83]=red;rgb[60:111,98:101]=red
        rgb[35:61,89:92]=red;rgb[110:136,89:92]=red
        rgb[5:35,89:92]=wire;rgb[136:175,89:92]=wire
        b=BBox(x1=80,y1=60,x2=100,y2=110)
        pins=[Pin(internal_id='p1',component_id='r',orientation='T',point=Point(x=90,y=49),
                  number='1',name='a',score=1),
              Pin(internal_id='p2',component_id='r',orientation='B',point=Point(x=90,y=121),
                  number='2',name='b',score=1)]
        if horizontal:
            rgb=rgb.transpose(1,0,2).copy()
            b=BBox(x1=60,y1=80,x2=110,y2=100)
            for pin,side in zip(pins,'LR'):
                pin.orientation=side;pin.point=Point(x=pin.point.y,y=pin.point.x)
        evidence=[Evidence(evidence_id='rl:r',source='geometry',
            metadata={'method':'printed-reference-and-two-terminal-strokes'})] if verified else []
        c=Component(internal_id='r',category='R',bbox=b,body_bbox=b,score=1,key='R7',evidence=evidence)
        return CircuitIR(image=ImageInfo(image_id='synthetic',path='unused',width=180,height=180),
                         components=[c],pins=pins),rgb

    def test_both_axes_use_pixels_and_preserve_identity(self):
        for horizontal in (False,True):
            with self.subTest(horizontal=horizontal):
                ir,rgb=self.sample(horizontal)
                before=[(p.internal_id,p.component_id,p.name,p.number) for p in ir.pins]
                report=refine_observed_terminal_boundaries(ir,rgb)
                self.assertEqual(len(report['updates']),2)
                self.assertEqual(before,[(p.internal_id,p.component_id,p.name,p.number) for p in ir.pins])
                for pin,expected in zip(ir.pins,(35,135)):
                    self.assertLessEqual(abs((pin.point.x if horizontal else pin.point.y)-expected),2)

    def test_same_color_and_blank_are_not_measurable_boundaries(self):
        for wire in ((200,0,0),(255,255,255)):
            ir,rgb=self.sample(wire=wire)
            self.assertEqual(refine_observed_terminal_boundaries(ir,rgb)['updates'],[])

    def test_unverified_symbol_does_not_move(self):
        ir,rgb=self.sample(verified=False)
        self.assertEqual(refine_observed_terminal_boundaries(ir,rgb)['updates'],[])

    def test_misaligned_and_same_side_terminals_abstain(self):
        for kind in ('offset','same_side','extra','missing'):
            ir,rgb=self.sample()
            if kind=='offset':ir.pins[1].point.x+=5
            if kind=='same_side':ir.pins[1].orientation='T'
            if kind=='extra':ir.pins.append(ir.pins[0].model_copy(update={'internal_id':'p3'}))
            if kind=='missing':ir.pins.pop()
            with self.subTest(kind=kind):
                self.assertEqual(refine_observed_terminal_boundaries(ir,rgb)['updates'],[])

    def test_an_unrelated_two_pin_component_is_not_eligible(self):
        ir,rgb=self.sample();ir.components[0].category='D'
        self.assertEqual(refine_observed_terminal_boundaries(ir,rgb)['updates'],[])

    def test_neighboring_symbol_interior_vetoes_the_tip(self):
        ir,rgb=self.sample();b=BBox(x1=86,y1=30,x2=95,y2=40)
        ir.components.append(Component(internal_id='neighbor',category='R',bbox=b,body_bbox=b))
        self.assertEqual(len(refine_observed_terminal_boundaries(ir,rgb)['updates']),1)


if __name__=='__main__':unittest.main()
