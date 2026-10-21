"""Long trims require matching current pixel evidence; branches stay immutable."""
from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from pcbparse.ir.models import BBox,CircuitIR,Component,Evidence,ImageInfo,Pin,Point,PhysicalNet,WireSegment
from pcbparse.perception.observed_terminal_boundaries import refine_observed_terminal_boundaries
from pcbparse.graph.terminal_wire_geometry import trim_terminal_leads

class MeasuredLeadTrimmingTests(unittest.TestCase):
    def sample(self,wire=(0,130,0)):
        rgb=np.full((220,220,3),255,np.uint8);red=(200,0,0)
        rgb[100:103,80:121]=red;rgb[110:113,80:121]=red
        rgb[40:101,99:102]=red;rgb[112:171,99:102]=red
        rgb[5:40,99:102]=wire;rgb[171:215,99:102]=wire
        b=BBox(x1=80,y1=101,x2=120,y2=111)
        c=Component(internal_id='c',category='C',bbox=b,body_bbox=b,score=1,
            evidence=[Evidence(evidence_id='cap',source='geometry',metadata={'method':'visible-capacitor-plates-and-reference'})])
        p=Pin(internal_id='top',component_id='c',orientation='T',point=Point(x=100,y=40),
              base=Point(x=100,y=101),score=1)
        line=WireSegment(internal_id='s',start=Point(x=100,y=100),end=Point(x=100,y=5))
        ir=CircuitIR(image=ImageInfo(image_id='synthetic',path='unused',width=220,height=220),components=[c],
            pins=[p],segments=[line],physical_nets=[PhysicalNet(internal_id='n',pins=['top'],segments=['s'])])
        return ir,rgb

    def test_already_correct_tip_retains_measurement_for_long_trim(self):
        ir,rgb=self.sample();report=refine_observed_terminal_boundaries(ir,rgb)
        self.assertEqual(report['updates'],[])
        self.assertEqual(len(report['measurements']),1)
        before=ir.pins[0].point.model_copy()
        updates=trim_terminal_leads(ir)
        self.assertEqual(len(updates),1)
        self.assertEqual(ir.segments[0].start.y,40)
        self.assertEqual(ir.pins[0].point,before)
        self.assertEqual(ir.physical_nets[0].pins,['top'])

    def test_model_point_without_visible_boundary_cannot_authorize_long_trim(self):
        ir,rgb=self.sample(wire=(200,0,0));refine_observed_terminal_boundaries(ir,rgb)
        self.assertEqual(trim_terminal_leads(ir),[])

    def test_stale_tip_base_and_wrong_owner_evidence_are_rejected(self):
        for change in ('tip','base','owner'):
            ir,rgb=self.sample();refine_observed_terminal_boundaries(ir,rgb)
            if change=='tip':ir.pins[0].point.y-=3
            if change=='base':ir.pins[0].base.y-=5
            if change=='owner':ir.pins[0].evidence[-1].metadata['component_id']='other'
            with self.subTest(change=change):self.assertEqual(trim_terminal_leads(ir),[])

    def test_branch_vertex_and_unrelated_parallel_stroke_are_preserved(self):
        for change in ('branch','offset'):
            ir,rgb=self.sample();refine_observed_terminal_boundaries(ir,rgb)
            if change=='branch':
                ir.segments.append(WireSegment(internal_id='branch',start=Point(x=100,y=100),end=Point(x=150,y=100)))
                ir.physical_nets[0].segments.append('branch')
            else:
                ir.segments[0].start.x+=5;ir.segments[0].end.x+=5
            with self.subTest(change=change):self.assertEqual(trim_terminal_leads(ir),[])

if __name__=='__main__':unittest.main()
