from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from pcbparse.graph.palette_wire import apply_palette_wire_filter, prepare_palette_wire
from pcbparse.ir.models import BBox, Component, Junction, Pin, Point, TextToken, WireSegment
from pcbparse.pipeline.providers import EmptyImageProvider


def color_fixture(path, wire='green', symbol='red'):
    bitmap=Image.new('RGB',(240,160),'white');draw=ImageDraw.Draw(bitmap)
    bitmap.save(path)
    ir=EmptyImageProvider().build_ir(path)
    for i in range(3):
        low=10+i*45
        ir.components.append(Component(internal_id=f'c{i}',key=f'U{i+1}',category='Box',
                                       bbox=BBox(x1=30,y1=low,x2=60,y2=low+35)))
        for j in range(2):
            y=low+10+j*15
            ir.pins.append(Pin(internal_id=f'p{i}{j}',component_id=f'c{i}',number=str(j+1),
                               orientation='R',point=Point(x=60,y=y)))
            draw.line((60,y,105,y),fill=wire,width=1)
            ir.segments.append(WireSegment(internal_id=f'w{i}{j}',start=Point(x=60,y=y),end=Point(x=105,y=y)))
    for i in range(2):
        y=25+i*40
        draw.line((145,y,200,y),fill=symbol,width=1)
        ir.segments.append(WireSegment(internal_id=f's{i}',start=Point(x=145,y=y),end=Point(x=200,y=y)))
    bitmap.save(path)
    ir.image=EmptyImageProvider().build_ir(path).image
    return ir


class PaletteWireTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'image.png'

    def test_wire_palette_changes_with_image_not_fixed_color(self):
        for wire,symbol in (('green','red'),('blue','red'),('purple','blue')):
            with self.subTest(wire=wire):
                ir=color_fixture(self.path,wire,symbol);identity=copy.deepcopy((ir.components,ir.pins,ir.texts))
                context=prepare_palette_wire(self.path,ir)
                report=apply_palette_wire_filter(ir,context)
                self.assertTrue(context.palette['enabled'])
                self.assertEqual(report['removed_segment_ids'],['s0','s1'])
                self.assertEqual(identity,(ir.components,ir.pins,ir.texts))

    def test_grayscale_and_mixed_palette_abstain_without_geometry_mutation(self):
        ir=color_fixture(self.path,'black','black')
        before=ir.model_dump(exclude={'metadata'})
        context=prepare_palette_wire(self.path,ir)
        self.assertFalse(context.palette['enabled'])
        apply_palette_wire_filter(ir,context)
        self.assertEqual(before,ir.model_dump(exclude={'metadata'}))
        ir=color_fixture(self.path)
        with Image.open(self.path) as original:bitmap=original.copy()
        draw=ImageDraw.Draw(bitmap)
        for pin in ir.pins[:3]:draw.line((60,pin.point.y,105,pin.point.y),fill='blue')
        bitmap.save(self.path);ir.image=EmptyImageProvider().build_ir(self.path).image
        context=prepare_palette_wire(self.path,ir)
        self.assertFalse(context.palette['enabled'])
        self.assertLess(context.palette['agreement'],.75)

    def test_erroneous_ocr_boxes_cannot_censor_wire_anchors(self):
        ir=color_fixture(self.path);first=prepare_palette_wire(self.path,ir)
        ir.texts=[TextToken(internal_id='bad-box',raw_text='bad',bbox=BBox(x1=0,y1=0,x2=240,y2=160))]
        second=prepare_palette_wire(self.path,ir)
        self.assertEqual(first.palette,second.palette)

    def test_missing_owner_and_orientation_do_not_create_anchors(self):
        ir=color_fixture(self.path)
        for pin in ir.pins[:3]:pin.component_id=None
        for pin in ir.pins[3:]:pin.orientation='UNKNOWN'
        self.assertFalse(prepare_palette_wire(self.path,ir).palette['enabled'])

    def test_original_pin_lead_survives_degraded_pruning(self):
        ir=color_fixture(self.path)
        with Image.open(self.path) as original:bitmap=original.copy()
        ImageDraw.Draw(bitmap).line((45,20,64,20),fill='red')
        bitmap.save(self.path);ir.image=EmptyImageProvider().build_ir(self.path).image
        ir.segments.append(WireSegment(internal_id='lead',start=Point(x=45,y=20),end=Point(x=64,y=20)))
        context=prepare_palette_wire(self.path,ir)
        ir.pins[0].point=Point(x=230,y=150)  # Association cannot move frozen protection geometry.
        ir.pins=[]
        report=apply_palette_wire_filter(ir,context)
        self.assertIn('lead',{s.internal_id for s in ir.segments})
        self.assertEqual(report['protected_pin_leads'],1)

    def test_explicit_membership_and_synthetic_segments_are_protected(self):
        ir=color_fixture(self.path)
        ir.segments[-2].source_edge_id='explicit'
        ir.pins[0].edge_ids=['explicit']
        ir.segments[-1].synthetic=True
        report=apply_palette_wire_filter(ir,prepare_palette_wire(self.path,ir))
        self.assertEqual(report['removed_segment_ids'],[])

    def test_insufficient_junctions_are_removed_and_valid_bends_remain(self):
        ir=color_fixture(self.path)
        ir.junctions=[Junction(internal_id='cross',point=Point(x=60,y=20),kind='connected_cross',
                              segment_ids=['s0','s1','w00','w01']),
                      Junction(internal_id='bend',point=Point(x=60,y=20),kind='bend',segment_ids=['w00','w01'])]
        report=apply_palette_wire_filter(ir,prepare_palette_wire(self.path,ir))
        self.assertEqual(report['removed_invalid_junction_ids'],['cross'])
        self.assertEqual([j.internal_id for j in ir.junctions],['bend'])

    def test_pixel_source_binding_rejects_hash_and_dimension_mismatches(self):
        ir=color_fixture(self.path);ir.image.sha256='0'*64
        with self.assertRaisesRegex(ValueError,'hash'):prepare_palette_wire(self.path,ir)
        ir.image=EmptyImageProvider().build_ir(self.path).image;ir.image.width+=1
        with self.assertRaisesRegex(ValueError,'dimensions'):prepare_palette_wire(self.path,ir)

    def test_order_independent_decisions_and_preserved_gray_segments(self):
        ir=color_fixture(self.path)
        with Image.open(self.path) as original:bitmap=original.copy()
        ImageDraw.Draw(bitmap).line((145,125,200,125),fill='black')
        bitmap.save(self.path);ir.image=EmptyImageProvider().build_ir(self.path).image
        ir.segments.append(WireSegment(internal_id='gray',start=Point(x=145,y=125),end=Point(x=200,y=125)))
        other=ir.model_copy(deep=True);other.pins.reverse();other.segments.reverse();other.components.reverse()
        a=apply_palette_wire_filter(ir,prepare_palette_wire(self.path,ir))
        b=apply_palette_wire_filter(other,prepare_palette_wire(self.path,other))
        self.assertEqual(a,b)
        self.assertIn('gray',{s.internal_id for s in ir.segments})


if __name__=='__main__':unittest.main()
