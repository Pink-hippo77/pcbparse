from __future__ import annotations

import unittest
from unittest.mock import patch

from pcbparse.association.ocr import associate_ocr
from pcbparse.association.ocr_common import OCRAssociationConfig
from pcbparse.association.pin_label_context import BodyPinLabelContext
from pcbparse.association.semantic_fields import infer_semantic_fields
from pcbparse.ir.models import BBox, CircuitIR, Component, Evidence, ImageInfo, Pin, Point, TextToken


class PinBodyLabelContextTests(unittest.TestCase):
    def setUp(self):
        self.body = Component(internal_id='component:body',category='Box',
                              bbox=BBox(x1=100,y1=100,x2=300,y2=200))
        self.pin = Pin(internal_id='pin:left',component_id=self.body.internal_id,
                       point=Point(x=100,y=120),base=Point(x=108,y=120),orientation='L')
        self.number = self.token('number','23',BBox(x1=101,y1=112,x2=112,y2=123))

    def token(self,name,value,bbox,*,details=''):
        return TextToken(internal_id='text:'+name,raw_text=value,score=.99,bbox=bbox,details=details,
                         evidence=[Evidence(evidence_id='ocr:'+name,source='ocr',metadata={'level':'global'})])

    def infer(self,texts,enabled=True):
        return infer_semantic_fields([self.body],[self.pin],texts,[],pin_max_distance_px=36.,
                                     pin_body_name_context_enabled=enabled)

    def test_paired_pin_function_beats_reference_shape_without_assigning_owners(self):
        name=self.token('name','PA15',BBox(x1=117,y1=113,x2=155,y2=125))
        summary=self.infer([name,self.number])
        self.assertEqual(name.details,'pinname')
        self.assertEqual(name.raw_text,'PA15')
        self.assertIsNone(name.component_id)
        self.assertIsNone(name.pin_id)
        self.assertIsNone(self.pin.number)
        self.assertIsNone(self.body.key)
        self.assertEqual(summary['pin_body_name_context']['field_overrides'],1)
        rule=name.evidence[-1].metadata['paired_body_pin_label']
        self.assertEqual(rule['number_token_id'],self.number.internal_id)
        self.assertEqual(rule['previous_field'],'designator')

    def test_logic_name_that_looks_like_amperes_still_keeps_exact_spelling(self):
        name=self.token('name','1A',BBox(x1=117,y1=113,x2=134,y2=125))
        self.infer([name,self.number])
        self.assertEqual(name.details,'pinname')
        self.assertEqual(name.raw_text,'1A')
        self.assertEqual(self.number.raw_text,'23')
        self.assertEqual(self.number.details,'pinnumber')

    def test_reference_and_long_part_identifier_are_protected(self):
        for value in ('U2','R10','LM2578A'):
            with self.subTest(value=value):
                name=self.token('name',value,BBox(x1=117,y1=113,x2=155,y2=125))
                self.infer([name,self.number])
                self.assertEqual(name.details,'designator')

    def test_missing_number_and_wrong_row_do_not_override(self):
        for numbers in ([],[self.token('far','23',BBox(x1=101,y1=155,x2=112,y2=166))]):
            with self.subTest(numbers=numbers):
                name=self.token('name','PA15',BBox(x1=117,y1=113,x2=155,y2=125))
                self.infer([name,*numbers])
                self.assertEqual(name.details,'designator')

    def test_body_center_outside_body_and_wrong_peer_owner_do_not_override(self):
        for bbox,owner in ((BBox(x1=210,y1=113,x2=250,y2=125),None),
                           (BBox(x1=80,y1=113,x2=95,y2=125),None),
                           (BBox(x1=117,y1=113,x2=155,y2=125),'component:other')):
            with self.subTest(bbox=bbox,owner=owner):
                name=self.token('name','PA15',bbox)
                number=self.number.model_copy(deep=True);number.component_id=owner
                context=BodyPinLabelContext([self.body],[self.pin],[name,number],36.)
                self.assertIsNone(context.hint(name,name.raw_text))

    def test_disabled_and_authoritative_tokens_keep_their_original_path(self):
        name=self.token('name','PA15',BBox(x1=117,y1=113,x2=155,y2=125))
        self.infer([name,self.number],enabled=False)
        self.assertEqual(name.details,'designator')
        typed=self.token('typed','PA15',name.bbox,details='value')
        self.infer([typed,self.number])
        self.assertEqual(typed.details,'value')
        annotation=self.token('annotation','PA15',name.bbox)
        annotation.evidence=[Evidence(evidence_id='dataset:one',source='dataset')]
        self.infer([annotation,self.number])
        self.assertEqual(annotation.details,'')

    def test_order_and_unicode_peer_do_not_change_context(self):
        name=self.token('name','PA15',BBox(x1=117,y1=113,x2=155,y2=125))
        number=self.number.model_copy(deep=True);number.raw_text='２３'
        a=BodyPinLabelContext([self.body],[self.pin],[name,number],36.).hint(name,'PA15')
        b=BodyPinLabelContext([self.body],[self.pin],[number,name],36.).hint(name,'PA15')
        self.assertEqual(a,b)
        self.assertIsNotNone(a)
        self.assertEqual(number.raw_text,'２３')

    def test_pairing_follows_right_top_and_bottom_pin_axes(self):
        body=Component(internal_id='body',category='IC',bbox=BBox(x1=100,y1=100,x2=300,y2=300))
        cases=(('R',Point(x=300,y=120),Point(x=292,y=120),
                BBox(x1=249,y1=113,x2=282,y2=125),BBox(x1=291,y1=113,x2=299,y2=123)),
               ('T',Point(x=120,y=100),Point(x=120,y=108),
                BBox(x1=114,y1=117,x2=126,y2=146),BBox(x1=115,y1=101,x2=125,y2=113)),
               ('B',Point(x=120,y=300),Point(x=120,y=292),
                BBox(x1=114,y1=255,x2=126,y2=285),BBox(x1=115,y1=289,x2=125,y2=299)))
        for side,point,base,name_box,number_box in cases:
            with self.subTest(side=side):
                pin=Pin(internal_id='pin',component_id='body',point=point,base=base,orientation=side)
                name=self.token('name','PA15',name_box);number=self.token('number','23',number_box)
                hint=BodyPinLabelContext([body],[pin],[name,number],36.).hint(name,'PA15')
                self.assertIsNotNone(hint)
                self.assertEqual(hint['pin_id'],'pin')

    def test_configuration_requires_an_explicit_boolean(self):
        for value in ('true',1,None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    OCRAssociationConfig(pin_body_name_context_enabled=value)

    def component_token(self, value, details=''):
        token = self.token('roi-name', value, BBox(x1=117, y1=113, x2=155, y2=125), details=details)
        token.evidence[0].metadata.update(level='component', component_id=self.body.internal_id)
        return token

    def infer_component_context(self, texts, enabled=True, **kwargs):
        return infer_semantic_fields(
            [self.body], [self.pin], texts, [], pin_max_distance_px=36.,
            pin_body_name_context_enabled=True,
            pin_body_component_roi_context_enabled=enabled, **kwargs,
        )

    def test_component_roi_opt_in_classifies_a_role_without_creating_identity(self):
        control = self.component_token('PA15')
        control_summary = self.infer_component_context([control, self.number], enabled=False)
        self.assertNotEqual(control.details, 'pinname')
        self.assertNotIn('component_roi_enabled', control_summary['pin_body_name_context'])
        name = self.component_token('PA15')
        summary = self.infer_component_context([name, self.number])
        self.assertEqual(name.details, 'pinname')
        self.assertEqual(name.raw_text, 'PA15')
        self.assertIsNone(name.pin_id)
        self.assertIsNone(name.component_id)
        self.assertIsNone(self.pin.number)
        self.assertIsNone(self.body.key)
        self.assertEqual(summary['pin_body_name_context']['component_roi_field_overrides'], 1)

    def test_component_roi_preserves_typed_dataset_reference_and_model_tokens(self):
        for value, details, dataset, models in (
            ('U2', '', False, frozenset()), ('R10', '', False, frozenset()),
            ('PA15', 'value', False, frozenset()), ('PA15', '', True, frozenset()),
            ('LM2578A', '', False, frozenset({'LM2578A'})),
        ):
            with self.subTest(value=value, details=details, dataset=dataset):
                name = self.component_token(value, details)
                if dataset:
                    name.evidence.append(Evidence(evidence_id='dataset:one', source='dataset'))
                self.infer_component_context([name, self.number], known_model_names=models)
                self.assertNotEqual(name.details, 'pinname')
                self.assertEqual(name.raw_text, value)

    def test_component_roi_requires_visible_number_on_the_correct_row_and_owner(self):
        wrong_owner = self.number.model_copy(deep=True)
        wrong_owner.component_id = 'component:other'
        for peers in (
            [], [self.token('far', '23', BBox(x1=101, y1=155, x2=112, y2=166))], [wrong_owner],
        ):
            with self.subTest(peers=peers):
                name = self.component_token('PA15')
                self.infer_component_context([name, *peers])
                self.assertNotEqual(name.details, 'pinname')

    def test_component_roi_switch_does_not_change_side_strip_classification(self):
        tokens = []
        for enabled in (False, True):
            token = self.component_token('PA15')
            token.evidence[0].metadata.update(level='side_strip', side='L')
            self.infer_component_context([token, self.number.model_copy(deep=True)], enabled=enabled)
            tokens.append(token.model_dump(mode='json'))
        self.assertEqual(tokens[0], tokens[1])

    def test_component_roi_switch_rejects_invalid_boolean_and_missing_parent(self):
        for value in ('true', 1, None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    OCRAssociationConfig(pin_body_name_context_enabled=True,
                                         pin_body_component_roi_context_enabled=value)
                with self.assertRaises(ValueError):
                    infer_semantic_fields([], [], [], [], pin_body_name_context_enabled=True,
                                          pin_body_component_roi_context_enabled=value)
        with self.assertRaises(ValueError):
            OCRAssociationConfig(pin_body_component_roi_context_enabled=True)
        with self.assertRaises(ValueError):
            infer_semantic_fields([], [], [], [], pin_body_component_roi_context_enabled=True)

    def test_association_forwards_component_roi_opt_in(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                ir = CircuitIR(image=ImageInfo(image_id='synthetic', path='synthetic.png', width=400, height=300))
                with patch('pcbparse.association.ocr.infer_semantic_fields', wraps=infer_semantic_fields) as classifier:
                    associate_ocr(ir, {'association': {
                        'pin_body_name_context_enabled': True,
                        'pin_body_component_roi_context_enabled': enabled,
                    }})
                self.assertEqual(classifier.call_args.kwargs.get('pin_body_component_roi_context_enabled', False), enabled)


if __name__=='__main__':unittest.main()
