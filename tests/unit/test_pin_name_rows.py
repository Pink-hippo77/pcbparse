from __future__ import annotations

import unittest

from pcbparse.association.ocr_common import OCRAssociationConfig
from pcbparse.association.text_pin import _score_candidate, associate_pin_texts
from pcbparse.ir.models import BBox, Pin, Point, TextToken


class PinNameRowTests(unittest.TestCase):
    def score(self, token, pin, *, field='name', penalty=0.22, explicit=False):
        return _score_candidate(token, pin, field=field, value=token.raw_text,
                                bound_values=[token.raw_text], explicit=explicit,
                                config=OCRAssociationConfig(pin_name_row_penalty=penalty))

    def token(self, bbox, value='PA15'):
        return TextToken(internal_id='text:name', raw_text=value, details='pinname',
                         score=.99, bbox=bbox)

    def test_name_row_can_resolve_the_closer_wrong_row_without_inventing_number(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                same = Pin(internal_id='pin:same', component_id='body', orientation='L',
                           point=Point(x=100, y=110), base=Point(x=108, y=110))
                wrong = Pin(internal_id='pin:wrong', component_id='body', orientation='L',
                            point=Point(x=135, y=120), base=Point(x=143, y=120))
                token = self.token(BBox(x1=130, y1=105, x2=170, y2=115))
                self.assertGreater(self.score(token, wrong, penalty=0).score,
                                   self.score(token, same, penalty=0).score)
                pins = [same, wrong] if not reverse else [wrong, same]
                result = associate_pin_texts(pins, [token], [], OCRAssociationConfig(
                    pin_min_score=.3, pin_unique_margin=0., pin_name_row_penalty=.22))
                self.assertEqual(result['assignments'], {'pin:same': {'name': 'PA15'}})
                self.assertEqual(token.pin_id, same.internal_id)
                self.assertEqual(token.raw_text, 'PA15')
                self.assertIsNone(same.number)
                self.assertIsNone(wrong.number)
                self.assertEqual(wrong.name, '')
                self.assertEqual(same.point, Point(x=100, y=110))

    def test_number_candidates_are_identical_even_when_penalty_is_enabled(self):
        token = self.token(BBox(x1=130, y1=105, x2=140, y2=115), value='23')
        token.details = 'pinnumber'
        pin = Pin(internal_id='pin', orientation='L', point=Point(x=130, y=120))
        self.assertEqual(self.score(token, pin, field='number', penalty=0),
                         self.score(token, pin, field='number', penalty=.22))

    def test_explicit_target_and_unknown_orientation_do_not_get_row_penalties(self):
        token = self.token(BBox(x1=130, y1=105, x2=170, y2=115))
        pin = Pin(internal_id='pin', orientation='L', point=Point(x=135, y=120))
        self.assertEqual(self.score(token, pin, explicit=True, penalty=0),
                         self.score(token, pin, explicit=True, penalty=.22))
        pin.orientation = 'UNKNOWN'
        self.assertEqual(self.score(token, pin, penalty=0), self.score(token, pin, penalty=.22))

    def test_each_side_uses_the_correct_axis_and_keeps_points(self):
        cases = (('L', BBox(x1=110,y1=105,x2=130,y2=115)),
                 ('R', BBox(x1=70,y1=105,x2=90,y2=115)),
                 ('T', BBox(x1=105,y1=110,x2=115,y2=130)),
                 ('B', BBox(x1=105,y1=70,x2=115,y2=90)))
        for side, box in cases:
            with self.subTest(side=side):
                pin = Pin(internal_id='pin', orientation=side, point=Point(x=100,y=100))
                before = pin.model_dump()
                token = self.token(box)
                control, candidate = self.score(token,pin,penalty=0), self.score(token,pin,penalty=.22)
                self.assertAlmostEqual(control.score - candidate.score, .22)
                self.assertEqual(candidate.score_components['row_alignment'], 0.)
                self.assertEqual(before, pin.model_dump())

    def test_existing_fields_are_preserved_during_matching(self):
        pin = Pin(internal_id='pin',component_id='body',orientation='L',point=Point(x=100,y=110),
                  name='existing', number='7')
        token = self.token(BBox(x1=104,y1=105,x2=140,y2=115))
        associate_pin_texts([pin],[token],[],OCRAssociationConfig(pin_name_row_penalty=.22))
        self.assertEqual((pin.name,pin.number),('existing','7'))

    def test_same_row_has_no_penalty_and_default_is_disabled(self):
        pin = Pin(internal_id='pin',orientation='L',point=Point(x=100,y=110))
        token = self.token(BBox(x1=105,y1=105,x2=140,y2=115))
        self.assertEqual(OCRAssociationConfig().pin_name_row_penalty,0.)
        self.assertEqual(self.score(token,pin,penalty=0).score,self.score(token,pin,penalty=.22).score)
        self.assertEqual(self.score(token,pin,penalty=.22).score_components['row_alignment'],1.)

    def test_invalid_or_nonfinite_penalties_are_rejected(self):
        for value in (-.01,1.01,float('nan'),float('inf'),True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    OCRAssociationConfig(pin_name_row_penalty=value)


if __name__ == '__main__':
    unittest.main()
