from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from pcbparse.perception.recognizer_backend import PaddleRecognizerBackend
from pcbparse.pipeline.budget import BudgetExceeded


class PaddleBackendTests(unittest.TestCase):
    def backend(self, predict, **kwargs):
        model = Mock(predict=predict)
        with patch.dict(sys.modules, {'paddleocr': SimpleNamespace(TextRecognition=lambda **_: model)}):
            backend = PaddleRecognizerBackend(**kwargs)
        return backend

    def test_batch_cache_dedup_preserves_input_order_and_spelling(self):
        calls = []
        def predict(*, input, batch_size):
            calls.append(len(input))
            return [{'rec_text': '10k' if c.flat[0] == 1 else '10K', 'rec_score': 0.9} for c in input]
        backend = self.backend(predict, batch_size=2)
        a, b = np.ones((4, 8, 3), dtype=np.uint8), np.zeros((4, 8, 3), dtype=np.uint8)
        self.assertEqual([('10k', .9), ('10K', .9), ('10k', .9), (None, 0.)],
                         backend.recognize_crops([a, b, a.copy(), None]))
        self.assertEqual([2], calls)
        self.assertEqual(1, backend.last_report['duplicate_inputs'])
        self.assertEqual([('10K', .9), ('10k', .9)], backend.recognize_crops([b, a]))
        self.assertEqual([2], calls)
        self.assertEqual(2, backend.last_report['cache_hits'])

    def test_cardinality_error_is_traceable_and_not_cached(self):
        model = Mock(side_effect=[[], [{'rec_text': 'R1', 'rec_score': 0.8}]])
        backend = self.backend(model)
        crop = np.ones((3, 8), dtype=np.uint8)
        self.assertEqual([(None, 0.)], backend.recognize_crops([crop]))
        self.assertEqual('ValueError', backend.last_report['errors'][0]['type'])
        self.assertEqual([('R1', .8)], backend.recognize_crops([crop]))
        self.assertEqual(2, model.call_count)

    def test_batch_failure_does_not_shift_later_outputs(self):
        predict = Mock(side_effect=[RuntimeError('fixture'), [{'rec_text': 'U2', 'rec_score': .7}]])
        backend = self.backend(predict, batch_size=1)
        self.assertEqual([(None, 0.), ('U2', .7)], backend.recognize_crops([
            np.zeros((3, 8), dtype=np.uint8), np.ones((3, 8), dtype=np.uint8)]))
        self.assertEqual([0], backend.last_report['errors'][0]['indices'])

    def test_budget_exhaustion_propagates_without_model_call(self):
        predict = Mock()
        backend = self.backend(predict)
        def exhausted():
            raise BudgetExceeded('fixture deadline')
        with self.assertRaises(BudgetExceeded):
            backend.recognize_crops([np.zeros((3, 8), dtype=np.uint8)], budget_check=exhausted)
        predict.assert_not_called()

    def test_shape_is_part_of_pixel_cache_identity(self):
        predict = Mock(return_value=[{'rec_text': '1', 'rec_score': .8}, {'rec_text': '2', 'rec_score': .8}])
        backend = self.backend(predict)
        self.assertEqual([('1', .8), ('2', .8)], backend.recognize_crops([
            np.zeros((3, 8), dtype=np.uint8), np.zeros((4, 6), dtype=np.uint8)]))
        self.assertEqual(2, len(predict.call_args.kwargs['input']))

    def test_grayscale_pixels_are_expanded_for_paddle_and_reuse_rgb_cache(self):
        predict = Mock(return_value=[{'rec_text': 'U3', 'rec_score': .8}])
        backend = self.backend(predict)
        gray = np.arange(24, dtype=np.uint8).reshape(3, 8)
        self.assertEqual([('U3', .8)], backend.recognize_crops([gray]))
        sent = predict.call_args.kwargs['input'][0]
        self.assertEqual((3, 8, 3), sent.shape)
        for channel in range(3):
            np.testing.assert_array_equal(gray, sent[:, :, channel])
        self.assertEqual([('U3', .8)], backend.recognize_crops([sent]))
        self.assertEqual(1, predict.call_count)


if __name__ == '__main__':
    unittest.main()
