from __future__ import annotations

from math import hypot
import unittest
from unittest.mock import patch

import numpy as np

from pcbparse.perception import wire_vectorizer as vectorizer


def full_scan(labels, left_id, right_id, radius):
    """Independent specification: closest qualifying pair in row-major order."""
    left = [tuple(map(int, point)) for point in np.argwhere(labels == left_id)]
    right = [tuple(map(int, point)) for point in np.argwhere(labels == right_id)]
    choices = [(hypot(a[0] - b[0], a[1] - b[1]), i, j, a, b)
               for i, a in enumerate(left) for j, b in enumerate(right)
               if hypot(a[0] - b[0], a[1] - b[1]) <= 2 * radius + 1]
    if not choices:
        return float(2 * radius), None, None
    distance, _, _, a, b = min(choices)
    return max(0., distance - 1.), a, b


class WireGapPixelPairTests(unittest.TestCase):
    def test_large_component_ties_keep_first_pixels(self):
        labels = np.zeros((30, 40), dtype=np.int32)
        labels[3:27, 2:16] = 1
        labels[3:27, 17:35] = 2
        self.assertEqual(vectorizer._closest_component_pair(labels, 1, 2, 1),
                         (1., (3, 15), (3, 17)))

    def test_missing_and_far_components_preserve_the_no_bridge_sentinel(self):
        labels = np.zeros((80, 90), dtype=np.int32)
        labels[:10, :10] = 1
        labels[70:, 80:] = 2
        for pair in ((1, 2), (1, 3), (3, 2)):
            with self.subTest(pair=pair):
                self.assertEqual(vectorizer._closest_component_pair(labels, *pair, 1), (2., None, None))

    def test_random_masks_and_different_search_radii_match_pixel_order_specification(self):
        rng = np.random.default_rng(20261002)
        for case in range(12):
            labels = rng.choice([0, 1, 2], (18, 21), p=[.5, .25, .25]).astype(np.int32)
            for radius in (0, 1, 2, 16):
                with self.subTest(case=case, radius=radius):
                    self.assertEqual(vectorizer._closest_component_pair(labels, 1, 2, radius),
                                     full_scan(labels, 1, 2, radius))

    def test_component_id_order_and_small_component_fallback_keep_exact_geometry(self):
        labels = np.zeros((8, 11), dtype=np.int32)
        labels[1:7, 2] = 1
        labels[2, 4] = labels[4, 4] = 2
        for left, right in ((1, 2), (2, 1), (1, 1)):
            with self.subTest(left=left, right=right):
                self.assertEqual(vectorizer._closest_component_pair(labels, left, right, 1),
                                 full_scan(labels, left, right, 1))

    def test_virtual_closing_repair_and_complete_vectorization_match_full_scan(self):
        masks = []
        straight = np.zeros((21, 80), dtype=np.uint8)
        straight[10, 3:38] = straight[10, 40:77] = 1
        masks.append(straight)
        branched = np.zeros((45, 80), dtype=np.uint8)
        branched[20, 3:38] = branched[20, 40:77] = 1
        branched[3:20, 38] = 1
        masks.append(branched)
        for index, mask in enumerate(masks):
            with self.subTest(mask=index):
                config = vectorizer.WireVectorizerConfig(gap_closing_radius=1)
                optimized = vectorizer.vectorize_wire_mask(mask, config=config, id_namespace='test')
                with patch.object(vectorizer, '_closest_component_pair', full_scan):
                    reference = vectorizer.vectorize_wire_mask(mask, config=config, id_namespace='test')
                self.assertEqual(optimized, reference)
                self.assertTrue(optimized.virtual_repairs)


if __name__ == '__main__':
    unittest.main()
