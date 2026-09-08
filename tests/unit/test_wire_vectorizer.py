from __future__ import annotations

import unittest

import numpy as np

from pcbparse.perception.wire_vectorizer import (
    SkeletonPolyline,
    WireVectorizerConfig,
    deduplicate_polylines,
    thin_binary_mask,
    vectorize_wire_mask,
)


def _mask(height: int = 15, width: int = 15) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)


class WireVectorizerTests(unittest.TestCase):
    def test_empty_mask(self) -> None:
        result = vectorize_wire_mask(_mask())

        self.assertEqual((), result.polylines)
        self.assertEqual((), result.junctions)
        self.assertEqual(0, result.foreground_pixels)
        self.assertEqual(0, result.skeleton_pixels)

    def test_horizontal_and_vertical_lines_map_to_global_coordinates(self) -> None:
        horizontal = _mask()
        horizontal[7, 2:13] = 1
        vertical = _mask()
        vertical[2:13, 7] = 1

        horizontal_result = vectorize_wire_mask(horizontal, origin=(100.0, 200.0), pixel_size=(2.0, 3.0))
        vertical_result = vectorize_wire_mask(vertical)

        self.assertEqual(((104.0, 221.0), (124.0, 221.0)), horizontal_result.polylines[0].points)
        self.assertEqual(((7.0, 2.0), (7.0, 12.0)), vertical_result.polylines[0].points)
        self.assertEqual(["endpoint", "endpoint"], [item.kind for item in horizontal_result.junctions])
        self.assertEqual(2, len(vertical_result.junctions))

    def test_diagonal_line_is_not_lost(self) -> None:
        diagonal = _mask()
        np.fill_diagonal(diagonal[2:13, 2:13], 1)

        result = vectorize_wire_mask(diagonal)

        self.assertEqual(1, len(result.polylines))
        self.assertEqual(((2.0, 2.0), (12.0, 12.0)), result.polylines[0].points)
        self.assertAlmostEqual(10.0 * 2.0**0.5, result.polylines[0].length)

    def test_right_angle_bend_survives_rdp(self) -> None:
        bend = _mask()
        bend[3, 3:11] = 1
        bend[3:12, 10] = 1

        result = vectorize_wire_mask(bend)

        self.assertEqual(1, len(result.polylines))
        self.assertEqual(((3.0, 3.0), (10.0, 3.0), (10.0, 11.0)), result.polylines[0].points)

    def test_t_junction_is_force_split_into_three_paths(self) -> None:
        tee = _mask()
        tee[7, 2:13] = 1
        tee[2:8, 7] = 1

        result = vectorize_wire_mask(tee)

        self.assertEqual(3, len(result.polylines))
        candidates = [item for item in result.junctions if item.kind == "t_junction"]
        self.assertEqual(1, len(candidates))
        self.assertEqual((7.0, 7.0), candidates[0].point)
        self.assertEqual(3, candidates[0].degree)
        self.assertEqual(3, len(candidates[0].polyline_ids))
        self.assertTrue(all((7.0, 7.0) in item.points for item in result.polylines))

    def test_cross_junction_is_force_split_into_four_paths(self) -> None:
        cross = _mask()
        cross[7, 2:13] = 1
        cross[2:13, 7] = 1

        result = vectorize_wire_mask(cross)

        self.assertEqual(4, len(result.polylines))
        candidates = [item for item in result.junctions if item.kind == "connected_cross"]
        self.assertEqual(1, len(candidates))
        self.assertEqual((7.0, 7.0), candidates[0].point)
        self.assertEqual(4, len(candidates[0].polyline_ids))

    def test_thick_stroke_thins_to_one_path_instead_of_parallel_runs(self) -> None:
        thick = _mask(21, 31)
        thick[8:13, 3:28] = 1

        skeleton = thin_binary_mask(thick)
        result = vectorize_wire_mask(thick)

        self.assertLessEqual(int(skeleton.sum()), 25)
        self.assertEqual(1, len(result.polylines))
        self.assertEqual(2, len(result.junctions))
        self.assertEqual(2, len(result.polylines[0].points))

    def test_configurable_closing_repairs_a_one_pixel_gap(self) -> None:
        broken = _mask()
        broken[7, 2:7] = 1
        broken[7, 8:13] = 1

        separate = vectorize_wire_mask(broken)
        repaired = vectorize_wire_mask(broken, config=WireVectorizerConfig(gap_closing_radius=1))

        self.assertEqual(2, len(separate.polylines))
        self.assertEqual(1, len(repaired.polylines))

    def test_configurable_small_hole_filling_is_bounded_by_area(self) -> None:
        stroke = _mask()
        stroke[6:9, 3:12] = 1
        stroke[7, 7] = 0

        unfilled = vectorize_wire_mask(stroke)
        filled = vectorize_wire_mask(stroke, config=WireVectorizerConfig(maximum_hole_area=1))

        self.assertEqual(unfilled.foreground_pixels + 1, filled.foreground_pixels)
        self.assertEqual(1, len(filled.polylines))

    def test_tile_duplicate_removal_uses_global_geometry_and_is_order_independent(self) -> None:
        first_mask = _mask(9, 12)
        first_mask[4, 1:11] = 1
        second_mask = _mask(9, 12)
        second_mask[4, 1:11] = 1
        first = vectorize_wire_mask(first_mask, origin=(10.0, 20.0), id_namespace="tile-a").polylines[0]
        second = vectorize_wire_mask(second_mask, origin=(10.2, 20.1), id_namespace="tile-b").polylines[0]

        forward = deduplicate_polylines((first, second), tolerance=0.25)
        reverse = deduplicate_polylines((second, first), tolerance=0.25)

        self.assertEqual(1, len(forward))
        self.assertEqual(forward, reverse)
        partial = SkeletonPolyline("partial", ((11.0, 24.0), (15.0, 24.0)), 5, 4.0)
        self.assertEqual(2, len(deduplicate_polylines((first, partial), tolerance=0.25)))

    def test_result_and_ids_are_deterministic(self) -> None:
        shape = _mask()
        shape[7, 2:13] = 1
        shape[2:13, 7] = 1

        first = vectorize_wire_mask(shape, id_namespace="det")
        second = vectorize_wire_mask(shape.copy(), id_namespace="det")

        self.assertEqual(first, second)
        self.assertEqual(
            [item.internal_id for item in first.polylines],
            [item.internal_id for item in second.polylines],
        )

    def test_tile_size_safety_limit_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "split it into tiles"):
            vectorize_wire_mask(_mask(11, 11), config=WireVectorizerConfig(maximum_mask_pixels=100))


if __name__ == "__main__":
    unittest.main()
