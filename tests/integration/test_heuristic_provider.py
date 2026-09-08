from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from pcbparse.perception.heuristic_provider import HeuristicVisionProvider


class HeuristicProviderTests(unittest.TestCase):
    def test_extracts_axis_aligned_wire_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wire.png"
            image = Image.new("RGB", (100, 80), "white")
            draw = ImageDraw.Draw(image)
            draw.line((10, 20, 80, 20), fill=(0, 160, 0), width=2)
            draw.line((60, 10, 60, 70), fill=(0, 160, 0), width=2)
            image.save(path)
            ir = HeuristicVisionProvider(minimum_run_length_px=10).build_ir(path)
            self.assertTrue(any(segment.channel == "horizontal" for segment in ir.segments))
            self.assertTrue(any(segment.channel == "vertical" for segment in ir.segments))
            self.assertTrue(all(segment.score < 0.5 for segment in ir.segments))


if __name__ == "__main__":
    unittest.main()

