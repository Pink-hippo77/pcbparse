from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pcbparse.perception.pin_package_adapter import (
    PinPackageConfig, PinPackageError, _origins, fuse_points, verify_descriptor,
)
from pcbparse.perception.pin_package_provider import load_pin_package_experiment


class PinPackageAdapterTests(unittest.TestCase):
    def test_conservative_fusion_preserves_primary_and_suppresses_nearby_supplement(self):
        primary = np.zeros((32, 64), dtype=np.uint8)
        secondary = np.zeros_like(primary)
        primary[2:4, 2:4] = 100
        secondary[0:8, 0:8] = 200
        secondary[16:24, 40:48] = 200
        secondary[16:18, 10:12] = 250  # too small for a multitask supplement
        points = fuse_points(primary, secondary, PinPackageConfig())
        self.assertEqual([p["source"] for p in points], ["primary", "multitask_supplement"])
        self.assertEqual([(p["x"], p["y"]) for p in points], [(2.5, 2.5), (43.5, 19.5)])

    def test_primary_uses_diagonal_connectivity_and_at_least_two_pixels(self):
        primary = np.zeros((8, 8), dtype=np.uint8)
        primary[0, 0] = primary[1, 1] = primary[7, 7] = 100
        points = fuse_points(primary, np.zeros_like(primary), PinPackageConfig())
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["area"], 2)

    def test_windows_cover_tail_and_small_images_without_resizing(self):
        self.assertEqual(_origins(33, 512, 384), [0])
        self.assertEqual(_origins(1024, 512, 384), [0, 384, 512])
        self.assertEqual(_origins(1280, 512, 384), [0, 384, 768])

    def test_invalid_policy_and_map_contracts_fail_closed(self):
        for kwargs in ({"stride": 513}, {"patch_size": 17}, {"branch_merge_radius": float("nan")},
                       {"primary_threshold": True}, {"multitask_min_area": 1}):
            with self.assertRaises(ValueError):
                PinPackageConfig(**kwargs)
        with self.assertRaises(PinPackageError):
            fuse_points(np.zeros((2, 2)), np.zeros((2, 2)), PinPackageConfig())

    def test_weight_tampering_and_path_escape_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "primary.pt").write_bytes(b"original")
            digest = hashlib.sha256(b"original").hexdigest()
            descriptor = {"schema_version": 1, "task": "pin", "status": "candidate",
                          "inference_scope": "full_image", "training_lineage_verified": False,
                          "weights": {name: {"path": "primary.pt", "sha256": digest}
                                      for name in ("primary", "multitask")}, "policy": {}}
            path = root / "pin.json"
            def write():
                path.write_text(json.dumps(descriptor), encoding="utf-8")
                return hashlib.sha256(path.read_bytes()).hexdigest()
            expected = write()
            verify_descriptor(path, expected)
            (root / "primary.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(PinPackageError, "asset SHA256"):
                verify_descriptor(path, expected)
            descriptor["weights"]["primary"]["path"] = "../escape.pt"
            with self.assertRaisesRegex(PinPackageError, "inside its bundle"):
                verify_descriptor(path, write())

    def test_experiment_cannot_run_under_production_or_degraded_profile(self):
        for run in ({"profile": "baseline"}, {"profile": "pin_package_experiment", "submission_mode": "degraded"}):
            with self.assertRaisesRegex(ValueError, "profile and strict"):
                load_pin_package_experiment("unused.lock", {"run": run})


if __name__ == "__main__":
    unittest.main()
