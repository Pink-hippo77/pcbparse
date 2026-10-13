from __future__ import annotations

import unittest

from data_tools.pin_package_infer import _reject_sidecar_reads
from pcbparse.cli import _provider


class PinPackageInferenceBoundaryTests(unittest.TestCase):
    def test_production_entry_rejects_experimental_config_instead_of_loading_old_pin(self):
        with self.assertRaisesRegex(ValueError, "config is experimental"):
            _provider("production", pipeline_config={"pin_package": {"experimental": True}})

    def test_targets_and_sidecars_are_rejected(self):
        for path in ("/dataset/0004_CVJsonStd.json", "E:\\images\\0004_CVJsonStd.json",
                     "/run/oracle_diagnostic/geometry.json", "/run/offline_evaluation/targets.json",
                     "/run/0004_geometry.json", "/run/0004.eds", "/run/0004_spice.net"):
            with self.assertRaises(RuntimeError):
                _reject_sidecar_reads("open", (path,))

    def test_image_model_config_and_ir_files_remain_readable(self):
        for path in ("/images/0004.png", "/models/model.pt", "/models/pin.descriptor.json",
                     "/configs/pipeline.json", "/output/perception.circuit-ir.json"):
            _reject_sidecar_reads("open", (path,))


if __name__ == "__main__":
    unittest.main()
