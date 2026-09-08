from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from pcbparse.perception.production_provider import (
    DETERMINISTIC_JUNCTION_MODEL_TASKS,
    ProductionPerceptionProvider,
    ProductionProviderError,
    composite_provider_config_from_pipeline,
    load_production_provider,
    required_model_tasks_for_config,
)
from pcbparse.service.readiness import ReadinessGateError
from tests.model_lock_fixtures import valid_inference_contract


class _Adapter:
    version = "fixture-v1"

    threshold = 0.5
    detector_threshold = 0.5

    def predict(self, *args, **kwargs):
        return []

    def recognize(self, *args, **kwargs):
        return []

    def segment(self, *args, **kwargs):
        return []

    def classify(self, *args, **kwargs):
        return []


class ProductionProviderFactoryTests(unittest.TestCase):
    def test_wire_threshold_override_is_validated_and_retained(self) -> None:
        config = composite_provider_config_from_pipeline({'wire': {'threshold': 0.6}})
        self.assertEqual({'threshold': 0.6}, config.wire_inference_overrides)
        with self.assertRaisesRegex(ValueError, 'pipeline wire contains unsupported'):
            composite_provider_config_from_pipeline({'wire': {'unknown': 0.6}})
        with self.assertRaises(ValueError):
            composite_provider_config_from_pipeline({'wire': {'threshold': 1.5}})

    def test_component_nms_iou_is_loaded_from_pipeline_image(self) -> None:
        config = composite_provider_config_from_pipeline(
            {"image": {"component_nms_iou": 0.45}}
        )
        self.assertEqual(0.45, config.component_nms_iou)
        with self.assertRaises(ValueError):
            composite_provider_config_from_pipeline(
                {"image": {"component_nms_iou": 1.01}}
            )

    def test_adaptive_component_fallback_config_is_strict_and_serialized(self) -> None:
        config = composite_provider_config_from_pipeline(
            {
                "image": {
                    "component_adaptive_fallback_enabled": True,
                    "component_fallback_tile_size": 256,
                    "component_fallback_tile_overlap": 32,
                    "component_fallback_trigger_max_count": 24,
                    "component_fallback_trigger_max_box_fraction": 0.2,
                    "component_fallback_trigger_max_bbox_area_fraction": 0.02,
                }
            }
        )
        self.assertTrue(config.component_adaptive_fallback_enabled)
        self.assertEqual(256, config.component_fallback_tile_size)
        self.assertEqual(32, config.component_fallback_tile_overlap)
        self.assertEqual(24, config.component_fallback_trigger_max_count)
        self.assertEqual(0.2, config.component_fallback_trigger_max_box_fraction)
        self.assertEqual(0.02, config.component_fallback_trigger_max_bbox_area_fraction)
        resolved = config.to_dict()
        self.assertTrue(resolved["component_adaptive_fallback_enabled"])
        with self.assertRaisesRegex(ValueError, "component_fallback_tile_overlap"):
            composite_provider_config_from_pipeline(
                {"image": {"component_fallback_tile_size": 32, "component_fallback_tile_overlap": 32}}
            )
        with self.assertRaisesRegex(ValueError, "component_fallback_trigger_max_box_fraction"):
            composite_provider_config_from_pipeline(
                {"image": {"component_fallback_trigger_max_box_fraction": 1.1}}
            )
        with self.assertRaisesRegex(ValueError, "component_fallback_trigger_max_bbox_area_fraction"):
            composite_provider_config_from_pipeline(
                {"image": {"component_fallback_trigger_max_bbox_area_fraction": 1.1}}
            )

    def test_ocr_detector_threshold_override_is_validated_and_retained(self) -> None:
        config = composite_provider_config_from_pipeline(
            {"ocr": {"detector_threshold": 0.2}}
        )
        self.assertEqual(
            {"detector_threshold": 0.2},
            config.ocr_inference_overrides,
        )
        with self.assertRaisesRegex(ValueError, "pipeline ocr contains unsupported"):
            composite_provider_config_from_pipeline({"ocr": {"unknown": 0.2}})
        with self.assertRaises(ValueError):
            composite_provider_config_from_pipeline(
                {"ocr": {"detector_threshold": 1.5}}
            )

    def test_verified_adapter_lock_constructs_composite_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module = types.ModuleType("fixture_model_adapters")
            module.load = lambda path, entry: _Adapter()
            sys.modules[module.__name__] = module
            self.addCleanup(sys.modules.pop, module.__name__, None)
            entries = []
            for task in ("component", "junction", "ocr", "pin", "wire"):
                payload = f"{task}-weights".encode()
                path = root / f"{task}.bin"
                path.write_bytes(payload)
                entries.append(
                    {
                        "task": task,
                        "path": path.name,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "status": "blessed",
                        "blessed": True,
                        "version": "1.0",
                        "license": "test-only",
                        "precision": "fp32",
                        "adapter": "fixture_model_adapters:load",
                        "inference": valid_inference_contract(task),
                    }
                )
            lock = root / "models.lock.json"
            lock.write_text(
                json.dumps({"schema_version": "1.0", "models": entries}),
                encoding="utf-8",
            )

            perception_config = composite_provider_config_from_pipeline(
                {
                    "image": {
                        "tile_size": 320,
                        "tile_overlap": 48,
                        "component_tile_size": 256,
                        "component_tile_overlap": 32,
                        "wire_tile_size": 640,
                        "wire_tile_overlap": 96,
                        "component_roi_padding_px": 8,
                        "ocr_component_roi_padding_px": 12,
                        "pin_component_roi_padding_px": 64,
                    },
                    "graph": {
                        "rdp_epsilon_px": 1.25,
                        "wire_gap_closing_radius_px": 1,
                        "wire_maximum_hole_area_px": 9,
                        "wire_maximum_component_pixels": 6000,
                        "learned_junctions_enabled": True,
                    },
                }
            )
            perception_config.wire_inference_overrides = {'threshold': 0.6}
            perception_config.ocr_inference_overrides = {'detector_threshold': 0.2}
            provider = load_production_provider(lock, config=perception_config)

            self.assertIsInstance(provider, ProductionPerceptionProvider)
            self.assertEqual("fixture-v1", provider.component_provider.version)
            self.assertEqual("fixture-v1", provider.junction_provider.version)
            self.assertEqual(320, provider.config.tile_size)
            self.assertEqual(48, provider.config.tile_overlap)
            self.assertEqual(256, provider.config.resolved_component_tile_size)
            self.assertEqual(32, provider.config.resolved_component_tile_overlap)
            self.assertEqual(640, provider.config.resolved_wire_tile_size)
            self.assertEqual(96, provider.config.resolved_wire_tile_overlap)
            self.assertEqual(8, provider.config.component_roi_padding_px)
            self.assertEqual(12, provider.config.resolved_ocr_component_roi_padding_px)
            self.assertEqual(64, provider.config.resolved_pin_component_roi_padding_px)
            self.assertEqual(1.25, provider.config.wire_rdp_epsilon_px)
            self.assertEqual(1, provider.config.wire_gap_closing_radius_px)
            self.assertEqual(9, provider.config.wire_maximum_hole_area_px)
            self.assertEqual(6000, provider.config.wire_maximum_component_pixels)
            self.assertTrue(provider.config.learned_junctions_enabled)
            self.assertEqual(512, provider.config.ocr_tile_size)
            self.assertEqual(0.6, provider.wire_provider.threshold)
            self.assertEqual(0.2, provider.ocr_provider.detector_threshold)

    def test_deterministic_junction_mode_loads_exactly_four_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loaded_paths: list[str] = []
            module = types.ModuleType("fixture_four_model_adapters")

            def load(path, entry):
                loaded_paths.append(Path(path).name)
                return _Adapter()

            module.load = load
            sys.modules[module.__name__] = module
            self.addCleanup(sys.modules.pop, module.__name__, None)
            entries = []
            for task in sorted(DETERMINISTIC_JUNCTION_MODEL_TASKS):
                payload = f"{task}-weights".encode()
                path = root / f"{task}.bin"
                path.write_bytes(payload)
                entries.append(
                    {
                        "task": task,
                        "path": path.name,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "status": "blessed",
                        "blessed": True,
                        "version": "1.0",
                        "license": "test-only",
                        "precision": "fp32",
                        "adapter": "fixture_four_model_adapters:load",
                        "inference": valid_inference_contract(task),
                    }
                )
            lock = root / "models.lock.json"
            lock.write_text(
                json.dumps({"schema_version": "1.0", "models": entries}),
                encoding="utf-8",
            )

            config = composite_provider_config_from_pipeline(
                {"graph": {"learned_junctions_enabled": False}}
            )
            provider = load_production_provider(lock, config=config)

            self.assertIsNone(provider.junction_provider)
            self.assertEqual(
                sorted(DETERMINISTIC_JUNCTION_MODEL_TASKS), sorted(required_model_tasks_for_config(config))
            )
            self.assertEqual(
                sorted(f"{task}.bin" for task in DETERMINISTIC_JUNCTION_MODEL_TASKS),
                sorted(loaded_paths),
            )
            self.assertEqual(
                {
                    "contract": "production_model_task_plan_v1",
                    "junction_mode": "deterministic_vectorizer",
                    "required_model_tasks": ["component", "ocr", "pin", "wire"],
                    "loaded_model_tasks": ["component", "ocr", "pin", "wire"],
                    "learned_junction_model_required": False,
                    "learned_junction_model_loaded": False,
                    "learned_junction_model_invoked": False,
                },
                provider.model_task_plan,
            )

    def test_pipeline_image_tiling_is_strict_and_missing_section_is_compatible(self) -> None:
        defaults = composite_provider_config_from_pipeline({})
        self.assertEqual(1024, defaults.tile_size)
        self.assertEqual(160, defaults.tile_overlap)
        self.assertEqual(1024, defaults.resolved_component_tile_size)
        self.assertEqual(160, defaults.resolved_component_tile_overlap)
        self.assertEqual(1024, defaults.resolved_wire_tile_size)
        self.assertEqual(160, defaults.resolved_wire_tile_overlap)
        self.assertEqual(0, defaults.component_roi_padding_px)
        self.assertEqual(0, defaults.resolved_ocr_component_roi_padding_px)
        self.assertEqual(0, defaults.resolved_pin_component_roi_padding_px)
        self.assertFalse(defaults.learned_junctions_enabled)

        pin_candidate = composite_provider_config_from_pipeline({
            "pin": {
                "score_threshold": 0.5,
                "max_detections_per_component": 4,
                "box_prior_mode": "filled_rect_v1",
                "box_margin_px": 4.0,
            }
        })
        self.assertEqual({
            "score_threshold": 0.5,
            "max_detections_per_component": 4,
            "box_prior_mode": "filled_rect_v1",
            "box_margin_px": 4.0,
        }, pin_candidate.pin_inference_overrides)

        for config, message in (
            ({"image": []}, "image section"),
            ({"graph": []}, "graph section"),
            ({"image": {"tile_size": True}}, "tile_size"),
            ({"image": {"tile_size": 0}}, "positive"),
            ({"image": {"tile_size": 64, "tile_overlap": 64}}, "tile_overlap"),
            (
                {"image": {"component_tile_size": 64, "component_tile_overlap": 64}},
                "component_tile_overlap",
            ),
            ({"image": {"wire_tile_size": True}}, "wire_tile_size"),
            ({"image": {"component_roi_padding_px": -1}}, "non-negative"),
            ({"image": {"ocr_component_roi_padding_px": True}}, "finite number"),
            ({"image": {"pin_component_roi_padding_px": -1}}, "non-negative"),
            ({"graph": {"rdp_epsilon_px": -1}}, "non-negative"),
            ({"graph": {"wire_gap_closing_radius_px": -1}}, "non-negative integer"),
            ({"graph": {"wire_minimum_run_px": -1}}, "must be positive"),
            ({"graph": {"wire_maximum_hole_area_px": 1.5}}, "non-negative integer"),
            ({"graph": {"wire_maximum_component_pixels": -1}}, "non-negative integer"),
            (
                {"graph": {"learned_junctions_enabled": 1}},
                "must be a boolean",
            ),
        ):
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, message):
                    composite_provider_config_from_pipeline(config)

    def test_bad_adapter_import_fails_after_lock_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for task in ("component", "junction", "ocr", "pin", "wire"):
                payload = f"{task}-weights".encode()
                path = root / f"{task}.bin"
                path.write_bytes(payload)
                entries.append(
                    {
                        "task": task,
                        "path": path.name,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "status": "blessed",
                        "blessed": True,
                        "version": "1.0",
                        "license": "test-only",
                        "precision": "fp32",
                        "adapter": "does_not_exist.adapters:load",
                        "inference": valid_inference_contract(task),
                    }
                )
            lock = root / "models.lock.json"
            lock.write_text(
                json.dumps({"schema_version": "1.0", "models": entries}),
                encoding="utf-8",
            )

            with self.assertRaises(ProductionProviderError):
                load_production_provider(
                    lock,
                    config=composite_provider_config_from_pipeline(
                        {"graph": {"learned_junctions_enabled": True}}
                    ),
                )

    def test_deterministic_mode_rejects_an_extra_junction_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for task in ("component", "junction", "ocr", "pin", "wire"):
                payload = f"{task}-weights".encode()
                path = root / f"{task}.bin"
                path.write_bytes(payload)
                entries.append(
                    {
                        "task": task,
                        "path": path.name,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "status": "blessed",
                        "blessed": True,
                        "version": "1.0",
                        "license": "test-only",
                        "precision": "fp32",
                        "adapter": "fixture.adapters:load",
                        "inference": valid_inference_contract(task),
                    }
                )
            lock = root / "models.lock.json"
            lock.write_text(
                json.dumps({"schema_version": "1.0", "models": entries}),
                encoding="utf-8",
            )

            with self.assertRaises(ReadinessGateError) as raised:
                load_production_provider(lock)
            self.assertIn(
                "unexpected_task",
                {item["code"] for item in raised.exception.report["issues"]},
            )


if __name__ == "__main__":
    unittest.main()