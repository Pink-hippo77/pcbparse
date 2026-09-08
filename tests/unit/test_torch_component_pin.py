from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pcbparse.training.torch_component_pin import (
    ComponentDetectionDataset,
    PinKeypointDataset,
    TorchDependencyError,
    component_detection_collate,
    pin_keypoint_collate,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
BINDING = "b" * 64


class TorchOptionalImportTests(unittest.TestCase):
    def test_core_module_imports_and_fails_closed_without_torch(self) -> None:
        if TORCH_AVAILABLE:
            self.skipTest("this interpreter has torch; no-torch path runs under bundled Python")
        with self.assertRaisesRegex(TorchDependencyError, r"\.venv-train"):
            ComponentDetectionDataset("missing.json", "missing-source")
        with self.assertRaisesRegex(TorchDependencyError, "torch-optional"):
            PinKeypointDataset("missing.jsonl", "missing-source")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch CPU-path tests require .venv-train")
class TorchComponentPinDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "source"
        image_path = self.source / "0001/fixture.png"
        image_path.parent.mkdir(parents=True)
        image = Image.new("RGB", (64, 64))
        for y in range(64):
            for x in range(64):
                image.putpixel((x, y), (x * 4, y * 4, 128))
        image.save(image_path)
        self._write_indexes()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _view(view_id: str, view_kind: str, side: str | None = None) -> dict[str, object]:
        return {
            "view_id": view_id,
            "view_kind": view_kind,
            "side": side,
            "sample_id": "0001",
            "split": "train",
            "source_image": "0001/fixture.png",
            "source_image_sha256": "c" * 64,
            "crop_bbox": {"x1": 0, "y1": 0, "x2": 64, "y2": 64},
            "source_size": [64, 64],
            "resized_size": [64, 64],
            "output_size": [64, 64],
            "padding": [0, 0, 0, 0],
            "transform": {
                "convention": "original_xy_to_view_xy_homogeneous",
                "forward": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "inverse": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "roundtrip_tolerance_pixels": 1e-8,
            },
            "decode": {
                "mode": "lazy",
                "color_mode": "RGB",
                "resample": "bilinear",
                "padding_rgb": [255, 255, 255],
            },
        }

    @classmethod
    def _targets(cls) -> list[dict[str, object]]:
        descriptor = {
            "rasterized": False,
            "target_type": "gaussian_2d_peak_1",
            "output_stride": 4,
            "output_size": [16, 16],
            "point": [5.0, 6.0],
            "sigma_heatmap_pixels": 1.5,
            "truncate_sigma": 3.0,
            "ignore_when_visibility_zero": True,
        }
        return [
            {
                "pin_id": "PIN_VISIBLE",
                "pin_key": "P_1",
                "source_record_id": "0001:pin:P_1",
                "tip_original": [20.0, 24.0],
                "base_original": [12.0, 24.0],
                "tip_view": [20.0, 24.0],
                "base_view": [12.0, 24.0],
                "visibility": 2,
                "visibility_convention": "coco_keypoint_0_absent_2_visible",
                "orientation_original": {"unit_vector": [1.0, 0.0]},
                "orientation_view": {
                    "vector": [8.0, 0.0],
                    "unit_vector": [1.0, 0.0],
                    "angle_degrees_image_y_down": 0.0,
                },
                "heatmap": descriptor,
            },
            {
                "pin_id": "PIN_HIDDEN",
                "pin_key": "P_2",
                "source_record_id": "0001:pin:P_2",
                "tip_original": [80.0, 20.0],
                "base_original": [70.0, 20.0],
                "tip_view": None,
                "base_view": [70.0, 20.0],
                "visibility": 0,
                "visibility_convention": "coco_keypoint_0_absent_2_visible",
                "orientation_original": {"unit_vector": [1.0, 0.0]},
                "orientation_view": {
                    "vector": [10.0, 0.0],
                    "unit_vector": [1.0, 0.0],
                    "angle_degrees_image_y_down": 0.0,
                },
                "heatmap": {**descriptor, "point": None},
            },
        ]

    def _write_indexes(self) -> None:
        component_view = self._view("component:0001:tile:0000:0000", "tile")
        component_view.pop("side")
        component_view.update({"id": 1, "binding_sha256": BINDING})
        component = {
            "info": {
                "description": "fixture",
                "version": "component-pin-index-v1",
                "schema_version": 1,
                "binding_sha256": BINDING,
            },
            "licenses": [],
            "categories": [
                {
                    "id": 1,
                    "name": "V",
                    "supercategory": "pcb_component",
                    "source_category_id": "0",
                },
                {
                    "id": 2,
                    "name": "Motor",
                    "supercategory": "pcb_component",
                    "source_category_id": "35",
                },
            ],
            "images": [component_view],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10.0, 10.0, 20.0, 20.0],
                    "area": 400.0,
                    "iscrowd": 0,
                    "ignore": 0,
                    "segmentation": [],
                    "partial_box": False,
                    "partial_box_policy": "complete",
                    "component_id": "RAW_V1",
                    "component_key": "C_1",
                    "source_record_id": "0001:component:C_1",
                    "source_bbox": [10.0, 10.0, 30.0, 30.0],
                    "tile_intersection_bbox": [10.0, 10.0, 30.0, 30.0],
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 2,
                    "bbox": [32.0, 10.0, 20.0, 20.0],
                    "area": 400.0,
                    "iscrowd": 0,
                    "ignore": 0,
                    "segmentation": [],
                    "partial_box": False,
                    "partial_box_policy": "complete",
                    "component_id": "RAW_M1",
                    "component_key": "C_M1",
                    "source_record_id": "0001:component:C_M1",
                    "source_bbox": [32.0, 10.0, 52.0, 30.0],
                    "tile_intersection_bbox": [32.0, 10.0, 52.0, 30.0],
                },
                {
                    "id": 3,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 5.0, 5.0],
                    "area": 25.0,
                    "iscrowd": 1,
                    "ignore": 1,
                    "segmentation": [],
                    "partial_box": True,
                    "partial_box_policy": "clip_and_ignore",
                    "component_id": "RAW_PARTIAL",
                    "component_key": "C_2",
                    "source_record_id": "0001:component:C_2",
                    "source_bbox": [-5.0, -5.0, 5.0, 5.0],
                    "tile_intersection_bbox": [0.0, 0.0, 5.0, 5.0],
                },
            ],
        }
        self.component_path = self.root / "component_coco.json"
        _write_json(component, self.component_path)

        views: list[dict[str, object]] = []
        for kind, side in (("roi", None), ("side_strip", "L"), ("side_strip", "R"), ("side_strip", "T"), ("side_strip", "B")):
            view = self._view(f"pin:0001:C_1:{side or 'roi'}", kind, side)
            view["targets"] = self._targets()
            views.append(view)
        positive = {
            "schema_version": 1,
            "materializer_version": "component-pin-index-v1",
            "binding_sha256": BINDING,
            "record_id": "pin-component:0001:C_1",
            "record_number": 1,
            "sample_id": "0001",
            "split": "train",
            "duplicate_group": "sample-0001",
            "source_image": "0001/fixture.png",
            "source_image_sha256": "c" * 64,
            "component_id": "RAW_R1",
            "component_key": "C_1",
            "source_component_record_id": "0001:component:C_1",
            "component_bbox": [10.0, 10.0, 30.0, 30.0],
            "component_category": "R",
            "negative_component": False,
            "pins": [],
            "bbox_jitter": {},
            "views": views,
        }
        negative_views: list[dict[str, object]] = []
        for kind, side in (("roi", None), ("side_strip", "L"), ("side_strip", "R"), ("side_strip", "T"), ("side_strip", "B")):
            view = self._view(f"pin:0001:C_NEG:{side or 'roi'}", kind, side)
            view["targets"] = []
            negative_views.append(view)
        negative = {
            **positive,
            "record_id": "pin-component:0001:C_NEG",
            "record_number": 2,
            "component_id": "RAW_NEG",
            "component_key": "C_NEG",
            "negative_component": True,
            "views": negative_views,
        }
        self.pin_path = self.root / "pin_index.jsonl"
        _write_jsonl([positive, negative], self.pin_path)
        _write_json(
            {
                "schema_version": 1,
                "materializer_version": "component-pin-index-v1",
                "binding_sha256": BINDING,
                "config": {"heatmap_stride": 4},
                "outputs": {
                    "component_coco.json": {"sha256": _sha256(self.component_path)},
                    "pin_index.jsonl": {"sha256": _sha256(self.pin_path)},
                },
            },
            self.root / "summary.json",
        )

    def test_component_lazy_decode_filters_ignore_and_returns_detection_tensors(self) -> None:
        dataset = ComponentDetectionDataset(self.component_path, self.source)
        self.assertEqual(1, len(dataset))
        image, target = dataset[0]
        self.assertEqual((3, 64, 64), tuple(image.shape))
        self.assertEqual("cpu", image.device.type)
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)
        self.assertEqual(
            [[10.0, 10.0, 30.0, 30.0], [32.0, 10.0, 52.0, 30.0]],
            target["boxes"].tolist(),
        )
        self.assertEqual([1, 2], target["labels"].tolist())
        self.assertEqual([0, 35], target["official_category_ids"].tolist())
        self.assertGreaterEqual(min(target["labels"].tolist()), 1)
        self.assertEqual(1, target["ignored_annotation_count"])
        self.assertEqual(("RAW_V1", "RAW_M1"), target["component_ids"])
        self.assertEqual({0: 1, 35: 2}, dataset.official_to_model_label)
        self.assertEqual({1: 0, 2: 35}, dataset.model_label_to_official)
        self.assertEqual(
            (0, 35), tuple(item["official_id"] for item in dataset.categories)
        )

    def test_component_class_balanced_weights_are_capped_and_ignore_partial_boxes(self) -> None:
        dataset = ComponentDetectionDataset(self.component_path, self.source)
        dataset._images = (
            {"id": 1},
            {"id": 2},
            {"id": 3},
            {"id": 4},
        )
        dataset._annotations = {
            1: [{"category_id": 1, "ignore": 0}],
            2: [{"category_id": 1, "ignore": 0}],
            3: [{"category_id": 2, "ignore": 0}],
            4: [{"category_id": 2, "ignore": 1}],
        }
        self.assertEqual(
            (1.0, 1.0, 2.0**0.5, 1.0),
            dataset.class_balanced_sampling_weights(power=0.5),
        )
        self.assertEqual(
            (1.0, 1.0, 1.2, 1.0),
            dataset.class_balanced_sampling_weights(power=1.0, maximum_weight=1.2),
        )
        with self.assertRaisesRegex(ValueError, "power"):
            dataset.class_balanced_sampling_weights(power=1.1)

    def test_component_full_view_mixture_composes_with_class_weights(self) -> None:
        dataset = ComponentDetectionDataset(self.component_path, self.source)
        dataset._images = (
            {"id": 1, "view_kind": "full"},
            {"id": 2, "view_kind": "full"},
            {"id": 3, "view_kind": "tile"},
            {"id": 4, "view_kind": "tile"},
        )
        weights = dataset.full_view_mixture_sampling_weights(
            full_fraction=0.4,
            base_weights=(1.0, 2.0, 3.0, 4.0),
        )
        summary = dataset.sampling_view_kind_summary(weights)
        self.assertAlmostEqual(0.4, summary["full"]["expected_fraction"])
        self.assertAlmostEqual(0.6, summary["tile"]["expected_fraction"])
        self.assertAlmostEqual(2.0, weights[1] / weights[0])
        self.assertAlmostEqual(4.0 / 3.0, weights[3] / weights[2])
        dataset._images = (
            {"id": 1, "view_kind": "full"},
            {"id": 2, "view_kind": "full"},
        )
        with self.assertRaisesRegex(ValueError, "both full and tile"):
            dataset.full_view_mixture_sampling_weights(
                full_fraction=0.4,
                base_weights=(1.0, 1.0),
            )

    def test_pin_offset_index_view_selection_and_dense_targets(self) -> None:
        roi = PinKeypointDataset(self.pin_path, self.source, view_mode="roi")
        sides = PinKeypointDataset(self.pin_path, self.source, view_mode="side")
        left = PinKeypointDataset(self.pin_path, self.source, view_mode="L")
        without_negatives = PinKeypointDataset(
            self.pin_path, self.source, view_mode="all", include_negative=False
        )
        self.assertEqual(2, len(roi))
        self.assertEqual(8, len(sides))
        self.assertEqual(2, len(left))
        self.assertEqual(5, len(without_negatives))

        image, target = roi[0]
        self.assertEqual((3, 64, 64), tuple(image.shape))
        self.assertEqual((1, 16, 16), tuple(target["heatmap"].shape))
        self.assertAlmostEqual(1.0, float(target["heatmap"][0, 6, 5]), places=6)
        self.assertEqual([0.0, 0.0], target["offset"][:, 6, 5].tolist())
        self.assertEqual([1.0, 0.0], target["orientation"][:, 6, 5].tolist())
        self.assertEqual(1.0, float(target["regression_mask"][0, 6, 5]))
        self.assertEqual([2, 0], target["visibility"].tolist())
        self.assertEqual([[20.0, 24.0], [-1.0, -1.0]], target["tip_points"].tolist())
        self.assertEqual(
            [[5.0, 6.0], [-1.0, -1.0]], target["tip_points_heatmap"].tolist()
        )
        self.assertEqual(("PIN_VISIBLE", "PIN_HIDDEN"), target["pin_ids"])

        _, negative = roi[1]
        self.assertEqual((1, 16, 16), tuple(negative["heatmap"].shape))
        self.assertEqual(0.0, float(negative["heatmap"].sum()))
        self.assertEqual((0,), tuple(negative["visibility"].shape))

    def test_component_and_pin_collate_stack_dense_cpu_tensors(self) -> None:
        components = ComponentDetectionDataset(self.component_path, self.source)
        component_images, component_targets = component_detection_collate(
            [components[0], components[0]]
        )
        self.assertEqual((2, 3, 64, 64), tuple(component_images.shape))
        self.assertEqual(2, len(component_targets))

        pins = PinKeypointDataset(
            self.pin_path, self.source, view_mode="roi", include_negative=False
        )
        pin_images, pin_targets = pin_keypoint_collate([pins[0], pins[0]])
        self.assertEqual((2, 3, 64, 64), tuple(pin_images.shape))
        self.assertEqual((2, 1, 16, 16), tuple(pin_targets["heatmap"].shape))
        self.assertEqual((2, 2, 16, 16), tuple(pin_targets["offset"].shape))
        self.assertEqual(2, len(pin_targets["visibility"]))
        self.assertEqual("pin:0001:C_1:roi", pin_targets["metadata"][0]["view_id"])

    def test_pin_offset_reader_detects_file_mutation(self) -> None:
        dataset = PinKeypointDataset(self.pin_path, self.source, view_mode="roi")
        with self.pin_path.open("ab") as handle:
            handle.write(b"\n")
        with self.assertRaisesRegex(RuntimeError, "changed after"):
            _ = dataset[0]

    def test_transform_contract_corruption_fails_on_read(self) -> None:
        records = [
            json.loads(line)
            for line in self.pin_path.read_text(encoding="utf-8").splitlines()
        ]
        records[0]["views"][0]["targets"][0]["tip_view"] = [21.0, 24.0]
        _write_jsonl(records, self.pin_path)
        # The test intentionally bypasses the now-stale commit marker to prove
        # item-level geometry validation is independent of file hashing.
        dataset = PinKeypointDataset(
            self.pin_path, self.source, view_mode="roi", verify_commit=False
        )
        with self.assertRaisesRegex(ValueError, "tip_view violates transform"):
            _ = dataset[0]


def _write_json(value: object, path: Path) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(values: list[dict[str, object]], path: Path) -> None:
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
