from __future__ import annotations

import hashlib
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

from pcbparse.data.manifest import write_json, write_jsonl
from pcbparse.training.component_pin_materialize import (
    ComponentPinMaterializeConfig,
    decode_indexed_view,
    materialize_component_pin_training_data,
    transform_point,
)


class ComponentPinMaterializeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name) / "workspace"
        self.workspace.mkdir()
        self._write_fixture()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_fixture(self) -> None:
        source_root = self.workspace / "source"
        image_path = source_root / "0001" / "fixture.png"
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (800, 600), (245, 245, 245)).save(image_path)
        annotation_path = source_root / "0001" / "fixture_CVJsonStd.json"
        annotation_path.write_text("{}\n", encoding="utf-8")

        source_row = {
            "schema_version": 1,
            "sample_id": "0001",
            "base_name": "fixture",
            "source_class": "kicad",
            "duplicate_group": "sample-0001",
            "duplicate_group_size": 1,
            "canvas": {"width": 800, "height": 600},
            "files": {
                "image": {
                    "path": "0001/fixture.png",
                    "sha256": _sha256(image_path),
                },
                "json": {
                    "path": "0001/fixture_CVJsonStd.json",
                    "sha256": _sha256(annotation_path),
                },
            },
        }
        source_manifest = self.workspace / "data/manifests/source.jsonl"
        split_manifest = self.workspace / "data/splits/split.jsonl"
        write_jsonl([source_row], source_manifest)
        write_jsonl(
            [
                {
                    **source_row,
                    "split": "train",
                }
            ],
            split_manifest,
        )

        config_hash = "a" * 64
        common = {
            "schema_version": 1,
            "derivation_version": "fixture-v2",
            "config_hash": config_hash,
            "sample_id": "0001",
            "source_domain": "kicad",
            "source_image": "0001/fixture.png",
            "source_json": "0001/fixture_CVJsonStd.json",
            "duplicate_group": "sample-0001",
            "split": "train",
        }
        components = [
            {
                **common,
                "task": "component_detection",
                "record_id": "0001:component:C_LARGE",
                "component_id": "RAW/LARGE",
                "component_key": "C_LARGE",
                "category": "Box",
                "category_id": "1",
                "bbox": {"x1": 100, "y1": 100, "x2": 700, "y2": 500},
                "raw_bbox": {"x1": 100, "y1": 100, "x2": 700, "y2": 500},
                "real_bbox": {"x1": 110, "y1": 110, "x2": 690, "y2": 490},
                "issue_flags": [],
            },
            {
                **common,
                "task": "component_detection",
                "record_id": "0001:component:C_SMALL",
                "component_id": "RAW_SMALL",
                "component_key": "C_SMALL",
                "category": "R",
                "category_id": "3",
                "bbox": {"x1": 20, "y1": 20, "x2": 80, "y2": 80},
                "raw_bbox": {"x1": 20, "y1": 20, "x2": 80, "y2": 80},
                "real_bbox": {"x1": 22, "y1": 22, "x2": 78, "y2": 78},
                "issue_flags": [],
            },
        ]
        pins = [
            {
                **common,
                "task": "pin_keypoints",
                "record_id": "0001:pin:P_LEFT",
                "pin_id": "RAW_PIN_LEFT",
                "pin_key": "P_LEFT",
                "owner_component_id": "RAW/LARGE",
                "owner_component_key": "C_LARGE",
                "owner_category": "Box",
                "edge_id": "EDGE_1",
                "tip": [100.0, 300.0],
                "base": [120.0, 300.0],
                "orientation": {
                    "label": "left",
                    "angle_degrees": 180.0,
                    "unit_vector": [-1.0, 0.0],
                    "vector": [-20.0, 0.0],
                    "image_y_axis": "down",
                },
                "label_refs": {},
                "missing_label_mask": {},
                "issue_flags": [],
            }
        ]
        derived = self.workspace / "derived"
        write_jsonl(components, derived / "component_detection.jsonl")
        write_jsonl(pins, derived / "pin_keypoints.jsonl")
        write_json(
            {
                "schema_version": 1,
                "derivation_version": "fixture-v2",
                "config_hash": config_hash,
                "sample_count": 1,
                "config": {"split_manifest_sha256": _sha256(split_manifest)},
                "counts": {
                    "component_detection": 2,
                    "pin_keypoints": 1,
                },
                "task_files": {
                    "component_detection": "component_detection.jsonl",
                    "pin_keypoints": "pin_keypoints.jsonl",
                },
            },
            derived / "summary.json",
        )

    def _materialize(self, name: str, *, crops: bool = False):
        return materialize_component_pin_training_data(
            self.workspace,
            self.workspace / name,
            derived_root="derived",
            source_root="source",
            source_manifest="data/manifests/source.jsonl",
            split_manifest="data/splits/split.jsonl",
            write_crops=crops,
        )

    def test_component_full_and_tiles_make_partial_policy_explicit(self) -> None:
        result = self._materialize("output")
        coco = json.loads(result.component_index.read_text(encoding="utf-8"))
        self.assertEqual(5, len(coco["images"]))  # one full view plus four tiles
        full = next(image for image in coco["images"] if image["view_kind"] == "full")
        full_large = next(
            annotation
            for annotation in coco["annotations"]
            if annotation["image_id"] == full["id"]
            and annotation["component_key"] == "C_LARGE"
        )
        self.assertEqual(0, full_large["ignore"])
        self.assertFalse(full_large["partial_box"])
        tile_large = [
            annotation
            for annotation in coco["annotations"]
            if annotation["component_key"] == "C_LARGE"
            and annotation["image_id"] != full["id"]
        ]
        self.assertTrue(tile_large)
        self.assertTrue(all(item["ignore"] == 1 for item in tile_large))
        self.assertTrue(all(item["iscrowd"] == 1 for item in tile_large))
        self.assertTrue(
            all(item["partial_box_policy"] == "clip_and_ignore" for item in tile_large)
        )
        self.assertTrue(all(item["source_bbox"] == [100.0, 100.0, 700.0, 500.0] for item in tile_large))

    def test_pin_views_have_reversible_transforms_targets_and_jitter_contract(self) -> None:
        result = self._materialize("output")
        records = [
            json.loads(line)
            for line in result.pin_index.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(2, len(records))
        large = next(record for record in records if record["component_key"] == "C_LARGE")
        self.assertEqual(["roi", "side_strip", "side_strip", "side_strip", "side_strip"], [view["view_kind"] for view in large["views"]])
        self.assertEqual([None, "L", "R", "T", "B"], [view["side"] for view in large["views"]])
        self.assertTrue(large["bbox_jitter"]["enabled"])
        self.assertFalse(large["bbox_jitter"]["materialized_jitter_applied"])

        roi = large["views"][0]
        original = (321.25, 222.75)
        in_view = transform_point(original, roi["transform"]["forward"])
        restored = transform_point(in_view, roi["transform"]["inverse"])
        self.assertAlmostEqual(original[0], restored[0], places=7)
        self.assertAlmostEqual(original[1], restored[1], places=7)

        left = next(view for view in large["views"] if view["side"] == "L")
        right = next(view for view in large["views"] if view["side"] == "R")
        top = next(view for view in large["views"] if view["side"] == "T")
        bottom = next(view for view in large["views"] if view["side"] == "B")
        self.assertEqual(2, left["targets"][0]["visibility"])
        self.assertEqual(0, right["targets"][0]["visibility"])
        self.assertEqual(0, top["targets"][0]["visibility"])
        self.assertEqual(0, bottom["targets"][0]["visibility"])
        self.assertIsNotNone(left["targets"][0]["heatmap"]["point"])
        self.assertIsNone(right["targets"][0]["heatmap"]["point"])
        self.assertEqual("RAW_PIN_LEFT", large["pins"][0]["pin_id"])
        negative = next(record for record in records if record["component_key"] == "C_SMALL")
        self.assertTrue(negative["negative_component"])
        self.assertEqual([], negative["pins"])

    def test_lazy_decode_optional_atomic_crops_and_determinism(self) -> None:
        first = self._materialize("first", crops=True)
        second = self._materialize("second", crops=True)
        self.assertEqual(15, first.crop_count)
        self.assertEqual(first.binding_sha256, second.binding_sha256)
        self.assertEqual(_sha256(first.component_index), _sha256(second.component_index))
        self.assertEqual(_sha256(first.pin_index), _sha256(second.pin_index))
        self.assertEqual(_sha256(first.summary), _sha256(second.summary))

        coco = json.loads(first.component_index.read_text(encoding="utf-8"))
        view = coco["images"][0]
        decoded = decode_indexed_view(view, self.workspace / "source")
        self.assertEqual((512, 512), decoded.size)
        decoded.close()
        crop = self.workspace / "first" / view["materialized_crop"]["path"]
        self.assertTrue(crop.is_file())
        self.assertEqual(view["materialized_crop"]["sha256"], _sha256(crop))
        summary = json.loads(first.summary.read_text(encoding="utf-8"))
        self.assertEqual(_sha256(first.component_index), summary["outputs"]["component_coco.json"]["sha256"])

    def test_split_lineage_hash_mismatch_fails_closed(self) -> None:
        summary_path = self.workspace / "derived/summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["config"]["split_manifest_sha256"] = "f" * 64
        write_json(summary, summary_path)
        with self.assertRaisesRegex(ValueError, "split manifest SHA-256"):
            self._materialize("invalid")

    def test_invalid_config_rejects_side_strips_outside_roi_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "contain the configured side-strip"):
            ComponentPinMaterializeConfig(
                pin_roi_margin_px=8,
                pin_strip_outside_px=64,
            )


@unittest.skipUnless(
    (REPOSITORY_ROOT / "data/derived/dataset300-v2/summary.json").is_file()
    and (REPOSITORY_ROOT / "Dataset300/Dataset300/0002/0002-Datasheet.png").is_file(),
    "official Dataset300-v2 is not available",
)
class OfficialDatasetComponentPinSmokeTests(unittest.TestCase):
    def test_real_sample_materializes_against_current_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = materialize_component_pin_training_data(
                REPOSITORY_ROOT,
                Path(temporary_directory) / "materialized",
                sample_ids=["0002"],
                write_crops=False,
                verify_source_hashes=True,
            )
            summary = json.loads(result.summary.read_text(encoding="utf-8"))
            self.assertEqual(["0002"], summary["selection"])
            self.assertGreater(summary["counts"]["component_views"], 1)
            self.assertGreater(summary["counts"]["pin_components"], 0)
            self.assertEqual(0, summary["counts"]["written_crops"])
            self.assertEqual(64, len(summary["binding_sha256"]))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
