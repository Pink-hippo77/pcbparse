from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from PIL import Image

from data_tools.export_yolo_dataset import build_tiles, export_yolo_datasets
from data_tools.validate_yolo_dataset import validate_yolo_d1


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _component(
    sample_id: str,
    split: str,
    *,
    index: int,
    category: str,
    category_id: str,
    bbox: tuple[int, int, int, int],
    domain: str,
) -> dict[str, Any]:
    return {
        "task": "component_detection",
        "record_id": f"{sample_id}:component:{index}",
        "sample_id": sample_id,
        "split": split,
        "state": "active",
        "source_domain": domain,
        "source_image": f"{sample_id}/image.png",
        "category": category,
        "category_id": category_id,
        "bbox": dict(zip(("x1", "y1", "x2", "y2"), bbox)),
    }


def _text(
    sample_id: str,
    split: str,
    *,
    index: int,
    bbox: tuple[int, int, int, int],
    domain: str,
) -> dict[str, Any]:
    return {
        "task": "ocr_text",
        "record_id": f"{sample_id}:text:{index}",
        "sample_id": sample_id,
        "split": split,
        "state": "active",
        "source_domain": domain,
        "source_image": f"{sample_id}/image.png",
        "category": "params",
        "category_id": 0,
        "content": "R1",
        "details": "designator",
        "bbox": dict(zip(("x1", "y1", "x2", "y2"), bbox)),
    }


class ExportYoloDatasetTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Path]:
        dataset = root / "dataset"
        samples = [
            ("train-a", "train", "datasheet", "group-train"),
            ("val-k", "val", "kicad", "group-val"),
            ("test-a", "test", "ad", "group-test"),
        ]
        source_rows: list[dict[str, Any]] = []
        split_rows: list[dict[str, Any]] = []
        for sample_id, split, domain, group in samples:
            image_path = dataset / sample_id / "image.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (100, 80), (250, 250, 250)).save(image_path)
            source_rows.append(
                {
                    "sample_id": sample_id,
                    "source_class": domain,
                    "duplicate_group": group,
                    "canvas": {"width": 100, "height": 80},
                    "files": {
                        "image": {
                            "path": f"{sample_id}/image.png",
                            "width": 100,
                            "height": 80,
                            "sha256": _sha(image_path),
                        }
                    },
                }
            )
            split_rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "duplicate_group": group,
                }
            )

        components = [
            _component(
                "train-a",
                "train",
                index=1,
                category="Common",
                category_id="3",
                bbox=(5, 5, 25, 25),
                domain="datasheet",
            ),
            _component(
                "train-a",
                "train",
                index=2,
                category="Common",
                category_id="3",
                bbox=(55, 25, 90, 60),
                domain="datasheet",
            ),
            _component(
                "train-a",
                "train",
                index=3,
                category="Rare",
                category_id="9",
                bbox=(40, 10, 68, 35),
                domain="datasheet",
            ),
            _component(
                "val-k",
                "val",
                index=1,
                category="Rare",
                category_id="9",
                bbox=(12, 12, 50, 45),
                domain="kicad",
            ),
            _component(
                "test-a",
                "test",
                index=1,
                category="Common",
                category_id="3",
                bbox=(10, 10, 30, 30),
                domain="ad",
            ),
        ]
        texts = [
            _text("train-a", "train", index=1, bbox=(10, 45, 30, 55), domain="datasheet"),
            _text("val-k", "val", index=1, bbox=(50, 60, 80, 72), domain="kicad"),
            _text("test-a", "test", index=1, bbox=(10, 40, 30, 50), domain="ad"),
        ]
        paths = {
            "dataset": dataset,
            "component": root / "component.jsonl",
            "text": root / "text.jsonl",
            "source": root / "source.jsonl",
            "split": root / "split.jsonl",
            "output": root / "output",
        }
        _write_jsonl(paths["component"], components)
        _write_jsonl(paths["text"], texts)
        _write_jsonl(paths["source"], source_rows)
        _write_jsonl(paths["split"], split_rows)
        return paths

    def test_build_tiles_covers_edges_without_duplicate_final_origin(self) -> None:
        tiles = build_tiles(100, 80, 64, 16)
        self.assertEqual(
            [(0, 0), (36, 0), (0, 16), (36, 16)],
            [(tile.x1, tile.y1) for tile in tiles],
        )
        self.assertTrue(all(tile.output_width == 64 for tile in tiles))

    def test_exports_two_yolo_datasets_and_special_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._fixture(root)
            result = export_yolo_datasets(
                workspace=root,
                dataset_root=paths["dataset"],
                component_manifest=paths["component"],
                text_manifest=paths["text"],
                source_manifest=paths["source"],
                split_manifest=paths["split"],
                output=paths["output"],
                tile_size=64,
                tile_overlap=16,
                component_min_visible_fraction=0.25,
                text_min_visible_fraction=0.50,
                minimum_clipped_box_pixels=2.0,
                long_tail_max_train_instances=1,
                special_validation_split="val",
                verify_source_hashes=True,
                overwrite=False,
            )

            self.assertTrue(result["ok"])
            output = paths["output"]
            self.assertTrue((output / "component_yolo/dataset.yaml").is_file())
            self.assertTrue((output / "text_yolo/dataset.yaml").is_file())
            component_yaml = (output / "component_yolo/dataset.yaml").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("path: .", component_yaml)
            self.assertIn("train: images/train", component_yaml)
            self.assertIn("val: images/val", component_yaml)
            classes = json.loads((output / "component_yolo/classes.json").read_text())
            self.assertEqual([0, 1], [row["yolo_class_id"] for row in classes["classes"]])
            self.assertEqual("Common", classes["classes"][0]["source_category_name"])
            self.assertEqual("Rare", classes["classes"][1]["source_category_name"])

            audit = json.loads((output / "conversion_audit.json").read_text())
            self.assertEqual("passed", audit["status"])
            self.assertEqual(0, audit["split_contract"]["test_rows_materialized_for_training"])
            self.assertEqual(
                audit["coverage"]["component_expected_records"],
                audit["coverage"]["component_emitted_records"],
            )
            self.assertEqual(
                audit["coverage"]["text_expected_records"],
                audit["coverage"]["text_emitted_records"],
            )

            all_training_paths = [path.as_posix() for path in output.rglob("*")]
            self.assertFalse(any("test-a" in path for path in all_training_paths))

            special = json.loads((output / "special_validation/summary.json").read_text())
            self.assertEqual("val", special["source_split"])
            self.assertFalse(special["training_rows_included"])
            self.assertEqual(1, special["kicad_sample_count"])
            self.assertEqual(["val-k"], special["sample_ids"])
            self.assertIn("Rare", special["long_tail_definition"]["classes"])
            self.assertTrue(
                list((output / "special_validation/component_yolo/images/val").glob("*.png"))
            )
            self.assertTrue(
                (output / "special_validation/kicad/component_yolo/dataset.yaml").is_file()
            )
            self.assertTrue(
                (output / "special_validation/long_tail/component_yolo/dataset.yaml").is_file()
            )
            visual_manifest = [
                json.loads(line)
                for line in (output / "visual_audit/manifest.jsonl").read_text().splitlines()
            ]
            self.assertTrue(any(row["task"] == "component" for row in visual_manifest))
            self.assertTrue(any(row["task"] == "text" for row in visual_manifest))
            self.assertTrue(
                all((output / row["overlay_path"]).is_file() for row in visual_manifest)
            )

            for label_path in output.glob("component_yolo/labels/*/*.txt"):
                for line in label_path.read_text().splitlines():
                    fields = line.split()
                    self.assertEqual(5, len(fields))
                    self.assertIn(int(fields[0]), {0, 1})
                    self.assertTrue(all(0.0 <= float(value) <= 1.0 for value in fields[1:]))

            manifest_rows = [
                json.loads(line)
                for line in (output / "tile_manifest.jsonl").read_text().splitlines()
            ]
            self.assertTrue(manifest_rows)
            self.assertTrue(all("crop_xyxy" in row for row in manifest_rows))
            self.assertTrue(all("source_image_sha256" in row for row in manifest_rows))
            self.assertTrue(all(row["split"] in {"train", "val"} for row in manifest_rows))
            text_labels = [
                label
                for row in manifest_rows
                for label in row["tasks"]["text"]["labels"]
            ]
            self.assertTrue(text_labels)
            self.assertEqual("R1", text_labels[0]["source_annotation"]["content"])
            self.assertEqual("designator", text_labels[0]["source_annotation"]["details"])

            validation = validate_yolo_d1(output, verify_inventory=True)
            self.assertTrue(validation["ok"])
            self.assertTrue(validation["inventory"]["verified"])

            first_label = next(output.glob("component_yolo/labels/*/*.txt"))
            first_label.write_text("99 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "class id outside"):
                validate_yolo_d1(output, verify_inventory=False)

    def test_rejects_manifest_split_that_differs_from_source_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._fixture(root)
            rows = [json.loads(line) for line in paths["component"].read_text().splitlines()]
            rows[0]["split"] = "val"
            _write_jsonl(paths["component"], rows)
            with self.assertRaisesRegex(ValueError, "split differs"):
                export_yolo_datasets(
                    workspace=root,
                    dataset_root=paths["dataset"],
                    component_manifest=paths["component"],
                    text_manifest=paths["text"],
                    source_manifest=paths["source"],
                    split_manifest=paths["split"],
                    output=paths["output"],
                    tile_size=64,
                    tile_overlap=16,
                    component_min_visible_fraction=0.25,
                    text_min_visible_fraction=0.50,
                    minimum_clipped_box_pixels=2.0,
                    long_tail_max_train_instances=1,
                    special_validation_split="val",
                    verify_source_hashes=True,
                    overwrite=False,
                )


if __name__ == "__main__":
    unittest.main()
