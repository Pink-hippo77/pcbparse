from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from data_tools.derive_datasets import main as derive_main
from pcbparse.data.derive import (
    TASK_NAMES,
    derive_training_datasets,
    write_derived_datasets,
)
from pcbparse.data.manifest import read_jsonl, write_jsonl


class DerivedDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "Dataset300"
        self.root.mkdir()
        self._create_sample("0001", first=True)
        self._create_sample("0002", first=False)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _create_sample(self, sample_id: str, *, first: bool) -> None:
        directory = self.root / sample_id
        directory.mkdir()
        base = f"{sample_id}-derive"
        image_path = directory / f"{base}.png"
        Image.new(
            "RGB",
            (100, 80),
            color=(30 + int(sample_id), 90, 150),
        ).save(image_path, format="PNG")

        if first:
            pin = {
                "pinId": "P_RAW_1",
                "edgeId": "E_1",
                "pinNumberByOCR": "T_NUM",
                "pinNumberByVLM": None,
                "pinNameByOCR": None,
                "pinNameByVLM": "",
                "tip": [5, 10],
                "base": [10, 10],
            }
            orphan = {
                "pinId": "P_ORPHAN_RAW",
                "edgeId": None,
                "pinNumberByOCR": None,
                "pinNumberByVLM": None,
                "pinNameByOCR": None,
                "pinNameByVLM": None,
                "tip": [50, 50],
                "base": [55, 50],
            }
            components = {
                "C_1": {
                    "cpntId": "C_RAW_1",
                    "state": "active",
                    "category": "R",
                    "categoryId": "4",
                    "location": {"x1": -5, "y1": 10, "x2": 20, "y2": 30},
                    "realBBox": {"x1": 0, "y1": 12, "x2": 18, "y2": 28},
                    "rotation": None,
                    "pins": {"P_1": pin},
                    "associatedTexts": {"T_1": {}},
                },
                "C_DELETED": {
                    "cpntId": "C_RAW_DELETED",
                    "state": "deleted",
                    "category": "C",
                    "categoryId": "3",
                    "location": {"x1": 30, "y1": 30, "x2": 40, "y2": 40},
                    "realBBox": {"x1": 31, "y1": 31, "x2": 39, "y2": 39},
                    "pins": {},
                    "associatedTexts": {},
                },
            }
            pins = {"P_1": pin, "P_ORPHAN": orphan}
            texts = {
                "T_1": {
                    "tid": "T_RAW_1",
                    "state": "active",
                    "category": "params",
                    "categoryId": 0,
                    "content": "  RΩ\nRaw Text  ",
                    "details": "designator ",
                    "location": {"x1": 90, "y1": 5, "x2": 105, "y2": 20},
                }
            }
            edges = {
                "E_1": self._edge(
                    "E_RAW_1",
                    skeleton=[[0, 10], [1, 10], [2, 10], [5, 5]],
                    endpoints=[[0, 10], [5, 5]],
                    diverge=[[2, 10]],
                    turn=[[1, 10]],
                    paths=[[[0, 10], [5, 5]]],
                ),
                "E_2": self._edge(
                    "E_RAW_2",
                    skeleton=[[2, 10], [2, 11], [2, 12]],
                    endpoints=[[2, 10], [2, 12]],
                    diverge=[],
                    turn=[],
                    paths=[[[2, 10], [2, 12]]],
                ),
                "E_DELETED": {
                    **self._edge(
                        "E_RAW_DELETED",
                        skeleton=[[60, 60]],
                        endpoints=[[60, 60]],
                        diverge=[],
                        turn=[],
                        paths=[],
                    ),
                    "state": "deleted",
                },
            }
            source_class = "datasheet"
        else:
            pin = {
                "pinId": "P_RAW_2",
                "edgeId": "E_3",
                "pinNumberByOCR": None,
                "pinNumberByVLM": None,
                "pinNameByOCR": "T_NAME",
                "pinNameByVLM": None,
                "tip": [20, 20],
                "base": [20, 25],
            }
            components = {
                "C_2": {
                    "cpntId": "C_RAW_2",
                    "state": "active",
                    "category": "C",
                    "categoryId": "3",
                    "location": {"x1": 10, "y1": 10, "x2": 30, "y2": 30},
                    "realBBox": {"x1": 12, "y1": 12, "x2": 28, "y2": 28},
                    "rotation": 90,
                    "pins": {"P_2": pin},
                    "associatedTexts": {},
                }
            }
            pins = {"P_2": pin}
            texts = {
                "T_2": {
                    "tid": "T_RAW_2",
                    "state": "active",
                    "category": "normal",
                    "categoryId": 1,
                    "content": "C2",
                    "details": "",
                    "location": {"x1": 10, "y1": 35, "x2": 20, "y2": 45},
                }
            }
            edges = {
                "E_3": self._edge(
                    "E_RAW_3",
                    skeleton=[[10, 10], [15, 10], [20, 10]],
                    endpoints=[[10, 10], [20, 10]],
                    diverge=[],
                    turn=[],
                    paths=[[[10, 10], [20, 10]]],
                )
            }
            source_class = "jlc"

        annotation = {
            "canvas": {"width": 100, "height": 80},
            "nowCls": source_class,
            "components": components,
            "pins": pins,
            "texts": texts,
            "edges": edges,
        }
        (directory / f"{base}_CVJsonStd.json").write_text(
            json.dumps(
                annotation,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        (directory / f"{base}_CVJsonStd.eds").write_text(
            f"(edif SAMPLE_{sample_id} (library LIB))\n", encoding="utf-8"
        )
        (directory / f"{base}_spice.net").write_text(
            ".title derive\nR1 1 0 1k\n.end\n", encoding="utf-8"
        )

    @staticmethod
    def _edge(
        edge_id: str,
        *,
        skeleton: list[list[int]],
        endpoints: list[list[int]],
        diverge: list[list[int]],
        turn: list[list[int]],
        paths: list[list[list[int]]],
    ) -> dict[str, object]:
        return {
            "edgeId": edge_id,
            "state": "active",
            "category": "wire",
            "netlabel": "RAW_NET",
            "isFlywire": False,
            "text": {},
            "location": [],
            "originalCoordinatesCount": len(skeleton),
            "connect": {},
            "keyPoints": {
                "skeleton": {"wire": skeleton},
                "endPoints": {"wire": endpoints},
                "divergePoints": {"wire": diverge},
                "turnPoints": {"wire": turn},
            },
            "keyPointPaths": {"wire": paths},
            # This sentinel must never be copied to wire-path output.
            "coordinates": [[999, 999], [998, 998]],
            "associatedTexts": {},
        }

    def test_five_tasks_preserve_raw_fields_and_lazy_geometry(self) -> None:
        first = derive_training_datasets(
            self.root, version="fixture-v1", patch_size=16, expected_count=2
        )
        second = derive_training_datasets(
            self.root, version="fixture-v1", patch_size=16, expected_count=2
        )

        self.assertEqual(first.records, second.records)
        self.assertEqual(first.summary, second.summary)
        self.assertEqual(set(TASK_NAMES), set(first.records))
        self.assertEqual(
            {
                "component_detection": 2,
                "pin_keypoints": 2,
                "ocr_text": 2,
                "wire_paths": 3,
                "junction_patches": 7,
            },
            first.summary["counts"],
        )

        component = next(
            record
            for record in first.records["component_detection"]
            if record["sample_id"] == "0001"
        )
        self.assertEqual(-5, component["raw_bbox"]["x1"])
        self.assertEqual(0, component["bbox"]["x1"])
        self.assertIn("bbox_clipped", component["issue_flags"])
        self.assertEqual("C_RAW_1", component["component_id"])

        pin = next(
            record
            for record in first.records["pin_keypoints"]
            if record["sample_id"] == "0001"
        )
        self.assertEqual("left", pin["orientation"]["label"])
        self.assertFalse(pin["missing_label_mask"]["pin_number_by_ocr"])
        self.assertTrue(pin["missing_label_mask"]["pin_name_by_ocr"])
        self.assertEqual(1, first.summary["skipped"]["unowned_root_pins"])

        text = next(
            record
            for record in first.records["ocr_text"]
            if record["sample_id"] == "0001"
        )
        self.assertEqual("  RΩ\nRaw Text  ", text["content"])
        self.assertEqual("designator ", text["details"])
        self.assertFalse(text["normalization_applied"])
        self.assertEqual(100, text["bbox"]["x2"])
        self.assertIn("bbox_clipped", text["issue_flags"])

        wire = first.records["wire_paths"][0]
        self.assertNotIn("coordinates", wire)
        self.assertFalse(wire["dense_coordinates_included"])
        self.assertIn("json_pointer", wire["key_point_paths_ref"])
        self.assertIn("json_pointer", wire["skeleton_ref"])

        cross = next(
            record
            for record in first.records["junction_patches"]
            if record["sample_id"] == "0001"
            and record["center"] == [2, 10]
        )
        self.assertEqual(["cross", "diverge", "endpoint"], cross["candidate_types"])
        self.assertTrue(cross["cross"]["is_cross"])
        self.assertEqual(["E_1", "E_2"], cross["cross"]["edge_keys"])
        self.assertEqual(["E_RAW_1", "E_RAW_2"], cross["cross"]["edge_ids"])

        border = next(
            record
            for record in first.records["junction_patches"]
            if record["sample_id"] == "0001"
            and record["center"] == [0, 10]
        )
        self.assertEqual(0, border["patch_bbox"]["x1"])
        self.assertEqual(8, border["padding"]["left"])
        self.assertIn("patch_bbox_clipped", border["issue_flags"])

    def test_split_manifest_propagation_atomic_outputs_and_cli(self) -> None:
        split_manifest = Path(self.temporary_directory.name) / "splits.jsonl"
        write_jsonl(
            [
                {"sample_id": "0001", "split": "train"},
                {"sample_id": "0002", "split": "val"},
            ],
            split_manifest,
        )
        result = derive_training_datasets(
            self.root,
            version="fixture-split-v1",
            patch_size=20,
            split_manifest=split_manifest,
            expected_count=2,
        )
        output = Path(self.temporary_directory.name) / "derived"
        written = write_derived_datasets(result, output)

        self.assertEqual(set(TASK_NAMES) | {"summary"}, set(written))
        for task in TASK_NAMES:
            records = [
                json.loads(line)
                for line in written[task].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(result.summary["counts"][task], len(records))
            self.assertEqual(
                {"train", "val"}, {record["split"] for record in records}
            )
        summary = json.loads(written["summary"].read_text(encoding="utf-8"))
        self.assertEqual(64, len(summary["config_hash"]))
        self.assertEqual(result.summary["counts"], summary["counts"])

        cli_output = Path(self.temporary_directory.name) / "cli-derived"
        with contextlib.redirect_stdout(io.StringIO()):
            return_code = derive_main(
                [
                    "--src",
                    str(self.root),
                    "--out",
                    str(cli_output),
                    "--version",
                    "cli-v1",
                    "--split-manifest",
                    str(split_manifest),
                    "--patch-size",
                    "18",
                    "--expected-count",
                    "2",
                ]
            )
        self.assertEqual(0, return_code)
        self.assertTrue((cli_output / "summary.json").is_file())

    def test_split_manifest_requires_exact_sample_coverage(self) -> None:
        split_manifest = Path(self.temporary_directory.name) / "incomplete.jsonl"
        write_jsonl([{"sample_id": "0001", "split": "train"}], split_manifest)

        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            derive_training_datasets(
                self.root,
                version="bad-split",
                split_manifest=split_manifest,
                expected_count=2,
            )


if __name__ == "__main__":
    unittest.main()
