from __future__ import annotations

import csv
import hashlib
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

from pcbparse.training.wire_junction_materialize import (
    JUNCTION_CLASSES,
    MaterializationError,
    WireJunctionMaterializeConfig,
    _load_annotation_without_dense_coordinates,
    materialize_wire_junction,
)


class WireJunctionMaterializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.fixture = self._write_fixture(self.root / "fixture")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_materializes_thin_masks_ignore_regions_and_safe_junction_labels(self) -> None:
        result = materialize_wire_junction(
            **self.fixture,
            output_root=self.root / "output",
            config=WireJunctionMaterializeConfig(tile_size=16),
        )

        self.assertTrue(result.summary["materialization"]["ok"])
        self.assertFalse(result.training_ready)
        self.assertEqual(4, result.summary["materialization"]["wire_tile_count"])
        self.assertEqual(
            {
                "endpoint": 1,
                "t_junction": 1,
                "bend": 1,
                "connected_cross": 0,
                "non_connected_cross": 0,
                "noise": 0,
            },
            result.summary["junction_class_counts"]["trusted"],
        )
        self.assertEqual(1, result.summary["junction_class_counts"]["UNKNOWN"])
        self.assertEqual(
            ["connected_cross", "non_connected_cross", "noise"],
            result.summary["training_gate"]["missing_classes_global"],
        )
        self.assertEqual(
            ["connected_cross", "non_connected_cross", "noise"],
            result.summary["training_gate"]["missing_classes_in_train"],
        )

        tile_rows = _read_jsonl(result.run_dir / "wire_tiles.jsonl")
        first = next(row for row in tile_rows if row["tile_bbox"] == {"x1": 0, "y1": 0, "x2": 16, "y2": 16})
        with Image.open(result.run_dir / first["target_mask"]["path"]) as target:
            self.assertEqual(255, target.getpixel((1, 10)))
            self.assertEqual(0, target.getpixel((3, 10)))  # text bbox
            self.assertEqual(0, target.getpixel((7, 10)))  # component bbox
            self.assertEqual(0, target.getpixel((12, 10)))  # ignored pin line
            self.assertEqual(255, target.getpixel((15, 10)))
            # The target is a one-pixel centerline outside explicit exclusions.
            self.assertEqual(0, target.getpixel((1, 9)))
        with Image.open(result.run_dir / first["ignore_mask"]["path"]) as ignore:
            self.assertEqual(255, ignore.getpixel((3, 10)))
            self.assertEqual(255, ignore.getpixel((7, 10)))
            self.assertEqual(255, ignore.getpixel((12, 10)))
            self.assertEqual(0, ignore.getpixel((1, 10)))
        self.assertTrue(first["supervision"]["eligible"])
        self.assertFalse(first["lineage"]["dense_coordinates_loaded"])
        self.assertEqual(0.0, first["transform"]["tile_to_image"][0][2])

        junction_rows = _read_jsonl(result.run_dir / "junction_annotations.jsonl")
        labels = {row["record_id"]: row for row in junction_rows}
        self.assertEqual("endpoint", labels["s1:junction:endpoint"]["label"])
        self.assertEqual("t_junction", labels["s1:junction:diverge"]["label"])
        self.assertEqual("bend", labels["s1:junction:turn"]["label"])
        cross = labels["s1:junction:cross"]
        self.assertEqual("UNKNOWN", cross["label"])
        self.assertEqual(-1, cross["label_id"])
        self.assertEqual("needs_review", cross["label_status"])
        self.assertEqual(
            "cross_connectivity_not_encoded_by_gt", cross["review_reason"]
        )
        self.assertNotIn(cross["label"], {"connected_cross", "non_connected_cross"})
        self.assertTrue((result.run_dir / cross["patch"]["path"]).is_file())

        review = _read_jsonl(result.run_dir / "junction_review_queue.jsonl")
        self.assertEqual(["s1:junction:cross"], [row["record_id"] for row in review])
        with (result.run_dir / "junction_review_template.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            template_rows = list(csv.DictReader(handle))
        self.assertEqual("pending", template_rows[0]["review_status"])
        self.assertEqual("", template_rows[0]["reviewed_label"])

        schema = json.loads(
            (result.run_dir / "junction_label_schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(list(JUNCTION_CLASSES), [row["name"] for row in schema["classes"]])
        self.assertEqual("UNKNOWN", schema["cross_policy"]["automatic_label"])
        self.assertTrue(schema["cross_policy"]["needs_review"])
        self.assertTrue((self.root / "output" / "latest.json").is_file())

    def test_dense_coordinates_are_lexically_discarded_and_outputs_are_deterministic(self) -> None:
        annotation, stats = _load_annotation_without_dense_coordinates(
            self.fixture["source_root"] / "s1" / "source.json"
        )
        self.assertIsNone(annotation["edges"]["E_1"]["coordinates"])
        self.assertEqual(1, stats["fields"])
        self.assertGreater(stats["bytes"], 0)

        first = materialize_wire_junction(
            **self.fixture,
            output_root=self.root / "deterministic-a",
            config=WireJunctionMaterializeConfig(tile_size=16),
        )
        second = materialize_wire_junction(
            **self.fixture,
            output_root=self.root / "deterministic-b",
            config=WireJunctionMaterializeConfig(tile_size=16),
        )
        self.assertEqual(first.run_id, second.run_id)
        self.assertEqual(first.summary, second.summary)
        self.assertEqual(
            _sha256(first.run_dir / "wire_tiles.jsonl"),
            _sha256(second.run_dir / "wire_tiles.jsonl"),
        )
        self.assertEqual(
            first.summary["assets"]["tree_sha256"],
            second.summary["assets"]["tree_sha256"],
        )
        self.assertFalse(first.summary["sources"][0]["dense_coordinates_loaded"])

        # Positive is an explicit, hash-bound alternative to the safe default.
        positive = materialize_wire_junction(
            **self.fixture,
            output_root=self.root / "positive",
            config=WireJunctionMaterializeConfig(
                tile_size=16, pin_line_policy="positive"
            ),
        )
        self.assertNotEqual(first.run_id, positive.run_id)
        tile = next(
            row
            for row in _read_jsonl(positive.run_dir / "wire_tiles.jsonl")
            if row["tile_bbox"] == {"x1": 0, "y1": 0, "x2": 16, "y2": 16}
        )
        with Image.open(positive.run_dir / tile["target_mask"]["path"]) as target:
            self.assertEqual(255, target.getpixel((12, 10)))
        with Image.open(positive.run_dir / tile["ignore_mask"]["path"]) as ignore:
            self.assertEqual(0, ignore.getpixel((12, 10)))

    def test_split_lineage_mismatch_fails_before_publication(self) -> None:
        split_path = self.fixture["split_manifest"]
        split_row = json.loads(split_path.read_text(encoding="utf-8"))
        split_row["split"] = "val"
        split_path.write_text(_canonical(split_row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(MaterializationError, "split"):
            materialize_wire_junction(
                **self.fixture,
                output_root=self.root / "must-not-publish",
            )
        self.assertFalse((self.root / "must-not-publish" / "latest.json").exists())

    @staticmethod
    def _write_fixture(root: Path) -> dict[str, Path]:
        source_root = root / "source"
        sample_root = source_root / "s1"
        sample_root.mkdir(parents=True)
        image_path = sample_root / "source.png"
        image = Image.new("RGB", (20, 20), (255, 255, 255))
        for x in range(20):
            image.putpixel((x, 10), (0, 0, 0))
        image.save(image_path, format="PNG")

        skeleton = [[x, 10] for x in range(20)]
        annotation = {
            "canvas": {"width": 20, "height": 20},
            "components": {
                "C_1": {
                    "state": "active",
                    "location": {"x1": 7, "y1": 9, "x2": 9, "y2": 12},
                }
            },
            "texts": {
                "T_1": {
                    "state": "active",
                    "location": {"x1": 3, "y1": 9, "x2": 5, "y2": 12},
                }
            },
            "pins": {"P_1": {"tip": [12, 8], "base": [12, 12]}},
            "edges": {
                "E_1": {
                    "edgeId": "RAW_E_1",
                    "state": "active",
                    "category": "wire",
                    "keyPointPaths": {"wire": [[[0, 10], [19, 10]]]},
                    "keyPoints": {
                        "skeleton": {"wire": skeleton},
                        "endPoints": {"wire": [[0, 10], [19, 10]]},
                        "divergePoints": {"wire": [[10, 10]]},
                        "turnPoints": {"wire": [[5, 10]]},
                    },
                    # This value is never decoded by the materializer.
                    "coordinates": [[x, 10] for x in range(20)],
                }
            },
        }
        json_path = sample_root / "source.json"
        json_path.write_text(_canonical(annotation), encoding="utf-8")
        config_hash = "a" * 64
        common = {
            "schema_version": 1,
            "derivation_version": "fixture-v2",
            "config_hash": config_hash,
            "sample_id": "s1",
            "source_domain": "kicad",
            "source_json": "s1/source.json",
            "source_image": "s1/source.png",
            "duplicate_group": "sample-s1",
            "split": "train",
        }
        wire = {
            **common,
            "task": "wire_paths",
            "record_id": "s1:edge:E_1",
            "edge_key": "E_1",
            "edge_id": "RAW_E_1",
            "category": "wire",
            "key_point_paths_ref": {
                "json_pointer": "/edges/E_1/keyPointPaths",
                "geometry": {
                    "wire": {"path_count": 1, "point_count": 2, "valid": True}
                },
            },
            "skeleton_ref": {
                "json_pointer": "/edges/E_1/keyPoints/skeleton",
                "geometry": {"wire": {"point_count": 20, "valid": True}},
            },
        }
        wire_manifest = root / "wire_paths.jsonl"
        wire_manifest.write_text(_canonical(wire) + "\n", encoding="utf-8")

        junction_specs = (
            ("endpoint", [0, 10], "/edges/E_1/keyPoints/endPoints/wire/0"),
            ("diverge", [10, 10], "/edges/E_1/keyPoints/divergePoints/wire/0"),
            ("turn", [5, 10], "/edges/E_1/keyPoints/turnPoints/wire/0"),
            ("cross", [15, 10], "/edges/E_1/keyPoints/skeleton"),
        )
        junctions = []
        for kind, center, pointer in junction_specs:
            raw_bbox, clipped_bbox, padding = _patch_geometry(center, 8, 20, 20)
            junctions.append(
                {
                    **common,
                    "task": "junction_patches",
                    "record_id": f"s1:junction:{kind}",
                    "candidate_types": [kind],
                    "center": center,
                    "patch_size": 8,
                    "raw_patch_bbox": raw_bbox,
                    "patch_bbox": clipped_bbox,
                    "padding": padding,
                    "source_refs": [
                        {"kind": kind, "edge_key": "E_1", "json_pointer": pointer}
                    ],
                }
            )
        junction_manifest = root / "junction_patches.jsonl"
        junction_manifest.write_text(
            "".join(_canonical(row) + "\n" for row in junctions),
            encoding="utf-8",
        )

        source = {
            "schema_version": 1,
            "sample_id": "s1",
            "source_class": "kicad",
            "duplicate_group": "sample-s1",
            "canvas": {"width": 20, "height": 20},
            "files": {
                "json": {
                    "path": "s1/source.json",
                    "bytes": json_path.stat().st_size,
                    "sha256": _sha256(json_path),
                },
                "image": {
                    "path": "s1/source.png",
                    "bytes": image_path.stat().st_size,
                    "sha256": _sha256(image_path),
                },
            },
        }
        source_manifest = root / "source.jsonl"
        source_manifest.write_text(_canonical(source) + "\n", encoding="utf-8")
        split_manifest = root / "split.jsonl"
        split_manifest.write_text(
            _canonical({**source, "split": "train"}) + "\n", encoding="utf-8"
        )
        return {
            "wire_manifest": wire_manifest,
            "junction_manifest": junction_manifest,
            "source_manifest": source_manifest,
            "split_manifest": split_manifest,
            "source_root": source_root,
        }


class OfficialDatasetWireJunctionSmokeTests(unittest.TestCase):
    def test_sample_0001_resolves_real_lazy_geometry_without_dense_coordinates(self) -> None:
        required = (
            REPOSITORY_ROOT / "data/derived/dataset300-v2/wire_paths.jsonl",
            REPOSITORY_ROOT / "data/derived/dataset300-v2/junction_patches.jsonl",
            REPOSITORY_ROOT / "data/manifests/dataset300.jsonl",
            REPOSITORY_ROOT / "data/splits/dataset300_split.jsonl",
            REPOSITORY_ROOT / "Dataset300/Dataset300/0001/0001-KiCad_CVJsonStd.json",
        )
        if not all(path.is_file() for path in required):
            self.skipTest("official Dataset300-v2 fixture is not present")
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = materialize_wire_junction(
                wire_manifest=required[0],
                junction_manifest=required[1],
                source_manifest=required[2],
                split_manifest=required[3],
                source_root=REPOSITORY_ROOT / "Dataset300/Dataset300",
                output_root=Path(temporary_directory) / "output",
                sample_ids=("0001",),
                expected_derivation_version="dataset300-v2",
                expected_config_hash=(
                    "3381e705c06642cacf8d3e9d82afbf2f725d3e925b67b049a57bd9b858632009"
                ),
            )
            self.assertTrue(result.summary["materialization"]["ok"])
            self.assertEqual(9, result.summary["materialization"]["wire_tile_count"])
            self.assertEqual(340, result.summary["materialization"]["junction_annotation_count"])
            source = result.summary["sources"][0]
            self.assertFalse(source["dense_coordinates_loaded"])
            self.assertEqual(141, source["dense_coordinate_fields_stripped"])
            self.assertGreater(source["dense_coordinate_source_bytes_skipped"], 300_000)
            self.assertEqual(
                ["connected_cross", "non_connected_cross", "noise"],
                result.summary["training_gate"]["missing_classes_global"],
            )
            self.assertEqual(
                ["0001:edge:E_141"],
                result.summary["wire"]["unusable_sparse_geometry_record_ids"],
            )
            tile_rows = _read_jsonl(result.run_dir / "wire_tiles.jsonl")
            self.assertTrue(all(not row["supervision"]["eligible"] for row in tile_rows))
            self.assertTrue(
                all(
                    row["ignore_mask"]["foreground_pixels"]
                    == row["tile_size"][0] * row["tile_size"][1]
                    for row in tile_rows
                )
            )


def _patch_geometry(
    center: list[int], patch_size: int, width: int, height: int
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    half = patch_size // 2
    raw = {
        "x1": center[0] - half,
        "y1": center[1] - half,
        "x2": center[0] - half + patch_size,
        "y2": center[1] - half + patch_size,
    }
    clipped = {
        "x1": max(0, raw["x1"]),
        "y1": max(0, raw["y1"]),
        "x2": min(width, raw["x2"]),
        "y2": min(height, raw["y2"]),
    }
    padding = {
        "left": clipped["x1"] - raw["x1"],
        "top": clipped["y1"] - raw["y1"],
        "right": raw["x2"] - clipped["x2"],
        "bottom": raw["y2"] - clipped["y2"],
    }
    return raw, clipped, padding


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
