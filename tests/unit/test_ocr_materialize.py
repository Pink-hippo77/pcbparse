from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pcbparse.training.ocr_materialize import (
    OCRMaterializationError,
    OCRMaterializationOptions,
    materialize_ocr_training_data,
)


class OCRMaterializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.fixture = _make_fixture(self.root)

    def test_materializes_strict_recognition_and_tiled_detection_labels(self) -> None:
        output = self.root / "materialized"

        manifest = materialize_ocr_training_data(
            **self.fixture,
            output_root=output,
            options=OCRMaterializationOptions(materialize_images=True),
        )

        self.assertTrue(manifest["complete"])
        self.assertEqual(4, manifest["counts"]["recognition_included"])
        self.assertEqual(2, manifest["counts"]["recognition_excluded"])
        self.assertEqual(4, manifest["counts"]["detection_tiles"])
        self.assertEqual(8, manifest["image_set"]["count"])
        self.assertTrue((output / "manifest.json").is_file())

        rec_rows = _read_jsonl(output / "recognition_index.jsonl")
        by_id = {row["record_id"]: row for row in rec_rows}
        self.assertEqual(
            {"left": 0, "top": 0, "right": 13, "bottom": 11},
            by_id["0001:text:T1"]["crop_box"],
        )
        self.assertEqual(
            [1, 0, 0, 0, 1, 0],
            by_id["0001:text:T1"]["transform"]["crop_to_source"],
        )
        self.assertEqual("contains_line_feed", by_id["0001:text:T2"]["exclusion_reason"])
        self.assertEqual("over_max_text_length", by_id["0001:text:T3"]["exclusion_reason"])
        self.assertTrue(by_id["0001:text:T5"]["vertical_long"])
        self.assertEqual("PNG", by_id["0002:text:T1"]["source_format"])
        for row in rec_rows:
            if row["status"] == "included":
                path = output / row["output_image"]
                self.assertTrue(path.is_file())
                self.assertEqual(_sha(path), row["output_image_sha256"])

        rec_train = (output / "rec_train.txt").read_text(encoding="utf-8")
        self.assertIn("\t R惟 \n", rec_train)
        self.assertNotIn("bad\nline", rec_train)
        self.assertNotIn("x" * 65, rec_train)

        det_rows = _read_jsonl(output / "detection_index.jsonl")
        first_train = next(
            row
            for row in det_rows
            if row["sample_id"] == "0001" and row["source_box"]["left"] == 0
        )
        annotations = {item["record_id"]: item for item in first_train["annotations"]}
        self.assertEqual(
            [[0.2, 0.2], [10.2, 0.2], [10.2, 8.2], [0.2, 8.2]],
            annotations["0001:text:T1"]["points"],
        )
        self.assertEqual("###", annotations["0001:text:T2"]["transcription"])
        self.assertEqual("contains_line_feed", annotations["0001:text:T2"]["ignore_reason"])
        self.assertEqual("###", annotations["0001:text:T3"]["transcription"])
        self.assertEqual("partial_intersection", annotations["0001:text:T3"]["ignore_reason"])
        self.assertEqual("###", annotations["0001:text:T4"]["transcription"])
        self.assertEqual("partial_intersection", annotations["0001:text:T4"]["ignore_reason"])
        self.assertEqual([1024, 1024], first_train["output_size"])
        test_tile = next(row for row in det_rows if row["sample_id"] == "0003")
        self.assertTrue(test_tile["empty_tile"])
        test_label = (output / "det_test.txt").read_text(encoding="utf-8")
        self.assertTrue(test_label.rstrip("\n").endswith("\t[]"))

        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(1, report["detection"]["oversize_count"])
        self.assertEqual("0001:text:T4", report["detection"]["oversize_records"][0]["record_id"])
        self.assertIn("Ω", report["charset"]["oov_by_split"]["val"]["characters"])
        self.assertEqual(1, report["recognition"]["vertical_long_included"])
        self.assertEqual("exclude_not_truncate", report["policy"]["overlength"])

    def test_index_only_is_reproducible_and_sample_limited(self) -> None:
        first = self.root / "index-first"
        second = self.root / "index-second"
        options = OCRMaterializationOptions(
            sample_limit=1,
            materialize_images=False,
            verify_all_source_hashes=True,
        )

        one = materialize_ocr_training_data(
            **self.fixture, output_root=first, options=options
        )
        two = materialize_ocr_training_data(
            **self.fixture, output_root=second, options=options
        )

        self.assertEqual(one, two)
        self.assertEqual(3, one["counts"]["recognition_included"])
        self.assertEqual(0, one["image_set"]["count"])
        self.assertFalse(any(first.rglob("*.png")))
        self.assertEqual(
            _sha(first / "recognition_index.jsonl"),
            _sha(second / "recognition_index.jsonl"),
        )
        report = json.loads((first / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(["0001"], report["selection"]["selected_sample_ids"])
        self.assertEqual("", (first / "rec_val.txt").read_text(encoding="utf-8"))

    def test_source_tampering_and_split_lineage_fail_closed(self) -> None:
        source_image = self.root / "source" / "0001" / "diagram.png"
        source_image.write_bytes(b"tampered")
        with self.assertRaisesRegex(OCRMaterializationError, "SHA-256 mismatch"):
            materialize_ocr_training_data(
                **self.fixture,
                output_root=self.root / "tampered",
                options=OCRMaterializationOptions(materialize_images=False),
            )

        # Restore the fixture and alter the split assignment while keeping the
        # derived summary binding stale. The manifest hash gate must fire first.
        self.fixture = _make_fixture(self.root / "fresh")
        split_path = Path(self.fixture["split_manifest"])
        rows = _read_jsonl(split_path)
        rows[0]["split"] = "val"
        _write_jsonl(split_path, rows)
        with self.assertRaisesRegex(OCRMaterializationError, "split manifest SHA-256"):
            materialize_ocr_training_data(
                **self.fixture,
                output_root=self.root / "split-mismatch",
                options=OCRMaterializationOptions(materialize_images=False),
            )


def _make_fixture(root: Path) -> dict[str, Path]:
    source_root = root / "source"
    derived_root = root / "derived"
    source_root.mkdir(parents=True, exist_ok=True)
    derived_root.mkdir(parents=True, exist_ok=True)
    specifications = (
        ("0001", "train", (1200, 100), "diagram.png", "kicad"),
        # The suffix is intentionally wrong: decoding must use image magic.
        ("0002", "val", (80, 50), "diagram.jpg", "datasheet"),
        ("0003", "test", (80, 50), "diagram.webp", "jlc"),
    )
    source_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    for sample_id, split, size, filename, source_class in specifications:
        sample_root = source_root / sample_id
        sample_root.mkdir()
        image_path = sample_root / filename
        Image.new("RGB", size, (255, 255, 255)).save(image_path, format="PNG")
        relative = f"{sample_id}/{filename}"
        duplicate_group = f"group-{sample_id}"
        source_rows.append(
            {
                "schema_version": 1,
                "sample_id": sample_id,
                "source_class": source_class,
                "duplicate_group": duplicate_group,
                "canvas": {"width": size[0], "height": size[1]},
                "files": {
                    "image": {
                        "path": relative,
                        "sha256": _sha(image_path),
                        "width": size[0],
                        "height": size[1],
                        "detected_format": "PNG",
                    }
                },
            }
        )
        split_rows.append(
            {
                "sample_id": sample_id,
                "duplicate_group": duplicate_group,
                "split": split,
            }
        )
    source_manifest = root / "source.jsonl"
    split_manifest = root / "split.jsonl"
    _write_jsonl(source_manifest, source_rows)
    _write_jsonl(split_manifest, split_rows)

    config_hash = "a" * 64
    common = {
        "schema_version": 1,
        "task": "ocr_text",
        "derivation_version": "fixture-ocr-v1",
        "config_hash": config_hash,
        "source_domain": "kicad",
        "source_image": "0001/diagram.png",
        "duplicate_group": "group-0001",
        "sample_id": "0001",
        "split": "train",
        "state": "active",
        "category": "params",
        "details": "value",
        "normalization_applied": False,
    }
    ocr_rows = [
        {
            **common,
            "record_id": "0001:text:T1",
            "content": " R惟 ",
            "bbox": {"x1": 0.2, "y1": 0.2, "x2": 10.2, "y2": 8.2},
        },
        {
            **common,
            "record_id": "0001:text:T2",
            "content": "bad\nline",
            "bbox": {"x1": 900, "y1": 10, "x2": 950, "y2": 20},
        },
        {
            **common,
            "record_id": "0001:text:T3",
            "content": "x" * 65,
            "bbox": {"x1": 1000, "y1": 10, "x2": 1100, "y2": 20},
        },
        {
            **common,
            "record_id": "0001:text:T4",
            "content": "wide",
            "bbox": {"x1": 10, "y1": 30, "x2": 1100, "y2": 40},
        },
        {
            **common,
            "record_id": "0001:text:T5",
            "content": "V",
            "bbox": {"x1": 50, "y1": 10, "x2": 55, "y2": 90},
        },
        {
            **common,
            "record_id": "0002:text:T1",
            "sample_id": "0002",
            "split": "val",
            "source_domain": "datasheet",
            "source_image": "0002/diagram.jpg",
            "duplicate_group": "group-0002",
            "content": "Ω",
            "bbox": {"x1": 10, "y1": 10, "x2": 20, "y2": 20},
        },
    ]
    ocr_manifest = derived_root / "ocr_text.jsonl"
    _write_jsonl(ocr_manifest, ocr_rows)
    summary = {
        "schema_version": 1,
        "derivation_version": "fixture-ocr-v1",
        "config_hash": config_hash,
        "sample_count": 3,
        "counts": {"ocr_text": len(ocr_rows)},
        "task_files": {"ocr_text": "ocr_text.jsonl"},
        "config": {"split_manifest_sha256": _sha(split_manifest)},
    }
    summary_path = derived_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return {
        "ocr_manifest": ocr_manifest,
        "derived_summary": summary_path,
        "source_root": source_root,
        "source_manifest": source_manifest,
        "split_manifest": split_manifest,
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
