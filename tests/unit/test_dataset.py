from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from data_tools.audit import main as audit_main
from data_tools.pcbnet_ingest import main as ingest_main
from data_tools.split_grouped import main as split_main
from pcbparse.data import (
    assign_grouped_splits,
    audit_dataset,
    read_jsonl,
    validate_grouped_splits,
    write_jsonl,
)
from pcbparse.data.hashing import apply_duplicate_groups


class DatasetFoundationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "Dataset300"
        self.root.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _create_sample(
        self,
        sample_id: str,
        *,
        annotation_token: str | None = None,
        source_class: str = "datasheet",
        image_extension: str = ".png",
        image_format: str = "PNG",
        image_size: tuple[int, int] = (12, 8),
        image_copy: Path | None = None,
    ) -> dict[str, Path]:
        directory = self.root / sample_id
        directory.mkdir()
        base_name = f"{sample_id}-fixture"
        image_path = directory / f"{base_name}{image_extension}"
        if image_copy is None:
            color_offset = int(sample_id) % 200
            image = Image.new(
                "RGB",
                image_size,
                color=(20 + color_offset, 80, 140),
            )
            image.save(image_path, format=image_format)
        else:
            shutil.copyfile(image_copy, image_path)

        token = annotation_token if annotation_token is not None else sample_id
        annotation = {
            "canvas": {"width": image_size[0], "height": image_size[1]},
            "nowCls": source_class,
            "components": {f"C_{token}": {"cpntId": f"C_{token}"}},
            "pins": {f"P_{token}": {"pinId": f"P_{token}"}},
            "texts": {f"T_{token}": {"tid": f"T_{token}"}},
            "edges": {f"E_{token}": {"edgeId": f"E_{token}"}},
        }
        annotation_path = directory / f"{base_name}_CVJsonStd.json"
        annotation_path.write_text(
            json.dumps(
                annotation,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        eds_path = directory / f"{base_name}_CVJsonStd.eds"
        eds_path.write_text(
            f'(edif "fixture ({sample_id})" (library LIB))\n', encoding="utf-8"
        )
        net_path = directory / f"{base_name}_spice.net"
        net_path.write_text(".title fixture\nR1 1 0 1k\n.end\n", encoding="utf-8")
        return {
            "image": image_path,
            "json": annotation_path,
            "eds": eds_path,
            "net": net_path,
        }

    def test_audit_discovers_samples_and_uses_image_magic(self) -> None:
        self._create_sample("0001")
        self._create_sample(
            "0002", image_extension=".png", image_format="JPEG"
        )
        self._create_sample("0003", image_extension=".jpg", image_format="JPEG")
        (self.root / ".DS_Store").write_bytes(b"ignored")
        (self.root / "__MACOSX").mkdir()

        result = audit_dataset(self.root, expected_count=3)

        self.assertTrue(result.ok)
        self.assertFalse(result.strict_ok)
        self.assertEqual(3, len(result.records))
        by_id = {record["sample_id"]: record for record in result.records}
        self.assertEqual("JPEG", by_id["0002"]["files"]["image"]["detected_format"])
        self.assertEqual(".png", by_id["0002"]["files"]["image"]["declared_extension"])
        self.assertEqual(
            1, result.report["summary"]["issues_by_code"]["image_extension_mismatch"]
        )

    def test_audit_reports_missing_file_and_dimension_mismatch(self) -> None:
        first = self._create_sample("0001")
        second = self._create_sample("0002")
        annotation = json.loads(first["json"].read_text(encoding="utf-8"))
        annotation["canvas"]["width"] += 1
        first["json"].write_text(
            json.dumps(annotation, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        second["net"].unlink()

        result = audit_dataset(self.root, expected_count=2)

        self.assertFalse(result.ok)
        self.assertEqual([], result.records)
        issue_codes = {issue["code"] for issue in result.report["issues"]}
        self.assertIn("canvas_image_dimension_mismatch", issue_codes)
        self.assertIn("missing_net_file", issue_codes)
        for issue in result.report["issues"]:
            self.assertTrue(issue["action"])

    def test_duplicate_groups_are_transitive_and_split_safe(self) -> None:
        first = self._create_sample("0001", annotation_token="shared-json")
        second = self._create_sample("0002", annotation_token="shared-json")
        self._create_sample(
            "0003",
            annotation_token="third-json",
            image_copy=second["image"],
        )
        self._create_sample("0004", annotation_token="unique")

        result = audit_dataset(self.root, expected_count=4)

        self.assertTrue(result.ok)
        self.assertEqual(1, len(result.report["duplicate_groups"]))
        self.assertEqual(
            ["0001", "0002", "0003"],
            result.report["duplicate_groups"][0]["sample_ids"],
        )
        first_split, first_summary = assign_grouped_splits(
            result.records,
            ratios={"train": 0.5, "val": 0.25, "test": 0.25},
            seed="fixture-seed",
        )
        second_split, second_summary = assign_grouped_splits(
            list(reversed(result.records)),
            ratios={"train": 0.5, "val": 0.25, "test": 0.25},
            seed="fixture-seed",
        )

        self.assertEqual(first_split, second_split)
        self.assertEqual(first_summary, second_summary)
        validate_grouped_splits(first_split)
        duplicate_splits = {
            record["split"]
            for record in first_split
            if record["sample_id"] in {"0001", "0002", "0003"}
        }
        self.assertEqual(1, len(duplicate_splits))

    def test_perceptual_near_duplicates_are_grouped_without_cross_domain_collision(self) -> None:
        def record(sample_id: str, perceptual_hash: str, source: str) -> dict:
            return {
                "sample_id": sample_id,
                "source_class": source,
                "files": {
                    "json": {"sha256": f"{int(sample_id):064x}"},
                    "image": {
                        "sha256": f"{int(sample_id) + 1000:064x}",
                        "perceptual_hash": perceptual_hash,
                        "perceptual_hash_eligible": True,
                        "width": 1000,
                        "height": 500,
                    },
                },
            }

        records = [
            record("0001", "0" * 64, "datasheet"),
            record("0002", "0" * 63 + "3", "datasheet"),
            record("0003", "0" * 63 + "1", "kicad"),
        ]

        groups = apply_duplicate_groups(records)

        self.assertEqual(1, len(groups))
        self.assertEqual(["0001", "0002"], groups[0]["sample_ids"])
        evidence = [item for item in groups[0]["evidence"] if item["kind"] == "image_perceptual"]
        self.assertEqual(1, len(evidence))
        self.assertEqual(2, evidence[0]["hamming_distance"])

    def test_grouped_split_keeps_small_source_domain_in_each_split(self) -> None:
        def record(sample_id: int, source: str) -> dict:
            token = f"{sample_id:064x}"
            return {
                "sample_id": f"{sample_id:04d}",
                "source_class": source,
                "files": {
                    "json": {"sha256": token},
                    "image": {
                        "sha256": f"{sample_id + 1000:064x}",
                        "perceptual_hash": f"{sample_id + 2000:064x}",
                        "perceptual_hash_eligible": False,
                    },
                },
            }

        records = [record(index, "common") for index in range(1, 25)]
        records.extend(record(100 + index, "rare") for index in range(1, 7))
        split, summary = assign_grouped_splits(records, seed="stratified-fixture")

        rare_splits = Counter(
            item["split"] for item in split if item["source_class"] == "rare"
        )
        self.assertEqual({"train": 4, "val": 1, "test": 1}, dict(rare_splits))
        self.assertEqual("source_class_grouped_hamilton_v2", summary["stratification"])

    def test_high_confidence_source_family_is_grouped(self) -> None:
        def record(sample_id: str, family: str | None) -> dict:
            return {
                "sample_id": sample_id,
                "source_class": "datasheet",
                "source_family": family,
                "files": {
                    "json": {"sha256": f"{int(sample_id):064x}"},
                    "image": {
                        "sha256": f"{int(sample_id) + 1000:064x}",
                        "perceptual_hash_eligible": False,
                    },
                },
            }

        records = [
            record("0208", "006_02"),
            record("0209", "006_02"),
            record("0210", "006_02"),
            record("0211", None),
        ]
        groups = apply_duplicate_groups(records)

        family_group = next(item for item in groups if "0208" in item["sample_ids"])
        self.assertEqual(["0208", "0209", "0210"], family_group["sample_ids"])
        self.assertEqual(
            ["source_family"],
            [item["kind"] for item in family_group["evidence"]],
        )

    def test_manifest_round_trip_is_canonical(self) -> None:
        self._create_sample("0001")
        self._create_sample("0002")
        result = audit_dataset(self.root, expected_count=2)
        manifest_path = Path(self.temporary_directory.name) / "manifest.jsonl"

        write_jsonl(result.records, manifest_path)
        first_bytes = manifest_path.read_bytes()
        records = read_jsonl(manifest_path)
        write_jsonl(records, manifest_path)

        self.assertEqual(result.records, records)
        self.assertEqual(first_bytes, manifest_path.read_bytes())

    def test_cli_ingest_audit_and_grouped_split(self) -> None:
        self._create_sample("0001", annotation_token="same")
        self._create_sample("0002", annotation_token="same")
        output_root = Path(self.temporary_directory.name) / "output"
        manifest_path = output_root / "manifest.jsonl"
        report_path = output_root / "audit.json"
        split_path = output_root / "split.jsonl"
        split_summary_path = output_root / "split-summary.json"

        with contextlib.redirect_stdout(io.StringIO()):
            ingest_code = ingest_main(
                [
                    "--dataset-root",
                    str(self.root),
                    "--expected-count",
                    "2",
                    "--output",
                    str(manifest_path),
                    "--report",
                    str(report_path),
                ]
            )
            audit_code = audit_main(
                [
                    "--dataset-root",
                    str(self.root),
                    "--expected-count",
                    "2",
                    "--output",
                    str(output_root / "audit-second.json"),
                ]
            )
            split_code = split_main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(split_path),
                    "--summary",
                    str(split_summary_path),
                    "--train",
                    "0.5",
                    "--val",
                    "0.5",
                    "--test",
                    "0",
                    "--seed",
                    "cli-test",
                ]
            )

        self.assertEqual(0, ingest_code)
        self.assertEqual(0, audit_code)
        self.assertEqual(0, split_code)
        self.assertTrue(report_path.is_file())
        split_records = read_jsonl(split_path)
        validate_grouped_splits(split_records)
        self.assertEqual(1, len({record["split"] for record in split_records}))


if __name__ == "__main__":
    unittest.main()
