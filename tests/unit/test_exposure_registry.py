from __future__ import annotations

import json
import shutil
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

from data_tools.exposure_registry import (  # noqa: E402
    HISTORICAL_EXPOSURE_EVENTS,
    build_registry,
    classify_label_payload,
    inspect_cvjsonstd_fields,
    inspect_target_fields,
    pixel_hash_image,
    write_outputs,
)
from pcbparse.data import read_jsonl, write_jsonl  # noqa: E402


def _write_image(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (16, 12)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path, format="PNG")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _cvjsonstd(token: str, *, named: bool = True, numbered: bool = False) -> dict:
    return {
        "canvas": {"width": 16, "height": 12},
        "nowCls": "datasheet",
        "components": {
            f"C_{token}": {
                "cpntId": f"C_{token}",
                "category": "R",
                "searchNameByOCR": f"R{token}" if named else None,
                "searchNameByVLM": None,
                "searchNameByLib": None,
                "location": {"x1": 1, "y1": 1, "x2": 8, "y2": 8},
                "realBBox": {"x1": 1, "y1": 1, "x2": 8, "y2": 8},
                "pins": {f"P_{token}": {"pinId": f"P_{token}"}},
            }
        },
        "pins": {
            f"P_{token}": {
                "pinId": f"P_{token}",
                "edgeId": f"E_{token}",
                "pinNumberByOCR": f"{token}" if numbered else None,
                "pinNumberByVLM": None,
                "pinNameByOCR": None,
                "tip": [8.0, 4.0],
                "base": [1.0, 4.0],
            }
        },
        "texts": {f"T_{token}": {"tid": f"T_{token}"}},
        "edges": {f"E_{token}": {"edgeId": f"E_{token}"}},
    }


def _target_named(token: str) -> dict:
    return {
        "components": {
            f"R{token}": {
                "Name": f"R{token}",
                "type": "r",
                "value": "10k",
                "bbox": [1.0, 1.0, 8.0, 8.0],
            }
        },
        "pins": {f"P_{token}": {"number": "1", "name": "A", "owner": f"R{token}"}},
        "nets": {},
    }


class ExposureRegistryFieldTests(unittest.TestCase):
    def test_cvjsonstd_missing_designator_and_pin_number_are_ambiguous(self) -> None:
        present, usable, ambiguous = inspect_cvjsonstd_fields(
            _cvjsonstd("1", named=False, numbered=False)
        )
        self.assertIn("components", usable)
        self.assertIn("component_designator", ambiguous)
        self.assertIn("pin_number", ambiguous)
        self.assertIn("component_designator_missing_all", ambiguous)
        self.assertIn("pin_number_missing_all", ambiguous)

    def test_cvjsonstd_complete_fields_are_usable(self) -> None:
        present, usable, ambiguous = inspect_cvjsonstd_fields(
            _cvjsonstd("1", named=True, numbered=True)
        )
        self.assertIn("component_designator", usable)
        self.assertIn("pin_number", usable)
        self.assertNotIn("component_designator_missing_all", ambiguous)

    def test_target_named_marks_designators_usable(self) -> None:
        present, usable, ambiguous = inspect_target_fields(_target_named("1"), "target_named")
        self.assertIn("component_designator", usable)
        self.assertIn("component_name", usable)
        self.assertIn("component_value", usable)
        self.assertIn("pin_number", usable)

    def test_classify_payload_dispatches(self) -> None:
        present, usable, ambiguous = classify_label_payload("CVJsonStd", _cvjsonstd("2"))
        self.assertIn("components", present)
        present2, usable2, ambiguous2 = classify_label_payload("target", _target_named("2"))
        self.assertIn("gtcase_target_not_training_gt", ambiguous2)

    def test_pixel_hash_is_deterministic_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.png"
            b = root / "b.png"
            c = root / "c.png"
            _write_image(a, (10, 20, 30))
            _write_image(b, (10, 20, 30))
            _write_image(c, (11, 20, 30))
            ha, wa, hha = pixel_hash_image(a)
            hb, _, _ = pixel_hash_image(b)
            hc, _, _ = pixel_hash_image(c)
            self.assertEqual(ha, hb)
            self.assertNotEqual(ha, hc)
            self.assertEqual((wa, hha), (16, 12))


class ExposureRegistryBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset300 = self.root / "Dataset300" / "Dataset300"
        self.public = self.root / "public"
        self.work = self.root / "work"
        self.configs = self.root / "configs"
        self.dataset300.mkdir(parents=True)
        self.public.mkdir()
        self.work.mkdir()
        self.configs.mkdir()

        # Three distinct images. Public cases byte-match 0001 and 0002.
        self._make_d300_sample("0001", (10, 20, 30), source_class="datasheet")
        self._make_d300_sample("0002", (200, 10, 10), source_class="ad")
        self._make_d300_sample("0003", (5, 5, 200), source_class="datasheet")

        self._make_public_case("200_train_cases", "0001-other", (10, 20, 30), "target_named")
        self._make_public_case("10_GTcase", "0002-Altium Designer", (200, 10, 10), "target")

        self.split_manifest = self.root / "split.jsonl"
        self.release50_manifest = self.root / "release50.jsonl"
        split_rows = [
            {
                "sample_id": "0001",
                "split": "test",
                "source_class": "datasheet",
                "source_family": "0001-fixture",
                "base_name": "0001-fixture",
                "files": {
                    "image": {
                        "sha256": self._sha_of(self.dataset300 / "0001" / "0001-fixture.png")
                    }
                },
            },
            {
                "sample_id": "0002",
                "split": "val",
                "source_class": "ad",
                "source_family": "0002-fixture",
                "base_name": "0002-fixture",
                "files": {
                    "image": {
                        "sha256": self._sha_of(self.dataset300 / "0002" / "0002-fixture.png")
                    }
                },
            },
            {
                "sample_id": "0003",
                "split": "train",
                "source_class": "datasheet",
                "source_family": "0003-fixture",
                "base_name": "0003-fixture",
                "files": {
                    "image": {
                        "sha256": self._sha_of(self.dataset300 / "0003" / "0003-fixture.png")
                    }
                },
            },
        ]
        write_jsonl(split_rows, self.split_manifest)
        write_jsonl(
            [
                {
                    "sample_id": "0001",
                    "split": "release50",
                    "release_source_split": "test",
                    "base_name": "0001-fixture",
                    "files": {"image": {"sha256": split_rows[0]["files"]["image"]["sha256"]}},
                },
                {
                    "sample_id": "0002",
                    "split": "release50",
                    "release_source_split": "val",
                    "base_name": "0002-fixture",
                    "files": {"image": {"sha256": split_rows[1]["files"]["image"]["sha256"]}},
                },
            ],
            self.release50_manifest,
        )

        run = self.work / "eval-component-val"
        run.mkdir()
        write_jsonl([{"sample_id": "0001"}, {"sample_id": "0002"}], run / "predictions.jsonl")

        _write_json(self.configs / "train" / "tiny.json", {"junction_synthetic_per_class": 3})

        self.out_identity = self.root / "dataset_identity.jsonl"
        self.out_exposure = self.root / "split_exposure.json"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _sha_of(self, path: Path) -> str:
        import hashlib

        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _make_d300_sample(self, sample_id: str, color: tuple[int, int, int], *, source_class: str) -> None:
        directory = self.dataset300 / sample_id
        base = f"{sample_id}-fixture"
        _write_image(directory / f"{base}.png", color)
        _write_json(directory / f"{base}_CVJsonStd.json", _cvjsonstd(sample_id, named=True, numbered=False))
        (directory / f"{base}_CVJsonStd.eds").write_text("eds", encoding="utf-8")
        (directory / f"{base}_spice.net").write_text(".end\n", encoding="utf-8")

    def _make_public_case(
        self, subset: str, case_name: str, color: tuple[int, int, int], label_version: str
    ) -> None:
        directory = self.public / subset / case_name
        sample_id = case_name[:4]
        _write_image(directory / f"{case_name}.png", color)
        if label_version == "target_named":
            _write_json(directory / f"{case_name}_target_named.json", _target_named(sample_id))
        else:
            _write_json(directory / f"{case_name}_target.json", _target_named(sample_id))

    def _build(self, *, near_duplicate_distance: int = -1):
        return build_registry(
            dataset300_root=self.dataset300,
            public_root=self.public,
            split_manifest=self.split_manifest,
            release50_manifest=self.release50_manifest,
            work_root=self.work,
            configs_root=self.configs,
            near_duplicate_distance=near_duplicate_distance,
            compute_pixel_hashes=True,
        )

    def test_exact_duplicates_merge_into_one_image_entity(self) -> None:
        records, doc = self._build()
        self.assertEqual(doc["counts"]["unique_images"], 3)
        matching = [r for r in records if r.get("public_duplicate_of_dataset300")]
        self.assertEqual(len(matching), 2)
        multi = [r for r in records if len(r["label_version_names"]) >= 2]
        self.assertEqual(len(multi), 2)
        version_sets = {tuple(r["label_version_names"]) for r in multi}
        self.assertIn(("CVJsonStd", "target_named"), version_sets)
        self.assertIn(("CVJsonStd", "target"), version_sets)

    def test_exact_duplicate_samples_merge_within_split(self) -> None:
        src = self.dataset300 / "0001"
        dst = self.dataset300 / "0004"
        shutil.copytree(src, dst)
        for path in list(dst.iterdir()):
            path.rename(dst / path.name.replace("0001", "0004"))
        rows = read_jsonl(self.split_manifest)
        rows.append(
            {
                "sample_id": "0004",
                "split": "test",
                "source_class": "datasheet",
                "source_family": "0004-fixture",
                "base_name": "0004-fixture",
                "files": {
                    "image": {
                        "sha256": self._sha_of(self.dataset300 / "0004" / "0004-fixture.png")
                    }
                },
            }
        )
        write_jsonl(rows, self.split_manifest)
        records, doc = self._build()
        self.assertEqual(doc["counts"]["unique_images"], 3)
        merged = [r for r in records if {"0001", "0004"}.issubset(set(r["sample_ids"]))]
        self.assertEqual(len(merged), 1)
        self.assertEqual(doc["constraints"]["same_family_no_cross_split"]["status"], "passed")

    def test_public_210_not_new_holdout(self) -> None:
        records, doc = self._build()
        self.assertEqual(doc["constraints"]["public_210_not_new_holdout"]["status"], "passed")
        self.assertFalse(doc["independence_claims"]["independent_test"])
        for record in records:
            self.assertFalse(record["independent_holdout_eligible"])
            if set(record["datasets"]) & {"public_200_train", "public_10_gtcase"}:
                self.assertTrue(record["public_duplicate_of_dataset300"])
                self.assertIn("hist-no-independent-holdout-claim", record["exposure_events"])

    def test_historical_exposure_is_preserved(self) -> None:
        records, doc = self._build()
        self.assertEqual(doc["constraints"]["historical_exposure_preserved"]["status"], "passed")
        ids = {e["event_id"] for e in doc["exposure_events"]}
        for event in HISTORICAL_EXPOSURE_EVENTS:
            self.assertIn(event["event_id"], ids)
            self.assertFalse(event["independence_claimed"])

    def test_release50_and_test_are_exposed_without_independence(self) -> None:
        records, doc = self._build()
        exposed_test = [r for r in records if r["split_role"] == "test" and r.get("in_release50")]
        self.assertEqual(len(exposed_test), 1)
        for record in exposed_test:
            self.assertTrue(record["exposed"])
            self.assertFalse(record["independent_holdout_eligible"])
            self.assertIn("hist-release50-construction", record["exposure_events"])
            self.assertIn("hist-release50-parameter-selection", record["exposure_events"])

    def test_same_family_no_cross_split(self) -> None:
        records, doc = self._build()
        self.assertEqual(doc["constraints"]["same_family_no_cross_split"]["status"], "passed")

    def test_family_no_cross_split_detects_leak(self) -> None:
        rows = read_jsonl(self.split_manifest)
        for row in rows:
            if row["sample_id"] in {"0001", "0003"}:
                row["source_family"] = "leaked-family"
        write_jsonl(rows, self.split_manifest)
        records, doc = self._build()
        self.assertEqual(doc["constraints"]["same_family_no_cross_split"]["status"], "failed")
        self.assertGreaterEqual(len(doc["constraints"]["same_family_no_cross_split"]["violations"]), 1)

    def test_pixel_duplicates_group_into_family(self) -> None:
        src = self.dataset300 / "0001" / "0001-fixture.png"
        dst_dir = self.dataset300 / "0004"
        dst_dir.mkdir()
        # Distinct PNG bytes, identical decoded pixels.
        Image.open(src).convert("RGB").save(
            dst_dir / "0004-fixture.png", format="PNG", compress_level=0
        )
        _write_json(dst_dir / "0004-fixture_CVJsonStd.json", _cvjsonstd("4"))
        (dst_dir / "0004-fixture_CVJsonStd.eds").write_text("eds", encoding="utf-8")
        (dst_dir / "0004-fixture_spice.net").write_text(".end\n", encoding="utf-8")
        rows = read_jsonl(self.split_manifest)
        rows.append(
            {
                "sample_id": "0004",
                "split": "test",
                "source_class": "datasheet",
                "source_family": "0004-fixture",
                "base_name": "0004-fixture",
                "files": {"image": {"sha256": self._sha_of(dst_dir / "0004-fixture.png")}},
            }
        )
        write_jsonl(rows, self.split_manifest)
        # Disable perceptual near-dup so only decoded pixel_hash links the pair.
        records, doc = self._build(near_duplicate_distance=-1)
        self.assertEqual(doc["counts"]["unique_images"], 4)
        by_sample_family = {}
        for record in records:
            for sample_id in record["sample_ids"]:
                by_sample_family[sample_id] = record["family_id"]
        self.assertEqual(by_sample_family.get("0001"), by_sample_family.get("0004"))
        self.assertEqual(doc["constraints"]["same_family_no_cross_split"]["status"], "passed")

    def test_teacher_exposure_registered_without_independence(self) -> None:
        records, doc = self._build()
        teacher_events = [e for e in doc["exposure_events"] if e["kind"] == "teacher_generation"]
        self.assertGreaterEqual(len(teacher_events), 1)
        for event in teacher_events:
            self.assertFalse(event["independence_claimed"])
        self.assertIn(
            "teacher-synthetic-junction-train-augmentation",
            {e["event_id"] for e in teacher_events},
        )

    def test_work_exposure_scanned(self) -> None:
        records, doc = self._build()
        work_events = [e for e in doc["exposure_events"] if e.get("source") == "work_scan"]
        self.assertGreaterEqual(len(work_events), 1)
        self.assertTrue(any(e["event_id"] == "work-eval-component-val" for e in work_events))

    def test_outputs_written_and_readable(self) -> None:
        records, doc = self._build()
        write_outputs(records, doc, self.out_identity, self.out_exposure)
        loaded = read_jsonl(self.out_identity)
        self.assertEqual(len(loaded), len(records))
        payload = json.loads(self.out_exposure.read_text(encoding="utf-8"))
        self.assertEqual(payload["kind"], "split_exposure_registry")
        self.assertEqual(payload["task_id"], "A02")
        for record in loaded:
            self.assertTrue(record["image_id"].startswith("img-"))
            self.assertTrue(record["sample_id"])
            self.assertTrue(record["sha256"])
            self.assertIn("label_versions", record)
            self.assertIn("exposure_events", record)
            self.assertIn("usable_fields", record)
            self.assertIn("ambiguous_fields", record)
            self.assertIn("pixel_hash", record)
            self.assertIn("family_id", record)

    def test_original_gt_not_rewritten(self) -> None:
        before = {str(path): path.read_bytes() for path in self.dataset300.rglob("*_CVJsonStd.json")}
        self._build()
        after = {str(path): path.read_bytes() for path in self.dataset300.rglob("*_CVJsonStd.json")}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
