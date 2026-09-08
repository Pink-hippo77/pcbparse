from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from data_tools.junction_review_pack import (  # noqa: E402
    CSV_FIELDS,
    JunctionReviewPackError,
    build_review_pack,
    import_human_reviews,
    physical_candidate_id,
    validate_review_file,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_canonical(row) + b"\n" for row in rows))


class JunctionReviewPackTests(unittest.TestCase):
    def test_physical_id_is_stable_under_list_ordering(self) -> None:
        first = {
            "sample_id": "0001",
            "edge_keys": ["E_2", "E_1"],
            "bbox": {"x1": 10, "y1": 11, "x2": 10, "y2": 12},
            "record_ids": ["r2", "r1"],
        }
        second = {
            **first,
            "edge_keys": list(reversed(first["edge_keys"])),
            "record_ids": list(reversed(first["record_ids"])),
        }

        self.assertEqual(physical_candidate_id(first), physical_candidate_id(second))
        self.assertRegex(physical_candidate_id(first), r"^JPC1-[0-9a-f]{24}$")

    def test_build_validate_and_import_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            first = build_review_pack(
                audit_path=fixture["audit"],
                derived_manifest=fixture["derived"],
                source_manifest=fixture["source_manifest"],
                dataset_root=fixture["dataset"],
                output_root=root / "pack-a",
                batch_size=1,
                context_size=96,
            )
            second = build_review_pack(
                audit_path=fixture["audit"],
                derived_manifest=fixture["derived"],
                source_manifest=fixture["source_manifest"],
                dataset_root=fixture["dataset"],
                output_root=root / "pack-b",
                batch_size=1,
                context_size=96,
            )

            self.assertEqual(first.package_id, second.package_id)
            self.assertEqual(3, first.manifest["counts"]["candidate_pixel_records"])
            self.assertEqual(2, first.manifest["counts"]["physical_candidate_clusters"])
            self.assertEqual({"train": 1, "val": 1}, first.manifest["counts"]["by_split"])
            self.assertEqual(
                _sha256(first.run_dir / "review_queue.jsonl"),
                _sha256(second.run_dir / "review_queue.jsonl"),
            )

            queue = [
                json.loads(line)
                for line in (first.run_dir / "review_queue.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            train = next(row for row in queue if row["split"] == "train")
            self.assertEqual(2, train["member_record_count"])
            self.assertEqual("mixed", train["proposal_consensus"])
            self.assertEqual("", train["current_proposal"])
            self.assertFalse(train["proposal_accepted_as_ground_truth"])
            self.assertFalse(train["synthetic_source"])
            for field in ("context_crop_path", "overlay_crop_path", "representative_patch_path"):
                self.assertTrue((first.run_dir / train[field]).is_file())

            pending_csv = first.run_dir / "batches" / "train-0001.review.csv"
            pending = validate_review_file(
                package_dir=first.run_dir,
                reviews_path=pending_csv,
                verify_assets=True,
            )
            self.assertEqual(1, pending.pending)

            incomplete_jsonl_row = dict(train)
            incomplete_jsonl_row.pop("reviewer")
            incomplete_jsonl = root / "incomplete.review.jsonl"
            _write_jsonl(incomplete_jsonl, [incomplete_jsonl_row])
            with self.assertRaisesRegex(JunctionReviewPackError, "JSONL fields differ"):
                validate_review_file(
                    package_dir=first.run_dir,
                    reviews_path=incomplete_jsonl,
                )

            reviewed_csv = root / "train-reviewed.csv"
            rows = self._read_csv(pending_csv)
            # The human label is intentionally different from every proposal;
            # the importer must validate identity, not force a heuristic guess.
            rows[0].update(
                {
                    "review_status": "reviewed",
                    "human_label": "bend",
                    "reviewer": "Alice Chen",
                    "reviewed_at": "2026-08-24T10:30:00+08:00",
                    "review_notes": "visual inspection",
                }
            )
            self._write_csv(reviewed_csv, rows)
            reviewed = validate_review_file(
                package_dir=first.run_dir,
                reviews_path=reviewed_csv,
                require_complete=True,
            )
            self.assertEqual(1, reviewed.reviewed)

            receipt = import_human_reviews(
                package_dir=first.run_dir,
                reviews_path=reviewed_csv,
                output_dir=root / "imported-train",
            )
            self.assertFalse(receipt["existing_truth_modified"])
            imported = json.loads(
                (root / "imported-train" / "junction_human_reviews.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual("bend", imported["label"])
            self.assertFalse(imported["synthetic_source"])
            self.assertFalse(imported["proposal_used_as_ground_truth"])
            self.assertEqual(
                "separate_train_supervision_candidate", imported["training_eligibility"]
            )

            val_csv = first.run_dir / "batches" / "val-0001.review.csv"
            val_rows = self._read_csv(val_csv)
            val_rows[0].update(
                {
                    "review_status": "reviewed",
                    "human_label": "connected_cross",
                    "reviewer": "Bob Li",
                    "reviewed_at": "2026-08-24T11:00:00Z",
                }
            )
            reviewed_val = root / "val-reviewed.csv"
            self._write_csv(reviewed_val, val_rows)
            import_human_reviews(
                package_dir=first.run_dir,
                reviews_path=reviewed_val,
                output_dir=root / "imported-val",
            )
            val_import = json.loads(
                (root / "imported-val" / "junction_human_reviews.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual("never_use_for_training", val_import["training_eligibility"])

            tampered = [dict(rows[0])]
            tampered[0]["center_x"] = "999"
            tampered_path = root / "tampered.csv"
            self._write_csv(tampered_path, tampered)
            with self.assertRaisesRegex(JunctionReviewPackError, "immutable field"):
                validate_review_file(
                    package_dir=first.run_dir,
                    reviews_path=tampered_path,
                    require_complete=True,
                )

            invalid = [dict(rows[0])]
            invalid[0]["human_label"] = "heuristic_guess"
            invalid_path = root / "invalid-label.csv"
            self._write_csv(invalid_path, invalid)
            with self.assertRaisesRegex(JunctionReviewPackError, "invalid human_label"):
                validate_review_file(
                    package_dir=first.run_dir,
                    reviews_path=invalid_path,
                    require_complete=True,
                )

            invalid_time = [dict(rows[0])]
            invalid_time[0]["reviewed_at"] = "2026-08-24T10:30:00"
            invalid_time_path = root / "invalid-time.csv"
            self._write_csv(invalid_time_path, invalid_time)
            with self.assertRaisesRegex(JunctionReviewPackError, "explicit timezone"):
                validate_review_file(
                    package_dir=first.run_dir,
                    reviews_path=invalid_time_path,
                    require_complete=True,
                )

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return [dict(row) for row in csv.DictReader(stream)]

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(CSV_FIELDS), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _fixture(root: Path) -> dict[str, Path]:
        dataset = root / "dataset"
        materialized = root / "materialized"
        source_rows: list[dict] = []
        image_hashes: dict[str, dict[str, str]] = {}
        for sample_id, color in (("0001", "white"), ("0002", "ivory")):
            sample_dir = dataset / sample_id
            sample_dir.mkdir(parents=True)
            image_path = sample_dir / f"{sample_id}.png"
            image = Image.new("RGB", (128, 128), color)
            draw = ImageDraw.Draw(image)
            draw.line((10, 64, 118, 64), fill="black", width=2)
            draw.line((64, 10, 64, 118), fill="black", width=2)
            image.save(image_path)
            image_hash = _sha256(image_path)
            source_rows.append(
                {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "source_class": "kicad",
                    "files": {
                        "image": {
                            "path": f"{sample_id}/{sample_id}.png",
                            "sha256": image_hash,
                        }
                    },
                }
            )
            image_hashes[sample_id] = {"image": image_hash}

        source_manifest = root / "source.jsonl"
        _write_jsonl(source_manifest, source_rows)
        patch_dir = materialized / "junction" / "patches"
        patch_dir.mkdir(parents=True)
        for name in ("r1", "r2", "r3"):
            Image.new("RGB", (64, 64), "white").save(patch_dir / f"{name}.png")

        def candidate(
            record_id: str,
            sample_id: str,
            split: str,
            center: list[int],
            proposal: str,
            confidence: str,
        ) -> dict:
            return {
                "record_id": record_id,
                "sample_id": sample_id,
                "split": split,
                "source_domain": "kicad",
                "center": center,
                "edge_keys": ["E_1", "E_2"],
                "patch_path": f"junction/patches/{record_id}.png",
                "review_proposal": {
                    "proposed_label": proposal,
                    "confidence": confidence,
                    "accepted_as_ground_truth": False,
                    "may_enter_training_without_human_review": False,
                },
            }

        candidates = [
            candidate("r1", "0001", "train", [64, 64], "connected_cross", "high"),
            candidate("r2", "0001", "train", [64, 65], "non_connected_cross", "low"),
            candidate("r3", "0002", "val", [64, 64], "noise", "high"),
        ]
        clusters = [
            {
                "sample_id": "0001",
                "split": "train",
                "source_domain": "kicad",
                "edge_keys": ["E_1", "E_2"],
                "candidate_pixel_count": 2,
                "bbox": {"x1": 64, "y1": 64, "x2": 64, "y2": 65},
                "record_ids": ["r1", "r2"],
            },
            {
                "sample_id": "0002",
                "split": "val",
                "source_domain": "kicad",
                "edge_keys": ["E_1", "E_2"],
                "candidate_pixel_count": 1,
                "bbox": {"x1": 64, "y1": 64, "x2": 64, "y2": 64},
                "record_ids": ["r3"],
            },
        ]
        audit = {
            "schema_version": 1,
            "kind": "junction_truth_audit",
            "scope": {
                "materialized_run_dir": str(materialized.resolve()),
                "materialized_run_id": "fixture-run",
            },
            "lineage": {
                "source_manifest_sha256": _sha256(source_manifest),
                "sample_source_sha256": image_hashes,
            },
            "summary": {
                "accepted_as_ground_truth_count": 0,
                "may_enter_training_without_human_review_count": 0,
            },
            "policy": {
                "no_ground_truth_promotion": True,
                "human_review_required": True,
            },
            "candidates": candidates,
            "clusters": clusters,
        }
        audit_path = root / "audit.json"
        _write_json(audit_path, audit)
        derived_rows = [
            {
                "record_id": value["record_id"],
                "sample_id": value["sample_id"],
                "split": value["split"],
                "source_domain": value["source_domain"],
                "center": value["center"],
                "edge_keys": value["edge_keys"],
            }
            for value in candidates
        ]
        derived_path = root / "derived.jsonl"
        _write_jsonl(derived_path, derived_rows)
        return {
            "audit": audit_path,
            "derived": derived_path,
            "source_manifest": source_manifest,
            "dataset": dataset,
        }


if __name__ == "__main__":
    unittest.main()
