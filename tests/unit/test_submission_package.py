from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from pcbparse.output.submission_package import (
    SubmissionPackageError,
    package_submission,
)


def _official(case_id: str) -> dict:
    return {
        "components": {
            f"R{case_id}": {
                "type": "resistor",
                "Name": None,
                "value": "10k",
                "bbox": [10, 10, 20, 20],
            }
        },
        "pins": {
            f"R{case_id}": {
                "pin_1": {"pinname": "1", "point": {"x": 10, "y": 15}}
            }
        },
        "nets": {
            "net_1": {
                "hyperGraph": f"(R{case_id}.1)",
                "edges": {
                    "edge_1": [{"x": 10, "y": 15}, {"x": 30, "y": 15}]
                },
            }
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SubmissionPackageTests(unittest.TestCase):
    def _evaluation(self, root: Path, case_ids: tuple[str, ...] = ("2", "10")) -> Path:
        evaluation = root / "evaluation"
        official = evaluation / "official"
        circuit_ir = evaluation / "circuit_ir"
        official.mkdir(parents=True)
        circuit_ir.mkdir()
        rows = []
        for case_id in case_ids:
            official_path = official / f"{case_id}.json"
            official_path.write_text(
                json.dumps(_official(case_id), ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            ir_path = circuit_ir / f"{case_id}.circuit-ir.json"
            ir_path.write_text("{}\n", encoding="utf-8")
            digest = _sha256(official_path)
            rows.append(
                {
                    "case_id": case_id,
                    "duplicate_group": f"group:{case_id}",
                    "image_sha256": "a" * 64,
                    "ir_path": ir_path.relative_to(evaluation).as_posix(),
                    "ir_sha256": _sha256(ir_path),
                    "official_path": official_path.relative_to(evaluation).as_posix(),
                    "official_sha256": digest,
                    "official_output_sha256": digest,
                    "provider_kind": "production",
                    "model_lock_sha256": "b" * 64,
                    "config_hash": "c" * 64,
                    "sidecar_access": False,
                    "input_paths_read": [f"board-{case_id}.png"],
                    "elapsed_seconds": 1.0,
                    "status": "ok",
                    "error": None,
                }
            )
        manifest = evaluation / "predictions.jsonl"
        manifest.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        summary = {
            "schema_version": 1,
            "kind": "production-image-only-inference",
            "accuracy_mode": "image-only",
            "execution_mode": "production",
            "provider_kind": "production",
            "sidecar_access": False,
            "status": "ok",
            "all_successful": True,
            "selected_cases": len(rows),
            "succeeded": len(rows),
            "failed": 0,
            "not_run": 0,
            "prediction_manifest": "predictions.jsonl",
            "prediction_manifest_sha256": _sha256(manifest),
            "model_lock_sha256": "b" * 64,
            "config_hash": "c" * 64,
            "readiness": {"status": "ready", "ready": True},
        }
        (evaluation / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return evaluation

    def test_packages_exact_guide_layout_and_deterministic_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = self._evaluation(root)
            first = package_submission(
                evaluation,
                root / "first" / "submission",
                archive_path=root / "first" / "submission.zip",
                audit_path=root / "first" / "package.audit.json",
            )
            second = package_submission(
                evaluation,
                root / "second" / "submission",
                archive_path=root / "second" / "submission.zip",
            )

            expected = {
                "case_2/case_2.json",
                "case_10/case_10.json",
            }
            self.assertEqual(
                expected,
                {
                    path.relative_to(first.output_dir).as_posix()
                    for path in first.output_dir.rglob("*.json")
                },
            )
            self.assertEqual(
                first.report["submission_tree_sha256"],
                second.report["submission_tree_sha256"],
            )
            self.assertEqual(_sha256(first.archive_path), _sha256(second.archive_path))
            with zipfile.ZipFile(first.archive_path) as archive:
                self.assertEqual(
                    [
                        "submission/case_2/case_2.json",
                        "submission/case_10/case_10.json",
                    ],
                    archive.namelist(),
                )
                for entry in archive.infolist():
                    self.assertEqual((1980, 1, 1, 0, 0, 0), entry.date_time)
                    self.assertEqual(3, entry.create_system)
                    self.assertEqual(0o100644, entry.external_attr >> 16)
            audit = json.loads(
                (root / "first" / "package.audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(2, audit["case_count"])
            self.assertFalse(audit["official_evaluator_available"])
            self.assertEqual(
                _sha256(evaluation / "summary.json"),
                audit["source"]["evaluation_summary_sha256"],
            )
            self.assertEqual(
                _sha256(evaluation / "predictions.jsonl"),
                audit["source"]["prediction_manifest_sha256"],
            )
            self.assertEqual(_sha256(first.archive_path), audit["archive"]["sha256"])

    def test_rejects_unsuccessful_or_tampered_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = self._evaluation(root)
            summary_path = evaluation / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary.update({"status": "partial", "all_successful": False})
            summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(SubmissionPackageError, "status"):
                package_submission(evaluation, root / "failed" / "submission")

            evaluation = self._evaluation(root / "tampered")
            (evaluation / "official" / "2.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(SubmissionPackageError, "SHA-256"):
                package_submission(evaluation, root / "tampered-out" / "submission")

    def test_rejects_normalized_case_name_collision_and_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = self._evaluation(root, ("1", "case_1"))
            with self.assertRaisesRegex(SubmissionPackageError, "collides"):
                package_submission(evaluation, root / "collision" / "submission")

            evaluation = self._evaluation(root / "existing")
            destination = root / "occupied" / "submission"
            destination.mkdir(parents=True)
            marker = destination / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(SubmissionPackageError, "already exists"):
                package_submission(evaluation, destination)
            self.assertEqual("keep", marker.read_text(encoding="utf-8"))

    def test_rejects_manifest_lineage_mixed_across_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = self._evaluation(root)
            manifest = evaluation / "predictions.jsonl"
            rows = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
            ]
            rows[1]["config_hash"] = "d" * 64
            manifest.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            summary_path = evaluation / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["prediction_manifest_sha256"] = _sha256(manifest)
            summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(SubmissionPackageError, "configuration lineage"):
                package_submission(evaluation, root / "mixed" / "submission")

    def test_rejects_windows_portable_name_collisions_and_trailing_period(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = self._evaluation(root, ("X", "case_x"))
            with self.assertRaisesRegex(SubmissionPackageError, "collides"):
                package_submission(evaluation, root / "casefold" / "submission")

            evaluation = self._evaluation(root / "period", ("x.",))
            official_path = evaluation / "official" / "x..json"
            official_path.write_text(
                json.dumps(_official("safe"), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest = evaluation / "predictions.jsonl"
            row = json.loads(manifest.read_text(encoding="utf-8"))
            row["official_sha256"] = _sha256(official_path)
            row["official_output_sha256"] = row["official_sha256"]
            manifest.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
            summary_path = evaluation / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["prediction_manifest_sha256"] = _sha256(manifest)
            summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(SubmissionPackageError, "Windows-portable"):
                package_submission(evaluation, root / "period-out" / "submission")

    def test_binds_each_case_to_its_exact_official_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = self._evaluation(root)
            manifest = evaluation / "predictions.jsonl"
            rows = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
            ]
            for field in (
                "official_path",
                "official_sha256",
                "official_output_sha256",
            ):
                rows[0][field], rows[1][field] = rows[1][field], rows[0][field]
            manifest.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            summary_path = evaluation / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["prediction_manifest_sha256"] = _sha256(manifest)
            summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(SubmissionPackageError, "official_path must"):
                package_submission(evaluation, root / "swapped" / "submission")

    def test_rejects_duplicate_json_keys_and_output_inside_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = self._evaluation(root)
            manifest = evaluation / "predictions.jsonl"
            lines = manifest.read_text(encoding="utf-8").splitlines()
            lines[0] = '{"case_id":"2",' + lines[0][1:]
            manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
            summary_path = evaluation / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["prediction_manifest_sha256"] = _sha256(manifest)
            summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(SubmissionPackageError, "duplicate JSON"):
                package_submission(evaluation, root / "duplicate" / "submission")

            evaluation = self._evaluation(root / "nested")
            with self.assertRaisesRegex(SubmissionPackageError, "outside the evaluation"):
                package_submission(evaluation, evaluation / "submission")

    def test_archive_and_audit_are_distinct_and_existing_files_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = self._evaluation(root)
            shared = root / "shared.out"
            with self.assertRaisesRegex(SubmissionPackageError, "must be different"):
                package_submission(
                    evaluation,
                    root / "same" / "submission",
                    archive_path=shared,
                    audit_path=shared,
                )
            self.assertFalse(shared.exists())
            self.assertFalse((root / "same" / "submission").exists())

            with self.assertRaisesRegex(SubmissionPackageError, "Windows-portable"):
                package_submission(
                    evaluation,
                    root / "portable" / "submission",
                    archive_path=root / "portable" / "archive.",
                )
            self.assertFalse((root / "portable" / "submission").exists())

            for label in ("archive", "audit"):
                target = root / f"existing-{label}.out"
                target.write_bytes(b"keep")
                arguments = {f"{label}_path": target}
                with self.assertRaisesRegex(SubmissionPackageError, "already exists"):
                    package_submission(
                        evaluation,
                        root / label / "submission",
                        **arguments,
                    )
                self.assertEqual(b"keep", target.read_bytes())
                self.assertFalse((root / label / "submission").exists())

    def test_publication_failure_never_exposes_partial_submission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = self._evaluation(root)
            destination = root / "failed-archive" / "submission"
            archive = root / "failed-archive" / "submission.zip"
            with mock.patch(
                "pcbparse.output.submission_package.zipfile.ZipFile.writestr",
                side_effect=OSError("forced archive failure"),
            ):
                with self.assertRaisesRegex(OSError, "forced archive failure"):
                    package_submission(
                        evaluation, destination, archive_path=archive
                    )
            self.assertFalse(destination.exists())
            self.assertFalse(archive.exists())

            destination = root / "raced" / "submission"
            archive = root / "raced" / "submission.zip"
            audit = root / "raced" / "package.audit.json"

            def create_racing_destination(_source: object, target: object) -> None:
                raced = Path(target)
                raced.mkdir()
                (raced / "keep.txt").write_text("keep", encoding="utf-8")
                raise FileExistsError("forced destination race")

            with mock.patch(
                "pcbparse.output.submission_package.os.rename",
                side_effect=create_racing_destination,
            ):
                with self.assertRaisesRegex(SubmissionPackageError, "already exists"):
                    package_submission(
                        evaluation,
                        destination,
                        archive_path=archive,
                        audit_path=audit,
                    )
            self.assertEqual(
                "keep", (destination / "keep.txt").read_text(encoding="utf-8")
            )
            self.assertFalse(archive.exists())
            self.assertFalse(audit.exists())


if __name__ == "__main__":
    unittest.main()
