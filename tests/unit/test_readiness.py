from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pcbparse.service.readiness import REQUIRED_MODEL_TASKS, readiness_report
from tests.model_lock_fixtures import valid_inference_contract


class ReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.lock_path = self.root / "models.lock.json"

    def _valid_entries(self) -> list[dict[str, object]]:
        weights = self.root / "weights"
        weights.mkdir(exist_ok=True)
        entries: list[dict[str, object]] = []
        for task in sorted(REQUIRED_MODEL_TASKS):
            relative_path = Path("weights") / f"{task}.bin"
            payload = f"verified-{task}-weights\n".encode()
            (self.root / relative_path).write_bytes(payload)
            entries.append(
                {
                    "task": task,
                    "path": relative_path.as_posix(),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "status": "blessed",
                    "blessed": True,
                    "version": f"{task}-v1",
                    "license": "Apache-2.0",
                    "precision": "fp32",
                    "adapter": "fixture.adapters:load",
                    "inference": valid_inference_contract(task),
                }
            )
        return entries

    def _write_lock(self, entries: list[dict[str, object]]) -> None:
        self.lock_path.write_text(
            json.dumps({"schema_version": "1.0", "models": entries}),
            encoding="utf-8",
        )

    def test_complete_lock_hashes_every_file_and_becomes_ready(self) -> None:
        self._write_lock(self._valid_entries())
        report = readiness_report(
            provider_name="CompositePerceptionProvider",
            models_lock=self.lock_path,
        )

        self.assertTrue(report["ready"])
        self.assertEqual("ready", report["status"])
        self.assertEqual(sorted(REQUIRED_MODEL_TASKS), report["verified_learned_tasks"])
        self.assertTrue(all(model["verified"] for model in report["models"]))
        self.assertTrue(all(model["actual_sha256"] == model["sha256"] for model in report["models"]))

    def test_explicit_four_task_plan_is_exact_and_rejects_junction_artifacts(self) -> None:
        deterministic_tasks = REQUIRED_MODEL_TASKS - {"junction"}
        entries = [
            entry
            for entry in self._valid_entries()
            if entry["task"] in deterministic_tasks
        ]
        self._write_lock(entries)

        report = readiness_report(
            provider_name="production",
            models_lock=self.lock_path,
            required_tasks=deterministic_tasks,
        )
        self.assertTrue(report["ready"])
        self.assertEqual(sorted(deterministic_tasks), report["required_learned_tasks"])
        self.assertEqual(sorted(deterministic_tasks), report["verified_learned_tasks"])

        all_entries = self._valid_entries()
        self._write_lock(all_entries)
        rejected = readiness_report(
            provider_name="production",
            models_lock=self.lock_path,
            required_tasks=deterministic_tasks,
        )
        self.assertFalse(rejected["ready"])
        unexpected = [
            issue
            for issue in rejected["issues"]
            if issue["code"] == "unexpected_task"
        ]
        self.assertEqual(["junction"], [issue["task"] for issue in unexpected])

    def test_empty_lock_is_explicitly_unready(self) -> None:
        self._write_lock([])
        report = readiness_report(provider_name="production", models_lock=self.lock_path)

        self.assertFalse(report["ready"])
        self.assertEqual([], report["verified_learned_tasks"])
        codes = [issue["code"] for issue in report["issues"]]
        self.assertIn("models_empty", codes)
        self.assertEqual(len(REQUIRED_MODEL_TASKS), codes.count("missing_task"))

    def test_tampered_file_fails_true_sha256_verification(self) -> None:
        entries = self._valid_entries()
        self._write_lock(entries)
        (self.root / "weights" / "component.bin").write_bytes(b"tampered after lock creation")

        report = readiness_report(provider_name="production", models_lock=self.lock_path)
        mismatch = next(issue for issue in report["issues"] if issue["code"] == "sha256_mismatch")
        self.assertEqual("component", mismatch["task"])
        self.assertNotEqual(mismatch["expected_sha256"], mismatch["actual_sha256"])
        self.assertNotIn("component", report["verified_learned_tasks"])

    def test_empty_artifact_and_duplicate_json_keys_are_rejected(self) -> None:
        entries = self._valid_entries()
        empty_path = self.root / "weights" / "component.bin"
        empty_path.write_bytes(b"")
        entries[0]["sha256"] = hashlib.sha256(b"").hexdigest()
        self._write_lock(entries)
        report = readiness_report(provider_name="production", models_lock=self.lock_path)
        self.assertIn("file_empty", {issue["code"] for issue in report["issues"]})

        self.lock_path.write_text(
            '{"schema_version":"1.0","models":[],"models":[]}',
            encoding="utf-8",
        )
        duplicate = readiness_report(provider_name="production", models_lock=self.lock_path)
        self.assertIn("lock_unreadable", {issue["code"] for issue in duplicate["issues"]})

    def test_each_compatibility_metadata_field_is_fail_closed(self) -> None:
        baseline = self._valid_entries()
        mutations = {
            "status_not_blessed": ("status", "candidate"),
            "blessed_not_true": ("blessed", False),
            "version_invalid": ("version", "unknown"),
            "license_invalid": ("license", ""),
            "precision_incompatible": ("precision", "fp64"),
            "adapter_invalid": ("adapter", "bad module:factory"),
            "path_missing": ("path", "weights/not-present.bin"),
            "sha256_invalid": ("sha256", "not-a-digest"),
        }
        for expected_code, (field, value) in mutations.items():
            with self.subTest(field=field):
                entries = copy.deepcopy(baseline)
                entries[0][field] = value
                self._write_lock(entries)
                report = readiness_report(provider_name="production", models_lock=self.lock_path)
                self.assertFalse(report["ready"])
                self.assertIn(expected_code, {issue["code"] for issue in report["issues"]})

    def test_inference_object_and_duplicate_options_are_fail_closed(self) -> None:
        entries = self._valid_entries()
        entries[0]["inference"] = []
        self._write_lock(entries)
        invalid = readiness_report(provider_name="production", models_lock=self.lock_path)
        self.assertIn("inference_invalid", {issue["code"] for issue in invalid["issues"]})

        entries = self._valid_entries()
        entries[0]["device"] = "cpu"
        entries[0]["inference"]["device"] = "cuda"
        self._write_lock(entries)
        conflict = readiness_report(provider_name="production", models_lock=self.lock_path)
        self.assertIn("inference_conflict", {issue["code"] for issue in conflict["issues"]})

        entries = self._valid_entries()
        pin = next(row for row in entries if row["task"] == "pin")
        pin["adapter_config"] = {"score_threshold": 0.95}
        self._write_lock(entries)
        compatibility_override = readiness_report(
            provider_name="production", models_lock=self.lock_path
        )
        issue = next(
            issue
            for issue in compatibility_override["issues"]
            if issue["code"] == "inference_conflict"
            and issue.get("task") == "pin"
        )
        self.assertEqual("adapter_config.score_threshold", issue["field"])

    def test_every_task_requires_the_exact_bundle_inference_contract(self) -> None:
        baseline = self._valid_entries()
        for task in sorted(REQUIRED_MODEL_TASKS):
            with self.subTest(task=task):
                entries = copy.deepcopy(baseline)
                entry = next(row for row in entries if row["task"] == task)
                inference = entry["inference"]
                self.assertIsInstance(inference, dict)
                removed = next(name for name in inference if name != "device")
                top_level_fallback = inference.pop(removed)
                # Production adapters historically accepted top-level values.
                # Readiness must validate the nested bundle contract itself and
                # cannot use that compatibility path to fill a missing key.
                entry[removed] = top_level_fallback
                self._write_lock(entries)

                report = readiness_report(
                    provider_name="production", models_lock=self.lock_path
                )
                matching = [
                    issue
                    for issue in report["issues"]
                    if issue["code"] == "inference_contract_invalid"
                    and issue.get("task") == task
                ]
                self.assertEqual(1, len(matching))
                self.assertIn("missing=", matching[0]["message"])
                self.assertNotIn(task, report["verified_learned_tasks"])

    def test_inference_typo_and_extra_key_are_fail_closed(self) -> None:
        entries = self._valid_entries()
        component = next(row for row in entries if row["task"] == "component")
        component["inference"]["score_threhsold"] = component["inference"].pop(
            "score_threshold"
        )
        self._write_lock(entries)

        report = readiness_report(provider_name="production", models_lock=self.lock_path)
        issue = next(
            issue
            for issue in report["issues"]
            if issue["code"] == "inference_contract_invalid"
            and issue.get("task") == "component"
        )
        self.assertIn("missing=", issue["message"])
        self.assertIn("extra=", issue["message"])
        self.assertFalse(report["ready"])

    def test_legacy_candidate_contracts_are_diagnostic_not_exceptions(self) -> None:
        entries = self._valid_entries()
        junction = next(row for row in entries if row["task"] == "junction")
        junction["status"] = "candidate"
        junction["blessed"] = False
        junction["inference"] = {"device": "auto"}
        ocr = next(row for row in entries if row["task"] == "ocr")
        del ocr["inference"]
        self._write_lock(entries)

        report = readiness_report(provider_name="production", models_lock=self.lock_path)

        self.assertFalse(report["ready"])
        codes_by_task = {
            (issue["code"], issue.get("task")) for issue in report["issues"]
        }
        self.assertIn(("status_not_blessed", "junction"), codes_by_task)
        self.assertIn(("blessed_not_true", "junction"), codes_by_task)
        self.assertIn(("inference_contract_invalid", "junction"), codes_by_task)
        self.assertIn(("inference_missing", "ocr"), codes_by_task)
        self.assertTrue(
            any("missing=" in problem for problem in report["problems"]),
            report["problems"],
        )

    def test_task_set_rejects_missing_duplicate_and_unexpected_tasks(self) -> None:
        entries = self._valid_entries()
        entries = [entry for entry in entries if entry["task"] != "junction"]
        duplicate = copy.deepcopy(next(entry for entry in entries if entry["task"] == "component"))
        extra = copy.deepcopy(entries[0])
        extra["task"] = "qwen"
        entries.extend([duplicate, extra])
        self._write_lock(entries)

        report = readiness_report(provider_name="production", models_lock=self.lock_path)
        codes = {issue["code"] for issue in report["issues"]}
        self.assertTrue({"missing_task", "duplicate_task", "unexpected_task"}.issubset(codes))
        self.assertTrue(all(model["actual_sha256"] for model in report["models"]))
        self.assertFalse(report["ready"])

    def test_debug_and_disaster_recovery_providers_never_become_ready(self) -> None:
        self._write_lock(self._valid_entries())
        for provider in (
            "dataset_debug",
            "dataset_golden",
            "DatasetGoldenProvider",
            "empty",
            "EmptyImageProvider",
            "heuristic",
            "HeuristicVisionProvider",
        ):
            with self.subTest(provider=provider):
                report = readiness_report(provider_name=provider, models_lock=self.lock_path)
                self.assertFalse(report["ready"])
                self.assertIn("provider_not_production", {issue["code"] for issue in report["issues"]})


if __name__ == "__main__":
    unittest.main()
