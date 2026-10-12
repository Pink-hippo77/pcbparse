from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from PIL import Image

from pcbparse.evaluation.infer_manifest import (
    ProductionEvaluationError,
    run_production_evaluation,
)
from pcbparse.evaluation.raw_cli import AccuracyManifestError, read_prediction_manifest
from pcbparse.ir.models import CircuitIR, ImageInfo, IRState
from pcbparse.output import validate_official_output
from pcbparse.service.readiness import ReadinessGateError


TASKS = ("component", "ocr", "pin", "wire")


MODEL_TASK_PLAN = {
    "contract": "production_model_task_plan_v1",
    "junction_mode": "deterministic_vectorizer",
    "required_model_tasks": list(TASKS),
    "loaded_model_tasks": list(TASKS),
    "learned_junction_model_required": False,
    "learned_junction_model_loaded": False,
    "learned_junction_model_invoked": False,
}


class StubProductionProvider:
    provider_kind = "production"
    version = "stub-production-v1"
    model_task_plan = MODEL_TASK_PLAN

    def build_ir(self, input_path, ctx=None):
        path = Path(input_path)
        if path.suffix.lower() != ".png":
            raise AssertionError(f"provider received a non-image path: {path}")
        with Image.open(path) as image:
            width, height = image.size
            actual_format = image.format
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return CircuitIR(
            state=IRState.PERCEIVED,
            image=ImageInfo(
                image_id=ctx.image_id if ctx else path.stem,
                path=str(path.resolve()),
                width=width,
                height=height,
                sha256=digest,
                actual_format=actual_format,
            ),
            metadata={
                "provider": {
                    "name": "StubProductionProvider",
                    "version": self.version,
                    "submission_safe": True,
                },
                "model_task_plan": MODEL_TASK_PLAN,
                "model_versions": {task: "stub-v1" for task in TASKS},
                "quarantined_model_versions": {},
            },
        )


class FailingStubProductionProvider(StubProductionProvider):
    version = "stub-production-failing-v1"

    def build_ir(self, input_path, ctx=None):
        if Path(input_path).parent.name == "2":
            raise RuntimeError("intentional fixture inference failure")
        return super().build_ir(input_path, ctx)


def ready_report(*, provider_name, models_lock, required_tasks):
    required = sorted(required_tasks)
    if required != list(TASKS):
        raise AssertionError(f"unexpected required task plan: {required}")
    return {
        "ready": True,
        "status": "ready",
        "provider": provider_name,
        "models_lock": str(models_lock),
        "required_learned_tasks": required,
        "registered_learned_tasks": required,
        "verified_learned_tasks": required,
        "issues": [],
        "problems": [],
    }


class ProductionEvaluationHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.config_path = self.root / "pipeline.json"
        self.profile_path = self.root / "profile.json"
        self.lock_path = self.root / "models.lock.json"
        self.split_path = self.root / "split.jsonl"

        source_config = json.loads(Path("configs/pipeline.yaml").read_text(encoding="utf-8"))
        source_config["image"]["tile_size"] = 321
        source_config["image"]["tile_overlap"] = 17
        self.config_path.write_text(
            json.dumps(source_config, ensure_ascii=False), encoding="utf-8"
        )
        self.profile_path.write_text(
            json.dumps(
                {
                    "name": "baseline",
                    "provider": "production",
                    "qwen_enabled": False,
                    "artifact_policy": "error",
                    "strict_validation": True,
                }
            ),
            encoding="utf-8",
        )
        models = []
        for task in TASKS:
            model = self.root / f"{task}.bin"
            model.write_bytes(f"fixture-{task}".encode("ascii"))
            models.append({"task": task, "path": model.name})
        self.lock_path.write_text(
            json.dumps({"schema_version": "1.0", "models": models}),
            encoding="utf-8",
        )
        self.rows = [self._make_case("10", color="white"), self._make_case("2", color="gray")]
        self._write_split(self.rows)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make_case(self, case_id: str, *, color: str) -> dict:
        case_root = self.dataset / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        image_path = case_root / f"board-{case_id}.png"
        Image.new("RGB", (24, 16), color).save(image_path)
        # These files are deliberately not valid JSON/EDS/NET.  Any accidental
        # annotation-sidecar load in the image-only harness or provider fails.
        (case_root / "invalid_CVJsonStd.json").write_bytes(b"{not-json")
        (case_root / "invalid.eds").write_bytes(b"\xff\x00invalid-eds")
        (case_root / "invalid.net").write_text("not a netlist", encoding="utf-8")
        return {
            "sample_id": case_id,
            "split": "test",
            "duplicate_group": "near-duplicate-fixture",
            "duplicate_group_size": 2,
            "files": {
                "image": {
                    "path": image_path.relative_to(self.dataset).as_posix(),
                    "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                },
                "json": {"path": f"{case_id}/invalid_CVJsonStd.json"},
                "eds": {"path": f"{case_id}/invalid.eds"},
                "net": {"path": f"{case_id}/invalid.net"},
            },
        }

    def _write_split(self, rows: list[dict]) -> None:
        self.split_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _run(self, output: Path, *, provider=None, fail_fast=False):
        instance = provider or StubProductionProvider()
        return run_production_evaluation(
            dataset_root=self.dataset,
            split_manifest=self.split_path,
            split="test",
            models_lock=self.lock_path,
            pipeline_config=self.config_path,
            profile=self.profile_path,
            output_dir=output,
            fail_fast=fail_fast,
            readiness_checker=ready_report,
            provider_loader=lambda _lock: instance,
        )

    def test_three_same_environment_runs_are_stable_strict_and_manifest_compatible(self) -> None:
        first = self._run(self.root / "first")
        second = self._run(self.root / "second")
        third = self._run(self.root / "third")

        self.assertEqual("ok", first.summary["status"])
        self.assertTrue(first.summary["all_successful"])
        self.assertEqual(2, first.summary["succeeded"])
        self.assertEqual(1, first.summary["selected_duplicate_groups"])
        self.assertTrue(first.summary["coverage"]["exact_case_coverage"])
        self.assertTrue(first.summary["coverage"]["exact_duplicate_group_coverage"])
        self.assertEqual(321, first.summary["resolved_perception_config"]["tile_size"])
        self.assertEqual(17, first.summary["resolved_perception_config"]["tile_overlap"])
        self.assertEqual(512, first.summary["resolved_perception_config"]["ocr_tile_size"])
        self.assertRegex(first.summary["resolved_perception_config_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(MODEL_TASK_PLAN, first.summary["model_task_plan"])
        self.assertEqual(list(TASKS), first.summary["readiness"]["required_learned_tasks"])
        self.assertEqual(list(TASKS), first.summary["readiness"]["verified_learned_tasks"])
        runtime_sources = first.summary["runtime_source_manifest"]
        self.assertEqual("production_runtime_source_manifest_v1", runtime_sources["contract"])
        self.assertGreater(runtime_sources["file_count"], 0)
        self.assertEqual(runtime_sources["file_count"], len(runtime_sources["files"]))
        self.assertRegex(runtime_sources["source_set_sha256"], r"^[0-9a-f]{64}$")
        timing = first.summary["timing"]
        self.assertEqual("production_wall_clock_scope_v1", timing["contract"])
        self.assertTrue(timing["cold_start_included"])
        self.assertTrue(timing["config_and_profile_load_included"])
        self.assertTrue(timing["readiness_included"])
        self.assertTrue(timing["provider_and_model_load_included"])
        self.assertTrue(timing["case_loop_included"])
        self.assertTrue(timing["prediction_manifest_write_included"])
        self.assertFalse(timing["summary_write_and_cli_teardown_included"])
        self.assertAlmostEqual(
            first.summary["elapsed_seconds"],
            timing["total_elapsed_seconds"],
            places=9,
        )
        self.assertAlmostEqual(
            timing["total_elapsed_seconds"],
            timing["initialization_seconds"]
            + timing["case_loop_seconds"]
            + timing["evidence_finalize_seconds"],
            places=9,
        )
        self.assertEqual(7200.0, timing["competition_time_limit_seconds"])
        self.assertTrue(timing["competition_time_limit_enforced"])
        self.assertTrue(timing["within_competition_time_limit"])
        self.assertEqual(2, timing["case_latency_seconds"]["count"])
        self.assertGreaterEqual(
            timing["case_latency_seconds"]["p95_seconds"],
            timing["case_latency_seconds"]["p50_seconds"],
        )
        self.assertEqual(
            "run_context_token_usage_summary_v1",
            first.summary["token_usage"]["contract"],
        )
        self.assertEqual(2, first.summary["token_usage"]["cases_with_evidence"])
        self.assertEqual(
            first.summary["prediction_set_sha256"],
            second.summary["prediction_set_sha256"],
        )
        self.assertEqual(
            first.summary["prediction_set_sha256"],
            third.summary["prediction_set_sha256"],
        )
        run_ids = {
            first.summary["inference_run_id"],
            second.summary["inference_run_id"],
            third.summary["inference_run_id"],
        }
        self.assertEqual(3, len(run_ids))
        for result in (first, second, third):
            self.assertRegex(result.summary["inference_run_id"], r"^[0-9a-f]{64}$")
            started_at = datetime.fromisoformat(
                result.summary["started_at_utc"].replace("Z", "+00:00")
            )
            completed_at = datetime.fromisoformat(
                result.summary["completed_at_utc"].replace("Z", "+00:00")
            )
            self.assertLessEqual(started_at, completed_at)
            self.assertLessEqual(completed_at, datetime.now(timezone.utc))
            self.assertEqual(
                result.summary["completed_at_utc"],
                result.summary["generated_at_utc"],
            )

        first_rows = [
            json.loads(line)
            for line in first.prediction_manifest.read_text(encoding="utf-8").splitlines()
        ]
        second_rows = [
            json.loads(line)
            for line in second.prediction_manifest.read_text(encoding="utf-8").splitlines()
        ]
        third_rows = [
            json.loads(line)
            for line in third.prediction_manifest.read_text(encoding="utf-8").splitlines()
        ]
        for summary, rows in (
            (first.summary, first_rows),
            (second.summary, second_rows),
            (third.summary, third_rows),
        ):
            self.assertTrue(rows)
            self.assertEqual(
                {summary["inference_run_id"]},
                {row["inference_run_id"] for row in rows},
            )
        self.assertEqual(["2", "10"], [row["case_id"] for row in first_rows])
        self.assertEqual(
            [row["ir_sha256"] for row in first_rows],
            [row["ir_sha256"] for row in second_rows],
        )
        self.assertEqual(
            [row["official_sha256"] for row in first_rows],
            [row["official_sha256"] for row in second_rows],
        )
        self.assertEqual(
            [row["official_sha256"] for row in first_rows],
            [row["official_sha256"] for row in third_rows],
        )
        for row in first_rows:
            self.assertEqual("production", row["provider_kind"])
            self.assertFalse(row["sidecar_access"])
            self.assertEqual("ok", row["status"])
            self.assertIsNone(row["error"])
            self.assertEqual("empty_result", row["quality"]["status"])
            self.assertTrue(row["quality"]["legitimate_empty_input"])
            self.assertEqual(
                {"perception", "association", "pre_prune", "final"},
                set(row["stage_evidence"]),
            )
            for stage, evidence in row["stage_evidence"].items():
                evidence_path = first.output_dir / evidence["path"]
                payload = evidence_path.read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), evidence["sha256"])
                snapshot = json.loads(payload)
                self.assertEqual(evidence["counts"]["components"], len(snapshot["components"]))
                self.assertEqual("SERIALIZED" if stage == "final" else (
                    "PERCEIVED" if stage == "perception" else "ASSOCIATED"
                ), snapshot["state"])
            self.assertEqual(
                "run_context_runtime_evidence_v1",
                row["runtime_evidence"]["contract"],
            )
            self.assertIn("trace", row["runtime_evidence"])
            self.assertIn("tokens", row["runtime_evidence"])
            self.assertTrue(row["ir_path"].startswith("circuit_ir/"))
            self.assertTrue(row["official_path"].startswith("official/"))
            lowered = "\n".join(row["input_paths_read"]).lower()
            self.assertNotIn("cvjsonstd", lowered)
            self.assertNotIn(".eds", lowered)
            self.assertNotIn(".net", lowered)
            self.assertEqual(
                str((self.dataset / row["case_id"] / f"board-{row['case_id']}.png").resolve()),
                row["input_paths_read"][0],
            )
            official = json.loads(
                (first.output_dir / row["official_path"]).read_text(encoding="utf-8")
            )
            # This exercises exact guide-v1 keys and dangling-reference checks
            # at the production harness boundary.  Official evaluator authority
            # remains a separate release requirement.
            validate_official_output(official)

        parsed = read_prediction_manifest(first.prediction_manifest)
        self.assertEqual(["2", "10"], [record.case_id for record in parsed])
        self.assertFalse(list(first.output_dir.rglob("*.tmp")))
        self.assertEqual(b"{not-json", (self.dataset / "2" / "invalid_CVJsonStd.json").read_bytes())

    def test_real_loader_receives_resolved_custom_tiling(self) -> None:
        with mock.patch(
            "pcbparse.evaluation.infer_manifest._load_production_provider",
            return_value=StubProductionProvider(),
        ) as loader:
            result = run_production_evaluation(
                dataset_root=self.dataset,
                split_manifest=self.split_path,
                split="test",
                models_lock=self.lock_path,
                pipeline_config=self.config_path,
                profile=self.profile_path,
                output_dir=self.root / "configured-loader",
                readiness_checker=ready_report,
            )

        loader.assert_called_once()
        resolved = loader.call_args.args[1]
        self.assertEqual(321, resolved.tile_size)
        self.assertEqual(17, resolved.tile_overlap)
        self.assertEqual(
            resolved.to_dict(),
            result.summary["resolved_perception_config"],
        )

    def test_fail_fast_records_error_and_remaining_case_is_not_run(self) -> None:
        result = self._run(
            self.root / "failure",
            provider=FailingStubProductionProvider(),
            fail_fast=True,
        )
        rows = [
            json.loads(line)
            for line in result.prediction_manifest.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(["error", "not_run"], [row["status"] for row in rows])
        self.assertEqual("failed", result.summary["status"])
        self.assertFalse(result.summary["all_successful"])
        self.assertEqual(1, result.summary["failed"])
        self.assertEqual(1, result.summary["not_run"])
        self.assertIsNone(rows[0]["ir_path"])
        self.assertEqual("RuntimeError", rows[0]["error"]["type"])
        self.assertEqual("inference", rows[0]["error"]["stage"])
        self.assertEqual(0, rows[0]["error"]["diagnostic_count"])
        self.assertEqual({}, rows[0]["error"]["diagnostic_codes"])
        self.assertEqual([], rows[0]["error"]["diagnostics"])
        self.assertRegex(rows[0]["error"]["error_id"], r"^[0-9a-f]{64}$")
        self.assertEqual("FailFastSkipped", rows[1]["error"]["type"])
        with self.assertRaises(AccuracyManifestError):
            read_prediction_manifest(result.prediction_manifest)
        self.assertFalse(list(result.output_dir.rglob("*.tmp")))

    def test_whole_program_time_limit_records_unrun_cases(self) -> None:
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["run"]["max_total_seconds"] = 1e-9
        self.config_path.write_text(
            json.dumps(config, ensure_ascii=False), encoding="utf-8"
        )
        result = self._run(self.root / "time-limit")
        rows = [
            json.loads(line)
            for line in result.prediction_manifest.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(["not_run", "not_run"], [row["status"] for row in rows])
        self.assertEqual("failed", result.summary["status"])
        self.assertEqual(0, result.summary["attempted"])
        self.assertEqual(2, result.summary["not_run"])
        self.assertFalse(result.summary["timing"]["within_competition_time_limit"])
        self.assertEqual(1e-9, result.summary["timing"]["competition_time_limit_seconds"])
        self.assertEqual(
            ["CompetitionTimeLimitExceeded", "CompetitionTimeLimitExceeded"],
            [row["error"]["type"] for row in rows],
        )

    def test_unready_real_gate_fails_before_creating_output(self) -> None:
        destination = self.root / "must-not-exist"
        with self.assertRaises(ReadinessGateError):
            run_production_evaluation(
                dataset_root=self.dataset,
                split_manifest=self.split_path,
                split="test",
                models_lock=self.lock_path,
                pipeline_config=self.config_path,
                profile=self.profile_path,
                output_dir=destination,
            )
        self.assertFalse(destination.exists())

    def test_accuracy_mode_rejects_debug_profile_and_duplicate_leakage(self) -> None:
        debug_profile = self.root / "debug-profile.json"
        debug_profile.write_text(
            json.dumps(
                {
                    "name": "debug",
                    "provider": "dataset_debug",
                    "qwen_enabled": False,
                    "artifact_policy": "error",
                    "strict_validation": True,
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ProductionEvaluationError, "accuracy mode"):
            run_production_evaluation(
                dataset_root=self.dataset,
                split_manifest=self.split_path,
                split="test",
                models_lock=self.lock_path,
                pipeline_config=self.config_path,
                profile=debug_profile,
                output_dir=self.root / "debug-output",
                readiness_checker=ready_report,
                provider_loader=lambda _lock: StubProductionProvider(),
            )

        leaking = [dict(row) for row in self.rows]
        leaking[1] = {**leaking[1], "split": "val"}
        self._write_split(leaking)
        with self.assertRaisesRegex(ProductionEvaluationError, "leak across splits"):
            self._run(self.root / "leaking")


if __name__ == "__main__":
    unittest.main()
