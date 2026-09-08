from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from pcbparse.cli import _provider, main
from pcbparse.evaluation.raw_cli import AccuracyManifestError, read_prediction_manifest
from tests.model_lock_fixtures import valid_inference_contract


WORKSPACE = Path(__file__).resolve().parents[2]
DATASET = WORKSPACE / "Dataset300" / "Dataset300"


class CLIReadinessIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.empty_lock = self.root / "empty-models.lock.json"
        self.empty_lock.write_text(json.dumps({"schema_version": "1.0", "models": []}), encoding="utf-8")
        self.image = self.root / "fixture.png"
        Image.new("RGB", (64, 48), "white").save(self.image)

    def _run(self, arguments: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(arguments)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_production_profiles_fail_closed_before_creating_output(self) -> None:
        for profile in ("baseline", "competition"):
            with self.subTest(profile=profile):
                output = self.root / f"blocked-{profile}"
                code, stdout, stderr = self._run(
                    [
                        "infer",
                        "--input",
                        str(self.image),
                        "--output",
                        str(output),
                        "--profile",
                        profile,
                        "--models-lock",
                        str(self.empty_lock),
                    ]
                )
                self.assertEqual(2, code)
                self.assertEqual("", stdout)
                payload = json.loads(stderr)
                self.assertEqual("unready", payload["status"])
                self.assertFalse(payload["readiness"]["ready"])
                self.assertFalse(output.exists())

    def test_explicit_heuristic_override_is_marked_degraded_debug(self) -> None:
        output = self.root / "degraded"
        evaluation = self.root / "evaluation-ir"
        code, stdout, stderr = self._run(
            [
                "infer",
                "--input",
                str(self.image),
                "--output",
                str(output),
                "--profile",
                "baseline",
                "--provider",
                "heuristic",
                "--models-lock",
                str(self.empty_lock),
                "--allow-unready-debug",
                "--evaluation-ir-dir",
                str(evaluation),
                "--fail-fast",
            ]
        )

        self.assertEqual(0, code, stderr)
        payload = json.loads(stdout)
        self.assertEqual("degraded", payload["status"])
        self.assertEqual("ok", payload["pipeline_status"])
        self.assertEqual("unready_debug", payload["execution_mode"])
        self.assertTrue(payload["degraded"])
        self.assertTrue(payload["debug"])
        self.assertFalse(payload["readiness"]["ready"])
        self.assertTrue((output / "fixture.json").is_file())
        self.assertEqual(evaluation / "predictions.jsonl", Path(payload["evaluation_manifest"]))
        records = read_prediction_manifest(evaluation / "predictions.jsonl")
        self.assertEqual("fixture", records[0].case_id)
        self.assertFalse(records[0].sidecar_access)

    def test_heuristic_requires_explicit_override_even_with_debug_profile(self) -> None:
        output = self.root / "blocked-debug-heuristic"
        code, stdout, stderr = self._run(
            [
                "infer",
                "--input",
                str(self.image),
                "--output",
                str(output),
                "--profile",
                "debug",
                "--provider",
                "heuristic",
                "--models-lock",
                str(self.empty_lock),
            ]
        )

        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("--allow-unready-debug", json.loads(stderr)["message"])
        self.assertFalse(output.exists())

    def test_override_does_not_permit_empty_or_annotation_provider_in_production(self) -> None:
        for provider in ("empty", "dataset_debug", "dataset_golden"):
            with self.subTest(provider=provider):
                output = self.root / f"blocked-{provider}"
                code, stdout, stderr = self._run(
                    [
                        "infer",
                        "--input",
                        str(self.image),
                        "--output",
                        str(output),
                        "--profile",
                        "baseline",
                        "--provider",
                        provider,
                        "--models-lock",
                        str(self.empty_lock),
                        "--allow-unready-debug",
                    ]
                )
                self.assertEqual(2, code)
                self.assertEqual("", stdout)
                self.assertEqual("unready", json.loads(stderr)["status"])
                self.assertFalse(output.exists())

    def test_debug_profile_dataset_debug_remains_runnable_without_override(self) -> None:
        if not (DATASET / "0001" / "0001-KiCad.png").is_file():
            self.skipTest("Dataset300 sample 0001 is not available")
        output = self.root / "dataset-debug"
        code, stdout, stderr = self._run(
            [
                "infer",
                "--input",
                str(DATASET / "0001" / "0001-KiCad.png"),
                "--output",
                str(output),
                "--profile",
                "debug",
                "--provider",
                "dataset_debug",
                "--models-lock",
                str(self.empty_lock),
                "--fail-fast",
            ]
        )

        self.assertEqual(0, code, stderr)
        payload = json.loads(stdout)
        self.assertEqual("debug", payload["status"])
        self.assertEqual("debug", payload["execution_mode"])
        self.assertFalse(payload["degraded"])
        self.assertTrue(payload["debug"])
        self.assertEqual("dataset_debug", payload["provider"])
        self.assertTrue((output / "0001-KiCad.json").is_file())

    def test_annotation_ir_artifact_is_transparently_rejected_by_accuracy_mode(self) -> None:
        if not (DATASET / "0001" / "0001-KiCad.png").is_file():
            self.skipTest("Dataset300 sample 0001 is not available")
        evaluation = self.root / "annotation-evaluation"
        code, _, stderr = self._run(
            [
                "infer",
                "--input",
                str(DATASET / "0001" / "0001-KiCad.png"),
                "--output",
                str(self.root / "annotation-output"),
                "--profile",
                "debug",
                "--provider",
                "dataset_debug",
                "--models-lock",
                str(self.empty_lock),
                "--evaluation-ir-dir",
                str(evaluation),
                "--fail-fast",
            ]
        )
        self.assertEqual(0, code, stderr)
        with self.assertRaisesRegex(AccuracyManifestError, "sidecar_access=false"):
            read_prediction_manifest(evaluation / "predictions.jsonl")

    def test_serve_passes_model_lock_and_server_side_debug_authority(self) -> None:
        input_root = self.root / "api-input"
        output_root = self.root / "api-output"
        input_root.mkdir()
        custom_lock = self.root / "custom-models.lock.json"
        custom_lock.write_text('{"schema_version":"1.0","models":[]}', encoding="utf-8")

        with mock.patch("pcbparse.cli.serve") as serve_mock:
            code, stdout, stderr = self._run(
                [
                    "serve",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "0",
                    "--input-root",
                    str(input_root),
                    "--output-root",
                    str(output_root),
                    "--profile",
                    "debug",
                    "--provider",
                    "dataset_debug",
                    "--models-lock",
                    str(custom_lock),
                    "--allow-unready-debug",
                ]
            )

        self.assertEqual(0, code, stderr)
        self.assertEqual("serving", json.loads(stdout)["status"])
        serve_mock.assert_called_once()
        api_config = serve_mock.call_args.args[2]
        self.assertEqual(custom_lock, api_config.models_lock)
        self.assertTrue(api_config.allow_unready_debug)

    def test_production_provider_receives_loaded_pipeline_tiling(self) -> None:
        sentinel = object()
        with mock.patch(
            "pcbparse.perception.production_provider.load_production_provider",
            return_value=sentinel,
        ) as loader:
            provider = _provider(
                "production",
                models_lock=self.empty_lock,
                pipeline_config={"image": {"tile_size": 384, "tile_overlap": 32}},
            )

        self.assertIs(sentinel, provider)
        loader.assert_called_once()
        resolved = loader.call_args.kwargs["config"]
        self.assertEqual(384, resolved.tile_size)
        self.assertEqual(32, resolved.tile_overlap)

    def test_required_healthcheck_loads_adapters_after_metadata_verification(self) -> None:
        entries = []
        # The packaged pipeline uses deterministic Junction topology, so the
        # exact production model plan contains four learned artifacts.
        for task in ("component", "ocr", "pin", "wire"):
            payload = f"{task}-fixture".encode()
            model = self.root / f"{task}.bin"
            model.write_bytes(payload)
            entries.append(
                {
                    "task": task,
                    "path": model.name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "status": "blessed",
                    "blessed": True,
                    "version": "1.0",
                    "license": "test-only",
                    "precision": "fp32",
                    "adapter": "missing_fixture_adapters:load",
                    "inference": valid_inference_contract(task),
                }
            )
        lock = self.root / "verified-but-unloadable.lock.json"
        lock.write_text(
            json.dumps({"schema_version": "1.0", "models": entries}), encoding="utf-8"
        )

        code, stdout, stderr = self._run(
            [
                "healthcheck",
                "--provider",
                "production",
                "--models-lock",
                str(lock),
                "--require-ready",
            ]
        )

        self.assertEqual(1, code, stderr)
        payload = json.loads(stdout)
        self.assertEqual("unready", payload["status"])
        self.assertFalse(payload["readiness"]["ready"])
        self.assertEqual("failed", payload["runtime_smoke"]["status"])
        self.assertIn(
            "provider_load_or_smoke_failed",
            {item["code"] for item in payload["readiness"]["issues"]},
        )


if __name__ == "__main__":
    unittest.main()
