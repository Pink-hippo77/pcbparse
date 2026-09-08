from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import unittest
from pathlib import Path

from pcbparse.training.config import (
    load_training_config,
    training_config_resume_sha256s,
    training_config_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
OCR_V4 = ROOT / "configs" / "train" / "dataset300_v2_ocr_v4_rare_prefix.json"
OCR_V5 = ROOT / "configs" / "train" / "dataset300_v2_ocr_v5_allgate.json"
COMPONENT_V5 = (
    ROOT / "configs" / "train" / "dataset300_v2_component_v5_fullmix.json"
)
PREFLIGHT = ROOT / "work" / "preflight-ocr-v5-allgate.json"
LAUNCHER = ROOT / "scripts" / "auto_launch_ocr_v5.ps1"


class OCRV5LaunchContractTests(unittest.TestCase):
    def test_v5_changes_only_run_identity_and_hash_bound_warm_start_from_v4(self) -> None:
        v4 = json.loads(OCR_V4.read_text(encoding="utf-8"))
        v5 = json.loads(OCR_V5.read_text(encoding="utf-8"))

        self.assertNotEqual(v4.pop("name"), v5.pop("name"))
        warm_start = v5["tasks"]["ocr"].pop("pretrained_artifact")
        v4["tasks"]["ocr"].pop("pretrained_artifact")
        self.assertEqual(v4, v5)
        self.assertEqual(
            "work/train-ocr-v4-rare-prefix/checkpoint_epoch_008.pt",
            warm_start["path"],
        )
        self.assertEqual(
            "f9d7aea4fbdc3d6be63c4566aa8376a9b373571a331ac70e7b4c24569f404ee9",
            warm_start["sha256"],
        )
        self.assertIn("weights-only", warm_start["source"])
        warm_start_path = ROOT / warm_start["path"]
        self.assertTrue(warm_start_path.is_file())
        self.assertEqual(
            warm_start["sha256"],
            hashlib.sha256(warm_start_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            [
                ("normalized_exact_match", "gte", 0.65),
                ("detector_hmean", "gte", 0.45),
                ("recognition_designator_exact_match", "gte", 0.9),
                ("recognition_pinnumber_exact_match", "gte", 0.74),
            ],
            [
                (row["metric"], row["operator"], row["threshold"])
                for row in v5["tasks"]["ocr"]["evaluation_gates"]
            ],
        )

    def test_preflight_is_bound_to_the_v5_config_and_frozen_data(self) -> None:
        config = load_training_config(OCR_V5)
        preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))

        self.assertTrue(preflight["ok"])
        self.assertEqual([], preflight["issues"])
        self.assertEqual(
            training_config_sha256(config), preflight["training_config_sha256"]
        )
        self.assertEqual(config.dataset.config_hash, preflight["dataset"]["config_hash"])
        self.assertEqual(
            config.dataset.source_manifest_sha256,
            preflight["hashes"]["source_manifest"],
        )
        self.assertEqual(
            config.dataset.split_manifest_sha256,
            preflight["hashes"]["split_manifest"],
        )
        self.assertEqual(
            config.dataset.task_manifest_sha256["ocr_text"],
            preflight["tasks"]["ocr"]["sha256"],
        )

    def test_launcher_is_lineage_bound_duplicate_safe_and_v2_only(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        ocr_v5 = load_training_config(OCR_V5)
        component_v5 = load_training_config(COMPONENT_V5)

        self.assertIn(training_config_sha256(component_v5), launcher)
        self.assertIn(training_config_sha256(ocr_v5), launcher)
        self.assertIn("train-component-v5-fullmix", launcher)
        self.assertIn('status -in @("candidate_evidence_ready", "gate_failed")', launcher)
        self.assertIn("formal_evidence -eq $true", launcher)
        self.assertIn("Get-RecordedProcessState", launcher)
        self.assertIn("reports running without a provably inactive", launcher)
        self.assertIn("[Threading.Mutex]", launcher)
        self.assertEqual(1, launcher.count("return Start-Process"))
        self.assertEqual(1, launcher.count("$process = Start-BoundPythonProcess"))
        self.assertIn('"--task", "ocr"', launcher)
        self.assertIn('"--resume-checkpoint"', launcher)
        self.assertIn("pcbparse.torch_training_resume.v2", launcher)
        self.assertIn("all_gate_normalized_violation_v1", launcher)
        self.assertIn("exact_lexicographic_key", launcher)
        self.assertIn("legacy_primary_metric_direction_with_absolute_epsilon", launcher)
        self.assertIn("Test-OcrAllGatePolicy", launcher)
        self.assertIn("Test-OcrStatusLineage", launcher)
        self.assertIn("training_config_sha256 = $expectedOcrTrainingConfigSha256", launcher)
        self.assertIn("resume_schema = $expectedResumeSchema", launcher)
        self.assertIn("model_selection_policy = $expectedSelectionPolicy", launcher)
        self.assertIn("$pidTemporary", launcher)
        self.assertIn("Move-Item -LiteralPath $pidTemporary", launcher)
        self.assertIn("unhandled launcher error", launcher)
        self.assertIn(
            "[hashtable]::new([StringComparer]::OrdinalIgnoreCase)", launcher
        )
        self.assertIn('[Environment]::GetEnvironmentVariables("Process")', launcher)
        self.assertIn(
            '$name.Equals("Path", [StringComparison]::OrdinalIgnoreCase)', launcher
        )
        self.assertIn('$launchEnvironment["Path"] = $pathValue', launcher)
        self.assertIn(
            '$name.Equals("PYTHONPATH", [StringComparison]::OrdinalIgnoreCase)',
            launcher,
        )
        self.assertIn(
            '$pythonWorkingDirectory = Join-Path $workspaceRoot "src"', launcher
        )
        self.assertIn("-WorkingDirectory $pythonWorkingDirectory", launcher)
        self.assertNotIn('$launchEnvironment["PYTHONPATH"]', launcher)
        self.assertIn("-UseNewEnvironment", launcher)
        self.assertIn("-Environment $launchEnvironment", launcher)
        self.assertIn("Invoke-LaunchEnvironmentSmoke", launcher)
        self.assertIn('"-m", "pcbparse.training.torch_train_cli", "--help"', launcher)
        self.assertIn("rejected v1 resume checkpoint", launcher)
        self.assertTrue(
            set(training_config_resume_sha256s(ocr_v5)).isdisjoint(
                training_config_resume_sha256s(load_training_config(OCR_V4))
            ),
            "an OCR-v4 checkpoint must not be resume-compatible with the v5 run name",
        )

    def test_executable_launch_environment_smoke_imports_from_unicode_workspace(self) -> None:
        pwsh = shutil.which("pwsh")
        if pwsh is None:
            self.skipTest("PowerShell 7 is unavailable")
        output = ROOT / "work" / "train-ocr-v5-allgate"
        pid_file = ROOT / "work" / "ocr-v5.pid.json"
        output_existed = output.exists()
        pid_existed = pid_file.exists()
        pid_bytes = pid_file.read_bytes() if pid_existed else None

        completed = subprocess.run(
            [
                pwsh,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-File",
                str(LAUNCHER),
                "-Workspace",
                str(ROOT),
                "-LaunchEnvironmentSmoke",
            ],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(
            0,
            completed.returncode,
            f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}",
        )
        self.assertIn("OCR_V5_LAUNCH_ENVIRONMENT_SMOKE_OK", completed.stdout)
        self.assertEqual(output_existed, output.exists())
        self.assertEqual(pid_existed, pid_file.exists())
        if pid_existed:
            self.assertEqual(pid_bytes, pid_file.read_bytes())


if __name__ == "__main__":
    unittest.main()
