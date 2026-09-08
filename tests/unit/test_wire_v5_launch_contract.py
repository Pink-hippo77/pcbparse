from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import unittest
from pathlib import Path

from pcbparse.training.config import (
    load_training_config,
    training_config_resume_sha256s,
    training_config_sha256,
)
from pcbparse.training.preflight import preflight_training_data


ROOT = Path(__file__).resolve().parents[2]
BASE_WIRE_CONFIG = ROOT / "configs" / "train" / "dataset300_v2_torch.json"
WIRE_V5_CONFIG = (
    ROOT / "configs" / "train" / "dataset300_v2_wire_v5_fresh_allgate.json"
)
OCR_V5_CONFIG = ROOT / "configs" / "train" / "dataset300_v2_ocr_v5_allgate.json"
FORMAL4_STATUS = ROOT / "work" / "train-wire-formal4" / "torch_run_status.json"
LAUNCHER = ROOT / "scripts" / "auto_launch_wire_v5.ps1"

EXPECTED_WIRE_SEMANTIC_SHA256 = (
    "e9b4003e3352a3ab4ea647fc1f3bec281e0cb4463fe3147d3851a28b1639716a"
)
EXPECTED_WIRE_FILE_SHA256 = (
    "e2a1b78b1f766d3075582d8960c2c03871d01ac16f7a1fe1fef56c66bfc5afee"
)
EXPECTED_OCR_SEMANTIC_SHA256 = (
    "80508a46009e160eef0f6c228aeccac152f148205ab7e917aaf2a0025c5ea229"
)
EXPECTED_OCR_FILE_SHA256 = (
    "833752c16936f9bb02ccec7d9344907d40719636cb219bbbcf156d024095718b"
)


class WireV5LaunchContractTests(unittest.TestCase):
    def test_fresh_v5_changes_only_identity_and_has_no_wire_warm_start(self) -> None:
        base = json.loads(BASE_WIRE_CONFIG.read_text(encoding="utf-8"))
        v5 = json.loads(WIRE_V5_CONFIG.read_text(encoding="utf-8"))

        self.assertEqual("dataset300-v2-pytorch-4gb", base.pop("name"))
        self.assertEqual(
            "dataset300-v2-pytorch-4gb-wire-v5-fresh-allgate", v5.pop("name")
        )
        self.assertEqual(base, v5)
        self.assertNotIn("pretrained_artifact", v5["tasks"]["wire"])
        self.assertEqual("tiny_wire_unet_v1", v5["tasks"]["wire"]["architecture"])
        self.assertEqual(
            [("skeleton_f1", "gte", 0.55)],
            [
                (gate["metric"], gate["operator"], gate["threshold"])
                for gate in v5["tasks"]["wire"]["evaluation_gates"]
            ],
        )

    def test_semantic_and_file_hashes_are_exact_and_preflight_is_clean(self) -> None:
        wire = load_training_config(WIRE_V5_CONFIG)
        ocr = load_training_config(OCR_V5_CONFIG)

        self.assertEqual(EXPECTED_WIRE_SEMANTIC_SHA256, training_config_sha256(wire))
        self.assertEqual(EXPECTED_OCR_SEMANTIC_SHA256, training_config_sha256(ocr))
        self.assertEqual(
            EXPECTED_WIRE_FILE_SHA256,
            hashlib.sha256(WIRE_V5_CONFIG.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            EXPECTED_OCR_FILE_SHA256,
            hashlib.sha256(OCR_V5_CONFIG.read_bytes()).hexdigest(),
        )

        preflight = preflight_training_data(WIRE_V5_CONFIG, workspace_root=ROOT)
        self.assertTrue(preflight["ok"], preflight["issues"])
        self.assertEqual([], preflight["issues"])
        self.assertEqual(
            EXPECTED_WIRE_SEMANTIC_SHA256,
            preflight["training_config_sha256"],
        )
        self.assertEqual(
            str(WIRE_V5_CONFIG.resolve()), preflight["config_path"]
        )
        self.assertEqual(
            wire.dataset.task_manifest_sha256["wire_paths"],
            preflight["tasks"]["wire"]["sha256"],
        )

    def test_v5_resume_lineage_is_disjoint_from_historical_formal4(self) -> None:
        v5 = load_training_config(WIRE_V5_CONFIG)
        formal4 = json.loads(FORMAL4_STATUS.read_text(encoding="utf-8"))
        historical_hash = formal4["lineage"]["training_config_sha256"]

        self.assertEqual(
            "04e7cc22e2405dfdc9e2f34d231e2d56f4d991b9d37fbc3467f07368d15cb0e7",
            historical_hash,
        )
        self.assertNotIn(historical_hash, training_config_resume_sha256s(v5))
        self.assertEqual("random", formal4["lineage"]["initialization"]["mode"])
        self.assertNotEqual(
            formal4["lineage"]["training_config_sha256"],
            EXPECTED_WIRE_SEMANTIC_SHA256,
        )

        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertNotIn("train-wire-formal4", launcher)
        self.assertNotIn("work/train-wire-formal4", WIRE_V5_CONFIG.read_text(encoding="utf-8"))
        self.assertIn("train-wire-v5-fresh-allgate", launcher)

    def test_upstream_requires_exact_ocr_v5_success_and_every_gate(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")

        self.assertIn("train-ocr-v5-allgate\\torch_run_status.json", launcher)
        self.assertIn(EXPECTED_OCR_SEMANTIC_SHA256, launcher)
        self.assertIn(EXPECTED_OCR_FILE_SHA256, launcher)
        self.assertIn('Status.status -eq "candidate_evidence_ready"', launcher)
        self.assertIn("Status.registerable -eq $true", launcher)
        self.assertIn("Status.formal_evidence -eq $true", launcher)
        self.assertIn("Status.gates_passed -eq $true", launcher)
        self.assertIn('Status.task -eq "ocr"', launcher)
        self.assertIn('Status.lineage.task -eq "ocr"', launcher)
        self.assertIn("Test-OcrAllGatePolicy", launcher)
        self.assertIn("Test-OcrPassedTerminal", launcher)
        self.assertIn('return "rejected"', launcher)
        self.assertIn("without passing every bound gate", launcher)

        expected_gate_calls = (
            '(Test-OcrGateResult $gates[0] "normalized_exact_match" 0.65)',
            '(Test-OcrGateResult $gates[1] "detector_hmean" 0.45)',
            '(Test-OcrGateResult $gates[2] "recognition_designator_exact_match" 0.90)',
            '(Test-OcrGateResult $gates[3] "recognition_pinnumber_exact_match" 0.74)',
        )
        for call in expected_gate_calls:
            self.assertIn(call, launcher)
        self.assertIn('selection.comparison.multi_gate -eq "exact_lexicographic_key"', launcher)
        self.assertIn("all_gate_normalized_violation_v1", launcher)
        self.assertIn("raw_violation/max(abs(threshold),1e-12)", launcher)

    def test_wire_launch_resume_pid_and_config_bindings_fail_closed(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")

        self.assertIn(EXPECTED_WIRE_SEMANTIC_SHA256, launcher)
        self.assertIn(EXPECTED_WIRE_FILE_SHA256, launcher)
        self.assertIn("Get-SemanticConfigSha256", launcher)
        self.assertEqual(2, launcher.count("Assert-ConfigBinding `"))
        self.assertIn('"--task", "wire"', launcher)
        self.assertIn('"--resume-checkpoint", $resume', launcher)
        self.assertIn("pcbparse.torch_training_resume.v2", launcher)
        self.assertIn("Test-WireAllGatePolicy", launcher)
        self.assertIn('selection.primary_metric -eq "skeleton_f1"', launcher)
        self.assertIn('[double]$gates[0].threshold -eq 0.55', launcher)
        self.assertIn("rejected v1 resume checkpoint", launcher)
        self.assertIn("its own resume checkpoint", launcher)
        self.assertIn("fresh-v5/v2/all-gate lineage", launcher)

        self.assertIn("Get-RecordedProcessState", launcher)
        for state in ("missing", "active", "inactive", "ambiguous"):
            self.assertIn(f'return "{state}"', launcher)
        self.assertIn(r'Local\pcbparse-wire-v5-$workspaceDigest', launcher)
        self.assertIn("another wire-v5 watcher is already active", launcher)
        self.assertEqual(1, launcher.count("return Start-Process"))
        self.assertEqual(1, launcher.count("$process = Start-BoundPythonProcess"))
        self.assertIn('schema = $pidSchema', launcher)
        self.assertIn('task = "wire"', launcher)
        self.assertIn('upstream_task = "ocr"', launcher)
        self.assertIn("training_config_path = $wireConfig", launcher)
        self.assertIn("training_config_sha256 = $expectedWireTrainingConfigSha256", launcher)
        self.assertIn("training_config_file_sha256 = $expectedWireConfigFileSha256", launcher)
        self.assertIn("upstream_status_path = $ocrStatus", launcher)
        self.assertIn(
            "upstream_training_config_sha256 = $expectedOcrTrainingConfigSha256",
            launcher,
        )
        self.assertIn("resume_schema = $expectedResumeSchema", launcher)
        self.assertIn("model_selection_policy = $expectedSelectionPolicy", launcher)
        self.assertIn("launch_environment_mode = $launchEnvironmentMode", launcher)
        self.assertIn("launch_working_directory = $sourceRoot", launcher)
        self.assertIn('pythonpath_mode = "working_directory_src_only"', launcher)
        self.assertIn("$pidTemporary", launcher)
        self.assertIn("Move-Item -LiteralPath $pidTemporary", launcher)
        self.assertIn("Stop-Process -Id $process.Id", launcher)
        self.assertIn("unhandled launcher error", launcher)

    def test_launch_environment_is_deduplicated_and_never_uses_pythonpath(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")

        self.assertNotIn("$env:PYTHONPATH =", launcher)
        self.assertNotIn('$launchEnvironment["PYTHONPATH"] =', launcher)
        self.assertIn("New-DeduplicatedLaunchEnvironment", launcher)
        self.assertIn(
            "[hashtable]::new([StringComparer]::OrdinalIgnoreCase)", launcher
        )
        self.assertIn(
            '[Environment]::GetEnvironmentVariables("Process")', launcher
        )
        self.assertIn(
            '$name.Equals("Path", [StringComparison]::OrdinalIgnoreCase)', launcher
        )
        self.assertIn('$launchEnvironment["Path"] = $pathValue', launcher)
        self.assertIn(
            '$name.Equals("PYTHONPATH", [StringComparison]::OrdinalIgnoreCase)',
            launcher,
        )
        self.assertIn('-WorkingDirectory $sourceRoot', launcher)
        self.assertIn("-UseNewEnvironment", launcher)
        self.assertIn("-Environment $launchEnvironment", launcher)
        self.assertIn("Invoke-LaunchEnvironmentSmoke", launcher)
        self.assertIn(
            '"-m", "pcbparse.training.torch_train_cli", "--help"', launcher
        )

    def test_executable_launch_environment_smoke_imports_from_unicode_workspace(self) -> None:
        pwsh = shutil.which("pwsh")
        if pwsh is None:
            self.skipTest("PowerShell 7 is unavailable")
        output = ROOT / "work" / "train-wire-v5-fresh-allgate"
        pid_file = ROOT / "work" / "wire-v5.pid.json"
        output_existed = output.exists()
        pid_existed = pid_file.exists()

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
        self.assertIn(
            "WIRE_V5_LAUNCH_ENVIRONMENT_SMOKE_OK", completed.stdout
        )
        self.assertEqual(output_existed, output.exists())
        self.assertEqual(pid_existed, pid_file.exists())


if __name__ == "__main__":
    unittest.main()
