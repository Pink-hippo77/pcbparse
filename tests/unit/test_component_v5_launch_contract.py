from __future__ import annotations

import json
import unittest
from pathlib import Path

from pcbparse.training.config import (
    load_training_config,
    training_config_resume_sha256s,
    training_config_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
OCR_V4 = ROOT / "configs" / "train" / "dataset300_v2_ocr_v4_rare_prefix.json"
COMPONENT_V5 = (
    ROOT / "configs" / "train" / "dataset300_v2_component_v5_fullmix.json"
)
OCR_V4_PREFLIGHT = ROOT / "work" / "preflight-ocr-v4-rare-prefix.json"
COMPONENT_V5_PREFLIGHT = ROOT / "work" / "preflight-component-v5-fullmix.json"
LAUNCHER = ROOT / "scripts" / "auto_launch_component_v5.ps1"


class ComponentV5LaunchContractTests(unittest.TestCase):
    def test_preflights_bind_exact_upstream_run_and_component_config(self) -> None:
        ocr_v4 = load_training_config(OCR_V4)
        component_v5 = load_training_config(COMPONENT_V5)
        ocr_preflight = json.loads(OCR_V4_PREFLIGHT.read_text(encoding="utf-8"))
        component_preflight = json.loads(
            COMPONENT_V5_PREFLIGHT.read_text(encoding="utf-8")
        )

        self.assertTrue(ocr_preflight["ok"])
        self.assertTrue(component_preflight["ok"])
        self.assertEqual([], ocr_preflight["issues"])
        self.assertEqual([], component_preflight["issues"])
        self.assertIn(
            ocr_preflight["training_config_sha256"],
            training_config_resume_sha256s(ocr_v4),
            "the live OCR-v4 run hash must remain a known exact lineage",
        )
        self.assertEqual(
            training_config_sha256(component_v5),
            component_preflight["training_config_sha256"],
        )
        self.assertEqual(
            [("bbox_map50", "gte", 0.25)],
            [
                (gate.metric, gate.operator, gate.threshold)
                for gate in component_v5.tasks["component"].evaluation_gates
            ],
        )

    def test_upstream_terminal_is_exact_formal_ocr_v4_only(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        ocr_preflight = json.loads(OCR_V4_PREFLIGHT.read_text(encoding="utf-8"))
        current_ocr_config_sha = training_config_sha256(load_training_config(OCR_V4))

        self.assertIn(ocr_preflight["training_config_sha256"], launcher)
        if current_ocr_config_sha != ocr_preflight["training_config_sha256"]:
            self.assertNotIn(
                current_ocr_config_sha,
                launcher,
                "the watcher must bind the live OCR run, not a later schema hash",
            )
        self.assertIn("train-ocr-v4-rare-prefix", launcher)
        self.assertIn('status -in @("candidate_evidence_ready", "gate_failed")', launcher)
        self.assertIn('$status.task -eq "ocr"', launcher)
        self.assertIn('$status.smoke -eq $false', launcher)
        self.assertIn('$status.formal_evidence -eq $true', launcher)
        self.assertIn('$status.lineage.task -eq "ocr"', launcher)
        self.assertIn('$status.lineage.smoke -eq $false', launcher)
        self.assertIn('$status.lineage.formal_evidence -eq $true', launcher)
        self.assertIn(
            "$status.lineage.training_config_sha256 -eq "
            "$expectedOcrTrainingConfigSha256",
            launcher,
        )

    def test_component_is_task_config_v2_and_single_gate_policy_bound(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        component_v5 = load_training_config(COMPONENT_V5)

        self.assertIn(training_config_sha256(component_v5), launcher)
        self.assertIn('"--task", "component"', launcher)
        self.assertIn('Status.task -eq "component"', launcher)
        self.assertIn('Status.lineage.task -eq "component"', launcher)
        self.assertIn("pcbparse.torch_training_resume.v2", launcher)
        self.assertIn("all_gate_normalized_violation_v1", launcher)
        self.assertIn("legacy_primary_metric_direction_with_absolute_epsilon", launcher)
        self.assertIn('expectedPrimaryMetric = "bbox_map50"', launcher)
        self.assertIn('expectedPrimaryOperator = "gte"', launcher)
        self.assertIn("expectedGateThreshold = 0.25", launcher)
        self.assertIn("gates.Count -eq 1", launcher)
        self.assertIn("Test-ComponentAllGatePolicy", launcher)
        self.assertIn('"--resume-checkpoint"', launcher)
        self.assertIn("rejected v1 resume checkpoint", launcher)
        self.assertIn("predates the all-gate model-selection policy", launcher)

    def test_launcher_has_four_pid_states_workspace_mutex_and_one_launch(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")

        self.assertIn("Get-RecordedProcessState", launcher)
        for state in ("missing", "active", "inactive", "ambiguous"):
            self.assertIn(f'return "{state}"', launcher)
        self.assertIn("reports running without a provably inactive", launcher)
        self.assertIn("component-v5 PID metadata is ambiguous", launcher)
        self.assertIn("component-v5.pid.json", launcher)
        self.assertIn("training_config_sha256 = $expectedComponentTrainingConfigSha256", launcher)
        self.assertIn("resume_schema = $expectedResumeSchema", launcher)
        self.assertIn("model_selection_policy = $expectedSelectionPolicy", launcher)
        self.assertIn("[Threading.Mutex]", launcher)
        self.assertIn(r"Local\pcbparse-component-v5-$workspaceDigest", launcher)
        self.assertIn("another component-v5 watcher is already active", launcher)
        self.assertIn("$pidTemporary", launcher)
        self.assertIn("Move-Item -LiteralPath $pidTemporary", launcher)
        self.assertIn("unhandled launcher error", launcher)
        self.assertEqual(1, launcher.count("$process = Start-Process"))


if __name__ == "__main__":
    unittest.main()
