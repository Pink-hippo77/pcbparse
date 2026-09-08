from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "audit_pin_validation_thresholds.py"
)
_SPEC = importlib.util.spec_from_file_location("pin_validation_threshold_audit", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


class PinValidationThresholdAuditTests(unittest.TestCase):
    def test_required_grid_cannot_be_removed_and_extra_values_are_sorted(self) -> None:
        self.assertEqual(
            (0.15, 0.2, 0.3, 0.45, 0.6, 0.75),
            audit._thresholds([0.2, 0.15]),
        )
        with self.assertRaisesRegex(ValueError, r"in \[0, 1\]"):
            audit._thresholds([1.1])

    def test_micro_summary_uses_shared_one_to_one_denominators(self) -> None:
        summary = audit._micro_summary(matched=3, predicted=5, truth=4)
        self.assertEqual(3, summary["matched"])
        self.assertAlmostEqual(0.6, summary["micro_precision_4px"])
        self.assertAlmostEqual(0.75, summary["micro_recall_4px"])
        self.assertAlmostEqual(2.0 / 3.0, summary["micro_f1_4px"])
        self.assertAlmostEqual(1.25, summary["pred_to_gt_ratio"])
        with self.assertRaisesRegex(ValueError, "invalid Pin detection totals"):
            audit._micro_summary(matched=5, predicted=4, truth=6)

    def test_incomplete_or_interim_checkpoint_is_rejected(self) -> None:
        running = {"status": "running"}
        with self.assertRaisesRegex(RuntimeError, "not a completed formal run"):
            audit._require_checkpoint_role(
                running,
                Path("model.pt"),
                interim_warm_start_diagnostic=False,
            )

        complete = {
            "status": "gate_failed",
            "formal_evidence": True,
            "finished_at_utc": "2026-08-23T00:00:00Z",
            "planned_epochs": 30,
            "completed_epochs": 30,
            "early_stopped": False,
        }
        with self.assertRaisesRegex(RuntimeError, "requires model.pt"):
            audit._require_checkpoint_role(
                complete,
                Path("best_checkpoint.pt"),
                interim_warm_start_diagnostic=False,
            )
        self.assertEqual(
            "completed_formal_best_model",
            audit._require_checkpoint_role(
                complete,
                Path("model.pt"),
                interim_warm_start_diagnostic=False,
            ),
        )

    def test_interim_mode_is_explicit_and_requires_formal_epoch_checkpoint(self) -> None:
        interrupted = {
            "status": "running",
            "task": "pin",
            "smoke": False,
            "lineage": {"formal_evidence": True},
        }
        self.assertEqual(
            "interim_formal_epoch_warm_start_diagnostic",
            audit._require_checkpoint_role(
                interrupted,
                Path("checkpoint_epoch_006.pt"),
                interim_warm_start_diagnostic=True,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "checkpoint_epoch_NNN"):
            audit._require_checkpoint_role(
                interrupted,
                Path("best_checkpoint.pt"),
                interim_warm_start_diagnostic=True,
            )


if __name__ == "__main__":
    unittest.main()
