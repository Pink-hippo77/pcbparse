from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "evaluate_yolo_component_tiled_validation.py"
)
if str(_SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("yolo_component_eval", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


class YoloTrainingCompletionTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
        results = root / "results.csv"
        with results.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)"],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "epoch": 1,
                        "metrics/mAP50(B)": 0.70,
                        "metrics/mAP50-95(B)": 0.45,
                    },
                    {
                        "epoch": 2,
                        "metrics/mAP50(B)": 0.75,
                        "metrics/mAP50-95(B)": 0.50,
                    },
                ]
            )
        checkpoint = root / "best.pt"
        checkpoint.write_bytes(b"checkpoint")
        config = root / "train.json"
        config.write_text("{}\n", encoding="utf-8")
        status = root / "launcher_status.json"
        status.write_text(
            json.dumps(
                {
                    "kind": "pcbparse_yolo_component_training_launcher_status",
                    "status": "training_complete",
                    "ok": True,
                    "mode": "train",
                    "models_lock_modified": False,
                    "configured_epochs": 10,
                    "completed_epochs": 2,
                    "stopped_before_epoch_limit": True,
                    "epoch_indexing": "results_csv_one_based",
                    "best_epoch": 2,
                    "best_epoch_zero_based": 1,
                    "best_map50": 0.75,
                    "best_map50_95": 0.50,
                    "training_config": {"sha256": audit.base._sha256(config)},
                    "artifacts": {
                        "results.csv": {"sha256": audit.base._sha256(results)},
                        "weights/best.pt": {
                            "sha256": audit.base._sha256(checkpoint)
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        return results, checkpoint, config, status

    def test_accepts_hash_bound_launcher_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results, checkpoint, config, status = self._fixture(Path(directory))
            completion = audit._training_completion(
                results,
                None,
                status_path=status,
                checkpoint_path=checkpoint,
                training_config_path=config,
            )
        self.assertEqual(2, completion["completed_epochs"])
        self.assertEqual(2, completion["best_epoch"])
        self.assertTrue(completion["early_stopping_confirmed"])
        self.assertEqual("launcher_status", completion["completion_evidence"])

    def test_rejects_tampered_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results, checkpoint, config, status = self._fixture(Path(directory))
            results.write_text(results.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "results.csv hash mismatch"):
                audit._training_completion(
                    results,
                    None,
                    status_path=status,
                    checkpoint_path=checkpoint,
                    training_config_path=config,
                )

    def test_accepts_legacy_launcher_epoch_fields_from_inflight_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results, checkpoint, config, status = self._fixture(Path(directory))
            payload = json.loads(status.read_text(encoding="utf-8"))
            payload.pop("epoch_indexing")
            payload.pop("best_epoch")
            payload["completed_epochs"] = 3
            payload["best_epoch_zero_based"] = 2
            status.write_text(json.dumps(payload), encoding="utf-8")
            completion = audit._training_completion(
                results,
                None,
                status_path=status,
                checkpoint_path=checkpoint,
                training_config_path=config,
            )
        self.assertEqual(2, completion["completed_epochs"])
        self.assertEqual(2, completion["best_epoch"])
        self.assertTrue(completion["launcher_epoch_fields_normalized"])


if __name__ == "__main__":
    unittest.main()
