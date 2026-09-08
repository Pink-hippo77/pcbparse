from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "audit_ocr_detector_threshold.py"
)
_SPEC = importlib.util.spec_from_file_location("ocr_detector_threshold_audit", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


class OCRDetectorThresholdAuditTests(unittest.TestCase):
    def test_refinement_grid_is_predeclared_and_contains_baseline(self) -> None:
        thresholds = audit._thresholds_around(0.325)
        self.assertIn(0.3, thresholds)
        self.assertIn(0.325, thresholds)
        self.assertIn(0.35, thresholds)
        self.assertIn(0.5, thresholds)
        self.assertTrue(set(audit.COARSE_THRESHOLDS).issubset(thresholds))
        self.assertEqual(tuple(sorted(set(thresholds))), thresholds)

    def test_selection_tie_breaks_without_test_information(self) -> None:
        rows = [
            {"threshold": 0.30, "hmean": 0.5, "recall": 0.6, "precision": 0.45},
            {"threshold": 0.40, "hmean": 0.5, "recall": 0.6, "precision": 0.50},
            {"threshold": 0.50, "hmean": 0.5, "recall": 0.55, "precision": 0.70},
        ]
        self.assertEqual(0.40, audit._select_best(rows)["threshold"])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            audit._select_best([rows[0], copy.deepcopy(rows[0])])

    def test_micro_metrics_are_recomputed_from_integer_counts(self) -> None:
        summary = audit._aggregate_counts(
            {
                "true_positives": 3,
                "false_positives": 2,
                "false_negatives": 1,
                "predictions": 5,
                "targets": 4,
                "ignored_predictions": 0,
                "mean_matched_iou": 0.75,
                # These deliberately wrong values must not be trusted.
                "precision": 0.0,
                "recall": 0.0,
                "hmean": 0.0,
            }
        )
        self.assertAlmostEqual(0.6, summary["precision"])
        self.assertAlmostEqual(0.75, summary["recall"])
        self.assertAlmostEqual(2.0 / 3.0, summary["hmean"])

    def test_checkpoint_role_rejects_running_model_pt_and_allows_explicit_epoch(self) -> None:
        running = {
            "task": "ocr",
            "smoke": False,
            "lineage": {"formal_evidence": True},
            "status": "running",
        }
        with self.assertRaisesRegex(RuntimeError, "not a completed formal run"):
            audit._checkpoint_role(running, Path("model.pt"), interim=False)
        self.assertEqual(
            "interim_formal_epoch_diagnostic",
            audit._checkpoint_role(
                running, Path("checkpoint_epoch_007.pt"), interim=True
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "checkpoint_epoch_NNN"):
            audit._checkpoint_role(running, Path("best_checkpoint.pt"), interim=True)

    def test_selection_verifier_rejects_any_test_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"model")
            document = audit._with_payload_hash(
                {
                    "schema_version": 1,
                    "kind": audit.SELECTION_KIND,
                    "status": "validation_threshold_selected",
                    "selection_boundary": {
                        "split": "val",
                        "test_tiles_loaded": 0,
                        "test_labels_scored": 0,
                        "test_thresholds_searched": 1,
                    },
                    "selected_threshold": 0.35,
                    "checkpoint": {
                        "path": "model.pt",
                        "sha256": audit._sha256_file(checkpoint),
                    },
                }
            )
            selection = root / "selection.json"
            selection.write_text(
                json.dumps(document, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "not validation-only"):
                audit._verify_selection(root, selection)

    def test_payload_hash_and_probability_file_tamper_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            source_names = (
                "config.json",
                "status.json",
                "model.pt",
                "manifest.json",
                "detection.jsonl",
                "audit.py",
                "model.py",
                "dataset.py",
            )
            paths = []
            for name in source_names:
                path = root / name
                path.write_bytes(name.encode("utf-8"))
                paths.append(path)
            source_keys = (
                "training_config",
                "run_status",
                "checkpoint",
                "materialization_manifest",
                "detection_index",
                "audit_script",
                "detector_evaluator",
                "detection_dataset",
            )
            sources = {
                key: {"path": path.name, "sha256": audit._sha256_file(path)}
                for key, path in zip(source_keys, paths, strict=True)
            }
            probabilities = np.zeros((1, 2, 2), dtype=np.float32)
            probability_path = cache_dir / "probabilities.npy"
            np.save(probability_path, probabilities, allow_pickle=False)
            tile_ids = ["tile-1"]
            tile_path = cache_dir / "tile_ids.json"
            tile_path.write_text(json.dumps(tile_ids), encoding="utf-8")
            manifest = audit._with_payload_hash(
                {
                    "schema_version": 1,
                    "kind": audit.CACHE_KIND,
                    "status": "complete",
                    "split": "val",
                    "model_forward_device": "cpu",
                    "labels_scored_while_caching": False,
                    "threshold_search_while_caching": False,
                    "sources": sources,
                    "coverage": {"tiles": 1},
                    "probability_file": {
                        "path": "probabilities.npy",
                        "sha256": audit._sha256_file(probability_path),
                        "shape": [1, 2, 2],
                    },
                    "tile_id_file": {
                        "path": "tile_ids.json",
                        "sha256": audit._sha256_file(tile_path),
                        "canonical_sha256": audit._canonical_sha256(tile_ids),
                    },
                }
            )
            (cache_dir / "cache.manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            audit._verify_cache(root, cache_dir, expected_split="val")
            with probability_path.open("r+b") as handle:
                handle.seek(-1, 2)
                last = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([last[0] ^ 0x01]))
            with self.assertRaisesRegex(ValueError, "probabilities SHA-256"):
                audit._verify_cache(root, cache_dir, expected_split="val")


if __name__ == "__main__":
    unittest.main()
