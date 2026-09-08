from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "audit_wire_validation_strata.py"
)
_SPEC = importlib.util.spec_from_file_location("wire_stratified_audit", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_mask(path: Path, mask: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(path)
    return {
        "path": path.as_posix(),
        "sha256": _sha(path),
        "foreground_pixels": int(np.asarray(mask, dtype=bool).sum()),
    }


def _prediction_row(record_id: str, sample_id: str, mask: np.ndarray) -> dict[str, Any]:
    height, width = mask.shape
    packed = np.packbits(np.asarray(mask, dtype=np.uint8).reshape(-1), bitorder="little")
    return {
        "record_id": record_id,
        "sample_id": sample_id,
        "shape": [height, width],
        "encoding": "base64-packbits-little",
        "mask": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def _fixture(root: Path, *, include_test_prediction: bool = False) -> dict[str, Path | None]:
    run_id = "a" * 64
    materialized_run = root / "materialized" / run_id
    paths: dict[str, Path | None] = {
        "predictions_path": root / "predictions.jsonl",
        "status_path": root / "status.json",
        "metrics_path": root / "metrics.json",
        "model_path": root / "model.pt",
        "materialized_run": materialized_run,
        "wire_manifest_path": root / "wire_paths.jsonl",
        "source_manifest_path": root / "source.jsonl",
        "split_manifest_path": root / "split.jsonl",
        "candidate_artifact_path": None,
        "models_lock_path": None,
        "training_config_path": None,
    }
    materialized_run.mkdir(parents=True)
    (root / "model.pt").write_bytes(b"fixture-wire-model")
    runner = root / "src" / "pcbparse" / "training" / "torch_backend.py"
    runner.parent.mkdir(parents=True)
    runner.write_text(
        'WireTorchDataset(run_dir, source_root=source_root, split="train")\n'
        'WireTorchDataset(run_dir, source_root=source_root, split="val")\n',
        encoding="utf-8",
    )
    (runner.parent / "torch_wire_model.py").write_text(
        "# fixture metric implementation\n", encoding="utf-8"
    )

    split_rows = [
        {"sample_id": "train-sample", "split": "train"},
        {"sample_id": "val-sample", "split": "val"},
        {"sample_id": "test-sample", "split": "test"},
    ]
    source_rows = [
        {
            "sample_id": row["sample_id"],
            "source_class": "kicad" if row["split"] != "test" else "orcad",
            "canvas": {"width": 1000, "height": 800},
        }
        for row in split_rows
    ]
    _write_jsonl(paths["split_manifest_path"], split_rows)  # type: ignore[arg-type]
    _write_jsonl(paths["source_manifest_path"], source_rows)  # type: ignore[arg-type]
    _write_jsonl(
        paths["wire_manifest_path"],  # type: ignore[arg-type]
        [
            {"record_id": "train-wire", "sample_id": "train-sample"},
            {"record_id": "val-wire", "sample_id": "val-sample"},
            {"record_id": "test-wire", "sample_id": "test-sample"},
        ],
    )

    empty = np.zeros((8, 8), dtype=bool)
    line = empty.copy()
    line[4, 1:7] = True
    partial = empty.copy()
    partial[4, 1:4] = True
    mask_specs = {
        "train:tile:0": ("train-sample", "train", line, line),
        "val:tile:empty": ("val-sample", "val", empty, empty),
        "val:tile:line": ("val-sample", "val", line, partial),
        "test:tile:0": ("test-sample", "test", line, line),
    }
    tile_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    validation_metrics: list[dict[str, Any]] = []
    for index, (record_id, (sample_id, split, target, prediction)) in enumerate(
        mask_specs.items()
    ):
        relative_target = Path("wire") / "target_masks" / f"{index}.png"
        relative_ignore = Path("wire") / "ignore_masks" / f"{index}.png"
        target_info = _write_mask(materialized_run / relative_target, target)
        ignore_info = _write_mask(materialized_run / relative_ignore, empty)
        target_info["path"] = relative_target.as_posix()
        ignore_info["path"] = relative_ignore.as_posix()
        tile_rows.append(
            {
                "record_id": record_id,
                "sample_id": sample_id,
                "split": split,
                "tile_size": [8, 8],
                "supervision": {"eligible": True},
                "target_mask": target_info,
                "ignore_mask": ignore_info,
            }
        )
        if split == "val" or (include_test_prediction and split == "test"):
            prediction_rows.append(_prediction_row(record_id, sample_id, prediction))
        if split == "val":
            validation_metrics.append(
                {"sample_id": sample_id, **audit.evaluate_masks(prediction, target, empty)}
            )
    wire_tiles = materialized_run / "wire_tiles.jsonl"
    _write_jsonl(wire_tiles, tile_rows)
    _write_jsonl(paths["predictions_path"], prediction_rows)  # type: ignore[arg-type]

    summary = {
        "kind": "wire_junction_materialization",
        "complete": True,
        "run_id": run_id,
        "outputs": {"wire_tiles": {"sha256": _sha(wire_tiles)}},
        "lineage": {
            "inputs": {
                "wire_manifest_sha256": _sha(paths["wire_manifest_path"]),  # type: ignore[arg-type]
                "source_manifest_sha256": _sha(paths["source_manifest_path"]),  # type: ignore[arg-type]
                "split_manifest_sha256": _sha(paths["split_manifest_path"]),  # type: ignore[arg-type]
            }
        },
    }
    summary_path = materialized_run / "summary.json"
    _write_json(summary_path, summary)
    calculated = audit.summarize(validation_metrics)["tile_macro"]
    historical = {
        "centerline_precision": calculated["centerline_precision"],
        "centerline_recall": calculated["centerline_recall"],
        "skeleton_f1": calculated["centerline_f1"],
        "connectivity_score": calculated["connectivity_score"],
        "component_count_error": calculated["component_count_error"],
    }
    _write_json(
        paths["metrics_path"],  # type: ignore[arg-type]
        {"split": "val", "formal_evidence": True, "metrics": historical},
    )
    _write_json(
        paths["status_path"],  # type: ignore[arg-type]
        {
            "task": "wire",
            "status": "candidate_evidence_ready",
            "formal_evidence": True,
            "smoke": False,
            "finished_at_utc": "2026-08-24T00:00:00Z",
            "best_epoch": 1,
            "completed_epochs": 1,
            "gates_passed": True,
            "artifact": {"sha256": _sha(paths["model_path"])},  # type: ignore[arg-type]
            "predictions": {"sha256": _sha(paths["predictions_path"])},  # type: ignore[arg-type]
            "metrics": historical,
            "lineage": {
                "task": "wire",
                "formal_evidence": True,
                "smoke": False,
                "binding_sha256": run_id,
                "materialization_commit_id": run_id,
                "materialization_commit_sha256": _sha(summary_path),
                "source_manifest_sha256": _sha(paths["source_manifest_path"]),  # type: ignore[arg-type]
                "split_manifest_sha256": _sha(paths["split_manifest_path"]),  # type: ignore[arg-type]
                "task_manifest_sha256": _sha(paths["wire_manifest_path"]),  # type: ignore[arg-type]
            },
        },
    )
    return paths


class WireMetricAggregationTests(unittest.TestCase):
    def test_strict_packbits_decoder_rejects_nonzero_padding(self) -> None:
        good = {
            "shape": [1, 1],
            "encoding": "base64-packbits-little",
            "mask": base64.b64encode(b"\x01").decode("ascii"),
        }
        self.assertTrue(bool(audit.decode_prediction_mask(good)[0, 0]))
        bad = dict(good)
        bad["mask"] = base64.b64encode(b"\x02").decode("ascii")
        with self.assertRaisesRegex(ValueError, "padding bits"):
            audit.decode_prediction_mask(bad)

    def test_global_micro_is_count_weighted_not_tile_macro(self) -> None:
        records = [
            {
                "sample_id": "empty",
                "precision_hits": 0,
                "recall_hits": 0,
                "predicted_skeleton_pixels": 0,
                "target_skeleton_pixels": 0,
                "valid_pixels": 100,
                "raw_predicted_positive_pixels": 0,
                "raw_target_positive_pixels": 0,
                "centerline_precision": 1.0,
                "centerline_recall": 1.0,
                "centerline_f1": 1.0,
                "connectivity_score": 1.0,
                "component_count_error": 0.0,
            },
            {
                "sample_id": "line",
                "precision_hits": 5,
                "recall_hits": 5,
                "predicted_skeleton_pixels": 10,
                "target_skeleton_pixels": 5,
                "valid_pixels": 100,
                "raw_predicted_positive_pixels": 10,
                "raw_target_positive_pixels": 5,
                "centerline_precision": 0.5,
                "centerline_recall": 1.0,
                "centerline_f1": 2.0 / 3.0,
                "connectivity_score": 1.0,
                "component_count_error": 0.0,
            },
        ]
        result = audit.summarize(records)
        self.assertAlmostEqual(2.0 / 3.0, result["micro"]["centerline_f1"])
        self.assertAlmostEqual(5.0 / 6.0, result["tile_macro"]["centerline_f1"])
        self.assertEqual(1, result["empty_target_tiles"])

    def test_historical_per_tile_zero_denominator_convention_is_reproduced(self) -> None:
        target = np.zeros((8, 8), dtype=bool)
        prediction = target.copy()
        prediction[4, 1:7] = True
        result = audit.evaluate_masks(prediction, target, target)
        self.assertGreater(result["predicted_skeleton_pixels"], 0)
        self.assertEqual(0.0, result["centerline_precision"])
        self.assertEqual(1.0, result["centerline_recall"])
        self.assertEqual(0.0, result["centerline_f1"])

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "Torch parity check requires the training runtime",
    )
    def test_numpy_skeleton_replay_matches_training_implementation(self) -> None:
        import torch

        from pcbparse.training.torch_wire_model import soft_skeletonize

        generator = np.random.default_rng(20260824)
        for shape in ((3, 5), (8, 8), (11, 7)):
            for _ in range(8):
                values = generator.random(shape) > 0.65
                valid = generator.random(shape) > 0.15
                expected = soft_skeletonize(
                    torch.from_numpy(values.astype(np.float32))[None, None],
                    iterations=8,
                    valid=torch.from_numpy(valid.astype(np.float32))[None, None],
                )[0, 0].numpy()
                actual = audit._soft_skeletonize_numpy(
                    values,
                    iterations=8,
                    valid=valid,
                )
                np.testing.assert_allclose(expected, actual, rtol=0.0, atol=0.0)


class WireStratifiedEvidenceTests(unittest.TestCase):
    def test_hash_bound_build_uses_only_exact_frozen_val_population(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            report = audit.build_audit(workspace=root, **_fixture(root))
        boundary = report["selection_boundary"]
        self.assertEqual(2, boundary["val_eligible_tile_rows_used"])
        self.assertEqual(0, boundary["train_rows_used_for_metrics"])
        self.assertEqual(0, boundary["test_rows_used"])
        self.assertEqual(0, boundary["prediction_rows_overlapping_test"])
        self.assertTrue(
            all(report["lineage_checks"]["historical_tile_macro_reproduction"].values())
        )
        self.assertEqual(1, report["population"]["empty_gt_tiles"])
        self.assertNotEqual(
            report["historical_vs_corrected"]["historical_tile_macro_f1"],
            report["historical_vs_corrected"]["corrected_micro_f1"],
        )
        self.assertEqual(64, len(report["audit_payload_sha256"]))

    def test_test_prediction_is_rejected_instead_of_scored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with self.assertRaisesRegex(ValueError, "exact frozen eligible-val population"):
                audit.build_audit(
                    workspace=root,
                    **_fixture(root, include_test_prediction=True),
                )

    def test_manifest_hash_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            paths = _fixture(root)
            wire_manifest = paths["wire_manifest_path"]
            assert isinstance(wire_manifest, Path)
            with wire_manifest.open("a", encoding="utf-8") as handle:
                handle.write(" \n")
            with self.assertRaisesRegex(ValueError, "wire_manifest/summary SHA"):
                audit.build_audit(workspace=root, **paths)


if __name__ == "__main__":
    unittest.main()
