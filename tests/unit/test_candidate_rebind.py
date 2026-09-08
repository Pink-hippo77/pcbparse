from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from pcbparse.training.artifacts import (
    CandidateBundleError,
    build_candidate_lock_entry,
    build_inference_contract,
    finalize_candidate_bundle,
)
from pcbparse.training.config import (
    load_training_config,
    training_config_resume_sha256s,
    training_config_sha256,
)
from pcbparse.training.rebind import (
    CandidateRebindError,
    rebind_legacy_candidate_bundle,
)


PREDICTIONS = b'{"sample_id":"fixture","prediction":[]}\n'


class CandidateRebindTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repository_root = Path(__file__).resolve().parents[2]
        self.config_path = (
            self.repository_root
            / "configs"
            / "train"
            / "dataset300_v2_pin_v5_boundary.json"
        )
        self.config = load_training_config(self.config_path)
        module = types.ModuleType("candidate_rebind_fixture_adapter")
        module.load = lambda path, entry: object()
        sys.modules[module.__name__] = module
        self.addCleanup(sys.modules.pop, module.__name__, None)

    def test_legacy_candidate_is_rebound_without_changing_evidence_bytes(self) -> None:
        source = self._make_legacy_source()
        destination = self.root / "rebound"

        result = rebind_legacy_candidate_bundle(
            source / "bundle.manifest.json",
            self.config_path,
            destination,
        )

        self.assertTrue(result["ok"])
        self.assertFalse(result["blessed"])
        self.assertFalse(result["models_lock_modified"])
        manifest = destination / "bundle.manifest.json"
        entry = build_candidate_lock_entry(manifest, lock_directory=destination)
        self.assertEqual("candidate", entry["status"])
        self.assertEqual([384, 384], entry["inference"]["input_size"])
        self.assertEqual(0.75, entry["inference"]["score_threshold"])
        self.assertEqual("boundary_band_v2", entry["inference"]["box_prior_mode"])
        self.assertEqual(
            _sha(source / "model.onnx"), _sha(destination / "model.onnx")
        )
        self.assertEqual(
            _sha(source / "predictions.jsonl"),
            _sha(destination / "predictions.jsonl"),
        )
        audit = json.loads(
            (destination / "rebind.audit.json").read_text(encoding="utf-8")
        )
        self.assertTrue(audit["checks"]["candidate_lock_entry_buildable"])
        self.assertFalse(audit["boundaries"]["model_retrained"])
        self.assertFalse(audit["boundaries"]["predictions_recomputed"])

    def test_source_prediction_tamper_and_existing_destination_fail_closed(self) -> None:
        source = self._make_legacy_source()
        (source / "predictions.jsonl").write_bytes(b"tampered")
        with self.assertRaisesRegex(CandidateRebindError, "predictions SHA-256"):
            rebind_legacy_candidate_bundle(
                source / "bundle.manifest.json",
                self.config_path,
                self.root / "blocked",
            )

        destination = self.root / "already-exists"
        destination.mkdir()
        with self.assertRaisesRegex(CandidateRebindError, "refusing to overwrite"):
            rebind_legacy_candidate_bundle(
                source / "bundle.manifest.json",
                self.config_path,
                destination,
            )

    def test_resume_compatible_historical_config_hash_is_accepted(self) -> None:
        source = self._make_legacy_source()
        compatible = training_config_resume_sha256s(self.config)
        historical_hash = compatible[-1]
        self.assertNotEqual(training_config_sha256(self.config), historical_hash)
        card_path = source / "model_card.json"
        manifest_path = source / "bundle.manifest.json"
        card = json.loads(card_path.read_text(encoding="utf-8"))
        card["training_config_sha256"] = historical_hash
        card_path.write_text(
            json.dumps(card, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["training_config_sha256"] = historical_hash
        manifest["model_card_sha256"] = _sha(card_path)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

        destination = self.root / "historical-rebound"
        rebind_legacy_candidate_bundle(
            manifest_path,
            self.config_path,
            destination,
        )

        audit = json.loads(
            (destination / "rebind.audit.json").read_text(encoding="utf-8")
        )
        self.assertFalse(audit["training_config"]["exact_semantic_match"])
        self.assertTrue(audit["training_config"]["resume_compatible"])
        self.assertEqual(
            historical_hash,
            audit["training_config"]["source_candidate_sha256"],
        )

    def test_lock_entry_rejects_rehashed_card_lineage_divergence(self) -> None:
        source = self._make_legacy_source()
        destination = self.root / "rebound-for-card-tamper"
        rebind_legacy_candidate_bundle(
            source / "bundle.manifest.json",
            self.config_path,
            destination,
        )
        card_path = destination / "model_card.json"
        manifest_path = destination / "bundle.manifest.json"
        card = json.loads(card_path.read_text(encoding="utf-8"))
        card["source_manifest_sha256"] = "f" * 64
        card_path.write_text(
            json.dumps(card, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["model_card_sha256"] = _sha(card_path)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(CandidateBundleError, "lineage differs"):
            build_candidate_lock_entry(manifest_path)

    def test_finalizer_rejects_model_as_prediction_evidence(self) -> None:
        source = self._make_legacy_source()
        artifact = source / "model.onnx"
        manifest = json.loads(
            (source / "bundle.manifest.json").read_text(encoding="utf-8")
        )
        card = json.loads((source / "model_card.json").read_text(encoding="utf-8"))
        evaluation = dict(manifest["evaluation"])
        evaluation["predictions_sha256"] = _sha(artifact)

        with self.assertRaisesRegex(
            CandidateBundleError, "must be different files"
        ):
            finalize_candidate_bundle(
                source,
                artifact_path=artifact,
                evaluation_predictions_path=artifact,
                task="pin",
                version=manifest["version"],
                license_name=manifest["license"],
                precision=manifest["precision"],
                adapter_entrypoint=manifest["adapter_smoke"]["entrypoint"],
                inference=build_inference_contract(
                    "pin", self.config.tasks["pin"]
                ),
                training_config_sha256=manifest["training_config_sha256"],
                dataset_config_hash=manifest["dataset_config_hash"],
                source_manifest_sha256=manifest["source_manifest_sha256"],
                split_manifest_sha256=manifest["split_manifest_sha256"],
                gate_policy=manifest["gate_policy"],
                export_consistency=manifest["export_consistency"],
                evaluation=evaluation,
                intended_use=card["intended_use"],
                limitations=card["limitations"],
                out_of_scope_use=card["out_of_scope_use"],
            )

    def _make_legacy_source(self) -> Path:
        source = self.root / "legacy"
        source.mkdir()
        artifact = source / "model.onnx"
        artifact.write_bytes(b"immutable-pin-model")
        predictions = source / "predictions.jsonl"
        predictions.write_bytes(PREDICTIONS)
        task = self.config.tasks["pin"]
        task_manifest_sha = self.config.dataset.task_manifest_sha256[
            task.dataset_task
        ]
        manifest_path = finalize_candidate_bundle(
            source,
            artifact_path=artifact,
            evaluation_predictions_path=predictions,
            task="pin",
            version="pin-fixture-v1",
            license_name="test-only",
            precision="fp32",
            adapter_entrypoint="candidate_rebind_fixture_adapter:load",
            inference={
                "device": "auto",
                "input_size": [384, 384],
                "output_stride": 4,
                "score_threshold": 0.75,
                "nms_kernel": 3,
                "nms_radius_heatmap": 2.0,
                "max_detections_per_component": 128,
                "base_length_pixels": 8.0,
                "box_margin_px": 4.0,
                "box_prior_mode": "boundary_band_v2",
            },
            training_config_sha256=training_config_sha256(self.config),
            dataset_config_hash=self.config.dataset.config_hash,
            source_manifest_sha256=self.config.dataset.source_manifest_sha256,
            split_manifest_sha256=self.config.dataset.split_manifest_sha256,
            gate_policy={
                "task_manifest_sha256": task_manifest_sha,
                "export_min_samples": task.export_min_samples,
                "export_max_abs_error": task.export_max_abs_error,
                "evaluation_gates": [
                    gate.model_dump(mode="json") for gate in task.evaluation_gates
                ],
            },
            export_consistency={
                "reference_backend": "fixture-torch",
                "export_backend": "fixture-onnx",
                "sample_count": task.export_min_samples,
                "minimum_samples": task.export_min_samples,
                "max_abs_error": 0.01,
                "tolerance": task.export_max_abs_error,
                "input_manifest_sha256": task_manifest_sha,
                "passed": True,
            },
            evaluation={
                "evaluator": "fixture-pin",
                "evaluator_version": "1",
                "split": "val",
                "dataset_config_hash": self.config.dataset.config_hash,
                "split_manifest_sha256": self.config.dataset.split_manifest_sha256,
                "predictions_sha256": hashlib.sha256(PREDICTIONS).hexdigest(),
                "metrics": {
                    "pin_f1_4px": 0.64,
                    "pin_precision_4px": 0.85,
                    "pin_recall_4px": 0.51,
                    "pred_to_gt_ratio": 0.60,
                },
                "gates": [
                    {
                        "metric": gate.metric,
                        "operator": gate.operator,
                        "threshold": gate.threshold,
                        "value": (
                            0.64
                            if gate.metric == "pin_f1_4px"
                            else 0.85
                            if gate.metric == "pin_precision_4px"
                            else 0.51
                            if gate.metric == "pin_recall_4px"
                            else 0.60
                        ),
                        "passed": True,
                    }
                    for gate in task.evaluation_gates
                ],
            },
            intended_use="fixture",
            limitations=["fixture only"],
        )
        card_path = source / "model_card.json"
        card = json.loads(card_path.read_text(encoding="utf-8"))
        card.pop("inference")
        card.pop("evaluation_predictions_path")
        card_path.write_text(
            json.dumps(card, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.pop("inference")
        manifest.pop("evaluation_predictions_path")
        manifest["model_card_sha256"] = _sha(card_path)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(CandidateBundleError):
            build_candidate_lock_entry(manifest_path)
        return source


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
