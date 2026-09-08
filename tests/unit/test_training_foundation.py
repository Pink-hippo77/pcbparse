from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from pcbparse.training.artifacts import (
    CandidateBundleError,
    build_candidate_lock_entry,
    build_inference_contract,
    finalize_candidate_bundle,
    load_candidate_bundle,
)
from pcbparse.training.config import (
    MODEL_TASKS,
    PretrainedArtifact,
    load_training_config,
    training_config_resume_sha256s,
    training_config_sha256,
)
from pcbparse.training.preflight import preflight_training_data
from pcbparse.training.runner import TrainingRunnerError, run_configured_training
from pcbparse.training.torch_train_cli import main as torch_train_main


DATASET_TASKS = {
    "component": "component_detection",
    "pin": "pin_keypoints",
    "ocr": "ocr_text",
    "wire": "wire_paths",
    "junction": "junction_patches",
}
_PREDICTIONS_BYTES = b'{"fixture_prediction":true}\n'


class TrainingFoundationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def test_preflight_verifies_hashes_sources_records_and_zero_leakage(self) -> None:
        config_path = _make_fixture(self.root)

        report = preflight_training_data(config_path, workspace_root=self.root)

        self.assertTrue(report["ok"], report["issues"])
        self.assertEqual(6, report["dataset"]["source_files"]["verified"])
        self.assertEqual(set(MODEL_TASKS), set(report["tasks"]))
        self.assertTrue(all(item["invalid_records"] == 0 for item in report["tasks"].values()))
        self.assertTrue(all(check["ok"] for check in report["checks"]))

    def test_preflight_detects_source_tampering_and_group_leakage(self) -> None:
        config_path = _make_fixture(self.root)
        (self.root / "source" / "001" / "image.png").write_bytes(b"tampered")
        source_path = self.root / "source.jsonl"
        source_rows = [
            json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines()
        ]
        source_rows[1]["duplicate_group"] = source_rows[0]["duplicate_group"]
        _write_jsonl(source_path, source_rows)
        split_path = self.root / "split.jsonl"
        rows = [json.loads(line) for line in split_path.read_text(encoding="utf-8").splitlines()]
        rows[1]["duplicate_group"] = rows[0]["duplicate_group"]
        _write_jsonl(split_path, rows)

        report = preflight_training_data(config_path, workspace_root=self.root)

        self.assertFalse(report["ok"])
        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("source_hash_mismatch", codes)
        self.assertIn("duplicate_group_leakage", codes)
        self.assertIn("split_manifest_hash_mismatch", codes)

    def test_shipped_config_has_five_4gb_tiny_plans_but_no_fake_runner(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        config_path = repository_root / "configs" / "train" / "dataset300_v2_tiny.json"

        config = load_training_config(config_path)

        self.assertEqual(set(MODEL_TASKS), set(config.tasks))
        self.assertLessEqual(config.runtime.max_vram_mb, 4096)
        self.assertTrue(all(task.precision == "fp16" for task in config.tasks.values()))
        self.assertTrue(all(task.runner_entrypoint is None for task in config.tasks.values()))
        self.assertTrue(all(task.evaluation_gates for task in config.tasks.values()))

    def test_shipped_torch_config_wires_real_backends_and_pins_pretraining(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        config = load_training_config(
            repository_root / "configs" / "train" / "dataset300_v2_torch.json"
        )

        self.assertEqual(set(MODEL_TASKS), set(config.tasks))
        for task in ("component", "pin", "ocr", "wire"):
            self.assertEqual(
                "pcbparse.training.torch_backend:run_torch_task",
                config.tasks[task].runner_entrypoint,
            )
        self.assertEqual(
            "pcbparse.training.torch_backend:run_torch_task",
            config.tasks["junction"].runner_entrypoint,
        )
        self.assertGreater(config.tasks["junction"].junction_synthetic_per_class, 0)
        pretrained = config.tasks["component"].pretrained_artifact
        self.assertIsNotNone(pretrained)
        assert pretrained is not None
        self.assertEqual(
            pretrained.sha256,
            _sha(repository_root / pretrained.path),
        )

    def test_pretrained_artifact_rejects_absolute_or_parent_paths(self) -> None:
        common = {
            "sha256": "a" * 64,
            "source": "fixture",
            "license": "fixture",
        }
        with self.assertRaises(ValueError):
            PretrainedArtifact(path="../escape.pt", **common)
        with self.assertRaises(ValueError):
            PretrainedArtifact(path=str(self.root / "absolute.pt"), **common)

    def test_resume_hash_accepts_only_noop_additive_schema_defaults(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        base = load_training_config(
            repository_root / "configs" / "train" / "dataset300_v2_pin_v4.json"
        )

        compatible = training_config_resume_sha256s(base)

        self.assertEqual(training_config_sha256(base), compatible[0])
        self.assertEqual(5, len(compatible))
        self.assertEqual(
            "c5be64859df587bf9693ebe9ee29881accf8a7acd4976dfcf91612e24e161d6f",
            compatible[1],
        )
        self.assertEqual(
            "5407b1f305470a27575d74eaff55d8b92b80b5c6481f02b373e4186dc8a15269",
            compatible[2],
        )
        self.assertEqual(
            "69c94a67132ee5c540f2336178ab9d36f703168f899198102e1b0b5c9b1a7ab4",
            compatible[3],
        )
        self.assertEqual(
            "96dafdb639a4939ef4bc4e9342171dc83256dc81be69a9c21f5b38dc3350224a",
            compatible[4],
        )

        changed_document = base.model_dump(mode="json")
        changed_document["tasks"]["ocr"]["ocr_target_shrink_ratio"] = 0.1
        changed = type(base).model_validate(changed_document)
        changed_compatible = training_config_resume_sha256s(changed)
        self.assertEqual(training_config_sha256(changed), changed_compatible[0])
        # The three newest schema generations may be removed, but the non-default
        # OCR target policy prevents matching the oldest pre-policy schema.
        self.assertEqual(4, len(changed_compatible))
        self.assertNotIn(compatible[-1], changed_compatible)

    def test_runner_without_real_backend_fails_without_claiming_training(self) -> None:
        config_path = _make_fixture(self.root)

        with self.assertRaisesRegex(TrainingRunnerError, "no model has been trained"):
            run_configured_training(
                config_path,
                task="component",
                output_directory=self.root / "run",
                workspace_root=self.root,
            )

    def test_torch_cli_reports_candidate_binding_failure_without_traceback(self) -> None:
        stderr = io.StringIO()
        with patch(
            "pcbparse.training.torch_train_cli.load_training_config",
            side_effect=CandidateBundleError("prediction bytes drifted"),
        ), redirect_stderr(stderr):
            result = torch_train_main(
                [
                    "--config",
                    "train.json",
                    "--task",
                    "component",
                    "--output",
                    "candidate",
                    "--workspace-root",
                    str(self.root),
                ]
            )

        self.assertEqual(2, result)
        failure = json.loads(stderr.getvalue())
        self.assertEqual("failed", failure["status"])
        self.assertEqual("CandidateBundleError", failure["error"]["type"])
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_candidate_bundle_requires_passed_gates_hashes_and_live_adapter(self) -> None:
        module = types.ModuleType("training_fixture_adapter")
        module.load = lambda path, entry: object()
        sys.modules[module.__name__] = module
        self.addCleanup(sys.modules.pop, module.__name__, None)
        bundle = self.root / "bundle"
        bundle.mkdir()
        artifact = bundle / "component.onnx"
        artifact.write_bytes(b"real-exported-model-bytes")
        predictions = _write_predictions(bundle)
        manifest = finalize_candidate_bundle(
            bundle,
            artifact_path=artifact,
            evaluation_predictions_path=predictions,
            task="component",
            version="component-v1",
            license_name="test-only",
            precision="fp16",
            adapter_entrypoint="training_fixture_adapter:load",
            inference=_component_inference(),
            training_config_sha256="a" * 64,
            dataset_config_hash="b" * 64,
            source_manifest_sha256="c" * 64,
            split_manifest_sha256="d" * 64,
            gate_policy=_gate_policy(),
            export_consistency=_export_evidence(),
            evaluation=_evaluation_evidence(),
            intended_use="fixture component detection",
            limitations=["fixture evidence only"],
        )

        entry = build_candidate_lock_entry(manifest, lock_directory=self.root)
        manifest_document = json.loads(manifest.read_text(encoding="utf-8"))
        card_document = json.loads(
            (bundle / "model_card.json").read_text(encoding="utf-8")
        )

        self.assertEqual("candidate", entry["status"])
        self.assertIs(entry["blessed"], False)
        self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), entry["sha256"])
        self.assertEqual(_component_inference(), entry["inference"])
        self.assertEqual(
            "predictions.jsonl",
            manifest_document["evaluation_predictions_path"],
        )
        self.assertEqual(
            manifest_document["evaluation_predictions_path"],
            card_document["evaluation_predictions_path"],
        )
        self.assertFalse((self.root / "models.lock.json").exists())
        predictions.write_bytes(b"changed-prediction-evidence")
        with self.assertRaisesRegex(CandidateBundleError, "SHA-256"):
            build_candidate_lock_entry(manifest)
        predictions.write_bytes(_PREDICTIONS_BYTES)
        artifact.write_bytes(b"changed-after-finalization")
        with self.assertRaisesRegex(CandidateBundleError, "byte size|SHA-256"):
            build_candidate_lock_entry(manifest)

    def test_candidate_bundle_hash_binds_evidence_attachments(self) -> None:
        module = types.ModuleType("attachment_fixture_adapter")
        module.load = lambda path, entry: object()
        sys.modules[module.__name__] = module
        self.addCleanup(sys.modules.pop, module.__name__, None)
        bundle = self.root / "attachment-bundle"
        bundle.mkdir()
        artifact = bundle / "component.onnx"
        artifact.write_bytes(b"attachment-model")
        predictions = _write_predictions(bundle)
        evidence = bundle / "independent-test.json"
        evidence.write_bytes(b'{"status":"pass"}\n')
        manifest = finalize_candidate_bundle(
            bundle,
            artifact_path=artifact,
            evaluation_predictions_path=predictions,
            task="component",
            version="component-attachment-v1",
            license_name="test-only",
            precision="fp16",
            adapter_entrypoint="attachment_fixture_adapter:load",
            inference=_component_inference(),
            training_config_sha256="a" * 64,
            dataset_config_hash="b" * 64,
            source_manifest_sha256="c" * 64,
            split_manifest_sha256="d" * 64,
            gate_policy=_gate_policy(),
            export_consistency=_export_evidence(),
            evaluation=_evaluation_evidence(),
            intended_use="fixture component detection",
            limitations=["fixture evidence only"],
            evidence_attachments=[
                {
                    "kind": "independent_test_confirmation",
                    "path": evidence.name,
                    "sha256": _sha(evidence),
                }
            ],
        )

        entry = build_candidate_lock_entry(manifest)
        self.assertEqual("candidate", entry["status"])
        document = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(
            "independent_test_confirmation",
            document["evidence_attachments"][0]["kind"],
        )
        evidence.write_bytes(b'{"status":"tampered"}\n')
        with self.assertRaisesRegex(CandidateBundleError, "evidence attachment.*SHA-256"):
            build_candidate_lock_entry(manifest)

    def test_candidate_bundle_accepts_lower_and_upper_bounds_for_one_metric(self) -> None:
        module = types.ModuleType("bounded_gate_fixture_adapter")
        module.load = lambda path, entry: object()
        sys.modules[module.__name__] = module
        self.addCleanup(sys.modules.pop, module.__name__, None)
        bundle = self.root / "bounded-gate-bundle"
        bundle.mkdir()
        artifact = bundle / "pin.onnx"
        artifact.write_bytes(b"bounded-gate-model")
        predictions = _write_predictions(bundle)
        policy = _gate_policy()
        policy["evaluation_gates"] = [
            {
                "metric": "pred_to_gt_ratio",
                "operator": "gte",
                "threshold": 0.45,
            },
            {
                "metric": "pred_to_gt_ratio",
                "operator": "lte",
                "threshold": 0.70,
            },
        ]
        evaluation = _evaluation_evidence()
        evaluation["metrics"] = {"pred_to_gt_ratio": 0.60}
        evaluation["gates"] = [
            {
                "metric": "pred_to_gt_ratio",
                "operator": "gte",
                "threshold": 0.45,
                "value": 0.60,
                "passed": True,
            },
            {
                "metric": "pred_to_gt_ratio",
                "operator": "lte",
                "threshold": 0.70,
                "value": 0.60,
                "passed": True,
            },
        ]

        manifest = finalize_candidate_bundle(
            bundle,
            artifact_path=artifact,
            evaluation_predictions_path=predictions,
            task="pin",
            version="pin-bounded-gate-v1",
            license_name="test-only",
            precision="fp32",
            adapter_entrypoint="bounded_gate_fixture_adapter:load",
            inference=_pin_inference(),
            training_config_sha256="a" * 64,
            dataset_config_hash="b" * 64,
            source_manifest_sha256="c" * 64,
            split_manifest_sha256="d" * 64,
            gate_policy=policy,
            export_consistency=_export_evidence(),
            evaluation=evaluation,
            intended_use="fixture pin detection",
            limitations=["fixture evidence only"],
        )

        document = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(2, len(document["gate_policy"]["evaluation_gates"]))
        self.assertEqual(2, len(document["evaluation"]["gates"]))

    def test_runtime_inference_contract_is_derived_from_hashed_task_config(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        config = load_training_config(
            repository_root / "configs" / "train" / "dataset300_v2_pin_v5_boundary.json"
        )

        pin = build_inference_contract("pin", config.tasks["pin"])
        component = build_inference_contract("component", config.tasks["component"])

        self.assertEqual([384, 384], pin["input_size"])
        self.assertEqual(0.75, pin["score_threshold"])
        self.assertEqual("boundary_band_v2", pin["box_prior_mode"])
        self.assertEqual(4.0, pin["box_margin_px"])
        self.assertEqual(0.0, component["score_threshold"])
        self.assertEqual(300, component["max_detections"])

        rectangular_pin_config = config.tasks["pin"].model_copy(
            update={"input_size": (256, 384)}
        )
        rectangular_pin = build_inference_contract(
            "pin", rectangular_pin_config
        )
        self.assertEqual(
            [384, 256],
            rectangular_pin["input_size"],
            "training uses (height,width), while the Pin adapter uses (width,height)",
        )

    def test_legacy_bundle_without_inference_contract_is_readable_but_not_lockable(self) -> None:
        module = types.ModuleType("legacy_fixture_adapter")
        module.load = lambda path, entry: object()
        sys.modules[module.__name__] = module
        self.addCleanup(sys.modules.pop, module.__name__, None)
        bundle = self.root / "legacy-bundle"
        bundle.mkdir()
        artifact = bundle / "component.onnx"
        artifact.write_bytes(b"legacy-model")
        predictions = _write_predictions(bundle)
        manifest_path = finalize_candidate_bundle(
            bundle,
            artifact_path=artifact,
            evaluation_predictions_path=predictions,
            task="component",
            version="component-legacy-v1",
            license_name="test-only",
            precision="fp16",
            adapter_entrypoint="legacy_fixture_adapter:load",
            inference=_component_inference(),
            training_config_sha256="a" * 64,
            dataset_config_hash="b" * 64,
            source_manifest_sha256="c" * 64,
            split_manifest_sha256="d" * 64,
            gate_policy=_gate_policy(),
            export_consistency=_export_evidence(),
            evaluation=_evaluation_evidence(),
            intended_use="fixture",
            limitations=["fixture"],
        )
        card_path = bundle / "model_card.json"
        card = json.loads(card_path.read_text(encoding="utf-8"))
        card.pop("inference")
        card_path.write_text(
            json.dumps(card, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.pop("inference")
        manifest["model_card_sha256"] = hashlib.sha256(card_path.read_bytes()).hexdigest()
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(CandidateBundleError, "lacks a hash-bound"):
            build_candidate_lock_entry(manifest_path)

    def test_legacy_bundle_without_prediction_path_is_readable_but_not_lockable(self) -> None:
        module = types.ModuleType("legacy_prediction_fixture_adapter")
        module.load = lambda path, entry: object()
        sys.modules[module.__name__] = module
        self.addCleanup(sys.modules.pop, module.__name__, None)
        bundle = self.root / "legacy-prediction-bundle"
        bundle.mkdir()
        artifact = bundle / "component.onnx"
        artifact.write_bytes(b"legacy-prediction-model")
        predictions = _write_predictions(bundle)
        manifest_path = finalize_candidate_bundle(
            bundle,
            artifact_path=artifact,
            evaluation_predictions_path=predictions,
            task="component",
            version="component-legacy-prediction-v1",
            license_name="test-only",
            precision="fp16",
            adapter_entrypoint="legacy_prediction_fixture_adapter:load",
            inference=_component_inference(),
            training_config_sha256="a" * 64,
            dataset_config_hash="b" * 64,
            source_manifest_sha256="c" * 64,
            split_manifest_sha256="d" * 64,
            gate_policy=_gate_policy(),
            export_consistency=_export_evidence(),
            evaluation=_evaluation_evidence(),
            intended_use="fixture",
            limitations=["fixture"],
        )
        card_path = bundle / "model_card.json"
        card = json.loads(card_path.read_text(encoding="utf-8"))
        card.pop("evaluation_predictions_path")
        card_path.write_text(
            json.dumps(card, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.pop("evaluation_predictions_path")
        manifest["model_card_sha256"] = hashlib.sha256(
            card_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

        loaded = load_candidate_bundle(manifest_path)
        self.assertIsNone(loaded.evaluation_predictions_path)
        with self.assertRaisesRegex(
            CandidateBundleError, "lacks hash-bound evaluation prediction bytes"
        ):
            build_candidate_lock_entry(manifest_path)

    def test_failed_metric_or_missing_adapter_cannot_create_bundle(self) -> None:
        bundle = self.root / "blocked"
        bundle.mkdir()
        artifact = bundle / "wire.onnx"
        artifact.write_bytes(b"wire")
        predictions = _write_predictions(bundle)
        failed_evaluation = _evaluation_evidence()
        failed_evaluation["metrics"]["bbox_map50"] = 0.1
        failed_evaluation["gates"][0]["value"] = 0.1
        failed_evaluation["gates"][0]["passed"] = False

        with self.assertRaises(ValueError):
            finalize_candidate_bundle(
                bundle,
                artifact_path=artifact,
                evaluation_predictions_path=predictions,
                task="wire",
                version="wire-v1",
                license_name="test-only",
                precision="fp16",
                adapter_entrypoint="missing_training_adapter:load",
                inference=_wire_inference(),
                training_config_sha256="a" * 64,
                dataset_config_hash="b" * 64,
                source_manifest_sha256="c" * 64,
                split_manifest_sha256="d" * 64,
                gate_policy=_gate_policy(),
                export_consistency=_export_evidence(),
                evaluation=failed_evaluation,
                intended_use="fixture",
                limitations=["fixture"],
            )

        with self.assertRaisesRegex(CandidateBundleError, "adapter import smoke failed"):
            finalize_candidate_bundle(
                bundle,
                artifact_path=artifact,
                evaluation_predictions_path=predictions,
                task="wire",
                version="wire-v1",
                license_name="test-only",
                precision="fp16",
                adapter_entrypoint="missing_training_adapter:load",
                inference=_wire_inference(),
                training_config_sha256="a" * 64,
                dataset_config_hash="b" * 64,
                source_manifest_sha256="c" * 64,
                split_manifest_sha256="d" * 64,
                gate_policy=_gate_policy(),
                export_consistency=_export_evidence(),
                evaluation=_evaluation_evidence(),
                intended_use="fixture",
                limitations=["fixture"],
            )


def _component_inference() -> dict:
    return {
        "device": "auto",
        "score_threshold": 0.2,
        "max_detections": 300,
    }


def _pin_inference() -> dict:
    return {
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
    }


def _wire_inference() -> dict:
    return {
        "device": "auto",
        "threshold": 0.5,
        "batch_size": 4,
        "output_kind": "logits",
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
    }


def _make_fixture(root: Path) -> Path:
    source_root = root / "source"
    derived_root = root / "derived"
    source_root.mkdir()
    derived_root.mkdir()
    splits = ("train", "val", "test")
    source_rows = []
    split_rows = []
    for index, split in enumerate(splits, start=1):
        sample_id = f"{index:03d}"
        sample_root = source_root / sample_id
        sample_root.mkdir()
        image = sample_root / "image.png"
        annotation = sample_root / "annotation.json"
        image.write_bytes(f"image-{sample_id}".encode())
        annotation.write_text("{}", encoding="utf-8")
        group = f"group-{sample_id}"
        source_rows.append(
            {
                "sample_id": sample_id,
                "duplicate_group": group,
                "files": {
                    "image": {
                        "path": f"{sample_id}/image.png",
                        "sha256": _sha(image),
                    },
                    "json": {
                        "path": f"{sample_id}/annotation.json",
                        "sha256": _sha(annotation),
                    },
                },
            }
        )
        split_rows.append(
            {"sample_id": sample_id, "duplicate_group": group, "split": split}
        )
    source_manifest = root / "source.jsonl"
    split_manifest = root / "split.jsonl"
    _write_jsonl(source_manifest, source_rows)
    _write_jsonl(split_manifest, split_rows)
    split_hash = _sha(split_manifest)
    config_hash = "1" * 64
    manifest_hashes: dict[str, str] = {}
    split_counts = {task: {split: 1 for split in splits} for task in DATASET_TASKS.values()}
    counts = {}
    task_files = {}
    for dataset_task in DATASET_TASKS.values():
        path = derived_root / f"{dataset_task}.jsonl"
        rows = []
        for source, split in zip(source_rows, split_rows, strict=True):
            rows.append(
                {
                    "record_id": f"{source['sample_id']}:{dataset_task}",
                    "task": dataset_task,
                    "derivation_version": "fixture-v2",
                    "config_hash": config_hash,
                    "sample_id": source["sample_id"],
                    "source_image": source["files"]["image"]["path"],
                    "source_json": source["files"]["json"]["path"],
                    "duplicate_group": source["duplicate_group"],
                    "split": split["split"],
                }
            )
        _write_jsonl(path, rows)
        manifest_hashes[dataset_task] = _sha(path)
        counts[dataset_task] = len(rows)
        task_files[dataset_task] = path.name
    summary = {
        "schema_version": 1,
        "derivation_version": "fixture-v2",
        "config_hash": config_hash,
        "sample_count": 3,
        "config": {"split_manifest_sha256": split_hash},
        "counts": counts,
        "split_counts": split_counts,
        "task_files": task_files,
    }
    summary_path = derived_root / "summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    tasks = {}
    for model_task, dataset_task in DATASET_TASKS.items():
        tasks[model_task] = {
            "dataset_task": dataset_task,
            "architecture": "fixture_tiny",
            "runner_entrypoint": None,
            "epochs": 1,
            "batch_size": 1,
            "gradient_accumulation": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "input_size": [32, 32],
            "precision": "fp16",
            "export_format": "onnx",
            "export_opset": 17,
            "export_max_abs_error": 0.001,
            "export_min_samples": 1,
            "evaluation_gates": [
                {"metric": "score", "operator": "gte", "threshold": 0.1}
            ],
        }
    config = {
        "schema_version": 1,
        "name": "fixture",
        "dataset": {
            "derived_root": "derived",
            "summary": "derived/summary.json",
            "source_root": "source",
            "source_manifest": "source.jsonl",
            "split_manifest": "split.jsonl",
            "derivation_version": "fixture-v2",
            "expected_sample_count": 3,
            "config_hash": config_hash,
            "summary_sha256": _sha(summary_path),
            "source_manifest_sha256": _sha(source_manifest),
            "split_manifest_sha256": split_hash,
            "task_manifest_sha256": manifest_hashes,
        },
        "runtime": {
            "seed": 1,
            "deterministic_algorithms": True,
            "max_vram_mb": 1024,
            "dataloader_workers": 0,
            "pin_memory": False,
            "checkpoint_every_epochs": 1,
            "early_stopping_patience": 1,
        },
        "tasks": tasks,
    }
    config_path = root / "train.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    return config_path


def _export_evidence() -> dict[str, object]:
    return {
        "reference_backend": "fixture-reference",
        "export_backend": "fixture-onnx",
        "sample_count": 16,
        "minimum_samples": 16,
        "max_abs_error": 0.0001,
        "tolerance": 0.001,
        "input_manifest_sha256": "e" * 64,
        "passed": True,
    }


def _gate_policy() -> dict[str, object]:
    return {
        "task_manifest_sha256": "e" * 64,
        "export_min_samples": 16,
        "export_max_abs_error": 0.001,
        "evaluation_gates": [
            {"metric": "bbox_map50", "operator": "gte", "threshold": 0.25}
        ],
    }


def _evaluation_evidence() -> dict[str, object]:
    return {
        "evaluator": "fixture-evaluator",
        "evaluator_version": "1.0",
        "split": "val",
        "dataset_config_hash": "b" * 64,
        "split_manifest_sha256": "d" * 64,
        "predictions_sha256": hashlib.sha256(_PREDICTIONS_BYTES).hexdigest(),
        "metrics": {"bbox_map50": 0.5},
        "gates": [
            {
                "metric": "bbox_map50",
                "operator": "gte",
                "threshold": 0.25,
                "value": 0.5,
                "passed": True,
            }
        ],
    }


def _write_predictions(bundle: Path) -> Path:
    path = bundle / "predictions.jsonl"
    path.write_bytes(_PREDICTIONS_BYTES)
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    )
    path.write_text(payload, encoding="utf-8", newline="\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
