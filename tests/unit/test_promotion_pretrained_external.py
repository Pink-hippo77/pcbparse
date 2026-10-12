from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from data_tools import model_promotion as promotion
from pcbparse.training.artifacts import (
    CandidateBundleError,
    CandidateBundleManifest,
    build_candidate_lock_entry,
    finalize_candidate_bundle,
    is_onnx_two_stage_backend,
    is_torchscript_two_stage_backend,
    load_candidate_bundle,
    validate_inference_contract,
    validate_source_repository_commit,
)


ADAPTER_MODULE = "pretrained_external_fixture_adapters"
_PREDICTIONS_BYTES = b'{"fixture_prediction":true}\n'
_DATASET_CONFIG_HASH = "b" * 64
_SOURCE_MANIFEST_SHA256 = "c" * 64
_SPLIT_MANIFEST_SHA256 = "d" * 64
_SOURCE_COMMIT = "PaddlePaddle/PaddleOCR@dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf"


class _StubProvider:
    """Fixture provider: empty typed lists satisfy the fixed protocol fixture."""

    def __init__(self, task: str) -> None:
        self.task = task
        self.version = "pretrained-external-fixture-stub/v1"

    def predict(self, *args: object) -> list:
        return []

    def recognize(self, *args: object) -> list:
        return []

    def segment(self, *args: object) -> list:
        return []


class PretrainedExternalPromotionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        module = types.ModuleType(ADAPTER_MODULE)
        for task in ("component", "pin", "wire", "ocr"):
            setattr(
                module,
                f"{task}_factory",
                (lambda name: (lambda path, entry: _StubProvider(name)))(task),
            )
        sys.modules[ADAPTER_MODULE] = module
        self.addCleanup(sys.modules.pop, ADAPTER_MODULE, None)

    # ------------------------------------------------------------------
    # descriptor bundle: legitimate shape
    # ------------------------------------------------------------------

    def test_descriptor_bundle_verifies_and_pins_external_source(self) -> None:
        manifest_path = self._make_descriptor_bundle()

        bundle = load_candidate_bundle(manifest_path)
        entry = build_candidate_lock_entry(manifest_path, lock_directory=self.root)

        self.assertEqual("pretrained_external", bundle.lineage_mode)
        self.assertIsNone(bundle.training_config_sha256)
        self.assertEqual(_SOURCE_COMMIT, bundle.source_repository_commit)
        self.assertEqual("apache-2.0", bundle.source_license)
        self.assertEqual(2, len(bundle.source_asset_sha256))
        self.assertEqual("candidate", entry["status"])
        self.assertIs(entry["blessed"], False)
        self.assertEqual("pretrained_external", entry["lineage_mode"])
        self.assertEqual(_SOURCE_COMMIT, entry["source_repository_commit"])
        self.assertEqual("apache-2.0", entry["source_license"])
        self.assertIsNone(entry["training_config_sha256"])
        self.assertEqual(
            sorted(bundle.source_asset_sha256), sorted(entry["source_asset_sha256"])
        )
        descriptor = json.loads(
            (manifest_path.parent / "ocr-medium-bundle.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            sorted(bundle.source_asset_sha256),
            sorted([descriptor["detector"]["sha256"], descriptor["recognizer"]["sha256"]]),
        )

    # ------------------------------------------------------------------
    # missing source pinning
    # ------------------------------------------------------------------

    def test_missing_source_commit_or_license_is_rejected(self) -> None:
        manifest_path = self._make_descriptor_bundle()

        for field_name in ("source_repository_commit", "source_license"):
            broken = _load_manifest_document(manifest_path)
            broken.pop(field_name)
            with self.assertRaisesRegex(ValueError, field_name):
                CandidateBundleManifest.model_validate(broken)
            broken = _load_manifest_document(manifest_path)
            broken[field_name] = None
            with self.assertRaisesRegex(ValueError, field_name):
                CandidateBundleManifest.model_validate(broken)

        empty_assets = _load_manifest_document(manifest_path)
        empty_assets["source_asset_sha256"] = []
        with self.assertRaisesRegex(ValueError, "pinned source asset"):
            CandidateBundleManifest.model_validate(empty_assets)

        repeated_assets = _load_manifest_document(manifest_path)
        digest = repeated_assets["source_asset_sha256"][0]
        repeated_assets["source_asset_sha256"] = [digest, digest]
        with self.assertRaisesRegex(ValueError, "source_asset_sha256"):
            CandidateBundleManifest.model_validate(repeated_assets)

    def test_source_commit_must_pin_a_full_revision(self) -> None:
        for bad in (
            "PaddlePaddle/PaddleOCR",
            "PaddlePaddle/PaddleOCR@main",
            "PaddlePaddle/PaddleOCR@dab3fe3",
            "dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf",
            "unknown",
            None,
        ):
            with self.assertRaises(ValueError):
                validate_source_repository_commit(bad)  # type: ignore[arg-type]
        self.assertEqual(
            _SOURCE_COMMIT, validate_source_repository_commit(_SOURCE_COMMIT)
        )

    def test_placeholder_source_text_is_rejected(self) -> None:
        manifest_path = self._make_descriptor_bundle()
        for placeholder in ("", "unknown", "TBD", "todo", "none", "unlicensed"):
            broken = _load_manifest_document(manifest_path)
            broken["source_license"] = placeholder
            with self.assertRaisesRegex(ValueError, "source_license"):
                CandidateBundleManifest.model_validate(broken)

        review = self._review_document(manifest_path)
        for field_name, placeholder in (
            ("reviewer", "todo"),
            ("review_id", "unknown"),
            ("source_license", "tbd"),
        ):
            broken = dict(review)
            broken[field_name] = placeholder
            with self.assertRaisesRegex(ValueError, field_name):
                promotion.PretrainedExternalReviewEvidence.model_validate(broken)

    # ------------------------------------------------------------------
    # hash binding
    # ------------------------------------------------------------------

    def test_tampered_asset_bytes_are_rejected(self) -> None:
        manifest_path = self._make_descriptor_bundle()
        (manifest_path.parent / "assets" / "det.onnx").write_bytes(b"tampered-detector")

        with self.assertRaisesRegex(CandidateBundleError, "SHA-256"):
            build_candidate_lock_entry(manifest_path)

    def test_source_asset_hashes_must_match_descriptor_binding(self) -> None:
        manifest_path = self._make_descriptor_bundle()
        broken = _load_manifest_document(manifest_path)
        broken["source_asset_sha256"] = ["9" * 64, "8" * 64]

        self._rewrite_manifest(manifest_path, broken)
        with self.assertRaisesRegex(CandidateBundleError, "source_asset_sha256"):
            build_candidate_lock_entry(manifest_path)

    def test_descriptor_asset_binding_cannot_drift_from_attachments(self) -> None:
        manifest_path = self._make_descriptor_bundle()
        broken = _load_manifest_document(manifest_path)
        broken["evidence_attachments"] = [
            attachment
            for attachment in broken["evidence_attachments"]
            if attachment["kind"] != "detector_onnx"
        ]

        self._rewrite_manifest(manifest_path, broken)
        with self.assertRaisesRegex(CandidateBundleError, "evidence attachments"):
            build_candidate_lock_entry(manifest_path)

    def test_review_evidence_must_bind_the_candidate_manifest(self) -> None:
        manifest_path = self._make_descriptor_bundle()
        review_path = self.root / "review.json"
        document = self._review_document(manifest_path)
        document["candidate_manifest_sha256"] = "a" * 64
        review_path.write_text(json.dumps(document), encoding="utf-8")

        review = promotion._load_pretrained_external_review(review_path)
        bundle = promotion._verify_bundle(manifest_path, self.root)
        with self.assertRaisesRegex(promotion.ModelPromotionError, "bind"):
            promotion._bind_pretrained_external_review(review, bundle)

        document = self._review_document(manifest_path)
        document["source_asset_sha256"] = ["7" * 64]
        review_path.write_text(json.dumps(document), encoding="utf-8")
        review = promotion._load_pretrained_external_review(review_path)
        with self.assertRaisesRegex(promotion.ModelPromotionError, "source_asset_sha256"):
            promotion._bind_pretrained_external_review(review, bundle)

    def test_review_evidence_requires_reviewer_attestation_fields(self) -> None:
        manifest_path = self._make_descriptor_bundle()
        base = self._review_document(manifest_path)
        for field_name in ("reviewer_confirmed", "provenance_verified"):
            broken = dict(base)
            broken.pop(field_name)
            with self.assertRaises(ValueError):
                promotion.PretrainedExternalReviewEvidence.model_validate(broken)
            broken = dict(base)
            broken[field_name] = False
            with self.assertRaises(ValueError):
                promotion.PretrainedExternalReviewEvidence.model_validate(broken)
        broken = dict(base)
        broken["reviewed_at_utc"] = "2026-10-07"
        with self.assertRaisesRegex(ValueError, "timezone"):
            promotion.PretrainedExternalReviewEvidence.model_validate(broken)

    # ------------------------------------------------------------------
    # dataset300-trained behavior stays exactly as before
    # ------------------------------------------------------------------

    def test_dataset300_trained_contract_is_unchanged(self) -> None:
        manifest_path = self._make_trained_bundle("component", "component-trained")

        bundle = load_candidate_bundle(manifest_path)
        entry = build_candidate_lock_entry(manifest_path, lock_directory=self.root)

        self.assertEqual("dataset300-trained", bundle.lineage_mode)
        self.assertEqual("a" * 64, bundle.training_config_sha256)
        self.assertIsNone(bundle.source_repository_commit)
        self.assertIsNone(bundle.source_license)
        self.assertEqual((), bundle.source_asset_sha256)
        self.assertEqual("dataset300-trained", entry["lineage_mode"])

        # Manifests written before the contract extension keep loading.
        legacy = _load_manifest_document(manifest_path)
        for field_name in (
            "lineage_mode",
            "source_repository_commit",
            "source_asset_sha256",
            "source_license",
        ):
            legacy.pop(field_name)
        reloaded = CandidateBundleManifest.model_validate(legacy)
        self.assertEqual("dataset300-trained", reloaded.lineage_mode)

    def test_dataset300_trained_rejects_missing_or_external_lineage(self) -> None:
        manifest_path = self._make_trained_bundle("pin", "pin-trained")
        broken = _load_manifest_document(manifest_path)
        broken["training_config_sha256"] = None
        with self.assertRaisesRegex(ValueError, "training_config_sha256"):
            CandidateBundleManifest.model_validate(broken)
        broken = _load_manifest_document(manifest_path)
        broken["training_config_sha256"] = "not-a-digest"
        with self.assertRaisesRegex(ValueError, "training_config_sha256"):
            CandidateBundleManifest.model_validate(broken)
        broken = _load_manifest_document(manifest_path)
        broken["source_repository_commit"] = _SOURCE_COMMIT
        with self.assertRaisesRegex(ValueError, "source pinning"):
            CandidateBundleManifest.model_validate(broken)

    def test_pretrained_external_rejects_fabricated_training_config(self) -> None:
        manifest_path = self._make_descriptor_bundle()
        broken = _load_manifest_document(manifest_path)
        broken["training_config_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "training config"):
            CandidateBundleManifest.model_validate(broken)

    # ------------------------------------------------------------------
    # export semantics and inference contract
    # ------------------------------------------------------------------

    def test_pretrained_external_requires_onnx_two_stage_export_semantics(self) -> None:
        manifest_path = self._make_descriptor_bundle()
        self.assertTrue(is_onnx_two_stage_backend("onnx-two-stage"))
        self.assertTrue(is_onnx_two_stage_backend("onnxruntime-two-stage"))
        self.assertFalse(is_onnx_two_stage_backend("torchscript-two-stage"))
        self.assertTrue(is_torchscript_two_stage_backend("torchscript-two-stage"))

        for backend in ("torchscript-two-stage", "torchscript", "fixture-onnx"):
            broken = _load_manifest_document(manifest_path)
            broken["export_consistency"]["export_backend"] = backend
            with self.assertRaisesRegex(ValueError, "onnx-two-stage"):
                CandidateBundleManifest.model_validate(broken)
        broken = _load_manifest_document(manifest_path)
        broken["export_consistency"]["export_backend"] = ""
        with self.assertRaises(ValueError):
            CandidateBundleManifest.model_validate(broken)

        for backend in ("onnx-two-stage", "onnxruntime-two-stage"):
            fixed = _load_manifest_document(manifest_path)
            fixed["export_consistency"]["export_backend"] = backend
            CandidateBundleManifest.model_validate(fixed)

    def test_inference_contract_separates_trained_and_descriptor_shapes(self) -> None:
        trained = validate_inference_contract("ocr", _trained_ocr_inference())
        descriptor = validate_inference_contract(
            "ocr", _descriptor_ocr_inference(), lineage_mode="pretrained_external"
        )
        self.assertEqual(_trained_ocr_inference(), trained)
        self.assertEqual(_descriptor_ocr_inference(), descriptor)

        with self.assertRaisesRegex(CandidateBundleError, "keys differ"):
            validate_inference_contract(
                "ocr", _trained_ocr_inference(), lineage_mode="pretrained_external"
            )
        with self.assertRaisesRegex(CandidateBundleError, "keys differ"):
            validate_inference_contract("ocr", _descriptor_ocr_inference())
        with self.assertRaisesRegex(CandidateBundleError, "not defined"):
            validate_inference_contract(
                "component",
                {"device": "auto"},
                lineage_mode="pretrained_external",
            )
        for field_name, bad_value in (
            ("detector_threshold", 0.0),
            ("detector_threshold", 1.0),
            ("box_threshold", 1.5),
            ("unclip_ratio", 0.0),
            ("max_candidates", 0),
            ("recognition_batch_size", 65),
            ("cpu_threads", 0),
            ("device", "tpu"),
        ):
            broken = _descriptor_ocr_inference()
            broken[field_name] = bad_value
            with self.assertRaises(CandidateBundleError):
                validate_inference_contract(
                    "ocr", broken, lineage_mode="pretrained_external"
                )

    # ------------------------------------------------------------------
    # promotion keeps the human-review requirement
    # ------------------------------------------------------------------

    def test_promotion_requires_pretrained_external_human_review(self) -> None:
        manifests = self._make_four_task_set()
        output = self.root / "lock.json"
        report = self.root / "report.json"

        with self.assertRaisesRegex(
            promotion.ModelPromotionError, "human review evidence"
        ):
            promotion.assemble_model_lock(
                manifests,
                output=output,
                promote=True,
                junction_mode="deterministic_vectorizer",
                report_output=report,
            )
        self.assertFalse(output.exists())

        review_path = self.root / "external-review.json"
        review_path.write_text(
            json.dumps(self._review_document(manifests[-1])), encoding="utf-8"
        )
        with self.assertRaisesRegex(
            promotion.ModelPromotionError, "only accepted with promote=True"
        ):
            promotion.assemble_model_lock(
                manifests,
                output=output,
                promote=False,
                junction_mode="deterministic_vectorizer",
                pretrained_external_review_evidence=[review_path],
                report_output=report,
            )
        self.assertFalse(output.exists())

    def test_promotion_binds_review_and_blesses_only_with_it(self) -> None:
        manifests = self._make_four_task_set()
        output = self.root / "lock.json"
        report = self.root / "report.json"
        review_path = self.root / "external-review.json"
        review_path.write_text(
            json.dumps(self._review_document(manifests[-1])), encoding="utf-8"
        )

        summary = promotion.assemble_model_lock(
            manifests,
            output=output,
            promote=True,
            junction_mode="deterministic_vectorizer",
            pretrained_external_review_evidence=[review_path],
            report_output=report,
        )

        self.assertTrue(summary["ok"])
        lock = json.loads(output.read_text(encoding="utf-8"))
        by_task = {entry["task"]: entry for entry in lock["models"]}
        ocr_entry = by_task["ocr"]
        self.assertEqual("blessed", ocr_entry["status"])
        self.assertIs(ocr_entry["blessed"], True)
        self.assertEqual("pretrained_external", ocr_entry["lineage_mode"])
        self.assertEqual(_SOURCE_COMMIT, ocr_entry["source_repository_commit"])
        self.assertEqual(
            hashlib.sha256(review_path.read_bytes()).hexdigest(),
            ocr_entry["pretrained_external_review_evidence_sha256"],
        )
        self.assertEqual("dataset300-trained", by_task["component"]["lineage_mode"])
        self.assertNotIn(
            "pretrained_external_review_evidence_sha256", by_task["component"]
        )
        self.assertNotIn("human_review_evidence_sha256", ocr_entry)
        report_document = json.loads(report.read_text(encoding="utf-8"))
        self.assertTrue(report_document["checks"]["pretrained_external_human_review"])
        self.assertTrue(
            report_document["checks"][
                "pretrained_external_source_pin_and_descriptor_binding"
            ]
        )
        self.assertEqual(1, len(report_document["pretrained_external_reviews"]))
        self.assertFalse((self.root / "models.lock.json").exists())

    # ------------------------------------------------------------------
    # fixtures
    # ------------------------------------------------------------------

    def _make_four_task_set(self) -> list[Path]:
        return [
            self._make_trained_bundle("component", "component-trained"),
            self._make_trained_bundle("pin", "pin-trained"),
            self._make_trained_bundle("wire", "wire-trained"),
            self._make_descriptor_bundle(),
        ]

    def _make_trained_bundle(self, task: str, name: str) -> Path:
        bundle = self.root / name
        bundle.mkdir(parents=True)
        artifact = bundle / f"{task}.onnx"
        artifact.write_bytes(f"fixture-{task}-model-bytes".encode("utf-8"))
        predictions = _write_predictions(bundle)
        return finalize_candidate_bundle(
            bundle,
            artifact_path=artifact,
            evaluation_predictions_path=predictions,
            task=task,
            version=f"{task}-fixture-v1",
            license_name="test-only",
            precision="fp32",
            adapter_entrypoint=f"{ADAPTER_MODULE}:{task}_factory",
            inference=_TRAINED_INFERENCE[task](),
            training_config_sha256="a" * 64,
            dataset_config_hash=_DATASET_CONFIG_HASH,
            source_manifest_sha256=_SOURCE_MANIFEST_SHA256,
            split_manifest_sha256=_SPLIT_MANIFEST_SHA256,
            gate_policy=_gate_policy(),
            export_consistency=_export_evidence(),
            evaluation=_evaluation_evidence(),
            intended_use="fixture trained candidate",
            limitations=["fixture evidence only"],
        )

    def _make_descriptor_bundle(self) -> Path:
        bundle = self.root / "ocr-descriptor"
        (bundle / "assets").mkdir(parents=True)
        detector = bundle / "assets" / "det.onnx"
        recognizer = bundle / "assets" / "rec.onnx"
        detector.write_bytes(b"fixture-detector-onnx-bytes")
        recognizer.write_bytes(b"fixture-recognizer-onnx-bytes")
        detector_sha256 = _sha(detector)
        recognizer_sha256 = _sha(recognizer)
        descriptor = {
            "schema_version": 1,
            "architecture": "db_ctc_onnx",
            "detector": {"path": "assets/det.onnx", "sha256": detector_sha256},
            "recognizer": {"path": "assets/rec.onnx", "sha256": recognizer_sha256},
            "characters": ["", "A", "B"],
            "preprocess": {"color_order": "BGR"},
            "provenance": {
                "promoted": False,
                "dataset300_v2_trained": False,
                "source_repositories": [
                    {
                        "repository": "PaddlePaddle/PP-OCRv6_medium_det_onnx",
                        "revision": "61323801669c338b7891481ec7bac61ce31b576a",
                        "path": "assets/det.onnx",
                        "sha256": detector_sha256,
                        "official_lfs_sha256": detector_sha256,
                        "license": "apache-2.0",
                    },
                    {
                        "repository": "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
                        "revision": "50c7eacafc52fa7bcf4194e8cd08e46f8558504b",
                        "path": "assets/rec.onnx",
                        "sha256": recognizer_sha256,
                        "official_lfs_sha256": recognizer_sha256,
                        "license": "apache-2.0",
                    },
                ],
            },
        }
        descriptor_path = bundle / "ocr-medium-bundle.json"
        descriptor_path.write_text(
            json.dumps(descriptor, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        descriptor_sha256 = _sha(descriptor_path)
        predictions = _write_predictions(bundle)
        return finalize_candidate_bundle(
            bundle,
            artifact_path=descriptor_path,
            evaluation_predictions_path=predictions,
            task="ocr",
            version="pp-ocr-medium-fixture-v1",
            license_name="apache-2.0",
            precision="fp32",
            adapter_entrypoint=f"{ADAPTER_MODULE}:ocr_factory",
            inference=_descriptor_ocr_inference(),
            training_config_sha256=None,
            dataset_config_hash=_DATASET_CONFIG_HASH,
            source_manifest_sha256=_SOURCE_MANIFEST_SHA256,
            split_manifest_sha256=_SPLIT_MANIFEST_SHA256,
            gate_policy={
                "task_manifest_sha256": descriptor_sha256,
                "export_min_samples": 2,
                "export_max_abs_error": 0.0,
                "evaluation_gates": [
                    {
                        "metric": "designator_accuracy",
                        "operator": "gte",
                        "threshold": 0.5,
                    }
                ],
            },
            export_consistency={
                "reference_backend": "upstream-lfs-sha256",
                "export_backend": "onnx-two-stage",
                "sample_count": 2,
                "minimum_samples": 2,
                "max_abs_error": 0.0,
                "tolerance": 0.0,
                "input_manifest_sha256": descriptor_sha256,
                "passed": True,
            },
            evaluation={
                "evaluator": "fixture-raw-v1",
                "evaluator_version": "1",
                "split": "val",
                "dataset_config_hash": _DATASET_CONFIG_HASH,
                "split_manifest_sha256": _SPLIT_MANIFEST_SHA256,
                "predictions_sha256": hashlib.sha256(_PREDICTIONS_BYTES).hexdigest(),
                "metrics": {"designator_accuracy": 0.75},
                "gates": [
                    {
                        "metric": "designator_accuracy",
                        "operator": "gte",
                        "threshold": 0.5,
                        "value": 0.75,
                        "passed": True,
                    }
                ],
            },
            intended_use="fixture pretrained-external OCR candidate",
            limitations=["fixture evidence only"],
            evidence_attachments=[
                {
                    "kind": "detector_onnx",
                    "path": "assets/det.onnx",
                    "sha256": detector_sha256,
                },
                {
                    "kind": "recognizer_onnx",
                    "path": "assets/rec.onnx",
                    "sha256": recognizer_sha256,
                },
            ],
            lineage_mode="pretrained_external",
            source_repository_commit=_SOURCE_COMMIT,
            source_asset_sha256=[detector_sha256, recognizer_sha256],
            source_license="apache-2.0",
        )

    def _review_document(self, manifest_path: Path) -> dict:
        manifest_path = Path(manifest_path)
        document = _load_manifest_document(manifest_path)
        return {
            "schema_version": 1,
            "kind": "pretrained_external_human_review",
            "decision": "approve_for_production_promotion",
            "task": document["task"],
            "lineage_mode": "pretrained_external",
            "candidate_manifest_sha256": _sha(manifest_path),
            "descriptor_sha256": document["artifact_sha256"],
            "source_repository_commit": document["source_repository_commit"],
            "source_asset_sha256": list(document["source_asset_sha256"]),
            "source_license": document["source_license"],
            "dataset_config_hash": document["dataset_config_hash"],
            "split_manifest_sha256": document["split_manifest_sha256"],
            "provenance_verified": True,
            "reviewer_confirmed": True,
            "reviewer": "r11b-fixture-reviewer",
            "review_id": "r11b-fixture-review-1",
            "reviewed_at_utc": "2026-10-07T12:00:00Z",
        }

    def _rewrite_manifest(self, manifest_path: Path, document: dict) -> None:
        manifest_path.write_text(
            json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )


def _load_manifest_document(manifest_path: Path) -> dict:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def _write_predictions(bundle: Path) -> Path:
    path = bundle / "predictions.jsonl"
    path.write_bytes(_PREDICTIONS_BYTES)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _gate_policy() -> dict[str, object]:
    return {
        "task_manifest_sha256": "e" * 64,
        "export_min_samples": 16,
        "export_max_abs_error": 0.001,
        "evaluation_gates": [
            {"metric": "bbox_map50", "operator": "gte", "threshold": 0.25}
        ],
    }


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


def _evaluation_evidence() -> dict[str, object]:
    return {
        "evaluator": "fixture-evaluator",
        "evaluator_version": "1.0",
        "split": "val",
        "dataset_config_hash": _DATASET_CONFIG_HASH,
        "split_manifest_sha256": _SPLIT_MANIFEST_SHA256,
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


def _component_inference() -> dict:
    return {"device": "auto", "score_threshold": 0.2, "max_detections": 300}


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


def _trained_ocr_inference() -> dict:
    return {
        "device": "auto",
        "detector_threshold": 0.5,
        "detector_min_area": 4,
        "recognition_threshold": 0.0,
        "detector_batch_size": 4,
        "recognition_batch_size": 8,
        "max_detections_per_view": 512,
        "max_recognition_width": 4096,
    }


def _descriptor_ocr_inference() -> dict:
    return {
        "device": "auto",
        "cpu_threads": 2,
        "detector_threshold": 0.3,
        "box_threshold": 0.6,
        "unclip_ratio": 1.5,
        "recognition_threshold": 0.0,
        "recognition_batch_size": 4,
        "max_candidates": 64,
    }


_TRAINED_INFERENCE = {
    "component": _component_inference,
    "pin": _pin_inference,
    "wire": _wire_inference,
}


if __name__ == "__main__":
    unittest.main()
