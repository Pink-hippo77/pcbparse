from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "audit_ocr_train_rare_prefix.py"
)
_SPEC = importlib.util.spec_from_file_location("ocr_train_rare_prefix_audit", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


class OCRTrainRarePrefixSelectionTests(unittest.TestCase):
    def test_complete_ascii_prefix_digit_policy_is_train_only(self) -> None:
        rows = [
            {
                "record_id": "train-upper",
                "split": "train",
                "status": "included",
                "content": "K1",
            },
            {
                "record_id": "train-lower",
                "split": "train",
                "status": "included",
                "content": "k22",
            },
            {
                "record_id": "leading-space",
                "split": "train",
                "status": "included",
                "content": " K3",
            },
            {
                "record_id": "no-digits",
                "split": "train",
                "status": "included",
                "content": "K",
            },
            {
                "record_id": "digit-prefix",
                "split": "train",
                "status": "included",
                "content": "1K",
            },
            {
                "record_id": "unicode-digit",
                "split": "train",
                "status": "included",
                "content": "K１",
            },
            {
                "record_id": "val",
                "split": "val",
                "status": "included",
                "content": "R1",
            },
            {
                "record_id": "test",
                "split": "test",
                "status": "included",
                "content": "C2",
            },
            {
                "record_id": "excluded",
                "split": "train",
                "status": "excluded",
                "content": "D3",
            },
        ]
        selected = audit.select_train_prefix_rows(rows)
        self.assertEqual(
            ["train-lower", "train-upper"],
            [row["record_id"] for row in selected],
        )
        self.assertEqual(["K", "K"], [row["audit_prefix"] for row in selected])
        self.assertTrue(all(row["split"] == "train" for row in selected))

    def test_prefix_function_rejects_partial_or_non_ascii_matches(self) -> None:
        self.assertEqual("GPIO", audit.ascii_prefix("GPIO12"))
        self.assertEqual("K", audit.ascii_prefix("k0"))
        for value in ("K", "12", "K1x", " K1", "K-1", "Ｋ1", "K１", None):
            self.assertIsNone(audit.ascii_prefix(value))


class OCRTrainRarePrefixMetricTests(unittest.TestCase):
    def test_exact_cer_support_and_domain_strata_share_denominators(self) -> None:
        rows = [
            {
                "record_id": "a",
                "sample_id": "s1",
                "prefix": "K",
                "prefix_case_variant": "K",
                "source_domain": "ad",
                "reference": "K1",
                "prediction": "K1",
                "score": 0.9,
            },
            {
                "record_id": "b",
                "sample_id": "s2",
                "prefix": "K",
                "prefix_case_variant": "k",
                "source_domain": "jlc",
                "reference": "K22",
                "prediction": "K2",
                "score": 0.7,
            },
            {
                "record_id": "c",
                "sample_id": "s2",
                "prefix": "R",
                "prefix_case_variant": "R",
                "source_domain": "jlc",
                "reference": "R1",
                "prediction": "C1",
                "score": 0.5,
            },
        ]
        summary = audit.summarize_predictions(rows)
        self.assertEqual(3, summary["aggregate"]["support"])
        self.assertEqual(1, summary["aggregate"]["exact_count"])
        self.assertEqual(2, summary["aggregate"]["edit_distance"])
        self.assertAlmostEqual(2.0 / 7.0, summary["aggregate"]["cer"])
        self.assertEqual(2, summary["by_prefix"]["K"]["support"])
        self.assertEqual(1, summary["by_prefix"]["K"]["exact_count"])
        self.assertEqual(["K", "k"], summary["by_prefix"]["K"]["prefix_case_variants"])
        self.assertEqual(2, summary["by_domain"]["jlc"]["support"])
        self.assertEqual(1, summary["by_prefix_domain"]["K"]["ad"]["support"])


class OCRTrainRarePrefixContractTests(unittest.TestCase):
    def _contract(self) -> tuple[dict, dict, dict, dict, dict]:
        digests = {
            "checkpoint": "1" * 64,
            "materialization_manifest": "2" * 64,
            "source_manifest": "3" * 64,
            "split_manifest": "4" * 64,
            "ocr_manifest": "5" * 64,
        }
        config = "6" * 64
        bundle = {
            "task": "ocr",
            "artifact_sha256": digests["checkpoint"],
            "training_config_sha256": config,
            "source_manifest_sha256": digests["source_manifest"],
            "split_manifest_sha256": digests["split_manifest"],
            "gate_policy": {"task_manifest_sha256": digests["ocr_manifest"]},
        }
        common_lineage = {
            "task": "ocr",
            "formal_evidence": True,
            "smoke": False,
            "training_config_sha256": config,
            "materialization_commit_kind": "manifest_sha256",
            "materialization_commit_id": digests["materialization_manifest"],
            "materialization_commit_sha256": digests["materialization_manifest"],
            "source_manifest_sha256": digests["source_manifest"],
            "split_manifest_sha256": digests["split_manifest"],
            "task_manifest_sha256": digests["ocr_manifest"],
        }
        status = {
            "task": "ocr",
            "status": "candidate_evidence_ready",
            "formal_evidence": True,
            "smoke": False,
            "artifact": {"sha256": digests["checkpoint"]},
            "lineage": dict(common_lineage),
        }
        checkpoint = dict(common_lineage)
        materialization = {
            "lineage": {
                "source_manifest_sha256": digests["source_manifest"],
                "split_manifest_sha256": digests["split_manifest"],
                "ocr_manifest_sha256": digests["ocr_manifest"],
            }
        }
        return bundle, status, checkpoint, materialization, digests

    def test_checkpoint_config_and_materialization_hash_chain_is_required(self) -> None:
        bundle, status, checkpoint, materialization, digests = self._contract()
        result = audit.validate_contract(
            bundle_manifest=bundle,
            run_status=status,
            checkpoint_lineage=checkpoint,
            materialization_manifest=materialization,
            hashes=digests,
        )
        self.assertTrue(result["hash_chain_verified"])
        self.assertEqual("6" * 64, result["training_config_sha256"])

    def test_config_or_materialization_drift_fails_closed(self) -> None:
        bundle, status, checkpoint, materialization, digests = self._contract()
        checkpoint["training_config_sha256"] = "7" * 64
        with self.assertRaisesRegex(ValueError, "training config hash mismatch"):
            audit.validate_contract(
                bundle_manifest=bundle,
                run_status=status,
                checkpoint_lineage=checkpoint,
                materialization_manifest=materialization,
                hashes=digests,
            )

        bundle, status, checkpoint, materialization, digests = self._contract()
        digests["materialization_manifest"] = "8" * 64
        with self.assertRaisesRegex(ValueError, "materialization_commit_id SHA-256"):
            audit.validate_contract(
                bundle_manifest=bundle,
                run_status=status,
                checkpoint_lineage=checkpoint,
                materialization_manifest=materialization,
                hashes=digests,
            )


if __name__ == "__main__":
    unittest.main()
