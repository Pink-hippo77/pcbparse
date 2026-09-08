from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_ocr_recognition_by_detail import build_audit


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


class OCRDetailAuditTests(unittest.TestCase):
    def test_val_join_stratifies_details_and_excludes_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            derived = root / "derived.jsonl"
            predictions = root / "predictions.jsonl"
            metrics = root / "metrics.json"
            model = root / "model.pt"
            _write_jsonl(
                derived,
                [
                    {
                        "record_id": "v:1",
                        "sample_id": "v",
                        "split": "val",
                        "details": "designator",
                        "content": "R1",
                    },
                    {
                        "record_id": "v:2",
                        "sample_id": "v",
                        "split": "val",
                        "details": "pinnumber",
                        "content": "12",
                    },
                    {
                        "record_id": "v:excluded",
                        "sample_id": "v",
                        "split": "val",
                        "details": "value",
                        "content": "10k",
                    },
                    {
                        "record_id": "t:1",
                        "sample_id": "t",
                        "split": "test",
                        "details": "designator",
                        "content": "C1",
                    },
                ],
            )
            _write_jsonl(
                predictions,
                [
                    {"kind": "ocr_detection", "tile_id": "v:0", "boxes": []},
                    {
                        "kind": "ocr_recognition",
                        "record_id": "v:1",
                        "sample_id": "v",
                        "reference": "R1",
                        "prediction": "R1",
                        "exact": True,
                        "score": 0.9,
                        "charset_oov": False,
                    },
                    {
                        "kind": "ocr_recognition",
                        "record_id": "v:2",
                        "sample_id": "v",
                        "reference": "12",
                        "prediction": "1",
                        "exact": False,
                        "score": 0.8,
                        "charset_oov": False,
                    },
                ],
            )
            metrics.write_text(
                json.dumps(
                    {
                        "metrics": {
                            "recognition_evaluated_records": 2.0,
                            "normalized_exact_match": 0.5,
                            "recognition_cer": 0.25,
                            "recognition_excluded_policy_records": 1.0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            model.write_bytes(b"fixture")

            audit = build_audit(
                predictions_path=predictions,
                derived_path=derived,
                metrics_path=metrics,
                model_path=model,
            )

            self.assertEqual("val", audit["selection_boundary"]["split"])
            self.assertEqual(0, audit["selection_boundary"]["test_rows_used"])
            self.assertEqual(2, audit["aggregate"]["support"])
            self.assertEqual(1.0, audit["by_detail"]["designator"]["exact_match"])
            self.assertEqual(0.5, audit["by_detail"]["pinnumber"]["cer"])
            self.assertEqual(
                ["v:excluded"], audit["join"]["excluded_policy_record_ids"]
            )
            self.assertEqual(
                hashlib.sha256(model.read_bytes()).hexdigest(),
                audit["sources"]["model"]["sha256"],
            )

    def test_reference_disagreement_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            derived = root / "derived.jsonl"
            predictions = root / "predictions.jsonl"
            metrics = root / "metrics.json"
            model = root / "model.pt"
            _write_jsonl(
                derived,
                [
                    {
                        "record_id": "v:1",
                        "sample_id": "v",
                        "split": "val",
                        "details": "designator",
                        "content": "R1",
                    }
                ],
            )
            _write_jsonl(
                predictions,
                [
                    {
                        "kind": "ocr_recognition",
                        "record_id": "v:1",
                        "sample_id": "v",
                        "reference": "C1",
                        "prediction": "C1",
                        "exact": True,
                        "score": 0.9,
                        "charset_oov": False,
                    }
                ],
            )
            metrics.write_text("{}", encoding="utf-8")
            model.write_bytes(b"fixture")

            with self.assertRaisesRegex(ValueError, "join is inconsistent"):
                build_audit(
                    predictions_path=predictions,
                    derived_path=derived,
                    metrics_path=metrics,
                    model_path=model,
                )


if __name__ == "__main__":
    unittest.main()
