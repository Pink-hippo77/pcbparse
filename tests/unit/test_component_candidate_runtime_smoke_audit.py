from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "audit_component_candidate_runtime_smoke.py"
)
SPEC = importlib.util.spec_from_file_location(
    "component_candidate_runtime_smoke", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
CUDA_BEFORE_IMPORT = os.environ.get("CUDA_VISIBLE_DEVICES")
SPEC.loader.exec_module(audit)
CUDA_AFTER_IMPORT = os.environ.get("CUDA_VISIBLE_DEVICES")


class ComponentCandidateRuntimeSmokePureTests(unittest.TestCase):
    def test_import_does_not_change_cuda_visibility(self) -> None:
        self.assertEqual(CUDA_BEFORE_IMPORT, CUDA_AFTER_IMPORT)

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            audit._loads('{"a":1,"a":2}', "fixture")
        with self.assertRaisesRegex(ValueError, "non-standard JSON constant"):
            audit._loads('{"a":NaN}', "fixture")

    def test_fixture_selection_is_first_stable_high_confidence_val_full_row(self) -> None:
        coco = {
            "images": [
                {
                    "id": 1,
                    "sample_id": "0030",
                    "view_id": "component:0030:full",
                    "split": "val",
                    "view_kind": "full",
                },
                {
                    "id": 2,
                    "sample_id": "0037",
                    "view_id": "component:0037:full",
                    "split": "val",
                    "view_kind": "full",
                },
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "ignore": 0, "iscrowd": 0},
                {"id": 2, "image_id": 2, "ignore": 0, "iscrowd": 0},
            ],
        }
        near_boundary = {
            "sample_id": "0030",
            "view_id": "component:0030:full",
            "boxes": [[0, 0, 20, 20], [30, 0, 50, 20], [60, 0, 80, 20]],
            "scores": [0.99, 0.95, 0.81],
            "labels": [1, 2, 3],
        }
        selected = {
            "sample_id": "0037",
            "view_id": "component:0037:full",
            "boxes": [
                [0, 0, 20, 20],
                [30, 0, 50, 20],
                [60, 0, 80, 20],
                [90, 0, 100, 20],
            ],
            "scores": [0.99, 0.95, 0.90, 0.50],
            "labels": [1, 2, 3, 1],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(row, separators=(",", ":"))
                    for row in (near_boundary, selected)
                )
                + "\n",
                encoding="utf-8",
            )
            fixture = audit._select_fixture(path, coco)
        self.assertEqual(2, fixture["prediction_line_number"])
        self.assertEqual("0037", fixture["image"]["sample_id"])
        self.assertEqual(1, fixture["usable_ground_truth_count"])
        self.assertRegex(fixture["prediction_line_sha256"], r"^[0-9a-f]{64}$")

    def test_replay_matching_is_class_aware_one_to_one_and_order_independent(self) -> None:
        formal = [
            {
                "category": "R",
                "score": 0.99,
                "box": [0.0, 0.0, 100.0, 100.0],
            },
            {
                "category": "C",
                "score": 0.95,
                "box": [200.0, 200.0, 300.0, 300.0],
            },
            {
                "category": "R",
                "score": 0.90,
                "box": [400.0, 400.0, 500.0, 500.0],
            },
        ]
        runtime = [
            {
                "category": "R",
                "score": 0.905,
                "box": [400.1, 400.0, 500.1, 500.0],
            },
            {
                "category": "R",
                "score": 0.985,
                "box": [0.1, 0.0, 100.1, 100.0],
            },
            {
                "category": "C",
                "score": 0.945,
                "box": [200.1, 200.0, 300.1, 300.0],
            },
        ]
        result = audit._compare_replay_rows(formal, runtime)
        self.assertTrue(result["passed"], result)
        self.assertEqual(3, result["formal_detection_count"])
        self.assertGreaterEqual(result["minimum_iou"], 0.99)
        self.assertLessEqual(result["maximum_box_absolute_error_px"], 0.25)

        runtime[2]["category"] = "LED"
        failed = audit._compare_replay_rows(formal, runtime)
        self.assertFalse(failed["passed"])

    def test_terminal_status_contract_is_fail_closed(self) -> None:
        status = {
            "status": "candidate_evidence_ready",
            "task": "component",
            "registerable": True,
            "formal_evidence": True,
            "smoke": False,
            "finished_at_utc": "2026-08-24T05:00:00Z",
            "gates_passed": True,
            "lineage": {
                "task": "component",
                "formal_evidence": True,
                "smoke": False,
                "training_config_sha256": "a" * 64,
            },
        }
        audit._terminal_status_contract(
            status, expected_training_config_sha256="a" * 64
        )
        for field, value in (
            ("status", "running"),
            ("registerable", False),
            ("formal_evidence", False),
            ("smoke", True),
            ("gates_passed", False),
        ):
            with self.subTest(field=field):
                corrupted = {**status, field: value}
                with self.assertRaises(ValueError):
                    audit._terminal_status_contract(
                        corrupted, expected_training_config_sha256="a" * 64
                    )

    def test_audit_output_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.json"
            audit._atomic_write_new(path, {"status": "pass"})
            first = path.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                audit._atomic_write_new(path, {"status": "fail"})
            self.assertEqual(first, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
