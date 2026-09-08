from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "audit_pin_candidate_runtime_smoke.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "pin_candidate_runtime_smoke_audit", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


class PinCandidateRuntimeSmokeAuditTests(unittest.TestCase):
    def test_fixture_is_first_nonempty_prediction_and_exact_val_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.jsonl"
            index = root / "pin_index.jsonl"
            predictions.write_text(
                "\n".join(
                    json.dumps(row, separators=(",", ":"))
                    for row in (
                        {
                            "record_id": "empty",
                            "sample_id": "0030",
                            "view_id": "empty-roi",
                            "detections_heatmap": [],
                        },
                        {
                            "record_id": "chosen",
                            "sample_id": "0030",
                            "view_id": "chosen-roi",
                            "detections_heatmap": [{"x": 1, "y": 2}],
                        },
                        {
                            "record_id": "later",
                            "sample_id": "0030",
                            "view_id": "later-roi",
                            "detections_heatmap": [{"x": 3, "y": 4}],
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            index.write_text(
                json.dumps(
                    {
                        "record_id": "chosen",
                        "sample_id": "0030",
                        "split": "val",
                        "pins": [{"pin_id": "P1"}],
                        "views": [
                            {"view_kind": "roi", "view_id": "chosen-roi"}
                        ],
                    },
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            selected = audit._select_fixture(predictions, index)
        self.assertEqual("chosen", selected["record"]["record_id"])
        self.assertEqual(2, selected["prediction_line_number"])
        self.assertEqual(1, selected["index_line_number"])
        self.assertRegex(selected["prediction_line_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(selected["index_line_sha256"], r"^[0-9a-f]{64}$")

    def test_letterbox_contract_uses_width_height_and_nchw_order(self) -> None:
        geometry = audit._expected_letterbox((146, 173), (384, 256))
        self.assertEqual([384, 256], geometry["input_size_width_height"])
        self.assertEqual([1, 3, 256, 384], geometry["tensor_shape_nchw"])
        self.assertEqual([216, 256], geometry["resized_size_width_height"])
        self.assertEqual(
            [84, 0, 84, 0], geometry["padding_left_top_right_bottom"]
        )

    def test_strict_json_rejects_duplicates_and_nonfinite_constants(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            audit._loads('{"a":1,"a":2}', "fixture")
        with self.assertRaisesRegex(ValueError, "non-standard JSON constant"):
            audit._loads('{"a":NaN}', "fixture")


if __name__ == "__main__":
    unittest.main()
