"""Tests for OCR loss decomposition logic (CPU-only, no models needed)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.ocr_loss_decomposition import (
    TokenTrace,
    _classify_field,
    _compute_loss_summary,
    _trace_tokens_from_ir,
)


class TestClassifyField(unittest.TestCase):
    def test_designator(self) -> None:
        self.assertEqual("designator", _classify_field("R1"))
        self.assertEqual("designator", _classify_field("C10"))
        self.assertEqual("designator", _classify_field("U5A"))

    def test_pin_number(self) -> None:
        self.assertEqual("pinnumber", _classify_field("1"))
        self.assertEqual("pinnumber", _classify_field("12"))
        self.assertEqual("pinnumber", _classify_field("999"))

    def test_value(self) -> None:
        self.assertEqual("value", _classify_field("10k"))
        self.assertEqual("value", _classify_field("4.7uF"))
        # "100" matches pinnumber first (3 digits)
        self.assertEqual("pinnumber", _classify_field("100"))

    def test_other(self) -> None:
        self.assertEqual("other", _classify_field("RUN"))
        self.assertEqual("other", _classify_field("VCC"))


class TestTraceFromIR(unittest.TestCase):
    def test_basic_ir_tracing(self) -> None:
        ir = {
            "components": [
                {"internal_id": "c1", "key": "R1", "category": "R", "value": "10k"},
                {"internal_id": "c2", "key": None, "category": "R", "value": ""},
            ],
            "texts": [
                {"internal_id": "t1", "raw_text": "R1", "resolved_text": "R1",
                 "details": "designator", "component_id": "c1", "score": 0.99},
                {"internal_id": "t2", "raw_text": "10k", "resolved_text": "10k",
                 "details": "value", "component_id": "c1", "score": 0.95},
                {"internal_id": "t3", "raw_text": "???", "resolved_text": "",
                 "details": "", "component_id": None, "score": 0.3},
                {"internal_id": "t4", "raw_text": "R2", "resolved_text": "R2",
                 "details": "designator", "component_id": "c2", "score": 0.9},
            ],
        }
        traces = _trace_tokens_from_ir(ir)
        self.assertEqual(4, len(traces))

        # t1: designator on c1 (has key R1) → no loss
        self.assertEqual("none", traces[0].loss_stage)
        # t2: value on c1 (has value 10k) → no loss
        self.assertEqual("none", traces[1].loss_stage)
        # t3: empty resolved_text falls back to raw "???", then unclassified → semantic
        self.assertEqual("semantic", traces[2].loss_stage)
        # t4: designator on c2 (no key) → matching loss
        self.assertEqual("matching", traces[3].loss_stage)

    def test_unclassified_token(self) -> None:
        ir = {
            "components": [],
            "texts": [
                {"internal_id": "t1", "raw_text": "hello", "resolved_text": "hello",
                 "details": "", "component_id": None, "score": 0.5},
            ],
        }
        traces = _trace_tokens_from_ir(ir)
        self.assertEqual(1, len(traces))
        self.assertEqual("semantic", traces[0].loss_stage)


class TestComputeLossSummary(unittest.TestCase):
    def test_empty_traces(self) -> None:
        result = _compute_loss_summary([])
        self.assertEqual(0, result["total"])

    def test_mixed_stages(self) -> None:
        traces = [
            TokenTrace("t1", "R1", "R1", "designator", "c1", 0.99, loss_stage="none"),
            TokenTrace("t2", "10k", "10k", "value", "c1", 0.95, loss_stage="none"),
            TokenTrace("t3", "???", "", "", None, 0.3, loss_stage="recognizer"),
            TokenTrace("t4", "R2", "R2", "designator", None, 0.9, loss_stage="owner"),
        ]
        result = _compute_loss_summary(traces)
        self.assertEqual(4, result["total_tokens"])
        self.assertEqual(2, result["tokens_with_loss"])
        self.assertEqual(2, result["tokens_lost"])
        self.assertEqual(0.5, result["loss_rate"])


if __name__ == "__main__":
    unittest.main()
