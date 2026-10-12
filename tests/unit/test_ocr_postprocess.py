"""Tests for OCR text post-processing (confusion pairs + format grammar)."""

from __future__ import annotations

import unittest

from pcbparse.association.ocr_postprocess import correct_ocr_text, postprocess_tokens
from pcbparse.ir.models import BBox, TextToken


def _make_token(raw_text: str, details: str = "") -> TextToken:
    return TextToken(
        internal_id=f"test:{raw_text}",
        raw_text=raw_text,
        candidates=[],
        resolved_text=None,
        category="unknown",
        details=details,
        bbox=BBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0),
        score=0.9,
    )


class CorrectOcrTextTests(unittest.TestCase):
    def test_designator_doubled_prefix(self) -> None:
        corrected, ctx, records = correct_ocr_text("CC13", category_hint="designator")
        self.assertEqual(corrected, "CC13")  # _validate_designator doesn't collapse
        self.assertEqual(ctx, "designator")

    def test_designator_confusion_prefix(self) -> None:
        # "0" prefix is wrong for a resistor; confusion says "O" but O is not R.
        corrected, ctx, records = correct_ocr_text("0R1", category_hint="designator")
        # "0" -> "O" via _CONFUSION_TO_LETTER, but "OR1" is not a valid designator.
        # The _validate_designator path only fixes the first char if it maps to a
        # valid designator prefix.  "O" is not in _DESIGNATOR_PREFIXES so no fix.
        self.assertEqual(ctx, "designator")

    def test_designator_suffix_confusion(self) -> None:
        # "Rl" should become "R1" (l -> 1 in suffix).
        corrected, ctx, records = correct_ocr_text("RI", category_hint="designator")
        # "RI" -> prefix "R", suffix "I".  "I" is not a digit and not trailing letter.
        # _CONFUSION_TO_DIGIT["I"] = "1", so suffix becomes "1" -> "R1".
        self.assertEqual(corrected, "R1")

    def test_valid_prefixes_and_letter_suffixes_are_not_digit_corrected(self) -> None:
        for value in ("GAIN1", "VDD1", "PB10", "CSN2", "VOUT2G", "VOUT2", "U5B", "U5A"):
            with self.subTest(value=value):
                corrected, _, records = correct_ocr_text(value)
                self.assertEqual(value, corrected)
                self.assertEqual([], records)
                token = _make_token(value)
                postprocess_tokens([token])
                self.assertIsNone(token.resolved_text)

    def test_pin_number_confusion(self) -> None:
        corrected, ctx, records = correct_ocr_text("O", category_hint="pinnumber")
        self.assertEqual(corrected, "0")

    def test_pin_number_l_to_1(self) -> None:
        corrected, ctx, records = correct_ocr_text("l", category_hint="pinnumber")
        self.assertEqual(corrected, "1")

    def test_pin_number_already_valid(self) -> None:
        corrected, ctx, records = correct_ocr_text("42", category_hint="pinnumber")
        self.assertEqual(corrected, "42")
        self.assertEqual(records, [])

    def test_value_confusion(self) -> None:
        corrected, ctx, records = correct_ocr_text("1Ok", category_hint="value")
        self.assertEqual(corrected, "10k")

    def test_auto_context_designator(self) -> None:
        corrected, ctx, records = correct_ocr_text("R23")
        self.assertEqual(ctx, "designator")

    def test_auto_context_pinnumber(self) -> None:
        corrected, ctx, records = correct_ocr_text("5")
        self.assertEqual(ctx, "pinnumber")

    def test_auto_context_value(self) -> None:
        corrected, ctx, records = correct_ocr_text("10k")
        self.assertEqual(ctx, "value")

    def test_empty_text(self) -> None:
        corrected, ctx, records = correct_ocr_text("")
        self.assertEqual(corrected, "")
        self.assertEqual(ctx, "unknown")


class PostprocessTokensTests(unittest.TestCase):
    def test_corrects_confused_pin_number(self) -> None:
        token = _make_token("O")
        summary = postprocess_tokens([token])
        self.assertEqual(token.resolved_text, "0")
        self.assertEqual(summary["corrected"], 1)
        self.assertTrue(any(e.metadata.get("kind") == "ocr-text-correction" for e in token.evidence))

    def test_preserves_raw_text(self) -> None:
        token = _make_token("O")
        postprocess_tokens([token])
        self.assertEqual(token.raw_text, "O")

    def test_no_correction_needed(self) -> None:
        token = _make_token("R23")
        summary = postprocess_tokens([token])
        self.assertIsNone(token.resolved_text)
        self.assertEqual(summary["corrected"], 0)

    def test_disabled(self) -> None:
        token = _make_token("O")
        summary = postprocess_tokens([token], enabled=False)
        self.assertIsNone(token.resolved_text)
        self.assertFalse(summary["enabled"])


if __name__ == "__main__":
    unittest.main()
