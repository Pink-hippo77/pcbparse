from __future__ import annotations

import unittest

from pcbparse.association.ocr_common import OCRAssociationConfig
from pcbparse.association.text_pin import associate_pin_texts
from pcbparse.ir.models import BBox, Pin, Point, TextToken


def _pin(pin_id: str, y: float, *, number: str | None = None) -> Pin:
    return Pin(
        internal_id=pin_id,
        component_id="component:1",
        number=number,
        point=Point(x=50, y=y),
        base=Point(x=45, y=y),
        orientation="R",
    )


def _token(token_id: str, value: str, y: float) -> TextToken:
    return TextToken(
        internal_id=token_id,
        raw_text=value,
        resolved_text=value,
        candidates=[value],
        category="params",
        details="pinNumber",
        bbox=BBox(x1=42, y1=y - 0.5, x2=48, y2=y + 0.5),
        score=0.99,
        component_id="component:1",
    )


class PinGlobalMatchingTests(unittest.TestCase):
    def test_global_assignment_solves_two_pin_greedy_counterexample(self) -> None:
        pins = [_pin("pin:1", 0), _pin("pin:2", 14)]
        # Token 1 is locally closer to pin 2, but taking it greedily would force
        # token 2 onto a much worse edge.  The global optimum is the diagonal.
        texts = [_token("text:1", "1", 8), _token("text:2", "2", 22)]
        diagnostics = []

        summary = associate_pin_texts(
            pins,
            texts,
            diagnostics,
            OCRAssociationConfig(pin_max_distance_px=24.0),
        )

        self.assertEqual([pin.number for pin in pins], ["1", "2"])
        self.assertEqual([token.pin_id for token in texts], ["pin:1", "pin:2"])
        self.assertEqual(summary["accepted_targets"], 2)
        evidence = next(
            item
            for item in texts[0].evidence
            if item.evidence_id == "association:pin-target:text:1"
        )
        self.assertEqual(evidence.metadata["assignment_method"], "hungarian")
        self.assertGreaterEqual(
            evidence.metadata["target_margin"],
            OCRAssociationConfig().pin_unique_margin,
        )

    def test_whole_side_translation_uses_monotonic_non_crossing_dp(self) -> None:
        pins = [_pin(f"pin:{index}", float(index * 10)) for index in range(4)]
        texts = [
            _token(f"text:{index}", str(index + 1), float(index * 10 + 6))
            for index in range(4)
        ]
        diagnostics = []

        associate_pin_texts(pins, texts, diagnostics, OCRAssociationConfig())

        self.assertEqual([pin.number for pin in pins], ["1", "2", "3", "4"])
        self.assertEqual(
            [token.pin_id for token in texts],
            ["pin:0", "pin:1", "pin:2", "pin:3"],
        )
        for token in texts:
            evidence = next(
                item
                for item in token.evidence
                if item.evidence_id == f"association:pin-target:{token.internal_id}"
            )
            self.assertEqual(evidence.metadata["assignment_method"], "monotonic_dp")

    def test_global_tie_is_withheld_as_unknown_with_candidates_and_diagnostics(self) -> None:
        pins = [_pin("pin:a", 0), _pin("pin:b", 10)]
        # Identical geometry creates two equally optimal complete assignments.
        texts = [_token("text:a", "A", 5), _token("text:b", "B", 5)]
        diagnostics = []

        summary = associate_pin_texts(pins, texts, diagnostics, OCRAssociationConfig())

        self.assertEqual([pin.number for pin in pins], [None, None])
        self.assertEqual([token.pin_id for token in texts], [None, None])
        self.assertEqual(summary["ambiguous_targets"], 2)
        self.assertIn(
            "ASSOCIATION.PIN_TARGET_AMBIGUOUS",
            {diagnostic.code for diagnostic in diagnostics},
        )
        for token in texts:
            evidence = next(
                item
                for item in token.evidence
                if item.evidence_id == f"association:pin-target:{token.internal_id}"
            )
            self.assertEqual(evidence.metadata["decision"], "ambiguous")
            self.assertEqual(evidence.metadata["target_margin"], 0.0)
            self.assertEqual(len(evidence.metadata["target_candidates"]), 2)
            self.assertIn(
                "score_components",
                evidence.metadata["target_candidates"][0],
            )

    def test_existing_fields_and_input_order_are_preserved(self) -> None:
        first = _pin("pin:z", 14, number="EXISTING")
        second = _pin("pin:a", 0)
        irrelevant = TextToken(
            internal_id="text:irrelevant",
            raw_text="note",
            resolved_text="note",
            details="comment",
            bbox=BBox(x1=0, y1=0, x2=2, y2=2),
        )
        token_z = _token("text:z", "9", 14)
        token_a = _token("text:a", "1", 0)
        pins = [first, second]
        texts = [token_z, irrelevant, token_a]
        pin_order = [id(item) for item in pins]
        text_order = [id(item) for item in texts]

        associate_pin_texts(pins, texts, [], OCRAssociationConfig())

        self.assertEqual([id(item) for item in pins], pin_order)
        self.assertEqual([id(item) for item in texts], text_order)
        self.assertEqual(first.number, "EXISTING")
        self.assertEqual(second.number, "1")


if __name__ == "__main__":
    unittest.main()
