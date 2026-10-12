"""A06: OCR field binding, match_key separation, and global association rules.

Hard constraints covered here:
- 10k / 10K / 10 k must NOT be silently normalized in output
- Duplicate designators must NOT merge merely because strings match
- match_key (casefold+NFKC) stays separate from raw_text
- _canonical_designator rewrites are recorded with raw preserved
- raw_text + resolved_text are bound for pin/semantic matching
- Alphanumeric pins (A5, USB_D-) use a dedicated recognition path
"""

from __future__ import annotations

import unittest
from typing import Any

from pcbparse.association.ocr_common import (
    field_choice,
    lexical_candidates,
    match_key,
    preferred_text,
    resolved_text,
)
from pcbparse.association.semantic_fields import infer_semantic_fields
from pcbparse.association.text_component import (
    _canonical_designator,
    associate_component_texts,
    resolve_component_keys,
)
from pcbparse.association.text_pin import associate_pin_texts
from pcbparse.association.ocr_common import OCRAssociationConfig
from pcbparse.ir.models import (
    BBox,
    Component,
    Diagnostic,
    Evidence,
    Pin,
    Point,
    TextToken,
    WireSegment,
)
from pcbparse.perception.ocr_multilevel import (
    MultilevelOCRConfig,
    run_multilevel_ocr,
)
from pcbparse.io import ImageView
from pcbparse.ir.transforms import AffineTransform, TransformChain
from PIL import Image


def _text(
    internal_id: str,
    raw_text: str,
    details: str,
    bbox: tuple[float, float, float, float],
    *,
    component_id: str | None = None,
    pin_id: str | None = None,
    resolved: str | None = None,
    candidates: list[str] | None = None,
    score: float = 0.99,
    level: str = "global",
    side: str | None = None,
    crop_component_id: str | None = None,
) -> TextToken:
    metadata: dict[str, Any] = {"fixture": True, "level": level}
    if side is not None:
        metadata["side"] = side
    if crop_component_id is not None:
        metadata["component_id"] = crop_component_id
    return TextToken(
        internal_id=internal_id,
        raw_text=raw_text,
        candidates=list(candidates or ([raw_text] if raw_text else [])),
        resolved_text=resolved if resolved is not None else raw_text,
        category="params",
        details=details,
        bbox=BBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
        score=score,
        component_id=component_id,
        pin_id=pin_id,
        evidence=[
            Evidence(
                evidence_id=f"ocr:{internal_id}",
                source="ocr",
                score=score,
                metadata=metadata,
            )
        ],
    )


def _component(
    internal_id: str,
    category: str,
    bbox: tuple[float, float, float, float],
    *,
    key: str | None = None,
    value: str = "",
    name: str = "",
) -> Component:
    return Component(
        internal_id=internal_id,
        category=category,
        bbox=BBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
        key=key,
        value=value,
        name=name,
    )


def _pin(
    internal_id: str,
    component_id: str | None,
    point: tuple[float, float],
    *,
    number: str | None = None,
    name: str = "",
    orientation: str = "UNKNOWN",
) -> Pin:
    return Pin(
        internal_id=internal_id,
        component_id=component_id,
        number=number,
        name=name,
        point=Point(x=point[0], y=point[1]),
        orientation=orientation,  # type: ignore[arg-type]
    )


def _identity_view(identifier: str = "original", size: tuple[int, int] = (200, 200)) -> ImageView:
    return ImageView(
        view_id=identifier,
        image=Image.new("L", size),
        transform=TransformChain([AffineTransform.identity(identifier)]),
    )


def _ml_token(
    identifier: str,
    view_id: str,
    text: str,
    bbox: BBox,
    *,
    score: float = 0.9,
    candidates: list[str] | None = None,
    resolved: str | None = None,
) -> TextToken:
    return TextToken(
        internal_id=identifier,
        raw_text=text,
        candidates=candidates or [],
        resolved_text=resolved,
        bbox=bbox,
        score=score,
        evidence=[Evidence(evidence_id=f"ocr:{view_id}:{identifier}", source="ocr", view_id=view_id)],
    )


class RecordingProvider:
    version = "recording"

    def __init__(self, outputs: dict[str, list[TextToken]]) -> None:
        self.outputs = outputs

    def recognize(self, views, ctx):
        del ctx
        return [
            token.model_copy(deep=True)
            for view in views
            for token in self.outputs.get(view.view_id, [])
        ]


class MatchKeySeparationTests(unittest.TestCase):
    def test_match_key_is_casefold_nfkc_but_raw_is_preserved(self) -> None:
        self.assertEqual(match_key("10k"), match_key("10K"))
        self.assertEqual(match_key("10k"), match_key("10K"))
        # Internal space is collapsed only in the identity key.
        self.assertEqual(match_key("10  k"), match_key("10 k"))
        self.assertNotEqual(match_key("10k"), match_key("10 k"))
        # preferred_text keeps the original spelling.
        token = _text("t", "10K", "value", (0, 0, 10, 10), resolved="10k")
        self.assertEqual(preferred_text(token), "10K")
        self.assertEqual(resolved_text(token), "10k")
        self.assertEqual(token.raw_text, "10K")

    def test_preferred_text_prefers_raw_over_resolved(self) -> None:
        token = _text("t", "10K", "value", (0, 0, 10, 10), resolved="10k", candidates=["10 k"])
        self.assertEqual(preferred_text(token), "10K")
        bound = lexical_candidates(token)
        self.assertIn("10K", bound)
        self.assertIn("10k", bound)
        self.assertIn("10 k", bound)

    def test_field_choice_does_not_casefold_output(self) -> None:
        rows = [
            {"target_status": "accepted", "eligible": True, "value": "10K", "score": 0.95},
            {"target_status": "accepted", "eligible": True, "value": "10k", "score": 0.80},
        ]
        status, chosen, score, margin = field_choice(rows, min_score=0.5, unique_margin=0.10)
        # Distinct originals compete separately; 10K wins on score and stays 10K.
        self.assertEqual(status, "accepted")
        self.assertEqual(chosen, "10K")
        self.assertAlmostEqual(score, 0.95)
        self.assertAlmostEqual(margin, 0.15)

    def test_field_choice_rejects_close_case_variants(self) -> None:
        rows = [
            {"target_status": "accepted", "eligible": True, "value": "10K", "score": 0.90},
            {"target_status": "accepted", "eligible": True, "value": "10k", "score": 0.90},
            {"target_status": "accepted", "eligible": True, "value": "10 k", "score": 0.90},
        ]
        status, chosen, score, margin = field_choice(rows, min_score=0.5, unique_margin=0.05)
        self.assertEqual(status, "ambiguous")
        self.assertIsNone(chosen)

    def test_field_choice_keeps_spaced_variant_distinct(self) -> None:
        rows = [
            {"target_status": "accepted", "eligible": True, "value": "10 k", "score": 0.95},
            {"target_status": "accepted", "eligible": True, "value": "10k", "score": 0.80},
        ]
        status, chosen, _score, _margin = field_choice(rows, min_score=0.5, unique_margin=0.10)
        self.assertEqual(status, "accepted")
        self.assertEqual(chosen, "10 k")


class MultilevelRawPreservationTests(unittest.TestCase):
    def test_case_variants_at_same_site_keep_winner_raw_and_record_suppressed(self) -> None:
        view = _identity_view("original")
        outputs = {
            view.view_id: [
                _ml_token("hi", view.view_id, "10K", BBox(x1=10, y1=10, x2=30, y2=20), score=0.9),
                _ml_token("lo", view.view_id, "10k", BBox(x1=10, y1=10, x2=30, y2=20), score=0.8),
            ]
        }
        result = run_multilevel_ocr(
            RecordingProvider(outputs), [view], [], None, MultilevelOCRConfig()
        )
        self.assertEqual(len(result), 1)
        # Winner raw_text is the higher-score original — never casefolded.
        self.assertEqual(result[0].raw_text, "10K")
        suppressed = result[0].evidence[0].metadata["suppressed_duplicates"]
        self.assertEqual(suppressed[0]["raw_text"], "10k")
        self.assertEqual(suppressed[0]["match_key"], match_key("10k"))
        self.assertEqual(result[0].raw_text, "10K")

    def test_independent_same_string_tokens_at_different_positions_are_not_merged(self) -> None:
        view = _identity_view("original")
        outputs = {
            view.view_id: [
                _ml_token("left", view.view_id, "10k", BBox(x1=10, y1=10, x2=30, y2=20), score=0.9),
                _ml_token("right", view.view_id, "10k", BBox(x1=100, y1=10, x2=120, y2=20), score=0.9),
                _ml_token("far", view.view_id, "R1", BBox(x1=10, y1=100, x2=30, y2=112), score=0.9),
                _ml_token("far2", view.view_id, "R1", BBox(x1=150, y1=150, x2=170, y2=162), score=0.9),
            ]
        }
        result = run_multilevel_ocr(
            RecordingProvider(outputs),
            [view],
            [],
            None,
            MultilevelOCRConfig(nms_iou_threshold=0.5, nms_center_distance_px=3.0),
        )
        raws = sorted(token.raw_text for token in result)
        self.assertEqual(raws, ["10k", "10k", "R1", "R1"])
        # Positions stay distinct.
        xs = sorted(token.bbox.x1 for token in result)
        self.assertEqual(xs, [10.0, 10.0, 100.0, 150.0])

    def test_spaced_variant_at_distinct_site_is_not_normalized_to_compact(self) -> None:
        view = _identity_view("original")
        outputs = {
            view.view_id: [
                _ml_token("spaced", view.view_id, "10 k", BBox(x1=10, y1=10, x2=40, y2=20), score=0.9),
                _ml_token("compact", view.view_id, "10k", BBox(x1=100, y1=10, x2=120, y2=20), score=0.9),
            ]
        }
        result = run_multilevel_ocr(RecordingProvider(outputs), [view], [], None)
        by_raw = {token.raw_text for token in result}
        self.assertEqual(by_raw, {"10 k", "10k"})


class DesignatorRewriteTests(unittest.TestCase):
    def test_canonical_designator_reports_rewrite_and_returns_canonical(self) -> None:
        canonical, rewrite = _canonical_designator("CC13", "C")
        self.assertEqual(canonical, "C13")
        self.assertIsNotNone(rewrite)
        assert rewrite is not None
        self.assertEqual(rewrite["raw_value"], "CC13")
        self.assertEqual(rewrite["canonical_value"], "C13")
        self.assertEqual(rewrite["rule"], "doubled_prefix_collapse")

    def test_canonical_designator_leaves_plain_value_without_rewrite(self) -> None:
        canonical, rewrite = _canonical_designator("R10", "R")
        self.assertEqual(canonical, "R10")
        self.assertIsNone(rewrite)

    def test_association_records_rewrite_evidence_and_keeps_raw(self) -> None:
        component = _component("component:cap", "C", (10, 10, 50, 50))
        token = _text("text:cc", "CC13", "designator", (20, 2, 38, 9), component_id=component.internal_id)
        diagnostics: list[Diagnostic] = []

        summary = associate_component_texts(
            [component],
            [token],
            diagnostics,
            OCRAssociationConfig(),
        )

        self.assertEqual(component.key, "C13")
        self.assertEqual(token.raw_text, "CC13")
        self.assertEqual(
            summary["assignments"],
            {"component:cap": {"designator": "C13"}},
        )
        rewrites = summary.get("designator_rewrites") or []
        self.assertTrue(any(item.get("canonical_value") == "C13" for item in rewrites))
        # Field evidence binds raw and resolved without overwriting raw.
        field_evidence = next(
            item
            for item in component.evidence
            if item.evidence_id == "association:component-field:component:cap:designator"
        )
        self.assertEqual(field_evidence.metadata["chosen_value"], "C13")
        self.assertEqual(field_evidence.metadata["candidates"][0]["raw_text"], "CC13")
        self.assertEqual(token.raw_text, "CC13")


class DuplicateDesignatorTests(unittest.TestCase):
    def test_duplicate_designators_are_not_merged_by_string_alone(self) -> None:
        left = _component("component:left", "R", (10, 10, 50, 50))
        right = _component("component:right", "R", (100, 10, 140, 50))
        token_left = _text(
            "text:left",
            "R1",
            "designator",
            (14, 2, 26, 8),
            component_id=left.internal_id,
        )
        token_right = _text(
            "text:right",
            "R1",
            "designator",
            (104, 2, 116, 8),
            component_id=right.internal_id,
        )
        diagnostics: list[Diagnostic] = []

        summary = associate_component_texts(
            [left, right],
            [token_left, token_right],
            diagnostics,
            OCRAssociationConfig(),
        )

        # Two components keep two objects; visible text is preserved on both.
        self.assertEqual(left.key, "R1")
        self.assertEqual(right.key, "R1")
        self.assertIsNot(left, right)
        assignments = summary["assignments"]
        self.assertEqual(assignments["component:left"]["designator"], "R1")
        self.assertEqual(assignments["component:right"]["designator"], "R1")
        # Tokens remain independent instances at different positions.
        self.assertNotEqual(token_left.bbox.x1, token_right.bbox.x1)
        self.assertEqual(token_left.raw_text, "R1")
        self.assertEqual(token_right.raw_text, "R1")

    def test_resolver_allows_same_visible_designator_with_explicit_owners(self) -> None:
        left = _component("component:left", "R", (10, 10, 50, 50))
        right = _component("component:right", "R", (100, 10, 140, 50))
        token_left = _text(
            "text:left", "R2", "designator", (14, 2, 26, 8), component_id=left.internal_id
        )
        token_right = _text(
            "text:right", "R2", "designator", (104, 2, 116, 8), component_id=right.internal_id
        )

        assignments = resolve_component_keys([left, right], [token_left, token_right])

        self.assertEqual(assignments, {"component:left": "R2", "component:right": "R2"})
        self.assertEqual(left.key, "R2")
        self.assertEqual(right.key, "R2")

    def test_independent_value_tokens_keep_distinct_spellings(self) -> None:
        left = _component("component:left", "R", (10, 10, 50, 55))
        right = _component("component:right", "R", (100, 10, 140, 55))
        token_left = _text("text:v1", "10k", "value", (15, 56, 30, 63), component_id=left.internal_id)
        token_right = _text("text:v2", "10K", "value", (105, 56, 120, 63), component_id=right.internal_id)
        token_spaced = _text(
            "text:v3", "10 k", "value", (200, 56, 230, 63), component_id=None
        )
        third = _component("component:third", "R", (190, 10, 230, 55))
        diagnostics: list[Diagnostic] = []

        associate_component_texts(
            [left, right, third],
            [token_left, token_right, token_spaced],
            diagnostics,
            OCRAssociationConfig(),
        )

        self.assertEqual(left.value, "10k")
        self.assertEqual(right.value, "10K")
        self.assertEqual(third.value, "10 k")
        self.assertEqual(token_left.raw_text, "10k")
        self.assertEqual(token_right.raw_text, "10K")
        self.assertEqual(token_spaced.raw_text, "10 k")


class AlphanumericPinBindingTests(unittest.TestCase):
    def test_alnum_pin_number_a5_binds_on_side_strip(self) -> None:
        component = _component("component:ic", "IC", (100, 100, 160, 140))
        token = _text(
            "text:a5",
            "A5",
            "",
            (102, 96, 110, 102),
            level="side_strip",
            side="T",
            crop_component_id=component.internal_id,
        )
        summary = infer_semantic_fields([component], [], [token], [])
        self.assertEqual(token.details, "pinnumber")
        self.assertEqual(summary["by_field"].get("pinnumber"), 1)

    def test_alnum_pin_number_a5_binds_near_pin_globally(self) -> None:
        component = _component("component:ic", "IC", (100, 100, 160, 140))
        pin = _pin("pin:a5", component.internal_id, (105, 105))
        token = _text("text:a5", "A5", "", (98, 100, 104, 108))
        infer_semantic_fields([component], [pin], [token], [])
        self.assertEqual(token.details, "pinnumber")

    def test_alnum_pin_number_inside_ic_roi_when_not_designator_prefix(self) -> None:
        component = _component("component:ic", "IC", (100, 100, 160, 140))
        token = _text(
            "text:a5",
            "A5",
            "",
            (120, 115, 128, 122),
            level="component",
            crop_component_id=component.internal_id,
        )
        infer_semantic_fields([component], [], [token], [])
        self.assertEqual(token.details, "pinnumber")

    def test_designator_r10_on_resistor_is_not_alnum_pin_number(self) -> None:
        component = _component("component:res", "RESISTOR", (100, 100, 140, 130))
        token = _text(
            "text:r10",
            "R10",
            "",
            (102, 96, 114, 102),
            level="side_strip",
            side="T",
            crop_component_id=component.internal_id,
        )
        infer_semantic_fields([component], [], [token], [])
        self.assertNotEqual(token.details, "pinnumber")

    def test_pin_name_usb_d_minus_binds_and_keeps_raw(self) -> None:
        component = _component("component:ic", "IC", (100, 100, 160, 140))
        pin = _pin("pin:usb", component.internal_id, (105, 105), orientation="L")
        token = _text(
            "text:usb",
            "USB_D-",
            "",
            (80, 100, 102, 108),
            level="side_strip",
            side="L",
            crop_component_id=component.internal_id,
        )
        infer_semantic_fields([component], [pin], [token], [])
        self.assertEqual(token.details, "pinname")
        self.assertEqual(token.raw_text, "USB_D-")

        diagnostics: list[Diagnostic] = []
        summary = associate_pin_texts([pin], [token], diagnostics, OCRAssociationConfig())
        self.assertEqual(pin.name, "USB_D-")
        self.assertEqual(token.raw_text, "USB_D-")
        self.assertGreaterEqual(summary["assigned_fields"], 1)

    def test_pin_name_iovdd_binds(self) -> None:
        component = _component("component:ic", "IC", (100, 100, 160, 140))
        pin = _pin("pin:vdd", component.internal_id, (160, 110), orientation="R")
        token = _text(
            "text:iovdd",
            "IOVDD",
            "",
            (165, 105, 195, 115),
            level="side_strip",
            side="R",
            crop_component_id=component.internal_id,
        )
        infer_semantic_fields([component], [pin], [token], [])
        self.assertEqual(token.details, "pinname")
        diagnostics: list[Diagnostic] = []
        associate_pin_texts([pin], [token], diagnostics, OCRAssociationConfig())
        self.assertEqual(pin.name, "IOVDD")

    def test_value_10k_variants_bind_without_normalization(self) -> None:
        component = _component("component:res", "RESISTOR", (100, 100, 140, 130))
        for index, spelling in enumerate(("10k", "10K", "10 k")):
            with self.subTest(spelling=spelling):
                local = _component(f"component:res{index}", "RESISTOR", (100 + 80 * index, 100, 140 + 80 * index, 130))
                token = _text(
                    f"text:v{index}",
                    spelling,
                    "",
                    (105 + 80 * index, 135, 130 + 80 * index, 142),
                )
                infer_semantic_fields([local], [], [token], [])
                self.assertEqual(token.details, "value")
                diagnostics: list[Diagnostic] = []
                associate_component_texts([local], [token], diagnostics, OCRAssociationConfig())
                self.assertEqual(local.value, spelling)
                self.assertEqual(token.raw_text, spelling)

    def test_pin_association_binds_raw_and_resolved_in_evidence(self) -> None:
        component = _component("component:ic", "IC", (100, 100, 160, 140))
        pin = _pin("pin:1", component.internal_id, (105, 105), orientation="L")
        token = _text(
            "text:num",
            "A5",
            "pinNumber",
            (80, 100, 100, 108),
            pin_id=pin.internal_id,
            resolved="a5",
            candidates=["A5", "A6"],
        )
        diagnostics: list[Diagnostic] = []
        summary = associate_pin_texts([pin], [token], diagnostics, OCRAssociationConfig())
        self.assertEqual(pin.number, "A5")
        self.assertEqual(token.raw_text, "A5")
        self.assertEqual(summary["assignments"]["pin:1"]["number"], "A5")
        evidence = next(
            item
            for item in token.evidence
            if item.evidence_id == "association:pin-target:text:num"
        )
        self.assertEqual(evidence.metadata["value"], "A5")
        self.assertEqual(evidence.metadata["raw_text"], "A5")
        self.assertEqual(evidence.metadata["resolved_value"], "a5")
        self.assertIn("A5", evidence.metadata["lexical_candidates"])
        self.assertIn("a5", evidence.metadata["lexical_candidates"])


class FieldExactAndMisAssignTests(unittest.TestCase):
    """Synthetic AB metrics for field exact and mis-assign rate."""

    def _run_case(
        self,
        components: list[Component],
        texts: list[TextToken],
        pins: list[Pin] | None = None,
    ) -> dict[str, Any]:
        pins = pins or []
        diagnostics: list[Diagnostic] = []
        infer_semantic_fields(components, pins, texts, [])
        associate_component_texts(components, texts, diagnostics, OCRAssociationConfig())
        if pins:
            associate_pin_texts(pins, texts, diagnostics, OCRAssociationConfig())
        return {"components": components, "pins": pins, "texts": texts, "diagnostics": diagnostics}

    def test_field_exact_and_mis_assign_on_regression_suite(self) -> None:
        # GT: (object_id, field, expected) for each case object.
        cases: list[dict[str, Any]] = []

        # 1. value 10k exact
        c1 = _component("c1", "RESISTOR", (10, 10, 50, 55))
        t1 = _text("t1", "10k", "value", (15, 56, 30, 63), component_id="c1")
        cases.append({"id": "value_10k", "gt": {("c1", "value"): "10k"}, "run": lambda: self._run_case([c1], [t1])})

        # 2. value 10K exact (case preserved)
        c2 = _component("c2", "RESISTOR", (10, 10, 50, 55))
        t2 = _text("t2", "10K", "value", (15, 56, 30, 63), component_id="c2")
        cases.append({"id": "value_10K", "gt": {("c2", "value"): "10K"}, "run": lambda: self._run_case([c2], [t2])})

        # 3. value 10 k exact (space preserved)
        c3 = _component("c3", "RESISTOR", (10, 10, 50, 55))
        t3 = _text("t3", "10 k", "value", (15, 56, 35, 63), component_id="c3")
        cases.append({"id": "value_10_spaced_k", "gt": {("c3", "value"): "10 k"}, "run": lambda: self._run_case([c3], [t3])})

        # 4. designator rewrite CC13 -> C13, raw kept
        c4 = _component("c4", "C", (10, 10, 50, 50))
        t4 = _text("t4", "CC13", "designator", (20, 2, 38, 9), component_id="c4")
        cases.append({"id": "designator_rewrite", "gt": {("c4", "designator"): "C13"}, "run": lambda: self._run_case([c4], [t4])})

        # 5. duplicate designators, two owners
        c5a = _component("c5a", "R", (10, 10, 50, 50))
        c5b = _component("c5b", "R", (100, 10, 140, 50))
        t5a = _text("t5a", "R1", "designator", (14, 2, 26, 8), component_id="c5a")
        t5b = _text("t5b", "R1", "designator", (104, 2, 116, 8), component_id="c5b")
        cases.append(
            {
                "id": "duplicate_designators",
                "gt": {("c5a", "designator"): "R1", ("c5b", "designator"): "R1"},
                "run": lambda: self._run_case([c5a, c5b], [t5a, t5b]),
            }
        )

        # 6. pin number A5
        c6 = _component("c6", "IC", (100, 100, 160, 140))
        p6 = _pin("p6", "c6", (105, 105), orientation="L")
        t6 = _text("t6", "A5", "pinNumber", (80, 100, 100, 108), pin_id="p6")
        cases.append({"id": "pin_A5", "gt": {("p6", "number"): "A5"}, "run": lambda: self._run_case([c6], [t6], [p6])})

        # 7. pin name USB_D-
        c7 = _component("c7", "IC", (100, 100, 160, 140))
        p7 = _pin("p7", "c7", (105, 105), orientation="L")
        t7 = _text("t7", "USB_D-", "pinName", (80, 100, 104, 108), pin_id="p7")
        cases.append({"id": "pin_USB_D-", "gt": {("p7", "name"): "USB_D-"}, "run": lambda: self._run_case([c7], [t7], [p7])})

        # 8. pin name IOVDD
        c8 = _component("c8", "IC", (100, 100, 160, 140))
        p8 = _pin("p8", "c8", (160, 110), orientation="R")
        t8 = _text("t8", "IOVDD", "pinName", (165, 105, 195, 115), pin_id="p8")
        cases.append({"id": "pin_IOVDD", "gt": {("p8", "name"): "IOVDD"}, "run": lambda: self._run_case([c8], [t8], [p8])})

        # 9. designator R10
        c9 = _component("c9", "RESISTOR", (10, 10, 50, 50))
        t9 = _text("t9", "R10", "designator", (14, 2, 30, 8), component_id="c9")
        cases.append({"id": "designator_R10", "gt": {("c9", "designator"): "R10"}, "run": lambda: self._run_case([c9], [t9])})

        # 10. mis-assign guard: wrong-owner token must not steal another component field
        c10a = _component("c10a", "R", (10, 10, 50, 50), value="")
        c10b = _component("c10b", "R", (200, 10, 240, 50), value="")
        t10 = _text("t10", "10k", "value", (205, 56, 220, 63), component_id="c10b")
        cases.append(
            {
                "id": "owner_guard_value",
                "gt": {("c10b", "value"): "10k", ("c10a", "value"): ""},
                "run": lambda: self._run_case([c10a, c10b], [t10]),
            }
        )

        field_exact_num = 0
        field_exact_den = 0
        mis_assign_num = 0
        mis_assign_den = 0
        unknown_num = 0
        per_case: list[dict[str, Any]] = []

        for case in cases:
            result = case["run"]()
            components = {c.internal_id: c for c in result["components"]}
            pins = {p.internal_id: p for p in result["pins"]}
            case_exact = 0
            case_den = 0
            case_mis = 0
            for (object_id, field), expected in case["gt"].items():
                case_den += 1
                field_exact_den += 1
                mis_assign_den += 1
                if object_id in components:
                    actual = {
                        "designator": components[object_id].key,
                        "value": components[object_id].value,
                        "searchName": components[object_id].name,
                    }.get(field)
                else:
                    actual = {
                        "number": pins[object_id].number,
                        "name": pins[object_id].name,
                    }.get(field)
                if actual == expected:
                    case_exact += 1
                    field_exact_num += 1
                else:
                    if expected == "" and (actual in (None, "")):
                        # Correctly left empty — not a mis-assign.
                        pass
                    elif actual not in (None, ""):
                        case_mis += 1
                        mis_assign_num += 1
                    else:
                        unknown_num += 1
            per_case.append(
                {
                    "case_id": case["id"],
                    "field_exact_numerator": case_exact,
                    "field_exact_denominator": case_den,
                    "mis_assign_numerator": case_mis,
                }
            )

        # All regression cases must be exact — this is the acceptance gate.
        self.assertEqual(field_exact_num, field_exact_den, per_case)
        self.assertEqual(mis_assign_num, 0, per_case)
        self.assertEqual(unknown_num, 0, per_case)

        self.METRICS = {
            "field_exact": {
                "numerator": field_exact_num,
                "denominator": field_exact_den,
            },
            "mis_assign_rate": {
                "numerator": mis_assign_num,
                "denominator": mis_assign_den,
            },
            "unknown_rate": {
                "numerator": unknown_num,
                "denominator": field_exact_den,
            },
            "per_case": per_case,
        }


if __name__ == "__main__":
    unittest.main()
