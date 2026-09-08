from __future__ import annotations

import unittest
from pathlib import Path

from pcbparse.association import (
    OCRAssociationConfig,
    associate_component_texts,
    associate_ocr,
    resolve_component_keys,
)
from pcbparse.ir.models import (
    BBox,
    CircuitIR,
    Component,
    Evidence,
    ImageInfo,
    NetLabel,
    PhysicalNet,
    Pin,
    Point,
    TextToken,
    WireSegment,
)
from pcbparse.perception.golden_provider import DatasetGoldenProvider, GoldenPolicy


WORKSPACE = Path(__file__).resolve().parents[2]
DATASET = WORKSPACE / "Dataset300" / "Dataset300"


def _text(
    internal_id: str,
    raw_text: str,
    details: str,
    bbox: tuple[float, float, float, float],
    *,
    component_id: str | None = None,
    pin_id: str | None = None,
    source_object_id: str | None = None,
    score: float = 0.99,
) -> TextToken:
    return TextToken(
        internal_id=internal_id,
        raw_text=raw_text,
        candidates=[raw_text],
        resolved_text=raw_text,
        category="params",
        details=details,
        bbox=BBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
        score=score,
        component_id=component_id,
        pin_id=pin_id,
        source_object_id=source_object_id,
        evidence=[
            Evidence(
                evidence_id=f"ocr:{internal_id}",
                source="ocr",
                score=score,
                metadata={"fixture": True},
            )
        ],
    )


def _fixture_ir() -> CircuitIR:
    components = [
        Component(
            internal_id="component:1",
            category="R",
            bbox=BBox(x1=10, y1=10, x2=50, y2=55),
        ),
        Component(
            internal_id="component:2",
            key="DATASET_C_2",
            category="C",
            bbox=BBox(x1=80, y1=10, x2=120, y2=55),
        ),
        Component(
            internal_id="component:3",
            category="C",
            bbox=BBox(x1=150, y1=10, x2=190, y2=55),
        ),
    ]
    pins = [
        Pin(
            internal_id="pin:1",
            component_id="component:1",
            point=Point(x=55, y=25),
            base=Point(x=50, y=25),
        ),
        Pin(
            internal_id="pin:2",
            component_id="component:2",
            number="DATASET_P_2",
            name="existing",
            point=Point(x=125, y=25),
            base=Point(x=120, y=25),
        ),
        Pin(
            internal_id="pin:3",
            component_id="component:3",
            point=Point(x=150, y=25),
            base=Point(x=155, y=25),
        ),
        Pin(
            internal_id="pin:4",
            component_id="component:3",
            point=Point(x=150, y=42),
            base=Point(x=155, y=42),
        ),
    ]
    texts = [
        _text("text:ref", "R1", "designator", (14, 2, 26, 8)),
        _text("text:value", "10k", "value", (15, 56, 30, 63)),
        _text("text:name", "RESISTOR", "searchName", (30, 56, 48, 63)),
        # Equal-distance component candidates: keep all rankings, assign none.
        _text("text:ambiguous-component", "C7", "designator", (132, 20, 138, 28)),
        _text(
            "text:pin-number",
            "1",
            "pinNumber",
            (48, 18, 52, 23),
            component_id="component:1",
        ),
        _text(
            "text:pin-name",
            "IN",
            "pinName",
            (48, 26, 54, 31),
            component_id="component:1",
        ),
        # Existing debug values must win over formal OCR candidates.
        _text(
            "text:debug-number",
            "9",
            "pinNumber",
            (119, 18, 124, 23),
            pin_id="pin:2",
        ),
        # Explicit but colliding numbers are retained as candidates only.
        _text("text:collision-a", "2", "pinNumber", (145, 20, 149, 24), pin_id="pin:3"),
        _text("text:collision-b", "2", "pinNumber", (145, 38, 149, 43), pin_id="pin:4"),
        _text(
            "text:gnd",
            "  GＮＤ  ",
            "netLabel",
            (20, 68, 36, 74),
            source_object_id="T_GND",
        ),
        _text("text:ambiguous-label", "SIG", "netlabel", (105, 87, 115, 97)),
        _text("text:far-label", "FAR", "netlabel", (180, 0, 195, 6)),
        _text(
            "text:existing-label",
            "OLD",
            "netlabel",
            (4, 68, 15, 74),
            source_object_id="T_EXISTING",
        ),
    ]
    segments = [
        WireSegment(
            internal_id="segment:main",
            start=Point(x=0, y=80),
            end=Point(x=80, y=80),
        ),
        WireSegment(
            internal_id="segment:parallel-a",
            start=Point(x=90, y=90),
            end=Point(x=140, y=90),
        ),
        WireSegment(
            internal_id="segment:parallel-b",
            start=Point(x=90, y=94),
            end=Point(x=140, y=94),
        ),
        WireSegment(
            internal_id="segment:far",
            start=Point(x=150, y=80),
            end=Point(x=190, y=80),
        ),
    ]
    existing_label = NetLabel(
        internal_id="net-label:existing",
        instance_text="OLD",
        resolved_text="OLD",
        point=Point(x=10, y=80),
        evidence=[
            Evidence(
                evidence_id="dataset:existing-label",
                source="dataset",
                metadata={"source_text_id": "T_EXISTING"},
            )
        ],
    )
    return CircuitIR(
        image=ImageInfo(image_id="fixture", path="fixture.png", width=200, height=100),
        components=components,
        pins=pins,
        texts=texts,
        segments=segments,
        net_labels=[existing_label],
        physical_nets=[
            PhysicalNet(
                internal_id="pnet:main",
                segments=["segment:main"],
            )
        ],
    )


class OCRAssociationTests(unittest.TestCase):
    def test_single_api_assigns_only_unique_component_and_pin_fields(self) -> None:
        ir = _fixture_ir()
        state_before = ir.state
        result = associate_ocr(ir)
        self.assertIs(result, ir)
        self.assertEqual(ir.state, state_before)

        components = ir.component_index()
        self.assertEqual(components["component:1"].key, "R1")
        self.assertEqual(components["component:1"].value, "10k")
        self.assertEqual(components["component:1"].name, "RESISTOR")
        self.assertEqual(components["component:2"].key, "DATASET_C_2")
        self.assertIsNone(components["component:3"].key)

        pins = ir.pin_index()
        self.assertEqual((pins["pin:1"].number, pins["pin:1"].name), ("1", "IN"))
        self.assertEqual((pins["pin:2"].number, pins["pin:2"].name), ("DATASET_P_2", "existing"))
        self.assertIsNone(pins["pin:3"].number)
        self.assertIsNone(pins["pin:4"].number)

        texts = {text.internal_id: text for text in ir.texts}
        self.assertEqual(texts["text:ref"].component_id, "component:1")
        self.assertIsNone(texts["text:ambiguous-component"].component_id)
        self.assertEqual(texts["text:pin-number"].pin_id, "pin:1")
        self.assertEqual(texts["text:pin-name"].pin_id, "pin:1")

        component_evidence = next(
            item
            for item in components["component:1"].evidence
            if item.evidence_id == "association:component-field:component:1:designator"
        )
        self.assertEqual(component_evidence.metadata["decision"], "accepted")
        self.assertEqual(component_evidence.metadata["chosen_value"], "R1")
        self.assertTrue(component_evidence.metadata["candidates"])
        self.assertIn("thresholds", component_evidence.metadata)

        conflict_evidence = next(
            item
            for item in pins["pin:3"].evidence
            if item.evidence_id == "association:pin-field:pin:3:number"
        )
        self.assertEqual(conflict_evidence.metadata["decision"], "component_number_conflict")
        self.assertEqual(conflict_evidence.metadata["candidates"][0]["value"], "2")
        codes = {item.code for item in ir.diagnostics}
        self.assertIn("ASSOCIATION.COMPONENT_TARGET_AMBIGUOUS", codes)
        self.assertIn("ASSOCIATION.PIN_NUMBER_CONFLICT", codes)

    def test_netlabels_are_normalized_attached_or_left_unknown_with_diagnostics(self) -> None:
        ir = associate_ocr(_fixture_ir())
        # One pre-existing Golden label plus three newly converted OCR labels.
        self.assertEqual(len(ir.net_labels), 4)
        labels = {label.resolved_text: label for label in ir.net_labels}
        gnd = labels["GND"]
        self.assertEqual(gnd.instance_text, "GＮＤ")
        self.assertEqual(gnd.net_type, "GROUND")
        self.assertEqual(gnd.attached_physical_net_id, "pnet:main")
        self.assertEqual((gnd.point.x, gnd.point.y), (28.0, 80.0))
        self.assertEqual(
            gnd.evidence[0].metadata["accepted_segment_id"],
            "segment:main",
        )
        self.assertTrue(gnd.evidence[0].metadata["segment_candidates"])
        self.assertIn("thresholds", gnd.evidence[0].metadata)

        self.assertEqual(labels["SIG"].net_type, "UNKNOWN")
        self.assertIsNone(labels["SIG"].attached_physical_net_id)
        self.assertEqual(labels["FAR"].net_type, "UNKNOWN")
        codes = {item.code for item in ir.diagnostics}
        self.assertIn("ASSOCIATION.NETLABEL_AMBIGUOUS", codes)
        self.assertIn("ASSOCIATION.NETLABEL_UNMATCHED", codes)
        report = ir.metadata["ocr_association"]
        self.assertEqual(report["netlabel"]["preserved_existing"], 1)
        self.assertEqual(report["netlabel"]["created"], 3)
        self.assertTrue(report["state_unchanged"])

        # Re-running is idempotent for label objects and association diagnostics.
        labels_before = [label.internal_id for label in ir.net_labels]
        diagnostics_before = [
            (item.code, item.message, tuple(item.object_ids))
            for item in ir.diagnostics
            if item.stage == "association.ocr"
        ]
        associate_ocr(ir)
        self.assertEqual([label.internal_id for label in ir.net_labels], labels_before)
        diagnostics_after = [
            (item.code, item.message, tuple(item.object_ids))
            for item in ir.diagnostics
            if item.stage == "association.ocr"
        ]
        self.assertEqual(diagnostics_after, diagnostics_before)

    def test_custom_thresholds_keep_low_confidence_candidates_unassigned(self) -> None:
        ir = _fixture_ir()
        token = next(text for text in ir.texts if text.internal_id == "text:ref")
        token.score = 0.70
        associate_ocr(
            ir,
            OCRAssociationConfig(component_min_score=0.95),
        )
        component = ir.component_index()["component:1"]
        self.assertIsNone(component.key)
        token_evidence = next(
            item
            for item in token.evidence
            if item.evidence_id == "association:component-target:text:ref"
        )
        self.assertEqual(token_evidence.metadata["decision"], "below_threshold")
        self.assertEqual(token_evidence.metadata["thresholds"]["min_score"], 0.95)

    def test_resolver_accepts_non_u_designator_prefix_for_box(self) -> None:
        component = Component(
            internal_id="component:box",
            category="Box",
            bbox=BBox(x1=10, y1=10, x2=50, y2=50),
        )
        # K12 has stronger OCR evidence.  Treating Box as U would incorrectly
        # add a semantic bonus to the weaker U9 alternative and select it.
        texts = [
            _text("text:k", "K12", "designator", (20, 20, 40, 40), score=0.90),
            _text("text:u", "U9", "designator", (20, 20, 40, 40), score=0.80),
        ]

        assignments = resolve_component_keys(
            [component],
            texts,
            unique_margin=0.03,
        )

        self.assertEqual(assignments, {"component:box": "K12"})
        self.assertEqual(component.key, "K12")

    def test_association_records_generic_box_prefix_as_semantically_compatible(self) -> None:
        component = Component(
            internal_id="component:box",
            category="Box",
            bbox=BBox(x1=10, y1=10, x2=50, y2=50),
        )
        token = _text(
            "text:box-designator",
            "SW2",
            "designator",
            (20, 2, 38, 9),
            component_id=component.internal_id,
        )
        diagnostics = []

        summary = associate_component_texts(
            [component],
            [token],
            diagnostics,
            OCRAssociationConfig(),
        )

        self.assertEqual(summary["assignments"], {"component:box": {"designator": "SW2"}})
        self.assertEqual(component.key, "SW2")
        target_evidence = next(
            item
            for item in token.evidence
            if item.evidence_id == "association:component-target:text:box-designator"
        )
        self.assertEqual(
            target_evidence.metadata["target_candidates"][0]["semantic_compatibility"],
            1.0,
        )
        self.assertEqual(diagnostics, [])

    def test_association_collapses_doubled_prefix_for_explicit_capacitor_owner(self) -> None:
        component = Component(
            internal_id="component:capacitor",
            category="C",
            bbox=BBox(x1=10, y1=10, x2=50, y2=50),
        )
        token = _text(
            "text:doubled-prefix",
            "CC13",
            "designator",
            (20, 2, 38, 9),
            component_id=component.internal_id,
        )

        summary = associate_component_texts(
            [component],
            [token],
            [],
            OCRAssociationConfig(),
        )

        self.assertEqual(
            summary["assignments"],
            {"component:capacitor": {"designator": "C13"}},
        )
        self.assertEqual(component.key, "C13")
        self.assertEqual(token.raw_text, "CC13")

    def test_resolver_respects_explicit_component_owner(self) -> None:
        components = [
            Component(
                internal_id="component:left",
                category="R",
                bbox=BBox(x1=10, y1=10, x2=50, y2=50),
            ),
            Component(
                internal_id="component:right",
                category="R",
                bbox=BBox(x1=60, y1=10, x2=100, y2=50),
            ),
        ]
        token = _text(
            "text:owned",
            "R2",
            "designator",
            (45, 20, 55, 30),
            component_id="component:right",
        )

        assignments = resolve_component_keys(components, [token], unique_margin=0.03)

        self.assertEqual(assignments, {"component:right": "R2"})
        self.assertIsNone(components[0].key)

    def test_resolver_withholds_same_key_when_targets_are_equally_plausible(self) -> None:
        components = [
            Component(
                internal_id="component:left",
                category="R",
                bbox=BBox(x1=10, y1=10, x2=50, y2=50),
            ),
            Component(
                internal_id="component:right",
                category="R",
                bbox=BBox(x1=50, y1=10, x2=90, y2=50),
            ),
        ]
        token = _text("text:shared", "R2", "designator", (45, 20, 55, 30))

        assignments = resolve_component_keys(components, [token], unique_margin=0.03)

        self.assertEqual(assignments, {})
        self.assertIsNone(components[0].key)
        self.assertIsNone(components[1].key)
    def test_real_dataset_debug_identifiers_and_existing_labels_are_preserved(self) -> None:
        ir = DatasetGoldenProvider(GoldenPolicy.DATASET_DEBUG).build_ir(DATASET / "0001")
        keys_before = {component.internal_id: component.key for component in ir.components}
        numbers_before = {pin.internal_id: pin.number for pin in ir.pins}
        labels_before = [label.internal_id for label in ir.net_labels]
        state_before = ir.state

        associate_ocr(ir)
        self.assertEqual(ir.state, state_before)
        self.assertEqual(
            {component.internal_id: component.key for component in ir.components},
            keys_before,
        )
        self.assertEqual({pin.internal_id: pin.number for pin in ir.pins}, numbers_before)
        self.assertEqual([label.internal_id for label in ir.net_labels], labels_before)
        self.assertEqual(
            ir.metadata["ocr_association"]["netlabel"]["preserved_existing"],
            len(labels_before),
        )
        self.assertTrue(any(key and key.startswith("DATASET_") for key in keys_before.values()))
        self.assertTrue(any(number and number.startswith("DATASET_") for number in numbers_before.values()))


if __name__ == "__main__":
    unittest.main()
