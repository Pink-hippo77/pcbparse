from __future__ import annotations

import unittest

from PIL import Image

from pcbparse.io import ComponentView, ImageView, crop_image_view
from pcbparse.ir.models import BBox, Evidence, Point, TextToken
from pcbparse.ir.transforms import AffineTransform, TransformChain
from pcbparse.perception.ocr_multilevel import MultilevelOCRConfig, run_multilevel_ocr


def _identity_view(identifier: str = "original", size: tuple[int, int] = (100, 100)) -> ImageView:
    return ImageView(
        view_id=identifier,
        image=Image.new("L", size),
        transform=TransformChain([AffineTransform.identity(identifier)]),
    )


def _component_view(original: ImageView, identifier: str = "component:000001") -> ComponentView:
    full = crop_image_view(
        original,
        BBox(x1=20, y1=30, x2=60, y2=70),
        view_id=f"{identifier}:full",
    )
    strips = {
        "L": crop_image_view(full, BBox(x1=0, y1=0, x2=8, y2=40), view_id=f"{identifier}:full:L"),
        "R": crop_image_view(full, BBox(x1=32, y1=0, x2=40, y2=40), view_id=f"{identifier}:full:R"),
        "T": crop_image_view(full, BBox(x1=0, y1=0, x2=40, y2=8), view_id=f"{identifier}:full:T"),
        "B": crop_image_view(full, BBox(x1=0, y1=32, x2=40, y2=40), view_id=f"{identifier}:full:B"),
    }
    return ComponentView(
        component_id=identifier,
        component_bbox=BBox(x1=20, y1=30, x2=60, y2=70),
        full_roi=full,
        side_strips=strips,
    )


def _token(
    identifier: str,
    view_id: str,
    text: str,
    bbox: BBox,
    *,
    score: float = 0.9,
    candidates: list[str] | None = None,
) -> TextToken:
    return TextToken(
        internal_id=identifier,
        raw_text=text,
        candidates=candidates or [],
        bbox=bbox,
        score=score,
        evidence=[Evidence(evidence_id=f"ocr:{view_id}:{identifier}", source="ocr", view_id=view_id)],
    )


class RecordingProvider:
    version = "recording"

    def __init__(self, outputs: dict[str, list[TextToken]]) -> None:
        self.outputs = outputs
        self.calls: list[list[str]] = []

    def recognize(self, views, ctx):
        del ctx
        self.calls.append([view.view_id for view in views])
        return [
            token.model_copy(deep=True)
            for view in views
            for token in self.outputs.get(view.view_id, [])
        ]


class MultilevelOCRTests(unittest.TestCase):
    def test_runs_three_levels_and_restores_each_coordinate_frame(self) -> None:
        original = _identity_view()
        tile = crop_image_view(
            original,
            BBox(x1=10, y1=15, x2=80, y2=85),
            view_id="tile:0",
        )
        component = _component_view(original)
        outputs = {
            "tile:0": [_token("same-local", "tile:0", "GLOBAL", BBox(x1=1, y1=2, x2=5, y2=6))],
            component.full_roi.view_id: [
                _token("same-local", component.full_roi.view_id, "ROI", BBox(x1=1, y1=2, x2=5, y2=6))
            ],
            component.side_strips["R"].view_id: [
                _token(
                    "same-local",
                    component.side_strips["R"].view_id,
                    "SIDE",
                    BBox(x1=1, y1=2, x2=5, y2=6),
                    candidates=["S1", "SI"],
                )
            ],
        }
        provider = RecordingProvider(outputs)

        result = run_multilevel_ocr(provider, [tile], [component], None, MultilevelOCRConfig())

        self.assertEqual([len(call) for call in provider.calls], [1, 1, 4])
        indexed = {token.raw_text: token for token in result}
        self.assertEqual(indexed["GLOBAL"].bbox, BBox(x1=11, y1=17, x2=15, y2=21))
        self.assertEqual(indexed["ROI"].bbox, BBox(x1=21, y1=32, x2=25, y2=36))
        self.assertEqual(indexed["SIDE"].bbox, BBox(x1=53, y1=32, x2=57, y2=36))
        self.assertEqual(len({token.internal_id for token in result}), 3)
        side_metadata = indexed["SIDE"].evidence[0].metadata
        self.assertEqual(side_metadata["level"], "side_strip")
        self.assertEqual(side_metadata["side"], "R")
        self.assertEqual(side_metadata["component_id"], "component:000001")
        self.assertEqual(side_metadata["candidates"], ["S1", "SI"])
        side_binding = side_metadata["local_to_global_binding"]
        self.assertEqual("image_view_local_to_original_v1", side_binding["contract"])
        self.assertEqual("text", side_binding["object_kind"])
        self.assertEqual([1.0, 2.0, 5.0, 6.0], side_binding["local_geometry"]["values"])
        self.assertEqual([53.0, 32.0, 57.0, 36.0], side_binding["original_geometry"]["values"])
        self.assertEqual(52.0, side_binding["local_to_original_matrix"][0][2])
        self.assertEqual(30.0, side_binding["local_to_original_matrix"][1][2])
        # Scheduling context is provenance, not a semantic assignment.
        self.assertIsNone(indexed["SIDE"].component_id)
        self.assertEqual(indexed["SIDE"].category, "unknown")

    def test_cross_level_nms_keeps_one_legal_view_and_merges_candidates(self) -> None:
        original = _identity_view()
        component = _component_view(original)
        outputs = {
            original.view_id: [
                _token("0", original.view_id, " R１ ", BBox(x1=20, y1=30, x2=30, y2=40), score=0.8, candidates=["R1"])
            ],
            component.full_roi.view_id: [
                _token("0", component.full_roi.view_id, "r1", BBox(x1=0, y1=0, x2=10, y2=10), score=0.9, candidates=["RI"])
            ],
        }

        result = run_multilevel_ocr(RecordingProvider(outputs), [original], [component], None)

        self.assertEqual(len(result), 1)
        token = result[0]
        self.assertEqual(token.raw_text, "r1")
        self.assertEqual(token.candidates, ["R1", "RI"])
        self.assertEqual({evidence.view_id for evidence in token.evidence}, {component.full_roi.view_id})
        metadata = token.evidence[0].metadata
        self.assertEqual(metadata["level"], "component")
        self.assertEqual(metadata["component_id"], component.component_id)
        self.assertEqual(metadata["suppressed_duplicates"][0]["level"], "global")

    def test_center_distance_suppresses_non_overlapping_duplicate(self) -> None:
        original = _identity_view()
        outputs = {
            original.view_id: [
                _token("a", original.view_id, "VCC", BBox(x1=10, y1=10, x2=12, y2=12), score=0.9),
                _token("b", original.view_id, "vcc", BBox(x1=13, y1=10, x2=15, y2=12), score=0.8),
            ]
        }
        result = run_multilevel_ocr(
            RecordingProvider(outputs),
            [original],
            [],
            None,
            MultilevelOCRConfig(nms_iou_threshold=0.8, nms_center_distance_px=3.0),
        )
        self.assertEqual(len(result), 1)

    def test_duplicate_from_different_component_rois_does_not_force_owner(self) -> None:
        original = _identity_view()
        first = _component_view(original, "component:first")
        second = _component_view(original, "component:second")
        outputs = {
            first.full_roi.view_id: [
                _token(
                    "first",
                    first.full_roi.view_id,
                    "R1",
                    BBox(x1=0, y1=0, x2=10, y2=10),
                    score=0.9,
                )
            ],
            second.full_roi.view_id: [
                _token(
                    "second",
                    second.full_roi.view_id,
                    "R1",
                    BBox(x1=0, y1=0, x2=10, y2=10),
                    score=0.8,
                )
            ],
        }

        result = run_multilevel_ocr(
            RecordingProvider(outputs), [], [first, second], None
        )

        self.assertEqual(1, len(result))
        metadata = result[0].evidence[0].metadata
        self.assertIsNone(metadata["component_id"])
        self.assertTrue(metadata["component_context_conflict"])
        self.assertEqual(
            ["component:first", "component:second"],
            metadata["component_context_candidates"],
        )

    def test_conflicting_text_merge_is_explicit_and_keeps_variants(self) -> None:
        first = _identity_view("first")
        second = _identity_view("second")
        outputs = {
            first.view_id: [
                _token("first-token", first.view_id, "C13", BBox(x1=20, y1=20, x2=40, y2=30), score=0.90)
            ],
            second.view_id: [
                _token("second-token", second.view_id, "CC13", BBox(x1=20, y1=20, x2=40, y2=30), score=0.80)
            ],
        }
        provider = RecordingProvider(outputs)

        unmerged = run_multilevel_ocr(provider, [first, second], [], None)
        self.assertEqual(2, len(unmerged))

        merged = run_multilevel_ocr(
            provider,
            [first, second],
            [],
            None,
            MultilevelOCRConfig(
                merge_conflicting_text=True,
                conflicting_text_iou_threshold=0.85,
            ),
        )
        self.assertEqual(1, len(merged))
        self.assertEqual("C13", merged[0].raw_text)
        self.assertEqual(["CC13"], merged[0].candidates)
        suppressed = merged[0].evidence[0].metadata["suppressed_duplicates"]
        self.assertEqual("CC13", suppressed[0]["raw_text"])

    def test_missing_multiple_and_out_of_batch_views_fail_closed(self) -> None:
        first = _identity_view("first")
        second = _identity_view("second")

        class InvalidProvider:
            version = "invalid"

            def __init__(self, mode: str) -> None:
                self.mode = mode

            def recognize(self, views, ctx):
                del ctx
                token = _token("bad", views[0].view_id, "X", BBox(x1=0, y1=0, x2=1, y2=1))
                if self.mode == "missing":
                    token.evidence = []
                elif self.mode == "multiple":
                    token.evidence.append(
                        Evidence(evidence_id="other", source="ocr", view_id=views[1].view_id)
                    )
                else:
                    token.evidence[0].view_id = "not-in-batch"
                return [token]

        with self.assertRaisesRegex(ValueError, "no source view"):
            run_multilevel_ocr(InvalidProvider("missing"), [first], [], None)
        with self.assertRaisesRegex(ValueError, "ambiguous source views"):
            run_multilevel_ocr(InvalidProvider("multiple"), [first, second], [], None)
        with self.assertRaisesRegex(ValueError, "outside its provider batch"):
            run_multilevel_ocr(InvalidProvider("unknown"), [first], [], None)

    def test_output_is_stable_under_provider_and_input_reordering(self) -> None:
        first = _identity_view("first")
        second = _identity_view("second")

        class ReversingProvider:
            version = "reversing"

            def __init__(self) -> None:
                self.reverse = False

            def recognize(self, views, ctx):
                del ctx
                output = [
                    _token(
                        "0",
                        view.view_id,
                        view.view_id.upper(),
                        BBox(x1=10 if view.view_id == "first" else 30, y1=10, x2=20 if view.view_id == "first" else 40, y2=20),
                    )
                    for view in views
                ]
                self.reverse = not self.reverse
                return list(reversed(output)) if self.reverse else output

        provider = ReversingProvider()
        left = run_multilevel_ocr(provider, [second, first], [], None)
        right = run_multilevel_ocr(provider, [first, second], [], None)
        self.assertEqual(
            [token.model_dump(mode="json") for token in left],
            [token.model_dump(mode="json") for token in right],
        )

    def test_batch_limits_and_four_strip_contract(self) -> None:
        views = [_identity_view(f"global:{index}") for index in range(5)]
        provider = RecordingProvider({})
        run_multilevel_ocr(
            provider,
            views,
            [],
            None,
            MultilevelOCRConfig(global_batch_size=2, max_global_views=3),
        )
        self.assertEqual(provider.calls, [["global:0", "global:1"], ["global:2"]])

        original = _identity_view()
        invalid = _component_view(original)
        invalid.side_strips = {"L": invalid.side_strips["L"]}
        with self.assertRaisesRegex(ValueError, "exactly L/R/T/B"):
            run_multilevel_ocr(RecordingProvider({}), [], [invalid], None)


if __name__ == "__main__":
    unittest.main()
