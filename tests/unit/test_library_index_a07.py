"""A07 unit tests: evidence-constrained cpntLibrary index and retrieval.

Covers the four M4 fixture classes (known / ambiguous / absent / duplicate pins)
and the hard constraints: pin rows != physical pin count, no fabricated
coordinates, UNKNOWN on ambiguity, hashed library version.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from pcbparse.knowledge.library_index import (
    build_library_index,
    default_library_path,
    load_raw_library,
    normalize_model_name,
    split_variant_suffix,
)
from pcbparse.knowledge.library_matcher import (
    RetrievalDecision,
    RetrievalQuery,
    VisiblePinEvidence,
    build_retrieval_ab_report,
    retrieve_component,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pins(*rows):
    return [
        {"pin_id": pid, "name": name, "number": number, "pin_type": ptype}
        for pid, name, number, ptype in rows
    ]


class NormalizeAndVariantTests(unittest.TestCase):
    def test_normalize_model_name(self) -> None:
        self.assertEqual(normalize_model_name("  ams1117-3.3 "), "AMS1117-3.3")
        self.assertEqual(normalize_model_name("ＡＭＳ１１１７"), "AMS1117")

    def test_split_variant_suffix(self) -> None:
        self.assertEqual(split_variant_suffix("AMS1117-3.3_C6186_1"), ("AMS1117-3.3", "C6186_1"))
        self.assertEqual(split_variant_suffix("CH340C_JX"), ("CH340C", "JX"))
        self.assertEqual(split_variant_suffix("1.25-2PWB"), ("1.25-2PWB", None))


class IndexAggregationTests(unittest.TestCase):
    def test_pin_rows_are_not_physical_pin_count(self) -> None:
        raw = {
            "DUP-A": {
                "esym_path": "DUP-A.esym",
                "pins": _pins(
                    ("e4", "1", "1", "IN"),
                    ("e5", "1", "1", "Undefined"),
                    ("e5", "1", "1", "Undefined"),
                    ("e8", "2", "2", "IN"),
                ),
                "same_hash_components": [],
            }
        }
        index = build_library_index(
            path="fixtures://dup", raw=raw, library_sha256="a" * 64, library_bytes=1
        )
        entry = index["components"]["DUP-A"]
        self.assertEqual(entry["pin_row_count"], 4)
        self.assertEqual(entry["physical_pin_count"], 2)
        self.assertFalse(entry["pin_rows_match_physical_count"])
        # Conflicts preserved: number 1 has name "1" with two pin_types and a
        # duplicated pin_id e5 — neither is collapsed away.
        num1 = next(i for i in entry["pins_by_number"] if i["number"] == "1")
        self.assertTrue(num1["name_conflict"] or num1["pin_type_conflict"] or num1["duplicate_pin_ids"])
        self.assertIn("e5", num1["duplicate_pin_ids"])
        self.assertEqual(num1["pin_row_count"], 3)

    def test_no_coordinates_in_index(self) -> None:
        raw = {
            "R1": {
                "esym_path": "R1.esym",
                "pins": _pins(("e1", "A", "1", "Passive"), ("e2", "B", "2", "Passive")),
                "same_hash_components": [],
            }
        }
        index = build_library_index(
            path="fixtures://r1", raw=raw, library_sha256="b" * 64, library_bytes=1
        )
        blob = json.dumps(index)
        self.assertNotIn('"x"', blob.replace(" ", ""))
        self.assertFalse(index["components"]["R1"]["has_geometry"])
        self.assertFalse(index["invariants"]["coordinates_emitted"])

    def test_same_hash_is_not_package_identity(self) -> None:
        raw = {
            "U1": {
                "esym_path": "U1.esym",
                "pins": _pins(("e1", "A", "1", "IN")),
                "same_hash_components": [
                    {"project": "1", "title": "U1", "symbol": "c" * 32}
                ],
            }
        }
        index = build_library_index(
            path="fixtures://u1", raw=raw, library_sha256="c" * 64, library_bytes=1
        )
        entry = index["components"]["U1"]
        self.assertFalse(entry["same_hash_is_package_identity"])
        self.assertEqual(entry["same_hash_components"][0]["symbol"], "c" * 32)


class RetrievalFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = build_library_index(
            path="fixtures://synthetic",
            raw={
                "CH340C": {
                    "esym_path": "CH340C.esym",
                    "pins": _pins(
                        ("e1", "GND", "1", "Power"),
                        ("e2", "TXD", "2", "OUT"),
                        ("e3", "RXD", "3", "IN"),
                        ("e4", "VCC", "4", "Power"),
                    ),
                    "same_hash_components": [],
                },
                "AMS1117-3.3": {
                    "esym_path": "AMS1117-3.3.esym",
                    "pins": _pins(
                        ("e1", "GND", "1", "Power"),
                        ("e2", "VOUT", "2", "OUT"),
                        ("e3", "VIN", "3", "IN"),
                        ("e4", "VOUT", "2", "OUT"),
                    ),
                    "same_hash_components": [],
                },
                "AMS1117-3.3_C6186": {
                    "esym_path": "AMS1117-3.3_C6186.esym",
                    "pins": _pins(
                        ("e1", "GND", "1", "Power"),
                        ("e2", "Out", "2", "OUT"),
                        ("e3", "IN", "3", "IN"),
                    ),
                    "same_hash_components": [],
                },
                "DUP-PIN-1": {
                    "esym_path": "DUP-PIN-1.esym",
                    "pins": _pins(
                        ("e4", "1", "1", "IN"),
                        ("e5", "1", "1", "Undefined"),
                        ("e5", "1", "1", "Undefined"),
                        ("e8", "2", "2", "IN"),
                    ),
                    "same_hash_components": [],
                },
            },
            library_sha256="f" * 64,
            library_bytes=0,
        )

    def _visible(self, *rows) -> tuple:
        return tuple(VisiblePinEvidence(number=n, name=m) for n, m in rows)

    def test_known_model_matches_with_visible_pins(self) -> None:
        result = retrieve_component(
            self.index,
            RetrievalQuery(
                ocr_model_text="CH340C",
                visible_pins=self._visible(
                    ("1", "GND"), ("2", "TXD"), ("3", "RXD"), ("4", "VCC")
                ),
            ),
        )
        self.assertEqual(result.decision, RetrievalDecision.MATCHED)
        self.assertEqual(result.matched_library_key, "CH340C")
        self.assertFalse(result.to_dict()["coordinates_emitted"])

    def test_ambiguous_family_keeps_unknown(self) -> None:
        result = retrieve_component(
            self.index,
            RetrievalQuery(
                ocr_model_text="AMS1117",
                visible_pins=self._visible(("1", "GND"), ("2", None), ("3", None)),
            ),
        )
        self.assertEqual(result.decision, RetrievalDecision.UNKNOWN)
        self.assertIsNone(result.matched_library_key)
        keys = [c["library_key"] for c in result.candidates]
        self.assertTrue(any(k.startswith("AMS1117") for k in keys))

    def test_absent_model_never_forced(self) -> None:
        result = retrieve_component(
            self.index,
            RetrievalQuery(
                ocr_model_text="XYZ1234_NOT_IN_LIBRARY",
                visible_pins=self._visible(("1", "A"), ("2", "B")),
            ),
        )
        self.assertEqual(result.decision, RetrievalDecision.NO_CANDIDATE)
        self.assertIsNone(result.matched_library_key)

    def test_duplicate_pin_data_index_and_retrieval(self) -> None:
        entry = self.index["components"]["DUP-PIN-1"]
        self.assertEqual(entry["pin_row_count"], 4)
        self.assertEqual(entry["physical_pin_count"], 2)
        result = retrieve_component(
            self.index,
            RetrievalQuery(
                ocr_model_text="DUP-PIN-1",
                visible_pins=self._visible(("1", "1"), ("2", "2")),
            ),
        )
        self.assertEqual(result.decision, RetrievalDecision.MATCHED)
        # Only visible pins receive name candidates — no phantom pins.
        self.assertLessEqual(set(result.pin_name_candidates), {"1", "2"})

    def test_no_visual_evidence_does_not_force_write(self) -> None:
        result = retrieve_component(
            self.index,
            RetrievalQuery(ocr_model_text="AMS1117-3.3", visible_pins=()),
        )
        # Exact key alone may match, but pin_name_candidates must stay empty
        # without visual pins (no forced image-side values).
        if result.decision == RetrievalDecision.MATCHED:
            self.assertEqual(result.matched_library_key, "AMS1117-3.3")
            self.assertEqual(result.pin_name_candidates, {})
        else:
            self.assertEqual(result.decision, RetrievalDecision.UNKNOWN)
            self.assertEqual(result.pin_name_candidates, {})

    def test_visible_pin_names_not_rewritten_when_conflicting(self) -> None:
        # Image says pin1 name "VCC" which is not a library candidate for "1".
        result = retrieve_component(
            self.index,
            RetrievalQuery(
                ocr_model_text="CH340C",
                visible_pins=self._visible(("1", "VCC"), ("2", "TXD")),
            ),
        )
        self.assertEqual(result.decision, RetrievalDecision.MATCHED)
        # Candidates list library names; they never invent coordinates.
        self.assertIn("1", result.pin_name_candidates)
        self.assertNotIn("VCC", result.pin_name_candidates.get("1", []))


class PublicLibrarySmokeTests(unittest.TestCase):
    """Optional smoke against the real extensionless file when present."""

    def test_public_library_hashes_and_row_vs_physical(self) -> None:
        root = _repo_root()
        path = root / default_library_path()
        if not path.is_file():
            self.skipTest("public cpntLibrary not present")
        raw, digest, size = load_raw_library(path)
        self.assertEqual(len(digest), 64)
        self.assertGreater(size, 0)
        index = build_library_index(path)
        self.assertEqual(index["library_sha256"], digest)
        self.assertEqual(index["component_count"], len(raw))
        self.assertGreater(
            index["totals"]["components_where_pin_rows_ne_physical"], 0
        )
        for entry in index["components"].values():
            self.assertFalse(entry["has_geometry"])


class RetrievalReportTests(unittest.TestCase):
    def test_ab_report_four_classes_and_metrics(self) -> None:
        report = build_retrieval_ab_report()
        kinds = {c["kind"] for c in report["cases"]}
        self.assertTrue({"known", "ambiguous", "absent", "duplicate_pins"} <= kinds)
        self.assertFalse(report["metrics"]["coordinates_emitted"])
        self.assertFalse(report["metrics"]["invisible_pins_emitted"])
        self.assertTrue(report["metrics"]["duplicate_pin_rows_ne_physical"])
        self.assertGreaterEqual(report["metrics"]["candidate_recall_at_5"], 0.5)
        self.assertLessEqual(report["metrics"]["wrong_completion_rate"], 0.0)
        self.assertEqual(
            report["metrics"]["forced_pin_writes_without_visual_evidence"], 0
        )
        self.assertEqual(report["metrics"]["cases_pass"], report["metrics"]["cases_total"])
        for case in report["cases"]:
            self.assertTrue(case["hard_invariants_pass"], case["case_id"])


if __name__ == "__main__":
    unittest.main()
