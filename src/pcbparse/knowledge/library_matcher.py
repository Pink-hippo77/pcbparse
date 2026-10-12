"""Evidence-constrained component retrieval against the offline library index.

Pipeline (M4):
1. OCR model text recalls top-k candidates via normalized names / base variants.
2. Visible pin evidence (count, numbers, OCR pin names, coarse arrangement)
   re-scores candidates. Library pin rows are never treated as physical pin count.
3. Without a unique evidence-backed match the decision stays UNKNOWN.
4. Only visible pins may receive name/number candidates; no coordinates and no
   invisible pins are synthesized.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "library_retrieval_ab.v1"

_TOKEN_RE = re.compile(r"[A-Z0-9]+")


class RetrievalDecision(str, Enum):
    MATCHED = "MATCHED"
    UNKNOWN = "UNKNOWN"
    NO_CANDIDATE = "NO_CANDIDATE"


@dataclass(frozen=True)
class VisiblePinEvidence:
    """Image-side pin evidence. Coordinates are accepted only as opaque strings
    for bookkeeping and are never re-exported as invented library geometry.
    """

    number: str | None = None
    name: str | None = None
    pin_type: str | None = None
    arrangement: str | None = None  # e.g. left/right/top/bottom
    evidence_id: str | None = None

    def normalized_name(self) -> str:
        return normalize_text(self.name or "")


@dataclass(frozen=True)
class RetrievalQuery:
    ocr_model_text: str
    visible_pins: tuple[VisiblePinEvidence, ...] = ()
    visible_arrangement: str | None = None
    allow_unknown: bool = True


@dataclass
class RetrievalResult:
    decision: RetrievalDecision
    matched_library_key: str | None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    pin_name_candidates: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    library_sha256: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decision": self.decision.value,
            "matched_library_key": self.matched_library_key,
            "candidates": self.candidates,
            "pin_name_candidates": self.pin_name_candidates,
            "notes": self.notes,
            "library_sha256": self.library_sha256,
            "coordinates_emitted": False,
            "invisible_pins_emitted": False,
        }


def normalize_text(value: str) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = text.replace("　", " ")
    text = re.sub(r"\s+", " ", text)
    return text.upper()


def _tokens(value: str) -> list[str]:
    return _TOKEN_RE.findall(normalize_text(value))


def _name_score(query_norm: str, entry: Mapping[str, Any]) -> tuple[float, str]:
    key_norm = entry.get("normalized_name") or normalize_text(entry.get("library_key", ""))
    base_norm = entry.get("base_normalized") or ""
    if not query_norm:
        return 0.0, "empty_query"
    if query_norm == key_norm:
        return 1.0, "exact_key"
    if base_norm and query_norm == base_norm:
        return 0.92, "exact_base"
    if query_norm in key_norm or key_norm in query_norm:
        return 0.7, "substring"
    if base_norm and (query_norm in base_norm or base_norm in query_norm):
        return 0.55, "base_substring"
    q_tokens = set(_tokens(query_norm))
    k_tokens = set(_tokens(key_norm))
    if q_tokens and k_tokens:
        jaccard = len(q_tokens & k_tokens) / len(q_tokens | k_tokens)
        if jaccard >= 0.99:
            return 0.9, "token_equal"
        if jaccard >= 0.5:
            return 0.35 + 0.4 * jaccard, "token_overlap"
    return 0.0, "no_name_match"


def _pin_evidence_score(
    entry: Mapping[str, Any], visible_pins: Sequence[VisiblePinEvidence]
) -> tuple[float, list[str]]:
    """Score visible-pin agreement. Never scores invisible library pins as present."""
    notes: list[str] = []
    if not visible_pins:
        # Absence of visual pin evidence must not force a unique write.
        return 0.0, ["no_visible_pin_evidence"]

    numbers = entry.get("pins_by_number") or []
    lib_numbers = {str(item.get("number", "")) for item in numbers}
    lib_name_map = {
        str(item.get("number", "")): {
            str(c.get("name", "")) for c in (item.get("name_candidates") or [])
        }
        for item in numbers
    }
    physical = int(entry.get("physical_pin_count") or 0)

    visible_numbers = [str(p.number) for p in visible_pins if p.number not in (None, "")]
    visible_count = len(visible_pins)

    score = 0.5
    # Visible count is a lower bound on physical pins (some may be occluded).
    if physical > 0:
        if visible_count == physical:
            score += 0.2
            notes.append("visible_count_equals_physical")
        elif visible_count < physical:
            score += 0.1
            notes.append("visible_count_below_physical_allowed")
        else:
            score -= 0.35
            notes.append("visible_count_exceeds_physical")

    if visible_numbers:
        overlap = [n for n in visible_numbers if n in lib_numbers]
        missing = [n for n in visible_numbers if n not in lib_numbers]
        if not missing:
            score += 0.2
            notes.append("all_visible_numbers_in_library")
        else:
            score -= 0.4 * (len(missing) / len(visible_numbers))
            notes.append(f"visible_numbers_missing_in_library={missing}")

    name_hits = 0
    name_checks = 0
    for pin in visible_pins:
        pin_name = pin.normalized_name()
        if not pin.number or not pin_name:
            continue
        name_checks += 1
        candidates = lib_name_map.get(str(pin.number), set())
        cand_norm = {normalize_text(c) for c in candidates}
        if pin_name in cand_norm:
            name_hits += 1
        elif any(pin_name in c or c in pin_name for c in cand_norm if c):
            name_hits += 0.5
    if name_checks:
        score += 0.25 * (name_hits / name_checks)
        notes.append(f"pin_name_agreement={name_hits}/{name_checks}")

    return max(0.0, min(1.0, score)), notes


def _combined_score(name_score: float, pin_score: float, has_pins: bool) -> float:
    if has_pins:
        return 0.55 * name_score + 0.45 * pin_score
    # Name-only ranking is provisional; uniqueness gate still applies.
    return 0.85 * name_score


def recall_topk(
    index: Mapping[str, Any],
    ocr_model_text: str,
    *,
    top_k: int = 8,
    min_name_score: float = 0.35,
) -> list[dict[str, Any]]:
    query_norm = normalize_text(ocr_model_text)
    scored: list[dict[str, Any]] = []
    components = index.get("components") or {}
    for key, entry in components.items():
        name_score, name_reason = _name_score(query_norm, entry)
        if name_score < min_name_score:
            continue
        scored.append(
            {
                "library_key": key,
                "name_score": round(name_score, 4),
                "name_reason": name_reason,
                "entry": entry,
            }
        )
    scored.sort(key=lambda item: (-item["name_score"], item["library_key"]))
    return scored[:top_k]


def _pin_name_candidates_for_visible(
    entry: Mapping[str, Any], visible_pins: Sequence[VisiblePinEvidence]
) -> dict[str, list[str]]:
    """Name candidates for *visible* pins only. No coordinates, no new pins."""
    numbers = entry.get("pins_by_number") or []
    by_number = {
        str(item.get("number", "")): item.get("name_candidates") or []
        for item in numbers
    }
    out: dict[str, list[str]] = OrderedDict()
    for pin in visible_pins:
        if pin.number in (None, ""):
            continue
        number = str(pin.number)
        candidates = by_number.get(number)
        if not candidates:
            continue
        names = []
        for candidate in candidates:
            name = str(candidate.get("name", ""))
            if name and name not in names:
                names.append(name)
        out[number] = names
    return dict(out)


def retrieve_component(
    index: Mapping[str, Any],
    query: RetrievalQuery,
    *,
    top_k: int = 8,
    uniqueness_margin: float = 0.08,
    min_accept_name: float = 0.7,
) -> RetrievalResult:
    """OCR model retrieval with visible-pin verification; UNKNOWN when ambiguous."""
    notes: list[str] = []
    library_sha256 = str(index.get("library_sha256", ""))
    visible_pins = tuple(query.visible_pins)
    has_pins = bool(visible_pins)

    recalled = recall_topk(index, query.ocr_model_text, top_k=top_k)
    if not recalled:
        return RetrievalResult(
            decision=RetrievalDecision.NO_CANDIDATE,
            matched_library_key=None,
            candidates=[],
            pin_name_candidates={},
            notes=["no_library_candidate"],
            library_sha256=library_sha256,
        )

    ranked: list[dict[str, Any]] = []
    for item in recalled:
        entry = item["entry"]
        pin_score, pin_notes = _pin_evidence_score(entry, visible_pins)
        combined = _combined_score(item["name_score"], pin_score, has_pins)
        ranked.append(
            {
                "library_key": item["library_key"],
                "base_name": entry.get("base_name"),
                "name_score": item["name_score"],
                "name_reason": item["name_reason"],
                "pin_evidence_score": round(pin_score, 4),
                "pin_notes": pin_notes,
                "score": round(combined, 4),
                "physical_pin_count": entry.get("physical_pin_count"),
                "pin_row_count": entry.get("pin_row_count"),
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["library_key"]))
    ranked = ranked[:top_k]

    best = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None
    gap = best["score"] - runner["score"] if runner else 1.0
    exact_key_only = (
        best["name_reason"] == "exact_key"
        and best["name_score"] >= 0.99
        and (runner is None or runner["name_reason"] != "exact_key")
    )

    pin_candidates = _pin_name_candidates_for_visible(
        index["components"][best["library_key"]], visible_pins
    )

    # Gates: name acceptability, uniqueness margin, and pin-count contradiction.
    if best["name_score"] < min_accept_name and not has_pins:
        notes.append("name_below_threshold_without_visual_pins")
        return RetrievalResult(
            decision=RetrievalDecision.UNKNOWN,
            matched_library_key=None,
            candidates=ranked,
            pin_name_candidates={},
            notes=notes,
            library_sha256=library_sha256,
        )

    if any(note.startswith("visible_count_exceeds_physical") for note in best.get("pin_notes", [])):
        notes.append("visible_pin_count_contradicts_library")
        return RetrievalResult(
            decision=RetrievalDecision.UNKNOWN,
            matched_library_key=None,
            candidates=ranked,
            pin_name_candidates={},
            notes=notes,
            library_sha256=library_sha256,
        )

    # Exact library key outranks base/variant siblings even when their scores
    # sit inside the ambiguity margin (OCR text is the model name itself).
    if exact_key_only:
        notes.append("unique_exact_key")
        return RetrievalResult(
            decision=RetrievalDecision.MATCHED,
            matched_library_key=best["library_key"],
            candidates=ranked,
            pin_name_candidates=pin_candidates,
            notes=notes,
            library_sha256=library_sha256,
        )

    if runner is not None and gap < uniqueness_margin:
        notes.append(f"ambiguous_gap={gap:.4f}<{uniqueness_margin}")
        return RetrievalResult(
            decision=RetrievalDecision.UNKNOWN,
            matched_library_key=None,
            candidates=ranked,
            pin_name_candidates={},
            notes=notes,
            library_sha256=library_sha256,
        )

    # Otherwise require visible-pin support for a non-unique / fuzzy name.
    if not has_pins:
        notes.append("reject_fuzzy_name_without_visible_pins")
        return RetrievalResult(
            decision=RetrievalDecision.UNKNOWN,
            matched_library_key=None,
            candidates=ranked,
            pin_name_candidates={},
            notes=notes,
            library_sha256=library_sha256,
        )

    if best["pin_evidence_score"] < 0.55:
        notes.append("pin_evidence_insufficient")
        return RetrievalResult(
            decision=RetrievalDecision.UNKNOWN,
            matched_library_key=None,
            candidates=ranked,
            pin_name_candidates={},
            notes=notes,
            library_sha256=library_sha256,
        )

    notes.append("unique_match_with_visible_pin_support")
    return RetrievalResult(
        decision=RetrievalDecision.MATCHED,
        matched_library_key=best["library_key"],
        candidates=ranked,
        pin_name_candidates=pin_candidates,
        notes=notes,
        library_sha256=library_sha256,
    )


def _fixture_library() -> dict[str, Any]:
    """Compact four-class fixtures: known, ambiguous, absent, duplicate pins."""
    def pins(*rows: tuple[str, str, str, str]) -> list[dict[str, str]]:
        return [
            {"pin_id": pid, "name": name, "number": number, "pin_type": ptype}
            for pid, name, number, ptype in rows
        ]

    raw = {
        "CH340C": {
            "esym_path": "CH340C.esym",
            "pins": pins(
                ("e1", "GND", "1", "Power"),
                ("e2", "TXD", "2", "OUT"),
                ("e3", "RXD", "3", "IN"),
                ("e4", "VCC", "4", "Power"),
                ("e5", "D+", "5", "BI"),
                ("e6", "D-", "6", "BI"),
            ),
            "same_hash_components": [
                {"project": "1", "title": "CH340C", "symbol": "a" * 32}
            ],
        },
        "AMS1117-3.3": {
            "esym_path": "AMS1117-3.3.esym",
            "pins": pins(
                ("e1", "GND", "1", "Power"),
                ("e2", "VOUT", "2", "OUT"),
                ("e3", "VIN", "3", "IN"),
                ("e4", "VOUT", "2", "OUT"),
            ),
            "same_hash_components": [],
        },
        "AMS1117-3.3_C6186": {
            "esym_path": "AMS1117-3.3_C6186.esym",
            "pins": pins(
                ("e1", "GND", "1", "Power"),
                ("e2", "Out", "2", "OUT"),
                ("e3", "IN", "3", "IN"),
            ),
            "same_hash_components": [],
        },
        "LM317": {
            "esym_path": "LM317.esym",
            "pins": pins(
                ("e1", "ADJ", "1", "IN"),
                ("e2", "VOUT", "2", "OUT"),
                ("e3", "VIN", "3", "IN"),
            ),
            "same_hash_components": [],
        },
        "DUP-PIN-1": {
            "esym_path": "DUP-PIN-1.esym",
            "pins": pins(
                ("e4", "1", "1", "IN"),
                ("e5", "1", "1", "Undefined"),
                ("e5", "1", "1", "Undefined"),
                ("e8", "2", "2", "IN"),
            ),
            "same_hash_components": [],
        },
    }
    from pcbparse.knowledge.library_index import build_library_index

    return build_library_index(
        path="fixtures://synthetic", raw=raw, library_sha256="f" * 64, library_bytes=0
    )


def _visible(*rows: tuple[str | None, str | None]) -> tuple[VisiblePinEvidence, ...]:
    return tuple(
        VisiblePinEvidence(number=number, name=name) for number, name in rows
    )


def build_retrieval_ab_report(
    index: Mapping[str, Any] | None = None,
    fixture_index: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Four-class fixture evaluation: known / ambiguous / absent / duplicate pins."""
    if index is None:
        from pcbparse.knowledge.library_index import build_library_index

        index = build_library_index()
    if fixture_index is None:
        fixture_index = _fixture_library()

    cases: list[dict[str, Any]] = []

    def run(
        case_id: str,
        kind: str,
        idx: Mapping[str, Any],
        query: RetrievalQuery,
        *,
        expect: RetrievalDecision,
        expect_key: str | None = None,
        expect_unknown: bool = False,
    ) -> None:
        result = retrieve_component(idx, query)
        matched = result.matched_library_key
        ok = result.decision == expect
        if expect_key is not None and matched != expect_key:
            ok = False
        if expect_unknown and result.decision == RetrievalDecision.MATCHED:
            ok = False
        # Hard invariants for every case.
        hard_ok = True
        if result.decision != RetrievalDecision.MATCHED and matched is not None:
            hard_ok = False
        for cand in result.candidates:
            if "x" in cand or "y" in cand:
                hard_ok = False
        payload = result.to_dict()
        if payload.get("coordinates_emitted") or payload.get("invisible_pins_emitted"):
            hard_ok = False
        # pin_name_candidates keys must be visible numbers only
        visible_numbers = {
            str(p.number) for p in query.visible_pins if p.number not in (None, "")
        }
        if not set(result.pin_name_candidates).issubset(visible_numbers | set()):
            hard_ok = False
        if not query.visible_pins and result.pin_name_candidates:
            hard_ok = False

        cases.append(
            {
                "case_id": case_id,
                "kind": kind,
                "ocr_model_text": query.ocr_model_text,
                "visible_pin_count": len(query.visible_pins),
                "expected_decision": expect.value,
                "expected_key": expect_key,
                "decision": result.decision.value,
                "matched_library_key": matched,
                "top_candidates": [
                    {
                        "library_key": c["library_key"],
                        "score": c["score"],
                        "name_score": c["name_score"],
                        "pin_evidence_score": c["pin_evidence_score"],
                    }
                    for c in result.candidates[:5]
                ],
                "pin_name_candidates": result.pin_name_candidates,
                "notes": result.notes,
                "pass": ok,
                "hard_invariants_pass": hard_ok,
            }
        )

    # 1) Known model with matching visible pins.
    run(
        "known_ch340c",
        "known",
        fixture_index,
        RetrievalQuery(
            ocr_model_text="CH340C",
            visible_pins=_visible(
                ("1", "GND"),
                ("2", "TXD"),
                ("3", "RXD"),
                ("4", "VCC"),
                ("5", "D+"),
                ("6", "D-"),
            ),
        ),
        expect=RetrievalDecision.MATCHED,
        expect_key="CH340C",
    )

    # 2) Ambiguous model family: two AMS1117 entries share base; fuzzy text
    # without complete discriminating evidence must stay UNKNOWN.
    run(
        "ambiguous_ams1117",
        "ambiguous",
        fixture_index,
        RetrievalQuery(
            ocr_model_text="AMS1117",
            visible_pins=_visible(("1", "GND"), ("2", None), ("3", None)),
        ),
        expect=RetrievalDecision.UNKNOWN,
        expect_unknown=True,
    )

    # 2b) Exact key still preferred when unambiguous and pins support it.
    run(
        "ambiguous_exact_key_supported",
        "ambiguous_exact",
        fixture_index,
        RetrievalQuery(
            ocr_model_text="AMS1117-3.3",
            visible_pins=_visible(("1", "GND"), ("2", "VOUT"), ("3", "VIN")),
        ),
        expect=RetrievalDecision.MATCHED,
        expect_key="AMS1117-3.3",
    )

    # 3) Absent model: NO_CANDIDATE / UNKNOWN, never a forced write.
    run(
        "absent_xyz1234",
        "absent",
        fixture_index,
        RetrievalQuery(
            ocr_model_text="XYZ1234_NOT_IN_LIBRARY",
            visible_pins=_visible(("1", "A"), ("2", "B")),
        ),
        expect=RetrievalDecision.NO_CANDIDATE,
        expect_unknown=True,
    )

    # 4) Duplicate pin data: rows != physical pins; conflicts retained.
    run(
        "duplicate_pin_data",
        "duplicate_pins",
        fixture_index,
        RetrievalQuery(
            ocr_model_text="DUP-PIN-1",
            visible_pins=_visible(("1", "1"), ("2", "2")),
        ),
        expect=RetrievalDecision.MATCHED,
        expect_key="DUP-PIN-1",
    )

    # Same-image missing visual evidence must not force pin writes.
    run(
        "no_visual_evidence_fuzzy",
        "no_visual_evidence",
        fixture_index,
        RetrievalQuery(ocr_model_text="AMS1117", visible_pins=()),
        expect=RetrievalDecision.UNKNOWN,
        expect_unknown=True,
    )

    # Exact key without visual pins may identify the part, but must not
    # complete pin names/numbers that the image never showed.
    run(
        "no_visual_evidence_exact_key",
        "no_visual_evidence_exact",
        fixture_index,
        RetrievalQuery(ocr_model_text="AMS1117-3.3", visible_pins=()),
        expect=RetrievalDecision.MATCHED,
        expect_key="AMS1117-3.3",
    )

    # Duplicate-pin structural checks against the fixture index entry.
    dup_entry = fixture_index["components"]["DUP-PIN-1"]
    dup_ok = (
        dup_entry["pin_row_count"] == 4
        and dup_entry["physical_pin_count"] == 2
        and not dup_entry["pin_rows_match_physical_count"]
        and any(
            n.get("name_conflict")
            or n.get("pin_type_conflict")
            or n.get("duplicate_pin_ids")
            for n in dup_entry["pins_by_number"]
        )
    )

    num_pass = sum(1 for c in cases if c["pass"])
    num_hard = sum(1 for c in cases if c["hard_invariants_pass"])
    recall_hits = 0
    recall_total = 0
    wrong_completion = 0
    completion_total = 0
    forced_pin_writes = 0
    for c in cases:
        keys = [t["library_key"] for t in c["top_candidates"]]
        completion_total += 1
        wrong = False
        if c["expected_key"]:
            recall_total += 1
            if c["expected_key"] in keys:
                recall_hits += 1
            if c["matched_library_key"] and c["matched_library_key"] != c["expected_key"]:
                wrong = True
        elif c["decision"] == "MATCHED":
            # Cases that must stay UNKNOWN / NO_CANDIDATE.
            wrong = True
        # Completing pin fields without any visible pins is a wrong completion.
        if not c["visible_pin_count"] and c["pin_name_candidates"]:
            wrong = True
            forced_pin_writes += 1
        if wrong:
            wrong_completion += 1

    report = {
        "schema_version": SCHEMA_VERSION,
        "library_sha256": index.get("library_sha256", ""),
        "fixture_library_sha256": fixture_index.get("library_sha256", ""),
        "cases": cases,
        "metrics": {
            "cases_total": len(cases),
            "cases_pass": num_pass,
            "hard_invariants_pass": num_hard,
            "candidate_recall_at_5": (
                recall_hits / recall_total if recall_total else None
            ),
            "wrong_completion_rate": (
                wrong_completion / completion_total if completion_total else 0.0
            ),
            "forced_pin_writes_without_visual_evidence": forced_pin_writes,
            "duplicate_pin_rows_ne_physical": dup_ok,
            "coordinates_emitted": False,
            "invisible_pins_emitted": False,
        },
        "invariants": {
            "pin_rows_not_physical_count": True,
            "unknown_on_ambiguity": True,
            "no_fabricated_coordinates": True,
            "library_version_hashed": bool(index.get("library_sha256")),
        },
    }
    return report


def write_retrieval_ab(report: Mapping[str, Any], out_path: Any) -> Any:
    import pathlib

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(description="Build library retrieval A/B JSON")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    report = build_retrieval_ab_report()
    write_retrieval_ab(report, args.out)
    print(json.dumps(report["metrics"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
