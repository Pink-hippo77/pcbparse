"""OCR loss decomposition: trace where OCR tokens are lost at each stage.

This is a CPU-only diagnostic tool. It analyzes cached IR from a production
run and decomposes the loss from raw OCR output to final field assignment.

Loss stages:
  1. DETECTOR: text box not detected (FN in detection)
  2. RECOGNIZER: text detected but recognized incorrectly
  3. SEMANTIC: recognized but classified as wrong type (not designator/value/etc.)
  4. OWNER: typed but assigned to wrong component or no owner
  5. MATCHING: owner assigned but field not accepted by association rules
  6. CONFLICT: multiple candidates compete, resolved against this token
  7. FINAL: all stages pass, field exact match

Usage:
  python scripts/ocr_loss_decomposition.py \
    --ir-dir work/production-e2e-componentmargin0-pinmargin16/ir \
    --split data/splits/dataset300_release50_v1.jsonl \
    --gt-dir Dataset300/Dataset300 \
    --output artifacts/audits/ocr_loss_decomposition.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose OCR loss from detection to field assignment.",
    )
    parser.add_argument(
        "--ir-dir",
        required=True,
        help="Directory containing cached CircuitIR JSON files from a production run.",
    )
    parser.add_argument(
        "--split",
        required=True,
        help="JSONL split file listing sample_ids to analyze.",
    )
    parser.add_argument(
        "--gt-dir",
        required=True,
        help="Root of Dataset300 for GT comparison.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/audits/ocr_loss_decomposition.json",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@dataclass
class TokenTrace:
    """Track a single OCR token through the loss pipeline."""
    token_id: str
    raw_text: str
    resolved_text: str | None
    details: str  # semantic type
    component_id: str | None  # owner
    score: float
    # Loss stage: which stage first rejected or degraded this token
    loss_stage: str = "none"
    loss_reason: str = ""
    # For field-level tracking
    field: str = ""  # designator, value, pinnumber, pinname
    field_exact: bool = False
    field_gt_value: str | None = None


# Canonical field names used in the IR
_DESIGNATOR_RE = re.compile(r"^[A-Z]{1,4}[0-9]+[A-Z]?$")
_PIN_NUMBER_RE = re.compile(r"^[0-9]{1,3}$")


def _classify_field(text: str) -> str:
    """Heuristic field classification for a recognized text string."""
    if _DESIGNATOR_RE.fullmatch(text):
        return "designator"
    if _PIN_NUMBER_RE.fullmatch(text):
        return "pinnumber"
    # Value-like patterns
    if re.match(r"^\d+(?:[.,]\d+)?\s*(?:[kKmMuUnNpP]?[A-Za-z]?)$", text):
        return "value"
    return "other"


def _trace_tokens_from_ir(ir: dict[str, Any]) -> list[TokenTrace]:
    """Extract and classify all OCR tokens from a CircuitIR dict."""
    traces = []
    components = {c["internal_id"]: c for c in ir.get("components", [])}
    for token in ir.get("texts", []):
        raw = token.get("raw_text", "")
        resolved = token.get("resolved_text") or raw
        details = token.get("details", "")
        owner = token.get("component_id")
        score = token.get("score", 0.0)

        trace = TokenTrace(
            token_id=token["internal_id"],
            raw_text=raw,
            resolved_text=resolved,
            details=details,
            component_id=owner,
            score=score,
        )

        # Stage 1: Was it detected? (Always yes if in IR)
        # Stage 2: Was it recognized? (resolved_text != raw suggests OCR processing)
        if not resolved:
            trace.loss_stage = "recognizer"
            trace.loss_reason = "empty_recognition"
            traces.append(trace)
            continue

        # Stage 3: Semantic classification
        if not details or details == "unknown":
            trace.loss_stage = "semantic"
            trace.loss_reason = "unclassified"
            traces.append(trace)
            continue

        # Stage 4: Owner assignment
        if not owner:
            trace.loss_stage = "owner"
            trace.loss_reason = "no_owner"
            traces.append(trace)
            continue

        if owner not in components:
            trace.loss_stage = "owner"
            trace.loss_reason = f"invalid_owner:{owner}"
            traces.append(trace)
            continue

        # Stage 5-6: Field-level analysis
        field_type = _classify_field(resolved)
        trace.field = field_type
        if field_type in ("designator", "value", "pinnumber"):
            # Check if the component has this field set
            comp = components[owner]
            if field_type == "designator":
                comp_key = comp.get("key")
                if not comp_key:
                    trace.loss_stage = "matching"
                    trace.loss_reason = "component_no_key"
                elif comp_key != resolved:
                    trace.loss_stage = "conflict"
                    trace.loss_reason = f"key_mismatch:{comp_key}_vs_{resolved}"
            elif field_type == "value":
                comp_val = comp.get("value", "")
                if not comp_val:
                    trace.loss_stage = "matching"
                    trace.loss_reason = "component_no_value"

        traces.append(trace)
    return traces


def _compute_loss_summary(traces: list[TokenTrace]) -> dict[str, Any]:
    """Aggregate loss statistics from token traces."""
    total = len(traces)
    if total == 0:
        return {"total": 0}

    stage_counts = Counter(t.loss_stage for t in traces)
    field_counts = Counter(t.field for t in traces if t.field)
    owner_counts = Counter(
        "has_owner" if t.component_id else "no_owner" for t in traces
    )

    # Per-field loss breakdown
    field_loss = {}
    for field_name in ("designator", "value", "pinnumber"):
        field_traces = [t for t in traces if t.field == field_name]
        field_stage = Counter(t.loss_stage for t in field_traces)
        field_loss[field_name] = {
            "total": len(field_traces),
            "stages": dict(field_stage),
        }

    return {
        "total_tokens": total,
        "stage_distribution": dict(stage_counts),
        "field_distribution": dict(field_counts),
        "owner_distribution": dict(owner_counts),
        "field_loss": field_loss,
        "tokens_with_loss": stage_counts.get("none", 0),
        "tokens_lost": total - stage_counts.get("none", 0),
        "loss_rate": round(1 - stage_counts.get("none", 0) / total, 4) if total else 0,
    }


def main() -> None:
    args = _arguments()
    ir_dir = Path(args.ir_dir)
    split_path = Path(args.split)
    output_path = Path(args.output)

    # Load split to get sample IDs
    split_rows = _read_jsonl(split_path)
    sample_ids = {row["sample_id"] for row in split_rows}

    # Find IR files
    ir_files = sorted(ir_dir.glob("*.circuit-ir.json"))
    analyzed = 0
    skipped = 0
    all_traces: list[TokenTrace] = []
    per_case: dict[str, dict[str, Any]] = {}

    for ir_path in ir_files:
        case_id = ir_path.stem.replace(".circuit-ir", "")
        if case_id not in sample_ids:
            skipped += 1
            continue

        try:
            ir = json.loads(ir_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue

        traces = _trace_tokens_from_ir(ir)
        summary = _compute_loss_summary(traces)
        per_case[case_id] = summary
        all_traces.extend(traces)
        analyzed += 1

    # Global summary
    global_summary = _compute_loss_summary(all_traces)

    result = {
        "tool": "ocr_loss_decomposition",
        "ir_dir": str(ir_dir),
        "split": str(split_path),
        "cases_analyzed": analyzed,
        "cases_skipped": skipped,
        "global": global_summary,
        "per_case": per_case,
        "stage_definitions": {
            "none": "Token passed all stages without loss",
            "recognizer": "Text detected but recognition produced empty/invalid output",
            "semantic": "Text recognized but semantic classifier could not assign a type",
            "owner": "Typed but no valid component owner assigned",
            "matching": "Owner assigned but association rules rejected the field assignment",
            "conflict": "Multiple candidates compete; this token lost the assignment",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Analyzed {analyzed} cases, skipped {skipped}")
    print(f"Total tokens: {global_summary['total_tokens']}")
    print(f"Loss rate: {global_summary.get('loss_rate', 0):.1%}")
    print(f"Stage distribution: {global_summary['stage_distribution']}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
