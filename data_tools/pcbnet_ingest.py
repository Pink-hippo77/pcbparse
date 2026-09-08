from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ._bootstrap import default_dataset_root, ensure_source_path

ensure_source_path()

from pcbparse.data import audit_dataset, write_json, write_jsonl  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Dataset300 and generate a canonical JSONL manifest."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root(),
        help="Inner directory containing 0001..0300 sample folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL manifest.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional destination for the full JSON audit report.",
    )
    parser.add_argument("--expected-count", type=int, default=300)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Refuse manifest generation when audit warnings are present.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit_dataset(args.dataset_root, expected_count=args.expected_count)
    if args.report is not None:
        write_json(result.report, args.report)

    accepted = result.strict_ok if args.strict else result.ok
    if accepted:
        write_jsonl(result.records, args.output)

    summary = {
        "ok": result.ok,
        "strict_ok": result.strict_ok,
        "manifest_written": bool(accepted),
        "manifest": str(args.output.resolve(strict=False)),
        "records": len(result.records),
        "errors": result.report["summary"]["errors"],
        "warnings": result.report["summary"]["warnings"],
        "duplicate_groups": result.report["summary"]["duplicate_groups"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())

