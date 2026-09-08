from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ._bootstrap import default_dataset_root, ensure_source_path

ensure_source_path()

from pcbparse.data import audit_dataset, write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a strict, actionable audit of the PCB schematic dataset."
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=default_dataset_root()
    )
    parser.add_argument("--expected-count", type=int, default=300)
    parser.add_argument(
        "--output", type=Path, help="Write the complete JSON report to this path."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return failure for warnings as well as errors.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit_dataset(args.dataset_root, expected_count=args.expected_count)
    if args.output is not None:
        write_json(result.report, args.output)
        console_value = {
            "ok": result.ok,
            "strict_ok": result.strict_ok,
            "report": str(args.output.resolve(strict=False)),
            "discovered_samples": result.report["discovered_samples"],
            "valid_manifest_records": result.report["valid_manifest_records"],
            **{
                key: result.report["summary"][key]
                for key in ("errors", "warnings", "duplicate_groups")
            },
        }
    else:
        console_value = result.report
    print(json.dumps(console_value, ensure_ascii=False, sort_keys=True, indent=2))
    accepted = result.strict_ok if args.strict else result.ok
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())

