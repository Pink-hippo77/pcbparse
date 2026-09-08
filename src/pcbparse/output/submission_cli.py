from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pcbparse.output.submission_package import package_submission


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcbparse-package-submission",
        description=(
            "Create a fail-closed submission/case_x/case_x.json package from a "
            "successful production evaluation run"
        ),
    )
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--audit", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = package_submission(
            args.evaluation_dir,
            args.output_dir,
            archive_path=args.archive,
            audit_path=args.audit,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"status": "error", "type": type(exc).__name__, "message": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps({"status": "ok", **result.report}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
