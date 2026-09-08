from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ._bootstrap import REPOSITORY_ROOT, ensure_source_path

ensure_source_path()

from pcbparse.training.preflight import preflight_training_data  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify Dataset300-v2 hashes, sources, grouped splits, and five training manifests."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train/dataset300_v2_tiny.json"),
        help="Strict JSON training configuration (default: configs/train/dataset300_v2_tiny.json).",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="Root used to resolve all bound paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the same machine-readable JSON report printed to stdout.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = preflight_training_data(
            args.config,
            workspace_root=args.workspace_root,
        )
    except Exception as exc:  # CLI boundary: always return machine-readable failure.
        report = {
            "schema_version": 1,
            "ok": False,
            "issues": [
                {
                    "code": "preflight_internal_error",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            ],
        }
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve(strict=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(payload, encoding="utf-8", newline="\n")
        temporary.replace(output)
    print(payload, end="")
    return 0 if report.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
