from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ._bootstrap import ensure_source_path

ensure_source_path()

from pcbparse.data.derive import (  # noqa: E402
    derive_training_datasets,
    write_derived_datasets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive deterministic component, pin, OCR, wire and junction JSONL "
            "training manifests from Dataset300."
        )
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Inner Dataset300 directory containing numbered sample folders.",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output directory for JSONL files."
    )
    parser.add_argument(
        "--version", required=True, help="User-controlled derivation version label."
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help="Optional grouped-split JSONL manifest to propagate into every record.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=64,
        help="Square junction patch size in source-image pixels (default: 64).",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=300,
        help="Expected number of numbered source samples (default: 300).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = derive_training_datasets(
            args.src,
            version=args.version,
            patch_size=args.patch_size,
            split_manifest=args.split_manifest,
            expected_count=args.expected_count,
        )
        written = write_derived_datasets(result, args.out)
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1

    console_summary = {
        "ok": True,
        "out": str(Path(args.out).expanduser().resolve(strict=False)),
        "sample_count": result.summary["sample_count"],
        "counts": result.summary["counts"],
        "total_records": result.summary["total_records"],
        "config_hash": result.summary["config_hash"],
        "files": {key: str(value) for key, value in sorted(written.items())},
    }
    print(json.dumps(console_summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

