from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ._bootstrap import ensure_source_path

ensure_source_path()

from pcbparse.data import (  # noqa: E402
    assign_grouped_splits,
    read_jsonl,
    write_json,
    write_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assign deterministic train/val/test splits while keeping exact "
            "JSON/image duplicate groups together."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", default="pcbparse-v1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        records = read_jsonl(args.manifest)
        split_records, summary = assign_grouped_splits(
            records,
            ratios={"train": args.train, "val": args.val, "test": args.test},
            seed=args.seed,
        )
        write_jsonl(split_records, args.output)
        if args.summary is not None:
            write_json(summary, args.summary)
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1

    console_value = {
        "ok": True,
        "manifest": str(args.output.resolve(strict=False)),
        **summary,
    }
    print(json.dumps(console_value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

