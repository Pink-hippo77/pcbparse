from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pcbparse.data import audit_dataset, assign_grouped_splits, read_jsonl, write_json, write_jsonl
from pcbparse.data.derive import derive_training_datasets, write_derived_datasets


def ingest_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Dataset300 and generate a canonical JSONL manifest.")
    parser.add_argument("--dataset-root", type=Path, default=_default_root())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--expected-count", type=int, default=300)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    result = audit_dataset(args.dataset_root, expected_count=args.expected_count)
    if args.report is not None:
        write_json(result.report, args.report)
    accepted = result.strict_ok if args.strict else result.ok
    if accepted:
        write_jsonl(result.records, args.output)
    print(
        json.dumps(
            {
                "ok": result.ok,
                "strict_ok": result.strict_ok,
                "manifest_written": bool(accepted),
                "manifest": str(args.output.resolve(strict=False)),
                "records": len(result.records),
                "errors": result.report["summary"]["errors"],
                "warnings": result.report["summary"]["warnings"],
                "duplicate_groups": result.report["summary"]["duplicate_groups"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if accepted else 1


def audit_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an actionable audit of Dataset300.")
    parser.add_argument("--dataset-root", type=Path, default=_default_root())
    parser.add_argument("--expected-count", type=int, default=300)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    result = audit_dataset(args.dataset_root, expected_count=args.expected_count)
    if args.output is not None:
        write_json(result.report, args.output)
        console = {
            "ok": result.ok,
            "strict_ok": result.strict_ok,
            "report": str(args.output.resolve(strict=False)),
            "discovered_samples": result.report["discovered_samples"],
            "valid_manifest_records": result.report["valid_manifest_records"],
            **{key: result.report["summary"][key] for key in ("errors", "warnings", "duplicate_groups")},
        }
    else:
        console = result.report
    print(json.dumps(console, ensure_ascii=False, sort_keys=True, indent=2))
    accepted = result.strict_ok if args.strict else result.ok
    return 0 if accepted else 1


def split_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create deterministic duplicate-safe Dataset300 splits.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", default="pcbparse-v1")
    args = parser.parse_args(argv)
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
    print(json.dumps({"ok": True, "manifest": str(args.output.resolve(strict=False)), **summary}, ensure_ascii=False, sort_keys=True))
    return 0


def derive_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive five deterministic Dataset300 training manifests.")
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--expected-count", type=int, default=300)
    args = parser.parse_args(argv)
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
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(Path(args.out).expanduser().resolve(strict=False)),
                "sample_count": result.summary["sample_count"],
                "counts": result.summary["counts"],
                "total_records": result.summary["total_records"],
                "config_hash": result.summary["config_hash"],
                "files": {key: str(value) for key, value in sorted(written.items())},
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _default_root() -> Path:
    return Path.cwd() / "Dataset300" / "Dataset300"

