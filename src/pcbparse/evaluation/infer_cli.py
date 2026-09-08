from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pcbparse.evaluation.infer_manifest import run_production_evaluation
from pcbparse.service.readiness import ReadinessGateError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcbparse-eval-infer",
        description=(
            "Run a production-ready PCBParse provider over one frozen split using "
            "raster images only"
        ),
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--models-lock", type=Path, required=True)
    parser.add_argument("--pipeline-config", type=Path, required=True)
    parser.add_argument("--profile", default="baseline")
    parser.add_argument("--official-profile", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_production_evaluation(
            dataset_root=args.dataset_root,
            split_manifest=args.split_manifest,
            split=args.split,
            models_lock=args.models_lock,
            pipeline_config=args.pipeline_config,
            profile=args.profile,
            official_profile=args.official_profile,
            output_dir=args.output_dir,
            fail_fast=args.fail_fast,
        )
    except ReadinessGateError as exc:
        print(
            json.dumps(
                {
                    "status": "unready",
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "readiness": exc.report,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
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
    print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0 if result.summary["all_successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
