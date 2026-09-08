from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC = REPOSITORY_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pcbparse.training.rebind import (  # noqa: E402
    CandidateRebindError,
    rebind_legacy_candidate_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebind one legacy candidate to a hash-verified runtime inference "
            "contract in a new directory without retraining or changing models.lock."
        )
    )
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        report = rebind_legacy_candidate_bundle(
            arguments.source_bundle,
            arguments.training_config,
            arguments.output,
        )
    except CandidateRebindError as exc:
        print(f"candidate rebind blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
