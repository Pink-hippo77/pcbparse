from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from .preflight import preflight_training_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify immutable data lineage before starting a PCBParse training task."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train/dataset300_v2_tiny.json"),
    )
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = preflight_training_data(args.config, workspace_root=args.workspace_root)
    except Exception as exc:
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
    payload = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    if args.output is not None:
        output = args.output.expanduser().resolve(strict=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, output)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    print(payload.decode("utf-8"), end="")
    return 0 if report.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
