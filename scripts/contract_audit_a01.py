"""A01 CLI: scan public targets and write contract_diff / contract_profile.

Never rewrites GT bytes. Run with PYTHONPATH=src.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pcbparse.evaluation.contract_audit import ContractAuditor  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A01 contract audit (read-only GT scan)")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "赛题六公开数据集",
        help="public dataset root containing 200_train_cases and 10_GTcase",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "audits",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=ROOT / "configs" / "official_schema.yaml",
    )
    args = parser.parse_args(argv)

    auditor = ContractAuditor()
    summary = auditor.run(
        args.root.resolve(),
        args.output_dir.resolve(),
        schema_path=args.schema.resolve() if args.schema.exists() else None,
        guide_rules=(
            "top-level keys components/pins/nets",
            "Name vs name spelling",
            "type aliases r->resistor etc",
            "null/emptystring/empty pin number markers",
            "duplicate designator natural sort",
            "multi pin number split",
            "coordinate y flip at official boundary",
            "zero-length edges kept as two-point arrays",
            "emptyset keys stay placeholders (no ID guessing)",
        ),
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "file_count": summary["file_count"],
                "conflicts": summary["conflicts"],
                "outputs": [
                    summary["inventory_path"],
                    summary["diff_path"],
                    summary["profile_path"],
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
