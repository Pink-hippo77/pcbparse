from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcbparse.evaluation.official_metrics import aggregate_reports, evaluate_official_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the non-official guide-v1 JSON regression matcher"
    )
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    pairs = _pairs(args.predicted, args.expected)
    per_case = []
    reports = []
    for case_id, predicted_path, expected_path in pairs:
        predicted = json.loads(predicted_path.read_text(encoding="utf-8"))
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        report = evaluate_official_json(predicted, expected)
        reports.append(report)
        per_case.append({"case_id": case_id, **report.as_dict()})
    payload = {
        "evaluator_kind": "guide_v1_assumption_regression",
        "official": False,
        "warning": (
            "This is not the competition evaluator. Use raw-v1 for local image-only "
            "accuracy work and replace both when the official scorer is released."
        ),
        "cases": per_case,
        "aggregate": aggregate_reports(reports).as_dict(),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    print(rendered, end="")
    return 0


def _pairs(predicted: Path, expected: Path) -> list[tuple[str, Path, Path]]:
    if predicted.is_file() and expected.is_file():
        return [(predicted.stem, predicted, expected)]
    if predicted.is_dir() and expected.is_dir():
        predicted_files = {path.name: path for path in predicted.glob("*.json") if path.name != "error_manifest.json"}
        expected_files = {path.name: path for path in expected.glob("*.json") if path.name != "error_manifest.json"}
        missing = sorted(set(expected_files) - set(predicted_files))
        extra = sorted(set(predicted_files) - set(expected_files))
        if missing or extra:
            raise ValueError(f"prediction/expected filenames differ; missing={missing}, extra={extra}")
        return [(name.removesuffix(".json"), predicted_files[name], expected_files[name]) for name in sorted(expected_files)]
    raise ValueError("predicted and expected must both be files or both be directories")


if __name__ == "__main__":
    raise SystemExit(main())
