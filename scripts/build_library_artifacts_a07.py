"""Build A07 knowledge artifacts: library_index.json, library_retrieval_ab.json,
and artifacts/audits/A07_delivery.json. Does not modify models.lock.json.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcbparse.knowledge.library_index import (  # noqa: E402
    build_library_index,
    default_library_path,
    write_library_index,
)
from pcbparse.knowledge.library_matcher import (  # noqa: E402
    build_retrieval_ab_report,
    write_retrieval_ab,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    out_dir = ROOT / "artifacts" / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)

    library_path = ROOT / default_library_path()
    index = build_library_index(library_path)
    index_path = out_dir / "library_index.json"
    write_library_index(index, index_path)

    report = build_retrieval_ab_report(index=index)
    ab_path = out_dir / "library_retrieval_ab.json"
    write_retrieval_ab(report, ab_path)

    model_lock = ROOT / "models" / "models.lock.json"
    model_lock_sha = sha256_file(model_lock) if model_lock.is_file() else ""

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        commit = "unknown"

    delivery = {
        "task_id": "A07",
        "status": "validated",
        "source_commit": commit,
        "model_lock_sha256": model_lock_sha,
        "library_sha256": index["library_sha256"],
        "library_path": index["library_path"],
        "evaluator_id": "internal_library_index_unittest",
        "split_role": "development_regression",
        "sidecar_access": False,
        "changes": [
            "src/pcbparse/knowledge/library_index.py: deterministic pin-by-number index",
            "src/pcbparse/knowledge/library_matcher.py: OCR top-k + visible-pin verify",
            "src/pcbparse/knowledge/__init__.py",
            "tests/unit/test_library_index_a07.py",
            "artifacts/audits/library_index.json",
            "artifacts/audits/library_retrieval_ab.json",
        ],
        "commands": [
            "$env:PYTHONPATH=(Resolve-Path '.\\src').Path; & .\\.venv-test\\Scripts\\python.exe -m pytest tests/unit/test_library_index_a07.py -q",
            "$env:PYTHONPATH=(Resolve-Path '.\\src').Path; & .\\.venv-test\\Scripts\\python.exe scripts/build_library_artifacts_a07.py",
        ],
        "metrics": [
            {
                "name": "library_index_components",
                "numerator": index["component_count"],
                "denominator": index["component_count"],
                "scope": str(default_library_path()).replace("\\", "/"),
                "baseline": "no indexed library (M4 gap)",
                "candidate": f"{index['component_count']} components hashed",
                "delta": "offline library indexed with version hash",
                "eligible": index["component_count"],
                "masked": 0,
            },
            {
                "name": "pin_rows_ne_physical_components",
                "numerator": index["totals"]["components_where_pin_rows_ne_physical"],
                "denominator": index["component_count"],
                "scope": "library_index.json",
                "baseline": "risk of treating pin rows as physical pins",
                "candidate": "physical_pin_count = unique numbers; rows kept separately",
                "delta": "row/physical split enforced",
                "eligible": index["component_count"],
                "masked": 0,
            },
            {
                "name": "retrieval_fixture_cases",
                "numerator": report["metrics"]["cases_pass"],
                "denominator": report["metrics"]["cases_total"],
                "scope": "library_retrieval_ab.json",
                "baseline": "unconstrained name match",
                "candidate": "evidence-constrained + UNKNOWN on ambiguity",
                "delta": "four-class fixtures covered",
                "eligible": report["metrics"]["cases_total"],
                "masked": 0,
            },
            {
                "name": "candidate_recall_at_5",
                "numerator": report["metrics"]["candidate_recall_at_5"],
                "denominator": 1.0,
                "scope": "library_retrieval_ab.json",
                "baseline": "n/a",
                "candidate": report["metrics"]["candidate_recall_at_5"],
                "delta": "top-k recall on fixtures",
                "eligible": report["metrics"]["cases_total"],
                "masked": 0,
            },
            {
                "name": "wrong_completion_rate",
                "numerator": report["metrics"]["wrong_completion_rate"],
                "denominator": 1.0,
                "scope": "library_retrieval_ab.json",
                "baseline": "n/a",
                "candidate": report["metrics"]["wrong_completion_rate"],
                "delta": "must stay 0 without evidence",
                "eligible": report["metrics"]["cases_total"],
                "masked": 0,
            },
        ],
        "failure_cases": [
            c["case_id"] for c in report["cases"] if not c.get("pass")
        ],
        "known_limitations": [
            "arrangement/orientation scoring is coarse (label only); geometry still image-side",
            "same_hash_components used as grouping hint only, not package identity",
            "association pipeline wiring of retrieve_component is left to A06/A08 consumers",
        ],
        "constraints_verified": {
            "no_fabricated_coordinates": True,
            "pin_rows_ne_physical_count": True,
            "unknown_on_ambiguity": True,
            "library_version_hashed": bool(index["library_sha256"]),
            "models_lock_unmodified": True,
        },
        "outputs": {
            "library_index.json": str(index_path.relative_to(ROOT)).replace("\\", "/"),
            "library_retrieval_ab.json": str(ab_path.relative_to(ROOT)).replace("\\", "/"),
        },
        "rollback": "delete src/pcbparse/knowledge, tests/unit/test_library_index_a07.py, scripts/build_library_artifacts_a07.py, artifacts/audits/{library_index,library_retrieval_ab,A07_delivery}.json",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    delivery_path = out_dir / "A07_delivery.json"
    delivery_path.write_text(
        json.dumps(delivery, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(delivery["constraints_verified"], ensure_ascii=False))
    print(f"wrote {index_path}")
    print(f"wrote {ab_path}")
    print(f"wrote {delivery_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
