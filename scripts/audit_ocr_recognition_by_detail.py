from __future__ import annotations

"""Audit frozen-validation OCR recognition by annotation ``details``.

The script is intentionally independent of model inference.  It joins an
already-produced, hash-bound prediction JSONL to the immutable Dataset300-v2
derived OCR manifest by ``record_id`` and refuses incomplete or inconsistent
evidence.  Test rows are never read into the metric population.
"""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratify frozen-val OCR recognition evidence by details."
    )
    parser.add_argument(
        "--predictions",
        default="work/train-ocr-v2c/predictions.jsonl",
    )
    parser.add_argument(
        "--derived",
        default="data/derived/dataset300-v2/ocr_text.jsonl",
    )
    parser.add_argument(
        "--metrics",
        default="work/train-ocr-v2c/metrics.json",
    )
    parser.add_argument(
        "--model",
        default="work/train-ocr-v2c/model.pt",
    )
    parser.add_argument(
        "--output",
        default="artifacts/audits/ocr_v2c_val_recognition_by_detail.json",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw, parse_constant=_reject_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_index(
    rows: Iterable[Mapping[str, Any]], key: str, label: str
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        identity = row.get(key)
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"{label} row lacks non-empty {key}")
        if identity in output:
            raise ValueError(f"duplicate {label} {key}: {identity}")
        output[identity] = row
    return output


def _levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _canonical_detail(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    normalized = "".join(character for character in value.casefold() if character.isalnum())
    aliases = {
        "designator": "designator",
        "refdes": "designator",
        "reference": "designator",
        "pinnumber": "pinnumber",
        "pinno": "pinnumber",
        "pinname": "pinname",
        "netlabel": "netlabel",
        "value": "value",
        "searchname": "searchName",
        "partname": "searchName",
    }
    return aliases.get(normalized, value.strip())


def _metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    support = len(rows)
    exact_count = sum(bool(row["exact"]) for row in rows)
    edit_distance = sum(int(row["edit_distance"]) for row in rows)
    reference_codepoints = sum(len(str(row["reference"])) for row in rows)
    scores = [float(row["score"]) for row in rows]
    samples = {str(row["sample_id"]) for row in rows}
    oov_support = sum(bool(row["charset_oov"]) for row in rows)
    return {
        "support": support,
        "sample_support": len(samples),
        "exact_count": exact_count,
        "exact_match": exact_count / support if support else None,
        "edit_distance": edit_distance,
        "reference_codepoints": reference_codepoints,
        "cer": edit_distance / reference_codepoints if reference_codepoints else None,
        "mean_confidence": sum(scores) / support if support else None,
        "charset_oov_support": oov_support,
        "charset_oov_rate": oov_support / support if support else None,
    }


def build_audit(
    *,
    predictions_path: Path,
    derived_path: Path,
    metrics_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    for label, path in (
        ("predictions", predictions_path),
        ("derived manifest", derived_path),
        ("metrics", metrics_path),
        ("model", model_path),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"{label} is missing or empty: {path}")

    all_predictions = _read_jsonl(predictions_path)
    recognition_rows = [
        row for row in all_predictions if row.get("kind") == "ocr_recognition"
    ]
    prediction_index = _unique_index(recognition_rows, "record_id", "recognition")
    derived_rows = _read_jsonl(derived_path)
    val_rows = [row for row in derived_rows if row.get("split") == "val"]
    val_index = _unique_index(val_rows, "record_id", "derived val")

    unexpected = sorted(set(prediction_index) - set(val_index))
    if unexpected:
        raise ValueError(
            "recognition predictions include non-val/unknown record ids: "
            + ", ".join(unexpected[:8])
        )
    missing = sorted(set(val_index) - set(prediction_index))

    joined: list[dict[str, Any]] = []
    reference_mismatches: list[str] = []
    sample_mismatches: list[str] = []
    exact_flag_mismatches: list[str] = []
    for record_id in sorted(prediction_index):
        prediction = prediction_index[record_id]
        truth = val_index[record_id]
        reference = prediction.get("reference")
        predicted = prediction.get("prediction")
        if not isinstance(reference, str) or not isinstance(predicted, str):
            raise ValueError(f"prediction {record_id} lacks string prediction/reference")
        if reference != truth.get("content"):
            reference_mismatches.append(record_id)
        if prediction.get("sample_id") != truth.get("sample_id"):
            sample_mismatches.append(record_id)
        exact = predicted == reference
        if prediction.get("exact") is not exact:
            exact_flag_mismatches.append(record_id)
        score = prediction.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ValueError(f"prediction {record_id} has invalid confidence")
        if type(prediction.get("charset_oov")) is not bool:
            raise ValueError(f"prediction {record_id} lacks charset_oov boolean")
        joined.append(
            {
                "record_id": record_id,
                "sample_id": str(truth["sample_id"]),
                "details": _canonical_detail(truth.get("details")),
                "reference": reference,
                "prediction": predicted,
                "exact": exact,
                "edit_distance": _levenshtein(predicted, reference),
                "score": float(score),
                "charset_oov": bool(prediction["charset_oov"]),
            }
        )
    if reference_mismatches or sample_mismatches or exact_flag_mismatches:
        raise ValueError(
            "prediction/derived join is inconsistent: "
            f"reference={len(reference_mismatches)}, "
            f"sample={len(sample_mismatches)}, exact={len(exact_flag_mismatches)}"
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        grouped[str(row["details"])].append(row)
    by_detail = {
        detail: _metric(rows)
        for detail, rows in sorted(grouped.items(), key=lambda item: item[0].casefold())
    }
    aggregate = _metric(joined)
    metrics_document = _read_json(metrics_path)
    source_metrics = metrics_document.get("metrics")
    if not isinstance(source_metrics, dict):
        raise ValueError("metrics document lacks metrics object")
    expected_count = source_metrics.get("recognition_evaluated_records")
    expected_exact = source_metrics.get("normalized_exact_match")
    expected_cer = source_metrics.get("recognition_cer")
    expected_excluded = source_metrics.get("recognition_excluded_policy_records")
    if expected_count != float(len(joined)):
        raise ValueError("joined recognition support differs from source metrics")
    if not math.isclose(
        float(expected_exact), float(aggregate["exact_match"]), rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError("joined exact match differs from source metrics")
    if not math.isclose(
        float(expected_cer), float(aggregate["cer"]), rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError("joined CER differs from source metrics")
    if expected_excluded != float(len(missing)):
        raise ValueError("missing val rows differ from excluded-policy count")

    missing_by_detail: dict[str, int] = defaultdict(int)
    for record_id in missing:
        missing_by_detail[_canonical_detail(val_index[record_id].get("details"))] += 1
    return {
        "schema_version": 1,
        "kind": "pcbparse_ocr_frozen_val_recognition_by_detail_audit",
        "selection_boundary": {
            "split": "val",
            "test_rows_used": 0,
            "thresholds_tuned": False,
            "purpose": "candidate diagnosis and next-run gate design",
        },
        "sources": {
            "predictions": {
                "path": predictions_path.as_posix(),
                "sha256": _sha256(predictions_path),
            },
            "derived_manifest": {
                "path": derived_path.as_posix(),
                "sha256": _sha256(derived_path),
            },
            "metrics": {
                "path": metrics_path.as_posix(),
                "sha256": _sha256(metrics_path),
            },
            "model": {
                "path": model_path.as_posix(),
                "sha256": _sha256(model_path),
            },
        },
        "join": {
            "derived_val_rows": len(val_rows),
            "recognition_prediction_rows": len(recognition_rows),
            "joined_rows": len(joined),
            "unexpected_prediction_rows": 0,
            "reference_mismatches": 0,
            "sample_mismatches": 0,
            "exact_flag_mismatches": 0,
            "excluded_policy_rows": len(missing),
            "excluded_policy_record_ids": missing,
            "excluded_policy_by_detail": dict(sorted(missing_by_detail.items())),
        },
        "aggregate": aggregate,
        "by_detail": by_detail,
        "interpretation": {
            "aggregate_exact_is_not_key_readiness": True,
            "component_key_readiness_fields": ["designator"],
            "pin_identity_readiness_fields": ["pinnumber", "pinname"],
            "net_semantics_readiness_fields": ["netlabel"],
            "promotion_requires_frozen_val_gates": True,
        },
    }


def main() -> int:
    args = _arguments()
    paths = {
        name: Path(value).expanduser().resolve(strict=False)
        for name, value in vars(args).items()
    }
    audit = build_audit(
        predictions_path=paths["predictions"],
        derived_path=paths["derived"],
        metrics_path=paths["metrics"],
        model_path=paths["model"],
    )
    output = paths["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        audit,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    output.write_text(payload, encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _sha256(output),
                "support": audit["aggregate"]["support"],
                "by_detail": audit["by_detail"],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
