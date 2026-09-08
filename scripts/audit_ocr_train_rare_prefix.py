from __future__ import annotations

"""Hash-bound OCR train-only ASCII-prefix long-tail diagnostic.

This script intentionally evaluates a model on selected *training* crops.  The
result measures memorisation/fit of rare alphanumeric prefixes, not
generalisation, and is therefore never promotion evidence.  Only complete raw
texts matching ``^[A-Za-z]+[0-9]+$`` from the frozen train materialization are
scored.  Validation and test records are structurally excluded.
"""

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_PREFIX_PATTERN = re.compile(r"([A-Za-z]+)[0-9]+\Z", flags=re.ASCII)
_FINAL_STATUSES = frozenset({"candidate_evidence_ready", "gate_failed"})


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Audit OCR rare ASCII letter-prefix + digit recognition on the "
            "hash-bound train split only (diagnostic; never promotion evidence)."
        )
    )
    parser.add_argument("--workspace-root", type=Path, default=root)
    parser.add_argument(
        "--bundle-manifest",
        type=Path,
        default=Path("work/train-ocr-v2c/bundle.manifest.json"),
    )
    parser.add_argument(
        "--run-status",
        type=Path,
        default=Path("work/train-ocr-v2c/torch_run_status.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("work/train-ocr-v2c/model.pt"),
    )
    parser.add_argument(
        "--materialization-root",
        type=Path,
        default=Path("work/materialized/ocr-v2-512"),
    )
    parser.add_argument(
        "--source-root", type=Path, default=Path("Dataset300/Dataset300")
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/manifests/dataset300.jsonl"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/splits/dataset300_split.jsonl"),
    )
    parser.add_argument(
        "--ocr-manifest",
        type=Path,
        default=Path("data/derived/dataset300-v2/ocr_text.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/ocr_v2c_train_rare_prefix.json"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def _resolve_within(
    root: Path,
    raw: Path,
    label: str,
    *,
    kind: str,
) -> Path:
    candidate = raw.expanduser()
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve(
        strict=False
    )
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside workspace: {resolved}") from exc
    if kind == "file" and (not resolved.is_file() or resolved.stat().st_size <= 0):
        raise FileNotFoundError(f"{label} is missing or empty: {resolved}")
    if kind == "directory" and not resolved.is_dir():
        raise FileNotFoundError(f"{label} directory is missing: {resolved}")
    if kind not in {"file", "directory", "output"}:
        raise ValueError(f"unsupported path kind {kind!r}")
    return resolved


def _safe_child(root: Path, raw: str, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} path must be a non-empty string")
    candidate = Path(raw)
    if candidate.is_absolute():
        raise ValueError(f"{label} path must be relative")
    resolved = (root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes its root: {raw}") from exc
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"{label} is missing or empty: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(
                    raw,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"invalid {label} JSONL at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{label} row {line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{label} is empty")
    return rows


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an array")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _expect_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{label} mismatch: expected={expected!r}, actual={actual!r}"
        )


def _expect_hash(actual: str, expected: Any, label: str) -> None:
    if not _is_sha256(expected):
        raise ValueError(f"{label} expected SHA-256 is invalid")
    _expect_equal(actual, str(expected).lower(), f"{label} SHA-256")


def _unique_index(
    rows: Iterable[Mapping[str, Any]], key: str, label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        identity = _text(row.get(key), f"{label}.{key}")
        if identity in result:
            raise ValueError(f"duplicate {label} {key}: {identity}")
        result[identity] = row
    return result


def ascii_prefix(value: Any) -> str | None:
    """Return an uppercase ASCII prefix only for a complete prefix+digit text."""

    if not isinstance(value, str):
        return None
    match = _PREFIX_PATTERN.fullmatch(value)
    return match.group(1).upper() if match is not None else None


def select_train_prefix_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select included train rows; val/test can never pass this boundary."""

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        record_id = _text(raw.get("record_id"), "recognition record_id")
        if record_id in seen:
            raise ValueError(f"duplicate recognition record_id: {record_id}")
        seen.add(record_id)
        if raw.get("split") != "train" or raw.get("status") != "included":
            continue
        reference = raw.get("content")
        prefix = ascii_prefix(reference)
        if prefix is None:
            continue
        row = dict(raw)
        row["audit_prefix"] = prefix
        row["audit_prefix_case_variant"] = str(reference)[: len(prefix)]
        selected.append(row)
    selected.sort(key=lambda row: str(row["record_id"]))
    if not selected:
        raise ValueError("no included train rows match the complete ASCII prefix+digits policy")
    if any(row.get("split") != "train" for row in selected):
        raise RuntimeError("non-train row crossed the prefix selection boundary")
    return selected


def levenshtein(left: str, right: str) -> int:
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


def metric(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    support = len(rows)
    exact_count = 0
    edit_distance = 0
    reference_codepoints = 0
    confidence = 0.0
    samples: set[str] = set()
    case_variants: set[str] = set()
    for row in rows:
        reference = _text(row.get("reference"), "metric reference")
        prediction = row.get("prediction")
        if not isinstance(prediction, str):
            raise ValueError("metric prediction must be a string")
        score = _finite(row.get("score"), "metric score")
        if not 0.0 <= score <= 1.0:
            raise ValueError("metric score must be in [0, 1]")
        exact_count += int(reference == prediction)
        edit_distance += levenshtein(prediction, reference)
        reference_codepoints += len(reference)
        confidence += score
        samples.add(_text(row.get("sample_id"), "metric sample_id"))
        variant = row.get("prefix_case_variant")
        if isinstance(variant, str) and variant:
            case_variants.add(variant)
    return {
        "support": support,
        "sample_support": len(samples),
        "exact_count": exact_count,
        "error_count": support - exact_count,
        "exact_match": exact_count / support if support else None,
        "edit_distance": edit_distance,
        "reference_codepoints": reference_codepoints,
        "cer": edit_distance / reference_codepoints if reference_codepoints else None,
        "mean_confidence": confidence / support if support else None,
        "prefix_case_variants": sorted(case_variants),
    }


def _rarity_bucket(support: int) -> str:
    if support == 1:
        return "singleton_1"
    if support <= 4:
        return "very_rare_2_4"
    if support <= 19:
        return "rare_5_19"
    if support <= 99:
        return "mid_20_99"
    return "head_100_plus"


def summarize_predictions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_prefix_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_domain_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_prefix_domain_rows: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        prefix = _text(row.get("prefix"), "prediction prefix")
        domain = _text(row.get("source_domain"), "prediction source_domain")
        by_prefix_rows[prefix].append(row)
        by_domain_rows[domain].append(row)
        by_prefix_domain_rows[prefix][domain].append(row)
    prefix_metrics = {
        prefix: metric(group) for prefix, group in sorted(by_prefix_rows.items())
    }
    domain_metrics = {
        domain: metric(group) for domain, group in sorted(by_domain_rows.items())
    }
    prefix_domain_metrics = {
        prefix: {
            domain: metric(group) for domain, group in sorted(domains.items())
        }
        for prefix, domains in sorted(by_prefix_domain_rows.items())
    }
    rarity_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for prefix, group in by_prefix_rows.items():
        rarity_rows[_rarity_bucket(len(group))].extend(group)
    rarity = {
        name: {
            **metric(group),
            "prefix_support": sum(
                _rarity_bucket(len(prefix_group)) == name
                for prefix_group in by_prefix_rows.values()
            ),
        }
        for name, group in sorted(rarity_rows.items())
    }
    supported_prefixes = len(prefix_metrics)
    zero_exact = sorted(
        prefix for prefix, values in prefix_metrics.items() if values["exact_count"] == 0
    )
    worst = sorted(
        (
            {
                "prefix": prefix,
                "support": values["support"],
                "exact_match": values["exact_match"],
                "cer": values["cer"],
            }
            for prefix, values in prefix_metrics.items()
            if int(values["support"]) >= 5
        ),
        key=lambda row: (
            float(row["exact_match"]),
            -float(row["cer"]),
            -int(row["support"]),
            str(row["prefix"]),
        ),
    )[:25]
    return {
        "aggregate": metric(rows),
        "by_prefix": prefix_metrics,
        "by_domain": domain_metrics,
        "by_prefix_domain": prefix_domain_metrics,
        "by_rarity_bucket": rarity,
        "diagnostics": {
            "prefix_count": supported_prefixes,
            "singleton_prefix_count": sum(
                int(values["support"]) == 1 for values in prefix_metrics.values()
            ),
            "prefixes_with_zero_exact": zero_exact,
            "prefixes_with_zero_exact_count": len(zero_exact),
            "worst_prefixes_min_support_5": worst,
        },
    }


def validate_contract(
    *,
    bundle_manifest: Mapping[str, Any],
    run_status: Mapping[str, Any],
    checkpoint_lineage: Mapping[str, Any],
    materialization_manifest: Mapping[str, Any],
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Validate the immutable checkpoint/config/materialization hash chain."""

    if bundle_manifest.get("task") != "ocr":
        raise ValueError("bundle manifest task is not OCR")
    if run_status.get("task") != "ocr":
        raise ValueError("run status task is not OCR")
    if run_status.get("status") not in _FINAL_STATUSES:
        raise ValueError("OCR run is not a completed formal run")
    if run_status.get("formal_evidence") is not True or run_status.get("smoke") is not False:
        raise ValueError("OCR run status is not formal non-smoke evidence")
    if checkpoint_lineage.get("task") != "ocr":
        raise ValueError("checkpoint lineage task is not OCR")
    if (
        checkpoint_lineage.get("formal_evidence") is not True
        or checkpoint_lineage.get("smoke") is not False
    ):
        raise ValueError("checkpoint lineage is not formal non-smoke evidence")

    checkpoint_hash = hashes["checkpoint"]
    _expect_hash(
        checkpoint_hash, bundle_manifest.get("artifact_sha256"), "checkpoint/bundle"
    )
    status_artifact = _mapping(run_status.get("artifact"), "status artifact")
    _expect_hash(checkpoint_hash, status_artifact.get("sha256"), "checkpoint/status")

    config_hash = bundle_manifest.get("training_config_sha256")
    if not _is_sha256(config_hash):
        raise ValueError("bundle training_config_sha256 is invalid")
    status_lineage = _mapping(run_status.get("lineage"), "status lineage")
    _expect_equal(
        status_lineage.get("training_config_sha256"),
        config_hash,
        "status/bundle training config hash",
    )
    _expect_equal(
        checkpoint_lineage.get("training_config_sha256"),
        config_hash,
        "checkpoint/bundle training config hash",
    )

    materialization_hash = hashes["materialization_manifest"]
    _expect_equal(
        checkpoint_lineage.get("materialization_commit_kind"),
        "manifest_sha256",
        "checkpoint materialization commit kind",
    )
    for key in ("materialization_commit_id", "materialization_commit_sha256"):
        _expect_hash(materialization_hash, checkpoint_lineage.get(key), f"checkpoint {key}")
        _expect_hash(materialization_hash, status_lineage.get(key), f"status {key}")

    material_lineage = _mapping(
        materialization_manifest.get("lineage"), "materialization lineage"
    )
    expected_hashes = {
        "source_manifest_sha256": hashes["source_manifest"],
        "split_manifest_sha256": hashes["split_manifest"],
        "ocr_manifest_sha256": hashes["ocr_manifest"],
    }
    for key, expected in expected_hashes.items():
        _expect_hash(expected, material_lineage.get(key), f"materialization {key}")
        checkpoint_key = (
            "task_manifest_sha256" if key == "ocr_manifest_sha256" else key
        )
        _expect_hash(
            expected,
            checkpoint_lineage.get(checkpoint_key),
            f"checkpoint {checkpoint_key}",
        )
        status_key = "task_manifest_sha256" if key == "ocr_manifest_sha256" else key
        _expect_hash(expected, status_lineage.get(status_key), f"status {status_key}")
    _expect_hash(
        hashes["source_manifest"],
        bundle_manifest.get("source_manifest_sha256"),
        "bundle source manifest",
    )
    _expect_hash(
        hashes["split_manifest"],
        bundle_manifest.get("split_manifest_sha256"),
        "bundle split manifest",
    )
    gate_policy = _mapping(bundle_manifest.get("gate_policy"), "bundle gate policy")
    _expect_hash(
        hashes["ocr_manifest"],
        gate_policy.get("task_manifest_sha256"),
        "bundle OCR manifest",
    )
    return {
        "checkpoint_sha256": checkpoint_hash,
        "training_config_sha256": str(config_hash),
        "materialization_manifest_sha256": materialization_hash,
        "source_manifest_sha256": hashes["source_manifest"],
        "split_manifest_sha256": hashes["split_manifest"],
        "ocr_manifest_sha256": hashes["ocr_manifest"],
        "hash_chain_verified": True,
    }


def _bbox_tuple(value: Any, label: str) -> tuple[float, float, float, float]:
    box = _mapping(value, label)
    result = tuple(_finite(box.get(key), label) for key in ("x1", "y1", "x2", "y2"))
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"{label} has non-positive size")
    return result  # type: ignore[return-value]


def _join_frozen_sources(
    *,
    selected: Sequence[Mapping[str, Any]],
    source_manifest_path: Path,
    split_manifest_path: Path,
    ocr_manifest_path: Path,
    source_root: Path,
) -> dict[str, Any]:
    selected_by_record = {
        str(row["record_id"]): row for row in selected
    }
    source_by_sample = _unique_index(
        _read_jsonl(source_manifest_path, "source manifest"),
        "sample_id",
        "source manifest",
    )
    split_by_sample = {
        sample_id: _text(row.get("split"), "split")
        for sample_id, row in _unique_index(
            _read_jsonl(split_manifest_path, "split manifest"),
            "sample_id",
            "split manifest",
        ).items()
    }
    if set(source_by_sample) != set(split_by_sample):
        raise ValueError("source/split manifest sample populations differ")
    derived_selected: dict[str, Mapping[str, Any]] = {}
    with ocr_manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(
                    raw,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"invalid OCR manifest JSONL at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"OCR manifest row {line_number} is not an object")
            record_id = row.get("record_id")
            if record_id not in selected_by_record:
                continue
            if record_id in derived_selected:
                raise ValueError(f"duplicate selected OCR manifest row: {record_id}")
            derived_selected[str(record_id)] = row
    if set(derived_selected) != set(selected_by_record):
        missing = sorted(set(selected_by_record) - set(derived_selected))
        raise ValueError(f"selected rows missing from frozen OCR manifest: {missing[:5]}")

    source_paths: dict[str, str] = {}
    sample_ids: set[str] = set()
    for record_id, materialized in selected_by_record.items():
        derived = derived_selected[record_id]
        sample_id = _text(materialized.get("sample_id"), "materialized sample_id")
        sample_ids.add(sample_id)
        if split_by_sample.get(sample_id) != "train":
            raise ValueError(f"selected row is not in frozen train split: {record_id}")
        for key in ("sample_id", "split", "content", "source_domain", "source_image"):
            _expect_equal(
                derived.get(key), materialized.get(key), f"derived/materialized {key}"
            )
        _expect_equal(
            _bbox_tuple(derived.get("bbox"), "derived bbox"),
            _bbox_tuple(materialized.get("bbox"), "materialized bbox"),
            "derived/materialized bbox",
        )
        source = source_by_sample[sample_id]
        source_image = _mapping(
            _mapping(source.get("files"), "source files").get("image"),
            "source image",
        )
        relative = _text(materialized.get("source_image"), "source image path")
        expected_hash = _text(
            materialized.get("source_image_sha256"), "source image SHA-256"
        )
        _expect_equal(source_image.get("path"), relative, "source image path")
        _expect_equal(source_image.get("sha256"), expected_hash, "source image hash")
        existing = source_paths.setdefault(relative, expected_hash)
        _expect_equal(existing, expected_hash, "source image hash across selected rows")

    verified_inventory: list[dict[str, str]] = []
    for relative, expected_hash in sorted(source_paths.items()):
        source_path = _safe_child(source_root, relative, "selected source image")
        actual_hash = _sha256(source_path)
        _expect_hash(actual_hash, expected_hash, f"selected source image {relative}")
        verified_inventory.append({"path": relative, "sha256": actual_hash})
    inventory_payload = json.dumps(
        verified_inventory,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "selected_derived_rows_joined": len(derived_selected),
        "selected_source_samples": len(sample_ids),
        "selected_source_images_verified": len(source_paths),
        "selected_source_inventory_sha256": hashlib.sha256(
            inventory_payload
        ).hexdigest(),
    }


def _relative(path: Path, workspace: Path) -> str:
    return path.relative_to(workspace).as_posix()


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if type(arguments.batch_size) is not int or not 1 <= arguments.batch_size <= 256:
        raise ValueError("batch size must be an integer in [1, 256]")
    workspace = arguments.workspace_root.expanduser().resolve(strict=True)
    bundle_manifest_path = _resolve_within(
        workspace, arguments.bundle_manifest, "bundle manifest", kind="file"
    )
    run_status_path = _resolve_within(
        workspace, arguments.run_status, "run status", kind="file"
    )
    checkpoint_path = _resolve_within(
        workspace, arguments.checkpoint, "OCR checkpoint", kind="file"
    )
    materialization_root = _resolve_within(
        workspace,
        arguments.materialization_root,
        "OCR materialization",
        kind="directory",
    )
    source_root = _resolve_within(
        workspace, arguments.source_root, "source root", kind="directory"
    )
    source_manifest_path = _resolve_within(
        workspace, arguments.source_manifest, "source manifest", kind="file"
    )
    split_manifest_path = _resolve_within(
        workspace, arguments.split_manifest, "split manifest", kind="file"
    )
    ocr_manifest_path = _resolve_within(
        workspace, arguments.ocr_manifest, "OCR manifest", kind="file"
    )
    output_path = _resolve_within(
        workspace, arguments.output, "audit output", kind="output"
    )
    materialization_manifest_path = materialization_root / "manifest.json"
    if not materialization_manifest_path.is_file():
        raise FileNotFoundError("OCR materialization manifest is missing")

    paths = {
        "bundle_manifest": bundle_manifest_path,
        "run_status": run_status_path,
        "checkpoint": checkpoint_path,
        "materialization_manifest": materialization_manifest_path,
        "source_manifest": source_manifest_path,
        "split_manifest": split_manifest_path,
        "ocr_manifest": ocr_manifest_path,
    }
    hashes = {name: _sha256(path) for name, path in paths.items()}
    bundle_manifest = _read_object(bundle_manifest_path, "bundle manifest")
    run_status = _read_object(run_status_path, "run status")
    materialization_manifest = _read_object(
        materialization_manifest_path, "materialization manifest"
    )
    if materialization_manifest.get("complete") is not True:
        raise ValueError("OCR materialization is incomplete")
    if materialization_manifest.get("materialize_images") is not True:
        raise ValueError("rare-prefix audit requires hash-bound materialized crops")
    material_artifacts = _mapping(
        materialization_manifest.get("artifacts"), "materialization artifacts"
    )
    verified_artifact_hashes: dict[str, str] = {}
    for name, expected in sorted(material_artifacts.items()):
        path = _safe_child(materialization_root, str(name), f"OCR artifact {name}")
        actual = _sha256(path)
        _expect_hash(actual, expected, f"OCR artifact {name}")
        verified_artifact_hashes[str(name)] = actual

    sys.path.insert(0, str(workspace / "src"))
    import torch

    from pcbparse.training.torch_ocr import (
        TorchOCRRecognitionDataset,
        ctc_collate,
    )
    from pcbparse.training.torch_ocr_model import (
        greedy_ctc_decode,
        load_ocr_bundle,
        recognition_output_lengths,
    )

    loaded = load_ocr_bundle(
        checkpoint_path,
        device="cpu",
        expected_sha256=hashes["checkpoint"],
        strict=True,
    )
    contract = validate_contract(
        bundle_manifest=bundle_manifest,
        run_status=run_status,
        checkpoint_lineage=loaded.lineage,
        materialization_manifest=materialization_manifest,
        hashes=hashes,
    )
    configured_artifact = _safe_child(
        bundle_manifest_path.parent,
        _text(bundle_manifest.get("artifact_path"), "bundle artifact_path"),
        "bundle checkpoint",
    )
    _expect_equal(configured_artifact, checkpoint_path, "bundle checkpoint path")

    recognition_height = loaded.preprocess.get("recognition_height")
    pad_value = loaded.preprocess.get("recognition_pad_value")
    if type(recognition_height) is not int or recognition_height <= 0:
        raise ValueError("checkpoint recognition_height is invalid")
    pad_value = _finite(pad_value, "checkpoint recognition_pad_value")
    dataset = TorchOCRRecognitionDataset(
        materialization_root,
        split="train",
        image_mode="materialized",
        image_height=recognition_height,
        verify_image_hashes=True,
    )
    if tuple(dataset.charset.characters) != tuple(loaded.charset.characters):
        raise ValueError("checkpoint charset differs from frozen materialization")
    selected_rows = select_train_prefix_rows(dataset.records)
    selected_ids = {str(row["record_id"]) for row in selected_rows}
    selected_indices = [
        index
        for index, row in enumerate(dataset.records)
        if str(row["record_id"]) in selected_ids
    ]
    if len(selected_indices) != len(selected_rows):
        raise RuntimeError("selected recognition index cardinality mismatch")
    source_join = _join_frozen_sources(
        selected=selected_rows,
        source_manifest_path=source_manifest_path,
        split_manifest_path=split_manifest_path,
        ocr_manifest_path=ocr_manifest_path,
        source_root=source_root,
    )

    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, selected_indices),
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=lambda batch: ctc_collate(batch, pad_value=pad_value),
    )
    selected_by_id = {str(row["record_id"]): row for row in selected_rows}
    predictions: list[dict[str, Any]] = []
    seen_predictions: set[str] = set()
    loaded.recognizer.to("cpu")
    loaded.recognizer.eval()
    with torch.inference_mode():
        for batch in loader:
            metadata = batch.get("metadata")
            if not isinstance(metadata, list) or any(
                row.get("split") != "train" for row in metadata
            ):
                raise RuntimeError("non-train row reached rare-prefix inference")
            images = batch["images"].to(device="cpu", dtype=torch.float32)
            widths = batch["image_widths"].to(device="cpu", dtype=torch.long)
            logits = loaded.recognizer(images, widths)
            decoded, scores = greedy_ctc_decode(
                logits,
                recognition_output_lengths(widths),
                loaded.charset,
            )
            for index, (prediction, score) in enumerate(
                zip(decoded, scores, strict=True)
            ):
                record_id = str(batch["record_ids"][index])
                if record_id in seen_predictions or record_id not in selected_by_id:
                    raise RuntimeError("rare-prefix prediction identity mismatch")
                seen_predictions.add(record_id)
                selected = selected_by_id[record_id]
                reference = str(selected["content"])
                prefix = str(selected["audit_prefix"])
                domain = _text(metadata[index].get("source_domain"), "source domain")
                score_value = _finite(score, "recognition confidence")
                if not 0.0 <= score_value <= 1.0:
                    raise ValueError("recognition confidence is outside [0, 1]")
                predictions.append(
                    {
                        "record_id": record_id,
                        "sample_id": str(batch["sample_ids"][index]),
                        "prefix": prefix,
                        "prefix_case_variant": str(
                            selected["audit_prefix_case_variant"]
                        ),
                        "source_domain": domain,
                        "reference": reference,
                        "prediction": prediction,
                        "score": score_value,
                        "exact": prediction == reference,
                        "edit_distance": levenshtein(prediction, reference),
                    }
                )
    if seen_predictions != selected_ids:
        missing = sorted(selected_ids - seen_predictions)
        raise RuntimeError(f"rare-prefix inference is incomplete: {missing[:5]}")
    predictions.sort(key=lambda row: str(row["record_id"]))
    prediction_payload = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in predictions
    ).encode("utf-8")
    summary = summarize_predictions(predictions)
    k_errors = [
        {
            "record_id": row["record_id"],
            "sample_id": row["sample_id"],
            "source_domain": row["source_domain"],
            "reference": row["reference"],
            "prediction": row["prediction"],
            "score": row["score"],
        }
        for row in predictions
        if row["prefix"] == "K" and not row["exact"]
    ]

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pcbparse_ocr_train_rare_prefix_diagnostic",
        "selection_boundary": {
            "split": "train",
            "selection_regex": "^[A-Za-z]+[0-9]+$",
            "complete_raw_text_match_required": True,
            "ascii_only": True,
            "included_materialization_rows_only": True,
            "train_rows_used": len(predictions),
            "val_rows_used": 0,
            "test_rows_used": 0,
            "device": "cpu",
            "gpu_used": False,
            "thresholds_tuned": False,
        },
        "evidence_role": {
            "diagnostic_only": True,
            "promotion_eligible": False,
            "may_not_be_used_for_promotion": True,
            "reason": (
                "metrics are measured on selected training crops and quantify "
                "long-tail fit/memorisation, not frozen-validation generalisation"
            ),
        },
        "hash_contract": contract,
        "sources": {
            name: {
                "path": _relative(path, workspace),
                "sha256": hashes[name],
                "verified": True,
            }
            for name, path in paths.items()
        },
        "materialization": {
            "root": _relative(materialization_root, workspace),
            "manifest_sha256": hashes["materialization_manifest"],
            "artifact_hashes_verified": verified_artifact_hashes,
            "selected_crop_hashes_verified": len(predictions),
            **source_join,
        },
        "checkpoint": {
            "epoch": loaded.epoch,
            "version": loaded.version,
            "charset_codepoints": len(loaded.charset.characters),
            "preprocess": dict(loaded.preprocess),
        },
        "population": {
            "materialized_train_included_rows": len(dataset.records),
            "selected_complete_ascii_prefix_digit_rows": len(predictions),
            "selected_fraction_of_train": len(predictions) / len(dataset.records),
            "prediction_population_sha256": hashlib.sha256(
                prediction_payload
            ).hexdigest(),
        },
        **summary,
        "spot_checks": {
            "K": summary["by_prefix"].get("K"),
            "K_errors": k_errors,
        },
    }
    canonical = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    report["audit_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_path)
    print(f"audit_payload_sha256={report['audit_payload_sha256']}")
    print(
        "K="
        + json.dumps(report["spot_checks"]["K"], ensure_ascii=False, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
