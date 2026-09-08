from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from pydantic import ValidationError

from .config import DATASET_TASK_BY_MODEL_TASK, TrainingConfig, load_training_config, training_config_sha256


_ALLOWED_SPLITS = frozenset({"train", "val", "test"})


def preflight_training_data(
    config_path: str | Path,
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate immutable Dataset300-v2 lineage before a trainer can start.

    The report is JSON-serializable and intentionally accumulates errors rather
    than failing on the first corrupt record.  A backend must require ``ok`` to
    be true; existence of the JSONL files alone is not sufficient.
    """

    requested_config = Path(config_path).expanduser()
    root = (
        Path(workspace_root).expanduser().resolve(strict=False)
        if workspace_root is not None
        else Path.cwd().resolve()
    )
    resolved_config = _resolve_path(requested_config, root)
    issues: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": 1,
        "ok": False,
        "workspace_root": str(root),
        "config_path": str(resolved_config),
        "checks": checks,
        "issues": issues,
    }
    try:
        config = load_training_config(resolved_config)
    except (ValueError, ValidationError) as exc:
        _issue(issues, "config_invalid", str(exc), path=resolved_config)
        return report

    report["training_config_sha256"] = training_config_sha256(config)
    binding = config.dataset
    try:
        paths = {
            "derived_root": _resolve_within(root, binding.derived_root),
            "summary": _resolve_within(root, binding.summary),
            "source_root": _resolve_within(root, binding.source_root),
            "source_manifest": _resolve_within(root, binding.source_manifest),
            "split_manifest": _resolve_within(root, binding.split_manifest),
        }
    except ValueError as exc:
        _issue(issues, "path_outside_workspace", str(exc))
        return report
    report["paths"] = {key: str(value) for key, value in paths.items()}

    expected_file_hashes = {
        "summary": binding.summary_sha256,
        "source_manifest": binding.source_manifest_sha256,
        "split_manifest": binding.split_manifest_sha256,
    }
    actual_hashes: dict[str, str | None] = {}
    for name, expected in expected_file_hashes.items():
        actual = _verify_file_hash(paths[name], expected, name, issues)
        actual_hashes[name] = actual
    report["hashes"] = actual_hashes

    summary = _read_json_object(paths["summary"], "summary", issues)
    if summary is None:
        return _finish(report)
    _check_equal(
        summary.get("derivation_version"),
        binding.derivation_version,
        "derivation_version",
        issues,
    )
    _check_equal(summary.get("config_hash"), binding.config_hash, "config_hash", issues)
    _check_equal(
        summary.get("sample_count"),
        binding.expected_sample_count,
        "sample_count",
        issues,
    )
    summary_config = summary.get("config")
    if not isinstance(summary_config, dict):
        _issue(issues, "summary_config_invalid", "summary.config must be an object")
    else:
        _check_equal(
            summary_config.get("split_manifest_sha256"),
            binding.split_manifest_sha256,
            "summary.config.split_manifest_sha256",
            issues,
        )
    _check(
        checks,
        "summary_lineage",
        not any(issue["code"].startswith("summary_") or issue["code"] == "mismatch" for issue in issues),
        "summary version, config hash, sample count, and split binding",
    )

    source_rows = _read_jsonl(paths["source_manifest"], "source_manifest", issues)
    split_rows = _read_jsonl(paths["split_manifest"], "split_manifest", issues)
    source_by_id = _index_samples(source_rows, "source_manifest", issues)
    split_by_id = _index_samples(split_rows, "split_manifest", issues)
    expected_ids = set(source_by_id)
    if len(expected_ids) != binding.expected_sample_count:
        _issue(
            issues,
            "source_sample_count",
            f"source manifest contains {len(expected_ids)} unique samples, expected {binding.expected_sample_count}",
        )
    if set(split_by_id) != expected_ids:
        _issue(
            issues,
            "split_sample_set_mismatch",
            "split manifest sample IDs differ from the source manifest",
            missing=sorted(expected_ids - set(split_by_id))[:20],
            unexpected=sorted(set(split_by_id) - expected_ids)[:20],
        )
    split_counts, leakage_groups = _validate_split_rows(
        split_by_id, source_by_id, issues
    )
    if leakage_groups:
        _issue(
            issues,
            "duplicate_group_leakage",
            "one or more duplicate groups cross train/val/test boundaries",
            groups=leakage_groups[:20],
        )
    _check(
        checks,
        "split_zero_leakage",
        not leakage_groups and set(split_by_id) == expected_ids,
        f"{len(split_by_id)} samples; split counts {dict(sorted(split_counts.items()))}",
    )

    source_file_results = _validate_sources(
        paths["source_root"], source_by_id, issues
    )
    _check(
        checks,
        "source_files",
        source_file_results["invalid"] == 0,
        f"{source_file_results['verified']} image/annotation files verified by SHA-256",
    )

    summary_counts = summary.get("counts") if isinstance(summary.get("counts"), dict) else {}
    summary_split_counts = (
        summary.get("split_counts") if isinstance(summary.get("split_counts"), dict) else {}
    )
    summary_task_files = (
        summary.get("task_files") if isinstance(summary.get("task_files"), dict) else {}
    )
    expected_dataset_tasks = set(DATASET_TASK_BY_MODEL_TASK.values())
    for field_name, mapping in (
        ("counts", summary_counts),
        ("split_counts", summary_split_counts),
        ("task_files", summary_task_files),
    ):
        if set(mapping) != expected_dataset_tasks:
            _issue(
                issues,
                "summary_task_matrix_invalid",
                f"summary.{field_name} must contain exactly the five derived tasks",
                field=field_name,
            )
    task_report: dict[str, Any] = {}
    for model_task, dataset_task in DATASET_TASK_BY_MODEL_TASK.items():
        expected_filename = summary_task_files.get(dataset_task, f"{dataset_task}.jsonl")
        if not isinstance(expected_filename, str) or not expected_filename:
            _issue(
                issues,
                "task_filename_invalid",
                f"summary task file for {dataset_task!r} is invalid",
                task=dataset_task,
            )
            continue
        try:
            task_path = _resolve_child(paths["derived_root"], expected_filename)
        except ValueError as exc:
            _issue(issues, "task_path_escape", str(exc), task=dataset_task)
            continue
        expected_hash = binding.task_manifest_sha256[dataset_task]
        actual_hash = _verify_file_hash(
            task_path, expected_hash, f"task:{dataset_task}", issues
        )
        stats = _validate_task_manifest(
            task_path,
            dataset_task=dataset_task,
            config=config,
            source_by_id=source_by_id,
            split_by_id=split_by_id,
            issues=issues,
        )
        expected_count = summary_counts.get(dataset_task)
        if stats["records"] != expected_count:
            _issue(
                issues,
                "task_record_count_mismatch",
                f"{dataset_task} contains {stats['records']} records, summary declares {expected_count!r}",
                task=dataset_task,
            )
        declared_split_counts = summary_split_counts.get(dataset_task)
        if isinstance(declared_split_counts, dict) and stats["splits"] != declared_split_counts:
            _issue(
                issues,
                "task_split_count_mismatch",
                f"{dataset_task} split counts do not match summary",
                task=dataset_task,
                actual=stats["splits"],
                expected=declared_split_counts,
            )
        task_report[model_task] = {
            "dataset_task": dataset_task,
            "path": str(task_path),
            "sha256": actual_hash,
            **stats,
        }
    report["tasks"] = task_report
    actual_total_records = sum(item["records"] for item in task_report.values())
    if summary.get("total_records", actual_total_records) != actual_total_records:
        _issue(
            issues,
            "summary_total_records_mismatch",
            "summary.total_records differs from streamed task record count",
            actual=actual_total_records,
            expected=summary.get("total_records"),
        )
    report["dataset"] = {
        "derivation_version": binding.derivation_version,
        "config_hash": binding.config_hash,
        "sample_count": len(source_by_id),
        "split_counts": dict(sorted(split_counts.items())),
        "source_files": source_file_results,
    }
    task_error_codes = {
        "task_hash_mismatch",
        "task_record_invalid",
        "task_record_count_mismatch",
        "task_split_count_mismatch",
    }
    _check(
        checks,
        "five_task_manifests",
        len(task_report) == 5 and not any(issue["code"] in task_error_codes for issue in issues),
        f"validated {actual_total_records} records",
    )
    return _finish(report)


def _validate_task_manifest(
    path: Path,
    *,
    dataset_task: str,
    config: TrainingConfig,
    source_by_id: dict[str, dict[str, Any]],
    split_by_id: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    records = 0
    invalid = 0
    splits: Counter[str] = Counter()
    record_ids: set[str] = set()
    try:
        iterator = _iter_jsonl(path)
        for line_number, record in iterator:
            records += 1
            if not isinstance(record, dict):
                invalid += 1
                _limited_issue(
                    issues,
                    "task_record_invalid",
                    f"{dataset_task} line {line_number} must be an object",
                    task=dataset_task,
                    line=line_number,
                )
                continue
            sample_id = record.get("sample_id")
            split = record.get("split")
            record_id = record.get("record_id")
            problems: list[str] = []
            if record.get("task") != dataset_task:
                problems.append("task")
            if record.get("derivation_version") != config.dataset.derivation_version:
                problems.append("derivation_version")
            if record.get("config_hash") != config.dataset.config_hash:
                problems.append("config_hash")
            if not isinstance(sample_id, str) or sample_id not in source_by_id:
                problems.append("sample_id")
            elif sample_id not in split_by_id or split != split_by_id[sample_id].get("split"):
                problems.append("split")
            else:
                source = source_by_id[sample_id]
                if record.get("source_image") != source.get("files", {}).get("image", {}).get("path"):
                    problems.append("source_image")
                if record.get("source_json") != source.get("files", {}).get("json", {}).get("path"):
                    problems.append("source_json")
                if record.get("duplicate_group") != source.get("duplicate_group"):
                    problems.append("duplicate_group")
            if split in _ALLOWED_SPLITS:
                splits[str(split)] += 1
            else:
                problems.append("split_value")
            if not isinstance(record_id, str) or not record_id:
                problems.append("record_id")
            elif record_id in record_ids:
                problems.append("duplicate_record_id")
            else:
                record_ids.add(record_id)
            if problems:
                invalid += 1
                _limited_issue(
                    issues,
                    "task_record_invalid",
                    f"{dataset_task} line {line_number} failed fields: {', '.join(sorted(set(problems)))}",
                    task=dataset_task,
                    line=line_number,
                )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        invalid += 1
        _issue(
            issues,
            "task_manifest_unreadable",
            f"cannot stream {dataset_task}: {type(exc).__name__}: {exc}",
            task=dataset_task,
            path=path,
        )
    return {
        "records": records,
        "invalid_records": invalid,
        "splits": dict(sorted(splits.items())),
    }


def _validate_sources(
    source_root: Path,
    source_by_id: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
) -> dict[str, int]:
    verified = 0
    invalid = 0
    for sample_id, record in sorted(source_by_id.items()):
        files = record.get("files")
        if not isinstance(files, dict):
            invalid += 2
            _limited_issue(
                issues,
                "source_record_invalid",
                f"sample {sample_id} has no files object",
                sample_id=sample_id,
            )
            continue
        for kind in ("image", "json"):
            metadata = files.get(kind)
            if not isinstance(metadata, dict):
                invalid += 1
                _limited_issue(
                    issues,
                    "source_record_invalid",
                    f"sample {sample_id} has invalid {kind} metadata",
                    sample_id=sample_id,
                )
                continue
            relative = metadata.get("path")
            expected_hash = metadata.get("sha256")
            try:
                file_path = _resolve_child(source_root, relative)
            except (TypeError, ValueError) as exc:
                invalid += 1
                _limited_issue(
                    issues,
                    "source_path_invalid",
                    f"sample {sample_id} {kind}: {exc}",
                    sample_id=sample_id,
                )
                continue
            if not file_path.is_file():
                invalid += 1
                _limited_issue(
                    issues,
                    "source_missing",
                    f"sample {sample_id} {kind} file is missing: {file_path}",
                    sample_id=sample_id,
                )
                continue
            actual_hash = _sha256_file(file_path)
            if actual_hash != expected_hash:
                invalid += 1
                _limited_issue(
                    issues,
                    "source_hash_mismatch",
                    f"sample {sample_id} {kind} SHA-256 mismatch",
                    sample_id=sample_id,
                    expected=expected_hash,
                    actual=actual_hash,
                )
                continue
            verified += 1
    return {"verified": verified, "invalid": invalid}


def _validate_split_rows(
    split_by_id: dict[str, dict[str, Any]],
    source_by_id: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
) -> tuple[Counter[str], list[str]]:
    counts: Counter[str] = Counter()
    group_splits: dict[str, set[str]] = defaultdict(set)
    for sample_id, record in split_by_id.items():
        split = record.get("split")
        declared_group = record.get("duplicate_group")
        source = source_by_id.get(sample_id)
        group = source.get("duplicate_group") if isinstance(source, dict) else None
        if split not in _ALLOWED_SPLITS:
            _limited_issue(
                issues,
                "split_value_invalid",
                f"sample {sample_id} has invalid split {split!r}",
                sample_id=sample_id,
            )
            continue
        counts[str(split)] += 1
        if not isinstance(group, str) or not group:
            _limited_issue(
                issues,
                "duplicate_group_invalid",
                f"sample {sample_id} has no duplicate_group",
                sample_id=sample_id,
            )
            continue
        if declared_group != group:
            _limited_issue(
                issues,
                "split_duplicate_group_mismatch",
                f"sample {sample_id} split duplicate_group differs from source manifest",
                sample_id=sample_id,
                expected=group,
                actual=declared_group,
            )
        group_splits[group].add(str(split))
    leakage = sorted(group for group, values in group_splits.items() if len(values) > 1)
    return counts, leakage


def _index_samples(
    rows: list[dict[str, Any]], label: str, issues: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(rows, start=1):
        sample_id = record.get("sample_id") if isinstance(record, dict) else None
        if not isinstance(sample_id, str) or not sample_id:
            _limited_issue(
                issues,
                f"{label}_sample_id_invalid",
                f"{label} line {index} has no valid sample_id",
                line=index,
            )
        elif sample_id in result:
            _limited_issue(
                issues,
                f"{label}_duplicate_sample",
                f"{label} repeats sample_id {sample_id!r}",
                sample_id=sample_id,
            )
        else:
            result[sample_id] = record
    return result


def _read_json_object(
    path: Path, label: str, issues: list[dict[str, Any]]
) -> dict[str, Any] | None:
    try:
        value = _load_json(path)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _issue(
            issues,
            f"{label}_unreadable",
            f"cannot read {label}: {type(exc).__name__}: {exc}",
            path=path,
        )
        return None
    if not isinstance(value, dict):
        _issue(issues, f"{label}_invalid", f"{label} root must be an object")
        return None
    return value


def _read_jsonl(
    path: Path, label: str, issues: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, value in _iter_jsonl(path):
            if not isinstance(value, dict):
                _limited_issue(
                    issues,
                    f"{label}_record_invalid",
                    f"{label} line {line_number} must be an object",
                    line=line_number,
                )
            else:
                rows.append(value)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _issue(
            issues,
            f"{label}_unreadable",
            f"cannot read {label}: {type(exc).__name__}: {exc}",
            path=path,
        )
    return rows


def _iter_jsonl(path: Path) -> Iterator[tuple[int, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at line {line_number}")
            yield line_number, json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )


def _load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def _resolve_path(path: Path, root: Path) -> Path:
    return (path if path.is_absolute() else root / path).resolve(strict=False)


def _resolve_within(root: Path, raw_path: str) -> Path:
    path = _resolve_path(Path(raw_path), root)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"configured path escapes workspace: {raw_path!r}") from exc
    return path


def _resolve_child(parent: Path, raw_path: Any) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise TypeError("relative path must be a non-empty string")
    path = (parent / Path(raw_path)).resolve(strict=False)
    try:
        path.relative_to(parent.resolve(strict=False))
    except ValueError as exc:
        raise ValueError(f"relative path escapes root: {raw_path!r}") from exc
    return path


def _verify_file_hash(
    path: Path,
    expected: str,
    label: str,
    issues: list[dict[str, Any]],
) -> str | None:
    if not path.is_file():
        _issue(issues, f"{_code(label)}_missing", f"{label} file is missing: {path}", path=path)
        return None
    try:
        actual = _sha256_file(path)
    except OSError as exc:
        _issue(
            issues,
            f"{_code(label)}_unreadable",
            f"cannot hash {label}: {type(exc).__name__}: {exc}",
            path=path,
        )
        return None
    if actual != expected:
        _issue(
            issues,
            f"{_code(label)}_hash_mismatch",
            f"{label} SHA-256 does not match the training binding",
            path=path,
            expected=expected,
            actual=actual,
        )
    return actual


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _check_equal(actual: Any, expected: Any, field: str, issues: list[dict[str, Any]]) -> None:
    if actual != expected:
        _issue(
            issues,
            "mismatch",
            f"{field} mismatch: expected {expected!r}, got {actual!r}",
            field=field,
            expected=expected,
            actual=actual,
        )


def _check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def _issue(issues: list[dict[str, Any]], code: str, message: str, **context: Any) -> None:
    issue: dict[str, Any] = {"code": code, "message": message}
    for key, value in context.items():
        if isinstance(value, Path):
            issue[key] = str(value)
        else:
            issue[key] = value
    issues.append(issue)


def _limited_issue(
    issues: list[dict[str, Any]], code: str, message: str, **context: Any
) -> None:
    if sum(1 for issue in issues if issue["code"] == code) < 20:
        _issue(issues, code, message, **context)


def _code(label: str) -> str:
    return label.lower().replace(":", "_").replace("-", "_")


def _finish(report: dict[str, Any]) -> dict[str, Any]:
    report["ok"] = not report["issues"] and all(
        check.get("ok") is True for check in report["checks"]
    )
    return report


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON object key {key!r}")
        output[key] = value
    return output


__all__ = ["preflight_training_data"]
