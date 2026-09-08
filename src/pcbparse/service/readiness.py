from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, cast

from pcbparse.training.artifacts import (
    CandidateBundleError,
    build_candidate_lock_entry,
    load_candidate_bundle,
    validate_inference_contract,
)
from pcbparse.training.config import ModelTask


LOCK_SCHEMA_VERSION = "1.0"
REQUIRED_MODEL_TASKS = frozenset({"component", "junction", "ocr", "pin", "wire"})
# Compatibility name used by the first readiness implementation.
REQUIRED_LEARNED_TASKS = REQUIRED_MODEL_TASKS
SUPPORTED_PRECISIONS = frozenset(
    {
        "bf16",
        "float16",
        "float32",
        "fp16",
        "fp32",
        "int4",
        "int8",
        "mixed",
        "uint8",
    }
)

_HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+\-]*$")
_ADAPTER = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$"
)
_PLACEHOLDERS = frozenset({"", "dev", "none", "null", "tbd", "unknown", "unlicensed", "unversioned"})
_UNREADY_PROVIDERS = {
    "dataset_debug": "dataset_debug depends on annotation sidecars and emits debug-only surrogate identifiers",
    "dataset_golden": "dataset_golden depends on annotation sidecars and is not a submission provider",
    "datasetgoldenprovider": "DatasetGoldenProvider depends on annotation sidecars and is not a submission provider",
    "empty": "empty disaster-recovery provider has no perception accuracy",
    "emptyimageprovider": "EmptyImageProvider has no perception accuracy",
    "heuristic": "heuristic provider only implements low-confidence wire candidates",
    "heuristicvisionprovider": "HeuristicVisionProvider only implements low-confidence wire candidates",
}


class ReadinessGateError(RuntimeError):
    """Raised when a production CLI action is attempted while unready."""

    def __init__(self, report: dict[str, Any], *, allow_hint: bool = False) -> None:
        problems = report.get("problems") or ["unknown readiness failure"]
        detail = "; ".join(str(item) for item in problems[:3])
        if len(problems) > 3:
            detail += f"; and {len(problems) - 3} more problem(s)"
        hint = (
            " Use --allow-unready-debug only for explicit degraded heuristic debugging."
            if allow_hint
            else ""
        )
        message = f"production readiness gate failed: {detail}"
        if not message.endswith("."):
            message += "."
        super().__init__(message + hint)
        self.report = report


def readiness_report(
    *,
    provider_name: str,
    models_lock: str | Path | None = None,
    required_tasks: Iterable[str] = REQUIRED_MODEL_TASKS,
    require_lineage: bool = False,
) -> dict[str, Any]:
    """Validate the production model lock and provider without loading models.

    Relative model paths are resolved from the directory containing the lock,
    not from the process working directory. A task becomes verified only when
    all required metadata is compatible and the declared SHA-256 equals the
    digest of the actual file bytes.
    """

    required = frozenset(str(item).strip().lower() for item in required_tasks)
    issues: list[dict[str, Any]] = []
    lock_path = _resolve_lock_path(models_lock)
    lock: dict[str, Any] = {}

    if not lock_path.is_file():
        _add_issue(issues, "lock_missing", f"model lock is missing: {lock_path}", path=lock_path)
    else:
        try:
            loaded = json.loads(
                lock_path.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            _add_issue(
                issues,
                "lock_unreadable",
                f"model lock is unreadable: {type(exc).__name__}: {exc}",
                path=lock_path,
            )
        else:
            if isinstance(loaded, dict):
                lock = loaded
            else:
                _add_issue(issues, "lock_invalid_root", "model lock root must be an object", path=lock_path)

    schema_version = lock.get("schema_version")
    lineage_required = bool(
        require_lineage
        and lock
        and (
            lock.get("lineage_contract") is not None
            or "status" in lock
            or "blessed" in lock
        )
    )
    if lineage_required:
        if lock.get("lineage_contract") != "candidate_bundle_v1":
            _add_issue(
                issues,
                "lineage_contract_missing",
                "production model lock must declare lineage_contract='candidate_bundle_v1'",
                path=lock_path,
            )
        if lock.get("status") != "blessed":
            _add_issue(
                issues,
                "lock_status_not_blessed",
                "production model lock status must be 'blessed'",
                path=lock_path,
            )
        if lock.get("blessed") is not True:
            _add_issue(
                issues,
                "lock_blessed_not_true",
                "production model lock blessed must be boolean true",
                path=lock_path,
            )
    if lock and schema_version != LOCK_SCHEMA_VERSION:
        _add_issue(
            issues,
            "schema_version_incompatible",
            f"model lock schema_version must be {LOCK_SCHEMA_VERSION!r}",
            path=lock_path,
        )

    raw_entries = lock.get("models", []) if lock else []
    if not isinstance(raw_entries, list):
        _add_issue(issues, "models_not_list", "model lock 'models' must be a list", path=lock_path)
        raw_entries = []
    if not raw_entries:
        _add_issue(issues, "models_empty", "model lock contains no model entries", path=lock_path)

    by_task: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    model_summaries: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            _add_issue(issues, "entry_not_object", f"model entry {index} must be an object", index=index)
            continue
        raw_task = raw_entry.get("task")
        if not isinstance(raw_task, str) or not raw_task.strip():
            _add_issue(issues, "task_invalid", f"model entry {index} has no canonical task", index=index)
            task = ""
        else:
            task = raw_task.strip().lower()
        if isinstance(raw_task, str) and raw_task != task:
            _add_issue(
                issues,
                "task_not_canonical",
                f"model entry {index} task must be lowercase without surrounding whitespace",
                index=index,
                task=task,
            )
        summary = _validate_entry(
            task,
            index,
            raw_entry,
            lock_path,
            issues,
            require_lineage=lineage_required,
        )
        model_summaries.append(summary)
        if task:
            by_task.setdefault(task, []).append((index, raw_entry))

    for task in sorted(set(by_task) - required):
        for index, _entry in by_task[task]:
            _add_issue(
                issues,
                "unexpected_task",
                f"model lock contains unsupported task {task!r}",
                task=task,
                index=index,
            )

    registered_tasks = sorted(set(by_task) & required)
    verified_tasks: list[str] = []
    for task in sorted(required):
        task_entries = by_task.get(task, [])
        if not task_entries:
            _add_issue(issues, "missing_task", f"blessed model is missing for task {task!r}", task=task)
            continue
        if len(task_entries) != 1:
            _add_issue(
                issues,
                "duplicate_task",
                f"task {task!r} must have exactly one model entry, found {len(task_entries)}",
                task=task,
            )
            continue

        index, _entry = task_entries[0]
        if not any(issue.get("index") == index for issue in issues):
            verified_tasks.append(task)

    verified_indices = {
        by_task[task][0][0]
        for task in verified_tasks
    }
    for summary in model_summaries:
        summary["verified"] = summary["index"] in verified_indices

    provider = str(provider_name).strip()
    if not provider:
        _add_issue(issues, "provider_missing", "provider name is empty")
    else:
        reason = _UNREADY_PROVIDERS.get(provider.lower())
        if reason:
            _add_issue(issues, "provider_not_production", reason, provider=provider)

    ready = not issues and set(verified_tasks) == set(required)
    return {
        "ready": ready,
        "status": "ready" if ready else "unready",
        "provider": provider,
        "models_lock": str(lock_path),
        "schema_version": schema_version,
        "lineage_required": lineage_required,
        "lineage_verified": lineage_required and not any(
            issue["code"].startswith("lineage_") or issue["code"].startswith("lock_")
            for issue in issues
        ),
        "required_learned_tasks": sorted(required),
        "registered_learned_tasks": registered_tasks,
        "verified_learned_tasks": sorted(verified_tasks),
        "models": sorted(model_summaries, key=lambda item: (item["task"], item["index"])),
        "issues": issues,
        "problems": [str(item["message"]) for item in issues],
    }


def _validate_entry(
    task: str,
    index: int,
    entry: dict[str, Any],
    lock_path: Path,
    issues: list[dict[str, Any]],
    *,
    require_lineage: bool = False,
) -> dict[str, Any]:
    status = entry.get("status")
    if status != "blessed":
        _add_issue(
            issues,
            "status_not_blessed",
            f"task {task!r} status must be 'blessed'",
            task=task,
            index=index,
        )
    if entry.get("blessed") is not True:
        _add_issue(
            issues,
            "blessed_not_true",
            f"task {task!r} blessed must be boolean true",
            task=task,
            index=index,
        )

    version = entry.get("version")
    if not _valid_declared_text(version, version=True):
        _add_issue(
            issues,
            "version_invalid",
            f"task {task!r} version must be a non-placeholder compatible identifier",
            task=task,
            index=index,
        )

    license_name = entry.get("license")
    if not _valid_declared_text(license_name):
        _add_issue(
            issues,
            "license_invalid",
            f"task {task!r} license must be explicitly declared",
            task=task,
            index=index,
        )

    precision = entry.get("precision")
    normalized_precision = precision.strip().lower() if isinstance(precision, str) else ""
    if normalized_precision not in SUPPORTED_PRECISIONS:
        _add_issue(
            issues,
            "precision_incompatible",
            f"task {task!r} precision {precision!r} is not supported",
            task=task,
            index=index,
        )

    adapter = entry.get("adapter")
    if not isinstance(adapter, str) or not _ADAPTER.fullmatch(adapter):
        _add_issue(
            issues,
            "adapter_invalid",
            f"task {task!r} adapter must be declared as 'package.module:factory'",
            task=task,
            index=index,
        )

    inference = entry.get("inference")
    if inference is None:
        if task in REQUIRED_MODEL_TASKS:
            _add_issue(
                issues,
                "inference_missing",
                f"task {task!r} must declare a complete hash-bound inference contract",
                task=task,
                index=index,
            )
    elif not isinstance(inference, dict):
        _add_issue(
            issues,
            "inference_invalid",
            f"task {task!r} inference must be an object",
            task=task,
            index=index,
        )
    else:
        for name, nested_value in sorted(inference.items()):
            if name in entry and entry[name] != nested_value:
                _add_issue(
                    issues,
                    "inference_conflict",
                    f"task {task!r} inference.{name} conflicts with top-level {name}",
                    task=task,
                    index=index,
                    field=name,
                )
        adapter_config = entry.get("adapter_config")
        if adapter_config is not None and not isinstance(adapter_config, dict):
            _add_issue(
                issues,
                "inference_conflict",
                f"task {task!r} adapter_config must be an object and cannot override inference",
                task=task,
                index=index,
                field="adapter_config",
            )
        elif isinstance(adapter_config, dict):
            for name, compatibility_value in sorted(adapter_config.items()):
                if name not in inference or inference[name] != compatibility_value:
                    _add_issue(
                        issues,
                        "inference_conflict",
                        f"task {task!r} adapter_config.{name} conflicts with hash-bound inference",
                        task=task,
                        index=index,
                        field=f"adapter_config.{name}",
                    )
        if task in REQUIRED_MODEL_TASKS:
            try:
                # Validate the nested object itself.  Top-level compatibility
                # fields must never fill a missing nested key, otherwise an old
                # lock could bypass the same exact contract enforced for a
                # candidate bundle and silently select adapter defaults.
                validate_inference_contract(
                    cast(ModelTask, task),
                    inference,
                )
            except (CandidateBundleError, KeyError, TypeError, ValueError) as exc:
                _add_issue(
                    issues,
                    "inference_contract_invalid",
                    f"task {task!r} inference contract is invalid: {exc}",
                    task=task,
                    index=index,
                    error_type=type(exc).__name__,
                )

    if require_lineage:
        _validate_bound_bundle(task, index, entry, lock_path, issues)

    model_path = _entry_path(entry.get("path"), lock_path)
    if model_path is None:
        _add_issue(
            issues,
            "path_invalid",
            f"task {task!r} path must be a non-empty file path",
            task=task,
            index=index,
        )
    elif not model_path.is_file():
        _add_issue(
            issues,
            "path_missing",
            f"task {task!r} model file is missing: {model_path}",
            task=task,
            index=index,
            path=model_path,
        )

    expected_digest = entry.get("sha256")
    digest_is_valid = isinstance(expected_digest, str) and bool(_HEX_SHA256.fullmatch(expected_digest))
    if not digest_is_valid:
        _add_issue(
            issues,
            "sha256_invalid",
            f"task {task!r} sha256 must contain exactly 64 hexadecimal characters",
            task=task,
            index=index,
        )

    actual_digest: str | None = None
    size_bytes: int | None = None
    if model_path is not None and model_path.is_file():
        try:
            size_bytes = model_path.stat().st_size
            if size_bytes <= 0:
                _add_issue(
                    issues,
                    "file_empty",
                    f"task {task!r} model file is empty",
                    task=task,
                    index=index,
                    path=model_path,
                )
            actual_digest = _sha256_file(model_path)
        except OSError as exc:
            _add_issue(
                issues,
                "sha256_unreadable",
                f"task {task!r} model file cannot be hashed: {type(exc).__name__}: {exc}",
                task=task,
                index=index,
                path=model_path,
            )
        else:
            if digest_is_valid and actual_digest != expected_digest.lower():
                _add_issue(
                    issues,
                    "sha256_mismatch",
                    f"task {task!r} model SHA-256 does not match lock metadata",
                    task=task,
                    index=index,
                    path=model_path,
                    expected_sha256=expected_digest.lower(),
                    actual_sha256=actual_digest,
                )

    return {
        **_entry_summary(task, index, entry, lock_path, verified=False),
        "size_bytes": size_bytes,
        "actual_sha256": actual_digest,
    }


def _validate_bound_bundle(
    task: str,
    index: int,
    entry: dict[str, Any],
    lock_path: Path,
    issues: list[dict[str, Any]],
) -> None:
    """Verify the blessed lock entry still matches its candidate bundle."""

    manifest_value = entry.get("candidate_manifest_path")
    manifest_path = _entry_path(manifest_value, lock_path)
    if not _portable_relative_path(manifest_value):
        _add_issue(
            issues,
            "lineage_manifest_path_not_portable",
            f"task {task!r} candidate manifest path must be bundle-relative to the lock",
            task=task,
            index=index,
        )
        return
    if manifest_path is None or not manifest_path.is_file():
        _add_issue(
            issues,
            "lineage_manifest_missing",
            f"task {task!r} candidate manifest is missing",
            task=task,
            index=index,
            path=manifest_path,
        )
        return

    expected_manifest_sha = entry.get("bundle_manifest_sha256")
    actual_manifest_sha = _sha256_file(manifest_path)
    if not isinstance(expected_manifest_sha, str) or actual_manifest_sha != expected_manifest_sha.lower():
        _add_issue(
            issues,
            "lineage_manifest_sha256_mismatch",
            f"task {task!r} candidate manifest SHA-256 does not match lock metadata",
            task=task,
            index=index,
            expected_sha256=expected_manifest_sha,
            actual_sha256=actual_manifest_sha,
        )
        return

    try:
        candidate = build_candidate_lock_entry(
            manifest_path,
            lock_directory=lock_path.parent,
        )
        bundle = load_candidate_bundle(manifest_path)
    except CandidateBundleError as exc:
        _add_issue(
            issues,
            "lineage_bundle_invalid",
            f"task {task!r} candidate bundle is invalid: {exc}",
            task=task,
            index=index,
        )
        return

    for field in (
        "task",
        "path",
        "sha256",
        "version",
        "license",
        "precision",
        "adapter",
        "inference",
        "training_config_sha256",
        "dataset_config_hash",
        "gate_policy",
        "evaluation",
    ):
        if candidate.get(field) != entry.get(field):
            _add_issue(
                issues,
                "lineage_entry_mismatch",
                f"task {task!r} lock field {field!r} differs from candidate bundle",
                task=task,
                index=index,
                field=field,
            )

    if entry.get("bundle_manifest_sha256") != candidate.get("bundle_manifest_sha256"):
        _add_issue(
            issues,
            "lineage_manifest_binding_invalid",
            f"task {task!r} bundle manifest digest is not bound to the lock entry",
            task=task,
            index=index,
        )

    card_path = _entry_path(entry.get("model_card_path"), lock_path)
    expected_card_path = (manifest_path.parent / bundle.model_card_path).resolve(strict=False)
    if not _portable_relative_path(entry.get("model_card_path")) or card_path != expected_card_path:
        _add_issue(
            issues,
            "lineage_model_card_path_mismatch",
            f"task {task!r} model card path differs from the candidate bundle",
            task=task,
            index=index,
        )
    card_sha = entry.get("model_card_sha256")
    if not isinstance(card_sha, str) or card_path is None or not card_path.is_file():
        _add_issue(
            issues,
            "lineage_model_card_missing",
            f"task {task!r} model card is missing",
            task=task,
            index=index,
            path=card_path,
        )
    elif _sha256_file(card_path) != card_sha.lower():
        _add_issue(
            issues,
            "lineage_model_card_sha256_mismatch",
            f"task {task!r} model card SHA-256 does not match lock metadata",
            task=task,
            index=index,
        )


def _portable_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    raw = value.strip().replace("\\", "/")
    path = Path(raw)
    return (
        not path.is_absolute()
        and not re.match(r"^[A-Za-z]:/", raw)
        and ".." not in path.parts
    )


def _entry_summary(
    task: str,
    index: int,
    entry: dict[str, Any],
    lock_path: Path,
    *,
    verified: bool,
) -> dict[str, Any]:
    model_path = _entry_path(entry.get("path"), lock_path)
    return {
        "task": task,
        "index": index,
        "path": str(model_path) if model_path is not None else None,
        "sha256": entry.get("sha256"),
        "status": entry.get("status"),
        "blessed": entry.get("blessed"),
        "version": entry.get("version"),
        "license": entry.get("license"),
        "precision": entry.get("precision"),
        "adapter": entry.get("adapter"),
        "verified": verified,
    }


def _resolve_lock_path(models_lock: str | Path | None) -> Path:
    if models_lock is not None:
        return Path(models_lock).expanduser().resolve(strict=False)
    environment_path = os.environ.get("PCBPARSE_MODELS_LOCK")
    if environment_path:
        return Path(environment_path).expanduser().resolve(strict=False)
    working_tree = Path("models/models.lock.json")
    if working_tree.is_file():
        return working_tree.resolve()
    source_tree = Path(__file__).resolve().parents[3] / "models" / "models.lock.json"
    if source_tree.is_file():
        return source_tree.resolve()
    return working_tree.resolve(strict=False)


def _entry_path(raw_path: Any, lock_path: Path) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    try:
        path = Path(raw_path.strip()).expanduser()
        if not path.is_absolute():
            path = lock_path.parent / path
        return path.resolve(strict=False)
    except (OSError, ValueError):
        return None


def _valid_declared_text(value: Any, *, version: bool = False) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip()
    if normalized.lower() in _PLACEHOLDERS:
        return False
    return bool(_VERSION.fullmatch(normalized)) if version else bool(normalized)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _add_issue(issues: list[dict[str, Any]], code: str, message: str, **context: Any) -> None:
    issue: dict[str, Any] = {"code": code, "message": message}
    for key, value in context.items():
        if value is not None:
            issue[key] = str(value) if isinstance(value, Path) else value
    issues.append(issue)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON object key {key!r}")
        output[key] = value
    return output


__all__ = [
    "LOCK_SCHEMA_VERSION",
    "REQUIRED_LEARNED_TASKS",
    "REQUIRED_MODEL_TASKS",
    "SUPPORTED_PRECISIONS",
    "ReadinessGateError",
    "readiness_report",
]
