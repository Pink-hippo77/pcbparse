from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pcbparse.output.official_schema import validate_official_output
from pcbparse.output.official_serializer import natural_sort_key


_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_WINDOWS_DEVICE_NAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}


class SubmissionPackageError(ValueError):
    """Raised when an evaluation run cannot form a safe submission package."""


@dataclass(frozen=True, slots=True)
class SubmissionPackageResult:
    output_dir: Path
    archive_path: Path | None
    report: dict[str, Any]


def package_submission(
    evaluation_dir: str | Path,
    output_dir: str | Path,
    *,
    archive_path: str | Path | None = None,
    audit_path: str | Path | None = None,
) -> SubmissionPackageResult:
    """Package a successful production evaluation in the guide directory layout.

    The source must be a complete image-only production run created by the
    evaluation harness.  Existing destinations are never overwritten.  The
    official scorer remains authoritative; this function only enforces the
    public guide's ``submission/case_x/case_x.json`` layout and local schema.
    """

    source = Path(evaluation_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    archive = (
        Path(archive_path).expanduser().resolve() if archive_path is not None else None
    )
    audit = Path(audit_path).expanduser().resolve() if audit_path is not None else None
    if not source.is_dir():
        raise FileNotFoundError(f"evaluation directory does not exist: {source}")
    if destination.name != "submission":
        raise SubmissionPackageError(
            "output directory must be named 'submission' to match the guide"
        )
    if archive is not None and audit is not None and archive == audit:
        raise SubmissionPackageError("archive and audit destinations must be different")
    for target, label in ((destination, "output"), (archive, "archive"), (audit, "audit")):
        if target is not None:
            _assert_windows_portable_target(target, label=label)
        if target is not None and target.exists():
            raise SubmissionPackageError(f"{label} destination already exists: {target}")
        if target is not None and _is_within(target, source):
            raise SubmissionPackageError(
                f"{label} destination must be outside the evaluation directory"
            )
    if archive is not None and _is_within(archive, destination):
        raise SubmissionPackageError("archive path must be outside the submission directory")
    if audit is not None and _is_within(audit, destination):
        raise SubmissionPackageError("audit path must be outside the submission directory")

    manifest_path = source / "predictions.jsonl"
    summary_path = source / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise SubmissionPackageError(
            "evaluation directory must contain predictions.jsonl and summary.json"
        )
    manifest_bytes = manifest_path.read_bytes()
    summary_bytes = summary_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    summary_sha256 = hashlib.sha256(summary_bytes).hexdigest()
    summary = _read_object(summary_bytes, "evaluation summary")
    rows = _read_manifest(manifest_bytes)
    _validate_summary(summary, manifest_sha256=manifest_sha256, row_count=len(rows))
    prepared = _prepare_rows(
        source,
        rows,
        expected_model_lock_sha256=str(summary["model_lock_sha256"]),
        expected_config_hash=str(summary["config_hash"]),
    )
    _assert_exact_official_coverage(source, prepared)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=".submission-stage-", dir=str(destination.parent))
    )
    archive_stage: Path | None = None
    archive_stage_dir: Path | None = None
    audit_stage: Path | None = None
    audit_stage_dir: Path | None = None
    published_sidecars: list[Path] = []
    try:
        packaged_rows: list[dict[str, Any]] = []
        for row in prepared:
            assert temporary is not None
            case_directory = temporary / row["package_case_name"]
            case_directory.mkdir()
            target = case_directory / f"{row['package_case_name']}.json"
            _write_new_file(target, row["payload"])
            target_sha256 = _sha256_file(target)
            if target_sha256 != row["official_sha256"]:
                raise SubmissionPackageError(
                    f"case {row['case_id']}: packaged bytes changed unexpectedly"
                )
            packaged_rows.append(
                {
                    "case_id": row["case_id"],
                    "path": target.relative_to(temporary).as_posix(),
                    "sha256": target_sha256,
                    "bytes": len(row["payload"]),
                }
            )
        archive_sha256 = None
        archive_bytes = None
        if archive is not None:
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive_stage_dir = Path(
                tempfile.mkdtemp(
                    prefix=".pcbparse-archive-stage-", dir=str(archive.parent)
                )
            )
            archive_stage = archive_stage_dir / "submission.zip"
            _write_deterministic_archive(
                temporary, archive_stage, root_name=destination.name
            )
            archive_sha256 = _sha256_file(archive_stage)
            archive_bytes = archive_stage.stat().st_size

        tree_sha256 = _json_sha256(
            [{"path": row["path"], "sha256": row["sha256"]} for row in packaged_rows]
        )
        report = {
            "schema_version": 1,
            "kind": "official-guide-submission-layout-package",
            "official_evaluator_available": False,
            "layout": "submission/case_<case_id>/case_<case_id>.json",
            "case_count": len(packaged_rows),
            "source": {
                "evaluation_summary_sha256": summary_sha256,
                "prediction_manifest_sha256": manifest_sha256,
                "model_lock_sha256": summary["model_lock_sha256"],
                "config_hash": summary["config_hash"],
            },
            "files": packaged_rows,
            "submission_tree_sha256": tree_sha256,
            "archive": (
                {
                    "path": str(archive),
                    "sha256": archive_sha256,
                    "bytes": archive_bytes,
                }
                if archive is not None
                else None
            ),
            "warning": (
                "Layout and JSON shape follow the public guide; the unreleased official "
                "sample submission and evaluator remain authoritative."
            ),
        }
        if audit is not None:
            audit.parent.mkdir(parents=True, exist_ok=True)
            audit_stage_dir = Path(
                tempfile.mkdtemp(
                    prefix=".pcbparse-audit-stage-", dir=str(audit.parent)
                )
            )
            audit_stage = audit_stage_dir / "package.audit.json"
            _write_new_file(audit_stage, _pretty_json_bytes(report))

        # Publish optional files first and the complete directory last.  A failed
        # sidecar publication therefore never leaves a partial submission tree.
        if archive is not None and archive_stage is not None:
            _publish_new_file(archive_stage, archive, label="archive")
            published_sidecars.append(archive)
        if audit is not None and audit_stage is not None:
            _publish_new_file(audit_stage, audit, label="audit")
            published_sidecars.append(audit)
        _publish_new_directory(temporary, destination)
        temporary = None
    except Exception:
        for published in reversed(published_sidecars):
            try:
                published.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for stage_dir in (audit_stage_dir, archive_stage_dir, temporary):
            if stage_dir is not None and stage_dir.exists():
                shutil.rmtree(stage_dir)
    return SubmissionPackageResult(destination, archive, report)


def _validate_summary(
    summary: Mapping[str, Any], *, manifest_sha256: str, row_count: int
) -> None:
    expected = {
        "schema_version": 1,
        "kind": "production-image-only-inference",
        "accuracy_mode": "image-only",
        "execution_mode": "production",
        "provider_kind": "production",
        "sidecar_access": False,
        "status": "ok",
        "all_successful": True,
    }
    for field, value in expected.items():
        actual = summary.get(field)
        if type(actual) is not type(value) or actual != value:
            raise SubmissionPackageError(
                f"evaluation summary {field} must be {value!r}"
            )
    if (
        type(summary.get("selected_cases")) is not int
        or type(summary.get("succeeded")) is not int
        or summary.get("selected_cases") != row_count
        or summary.get("succeeded") != row_count
    ):
        raise SubmissionPackageError(
            "evaluation summary case counts do not match the prediction manifest"
        )
    if (
        type(summary.get("failed")) is not int
        or type(summary.get("not_run")) is not int
        or summary.get("failed") != 0
        or summary.get("not_run") != 0
    ):
        raise SubmissionPackageError("evaluation summary contains failed or unrun cases")
    if summary.get("prediction_manifest") != "predictions.jsonl":
        raise SubmissionPackageError("evaluation summary points at a different manifest")
    if summary.get("prediction_manifest_sha256") != manifest_sha256:
        raise SubmissionPackageError("prediction manifest SHA-256 differs from summary")
    readiness = summary.get("readiness")
    if not isinstance(readiness, Mapping) or readiness.get("ready") is not True or readiness.get("status") != "ready":
        raise SubmissionPackageError("evaluation summary does not prove production readiness")
    for field in ("model_lock_sha256", "config_hash"):
        value = summary.get(field)
        if not isinstance(value, str) or not _HEX_SHA256.fullmatch(value):
            raise SubmissionPackageError(f"evaluation summary {field} is not SHA-256")


def _read_manifest(payload: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SubmissionPackageError("prediction manifest is not UTF-8") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = _strict_json_loads(line)
        except (json.JSONDecodeError, ValueError) as exc:
            raise SubmissionPackageError(
                f"invalid prediction manifest JSON on line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise SubmissionPackageError(
                f"prediction manifest line {line_number} must be an object"
            )
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not _SAFE_CASE_ID.fullmatch(case_id):
            raise SubmissionPackageError(
                f"prediction manifest line {line_number} has unsafe case_id"
            )
        if case_id in seen:
            raise SubmissionPackageError(f"duplicate prediction case_id: {case_id}")
        seen.add(case_id)
        rows.append(row)
    if not rows:
        raise SubmissionPackageError("prediction manifest is empty")
    return sorted(rows, key=lambda row: natural_sort_key(str(row["case_id"])))


def _prepare_rows(
    source: Path,
    rows: list[dict[str, Any]],
    *,
    expected_model_lock_sha256: str,
    expected_config_hash: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    package_names: set[str] = set()
    for row in rows:
        case_id = str(row["case_id"])
        if row.get("status") != "ok" or row.get("error") is not None:
            raise SubmissionPackageError(f"case {case_id}: inference was not successful")
        if row.get("provider_kind") != "production" or row.get("sidecar_access") is not False:
            raise SubmissionPackageError(
                f"case {case_id}: manifest is not image-only production inference"
            )
        if row.get("model_lock_sha256") != expected_model_lock_sha256:
            raise SubmissionPackageError(
                f"case {case_id}: model-lock lineage differs from evaluation summary"
            )
        if row.get("config_hash") != expected_config_hash:
            raise SubmissionPackageError(
                f"case {case_id}: configuration lineage differs from evaluation summary"
            )
        relative = row.get("official_path")
        digest = row.get("official_sha256")
        output_digest = row.get("official_output_sha256")
        if not isinstance(relative, str) or not relative:
            raise SubmissionPackageError(f"case {case_id}: official_path is missing")
        expected_relative = f"official/{case_id}.json"
        if relative != expected_relative:
            raise SubmissionPackageError(
                f"case {case_id}: official_path must be {expected_relative!r}"
            )
        if not isinstance(digest, str) or not _HEX_SHA256.fullmatch(digest):
            raise SubmissionPackageError(f"case {case_id}: official_sha256 is invalid")
        if output_digest != digest:
            raise SubmissionPackageError(
                f"case {case_id}: serializer and file SHA-256 evidence differ"
            )
        official_path = (source / relative).resolve()
        if not _is_within(official_path, source) or not official_path.is_file():
            raise SubmissionPackageError(
                f"case {case_id}: official_path escapes or is missing"
            )
        payload = official_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise SubmissionPackageError(
                f"case {case_id}: official output SHA-256 mismatch"
            )
        try:
            document = _strict_json_loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise SubmissionPackageError(
                f"case {case_id}: official output is not UTF-8 JSON"
            ) from exc
        validate_official_output(document)
        package_name = case_id if case_id.startswith("case_") else f"case_{case_id}"
        if package_name.endswith((".", " ")) or len(f"{package_name}.json") > 255:
            raise SubmissionPackageError(
                f"case {case_id}: normalized submission name is not Windows-portable"
            )
        portable_name = package_name.casefold()
        if portable_name in package_names:
            raise SubmissionPackageError(
                f"case {case_id}: normalized submission name collides"
            )
        package_names.add(portable_name)
        result.append(
            {
                "case_id": case_id,
                "package_case_name": package_name,
                "official_path": official_path,
                "official_sha256": digest,
                "payload": payload,
            }
        )
    return result


def _assert_exact_official_coverage(source: Path, rows: list[dict[str, Any]]) -> None:
    official_root = (source / "official").resolve()
    if not official_root.is_dir():
        raise SubmissionPackageError("evaluation official directory is missing")
    expected = {row["official_path"] for row in rows}
    actual = {path.resolve() for path in official_root.rglob("*.json") if path.is_file()}
    if actual != expected:
        missing = sorted(str(path) for path in expected - actual)
        extra = sorted(str(path) for path in actual - expected)
        raise SubmissionPackageError(
            f"official output coverage differs; missing={missing}, extra={extra}"
        )


def _write_deterministic_archive(
    source: Path, target: Path, *, root_name: str
) -> None:
    if target.exists():
        raise SubmissionPackageError(f"archive staging path already exists: {target}")
    with zipfile.ZipFile(
        target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        archive.comment = b""
        for path in sorted(
            (item for item in source.rglob("*") if item.is_file()),
            key=lambda item: natural_sort_key(item.relative_to(source).as_posix()),
        ):
            relative = Path(root_name) / path.relative_to(source)
            info = zipfile.ZipInfo(
                relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.flag_bits |= 0x800
            archive.writestr(
                info,
                path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def _read_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = _strict_json_loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SubmissionPackageError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SubmissionPackageError(f"{label} must be an object")
    return value


def _write_new_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _publish_new_file(source: Path, target: Path, *, label: str) -> None:
    """Atomically expose a staged file without replacing an existing path."""

    try:
        os.link(source, target)
    except FileExistsError as exc:
        raise SubmissionPackageError(
            f"{label} destination already exists: {target}"
        ) from exc
    except OSError as exc:
        if target.exists():
            raise SubmissionPackageError(
                f"{label} destination already exists: {target}"
            ) from exc
        raise SubmissionPackageError(
            f"could not atomically publish {label} destination: {target}"
        ) from exc


def _publish_new_directory(source: Path, target: Path) -> None:
    """Atomically publish a same-parent directory, fail-closed on Windows races."""

    if target.exists():
        raise SubmissionPackageError(f"output destination already exists: {target}")
    try:
        # Unlike os.replace(), os.rename() does not replace an existing directory
        # on Windows.  Source and target share a parent, so the move is atomic.
        os.rename(source, target)
    except OSError as exc:
        if target.exists():
            raise SubmissionPackageError(
                f"output destination already exists: {target}"
            ) from exc
        raise SubmissionPackageError(
            f"could not atomically publish output destination: {target}"
        ) from exc


def _strict_json_loads(payload: str) -> Any:
    return json.loads(
        payload,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _assert_windows_portable_target(path: Path, *, label: str) -> None:
    name = path.name
    device_name = name.split(".", 1)[0].casefold()
    if (
        name.endswith((".", " "))
        or device_name in _WINDOWS_DEVICE_NAMES
        or len(name) > 255
    ):
        raise SubmissionPackageError(
            f"{label} destination is not Windows-portable: {path}"
        )


__all__ = [
    "SubmissionPackageError",
    "SubmissionPackageResult",
    "package_submission",
]
