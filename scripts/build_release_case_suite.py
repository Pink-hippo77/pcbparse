#!/usr/bin/env python3
"""Build a hash-bound, image-only production release case suite.

The suite is for deterministic replay and performance validation, not for
claiming held-out accuracy: it contains every case in the primary split and a
deterministic set of complete singleton groups from a supplemental split.
Only verified image bytes are copied to the materialized dataset root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_NATURAL_PART = re.compile(r"(\d+)")


class ReleaseCaseSuiteError(ValueError):
    """Raised when a release suite cannot be built without ambiguity."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic, hash-bound, image-only case suite for "
            "production replay and performance gates."
        )
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument(
        "--source-split-manifest",
        type=Path,
        default=Path("data/splits/dataset300_split.jsonl"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset300/Dataset300"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/splits/dataset300_release50_v1.jsonl"),
    )
    parser.add_argument(
        "--output-images-root",
        type=Path,
        default=Path("work/release50-images-only"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/release/release50_suite_build.json"),
    )
    parser.add_argument("--primary-split", default="test")
    parser.add_argument("--supplemental-split", default="val")
    parser.add_argument("--output-split", default="release50")
    parser.add_argument("--target-count", type=int, default=50)
    return parser


def _natural_key(value: str) -> tuple[tuple[int, Any], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in _NATURAL_PART.split(value)
        if part
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseCaseSuiteError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ReleaseCaseSuiteError(f"non-finite JSON constant is forbidden: {value}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (json.JSONDecodeError, ReleaseCaseSuiteError) as exc:
            raise ReleaseCaseSuiteError(
                f"invalid source split manifest line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ReleaseCaseSuiteError(
                f"source split manifest line {line_number} must be an object"
            )
        rows.append(value)
    if not rows:
        raise ReleaseCaseSuiteError("source split manifest is empty")
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) for row in rows)


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _resolve_input(root: Path, value: str | Path, label: str) -> Path:
    candidate = Path(value).expanduser()
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _resolve_directory(root: Path, value: str | Path, label: str) -> Path:
    candidate = Path(value).expanduser()
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _resolve_output(root: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else root / candidate).resolve(
        strict=False
    )


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReleaseCaseSuiteError(f"{label} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ReleaseCaseSuiteError(f"{label} must be a positive integer")
    return value


def _normalize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(rows, start=1):
        sample_id = _nonempty_text(raw.get("sample_id"), f"row {index} sample_id")
        if not _SAFE_CASE_ID.fullmatch(sample_id) or sample_id in {".", ".."}:
            raise ReleaseCaseSuiteError(f"unsafe sample_id: {sample_id!r}")
        if sample_id in seen_ids:
            raise ReleaseCaseSuiteError(f"duplicate sample_id: {sample_id!r}")
        seen_ids.add(sample_id)
        duplicate_group = _nonempty_text(
            raw.get("duplicate_group"), f"row {index} duplicate_group"
        )
        split = _nonempty_text(raw.get("split"), f"row {index} split")
        declared_group_size = _positive_int(
            raw.get("duplicate_group_size"),
            f"row {index} duplicate_group_size",
        )
        files = raw.get("files")
        image = files.get("image") if isinstance(files, Mapping) else None
        if not isinstance(image, Mapping):
            raise ReleaseCaseSuiteError(f"row {index} files.image must be an object")
        image_path = _nonempty_text(image.get("path"), f"row {index} image path")
        relative_image = Path(image_path)
        if relative_image.is_absolute() or any(part in {"..", ""} for part in relative_image.parts):
            raise ReleaseCaseSuiteError(
                f"row {index} image path must be safe and relative: {image_path!r}"
            )
        image_sha = _nonempty_text(
            image.get("sha256"), f"row {index} image SHA-256"
        ).lower()
        if not _HEX_SHA256.fullmatch(image_sha):
            raise ReleaseCaseSuiteError(
                f"row {index} image SHA-256 is malformed"
            )
        normalized.append(
            {
                "raw": dict(raw),
                "sample_id": sample_id,
                "duplicate_group": duplicate_group,
                "duplicate_group_size": declared_group_size,
                "split": split,
                "image": dict(image),
                "image_path": relative_image,
                "image_sha256": image_sha,
            }
        )

    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    split_by_group: dict[str, set[str]] = defaultdict(set)
    for row in normalized:
        members[row["duplicate_group"]].append(row)
        split_by_group[row["duplicate_group"]].add(row["split"])
    leaking = sorted(group for group, splits in split_by_group.items() if len(splits) != 1)
    if leaking:
        raise ReleaseCaseSuiteError(
            f"duplicate groups leak across source splits: {leaking[:10]}"
        )
    for group, group_rows in members.items():
        declared = {row["duplicate_group_size"] for row in group_rows}
        if declared != {len(group_rows)}:
            raise ReleaseCaseSuiteError(
                f"duplicate group {group!r} declares {sorted(declared)}, "
                f"but contains {len(group_rows)} row(s)"
            )
    return normalized


def _select_rows(
    rows: Sequence[dict[str, Any]],
    *,
    primary_split: str,
    supplemental_split: str,
    target_count: int,
) -> list[dict[str, Any]]:
    if not primary_split or not supplemental_split or primary_split == supplemental_split:
        raise ReleaseCaseSuiteError(
            "primary and supplemental splits must be distinct non-empty strings"
        )
    if target_count <= 0:
        raise ReleaseCaseSuiteError("target_count must be positive")
    primary = [row for row in rows if row["split"] == primary_split]
    if not primary:
        raise ReleaseCaseSuiteError(f"primary split has no rows: {primary_split!r}")
    if len(primary) > target_count:
        raise ReleaseCaseSuiteError(
            f"primary split contains {len(primary)} rows, exceeding target {target_count}"
        )
    needed = target_count - len(primary)
    supplemental = sorted(
        (
            row
            for row in rows
            if row["split"] == supplemental_split
            and row["duplicate_group_size"] == 1
        ),
        key=lambda row: _natural_key(row["sample_id"]),
    )
    if len(supplemental) < needed:
        raise ReleaseCaseSuiteError(
            f"supplemental split has only {len(supplemental)} singleton rows; "
            f"{needed} required"
        )
    selected = [*primary, *supplemental[:needed]]
    selected.sort(key=lambda row: _natural_key(row["sample_id"]))
    if len(selected) != target_count:
        raise AssertionError("internal release-suite selection cardinality mismatch")
    return selected


def _release_row(row: Mapping[str, Any], *, output_split: str) -> dict[str, Any]:
    raw = row["raw"]
    retained_image_fields = {
        key: value
        for key, value in row["image"].items()
        if key
        in {
            "bytes",
            "declared_extension",
            "detected_format",
            "grayscale_stddev",
            "height",
            "path",
            "perceptual_hash",
            "perceptual_hash_algorithm",
            "perceptual_hash_eligible",
            "sha256",
            "width",
        }
    }
    retained_image_fields["path"] = row["image_path"].as_posix()
    retained_image_fields["sha256"] = row["image_sha256"]
    output: dict[str, Any] = {
        "schema_version": 1,
        "sample_id": row["sample_id"],
        "duplicate_group": row["duplicate_group"],
        "duplicate_group_size": row["duplicate_group_size"],
        "split": output_split,
        "release_source_split": row["split"],
        "source_class": raw.get("source_class"),
        "source_family": raw.get("source_family"),
        "base_name": raw.get("base_name"),
        "canvas": raw.get("canvas"),
        "files": {"image": retained_image_fields},
    }
    return output


def build_release_case_suite(
    *,
    workspace_root: str | Path,
    source_split_manifest: str | Path,
    dataset_root: str | Path,
    output_manifest: str | Path,
    output_images_root: str | Path,
    report: str | Path,
    primary_split: str = "test",
    supplemental_split: str = "val",
    output_split: str = "release50",
    target_count: int = 50,
) -> dict[str, Any]:
    workspace = Path(workspace_root).expanduser().resolve()
    if not workspace.is_dir():
        raise FileNotFoundError(f"workspace root does not exist: {workspace}")
    source_manifest_path = _resolve_input(
        workspace, source_split_manifest, "source split manifest"
    )
    source_dataset = _resolve_directory(workspace, dataset_root, "dataset root")
    manifest_path = _resolve_output(workspace, output_manifest)
    images_root = _resolve_output(workspace, output_images_root)
    report_path = _resolve_output(workspace, report)

    if len({manifest_path, images_root, report_path}) != 3:
        raise ReleaseCaseSuiteError("output manifest, image root, and report must differ")
    if _within(images_root, source_dataset) or _within(source_dataset, images_root):
        raise ReleaseCaseSuiteError(
            "output image root must not overlap the source dataset root"
        )
    if _within(manifest_path, images_root) or _within(report_path, images_root):
        raise ReleaseCaseSuiteError(
            "manifest and report must remain outside the image-only dataset root"
        )
    existing = [path for path in (manifest_path, images_root, report_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "release-suite outputs must be new: " + ", ".join(str(path) for path in existing)
        )
    for parent in {manifest_path.parent, images_root.parent, report_path.parent}:
        parent.mkdir(parents=True, exist_ok=True)

    source_rows = _normalize_rows(_read_jsonl(source_manifest_path))
    selected = _select_rows(
        source_rows,
        primary_split=_nonempty_text(primary_split, "primary_split"),
        supplemental_split=_nonempty_text(
            supplemental_split, "supplemental_split"
        ),
        target_count=target_count,
    )
    output_split_value = _nonempty_text(output_split, "output_split")
    release_rows = [
        _release_row(row, output_split=output_split_value) for row in selected
    ]
    manifest_payload = _canonical_jsonl_bytes(release_rows)
    manifest_sha = _sha256_bytes(manifest_payload)

    staging_images = Path(
        tempfile.mkdtemp(prefix=f".{images_root.name}.", dir=images_root.parent)
    ).resolve()
    published_images = False
    temporary_manifest: Path | None = None
    temporary_report: Path | None = None
    try:
        image_bindings: list[dict[str, Any]] = []
        copied_paths: set[Path] = set()
        for row in selected:
            relative = row["image_path"]
            source_image = (source_dataset / relative).resolve()
            if not _within(source_image, source_dataset):
                raise ReleaseCaseSuiteError(
                    f"image path escapes source dataset: {relative.as_posix()}"
                )
            if not source_image.is_file() or source_image.is_symlink():
                raise ReleaseCaseSuiteError(
                    f"source image is missing, non-regular, or a symlink: {source_image}"
                )
            actual_source_sha = _sha256_file(source_image)
            if actual_source_sha != row["image_sha256"]:
                raise ReleaseCaseSuiteError(
                    f"source image SHA-256 mismatch for case {row['sample_id']}"
                )
            destination_image = (staging_images / relative).resolve()
            if not _within(destination_image, staging_images):
                raise ReleaseCaseSuiteError(
                    f"destination image escapes staging root: {relative.as_posix()}"
                )
            if destination_image in copied_paths:
                raise ReleaseCaseSuiteError(
                    f"two cases select the same image path: {relative.as_posix()}"
                )
            copied_paths.add(destination_image)
            destination_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_image, destination_image)
            copied_sha = _sha256_file(destination_image)
            if copied_sha != actual_source_sha:
                raise ReleaseCaseSuiteError(
                    f"copied image SHA-256 mismatch for case {row['sample_id']}"
                )
            image_bindings.append(
                {
                    "sample_id": row["sample_id"],
                    "path": relative.as_posix(),
                    "sha256": copied_sha,
                }
            )

        staged_files = sorted(path for path in staging_images.rglob("*") if path.is_file())
        if len(staged_files) != target_count or len(image_bindings) != target_count:
            raise ReleaseCaseSuiteError("image-only staging population is incomplete")
        unexpected_suffixes = sorted(
            path.relative_to(staging_images).as_posix()
            for path in staged_files
            if path.suffix.casefold() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        )
        if unexpected_suffixes:
            raise ReleaseCaseSuiteError(
                f"image-only staging contains non-image files: {unexpected_suffixes[:5]}"
            )

        membership: dict[str, list[str]] = defaultdict(list)
        for row in selected:
            membership[row["duplicate_group"]].append(row["sample_id"])
        membership = {
            group: sorted(sample_ids, key=_natural_key)
            for group, sample_ids in sorted(membership.items(), key=lambda item: _natural_key(item[0]))
        }
        selected_source_counts = Counter(row["split"] for row in selected)
        report_payload: dict[str, Any] = {
            "schema_version": 1,
            "kind": "pcbparse_image_only_release_case_suite_v1",
            "status": "ready",
            "diagnostic_only": False,
            "accuracy_claim": False,
            "purpose": [
                "production_determinism",
                "production_performance",
                "coordinate_and_schema_replay",
            ],
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "selection": {
                "algorithm": "all_primary_plus_natural_order_supplemental_singletons_v1",
                "primary_split": primary_split,
                "supplemental_split": supplemental_split,
                "output_split": output_split_value,
                "target_count": target_count,
                "selected_count": len(selected),
                "selected_source_split_counts": dict(
                    sorted(selected_source_counts.items())
                ),
                "selected_case_ids": [row["sample_id"] for row in selected],
                "selected_case_ids_sha256": _sha256_bytes(
                    _canonical_json_bytes([row["sample_id"] for row in selected])
                ),
                "duplicate_group_count": len(membership),
                "duplicate_group_membership": membership,
                "duplicate_group_membership_sha256": _sha256_bytes(
                    _canonical_json_bytes(membership)
                ),
                "complete_duplicate_groups": True,
            },
            "sources": {
                "source_split_manifest": {
                    "path": str(source_manifest_path),
                    "sha256": _sha256_file(source_manifest_path),
                    "row_count": len(source_rows),
                },
                "source_dataset_root": str(source_dataset),
            },
            "outputs": {
                "manifest": {
                    "path": str(manifest_path),
                    "sha256": manifest_sha,
                    "row_count": len(release_rows),
                },
                "images_root": str(images_root),
                "image_count": len(image_bindings),
                "image_set_sha256": _sha256_bytes(
                    _canonical_json_bytes(image_bindings)
                ),
                "only_verified_images_materialized": True,
                "report": str(report_path),
            },
            "images": image_bindings,
        }
        report_without_digest = _pretty_json_bytes(report_payload)
        report_payload["report_payload_sha256"] = _sha256_bytes(
            report_without_digest
        )
        report_bytes = _pretty_json_bytes(report_payload)

        manifest_descriptor, temporary_manifest_name = tempfile.mkstemp(
            prefix=f".{manifest_path.name}.", suffix=".tmp", dir=manifest_path.parent
        )
        temporary_manifest = Path(temporary_manifest_name)
        with os.fdopen(manifest_descriptor, "wb") as stream:
            stream.write(manifest_payload)
            stream.flush()
            os.fsync(stream.fileno())
        report_descriptor, temporary_report_name = tempfile.mkstemp(
            prefix=f".{report_path.name}.", suffix=".tmp", dir=report_path.parent
        )
        temporary_report = Path(temporary_report_name)
        with os.fdopen(report_descriptor, "wb") as stream:
            stream.write(report_bytes)
            stream.flush()
            os.fsync(stream.fileno())

        if any(path.exists() for path in (manifest_path, images_root, report_path)):
            raise FileExistsError("release-suite output appeared during staging")
        staging_images.replace(images_root)
        published_images = True
        temporary_manifest.replace(manifest_path)
        temporary_manifest = None
        temporary_report.replace(report_path)
        temporary_report = None
        return report_payload
    except Exception:
        if published_images and images_root.exists():
            shutil.rmtree(images_root)
        raise
    finally:
        if staging_images.exists():
            shutil.rmtree(staging_images)
        if temporary_manifest is not None:
            temporary_manifest.unlink(missing_ok=True)
        if temporary_report is not None:
            temporary_report.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_release_case_suite(
        workspace_root=args.workspace_root,
        source_split_manifest=args.source_split_manifest,
        dataset_root=args.dataset_root,
        output_manifest=args.output_manifest,
        output_images_root=args.output_images_root,
        report=args.report,
        primary_split=args.primary_split,
        supplemental_split=args.supplemental_split,
        output_split=args.output_split,
        target_count=args.target_count,
    )
    print(report["outputs"]["manifest"]["path"])
    print(f"manifest_sha256={report['outputs']['manifest']['sha256']}")
    print(f"image_set_sha256={report['outputs']['image_set_sha256']}")
    print(f"report_payload_sha256={report['report_payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
