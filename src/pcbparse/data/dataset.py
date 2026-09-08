from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .hashing import PERCEPTUAL_HASH_ALGORITHM, apply_duplicate_groups, perceptual_dhash, sha256_file
from .models import AuditIssue, DatasetAuditResult


EXPECTED_ROOT_KEYS = frozenset(
    {"canvas", "nowCls", "components", "pins", "texts", "edges"}
)
COLLECTION_KEYS = ("components", "pins", "texts", "edges")
SAMPLE_ID_PATTERN = re.compile(r"^[0-9]{4}$")
IGNORED_NAMES = frozenset({".DS_Store", "__MACOSX"})
IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
)
FORMAT_EXTENSIONS = {
    "PNG": frozenset({".png"}),
    "JPEG": frozenset({".jpg", ".jpeg"}),
    "WEBP": frozenset({".webp"}),
    "BMP": frozenset({".bmp"}),
    "TIFF": frozenset({".tif", ".tiff"}),
}


@dataclass(frozen=True, slots=True)
class _SampleFiles:
    sample_id: str
    directory: Path
    base_name: str
    image: Path
    annotation: Path
    eds: Path
    net: Path


def audit_dataset(
    dataset_root: str | Path = Path("Dataset300") / "Dataset300",
    expected_count: int = 300,
) -> DatasetAuditResult:
    """Discover, validate and fingerprint a PCB schematic dataset.

    Errors exclude a sample from the returned records. Warnings (for example a
    misleading image filename extension) remain visible in the report but do
    not make an otherwise usable sample disappear.
    """

    if expected_count <= 0:
        raise ValueError("expected_count must be positive")

    root = Path(dataset_root).expanduser().resolve(strict=False)
    issues: list[AuditIssue] = []
    records: list[dict[str, Any]] = []
    discovered_count = 0

    if not root.exists():
        issues.append(
            AuditIssue(
                code="dataset_root_missing",
                severity="error",
                message=f"Dataset root does not exist: {root}",
                action="Point --dataset-root at the inner Dataset300/Dataset300 directory.",
                path=str(root),
            )
        )
        report = _build_report(root, expected_count, 0, records, [], issues)
        return DatasetAuditResult(root, records, report)
    if not root.is_dir():
        issues.append(
            AuditIssue(
                code="dataset_root_not_directory",
                severity="error",
                message=f"Dataset root is not a directory: {root}",
                action="Provide a directory containing numbered sample folders.",
                path=str(root),
            )
        )
        report = _build_report(root, expected_count, 0, records, [], issues)
        return DatasetAuditResult(root, records, report)

    numbered: dict[str, Path] = {}
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if _is_ignored(child.name):
            continue
        if child.is_dir() and SAMPLE_ID_PATTERN.fullmatch(child.name):
            numbered[child.name] = child
        elif child.is_dir():
            issues.append(
                AuditIssue(
                    code="unexpected_dataset_directory",
                    severity="error",
                    message=f"Unexpected directory at dataset root: {child.name}",
                    action="Remove it from the dataset view or point at the inner numbered directory.",
                    path=child.name,
                )
            )
        else:
            issues.append(
                AuditIssue(
                    code="unexpected_dataset_file",
                    severity="error",
                    message=f"Unexpected file at dataset root: {child.name}",
                    action="Move non-sample files outside the dataset root.",
                    path=child.name,
                )
            )

    discovered_count = len(numbered)
    expected_ids = {f"{index:04d}" for index in range(1, expected_count + 1)}
    actual_ids = set(numbered)
    if discovered_count != expected_count:
        issues.append(
            AuditIssue(
                code="sample_count_mismatch",
                severity="error",
                message=(
                    f"Expected {expected_count} numbered samples but discovered "
                    f"{discovered_count}."
                ),
                action="Restore missing sample folders and remove unexpected numbered folders.",
                path=".",
            )
        )
    for missing_id in sorted(expected_ids - actual_ids):
        issues.append(
            AuditIssue(
                code="missing_sample_directory",
                severity="error",
                message=f"Expected sample directory {missing_id} is missing.",
                action=f"Restore sample {missing_id} from the official dataset archive.",
                sample_id=missing_id,
                path=missing_id,
            )
        )
    for extra_id in sorted(actual_ids - expected_ids):
        issues.append(
            AuditIssue(
                code="unexpected_sample_directory",
                severity="error",
                message=f"Numbered sample {extra_id} is outside the expected range.",
                action="Move this folder out of the audited dataset or adjust --expected-count.",
                sample_id=extra_id,
                path=extra_id,
            )
        )

    for sample_id, sample_directory in sorted(numbered.items()):
        issue_start = len(issues)
        sample_files = _discover_sample_files(root, sample_id, sample_directory, issues)
        if sample_files is None:
            continue

        annotation_metadata = _inspect_annotation(root, sample_files, issues)
        image_metadata = _inspect_image(root, sample_files, issues)
        _inspect_eds(root, sample_files, issues)
        _inspect_net(root, sample_files, issues)

        if annotation_metadata is not None and image_metadata is not None:
            canvas = annotation_metadata["canvas"]
            actual_size = (image_metadata["width"], image_metadata["height"])
            expected_size = (canvas["width"], canvas["height"])
            if actual_size != expected_size:
                issues.append(
                    AuditIssue(
                        code="canvas_image_dimension_mismatch",
                        severity="error",
                        message=(
                            f"Image dimensions {actual_size[0]}x{actual_size[1]} do not "
                            f"match JSON canvas {expected_size[0]}x{expected_size[1]}."
                        ),
                        action="Use the matching image/annotation pair from the official dataset.",
                        sample_id=sample_id,
                        path=_relative(sample_files.image, root),
                    )
                )

        sample_has_error = any(
            issue.severity == "error" for issue in issues[issue_start:]
        )
        if sample_has_error or annotation_metadata is None or image_metadata is None:
            continue

        try:
            file_metadata = {
                "image": _fingerprint_file(sample_files.image, root),
                "json": _fingerprint_file(sample_files.annotation, root),
                "eds": _fingerprint_file(sample_files.eds, root),
                "net": _fingerprint_file(sample_files.net, root),
            }
        except OSError as exc:
            issues.append(
                AuditIssue(
                    code="file_hash_failed",
                    severity="error",
                    message=f"Could not hash every file for sample {sample_id}: {exc}",
                    action="Check file permissions and restore unreadable files.",
                    sample_id=sample_id,
                    path=sample_id,
                )
            )
            continue

        file_metadata["image"].update(image_metadata)
        records.append(
            {
                "schema_version": 1,
                "sample_id": sample_id,
                "base_name": sample_files.base_name,
                "source_family": _source_family(sample_files.base_name),
                "source_class": annotation_metadata["source_class"],
                "canvas": annotation_metadata["canvas"],
                "counts": annotation_metadata["counts"],
                "files": file_metadata,
            }
        )

    duplicate_groups = apply_duplicate_groups(records)
    for group in duplicate_groups:
        evidence_kinds = sorted({item["kind"] for item in group["evidence"]})
        members = ", ".join(group["sample_ids"])
        issues.append(
            AuditIssue(
                code="duplicate_samples",
                severity="warning",
                message=(
                    f"Samples {members} share identical "
                    f"{' and '.join(evidence_kinds)} content."
                ),
                action=(
                    "Keep every member in one train/validation/test split; "
                    "exclude duplicates from independent evaluation if appropriate."
                ),
            )
        )

    report = _build_report(
        root,
        expected_count,
        discovered_count,
        records,
        duplicate_groups,
        issues,
    )
    return DatasetAuditResult(root, records, report)


def _source_family(base_name: str) -> str | None:
    """Extract only a high-confidence multi-sheet project identifier."""

    match = re.fullmatch(r"(?P<family>\d{3}_\d{2})_\d{2}", base_name)
    return match.group("family") if match is not None else None


def _is_ignored(name: str) -> bool:
    return name in IGNORED_NAMES or name.startswith("._")


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _discover_sample_files(
    root: Path,
    sample_id: str,
    directory: Path,
    issues: list[AuditIssue],
) -> _SampleFiles | None:
    files: list[Path] = []
    for child in sorted(directory.iterdir(), key=lambda item: item.name):
        if _is_ignored(child.name):
            continue
        if child.is_dir():
            issues.append(
                AuditIssue(
                    code="unexpected_sample_subdirectory",
                    severity="error",
                    message=f"Sample {sample_id} contains subdirectory {child.name}.",
                    action="Keep exactly four aligned files directly inside each sample folder.",
                    sample_id=sample_id,
                    path=_relative(child, root),
                )
            )
        else:
            files.append(child)

    annotation_candidates = [
        path for path in files if path.name.endswith("_CVJsonStd.json")
    ]
    eds_candidates = [path for path in files if path.name.endswith("_CVJsonStd.eds")]
    net_candidates = [path for path in files if path.name.endswith("_spice.net")]
    known = set(annotation_candidates + eds_candidates + net_candidates)
    image_candidates = [
        path
        for path in files
        if path not in known and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    recognized = known.union(image_candidates)
    for unexpected in sorted(set(files) - recognized):
        issues.append(
            AuditIssue(
                code="unexpected_sample_file",
                severity="error",
                message=f"Sample {sample_id} contains unexpected file {unexpected.name}.",
                action="Keep one image, one CVJsonStd JSON, one EDIF and one SPICE netlist.",
                sample_id=sample_id,
                path=_relative(unexpected, root),
            )
        )

    image = _require_one(
        "image", image_candidates, root, sample_id, issues, "schematic image"
    )
    annotation = _require_one(
        "json", annotation_candidates, root, sample_id, issues, "CVJsonStd JSON"
    )
    eds = _require_one("eds", eds_candidates, root, sample_id, issues, "EDIF")
    net = _require_one("net", net_candidates, root, sample_id, issues, "SPICE netlist")
    if image is None or annotation is None or eds is None or net is None:
        return None

    bases = {
        image.stem,
        annotation.name[: -len("_CVJsonStd.json")],
        eds.name[: -len("_CVJsonStd.eds")],
        net.name[: -len("_spice.net")],
    }
    if len(bases) != 1:
        issues.append(
            AuditIssue(
                code="sample_basename_mismatch",
                severity="error",
                message=(
                    f"Sample {sample_id} files do not share one aligned base name: "
                    f"{', '.join(sorted(bases))}."
                ),
                action="Restore a matching four-file sample from the official dataset.",
                sample_id=sample_id,
                path=sample_id,
            )
        )
        return None

    return _SampleFiles(
        sample_id=sample_id,
        directory=directory,
        base_name=next(iter(bases)),
        image=image,
        annotation=annotation,
        eds=eds,
        net=net,
    )


def _require_one(
    role: str,
    candidates: list[Path],
    root: Path,
    sample_id: str,
    issues: list[AuditIssue],
    label: str,
) -> Path | None:
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        issues.append(
            AuditIssue(
                code=f"missing_{role}_file",
                severity="error",
                message=f"Sample {sample_id} has no {label} file.",
                action=f"Restore the missing {label} file from the official dataset.",
                sample_id=sample_id,
                path=sample_id,
            )
        )
    else:
        issues.append(
            AuditIssue(
                code=f"multiple_{role}_files",
                severity="error",
                message=(
                    f"Sample {sample_id} has {len(candidates)} candidate {label} files: "
                    f"{', '.join(path.name for path in candidates)}."
                ),
                action=f"Keep exactly one aligned {label} file.",
                sample_id=sample_id,
                path=_relative(candidates[0].parent, root),
            )
        )
    return None


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _inspect_annotation(
    root: Path,
    sample: _SampleFiles,
    issues: list[AuditIssue],
) -> dict[str, Any] | None:
    try:
        raw = sample.annotation.read_text(encoding="utf-8")
        value = json.loads(raw, parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        issues.append(
            AuditIssue(
                code="invalid_json",
                severity="error",
                message=f"Annotation JSON is not strict UTF-8 JSON: {exc}",
                action="Regenerate or restore this CVJsonStd JSON file.",
                sample_id=sample.sample_id,
                path=_relative(sample.annotation, root),
            )
        )
        return None
    if not isinstance(value, dict):
        issues.append(
            AuditIssue(
                code="json_root_not_object",
                severity="error",
                message="Annotation JSON root must be an object.",
                action="Restore a CVJsonStd document with the expected root schema.",
                sample_id=sample.sample_id,
                path=_relative(sample.annotation, root),
            )
        )
        return None

    root_keys = set(value)
    for key in sorted(EXPECTED_ROOT_KEYS - root_keys):
        issues.append(
            AuditIssue(
                code="json_missing_root_key",
                severity="error",
                message=f"Annotation JSON is missing root key {key!r}.",
                action="Restore or regenerate the annotation using the CVJsonStd schema.",
                sample_id=sample.sample_id,
                path=_relative(sample.annotation, root),
            )
        )
    for key in sorted(root_keys - EXPECTED_ROOT_KEYS):
        issues.append(
            AuditIssue(
                code="json_unexpected_root_key",
                severity="error",
                message=f"Annotation JSON has unexpected root key {key!r}.",
                action="Confirm the schema version or remove non-CVJsonStd content.",
                sample_id=sample.sample_id,
                path=_relative(sample.annotation, root),
            )
        )

    canvas = value.get("canvas")
    canvas_valid = isinstance(canvas, dict)
    if not canvas_valid:
        issues.append(
            AuditIssue(
                code="invalid_canvas",
                severity="error",
                message="JSON canvas must be an object with positive integer width and height.",
                action="Restore valid canvas metadata from the official annotation.",
                sample_id=sample.sample_id,
                path=_relative(sample.annotation, root),
            )
        )
    else:
        for dimension in ("width", "height"):
            dimension_value = canvas.get(dimension)
            if type(dimension_value) is not int or dimension_value <= 0:
                canvas_valid = False
                issues.append(
                    AuditIssue(
                        code="invalid_canvas_dimension",
                        severity="error",
                        message=f"JSON canvas {dimension} must be a positive integer.",
                        action="Restore valid canvas dimensions from the official annotation.",
                        sample_id=sample.sample_id,
                        path=_relative(sample.annotation, root),
                    )
                )

    source_class = value.get("nowCls")
    if not isinstance(source_class, str) or not source_class.strip():
        issues.append(
            AuditIssue(
                code="invalid_source_class",
                severity="error",
                message="JSON nowCls must be a non-empty string.",
                action="Restore the source class in the annotation metadata.",
                sample_id=sample.sample_id,
                path=_relative(sample.annotation, root),
            )
        )

    counts: dict[str, int] = {}
    collections_valid = True
    for key in COLLECTION_KEYS:
        collection = value.get(key)
        if not isinstance(collection, dict):
            collections_valid = False
            issues.append(
                AuditIssue(
                    code="invalid_json_collection",
                    severity="error",
                    message=f"JSON {key} must be an object keyed by stable IDs.",
                    action=f"Restore the {key} mapping from a valid CVJsonStd annotation.",
                    sample_id=sample.sample_id,
                    path=_relative(sample.annotation, root),
                )
            )
        else:
            counts[key] = len(collection)

    if (
        root_keys != EXPECTED_ROOT_KEYS
        or not canvas_valid
        or not isinstance(source_class, str)
        or not source_class.strip()
        or not collections_valid
    ):
        return None
    return {
        "source_class": source_class,
        "canvas": {"width": canvas["width"], "height": canvas["height"]},
        "counts": counts,
    }


def _inspect_image(
    root: Path,
    sample: _SampleFiles,
    issues: list[AuditIssue],
) -> dict[str, Any] | None:
    try:
        with Image.open(sample.image) as image:
            image_format = image.format
            width, height = image.size
            # Force pixel decoding; Image.open alone only validates the header.
            image.load()
            perceptual_hash = perceptual_dhash(image)
            perceptual_stddev = float(np.asarray(image.convert("L"), dtype=np.float32).std())
    except (OSError, ValueError) as exc:
        issues.append(
            AuditIssue(
                code="invalid_image",
                severity="error",
                message=f"Pillow could not decode the schematic image: {exc}",
                action="Restore a complete, supported schematic image.",
                sample_id=sample.sample_id,
                path=_relative(sample.image, root),
            )
        )
        return None
    if not image_format:
        issues.append(
            AuditIssue(
                code="unknown_image_format",
                severity="error",
                message="Pillow decoded the image but could not identify its format.",
                action="Normalize the image into PNG, JPEG or WEBP in a derived dataset.",
                sample_id=sample.sample_id,
                path=_relative(sample.image, root),
            )
        )
        return None

    normalized_format = image_format.upper()
    extension = sample.image.suffix.lower()
    expected_extensions = FORMAT_EXTENSIONS.get(normalized_format)
    if expected_extensions is not None and extension not in expected_extensions:
        issues.append(
            AuditIssue(
                code="image_extension_mismatch",
                severity="warning",
                message=(
                    f"File extension {extension!r} declares a different type; "
                    f"Pillow magic identifies {normalized_format}."
                ),
                action=(
                    "Decode by detected content format, or rename/normalize the image "
                    "only in a derived copy of the dataset."
                ),
                sample_id=sample.sample_id,
                path=_relative(sample.image, root),
            )
        )
    return {
        "declared_extension": extension,
        "detected_format": normalized_format,
        "width": width,
        "height": height,
        "perceptual_hash": perceptual_hash,
        "perceptual_hash_algorithm": PERCEPTUAL_HASH_ALGORITHM,
        "perceptual_hash_eligible": perceptual_stddev >= 5.0,
        "grayscale_stddev": round(perceptual_stddev, 6),
    }


def _decode_utf8_file(
    path: Path,
    root: Path,
    sample_id: str,
    label: str,
    issues: list[AuditIssue],
) -> str | None:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        issues.append(
            AuditIssue(
                code=f"invalid_{label}_encoding",
                severity="error",
                message=f"{label.upper()} file is not readable UTF-8: {exc}",
                action=f"Restore a UTF-8 {label.upper()} file from the official dataset.",
                sample_id=sample_id,
                path=_relative(path, root),
            )
        )
        return None
    if not content.strip():
        issues.append(
            AuditIssue(
                code=f"empty_{label}_file",
                severity="error",
                message=f"{label.upper()} file is empty.",
                action=f"Restore a non-empty {label.upper()} file from the official dataset.",
                sample_id=sample_id,
                path=_relative(path, root),
            )
        )
        return None
    return content


def _inspect_eds(
    root: Path,
    sample: _SampleFiles,
    issues: list[AuditIssue],
) -> None:
    content = _decode_utf8_file(
        sample.eds, root, sample.sample_id, "eds", issues
    )
    if content is None:
        return
    if not content.lstrip().lower().startswith("(edif"):
        issues.append(
            AuditIssue(
                code="invalid_eds_header",
                severity="error",
                message="EDIF file does not begin with an (edif form.",
                action="Regenerate or restore the EDIF output.",
                sample_id=sample.sample_id,
                path=_relative(sample.eds, root),
            )
        )
    balance, minimum, open_quote = _edif_parenthesis_state(content)
    if balance != 0 or minimum < 0 or open_quote:
        issues.append(
            AuditIssue(
                code="invalid_eds_structure",
                severity="error",
                message=(
                    "EDIF parentheses/quoted strings are structurally unbalanced "
                    f"(balance={balance}, open_quote={open_quote})."
                ),
                action="Regenerate or restore the EDIF output.",
                sample_id=sample.sample_id,
                path=_relative(sample.eds, root),
            )
        )


def _edif_parenthesis_state(content: str) -> tuple[int, int, bool]:
    balance = 0
    minimum = 0
    quoted = False
    escaped = False
    for character in content:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character == "(":
            balance += 1
        elif character == ")":
            balance -= 1
            minimum = min(minimum, balance)
    return balance, minimum, quoted


def _inspect_net(
    root: Path,
    sample: _SampleFiles,
    issues: list[AuditIssue],
) -> None:
    content = _decode_utf8_file(
        sample.net, root, sample.sample_id, "net", issues
    )
    if content is None:
        return
    nonempty_lines = [line.strip().lower() for line in content.splitlines() if line.strip()]
    if not nonempty_lines or nonempty_lines[-1] != ".end":
        issues.append(
            AuditIssue(
                code="invalid_net_terminator",
                severity="error",
                message="SPICE netlist does not end with a standalone .end directive.",
                action="Regenerate or restore the SPICE netlist.",
                sample_id=sample.sample_id,
                path=_relative(sample.net, root),
            )
        )


def _fingerprint_file(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": _relative(path, root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _build_report(
    root: Path,
    expected_count: int,
    discovered_count: int,
    records: list[dict[str, Any]],
    duplicate_groups: list[dict[str, Any]],
    issues: list[AuditIssue],
) -> dict[str, Any]:
    severity_order = {"error": 0, "warning": 1, "info": 2}
    ordered_issues = sorted(
        issues,
        key=lambda issue: (
            severity_order.get(issue.severity, 99),
            issue.sample_id or "",
            issue.code,
            issue.path or "",
        ),
    )
    severity_counts = Counter(issue.severity for issue in ordered_issues)
    code_counts = Counter(issue.code for issue in ordered_issues)
    image_formats = Counter(
        record["files"]["image"]["detected_format"] for record in records
    )
    source_classes = Counter(record["source_class"] for record in records)
    object_totals = {
        key: sum(record["counts"][key] for record in records)
        for key in COLLECTION_KEYS
    }
    duplicate_sample_count = sum(group["size"] for group in duplicate_groups)

    error_count = severity_counts.get("error", 0)
    warning_count = severity_counts.get("warning", 0)
    return {
        "schema_version": 1,
        "dataset_root": str(root),
        "ok": error_count == 0,
        "strict_ok": error_count == 0 and warning_count == 0,
        "expected_samples": expected_count,
        "discovered_samples": discovered_count,
        "valid_manifest_records": len(records),
        "summary": {
            "errors": error_count,
            "warnings": warning_count,
            "infos": severity_counts.get("info", 0),
            "issues_by_code": dict(sorted(code_counts.items())),
            "image_formats": dict(sorted(image_formats.items())),
            "source_classes": dict(sorted(source_classes.items())),
            "object_totals": object_totals,
            "duplicate_groups": len(duplicate_groups),
            "samples_in_duplicate_groups": duplicate_sample_count,
            "effective_groups": len(records) - duplicate_sample_count + len(duplicate_groups),
        },
        "ignored_metadata": [".DS_Store", "._*", "__MACOSX"],
        "duplicate_groups": duplicate_groups,
        "issues": [issue.to_dict() for issue in ordered_issues],
    }
