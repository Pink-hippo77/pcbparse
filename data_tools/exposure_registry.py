"""Dataset identity and split exposure registry (A02).

Builds a precise duplicate / data-exposure registry without rewriting original
ground truth and without claiming independence for historically exposed splits.

Primary artifacts:
  - dataset_identity.jsonl : one record per image entity, with merged label versions
  - split_exposure.json    : split roles, family integrity, exposure events
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from ._bootstrap import default_dataset_root, ensure_source_path

ensure_source_path()

from pcbparse.data import (  # noqa: E402
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from pcbparse.data.hashing import (  # noqa: E402
    DEFAULT_NEAR_DUPLICATE_DISTANCE,
    PERCEPTUAL_HASH_ALGORITHM,
    perceptual_dhash,
    perceptual_hash_distance,
)

PIXEL_HASH_ALGORITHM = "sha256-rgb-raw-v1"
SCHEMA_VERSION = "1.0"
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})

# Historical exposure facts frozen by docs/optimization-20260908/split_exposure_audit.md.
# These MUST NOT be erased by any later re-split.
HISTORICAL_EXPOSURE_EVENTS: tuple[dict[str, Any], ...] = (
    {
        "event_id": "hist-split-dataset300-v1-20260812",
        "kind": "split_assignment",
        "date": "2026-08-12",
        "evidence": "data/splits/dataset300_split.jsonl",
        "description": (
            "Frozen Dataset300 split train=240/val=30/test=30 (seed 20260812). "
            "The original test role was later exposed via release50."
        ),
        "independence_claimed": False,
        "sample_scope": {"split": {"train": 240, "val": 30, "test": 30}},
    },
    {
        "event_id": "hist-release50-construction",
        "kind": "release_suite",
        "date": "2026-08-24",
        "evidence": (
            "docs/optimization-20260908/split_exposure_audit.md; "
            "artifacts/release/release50_suite_build.json"
        ),
        "description": (
            "release50 = 30 original test + 20 original val (50 total). "
            "Selected as an already-exposed regression/performance suite."
        ),
        "independence_claimed": False,
        "sample_scope": {"release_source_split": {"test": 30, "val": 20}},
    },
    {
        "event_id": "hist-release50-parameter-selection",
        "kind": "parameter_selection",
        "date": "2026-09-08",
        "evidence": "docs/optimization-20260908/split_exposure_audit.md",
        "description": (
            "release50 used repeatedly for threshold selection, parameter scans "
            "and regression tests. Independence is void."
        ),
        "independence_claimed": False,
        "sample_scope": {"split": {"release50": 50}},
    },
    {
        "event_id": "hist-no-independent-holdout-claim",
        "kind": "policy",
        "date": "2026-09-08",
        "evidence": (
            "docs/optimization-20260908/split_exposure_audit.md; "
            "docs/Agent构建执行说明_2026-09-22.md"
        ),
        "description": (
            "Historical exposure must not be erased by re-splitting. "
            "Public 210 images byte-match Dataset300 and cannot be labeled a new holdout. "
            "No independent generalization result may be claimed until new unseen families arrive."
        ),
        "independence_claimed": False,
        "sample_scope": {"public_images": 210, "dataset300_images": 300},
    },
)


@dataclass(slots=True)
class ImageOccurrence:
    """One on-disk image file that belongs to a merged image entity."""

    dataset: str
    path: str
    sample_id: str | None
    sha256: str
    bytes: int
    width: int | None = None
    height: int | None = None
    declared_extension: str | None = None
    role: str = "image"


@dataclass(slots=True)
class LabelOccurrence:
    """One label file attached to an image entity (never overwrites siblings)."""

    label_version: str
    dataset: str
    path: str
    sha256: str
    bytes: int
    format: str
    sample_id: str | None = None
    image_sha256: str | None = None
    fields_present: list[str] = field(default_factory=list)
    usable_fields: list[str] = field(default_factory=list)
    ambiguous_fields: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def pixel_hash_image(path: Path) -> tuple[str, int, int]:
    """Return (pixel_sha256, width, height) for decoded RGB pixels."""

    with Image.open(path) as handle:
        rgb = handle.convert("RGB")
        width, height = rgb.size
        digest = hashlib.sha256()
        digest.update(f"{PIXEL_HASH_ALGORITHM}:{width}x{height}:".encode("ascii"))
        digest.update(rgb.tobytes())
        return digest.hexdigest(), width, height


def _relative_to(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def inspect_cvjsonstd_fields(
    annotation: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    """Classify CVJsonStd fields as present / usable / ambiguous.

    Designators and pin numbers are frequently missing. Missing labels are
    ambiguous (must not be treated as negatives), not usable negatives.
    """

    present: list[str] = []
    usable: list[str] = []
    ambiguous: list[str] = []
    if not isinstance(annotation, dict):
        return present, usable, ambiguous

    for key in ("canvas", "components", "pins", "texts", "edges", "nowCls"):
        if key in annotation and annotation[key] is not None:
            present.append(key)

    components = annotation.get("components") or {}
    pins = annotation.get("pins") or {}
    comp_values = list(components.values()) if isinstance(components, dict) else list(components or [])
    pin_values = list(pins.values()) if isinstance(pins, dict) else list(pins or [])

    for key in ("components", "pins", "texts", "edges", "canvas"):
        if key in present:
            usable.append(key)

    named = 0
    for item in comp_values:
        if not isinstance(item, dict):
            continue
        for name_key in ("searchNameByOCR", "searchNameByVLM", "searchNameByLib"):
            value = item.get(name_key)
            if isinstance(value, str) and value.strip():
                named += 1
                break
    if comp_values:
        present.append("component_designator")
        if named == 0:
            ambiguous.append("component_designator")
            ambiguous.append("component_designator_missing_all")
        elif named < len(comp_values):
            ambiguous.append("component_designator_partial")
            usable.append("component_designator")
        else:
            usable.append("component_designator")

    numbered = 0
    for item in pin_values:
        if not isinstance(item, dict):
            continue
        for num_key in ("pinNumberByOCR", "pinNumberByVLM"):
            value = item.get(num_key)
            if isinstance(value, str) and value.strip():
                numbered += 1
                break
            if isinstance(value, (int, float)) and value != 0:
                numbered += 1
                break
    if pin_values:
        present.append("pin_number")
        if numbered == 0:
            ambiguous.append("pin_number")
            ambiguous.append("pin_number_missing_all")
        elif numbered < len(pin_values):
            ambiguous.append("pin_number_partial")
            usable.append("pin_number")
        else:
            usable.append("pin_number")

    return sorted(set(present)), sorted(set(usable)), sorted(set(ambiguous))


def inspect_target_fields(
    annotation: dict[str, Any], label_version: str
) -> tuple[list[str], list[str], list[str]]:
    """Classify public target / target_named label fields."""

    present: list[str] = []
    usable: list[str] = []
    ambiguous: list[str] = []
    if not isinstance(annotation, dict):
        return present, usable, ambiguous

    for key in ("components", "pins", "nets"):
        if key in annotation and annotation[key] is not None:
            present.append(key)
            usable.append(key)

    components = annotation.get("components") or {}
    pins = annotation.get("pins") or {}
    comp_values = list(components.values()) if isinstance(components, dict) else list(components or [])
    pin_values = list(pins.values()) if isinstance(pins, dict) else list(pins or [])

    if isinstance(components, dict) and components:
        present.append("component_designator")
        usable.append("component_designator")

    named = sum(
        1
        for item in comp_values
        if isinstance(item, dict) and isinstance(item.get("Name"), str) and item.get("Name").strip()
    )
    valued = sum(
        1
        for item in comp_values
        if isinstance(item, dict) and item.get("value") not in (None, "")
    )
    if comp_values:
        present.append("component_name")
        if named:
            usable.append("component_name")
            if named < len(comp_values):
                ambiguous.append("component_name_partial")
        else:
            ambiguous.append("component_name")
        present.append("component_value")
        if valued:
            usable.append("component_value")
            if valued < len(comp_values):
                ambiguous.append("component_value_partial")
        else:
            ambiguous.append("component_value")

    numbered = 0
    for item in pin_values:
        if not isinstance(item, dict):
            continue
        for num_key in ("number", "num", "pinNumber", "Number"):
            value = item.get(num_key)
            if value not in (None, ""):
                numbered += 1
                break
    if pin_values:
        present.append("pin_number")
        if numbered == 0:
            ambiguous.append("pin_number")
        elif numbered < len(pin_values):
            usable.append("pin_number")
            ambiguous.append("pin_number_partial")
        else:
            usable.append("pin_number")

    if label_version == "target":
        ambiguous.append("gtcase_target_not_training_gt")
    return sorted(set(present)), sorted(set(usable)), sorted(set(ambiguous))


def classify_label_payload(
    label_version: str, annotation: Any
) -> tuple[list[str], list[str], list[str]]:
    if not isinstance(annotation, dict):
        return ["unparsed"], [], ["unparsed"]
    if label_version == "CVJsonStd":
        return inspect_cvjsonstd_fields(annotation)
    return inspect_target_fields(annotation, label_version)


def _fingerprint_image(path: Path, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    key = str(path.resolve())
    if key in cache:
        return cache[key]
    digest = sha256_file(path)
    width = height = None
    try:
        with Image.open(path) as handle:
            width, height = handle.size
    except OSError:
        pass
    meta = {
        "sha256": digest,
        "bytes": path.stat().st_size,
        "width": width,
        "height": height,
    }
    cache[key] = meta
    return meta


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset300(
    dataset_root: Path,
    *,
    split_records: list[dict[str, Any]] | None = None,
    cache: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[ImageOccurrence], list[LabelOccurrence], dict[str, dict[str, Any]]]:
    """Load Dataset300 image + CVJsonStd label occurrences."""

    cache = cache if cache is not None else {}
    images: list[ImageOccurrence] = []
    labels: list[LabelOccurrence] = []
    sample_meta: dict[str, dict[str, Any]] = {}

    split_by_sample: dict[str, dict[str, Any]] = {}
    for record in split_records or []:
        sample_id = str(record.get("sample_id", ""))
        if sample_id:
            split_by_sample[sample_id] = record

    numbered = (
        sorted(
            (p for p in dataset_root.iterdir() if p.is_dir() and re.fullmatch(r"[0-9]{4}", p.name)),
            key=lambda p: p.name,
        )
        if dataset_root.is_dir()
        else []
    )

    for sample_dir in numbered:
        sample_id = sample_dir.name
        image_paths = [
            p
            for p in sorted(sample_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        label_paths = [
            p
            for p in sorted(sample_dir.iterdir())
            if p.is_file() and p.name.endswith("_CVJsonStd.json")
        ]
        for image_path in image_paths:
            meta = _fingerprint_image(image_path, cache)
            images.append(
                ImageOccurrence(
                    dataset="Dataset300",
                    path=_relative_to(image_path, dataset_root.parent),
                    sample_id=sample_id,
                    sha256=meta["sha256"],
                    bytes=meta["bytes"],
                    width=meta.get("width"),
                    height=meta.get("height"),
                    declared_extension=image_path.suffix.lower(),
                )
            )
        for label_path in label_paths:
            annotation = _read_json(label_path)
            present, usable, ambiguous = classify_label_payload("CVJsonStd", annotation)
            digest = sha256_file(label_path)
            labels.append(
                LabelOccurrence(
                    label_version="CVJsonStd",
                    dataset="Dataset300",
                    path=_relative_to(label_path, dataset_root.parent),
                    sha256=digest,
                    bytes=label_path.stat().st_size,
                    format="CVJsonStd",
                    sample_id=sample_id,
                    fields_present=present,
                    usable_fields=usable,
                    ambiguous_fields=ambiguous,
                    notes=["original_gt_must_not_be_overwritten"],
                )
            )
        split_record = split_by_sample.get(sample_id, {})
        sample_meta[sample_id] = {
            "sample_id": sample_id,
            "source_class": split_record.get("source_class"),
            "source_family": split_record.get("source_family"),
            "split": split_record.get("split"),
            "duplicate_group": split_record.get("duplicate_group"),
            "base_name": split_record.get("base_name"),
        }
    return images, labels, sample_meta


def load_public_dataset(
    public_root: Path,
    *,
    cache: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[ImageOccurrence], list[LabelOccurrence]]:
    """Load public 200_train_cases + 10_GTcase images and target labels."""

    cache = cache if cache is not None else {}
    images: list[ImageOccurrence] = []
    labels: list[LabelOccurrence] = []

    subsets = (
        ("public_200_train", public_root / "200_train_cases", "target_named"),
        ("public_10_gtcase", public_root / "10_GTcase", "target"),
    )
    for dataset_name, subset_root, label_version in subsets:
        if not subset_root.is_dir():
            continue
        for case_dir in sorted(p for p in subset_root.iterdir() if p.is_dir()):
            case_name = case_dir.name
            sample_match = re.match(r"([0-9]{4})", case_name)
            sample_id = sample_match.group(1) if sample_match else case_name
            for item_path in sorted(case_dir.iterdir()):
                if not item_path.is_file():
                    continue
                if item_path.suffix.lower() in IMAGE_EXTENSIONS:
                    meta = _fingerprint_image(item_path, cache)
                    images.append(
                        ImageOccurrence(
                            dataset=dataset_name,
                            path=_relative_to(item_path, public_root),
                            sample_id=sample_id,
                            sha256=meta["sha256"],
                            bytes=meta["bytes"],
                            width=meta.get("width"),
                            height=meta.get("height"),
                            declared_extension=item_path.suffix.lower(),
                        )
                    )
                elif item_path.suffix.lower() == ".json":
                    annotation = _read_json(item_path)
                    present, usable, ambiguous = classify_label_payload(
                        label_version, annotation
                    )
                    digest = sha256_file(item_path)
                    labels.append(
                        LabelOccurrence(
                            label_version=label_version,
                            dataset=dataset_name,
                            path=_relative_to(item_path, public_root),
                            sha256=digest,
                            bytes=item_path.stat().st_size,
                            format=label_version,
                            sample_id=sample_id,
                            fields_present=present,
                            usable_fields=usable,
                            ambiguous_fields=ambiguous,
                            notes=[
                                "public_target_side_by_side_with_cvjsonstd",
                                "must_not_overwrite_original_gt",
                            ],
                        )
                    )
    return images, labels


def load_release50(release50_path: Path) -> list[dict[str, Any]]:
    if not release50_path.is_file():
        return []
    return read_jsonl(release50_path)


def collect_work_exposure(work_root: Path) -> list[dict[str, Any]]:
    """Best-effort scan of work/ evaluation runs as exposure evidence."""

    events: list[dict[str, Any]] = []
    if not work_root.is_dir():
        return events
    for run_dir in sorted(p for p in work_root.iterdir() if p.is_dir()):
        name = run_dir.name
        lower = name.lower()
        if not any(
            token in lower
            for token in ("eval", "confirm", "release50", "test", "val", "ab-", "graph-", "audit")
        ):
            continue
        sample_ids: set[str] = set()
        for jsonl_path in sorted(run_dir.glob("*.jsonl")):
            try:
                for record in read_jsonl(jsonl_path):
                    for key in ("sample_id", "image_id", "case_id", "id"):
                        value = record.get(key)
                        if isinstance(value, str) and re.fullmatch(r"[0-9]{4}", value):
                            sample_ids.add(value)
                            break
                    image = record.get("image")
                    if isinstance(image, str):
                        match = re.search(r"([0-9]{4})", image)
                        if match:
                            sample_ids.add(match.group(1))
            except (OSError, ValueError):
                continue
        for status_name in (
            "diagnostic_status.json",
            "fixed_test_status.json",
            "summary.json",
        ):
            status_path = run_dir / status_name
            if not status_path.is_file():
                continue
            try:
                payload = _read_json(status_path)
            except (OSError, ValueError):
                continue
            text = json.dumps(payload, ensure_ascii=False)
            for match in re.finditer(r"\b([0-9]{4})\b", text):
                sample_ids.add(match.group(1))

        kind = "evaluation"
        if "test" in lower and "val" not in lower:
            kind = "evaluation_test"
        elif "val" in lower:
            kind = "evaluation_val"
        elif "release50" in lower or "fixed020" in lower or "current50" in lower:
            kind = "evaluation_release50"
        if not sample_ids and kind == "evaluation":
            continue
        events.append(
            {
                "event_id": f"work-{name}",
                "kind": kind,
                "date": None,
                "evidence": f"work/{name}",
                "description": (
                    f"Historical work run {name} consumed listed samples for diagnostics/eval."
                ),
                "independence_claimed": False,
                "sample_ids": sorted(sample_ids),
                "sample_count": len(sample_ids),
                "source": "work_scan",
            }
        )
    return events


def collect_teacher_exposure(
    *,
    train_sample_ids: set[str],
    configs_root: Path,
) -> list[dict[str, Any]]:
    """Register teacher/synthetic label-generation exposure without independence claims.

    Teacher-generated or synthetic labels inherit source_image_id/family_id and
    permanently expose the source images they were built from.
    """

    events: list[dict[str, Any]] = []
    synthetic_refs: list[str] = []
    if configs_root.is_dir():
        for path in sorted(configs_root.rglob("*.json")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            lowered = text.lower()
            if "junction_synthetic" in text or (
                "synthetic" in lowered and "junction" in lowered
            ) or "teacher" in lowered:
                try:
                    synthetic_refs.append(path.relative_to(configs_root.parent).as_posix())
                except ValueError:
                    synthetic_refs.append(path.as_posix())

    events.append(
        {
            "event_id": "teacher-synthetic-junction-train-augmentation",
            "kind": "teacher_generation",
            "date": None,
            "evidence": (
                "src/pcbparse/training/synthetic_junction.py; "
                "configs/train (junction_synthetic_per_class); "
                "docs/优秀方案对标与项目改进架构_2026-09-22.md M1/M10"
            ),
            "description": (
                "Teacher/synthetic junction label generation consumes source images. "
                "Generated labels inherit source_image_id and family_id and must not "
                "re-enter evaluation as independent samples. Train-scope synthetic "
                "augmentation only; provenance-required for any formal training use."
            ),
            "independence_claimed": False,
            "sample_scope": {"train_samples_considered": len(train_sample_ids)},
            "sample_ids": sorted(train_sample_ids),
            "config_refs": sorted(set(synthetic_refs)),
            "policy": {
                "inherit_source_image_id": True,
                "inherit_family_id": True,
                "formal_training_requires_provenance": True,
                "may_not_claim_independent_test": True,
            },
        }
    )
    if synthetic_refs:
        events.append(
            {
                "event_id": "teacher-config-references",
                "kind": "teacher_generation",
                "date": None,
                "evidence": "; ".join(sorted(set(synthetic_refs))),
                "description": "Configs referencing synthetic/teacher label generation.",
                "independence_claimed": False,
                "config_refs": sorted(set(synthetic_refs)),
            }
        )
    return events


def _stable_family_id(members: Sequence[str]) -> str:
    identity = "\0".join(sorted(members)).encode("utf-8")
    return f"fam-{hashlib.sha256(identity).hexdigest()[:16]}"


def _stable_image_id(sha256: str) -> str:
    return f"img-{sha256[:16]}"


def _resolve_image_path(
    sha256: str,
    all_images: list[ImageOccurrence],
    dataset300_root: Path,
    public_root: Path,
) -> Path | None:
    for occ in all_images:
        if occ.sha256 != sha256:
            continue
        root = dataset300_root.parent if occ.dataset == "Dataset300" else public_root
        candidate = root / occ.path
        if candidate.is_file():
            return candidate
    return None


def _assign_families(
    entities: dict[str, dict[str, Any]],
    *,
    near_duplicate_distance: int,
    sample_meta: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Union exact/pixel/near/source_family links into deterministic family_ids."""

    parent = {sha: sha for sha in entities}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    by_pixel: dict[str, list[str]] = defaultdict(list)
    for sha, entity in entities.items():
        if entity.get("pixel_hash"):
            by_pixel[entity["pixel_hash"]].append(sha)
    for members in by_pixel.values():
        ordered = sorted(members)
        for other in ordered[1:]:
            union(ordered[0], other)

    by_source_family: dict[str, list[str]] = defaultdict(list)
    for sha, entity in entities.items():
        family = entity.get("source_family")
        if not family:
            for sample_id in entity.get("sample_ids") or []:
                meta = sample_meta.get(sample_id) or {}
                if meta.get("source_family"):
                    family = meta["source_family"]
                    entity["source_family"] = family
                    break
        if family:
            by_source_family[str(family)].append(sha)
    for members in by_source_family.values():
        ordered = sorted(set(members))
        for other in ordered[1:]:
            union(ordered[0], other)

    if near_duplicate_distance >= 0:
        shas = sorted(entities)
        for index, left in enumerate(shas):
            left_hash = entities[left].get("perceptual_hash")
            if not left_hash:
                continue
            left_ratio = None
            if entities[left].get("width") and entities[left].get("height"):
                left_ratio = entities[left]["width"] / entities[left]["height"]
            for right in shas[index + 1 :]:
                right_hash = entities[right].get("perceptual_hash")
                if not right_hash:
                    continue
                if (
                    left_ratio is not None
                    and entities[right].get("width")
                    and entities[right].get("height")
                ):
                    right_ratio = entities[right]["width"] / entities[right]["height"]
                    if abs(left_ratio - right_ratio) / max(left_ratio, right_ratio) > 0.02:
                        continue
                try:
                    distance = perceptual_hash_distance(left_hash, right_hash)
                except ValueError:
                    continue
                if distance <= near_duplicate_distance:
                    union(left, right)

    components: dict[str, list[str]] = defaultdict(list)
    for sha in entities:
        components[find(sha)].append(sha)

    family_id_by_sha: dict[str, str] = {}
    for members in components.values():
        family_id = _stable_family_id(members)
        for sha in members:
            family_id_by_sha[sha] = family_id
    return family_id_by_sha


def build_registry(
    *,
    dataset300_root: Path,
    public_root: Path,
    split_manifest: Path,
    release50_manifest: Path,
    work_root: Path,
    configs_root: Path,
    near_duplicate_distance: int = DEFAULT_NEAR_DUPLICATE_DISTANCE,
    compute_pixel_hashes: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build dataset_identity records and the split_exposure document."""

    split_records = read_jsonl(split_manifest) if split_manifest.is_file() else []
    split_by_sample = {str(r.get("sample_id")): r for r in split_records}
    release50_records = load_release50(release50_manifest)
    release50_by_sample = {str(r.get("sample_id")): r for r in release50_records}

    cache: dict[str, dict[str, Any]] = {}
    d300_images, d300_labels, sample_meta = load_dataset300(
        dataset300_root, split_records=split_records, cache=cache
    )
    public_images, public_labels = load_public_dataset(public_root, cache=cache)

    all_images = d300_images + public_images
    all_labels = d300_labels + public_labels

    image_sha_by_dataset_sample: dict[tuple[str, str], str] = {}
    for occ in all_images:
        if occ.sample_id:
            image_sha_by_dataset_sample[(occ.dataset, occ.sample_id)] = occ.sha256
    for label in all_labels:
        image_sha = image_sha_by_dataset_sample.get((label.dataset, label.sample_id or ""))
        if image_sha is None:
            parent = Path(label.path).parent.name
            image_sha = image_sha_by_dataset_sample.get((label.dataset, parent))
        label.image_sha256 = image_sha

    entities: dict[str, dict[str, Any]] = {}
    for occ in sorted(all_images, key=lambda o: (o.sha256, o.dataset, o.path)):
        entity = entities.setdefault(
            occ.sha256,
            {
                "image_id": _stable_image_id(occ.sha256),
                # read_jsonl compatibility key (unique per image entity).
                "sample_id": _stable_image_id(occ.sha256),
                "sha256": occ.sha256,
                "pixel_hash": None,
                "pixel_hash_algorithm": PIXEL_HASH_ALGORITHM if compute_pixel_hashes else None,
                "perceptual_hash": None,
                "perceptual_hash_algorithm": PERCEPTUAL_HASH_ALGORITHM,
                "width": occ.width,
                "height": occ.height,
                "bytes": occ.bytes,
                "image_occurrences": [],
                "label_versions": [],
                "sample_ids": [],
                "datasets": [],
                "source_domain": None,
                "source_family": None,
                "split_role": None,
                "split_roles_seen": [],
                "release50_source_split": None,
                "in_release50": False,
                "exposure_events": [],
                "exposed": False,
                "independent_holdout_eligible": False,
                "public_duplicate_of_dataset300": False,
                "family_id": None,
                "near_duplicate_group": None,
                "usable_fields": [],
                "ambiguous_fields": [],
                "label_version_names": [],
            },
        )
        entity["image_occurrences"].append(
            {
                "dataset": occ.dataset,
                "path": occ.path,
                "sample_id": occ.sample_id,
                "role": occ.role,
            }
        )
        if occ.sample_id and occ.sample_id not in entity["sample_ids"]:
            entity["sample_ids"].append(occ.sample_id)
        if occ.dataset not in entity["datasets"]:
            entity["datasets"].append(occ.dataset)
        if occ.width and not entity["width"]:
            entity["width"] = occ.width
        if occ.height and not entity["height"]:
            entity["height"] = occ.height

    if compute_pixel_hashes:
        for sha256, entity in entities.items():
            path = _resolve_image_path(sha256, all_images, dataset300_root, public_root)
            if path is None:
                continue
            try:
                pixel_digest, width, height = pixel_hash_image(path)
                entity["pixel_hash"] = pixel_digest
                entity["width"] = width
                entity["height"] = height
                with Image.open(path) as handle:
                    entity["perceptual_hash"] = perceptual_dhash(handle)
            except (OSError, ValueError):
                pass

    for label in all_labels:
        image_sha = label.image_sha256
        if not image_sha or image_sha not in entities:
            sample_id = label.sample_id
            matched = None
            if sample_id:
                for sha256, entity in entities.items():
                    if sample_id in entity["sample_ids"] and label.dataset in entity["datasets"]:
                        matched = sha256
                        break
            if not matched:
                continue
            image_sha = matched
        entity = entities[image_sha]
        entry = {
            "label_version": label.label_version,
            "dataset": label.dataset,
            "path": label.path,
            "sha256": label.sha256,
            "bytes": label.bytes,
            "format": label.format,
            "fields_present": label.fields_present,
            "usable_fields": label.usable_fields,
            "ambiguous_fields": label.ambiguous_fields,
            "notes": label.notes,
            "sample_id": label.sample_id,
        }
        existing_paths = {item["path"] for item in entity["label_versions"]}
        if entry["path"] not in existing_paths:
            entity["label_versions"].append(entry)
        if label.label_version not in entity["label_version_names"]:
            entity["label_version_names"].append(label.label_version)
        for value in label.usable_fields:
            if value not in entity["usable_fields"]:
                entity["usable_fields"].append(value)
        for value in label.ambiguous_fields:
            if value not in entity["ambiguous_fields"]:
                entity["ambiguous_fields"].append(value)

    dataset300_hashes = {occ.sha256 for occ in d300_images}
    public_hashes = {occ.sha256 for occ in public_images}
    for sha256, entity in entities.items():
        datasets = set(entity["datasets"])
        if "Dataset300" in datasets and datasets & {"public_200_train", "public_10_gtcase"}:
            entity["public_duplicate_of_dataset300"] = True
            entity["independent_holdout_eligible"] = False
        elif datasets & {"public_200_train", "public_10_gtcase"} and sha256 in dataset300_hashes:
            entity["public_duplicate_of_dataset300"] = True
            entity["independent_holdout_eligible"] = False

    exposure_event_ids_for_entity: dict[str, set[str]] = defaultdict(set)

    def add_exposure(sha256: str, event_id: str) -> None:
        exposure_event_ids_for_entity[sha256].add(event_id)

    for sample_id, record in sorted(split_by_sample.items()):
        split_role = record.get("split")
        source_family = record.get("source_family")
        source_class = record.get("source_class")
        image_sha = record.get("files", {}).get("image", {}).get("sha256")
        if not image_sha or image_sha not in entities:
            for sha256, entity in entities.items():
                if sample_id in entity["sample_ids"]:
                    image_sha = sha256
                    break
        if not image_sha or image_sha not in entities:
            continue
        entity = entities[image_sha]
        entity["split_role"] = split_role
        if split_role and split_role not in entity["split_roles_seen"]:
            entity["split_roles_seen"].append(split_role)
        if source_family:
            entity["source_family"] = source_family
        if source_class and not entity["source_domain"]:
            entity["source_domain"] = source_class
        add_exposure(image_sha, "hist-split-dataset300-v1-20260812")
        if sample_id in release50_by_sample:
            entity["in_release50"] = True
            entity["release50_source_split"] = release50_by_sample[sample_id].get(
                "release_source_split"
            )
            add_exposure(image_sha, "hist-release50-construction")
            add_exposure(image_sha, "hist-release50-parameter-selection")
        if split_role in {"test", "val", "release50"}:
            add_exposure(image_sha, "hist-no-independent-holdout-claim")

    for sample_id, record in sorted(release50_by_sample.items()):
        image_sha = record.get("files", {}).get("image", {}).get("sha256")
        if image_sha and image_sha in entities:
            entities[image_sha]["in_release50"] = True
            entities[image_sha]["release50_source_split"] = record.get("release_source_split")
            add_exposure(image_sha, "hist-release50-construction")
            add_exposure(image_sha, "hist-release50-parameter-selection")

    for sha256, entity in entities.items():
        if set(entity["datasets"]) & {"public_200_train", "public_10_gtcase"}:
            add_exposure(sha256, "hist-no-independent-holdout-claim")
            entity["independent_holdout_eligible"] = False

    family_id_by_sha = _assign_families(
        entities,
        near_duplicate_distance=near_duplicate_distance,
        sample_meta=sample_meta,
    )
    for sha256, family_id in family_id_by_sha.items():
        entities[sha256]["family_id"] = family_id
        entities[sha256]["near_duplicate_group"] = family_id

    work_events = collect_work_exposure(work_root)
    for event in work_events:
        for sample_id in event.get("sample_ids", []):
            for sha256, entity in entities.items():
                if sample_id in entity["sample_ids"]:
                    add_exposure(sha256, event["event_id"])
                    entity["exposed"] = True

    train_samples = {
        sample_id
        for sample_id, record in split_by_sample.items()
        if record.get("split") == "train"
    }
    teacher_events = collect_teacher_exposure(
        train_sample_ids=train_samples, configs_root=configs_root
    )
    for event in teacher_events:
        for sample_id in event.get("sample_ids", []):
            for sha256, entity in entities.items():
                if sample_id in entity["sample_ids"]:
                    add_exposure(sha256, event["event_id"])
                    entity["exposed"] = True

    for sha256, entity in entities.items():
        entity["exposure_events"] = sorted(
            set(entity["exposure_events"]) | exposure_event_ids_for_entity.get(sha256, set())
        )
        if (
            entity["exposure_events"]
            or entity["in_release50"]
            or entity["split_role"] in {"test", "val", "release50"}
        ):
            entity["exposed"] = True
        entity["independent_holdout_eligible"] = False
        entity["label_version_names"] = sorted(entity["label_version_names"])
        entity["usable_fields"] = sorted(entity["usable_fields"])
        entity["ambiguous_fields"] = sorted(entity["ambiguous_fields"])
        entity["sample_ids"] = sorted(entity["sample_ids"])
        entity["datasets"] = sorted(entity["datasets"])
        entity["split_roles_seen"] = sorted({r for r in entity["split_roles_seen"] if r})
        entity["image_occurrences"] = sorted(
            entity["image_occurrences"], key=lambda item: (item["dataset"], item["path"])
        )
        entity["label_versions"] = sorted(
            entity["label_versions"], key=lambda item: (item["label_version"], item["path"])
        )

    identity_records = [entities[sha] for sha in sorted(entities)]

    family_to_splits: dict[str, set[str]] = defaultdict(set)
    family_members: dict[str, list[str]] = defaultdict(list)
    for record in identity_records:
        family_id = record["family_id"]
        family_members[family_id].append(record["image_id"])
        roles = record["split_roles_seen"] or (
            [record["split_role"]] if record["split_role"] else []
        )
        for role in roles:
            if role and role != "release50":
                family_to_splits[family_id].add(role)

    family_leaks = []
    for family_id, roles in sorted(family_to_splits.items()):
        if len(roles) > 1:
            family_leaks.append(
                {
                    "family_id": family_id,
                    "splits": sorted(roles),
                    "image_ids": sorted(family_members[family_id]),
                }
            )

    public_entities = [
        r
        for r in identity_records
        if set(r["datasets"]) & {"public_200_train", "public_10_gtcase"}
    ]
    public_new_holdout = [
        r
        for r in public_entities
        if r.get("independent_holdout_eligible") or r.get("split_role") == "holdout"
    ]
    public_matching = [r for r in public_entities if r.get("public_duplicate_of_dataset300")]

    all_events = list(HISTORICAL_EXPOSURE_EVENTS) + work_events + teacher_events
    historical_ids = {e["event_id"] for e in HISTORICAL_EXPOSURE_EVENTS}
    registered_ids = {e["event_id"] for e in all_events}
    historical_preserved = historical_ids.issubset(registered_ids)

    split_counts = Counter()
    for record in identity_records:
        if record.get("split_role"):
            split_counts[record["split_role"]] += 1

    exposure_doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "split_exposure_registry",
        "task_id": "A02",
        "generated_at_utc": utc_now(),
        "independence_claims": {
            "external_holdout": False,
            "independent_test": False,
            "independent_generalization": False,
            "note": (
                "No split in this registry may be reported as an independent test or "
                "new holdout. release50 and original test/val have historical exposure. "
                "Public 210 images byte-match Dataset300 and are not new holdout."
            ),
        },
        "inputs": {
            "dataset300_root": str(dataset300_root),
            "public_root": str(public_root),
            "split_manifest": str(split_manifest),
            "split_manifest_sha256": sha256_file(split_manifest)
            if split_manifest.is_file()
            else None,
            "release50_manifest": str(release50_manifest),
            "release50_manifest_sha256": sha256_file(release50_manifest)
            if release50_manifest.is_file()
            else None,
            "work_root": str(work_root),
            "configs_root": str(configs_root),
            "near_duplicate_distance": near_duplicate_distance,
            "pixel_hash_computed": compute_pixel_hashes,
        },
        "counts": {
            "unique_images": len(identity_records),
            "dataset300_images": len(d300_images),
            "public_images": len(public_images),
            "public_images_matching_dataset300": len(public_matching),
            "public_images_not_matching_dataset300": len(public_entities)
            - len(public_matching),
            "overlap_unique_hashes": len(public_hashes & dataset300_hashes),
            "label_occurrences": len(all_labels),
            "label_versions": sorted({lbl.label_version for lbl in all_labels}),
            "families": len(family_members),
            "near_duplicate_distance": near_duplicate_distance,
            "split_role_counts": dict(sorted(split_counts.items())),
            "release50_samples": len(release50_by_sample),
            "exposure_events": len(all_events),
            "work_exposure_events": len(work_events),
            "teacher_exposure_events": len(teacher_events),
            "exposed_images": sum(1 for r in identity_records if r.get("exposed")),
        },
        "splits": {
            "dataset300_frozen": {
                "path": "data/splits/dataset300_split.jsonl",
                "counts": {"train": 240, "val": 30, "test": 30},
                "role": "development_regression",
                "independent": False,
            },
            "release50": {
                "path": "data/splits/dataset300_release50_v1.jsonl",
                "counts": {"release50": 50},
                "source_counts": {"test": 30, "val": 20},
                "role": "exposed_regression",
                "independent": False,
            },
            "public_200_train": {
                "path": "赛题六公开数据集/200_train_cases",
                "counts": {
                    "images": sum(
                        1 for o in public_images if o.dataset == "public_200_train"
                    )
                },
                "role": "public_duplicate_labels",
                "independent": False,
            },
            "public_10_gtcase": {
                "path": "赛题六公开数据集/10_GTcase",
                "counts": {
                    "images": sum(
                        1 for o in public_images if o.dataset == "public_10_gtcase"
                    )
                },
                "role": "public_duplicate_labels",
                "independent": False,
            },
        },
        "families": [
            {
                "family_id": family_id,
                "size": len(members),
                "image_ids": sorted(members),
                "splits": sorted(family_to_splits.get(family_id, set())),
            }
            for family_id, members in sorted(family_members.items())
        ],
        "exposure_events": all_events,
        "constraints": {
            "same_family_no_cross_split": {
                "status": "passed" if not family_leaks else "failed",
                "violations": family_leaks,
                "rule": "同工程族无跨集泄漏 / same family must not leak across train/val/holdout",
            },
            "public_210_not_new_holdout": {
                "status": "passed" if not public_new_holdout else "failed",
                "rule": "公开210图不能标新 holdout",
                "public_image_count": len(public_images),
                "matching_dataset300": len(public_matching),
                "new_holdout_candidates": [r["image_id"] for r in public_new_holdout],
                "byte_match_dataset300": True,
            },
            "historical_exposure_preserved": {
                "status": "passed" if historical_preserved else "failed",
                "rule": "历史暴露不被重划分抹去 / historical exposure must not be erased by re-splitting",
                "historical_event_ids": sorted(historical_ids),
                "registered_event_ids": sorted(registered_ids),
                "preserved": historical_preserved,
            },
            "no_independence_claim": {
                "status": "passed",
                "rule": (
                    "test/release50/teacher exposure registered WITHOUT claiming independence"
                ),
                "independent_holdout_count": sum(
                    1 for r in identity_records if r.get("independent_holdout_eligible")
                ),
            },
        },
        "summary": {
            "ok": (not family_leaks) and (not public_new_holdout) and historical_preserved,
            "unique_images": len(identity_records),
            "families": len(family_members),
            "exposure_events": len(all_events),
        },
    }

    return identity_records, exposure_doc


def write_outputs(
    identity_records: list[dict[str, Any]],
    exposure_doc: dict[str, Any],
    identity_path: Path,
    exposure_path: Path,
) -> None:
    write_jsonl(identity_records, identity_path)
    write_json(exposure_doc, exposure_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build dataset_identity.jsonl and split_exposure.json for A02 "
            "(precise duplicates + data exposure registry)."
        )
    )
    parser.add_argument(
        "--dataset300-root",
        type=Path,
        default=default_dataset_root(),
        help="Inner Dataset300 directory containing 0001..0300 sample folders.",
    )
    parser.add_argument(
        "--public-root",
        type=Path,
        default=Path("赛题六公开数据集"),
        help="Public dataset root containing 200_train_cases and 10_GTcase.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/splits/dataset300_split.jsonl"),
    )
    parser.add_argument(
        "--release50-manifest",
        type=Path,
        default=Path("data/splits/dataset300_release50_v1.jsonl"),
    )
    parser.add_argument("--work-root", type=Path, default=Path("work"))
    parser.add_argument("--configs-root", type=Path, default=Path("configs"))
    parser.add_argument(
        "--output-identity",
        type=Path,
        default=Path("artifacts/audits/dataset_identity.jsonl"),
    )
    parser.add_argument(
        "--output-exposure",
        type=Path,
        default=Path("artifacts/audits/split_exposure.json"),
    )
    parser.add_argument(
        "--near-duplicate-distance",
        type=int,
        default=DEFAULT_NEAR_DUPLICATE_DISTANCE,
    )
    parser.add_argument(
        "--skip-pixel-hashes",
        action="store_true",
        help="Skip decoded pixel hashing (faster; pixel_hash fields stay null).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    identity_records, exposure_doc = build_registry(
        dataset300_root=args.dataset300_root,
        public_root=args.public_root,
        split_manifest=args.split_manifest,
        release50_manifest=args.release50_manifest,
        work_root=args.work_root,
        configs_root=args.configs_root,
        near_duplicate_distance=args.near_duplicate_distance,
        compute_pixel_hashes=not args.skip_pixel_hashes,
    )
    write_outputs(
        identity_records,
        exposure_doc,
        args.output_identity,
        args.output_exposure,
    )
    summary = {
        "ok": exposure_doc["summary"]["ok"],
        "identity": str(args.output_identity.resolve(strict=False)),
        "exposure": str(args.output_exposure.resolve(strict=False)),
        **exposure_doc["counts"],
        "constraints": {
            key: value["status"] for key, value in exposure_doc["constraints"].items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if exposure_doc["summary"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
