from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a lowercase SHA-256 digest without loading the file at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


PERCEPTUAL_HASH_ALGORITHM = "dhash-16x16-v1"
PERCEPTUAL_HASH_BITS = 256
DEFAULT_NEAR_DUPLICATE_DISTANCE = 12


def perceptual_dhash(image: Image.Image) -> str:
    """Return a deterministic 256-bit difference hash for near-duplicate grouping."""

    grayscale = image.convert("L").resize((17, 16), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).reshape(-1)
    value = 0
    for index, bit in enumerate(bits):
        if bool(bit):
            value |= 1 << index
    return f"{value:064x}"


def perceptual_hash_distance(left: str, right: str) -> int:
    if len(left) != 64 or len(right) != 64:
        raise ValueError("perceptual hashes must be 64 hexadecimal characters")
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError as exc:
        raise ValueError("perceptual hashes must be hexadecimal") from exc


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        # Lexical roots make the operation independent of input order.
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def _file_hash(record: dict[str, Any], role: str) -> str:
    try:
        value = record["files"][role]["sha256"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"record {record.get('sample_id')!r} lacks files.{role}.sha256"
        ) from exc
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(
            f"record {record.get('sample_id')!r} has an invalid {role} SHA-256"
        )
    return value


def apply_duplicate_groups(
    records: list[dict[str, Any]],
    *,
    near_duplicate_distance: int = DEFAULT_NEAR_DUPLICATE_DISTANCE,
    maximum_aspect_ratio_delta: float = 0.02,
) -> list[dict[str, Any]]:
    """Add deterministic duplicate-group fields to records.

    Samples are joined transitively when they share annotation/image bytes or
    have a close perceptual hash, compatible aspect ratio and source domain.
    The returned list contains duplicate components only (singletons still
    receive a stable ``duplicate_group`` value).
    """

    sample_ids = [record.get("sample_id") for record in records]
    if any(not isinstance(sample_id, str) for sample_id in sample_ids):
        raise ValueError("every manifest record must have a string sample_id")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("manifest sample_id values must be unique")

    typed_ids = [str(sample_id) for sample_id in sample_ids]
    union_find = _UnionFind(typed_ids)
    collisions: dict[str, dict[str, list[str]]] = {
        "json": defaultdict(list),
        "image": defaultdict(list),
    }

    for record in records:
        sample_id = record["sample_id"]
        for role in ("json", "image"):
            collisions[role][_file_hash(record, role)].append(sample_id)

    for role_groups in collisions.values():
        for members in role_groups.values():
            ordered = sorted(members)
            for member in ordered[1:]:
                union_find.union(ordered[0], member)

    family_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in records:
        family = record.get("source_family")
        if isinstance(family, str) and family:
            family_groups[(str(record.get("source_class", "")), family)].append(
                record["sample_id"]
            )
    family_evidence: list[dict[str, Any]] = []
    for (source_class, family), members in sorted(family_groups.items()):
        ordered = sorted(members)
        if len(ordered) < 2:
            continue
        for member in ordered[1:]:
            union_find.union(ordered[0], member)
        family_evidence.append(
            {
                "kind": "source_family",
                "source_class": source_class,
                "family": family,
                "sample_ids": ordered,
            }
        )

    near_evidence: list[dict[str, Any]] = []
    if near_duplicate_distance >= 0:
        ordered_records = sorted(records, key=lambda item: item["sample_id"])
        for index, left in enumerate(ordered_records):
            left_hash = _perceptual_hash(left)
            if left_hash is None:
                continue
            for right in ordered_records[index + 1 :]:
                right_hash = _perceptual_hash(right)
                if right_hash is None:
                    continue
                if str(left.get("source_class", "")) != str(right.get("source_class", "")):
                    continue
                if not _compatible_aspect_ratio(left, right, maximum_aspect_ratio_delta):
                    continue
                distance = perceptual_hash_distance(left_hash, right_hash)
                if distance > near_duplicate_distance:
                    continue
                union_find.union(left["sample_id"], right["sample_id"])
                near_evidence.append(
                    {
                        "kind": "image_perceptual",
                        "algorithm": PERCEPTUAL_HASH_ALGORITHM,
                        "hamming_distance": distance,
                        "sample_ids": [left["sample_id"], right["sample_id"]],
                    }
                )

    components: dict[str, list[str]] = defaultdict(list)
    for sample_id in typed_ids:
        components[union_find.find(sample_id)].append(sample_id)

    group_for_sample: dict[str, tuple[str, list[str]]] = {}
    duplicate_summaries: list[dict[str, Any]] = []
    for members in sorted((sorted(value) for value in components.values())):
        if len(members) == 1:
            group_id = f"sample-{members[0]}"
        else:
            identity = "\0".join(members).encode("utf-8")
            group_id = f"dup-{hashlib.sha256(identity).hexdigest()[:16]}"

        for sample_id in members:
            group_for_sample[sample_id] = (group_id, members)

        if len(members) <= 1:
            continue

        member_set = set(members)
        evidence: list[dict[str, Any]] = []
        for role in ("json", "image"):
            for digest, digest_members in sorted(collisions[role].items()):
                shared = sorted(member_set.intersection(digest_members))
                if len(shared) > 1:
                    evidence.append(
                        {
                            "kind": role,
                            "sha256": digest,
                            "sample_ids": shared,
                        }
                    )
        for item in near_evidence:
            if set(item["sample_ids"]).issubset(member_set):
                evidence.append(dict(item))
        for item in family_evidence:
            if set(item["sample_ids"]).issubset(member_set):
                evidence.append(dict(item))
        duplicate_summaries.append(
            {
                "group_id": group_id,
                "sample_ids": members,
                "size": len(members),
                "evidence": evidence,
            }
        )

    for record in records:
        group_id, members = group_for_sample[record["sample_id"]]
        record["duplicate_group"] = group_id
        record["duplicate_group_size"] = len(members)

    return sorted(duplicate_summaries, key=lambda item: item["group_id"])


def _perceptual_hash(record: dict[str, Any]) -> str | None:
    try:
        image = record["files"]["image"]
        if image.get("perceptual_hash_eligible") is not True:
            return None
        value = image.get("perceptual_hash")
    except (KeyError, TypeError, AttributeError):
        return None
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(
            f"record {record.get('sample_id')!r} has an invalid image perceptual hash"
        )
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(
            f"record {record.get('sample_id')!r} has a non-hex image perceptual hash"
        ) from exc
    return value.lower()


def _compatible_aspect_ratio(
    left: dict[str, Any],
    right: dict[str, Any],
    maximum_delta: float,
) -> bool:
    try:
        left_image = left["files"]["image"]
        right_image = right["files"]["image"]
        left_ratio = float(left_image["width"]) / float(left_image["height"])
        right_ratio = float(right_image["width"]) / float(right_image["height"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    return abs(left_ratio - right_ratio) / max(left_ratio, right_ratio) <= maximum_delta
