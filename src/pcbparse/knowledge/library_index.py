"""Deterministic index over the offline cpntLibrary JSON file.

Evidence constraints (A07 / M4):
- Pin rows are multi-source symbol aggregates: `pin_row_count` is never the
  physical pin count. Physical pin count is the count of unique `number`s.
- `pin_id` is not unique and is retained as source provenance only.
- Name/type conflicts inside one `number` are preserved, never collapsed.
- No image coordinates exist in the library and none are synthesized here.
- `same_hash_components` is a grouping hint only, not proof of identical package.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = "library_index.v1"

# JLC / house suffixes and trailing instance indices seen on library keys.
_VARIANT_SUFFIX = re.compile(r"_(?:C\d+|[A-Z]{1,4}|\d+)$")

_PIN_TYPES = frozenset(
    {"IN", "OUT", "BI", "Passive", "Power", "Undefined", ""}
)


class LibraryIndexError(ValueError):
    """Raised when the offline library cannot be indexed safely."""


def default_library_path() -> Path:
    """Return the extensionless public library file path."""
    return Path("赛题六公开数据集") / "cpntLibrary"


def normalize_model_name(value: str) -> str:
    """Normalize a model string for retrieval (NFKC, case-fold, squeeze)."""
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(value)).strip()
    normalized = normalized.replace("　", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.upper()


def split_variant_suffix(key: str) -> tuple[str, str | None]:
    """Split trailing instance/part suffixes from a library key.

    Repeatedly strips `_C12345`, `_JX`, `_0`-style suffixes so
    `AMS1117-3.3_C6186_1` yields base `AMS1117-3.3`.
    """
    base = key
    parts: list[str] = []
    while True:
        match = _VARIANT_SUFFIX.search(base)
        if not match:
            break
        token = match.group(0)[1:]
        # Keep at least one meaningful segment on the left.
        remainder = base[: match.start()]
        if not remainder:
            break
        parts.append(token)
        base = remainder
    variant = "_".join(reversed(parts)) if parts else None
    return base, variant


def _as_pin_number(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _aggregate_pins(
    pins: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int, int, list[str]]:
    """Aggregate pin rows by `number`, preserving name/type conflicts.

    Returns (aggregates, pin_row_count, physical_pin_count, conflict_notes).
    """
    rows = list(pins)
    by_number: "OrderedDict[str, list[Mapping[str, Any]]]" = OrderedDict()
    for row in rows:
        number = _as_pin_number(row.get("number"))
        by_number.setdefault(number, []).append(row)

    aggregates: list[dict[str, Any]] = []
    conflict_notes: list[str] = []
    for number, group_rows in by_number.items():
        name_counter: Counter[str] = Counter()
        name_pin_types: dict[str, set[str]] = {}
        name_pin_ids: dict[str, list[str]] = {}
        all_pin_ids: list[str] = []
        for row in group_rows:
            name = "" if row.get("name") is None else str(row.get("name"))
            pin_type = "" if row.get("pin_type") is None else str(row.get("pin_type"))
            if pin_type not in _PIN_TYPES:
                conflict_notes.append(
                    f"number={number!r}: unexpected pin_type={pin_type!r}"
                )
            pin_id = "" if row.get("pin_id") is None else str(row.get("pin_id"))
            name_counter[name] += 1
            name_pin_types.setdefault(name, set()).add(pin_type)
            name_pin_ids.setdefault(name, []).append(pin_id)
            all_pin_ids.append(pin_id)

        name_candidates = []
        for name, row_count in sorted(
            name_counter.items(), key=lambda item: (-item[1], item[0])
        ):
            name_candidates.append(
                {
                    "name": name,
                    "row_count": row_count,
                    "pin_types": sorted(name_pin_types.get(name, set())),
                    "pin_ids": name_pin_ids.get(name, []),
                }
            )

        pin_id_counts = Counter(all_pin_ids)
        duplicate_pin_ids = sorted(
            pin_id for pin_id, count in pin_id_counts.items() if pin_id and count > 1
        )
        name_conflict = len(name_candidates) > 1
        type_conflict = any(
            len(candidate["pin_types"]) > 1 for candidate in name_candidates
        )
        if name_conflict:
            conflict_notes.append(
                f"number={number!r}: name_candidates="
                f"{[c['name'] for c in name_candidates]}"
            )
        if duplicate_pin_ids:
            conflict_notes.append(
                f"number={number!r}: duplicate_pin_id={duplicate_pin_ids}"
            )

        aggregates.append(
            {
                "number": number,
                "pin_row_count": len(group_rows),
                "name_conflict": name_conflict,
                "pin_type_conflict": type_conflict,
                "duplicate_pin_ids": duplicate_pin_ids,
                "name_candidates": name_candidates,
            }
        )

    return aggregates, len(rows), len(by_number), conflict_notes


def load_raw_library(path: Path | str | None = None) -> tuple[dict[str, Any], str, int]:
    """Load the raw library mapping and return (data, sha256, byte_size)."""
    library_path = Path(path) if path is not None else default_library_path()
    if not library_path.is_file():
        raise LibraryIndexError(f"library file not found: {library_path}")
    raw = library_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LibraryIndexError(f"library JSON decode failed: {exc}") from exc
    if not isinstance(data, dict):
        raise LibraryIndexError("library root must be a JSON object keyed by name")
    return data, digest, len(raw)


def _index_component(key: str, record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise LibraryIndexError(f"component {key!r} record must be an object")
    pins = record.get("pins")
    if pins is None:
        pins = []
    if not isinstance(pins, list):
        raise LibraryIndexError(f"component {key!r} pins must be a list")
    aggregates, pin_row_count, physical_pin_count, conflict_notes = _aggregate_pins(
        pins
    )
    base_name, variant_suffix = split_variant_suffix(key)
    same_hash = record.get("same_hash_components") or []
    if not isinstance(same_hash, list):
        raise LibraryIndexError(
            f"component {key!r} same_hash_components must be a list"
        )
    # Grouping hint only — never treated as identical footprint/package.
    same_hash_components = []
    for item in same_hash:
        if not isinstance(item, Mapping):
            continue
        same_hash_components.append(
            {
                "project": str(item.get("project", "")),
                "title": str(item.get("title", "")),
                "symbol": str(item.get("symbol", "")),
            }
        )

    return {
        "library_key": key,
        "model_raw": key,
        "normalized_name": normalize_model_name(key),
        "base_name": base_name,
        "base_normalized": normalize_model_name(base_name),
        "variant_suffix": variant_suffix,
        "esym_path": str(record.get("esym_path", "")),
        "pin_row_count": pin_row_count,
        "physical_pin_count": physical_pin_count,
        "pin_rows_match_physical_count": pin_row_count == physical_pin_count,
        "pins_by_number": aggregates,
        "conflict_notes": conflict_notes,
        "has_geometry": False,
        "geometry_source": "absent_in_library",
        "same_hash_components": same_hash_components,
        "same_hash_is_package_identity": False,
    }


def build_library_index(
    path: Path | str | None = None,
    *,
    raw: Mapping[str, Any] | None = None,
    library_sha256: str | None = None,
    library_bytes: int | None = None,
) -> dict[str, Any]:
    """Build a deterministic, evidence-constrained index document."""
    library_path = Path(path) if path is not None else default_library_path()
    if raw is None:
        data, digest, size = load_raw_library(library_path)
    else:
        data = dict(raw)
        digest = library_sha256 or ""
        size = library_bytes if library_bytes is not None else 0

    components = {}
    total_pin_rows = 0
    total_physical = 0
    conflict_components = 0
    row_not_physical = 0
    for key in sorted(data.keys(), key=lambda item: (item.casefold(), item)):
        entry = _index_component(key, data[key])
        components[key] = entry
        total_pin_rows += entry["pin_row_count"]
        total_physical += entry["physical_pin_count"]
        if entry["conflict_notes"]:
            conflict_components += 1
        if not entry["pin_rows_match_physical_count"]:
            row_not_physical += 1

    index = {
        "schema_version": SCHEMA_VERSION,
        "library_path": str(library_path).replace("\\", "/"),
        "library_sha256": digest,
        "library_bytes": size,
        "component_count": len(components),
        "totals": {
            "pin_row_count": total_pin_rows,
            "physical_pin_count_sum": total_physical,
            "components_with_conflicts": conflict_components,
            "components_where_pin_rows_ne_physical": row_not_physical,
        },
        "invariants": {
            "pin_rows_equal_physical_count": False,
            "coordinates_emitted": False,
            "pin_id_is_unique": False,
            "same_hash_is_package_identity": False,
        },
        "components": components,
    }
    return index


def write_library_index(index: Mapping[str, Any], out_path: Path | str) -> Path:
    """Write the index JSON (UTF-8, stable separators)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build cpntLibrary index JSON")
    parser.add_argument("--library", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    index = build_library_index(args.library)
    write_library_index(index, args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "component_count": index["component_count"],
                "library_sha256": index["library_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
