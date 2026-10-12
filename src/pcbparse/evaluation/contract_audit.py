"""A01 ContractAuditor: public target vs official schema, versioned and explicit.

Hard rules:
- never rewrite original GT bytes
- never invent unobservable internal IDs (emptyset / ``∅*`` keys stay placeholders)
- every conflict is recorded with status ``confirmed`` | ``conflicting`` | ``unresolved``
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pcbparse.output.official_schema import (
    DEFAULT_OFFICIAL_SCHEMA,
    EMPTY_PIN_NUMBER_TOKENS,
    PIN_NUMBER_SPLIT_RE,
    load_official_schema,
    split_official_pin_numbers,
)

CONTRACT_PROFILE_VERSION = "public_targets_contract_profile_v2"
EMPTYSET_RE = re.compile(r"^[∅Ø]")
EMPTY_PIN_TOKENS = EMPTY_PIN_NUMBER_TOKENS
# Official guide allows comma-delimited multi pin numbers such as ``5,6,7``.
MULTI_PIN_SPLIT_RE = PIN_NUMBER_SPLIT_RE

PIN_KEY_TEMPLATE_RE = re.compile(r"^pin_(?P<number>.*)$")
# GT lowercase spellings observed in public targets -> official canonical types.
# Keys match real ``component.type`` strings; values come from
# ``DEFAULT_OFFICIAL_SCHEMA['component']['type_aliases']`` targets.
GT_TYPE_TO_OFFICIAL: dict[str, str] = {
    "box": "ic",
    "boxic": "ic",
    "r": "resistor",
    "c": "capacitor",
    "l": "inductor",
    "crystal": "crystal",
    "switch": "switch",
    "gnd": "gnd",
    "net_input": "vbus_in",
}

STATUS_CONFIRMED = "confirmed"
STATUS_CONFLICTING = "conflicting"
STATUS_UNRESOLVED = "unresolved"
STATUS_MAPPED = "mapped"
STATUS_MISSING = "missing"


@dataclass(frozen=True, slots=True)
class ConflictRecord:
    item: str
    status: str
    gt_kind: str = ""
    official_alias_target: str = ""
    count: int = 0
    note: str = ""
    example_paths: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "item": self.item,
            "status": self.status,
            "gt_kind": self.gt_kind,
            "official_alias_target": self.official_alias_target,
            "count": self.count,
        }
        if self.note:
            payload["note"] = self.note
        if self.example_paths:
            payload["example_paths"] = list(self.example_paths)
        return payload


@dataclass(slots=True)
class TargetInspection:
    path: str
    kind: str
    sha256: str
    top_keys: list[str]
    type_counts: dict[str, int]
    field_presence: dict[str, int]
    missing_markers: dict[str, int]
    placeholder_keys: dict[str, int]
    pin_number_shapes: dict[str, int]
    edge_shapes: dict[str, int]
    examples: dict[str, list[str]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "sha256": self.sha256,
            "top_keys": self.top_keys,
            "type_counts": self.type_counts,
            "field_presence": self.field_presence,
            "missing_markers": self.missing_markers,
            "placeholder_keys": self.placeholder_keys,
            "pin_number_shapes": self.pin_number_shapes,
            "edge_shapes": self.edge_shapes,
            "examples": self.examples,
        }


def discover_targets(root: Path) -> list[Path]:
    """Discover both public label patterns under ``root``.

    Train labels are ``*_target_named.json`` (200). Golden labels are
    ``*_target.json`` without the ``_named`` suffix (10).  Searching only one
    pattern misses half the public set.
    """
    named = sorted(root.rglob("*_target_named.json"))
    plain = sorted(
        path
        for path in root.rglob("*_target.json")
        if not path.name.endswith("_target_named.json")
    )
    return named + plain


def classify_target(path: Path) -> str:
    return (
        "train200_target_named"
        if path.name.endswith("_target_named.json")
        else "golden10_target"
    )


def _is_emptyset_key(key: object) -> bool:
    return bool(EMPTYSET_RE.match(str(key)))


def _pin_number_shape(pin_key: object) -> str:
    text = str(pin_key)
    if text in EMPTY_PIN_TOKENS:
        return "empty_or_placeholder"
    match = PIN_KEY_TEMPLATE_RE.match(text)
    if not match:
        return "non_template"
    number = match.group("number")
    if number in EMPTY_PIN_TOKENS:
        return "empty_or_placeholder"
    parts = [part for part in MULTI_PIN_SPLIT_RE.split(number) if part != ""]
    if len(parts) > 1:
        return "multi_number"
    if number.isdigit():
        return "single_numeric"
    return "single_other"


def _is_zero_length_edge(edge: object) -> bool:
    """Zero-length official edge: two identical points (not a whole line->point rewrite)."""
    if not isinstance(edge, Sequence) or isinstance(edge, (str, bytes)):
        return False
    if len(edge) != 2:
        return False
    first, second = edge[0], edge[1]
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        try:
            return float(first["x"]) == float(second["x"]) and float(first["y"]) == float(
                second["y"]
            )
        except (KeyError, TypeError, ValueError):
            return False
    return first == second


def _edge_shape(edge: object) -> str:
    if isinstance(edge, Mapping):
        return "object"
    if not isinstance(edge, Sequence) or isinstance(edge, (str, bytes)):
        return "other"
    if len(edge) == 2:
        return "two_point_array"
    return f"point_count_{len(edge)}"


def inspect_target(path: Path) -> TargetInspection:
    """Read-only inventory of one public target JSON. Never writes back."""
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    components = data.get("components") or {}
    pins = data.get("pins") or {}
    nets = data.get("nets") or {}

    type_counts: Counter[str] = Counter()
    field_presence: Counter[str] = Counter()
    missing_markers: Counter[str] = Counter()
    placeholder_keys: Counter[str] = Counter()
    pin_number_shapes: Counter[str] = Counter()
    edge_shapes: Counter[str] = Counter()
    examples: dict[str, list[str]] = {
        "emptyset_keys": [],
        "null_values": [],
        "empty_pin_numbers": [],
        "empty_pin_names": [],
        "multi_pin_numbers": [],
        "zero_length_edges": [],
        "duplicate_designator_names": [],
    }

    def _note(bucket: str, value: str, limit: int = 5) -> None:
        if len(examples[bucket]) < limit:
            examples[bucket].append(value)

    if isinstance(components, Mapping):
        names: list[str] = []
        for key, value in components.items():
            key_text = str(key)
            if _is_emptyset_key(key_text):
                placeholder_keys["emptyset_component_keys"] += 1
                _note("emptyset_keys", key_text)
            if not isinstance(value, Mapping):
                missing_markers["component.non_object"] += 1
                continue
            for field_name, field_value in value.items():
                field_presence[f"component.{field_name}"] += 1
                if field_value is None:
                    missing_markers[f"component.{field_name}.null"] += 1
                    if field_name == "value":
                        _note("null_values", f"{key_text}.value")
                elif isinstance(field_value, str) and field_value == "":
                    missing_markers[f"component.{field_name}.empty_string"] += 1
            type_counts[str(value.get("type", ""))] += 1
            name_value = value.get("Name", value.get("name"))
            if isinstance(name_value, str):
                names.append(name_value)
        name_counts = Counter(names)
        for name, count in sorted(name_counts.items()):
            if count > 1 and name:
                placeholder_keys["duplicate_designator_names"] += count
                _note("duplicate_designator_names", f"{name}x{count}")

    if isinstance(pins, Mapping):
        for owner, group in pins.items():
            owner_text = str(owner)
            if not isinstance(group, Mapping):
                missing_markers["pin.non_object_group"] += 1
                continue
            if not group:
                placeholder_keys["empty_pin_groups"] += 1
                missing_markers["pin.empty_group"] += 1
                continue
            for pin_key, pin_value in group.items():
                shape = _pin_number_shape(pin_key)
                pin_number_shapes[shape] += 1
                if shape == "empty_or_placeholder":
                    _note("empty_pin_numbers", f"{owner_text}:{pin_key}")
                if shape == "multi_number":
                    _note("multi_pin_numbers", f"{owner_text}:{pin_key}")
                if not isinstance(pin_value, Mapping):
                    missing_markers["pin.non_object"] += 1
                    continue
                for field_name, field_value in pin_value.items():
                    field_presence[f"pin.{field_name}"] += 1
                    if field_value is None:
                        missing_markers[f"pin.{field_name}.null"] += 1
                    elif isinstance(field_value, str) and field_value == "":
                        missing_markers[f"pin.{field_name}.empty_string"] += 1
                        if field_name == "pinname":
                            _note("empty_pin_names", f"{owner_text}:{pin_key}")

    if isinstance(nets, Mapping):
        for net_key, net_value in nets.items():
            if not isinstance(net_value, Mapping):
                missing_markers["net.non_object"] += 1
                continue
            for field_name, field_value in net_value.items():
                field_presence[f"net.{field_name}"] += 1
                if field_value is None:
                    missing_markers[f"net.{field_name}.null"] += 1
                elif isinstance(field_value, str) and field_value == "":
                    missing_markers[f"net.{field_name}.empty_string"] += 1
            edges = net_value.get("edges") or {}
            if isinstance(edges, Mapping):
                for edge_key, edge in edges.items():
                    edge_shapes[_edge_shape(edge)] += 1
                    if _is_zero_length_edge(edge):
                        placeholder_keys["zero_length_edges"] += 1
                        _note("zero_length_edges", f"{net_key}:{edge_key}")

    return TargetInspection(
        path=path.as_posix(),
        kind=classify_target(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        top_keys=list(data.keys()),
        type_counts=dict(type_counts),
        field_presence=dict(field_presence),
        missing_markers=dict(missing_markers),
        placeholder_keys=dict(placeholder_keys),
        pin_number_shapes=dict(pin_number_shapes),
        edge_shapes=dict(edge_shapes),
        examples=examples,
    )


def _official_type_aliases() -> dict[str, str]:
    return dict(DEFAULT_OFFICIAL_SCHEMA["component"]["type_aliases"])


def _resolve_gt_type(gt_type: str) -> tuple[str, str, str]:
    """Return (mapped_target, status, note) for one observed GT type string."""
    if not gt_type:
        return "", STATUS_MISSING, "empty type string"
    aliases = _official_type_aliases()
    if gt_type in GT_TYPE_TO_OFFICIAL:
        mapped = GT_TYPE_TO_OFFICIAL[gt_type]
        official_direct = aliases.get(gt_type)
        if official_direct and official_direct != mapped:
            return (
                mapped,
                STATUS_CONFLICTING,
                f"GT lower alias maps to {mapped!r} but official_schema alias {gt_type!r}->{official_direct!r}",
            )
        note = ""
        # Case-variant official keys such as R/C/L/NET_input share the same target.
        case_variants = [
            key
            for key, value in aliases.items()
            if value == mapped and key.casefold() == gt_type.casefold()
        ]
        if case_variants:
            note = f"official_schema case variants: {sorted(case_variants)}"
        return mapped, STATUS_MAPPED, note
    if gt_type in aliases:
        return aliases[gt_type], STATUS_MAPPED, "direct official_schema alias"
    lowered = {key.casefold(): value for key, value in aliases.items()}
    if gt_type.casefold() in lowered:
        return (
            lowered[gt_type.casefold()],
            STATUS_MAPPED,
            "matched official_schema alias case-insensitively",
        )
    return (
        gt_type,
        STATUS_CONFLICTING,
        "no official_schema type alias; keep GT spelling, do not invent a mapping",
    )


@dataclass(slots=True)
class ContractAuditor:
    """Compare public targets against the versioned official contract."""

    schema_path: Path | None = None

    def official_profile(self) -> dict[str, Any]:
        if self.schema_path is not None and self.schema_path.exists():
            loaded = json.loads(self.schema_path.read_text(encoding="utf-8"))
            profile = loaded.get("official_schema", loaded)
            return dict(load_official_schema(profile))
        return dict(load_official_schema(DEFAULT_OFFICIAL_SCHEMA))

    def inspect_many(self, paths: Sequence[Path]) -> list[TargetInspection]:
        return [inspect_target(path) for path in paths]

    def compare(
        self,
        targets: Sequence[TargetInspection],
        profile: Mapping[str, Any] | None = None,
        guide_rules: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        del profile  # always bound to the loaded official profile below
        official = self.official_profile()
        guide_rules = tuple(guide_rules or ())
        all_types: Counter[str] = Counter()
        top_key_sets: Counter[tuple[str, ...]] = Counter()
        fields: Counter[str] = Counter()
        missing_markers: Counter[str] = Counter()
        placeholders: Counter[str] = Counter()
        pin_shapes: Counter[str] = Counter()
        edge_shapes: Counter[str] = Counter()
        example_paths: dict[str, list[str]] = {}
        by_kind: dict[str, int] = {}

        def _remember(bucket: str, path: str, limit: int = 5) -> None:
            examples = example_paths.setdefault(bucket, [])
            if len(examples) < limit and path not in examples:
                examples.append(path)

        for report in targets:
            all_types.update(report.type_counts)
            top_key_sets[tuple(report.top_keys)] += 1
            fields.update(report.field_presence)
            missing_markers.update(report.missing_markers)
            placeholders.update(report.placeholder_keys)
            pin_shapes.update(report.pin_number_shapes)
            edge_shapes.update(report.edge_shapes)
            by_kind[report.kind] = by_kind.get(report.kind, 0) + 1
            for bucket, values in report.examples.items():
                if values:
                    _remember(bucket, report.path)

        conflicts: list[ConflictRecord] = []

        official_top = list(official["top_level_keys"])
        for keys, file_count in sorted(top_key_sets.items(), key=lambda item: (-item[1], item[0])):
            status = STATUS_CONFIRMED if list(keys) == official_top else STATUS_CONFLICTING
            conflicts.append(
                ConflictRecord(
                    item="top_level_keys",
                    status=status,
                    gt_kind=",".join(keys),
                    official_alias_target=",".join(official_top),
                    count=file_count,
                    note="file count with this exact top-level key set",
                )
            )

        for type_name in sorted(all_types):
            mapped, status, note = _resolve_gt_type(type_name)
            conflicts.append(
                ConflictRecord(
                    item=f"component.type:{type_name or '<empty>'}",
                    status=status,
                    gt_kind=type_name,
                    official_alias_target=mapped,
                    count=all_types[type_name],
                    note=note,
                )
            )

        # Name vs name spelling: official examples use ``Name``; profile allows both.
        name_count = fields.get("component.Name", 0)
        lower_name_count = fields.get("component.name", 0)
        if name_count or lower_name_count:
            if name_count and not lower_name_count:
                status = STATUS_CONFIRMED
                note = "public targets use official-example 'Name'"
            elif lower_name_count and not name_count:
                status = STATUS_CONFLICTING
                note = "public targets use official-table 'name'"
            else:
                status = STATUS_UNRESOLVED
                note = "both 'Name' and 'name' present; keep spelling explicit"
            conflicts.append(
                ConflictRecord(
                    item="component.name_spelling",
                    status=status,
                    gt_kind=f"Name={name_count},name={lower_name_count}",
                    official_alias_target=str(official["component"]["fields"][1]),
                    count=name_count + lower_name_count,
                    note=note,
                )
            )

        value_null = missing_markers.get("component.value.null", 0)
        conflicts.append(
            ConflictRecord(
                item="component.value.null",
                status=STATUS_CONFIRMED,
                gt_kind="null",
                official_alias_target="string|null",
                count=value_null,
                note="null value is an explicit missing marker, not empty string",
                example_paths=tuple(example_paths.get("null_values", ())),
            )
        )

        emptyset_count = placeholders.get("emptyset_component_keys", 0)
        conflicts.append(
            ConflictRecord(
                item="component.key.placeholder",
                status=STATUS_CONFLICTING,
                gt_kind="emptyset_prefix_keys",
                official_alias_target="UNKNOWN_<internal_id>",
                count=emptyset_count,
                note=(
                    "Do not invent emptyset IDs at prediction time — unobservable. "
                    "Keep as placeholders in the diff."
                ),
                example_paths=tuple(example_paths.get("emptyset_keys", ())),
            )
        )

        empty_pin_groups = placeholders.get("empty_pin_groups", 0)
        conflicts.append(
            ConflictRecord(
                item="pins.empty_group",
                status=STATUS_UNRESOLVED,
                gt_kind="empty pin dict",
                official_alias_target="auto-numbered pin_N",
                count=empty_pin_groups,
                note=(
                    "Empty pin number is unrepresentable (PIN.MISSING_NUMBER); "
                    "fully empty pin groups need official clarification."
                ),
            )
        )

        empty_pin_numbers = pin_shapes.get("empty_or_placeholder", 0)
        conflicts.append(
            ConflictRecord(
                item="pins.empty_number",
                status=STATUS_CONFIRMED,
                gt_kind="empty_or_placeholder pin key",
                official_alias_target="rejected: PIN.MISSING_NUMBER",
                count=empty_pin_numbers,
                note="empty/placeholder pin numbers are not OCR success; prune or reject, never invent",
                example_paths=tuple(example_paths.get("empty_pin_numbers", ())),
            )
        )

        multi_pin = pin_shapes.get("multi_number", 0)
        conflicts.append(
            ConflictRecord(
                item="pins.multi_number",
                status=STATUS_CONFIRMED,
                gt_kind="comma-delimited pin numbers",
                official_alias_target="split into pin_5, pin_6, pin_7",
                count=multi_pin,
                note="split_official_pin_numbers expands 5,6,7 without rewriting source",
                example_paths=tuple(example_paths.get("multi_pin_numbers", ())),
            )
        )

        zero_length = placeholders.get("zero_length_edges", 0)
        conflicts.append(
            ConflictRecord(
                item="net.zero_length_edges",
                status=STATUS_CONFIRMED,
                gt_kind="two identical points",
                official_alias_target="legal two_point_array edge",
                count=zero_length,
                note="do not rewrite all segments into points; only identical endpoints count",
                example_paths=tuple(example_paths.get("zero_length_edges", ())),
            )
        )

        duplicate_names = placeholders.get("duplicate_designator_names", 0)
        conflicts.append(
            ConflictRecord(
                item="component.duplicate_designator",
                status=STATUS_CONFIRMED,
                gt_kind="repeated Name/designator",
                official_alias_target="KEY_1, KEY_2 by natural internal order",
                count=duplicate_names,
                note="serializer disambiguates; IR keeps the original labels",
                example_paths=tuple(example_paths.get("duplicate_designator_names", ())),
            )
        )

        coord = official.get("numbers", {})
        conflicts.append(
            ConflictRecord(
                item="coordinates.frame",
                status=STATUS_CONFIRMED,
                gt_kind=str(coord.get("source_coordinate_system", "image_top_left_xy")),
                official_alias_target=str(coord.get("coordinate_system", "original_bottom_left_xy")),
                count=len(targets),
                note=(
                    "Flip y only at the official boundary: y_out = image_height - y. "
                    "Double flip must restore source coordinates."
                ),
            )
        )

        conflicts.append(
            ConflictRecord(
                item="component.extra_fields",
                status=STATUS_CONFIRMED if official.get("extra_fields") == "forbid" else STATUS_CONFLICTING,
                gt_kind="extra_fields",
                official_alias_target=str(official.get("extra_fields")),
                count=0,
                note="official output forbids extra fields; GT inventory records observed fields only",
            )
        )

        for rule in guide_rules:
            conflicts.append(
                ConflictRecord(
                    item=f"guide_rule:{rule[:48]}",
                    status=STATUS_UNRESOLVED,
                    gt_kind=rule,
                    official_alias_target="",
                    count=0,
                    note="guide rule retained as evidence; not a measured conflict",
                )
            )

        return {
            "schema_version": "1.0",
            "kind": "contract_diff",
            "task_id": "A01",
            "profile_id": CONTRACT_PROFILE_VERSION,
            "official_schema_version": official.get("version"),
            "target_counts": by_kind,
            "file_count": len(targets),
            "top_level_key_sets": [
                {"keys": list(keys), "files": count}
                for keys, count in sorted(
                    top_key_sets.items(), key=lambda item: (-item[1], item[0])
                )
            ],
            "official_top_level": official_top,
            "gt_type_counts": dict(sorted(all_types.items(), key=lambda item: (-item[1], item[0]))),
            "gt_field_presence": dict(sorted(fields.items())),
            "missing_markers": dict(sorted(missing_markers.items())),
            "placeholder_key_counts": dict(sorted(placeholders.items())),
            "pin_number_shapes": dict(sorted(pin_shapes.items())),
            "edge_shapes": dict(sorted(edge_shapes.items())),
            "type_alias_table": {
                "gt_to_official": dict(sorted(GT_TYPE_TO_OFFICIAL.items())),
                "official_schema_type_aliases": _official_type_aliases(),
            },
            "conflicts": [item.as_dict() for item in conflicts],
            "policy": {
                "rewrite_original_gt": False,
                "guess_unobservable_internal_ids": False,
                "raw_gt_preserved": True,
                "emptyset_keys_remain_placeholders": True,
            },
            "example_paths": example_paths,
        }

    def build_profile(self, diff: Mapping[str, Any]) -> dict[str, Any]:
        official = self.official_profile()
        conflicts = list(diff.get("conflicts") or [])
        unmapped = sorted(
            {
                item.get("gt_kind", "")
                for item in conflicts
                if str(item.get("item", "")).startswith("component.type:")
                and item.get("status") == STATUS_CONFLICTING
                and item.get("gt_kind")
            }
        )
        return {
            "schema_version": "1.0",
            "kind": "contract_profile",
            "profile_id": CONTRACT_PROFILE_VERSION,
            "status": "versioned_with_unresolved",
            "official_schema_version": official.get("version"),
            "official_component_fields": list(official["component"]["fields"]),
            "official_pin_fields": list(official["pin"]["fields"]),
            "official_net_fields": list(official["net"]["fields"]),
            "type_aliases_gt_to_official": dict(sorted(GT_TYPE_TO_OFFICIAL.items())),
            "official_schema_type_aliases": _official_type_aliases(),
            "unmapped_gt_types": unmapped,
            "missing_markers": dict(diff.get("missing_markers") or {}),
            "placeholder_key_policy": {
                "gt_prefix": "emptyset",
                "serializer_output_prefix": "UNKNOWN_",
                "prediction_may_invent_gt_placeholder": False,
                "status": STATUS_CONFLICTING,
                "note": "∅75 style keys stay placeholders in the diff; never guessed at prediction time",
            },
            "pin_number_policy": {
                "empty_allowed_auto_number": True,
                "multi_number_split_on_comma": True,
                "status": STATUS_CONFIRMED,
            },
            "duplicate_designator_policy": {
                "ir_keeps_original_labels": True,
                "serializer_emits": "KEY_1, KEY_2 by natural internal_id order",
                "status": STATUS_CONFIRMED,
            },
            "coordinate_policy": {
                "source": official.get("numbers", {}).get("source_coordinate_system"),
                "official": official.get("numbers", {}).get("coordinate_system"),
                "transform": "y_out = image_height - y at official boundary only",
                "status": STATUS_CONFIRMED,
            },
            "zero_length_edge_policy": {
                "definition": "edge with two identical points",
                "rewrite_all_segments_to_points": False,
                "status": STATUS_CONFIRMED,
            },
            "acceptance": {
                "raw_gt_not_rewritten": True,
                "conflicts_explicit": True,
                "no_guessing_unobservable_ids": True,
            },
        }

    def run(
        self,
        root: Path,
        output_dir: Path,
        *,
        schema_path: Path | None = None,
        guide_rules: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        if schema_path is not None:
            self.schema_path = schema_path
        paths = discover_targets(root)
        reports = self.inspect_many(paths)
        diff = self.compare(reports, guide_rules=guide_rules)
        profile = self.build_profile(diff)
        output_dir.mkdir(parents=True, exist_ok=True)
        inventory_path = output_dir / "contract_target_inventory.json"
        diff_path = output_dir / "contract_diff.json"
        profile_path = output_dir / "contract_profile.json"
        inventory = {
            "schema_version": "1.0",
            "kind": "contract_target_inventory",
            "profile_id": CONTRACT_PROFILE_VERSION,
            "file_count": len(reports),
            "files": [item.as_dict() for item in reports],
        }
        inventory_path.write_text(
            json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        diff_path.write_text(
            json.dumps(diff, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        profile_path.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "file_count": len(reports),
            "conflicts": len(diff["conflicts"]),
            "inventory_path": inventory_path.as_posix(),
            "diff_path": diff_path.as_posix(),
            "profile_path": profile_path.as_posix(),
            "target_sha256": {item.path: item.sha256 for item in reports},
        }


def flip_y(value: float, image_height: float) -> float:
    """Official boundary y flip. Source image uses top-left origin."""
    return image_height - value


def flip_y_twice(value: float, image_height: float) -> float:
    return flip_y(flip_y(value, image_height), image_height)


def expand_official_pin_numbers(value: object) -> tuple[str, ...]:
    """Public multi-number pin keys expand without rewriting source JSON."""
    text = "" if value is None else str(value).strip()
    match = PIN_KEY_TEMPLATE_RE.match(text)
    number = match.group("number") if match else text
    return split_official_pin_numbers(number)


def classify_edge(edge: object) -> str:
    if _is_zero_length_edge(edge):
        return "zero_length"
    return _edge_shape(edge)
