from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import math
import re
from string import Formatter
from typing import Any

from pcbparse.config import load_config


class OfficialSchemaError(ValueError):
    """Raised when an official profile or serialized submission is invalid."""


EMPTY_PIN_NUMBER_TOKENS = frozenset({"", "∅", "Ø", "None", "null", "NULL", "none"})
PIN_NUMBER_SPLIT_RE = re.compile(r"[,，]")


def split_official_pin_numbers(value: object) -> tuple[str, ...]:
    """Return explicit official pin numbers, or empty when unrepresentable.

    The guide requires comma-delimited values such as ``5,6,7`` to become
    separate pins.  Blank/placeholder/OCR-noise values return empty: callers
    must resolve the identity before serialization: Q&A permits local output
    IDs for non-Box pins; missing scored Box numbers require evidence. Dots and
    parentheses cannot be represented without colliding with the documented
    ``(U1.A5,R1.1)`` HyperGraph grammar.
    """

    if not isinstance(value, str):
        return ()
    text = value.strip()
    if not text or text in EMPTY_PIN_NUMBER_TOKENS:
        return ()
    parts = tuple(part.strip() for part in PIN_NUMBER_SPLIT_RE.split(text))
    if not parts or any(
        not part
        or part in EMPTY_PIN_NUMBER_TOKENS
        or any(char in part for char in ".()")
        for part in parts
    ):
        return ()
    return tuple(dict.fromkeys(parts))


DEFAULT_OFFICIAL_SCHEMA: dict[str, Any] = {
    "version": "qna-type-list-10gt-0930-v2",
    "top_level_keys": ["components", "pins", "nets"],
    "component": {
        # The guide's prose table says ``name`` while every machine-readable
        # JSON example on pages 13, 15, 25 uses ``Name``.  The profile makes
        # that ambiguity explicit: the default follows the examples, while
        # validate_profile also permits an explicitly selected official-table
        # lowercase-name profile.
        "fields": ["type", "Name", "value", "bbox"],
        "bbox_encoding": "xyxy_list",
        "allowed_types": [
            'gnd', 'c', 'r', 'v', 'box',
            'net_input', 'net_short', 'net_output', 'net_bidirection', 'd',
            'led', 'pin', 'switch', 'l', 'mosfet',
            'other', 'm3螺丝', 'bjt', 'testpoint', 'crystal',
            'buzzer', 'battary', 'fuse', 'block', 'amp',
            'jumper', 'motor', 'esd', 'dc', 'transfomer',
            'ant', 'circle_header', 'opt', 'seg',
        ],
        "type_aliases": {
            'gnd': 'gnd',
            'c': 'c',
            'r': 'r',
            'v': 'v',
            'box': 'box',
            'net_input': 'net_input',
            'net_short': 'net_short',
            'net_output': 'net_output',
            'net_bidirection': 'net_bidirection',
            'd': 'd',
            'led': 'led',
            'pin': 'pin',
            'switch': 'switch',
            'l': 'l',
            'mosfet': 'mosfet',
            'other': 'other',
            'm3螺丝': 'm3螺丝',
            'bjt': 'bjt',
            'testpoint': 'testpoint',
            'crystal': 'crystal',
            'buzzer': 'buzzer',
            'battary': 'battary',
            'fuse': 'fuse',
            'block': 'block',
            'amp': 'amp',
            'jumper': 'jumper',
            'motor': 'motor',
            'esd': 'esd',
            'dc': 'dc',
            'transfomer': 'transfomer',
            'ant': 'ant',
            'circle_header': 'circle_header',
            'opt': 'opt',
            'seg': 'seg',
            'Box': 'box',
            'BoxIc': 'box',
            'boxic': 'box',
            'box_ic': 'box',
            'ic': 'box',
            'IC': 'box',
            'R': 'r',
            'Resistor': 'r',
            'resistor': 'r',
            'C': 'c',
            'Capacitor': 'c',
            'capacitor': 'c',
            'L': 'l',
            'Inductor': 'l',
            'inductor': 'l',
            'D': 'd',
            'Diode': 'd',
            'diode': 'd',
            'MOSFET': 'mosfet',
            'Mosfet': 'mosfet',
            'BJT': 'bjt',
            'Bjt': 'bjt',
            'LED': 'led',
            'Led': 'led',
            'Fuse': 'fuse',
            'FUSE': 'fuse',
            'Battary': 'battary',
            'Battery': 'battary',
            'battery': 'battary',
            'Jumper': 'jumper',
            'Testpoint': 'testpoint',
            'TestPoint': 'testpoint',
            'Crystal': 'crystal',
            'Switch': 'switch',
            'Gnd': 'gnd',
            'GND': 'gnd',
            'V': 'v',
            'VCC': 'v',
            'Vcc': 'v',
            'vcc': 'v',
            'NET_input': 'net_input',
            'Net_input': 'net_input',
            'vbus_in': 'net_input',
            'NET_short': 'net_short',
            'Net_short': 'net_short',
            'NET_output': 'net_output',
            'Net_output': 'net_output',
            'vbus_out': 'net_output',
            'NET_bidirection': 'net_bidirection',
            'Net_bidirection': 'net_bidirection',
            'Other': 'other',
            'Pin': 'pin',
            'M3螺丝': 'm3螺丝',
            'transformer': 'transfomer',
            'BOX': 'box',
            'NET_INPUT': 'net_input',
            'Net_Input': 'net_input',
            'NET_SHORT': 'net_short',
            'Net_Short': 'net_short',
            'NET_OUTPUT': 'net_output',
            'Net_Output': 'net_output',
            'NET_BIDIRECTION': 'net_bidirection',
            'Net_Bidirection': 'net_bidirection',
            'PIN': 'pin',
            'SWITCH': 'switch',
            'OTHER': 'other',
            'TESTPOINT': 'testpoint',
            'CRYSTAL': 'crystal',
            'BUZZER': 'buzzer',
            'Buzzer': 'buzzer',
            'BATTARY': 'battary',
            'BLOCK': 'block',
            'Block': 'block',
            'AMP': 'amp',
            'Amp': 'amp',
            'JUMPER': 'jumper',
            'MOTOR': 'motor',
            'Motor': 'motor',
            'ESD': 'esd',
            'Esd': 'esd',
            'DC': 'dc',
            'Dc': 'dc',
            'TRANSFOMER': 'transfomer',
            'Transfomer': 'transfomer',
            'ANT': 'ant',
            'Ant': 'ant',
            'CIRCLE_HEADER': 'circle_header',
            'Circle_Header': 'circle_header',
            'OPT': 'opt',
            'Opt': 'opt',
            'SEG': 'seg',
            'Seg': 'seg',
        },
    },
    "pin": {
        "key_template": "pin_{number}",
        "fields": ["pinname", "point"],
        "point_encoding": "object_xy",
    },
    "net": {
        "key_template": "net_{index}",
        "fields": ["hyperGraph", "edges"],
        "hypergraph_pin_template": "{component_key}.{pin_number}",
        "hypergraph_separator": ",",
        "edge_key_template": "edge_{index}",
        "edge_encoding": "two_point_array",
    },
    "numbers": {
        "coordinate_type": "number",
        "coordinate_system": "original_bottom_left_xy",
        "source_coordinate_system": "image_top_left_xy",
    },
    "ordering": {"deterministic": True, "natural_sort_keys": True},
    "extra_fields": "forbid",
}


def _unwrap_profile(data: Mapping[str, Any]) -> dict[str, Any]:
    profile = data.get("official_schema", data)
    if not isinstance(profile, Mapping):
        raise OfficialSchemaError("official_schema must be an object")
    return deepcopy(dict(profile))


def validate_profile(profile: Mapping[str, Any]) -> None:
    if profile.get("top_level_keys") != ["components", "pins", "nets"]:
        raise OfficialSchemaError(
            "top_level_keys must be exactly ['components', 'pins', 'nets']"
        )
    component_profile = profile.get("component")
    if not isinstance(component_profile, Mapping):
        raise OfficialSchemaError("component must be an object")
    component_fields = component_profile.get("fields")
    if component_fields not in (
        ["type", "Name", "value", "bbox"],
        ["type", "name", "value", "bbox"],
    ):
        raise OfficialSchemaError(
            "component.fields must explicitly select either official-example 'Name' "
            "or official-table 'name' spelling"
        )
    expected_fields = {
        "pin": ["pinname", "point"],
        "net": ["hyperGraph", "edges"],
    }
    for section, fields in expected_fields.items():
        value = profile.get(section)
        if not isinstance(value, Mapping) or value.get("fields") != fields:
            raise OfficialSchemaError(f"{section}.fields must be exactly {fields!r}")
    aliases = component_profile.get("type_aliases", {})
    if not isinstance(aliases, Mapping) or any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or not value
        for key, value in aliases.items()
    ):
        raise OfficialSchemaError("component.type_aliases must map non-empty strings to strings")
    allowed_types = component_profile.get("allowed_types")
    if (
        not isinstance(allowed_types, list)
        or not allowed_types
        or any(not isinstance(item, str) or not item for item in allowed_types)
        or len(allowed_types) != len(set(allowed_types))
    ):
        raise OfficialSchemaError("component.allowed_types must be a non-empty unique string list")
    allowed_type_set = set(allowed_types)
    if any(value not in allowed_type_set for value in aliases.values()):
        raise OfficialSchemaError("component.type_aliases targets must belong to allowed_types")
    if component_profile.get("bbox_encoding") != "xyxy_list":
        raise OfficialSchemaError("only component.bbox_encoding=xyxy_list is supported")
    if profile.get("pin", {}).get("point_encoding") != "object_xy":
        raise OfficialSchemaError("only pin.point_encoding=object_xy is supported")
    if profile.get("net", {}).get("edge_encoding") != "two_point_array":
        raise OfficialSchemaError("only net.edge_encoding=two_point_array is supported")
    if profile.get("numbers") != {
        "coordinate_type": "number",
        "coordinate_system": "original_bottom_left_xy",
        "source_coordinate_system": "image_top_left_xy",
    }:
        raise OfficialSchemaError(
            "numbers must select finite JSON numbers transformed from image-top-left "
            "to original-bottom-left coordinates"
        )
    if profile.get("extra_fields") != "forbid":
        raise OfficialSchemaError("extra_fields must be 'forbid'")
    for section, field in (
        ("pin", "key_template"),
        ("net", "key_template"),
        ("net", "hypergraph_pin_template"),
        ("net", "edge_key_template"),
    ):
        value = profile.get(section, {}).get(field)
        if not isinstance(value, str) or not value:
            raise OfficialSchemaError(f"{section}.{field} must be a non-empty string")
    separator = profile.get("net", {}).get("hypergraph_separator", ",")
    if not isinstance(separator, str) or separator == "":
        raise OfficialSchemaError("net.hypergraph_separator must be a non-empty string")

    templates = {
        "pin.key_template": (
            profile["pin"]["key_template"],
            {"number": "1"},
            {"number": "2"},
        ),
        "net.key_template": (
            profile["net"]["key_template"],
            {"index": 1},
            {"index": 2},
        ),
        "net.edge_key_template": (
            profile["net"]["edge_key_template"],
            {"index": 1},
            {"index": 2},
        ),
    }
    for location, (template, first_args, second_args) in templates.items():
        try:
            first = template.format(**first_args)
            second = template.format(**second_args)
        except (KeyError, ValueError) as exc:
            raise OfficialSchemaError(f"invalid {location}: {exc}") from exc
        if first == second:
            raise OfficialSchemaError(f"{location} must encode its variable field")
    pin_ref_template = profile["net"]["hypergraph_pin_template"]
    try:
        base = pin_ref_template.format(component_key="R2", pin_number="1")
        other_component = pin_ref_template.format(component_key="R3", pin_number="1")
        other_pin = pin_ref_template.format(component_key="R2", pin_number="2")
    except (KeyError, ValueError) as exc:
        raise OfficialSchemaError(f"invalid net.hypergraph_pin_template: {exc}") from exc
    if base == other_component or base == other_pin:
        raise OfficialSchemaError(
            "net.hypergraph_pin_template must encode component_key and pin_number"
        )


def load_official_schema(
    source: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if source is None:
        profile = deepcopy(DEFAULT_OFFICIAL_SCHEMA)
    elif isinstance(source, (str, Path)):
        profile = _unwrap_profile(load_config(source))
    elif isinstance(source, Mapping):
        profile = _unwrap_profile(source)
    else:
        raise TypeError("official schema source must be a path, mapping, or None")
    validate_profile(profile)
    return profile


def _require_exact_keys(value: object, expected: list[str], location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OfficialSchemaError(f"{location} must be an object")
    actual = set(value)
    wanted = set(expected)
    if actual != wanted:
        missing = sorted(wanted - actual)
        extra = sorted(actual - wanted)
        raise OfficialSchemaError(
            f"{location} keys differ; missing={missing!r}, extra={extra!r}"
        )
    return value


def _require_number(value: object, location: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise OfficialSchemaError(f"{location} must be a finite number")


def _validate_point(value: object, location: str) -> None:
    point = _require_exact_keys(value, ["x", "y"], location)
    _require_number(point["x"], f"{location}.x")
    _require_number(point["y"], f"{location}.y")


def _template_pattern(template: str, fields: set[str], location: str) -> re.Pattern[str]:
    pattern = "^"
    seen: set[str] = set()
    try:
        parts = Formatter().parse(template)
        for literal, field_name, _format_spec, _conversion in parts:
            pattern += re.escape(literal)
            if field_name is not None:
                if field_name not in fields or field_name in seen:
                    raise OfficialSchemaError(
                        f"{location} has unsupported or repeated field {field_name!r}"
                    )
                pattern += f"(?P<{field_name}>.+?)"
                seen.add(field_name)
    except ValueError as exc:
        raise OfficialSchemaError(f"invalid {location}: {exc}") from exc
    if seen != fields:
        raise OfficialSchemaError(f"{location} must contain fields {sorted(fields)!r}")
    return re.compile(pattern + "$")


def validate_official_output(
    output: object,
    profile: str | Path | Mapping[str, Any] | None = None,
) -> None:
    """Guard the exact public submission shape, including extra fields."""

    schema = load_official_schema(profile)
    root = _require_exact_keys(output, schema["top_level_keys"], "output")
    components = root["components"]
    pins = root["pins"]
    nets = root["nets"]
    if not isinstance(components, Mapping):
        raise OfficialSchemaError("output.components must be an object")
    if not isinstance(pins, Mapping):
        raise OfficialSchemaError("output.pins must be an object")
    if not isinstance(nets, Mapping):
        raise OfficialSchemaError("output.nets must be an object")
    if set(pins) != set(components):
        raise OfficialSchemaError("output.pins component groups must exactly match components")
    for component_key in components:
        if not isinstance(component_key, str) or not component_key:
            raise OfficialSchemaError(
                "output.components keys must be non-empty strings"
            )
    for component_key in pins:
        if not isinstance(component_key, str) or not component_key:
            raise OfficialSchemaError("output.pins keys must be non-empty strings")
    for net_key in nets:
        if not isinstance(net_key, str) or not net_key:
            raise OfficialSchemaError("output.nets keys must be non-empty strings")

    pin_key_pattern = _template_pattern(
        schema["pin"]["key_template"], {"number"}, "pin.key_template"
    )
    pin_ref_pattern = _template_pattern(
        schema["net"]["hypergraph_pin_template"],
        {"component_key", "pin_number"},
        "net.hypergraph_pin_template",
    )

    for key, raw_component in components.items():
        component = _require_exact_keys(
            raw_component, schema["component"]["fields"], f"components.{key}"
        )
        name_field = schema["component"]["fields"][1]
        if not isinstance(component["type"], str) or not component["type"]:
            raise OfficialSchemaError(f"components.{key}.type must be a non-empty string")
        if component["type"] not in schema["component"]["allowed_types"]:
            raise OfficialSchemaError(
                f"components.{key}.type {component['type']!r} is not in component.allowed_types"
            )
        for text_field in (name_field, "value"):
            if component[text_field] is not None and not isinstance(component[text_field], str):
                raise OfficialSchemaError(
                    f"components.{key}.{text_field} must be a string or null"
                )
        bbox = component["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise OfficialSchemaError(f"components.{key}.bbox must contain four coordinates")
        for index, coordinate in enumerate(bbox):
            _require_number(coordinate, f"components.{key}.bbox[{index}]")
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise OfficialSchemaError(
                f"components.{key}.bbox must have positive area"
            )

    available_pin_refs: set[str] = set()
    for component_key, raw_group in pins.items():
        if not isinstance(raw_group, Mapping):
            raise OfficialSchemaError(f"pins.{component_key} must be an object")
        for pin_key, raw_pin in raw_group.items():
            if not isinstance(pin_key, str):
                raise OfficialSchemaError(
                    f"pins.{component_key} keys must be strings"
                )
            match = pin_key_pattern.fullmatch(pin_key)
            if match is None:
                raise OfficialSchemaError(
                    f"pins.{component_key}.{pin_key} does not match pin.key_template"
                )
            # Resolve against the actual component/pin groups. Splitting a
            # reference at its first dot misreads multi-unit keys (U8.1.3).
            available_pin_refs.add(schema["net"]["hypergraph_pin_template"].format(
                component_key=component_key, pin_number=match.group("number")
            ))
            pin = _require_exact_keys(
                raw_pin, schema["pin"]["fields"], f"pins.{component_key}.{pin_key}"
            )
            if not isinstance(pin["pinname"], str):
                raise OfficialSchemaError(
                    f"pins.{component_key}.{pin_key}.pinname must be a string"
                )
            _validate_point(pin["point"], f"pins.{component_key}.{pin_key}.point")

    separator = schema["net"].get("hypergraph_separator", ",")
    for net_key, raw_net in nets.items():
        net = _require_exact_keys(raw_net, schema["net"]["fields"], f"nets.{net_key}")
        if not isinstance(net["hyperGraph"], str):
            raise OfficialSchemaError(f"nets.{net_key}.hyperGraph must be a string")
        hypergraph = net["hyperGraph"]
        if not hypergraph.startswith("(") or not hypergraph.endswith(")"):
            raise OfficialSchemaError(f"nets.{net_key}.hyperGraph must be parenthesized")
        encoded_refs = hypergraph[1:-1].split(separator) if hypergraph[1:-1] else []
        if len(encoded_refs) != len(set(encoded_refs)):
            raise OfficialSchemaError(f"nets.{net_key}.hyperGraph contains duplicate pin refs")
        for encoded_ref in encoded_refs:
            match = pin_ref_pattern.fullmatch(encoded_ref)
            if match is None:
                raise OfficialSchemaError(
                    f"nets.{net_key}.hyperGraph ref {encoded_ref!r} does not match profile"
                )
            if encoded_ref not in available_pin_refs:
                raise OfficialSchemaError(
                    f"nets.{net_key}.hyperGraph references unknown pin {encoded_ref!r}"
                )
        edges = net["edges"]
        if not isinstance(edges, Mapping):
            raise OfficialSchemaError(f"nets.{net_key}.edges must be an object")
        for edge_key, edge in edges.items():
            if not isinstance(edge_key, str) or not edge_key:
                raise OfficialSchemaError(
                    f"nets.{net_key}.edges keys must be non-empty strings"
                )
            if not isinstance(edge, list) or len(edge) != 2:
                raise OfficialSchemaError(
                    f"nets.{net_key}.edges.{edge_key} must contain exactly two points"
                )
            _validate_point(edge[0], f"nets.{net_key}.edges.{edge_key}[0]")
            _validate_point(edge[1], f"nets.{net_key}.edges.{edge_key}[1]")
