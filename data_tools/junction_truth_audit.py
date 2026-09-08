from __future__ import annotations

"""Audit ambiguous junction candidates without manufacturing ground truth.

The Dataset300-v2 derivation deliberately marks every point shared by two raw
edge skeletons as ``UNKNOWN``.  Shared pixels are only *candidates*: duplicate
or collinear edges, table grids and real schematic crossings all satisfy that
test.  This tool combines four auditable evidence channels and emits review
proposals.  It never edits the frozen materialization and never promotes a
proposal to training ground truth.

Dense ``coordinates`` arrays are lexically removed before JSON decoding by the
same fail-closed reader used by the wire/junction materializer.  Only sparse
``keyPointPaths``, ``keyPoints`` and ``connect`` metadata are inspected.
"""

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict, deque
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

from ._bootstrap import REPOSITORY_ROOT, default_dataset_root, ensure_source_path

ensure_source_path()

from pcbparse.training.wire_junction_materialize import (  # noqa: E402
    _load_annotation_without_dense_coordinates,
)


AUDIT_SCHEMA_VERSION = 1
AUDIT_VERSION = "junction-truth-audit-v1"
_POINT_TOLERANCE = 2.0
_PARALLEL_ANGLE_DEGREES = 25.0
_CROSS_ANGLE_DEGREES = 60.0
_IMAGE_THRESHOLD = 35.0


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _points(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []
    output: list[tuple[float, float]] = []
    for raw in value:
        if (
            isinstance(raw, list)
            and len(raw) >= 2
            and isinstance(raw[0], (int, float))
            and not isinstance(raw[0], bool)
            and isinstance(raw[1], (int, float))
            and not isinstance(raw[1], bool)
            and math.isfinite(float(raw[0]))
            and math.isfinite(float(raw[1]))
        ):
            output.append((float(raw[0]), float(raw[1])))
    return output


def _geometry_points(edge: Mapping[str, Any], field: str) -> list[tuple[float, float]]:
    key_points = edge.get("keyPoints")
    if not isinstance(key_points, Mapping):
        return []
    geometry = key_points.get(field)
    if not isinstance(geometry, Mapping):
        return []
    output: list[tuple[float, float]] = []
    for raw in geometry.values():
        output.extend(_points(raw))
    return output


def _skeleton(edge: Mapping[str, Any]) -> list[tuple[float, float]]:
    return _geometry_points(edge, "skeleton")


def _paths(edge: Mapping[str, Any]) -> list[list[tuple[float, float]]]:
    raw_paths = edge.get("keyPointPaths")
    if not isinstance(raw_paths, Mapping):
        return []
    output: list[list[tuple[float, float]]] = []
    for geometry_paths in raw_paths.values():
        if not isinstance(geometry_paths, list):
            continue
        for raw_path in geometry_paths:
            path = _points(raw_path)
            if path:
                output.append(path)
    return output


def _distance_to_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float, tuple[float, float]]:
    vx, vy = end[0] - start[0], end[1] - start[1]
    denominator = vx * vx + vy * vy
    if denominator <= 1.0e-12:
        return math.dist(point, start), 0.0, (0.0, 0.0)
    t = ((point[0] - start[0]) * vx + (point[1] - start[1]) * vy) / denominator
    clipped = min(1.0, max(0.0, t))
    closest = (start[0] + clipped * vx, start[1] + clipped * vy)
    length = math.sqrt(denominator)
    return math.dist(point, closest), clipped, (vx / length, vy / length)


def nearest_sparse_segment(
    edge: Mapping[str, Any], center: tuple[float, float]
) -> dict[str, Any]:
    """Return the closest non-dense keyPointPaths segment to ``center``."""

    best: tuple[float, float, tuple[float, float], int, int] | None = None
    for path_index, path in enumerate(_paths(edge)):
        if len(path) == 1:
            distance = math.dist(center, path[0])
            candidate = (distance, 0.0, (0.0, 0.0), path_index, 0)
            if best is None or candidate[0] < best[0]:
                best = candidate
            continue
        for segment_index, (start, end) in enumerate(zip(path, path[1:])):
            distance, projection, direction = _distance_to_segment(center, start, end)
            candidate = (
                distance,
                projection,
                direction,
                path_index,
                segment_index,
            )
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        return {
            "available": False,
            "distance_px": None,
            "direction": None,
            "projection": None,
            "near_endpoint": None,
        }
    distance, projection, direction, path_index, segment_index = best
    return {
        "available": True,
        "distance_px": round(distance, 6),
        "direction": [round(direction[0], 8), round(direction[1], 8)],
        "projection": round(projection, 8),
        "near_endpoint": bool(projection <= 0.05 or projection >= 0.95),
        "path_index": path_index,
        "segment_index": segment_index,
        "supports_center": bool(distance <= _POINT_TOLERANCE),
    }


def _axial_angle(first: Sequence[float], second: Sequence[float]) -> float:
    dot = abs(float(first[0]) * float(second[0]) + float(first[1]) * float(second[1]))
    return math.degrees(math.acos(min(1.0, max(0.0, dot))))


def _ordered_skeleton_direction(
    edge: Mapping[str, Any], center: tuple[float, float]
) -> list[float] | None:
    points = _skeleton(edge)
    indexes = [index for index, value in enumerate(points) if value == center]
    vectors: list[tuple[float, float]] = []
    for index in indexes:
        for other_index in (index - 2, index - 1, index + 1, index + 2):
            if 0 <= other_index < len(points):
                dx = points[other_index][0] - center[0]
                dy = points[other_index][1] - center[1]
                if dx != 0.0 or dy != 0.0:
                    vectors.append((dx, dy))
    if not vectors:
        return None
    cosine = sum(
        (dx * dx - dy * dy) / (dx * dx + dy * dy) for dx, dy in vectors
    )
    sine = sum(2.0 * dx * dy / (dx * dx + dy * dy) for dx, dy in vectors)
    angle = 0.5 * math.atan2(sine, cosine)
    return [round(math.cos(angle), 8), round(math.sin(angle), 8)]


def _component_size(points: set[tuple[float, float]], center: tuple[float, float]) -> int:
    if center not in points:
        return 0
    pending = deque([center])
    seen = {center}
    while pending:
        x, y = pending.popleft()
        for dx in (-1.0, 0.0, 1.0):
            for dy in (-1.0, 0.0, 1.0):
                neighbor = (x + dx, y + dy)
                if neighbor in points and neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
    return len(seen)


def analyze_geometry(
    edges: Mapping[str, Mapping[str, Any]],
    edge_keys: Sequence[str],
    center: tuple[float, float],
) -> dict[str, Any]:
    segment_evidence = [nearest_sparse_segment(edges[key], center) for key in edge_keys]
    directions: list[list[float]] = []
    direction_sources: list[str] = []
    for key, segment in zip(edge_keys, segment_evidence):
        direction = segment.get("direction")
        if segment.get("supports_center") and direction != [0.0, 0.0]:
            directions.append(direction)
            direction_sources.append("keyPointPaths")
        else:
            fallback = _ordered_skeleton_direction(edges[key], center)
            if fallback is not None:
                directions.append(fallback)
                direction_sources.append("keyPoints/skeleton")
    angles = [
        _axial_angle(directions[first], directions[second])
        for first in range(len(directions))
        for second in range(first + 1, len(directions))
    ]
    skeleton_sets = [set(_skeleton(edges[key])) for key in edge_keys]
    shared = set.intersection(*skeleton_sets) if skeleton_sets else set()
    shared_component = _component_size(shared, center)
    identical_sets = bool(skeleton_sets) and all(
        value == skeleton_sets[0] for value in skeleton_sets[1:]
    )
    maximum_angle = max(angles) if angles else None
    if maximum_angle is None:
        shape = "orientation_unavailable"
    elif maximum_angle < _PARALLEL_ANGLE_DEGREES:
        shape = "collinear_overlap"
    elif maximum_angle >= _CROSS_ANGLE_DEGREES:
        shape = "orthogonal_intersection"
    else:
        shape = "oblique_intersection"
    endpoint_flags = {
        field: any(center in set(_geometry_points(edges[key], field)) for key in edge_keys)
        for field in ("endPoints", "divergePoints", "turnPoints")
    }
    near_endpoint_edges = sum(
        bool(item.get("supports_center") and item.get("near_endpoint"))
        for item in segment_evidence
    )
    if near_endpoint_edges >= 2:
        path_relation = "endpoint_endpoint"
    elif near_endpoint_edges == 1:
        path_relation = "endpoint_interior"
    elif all(item.get("supports_center") for item in segment_evidence):
        path_relation = "interior_interior"
    else:
        path_relation = "partly_unsupported"
    return {
        "shape": shape,
        "owner_edge_count": len(edge_keys),
        "same_raw_edge": len(set(edge_keys)) != len(edge_keys),
        "different_raw_edges": len(set(edge_keys)) == len(edge_keys),
        "identical_skeleton_sets": identical_sets,
        "shared_skeleton_pixels": len(shared),
        "shared_component_pixels": shared_component,
        "maximum_pair_angle_degrees": None
        if maximum_angle is None
        else round(maximum_angle, 4),
        "direction_sources": direction_sources,
        "sparse_path_support_count": sum(
            bool(item.get("supports_center")) for item in segment_evidence
        ),
        "sparse_path_relation": path_relation,
        "raw_keypoint_at_center": endpoint_flags,
        "segments": segment_evidence,
    }


def _connect_entries(edge: Mapping[str, Any]) -> list[dict[str, Any]]:
    connect = edge.get("connect")
    if not isinstance(connect, Mapping):
        return []
    output: list[dict[str, Any]] = []
    for key, raw in connect.items():
        if not isinstance(raw, Mapping):
            continue
        location = raw.get("location")
        point = _points([location])
        output.append(
            {
                "pin_id": str(raw.get("pinId") or key),
                "component_id": None
                if raw.get("cpntId") is None
                else str(raw.get("cpntId")),
                "location": None if not point else [point[0][0], point[0][1]],
            }
        )
    return output


def _normal_netlabel(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.strip().split()).casefold()
    if not normalized or normalized in {"none", "null", "unknown"}:
        return None
    return normalized


def parse_edif_pin_nets(text: str) -> dict[str, set[str]]:
    """Parse only ``net``/``joined`` pin membership from generated EDIF."""

    output: dict[str, set[str]] = defaultdict(set)
    start_pattern = re.compile(r"(?m)^\s*\(net\s+")
    for ordinal, match in enumerate(start_pattern.finditer(text)):
        start = text.find("(net", match.start())
        depth = 0
        index = start
        quoted = False
        escaped = False
        while index < len(text):
            char = text[index]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    index += 1
                    break
            index += 1
        block = text[start:index]
        header = block.splitlines()[0].strip() if block else ""
        net_id = f"{ordinal}:{header}"
        for pin_id in re.findall(r"\(portRef\s+PORT_(P_[^\s\)]+)", block):
            output[pin_id].add(net_id)
    return dict(output)


def parse_spice_edge_nodes(text: str) -> dict[str, str]:
    output: dict[str, str] = {}
    pattern = re.compile(r"^\*\s+(E_\S+)\s*=\s*([^\s:（(]+)", re.MULTILINE)
    for edge, node in pattern.findall(text):
        output[edge] = node.casefold()
    return output


def _set_relation(values: Sequence[set[str]]) -> str:
    if not values or any(not value for value in values):
        return "unknown"
    if set.intersection(*values):
        return "same"
    if all(values[first].isdisjoint(values[second]) for first in range(len(values)) for second in range(first + 1, len(values))):
        return "different"
    return "mixed"


def analyze_topology(
    edges: Mapping[str, Mapping[str, Any]],
    edge_keys: Sequence[str],
    edif_pin_nets: Mapping[str, set[str]],
    spice_edge_nodes: Mapping[str, str],
) -> dict[str, Any]:
    connects = [_connect_entries(edges[key]) for key in edge_keys]
    pin_sets = [{item["pin_id"] for item in values} for values in connects]
    locations = [
        {tuple(item["location"]) for item in values if item["location"] is not None}
        for values in connects
    ]
    components = [
        {item["component_id"] for item in values if item["component_id"] is not None}
        for values in connects
    ]
    labels = [_normal_netlabel(edges[key].get("netlabel")) for key in edge_keys]
    shared_pin = any(
        pin_sets[first] & pin_sets[second]
        for first in range(len(pin_sets))
        for second in range(first + 1, len(pin_sets))
    )
    shared_location = any(
        locations[first] & locations[second]
        for first in range(len(locations))
        for second in range(first + 1, len(locations))
    )
    shared_component = any(
        components[first] & components[second]
        for first in range(len(components))
        for second in range(first + 1, len(components))
    )
    same_label = bool(labels) and all(label is not None for label in labels) and len(set(labels)) == 1
    distinct_labels = (
        bool(labels)
        and all(label is not None for label in labels)
        and len(set(labels)) == len(labels)
    )
    edif_values = [
        set().union(*(edif_pin_nets.get(pin, set()) for pin in pin_set))
        for pin_set in pin_sets
    ]
    edif_relation = _set_relation(edif_values)
    spice_values = [
        set() if spice_edge_nodes.get(key) is None else {spice_edge_nodes[key]}
        for key in edge_keys
    ]
    spice_relation = _set_relation(spice_values)
    positive_raw = shared_pin or shared_location or same_label
    negative_raw = distinct_labels and not shared_pin and not shared_location
    positive_channels = sum(
        (positive_raw, edif_relation == "same", spice_relation == "same")
    )
    negative_channels = sum(
        (negative_raw, edif_relation == "different", spice_relation == "different")
    )
    if positive_channels and negative_channels:
        consensus = "conflict"
    elif positive_channels >= 2:
        consensus = "connected"
    elif negative_channels >= 2:
        consensus = "non_connected"
    else:
        consensus = "unknown"
    return {
        "raw_connect": {
            "shared_pin_id": shared_pin,
            "shared_pin_location": shared_location,
            "shared_component_id": shared_component,
            "all_same_nonempty_netlabel": same_label,
            "all_distinct_nonempty_netlabels": distinct_labels,
            "edge_pin_counts": [len(value) for value in pin_sets],
        },
        "edif_pin_net_relation": edif_relation,
        "spice_edge_node_relation": spice_relation,
        "consensus": consensus,
        "positive_channel_count": positive_channels,
        "negative_channel_count": negative_channels,
        "provenance_independence_warning": (
            "CVJsonStd, EDS and SPICE are co-packaged/generated views and are not "
            "statistically independent truth sources"
        ),
    }


def _background_color(image: Image.Image) -> tuple[float, float, float]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    border = [rgb.getpixel((x, 0)) for x in range(width)]
    border += [rgb.getpixel((x, height - 1)) for x in range(width)]
    border += [rgb.getpixel((0, y)) for y in range(height)]
    border += [rgb.getpixel((width - 1, y)) for y in range(height)]
    buckets = Counter(tuple(channel // 8 for channel in value) for value in border)
    bucket = buckets.most_common(1)[0][0]
    selected = [
        value for value in border if tuple(channel // 8 for channel in value) == bucket
    ]
    return tuple(float(median(value[index] for value in selected)) for index in range(3))


def _foreground_mask(image: Image.Image) -> tuple[list[list[bool]], tuple[float, float, float]]:
    rgb = image.convert("RGB")
    background = _background_color(rgb)
    width, height = rgb.size
    mask: list[list[bool]] = []
    for y in range(height):
        row: list[bool] = []
        for x in range(width):
            pixel = rgb.getpixel((x, y))
            distance = math.sqrt(
                sum((float(pixel[index]) - background[index]) ** 2 for index in range(3))
            )
            row.append(distance > _IMAGE_THRESHOLD)
        mask.append(row)
    return mask, background


def _sample_mask(
    mask: Sequence[Sequence[bool]], x: float, y: float, radius: int = 1
) -> float:
    height = len(mask)
    width = len(mask[0]) if height else 0
    cx, cy = int(round(x)), int(round(y))
    values = [
        mask[row][column]
        for row in range(max(0, cy - radius), min(height, cy + radius + 1))
        for column in range(max(0, cx - radius), min(width, cx + radius + 1))
    ]
    return 0.0 if not values else sum(values) / len(values)


def analyze_visual_symbol(
    image: Image.Image, directions: Sequence[Sequence[float]] | None = None
) -> dict[str, Any]:
    """Return conservative dot/gap/plain features for a centered review patch."""

    mask, background = _foreground_mask(image)
    width, height = image.size
    cx, cy = width / 2.0, height / 2.0
    axes: list[tuple[float, float]] = []
    if directions:
        for raw in directions:
            if len(raw) >= 2:
                norm = math.hypot(float(raw[0]), float(raw[1]))
                if norm > 1.0e-9:
                    value = (float(raw[0]) / norm, float(raw[1]) / norm)
                    if all(_axial_angle(value, present) >= 25.0 for present in axes):
                        axes.append(value)
            if len(axes) == 2:
                break
    if len(axes) < 2:
        axes = [(1.0, 0.0), (0.0, 1.0)]
    arm_scores: list[float] = []
    for dx, dy in axes[:2]:
        for sign in (-1.0, 1.0):
            arm_scores.append(
                sum(
                    _sample_mask(mask, cx + sign * dx * distance, cy + sign * dy * distance)
                    for distance in (8.0, 10.0, 12.0)
                )
                / 3.0
            )
    # A one-pixel anti-aliased stroke occupies roughly one third of the
    # three-pixel sampling strip; keep the threshold just below that floor.
    four_arms = min(arm_scores) >= 0.30
    direct_scores: list[float] = []
    line_widths: list[float] = []
    for dx, dy in axes[:2]:
        direct_scores.append(
            sum(_sample_mask(mask, cx + dx * value, cy + dy * value, 0) for value in range(-4, 5))
            / 9.0
        )
        perpendicular = (-dy, dx)
        widths: list[int] = []
        for sign in (-1.0, 1.0):
            for distance in (8.0, 10.0, 12.0):
                present = 0
                for offset in range(-6, 7):
                    x = cx + sign * dx * distance + perpendicular[0] * offset
                    y = cy + sign * dy * distance + perpendicular[1] * offset
                    if _sample_mask(mask, x, y, 0) >= 0.5:
                        present += 1
                widths.append(present)
        line_widths.append(float(median(widths)))
    central_radius = 6
    central_foreground = 0
    central_expected = 0
    central_excess = 0
    for oy in range(-central_radius, central_radius + 1):
        for ox in range(-central_radius, central_radius + 1):
            if ox * ox + oy * oy > central_radius * central_radius:
                continue
            foreground = _sample_mask(mask, cx + ox, cy + oy, 0) >= 0.5
            expected = False
            for (dx, dy), line_width in zip(axes[:2], line_widths):
                distance = abs(ox * (-dy) + oy * dx)
                expected = expected or distance <= max(0.75, line_width / 2.0 + 0.5)
            central_foreground += int(foreground)
            central_expected += int(expected)
            central_excess += int(foreground and not expected)
    center_score = _sample_mask(mask, cx, cy, 1)
    explicit_dot = (
        four_arms
        and central_excess >= 12
        and central_foreground >= max(24, int(central_expected * 1.15))
    )
    bridge_or_gap = (
        four_arms
        and not explicit_dot
        and min(direct_scores) <= 0.34
        and max(direct_scores) >= 0.67
    )
    if explicit_dot:
        symbol = "filled_dot"
    elif bridge_or_gap:
        symbol = "bridge_or_gap"
    elif four_arms:
        symbol = "plain_intersection"
    elif max(arm_scores, default=0.0) < 0.30:
        symbol = "no_local_stroke_support"
    else:
        symbol = "complex_or_ambiguous"
    return {
        "symbol": symbol,
        "background_rgb": [round(value, 3) for value in background],
        "foreground_threshold_rgb_distance": _IMAGE_THRESHOLD,
        "arm_scores": [round(value, 4) for value in arm_scores],
        "four_arm_support": four_arms,
        "direct_channel_scores": [round(value, 4) for value in direct_scores],
        "estimated_line_widths_px": [round(value, 3) for value in line_widths],
        "center_foreground_score": round(center_score, 4),
        "central_foreground_pixels": central_foreground,
        "central_expected_strip_pixels": central_expected,
        "central_excess_pixels": central_excess,
        "limitations": (
            "Raster evidence is heuristic: antialiasing, table grids, text and "
            "domain-specific drawing conventions can mimic dots or gaps"
        ),
    }


def review_proposal(
    candidate_types: Sequence[str],
    geometry: Mapping[str, Any],
    topology: Mapping[str, Any],
    visual: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a fail-closed review suggestion, never a GT label."""

    raw_points = geometry.get("raw_keypoint_at_center", {})
    shape = geometry.get("shape")
    proposed_label: str | None = None
    rule_id: str | None = None
    confidence = "insufficient"
    rationale: list[str] = []
    if shape == "collinear_overlap":
        if raw_points.get("divergePoints"):
            proposed_label, rule_id = "t_junction", "R_RESTORE_DIVERGE"
        elif raw_points.get("endpoint") or raw_points.get("endPoints"):
            proposed_label, rule_id = "endpoint", "R_RESTORE_ENDPOINT"
        elif raw_points.get("turnPoints"):
            proposed_label, rule_id = "bend", "R_RESTORE_TURN"
        else:
            proposed_label, rule_id = "noise", "R_COLLINEAR_NOT_A_CROSS"
        confidence = "high"
        rationale.append("different edge skeletons overlap collinearly, so this pixel is not a crossing")
    elif geometry.get("identical_skeleton_sets"):
        proposed_label, rule_id, confidence = "noise", "R_DUPLICATE_SKELETON", "high"
        rationale.append("all owning edges have identical sparse skeleton sets")
    elif (
        shape == "orthogonal_intersection"
        and topology.get("consensus") == "connected"
        and visual.get("symbol") == "filled_dot"
        and topology.get("negative_channel_count") == 0
    ):
        proposed_label, rule_id, confidence = (
            "connected_cross",
            "R_CONNECTED_TOPOLOGY_PLUS_DOT",
            "medium",
        )
        rationale.append(
            "topology channels agree and the raster has an explicit dot, but the "
            "topology files share provenance and the raster rule is not validated GT"
        )
    elif (
        shape == "orthogonal_intersection"
        and topology.get("consensus") == "non_connected"
        and visual.get("symbol") == "bridge_or_gap"
        and topology.get("positive_channel_count") == 0
    ):
        proposed_label, rule_id, confidence = (
            "non_connected_cross",
            "R_DISTINCT_TOPOLOGY_PLUS_GAP",
            "medium",
        )
        rationale.append(
            "topology channels are distinct and the raster has a bridge/gap cue, but "
            "datasheet grids and one-pixel alignment can produce the same signature"
        )
    else:
        rationale.append("the available channels do not satisfy a conservative agreement rule")
    auto_accept = confidence == "high" and proposed_label is not None
    return {
        "proposed_label": proposed_label,
        "confidence": confidence,
        "rule_id": rule_id,
        "eligible_for_review_auto_accept": auto_accept,
        "accepted_as_ground_truth": False,
        "may_enter_training_without_human_review": False,
        "rationale": rationale,
        "candidate_types": list(candidate_types),
    }


def _edge_keys(record: Mapping[str, Any]) -> list[str]:
    refs = record.get("source", {}).get("source_refs", [])
    keys = sorted(
        {
            str(ref.get("edge_key"))
            for ref in refs
            if isinstance(ref, Mapping) and ref.get("kind") == "cross" and ref.get("edge_key")
        }
    )
    return keys


def _directions_from_geometry(geometry: Mapping[str, Any]) -> list[list[float]]:
    output: list[list[float]] = []
    for segment in geometry.get("segments", []):
        if segment.get("supports_center") and segment.get("direction") not in (None, [0.0, 0.0]):
            output.append(segment["direction"])
    return output


def _connected_clusters(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], set[tuple[int, int]]] = defaultdict(set)
    ids: dict[tuple[tuple[str, ...], tuple[int, int]], str] = {}
    for record in records:
        center = tuple(int(value) for value in record["center"])
        keys = tuple(_edge_keys(record))
        grouped[keys].add(center)
        ids[(keys, center)] = str(record["record_id"])
    output: list[dict[str, Any]] = []
    for keys, all_points in sorted(grouped.items()):
        remaining = set(all_points)
        while remaining:
            seed = min(remaining)
            remaining.remove(seed)
            pending = [seed]
            component = [seed]
            while pending:
                x, y = pending.pop()
                neighbors = {
                    value
                    for value in remaining
                    if abs(value[0] - x) <= 1 and abs(value[1] - y) <= 1
                }
                remaining.difference_update(neighbors)
                pending.extend(neighbors)
                component.extend(neighbors)
            record_ids = sorted(ids[(keys, point)] for point in component)
            output.append(
                {
                    "edge_keys": list(keys),
                    "candidate_pixel_count": len(component),
                    "bbox": {
                        "x1": min(point[0] for point in component),
                        "y1": min(point[1] for point in component),
                        "x2": max(point[0] for point in component),
                        "y2": max(point[1] for point in component),
                    },
                    "record_ids": record_ids,
                }
            )
    return output


def _increment_distribution(
    output: dict[str, dict[str, int]], key: str, value: str
) -> None:
    output.setdefault(key, {})[value] = output.setdefault(key, {}).get(value, 0) + 1


def audit_junction_truth(
    *,
    dataset_root: Path,
    source_manifest: Path,
    split_manifest: Path,
    materialized_root: Path,
) -> dict[str, Any]:
    latest_path = materialized_root / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    run_dir = materialized_root / str(latest["run_dir"])
    annotations_path = run_dir / "junction_annotations.jsonl"
    rows = [row for row in _read_jsonl(annotations_path) if row.get("label") == "UNKNOWN"]
    manifest_rows = {row["sample_id"]: row for row in _read_jsonl(source_manifest)}
    split_rows = {row["sample_id"]: row for row in _read_jsonl(split_manifest)}
    duplicate_group_splits: dict[str, set[str]] = defaultdict(set)
    for sample_id, manifest in manifest_rows.items():
        duplicate_group_splits[str(manifest["duplicate_group"])].add(
            str(split_rows[sample_id]["split"])
        )
    leaking_duplicate_groups = sorted(
        group for group, splits in duplicate_group_splits.items() if len(splits) > 1
    )
    if not rows:
        raise ValueError("materialized junction annotations contain no UNKNOWN records")
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sample[str(row["sample_id"])].append(row)

    candidates: list[dict[str, Any]] = []
    clusters: list[dict[str, Any]] = []
    dense_totals = Counter()
    file_hashes: dict[str, dict[str, str]] = {}
    for sample_id, sample_rows in sorted(by_sample.items()):
        manifest = manifest_rows[sample_id]
        split = str(split_rows[sample_id]["split"])
        domain = str(manifest.get("source_class") or "unknown")
        files = manifest["files"]
        json_path = (dataset_root / files["json"]["path"]).resolve()
        image_path = (dataset_root / files["image"]["path"]).resolve()
        eds_path = (dataset_root / files["eds"]["path"]).resolve()
        net_path = (dataset_root / files["net"]["path"]).resolve()
        annotation, dense_stats = _load_annotation_without_dense_coordinates(json_path)
        dense_totals.update(dense_stats)
        edges = annotation.get("edges")
        if not isinstance(edges, Mapping):
            raise ValueError(f"sample {sample_id} has no edge map")
        edif_pin_nets = parse_edif_pin_nets(eds_path.read_text(encoding="utf-8"))
        spice_edge_nodes = parse_spice_edge_nodes(net_path.read_text(encoding="utf-8"))
        file_hashes[sample_id] = {
            "json": _sha256_file(json_path),
            "image": _sha256_file(image_path),
            "eds": _sha256_file(eds_path),
            "spice": _sha256_file(net_path),
        }
        clusters.extend(
            {"sample_id": sample_id, "split": split, "source_domain": domain, **item}
            for item in _connected_clusters(sample_rows)
        )
        for row in sorted(sample_rows, key=lambda value: value["record_id"]):
            edge_keys = _edge_keys(row)
            if len(edge_keys) < 2 or any(key not in edges for key in edge_keys):
                raise ValueError(f"{row['record_id']}: invalid cross edge references")
            center = (float(row["center"][0]), float(row["center"][1]))
            geometry = analyze_geometry(edges, edge_keys, center)
            topology = analyze_topology(edges, edge_keys, edif_pin_nets, spice_edge_nodes)
            patch_path = run_dir / row["patch"]["path"]
            with Image.open(patch_path) as patch:
                visual = analyze_visual_symbol(
                    patch, directions=_directions_from_geometry(geometry)
                )
            proposal = review_proposal(
                row.get("candidate_types", []), geometry, topology, visual
            )
            candidates.append(
                {
                    "record_id": row["record_id"],
                    "sample_id": sample_id,
                    "split": split,
                    "source_domain": domain,
                    "center": row["center"],
                    "edge_keys": edge_keys,
                    "patch_path": row["patch"]["path"],
                    "geometry": geometry,
                    "topology": topology,
                    "visual": visual,
                    "review_proposal": proposal,
                    "leakage_guard": (
                        "never_use_for_training"
                        if split in {"val", "test"}
                        else "human_review_required_before_training"
                    ),
                }
            )

    distributions: dict[str, dict[str, int]] = {}
    by_split_domain: dict[str, dict[str, Any]] = {}
    for record in candidates:
        geometry = record["geometry"]
        topology = record["topology"]
        visual = record["visual"]
        proposal = record["review_proposal"]
        _increment_distribution(distributions, "geometry_shape", geometry["shape"])
        _increment_distribution(
            distributions, "owner_edge_count", str(geometry["owner_edge_count"])
        )
        _increment_distribution(
            distributions, "edif_relation", topology["edif_pin_net_relation"]
        )
        _increment_distribution(
            distributions, "spice_relation", topology["spice_edge_node_relation"]
        )
        _increment_distribution(distributions, "topology_consensus", topology["consensus"])
        _increment_distribution(distributions, "visual_symbol", visual["symbol"])
        _increment_distribution(
            distributions,
            "proposal_label",
            str(proposal["proposed_label"] or "UNKNOWN"),
        )
        raw = topology["raw_connect"]
        for name in (
            "shared_pin_id",
            "shared_pin_location",
            "shared_component_id",
            "all_same_nonempty_netlabel",
            "all_distinct_nonempty_netlabels",
        ):
            _increment_distribution(distributions, name, str(bool(raw[name])).lower())
        bucket_key = f"{record['split']}|{record['source_domain']}"
        bucket = by_split_domain.setdefault(
            bucket_key,
            {
                "split": record["split"],
                "source_domain": record["source_domain"],
                "records": 0,
                "review_auto_accept_proposals": 0,
                "proposal_labels": {},
                "topology_conflicts": 0,
                "topology_unknown": 0,
            },
        )
        bucket["records"] += 1
        bucket["review_auto_accept_proposals"] += int(
            proposal["eligible_for_review_auto_accept"]
        )
        label = str(proposal["proposed_label"] or "UNKNOWN")
        bucket["proposal_labels"][label] = bucket["proposal_labels"].get(label, 0) + 1
        bucket["topology_conflicts"] += int(topology["consensus"] == "conflict")
        bucket["topology_unknown"] += int(topology["consensus"] == "unknown")

    auto_accept = [
        value
        for value in candidates
        if value["review_proposal"]["eligible_for_review_auto_accept"]
    ]
    auto_accept_train = [value for value in auto_accept if value["split"] == "train"]
    cross_suggestions = Counter(
        value["review_proposal"]["proposed_label"]
        for value in candidates
        if value["review_proposal"]["proposed_label"]
        in {"connected_cross", "non_connected_cross"}
    )
    topology_conflict = sum(
        value["topology"]["consensus"] == "conflict" for value in candidates
    )
    topology_unknown = sum(
        value["topology"]["consensus"] == "unknown" for value in candidates
    )
    same_edge = sum(value["geometry"]["same_raw_edge"] for value in candidates)
    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_version": AUDIT_VERSION,
        "kind": "junction_truth_audit",
        "scope": {
            "unknown_cross_records": len(candidates),
            "samples_with_unknown_crosses": len(by_sample),
            "materialized_run_id": latest.get("run_id"),
            "materialized_run_dir": str(run_dir.resolve()),
            "dense_coordinates_loaded": False,
            "dense_coordinate_fields_stripped": dense_totals["fields"],
            "dense_coordinate_source_bytes_skipped": dense_totals["bytes"],
            "same_edge_self_intersections_in_candidate_derivation": False,
        },
        "lineage": {
            "junction_annotations_sha256": _sha256_file(annotations_path),
            "source_manifest_sha256": _sha256_file(source_manifest),
            "split_manifest_sha256": _sha256_file(split_manifest),
            "latest_sha256": _sha256_file(latest_path),
            "sample_source_sha256": file_hashes,
        },
        "summary": {
            "same_raw_edge_records": same_edge,
            "different_raw_edge_records": len(candidates) - same_edge,
            "physical_candidate_clusters": len(clusters),
            "review_auto_accept_proposal_count": len(auto_accept),
            "review_auto_accept_train_split_count": len(auto_accept_train),
            "review_auto_accept_val_test_observation_only_count": (
                len(auto_accept) - len(auto_accept_train)
            ),
            "review_auto_accept_by_label": dict(
                sorted(Counter(value["review_proposal"]["proposed_label"] for value in auto_accept).items())
            ),
            "connected_cross_review_suggestion_count": int(
                cross_suggestions.get("connected_cross", 0)
            ),
            "non_connected_cross_review_suggestion_count": int(
                cross_suggestions.get("non_connected_cross", 0)
            ),
            "connected_or_non_connected_review_auto_accept_count": sum(
                value["review_proposal"]["eligible_for_review_auto_accept"]
                and value["review_proposal"]["proposed_label"]
                in {"connected_cross", "non_connected_cross"}
                for value in candidates
            ),
            "visual_explicit_dot_count": sum(
                value["visual"]["symbol"] == "filled_dot" for value in candidates
            ),
            "visual_bridge_or_gap_count": sum(
                value["visual"]["symbol"] == "bridge_or_gap" for value in candidates
            ),
            "still_manual_review_count": len(candidates) - len(auto_accept),
            "topology_conflict_count": topology_conflict,
            "topology_unknown_count": topology_unknown,
            "accepted_as_ground_truth_count": 0,
            "may_enter_training_without_human_review_count": 0,
        },
        "distributions": {key: dict(sorted(value.items())) for key, value in sorted(distributions.items())},
        "by_split_domain": [by_split_domain[key] for key in sorted(by_split_domain)],
        "clusters": clusters,
        "candidates": candidates,
        "policy": {
            "status": "review_proposals_only",
            "connected_cross_rule": (
                "orthogonal sparse geometry + connected topology consensus + explicit "
                "filled-dot raster cue + no negative topology channel"
            ),
            "non_connected_cross_rule": (
                "orthogonal sparse geometry + distinct topology consensus + explicit "
                "bridge/gap raster cue + no positive topology channel"
            ),
            "false_cross_rule": (
                "collinear or duplicate skeleton overlap is proposed as noise, except "
                "official sparse endpoint/diverge/turn keypoints are restored as their safe class"
            ),
            "no_ground_truth_promotion": True,
            "human_review_required": True,
            "val_test_training_exclusion": True,
        },
        "leakage_risk": {
            "rating": "high_if_proposals_are_used_as_labels",
            "duplicate_group_cross_split_count": len(leaking_duplicate_groups),
            "duplicate_groups_crossing_splits": leaking_duplicate_groups,
            "reasons": [
                "EDS and SPICE are co-packaged/generated from the same sample family as CVJsonStd, not independent annotations.",
                "Visual thresholds were designed from the visible corpus and are not a validated labeling model.",
                "Validation/test proposals must never be copied into training data.",
                "Duplicate groups are split-safe, but proposal logic must be frozen using train only before measuring val/test behavior.",
            ],
            "safe_use": (
                "Use this report to prioritize human review. Only signed-off train-split "
                "records may later enter a separately versioned supervision manifest."
            ),
        },
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Dataset300 UNKNOWN crossing candidates without creating GT"
    )
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "manifests" / "dataset300.jsonl",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "splits" / "dataset300_split.jsonl",
    )
    parser.add_argument(
        "--materialized-root",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "artifacts"
            / "training"
            / "materialized"
            / "dataset300-v2"
            / "wire-junction"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "audits" / "junction_truth_audit.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_junction_truth(
        dataset_root=args.dataset_root.resolve(),
        source_manifest=args.source_manifest.resolve(),
        split_manifest=args.split_manifest.resolve(),
        materialized_root=args.materialized_root.resolve(),
    )
    _atomic_write_json(args.output.resolve(), report)
    console = {
        "ok": True,
        "output": str(args.output.resolve()),
        "summary": report["summary"],
        "distributions": report["distributions"],
    }
    print(json.dumps(console, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
