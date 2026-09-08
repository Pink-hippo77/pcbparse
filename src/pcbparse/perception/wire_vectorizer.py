"""Dependency-free vectorisation of binary schematic-wire masks.

The vectoriser deliberately operates on one inference tile at a time.  The
default limits reject masks larger than 4,194,304 pixels (2048 x 2048) or with
more than 1,000,000 foreground pixels; callers handling larger drawings must
tile them and use :func:`deduplicate_polylines` after mapping tile coordinates
back to the original image.

Only NumPy is required.  Processing is deterministic and consists of optional
binary closing / small-hole filling, Zhang-Suen thinning, corner-safe 8-neighbour
pixel-graph tracing, mandatory splitting at graph junctions, and RDP
simplification.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
from math import hypot
from typing import Iterable, Literal, Sequence

import numpy as np


Coordinate = tuple[float, float]
Pixel = tuple[int, int]
JunctionKind = Literal["endpoint", "t_junction", "connected_cross", "ambiguous"]


@dataclass(frozen=True, slots=True)
class WireVectorizerConfig:
    """Controls mask cleanup and vectorisation.

    ``gap_closing_radius`` is a square binary-closing radius.  A value of one
    repairs one-pixel gaps in ordinary one-pixel strokes after thinning.
    Cleanup is conservative by default because closing can join legitimately
    separate nearby nets.
    """

    gap_closing_radius: int = 0
    maximum_hole_area: int = 0
    minimum_component_pixels: int = 2
    minimum_polyline_pixels: int = 2
    minimum_polyline_length: float = 0.0
    rdp_epsilon: float = 0.75
    maximum_thinning_iterations: int = 512
    maximum_mask_pixels: int = 4_194_304
    maximum_foreground_pixels: int = 1_000_000
    # Optional guard against a single mask component spanning unrelated
    # regions. Zero keeps the historical behavior.
    maximum_component_pixels: int = 0

    def __post_init__(self) -> None:
        integer_fields = (
            "gap_closing_radius",
            "maximum_hole_area",
            "minimum_component_pixels",
            "minimum_polyline_pixels",
            "maximum_thinning_iterations",
            "maximum_mask_pixels",
            "maximum_foreground_pixels",
            "maximum_component_pixels",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if self.gap_closing_radius < 0 or self.maximum_hole_area < 0:
            raise ValueError("morphology sizes must be non-negative")
        if self.minimum_component_pixels < 1 or self.minimum_polyline_pixels < 2:
            raise ValueError("component minimum must be >= 1 and polyline minimum >= 2")
        if self.maximum_thinning_iterations < 1:
            raise ValueError("maximum_thinning_iterations must be positive")
        if self.maximum_mask_pixels < 1 or self.maximum_foreground_pixels < 1:
            raise ValueError("mask safety limits must be positive")
        if self.maximum_component_pixels < 0:
            raise ValueError("maximum_component_pixels must be non-negative")
        if self.minimum_polyline_length < 0.0 or self.rdp_epsilon < 0.0:
            raise ValueError("length and RDP tolerances must be non-negative")


@dataclass(frozen=True, slots=True)
class SkeletonPolyline:
    """One junction-to-junction (or endpoint-to-junction) skeleton path."""

    internal_id: str
    points: tuple[Coordinate, ...]
    pixel_count: int
    length: float

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("a skeleton polyline needs at least two points")
        if self.pixel_count < 2:
            raise ValueError("pixel_count must be at least two")


@dataclass(frozen=True, slots=True)
class WireJunctionCandidate:
    """A graph endpoint or a possible electrically connected junction."""

    internal_id: str
    point: Coordinate
    kind: JunctionKind
    degree: int
    polyline_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WireVectorizationResult:
    polylines: tuple[SkeletonPolyline, ...]
    junctions: tuple[WireJunctionCandidate, ...]
    foreground_pixels: int
    skeleton_pixels: int
    thinning_iterations: int


def vectorize_wire_mask(
    mask: np.ndarray,
    *,
    origin: Coordinate = (0.0, 0.0),
    pixel_size: Coordinate = (1.0, 1.0),
    config: WireVectorizerConfig | None = None,
    id_namespace: str = "wire",
) -> WireVectorizationResult:
    """Convert a 2-D boolean/unsigned-integer tile mask into global polylines.

    Pixel ``(row, column)`` maps to ``(origin_x + column * pixel_size_x,
    origin_y + row * pixel_size_y)``.  Positive pixel sizes are required.
    Stable identifiers depend only on the namespace and mapped geometry, not on
    input traversal order.
    """

    options = config or WireVectorizerConfig()
    binary = _normalise_mask(mask, options)
    ox, oy = _finite_pair(origin, "origin")
    sx, sy = _finite_pair(pixel_size, "pixel_size")
    if sx <= 0.0 or sy <= 0.0:
        raise ValueError("pixel_size values must be positive")
    if not id_namespace or any(char.isspace() for char in id_namespace):
        raise ValueError("id_namespace must be a non-empty token")

    if options.gap_closing_radius:
        binary = _binary_close(binary, options.gap_closing_radius)
    if options.maximum_hole_area:
        binary = _fill_small_holes(binary, options.maximum_hole_area)
    if options.minimum_component_pixels > 1:
        binary = _remove_small_components(binary, options.minimum_component_pixels)
    if options.maximum_component_pixels:
        binary = _remove_large_components(binary, options.maximum_component_pixels)

    foreground_pixels = int(np.count_nonzero(binary))
    if foreground_pixels > options.maximum_foreground_pixels:
        raise ValueError(
            f"wire mask has {foreground_pixels} foreground pixels, exceeding the "
            f"configured tile limit {options.maximum_foreground_pixels}; split it into tiles"
        )
    if foreground_pixels == 0:
        return WireVectorizationResult((), (), 0, 0, 0)

    skeleton, thinning_iterations = _zhang_suen(binary, options.maximum_thinning_iterations)
    nodes, adjacency = _build_pixel_graph(skeleton)
    raw_paths = _trace_pixel_graph(nodes, adjacency)

    path_records: list[tuple[Pixel, Pixel, SkeletonPolyline]] = []
    geometry_seen: set[tuple[Coordinate, ...]] = set()
    for pixel_path in raw_paths:
        if len(pixel_path) < options.minimum_polyline_pixels:
            continue
        mapped = tuple((ox + column * sx, oy + row * sy) for row, column in pixel_path)
        simplified = _simplify_mapped_path(mapped, options.rdp_epsilon)
        simplified = _canonical_points(simplified)
        if len(simplified) < 2 or simplified in geometry_seen:
            continue
        length = _polyline_length(simplified)
        if length < options.minimum_polyline_length:
            continue
        geometry_seen.add(simplified)
        polyline = SkeletonPolyline(
            internal_id=_geometry_id(id_namespace, "polyline", simplified),
            points=simplified,
            pixel_count=len(pixel_path),
            length=length,
        )
        path_records.append((pixel_path[0], pixel_path[-1], polyline))

    path_records.sort(key=lambda item: _polyline_sort_key(item[2]))
    endpoint_memberships: dict[Pixel, list[str]] = {}
    for start, end, polyline in path_records:
        endpoint_memberships.setdefault(start, []).append(polyline.internal_id)
        endpoint_memberships.setdefault(end, []).append(polyline.internal_id)

    junctions: list[WireJunctionCandidate] = []
    for pixel in sorted(nodes):
        degree = len(adjacency[pixel])
        if degree == 1:
            kind: JunctionKind = "endpoint"
        elif degree == 3:
            kind = "t_junction"
        elif degree == 4:
            kind = "connected_cross"
        elif degree > 4:
            kind = "ambiguous"
        else:
            continue
        polyline_ids = tuple(sorted(set(endpoint_memberships.get(pixel, ()))))
        if not polyline_ids:
            continue
        row, column = pixel
        point = (ox + column * sx, oy + row * sy)
        junctions.append(
            WireJunctionCandidate(
                internal_id=_geometry_id(id_namespace, "junction", (point,), suffix=kind),
                point=point,
                kind=kind,
                degree=degree,
                polyline_ids=polyline_ids,
            )
        )

    junctions.sort(key=lambda item: (item.point[1], item.point[0], item.kind, item.internal_id))
    return WireVectorizationResult(
        polylines=tuple(item[2] for item in path_records),
        junctions=tuple(junctions),
        foreground_pixels=foreground_pixels,
        skeleton_pixels=len(nodes),
        thinning_iterations=thinning_iterations,
    )


def thin_binary_mask(
    mask: np.ndarray,
    *,
    maximum_iterations: int = 512,
    maximum_mask_pixels: int = 4_194_304,
) -> np.ndarray:
    """Return the deterministic one-pixel Zhang-Suen skeleton of ``mask``."""

    if not isinstance(maximum_iterations, int) or isinstance(maximum_iterations, bool):
        raise TypeError("maximum_iterations must be an integer")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")
    if not isinstance(maximum_mask_pixels, int) or isinstance(maximum_mask_pixels, bool):
        raise TypeError("maximum_mask_pixels must be an integer")
    if maximum_mask_pixels < 1:
        raise ValueError("maximum_mask_pixels must be positive")
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("wire mask must be two-dimensional")
    if array.dtype.kind not in {"b", "u"}:
        raise TypeError("wire mask must have boolean or unsigned-integer dtype")
    if array.size > maximum_mask_pixels:
        raise ValueError(
            f"wire mask has {array.size} pixels, exceeding the configured tile limit "
            f"{maximum_mask_pixels}; split it into tiles"
        )
    result, _ = _zhang_suen(array != 0, maximum_iterations)
    return result


def deduplicate_polylines(
    polylines: Iterable[SkeletonPolyline],
    *,
    tolerance: float = 0.75,
) -> tuple[SkeletonPolyline, ...]:
    """Remove repeated global curves emitted by overlapping inference tiles.

    Curves are considered duplicates only when their paired endpoints and their
    bidirectional vertex-to-curve distances are within ``tolerance``.  Thus a
    short collinear sub-path is not discarded as a duplicate of a longer net.
    The result is independent of input order.
    """

    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    candidates = sorted(tuple(polylines), key=_polyline_sort_key)
    kept: list[SkeletonPolyline] = []
    for candidate in candidates:
        if any(_same_curve(candidate.points, other.points, tolerance) for other in kept):
            continue
        kept.append(candidate)
    return tuple(kept)


def _normalise_mask(mask: np.ndarray, options: WireVectorizerConfig) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"wire mask must be two-dimensional, received shape {array.shape}")
    if array.dtype.kind not in {"b", "u"}:
        raise TypeError("wire mask must have boolean or unsigned-integer dtype")
    if array.size > options.maximum_mask_pixels:
        raise ValueError(
            f"wire mask has {array.size} pixels, exceeding the configured tile limit "
            f"{options.maximum_mask_pixels}; split it into tiles"
        )
    return np.array(array != 0, dtype=bool, copy=True, order="C")


def _finite_pair(value: Coordinate, name: str) -> Coordinate:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    result = (float(value[0]), float(value[1]))
    if not all(np.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _binary_close(mask: np.ndarray, radius: int) -> np.ndarray:
    # Work on a larger zero-padded canvas so erosion does not shorten strokes at
    # tile borders merely because their neighbourhood lies outside the tile.
    pad = 2 * radius
    expanded = np.pad(mask, pad, mode="constant", constant_values=False)
    dilated = _dilate_same(expanded, radius)
    closed = _erode_same(dilated, radius)
    return closed[pad : pad + mask.shape[0], pad : pad + mask.shape[1]]


def _dilate_same(mask: np.ndarray, radius: int) -> np.ndarray:
    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.zeros_like(mask)
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            result |= padded[row_offset : row_offset + height, column_offset : column_offset + width]
    return result


def _erode_same(mask: np.ndarray, radius: int) -> np.ndarray:
    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.ones_like(mask)
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            result &= padded[row_offset : row_offset + height, column_offset : column_offset + width]
    return result


def _fill_small_holes(mask: np.ndarray, maximum_area: int) -> np.ndarray:
    height, width = mask.shape
    visited = np.zeros_like(mask)
    exterior: deque[Pixel] = deque()
    for column in range(width):
        if not mask[0, column]:
            exterior.append((0, column))
        if height > 1 and not mask[height - 1, column]:
            exterior.append((height - 1, column))
    for row in range(1, max(1, height - 1)):
        if not mask[row, 0]:
            exterior.append((row, 0))
        if width > 1 and not mask[row, width - 1]:
            exterior.append((row, width - 1))
    _flood_background(mask, visited, exterior, collect=False)

    result = mask.copy()
    for row, column in np.argwhere(~mask & ~visited):
        pixel = (int(row), int(column))
        if visited[pixel]:
            continue
        component: list[Pixel] = []
        queue: deque[Pixel] = deque((pixel,))
        _flood_background(mask, visited, queue, collect=True, output=component)
        if len(component) <= maximum_area:
            for item in component:
                result[item] = True
    return result


def _flood_background(
    mask: np.ndarray,
    visited: np.ndarray,
    queue: deque[Pixel],
    *,
    collect: bool,
    output: list[Pixel] | None = None,
) -> None:
    height, width = mask.shape
    seeds = tuple(queue)
    queue.clear()
    for row, column in seeds:
        if 0 <= row < height and 0 <= column < width and not mask[row, column] and not visited[row, column]:
            visited[row, column] = True
            queue.append((row, column))
    while queue:
        row, column = queue.popleft()
        if collect and output is not None:
            output.append((row, column))
        for next_row, next_column in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            if (
                0 <= next_row < height
                and 0 <= next_column < width
                and not mask[next_row, next_column]
                and not visited[next_row, next_column]
            ):
                visited[next_row, next_column] = True
                queue.append((next_row, next_column))


def _remove_small_components(mask: np.ndarray, minimum: int) -> np.ndarray:
    result = mask.copy()
    visited = np.zeros_like(mask)
    height, width = mask.shape
    for raw_row, raw_column in np.argwhere(mask):
        start = (int(raw_row), int(raw_column))
        if visited[start]:
            continue
        component: list[Pixel] = []
        queue: deque[Pixel] = deque((start,))
        visited[start] = True
        while queue:
            row, column = queue.popleft()
            component.append((row, column))
            for row_delta in (-1, 0, 1):
                for column_delta in (-1, 0, 1):
                    next_row = row + row_delta
                    next_column = column + column_delta
                    if (
                        (row_delta or column_delta)
                        and 0 <= next_row < height
                        and 0 <= next_column < width
                        and mask[next_row, next_column]
                        and not visited[next_row, next_column]
                    ):
                        visited[next_row, next_column] = True
                        queue.append((next_row, next_column))
        if len(component) < minimum:
            for pixel in component:
                result[pixel] = False
    return result


def _remove_large_components(mask: np.ndarray, maximum: int) -> np.ndarray:
    result = mask.copy()
    visited = np.zeros_like(mask)
    height, width = mask.shape
    for raw_row, raw_column in np.argwhere(mask):
        start = (int(raw_row), int(raw_column))
        if visited[start]:
            continue
        component: list[Pixel] = []
        queue: deque[Pixel] = deque((start,))
        visited[start] = True
        while queue:
            row, column = queue.popleft()
            component.append((row, column))
            for row_delta in (-1, 0, 1):
                for column_delta in (-1, 0, 1):
                    next_row = row + row_delta
                    next_column = column + column_delta
                    if (
                        (row_delta or column_delta)
                        and 0 <= next_row < height
                        and 0 <= next_column < width
                        and mask[next_row, next_column]
                        and not visited[next_row, next_column]
                    ):
                        visited[next_row, next_column] = True
                        queue.append((next_row, next_column))
        if len(component) > maximum:
            for pixel in component:
                result[pixel] = False
    return result


def _zhang_suen(mask: np.ndarray, maximum_iterations: int) -> tuple[np.ndarray, int]:
    skeleton = np.array(mask, dtype=bool, copy=True, order="C")
    for iteration in range(1, maximum_iterations + 1):
        changed = False
        neighbours = _neighbour_planes(skeleton)
        removal = _thinning_removal(skeleton, neighbours, first=True)
        if np.any(removal):
            skeleton[removal] = False
            changed = True
        neighbours = _neighbour_planes(skeleton)
        removal = _thinning_removal(skeleton, neighbours, first=False)
        if np.any(removal):
            skeleton[removal] = False
            changed = True
        if not changed:
            return skeleton, iteration - 1
    raise RuntimeError(
        f"Zhang-Suen thinning did not converge within {maximum_iterations} iterations; "
        "use smaller tiles or increase maximum_thinning_iterations"
    )


def _neighbour_planes(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    # Clockwise P2 ... P9 in the notation of the Zhang-Suen paper.
    return (
        padded[:-2, 1:-1],
        padded[:-2, 2:],
        padded[1:-1, 2:],
        padded[2:, 2:],
        padded[2:, 1:-1],
        padded[2:, :-2],
        padded[1:-1, :-2],
        padded[:-2, :-2],
    )


def _thinning_removal(mask: np.ndarray, neighbours: tuple[np.ndarray, ...], *, first: bool) -> np.ndarray:
    p2, p3, p4, p5, p6, p7, p8, p9 = neighbours
    count = sum(plane.astype(np.uint8, copy=False) for plane in neighbours)
    transitions = sum(
        ((~left) & right).astype(np.uint8, copy=False)
        for left, right in zip(neighbours, (*neighbours[1:], neighbours[0]))
    )
    base = mask & (count >= 2) & (count <= 6) & (transitions == 1)
    if first:
        return base & ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
    return base & ~(p2 & p4 & p8) & ~(p2 & p6 & p8)


def _build_pixel_graph(skeleton: np.ndarray) -> tuple[tuple[Pixel, ...], dict[Pixel, tuple[Pixel, ...]]]:
    nodes = tuple(sorted((int(row), int(column)) for row, column in np.argwhere(skeleton)))
    node_set = set(nodes)
    adjacency: dict[Pixel, tuple[Pixel, ...]] = {}
    for row, column in nodes:
        neighbours: list[Pixel] = []
        for row_delta in (-1, 0, 1):
            for column_delta in (-1, 0, 1):
                if not (row_delta or column_delta):
                    continue
                candidate = (row + row_delta, column + column_delta)
                if candidate not in node_set:
                    continue
                if row_delta and column_delta:
                    # Keep true diagonal strokes but suppress the triangular
                    # shortcuts 8-connectivity creates around orthogonal T/X
                    # intersections and right-angle bends.
                    if (row + row_delta, column) in node_set or (row, column + column_delta) in node_set:
                        continue
                neighbours.append(candidate)
        adjacency[(row, column)] = tuple(sorted(neighbours))
    return nodes, adjacency


def _edge(left: Pixel, right: Pixel) -> tuple[Pixel, Pixel]:
    return (left, right) if left <= right else (right, left)


def _trace_pixel_graph(
    nodes: Sequence[Pixel],
    adjacency: dict[Pixel, tuple[Pixel, ...]],
) -> list[tuple[Pixel, ...]]:
    visited: set[tuple[Pixel, Pixel]] = set()
    paths: list[tuple[Pixel, ...]] = []
    key_nodes = tuple(node for node in nodes if len(adjacency[node]) != 2)

    for start in key_nodes:
        for neighbour in adjacency[start]:
            first_edge = _edge(start, neighbour)
            if first_edge in visited:
                continue
            path = [start, neighbour]
            visited.add(first_edge)
            previous, current = start, neighbour
            while len(adjacency[current]) == 2:
                choices = tuple(item for item in adjacency[current] if item != previous)
                if not choices:
                    break
                following = choices[0]
                next_edge = _edge(current, following)
                if next_edge in visited:
                    break
                visited.add(next_edge)
                path.append(following)
                previous, current = current, following
            paths.append(tuple(path))

    # Components with no degree != 2 node are closed cycles.  Pick their
    # lexicographically smallest unvisited edge to make traversal deterministic.
    all_edges = sorted(
        {_edge(node, neighbour) for node in nodes for neighbour in adjacency[node] if node != neighbour}
    )
    while True:
        remaining = next((item for item in all_edges if item not in visited), None)
        if remaining is None:
            break
        start, neighbour = remaining
        path = [start, neighbour]
        visited.add(remaining)
        previous, current = start, neighbour
        while current != start:
            choices = sorted(
                item
                for item in adjacency[current]
                if item != previous and _edge(current, item) not in visited
            )
            if not choices:
                break
            following = choices[0]
            visited.add(_edge(current, following))
            path.append(following)
            previous, current = current, following
        paths.append(tuple(path))
    return paths


def _simplify_mapped_path(points: tuple[Coordinate, ...], epsilon: float) -> tuple[Coordinate, ...]:
    if len(points) <= 2 or epsilon == 0.0:
        return points
    if points[0] != points[-1]:
        return _rdp(points, epsilon)

    ring = points[:-1]
    if len(ring) < 3:
        return points
    anchor = ring[0]
    farthest_index = max(
        range(1, len(ring)),
        key=lambda index: (_squared_distance(anchor, ring[index]), -index),
    )
    first_arc = _rdp(ring[: farthest_index + 1], epsilon)
    second_arc = _rdp((ring[0], *reversed(ring[farthest_index:])), epsilon)
    return (*first_arc, *reversed(second_arc[:-1]))


def _rdp(points: Sequence[Coordinate], epsilon: float) -> tuple[Coordinate, ...]:
    if len(points) <= 2:
        return tuple(points)
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        maximum_distance = -1.0
        maximum_index = -1
        for index in range(start + 1, end):
            distance = _point_segment_distance(points[index], points[start], points[end])
            if distance > maximum_distance:
                maximum_distance = distance
                maximum_index = index
        if maximum_index >= 0 and maximum_distance > epsilon:
            keep[maximum_index] = True
            stack.append((maximum_index, end))
            stack.append((start, maximum_index))
    return tuple(point for index, point in enumerate(points) if keep[index])


def _canonical_points(points: Sequence[Coordinate]) -> tuple[Coordinate, ...]:
    sequence = tuple(points)
    if sequence[0] != sequence[-1]:
        reverse = tuple(reversed(sequence))
        return min(sequence, reverse)
    ring = sequence[:-1]
    if not ring:
        return sequence
    candidates: list[tuple[Coordinate, ...]] = []
    for oriented in (ring, tuple(reversed(ring))):
        minimum = min(oriented)
        for index, point in enumerate(oriented):
            if point == minimum:
                rotated = oriented[index:] + oriented[:index]
                candidates.append((*rotated, rotated[0]))
    return min(candidates)


def _geometry_id(
    namespace: str,
    category: str,
    points: Sequence[Coordinate],
    *,
    suffix: str = "",
) -> str:
    payload = ";".join(f"{x:.12g},{y:.12g}" for x, y in points)
    digest = sha256(f"{category}|{suffix}|{payload}".encode("ascii")).hexdigest()[:20]
    return f"{namespace}:{category}:{digest}"


def _polyline_length(points: Sequence[Coordinate]) -> float:
    return sum(hypot(right[0] - left[0], right[1] - left[1]) for left, right in zip(points, points[1:]))


def _polyline_sort_key(polyline: SkeletonPolyline) -> tuple[object, ...]:
    return (polyline.points, polyline.internal_id, -polyline.pixel_count, polyline.length)


def _same_curve(left: Sequence[Coordinate], right: Sequence[Coordinate], tolerance: float) -> bool:
    left_closed = left[0] == left[-1]
    right_closed = right[0] == right[-1]
    if left_closed != right_closed:
        return False
    if not left_closed:
        forward = max(_distance(left[0], right[0]), _distance(left[-1], right[-1]))
        reverse = max(_distance(left[0], right[-1]), _distance(left[-1], right[0]))
        if min(forward, reverse) > tolerance:
            return False
    return max(_directed_curve_distance(left, right), _directed_curve_distance(right, left)) <= tolerance


def _directed_curve_distance(points: Sequence[Coordinate], curve: Sequence[Coordinate]) -> float:
    segments = tuple(zip(curve, curve[1:]))
    if not segments:
        return float("inf")
    maximum = 0.0
    for point in points:
        distance = min(_point_segment_distance(point, start, end) for start, end in segments)
        maximum = max(maximum, distance)
    return maximum


def _point_segment_distance(point: Coordinate, start: Coordinate, end: Coordinate) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator == 0.0:
        return _distance(point, start)
    ratio = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator
    ratio = min(1.0, max(0.0, ratio))
    projection = (start[0] + ratio * dx, start[1] + ratio * dy)
    return _distance(point, projection)


def _distance(left: Coordinate, right: Coordinate) -> float:
    return hypot(right[0] - left[0], right[1] - left[1])


def _squared_distance(left: Coordinate, right: Coordinate) -> float:
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    return dx * dx + dy * dy


__all__ = [
    "SkeletonPolyline",
    "WireJunctionCandidate",
    "WireVectorizationResult",
    "WireVectorizerConfig",
    "deduplicate_polylines",
    "thin_binary_mask",
    "vectorize_wire_mask",
]
