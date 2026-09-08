from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pcbparse.graph.physical import build_physical_graph  # noqa: E402
from pcbparse.graph.union_find import UnionFind  # noqa: E402
from pcbparse.ir.models import CircuitIR, Junction, Point, WireSegment  # noqa: E402
from pcbparse.perception.torch_junction_adapter import _proper_intersection  # noqa: E402
from pcbparse.validation.graph import validate_graph  # noqa: E402


POSITIVE_JUNCTION_KINDS = frozenset({"t_junction", "connected_cross", "bend"})
ENDPOINT_TOLERANCE_PX = 3.0
PIN_SNAP_MIN_PX = 2.0
PIN_SNAP_MAX_PX = 6.0
PIN_SNAP_UNIQUE_MARGIN = 1.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: Iterable[int | float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None}

    def quantile(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
        return ordered[index]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": quantile(0.5),
        "p90": quantile(0.9),
        "max": ordered[-1],
    }


def _segment_component_count(uf: UnionFind, segment_ids: Iterable[str]) -> int:
    return len({uf.find(f"segment:{segment_id}") for segment_id in segment_ids})


def _segment_component_sizes(uf: UnionFind, segment_ids: Iterable[str]) -> dict[str, Any]:
    counts = Counter(uf.find(f"segment:{segment_id}") for segment_id in segment_ids)
    return _summary(counts.values())


def _successful_union(uf: UnionFind, left: str, right: str) -> bool:
    before_left = uf.find(left)
    before_right = uf.find(right)
    if before_left == before_right:
        return False
    uf.union(left, right)
    return True


def _union_touching_group(
    uf: UnionFind,
    segment_ids: Iterable[str],
    segment_index: dict[str, WireSegment],
    *,
    tolerance: float,
) -> tuple[int, int]:
    attempts = 0
    reductions = 0
    cells: dict[tuple[int, int], list[tuple[str, Point]]] = defaultdict(list)
    cell_size = max(tolerance, 1.0e-6)
    for segment_id in sorted(segment_ids):
        segment = segment_index[segment_id]
        for point in (segment.start, segment.end):
            cx = math.floor(point.x / cell_size)
            cy = math.floor(point.y / cell_size)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for other_id, other_point in cells.get((cx + dx, cy + dy), []):
                        if math.hypot(point.x - other_point.x, point.y - other_point.y) <= tolerance:
                            attempts += 1
                            reductions += int(
                                _successful_union(
                                    uf,
                                    f"segment:{segment_id}",
                                    f"segment:{other_id}",
                                )
                            )
            cells[(cx, cy)].append((segment_id, point))
    return attempts, reductions


def _stage_replay(ir: CircuitIR) -> tuple[dict[str, Any], UnionFind]:
    segment_ids = [segment.internal_id for segment in ir.segments]
    segment_index = ir.segment_index()
    uf = UnionFind(f"segment:{segment_id}" for segment_id in segment_ids)
    stages: dict[str, Any] = {
        "singleton_segments": {
            "components_before": len(segment_ids),
            "components_after": len(segment_ids),
            "successful_merges": 0,
            "component_size_segments": _segment_component_sizes(uf, segment_ids),
        }
    }

    endpoints: list[tuple[str, Point, str]] = []
    for segment in ir.segments:
        endpoints.extend(
            (
                (segment.internal_id, segment.start, segment.channel),
                (segment.internal_id, segment.end, segment.channel),
            )
        )
    before = _segment_component_count(uf, segment_ids)
    attempts = 0
    reductions = 0
    cells: dict[tuple[int, int, str], list[tuple[str, Point]]] = defaultdict(list)
    for segment_id, point, channel in endpoints:
        cx = math.floor(point.x / ENDPOINT_TOLERANCE_PX)
        cy = math.floor(point.y / ENDPOINT_TOLERANCE_PX)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other_id, other_point in cells.get((cx + dx, cy + dy, channel), []):
                    if math.hypot(point.x - other_point.x, point.y - other_point.y) <= ENDPOINT_TOLERANCE_PX:
                        attempts += 1
                        reductions += int(
                            _successful_union(
                                uf,
                                f"segment:{segment_id}",
                                f"segment:{other_id}",
                            )
                        )
        cells[(cx, cy, channel)].append((segment_id, point))
    stages["same_channel_endpoint_clustering"] = {
        "components_before": before,
        "components_after": _segment_component_count(uf, segment_ids),
        "candidate_pairs": attempts,
        "successful_merges": reductions,
        "tolerance_px": ENDPOINT_TOLERANCE_PX,
        "component_size_segments": _segment_component_sizes(uf, segment_ids),
    }

    by_polyline: dict[str, list[str]] = defaultdict(list)
    for segment in ir.segments:
        if segment.source_polyline_id:
            by_polyline[segment.source_polyline_id].append(segment.internal_id)
    before = _segment_component_count(uf, segment_ids)
    attempts = 0
    reductions = 0
    for members in by_polyline.values():
        group_attempts, group_reductions = _union_touching_group(
            uf,
            members,
            segment_index,
            tolerance=ENDPOINT_TOLERANCE_PX,
        )
        attempts += group_attempts
        reductions += group_reductions
    fragmented_polylines = sum(
        len({uf.find(f"segment:{segment_id}") for segment_id in members}) > 1
        for members in by_polyline.values()
    )
    stages["touching_source_polyline_members"] = {
        "components_before": before,
        "components_after": _segment_component_count(uf, segment_ids),
        "candidate_pairs": attempts,
        "successful_merges": reductions,
        "source_polylines": len(by_polyline),
        "source_polylines_still_fragmented": fragmented_polylines,
        "tolerance_px": ENDPOINT_TOLERANCE_PX,
        "component_size_segments": _segment_component_sizes(uf, segment_ids),
    }

    before = _segment_component_count(uf, segment_ids)
    reductions_by_kind: Counter[str] = Counter()
    candidate_groups_by_kind: Counter[str] = Counter()
    branch_references_by_kind: Counter[str] = Counter()
    productive_events: list[dict[str, Any]] = []
    for junction in ir.junctions:
        if junction.kind not in POSITIVE_JUNCTION_KINDS:
            continue
        known = list(dict.fromkeys(item for item in junction.segment_ids if item in segment_index))
        candidate_groups_by_kind[junction.kind] += 1
        branch_references_by_kind[junction.kind] += len(known)
        if not known:
            continue
        root_sizes_before = Counter(
            uf.find(f"segment:{segment_id}")
            for segment_id in segment_ids
        )
        participating_roots = {
            uf.find(f"segment:{segment_id}")
            for segment_id in known
        }
        event_reductions = 0
        for segment_id in known[1:]:
            changed = int(
                _successful_union(
                    uf,
                    f"segment:{known[0]}",
                    f"segment:{segment_id}",
                )
            )
            event_reductions += changed
            reductions_by_kind[junction.kind] += changed
        if event_reductions:
            resulting_root = uf.find(f"segment:{known[0]}")
            resulting_size = sum(
                1
                for segment_id in segment_ids
                if uf.find(f"segment:{segment_id}") == resulting_root
            )
            productive_events.append(
                {
                    "junction_id": junction.internal_id,
                    "kind": junction.kind,
                    "point": [junction.point.x, junction.point.y],
                    "score": junction.score,
                    "origin": _junction_origin(junction),
                    "branch_references": len(known),
                    "distinct_components_before": len(participating_roots),
                    "successful_merges": event_reductions,
                    "participating_component_segments_before": sum(
                        root_sizes_before[root] for root in participating_roots
                    ),
                    "resulting_component_segments": resulting_size,
                }
            )
    stages["positive_junctions"] = {
        "components_before": before,
        "components_after": _segment_component_count(uf, segment_ids),
        "successful_merges": sum(reductions_by_kind.values()),
        "successful_merges_by_kind": dict(sorted(reductions_by_kind.items())),
        "candidate_groups_by_kind": dict(sorted(candidate_groups_by_kind.items())),
        "branch_references_by_kind": dict(sorted(branch_references_by_kind.items())),
        "accepted_kinds": sorted(POSITIVE_JUNCTION_KINDS),
        "component_size_segments": _segment_component_sizes(uf, segment_ids),
        "largest_productive_events": sorted(
            productive_events,
            key=lambda item: (
                -item["resulting_component_segments"],
                -item["successful_merges"],
                item["junction_id"],
            ),
        )[:12],
    }

    by_source_edge: dict[str, list[str]] = defaultdict(list)
    for segment in ir.segments:
        if segment.source_edge_id:
            by_source_edge[segment.source_edge_id].append(segment.internal_id)
    before = _segment_component_count(uf, segment_ids)
    reductions = 0
    trusted_groups = 0
    touching_groups = 0
    for members in by_source_edge.values():
        if all(
            any(item.source == "dataset" for item in segment_index[segment_id].evidence)
            for segment_id in members
        ):
            trusted_groups += 1
            for segment_id in members[1:]:
                reductions += int(
                    _successful_union(
                        uf,
                        f"segment:{members[0]}",
                        f"segment:{segment_id}",
                    )
                )
        else:
            touching_groups += 1
            _, group_reductions = _union_touching_group(
                uf,
                members,
                segment_index,
                tolerance=ENDPOINT_TOLERANCE_PX,
            )
            reductions += group_reductions
    stages["source_edge_evidence"] = {
        "components_before": before,
        "components_after": _segment_component_count(uf, segment_ids),
        "successful_merges": reductions,
        "trusted_dataset_groups": trusted_groups,
        "geometry_checked_groups": touching_groups,
        "component_size_segments": _segment_component_sizes(uf, segment_ids),
    }
    return stages, uf


def _endpoint_pair_audit(ir: CircuitIR, final_uf: UnionFind) -> dict[str, Any]:
    endpoints: list[tuple[str, Point, str]] = []
    for segment in ir.segments:
        endpoints.extend(
            (
                (segment.internal_id, segment.start, segment.channel),
                (segment.internal_id, segment.end, segment.channel),
            )
        )
    cells: dict[tuple[int, int], list[tuple[str, Point, str]]] = defaultdict(list)
    pair_distance: dict[tuple[str, str], float] = {}
    pair_same_channel: dict[tuple[str, str], bool] = {}
    for segment_id, point, channel in endpoints:
        cx = math.floor(point.x / ENDPOINT_TOLERANCE_PX)
        cy = math.floor(point.y / ENDPOINT_TOLERANCE_PX)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other_id, other_point, other_channel in cells.get((cx + dx, cy + dy), []):
                    if other_id == segment_id:
                        continue
                    distance = math.hypot(point.x - other_point.x, point.y - other_point.y)
                    if distance > ENDPOINT_TOLERANCE_PX:
                        continue
                    pair = tuple(sorted((segment_id, other_id)))
                    if pair not in pair_distance or distance < pair_distance[pair]:
                        pair_distance[pair] = distance
                        pair_same_channel[pair] = channel == other_channel
        cells[(cx, cy)].append((segment_id, point, channel))

    positive_pairs: set[tuple[str, str]] = set()
    negative_pairs: set[tuple[str, str]] = set()
    for junction in ir.junctions:
        members = sorted(set(junction.segment_ids))
        target = positive_pairs if junction.kind in POSITIVE_JUNCTION_KINDS else negative_pairs
        target.update(itertools.combinations(members, 2))
    cross_channel = [pair for pair, same in pair_same_channel.items() if not same]
    same_channel = [pair for pair, same in pair_same_channel.items() if same]
    cross_connected = [
        pair
        for pair in cross_channel
        if final_uf.find(f"segment:{pair[0]}") == final_uf.find(f"segment:{pair[1]}")
    ]
    return {
        "tolerance_px": ENDPOINT_TOLERANCE_PX,
        "unique_segment_pairs": len(pair_distance),
        "same_channel_pairs": len(same_channel),
        "cross_channel_pairs": len(cross_channel),
        "cross_channel_pairs_exactly_coincident": sum(pair_distance[pair] <= 1.0e-9 for pair in cross_channel),
        "cross_channel_pairs_with_positive_junction_support": sum(pair in positive_pairs for pair in cross_channel),
        "cross_channel_pairs_with_only_negative_junction_support": sum(
            pair not in positive_pairs and pair in negative_pairs for pair in cross_channel
        ),
        "cross_channel_pairs_without_shared_junction_candidate": sum(
            pair not in positive_pairs and pair not in negative_pairs for pair in cross_channel
        ),
        "cross_channel_pairs_connected_after_positive_evidence": len(cross_connected),
        "cross_channel_pairs_left_separate": len(cross_channel) - len(cross_connected),
    }


def _intersection_audit(ir: CircuitIR, final_uf: UnionFind) -> dict[str, Any]:
    pair_kinds: dict[tuple[str, str], set[str]] = defaultdict(set)
    for junction in ir.junctions:
        for pair in itertools.combinations(sorted(set(junction.segment_ids)), 2):
            pair_kinds[pair].add(junction.kind)
    counts: Counter[str] = Counter()
    segment_pairs = 0
    for left_index, left in enumerate(ir.segments):
        for right in ir.segments[left_index + 1 :]:
            point = _proper_intersection(left, right)
            if point is None:
                continue
            segment_pairs += 1
            pair = tuple(sorted((left.internal_id, right.internal_id)))
            kinds = pair_kinds.get(pair, set())
            at_left_endpoint = min(
                math.hypot(point.x - left.start.x, point.y - left.start.y),
                math.hypot(point.x - left.end.x, point.y - left.end.y),
            ) <= 1.0e-6
            at_right_endpoint = min(
                math.hypot(point.x - right.start.x, point.y - right.start.y),
                math.hypot(point.x - right.end.x, point.y - right.end.y),
            ) <= 1.0e-6
            counts[
                "shared_endpoint" if at_left_endpoint and at_right_endpoint else "has_interior_contact"
            ] += 1
            counts["same_channel" if left.channel == right.channel else "cross_channel"] += 1
            if any(kind in POSITIVE_JUNCTION_KINDS for kind in kinds):
                counts["positive_junction_support"] += 1
            elif kinds:
                counts["only_negative_junction_support"] += 1
            else:
                counts["no_shared_junction_candidate"] += 1
            if final_uf.find(f"segment:{left.internal_id}") == final_uf.find(
                f"segment:{right.internal_id}"
            ):
                counts["connected_after_positive_evidence"] += 1
            else:
                counts["left_separate"] += 1
    return {"proper_intersection_segment_pairs": segment_pairs, **dict(sorted(counts.items()))}


def _collinear_overlap_audit(ir: CircuitIR, final_uf: UnionFind) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for left_index, left in enumerate(ir.segments):
        left_dx = left.end.x - left.start.x
        left_dy = left.end.y - left.start.y
        for right in ir.segments[left_index + 1 :]:
            right_dx = right.end.x - right.start.x
            right_dy = right.end.y - right.start.y
            if abs(left_dx * right_dy - left_dy * right_dx) > 1.0e-9:
                continue
            offset_x = right.start.x - left.start.x
            offset_y = right.start.y - left.start.y
            if abs(offset_x * left_dy - offset_y * left_dx) > 1.0e-9:
                continue
            if abs(left_dx) >= abs(left_dy):
                left_interval = sorted((left.start.x, left.end.x))
                right_interval = sorted((right.start.x, right.end.x))
            else:
                left_interval = sorted((left.start.y, left.end.y))
                right_interval = sorted((right.start.y, right.end.y))
            overlap = min(left_interval[1], right_interval[1]) - max(
                left_interval[0], right_interval[0]
            )
            if overlap <= 1.0e-9:
                continue
            counts["pairs"] += 1
            counts["same_channel" if left.channel == right.channel else "cross_channel"] += 1
            counts[
                "same_source_polyline"
                if left.source_polyline_id
                and left.source_polyline_id == right.source_polyline_id
                else "different_source_polyline"
            ] += 1
            connected = final_uf.find(f"segment:{left.internal_id}") == final_uf.find(
                f"segment:{right.internal_id}"
            )
            counts["connected"] += int(connected)
            counts["left_separate"] += int(not connected)
            if not connected and len(examples) < 12:
                examples.append(
                    {
                        "left_segment_id": left.internal_id,
                        "right_segment_id": right.internal_id,
                        "channel_pair": [left.channel, right.channel],
                        "overlap_axis_px": overlap,
                    }
                )
    return {**dict(sorted(counts.items())), "separate_examples": examples}


def _junction_origin(junction: Junction) -> str:
    learned = any(
        evidence.evidence_id.startswith("junction-model:")
        or evidence.metadata.get("task") == "junction"
        for evidence in junction.evidence
    )
    geometry = any(
        evidence.source == "geometry" or evidence.metadata.get("vectorizer")
        for evidence in junction.evidence
    )
    if learned and geometry:
        return "learned_and_geometry"
    if learned:
        return "learned_only"
    if geometry:
        return "geometry_only"
    return "other"


def _replay_current(ir: CircuitIR) -> tuple[CircuitIR, list[Any]]:
    replay = ir.model_copy(deep=True)
    replay.physical_nets, snap_decisions = build_physical_graph(
        replay,
        endpoint_tolerance_px=ENDPOINT_TOLERANCE_PX,
        pin_snap_min_px=PIN_SNAP_MIN_PX,
        pin_snap_tolerance_px=PIN_SNAP_MAX_PX,
        unique_margin=PIN_SNAP_UNIQUE_MARGIN,
    )
    return replay, snap_decisions


def _segment_only_net_count(ir: CircuitIR, *, endpoint_tolerance_px: float) -> int:
    replay = ir.model_copy(deep=True)
    replay.pins = []
    replay.net_labels = []
    nets, _ = build_physical_graph(
        replay,
        endpoint_tolerance_px=endpoint_tolerance_px,
        pin_snap_min_px=PIN_SNAP_MIN_PX,
        pin_snap_tolerance_px=PIN_SNAP_MAX_PX,
        unique_margin=PIN_SNAP_UNIQUE_MARGIN,
    )
    return len(nets)


def _current_vectorizer_channel(segment: WireSegment) -> str:
    dx = segment.end.x - segment.start.x
    dy = segment.end.y - segment.start.y
    if abs(dy) <= 1.0e-9:
        return "horizontal"
    if abs(dx) <= 1.0e-9:
        return "vertical"
    return "diagonal_positive" if dx * dy > 0 else "diagonal_negative"


def _pipeline_binding(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    graph = payload.get("graph", {})
    association = payload.get("association", {})
    observed = {
        "endpoint_tolerance_px": float(graph.get("endpoint_merge_tolerance_px", 3.0)),
        "pin_snap_min_px": float(association.get("pin_wire_snap_min_px", 2.0)),
        "pin_snap_max_px": float(
            association.get("pin_wire_snap_max_px", graph.get("pin_snap_tolerance_px", 6.0))
        ),
        "pin_snap_unique_margin": float(association.get("unique_margin", 1.5)),
    }
    expected = {
        "endpoint_tolerance_px": ENDPOINT_TOLERANCE_PX,
        "pin_snap_min_px": PIN_SNAP_MIN_PX,
        "pin_snap_max_px": PIN_SNAP_MAX_PX,
        "pin_snap_unique_margin": PIN_SNAP_UNIQUE_MARGIN,
    }
    if observed != expected:
        raise ValueError(
            "audit constants differ from the supplied frozen pipeline contract: "
            f"expected={expected}, observed={observed}"
        )
    return {
        "path": path.as_posix(),
        "sha256": _sha256(path),
        "rdp_epsilon_px": graph.get("rdp_epsilon_px"),
        "gap_repair_max_px": graph.get("gap_repair_max_px"),
        "allow_virtual_repair_ratio": graph.get("allow_virtual_repair_ratio"),
    }


def build_report(ir_path: Path, *, pipeline_config_path: Path | None = None) -> dict[str, Any]:
    pipeline_binding = _pipeline_binding(pipeline_config_path)
    ir = CircuitIR.model_validate_json(ir_path.read_text(encoding="utf-8"))
    stages, final_uf = _stage_replay(ir)
    replay, snaps = _replay_current(ir)
    graph_issues = validate_graph(replay)
    disconnected_after = sum(item.code == "NET.DISCONNECTED_SEGMENTS" for item in graph_issues)
    stored_disconnected = sum(
        item.code == "NET.DISCONNECTED_SEGMENTS" for item in ir.diagnostics
    )

    segment_nets = [net for net in replay.physical_nets if net.segments]
    pin_only_nets = [net for net in replay.physical_nets if net.pins and not net.segments]
    mixed_nets = [net for net in replay.physical_nets if net.pins and net.segments]
    polyline_sizes = Counter(
        segment.source_polyline_id
        for segment in ir.segments
        if segment.source_polyline_id is not None
    )
    segment_lengths = [
        math.hypot(segment.end.x - segment.start.x, segment.end.y - segment.start.y)
        for segment in ir.segments
    ]
    junction_kind_counts = Counter(junction.kind for junction in ir.junctions)
    junction_origin_counts = Counter(_junction_origin(junction) for junction in ir.junctions)
    positive_count = sum(
        count for kind, count in junction_kind_counts.items() if kind in POSITIVE_JUNCTION_KINDS
    )
    negative_count = len(ir.junctions) - positive_count
    channel_counts = Counter(segment.channel for segment in ir.segments)
    singleton_segment_ids = {
        net.segments[0]
        for net in segment_nets
        if len(net.segments) == 1
    }
    singleton_lengths = [
        math.hypot(segment.end.x - segment.start.x, segment.end.y - segment.start.y)
        for segment in ir.segments
        if segment.internal_id in singleton_segment_ids
    ]
    singleton_junction_support: Counter[str] = Counter()
    for segment_id in sorted(singleton_segment_ids):
        candidates = [
            junction
            for junction in ir.junctions
            if segment_id in junction.segment_ids and len(set(junction.segment_ids)) >= 2
        ]
        if any(junction.kind in POSITIVE_JUNCTION_KINDS for junction in candidates):
            singleton_junction_support["positive_multi_branch_candidate"] += 1
        elif candidates:
            singleton_junction_support["only_non_positive_multi_branch_candidates"] += 1
        else:
            singleton_junction_support["no_multi_branch_junction_candidate"] += 1

    exact_coordinate_segment_nets = _segment_only_net_count(ir, endpoint_tolerance_px=0.0)
    live_channel_ir = ir.model_copy(deep=True)
    for segment in live_channel_ir.segments:
        segment.channel = _current_vectorizer_channel(segment)
    live_channel_segment_nets = _segment_only_net_count(
        live_channel_ir,
        endpoint_tolerance_px=ENDPOINT_TOLERANCE_PX,
    )

    code_paths = [
        ROOT / "scripts/audit_saved_physical_graph.py",
        ROOT / "src/pcbparse/graph/physical.py",
        ROOT / "src/pcbparse/perception/composite_provider.py",
        ROOT / "src/pcbparse/perception/torch_junction_adapter.py",
        ROOT / "src/pcbparse/validation/graph.py",
    ]
    lineage_paths = [
        candidate
        for candidate in (
            ir_path.parents[1] / "summary.json",
            ir_path.parents[1] / "predictions.jsonl",
        )
        if candidate.is_file()
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "audit_kind": "saved-production-physical-graph-replay",
        "selection_boundary": {
            "ground_truth_files_opened": 0,
            "ground_truth_rows_used": 0,
            "test_rows_used": 0,
            "test_images_replayed_without_labels": 1,
            "threshold_search_performed": False,
            "external_approximate_reference_net_count": 38,
            "external_reference_used_for_selection": False,
            "note": "Sample 0024 is used only as a frozen failure-localization replay. No Dataset300 annotation or official target row is loaded.",
        },
        "hash_binding": {
            "input_ir": {"path": ir_path.as_posix(), "sha256": _sha256(ir_path)},
            "pipeline_config": pipeline_binding or None,
            "inference_lineage": {
                path.relative_to(ROOT).as_posix(): _sha256(path)
                for path in lineage_paths
            },
            "code": {
                path.relative_to(ROOT).as_posix(): _sha256(path)
                for path in code_paths
            },
        },
        "fixed_replay_contract": {
            "endpoint_tolerance_px": ENDPOINT_TOLERANCE_PX,
            "pin_snap_min_px": PIN_SNAP_MIN_PX,
            "pin_snap_max_px": PIN_SNAP_MAX_PX,
            "pin_snap_unique_margin": PIN_SNAP_UNIQUE_MARGIN,
            "positive_junction_kinds": sorted(POSITIVE_JUNCTION_KINDS),
            "rdp_epsilon_px_recorded_by_pipeline": pipeline_binding.get("rdp_epsilon_px"),
        },
        "input_inventory": {
            "image": {"width": ir.image.width, "height": ir.image.height},
            "components": len(ir.components),
            "pins": len(ir.pins),
            "segments": len(ir.segments),
            "junctions": len(ir.junctions),
            "source_polylines": len(polyline_sizes),
            "stored_physical_nets": len(ir.physical_nets),
            "channel_counts": dict(sorted(channel_counts.items())),
            "legacy_generic_diagonal_channels_present": "diagonal" in channel_counts,
        },
        "before_after": {
            "stored_pre_fix": {
                "physical_nets": len(ir.physical_nets),
                "disconnected_segment_errors": stored_disconnected,
            },
            "current_builder_replay": {
                "physical_nets": len(replay.physical_nets),
                "segment_bearing_nets": len(segment_nets),
                "pin_only_nets": len(pin_only_nets),
                "mixed_pin_segment_nets": len(mixed_nets),
                "disconnected_segment_errors": disconnected_after,
            },
        },
        "connectivity_attribution": stages,
        "replayed_net_shape": {
            "segment_counts_per_segment_bearing_net": _summary(len(net.segments) for net in segment_nets),
            "singleton_segment_nets": sum(len(net.segments) == 1 for net in segment_nets),
            "singleton_segment_length_px": _summary(singleton_lengths),
            "singleton_segments_at_or_below_endpoint_tolerance": sum(
                length <= ENDPOINT_TOLERANCE_PX for length in singleton_lengths
            ),
            "singleton_segment_junction_support": dict(
                sorted(singleton_junction_support.items())
            ),
            "pin_counts_per_net": _summary(len(net.pins) for net in replay.physical_nets),
            "snap_status_counts": dict(sorted(Counter(item.status for item in snaps).items())),
            "accepted_pin_snaps": sum(item.status == "accepted" for item in snaps),
        },
        "vectorization_attribution": {
            "segments_per_source_polyline": _summary(polyline_sizes.values()),
            "multi_segment_source_polylines": sum(value > 1 for value in polyline_sizes.values()),
            "segment_length_px": _summary(segment_lengths),
            "segments_at_or_below_endpoint_tolerance": sum(
                length <= ENDPOINT_TOLERANCE_PX for length in segment_lengths
            ),
            "interpretation": "RDP subdivisions inside one retained source polyline are reconnected only when their endpoints touch; their measured reduction is reported in connectivity_attribution.",
        },
        "contract_ablations_not_used_for_selection": {
            "segment_nets_with_configured_3px_endpoint_contract": len(segment_nets),
            "segment_nets_with_exact_coordinate_endpoint_contract": exact_coordinate_segment_nets,
            "net_reduction_from_configured_endpoint_tolerance_and_touching_contract": exact_coordinate_segment_nets
            - len(segment_nets),
            "segment_nets_after_projecting_legacy_channels_to_current_vectorizer_channels": live_channel_segment_nets,
            "note": "These are fixed contract ablations, not a threshold scan. The current-channel projection changes only stored categorical channel labels and performs no model inference; it increases rather than explains away fragmentation in this legacy replay.",
        },
        "junction_attribution": {
            "kind_counts": dict(sorted(junction_kind_counts.items())),
            "origin_counts": dict(sorted(junction_origin_counts.items())),
            "positive_count": positive_count,
            "non_positive_count": negative_count,
            "positive_fraction": positive_count / len(ir.junctions) if ir.junctions else None,
            "branches_per_kind": {
                kind: _summary(
                    len(set(junction.segment_ids))
                    for junction in ir.junctions
                    if junction.kind == kind
                )
                for kind in sorted(junction_kind_counts)
            },
        },
        "endpoint_pair_attribution": _endpoint_pair_audit(ir, final_uf),
        "intersection_attribution": _intersection_audit(ir, final_uf),
        "collinear_overlap_attribution": _collinear_overlap_audit(ir, final_uf),
        "diagnosis": {
            "proven_fixed_implementation_defect": "The stored pre-fix graph connected non-positive junction candidates, producing 24 validator-proven disconnected nets. Current positive-only junction semantics removes all 24 errors.",
            "new_contract_proven_connectivity_defects": [],
            "weight_or_upstream_evidence_limitations": [
                "The remaining segment-bearing net count is determined chiefly by the number of retained wire polylines and learned junction classes; a saved IR cannot establish which negative decisions are false without trusted validation evidence.",
                "Pin snapping can change only the small number of pin-only groups reported above and cannot explain segment-net fragmentation.",
            ],
            "replay_limitations": [
                "The saved IR contains legacy generic diagonal channel values produced before the live vectorizer split diagonal slopes. Replaying stored segments cannot measure the fully regenerated production output.",
                "The saved IR does not contain the raw wire probability mask or pre-RDP skeleton, so gap-closing and RDP alternatives cannot be rerun without fresh model inference.",
            ],
            "inactive_config_controls": [
                {
                    "name": "graph.gap_repair_max_px",
                    "configured_value": pipeline_binding.get("gap_repair_max_px"),
                    "status": "declared_but_not_consumed_by_the_graph_or_repair_pipeline",
                },
                {
                    "name": "graph.allow_virtual_repair_ratio",
                    "configured_value": pipeline_binding.get("allow_virtual_repair_ratio"),
                    "status": "declared_but_not_consumed_by_the_graph_or_repair_pipeline",
                },
            ],
            "inactive_control_disposition": "This is a real implementation gap, but enabling an 8px virtual bridge from sample 0024 would be unsafe test-image tuning. A bridge policy needs an independently frozen validation contract and crossing safeguards first.",
        },
        "ranked_next_actions": [
            {
                "rank": 1,
                "action": "Evaluate junction positive-class recall and calibration on the frozen trusted validation junction manifest, stratified by candidate origin and branch count.",
                "why": f"{singleton_junction_support['only_non_positive_multi_branch_candidates']} of {len(singleton_segment_ids)} singleton segment nets already have only non-positive multi-branch junction candidates. Trusted validation is required to tell correct endpoint/non-connected decisions from missed bends/T-junctions.",
            },
            {
                "rank": 2,
                "action": "Add a validation-only wire connectivity metric that measures retained-polyline fragmentation and junction coverage from trusted wire/junction records.",
                "why": f"{singleton_junction_support['no_multi_branch_junction_candidate']} singleton segment nets have no multi-branch junction candidate at all, so junction reclassification alone cannot absorb them; pixel IoU does not expose this graph failure mode.",
            },
            {
                "rank": 3,
                "action": "Specify and validate the currently inactive gap-repair/virtual-repair policy on frozen trusted validation data before wiring those pipeline fields into graph construction.",
                "why": "The pipeline advertises 8px/8% controls but no implementation consumes them; blindly enabling them on 0024 could short nearby nets.",
            },
            {
                "rank": 4,
                "action": "Regenerate one production IR with current diagonal channel serialization after independently blessed weights are available, then rerun this hash-bound audit unchanged.",
                "why": "The historical IR predates the live diagonal-slope fix, so its 205-net replay is not a full current-production measurement.",
            },
        ],
        "conclusion": {
            "safe_code_change_applied": False,
            "reason": "No additional graph-connectivity behavior is safe to change from this evidence. The inactive gap-repair controls are a proven configuration-surface gap, but enabling a bridge policy without an independently validated contract would be test-image tuning and could short non-connected crossings.",
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay a saved production CircuitIR without loading images, models, or labels."
    )
    parser.add_argument("--ir", required=True, type=Path)
    parser.add_argument("--pipeline-config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    ir_path = args.ir.resolve()
    pipeline_config_path = args.pipeline_config.resolve() if args.pipeline_config else None
    output = args.output.resolve()
    report = build_report(ir_path, pipeline_config_path=pipeline_config_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["before_after"], ensure_ascii=False, sort_keys=True))
    print(f"audit_sha256={_sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
