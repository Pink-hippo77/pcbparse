from __future__ import annotations

import copy
import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Mapping

from .hashing import apply_duplicate_groups


DEFAULT_SPLIT_RATIOS: dict[str, float] = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}


def _integer_targets(total: int, ratio_items: list[tuple[str, float]]) -> dict[str, int]:
    """Hamilton allocation with deterministic ratio-order tie breaking."""

    raw = {name: ratio * total for name, ratio in ratio_items}
    targets = {name: math.floor(value) for name, value in raw.items()}
    remainder = total - sum(targets.values())
    order = {name: position for position, (name, _) in enumerate(ratio_items)}
    for name in sorted(raw, key=lambda item: (-(raw[item] - targets[item]), order[item]))[
        :remainder
    ]:
        targets[name] += 1
    return targets


def _stratified_targets(
    source_totals: Mapping[str, int], ratio_items: list[tuple[str, float]]
) -> dict[str, dict[str, int]]:
    """Return integer source-by-split quotas with useful rare-domain coverage.

    When a source domain has at least one sample per enabled split, every split
    receives one.  A minimum-cost reconciliation then makes the column totals
    equal the global Hamilton targets.  This prevents a six-sample domain such
    as Dataset300's KiCad subset from silently disappearing from validation and
    test while retaining exact global ratios whenever group sizes permit them.
    """

    enabled = [name for name, ratio in ratio_items if ratio > 0]
    quotas: dict[str, dict[str, int]] = {}
    ideals: dict[str, dict[str, float]] = {}
    minimums: dict[str, dict[str, int]] = {}
    for source, total in sorted(source_totals.items()):
        ideal = {name: ratio * total for name, ratio in ratio_items}
        row = _integer_targets(total, ratio_items)
        minimum = {
            name: int(total >= len(enabled) and name in enabled)
            for name, _ in ratio_items
        }
        if minimum and any(row[name] < minimum[name] for name in minimum):
            for receiver in [name for name in enabled if row[name] < minimum[name]]:
                donors = [
                    name
                    for name in enabled
                    if row[name] > minimum[name]
                ]
                if not donors:
                    break
                donor = min(
                    donors,
                    key=lambda name: (
                        (row[name] - 1 - ideal[name]) ** 2
                        - (row[name] - ideal[name]) ** 2,
                        enabled.index(name),
                    ),
                )
                row[donor] -= 1
                row[receiver] += 1
        quotas[source] = row
        ideals[source] = ideal
        minimums[source] = minimum

    global_targets = _integer_targets(sum(source_totals.values()), ratio_items)
    column_totals = {
        name: sum(row[name] for row in quotas.values()) for name, _ in ratio_items
    }
    # Moving one sample preserves every row total.  Squared-distance cost keeps
    # the resulting matrix as close as possible to proportional stratification.
    while column_totals != global_targets:
        excess = [name for name in enabled if column_totals[name] > global_targets[name]]
        deficit = [name for name in enabled if column_totals[name] < global_targets[name]]
        if not excess or not deficit:
            break
        moves: list[tuple[float, str, str, str]] = []
        for source, row in quotas.items():
            for donor in excess:
                if row[donor] <= minimums[source][donor]:
                    continue
                for receiver in deficit:
                    before = (
                        (row[donor] - ideals[source][donor]) ** 2
                        + (row[receiver] - ideals[source][receiver]) ** 2
                    )
                    after = (
                        (row[donor] - 1 - ideals[source][donor]) ** 2
                        + (row[receiver] + 1 - ideals[source][receiver]) ** 2
                    )
                    moves.append((after - before, source, donor, receiver))
        if not moves:
            break
        _, source, donor, receiver = min(moves)
        quotas[source][donor] -= 1
        quotas[source][receiver] += 1
        column_totals[donor] -= 1
        column_totals[receiver] += 1
    return quotas


def _validate_ratios(ratios: Mapping[str, float]) -> list[tuple[str, float]]:
    if not ratios:
        raise ValueError("at least one split ratio is required")
    result: list[tuple[str, float]] = []
    for name, raw_value in ratios.items():
        if not isinstance(name, str) or not name:
            raise ValueError("split names must be non-empty strings")
        value = float(raw_value)
        if value < 0:
            raise ValueError(f"split ratio for {name!r} cannot be negative")
        result.append((name, value))
    total = sum(value for _, value in result)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {total:.12g}")
    if not any(value > 0 for _, value in result):
        raise ValueError("at least one split ratio must be positive")
    return result


def assign_grouped_splits(
    records: list[dict[str, Any]],
    ratios: Mapping[str, float] | None = None,
    seed: str = "pcbparse-v1",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign deterministic splits without leaking JSON/image duplicates.

    Duplicate components are recomputed from SHA-256 values, so stale or
    user-edited ``duplicate_group`` fields cannot accidentally break grouping.
    """

    ratio_items = _validate_ratios(ratios or DEFAULT_SPLIT_RATIOS)
    output = copy.deepcopy(records)
    duplicate_groups = apply_duplicate_groups(output)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in output:
        grouped[record["duplicate_group"]].append(record)

    total_samples = len(output)
    sample_counts = {name: 0 for name, _ in ratio_items}
    group_counts = {name: 0 for name, _ in ratio_items}

    def group_order(group_id: str) -> tuple[str, str]:
        digest = hashlib.sha256(f"{seed}\0{group_id}".encode("utf-8")).hexdigest()
        return digest, group_id

    group_sources: dict[str, str] = {}
    source_totals: Counter[str] = Counter()
    source_groups: dict[str, list[str]] = defaultdict(list)
    for group_id, members in grouped.items():
        sources = {str(record.get("source_class", "unknown")) for record in members}
        # Duplicate groups crossing domains are kept intact and assigned to a
        # stable compound stratum instead of being split or guessed apart.
        source = "+".join(sorted(sources))
        group_sources[group_id] = source
        source_totals[source] += len(members)
        source_groups[source].append(group_id)

    source_targets = _stratified_targets(source_totals, ratio_items)
    for source in sorted(source_groups):
        assigned = {name: 0 for name, _ in ratio_items}
        # Large duplicate groups are placed first; the seed still determines
        # membership among equal-sized groups and therefore the concrete split.
        ordered_groups = sorted(
            source_groups[source],
            key=lambda group_id: (-len(grouped[group_id]), group_order(group_id)),
        )
        for group_id in ordered_groups:
            group_size = len(grouped[group_id])
            candidates: list[tuple[int, float, int, str]] = []
            for position, (name, ratio) in enumerate(ratio_items):
                if ratio <= 0:
                    continue
                remaining = source_targets[source][name] - assigned[name]
                overflow = max(0, group_size - remaining)
                target = source_targets[source][name]
                projected_fill = (assigned[name] + group_size) / max(target, 1)
                candidates.append((overflow, projected_fill, position, name))
            _, _, _, chosen = min(candidates)
            assigned[chosen] += group_size
            sample_counts[chosen] += group_size
            group_counts[chosen] += 1
            for record in grouped[group_id]:
                record["split"] = chosen

    output.sort(key=lambda record: record["sample_id"])
    validate_grouped_splits(output)

    class_counts: dict[str, Counter[str]] = {
        name: Counter() for name, _ in ratio_items
    }
    for record in output:
        class_counts[record["split"]][str(record.get("source_class", "unknown"))] += 1

    summary: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "total_samples": total_samples,
        "total_groups": len(grouped),
        "duplicate_groups": len(duplicate_groups),
        "stratification": "source_class_grouped_hamilton_v2",
        "leakage_check": "passed",
        "splits": {},
    }
    for name, requested_ratio in ratio_items:
        count = sample_counts[name]
        summary["splits"][name] = {
            "requested_ratio": requested_ratio,
            "samples": count,
            "groups": group_counts[name],
            "effective_ratio": count / total_samples if total_samples else 0.0,
            "source_classes": dict(sorted(class_counts[name].items())),
        }
    return output, summary


def validate_grouped_splits(records: list[dict[str, Any]]) -> None:
    """Raise when one duplicate group appears in more than one split."""

    seen: dict[str, str] = {}
    for record in records:
        group_id = record.get("duplicate_group")
        split = record.get("split")
        if not isinstance(group_id, str) or not isinstance(split, str):
            raise ValueError("every split record needs duplicate_group and split")
        previous = seen.setdefault(group_id, split)
        if previous != split:
            raise ValueError(
                f"duplicate group {group_id!r} leaks across {previous!r} and {split!r}"
            )
