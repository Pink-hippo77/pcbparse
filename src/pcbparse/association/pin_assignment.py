from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Mapping, Sequence


_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class PinAssignmentCandidate:
    """One scored text-to-pin edge used by the global assignment solver."""

    token_id: str
    pin_id: str
    component_id: str | None
    side: str
    score: float
    distance: float
    token_axis: float
    pin_axis: float
    explicit: bool = False
    score_components: Mapping[str, float] | None = None

    @property
    def group_key(self) -> tuple[str | None, str]:
        return self.component_id, self.side


@dataclass(frozen=True, slots=True)
class PinAssignmentResult:
    assignments: Mapping[str, str]
    pair_margins: Mapping[tuple[str, str], float]
    total_score: float
    next_best_score: float | None
    method: str


def solve_pin_assignment_group(
    candidates: Sequence[PinAssignmentCandidate],
    *,
    minimum_score: float,
    ordered_pin_threshold: int = 4,
) -> PinAssignmentResult:
    """Solve one component-side-field group as a deterministic 1:1 matching.

    A Hungarian assignment is used for ordinary groups.  Dense/high-pin side
    groups use an order-preserving dynamic program, which encodes the physical
    fact that labels along one component side should not cross one another.
    Each selected edge is then forbidden in turn.  The loss in the globally
    optimal objective is the edge's *global* best-vs-next margin.
    """

    eligible = tuple(
        candidate
        for candidate in candidates
        if math.isfinite(candidate.score) and candidate.score >= minimum_score
    )
    token_ids = tuple(sorted({item.token_id for item in eligible}))
    pin_ids = tuple(sorted({item.pin_id for item in eligible}))
    if not token_ids or not pin_ids:
        return PinAssignmentResult({}, {}, 0.0, None, "hungarian")

    ordered = (
        len(pin_ids) >= ordered_pin_threshold
        and len(token_ids) >= 2
        and {item.side for item in eligible}.issubset({"L", "R", "T", "B"})
    )
    method = "monotonic_dp" if ordered else "hungarian"
    solver: Callable[
        [frozenset[tuple[str, str]]], tuple[dict[str, str], float]
    ]
    if ordered:
        solver = lambda forbidden: _solve_monotonic(eligible, forbidden)
    else:
        solver = lambda forbidden: _solve_hungarian(eligible, forbidden)

    assignments, total_score = solver(frozenset())
    pair_margins: dict[tuple[str, str], float] = {}
    alternative_scores: list[float] = []
    for token_id, pin_id in sorted(assignments.items()):
        _, alternative_score = solver(frozenset({(token_id, pin_id)}))
        alternative_scores.append(alternative_score)
        pair_margins[(token_id, pin_id)] = max(0.0, total_score - alternative_score)

    return PinAssignmentResult(
        assignments=assignments,
        pair_margins=pair_margins,
        total_score=total_score,
        next_best_score=max(alternative_scores) if alternative_scores else None,
        method=method,
    )


def _candidate_lookup(
    candidates: Iterable[PinAssignmentCandidate],
    forbidden: frozenset[tuple[str, str]],
) -> dict[tuple[str, str], PinAssignmentCandidate]:
    lookup: dict[tuple[str, str], PinAssignmentCandidate] = {}
    for candidate in candidates:
        key = candidate.token_id, candidate.pin_id
        if key in forbidden:
            continue
        previous = lookup.get(key)
        if previous is None or _candidate_sort_key(candidate) < _candidate_sort_key(previous):
            lookup[key] = candidate
    return lookup


def _candidate_sort_key(candidate: PinAssignmentCandidate) -> tuple[float, float, str, str]:
    return (-candidate.score, candidate.distance, candidate.token_id, candidate.pin_id)


def _solve_hungarian(
    candidates: Sequence[PinAssignmentCandidate],
    forbidden: frozenset[tuple[str, str]],
) -> tuple[dict[str, str], float]:
    """Maximum-weight sparse bipartite matching with explicit dummy columns."""

    lookup = _candidate_lookup(candidates, forbidden)
    token_ids = tuple(sorted({item.token_id for item in candidates}))
    pin_ids = tuple(sorted({item.pin_id for item in candidates}))
    if not token_ids or not pin_ids:
        return {}, 0.0

    # There is one dummy column per token, so every row can remain unmatched.
    # The rectangular Hungarian implementation below requires rows <= columns.
    column_count = len(pin_ids) + len(token_ids)
    row_count = len(token_ids)
    impossible_cost = 1_000_000.0

    def cost(row: int, column: int) -> float:
        if column >= len(pin_ids):
            return 0.0
        edge = lookup.get((token_ids[row], pin_ids[column]))
        return -edge.score if edge is not None else impossible_cost

    # 1-indexed form of the O(n*m) shortest augmenting-path Hungarian method.
    u = [0.0] * (row_count + 1)
    v = [0.0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)
    for row in range(1, row_count + 1):
        matched_row[0] = row
        minimum = [math.inf] * (column_count + 1)
        used = [False] * (column_count + 1)
        column0 = 0
        while True:
            used[column0] = True
            row0 = matched_row[column0]
            delta = math.inf
            column1 = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                reduced = cost(row0 - 1, column - 1) - u[row0] - v[column]
                if reduced < minimum[column] - _EPSILON:
                    minimum[column] = reduced
                    predecessor[column] = column0
                if (
                    minimum[column] < delta - _EPSILON
                    or abs(minimum[column] - delta) <= _EPSILON
                    and (column1 == 0 or column < column1)
                ):
                    delta = minimum[column]
                    column1 = column
            for column in range(column_count + 1):
                if used[column]:
                    u[matched_row[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if matched_row[column0] == 0:
                break
        while True:
            column1 = predecessor[column0]
            matched_row[column0] = matched_row[column1]
            column0 = column1
            if column0 == 0:
                break

    assignments: dict[str, str] = {}
    total = 0.0
    for column in range(1, len(pin_ids) + 1):
        row = matched_row[column]
        if row == 0:
            continue
        token_id = token_ids[row - 1]
        pin_id = pin_ids[column - 1]
        edge = lookup.get((token_id, pin_id))
        if edge is not None:
            assignments[token_id] = pin_id
            total += edge.score
    return assignments, total


def _solve_monotonic(
    candidates: Sequence[PinAssignmentCandidate],
    forbidden: frozenset[tuple[str, str]],
) -> tuple[dict[str, str], float]:
    """Maximum-weight non-crossing matching for one physical component side."""

    lookup = _candidate_lookup(candidates, forbidden)
    token_axis = _minimum_axis(candidates, "token")
    pin_axis = _minimum_axis(candidates, "pin")
    token_ids = tuple(sorted(token_axis, key=lambda item: (token_axis[item], item)))
    pin_ids = tuple(sorted(pin_axis, key=lambda item: (pin_axis[item], item)))
    rows, columns = len(token_ids), len(pin_ids)
    scores = [[0.0] * (columns + 1) for _ in range(rows + 1)]
    actions = [[""] * (columns + 1) for _ in range(rows + 1)]

    for row in range(1, rows + 1):
        actions[row][0] = "skip_token"
    for column in range(1, columns + 1):
        actions[0][column] = "skip_pin"

    # Fixed action priority provides deterministic reconstruction at exact ties.
    priority = {"match": 0, "skip_pin": 1, "skip_token": 2}
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            options = [
                (scores[row - 1][column], "skip_token"),
                (scores[row][column - 1], "skip_pin"),
            ]
            edge = lookup.get((token_ids[row - 1], pin_ids[column - 1]))
            if edge is not None:
                options.append((scores[row - 1][column - 1] + edge.score, "match"))
            best_score, best_action = min(
                options,
                key=lambda item: (-item[0], priority[item[1]]),
            )
            scores[row][column] = best_score
            actions[row][column] = best_action

    assignments: dict[str, str] = {}
    row, column = rows, columns
    while row > 0 or column > 0:
        action = actions[row][column]
        if action == "match":
            assignments[token_ids[row - 1]] = pin_ids[column - 1]
            row -= 1
            column -= 1
        elif action == "skip_token":
            row -= 1
        else:
            column -= 1
    return assignments, scores[rows][columns]


def _minimum_axis(
    candidates: Sequence[PinAssignmentCandidate],
    kind: str,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for candidate in candidates:
        object_id = candidate.token_id if kind == "token" else candidate.pin_id
        axis = candidate.token_axis if kind == "token" else candidate.pin_axis
        result[object_id] = min(result.get(object_id, math.inf), axis)
    return result


__all__ = [
    "PinAssignmentCandidate",
    "PinAssignmentResult",
    "solve_pin_assignment_group",
]
