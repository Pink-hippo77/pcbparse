from __future__ import annotations

import copy
import math
from typing import Any, Sequence

from pcbparse.io import ImageView
from pcbparse.ir.models import BBox, Evidence, Point


LOCAL_TO_GLOBAL_BINDING_KEY = "local_to_global_binding"
LOCAL_TO_GLOBAL_BINDING_CONTRACT = "image_view_local_to_original_v1"


def bind_bbox_evidence(
    evidence: Sequence[Evidence],
    *,
    view: ImageView,
    local_bbox: BBox,
    original_bbox: BBox,
    object_kind: str,
) -> None:
    _attach(
        evidence,
        view=view,
        object_kind=object_kind,
        local_geometry={"kind": "bbox_xyxy", "values": _bbox_values(local_bbox)},
        original_geometry={
            "kind": "bbox_xyxy",
            "values": _bbox_values(original_bbox),
        },
    )


def bind_point_evidence(
    evidence: Sequence[Evidence],
    *,
    view: ImageView,
    local_point: Point,
    original_point: Point,
    object_kind: str,
    local_base: Point | None = None,
    original_base: Point | None = None,
) -> None:
    local_geometry: dict[str, Any] = {
        "kind": "point",
        "point_xy": _point_values(local_point),
    }
    original_geometry: dict[str, Any] = {
        "kind": "point",
        "point_xy": _point_values(original_point),
    }
    if local_base is not None or original_base is not None:
        if local_base is None or original_base is None:
            raise ValueError("local/original base coordinates must be present together")
        local_geometry["base_xy"] = _point_values(local_base)
        original_geometry["base_xy"] = _point_values(original_base)
    _attach(
        evidence,
        view=view,
        object_kind=object_kind,
        local_geometry=local_geometry,
        original_geometry=original_geometry,
    )


def bind_segment_evidence(
    evidence: Sequence[Evidence],
    *,
    view: ImageView,
    local_start: Point,
    local_end: Point,
    original_start: Point,
    original_end: Point,
) -> None:
    _attach(
        evidence,
        view=view,
        object_kind="segment",
        local_geometry={
            "kind": "segment",
            "start_xy": _point_values(local_start),
            "end_xy": _point_values(local_end),
        },
        original_geometry={
            "kind": "segment",
            "start_xy": _point_values(original_start),
            "end_xy": _point_values(original_end),
        },
    )


def _attach(
    evidence: Sequence[Evidence],
    *,
    view: ImageView,
    object_kind: str,
    local_geometry: dict[str, Any],
    original_geometry: dict[str, Any],
) -> None:
    if not evidence:
        raise ValueError(f"{object_kind} has no evidence to bind")
    if not isinstance(object_kind, str) or not object_kind:
        raise ValueError("object_kind must be non-empty")
    binding = {
        "contract": LOCAL_TO_GLOBAL_BINDING_CONTRACT,
        "object_kind": object_kind,
        "view_id": view.view_id,
        "view_size": [view.width, view.height],
        "local_to_original_matrix": _local_to_original_matrix(view),
        "local_geometry": local_geometry,
        "original_geometry": original_geometry,
    }
    for item in evidence:
        if item.view_id != view.view_id:
            raise ValueError(
                f"evidence {item.evidence_id} view {item.view_id!r} differs from "
                f"coordinate binding view {view.view_id!r}"
            )
        item.metadata[LOCAL_TO_GLOBAL_BINDING_KEY] = copy.deepcopy(binding)


def _local_to_original_matrix(view: ImageView) -> list[list[float]]:
    origin = view.to_original(Point(x=0.0, y=0.0))
    unit_x = view.to_original(Point(x=1.0, y=0.0))
    unit_y = view.to_original(Point(x=0.0, y=1.0))
    matrix = [
        [unit_x.x - origin.x, unit_y.x - origin.x, origin.x],
        [unit_x.y - origin.y, unit_y.y - origin.y, origin.y],
        [0.0, 0.0, 1.0],
    ]
    values = [float(value) for row in matrix for value in row]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"view {view.view_id} has a non-finite coordinate transform")
    return [[float(value) for value in row] for row in matrix]


def _point_values(point: Point) -> list[float]:
    values = [float(point.x), float(point.y)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("coordinate binding point is non-finite")
    return values


def _bbox_values(bbox: BBox) -> list[float]:
    values = [float(bbox.x1), float(bbox.y1), float(bbox.x2), float(bbox.y2)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("coordinate binding bbox is non-finite")
    return values


__all__ = [
    "LOCAL_TO_GLOBAL_BINDING_CONTRACT",
    "LOCAL_TO_GLOBAL_BINDING_KEY",
    "bind_bbox_evidence",
    "bind_point_evidence",
    "bind_segment_evidence",
]
