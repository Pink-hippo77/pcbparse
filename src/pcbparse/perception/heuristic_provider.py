from __future__ import annotations

import hashlib
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pcbparse.ir.models import CircuitIR, Evidence, IRState, ImageInfo, Point, WireSegment
from pcbparse.pipeline.context import RunContext


class HeuristicVisionProvider:
    """Dependency-light image-only baseline for wire candidates.

    This provider intentionally does not pretend to solve component/OCR/pin
    recognition without learned weights. It extracts long, saturated/thin dark
    axis-aligned runs as evidence-bearing wire segments, giving the production
    pipeline a real image-only fast path and an upgrade seam for trained models.
    """

    version = "heuristic-wire-v1"

    def __init__(self, *, minimum_run_length_px: int = 12, maximum_gap_px: int = 1) -> None:
        if minimum_run_length_px < 2:
            raise ValueError("minimum_run_length_px must be at least 2")
        self.minimum_run_length_px = minimum_run_length_px
        self.maximum_gap_px = max(0, maximum_gap_px)

    def build_ir(self, input_path: str | Path, ctx: RunContext | None = None) -> CircuitIR:
        path = Path(input_path)
        with Image.open(path) as image:
            actual_format = image.format
            rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        candidate_mask = _wire_candidate_mask(rgb)
        horizontal = _axis_runs(candidate_mask, axis=1, minimum=self.minimum_run_length_px, max_gap=self.maximum_gap_px)
        vertical = _axis_runs(candidate_mask, axis=0, minimum=self.minimum_run_length_px, max_gap=self.maximum_gap_px)
        segments: list[WireSegment] = []
        for index, (start, end, orientation) in enumerate([*horizontal, *vertical], start=1):
            segments.append(
                WireSegment(
                    internal_id=f"heuristic-segment:{index:06d}",
                    start=start,
                    end=end,
                    score=0.35,
                    source_polyline_id=f"heuristic:{orientation}:{index}",
                    channel=orientation,
                    evidence=[
                        Evidence(
                            evidence_id=f"heuristic:wire:{index:06d}",
                            source="detector",
                            view_id="original",
                            score=0.35,
                            metadata={
                                "algorithm": self.version,
                                "minimum_run_length_px": self.minimum_run_length_px,
                            },
                        )
                    ],
                )
            )
        return CircuitIR(
            state=IRState.PERCEIVED,
            image=ImageInfo(
                image_id=ctx.image_id if ctx else path.stem,
                path=str(path),
                width=width,
                height=height,
                sha256=digest,
                actual_format=actual_format,
            ),
            segments=segments,
            metadata={
                "provider": {"name": type(self).__name__, "version": self.version},
                "limitations": [
                    "No learned component detector is installed.",
                    "No OCR or pin keypoint weights are installed.",
                    "Wire candidates are low-confidence and axis-aligned only.",
                ],
            },
        )


def _wire_candidate_mask(rgb: np.ndarray) -> np.ndarray:
    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    green_wire = (green >= red + 18) & (green >= blue + 18) & (green < 235)
    blue_wire = (blue >= red + 22) & (blue >= green + 8) & (blue < 235)
    dark = (red < 95) & (green < 95) & (blue < 95)
    return green_wire | blue_wire | dark


def _axis_runs(
    mask: np.ndarray,
    *,
    axis: int,
    minimum: int,
    max_gap: int,
) -> list[tuple[Point, Point, str]]:
    view = mask if axis == 1 else mask.T
    orientation = "horizontal" if axis == 1 else "vertical"
    runs: list[tuple[Point, Point, str]] = []
    for row_index, row in enumerate(view):
        start: int | None = None
        last_true: int | None = None
        gap = 0
        for column_index, value in enumerate(row):
            if bool(value):
                if start is None:
                    start = column_index
                last_true = column_index
                gap = 0
            elif start is not None:
                gap += 1
                if gap > max_gap:
                    if last_true is not None and last_true - start + 1 >= minimum:
                        runs.append(_run_to_points(row_index, start, last_true, axis, orientation))
                    start = None
                    last_true = None
                    gap = 0
        if start is not None and last_true is not None and last_true - start + 1 >= minimum:
            runs.append(_run_to_points(row_index, start, last_true, axis, orientation))
    # Deduplicate adjacent scanlines representing a thick stroke by retaining
    # only the centerline of identical/similar runs.
    deduped: list[tuple[Point, Point, str]] = []
    for run in runs:
        if deduped and _parallel_overlap(deduped[-1], run, axis):
            previous = deduped[-1]
            center = _merge_parallel(previous, run, axis, orientation)
            deduped[-1] = center
        else:
            deduped.append(run)
    return deduped


def _run_to_points(row: int, start: int, end: int, axis: int, orientation: str) -> tuple[Point, Point, str]:
    if axis == 1:
        return Point(x=float(start), y=float(row)), Point(x=float(end), y=float(row)), orientation
    return Point(x=float(row), y=float(start)), Point(x=float(row), y=float(end)), orientation


def _parallel_overlap(
    first: tuple[Point, Point, str],
    second: tuple[Point, Point, str],
    axis: int,
) -> bool:
    if axis == 1:
        adjacent = abs(first[0].y - second[0].y) <= 1.0
        overlap = min(first[1].x, second[1].x) - max(first[0].x, second[0].x)
        span = min(first[1].x - first[0].x, second[1].x - second[0].x)
    else:
        adjacent = abs(first[0].x - second[0].x) <= 1.0
        overlap = min(first[1].y, second[1].y) - max(first[0].y, second[0].y)
        span = min(first[1].y - first[0].y, second[1].y - second[0].y)
    return adjacent and span > 0 and overlap / span >= 0.85


def _merge_parallel(
    first: tuple[Point, Point, str],
    second: tuple[Point, Point, str],
    axis: int,
    orientation: str,
) -> tuple[Point, Point, str]:
    if axis == 1:
        y = (first[0].y + second[0].y) / 2.0
        return Point(x=min(first[0].x, second[0].x), y=y), Point(x=max(first[1].x, second[1].x), y=y), orientation
    x = (first[0].x + second[0].x) / 2.0
    return Point(x=x, y=min(first[0].y, second[0].y)), Point(x=x, y=max(first[1].y, second[1].y)), orientation

