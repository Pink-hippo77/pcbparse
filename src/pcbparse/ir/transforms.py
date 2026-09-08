from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pcbparse.ir.models import BBox, Point


@dataclass(frozen=True, slots=True)
class AffineTransform:
    """A 3x3 homogeneous transform mapping source-view to target-view coordinates."""

    name: str
    matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    source_view: str
    target_view: str

    def as_array(self) -> np.ndarray:
        array = np.asarray(self.matrix, dtype=np.float64)
        if array.shape != (3, 3):
            raise ValueError("transform matrix must be 3x3")
        if abs(float(np.linalg.det(array))) < 1e-12:
            raise ValueError("transform matrix must be invertible")
        return array

    def inverse(self) -> "AffineTransform":
        matrix = np.linalg.inv(self.as_array())
        return AffineTransform(
            name=f"inverse({self.name})",
            matrix=_as_tuple(matrix),
            source_view=self.target_view,
            target_view=self.source_view,
        )

    def apply_point(self, point: Point) -> Point:
        result = self.as_array() @ np.asarray([point.x, point.y, 1.0], dtype=np.float64)
        if abs(float(result[2])) < 1e-12:
            raise ValueError("homogeneous coordinate collapsed to zero")
        return Point(x=float(result[0] / result[2]), y=float(result[1] / result[2]))

    def apply_bbox(self, bbox: BBox) -> BBox:
        corners = [
            self.apply_point(Point(x=bbox.x1, y=bbox.y1)),
            self.apply_point(Point(x=bbox.x2, y=bbox.y1)),
            self.apply_point(Point(x=bbox.x1, y=bbox.y2)),
            self.apply_point(Point(x=bbox.x2, y=bbox.y2)),
        ]
        xs = [point.x for point in corners]
        ys = [point.y for point in corners]
        return BBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))

    @classmethod
    def identity(cls, view_id: str = "original") -> "AffineTransform":
        return cls(
            name="identity",
            matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            source_view=view_id,
            target_view=view_id,
        )

    @classmethod
    def scale_translate(
        cls,
        *,
        scale_x: float,
        scale_y: float,
        offset_x: float,
        offset_y: float,
        source_view: str,
        target_view: str,
        name: str = "scale_translate",
    ) -> "AffineTransform":
        return cls(
            name=name,
            matrix=((scale_x, 0.0, offset_x), (0.0, scale_y, offset_y), (0.0, 0.0, 1.0)),
            source_view=source_view,
            target_view=target_view,
        )


@dataclass(slots=True)
class TransformChain:
    transforms: list[AffineTransform] = field(default_factory=list)

    def append(self, transform: AffineTransform) -> None:
        if self.transforms and self.transforms[-1].target_view != transform.source_view:
            raise ValueError("transform chain views are not contiguous")
        self.transforms.append(transform)

    def apply(self, point: Point) -> Point:
        result = point
        for transform in self.transforms:
            result = transform.apply_point(result)
        return result

    def to_original(self, point: Point) -> Point:
        result = point
        for transform in reversed(self.transforms):
            result = transform.inverse().apply_point(result)
        return result

    def max_roundtrip_error(self, width: int, height: int, step: int = 64) -> float:
        points = [
            Point(x=float(x), y=float(y))
            for x in range(0, width + 1, step)
            for y in range(0, height + 1, step)
        ]
        points.extend([Point(x=float(width), y=float(height)), Point(x=0.0, y=0.0)])
        errors = []
        for point in points:
            restored = self.to_original(self.apply(point))
            errors.append(float(np.hypot(restored.x - point.x, restored.y - point.y)))
        return max(errors, default=0.0)

    def to_records(self) -> list[dict[str, object]]:
        return [
            {
                "name": transform.name,
                "matrix": transform.matrix,
                "source_view": transform.source_view,
                "target_view": transform.target_view,
            }
            for transform in self.transforms
        ]


def _as_tuple(matrix: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    return tuple(tuple(float(value) for value in row) for row in matrix)  # type: ignore[return-value]

