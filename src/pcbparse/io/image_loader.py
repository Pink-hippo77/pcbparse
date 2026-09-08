from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from PIL import Image

from pcbparse.ir.models import BBox, Component, ImageInfo, Point
from pcbparse.ir.transforms import AffineTransform, TransformChain


class ImageLoadError(RuntimeError):
    """Raised when an input is not a decodable raster image."""


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class ImageView:
    """A raster view whose coordinates can be mapped back to the input image.

    ``transform`` maps original-image coordinates into this view.  Calling
    ``transform.to_original`` therefore restores provider output to the
    canonical image coordinate system.
    """

    view_id: str
    image: Image.Image
    transform: TransformChain
    source_path: Path | None = None
    actual_format: str | None = None
    sha256: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return int(self.image.width)

    @property
    def height(self) -> int:
        return int(self.image.height)

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    def to_original(self, point: Point) -> Point:
        return self.transform.to_original(point)

    def to_view(self, point: Point) -> Point:
        return self.transform.apply(point)

    def image_info(
        self,
        *,
        image_id: str | None = None,
        source_domain: str = "unknown",
    ) -> ImageInfo:
        if self.source_path is None:
            path = ""
            default_id = self.view_id
        else:
            path = str(self.source_path.resolve())
            default_id = self.source_path.stem
        return ImageInfo(
            image_id=image_id or default_id,
            path=path,
            width=self.width,
            height=self.height,
            sha256=self.sha256,
            source_domain=source_domain,
            actual_format=self.actual_format,
        )


@dataclass(slots=True)
class ComponentView:
    """The full component ROI and optional side strips used by pin models."""

    component_id: str
    component_bbox: BBox
    full_roi: ImageView
    side_strips: Mapping[str, ImageView] = field(default_factory=dict)

    @property
    def transform(self) -> TransformChain:
        return self.full_roi.transform


def detect_image_format(path: str | Path) -> str:
    """Return Pillow's magic-byte format, independent of the file suffix."""

    image_path = Path(path)
    try:
        with Image.open(image_path) as image:
            actual_format = image.format
            # Force decoder initialization so truncated or spoofed files fail
            # here rather than much later in a provider.
            image.verify()
    except (OSError, ValueError) as exc:
        raise ImageLoadError(f"cannot decode image {image_path}: {exc}") from exc
    if not actual_format:
        raise ImageLoadError(f"Pillow could not identify image format for {image_path}")
    return str(actual_format).upper()


def load_image(path: str | Path, *, view_id: str = "original") -> ImageView:
    """Decode an image with Pillow and create an identity-coordinate view.

    Pillow selects a decoder from the file signature, so a JPEG or WEBP named
    ``*.png`` is handled correctly.  EXIF orientation is deliberately not
    applied: Dataset300 annotations use the stored raster coordinates.
    """

    image_path = Path(path)
    if not image_path.is_file():
        raise ImageLoadError(f"image does not exist: {image_path}")
    try:
        with Image.open(image_path) as opened:
            actual_format = opened.format
            opened.load()
            image = opened.copy()
    except (OSError, ValueError) as exc:
        raise ImageLoadError(f"cannot decode image {image_path}: {exc}") from exc
    if not actual_format:
        raise ImageLoadError(f"Pillow could not identify image format for {image_path}")
    chain = TransformChain([AffineTransform.identity(view_id)])
    return ImageView(
        view_id=view_id,
        image=image,
        transform=chain,
        source_path=image_path.resolve(),
        actual_format=str(actual_format).upper(),
        sha256=_sha256(image_path),
        metadata={"filename_suffix": image_path.suffix.lower()},
    )


def crop_image_view(
    parent: ImageView,
    bbox: BBox,
    *,
    view_id: str,
) -> ImageView:
    """Crop a child view while extending the original-to-view transform."""

    left = max(0, min(parent.width, math.floor(bbox.x1)))
    top = max(0, min(parent.height, math.floor(bbox.y1)))
    right = max(left, min(parent.width, math.ceil(bbox.x2)))
    bottom = max(top, min(parent.height, math.ceil(bbox.y2)))
    if right <= left or bottom <= top:
        raise ValueError("crop bbox must have non-zero area inside the parent view")

    image = parent.image.crop((left, top, right, bottom))
    transforms = list(parent.transform.transforms)
    transforms.append(
        AffineTransform.scale_translate(
            scale_x=1.0,
            scale_y=1.0,
            offset_x=-float(left),
            offset_y=-float(top),
            source_view=parent.view_id,
            target_view=view_id,
            name="crop",
        )
    )
    return ImageView(
        view_id=view_id,
        image=image,
        transform=TransformChain(transforms),
        source_path=parent.source_path,
        actual_format=parent.actual_format,
        sha256=parent.sha256,
        metadata={
            **parent.metadata,
            "parent_view_id": parent.view_id,
            "crop_box": [left, top, right, bottom],
        },
    )


def make_component_view(
    image_view: ImageView,
    component: Component,
    *,
    padding: float = 0.0,
    strip_fraction: float = 0.2,
    view_id_prefix: str = "component",
) -> ComponentView:
    """Create a padded component ROI and four pin-side strips.

    Component bboxes are canonical original-image coordinates.  They are first
    mapped into ``image_view`` and every derived crop keeps an invertible chain.
    """

    if padding < 0:
        raise ValueError("padding cannot be negative")
    if not 0 < strip_fraction <= 1:
        raise ValueError("strip_fraction must be in (0, 1]")
    if not isinstance(view_id_prefix, str) or not view_id_prefix.strip():
        raise ValueError("view_id_prefix must be a non-empty string")

    corners = [
        image_view.to_view(Point(x=component.bbox.x1, y=component.bbox.y1)),
        image_view.to_view(Point(x=component.bbox.x2, y=component.bbox.y1)),
        image_view.to_view(Point(x=component.bbox.x1, y=component.bbox.y2)),
        image_view.to_view(Point(x=component.bbox.x2, y=component.bbox.y2)),
    ]
    xs = [point.x for point in corners]
    ys = [point.y for point in corners]
    roi_bbox = BBox(
        x1=min(xs) - padding,
        y1=min(ys) - padding,
        x2=max(xs) + padding,
        y2=max(ys) + padding,
    )
    full_id = f"{view_id_prefix}:{component.internal_id}:full"
    full = crop_image_view(image_view, roi_bbox, view_id=full_id)

    strip_width = max(1.0, full.width * strip_fraction)
    strip_height = max(1.0, full.height * strip_fraction)
    strip_boxes = {
        "L": BBox(x1=0, y1=0, x2=strip_width, y2=full.height),
        "R": BBox(x1=full.width - strip_width, y1=0, x2=full.width, y2=full.height),
        "T": BBox(x1=0, y1=0, x2=full.width, y2=strip_height),
        "B": BBox(x1=0, y1=full.height - strip_height, x2=full.width, y2=full.height),
    }
    strips = {
        side: crop_image_view(full, box, view_id=f"{full_id}:{side}")
        for side, box in strip_boxes.items()
    }
    return ComponentView(
        component_id=component.internal_id,
        component_bbox=component.bbox,
        full_roi=full,
        side_strips=strips,
    )


def tile_image_view(
    image_view: ImageView,
    *,
    tile_size: int = 1024,
    overlap: int = 160,
) -> list[ImageView]:
    """Create deterministic overlapping tiles with invertible coordinates."""

    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    if image_view.width <= tile_size and image_view.height <= tile_size:
        return [image_view]
    stride = tile_size - overlap

    def starts(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        values = list(range(0, max(1, length - tile_size + 1), stride))
        last = length - tile_size
        if values[-1] != last:
            values.append(last)
        return values

    views: list[ImageView] = []
    for row, top in enumerate(starts(image_view.height)):
        for column, left in enumerate(starts(image_view.width)):
            right = min(image_view.width, left + tile_size)
            bottom = min(image_view.height, top + tile_size)
            views.append(
                crop_image_view(
                    image_view,
                    BBox(x1=left, y1=top, x2=right, y2=bottom),
                    view_id=f"tile_{row:03d}_{column:03d}",
                )
            )
    return views


__all__ = [
    "ComponentView",
    "ImageLoadError",
    "ImageView",
    "crop_image_view",
    "detect_image_format",
    "load_image",
    "make_component_view",
    "tile_image_view",
]
