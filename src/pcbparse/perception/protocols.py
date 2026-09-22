from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np

from pcbparse.io import ComponentView, ImageView
from pcbparse.ir.models import Component, Evidence, Junction, Pin, TextToken, WireSegment


ComponentCandidate: TypeAlias = Component
PinCandidate: TypeAlias = Pin


@dataclass(slots=True)
class WireMask:
    """A provider wire mask in the coordinate system of ``view_id``."""

    view_id: str
    mask: np.ndarray
    score: float = 1.0
    evidence: list[Evidence] = field(default_factory=list)


class ProviderError(RuntimeError):
    """Base typed failure returned by a perception provider."""


class ProviderTimeoutError(ProviderError):
    """A provider exceeded its configured inference deadline."""


@runtime_checkable
class ComponentProvider(Protocol):
    version: str

    def predict(self, image: ImageView, ctx: Any) -> list[ComponentCandidate]: ...


@runtime_checkable
class BatchedComponentProvider(ComponentProvider, Protocol):
    """Optional throughput extension for providers that support image batches."""

    def predict_many(
        self,
        images: list[ImageView],
        ctx: Any,
    ) -> list[list[ComponentCandidate]]: ...


@runtime_checkable
class OCRProvider(Protocol):
    version: str

    def recognize(self, views: list[ImageView], ctx: Any) -> list[TextToken]: ...


@runtime_checkable
class PinProvider(Protocol):
    version: str

    def predict(
        self,
        component_views: list[ComponentView],
        ctx: Any,
    ) -> list[PinCandidate]: ...


@runtime_checkable
class WireProvider(Protocol):
    version: str

    def segment(self, views: list[ImageView], ctx: Any) -> list[WireMask]: ...


@runtime_checkable
class JunctionProvider(Protocol):
    version: str

    def classify(
        self,
        views: list[ImageView],
        segments: list[WireSegment],
        ctx: Any,
    ) -> list[Junction]: ...


__all__ = [
    "ComponentCandidate",
    "ComponentProvider",
    "BatchedComponentProvider",
    "OCRProvider",
    "JunctionProvider",
    "PinCandidate",
    "PinProvider",
    "ProviderError",
    "ProviderTimeoutError",
    "WireMask",
    "WireProvider",
]
