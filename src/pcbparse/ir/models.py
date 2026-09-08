from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class IRState(StrEnum):
    RAW = "RAW"
    PERCEIVED = "PERCEIVED"
    ASSOCIATED = "ASSOCIATED"
    GRAPHED = "GRAPHED"
    SEMANTIC = "SEMANTIC"
    VALIDATED = "VALIDATED"
    SERIALIZED = "SERIALIZED"


class Point(StrictModel):
    x: float
    y: float


class BBox(StrictModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def validate_order(self) -> "BBox":
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("bbox must satisfy x1 <= x2 and y1 <= y2")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Point:
        return Point(x=(self.x1 + self.x2) / 2.0, y=(self.y1 + self.y2) / 2.0)


class Evidence(StrictModel):
    evidence_id: str
    source: Literal["dataset", "detector", "ocr", "geometry", "qwen", "rule", "repair"]
    view_id: str = "full"
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    payload_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageInfo(StrictModel):
    image_id: str
    path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sha256: str = ""
    source_domain: str = "unknown"
    actual_format: str | None = None


class Component(StrictModel):
    internal_id: str
    key: str | None = None
    category: str
    name: str = ""
    value: str = ""
    bbox: BBox
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    key_score: float = Field(default=0.0, ge=0.0, le=1.0)
    rotation: int | None = None
    source_object_id: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class Pin(StrictModel):
    internal_id: str
    component_id: str | None = None
    number: str | None = None
    name: str = ""
    point: Point
    base: Point | None = None
    orientation: Literal["L", "R", "T", "B", "UNKNOWN"] = "UNKNOWN"
    edge_ids: list[str] = Field(default_factory=list)
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    source_object_id: str | None = None
    orphan: bool = False
    evidence: list[Evidence] = Field(default_factory=list)


class TextToken(StrictModel):
    internal_id: str
    raw_text: str
    candidates: list[str] = Field(default_factory=list)
    resolved_text: str | None = None
    category: str = "unknown"
    details: str = ""
    bbox: BBox
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    source_object_id: str | None = None
    component_id: str | None = None
    pin_id: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class WireSegment(StrictModel):
    internal_id: str
    start: Point
    end: Point
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    source_edge_id: str | None = None
    source_polyline_id: str | None = None
    synthetic: bool = False
    channel: str = "default"
    evidence: list[Evidence] = Field(default_factory=list)


class Junction(StrictModel):
    internal_id: str
    point: Point
    kind: Literal[
        "endpoint",
        "t_junction",
        "connected_cross",
        "non_connected_cross",
        "bend",
        "noise",
        "ambiguous",
    ]
    segment_ids: list[str] = Field(default_factory=list)
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)


class NetLabel(StrictModel):
    internal_id: str
    instance_text: str
    resolved_text: str
    net_type: Literal["SIGNAL", "POWER", "GROUND", "UNKNOWN"] = "UNKNOWN"
    point: Point | None = None
    attached_physical_net_id: str | None = None
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)


class PhysicalNet(StrictModel):
    internal_id: str
    pins: list[str] = Field(default_factory=list)
    segments: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    repair_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    score: float = Field(default=1.0, ge=0.0, le=1.0)


class ElectricalNet(StrictModel):
    internal_id: str
    pins: list[str] = Field(default_factory=list)
    segments: list[str] = Field(default_factory=list)
    label: str | None = None
    net_type: Literal["SIGNAL", "POWER", "GROUND", "UNKNOWN"] = "UNKNOWN"
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    physical_net_ids: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)


class Diagnostic(StrictModel):
    severity: Literal["ERROR", "WARNING", "INFO"]
    code: str
    message: str
    object_ids: list[str] = Field(default_factory=list)
    stage: str = "unknown"
    repair_id: str | None = None


class RepairRecord(StrictModel):
    repair_id: str
    rule_id: str
    object_ids: list[str] = Field(default_factory=list)
    before: Any = None
    after: Any = None
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    config_hash: str = ""


class CircuitIR(StrictModel):
    schema_version: str = "1.0"
    state: IRState = IRState.RAW
    image: ImageInfo
    transforms: list[dict[str, Any]] = Field(default_factory=list)
    components: list[Component] = Field(default_factory=list)
    pins: list[Pin] = Field(default_factory=list)
    texts: list[TextToken] = Field(default_factory=list)
    segments: list[WireSegment] = Field(default_factory=list)
    junctions: list[Junction] = Field(default_factory=list)
    net_labels: list[NetLabel] = Field(default_factory=list)
    physical_nets: list[PhysicalNet] = Field(default_factory=list)
    electrical_nets: list[ElectricalNet] = Field(default_factory=list)
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    repairs: list[RepairRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def component_index(self) -> dict[str, Component]:
        return {component.internal_id: component for component in self.components}

    def pin_index(self) -> dict[str, Pin]:
        return {pin.internal_id: pin for pin in self.pins}

    def segment_index(self) -> dict[str, WireSegment]:
        return {segment.internal_id: segment for segment in self.segments}

