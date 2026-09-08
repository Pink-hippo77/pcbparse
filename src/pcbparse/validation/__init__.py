from __future__ import annotations

from collections.abc import Sequence

from pcbparse.ir.models import CircuitIR, Diagnostic
from pcbparse.validation.common import diagnostic_sort_key, stable_diagnostics
from pcbparse.validation.geometry import GeometryValidator, validate_geometry
from pcbparse.validation.graph import GraphValidator, validate_graph
from pcbparse.validation.references import ReferenceValidator, validate_references
from pcbparse.validation.schema import SchemaValidator, validate_schema


def default_validators() -> tuple[
    SchemaValidator,
    ReferenceValidator,
    GeometryValidator,
    GraphValidator,
]:
    return (
        SchemaValidator(),
        ReferenceValidator(),
        GeometryValidator(),
        GraphValidator(),
    )


def run_validators(
    ir: CircuitIR,
    validators: Sequence[object] | None = None,
    ctx: object | None = None,
) -> list[Diagnostic]:
    selected = validators or default_validators()
    return stable_diagnostics(
        issue
        for validator in selected
        for issue in validator.validate(ir, ctx)  # type: ignore[attr-defined]
    )


from pcbparse.validation.repair import (  # noqa: E402  (avoids import cycle)
    Validator,
    repair_ir_once,
    validate_and_repair_once,
)
from pcbparse.validation.api import validate_ir  # noqa: E402

__all__ = [
    "GeometryValidator",
    "GraphValidator",
    "ReferenceValidator",
    "SchemaValidator",
    "Validator",
    "default_validators",
    "diagnostic_sort_key",
    "repair_ir_once",
    "run_validators",
    "stable_diagnostics",
    "validate_and_repair_once",
    "validate_geometry",
    "validate_graph",
    "validate_ir",
    "validate_references",
    "validate_schema",
]
