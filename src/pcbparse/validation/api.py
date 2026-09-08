from __future__ import annotations

from pcbparse.ir.models import CircuitIR, Diagnostic, IRState
from pcbparse.validation import run_validators
from pcbparse.validation.common import merge_diagnostics
from pcbparse.validation.repair import validate_and_repair_once


def validate_ir(
    ir: CircuitIR,
    *,
    repair: bool = False,
    config_hash: str = "",
) -> tuple[CircuitIR, list[Diagnostic]]:
    """Validate an IR, optionally applying one conservative repair pass.

    The input is never mutated.  A successful SEMANTIC or already-VALIDATED IR
    is returned in VALIDATED state; any ERROR leaves its state unchanged.
    """

    if repair:
        return validate_and_repair_once(ir, config_hash=config_hash)

    result = ir.model_copy(deep=True)
    issues = merge_diagnostics(result.diagnostics, run_validators(result))
    result.diagnostics = issues
    has_errors = any(issue.severity == "ERROR" for issue in issues)
    if not has_errors:
        if result.state in {IRState.SEMANTIC, IRState.VALIDATED}:
            result.state = IRState.VALIDATED
    elif result.state in {IRState.VALIDATED, IRState.SERIALIZED}:
        result.state = IRState.SEMANTIC
    return result, issues
