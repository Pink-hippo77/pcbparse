from __future__ import annotations

from pcbparse.ir.models import CircuitIR, IRState


_NEXT_STATE: dict[IRState, IRState] = {
    IRState.RAW: IRState.PERCEIVED,
    IRState.PERCEIVED: IRState.ASSOCIATED,
    IRState.ASSOCIATED: IRState.GRAPHED,
    IRState.GRAPHED: IRState.SEMANTIC,
    IRState.SEMANTIC: IRState.VALIDATED,
    IRState.VALIDATED: IRState.SERIALIZED,
}


def transition(ir: CircuitIR, target: IRState) -> CircuitIR:
    expected = _NEXT_STATE.get(ir.state)
    if expected != target:
        raise ValueError(f"invalid Circuit-IR transition: {ir.state} -> {target}; expected {expected}")
    ir.state = target
    return ir

