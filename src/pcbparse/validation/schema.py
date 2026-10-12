from __future__ import annotations

from collections import Counter, defaultdict

from pcbparse.ir.models import CircuitIR, Diagnostic
from pcbparse.output.official_schema import split_official_pin_numbers
from pcbparse.validation.common import error, stable_diagnostics


def _duplicate_ids(values: list[str]) -> set[str]:
    return {value for value, count in Counter(values).items() if count > 1}


def validate_schema(ir: CircuitIR) -> list[Diagnostic]:
    """Validate IR-level identity and official-output prerequisites.

    Pydantic already rejects unknown object fields and invalid field types.  This
    validator covers invariants spanning multiple objects, which cannot be
    expressed by an individual model.
    """

    issues: list[Diagnostic] = []
    collections = {
        "component": ir.components,
        "pin": ir.pins,
        "text": ir.texts,
        "segment": ir.segments,
        "junction": ir.junctions,
        "net_label": ir.net_labels,
        "physical_net": ir.physical_nets,
        "electrical_net": ir.electrical_nets,
    }
    all_ids: dict[str, list[str]] = defaultdict(list)
    for kind, objects in collections.items():
        ids = [obj.internal_id for obj in objects]
        for object_id in ids:
            all_ids[object_id].append(kind)
        for duplicate_id in sorted(_duplicate_ids(ids)):
            issues.append(
                error(
                    "SCHEMA.DUPLICATE_INTERNAL_ID",
                    f"duplicate {kind} internal_id: {duplicate_id}",
                    duplicate_id,
                    stage="schema",
                )
            )
        for obj in objects:
            if not obj.internal_id:
                issues.append(
                    error(
                        "SCHEMA.EMPTY_INTERNAL_ID",
                        f"{kind} internal_id must not be empty",
                        stage="schema",
                    )
                )

    for object_id, kinds in sorted(all_ids.items()):
        unique_kinds = sorted(set(kinds))
        if object_id and len(unique_kinds) > 1:
            issues.append(
                error(
                    "SCHEMA.CROSS_TYPE_ID_COLLISION",
                    f"internal_id is reused across object types: {unique_kinds!r}",
                    object_id,
                    stage="schema",
                )
            )

    for component in ir.components:
        if component.key is None or component.key == "":
            issues.append(
                error(
                    "COMPONENT.MISSING_KEY",
                    "component must have an official key before validation",
                    component.internal_id,
                    stage="schema",
                )
            )
    # Duplicate designators are part of the official guide contract.  They stay
    # unchanged in Circuit-IR so evidence remains faithful; the serializer
    # deterministically emits KEY_1, KEY_2, ... by natural component-ID order.

    pin_numbers: dict[tuple[str, str], list[str]] = defaultdict(list)
    for pin in ir.pins:
        if pin.component_id:
            numbers = split_official_pin_numbers(pin.number)
            if not numbers:
                issues.append(
                    error(
                        "PIN.MISSING_NUMBER",
                        "pin number is empty/placeholder and cannot be represented officially",
                        pin.internal_id,
                        stage="schema",
                    )
                )
            for number in numbers:
                pin_numbers[(pin.component_id, number)].append(pin.internal_id)
    for (component_id, number), pin_ids in sorted(pin_numbers.items()):
        if len(pin_ids) > 1:
            issues.append(
                error(
                    "PIN.DUPLICATE_NUMBER",
                    f"pin number {number!r} is not unique within component {component_id}",
                    *sorted(pin_ids),
                    stage="schema",
                )
            )

    return stable_diagnostics(issues)


class SchemaValidator:
    def validate(self, ir: CircuitIR, ctx: object | None = None) -> list[Diagnostic]:
        del ctx
        return validate_schema(ir)
