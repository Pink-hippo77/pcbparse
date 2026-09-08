from __future__ import annotations

from pcbparse.ir.models import CircuitIR, Diagnostic
from pcbparse.validation.common import error, stable_diagnostics


def validate_references(ir: CircuitIR) -> list[Diagnostic]:
    issues: list[Diagnostic] = []
    component_ids = {item.internal_id for item in ir.components}
    pin_ids = {item.internal_id for item in ir.pins}
    segment_ids = {item.internal_id for item in ir.segments}
    label_ids = {item.internal_id for item in ir.net_labels}
    physical_net_ids = {item.internal_id for item in ir.physical_nets}

    for pin in ir.pins:
        if not pin.component_id or pin.component_id not in component_ids:
            issues.append(
                error(
                    "PIN.UNKNOWN_COMPONENT",
                    f"pin references unknown component: {pin.component_id!r}",
                    pin.internal_id,
                    *([pin.component_id] if pin.component_id else []),
                    stage="references",
                )
            )
        for segment_id in pin.edge_ids:
            if segment_id not in segment_ids:
                issues.append(
                    error(
                        "PIN.DANGLING_EDGE",
                        f"pin references unknown segment: {segment_id}",
                        pin.internal_id,
                        segment_id,
                        stage="references",
                    )
                )

    for junction in ir.junctions:
        for segment_id in junction.segment_ids:
            if segment_id not in segment_ids:
                issues.append(
                    error(
                        "JUNCTION.DANGLING_SEGMENT",
                        f"junction references unknown segment: {segment_id}",
                        junction.internal_id,
                        segment_id,
                        stage="references",
                    )
                )

    for label in ir.net_labels:
        if (
            label.attached_physical_net_id is not None
            and label.attached_physical_net_id not in physical_net_ids
        ):
            issues.append(
                error(
                    "LABEL.DANGLING_PHYSICAL_NET",
                    "net label references unknown physical net: "
                    f"{label.attached_physical_net_id}",
                    label.internal_id,
                    label.attached_physical_net_id,
                    stage="references",
                )
            )

    for net in ir.physical_nets:
        for pin_id in net.pins:
            if pin_id not in pin_ids:
                issues.append(
                    error(
                        "NET.DANGLING_PIN",
                        f"physical net references unknown pin: {pin_id}",
                        net.internal_id,
                        pin_id,
                        stage="references",
                    )
                )
        for segment_id in net.segments:
            if segment_id not in segment_ids:
                issues.append(
                    error(
                        "NET.DANGLING_SEGMENT",
                        f"physical net references unknown segment: {segment_id}",
                        net.internal_id,
                        segment_id,
                        stage="references",
                    )
                )
        for label_id in net.labels:
            if label_id not in label_ids:
                issues.append(
                    error(
                        "NET.DANGLING_LABEL",
                        f"physical net references unknown label: {label_id}",
                        net.internal_id,
                        label_id,
                        stage="references",
                    )
                )

    for net in ir.electrical_nets:
        for pin_id in net.pins:
            if pin_id not in pin_ids:
                issues.append(
                    error(
                        "NET.DANGLING_PIN",
                        f"electrical net references unknown pin: {pin_id}",
                        net.internal_id,
                        pin_id,
                        stage="references",
                    )
                )
        for segment_id in net.segments:
            if segment_id not in segment_ids:
                issues.append(
                    error(
                        "NET.DANGLING_SEGMENT",
                        f"electrical net references unknown segment: {segment_id}",
                        net.internal_id,
                        segment_id,
                        stage="references",
                    )
                )
        for physical_net_id in net.physical_net_ids:
            if physical_net_id not in physical_net_ids:
                issues.append(
                    error(
                        "NET.DANGLING_PHYSICAL_NET",
                        f"electrical net references unknown physical net: {physical_net_id}",
                        net.internal_id,
                        physical_net_id,
                        stage="references",
                    )
                )

    return stable_diagnostics(issues)


class ReferenceValidator:
    def validate(self, ir: CircuitIR, ctx: object | None = None) -> list[Diagnostic]:
        del ctx
        return validate_references(ir)
