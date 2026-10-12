"""A04 per-object error funnels over raw-v1 replay.

Three object-consistent funnels replace the old mixed-stage funnel:

* text fields: one funnel over the same batch of visible GT text fields,
* pins: one funnel over the same batch of evaluable GT pins,
* nets: one funnel over the same batch of evaluable GT nets.

Every stage carries object IDs with inclusion and exclusion reasons.  Stages
or objects that cannot be measured report ``not_measured`` with a reason and a
``None`` count — never a fabricated zero.  Counts are cumulative prefix-AND:
an object leaves the funnel at its first failed stage and ``first_failed_stage``
names that stage.  Ground truth is consumed only inside this evaluator;
predictions stay image-only artifacts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from pcbparse.evaluation.masked_metrics import (
    _maximum_cardinality_matching,
    _natural_key,
    _normal_text,
    bbox_iou,
    raster_centerline_coverage,
)
from pcbparse.evaluation.raw_ground_truth import RawGroundTruth, TextTarget
from pcbparse.ir.models import CircuitIR

TEXT_FIELD_STAGES = (
    "owner_matched",
    "char_correct",
    "field_correct",
    "owner_correct",
    "output_correct",
)
PIN_STAGES = (
    "localized",
    "owner_correct",
    "number_name_correct",
    "complete_pin_hit",
)
NET_STAGES = ("matched", "membership_correct", "segments_correct")

STAGE_DEFINITIONS: dict[str, dict[str, str]] = {
    "text_field": {
        "owner_matched": (
            "owner object of the GT field is matched to a prediction "
            "(owner-match proxy, NOT text-box detection; "
            "use text_box_eval for real detection metrics)"
        ),
        "char_correct": "normalized field text is recognized on some predicted field slot",
        "field_correct": "exact field text is recognized on some predicted field slot",
        "owner_correct": "the exact text sits on the predicted owner matched to this GT owner",
        "output_correct": (
            "official output carries the exact text under the GT owner identity"
        ),
    },
    "pin": {
        "localized": "a predicted pin lies within the tip tolerance (owner-agnostic)",
        "owner_correct": (
            "a within-tolerance predicted pin is owned by the matched GT owner component"
        ),
        "number_name_correct": (
            "the owner-correct pin carries exact number and name where GT text is visible"
        ),
        "complete_pin_hit": (
            "localized + owner + visible fields hold on one pin and the owner designator "
            "is present; with official output the pin key and pinname are verified too"
        ),
    },
    "net": {
        "matched": "one predicted net holds at least two of the GT net's mapped pins",
        "membership_correct": (
            "the best predicted net holds exactly the GT net's eligible pins, all mapped, "
            "with no foreign mapped pins"
        ),
        "segments_correct": (
            "net wire geometry is complete in both directions within the wire tolerance"
        ),
    },
}

POLICY = {
    "gt_only_in_independent_evaluator": True,
    "predictions_are_image_only": True,
    "mixed_object_funnel_forbidden": True,
    "unmeasured_is_null_not_zero": True,
    "stage_counts_are_cumulative_prefix_and": True,
}


def _text_visible(target: TextTarget) -> bool:
    return bool(target.mask.eligible and target.value is not None)


def _mask_reasons(mask: Any) -> list[str]:
    return [str(item) for item in getattr(mask, "reasons", ())]


class _ObjectRecord:
    def __init__(self, object_id: str, **fields: Any) -> None:
        self.object_id = object_id
        self.fields = fields
        self.verdicts: dict[str, dict[str, Any]] = {}

    def record(self, stage: str, status: str, reason: str | None = None) -> None:
        if status not in {"passed", "failed", "not_measured"}:
            raise ValueError(f"unsupported verdict status {status!r}")
        self.verdicts[stage] = {"status": status, "reason": reason}

    def as_dict(self, stages: Sequence[str]) -> dict[str, Any]:
        first_failed = next(
            (
                stage
                for stage in stages
                if self.verdicts.get(stage, {}).get("status") == "failed"
            ),
            None,
        )
        return {
            "object_id": self.object_id,
            **self.fields,
            "first_failed_stage": first_failed,
            "stages": {
                stage: self.verdicts.get(
                    stage, {"status": "not_measured", "reason": "upstream_not_reached"}
                )
                for stage in stages
            },
        }


def _finish_funnel(
    *,
    object_kind: str,
    stages: Sequence[str],
    records: Sequence[_ObjectRecord],
    denominator_excluded: Sequence[Mapping[str, Any]],
    denominator_policy: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dropped: set[str] = set()
    stage_rows: list[dict[str, Any]] = []
    for stage in stages:
        included: list[str] = []
        excluded: list[dict[str, Any]] = []
        not_measured: list[dict[str, Any]] = []
        reaching = [item for item in records if item.object_id not in dropped]
        for record in reaching:
            verdict = record.verdicts.get(
                stage, {"status": "not_measured", "reason": "upstream_not_reached"}
            )
            status = verdict["status"]
            if status == "failed":
                dropped.add(record.object_id)
                excluded.append(
                    {
                        "object_id": record.object_id,
                        "reason": verdict.get("reason") or "unspecified",
                    }
                )
            elif status == "passed":
                included.append(record.object_id)
            else:
                not_measured.append(
                    {
                        "object_id": record.object_id,
                        "reason": verdict.get("reason") or "unspecified",
                    }
                )
        if not records:
            row_status = "not_measured"
            count = None
            note = "empty_denominator"
        elif included or excluded:
            row_status = "measured"
            count = len(included)
            note = None
        elif not reaching:
            row_status = "measured"
            count = 0
            note = "all_objects_failed_upstream"
        else:
            row_status = "not_measured"
            count = None
            note = "no_object_was_measurable_at_this_stage"
        stage_rows.append(
            {
                "stage": stage,
                "definition": STAGE_DEFINITIONS[object_kind][stage],
                "status": row_status,
                "count": count,
                "denominator": len(records),
                "reaching": len(reaching),
                "evaluated": len(included) + len(excluded),
                "note": note,
                "included_object_ids": included,
                "excluded": excluded,
                "not_measured": not_measured,
            }
        )
    for index in range(1, len(stage_rows)):
        previous = stage_rows[index - 1]["count"]
        current = stage_rows[index]["count"]
        stage_rows[index]["pass_through_from_prev"] = (
            current / previous if previous not in (None, 0) and current is not None else None
        )
    if stage_rows:
        stage_rows[0]["pass_through_from_prev"] = None
    funnel: dict[str, Any] = {
        "object_kind": object_kind,
        "denominator_policy": denominator_policy,
        "denominator": len(records),
        "denominator_excluded": list(denominator_excluded),
        "stages": stage_rows,
        "objects": [item.as_dict(stages) for item in records],
        "batch_definition": (
            "all stages share one denominator: the evaluable GT objects of this "
            "object kind for the case"
        ),
    }
    if extra:
        funnel.update(extra)
    return funnel


def _component_match_map(
    ir: CircuitIR, gt: RawGroundTruth, *, threshold: float
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    predicted = sorted(ir.components, key=lambda item: _natural_key(item.internal_id))
    expected = sorted(
        (item for item in gt.components if item.mask.eligible and item.bbox is not None),
        key=lambda item: _natural_key(item.raw_id),
    )
    candidates: list[tuple[int, int, float]] = []
    for pred_index, component in enumerate(predicted):
        predicted_bbox = (
            component.bbox.x1,
            component.bbox.y1,
            component.bbox.x2,
            component.bbox.y2,
        )
        for gt_index, target in enumerate(expected):
            assert target.bbox is not None
            score = bbox_iou(predicted_bbox, target.bbox)
            if score >= threshold:
                candidates.append((pred_index, gt_index, score))
    matches = _maximum_cardinality_matching(
        len(predicted),
        len(expected),
        candidates,
        left_keys=[item.internal_id for item in predicted],
        right_keys=[item.raw_id for item in expected],
    )
    gt_to_pred: dict[str, dict[str, Any]] = {}
    pred_to_gt: dict[str, str] = {}
    for left, right, score in matches:
        gt_to_pred[expected[right].raw_id] = {
            "predicted_id": predicted[left].internal_id,
            "iou": score,
            "predicted": predicted[left],
        }
        pred_to_gt[predicted[left].internal_id] = expected[right].raw_id
    return gt_to_pred, pred_to_gt


def _pin_matches(
    ir: CircuitIR,
    gt: RawGroundTruth,
    pred_to_gt_component: Mapping[str, str],
    *,
    tolerance: float,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, str]]:
    predicted = sorted(ir.pins, key=lambda item: _natural_key(item.internal_id))
    expected = sorted(
        (item for item in gt.pins if item.mask.eligible and item.tip is not None),
        key=lambda item: _natural_key(item.raw_id),
    )
    positional: dict[str, dict[str, Any]] = {}
    owner_correct: dict[str, dict[str, Any]] = {}
    for target in expected:
        assert target.tip is not None
        best_any: dict[str, Any] | None = None
        best_owner: dict[str, Any] | None = None
        for pin in predicted:
            distance = math.hypot(
                pin.point.x - target.tip[0], pin.point.y - target.tip[1]
            )
            if distance > tolerance:
                continue
            bound = (
                pin.component_id is not None
                and pred_to_gt_component.get(str(pin.component_id)) == target.owner_raw_id
            )
            entry = {
                "predicted_id": pin.internal_id,
                "distance_px": distance,
                "owner_bound": bound,
                "predicted": pin,
            }
            key = (distance, _natural_key(pin.internal_id))
            if best_any is None or key < (
                best_any["distance_px"],
                _natural_key(best_any["predicted_id"]),
            ):
                best_any = entry
            if bound and (
                best_owner is None
                or key
                < (
                    best_owner["distance_px"],
                    _natural_key(best_owner["predicted_id"]),
                )
            ):
                best_owner = entry
        if best_any is not None:
            positional[target.raw_id] = best_any
        if best_owner is not None:
            owner_correct[target.raw_id] = best_owner
    pred_to_gt: dict[str, str] = {}
    for gt_id, entry in sorted(
        owner_correct.items(), key=lambda item: _natural_key(item[0])
    ):
        pred_id = str(entry["predicted_id"])
        if pred_id not in pred_to_gt:
            pred_to_gt[pred_id] = gt_id
    return {"positional": positional, "owner_correct": owner_correct}, pred_to_gt


def _pred_field_values(ir: CircuitIR, field: str) -> list[str]:
    if field == "designator":
        return [str(item.key) for item in ir.components if isinstance(item.key, str)]
    if field == "value":
        return [str(item.value) for item in ir.components if isinstance(item.value, str)]
    if field == "search_name":
        return [str(item.name) for item in ir.components if isinstance(item.name, str)]
    if field == "number":
        return [str(item.number) for item in ir.pins if isinstance(item.number, str)]
    if field == "name":
        return [str(item.name) for item in ir.pins if isinstance(item.name, str)]
    raise ValueError(f"unsupported field {field!r}")


def _owner_field_value(owner: Any, owner_kind: str, field: str) -> str | None:
    if owner_kind == "component":
        attribute = {"designator": "key", "value": "value", "search_name": "name"}[field]
    else:
        attribute = {"number": "number", "name": "name"}[field]
    value = getattr(owner, attribute, None)
    return value if isinstance(value, str) else None


def _gt_designator_of(gt: RawGroundTruth, owner_kind: str, owner: Any) -> str | None:
    if owner_kind == "component":
        target = owner.designator
    else:
        owner_component = next(
            (item for item in gt.components if item.raw_id == owner.owner_raw_id), None
        )
        target = owner_component.designator if owner_component is not None else None
    if target is not None and _text_visible(target):
        return str(target.value)
    return None


def _output_slot(
    official_output: Mapping[str, Any] | None,
    *,
    owner_kind: str,
    gt_designator: str | None,
    field: str,
    gt_value: str,
) -> tuple[str, str | None]:
    if official_output is None:
        return "not_measured", "official_output_not_provided"
    if gt_designator is None:
        return "not_measured", "gt_designator_not_visible"
    components = official_output.get("components")
    pins = official_output.get("pins")
    if not isinstance(components, Mapping) or not isinstance(pins, Mapping):
        return "failed", "official_output_missing_containers"
    if owner_kind == "component":
        entry = components.get(gt_designator)
        if not isinstance(entry, Mapping):
            return "failed", "owner_key_absent_in_output"
        if field == "designator":
            return "passed", None
        name_field = "Name" if "Name" in entry else "name"
        slot = {"value": "value", "search_name": name_field}[field]
        actual = entry.get(slot)
        return (
            ("passed", None)
            if actual == gt_value
            else ("failed", f"output_{slot}_mismatch")
        )
    group = pins.get(gt_designator)
    if not isinstance(group, Mapping):
        return "failed", "owner_pin_group_absent_in_output"
    if field == "number":
        return (
            ("passed", None)
            if f"pin_{gt_value}" in group
            else ("failed", "pin_number_key_absent_in_output")
        )
    for _key, raw_pin in sorted(group.items(), key=lambda item: _natural_key(item[0])):
        if isinstance(raw_pin, Mapping) and raw_pin.get("pinname") == gt_value:
            return "passed", None
    return "failed", "pinname_absent_in_output"


def _text_field_funnel(
    ir: CircuitIR,
    gt: RawGroundTruth,
    *,
    component_match: Mapping[str, Mapping[str, Any]],
    pin_matches: Mapping[str, Mapping[str, Any]],
    official_output: Mapping[str, Any] | None,
) -> dict[str, Any]:
    records: list[_ObjectRecord] = []
    excluded: list[dict[str, Any]] = []
    slots: list[tuple[str, Any, str, str, TextTarget]] = []
    for component in gt.components:
        for field, target in (
            ("designator", component.designator),
            ("value", component.value),
            ("search_name", component.search_name),
        ):
            slots.append(
                (component.raw_id, "component", field, f"{component.raw_id}.{field}", target)
            )
    for pin in gt.pins:
        for field, target in (("number", pin.number), ("name", pin.name)):
            slots.append((pin.raw_id, "pin", field, f"{pin.raw_id}.{field}", target))

    for owner_id, owner_kind, field, object_id, target in slots:
        owner = next(
            (
                item
                for item in (gt.components if owner_kind == "component" else gt.pins)
                if item.raw_id == owner_id
            ),
            None,
        )
        assert owner is not None
        if not _text_visible(target):
            excluded.append(
                {
                    "object_id": object_id,
                    "reason": ";".join(_mask_reasons(target.mask)) or "field_not_visible",
                }
            )
            continue
        if not owner.mask.eligible:
            excluded.append(
                {
                    "object_id": object_id,
                    "reason": "owner_masked:" + ";".join(_mask_reasons(owner.mask)),
                }
            )
            continue
        gt_value = str(target.value)
        gt_designator = _gt_designator_of(gt, owner_kind, owner)
        record = _ObjectRecord(
            object_id,
            owner_kind=owner_kind,
            owner_id=owner_id,
            field=field,
            gt_value=gt_value,
            gt_text_bbox=list(target.bbox) if target.bbox is not None else None,
            gt_text_id=target.text_id,
        )
        records.append(record)

        detected_entry = (
            component_match.get(owner_id)
            if owner_kind == "component"
            else pin_matches.get("positional", {}).get(owner_id)
        )
        bound_entry = (
            component_match.get(owner_id)
            if owner_kind == "component"
            else pin_matches.get("owner_correct", {}).get(owner_id)
        )
        if detected_entry is None:
            record.record("owner_matched", "failed", "owner_object_not_matched")
        else:
            record.record("owner_matched", "passed")

        anywhere = _pred_field_values(ir, field)
        if any(_normal_text(value) == _normal_text(gt_value) for value in anywhere):
            record.record("char_correct", "passed")
        else:
            record.record("char_correct", "failed", "normalized_text_not_recognized")

        if any(value == gt_value for value in anywhere):
            record.record("field_correct", "passed")
        else:
            record.record("field_correct", "failed", "exact_text_not_recognized")

        bound_owner = bound_entry.get("predicted") if bound_entry is not None else None
        bound_value = (
            _owner_field_value(bound_owner, owner_kind, field)
            if bound_owner is not None
            else None
        )
        if bound_owner is None:
            record.record("owner_correct", "failed", "owner_binding_absent")
        elif bound_value == gt_value:
            record.record("owner_correct", "passed")
        else:
            record.record("owner_correct", "failed", "exact_text_on_wrong_owner")

        status, reason = _output_slot(
            official_output,
            owner_kind=owner_kind,
            gt_designator=gt_designator,
            field=field,
            gt_value=gt_value,
        )
        record.record("output_correct", status, reason)

    return _finish_funnel(
        object_kind="text_field",
        stages=TEXT_FIELD_STAGES,
        records=records,
        denominator_excluded=excluded,
        denominator_policy="visible_gt_text_fields_on_evaluable_owners",
    )


def _pin_funnel(
    ir: CircuitIR,
    gt: RawGroundTruth,
    *,
    pin_matches: Mapping[str, Mapping[str, Any]],
    component_match: Mapping[str, Mapping[str, Any]],
    official_output: Mapping[str, Any] | None,
    tolerance: float,
) -> dict[str, Any]:
    records: list[_ObjectRecord] = []
    excluded: list[dict[str, Any]] = []
    for pin in sorted(gt.pins, key=lambda item: _natural_key(item.raw_id)):
        if not pin.mask.eligible or pin.tip is None:
            excluded.append(
                {
                    "object_id": f"pin:{pin.raw_id}",
                    "reason": ";".join(_mask_reasons(pin.mask)) or "invalid_pin_tip",
                }
            )
            continue
        record = _ObjectRecord(
            f"pin:{pin.raw_id}",
            owner_id=pin.owner_raw_id,
            tip=list(pin.tip) if pin.tip is not None else None,
            tolerance_px=tolerance,
        )
        records.append(record)
        positional = pin_matches.get("positional", {}).get(pin.raw_id)
        owner_entry = pin_matches.get("owner_correct", {}).get(pin.raw_id)
        if positional is None:
            record.record("localized", "failed", "no_predicted_pin_within_tolerance")
            record.record(
                "owner_correct", "not_measured", "no_predicted_pin_within_tolerance"
            )
            record.record(
                "number_name_correct", "not_measured", "no_predicted_pin_within_tolerance"
            )
            record.record(
                "complete_pin_hit", "not_measured", "no_predicted_pin_within_tolerance"
            )
            continue
        record.record("localized", "passed")
        if owner_entry is None:
            record.record("owner_correct", "failed", "owner_binding_wrong")
            record.record("number_name_correct", "not_measured", "owner_binding_wrong")
            record.record("complete_pin_hit", "not_measured", "owner_binding_wrong")
            continue
        record.record("owner_correct", "passed")
        predicted_pin = owner_entry["predicted"]
        field_failures: list[str] = []
        visible_fields = 0
        for field, target in (("number", pin.number), ("name", pin.name)):
            if not _text_visible(target):
                continue
            visible_fields += 1
            actual = _owner_field_value(predicted_pin, "pin", field)
            if actual != str(target.value):
                field_failures.append(f"{field}_mismatch")
        if visible_fields == 0:
            record.record(
                "number_name_correct", "not_measured", "no_visible_pin_fields"
            )
        elif field_failures:
            record.record("number_name_correct", "failed", ";".join(field_failures))
        else:
            record.record("number_name_correct", "passed")

        if field_failures:
            record.record("complete_pin_hit", "failed", ";".join(field_failures))
            continue
        if visible_fields == 0:
            record.record("complete_pin_hit", "not_measured", "no_visible_pin_fields")
            continue
        owner_component = component_match.get(str(pin.owner_raw_id))
        predicted_component = (
            owner_component.get("predicted") if owner_component is not None else None
        )
        designator = getattr(predicted_component, "key", None)
        if not isinstance(designator, str) or not designator.strip():
            record.record("complete_pin_hit", "failed", "owner_designator_missing")
            continue
        if official_output is None:
            record.record("complete_pin_hit", "passed", "ir_identity_only")
            continue
        gt_designator = _gt_designator_of(gt, "pin", pin)
        status, reason = "passed", None
        if _text_visible(pin.number):
            status, reason = _output_slot(
                official_output,
                owner_kind="pin",
                gt_designator=gt_designator,
                field="number",
                gt_value=str(pin.number.value),
            )
        if status == "passed" and _text_visible(pin.name):
            status, reason = _output_slot(
                official_output,
                owner_kind="pin",
                gt_designator=gt_designator,
                field="name",
                gt_value=str(pin.name.value),
            )
        record.record("complete_pin_hit", status, reason)

    return _finish_funnel(
        object_kind="pin",
        stages=PIN_STAGES,
        records=records,
        denominator_excluded=excluded,
        denominator_policy="evaluable_gt_pins_with_valid_tip",
        extra={
            "false_positive_note": (
                "false-positive predicted pins are counted by raw-v1 pin detection, "
                "this funnel is GT-object scoped"
            )
        },
    )


def _net_funnel(
    ir: CircuitIR,
    gt: RawGroundTruth,
    *,
    pred_to_gt_pin: Mapping[str, str],
    wire_tolerance: int,
) -> dict[str, Any]:
    records: list[_ObjectRecord] = []
    excluded: list[dict[str, Any]] = []
    predicted_nets = list(ir.electrical_nets)
    segment_index = ir.segment_index()
    wire_by_id = {wire.raw_id: wire for wire in gt.wires}
    gt_pin_sets = [set(pins) for pins in gt.connectivity_nets]
    for index, pins in enumerate(gt.connectivity_nets):
        pin_ids = sorted(pins, key=_natural_key)
        wire_ids = (
            sorted(gt.net_wire_ids[index], key=_natural_key)
            if index < len(gt.net_wire_ids)
            else []
        )
        object_id = f"net:{index}"
        if len(pin_ids) < 2:
            excluded.append(
                {
                    "object_id": object_id,
                    "reason": "fewer_than_two_eligible_pins",
                    "pin_ids": pin_ids,
                }
            )
            continue
        record = _ObjectRecord(object_id, pin_ids=pin_ids, wire_ids=wire_ids, index=index)
        records.append(record)
        net_pin_set = gt_pin_sets[index]
        mapped_gt_ids = {
            gt_id for gt_id in pred_to_gt_pin.values() if gt_id in net_pin_set
        }
        best_net: Any = None
        shared_best: set[str] = set()
        for net in predicted_nets:
            shared = {
                pred_to_gt_pin[pid] for pid in net.pins if pid in pred_to_gt_pin
            } & net_pin_set
            if len(shared) > len(shared_best):
                shared_best = shared
                best_net = net
        if len(mapped_gt_ids) < 2:
            record.record(
                "matched", "failed", f"insufficient_mapped_pins:{len(mapped_gt_ids)}"
            )
            record.record("membership_correct", "not_measured", "not_matched")
            record.record("segments_correct", "not_measured", "not_matched")
            continue
        if best_net is None or len(shared_best) < 2:
            record.record("matched", "failed", "no_predicted_net_holds_two_mapped_pins")
            record.record("membership_correct", "not_measured", "not_matched")
            record.record("segments_correct", "not_measured", "not_matched")
            continue
        record.record("matched", "passed")
        missing = sorted(net_pin_set - shared_best, key=_natural_key)
        unmapped = sorted(net_pin_set - mapped_gt_ids, key=_natural_key)
        foreign = sorted(
            {
                pred_to_gt_pin[pid]
                for pid in best_net.pins
                if pid in pred_to_gt_pin and pred_to_gt_pin[pid] not in net_pin_set
            },
            key=_natural_key,
        )
        if missing or foreign or unmapped:
            reasons = []
            if unmapped:
                reasons.append(f"unmapped_eligible_pins:{','.join(unmapped)}")
            if missing:
                reasons.append(f"missing_from_pred_net:{','.join(missing)}")
            if foreign:
                reasons.append(f"foreign_mapped_pins:{','.join(foreign)}")
            record.record("membership_correct", "failed", ";".join(reasons))
            record.record(
                "segments_correct", "not_measured", "membership_incorrect"
            )
            continue
        record.record("membership_correct", "passed")

        gt_paths: list[list[tuple[float, float]]] = []
        gt_has_geometry = False
        for wire_id in wire_ids:
            wire = wire_by_id.get(wire_id)
            if wire is None or not wire.mask.eligible:
                continue
            gt_has_geometry = True
            gt_paths.extend([list(path) for path in wire.paths])
        pred_segments = [
            (
                (segment_index[seg_id].start.x, segment_index[seg_id].start.y),
                (segment_index[seg_id].end.x, segment_index[seg_id].end.y),
            )
            for seg_id in best_net.segments
            if seg_id in segment_index
        ]
        if not gt_has_geometry and not pred_segments:
            record.record("segments_correct", "passed", "no_wire_geometry_both_sides")
            continue
        if gt_has_geometry and not pred_segments:
            record.record("segments_correct", "failed", "missing_predicted_segments")
            continue
        if not gt_has_geometry and pred_segments:
            record.record("segments_correct", "failed", "unexpected_predicted_segments")
            continue
        coverage = raster_centerline_coverage(
            gt_paths,
            pred_segments,
            width=gt.width,
            height=gt.height,
            tolerance=wire_tolerance,
        )
        complete = (
            coverage["expected_hit_pixels"] == coverage["expected_pixels"]
            and coverage["predicted_hit_pixels"] == coverage["predicted_pixels"]
        )
        if complete:
            record.record("segments_correct", "passed")
        else:
            record.record(
                "segments_correct",
                "failed",
                "incomplete_centerline_coverage:"
                f"recall={coverage['recall']:.3f},precision={coverage['precision']:.3f}",
            )

    return _finish_funnel(
        object_kind="net",
        stages=NET_STAGES,
        records=records,
        denominator_excluded=excluded,
        denominator_policy="gt_connectivity_nets_with_at_least_two_eligible_pins",
    )


def build_error_funnel(
    ir: CircuitIR,
    ground_truth: RawGroundTruth,
    *,
    official_output: Mapping[str, Any] | None = None,
    component_iou_threshold: float = 0.5,
    pin_distance_tolerance_px: float = 5.0,
    wire_tolerance_px: int = 3,
    binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the three per-object funnels for one replayed case."""

    component_match, pred_to_gt_component = _component_match_map(
        ir, ground_truth, threshold=component_iou_threshold
    )
    pin_matches, pred_to_gt_pin = _pin_matches(
        ir,
        ground_truth,
        pred_to_gt_component,
        tolerance=pin_distance_tolerance_px,
    )
    text_funnel = _text_field_funnel(
        ir,
        ground_truth,
        component_match=component_match,
        pin_matches=pin_matches,
        official_output=official_output,
    )
    pin_funnel = _pin_funnel(
        ir,
        ground_truth,
        pin_matches=pin_matches,
        component_match=component_match,
        official_output=official_output,
        tolerance=pin_distance_tolerance_px,
    )
    net_funnel = _net_funnel(
        ir,
        ground_truth,
        pred_to_gt_pin=pred_to_gt_pin,
        wire_tolerance=wire_tolerance_px,
    )
    return {
        "schema_version": "2.0",
        "kind": "error_funnel",
        "task_id": "A04",
        "status": "measured",
        "case_id": ground_truth.case_id,
        "policy_version": ground_truth.policy_version,
        "binding": dict(binding) if binding is not None else empty_binding(),
        "thresholds": {
            "component_iou": component_iou_threshold,
            "pin_distance_px": pin_distance_tolerance_px,
            "wire_centerline_tolerance_px": wire_tolerance_px,
        },
        "official_output_checked": official_output is not None,
        "funnels": {
            "text_field": text_funnel,
            "pin": pin_funnel,
            "net": net_funnel,
        },
        "policy": dict(POLICY),
    }


def empty_binding() -> dict[str, Any]:
    return {
        "image_sha256": None,
        "source_sha256": None,
        "config_sha256": None,
        "model_lock_sha256": None,
        "status": "not_provided",
    }


def not_measured_funnel(reason: str) -> dict[str, Any]:
    """Scaffold for runs that were not executed; counts stay null, never zero."""

    funnels: dict[str, Any] = {}
    for object_kind, stages in (
        ("text_field", TEXT_FIELD_STAGES),
        ("pin", PIN_STAGES),
        ("net", NET_STAGES),
    ):
        funnels[object_kind] = {
            "object_kind": object_kind,
            "denominator": None,
            "denominator_excluded": [],
            "objects": [],
            "batch_definition": (
                "all stages share one denominator: the evaluable GT objects of this "
                "object kind for the case"
            ),
            "stages": [
                {
                    "stage": stage,
                    "definition": STAGE_DEFINITIONS[object_kind][stage],
                    "status": "not_measured",
                    "count": None,
                    "denominator": None,
                    "reaching": None,
                    "evaluated": None,
                    "note": reason,
                    "pass_through_from_prev": None,
                    "included_object_ids": [],
                    "excluded": [],
                    "not_measured": [{"object_id": None, "reason": reason}],
                }
                for stage in stages
            ],
        }
    return {
        "schema_version": "2.0",
        "kind": "error_funnel",
        "task_id": "A04",
        "status": "not_measured",
        "reason": reason,
        "binding": empty_binding(),
        "funnels": funnels,
        "policy": dict(POLICY),
    }


def aggregate_error_funnels(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-case funnels; object ids stay case-scoped and traceable."""

    measured = [item for item in reports if item.get("status") == "measured"]
    aggregate: dict[str, Any] = {}
    for object_kind, stages in (
        ("text_field", TEXT_FIELD_STAGES),
        ("pin", PIN_STAGES),
        ("net", NET_STAGES),
    ):
        rows: list[dict[str, Any]] = []
        previous_count: int | None = None
        for stage in stages:
            count = 0
            denominator = 0
            measured_cases = 0
            included: list[str] = []
            excluded: list[dict[str, Any]] = []
            for report in measured:
                case_id = str(report.get("case_id"))
                funnel = report["funnels"][object_kind]
                row = next(item for item in funnel["stages"] if item["stage"] == stage)
                if row["count"] is not None:
                    count += int(row["count"])
                    measured_cases += 1
                if row["denominator"] is not None:
                    denominator += int(row["denominator"])
                included.extend(
                    f"{case_id}:{value}" for value in row["included_object_ids"]
                )
                excluded.extend(
                    {
                        "object_id": f"{case_id}:{item['object_id']}",
                        "reason": item["reason"],
                    }
                    for item in row["excluded"]
                )
            pass_through = (
                count / previous_count
                if previous_count not in (None, 0) and measured_cases
                else None
            )
            rows.append(
                {
                    "stage": stage,
                    "definition": STAGE_DEFINITIONS[object_kind][stage],
                    "status": "measured" if measured_cases else "not_measured",
                    "count": count if measured_cases else None,
                    "denominator": denominator if measured_cases else None,
                    "measured_cases": measured_cases,
                    "pass_through_from_prev": pass_through,
                    "included_object_ids": included,
                    "excluded": excluded,
                }
            )
            previous_count = count if measured_cases else previous_count
        aggregate[object_kind] = {
            "object_kind": object_kind,
            "denominator_policy": "sum_of_per_case_denominators",
            "stages": rows,
        }
    return {
        "cases_total": len(reports),
        "cases_measured": len(measured),
        "cases_not_measured": [
            {
                "case_id": item.get("case_id"),
                "reason": item.get("reason") or "not_measured",
            }
            for item in reports
            if item.get("status") != "measured"
        ],
        "funnels": aggregate,
    }
