from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import replace
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pcbparse.ir.models import CircuitIR, Diagnostic, Evidence
from pcbparse.pipeline.budget import BudgetExceeded
from pcbparse.pipeline.context import RunContext
from pcbparse.reasoning.cache import (
    DEFAULT_REASONING_CACHE,
    ReasoningCache,
    reasoning_cache_key,
)
from pcbparse.reasoning.qwen_client import DisabledReasoner, HTTPQwenReasoner, Reasoner
from pcbparse.reasoning.router import AmbiguityTask, Decision, ValueAwareRouter
from pcbparse.reasoning.tasks import generate_ambiguity_tasks


REASONING_STAGE = "reasoning.selective"
_METADATA_KEY = "selective_reasoning"


def run_selective_reasoning(
    ir: CircuitIR,
    config: Mapping[str, Any],
    ctx: RunContext,
    reasoner: Reasoner | None = None,
) -> CircuitIR:
    """Resolve selected ambiguities without allowing generative IR edits.

    The reasoner sees candidate IDs and context, never the mutable Circuit-IR.
    Candidate decisions fill empty semantic fields or resolve an ``ambiguous``
    crossing. Optional number transcription requires two agreeing source crops
    before filling an empty field or replacing an explicitly weak association.
    Geometry and object counts are immutable. Populated fields require explicit
    visual-review evidence; other existing values cannot be replaced.
    """

    qwen_config = _qwen_config(config)
    if not bool(qwen_config.get("enabled", False)):
        return ir

    active_reasoner: Reasoner = reasoner if reasoner is not None else DisabledReasoner()
    reasoner_version = _reasoner_version(active_reasoner, qwen_config, ctx)
    config_fingerprint = _config_fingerprint(qwen_config)
    pre_state_signature = _reasoning_state_signature(ir)
    previous = ir.metadata.get(_METADATA_KEY)
    if (
        isinstance(previous, Mapping)
        and previous.get("completed") is True
        and previous.get("reasoner_version") == reasoner_version
        and previous.get("config_fingerprint") == config_fingerprint
        and previous.get("post_state_signature") == pre_state_signature
    ):
        return ir

    # Reprocessing after an actual upstream/config change replaces this stage's
    # own diagnostics and ledger rather than multiplying them.
    ir.diagnostics = [item for item in ir.diagnostics if item.stage != REASONING_STAGE]

    number_readback = None
    missing_numbers = qwen_config.get('visual_missing_numbers_enabled', False)
    if type(missing_numbers) is not bool:
        raise ValueError('visual_missing_numbers_enabled must be a boolean')
    numeric_cells = qwen_config.get('visual_numeric_cell_readback_enabled', False)
    if type(numeric_cells) is not bool:
        raise ValueError('visual_numeric_cell_readback_enabled must be a boolean')
    for option in ('visual_unsupported_numbers_enabled', 'visual_alphanumeric_numbers_enabled'):
        if type(qwen_config.get(option, False)) is not bool:
            raise ValueError(option+' must be a boolean')
    max_candidates = int(qwen_config.get("max_candidates_per_task", 5))
    max_candidates = max(2, min(5, max_candidates))
    tasks = generate_ambiguity_tasks(ir, max_candidates=max_candidates)
    if qwen_config.get('visual_ic_reselection_enabled') is True:
        from pcbparse.reasoning.visual_ic_tasks import visible_name_reselection_tasks, visible_number_reselection_tasks
        tasks.extend(visible_name_reselection_tasks(ir))
        tasks.extend(visible_number_reselection_tasks(ir))
    allowed_types = qwen_config.get('task_types')
    if isinstance(allowed_types, list):
        tasks = [task for task in tasks if task.task_type in allowed_types]
    if qwen_config.get('ic_only') is True:
        components, pins = ir.component_index(), ir.pin_index()
        def ic_task(task):
            kind, oid = task.context.get('owner_kind'), task.context.get('object_id')
            component = components.get(oid) if kind == 'component' else components.get(pins[oid].component_id) if kind == 'pin' and oid in pins else None
            return component is not None and component.category.upper() in {'BOX','IC','BOXIC','BOX_IC'}
        tasks = [task for task in tasks if ic_task(task)]
    router = ValueAwareRouter(
        threshold=float(qwen_config.get("route_threshold", 0.72)),
        min_time_reserve_seconds=float(qwen_config.get("min_time_reserve_seconds", 25.0)),
    )
    cache_enabled = bool(qwen_config.get("cache", True))
    cache = _cache_from_config(qwen_config)
    image_hash, image_hash_source = _image_hash(ir)
    minimum_confidence = _clamp01(qwen_config.get("decision_min_confidence", 0.0))

    ranked_tasks = sorted(
        tasks,
        key=lambda task: (-router.ambiguity_score(task), task.task_id),
    )
    ledger: list[dict[str, Any]] = []
    applied = 0
    routed = 0
    rejected = 0
    cache_hits = 0
    batch_size = qwen_config.get('visual_batch_size', 1)
    if type(batch_size) is not int or not 1 <= batch_size <= 6:
        raise ValueError('visual_batch_size must be an integer in [1, 6]')
    batch_size = min(batch_size, max(1, ctx.token_budget.max_output_tokens_per_call//32))
    batch_results = {}

    for task_index, task in enumerate(ranked_tasks):
        failure = None
        route_score = router.ambiguity_score(task)
        estimated_input_tokens = _estimated_input_tokens(task)
        cache_key = reasoning_cache_key(image_hash, task, reasoner_version)
        row = _ledger_base(
            task,
            route_score=route_score,
            estimated_input_tokens=estimated_input_tokens,
            cache_key=cache_key,
        )

        # Junction crossing kind is a visual decision.  The text-only HTTP
        # reasoner (send_images=False) has no crop to inspect, so routing
        # would only produce UNKNOWN and waste the call budget.  Capable
        # fixture/rule reasoners keep the ability to resolve junctions.
        if (
            task.task_type == "junction.kind"
            and not task.crop_path
            and isinstance(active_reasoner, HTTPQwenReasoner)
            and not active_reasoner.send_images
        ):
            row["status"] = "not_routed"
            row["reason"] = "junction resolution requires a visual crop"
            ledger.append(row)
            continue

        # A cached decision has no call/token cost, but it still has to satisfy
        # the configured value threshold.  This prevents a stale policy from
        # silently widening the set of routed tasks.
        threshold_eligible = (
            route_score >= router.threshold and 1 < len(task.candidates) <= 5
        )
        cached = cache.get(cache_key) if cache_enabled and threshold_eligible else None
        if task.task_id in batch_results:
            decision, failure, batch_id = batch_results.pop(task.task_id)
            row.update(routed=True, batch_id=batch_id)
            routed += 1
        elif cached is not None:
            decision = cached
            row["cache_hit"] = True
            row["routed"] = True
            routed += 1
            cache_hits += 1
        elif router.should_route(
            task,
            ctx.token_budget,
            ctx.time_budget,
            estimated_input_tokens=estimated_input_tokens,
        ):
            row["routed"] = True
            routed += 1
            group = [task]
            if (batch_size > 1 and isinstance(active_reasoner, HTTPQwenReasoner) and active_reasoner.send_images
                    and task.task_type in {'pin.readback_name', 'pin.readback_number'}):
                pin_index = ir.pin_index()
                target = pin_index.get(task.context.get('object_id'))
                if target is not None:
                    for other in ranked_tasks[task_index+1:]:
                        pin = pin_index.get(other.context.get('object_id'))
                        if (other.task_id in batch_results or other.task_type != task.task_type or pin is None
                                or not 2 <= len(other.candidates) <= 5
                                or router.ambiguity_score(other) < router.threshold):
                            continue
                        if cache_enabled and cache.get(reasoning_cache_key(image_hash, other, reasoner_version)) is not None:
                            continue
                        estimate = sum(_estimated_input_tokens(t) for t in [*group, other])+256
                        if estimate > ctx.token_budget.remaining_input_tokens:
                            continue
                        group.append(other)
                        if len(group) == batch_size:
                            break
            if len(group) > 1:
                batch_id = 'batch:'+hashlib.sha256('|'.join(t.task_id for t in group).encode()).hexdigest()[:16]
                decisions, failure = _choose_batch_with_accounting(active_reasoner, group, ir, ctx,
                    reserve_seconds=router.min_time_reserve_seconds)
                for target_task, choice in zip(group, decisions):
                    batch_results[target_task.task_id] = (choice, failure, batch_id)
                decision, failure, _ = batch_results.pop(task.task_id)
                row['batch_id'] = batch_id
            else:
                decision, failure = _choose_with_accounting(
                    active_reasoner, task, ctx, estimated_input_tokens=estimated_input_tokens,
                    ir=ir, reserve_seconds=router.min_time_reserve_seconds)
        else:
            row["status"] = "not_routed"
            row["reason"] = _not_routed_reason(
                task,
                router,
                ctx,
                estimated_input_tokens=estimated_input_tokens,
            )
            ledger.append(row)
            continue

        if failure is not None:
            row.update(status='reasoner_failed', reason=failure)
            rejected += 1
            _append_diagnostic(ir,severity='WARNING',code='REASONING.REASONER_FAILED',
                message=f'task {task.task_id} failed closed: {failure}',
                object_ids=[task.context.get('object_id', task.task_id)])
            ledger.append(row)
            continue

        validation_error = _decision_error(decision, task)
        if validation_error is not None:
            if cache_enabled and bool(row["cache_hit"]):
                cache.discard(cache_key)
            row["status"] = "decision_rejected"
            row["reason"] = validation_error
            row["decision"] = _decision_dict(decision)
            rejected += 1
            _append_diagnostic(
                ir,
                severity="WARNING",
                code="REASONING.DECISION_REJECTED",
                message=f"task {task.task_id} rejected: {validation_error}",
                object_ids=[task.context.get("object_id", task.task_id)],
            )
            ledger.append(row)
            continue

        if cache_enabled and not bool(row["cache_hit"]):
            cache.put(cache_key, decision)
        row["decision"] = _decision_dict(decision)
        if decision.selected_id == "UNKNOWN":
            row["status"] = "unknown"
            row["reason"] = decision.reason or "reasoner selected UNKNOWN"
            _append_diagnostic(
                ir,
                severity="INFO",
                code="REASONING.UNKNOWN",
                message=f"task {task.task_id} remains unresolved",
                object_ids=[task.context.get("object_id", task.task_id)],
            )
            ledger.append(row)
            continue
        if decision.confidence < minimum_confidence:
            row["status"] = "below_decision_threshold"
            row["reason"] = (
                f"decision confidence {decision.confidence:.6f} is below "
                f"{minimum_confidence:.6f}"
            )
            _append_diagnostic(
                ir,
                severity="INFO",
                code="REASONING.DECISION_BELOW_THRESHOLD",
                message=f"task {task.task_id} remains unresolved below decision threshold",
                object_ids=[task.context.get("object_id", task.task_id)],
            )
            ledger.append(row)
            continue

        candidate = next(
            item for item in task.candidates if item.candidate_id == decision.selected_id
        )
        status, reason = _safe_apply(ir, task, candidate.value, decision, cache_key)
        row["status"] = status
        row["reason"] = reason
        if status == "applied":
            row["applied_value"] = candidate.value
            applied += 1
            _append_diagnostic(
                ir,
                severity="INFO",
                code="REASONING.APPLIED",
                message=f"task {task.task_id} selected a supplied candidate",
                object_ids=[task.context.get("object_id", task.task_id)],
            )
        else:
            rejected += 1
            _append_diagnostic(
                ir,
                severity="WARNING",
                code="REASONING.WRITE_PROTECTED",
                message=f"task {task.task_id} was not applied: {reason}",
                object_ids=[task.context.get("object_id", task.task_id)],
            )
        ledger.append(row)

    # Each numbered image tile is already an independent closed-set task.
    # Batching across owners leaves every crop, candidate set and write gate
    # attached to its own pin. Spend the remaining budget on missing numbers
    # only after reviewing existing OCR conflicts, never before that work.
    if missing_numbers or numeric_cells:
        from pcbparse.reasoning.visual_number_readback import read_missing_visual_numbers
        number_readback = read_missing_visual_numbers(ir, active_reasoner, ctx, qwen_config)

    _append_diagnostic(
        ir,
        severity="INFO",
        code="REASONING.SUMMARY",
        message=(
            f"generated={len(tasks)}, routed={routed}, applied={applied}, "
            f"rejected={rejected}, cache_hits={cache_hits}"
        ),
        object_ids=[],
    )
    ir.metadata[_METADATA_KEY] = {
        "version": "1.0",
        "completed": True,
        "reasoner_version": reasoner_version,
        "config_fingerprint": config_fingerprint,
        "image_hash": image_hash,
        "image_hash_source": image_hash_source,
        "cache_enabled": cache_enabled,
        "generated_tasks": len(tasks),
        "routed_tasks": routed,
        "applied_tasks": applied,
        "rejected_tasks": rejected,
        "cache_hits": cache_hits,
        "ledger": ledger,
        "post_state_signature": "",  # populated after all evidence writes
    }
    ir.metadata[_METADATA_KEY]["post_state_signature"] = _reasoning_state_signature(ir)
    if number_readback is not None:
        ir.metadata[_METADATA_KEY]['visual_number_readback'] = number_readback
    return ir


def _qwen_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = config.get("qwen")
    return nested if isinstance(nested, Mapping) else config


def _cache_from_config(config: Mapping[str, Any]) -> ReasoningCache:
    supplied = config.get("cache_store")
    return supplied if isinstance(supplied, ReasoningCache) else DEFAULT_REASONING_CACHE


def _choose_with_accounting(
    reasoner: Reasoner,
    task: AmbiguityTask,
    ctx: RunContext,
    *,
    estimated_input_tokens: int,
    ir: CircuitIR | None = None,
    reserve_seconds: float = 1.0,
) -> tuple[Decision, str | None]:
    before_calls = ctx.token_budget.calls
    decision: Decision | None = None
    failure: str | None = None
    try:
        if isinstance(reasoner, HTTPQwenReasoner) and reasoner.send_images and ir is not None:
            with tempfile.TemporaryDirectory(prefix='pcbparse-qwen-') as folder:
                visual_task = _visual_task(task, ir, Path(folder))
                decision = reasoner.choose(visual_task, ctx.token_budget,
                    timeout_seconds=max(0., ctx.time_budget.remaining_seconds-reserve_seconds))
        elif isinstance(reasoner, HTTPQwenReasoner):
            decision = reasoner.choose(task, ctx.token_budget,
                timeout_seconds=max(0., ctx.time_budget.remaining_seconds-reserve_seconds))
        else:
            decision = reasoner.choose(task, ctx.token_budget)
    except Exception as exc:  # third-party model adapters must fail closed
        failure = type(exc).__name__
    finally:
        if ctx.token_budget.calls == before_calls and not (isinstance(decision, Decision) and decision.source == 'qwen_not_called'):
            try:
                ctx.token_budget.reserve(
                    estimated_input_tokens,
                    min(64, max(0, ctx.token_budget.max_output_tokens_per_call)),
                )
            except BudgetExceeded as exc:
                failure = failure or str(exc)
    if failure is not None:
        return Decision(task.task_id, "UNKNOWN", 0.0, "reasoner_error", failure), failure
    if not isinstance(decision, Decision):
        return (
            Decision(task.task_id, "UNKNOWN", 0.0, "reasoner_error", "invalid decision type"),
            "invalid decision type",
        )
    return decision, None


def _choose_batch_with_accounting(reasoner: HTTPQwenReasoner, tasks: list[AmbiguityTask],
                                  ir: CircuitIR, ctx: RunContext, *, reserve_seconds: float):
    """One bounded request; each returned decision retains its own write gate."""
    from PIL import Image, ImageDraw
    before_calls = ctx.token_budget.calls
    decisions = None
    failure = None
    try:
        with tempfile.TemporaryDirectory(prefix='pcbparse-qwen-batch-') as folder:
            folder = Path(folder)
            visual = []
            crops = []
            for index,task in enumerate(tasks,1):
                tile_folder = folder/str(index)
                tile_folder.mkdir()
                target = _visual_task(task, ir, tile_folder, tight_pin_crop=True)
                visual.append(target)
                with Image.open(target.crop_path) as image:
                    crops.append(image.convert('RGB'))
            width = max(c.width for c in crops)+12
            height = max(c.height for c in crops)+34
            collage = Image.new('RGB',(2*width,math.ceil(len(crops)/2)*height),'white')
            draw = ImageDraw.Draw(collage)
            positioned = []
            for index,(task,crop) in enumerate(zip(visual,crops)):
                x,y = (index%2)*width,(index//2)*height
                draw.text((x+6,y+4),str(index+1),fill='black')
                collage.paste(crop,(x+6,y+28))
                point = task.context['target_point_in_crop'].split(',')
                context = {**task.context, 'target_point_in_crop':
                    f'{float(point[0])+x+6:.1f},{float(point[1])+y+28:.1f}'}
                positioned.append(replace(task,context=context,crop_path=str(folder/'batch.png')))
            collage.save(folder/'batch.png')
            decisions = reasoner.choose_many(positioned,ctx.token_budget,
                timeout_seconds=max(0.,ctx.time_budget.remaining_seconds-reserve_seconds))
            if not isinstance(decisions,list) or len(decisions) != len(tasks):
                raise ValueError('batch decision cardinality mismatch')
            if any(not isinstance(d,Decision) or _decision_error(d,t) for d,t in zip(decisions,tasks)):
                raise ValueError('batch decision task mismatch')
    except Exception as exc:
        failure = type(exc).__name__
    finally:
        not_called = (isinstance(decisions,list) and len(decisions)==len(tasks)
                      and all(isinstance(d,Decision) and d.source=='qwen_not_called' for d in decisions))
        if ctx.token_budget.calls == before_calls and not not_called:
            try:
                ctx.token_budget.reserve(sum(_estimated_input_tokens(t)for t in tasks)+256,
                    min(192,max(0,ctx.token_budget.max_output_tokens_per_call)))
            except BudgetExceeded as exc:
                failure = failure or str(exc)
    if failure is not None:
        decisions = [Decision(t.task_id,'UNKNOWN',0.,'reasoner_error',failure) for t in tasks]
    return decisions, failure


def _visual_task(task: AmbiguityTask, ir: CircuitIR, folder: Path, *, tight_pin_crop: bool = False) -> AmbiguityTask:
    """Attach the actual source pixels around a fixed, existing IR object."""
    from PIL import Image
    from pcbparse.pipeline.context import stable_file_hash
    from pcbparse.ir.models import BBox

    kind, oid = task.context.get('owner_kind'), task.context.get('object_id')
    owner = ir.component_index().get(oid) if kind == 'component' else ir.pin_index().get(oid) if kind == 'pin' else next(
        (j for j in ir.junctions if j.internal_id == oid), None)
    if owner is None:
        raise ValueError('Qwen crop owner does not exist')
    context = dict(task.context)
    if kind == 'component':
        box = owner.body_bbox or owner.bbox
        padding = 35.
        context.update(component_type=owner.category, observed_reference=owner.key or '', observed_model=owner.name)
    elif kind == 'pin':
        component = ir.component_index().get(owner.component_id)
        if component is None:
            raise ValueError('Qwen pin crop has no component')
        p, b = owner.point, component.body_bbox or component.bbox
        box = BBox(x1=min(p.x-45, max(b.x1-25, p.x-140)), y1=max(b.y1-25, p.y-45),
                   x2=max(p.x+45, min(b.x2+25, p.x+140)), y2=min(b.y2+25, p.y+45))
        padding = 5.
        context.update(component_type=component.category, observed_reference=component.key or '',
                       observed_model=component.name, terminal_side=owner.orientation)
    else:
        p = owner.point
        box, padding = BBox(x1=p.x-40, y1=p.y-40, x2=p.x+40, y2=p.y+40), 0.
    source = Path(ir.image.path)
    if not ir.image.sha256 or stable_file_hash(source) != ir.image.sha256:
        raise ValueError('Qwen crop source image hash mismatch')
    bounds = (max(0, int(box.x1-padding)), max(0, int(box.y1-padding)),
              min(ir.image.width, math.ceil(box.x2+padding)), min(ir.image.height, math.ceil(box.y2+padding)))
    with Image.open(source) as original:
        if original.size != (ir.image.width, ir.image.height):
            raise ValueError('Qwen crop dimensions do not match image')
        original = original.convert('RGB')
        if tight_pin_crop and kind == 'pin':
            return _pin_row_visual_task(task,ir,owner,component,original,folder,context)
        crop = original.crop(bounds)
    scale = min(3., 1024/max(crop.size))
    crop = crop.resize((max(1, round(crop.width*scale)), max(1, round(crop.height*scale))))
    if kind in {'pin', 'junction'}:
        context['target_point_in_crop'] = f'{(owner.point.x-bounds[0])*scale:.1f},{(owner.point.y-bounds[1])*scale:.1f}'
    output = folder/'crop.png'
    crop.save(output)
    return replace(task, crop_path=str(output), context=context)


def _pin_row_visual_task(task,ir,pin,component,image,folder,context):
    """Keep the whole printed row, rotating vertical banks into reading order."""
    from PIL import Image
    from pcbparse.perception.ic_visual_readback import _transform_box
    observations=[e.metadata for e in pin.evidence if e.metadata.get('kind')=='visible-stroke-and-text-lanes']
    observation=observations[-1] if observations else {}
    side=pin.orientation
    if side not in {'L','R','T','B'}:
        raise ValueError('row review requires a measured terminal side')
    axis=pin.point.y if side in {'L','R'} else pin.point.x
    position=float(observation.get('position',axis))
    peers=[abs((p.point.y if side in {'L','R'} else p.point.x)-axis) for p in ir.pins
           if p.component_id==pin.component_id and p.orientation==side and p.internal_id!=pin.internal_id]
    pitch=max(8.,min([v for v in peers if v>3]+[40.]))
    body=component.body_bbox or component.bbox
    px,py=pin.point.x,pin.point.y
    rotation=int(observation.get('rotation',1)) if side in {'T','B'} else 0
    if rotation not in {0,1,3} or side in {'T','B'} and rotation==0:
        raise ValueError('invalid terminal text rotation')
    if rotation:
        body=_transform_box(body,ir.image.width,ir.image.height,rotation)
        if rotation==1:
            image=image.transpose(Image.Transpose.ROTATE_90)
            px,py=py,ir.image.width-1-px
            position=ir.image.width-1-position
            side={'T':'L','B':'R'}[side]
        else:
            image=image.transpose(Image.Transpose.ROTATE_270)
            px,py=ir.image.height-1-py,px
            side={'T':'R','B':'L'}[side]
    if task.context.get('field')=='number':
        if context.get('terminal_label_layout') == 'interior_numeric_cell':
            # Isolate the printed interior cell. A second observation changes
            # horizontal padding only, so it cannot expose a neighbouring row.
            margin = float(context.get('readback_margin_px', 0.))
            if not math.isfinite(margin) or not 0 <= margin <= 2:
                raise ValueError('invalid numeric-cell horizontal margin')
            left, right = ((body.x1+2-margin/2, body.center.x-.5) if side == 'L'
                           else (body.center.x+.5, body.x2-2+margin/2))
            top, bottom = position-.35*pitch, position+.65*pitch
            bounds = (max(0, math.floor(left)), max(0, math.ceil(body.y1+1), math.floor(top)),
                      min(image.width, math.ceil(right)), min(image.height, math.floor(body.y2-1), math.ceil(bottom)))
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                raise ValueError('empty numeric cell')
            crop = image.crop(bounds)
            scale = min(4., 1024/max(crop.size))
            crop = crop.resize((max(1, round(crop.width*scale)), max(1, round(crop.height*scale))))
            output = folder/'crop.png'
            crop.save(output)
            context.update(terminal_side=side, cell_bounds=list(bounds),
                           target_point_in_crop=f'{(px-bounds[0])*scale:.1f},{(position-bounds[1])*scale:.1f}')
            return replace(task, crop_path=str(output), context=context)
        extent=max(50.,pitch*2.)
        left,right=(min(body.x1-extent,px-3),body.x1+3) if side=='L' else (body.x2-3,max(body.x2+extent,px+3))
        top,bottom=position-min(30.,pitch*.9+2),position+3
        # Keep remote signal names outside a review of this measured lead.
        # Include the whole owned number token if its printed box extends
        # beyond the lead; this preserves long and alphanumeric terminal IDs.
        if side=='L':left=max(left,min(body.x1,px)-3)
        else:right=min(right,max(body.x2,px)+3)
        number_tokens={row.get('number_token') for row in observations}
        for token in ir.texts:
            if token.internal_id not in number_tokens or token.raw_text.strip()!=pin.number or token.score<.95:
                continue
            box=_transform_box(token.bbox,ir.image.width,ir.image.height,rotation)
            if box.height>pitch*1.6 or abs(box.center.y-position)>pitch*.9+2:
                continue
            left,right=min(left,box.x1-2),max(right,box.x2+2)
            top,bottom=min(top,box.y1-2),max(bottom,box.y2+2)
        if pin.name and pin.name.isdecimal():
            # Headers can print their sole terminal ID inside the outline.
            # Do not crop that actual glyph away and show only a remote net
            # label such as P4. No interior label is copied into the answer.
            interior=min(body.width,max(20.,pitch*2.))
            if side=='L':right=max(right,body.x1+interior)
            else:left=min(left,body.x2-interior)
            bottom=max(bottom,position+pitch*.6+2)
            context['terminal_label_layout']='terminal ID may be printed inside the outline; remote signal labels are not pin IDs'
            # Bound the exterior to this observed lead, rather than including
            # an unrelated signal label farther along the connected wire.
            if side=='L':left=max(left,px-3)
            else:right=min(right,px+3)
    else:
        extent=min(body.width,max(140.,pitch*5.))
        left,right=(min(body.x1-5,px-3),body.x1+extent) if side=='L' else (body.x2-extent,max(body.x2+5,px+3))
        top,bottom=position-(pitch*.6+2),position+(pitch*.6+2)
        # A fixed width can cut the last characters of a measured name.
        # Include its own complete OCR box in reading coordinates; do not
        # enlarge the crop using unrelated text or a neighbouring terminal.
        name_tokens={row.get('name_token')for row in observations}
        for token in ir.texts:
            if token.internal_id not in name_tokens or token.raw_text.strip()!=pin.name or token.score<.95:
                continue
            box=_transform_box(token.bbox,ir.image.width,ir.image.height,rotation)
            if (box.height>pitch*1.6 or abs(box.center.y-position)>pitch*.5+2
                    or not body.x1<=box.x1<box.x2<=body.x2):
                continue
            left,right=min(left,box.x1-2),max(right,box.x2+2)
            top,bottom=min(top,box.y1-2),max(bottom,box.y2+2)
    margin = float(context.get('readback_margin_px', 0.))
    if not math.isfinite(margin) or not 0 <= margin <= 2:
        raise ValueError('invalid number readback margin')
    bounds=(max(0,math.floor(left-margin)),max(0,math.floor(top-margin)),min(image.width,math.ceil(right+margin)),min(image.height,math.ceil(bottom+margin)))
    if bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:
        raise ValueError('empty row crop')
    crop=image.crop(bounds)
    scale=min(4.,1024/max(crop.size))
    crop=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))))
    context.update(terminal_side=side,target_point_in_crop=f'{(px-bounds[0])*scale:.1f},{(position-bounds[1])*scale:.1f}')
    output=folder/'crop.png';crop.save(output)
    return replace(task,crop_path=str(output),context=context)


def _decision_error(decision: Decision, task: AmbiguityTask) -> str | None:
    if decision.task_id != task.task_id:
        return f"decision task id {decision.task_id!r} does not match routed task"
    try:
        decision.validate_against(task)
    except ValueError as exc:
        return str(exc)
    return None


def _safe_apply(
    ir: CircuitIR,
    task: AmbiguityTask,
    value: str,
    decision: Decision,
    cache_key: str,
) -> tuple[str, str]:
    owner_kind = task.context.get("owner_kind", "")
    object_id = task.context.get("object_id", "")
    field = task.context.get("field", "")
    if owner_kind == "component" and field in {"key", "value", "name"}:
        owner = next((item for item in ir.components if item.internal_id == object_id), None)
        if owner is None:
            return "write_rejected", "component no longer exists"
        current = getattr(owner, field)
        if isinstance(current, str) and current.strip():
            return "write_rejected", "component field is already populated"
        if field == 'key' and any(c.internal_id != object_id and c.key == value for c in ir.components):
            return 'write_rejected', 'component key already belongs to another observed object'
        setattr(owner, field, value)
        _upsert_reasoning_evidence(owner.evidence, task, decision, cache_key, value)
        return "applied", "filled empty component field"
    if owner_kind == "pin" and field in {"number", "name"}:
        owner = next((item for item in ir.pins if item.internal_id == object_id), None)
        if owner is None:
            return "write_rejected", "pin no longer exists"
        current = getattr(owner, field)
        visual_review = ((task.task_type, field) in {('pin.readback_name', 'name'), ('pin.readback_number', 'number')}
                         and task.context.get('previous_value') == current
                         and any(e.metadata.get('kind') == 'visible-stroke-and-text-lanes' for e in owner.evidence))
        if visual_review and value != current:
            from pcbparse.reasoning.visual_ic_tasks import confirmed_numeric_terminal_label
            if confirmed_numeric_terminal_label(owner):
                return 'write_rejected', 'a confirmed interior numeral cannot be replaced by an alternate crop'
        if isinstance(current, str) and current.strip() and not visual_review:
            return "write_rejected", "pin field is already populated"
        if visual_review and field == 'number':
            from pcbparse.reasoning.visual_ic_tasks import number_candidate_echoes_name, number_candidate_is_remote_label
            if number_candidate_echoes_name(owner, value):
                return 'write_rejected', 'a pin-name echo cannot replace its observed decimal number'
            if number_candidate_is_remote_label(ir, owner, value):
                return 'write_rejected', 'a remote label cannot replace its observed terminal number'
        if field == 'number' and not current and _pin_tip_inside_observed_body(ir, owner):
            return 'write_rejected', 'text confirmation cannot validate a terminal tip inside the observed symbol body'
        if field == 'number' and any(p.internal_id != object_id and p.component_id == owner.component_id
                                     and p.number == value for p in ir.pins):
            return 'write_rejected', 'pin number already belongs to another observed terminal'
        setattr(owner, field, value)
        _upsert_reasoning_evidence(owner.evidence, task, decision, cache_key, value)
        return "applied", "filled empty pin field"
    if owner_kind == "junction" and field == "kind":
        owner = next((item for item in ir.junctions if item.internal_id == object_id), None)
        if owner is None:
            return "write_rejected", "junction no longer exists"
        if owner.kind != "ambiguous":
            return "write_rejected", "junction is no longer ambiguous"
        if value not in {"connected_cross", "non_connected_cross"}:
            return "write_rejected", "candidate is not a legal crossing resolution"
        owner.kind = value  # type: ignore[assignment]
        _upsert_reasoning_evidence(owner.evidence, task, decision, cache_key, value)
        return "applied", "resolved ambiguous crossing"
    return "write_rejected", "task target is outside the reasoning write allowlist"


def _pin_tip_inside_observed_body(ir: CircuitIR, pin: Any) -> bool:
    """A printed ID cannot make an interior detection an external terminal.

    Only use the independently measured visible outline, never a detector box.
    Allow two raster pixels for outline thickness/rounding. Existing terminal
    IDs and their geometry are not changed by this write gate.
    """
    component = ir.component_index().get(pin.component_id)
    if component is None or component.category.upper() not in {'BOX', 'IC', 'BOXIC', 'BOX_IC'}:
        return False
    body = component.body_bbox
    if body is None or pin.orientation not in {'L', 'R', 'T', 'B'}:
        return False
    observed = any(
        evidence.source == 'geometry'
        and evidence.metadata.get('method') == 'visible-symbol-body-v2'
        and evidence.metadata.get('gt_access') is False
        and evidence.metadata.get('body_bbox') == body.model_dump()
        for evidence in component.evidence
    )
    if not observed:
        return False
    exterior_distance = {
        'L': body.x1 - pin.point.x,
        'R': pin.point.x - body.x2,
        'T': body.y1 - pin.point.y,
        'B': pin.point.y - body.y2,
    }[pin.orientation]
    return exterior_distance < -2.0


def _upsert_reasoning_evidence(
    evidence: list[Evidence],
    task: AmbiguityTask,
    decision: Decision,
    cache_key: str,
    value: str,
) -> None:
    item = Evidence(
        evidence_id=f"reasoning:{cache_key[:24]}",
        source="qwen",
        view_id="original",
        score=decision.confidence,
        payload_ref=task.task_id,
        metadata={
            "kind": "selective-reasoning",
            "selected_id": decision.selected_id,
            "selected_value": value,
            "decision_source": decision.source,
            "candidate_ids": [candidate.candidate_id for candidate in task.candidates],
            "cache_key": cache_key,
        },
    )
    for index, existing in enumerate(evidence):
        if existing.evidence_id == item.evidence_id:
            evidence[index] = item
            return
    evidence.append(item)


def _ledger_base(
    task: AmbiguityTask,
    *,
    route_score: float,
    estimated_input_tokens: int,
    cache_key: str,
) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "object_id": task.context.get("object_id", ""),
        "field": task.context.get("field", ""),
        "route_score": round(route_score, 8),
        "estimated_input_tokens": estimated_input_tokens,
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "value": candidate.value,
                "score": round(candidate.score, 8),
            }
            for candidate in task.candidates
        ],
        "cache_key": cache_key,
        "cache_hit": False,
        "routed": False,
        "status": "pending",
        "reason": "",
        "decision": None,
    }


def _decision_dict(decision: Decision) -> dict[str, Any]:
    return {
        "task_id": decision.task_id,
        "selected_id": decision.selected_id,
        "confidence": round(float(decision.confidence), 8),
        "source": decision.source,
        "reason": decision.reason,
    }


def _not_routed_reason(
    task: AmbiguityTask,
    router: ValueAwareRouter,
    ctx: RunContext,
    *,
    estimated_input_tokens: int,
) -> str:
    if router.ambiguity_score(task) < router.threshold:
        return "below route threshold"
    if not 1 < len(task.candidates) <= 5:
        return "candidate count outside [2, 5]"
    if ctx.token_budget.remaining_calls < 1:
        return "call budget exhausted"
    if ctx.token_budget.remaining_input_tokens < estimated_input_tokens:
        return "input token budget exhausted"
    if ctx.time_budget.remaining_seconds < router.min_time_reserve_seconds:
        return "time reserve reached"
    return "router declined task"


def _estimated_input_tokens(task: AmbiguityTask) -> int:
    payload = {
        "task": task.task_type,
        "context": task.context,
        "candidates": [
            {"id": item.candidate_id, "value": item.value, "score": item.score}
            for item in task.candidates
        ],
    }
    size = len(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    )
    return max(1, (size + 3) // 4)


def _reasoner_version(
    reasoner: Reasoner,
    config: Mapping[str, Any],
    ctx: RunContext,
) -> str:
    for key in ("reasoner_version", "model_version", "version", "model"):
        value = config.get(key)
        if isinstance(value, (str, int, float)) and str(value):
            return str(value)
    ctx_version = ctx.model_versions.get("qwen")
    if ctx_version:
        return str(ctx_version)
    for attribute in ("version", "model_version", "model"):
        value = getattr(reasoner, attribute, None)
        if isinstance(value, (str, int, float)) and str(value):
            return str(value)
    cls = type(reasoner)
    return f"{cls.__module__}.{cls.__qualname__}:unversioned"


def _image_hash(ir: CircuitIR) -> tuple[str, str]:
    declared = ir.image.sha256.strip().lower()
    if declared:
        return declared, "ir.image.sha256"
    path = Path(ir.image.path)
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest(), "image_file"
    payload = {
        "image_id": ir.image.image_id,
        "path": ir.image.path,
        "width": ir.image.width,
        "height": ir.image.height,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), "image_identity_fallback"


def _reasoning_state_signature(ir: CircuitIR) -> str:
    def upstream_evidence(items: list[Evidence]) -> list[dict[str, Any]]:
        return [
            item.model_dump(mode="json")
            for item in sorted(items, key=lambda value: value.evidence_id)
            if item.source != "qwen"
        ]

    payload = {
        "components": [
            {
                "id": item.internal_id,
                "key": item.key,
                "value": item.value,
                "name": item.name,
                "evidence": upstream_evidence(item.evidence),
            }
            for item in sorted(ir.components, key=lambda value: value.internal_id)
        ],
        "pins": [
            {
                "id": item.internal_id,
                "number": item.number,
                "name": item.name,
                "evidence": upstream_evidence(item.evidence),
            }
            for item in sorted(ir.pins, key=lambda value: value.internal_id)
        ],
        "junctions": [
            {
                "id": item.internal_id,
                "kind": item.kind,
                "evidence": upstream_evidence(item.evidence),
            }
            for item in sorted(ir.junctions, key=lambda value: value.internal_id)
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    safe = {
        str(key): value
        for key, value in config.items()
        if key != "cache_store" and isinstance(value, (str, int, float, bool, type(None), list, tuple, dict))
    }
    encoded = json.dumps(
        safe,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _append_diagnostic(
    ir: CircuitIR,
    *,
    severity: str,
    code: str,
    message: str,
    object_ids: list[str],
) -> None:
    ir.diagnostics.append(
        Diagnostic(
            severity=severity,  # type: ignore[arg-type]
            code=code,
            message=message,
            object_ids=object_ids,
            stage=REASONING_STAGE,
        )
    )


def _clamp01(value: Any) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, converted))


__all__ = ["REASONING_STAGE", "run_selective_reasoning"]
