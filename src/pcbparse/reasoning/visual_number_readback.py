"""Bounded image transcription for missing or unsupported terminal numbers."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import math
from pathlib import Path
import tempfile

from pcbparse.ir.models import Evidence
from pcbparse.reasoning.qwen_client import HTTPQwenReasoner
from pcbparse.reasoning.router import AmbiguityTask


def _numeric_cell_observation(pin):
    return next((e.metadata for e in reversed(pin.evidence)
                 if e.source == 'geometry' and e.metadata.get('kind') == 'visible-stroke-and-text-lanes'
                 and e.metadata.get('gt_access') is False
                 and (e.metadata.get('interior_numeric_terminal_label') is True
                      or e.metadata.get('observed_numeric_terminal_layout') is True)), None)


def _unsupported_association_number(pin):
    """Grant pixel review only for a weak association into an empty field.

    Measured lane reads, model/library consensus and a prior visual review
    take precedence. This gate never clears or replaces the current value.
    """
    if not pin.number or not pin.name or pin.name.isdecimal():
        return None
    observations = [e.metadata for e in pin.evidence
                    if e.source == 'geometry' and e.metadata.get('kind') == 'visible-stroke-and-text-lanes'
                    and e.metadata.get('gt_access') is False]
    if not observations or any(row.get('number') for row in observations):
        return None
    if any(e.source in {'qwen', 'library', 'dataset'}
           or e.metadata.get('kind') == 'exact-model-visible-pin-consensus' for e in pin.evidence):
        return None
    associations = [e.metadata for e in pin.evidence
                    if e.metadata.get('kind') == 'pin-field-resolution'
                    and e.metadata.get('field') == 'number'
                    and e.metadata.get('decision') == 'accepted'
                    and not e.metadata.get('existing_value')
                    and e.metadata.get('chosen_value') == pin.number]
    if not associations:
        return None
    accepted = [c for c in associations[-1].get('candidates', ())
                if c.get('target_status') == 'accepted' and c.get('value') == pin.number]
    if len(accepted) != 1:
        return None
    confidence = accepted[0].get('score_components', {}).get('ocr')
    if type(confidence) not in {int, float} or not math.isfinite(confidence) or not 0 <= confidence < .9:
        return None
    return {'reason': 'weak_association_without_measured_number',
            'previous_number': pin.number, 'text_id': accepted[0].get('text_id'),
            'ocr_confidence': confidence}


def _geometry_supported(ir, targets, ctx, reserve_seconds):
    """Text recognition cannot validate an endpoint that contradicts pixels."""
    import numpy as np
    from PIL import Image
    from pcbparse.graph.palette_wire import hue_map
    from pcbparse.perception.decorated_ic_leads import decorated_terminal_strokes
    from pcbparse.pipeline.context import stable_file_hash
    ctx.time_budget.check(stage='readback_geometry', reserve_seconds=reserve_seconds)
    if not ir.image.sha256 or stable_file_hash(ir.image.path) != ir.image.sha256:
        raise ValueError('readback source image hash mismatch')
    with Image.open(ir.image.path) as image:
        if image.size != (ir.image.width, ir.image.height):
            raise ValueError('readback source dimensions mismatch')
        palette = hue_map(np.asarray(image.convert('RGB')))
    components, decorated, accepted, rejected = ir.component_index(), {}, [], []
    for pin in targets:
        ctx.time_budget.check(stage='readback_geometry', reserve_seconds=reserve_seconds)
        row = next(e.metadata for e in reversed(pin.evidence)
                   if e.metadata.get('kind') == 'visible-stroke-and-text-lanes')
        tip, position = row.get('tip'), row.get('position')
        if (type(tip) not in {int, float} or type(position) not in {int, float}
                or not math.isfinite(tip) or not math.isfinite(position)):
            rejected.append({'pin_id': pin.internal_id, 'reason': 'unmeasured_stroke_endpoint'})
            continue
        point = (tip, position) if pin.orientation in 'LR' else (position, tip)
        if math.dist((pin.point.x, pin.point.y), point) > 5.:
            rejected.append({'pin_id': pin.internal_id, 'reason': 'point_disagrees_with_observed_stroke'})
            continue
        cid = pin.component_id
        if cid not in decorated:
            body = components[cid].body_bbox
            x1, y1, x2, y2 = [round(v) for v in (body.x1, body.y1, body.x2, body.y2)]
            bins, _ = palette
            if not (0 <= x1 < x2 < bins.shape[1] and 0 <= y1 < y2 < bins.shape[0]):
                raise ValueError('observed body outside source image')
            border = np.concatenate([bins[y1:y2+1, x1], bins[y1:y2+1, x2], bins[y1, x1:x2+1], bins[y2, x1:x2+1]])
            colors = border[border >= 0]
            hue = int(np.bincount(colors).argmax()) if len(colors) else -1
            decorated[cid] = decorated_terminal_strokes((x1, y1, x2, y2, hue), palette,
                check=lambda: ctx.time_budget.check(stage='readback_geometry', reserve_seconds=reserve_seconds))
        shapes = [r for r in decorated[cid] if r[0] == pin.orientation]
        if shapes:
            near = [r for r in shapes if abs(r[1]-position) <= 3.]
            distances = [math.dist((pin.point.x, pin.point.y), (r[2], r[1]) if pin.orientation in 'LR' else (r[1], r[2])) for r in near]
            if not distances or min(distances) > 5.:
                rejected.append({'pin_id': pin.internal_id, 'reason': 'decorated_endpoint_disagrees_with_point'})
                continue
        accepted.append(pin)
    return accepted, rejected


def _readback_targets(ir, config):
    from pcbparse.reasoning.orchestrator import _pin_tip_inside_observed_body
    components = ir.component_index()
    targets, reviews = [], {}
    for pin in ir.pins:
        component = components.get(pin.component_id)
        review = _unsupported_association_number(pin) if config.get('visual_unsupported_numbers_enabled') else None
        if ((pin.number and review is None) or component is None or not component.key or not pin.name
                or component.category.upper() not in {'BOX', 'IC', 'BOXIC', 'BOX_IC'}
                or component.body_bbox is None or pin.orientation not in {'L', 'R', 'T', 'B'}):
            continue
        observed_body = any(e.source == 'geometry' and e.metadata.get('method') == 'visible-symbol-body-v2'
            and e.metadata.get('gt_access') is False and e.metadata.get('body_bbox') == component.body_bbox.model_dump()
            for e in component.evidence)
        measured = any(e.source == 'geometry' and e.metadata.get('kind') == 'visible-stroke-and-text-lanes'
            and e.metadata.get('gt_access') is False
            for e in pin.evidence)
        if not observed_body or not measured or _pin_tip_inside_observed_body(ir, pin):
            continue
        if pin.name.isdecimal():
            if not config.get('visual_numeric_cell_readback_enabled') or not _numeric_cell_observation(pin):
                continue
        elif not config.get('visual_missing_numbers_enabled'):
            continue
        targets.append(pin)
        if review:
            reviews[pin.internal_id] = review
    targets.sort(key=lambda p: (p.internal_id not in reviews,
        reviews.get(p.internal_id, {}).get('ocr_confidence', 0.),
        p.component_id, p.orientation, p.point.y, p.point.x, p.internal_id))
    return targets, reviews


def read_missing_visual_numbers(ir, reasoner, ctx, config):
    """Two crop reads confirm a number; geometry and ownership are fixed.

    An interior numeric label already marked as a duplicate may be corrected
    with the number, since the same printed cell supplies both header fields.
    Both reads use the source image, with no list of likely numbers, device
    identity, neighbour numbers, or GT. Agreement is a conservative write gate,
    not evidence of perfect transcription or independent statistical accuracy.
    """
    report = {'version': 3, 'eligible': 0, 'attempted': 0, 'applied': [], 'rejected': [],
              'new_pins': 0, 'coordinates_changed': False, 'gt_access': False}
    if not isinstance(reasoner, HTTPQwenReasoner) or not reasoner.send_images:
        return {**report, 'status': 'vision_reasoner_unavailable'}
    from pcbparse.reasoning.orchestrator import _visual_task
    targets, reviews = _readback_targets(ir, config)
    report['eligible'] = len(targets)
    report['unsupported_existing'] = len(reviews)
    if not targets:
        return {**report, 'status': 'no_eligible_missing_numbers'}
    reserve = max(1., float(config.get('min_time_reserve_seconds', 25.)))
    # At most three fields, two API calls, sharing the global budget.
    # Do not consume the last call on an unconfirmable first observation.
    if (ctx.token_budget.remaining_calls < 2 or ctx.token_budget.remaining_input_tokens < 2048
            or ctx.token_budget.max_output_tokens_per_call < 128
            or ctx.time_budget.remaining_seconds < reserve+2*reasoner.timeout_seconds):
        return {**report, 'status': 'insufficient_confirmation_budget'}
    try:
        targets, rejected = _geometry_supported(ir, targets, ctx, reserve)
        report['rejected'].extend(rejected)
    except Exception as exc:
        return {**report, 'status': 'geometry_check_failed', 'error_type': type(exc).__name__}
    if not targets:
        return {**report, 'status': 'no_geometry_supported_targets'}
    targets = targets[:3]
    tasks = [AmbiguityTask('transcribe-number:'+p.internal_id, 'pin.transcribe_number', (), .0,
             context={'owner_kind': 'pin', 'object_id': p.internal_id, 'field': 'number',
                      'allow_alphanumeric_number': bool(config.get('visual_alphanumeric_numbers_enabled')) and not p.name.isdecimal(),
                      'terminal_label_layout': 'interior_numeric_cell' if p.name.isdecimal()
                                               else 'exterior_number_above_lead'}) for p in targets]
    reads, hashes = [], []
    try:
        with tempfile.TemporaryDirectory(prefix='pcbparse-number-readback-') as directory:
            for view in range(2):
                visual, crop_hashes = [], []
                for i, task in enumerate(tasks):
                    folder = Path(directory)/f'{view}-{i}'
                    folder.mkdir()
                    context = {**task.context, 'readback_margin_px': str(2*view)}
                    crop = _visual_task(replace(task, context=context), ir, folder, tight_pin_crop=True)
                    visual.append(crop)
                    crop_hashes.append(hashlib.sha256(Path(crop.crop_path).read_bytes()).hexdigest())
                report['attempted'] = len(tasks)
                hashes.append(crop_hashes)
                if (ctx.token_budget.remaining_calls < 1 or ctx.time_budget.remaining_seconds <= reserve):
                    return {**report, 'status': 'confirmation_budget_exhausted'}
                reads.append(reasoner.read_numbers(visual, ctx.token_budget,
                    timeout_seconds=max(0., ctx.time_budget.remaining_seconds-reserve)))
                if not reads[-1]:
                    return {**report, 'status': 'unreadable_or_request_failed'}
    except Exception as exc:
        # Preserve valid non-API predictions on timeout, source mismatch or
        # invalid model output; never emit credentials or server response text.
        return {**report, 'status': 'readback_failed', 'error_type': type(exc).__name__}
    proposed = {}
    for pin, task in zip(targets, tasks):
        a, b = (read.get(task.task_id) for read in reads)
        if not a or not b or a[0] != b[0] or min(a[1], b[1]) < .95:
            report['rejected'].append({'pin_id': pin.internal_id, 'reason': 'two_crops_not_confidently_equal'})
            continue
        proposed[pin.internal_id] = a[0]
    counts = Counter((p.component_id, proposed[p.internal_id]) for p in targets if p.internal_id in proposed)
    for i, pin in enumerate(targets):
        value = proposed.get(pin.internal_id)
        if value is None:
            continue
        review = reviews.get(pin.internal_id)
        if (pin.number and (review is None or _unsupported_association_number(pin) != review)):
            report['rejected'].append({'pin_id': pin.internal_id, 'reason': 'existing_number_write_protected'})
            continue
        if (counts[(pin.component_id, value)] != 1
                or any(p.internal_id != pin.internal_id and p.component_id == pin.component_id and p.number == value for p in ir.pins)):
            report['rejected'].append({'pin_id': pin.internal_id, 'reason': 'number_already_in_use'})
            continue
        previous_name, previous_number = pin.name, pin.number
        if previous_number == value:
            report['rejected'].append({'pin_id': pin.internal_id, 'reason': 'existing_number_confirmed'})
            continue
        if pin.name.isdecimal() and pin.name != value:
            row = _numeric_cell_observation(pin)
            # One interior glyph supplies both header fields, but an existing
            # name may be replaced only when upstream explicitly rejected it
            # as a duplicate of a different terminal's observed number.
            if (row is None or row.get('rejected_duplicate_number') != pin.name
                    or not any(p.internal_id != pin.internal_id and p.component_id == pin.component_id
                               and p.number == pin.name for p in ir.pins)):
                report['rejected'].append({'pin_id': pin.internal_id, 'reason': 'existing_numeric_name_disagrees'})
                continue
            pin.name = value
        pin.number = value
        pin.evidence.append(Evidence(evidence_id='visual-number-readback:'+pin.internal_id, source='qwen',
            score=min(reads[0][tasks[i].task_id][1], reads[1][tasks[i].task_id][1]),
            metadata={'kind': 'visible-number-transcription', 'number': value,
                      'crop_sha256': [hashes[0][i], hashes[1][i]], 'confirmed_views': 2,
                      'label_layout': tasks[i].context['terminal_label_layout'],
                      'previous_name': previous_name, 'name_changed': previous_name != pin.name,
                      'previous_number': previous_number, 'replaced_weak_association': review,
                      'existing_terminal_only': True, 'gt_access': False}))
        report['applied'].append({'pin_id': pin.internal_id, 'number': value,
                                  'previous_number': previous_number,
                                  'name_changed': previous_name != pin.name})
    return {**report, 'status': 'completed'}
