"""Read small printed references without changing symbols or inventing IDs."""
from __future__ import annotations

import hashlib
import math
import re
import time

import numpy as np
from PIL import Image

from pcbparse.association.reference_prefixes import DESIGNATOR_PATTERN, designator_compatible
from pcbparse.graph.palette_wire import hue_map
from pcbparse.ir.models import BBox, Evidence, TextToken
from pcbparse.perception.ic_visual_readback import _read_batch

_REFERENCE = re.compile(DESIGNATOR_PATTERN)
_CATEGORIES = {'R', 'RESISTOR', 'C', 'CAPACITOR', 'L', 'INDUCTOR', 'D', 'DIODE',
               'LED', 'Q', 'BJT', 'MOSFET', 'TRANSISTOR', 'CRYSTAL', 'FUSE', 'SWITCH'}


def _reviewable(component):
    return not component.key or any(
        e.source == 'geometry' and e.metadata.get('gt_access') is False
        and e.metadata.get('method') in {'printed-reference-and-two-terminal-strokes', 'visible-capacitor-plates-and-reference'}
        and e.metadata.get('reference') == component.key for e in component.evidence)


def _completes_same_printed_reference(ir, component, value, box):
    previous = component.key
    if not previous or not value.startswith(previous) or not 1 <= len(value)-len(previous) <= 3:
        return False
    for token in ir.texts:
        if token.raw_text.strip() != previous or token.score < .95:
            continue
        old = token.bbox
        overlap = max(0., min(old.x2, box.x2)-max(old.x1, box.x1))*max(0., min(old.y2, box.y2)-max(old.y1, box.y1))
        if (overlap/max(1., old.width*old.height) >= .8 and .5 <= box.height/max(1., old.height) <= 2.
                and box.width > old.width+1):
            return True
    return False


def _owner(ir, value, text):
    """Use all compatible bodies as competitors, including already keyed ones."""
    if not _REFERENCE.fullmatch(value):
        return None
    center = text.center
    for component in ir.components:
        if component.category.upper() in {'BOX', 'IC', 'BOXIC', 'BOX_IC'}:
            body = component.body_bbox or component.bbox
            if body.x1 <= center.x <= body.x2 and body.y1 <= center.y <= body.y2:
                return None
    choices = []
    for component in ir.components:
        if component.category.upper() not in _CATEGORIES or not designator_compatible(component.category, value):
            continue
        body = component.body_bbox or component.bbox
        if body.x1 < center.x < body.x2 and body.y1 < center.y < body.y2:
            continue
        gap = math.hypot(max(body.x1-text.x2, text.x1-body.x2, 0.),
                         max(body.y1-text.y2, text.y1-body.y2, 0.))
        if gap > min(30., max(12., .8*max(body.width, body.height))):
            continue
        rank = gap + .15*math.hypot(center.x-body.center.x, center.y-body.center.y)
        choices.append((rank, component.internal_id, component))
    choices.sort(key=lambda row: row[:2])
    if not choices or len(choices) > 1 and choices[1][0]-choices[0][0] < 4.:
        return None
    return choices[0][2]


def _reference_observations(rgb, bounds, engine, ctx):
    """Detect enlarged lines in original pixels and up to three color layers."""
    x1, y1, x2, y2 = bounds
    crop = rgb[y1:y2, x1:x2]
    bins, ink = hue_map(crop)
    colors = sorted((int(h) for h in np.unique(bins[ink]) if h >= 0),
                    key=lambda h: (-int(((bins == h) & ink).sum()), h))[:3]
    versions = [('original', crop)]
    for hue in colors:
        mask = (bins == hue) & ink
        if mask.sum() < 12:
            continue
        pixels = np.full_like(crop, 255)
        pixels[mask] = crop[mask]
        versions.append(('hue:'+str(hue), pixels))
    found = []
    for mode, pixels in versions:
        enlarged = np.asarray(Image.fromarray(pixels).resize((3*pixels.shape[1], 3*pixels.shape[0])))
        for text, score, polygon, _ in engine.infer_bgr(enlarged[:, :, ::-1].copy(), ctx):
            value = text.strip()
            if not math.isfinite(float(score)) or score < .98 or not _REFERENCE.fullmatch(value):
                continue
            low, high = polygon.min(0)/3., polygon.max(0)/3.
            box = BBox(x1=float(low[0])+x1, y1=float(low[1])+y1,
                       x2=float(high[0])+x1, y2=float(high[1])+y1)
            if not 4 <= box.height <= 32 or not 1.2 <= box.width/max(1., box.height) <= 10:
                continue
            # A wider confirmation crop includes adjacent glyphs that a DB
            # proposal may have cut off (C1 from C17, for example).
            padding = max(3., .75*box.height)
            a, b = max(0, math.floor(low[0]-padding)), max(0, math.floor(low[1]-1))
            c, d = min(crop.shape[1], math.ceil(high[0]+padding)), min(crop.shape[0], math.ceil(high[1]+1))
            if min(low) < 2 or high[0] > crop.shape[1]-2 or high[1] > crop.shape[0]-2:
                continue
            found.append({'value': value, 'score': float(score), 'bbox': box,
                          'mode': mode, 'confirmation_crop': pixels[b:d, a:c]})
    return found, len(versions)


def _isolated_line_crop(crop, text_height):
    """Use a blank scanline to exclude a different annotation row.

    Keep the full horizontal confirmation margin: trimming to a detected
    prefix alone could falsely confirm C1 when the drawing says C17.
    """
    occupied = np.flatnonzero((crop.min(2) < 200).any(1))
    bands = np.split(occupied, np.where(np.diff(occupied) > 1)[0] + 1)
    center = (crop.shape[0] - 1) / 2
    selected = [band for band in bands if len(band) and band[0] <= center <= band[-1]]
    if len(selected) != 1:
        return None
    band = selected[0]
    low, high = max(0, int(band[0]) - 1), min(crop.shape[0], int(band[-1]) + 2)
    if not .55 * text_height <= high - low < crop.shape[0]:
        return None
    return crop[low:high]


def _retry_has_observed_reference(ir, component, rgb):
    """Use existing text as a cheap retry hint, never as a recovered ID."""
    body = component.body_bbox or component.bbox
    for token in ir.texts:
        if (not math.isfinite(token.score) or token.score < .4
                or any(e.source == 'dataset' for e in token.evidence)):
            continue
        text = token.raw_text.strip()
        # Partial R, =C11 and 1R41 are useful uncertainty observations. They
        # cannot be assigned by this screen; both later .98 gates still apply.
        fragments = re.findall(r'[A-Z]{1,8}(?:\d+[A-Z]?)?', text)
        fragments = [v for v in fragments if designator_compatible(component.category, v)]
        if not fragments:
            continue
        box = token.bbox
        gap = math.hypot(max(body.x1-box.x2, box.x1-body.x2, 0.),
                         max(body.y1-box.y2, box.y1-body.y2, 0.))
        if gap > min(30., max(12., .8*max(body.width, body.height))):
            continue
        if not component.key or any(v.startswith(component.key) and len(v) > len(component.key) for v in fragments):
            return True
        if text != component.key or token.score < .95:
            continue
        # A cropped trailing digit can leave ink just beyond a reference
        # box. Require a glyph-height cluster in the same color, not a thin
        # wire or the top pixels of a quantity in the next row.
        x1, y1 = max(0, math.floor(box.x1)), max(0, math.floor(box.y1))
        x2 = min(rgb.shape[1], math.ceil(box.x2 + max(3., .6*box.height)))
        y2 = min(rgb.shape[0], math.ceil(box.y2))
        if x2 <= x1 or y2 <= y1:
            continue
        bins, ink = hue_map(rgb[y1:y2, x1:x2])
        split = max(0, min(bins.shape[1], math.ceil(box.x2)-x1))
        inside = bins[:, :split][ink[:, :split]]
        if inside.size == 0 or split >= bins.shape[1]:
            continue
        colors, counts = np.unique(inside, return_counts=True)
        hue = int(colors[np.argmax(counts)])
        outside = ink[:, split:] & (bins[:, split:] == hue)
        rows = np.flatnonzero(outside.any(1))
        if len(rows) >= max(3., .5*box.height) and outside.sum() >= 4:
            return True
    return False


def _competing_printed_reference(ir, component, value):
    """Repeated visible IDs must not displace a different unkeyed owner."""
    for token in ir.texts:
        if (token.raw_text.strip() != value or not math.isfinite(token.score) or token.score < .98
                or any(e.source == 'dataset' for e in token.evidence)):
            continue
        owner = _owner(ir, value, token.bbox)
        if owner is not None and owner.internal_id != component.internal_id:
            return True
    return False


def recover_local_references(ir, rgb, engine, ctx=None, *, crop_retry=False,
                             line_isolation=False, selective_retry=False):
    """Recover small references only after exact visual readback agreement.

    Raw OCR is preserved; no values, pin IDs, geometry or connections change.
    This is a bounded supplement using the already loaded OCR model.
    """
    started = time.monotonic()
    targets = [c for c in ir.components if _reviewable(c) and c.category.upper() in _CATEGORIES
               and c.score >= .3 and 5 <= min(c.bbox.width, c.bbox.height)
               and max(c.bbox.width, c.bbox.height) <= 100]
    targets.sort(key=lambda c: (c.bbox.width*c.bbox.height, c.internal_id))
    proposals, crops, views, regions, retry_regions = [], {}, 0, 0, 0
    skipped_retries = 0
    skipped_complete = 0
    conflicts = set()
    for component in targets[:32]:
        if ctx is not None and ctx.time_budget.remaining_seconds < 30:
            break
        # A populated reference can only be extended on the same printed
        # glyphs below. Use the existing pixel/fragment screen before running
        # several enlarged OCR views, not just before repeating those views.
        # Unkeyed symbols always retain their first readback opportunity.
        if component.key and selective_retry and not _retry_has_observed_reference(ir, component, rgb):
            skipped_complete += 1
            continue
        body = component.bbox
        radius = min(90., max(30., body.width, body.height))
        bounds = (max(0, math.floor(body.x1-radius)), max(0, math.floor(body.y1-radius)),
                  min(ir.image.width, math.ceil(body.x2+radius)), min(ir.image.height, math.ceil(body.y2+radius)))
        regions += 1
        # OCR detector resize/padding can change at a one-pixel crop boundary.
        # Retry one contained crop only when the primary view has no eligible
        # observation for this owner. Both recognition thresholds stay .98.
        retry_bounds = (bounds[0], bounds[1], bounds[2]-1, bounds[3]-1)
        crops_to_try = [bounds]
        if crop_retry and retry_bounds[2] > bounds[0]+4 and retry_bounds[3] > bounds[1]+4:
            crops_to_try.append(retry_bounds)
        for attempt, region_bounds in enumerate(crops_to_try):
            if attempt and ctx is not None and ctx.time_budget.remaining_seconds < 30:
                break
            if attempt and selective_retry and not _retry_has_observed_reference(ir, component, rgb):
                skipped_retries += 1
                break
            if attempt:
                retry_regions += 1
            observations, count = _reference_observations(rgb, region_bounds, engine, ctx)
            views += count
            before = len(proposals)
            for observation in observations:
                owner = _owner(ir, observation['value'], observation['bbox'])
                if owner is None or owner.internal_id != component.internal_id:
                    continue
                if _competing_printed_reference(ir, component, observation['value']):
                    conflicts.add(observation['value'])
                    continue
                if component.key and not _completes_same_printed_reference(ir, component, observation['value'], observation['bbox']):
                    continue
                key = str(len(proposals))
                crops[key] = observation['confirmation_crop']
                observation = {k: v for k, v in observation.items() if k != 'confirmation_crop'}
                observation['crop_attempt'] = attempt
                proposals.append((component, observation))
            if len(proposals) > before:
                break
    confirmation = _read_batch(engine, crops, ctx) if crops else {}
    isolated, isolated_reads = {}, {}
    if line_isolation:
        for index, (_, observation) in enumerate(proposals):
            key = str(index)
            text, score = confirmation[key]
            if math.isfinite(float(score)) and score >= .98 and text == observation['value']:
                continue
            crop = _isolated_line_crop(crops[key], observation['bbox'].height)
            if crop is not None:
                isolated[key] = crop
        if isolated and (ctx is None or ctx.time_budget.remaining_seconds >= 30):
            isolated_reads = _read_batch(engine, isolated, ctx)
    confirmed = []
    for index, (component, observation) in enumerate(proposals):
        text, score = confirmation[str(index)]
        if str(index) in isolated_reads:
            value, confidence = isolated_reads[str(index)]
            if math.isfinite(float(confidence)) and confidence >= .98 and value == observation['value']:
                text, score = value, confidence
                observation['line_isolation_confirmed'] = True
        if math.isfinite(float(score)) and score >= .98 and text == observation['value']:
            confirmed.append((component, observation, float(score)))
    updates = []
    for component in targets:
        candidates = [(o, score) for c, o, score in confirmed if c is component]
        values = {o['value'] for o, _ in candidates}
        if len(values) != 1:
            continue
        value = next(iter(values))
        # Another owner claiming the same visible ID remains ambiguous.
        if any(c is not component and o['value'] == value for c, o, _ in confirmed):
            continue
        if any(c is not component and c.key == value for c in ir.components):
            continue
        observation, confidence = max(candidates, key=lambda row: (min(row[0]['score'], row[1]), row[0]['mode']))
        box = observation['bbox']
        if component.key and not _completes_same_printed_reference(ir, component, value, box):
            continue
        previous_key = component.key
        digest = hashlib.sha256((component.internal_id+'|'+value+'|'+str(box.model_dump())).encode()).hexdigest()[:20]
        evidence = Evidence(evidence_id='reference-readback:'+digest, source='ocr', score=min(observation['score'], confidence),
            metadata={'method': 'local-color-reference-readback-v1', 'component_id': component.internal_id,
                      'observed_reference': value, 'previous_reference': previous_key, 'coordinate_space': 'original',
                      'readback_mode': observation['mode'], 'detection_recognition_score': observation['score'],
                      'crop_attempt': observation['crop_attempt'],
                      'line_isolation_confirmed': observation.get('line_isolation_confirmed', False),
                      'confirmation_score': confidence, 'descriptor_sha256': engine.descriptor_sha256, 'gt_access': False})
        token = TextToken(internal_id='reference-readback:'+digest, raw_text=value, bbox=box,
                          score=evidence.score, category='designator', details='designator',
                          component_id=component.internal_id, evidence=[evidence])
        ir.texts.append(token)
        if component.name and component.name == previous_key:
            component.name = value
        component.key, component.key_score = value, evidence.score
        component.evidence.append(evidence.model_copy(deep=True))
        updates.append({'component_id': component.internal_id, 'previous_reference': previous_key, 'reference': value,
                        'token_id': token.internal_id, 'bbox': box.model_dump(), 'mode': observation['mode'],
                        'crop_attempt': observation['crop_attempt']})
        updates[-1]['line_isolation_confirmed'] = observation.get('line_isolation_confirmed', False)
    report = {'eligible_components': len(targets), 'processed_regions': regions, 'ocr_views': views,
              'proposed': len(proposals), 'confirmed': len(confirmed), 'updates': updates,
              'conflicting_printed_references': sorted(conflicts),
              'seconds': time.monotonic()-started, 'gt_access': False, 'coordinates_changed': False}
    report.update(crop_retry_enabled=crop_retry, retry_regions=retry_regions)
    report.update(line_isolation_enabled=line_isolation, line_isolation_crops=len(isolated_reads),
                  selective_retry_enabled=selective_retry, retries_skipped_no_observation=skipped_retries,
                  populated_regions_skipped_no_completion_hint=skipped_complete)
    ir.metadata['local_reference_readback'] = report
    return report
