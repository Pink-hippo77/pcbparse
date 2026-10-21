"""Explicit visual review of case ambiguity and cropped IC-label fragments."""
from __future__ import annotations
import hashlib
import re
from pcbparse.reasoning.router import AmbiguityTask, Candidate
from pcbparse.association.pin_identity import PIN_NAME_PATTERN


def visible_name_reselection_tasks(ir):
    components=ir.component_index()
    result=[]
    for pin in ir.pins:
        if confirmed_numeric_terminal_label(pin):
            continue
        observations=[e for e in pin.evidence if e.metadata.get('kind')=='visible-stroke-and-text-lanes']
        component=components.get(pin.component_id)
        if not observations or component is None or not pin.name or not re.fullmatch(PIN_NAME_PATTERN,pin.name):
            continue
        values={pin.name}
        # Letter case is a hypothesis to be checked against the crop, never
        # an unconditional normalization of the prediction.
        name_score=max((e.metadata.get('name_score',0.)for e in observations),default=0.)
        if pin.name != pin.name.upper() and (pin.name.islower() or name_score<.98):
            values.add(pin.name.upper())
        body=component.body_bbox or component.bbox
        anchor_tokens=[next((t for t in ir.texts if t.internal_id==e.metadata.get('name_token')),None)
                 for e in observations]
        anchor_tokens=[t for t in anchor_tokens if t is not None]
        anchors=[t.bbox for t in anchor_tokens]
        if pin.orientation in {'L','R','T','B'}:
            horizontal=pin.orientation in {'L','R'}
            axis=pin.point.y if horizontal else pin.point.x
            peers=[p.point.y if horizontal else p.point.x for p in ir.pins
                   if p.component_id==pin.component_id and p.orientation==pin.orientation and p.internal_id!=pin.internal_id]
            pitch=min([abs(p-axis) for p in peers]+[40.])
            for token in ir.texts:
                text,b=token.raw_text.strip(),token.bbox
                # A confident full read must not be replaced by a contained
                # crop fragment of those same pixels (ADC_Vref -> _Vref).
                clipped=(len(text)<len(pin.name)
                    and any(t.score>=.95 and t.raw_text.strip().casefold()==pin.name.casefold()
                        and max(0.,min(t.bbox.x2,b.x2)-max(t.bbox.x1,b.x1))
                            *max(0.,min(t.bbox.y2,b.y2)-max(t.bbox.y1,b.y1))
                            /max(1.,b.width*b.height)>=.8
                        and (text.casefold() in pin.name.casefold()
                             or (b.height if horizontal else b.width)
                                < .9*(t.bbox.height if horizontal else t.bbox.width)
                             and t.bbox.x1<=b.x1 and t.bbox.y1<=b.y1
                             and t.bbox.x2>=b.x2 and t.bbox.y2>=b.y2)
                        for t in anchor_tokens))
                if clipped:continue
                overlap=any(max(0.,min(a.x2,b.x2)-max(a.x1,b.x1))*max(0.,min(a.y2,b.y2)-max(a.y1,b.y1))
                            /max(1.,min(a.width*a.height,b.width*b.height))>=.7 for a in anchors)
                fragment=(len(text)>len(pin.name) and (text.casefold().startswith(pin.name.casefold())
                          or len(pin.name)<=2 and text.casefold().endswith(pin.name.casefold())))
                if (token.score<.85 or text==pin.name or not (overlap or fragment)
                        or token.pin_id not in {None,pin.internal_id}
                        or not re.fullmatch(PIN_NAME_PATTERN,text)
                        or (b.height if horizontal else b.width)>pitch*1.6
                        or abs((b.center.y if horizontal else b.center.x)-axis)>pitch*.5+2
                        or not body.x1+2<b.center.x<body.x2-2 or not body.y1+2<b.center.y<body.y2-2):
                    continue
                if (pin.orientation=='L' and b.center.x>body.center.x or pin.orientation=='R' and b.center.x<body.center.x
                        or pin.orientation=='T' and b.center.y>body.center.y or pin.orientation=='B' and b.center.y<body.center.y):
                    continue
                values.add(text)
                if text != text.upper() and (text.islower() or token.score<.98):
                    values.add(text.upper())
        if len(values)<2:continue
        # Keep the current value available even when many overlapping crop
        # hypotheses exist. Visual review may always retain that value.
        selected={pin.name,*sorted(values-{pin.name},key=lambda s:(-len(s),s))[:4]}
        candidates=tuple(Candidate('visual-name:'+hashlib.sha256(v.encode()).hexdigest()[:16],v,.5)
                         for v in sorted(selected,key=lambda s:(-len(s),s)))
        result.append(AmbiguityTask(task_id='reasoning:visual-name:'+pin.internal_id,task_type='pin.readback_name',
            candidates=candidates,primary_score=.5,graph_inconsistency_score=.5,downstream_value_score=1.,hardcase_prior=1.,
            context={'owner_kind':'pin','object_id':pin.internal_id,'field':'name','previous_value':pin.name,
                     'review_crop_version':'complete-observed-name-v2',
                     'review_reason':'printed case or potentially clipped OCR; select only the visible name at the target terminal'}))
    return result


def number_candidate_echoes_name(pin, value):
    """A signal/name crop cannot overturn an observed decimal terminal ID.

    Keep genuine alphanumeric IDs and other observed alternatives eligible.
    A clipped port label such as P1 from P1.0/ADC0 is also a name echo.
    Require a signal separator after the fragment: P1 must not suppress a
    distinct printed BGA terminal P10 merely because the characters overlap.
    """
    if not (pin.number and pin.number.isdecimal() and pin.name and value
            and not value.isdecimal()):
        return False
    name, candidate = pin.name.casefold(), value.casefold()
    if name == candidate:
        return True
    # Exterior port names and interior multiplexed names can share a stem.
    # A partial OCR crop of either is not a conflicting terminal number.
    return any(part.startswith(candidate) and len(part) > len(candidate)
               and part[len(candidate)] in '._[('
               for part in name.split('/'))


def visible_number_reselection_tasks(ir):
    """Review conflicting reads of an existing visible terminal, never invent IDs."""
    result = []
    for pin in ir.pins:
        if confirmed_numeric_terminal_label(pin):
            continue
        observations = [e.metadata for e in pin.evidence
                        if e.metadata.get('kind') == 'visible-stroke-and-text-lanes']
        values = {str(v) for row in observations for v in row.get('number_alternatives', [])}
        values = {v for v in values if not number_candidate_echoes_name(pin, v)
                  and not number_candidate_is_remote_label(ir, pin, v)}
        if not pin.number or not values:
            continue
        values.add(pin.number)
        if len(values) < 2:
            continue
        review_values = sorted({pin.number, *sorted(values-{pin.number})[:4]})
        # A single printed B/8 (or similar letter/digit) conflict changes the
        # terminal identity and every network containing it. Review these
        # observed alternatives before less specific crop disagreements, so a
        # small call budget is not exhausted in component-ID order.
        glyph_conflict = any(
            len(a) == len(b) and len(a) > 1
            and len(differences := [(x, y) for x, y in zip(a, b) if x != y]) == 1
            and ((differences[0][0].isalpha() and differences[0][1].isdigit())
                 or (differences[0][1].isalpha() and differences[0][0].isdigit()))
            for a in review_values for b in review_values if a < b)
        candidates = tuple(Candidate('visual-number:'+hashlib.sha256(v.encode()).hexdigest()[:16], v, .5)
                           for v in review_values)
        result.append(AmbiguityTask(task_id='reasoning:visual-number:'+pin.internal_id,
            task_type='pin.readback_number', candidates=candidates, primary_score=.1 if glyph_conflict else .3,
            graph_inconsistency_score=.75, downstream_value_score=1., hardcase_prior=1.,
            context={'owner_kind':'pin', 'object_id':pin.internal_id, 'field':'number',
                     'previous_value':pin.number,
                     'review_crop_version':'observed-terminal-number-v3',
                     'terminal_label_layout': ('interior_numeric_cell' if pin.name and pin.name.isdecimal() and any(
                         row.get('interior_numeric_terminal_label') for row in observations)
                         else 'exterior_number_lane'),
                     'review_reason':'conflicting OCR crops; read only the printed number at the target terminal'}))
    return result


def confirmed_numeric_terminal_label(pin):
    """Preserve an interior numeral agreed by strong name and number reads.

    Headers may use one printed numeral for both fields. A weak alternate
    crop must not turn that consistent identity into a different terminal.
    Uncertain, differing and nonnumeric reads remain eligible for review.
    """
    return bool(pin.number and pin.number == pin.name and pin.number.isdecimal()
                and any(row.get('kind') == 'visible-stroke-and-text-lanes'
                        and row.get('interior_numeric_terminal_label') is True
                        and row.get('name') == pin.name and row.get('number') == pin.number
                        and min(row.get('name_score', 0.), row.get('number_score', 0.)) >= .99
                        for row in (e.metadata for e in pin.evidence)))


def number_candidate_is_remote_label(ir, pin, value):
    """A label beyond the measured terminal cannot replace its confident ID.

    Require an actual OCR observation of the alternative, not a prefix or a
    device-specific naming rule. Overlapping observations of the terminal
    itself remain eligible, including alphanumeric package numbers.
    """
    if not (pin.number and value != pin.number and pin.orientation in {'L', 'R', 'T', 'B'}
            and pin.base is not None and any(
                e.metadata.get('kind') == 'visible-stroke-and-text-lanes'
                and e.metadata.get('number') == pin.number
                and e.metadata.get('number_score', 0.) >= .95 for e in pin.evidence)):
        return False
    horizontal = pin.orientation in {'L', 'R'}
    axis = pin.point.y if horizontal else pin.point.x
    peers = [abs((p.point.y if horizontal else p.point.x)-axis) for p in ir.pins
             if p.component_id == pin.component_id and p.orientation == pin.orientation
             and p.internal_id != pin.internal_id]
    pitch = max(8., min([v for v in peers if v > 3]+[40.]))
    owner=ir.component_index().get(pin.component_id)
    if owner is None:return False
    body=owner.body_bbox or owner.bbox
    def own_side(box):
        if pin.orientation=='L':return box.center.x<=body.center.x
        if pin.orientation=='R':return box.center.x>=body.center.x
        if pin.orientation=='T':return box.center.y<=body.center.y
        return box.center.y>=body.center.y
    tokens = [t for t in ir.texts if t.raw_text.strip() == value and t.score >= .85
              and own_side(t.bbox)
              and abs((t.bbox.center.y if horizontal else t.bbox.center.x)-axis) <= pitch*.9+2]
    if not tokens:
        return False
    def beyond(box):
        if pin.orientation == 'L':return box.x2 < min(pin.point.x, pin.base.x)-3
        if pin.orientation == 'R':return box.x1 > max(pin.point.x, pin.base.x)+3
        if pin.orientation == 'T':return box.y2 < min(pin.point.y, pin.base.y)-3
        return box.y1 > max(pin.point.y, pin.base.y)+3
    return all(beyond(t.bbox) for t in tokens)
