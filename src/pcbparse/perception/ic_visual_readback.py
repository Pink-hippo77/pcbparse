"""Read observed IC outlines, terminal strokes and their separate text lanes.

This is an image-only supplement to the learned detector. A closed rectangle
alone is insufficient: an external IC reference and multiple visibly labelled
terminal strokes must also agree. No package template supplies unseen pins.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from itertools import product

import numpy as np

from pcbparse.graph.palette_wire import hue_map
from pcbparse.ir.models import BBox, Component, Evidence, Pin, Point
from pcbparse.perception.onnx_ocr_processing import recognition_tensor, decode_ctc
from pcbparse.perception.symbol_body import groups, readback_rgb
from pcbparse.association.reference_prefixes import BOX_REFERENCE_PATTERN
from pcbparse.association.pin_identity import PIN_NAME_PATTERN

_REFERENCE = re.compile(BOX_REFERENCE_PATTERN)
_CONNECTOR_REFERENCE = re.compile(r'(?:J|P|H|CN|CON|JP|USB)\d+[A-Z]?$')
_NUMBER = re.compile(r'(?:\d{1,4}|[A-Z]{1,2}\d{1,3})$')
_NAME = re.compile(PIN_NAME_PATTERN)
_QUANTITY = re.compile(r'\d+(?:[.,]\d+)?\s*[A-Za-zΩµμ%]+$')
_IC = {'BOX', 'IC', 'BOXIC', 'BOX_IC'}
_NEARBY_BOX_CELLS = tuple(product((-1, 0, 1), repeat=4))


def _runs(mask, minimum):
    delta = np.diff(np.pad(mask.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    starts, ends = np.argwhere(delta == 1), np.argwhere(delta == -1)
    return [(int(y), int(x), int(z)) for (y, x), (_, z) in zip(starts, ends) if z-x >= minimum]


def _vertical_edges(mask):
    edges=mask.copy()
    edges[:,1:-1] &= ~(mask[:,:-2] & mask[:,2:])
    # Pin glyphs and lettering can interrupt an otherwise measured edge.
    # Join short gaps in a single column; retain the separate horizontal
    # edge checks before accepting any rectangle.
    runs=[]
    for x,a,b in _runs(edges.T,1):
        if runs and runs[-1][0]==x and a-runs[-1][2]<=8:
            runs[-1]=(x,runs[-1][1],b)
        else:runs.append((x,a,b))
    return [r for r in runs if r[2]-r[1]>=35]


def closed_rectangles(rgb, *, palette=None, check=None, edge_only=False, minimum_width=38):
    return closed_rectangle_sets(rgb, palette=palette, check=check, edge_only=edge_only,
                                 minimum_widths=(minimum_width,))[minimum_width]


def closed_rectangle_sets(rgb, *, palette=None, check=None, edge_only=False, minimum_widths=(12, 38)):
    """Share the image scan while preserving each width's own deduplication."""
    widths = tuple(minimum_widths)
    if not widths or len(set(widths)) != len(widths) or any(type(w) is not int or w < 1 for w in widths):
        raise ValueError('distinct positive integer widths required')
    smallest = min(widths)
    bins, ink = palette if palette is not None else hue_map(rgb)
    result = {w: [] for w in widths}
    accepted = {w: defaultdict(list) for w in widths}
    masks = [(-1, ink & (bins < 0))]
    masks += [(i, ink & (bins == i)) for i in range(12) if np.count_nonzero(bins == i) > 100]
    for hue, mask in masks:
        if check:
            check()
        vertical = _vertical_edges(mask) if edge_only else _runs(mask.T, 35)
        # Only lines whose top and bottom differ by at most three pixels can
        # form a rectangle. Preserve the original pair order and predicates.
        buckets = defaultdict(list)
        for j, (x, a, b) in enumerate(vertical):
            buckets[(a//4, b//4)].append((j, x, a, b))
        for i, (x1, a, b) in enumerate(vertical):
            if check and i % 128 == 0:
                check()
            neighbors = sorted((row for da in (-1, 0, 1) for db in (-1, 0, 1)
                                for row in buckets.get((a//4+da, b//4+db), ())
                                if row[0] > i), key=lambda row: row[0])
            for pair_index, (_, x2, c, d) in enumerate(neighbors):
                if check and pair_index % 256 == 0:
                    check()
                if x2-x1 < smallest or abs(a-c) > 3 or abs(b-d) > 3:
                    continue
                y1, y2 = round((a+c)/2), round((b+d)/2)-1
                if y2-y1 < 38:
                    continue
                edges = [mask[max(0, y-2):y+3, x1:x2+1].any(0).mean() for y in (y1, y2)]
                if min(edges) < .95:
                    continue
                coordinates = (x1, y1, x2, y2)
                cell = tuple(v//5 for v in coordinates)
                for width in widths:
                    if x2-x1 < width:
                        continue
                    seen = accepted[width]
                    if any(abs(old[0]-x1) < 5 and abs(old[1]-y1) < 5
                           and abs(old[2]-x2) < 5 and abs(old[3]-y2) < 5
                           for da, db, dc, dd in _NEARBY_BOX_CELLS
                           for old in seen.get((cell[0]+da, cell[1]+db, cell[2]+dc, cell[3]+dd), ())):
                        continue
                    result[width].append((x1, y1, x2, y2, hue))
                    seen[cell].append(coordinates)
    return {w: sorted(boxes, key=lambda b: (b[1], b[0])) for w, boxes in result.items()}


def _bending_wire_support(bins,ink,x,y,nx,ny,hue):
    """A wire may turn at the terminal instead of continuing along its ray."""
    from pcbparse.graph.raster_trace import label_pixels
    a,z=max(0,x-7),max(0,y-7)
    colors=bins[z:y+8,a:x+8];visible=ink[z:y+8,a:x+8]
    yy,xx=np.indices(colors.shape);xx=xx+a;yy=yy+z
    for wire_hue in np.unique(colors[visible&(colors>=0)]):
        if min((int(wire_hue)-hue)%12,(hue-int(wire_hue))%12)<2:continue
        distance=np.minimum((colors-wire_hue)%12,(wire_hue-colors)%12)
        foreign=visible&(colors>=0)&(distance<=1)&((xx-x)*nx+(yy-y)*ny>=-1)
        _,components=label_pixels(foreign)
        for pixels in components[1:]:
            if len(pixels)<6 or max(np.ptp(pixels,axis=0))<5:continue
            dx,dy=pixels[:,1]+a-x,pixels[:,0]+z-y
            if np.min(dx*dx+dy*dy)<=8:return True
    return False


def _colored_lead_end(bins, ink, fixed, position, normal, horizontal, hue):
    """Measure a symbol lead ending at a separately colored outgoing wire.

    Long printed leads must not fall back to the historical 10px default.
    An uninterrupted same-color wire supplies no observable terminal boundary.
    """
    symbol=[];foreign=[];occupied=[]
    for t in range(2,130):
        x,y=(fixed+normal*t,position) if horizontal else (position,fixed+normal*t)
        if not 1<=x<ink.shape[1]-1 or not 1<=y<ink.shape[0]-1:break
        colors=bins[y-1:y+2,x] if horizontal else bins[y,x-1:x+2]
        visible=ink[y-1:y+2,x] if horizontal else ink[y,x-1:x+2]
        symbol.append(bool(((colors==hue)&visible).any()))
        occupied.append(bool(visible.any()))
        colors=colors[visible&(colors>=0)]
        foreign.append([int(v) for v in colors if min((int(v)-hue)%12,(hue-int(v))%12)>=2])
    for i in range(len(symbol)-6):
        if not symbol[i] or any(symbol[i+1:i+4]):continue
        end=i+2
        if not 5<=end<=120 or not all(symbol[:3]) or np.mean(symbol[:i+1])<.95:return None
        # Antialiasing can leave a neutral pixel and a neighboring hue bin
        # at the transition. Require continuing ink and three agreeing color
        # cross sections beyond that small fringe, never across blank space.
        votes=foreign[i+2:i+7]
        if np.mean(occupied[i+1:i+7])>=.8:
            for wire_hue in range(12):
                if min((wire_hue-hue)%12,(hue-wire_hue)%12)<2:continue
                if sum(any(min((c-wire_hue)%12,(wire_hue-c)%12)<=1 for c in v) for v in votes)>=3:return end
        x,y=(fixed+normal*end,position) if horizontal else (position,fixed+normal*end)
        if _bending_wire_support(bins,ink,x,y,normal if horizontal else 0,0 if horizontal else normal,hue):return end
        return None
    return None


def terminal_strokes(rgb, rectangle, *, palette=None, shaped_leads=False, short_colored_leads=False):
    x1, y1, x2, y2, hue = rectangle
    bins, ink = palette if palette is not None else hue_map(rgb)
    found = []
    for side in 'LRTB':
        horizontal = side in 'LR'
        fixed = {'L': x1, 'R': x2, 'T': y1, 'B': y2}[side]
        lo, hi = (y1, y2) if horizontal else (x1, x2)
        normal = -1 if side in 'LT' else 1
        # Evaluate the same 15 samples and transverse tolerances for all
        # positions together. The arrays cover only a narrow exterior strip.
        # Clipped repeated edge samples preserve an ``any`` reduction; anchor
        # validity below still rejects incomplete rays at the image border.
        axis = np.arange(lo+6, hi-5)
        if not len(axis):
            continue
        distances = fixed+normal*np.arange(1, 16)
        transverse = axis[:, None, None]+np.arange(-2, 3)[None, None, :]
        if horizontal:
            yy = np.clip(transverse, 0, ink.shape[0]-1)
            xx = np.clip(distances[None, :, None], 0, ink.shape[1]-1)
            valid_axis = (axis >= 1) & (axis < ink.shape[0]-1)
            valid_distance = (distances >= 1) & (distances < ink.shape[1]-1)
        else:
            xx = np.clip(transverse, 0, ink.shape[1]-1)
            yy = np.clip(distances[None, :, None], 0, ink.shape[0]-1)
            valid_axis = (axis >= 1) & (axis < ink.shape[1]-1)
            valid_distance = (distances >= 1) & (distances < ink.shape[0]-1)
        strip = ink[yy, xx]
        support = strip[:, :, 1:4].any(2)
        complete = valid_axis & valid_distance.all()
        positions = axis[complete & support[:, :3].all(1) & (support.mean(1) >= .9)]
        wide = strip.any(2)
        wide_positions = (axis[complete & wide[:, :3].all(1) & (wide.mean(1) >= .9)]
                          if shaped_leads else [])
        colored_positions = []
        if hue >= 0 and valid_distance[:6].all():
            colors = bins[yy, xx][:, :6, 1:4]
            symbol = ((colors == hue) & strip[:, :6, 1:4]).any(2).all(1)
            colored_positions = axis[valid_axis & symbol]
        position_groups=[g for g in groups(positions) if len(g)<=8]
        if short_colored_leads and hue>=0:
            colored_groups=[g for g in groups(colored_positions) if len(g)<=5]
            if len(colored_groups)>=2:position_groups=colored_groups
        if shaped_leads:
            wide_groups=[]
            for p in wide_positions:
                if not wide_groups or p-wide_groups[-1][-1]>6:wide_groups.append([])
                wide_groups[-1].append(p)
            wide_groups=[g for g in wide_groups if g[-1]-g[0]+1<=11]
            if len(wide_groups)>1.5*len(position_groups)+1:
                position_groups=wide_groups
        # A short symbol lead can meet a wire that turns before the 15px
        # straight-ray window ends. Require the measured symbol-to-wire
        # transition before adding such a row; body lettering alone is not
        # sufficient evidence of a terminal.
        short_rows = []
        if hue >= 0 and not short_colored_leads:
            ordinary = [round(float(np.mean(g))) for g in position_groups]
            for g in groups(colored_positions):
                if len(g) > 5:
                    continue
                p = round(float(np.mean(g)))
                if any(abs(p-other) <= 4 for other in ordinary):
                    continue
                length = _colored_lead_end(bins, ink, fixed, p, normal, horizontal, hue)
                if length is not None:
                    short_rows.append((side, p, fixed+normal*length))
        for group in position_groups:
            p = round(float(np.mean(group)))
            # In color CAD drawings the red symbol lead ends where the wire
            # changes color. Use that visible end when it can be measured.
            length = 10
            if hue >= 0:
                same = []
                for t in range(2, 33):
                    x, y = (fixed+normal*t, p) if horizontal else (p, fixed+normal*t)
                    if not 1 <= x < ink.shape[1]-1 or not 1 <= y < ink.shape[0]-1:
                        break
                    local = bins[y-1:y+2, x] if horizontal else bins[y, x-1:x+2]
                    same.append(bool((local == hue).any()))
                ends = [i+2 for i in range(len(same)-2) if same[i] and not any(same[i+1:i+3])]
                if ends and 5 <= ends[0] <= 25:
                    length = ends[0]
                measured=_colored_lead_end(bins,ink,fixed,p,normal,horizontal,hue)
                if measured is not None:length=measured
            found.append((side, p, fixed+normal*length))
        found.extend(short_rows)
    return found


def _transform_box(box, width, height, rotation):
    if not rotation:
        return box
    if rotation == 1:
        return BBox(x1=box.y1, y1=width-1-box.x2, x2=box.y2, y2=width-1-box.x1)
    return BBox(x1=height-1-box.y2, y1=box.x1, x2=height-1-box.y1, y2=box.x2)


def has_no_connect_marker(rgb, point, side):
    """Verify an X beside the terminal, followed by blank outgoing space."""
    x0,y0=max(0,round(point.x)-30),max(0,round(point.y)-30)
    local=rgb[y0:round(point.y)+31,x0:round(point.x)+31]
    bins,ink=hue_map(local)
    chromatic=(bins>=0)&ink
    mask=chromatic if chromatic.any() else ink
    normal={'L':(-1,0),'R':(1,0),'T':(0,-1),'B':(0,1)}.get(side)
    if normal is None:return False
    def hit(x,y):
        x,y=round(x),round(y)
        return bool(1<=x<mask.shape[1]-1 and 1<=y<mask.shape[0]-1 and mask[y-1:y+2,x-1:x+2].any())
    for offset in range(0,13):
        x,y=point.x-x0+normal[0]*offset,point.y-y0+normal[1]*offset
        arms=[sum(hit(x+sx*d,y+sy*d) for d in range(2,7)) for sx,sy in ((-1,-1),(1,1),(-1,1),(1,-1))]
        if min(arms)<4:continue
        if sum(hit(x+normal[0]*d,y+normal[1]*d) for d in range(10,17))<=1:
            return True
    return False


def _has_functional_no_connect_marker(rgb, point, side):
    """Require a same-color X and clear surroundings for a functional pin.

    Unlike a terminal already printed NC, a functional label provides no
    supporting evidence of disconnection. Black wires must also veto the X;
    mixing differently colored text strokes can otherwise manufacture one.
    """
    x0, y0 = max(0, round(point.x)-30), max(0, round(point.y)-30)
    local = rgb[y0:round(point.y)+31, x0:round(point.x)+31]
    bins, ink = hue_map(local)
    normal = {'L': (-1, 0), 'R': (1, 0), 'T': (0, -1), 'B': (0, 1)}.get(side)
    if normal is None or not ink.any():
        return False

    def hit(mask, x, y):
        x, y = round(x), round(y)
        return bool(1 <= x < mask.shape[1]-1 and 1 <= y < mask.shape[0]-1
                    and mask[y-1:y+2, x-1:x+2].any())

    masks = [ink & (bins == int(hue)) for hue in np.unique(bins[ink])]
    for offset in range(13):
        x, y = point.x-x0+normal[0]*offset, point.y-y0+normal[1]*offset
        if sum(hit(ink, x+normal[0]*d, y+normal[1]*d) for d in range(10, 17)) > 1:
            continue
        if any(sum(hit(ink, x-normal[1]*sign*d, y+normal[0]*sign*d)
                   for d in range(4, 8)) > 1 for sign in (-1, 1)):
            continue
        for mask in masks:
            if not hit(mask, x, y):
                continue
            arms = [sum(hit(mask, x+sx*d, y+sy*d) for d in range(2, 7))
                    for sx, sy in ((-1, -1), (1, 1), (-1, 1), (1, -1))]
            if min(arms) >= 4:
                return True
    return False


def _read_crop(rgb, crop_box, *, neutral_text=False):
    x1, y1, x2, y2 = map(round, crop_box)
    crop = rgb[max(0, y1):min(rgb.shape[0], y2), max(0, x1):min(rgb.shape[1], x2)]
    if not crop.size:
        return None
    if neutral_text:
        # Colored no-connect crosses can overlap the black pin number.
        # Keep achromatic printed glyphs when they are present in the crop.
        chroma=crop.max(2).astype(int)-crop.min(2).astype(int)
        neutral=(chroma<35)&(crop.min(2)<170)
        if neutral.sum()>=4:
            crop=crop.copy()
            crop[chroma>=35]=255
    # Remove excessive blank margins, while preserving printed case and all
    # punctuation. This matters for short labels in wide component bodies.
    ink = crop.min(2) < 170
    ys, xs = np.nonzero(ink)
    if not len(xs):
        return None
    crop = crop[max(0, ys.min()-2):min(crop.shape[0], ys.max()+3),
                max(0, xs.min()-2):min(crop.shape[1], xs.max()+3)]
    return crop


def _read(engine, rgb, crop_box, ctx, *, neutral_text=False):
    engine._check(ctx, 'ic_text_readback')
    crop = _read_crop(rgb, crop_box, neutral_text=neutral_text)
    if crop is None:
        return '', 0.
    tensor = recognition_tensor(crop[:, :, ::-1].copy(), minimum_width=engine.minimum_width,
                                maximum_width=engine.maximum_width)
    output = engine.recognizer.run(None, {engine.recognizer_input: tensor[None]})[0]
    text, score = decode_ctc(output, engine.charset)[0]
    return text.strip(), score


def _read_batch(engine, crops, ctx):
    """Recognize independent crops in equal-width batches without extra padding."""
    widths = defaultdict(list)
    result = {}
    for key, rgb in crops.items():
        engine._check(ctx, 'rail_text_readback')
        crop = _read_crop(rgb, (0, 0, rgb.shape[1], rgb.shape[0]))
        if crop is None:
            result[key] = ('', 0.)
            continue
        tensor = recognition_tensor(crop[:, :, ::-1].copy(), minimum_width=engine.minimum_width,
                                    maximum_width=engine.maximum_width)
        widths[tensor.shape[-1]].append((key, tensor))
    for rows in widths.values():
        for start in range(0, len(rows), 8):
            engine._check(ctx, 'rail_text_readback')
            batch = rows[start:start+8]
            output = engine.recognizer.run(None, {engine.recognizer_input: np.stack([t for _, t in batch])})[0]
            values = decode_ctc(output, engine.charset)
            if len(values) != len(batch):
                raise ValueError('readback OCR batch cardinality mismatch')
            result.update((key, (text.strip(), score)) for (key, _), (text, score) in zip(batch, values))
    return result


def _nearest_number_box(rgb, body, side, position, pitch, *, baseline_text=False):
    """Isolate the printed glyph cluster nearest the symbol boundary.

    A broad OCR lane can also contain another component's number. Use only
    glyphs above this terminal and keep multi-character clusters. Printed
    numbers may themselves be colored; the body border is excluded explicitly.
    Coordinates are in the already rotated image, so all four sides share it.
    """
    font = min(pitch*.65, 25.)
    extent = max(30., pitch*1.9)
    x1, x2 = (body.x1-extent, body.x1) if side == 'L' else (body.x2+1, body.x2+extent)
    x1, x2 = max(0, round(x1)), min(rgb.shape[1], round(x2))
    y1 = max(0, round(position-font-2))
    y2 = min(rgb.shape[0], round(position+(2 if baseline_text else -1)))
    crop = rgb[y1:y2, x1:x2]
    if not crop.size:
        return None
    mask = crop.min(2) < 170
    if baseline_text:
        # Some small glyphs touch the terminal baseline. Measure achromatic
        # text separately from a visible colored lead before extending the
        # crop below the old cutoff. Monochrome/colored text keeps the normal
        # path; this retry must not join it to a black wire or adjacent row.
        chroma = crop.max(2).astype(int)-crop.min(2).astype(int)
        baseline = max(0, round(position)-y1)
        strip = slice(max(0,baseline-1), baseline+2)
        if not ((chroma[strip] >= 35) & mask[strip]).any():
            return None
        mask &= chroma < 35
    # Reject a full-height outline column without cropping a number which
    # touches it. Fixed inward margins can cut off an entire narrow digit.
    edge = range(max(0,mask.shape[1]-2),mask.shape[1]) if side=='L' else range(min(2,mask.shape[1]))
    for column in edge:
        if mask[:,column].all():
            mask[:,column] = False
    # A monochrome lead can touch the last scanline. It is not printed text.
    mask[mask.sum(1) > max(12, .7*mask.shape[1])] = False
    columns = np.flatnonzero(mask.any(0))
    if not len(columns):
        return None
    clusters = []
    for column in columns:
        if not clusters or column-clusters[-1][-1] > max(2, round(font*.35)):
            clusters.append([])
        clusters[-1].append(int(column))
    cluster = clusters[-1] if side == 'L' else clusters[0]
    lo, hi = cluster[0], cluster[-1]+1
    ys = np.flatnonzero(mask[:, lo:hi].any(1))
    if not len(ys) or mask[:, lo:hi].sum() < 5 or ys[-1]-ys[0] < 3:
        return None
    # Remote text is not proof of a terminal number.
    gap = x2-(x1+hi) if side == 'L' else lo
    if gap > max(6., font*1.8):
        return None
    return (x1+max(0, lo-1), y1+max(0, int(ys[0])-1),
            x1+min(mask.shape[1], hi+1), y1+min(mask.shape[0], int(ys[-1])+2))


def _read_baseline_number(engine, rgb, body, side, position, pitch, ctx):
    """Confirm an uncertain number at a colored lead using two text views."""
    box = _nearest_number_box(rgb,body,side,position,pitch,baseline_text=True)
    if box is None:
        return '', 0.
    neutral, score = _read(engine,rgb,box,ctx,neutral_text=True)
    if score < .98 or not _NUMBER.fullmatch(neutral) or neutral == '0':
        return '', 0.
    original, other_score = _read(engine,rgb,box,ctx)
    if original != neutral or other_score < .98:
        return '', 0.
    return neutral, min(score,other_score)


def _row_text(ir, rectangle, side, p, pitch, rotated, rotation, engine, ctx, *, refine_name_lanes=False, numeric_names=False, decorated_lead=False, single_name_bank=False, localized_numbers=False):
    body = BBox(x1=rectangle[0], y1=rectangle[1], x2=rectangle[2], y2=rectangle[3])
    width, height = ir.image.width, ir.image.height
    if rotation:
        body = _transform_box(body, width, height, rotation)
        p = width-1-p if rotation == 1 else p
        side = ({'T': 'L', 'B': 'R'} if rotation == 1 else {'T': 'R', 'B': 'L'})[side]
    localized_box = (_nearest_number_box(rotated,body,side,p,pitch)
                     if localized_numbers else None)
    names, numbers = [], []
    name_boxes = {}
    number_boxes = {}
    def valid_name(value, score):
        return bool(_NAME.fullmatch(value) or
                    numeric_names and re.fullmatch(r'\d{1,4}', value) and score >= .9)
    for token in ir.texts:
        value = token.raw_text.strip()
        box = _transform_box(token.bbox, width, height, rotation)
        center = box.center
        max_height=max(12.,pitch*1.6) if numeric_names else pitch*1.2
        if token.score < .55 or box.height > max_height:
            continue
        if (body.x1+2 <= center.x <= body.x2-2 and body.y1+2 <= center.y <= body.y2-2
                and abs(center.y-p) <= pitch*.43+1 and valid_name(value, token.score)
                and not _REFERENCE.fullmatch(value)):
            if not single_name_bank and (side == 'L' and center.x > body.center.x or side == 'R' and center.x < body.center.x):
                continue
            gap = max(0., box.x1-body.x1 if side == 'L' else body.x2-box.x2)
            rank = float(token.score)+.005*len(value)-(.02*(gap+abs(center.y-p)) if refine_name_lanes else 0.)
            names.append((rank, value, token.internal_id, float(token.score)))
            name_boxes[token.internal_id] = box
        outside = (body.x1-max(65, pitch*2.2) <= box.x1 and box.x2 <= body.x1+2) if side == 'L' else (
            body.x2-2 <= box.x1 and box.x2 <= body.x2+max(65, pitch*2.2))
        if outside and localized_numbers:
            # An occluded outline needs a separately located number glyph;
            # a nearby device reference cannot supply that terminal identity.
            outside = False
            if localized_box is not None:
                a,z,c,d = localized_box
                overlap = max(0.,min(box.x2,c)-max(box.x1,a))*max(0.,min(box.y2,d)-max(box.y1,z))
                outside = overlap/max(1.,min(box.width*box.height,(c-a)*(d-z))) >= .5
        if outside and _NUMBER.fullmatch(value) and (not value.isdecimal() or int(value) > 0):
            if p-pitch*.7 <= box.center.y <= p+2 and abs(box.y2-p) <= pitch*.55:
                # Close to the wire's upper edge beats the next row, even if
                # both OCR tokens have near-perfect confidence.
                gap = max(0., body.x1-box.x2 if side=='L' else box.x1-body.x2)
                rank = float(token.score)-.02*abs(box.y2-p)+.001*len(value)-.03*gap
                numbers.append((rank, value, token.internal_id, float(token.score)))
                number_boxes[token.internal_id] = box
    if refine_name_lanes:
        # A high-confidence clipped letter from a perpendicular label must
        # not beat a longer observation of those same printed pixels.
        selected = []
        for candidate in names:
            b = name_boxes[candidate[2]]
            clipped = False
            if len(candidate[1]) <= 2:
                for other in names:
                    if (len(other[1]) < len(candidate[1])+1
                            or len(other[1])==len(candidate[1])+1 and other[3]<candidate[3]-.03
                            or not other[1].casefold().startswith(candidate[1].casefold())):
                        continue
                    a = name_boxes[other[2]]
                    overlap = max(0., min(a.x2,b.x2)-max(a.x1,b.x1))*max(0., min(a.y2,b.y2)-max(a.y1,b.y1))
                    if overlap/max(1.,min(a.width*a.height,b.width*b.height)) >= .7:
                        clipped = True
                        break
            if not clipped:
                selected.append(candidate)
        names = selected
        # A partial read of the same pixels can end closer to the lead than
        # the complete number's OCR box. That small alignment advantage must
        # not replace a confident, overlapping multi-character observation.
        complete_numbers = []
        for candidate in numbers:
            b = number_boxes[candidate[2]]
            clipped = False
            for other in numbers:
                if (len(other[1]) <= len(candidate[1]) or other[3] < max(.97,candidate[3]-.02)
                        or not (other[1].startswith(candidate[1]) or other[1].endswith(candidate[1]))):
                    continue
                a = number_boxes[other[2]]
                overlap = max(0.,min(a.x2,b.x2)-max(a.x1,b.x1))*max(0.,min(a.y2,b.y2)-max(a.y1,b.y1))
                if overlap/max(1.,min(a.width*a.height,b.width*b.height)) >= .7:
                    clipped = True
                    break
            if not clipped:
                complete_numbers.append(candidate)
        numbers = complete_numbers
    name = max(names, default=None)
    number = max(numbers, default=None)
    same_glyph_names = set()
    if refine_name_lanes and name is not None and name[3] < .995:
        selected_box = name_boxes[name[2]]
        for candidate in names:
            if candidate[1] == name[1] or len(candidate[1]) != len(name[1]) or candidate[3] < .85:
                continue
            other = name_boxes[candidate[2]]
            overlap = max(0., min(other.x2, selected_box.x2)-max(other.x1, selected_box.x1))*max(
                0., min(other.y2, selected_box.y2)-max(other.y1, selected_box.y1))
            if overlap/max(1., min(other.width*other.height, selected_box.width*selected_box.height)) >= .7:
                same_glyph_names.add(candidate[1])
    number_pitch = max(14., pitch) if numeric_names else pitch
    font = min(number_pitch*.65, 25.)
    alternatives = []
    if number is not None and numeric_names:
        # Conflicting overlapping observations of one glyph should reach
        # visual review even when greedy OCR assigns its wrong read .99.
        selected_box=next(t.bbox for t in ir.texts if t.internal_id==number[2])
        for candidate in numbers:
            if candidate[1]==number[1] or candidate[3]<.9:continue
            other=next(t.bbox for t in ir.texts if t.internal_id==candidate[2])
            overlap=max(0.,min(other.x2,selected_box.x2)-max(other.x1,selected_box.x1))*max(0.,min(other.y2,selected_box.y2)-max(other.y1,selected_box.y1))
            if overlap/max(1.,min(other.width*other.height,selected_box.width*selected_box.height))>=.6:
                alternatives.append(candidate[1])
    measured_number = None
    if number is None or number[3] < .995 or len(number[1]) > 1:
        local_box = localized_box if localized_numbers else _nearest_number_box(rotated, body, side, p, number_pitch)
        if local_box is not None:
            local_value, local_score = _read(engine, rotated, local_box, ctx, neutral_text=True)
            # Connector signal labels can look like terminal numbers even
            # before their interior text lane has been recognized. Keep
            # their established lane resolution; this retry targets IC text.
            if local_score < .95 and not numeric_names:
                value, score = _read_baseline_number(engine,rotated,body,side,p,number_pitch,ctx)
                if score >= .98:
                    local_value, local_score = value, score
            if (numeric_names and number is not None and _NUMBER.fullmatch(local_value)
                    and re.search(r'[A-Z]',local_value) and local_score>=.7 and local_value!=number[1]):
                alternatives.append(local_value)
            if _NUMBER.fullmatch(local_value) and local_value != '0' and local_score >= .95:
                measured_number = local_value
                if number is None:
                    number = (local_score, local_value, None, local_score)
                elif local_value != number[1]:
                    old_token = next((t for t in ir.texts if t.internal_id == number[2]), None)
                    remote = False
                    if old_token is not None:
                        old_box = _transform_box(old_token.bbox, width, height, rotation)
                        gap = body.x1-old_box.x2 if side == 'L' else old_box.x1-body.x2
                        remote = gap > max(6., font*.9) and local_score >= .97
                        # A tight crop may omit a thin leading/trailing digit.
                        # When that crop covers the same printed number, its
                        # shorter read is not evidence of a different label.
                        overlap = max(0., min(old_box.x2,local_box[2])-max(old_box.x1,local_box[0]))*max(
                            0., min(old_box.y2,local_box[3])-max(old_box.y1,local_box[1]))
                        local_area = max(1.,(local_box[2]-local_box[0])*(local_box[3]-local_box[1]))
                        partial_same_number = (len(local_value) < len(number[1]) and
                            (number[1].startswith(local_value) or number[1].endswith(local_value)) and
                            number[3] >= .97 and overlap/local_area >= .7)
                        remote = remote and not partial_same_number
                    if remote:
                        alternatives.append(number[1])
                        number = (local_score, local_value, None, local_score)
                    elif not (len(local_value) < len(number[1]) and number[1].endswith(local_value)):
                        alternatives.append(local_value)
    if (number is None or number[3] < .995) and not localized_numbers:
        nb = (body.x1-max(30., pitch*1.9), p-font-2, body.x1-2, p-1) if side == 'L' else (
            body.x2+2, p-font-2, body.x2+max(30., pitch*1.9), p-1)
        value, score = _read(engine, rotated, nb, ctx, neutral_text=True)
        if (_NUMBER.fullmatch(value) and score >= .55 and value != '0'
                and (number is None or score > number[3]+.01)):
            number = (score, value, None, score)
    if decorated_lead:
        # Arrow outlines below a number can be decoded as another digit.
        # Two upper-lane crops must independently agree on the printed text;
        # neither a package sequence nor neighboring numbers supplies it.
        extent = max(35., min(55., pitch*3))
        reads = []
        selected = None
        for top_shift in (0, 1):
            for factor in (.6, .75, .85, .95):
                inset = factor*pitch
                nb = ((body.x1-extent, p-min(20., pitch*.9)+top_shift, body.x1-inset, p-3)
                      if side == 'L' else
                      (body.x2+inset, p-min(20., pitch*.9)+top_shift, body.x2+extent, p-3))
                reads.append(_read(engine, rotated, nb, ctx, neutral_text=True))
            confident = [(value, score) for value, score in reads
                         if score >= .98 and _NUMBER.fullmatch(value) and value != '0']
            if len({value for value, _ in confident}) == 1 and len(confident) >= 2:
                value = confident[0][0]
                selected = value, sorted((score for _, score in confident), reverse=True)[1]
                break
        if selected:
            value, score = selected
            clipped = (number is not None and number[3] >= .95 and len(value) < len(number[1])
                       and (number[1].startswith(value) or number[1].endswith(value)))
            if clipped:
                alternatives.append(value)
            else:
                if number is not None and number[1] != value:
                    alternatives.append(number[1])
                number = (score, value, None, score)
    if numeric_names and pitch<12 and number is not None and number[1].isdecimal():
        # A taller independent crop can reveal a letter mistaken for a digit
        # in tightly spaced rows. It supplies a review candidate only.
        taller_box=_nearest_number_box(rotated,body,side,p,18.)
        if taller_box is not None:
            value,score=_read(engine,rotated,taller_box,ctx,neutral_text=True)
            if _NUMBER.fullmatch(value) and re.search(r'[A-Z]',value) and score>=.75 and value!=number[1]:
                alternatives.append(value)
    tight_numeric_name = None
    if refine_name_lanes and name is not None and (name[3] < .95 or name[1].islower() or len(name[1]) <= 2 or same_glyph_names):
        b = name_boxes[name[2]]
        value, score = _read(engine, rotated, (b.x1-1,b.y1-1,b.x2+2,b.y2+2), ctx, neutral_text=True)
        if numeric_names and value.isdecimal() and score >= .97:
            tight_numeric_name = value
        observed_consensus = value in same_glyph_names and score >= max(.97, name[3])
        if valid_name(value, score) and score >= .90 and (score > name[3]+.02 or value.casefold() == name[1].casefold() or observed_consensus):
            name = (score,value,None,score)
    if name is None or name[1].islower() or len(name[1])<=2:
        extent = min(body.width*.54, max(100., pitch*4))
        nb = (body.x1+4, p-pitch*.44, body.x1+extent, p+pitch*.56) if side == 'L' else (
            body.x2-extent, p-pitch*.44, body.x2-4, p+pitch*.56)
        if (numeric_names or single_name_bank) and body.width<38:
            # A one-sided header has one interior text lane. Splitting its
            # already narrow body in half truncates labels such as 10 and 20.
            nb=(body.x1+2,p-pitch*.44,body.x2-2,p+pitch*.56)
        value, score = _read(engine, rotated, nb, ctx)
        if valid_name(value, score) and score >= .65:
            numeric_consensus = (numeric_names and tight_numeric_name == value and score >= .97)
            if name is None or value.casefold() == name[1].casefold() or score > name[3]+.08 or numeric_consensus:
                name = (score, value, None, score)
    if numeric_names and name is not None and name[1].isdecimal() and name[3] >= .95:
        # An unlocalized exterior crop or a distant signal label must not
        # displace a clearly observed interior header number. Real exterior
        # terminal labels retain priority when spatially adjacent to the body.
        remote_number = False
        if number is not None:
            token = next((t for t in ir.texts if t.internal_id == number[2]), None)
            if token is not None:
                box = _transform_box(token.bbox, width, height, rotation)
                gap = body.x1-box.x2 if side == 'L' else box.x1-body.x2
                remote_number = gap > max(6., font*1.8)
            else:
                remote_number = number[1] != measured_number
        if remote_number:
            alternatives.append(number[1])
            number = None
    if number is None and numeric_names and name is not None and name[1].isdecimal():
        # Many headers print their only terminal number inside the outline.
        # This uses that observed numeric label, never a package pin sequence.
        number = name
    if number is not None:
        number = (*number[:4], sorted(set(alternatives)-{number[1]}))
    return name, number


def _resolve_duplicate_numbers(rows, rectangle, strokes, rotations, engine, ctx, *, width, height, connector):
    """Prefer a number observed in its own glyph crop over a broad lane read."""
    body = BBox(x1=rectangle[0],y1=rectangle[1],x2=rectangle[2],y2=rectangle[3])
    for number in sorted({r['number'] for r in rows if r['number']}):
        duplicates = [r for r in rows if r['number'] == number]
        if len(duplicates) < 2:
            continue
        for row in duplicates:
            side,p = row['side'],row['position']
            peers = [v for s,v,_ in strokes if s==side and v!=p]
            pitch = max(8. if connector else 12.,min(min((abs(v-p) for v in peers),default=25.),40.))
            pitch = max(14.,pitch) if connector else pitch
            rotation = row['rotation']
            local_body = _transform_box(body,width,height,rotation) if rotation else body
            if rotation:
                p = width-1-p if rotation==1 else p
                side = ({'T':'L','B':'R'} if rotation==1 else {'T':'R','B':'L'})[side]
            box = _nearest_number_box(rotations[rotation],local_body,side,p,pitch)
            value,score = _read(engine,rotations[rotation],box,ctx,neutral_text=True) if box else ('',0.)
            row['duplicate_local_number'] = value
            row['duplicate_local_score'] = score
        duplicates.sort(key=lambda r:(r['duplicate_local_number']==number and r['duplicate_local_score']>=.9,
            r['duplicate_local_score'] if r['duplicate_local_number']==number and r['duplicate_local_score']>=.9
            else r['number_score']),reverse=True)
        for row in duplicates[1:]:
            row['rejected_duplicate_number'] = row['number']
            row['number'] = None
    return rows


def _prefer_interior_header_numbers(rows, rectangle):
    """Use observed interior integers in a narrow, one-sided numbered header.

    Adjacent signal labels such as MCU port names also match the alphanumeric
    terminal grammar. They do not displace a coherent interior number lane.
    An explicitly observed exterior integer still keeps its existing priority.
    """
    sides = {r['side'] for r in rows}
    if len(rows) < 3 or len(sides) != 1:
        return
    side = next(iter(sides))
    width, height = rectangle[2]-rectangle[0], rectangle[3]-rectangle[1]
    thickness, length = (width,height) if side in 'LR' else (height,width)
    if thickness >= 38 or length < 2*thickness:
        return
    numbered = [r for r in rows if r['name'].isdecimal() and int(r['name']) > 0
                and r['name_score'] >= .95]
    if len(numbered) < max(3,.8*len(rows)) or len({r['name'] for r in numbered}) != len(numbered):
        return
    for row in numbered:
        old = row['number']
        if old is not None and old.isdecimal():
            continue
        if old:
            row['exterior_signal_label'] = old
            row['exterior_signal_alternatives'] = row['number_alternatives']
        # These alternate reads came from the exterior signal lane. They are
        # not alternate reads of the chosen interior terminal-number glyphs.
        row['number_alternatives'] = []
        row['number'] = row['name']
        row['number_token'] = row['name_token']
        row['number_score'] = row['name_score']
        row['number_location'] = 'observed_interior_header_lane'


def _select_visible_title(titles):
    """Separate the nearest title row from later part lists or annotations."""
    if not titles:
        return None
    nearest = min(gap for gap, _ in titles)
    nearest_height = min(t.bbox.height for gap,t in titles if gap == nearest)
    # Overlapping OCR boxes can differ by a few pixels. Keep the observations
    # of that same line together, then prefer their confidence/completeness.
    candidates = [token for gap, token in titles
                  if gap <= nearest + max(2., .5*min(token.bbox.height,nearest_height))]
    return max(candidates, key=lambda t:(t.score+.002*len(t.raw_text.strip()),t.internal_id))


def _reference(ir, body):
    choices = []
    for token in ir.texts:
        value = token.raw_text.strip()
        if token.score < .75:
            continue
        if not _REFERENCE.fullmatch(value):
            # A single OCR line can contain both the printed reference and
            # the adjacent part title, e.g. "USB1 KH-TYPE-C-16P". Keep the
            # observed prefix rather than treating a pin function as the key.
            words = value.split(maxsplit=1)
            if (token.score < .95 or len(words) != 2 or not _REFERENCE.fullmatch(words[0])
                    or not re.fullmatch(r'[A-Za-z0-9_+./()\-]{3,64}', words[1])
                    or not re.search(r'[A-Za-z]', words[1]) or not re.search(r'\d', words[1])
                    or _QUANTITY.fullmatch(words[1])):
                continue
            b=token.bbox
            prefix_box=BBox(x1=b.x1,y1=b.y1,x2=b.x1+b.width*len(words[0])/len(value),y2=b.y2)
            token=token.model_copy(update={'internal_id':token.internal_id+':reference-prefix',
                'raw_text':words[0],'resolved_text':words[0],'candidates':[],
                'bbox':prefix_box,'component_id':None,'pin_id':None,'details':'','category':'',
                'evidence':[*token.evidence,Evidence(evidence_id='reference-prefix:'+token.internal_id,
                    source='rule',score=token.score,metadata={'kind':'observed-reference-prefix',
                    'parent_token':token.internal_id,'full_text':value,'suffix':words[1],
                    'bbox_method':'proportional_prefix_hint','gt_access':False})]})
        t = token.bbox
        top = (body.x1-t.width <= t.center.x <= body.x2+t.width
               and body.y1-max(35., 2*t.height) <= t.center.y <= body.y1+3)
        right = (body.x2 <= t.center.x <= body.x2+max(35., 2*t.width)
                 and body.y1-5 <= t.center.y <= body.y1+max(35., body.height*.25))
        if top or right:
            distance = min(math.hypot(t.center.x-body.x1, t.y2-body.y1),
                           math.hypot(t.x1-body.x2, t.y1-body.y1))
            choices.append((distance, -token.score, token.internal_id, token))
    if not choices:
        return None
    # Overlapping OCR crops are alternate reads of the same glyphs, not
    # independent neighboring references. Retain the complete observation
    # over a smaller, similarly confident fragment (e.g. U4 read as J4).
    independent=[]
    for candidate in choices:
        t=candidate[3].bbox
        area=t.width*t.height
        clipped=False
        for other in choices:
            if other is candidate or other[3].score < candidate[3].score-.03:
                continue
            b=other[3].bbox
            overlap=max(0.,min(t.x2,b.x2)-max(t.x1,b.x1))*max(0.,min(t.y2,b.y2)-max(t.y1,b.y1))
            if b.width*b.height >= 1.25*area and overlap/max(1.,area) >= .8:
                clipped=True
                break
        if not clipped:independent.append(candidate)
    choices=independent
    choices.sort(key=lambda x: x[:3])
    if len(choices) > 1 and choices[1][0]-choices[0][0] < 5 and choices[0][3].raw_text != choices[1][3].raw_text:
        return None
    selected=choices[0][3]
    if not any(t.internal_id==selected.internal_id for t in ir.texts):
        ir.texts.append(selected)
    return selected


def _interior_reference(ir, body):
    """Accept a unique printed IC reference inside a separately found outline."""
    candidates = [t for t in ir.texts if t.score >= .9 and _REFERENCE.fullmatch(t.raw_text.strip())
                  and body.x1+12 < t.bbox.center.x < body.x2-12
                  and body.y1+12 < t.bbox.center.y < body.y2-12]
    return candidates[0] if len(candidates) == 1 else None


def _learned_stroke_tip(previous_pins, side, position, body, ink, used):
    nx,ny={'L':(-1,0),'R':(1,0),'T':(0,-1),'B':(0,1)}[side]
    fixed={'L':body.x1,'R':body.x2,'T':body.y1,'B':body.y2}[side]
    h,w=ink.shape;choices=[]
    for pin in previous_pins:
        if pin.internal_id in used or pin.orientation!=side or pin.score<.5:
            continue
        x,y=pin.point.x,pin.point.y
        lateral=abs(y-position) if nx else abs(x-position)
        outward=(x-fixed)*nx if nx else (y-fixed)*ny
        if lateral>3 or not 3<=outward<=48:
            continue
        a,b=(fixed,y) if nx else (x,fixed);samples=[]
        for f in np.linspace(0,1,max(3,math.ceil(outward))):
            px,py=round(a+(x-a)*f),round(b+(y-b)*f)
            if 1<=px<w-1 and 1<=py<h-1:
                samples.append(ink[py-1:py+2,px-1:px+2].any())
        if not samples or np.mean(samples)<.85:
            continue
        choices.append((lateral-2*pin.score,pin.internal_id,pin))
    return min(choices,key=lambda c:c[:2])[2] if choices else None


def _nearby_stroke_tip(previous_pins, side, position, point, used):
    """A measured point belongs to one observed row and can be used only once."""
    choices=[]
    for old in previous_pins:
        if old.internal_id in used or old.orientation != side:
            continue
        lateral=abs(old.point.y-position) if side in 'LR' else abs(old.point.x-position)
        distance=math.hypot(old.point.x-point.x,old.point.y-point.y)
        if lateral <= 3 and distance <= 8:
            choices.append((distance,old.internal_id,old))
    return min(choices,key=lambda row:row[:2])[2] if choices else None


def refine_achromatic_ic_tips(ir, path, rgb, provider, config, ctx=None):
    """Reobserve verified IC leads with the primary model; preserve identities.

    Unlike the original detection pass, this also covers newly recovered
    visible ICs. Only continuous achromatic symbol leads accept a measured
    model point; colored terminals retain their independent visual evidence.
    """
    from pcbparse.io import load_image, make_component_view
    from pcbparse.perception.composite_provider import _restore_pins
    measured_decorated = {p.internal_id for p in ir.pins if any(
        e.metadata.get('kind') == 'visible-stroke-and-text-lanes' and e.metadata.get('decorated_lead')
        for e in p.evidence)}
    wanted={p.component_id for p in ir.pins if p.internal_id.startswith('visual-pin:')
            and p.internal_id not in measured_decorated}
    components=[]
    for c in ir.components:
        if c.internal_id not in wanted:continue
        raw=c.model_copy(deep=True)
        bounds=[e.metadata.get('local_to_global_binding',{}).get('original_geometry',{}).get('values')
                for e in c.evidence if e.source=='detector']
        bounds=[v for v in bounds if isinstance(v,list) and len(v)==4]
        if bounds:raw.bbox=BBox(**dict(zip(['x1','y1','x2','y2'],bounds[-1])))
        components.append(raw)
    report={'updates':[],'gt_access':False,'new_pins':0,'identities_changed':False}
    if components:
        original=load_image(path)
        views=[make_component_view(original,c,padding=config.resolved_pin_component_roi_padding_px,
                                   view_id_prefix='visual-ic-refiner') for c in components]
        index={v.full_roi.view_id:v.full_roi for v in views}
        owners={v.full_roi.view_id:v.component_id for v in views}
        previous_version=ctx.model_versions.get('pin') if ctx else None
        try:
            measured=_restore_pins(provider.predict(views,ctx),index,owners)
        finally:
            if ctx:
                if previous_version is None:ctx.model_versions.pop('pin',None)
                else:ctx.model_versions['pin']=previous_version
        chroma=rgb.max(2).astype(np.int16)-rgb.min(2).astype(np.int16)
        gray=(rgb.min(2)<180)&(chroma<30)
        by_owner={c.internal_id:[] for c in components}
        for p in measured:by_owner[p.component_id].append(p)
        used=set();current=ir.component_index()
        for pin in ir.pins:
            if not pin.internal_id.startswith('visual-pin:') or pin.orientation not in 'LRTB':continue
            if pin.internal_id in measured_decorated:continue
            body=current[pin.component_id].body_bbox
            if body is None:continue
            axis=pin.point.y if pin.orientation in 'LR' else pin.point.x
            selected=_learned_stroke_tip(by_owner.get(pin.component_id,[]),pin.orientation,axis,body,gray,used)
            if selected is None:continue
            used.add(selected.internal_id)
            previous=pin.point.model_dump();pin.point=selected.point.model_copy(deep=True)
            row={'pin_id':pin.internal_id,'previous':previous,'point':pin.point.model_dump(),
                 'model_observation':selected.internal_id,'model_score':selected.score,'gt_access':False}
            pin.evidence.append(Evidence(evidence_id='achromatic-ic-tip:'+pin.internal_id,source='geometry',
                metadata={'kind':'achromatic-lead-model-terminal-fusion',**row}))
            report['updates'].append(row)
        if ctx:
            ctx.time_budget.check(stage='visual_ic_pin_refiner')
            ctx.model_versions['visual_ic_pin_refiner']=provider.version
    ir.metadata['achromatic_ic_terminal_fusion']=report
    return report


def measure_ic_outlines(rgb, engine, ctx=None, *, learned_tip_fusion=False, outline_dedup=False,
                        palette=None, ordinary_rectangles=None):
    """Measure image geometry once for exterior and interior reference passes."""
    if palette is None:
        palette = hue_map(rgb)
    rectangles = (list(ordinary_rectangles) if ordinary_rectangles is not None else
                  closed_rectangles(rgb,palette=palette,minimum_width=12,
                                    check=lambda:engine._check(ctx,'ic_outline_readback')))
    if learned_tip_fusion or outline_dedup:
        # Keep ordinary outlined symbols unchanged. Edge candidates supplement
        # filled bodies whose original interior columns create false outlines.
        rectangles += closed_rectangles(rgb,palette=palette,edge_only=True,minimum_width=12,
                                       check=lambda:engine._check(ctx,'ic_outline_readback'))
        rectangles.sort(key=lambda b:-(b[2]-b[0])*(b[3]-b[1]))
    return palette, tuple(rectangles)


def _narrow_ic_model_owner(ir, body, reference, strokes, ink):
    """A narrow IC needs independent detector and exterior terminal support.

    A nearby reference and incidental rectangle cannot create an IC. Reuse
    one strongly detected owner, and require at least two distinct measured
    strokes to coincide with its existing model tips and continuous leads.
    """
    if reference.score < .95 or len(strokes) < 2:
        return None
    candidates = []
    for component in ir.components:
        if (component.category.upper() not in _IC or component.score < .75
                or component.key and component.key != reference.raw_text.strip()
                or not any(e.source == 'detector' for e in component.evidence)):
            continue
        b = component.bbox
        area = max(0., min(body.x2,b.x2)-max(body.x1,b.x1))*max(0., min(body.y2,b.y2)-max(body.y1,b.y1))
        if area/max(1., body.width*body.height+b.width*b.height-area) < .5:
            continue
        pins = [p for p in ir.pins if p.component_id == component.internal_id]
        used = set()
        for side, position, _ in strokes:
            pin = _learned_stroke_tip(pins, side, position, body, ink, used)
            if pin is not None:
                used.add(pin.internal_id)
        if len(used) >= 2:
            candidates.append(component)
    return candidates[0] if len(candidates) == 1 else None


def _observed_numeric_terminal_layout(ir, body, reference, strokes, ink):
    """Identify printed numeric name banks without relying on a J/P prefix.

    This determines the reading layout, never a numbering sequence. Every
    resulting name/number must still come from its own observed text lane.
    """
    if len(strokes) < 3 or _narrow_ic_model_owner(ir,body,reference,strokes,ink) is None:
        return False
    numeric, functional = set(), set()
    for side, position, _ in strokes:
        if side not in {'L','R','T','B'}:
            continue
        peers = [abs(position-p) for s,p,_ in strokes if s == side and p != position]
        pitch = max(8., min(peers or [40.]))
        for token in ir.texts:
            value, b = token.raw_text.strip(), token.bbox
            extent = b.height if side in {'L','R'} else b.width
            axis = b.center.y if side in {'L','R'} else b.center.x
            if (token.score < .95 or extent > pitch*1.6
                    or not body.x1+2 < b.center.x < body.x2-2
                    or not body.y1+2 < b.center.y < body.y2-2
                    or abs(axis-position) > pitch*.4+1):
                continue
            if (side == 'L' and b.center.x > body.center.x
                    or side == 'R' and b.center.x < body.center.x
                    or side == 'T' and b.center.y > body.center.y
                    or side == 'B' and b.center.y < body.center.y):
                continue
            # Count occupied rows, not overlapping OCR crops of one glyph.
            if value.isdecimal() and 0 < int(value) <= 9999:
                numeric.add((side,position))
            elif _NAME.fullmatch(value):
                functional.add((side,position))
    return len(numeric) >= max(3, .6*len(strokes)) and len(functional) <= .1*len(strokes)


def recover_visual_ics(ir, path, engine, ctx=None, *, rgb=None, interior_only=False, learned_tip_fusion=False, outline_dedup=False, shaped_leads=False, refine_name_lanes=False, decorated_leads=False, outline_geometry=None):
    if rgb is None:
        rgb = readback_rgb(ir, path)
    rotations = {0: rgb}
    report = list(ir.metadata.get('visual_ic_readback',{}).get('components',[])) if interior_only else []
    if outline_geometry is None:
        outline_geometry = measure_ic_outlines(rgb, engine, ctx,
            learned_tip_fusion=learned_tip_fusion, outline_dedup=outline_dedup)
    palette, rectangles = outline_geometry
    occlusions = {}
    if not interior_only:
        from pcbparse.perception.occluded_ic_outlines import occluded_ic_outlines
        supplements, occlusions = occluded_ic_outlines(ir, palette, rectangles,
            check=lambda: engine._check(ctx, 'ic_outline_readback'))
        rectangles = [*rectangles, *supplements]
    eligible = []
    narrow_support = {}
    for rectangle in rectangles:
        if rectangle[2]-rectangle[0] >= 38:
            eligible.append(rectangle)
            continue
        body = BBox(x1=rectangle[0],y1=rectangle[1],x2=rectangle[2],y2=rectangle[3])
        reference = _reference(ir, body)
        if reference is None:
            continue
        if _CONNECTOR_REFERENCE.fullmatch(reference.raw_text.strip()):
            eligible.append(rectangle)
            continue
        strokes = terminal_strokes(rgb, rectangle, palette=palette, shaped_leads=shaped_leads)
        owner = _narrow_ic_model_owner(ir,body,reference,strokes,palette[1])
        if owner is not None:
            eligible.append(rectangle)
            narrow_support[rectangle] = (strokes, owner)
    # Unsupported small rectangles do not consume the existing outline cap.
    rectangles = eligible
    for rectangle in rectangles[:128]:
        engine._check(ctx, 'ic_outline_readback')
        x1, y1, x2, y2, hue = rectangle
        body = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
        reference = _reference(ir, body)
        shape_reference = False
        if interior_only:
            # Page/section frames can contain a reference but are not ICs.
            if body.width*body.height > .5*ir.image.width*ir.image.height:
                continue
            if reference is not None:
                continue
            reference = _interior_reference(ir,body)
        if reference is None and not interior_only:
            from pcbparse.perception.ic_reference_geometry import (
                exterior_reference_candidate, shape_supported_reference)
            if exterior_reference_candidate(ir, body) is not None:
                measured_strokes = terminal_strokes(rgb, rectangle, palette=palette,
                                                    shaped_leads=shaped_leads)
                reference = shape_supported_reference(ir, body, measured_strokes)
                shape_reference = reference is not None
        if reference is None:
            continue
        connector = bool(_CONNECTOR_REFERENCE.fullmatch(reference.raw_text.strip()))
        if (learned_tip_fusion or outline_dedup) and any(r['reference_token']==reference.internal_id
                and r['body']['x1']<=x1+3 and r['body']['y1']<=y1+3
                and r['body']['x2']>=x2-3 and r['body']['y2']>=y2-3 for r in report):
            continue
        # Point refinement must not also change which terminal strokes exist.
        if rectangle in narrow_support:
            strokes, narrow_owner = narrow_support[rectangle]
        else:
            strokes = terminal_strokes(rgb, rectangle, palette=palette,shaped_leads=shaped_leads,
                                       short_colored_leads=connector)
            narrow_owner = None
        single_name_bank = (narrow_owner is not None and len({s for s,_,_ in strokes}) == 1
                            and strokes[0][0] in {'L','R'})
        numeric_layout = not connector and _observed_numeric_terminal_layout(ir,body,reference,strokes,palette[1])
        numeric_names = connector or numeric_layout
        decorated = []
        if decorated_leads and not connector:
            from pcbparse.perception.decorated_ic_leads import decorated_terminal_strokes
            decorated = decorated_terminal_strokes(rectangle, palette,
                check=lambda: engine._check(ctx, 'decorated_ic_leads'))
            sides = {row[0] for row in decorated}
            strokes = [row for row in strokes if row[0] not in sides] + decorated
        if len(strokes) < (4 if interior_only else 2):
            continue
        rows = []
        for side, p, fixed in strokes:
            peers = [v for s, v, _ in strokes if s == side and v != p]
            pitch = min((abs(v-p) for v in peers), default=25.)
            pitch = max(8. if numeric_names or single_name_bank else 12., min(pitch, 40.))
            alternatives = []
            for rotation in ([0] if side in 'LR' else [1, 3]):
                if rotation not in rotations:
                    rotations[rotation] = np.rot90(rgb, rotation)
                name, number = _row_text(ir, rectangle, side, p, pitch,
                                         rotations[rotation], rotation, engine, ctx,refine_name_lanes=refine_name_lanes,
                                         numeric_names=numeric_names,decorated_lead=(side,p,fixed) in decorated,
                                         single_name_bank=single_name_bank,localized_numbers=rectangle in occlusions)
                rank = sum(item[3] if item else 0 for item in (name, number))
                alternatives.append((rank, rotation, name, number))
            _, rotation, name, number = max(alternatives, key=lambda x: (x[0], -x[1]))
            if name is None and number is None:
                continue
            rows.append({'side': side, 'position': p, 'tip': fixed, 'name': name[1] if name else '',
                         'number': number[1] if number else None,
                         'name_token': name[2] if name else None, 'number_token': number[2] if number else None,
                         'name_score': name[3] if name else 0., 'number_score': number[3] if number else 0.,
                         'number_alternatives': number[4] if number else [],
                         'rotation': rotation})
            if (side,p,fixed) in decorated:
                rows[-1]['decorated_lead'] = True
            if single_name_bank:
                rows[-1]['narrow_single_name_bank'] = True
            if numeric_layout:
                rows[-1]['observed_numeric_terminal_layout'] = True
            if numeric_names:
                rows[-1]['interior_numeric_terminal_label'] = True
            if rectangle in occlusions:
                rows[-1]['bounded_exterior_number_lane'] = True
        if len(rows) < 2 or sum(bool(r['name']) for r in rows) < max(2, .5*len(strokes)):
            continue
        if numeric_names:
            _prefer_interior_header_numbers(rows,rectangle)
        if body.height < 38:
            # Compact connector bodies have few glyphs; an OCR reading of a
            # circular pin-one mark must not supply a function name. Require
            # a complete, distinct, high-confidence printed numeric bank.
            if (not connector or len(rows) != len(strokes)
                    or any(not r['name'].isdecimal() or int(r['name']) <= 0
                           or r['number'] != r['name']
                           or min(r['name_score'], r['number_score']) < .95 for r in rows)
                    or len({r['number'] for r in rows}) != len(rows)):
                continue
        # An OCR confusion must never create an invalid duplicate number. A
        # lower-confidence conflict remains unresolved for the exact-model
        # consensus pass, which can use its independently observed name.
        _resolve_duplicate_numbers(rows,rectangle,strokes,rotations,engine,ctx,
                                   width=ir.image.width,height=ir.image.height,connector=numeric_names)
        for duplicate in rows:
            if 'rejected_duplicate_number' in duplicate:
                if duplicate['number_token']:
                    token=next(t for t in ir.texts if t.internal_id==duplicate['number_token'])
                    token.details=token.category='ic-conflicting-text'
                duplicate['number'], duplicate['number_token'] = None, None
        candidates = []
        for component in ir.components:
            b = component.bbox
            area = max(0., min(x2, b.x2)-max(x1, b.x1))*max(0., min(y2, b.y2)-max(y1, b.y1))
            iou = area/max(1., body.width*body.height+b.width*b.height-area)
            if component.category.upper() not in _IC:
                # A detector can call a rectangular LED array a two-terminal
                # LED. Promote only a substantially coincident body after at
                # least four independent labelled terminal rows were read.
                if not (shape_reference and len(rows) >= 4 and iou >= .6
                        and (not component.key or component.key == reference.raw_text.strip())):
                    continue
            if iou >= .4:
                candidates.append((iou, component.internal_id, component))
        if narrow_owner is not None:
            component = narrow_owner
        elif candidates:
            component = max(candidates, key=lambda c: c[:2])[2]
        else:
            component = Component(internal_id=f'visual-ic:{x1}:{y1}:{x2}:{y2}', category='Box', bbox=body, score=.99)
            ir.components.append(component)
        cid = component.internal_id
        if shape_reference:
            previous_category = component.category
            component.category = 'Box'
            component.evidence.append(Evidence(evidence_id='shape-supported-ic-reference:'+cid,
                source='geometry', metadata={'kind':'measured-box-and-labelled-multiple-terminals',
                    'reference':reference.raw_text.strip(), 'reference_token':reference.internal_id,
                    'previous_category':previous_category, 'labelled_terminal_rows':len(rows),
                    'gt_access':False}))
        previous_pins = [p for p in ir.pins if p.component_id == cid]
        component.body_bbox = body
        # The verified outline also restores a detector ROI truncated by a
        # tile boundary. Keep physical terminal tips in the association ROI.
        component.bbox = BBox(x1=min([x1]+[r['tip'] for r in rows if r['side']=='L']),
                              y1=min([y1]+[r['tip'] for r in rows if r['side']=='T']),
                              x2=max([x2]+[r['tip'] for r in rows if r['side']=='R']),
                              y2=max([y2]+[r['tip'] for r in rows if r['side']=='B']))
        component.key, component.key_score = reference.raw_text.strip(), float(reference.score)
        reference.component_id, reference.details, reference.category = cid, 'designator', 'designator'
        for evidence in reference.evidence:
            if evidence.metadata.get('kind')=='observed-reference-prefix':
                component.name=evidence.metadata['suffix']
                parent=next(t for t in ir.texts if t.internal_id==evidence.metadata['parent_token'])
                parent.component_id,parent.details,parent.category=cid,'searchname','searchname'
        titles = []
        for token in ir.texts:
            text, t = token.raw_text.strip(), token.bbox
            if (token.score < .85 or len(text) < 5 or len(text) > 64 or not re.search(r'[A-Za-z]', text)
                    or not re.search(r'\d{2,}', text) or _QUANTITY.fullmatch(text)
                    or not re.fullmatch(r'[A-Za-z0-9_+./() \-]+', text)):
                continue
            overlap = max(0., min(x2, t.x2)-max(x1, t.x1))
            bottom = (y2-2 <= t.center.y <= y2+max(24., t.height*1.5)
                      and overlap >= .5*min(body.width, t.width))
            right = (x2 <= t.x1 <= x2+10 and y1 <= t.center.y <= y1+max(45., body.height*.35)
                     and reference.bbox.center.x >= x2)
            interior_title = (interior_only and x1+.2*body.width <= t.center.x <= x2-.2*body.width
                              and y1+.15*body.height <= t.center.y <= y2-.15*body.height
                              and abs(t.center.x-reference.bbox.center.x) <= max(30.,.2*body.width)
                              and abs(t.center.y-reference.bbox.center.y) <= max(50.,reference.bbox.height*3))
            if bottom or right or interior_title:
                gap = max(0.,t.y1-y2) if bottom else max(0.,t.x1-x2) if right else abs(t.center.y-reference.bbox.center.y)
                titles.append((gap, token))
        if titles:
            title = _select_visible_title(titles)
            title_text=title.raw_text.strip()
            words=title_text.split(maxsplit=1)
            component.name = words[1] if len(words)==2 and words[0]==component.key else title_text
            title.component_id, title.details, title.category = cid, 'searchname', 'searchname'
        used_tokens = set()
        used_previous_pins = set()
        new_pins = []
        for i, row in enumerate(rows):
            side, p, tip = row['side'], row['position'], row['tip']
            point = Point(x=tip, y=p) if side in 'LR' else Point(x=p, y=tip)
            base = Point(x=x1 if side=='L' else x2, y=p) if side in 'LR' else Point(x=p, y=y1 if side=='T' else y2)
            # An independent learned tip close to this very stroke retains
            # its measured position. Identity comes from the separate lanes.
            learned = (_learned_stroke_tip(previous_pins,side,p,body,palette[1],used_previous_pins)
                       if learned_tip_fusion and not row.get('decorated_lead') else None)
            close = (_nearby_stroke_tip(previous_pins,side,p,point,used_previous_pins)
                     if learned is None and not row.get('decorated_lead') else None)
            if learned is not None:
                point=learned.point
                used_previous_pins.add(learned.internal_id)
            elif close is not None:
                point=close.point
                used_previous_pins.add(close.internal_id)
            pin = Pin(internal_id=f'visual-pin:{cid}:{side}:{p}', component_id=cid, point=point, base=base,
                      orientation=side, number=row['number'], name=row['name'], score=.99,
                      evidence=[Evidence(evidence_id=f'visual-pin-evidence:{cid}:{i}', source='geometry',
                                         metadata={'kind': 'visible-stroke-and-text-lanes', 'gt_access': False,
                                                   'image_sha256': ir.image.sha256, **row})])
            new_pins.append(pin)
            if learned is not None:
                pin.evidence.append(Evidence(evidence_id='ic-tip-fusion:'+pin.internal_id,source='geometry',
                    metadata={'kind':'visible-lead-model-terminal-fusion','model_observation':learned.internal_id,
                              'model_score':learned.score,'gt_access':False}))
            disconnected = (has_no_connect_marker(rgb, point, side) if pin.name == 'NC'
                            else _has_functional_no_connect_marker(rgb, point, side))
            if disconnected:
                pin.evidence.append(Evidence(evidence_id='visual-noconnect:'+pin.internal_id,source='geometry',
                    metadata={'kind':'visible-no-connect-marker','gt_access':False}))
            for field in ('name', 'number'):
                tid = row[field+'_token']
                if tid:
                    used_tokens.add(tid)
                    token = next(t for t in ir.texts if t.internal_id == tid)
                    token.component_id, token.pin_id = cid, pin.internal_id
                    token.details = token.category = 'pin'+field
        if learned_tip_fusion:
            # Keep the full observed stroke extent even when the measured tip
            # lies nearer the body. Otherwise symbol pixels just outside the
            # shrunken ROI can become wire-color votes for the whole image.
            component.bbox = BBox(x1=min([component.bbox.x1]+[p.point.x for p in new_pins if p.orientation=='L']),
                                  y1=min([component.bbox.y1]+[p.point.y for p in new_pins if p.orientation=='T']),
                                  x2=max([component.bbox.x2]+[p.point.x for p in new_pins if p.orientation=='R']),
                                  y2=max([component.bbox.y2]+[p.point.y for p in new_pins if p.orientation=='B']))
        ir.pins[:] = [p for p in ir.pins if p.component_id != cid] + new_pins
        # Duplicate clipped OCR of interior labels must not leak into chip
        # values or consume unrelated references during generic association.
        for token in ir.texts:
            b = token.bbox
            if (x1+2 < b.center.x < x2-2 and y1+2 < b.center.y < y2-2
                    and token.internal_id not in used_tokens):
                token.component_id = cid
                token.details = token.category = 'ic-body-text'
        row = {'component_id': cid, 'reference': component.key, 'reference_token': reference.internal_id,
               'body': body.model_dump(), 'pins': rows, 'previous_pin_count': len(previous_pins),
               'gt_access': False, 'descriptor_sha256': engine.descriptor_sha256}
        if rectangle in occlusions:
            row['outline_occlusion'] = occlusions[rectangle]
        if decorated:
            row['decorated_terminal_strokes'] = len(decorated)
        if interior_only:
            row['reference_location'] = 'inside_observed_outline'
        component.evidence.append(Evidence(evidence_id='visual-ic-evidence:'+cid, source='geometry', metadata=row))
        report.append(row)
    ir.metadata['visual_ic_readback'] = {'components': report, 'gt_access': False,
                                        'descriptor_sha256': engine.descriptor_sha256}
    return report


def finalize_visual_ic_fields(ir):
    """Reject remote value text and retain observed reference ownership."""
    checked = set(ir.metadata.get('decorated_number_association_cleanup', {}).get('checked_pins', []))
    rejected = list(ir.metadata.get('decorated_number_association_cleanup', {}).get('rejected', []))
    for pin in ir.pins:
        observations = [e.metadata for e in pin.evidence
                        if e.metadata.get('kind') == 'visible-stroke-and-text-lanes'
                        and (e.metadata.get('decorated_lead') or e.metadata.get('narrow_single_name_bank')
                             or e.metadata.get('bounded_exterior_number_lane'))]
        if not observations or pin.internal_id in checked:
            continue
        checked.add(pin.internal_id)
        # Unreadable exterior decorations must not fall back to an unrelated
        # signal label. Later exact-library or Qwen evidence may resolve them.
        if not any(row.get('number') for row in observations) and pin.number:
            rejected.append({'pin_id': pin.internal_id, 'previous_number': pin.number})
            pin.number = None
    if checked:
        ir.metadata['decorated_number_association_cleanup'] = {
            'checked_pins': sorted(checked), 'rejected': rejected, 'gt_access': False}
    for row in ir.metadata.get('visual_ic_readback', {}).get('components', []):
        component = ir.component_index().get(row['component_id'])
        if component is None:
            continue
        component.key = row['reference']
        if component.value:
            b = component.body_bbox
            evidence = [t for t in ir.texts if (t.resolved_text or t.raw_text) == component.value
                        and b.x1 <= t.bbox.center.x <= b.x2 and b.y1 <= t.bbox.center.y <= b.y2]
            if not evidence:
                component.evidence.append(Evidence(evidence_id='visual-ic:reject-remote-value:'+component.internal_id,
                    source='rule', metadata={'kind': 'value-outside-observed-ic', 'previous_value': component.value,
                                             'gt_access': False}))
                component.value = ''
