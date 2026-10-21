"""Anchor repeated exterior signal names to independently visible CAD wires."""
from __future__ import annotations

from collections import defaultdict
import math
import re

import numpy as np

_NAME=re.compile(r'[A-Za-z_][A-Za-z0-9_.+\-/]{1,39}$')
_REFERENCE=re.compile(r'(?:R|C|L|D|Q|U|J|P|H|IC|CN|SW|FB|TP|LED|T|B)\d+[A-Z]?$')
_BARE={'TP','NC','IC','LED','POWER','TITLE','SIZE','DATE','REV','GND'}


def _conflicting_observation(token, nearby):
    """Reject clipped names and competing readings of the same printed glyphs.

    Check all OCR observations before grouping repeated names. A clipped COM
    inside COM1/COM2 must not authorize an electrical union, even if its OCR
    score exceeds the score of the complete label.
    """
    name=token.raw_text.strip();a=token.bbox
    for other in nearby:
        value=other.raw_text.strip();b=other.bbox
        if (other is token or value==name or other.score<.65 or not _NAME.fullmatch(value)
                or not .55*a.height<=b.height<=1.8*a.height
                or abs(a.center.y-b.center.y)>max(2.,.35*min(a.height,b.height))):continue
        overlap=max(0.,min(a.x2,b.x2)-max(a.x1,b.x1))*max(0.,min(a.y2,b.y2)-max(a.y1,b.y1))
        if overlap/max(1.,min(a.width*a.height,b.width*b.height))<.65:continue
        if len(value)>len(name) and b.width>a.width+2:
            if name.casefold() in value.casefold() or other.score>=.85 or len(value)>=len(name)+2:
                return True
        if len(value)==len(name) and other.score>=.94:
            return True
    return False


def signal_palette_supported(palette):
    """Local anchor checks can use a mixed sheet; full-sheet masking cannot."""
    return (palette.get('wire_hue_bin') is not None
            and palette.get('supporting_pins',0)>=12
            and palette.get('supporting_components',0)>=6
            and palette.get('agreement',0)>=.60)


def visible_signal_label_anchors(ir, bins, ink, wire_hue):
    """Use exact repeated OCR names; never turn an interior pin name into a net.

    Color drawings provide a useful independent check: a text stroke and the
    wire below it have different hues. Gray drawings abstain. Anchors must
    subsequently attach within two pixels of the traced wire.
    """
    h,w=bins.shape;wire=ink & (bins>=0) & (np.minimum((bins-wire_hue)%12,(wire_hue-bins)%12)<=1)
    bodies=[c.body_bbox or c.bbox for c in ir.components if c.category.upper() in {'BOX','IC','BOXIC','BOX_IC'}]
    occupied={c.name.strip() for c in ir.components if c.name}|{c.value.strip() for c in ir.components if c.value}
    groups=defaultdict(list)
    by_row=defaultdict(list)
    for t in ir.texts:
        by_row[math.floor(t.bbox.center.y/16)].append(t)
    for t in ir.texts:
        name=t.raw_text.strip();b=t.bbox
        if (t.score<.94 or not _NAME.fullmatch(name) or _REFERENCE.fullmatch(name)
                or name[-1] in '._/' or name.upper() in _BARE or name in occupied):continue
        if not (4<=b.height<=40 and 5<=b.width<=250):continue
        if any(c.x1+2<b.center.x<c.x2-2 and c.y1+2<b.center.y<c.y2-2 for c in bodies):continue
        nearby=[v for row in range(math.floor((b.center.y-16)/16),math.floor((b.center.y+16)/16)+1)
                for v in by_row.get(row,())]
        if _conflicting_observation(t,nearby):continue
        x1,y1=max(0,math.floor(b.x1)),max(0,math.floor(b.y1))
        x2,y2=min(w,math.ceil(b.x2)+1),min(h,math.ceil(b.y2)+1)
        stop=max(y1+1,round(y1+.65*(y2-y1)))
        sample=bins[y1:stop,x1:x2]
        glyphs=sample>=0
        # Dense label rows can include the preceding wire inside this OCR
        # box. Its long colored stroke must not determine the text's hue.
        # A label can start beside the gray endpoint of that preceding wire.
        # Require a continuous wire run covering most of the OCR width and
        # continuing outside an edge, independent of the text's glyphs.
        a,z=max(0,x1-6),min(w,x2+6)
        for y in range(y1,stop):
            delta=np.diff(np.pad(wire[y,a:z].astype(np.int8),1))
            for left,right in zip(np.flatnonzero(delta==1),np.flatnonzero(delta==-1)):
                left,right=a+int(left),a+int(right)
                overlap=max(0,min(right,x2)-max(left,x1))
                if overlap>=max(8,.75*(x2-x1)) and (left<=x1-4 or right>=x2+4):
                    lo,hi=max(y1,y-1),min(stop,y+2)
                    # Preserve differently colored letter pixels beside the
                    # wire, even when their row overlaps the wire's halo.
                    glyphs[lo-y1:hi-y1] &= ~wire[lo:hi,x1:x2]
                    break
        colors=sample[glyphs]
        if len(colors)<8:continue
        votes=np.bincount(colors,minlength=12);hue=int(votes.argmax())
        if votes[hue]/len(colors)<.60 or min((hue-wire_hue)%12,(wire_hue-hue)%12)<=1:continue
        a,z=max(0,x1-4),min(w,x2+8)
        top,bottom=max(0,round(b.center.y)),min(h,math.ceil(b.y2)+6)
        choices=[]
        for y in range(top,bottom):
            row=wire[y,a:z];delta=np.diff(np.pad(row.astype(np.int8),1))
            starts=np.flatnonzero(delta==1);ends=np.flatnonzero(delta==-1)
            for left,right in zip(starts,ends):
                if right-left<max(6,.35*b.width):continue
                overlap=max(0,min(a+right,b.x2)-max(a+left,b.x1))
                if overlap<max(4,.25*b.width):continue
                x=min(max(round(b.center.x),a+int(left)),a+int(right)-1)
                choices.append((abs(y-b.y2),-(right-left),y,x))
        if not choices:continue
        _,_,y,x=min(choices)
        groups[name].append({'name':name,'x':x,'y':y,'global':True,'scope':'sheet','radius':2,
                             'text_id':t.internal_id,'text_bbox':b.model_dump(),'score':float(t.score)})
    anchors=[]
    for name,rows in groups.items():
        unique=[]
        for row in sorted(rows,key=lambda v:-v['score']):
            if any(math.hypot(row['x']-v['x'],row['y']-v['y'])<8 for v in unique):continue
            unique.append(row)
        if len(unique)>=2:anchors.extend(unique)
    # Conflicting OCR on the same printed pixels cannot authorize a union.
    result=[]
    for row in anchors:
        a=row['text_bbox'];ambiguous=False
        for other in anchors:
            if other['name']==row['name']:continue
            b=other['text_bbox']
            area=max(0,min(a['x2'],b['x2'])-max(a['x1'],b['x1']))*max(0,min(a['y2'],b['y2'])-max(a['y1'],b['y1']))
            denom=min((a['x2']-a['x1'])*(a['y2']-a['y1']),(b['x2']-b['x1'])*(b['y2']-b['y1']))
            if area/max(1,denom)>.5:
                # Oversized OCR boxes on adjacent rows can overlap although
                # the printed baselines and visible wire anchors are separate.
                # Competing readings on one row still forbid the union.
                center_delta=abs((a['y1']+a['y2']-b['y1']-b['y2'])/2)
                separate_rows=(abs(row['y']-other['y'])>2
                    and center_delta>=max(3,.5*min(a['y2']-a['y1'],b['y2']-b['y1'])))
                if not separate_rows:ambiguous=True;break
        if not ambiguous:result.append(row)
    return result
