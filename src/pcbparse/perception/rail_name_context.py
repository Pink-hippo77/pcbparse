"""Keep rail labels outside IC bodies and local to one observed symbol."""
from __future__ import annotations

import math
from pcbparse.ir.models import BBox

_RAILS={'GND','GROUND','V','POWER','POWERPORT'}
_BOXES={'BOX','IC','BOXIC','BOX_IC'}


def label_contains_symbol_stroke(body, token_box, rgb=None):
    """A rail cannot be named by the same glyph stroke proposed as its bar.

    Preserve a detector box's bottom underline, which can include a real
    power bar. Only overlap with the printed glyph interior is rejected.
    """
    b = token_box
    if b.width <= 2 or b.height <= 3:
        return False
    left, top = b.x1+1, b.y1+1
    right, bottom = b.x2-1, b.y2-min(3, b.height*.2)
    area = max(0., min(body.x2, right)-max(body.x1, left))*max(
        0., min(body.y2, bottom)-max(body.y1, top))
    if area/max(1., body.width*body.height) <= .5:
        return False
    if rgb is not None:
        # A large text detector can enclose a real red supply and its blue
        # name. Allow that independently colored name, while rejecting a
        # blue T glyph supported by other blue glyphs in the same label.
        import numpy as np
        from pcbparse.graph.palette_wire import hue_map
        def crop(box):
            return rgb[max(0, math.floor(box.y1)):min(rgb.shape[0], math.ceil(box.y2)+1),
                       max(0, math.floor(box.x1)):min(rgb.shape[1], math.ceil(box.x2)+1)]
        bar = crop(body)
        region = crop(BBox(x1=left,y1=top,x2=right,y2=bottom))
        if bar.size and region.size:
            bar_bins, bar_ink = hue_map(bar)
            if bar_ink.any():
                bar_hue = int(np.bincount(bar_bins[bar_ink]+1,minlength=13).argmax())-1
                bins, ink = hue_map(region)
                counts = np.bincount(bins[ink]+1,minlength=13)
                dominant = int(counts.argmax())-1
                yy, xx = np.nonzero(ink & (bins==dominant))
                if (dominant!=bar_hue and counts[dominant+1]>=8
                        and counts[dominant+1]>counts[bar_hue+1]
                        and len(xx) and np.ptp(xx)>=3 and np.ptp(yy)>=3):
                    return False
    return True


def interior_terminal_label(ir,token):
    """A printed terminal function cannot name a neighboring supply symbol."""
    for c in ir.components:
        if c.category.upper() not in _BOXES:
            continue
        b=c.body_bbox or c.bbox
        if (b.x1+2<token.bbox.center.x<b.x2-2
                and b.y1+2<token.bbox.center.y<b.y2-2):
            return True
    return False


def local_rail_name_box(ir,body,token,*,split=False):
    """Separate merged OCR rows at neighboring rail centers, then check owner.

    This only bounds a new image read; no missing characters are supplied.
    The original complete token is retained when it already names one rail.
    """
    b=token.bbox
    bounds=[b.x1,b.y1,b.x2,b.y2]
    horizontal=b.width>=b.height
    axis=0 if horizontal else 1
    center=body.center.x if horizontal else body.center.y
    cross=body.center.y if horizontal else body.center.x
    font=b.height if horizontal else b.width
    peers=[]
    for c in ir.components:
        if c.category.upper() not in _RAILS:
            continue
        other=c.body_bbox or c.bbox
        if math.hypot(other.center.x-body.center.x,other.center.y-body.center.y)<8:
            continue
        peers.append(other)
        if not split:
            continue
        location=other.center.x if horizontal else other.center.y
        transverse=other.center.y if horizontal else other.center.x
        if abs(transverse-cross)>max(8.,2*font):
            continue
        middle=(center+location)/2
        if location<center:
            bounds[axis]=max(bounds[axis],middle)
        elif location>center:
            bounds[axis+2]=min(bounds[axis+2],middle)
    if bounds[2]-bounds[0]<3 or bounds[3]-bounds[1]<3:
        return None
    region=BBox(x1=bounds[0],y1=bounds[1],x2=bounds[2],y2=bounds[3])
    p=region.center
    distance=math.hypot(p.x-body.center.x,p.y-body.center.y)
    if any(math.hypot(p.x-peer.center.x,p.y-peer.center.y)+4<distance for peer in peers):
        return None
    return region
