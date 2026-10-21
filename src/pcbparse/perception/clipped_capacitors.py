"""Recover capacitor bars truncated by adjacent printed ink."""
import math
import numpy as np
from pcbparse.graph.palette_wire import hue_map
from pcbparse.ir.models import BBox, Point
from pcbparse.perception.rail_visual_readback import _bars, _stem


def clipped_plate_pairs(rgb, region, text_boxes=()):
    h, w = rgb.shape[:2]
    x0, y0 = max(0, math.floor(region.x1)), max(0, math.floor(region.y1))
    x2, y2 = min(w, math.ceil(region.x2)), min(h, math.ceil(region.y2))
    bins, ink = hue_map(rgb[y0:y2, x0:x2])
    found = []
    masks = [ink & (bins < 0)] + [(bins == hue) & ink for hue in range(12) if (bins == hue).sum() >= 15]
    for mask in masks:
        for transpose in [False, True]:
            m = mask.T if transpose else mask
            bars = _bars(m)
            for i, a in enumerate(bars):
                for b in bars[i+1:]:
                    left, right = min(a[2], b[2]), max(a[3], b[3])
                    common_left, common_right = max(a[2], b[2]), min(a[3], b[3])
                    width, overlap, gap = right-left, common_right-common_left, b[0]-a[1]
                    if not (10 <= width <= 60 and overlap >= .5*width and 3 <= gap <= min(18, .75*width)):
                        continue
                    choices = []
                    middle = (left+right)/2
                    margin = max(2, .12*width)
                    for axis in range(max(common_left+2, math.ceil(middle-margin)), min(common_right-2, math.floor(middle+margin))+1):
                        first = _stem(m, [a[0], a[1], axis, axis], -1)
                        second = _stem(m, [b[0], b[1], axis, axis], 1)
                        if min(first[0], second[0]) < 4:
                            continue
                        # Two localized external leads, with no central stroke
                        # through the dielectric gap, are required together.
                        # Text can split a thick bar into separate run records.
                        # Normalize only the inward plate edges at the measured
                        # lead axis; a continuous symbol stroke still fails.
                        inside_a, inside_b = a[1], b[0]
                        while inside_a+1 < inside_b and m[inside_a+1,max(0,axis-1):axis+2].any():
                            inside_a += 1
                        while inside_b-1 > inside_a and m[inside_b-1,max(0,axis-1):axis+2].any():
                            inside_b -= 1
                        if inside_a-a[1] > 3 or b[0]-inside_b > 3 or inside_b-inside_a < 3:
                            continue
                        if m[inside_a+1:inside_b,max(0,axis-1):axis+2].any(1).mean() > .25:
                            continue
                        # A closed body with conducting strokes on both sides
                        # is a rectangle, even when its center is empty. One
                        # contaminated side is tolerated for adjacent print.
                        dielectric = m[inside_a+1:inside_b]
                        if (dielectric[:,left:left+3].any(1).mean() > .75
                                and dielectric[:,max(left,right-2):right+1].any(1).mean() > .75):
                            continue
                        widths = [m[a[0]-step, max(0,axis-3):axis+4].sum() for step in range(1,5)]
                        widths += [m[b[1]+step, max(0,axis-3):axis+4].sum() for step in range(1,5)]
                        if max(widths) > 4:
                            continue
                        choices.append((min(first[0],second[0]), -abs(axis-middle), axis, first, second))
                    if not choices:
                        continue
                    _, _, axis, first, second = max(choices)
                    top, bottom = (a[0]+a[1])/2, (b[0]+b[1])/2
                    lead = min(15,max(5,width*.25))
                    ends = (top-min(first[0],lead), bottom+min(second[0],lead))
                    if transpose:
                        body = BBox(x1=x0+top,y1=y0+left,x2=x0+bottom,y2=y0+right)
                        points = [Point(x=x0+v,y=y0+axis) for v in ends]
                    else:
                        body = BBox(x1=x0+left,y1=y0+top,x2=x0+right,y2=y0+bottom)
                        points = [Point(x=x0+axis,y=y0+v) for v in ends]
                    # Neighboring glyphs can share the same bar pattern. A
                    # shape centered inside recognized print is not an
                    # independently observed capacitor. Partial edge overlap
                    # is allowed because print can touch a genuine plate.
                    if any(t.x1<body.center.x<t.x2 and t.y1<body.center.y<t.y2 for t in text_boxes):
                        continue
                    if not any(max(abs(getattr(old[0],k)-getattr(body,k)) for k in ['x1','y1','x2','y2']) < 3 for old in found):
                        found.append((body,points,'LR' if transpose else 'TB'))
    return found
