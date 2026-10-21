"""Retry a rectangular resistor obscured by a neighboring OCR box."""
from pcbparse.perception.passive_visual_readback import _contains


def retry_printed_rectangles(ir,rgb,region,kind,text_boxes,original):
    found=original(rgb,region,kind,text_boxes)
    if found or kind!='R' or not text_boxes:
        return found
    for body,points,sides in original(rgb,region,kind):
        # A retry addresses a box that clips a small part of the outline.
        # A rectangle inside the printed text is not a symbol proposal.
        overlaps=[_contains(body,t)*t.width*t.height/max(1.,body.width*body.height) for t in text_boxes]
        if not 0 < sum(overlaps) <= .2:
            continue
        # Existing differently typed bodies remain protected. This retry
        # recovers a missed symbol instead of relabeling another detection.
        if any(_contains(c.body_bbox or c.bbox,body)>.5 for c in ir.components):
            continue
        found.append((body,points,sides))
    return found
