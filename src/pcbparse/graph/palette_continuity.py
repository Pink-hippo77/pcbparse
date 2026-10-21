"""Preserve visible junction dots and terminal strokes across color changes."""
from __future__ import annotations

import numpy as np


def supported_palette_strokes(image, probability, wire, body_blocked, pins, geometry, check=None):
    """Return image-supported exceptions to the wire-color filter.

    A compact dot needs wire anchors on multiple sides. A terminal lead needs
    an existing pin, an outward direction, continuous visible ink and a wire
    anchor. This never joins across blank space or through a component body.
    Pin positions and identities remain unchanged.
    """
    from pcbparse.graph.raster_trace import box, label_pixels, palette_wire_support

    h, w = wire.shape
    ink = image.min(2) < 204
    recovered = np.zeros_like(wire)
    report = {'junctions': [], 'terminal_leads': [], 'gt_access': False}
    # Exclude the antialias fringe as well: otherwise one faint pixel along
    # an entire wire can join a small dot to distant glyphs in this mask.
    foreign = ink & ~palette_wire_support(wire) & ~body_blocked
    _, components = label_pixels(foreign, check=check)
    for coordinates in components[1:]:
        if len(coordinates) < 5:
            continue
        y1, x1 = coordinates.min(0)
        y2, x2 = coordinates.max(0)
        width, height = int(x2 - x1 + 1), int(y2 - y1 + 1)
        if min(width, height) < 3 or max(width, height) > 24:
            continue
        if len(coordinates) / (width * height) < .45:
            continue
        support = probability[coordinates[:, 0], coordinates[:, 1]]
        if support.mean() < .35 or support.max() < .5:
            continue
        x1, x2, y1, y2 = map(int, (x1, x2, y1, y2))
        sides = [wire[y1:y2+1, max(0, x1-5):x1].any(),
                 wire[y1:y2+1, x2+1:min(w, x2+6)].any(),
                 wire[max(0, y1-5):y1, x1:x2+1].any(),
                 wire[y2+1:min(h, y2+6), x1:x2+1].any()]
        if sum(sides) < 2:
            continue
        recovered[coordinates[:, 0], coordinates[:, 1]] = True
        report['junctions'].append({'bbox': [x1, y1, x2, y2], 'wire_sides': int(sum(sides))})

    directions = {'left': (-1, 0), 'right': (1, 0), 'top': (0, -1), 'bottom': (0, 1)}
    for pin in pins:
        if check:
            check()
        if not pin.get('source_pin_id'):
            continue
        direction = directions.get(pin.get('side') or pin.get('source_side'))
        component = geometry['components'].get(pin.get('component_hint'))
        bounds = box(component.get('realBBox')) if component else None
        if bounds:
            x, y = pin['x'], pin['y']
            # A vertical diode may have a detector L/R orientation. A point
            # clearly beyond one body edge supplies an independent direction.
            if bounds[0] <= x <= bounds[2] and not bounds[1] <= y <= bounds[3]:
                direction = (0, -1 if y < bounds[1] else 1)
            elif bounds[1] <= y <= bounds[3] and not bounds[0] <= x <= bounds[2]:
                direction = (-1 if x < bounds[0] else 1, 0)
        if direction is None:
            continue
        nx, ny = direction
        x0, y0 = float(pin['x']), float(pin['y'])
        samples = []
        for step in range(65):
            x, y = round(x0 + nx*step), round(y0 + ny*step)
            if not (1 <= x < w-1 and 1 <= y < h-1):
                break
            xs = np.asarray([x-ny, x, x+ny])
            ys = np.asarray([y+nx, y, y-nx])
            visible = ink[ys, xs] & ~body_blocked[ys, xs]
            samples.append((xs, ys, visible, bool((wire[ys, xs] & visible).any())))
        if len(samples) < 5 or any(s[3] for s in samples[:2]):
            continue
        for end in range(2, len(samples)-2):
            if not all(s[3] for s in samples[end:end+3]):
                continue
            lead = samples[:end+1]
            # Every scan position must contain actual ink outside the body.
            if not all(s[2].any() for s in lead):
                break
            model_support = [float(probability[s[1], s[0]][s[2]].max()) for s in lead]
            if np.mean(model_support) < .15 or max(model_support) < .4:
                break
            for xs, ys, visible, _ in lead:
                recovered[ys[visible], xs[visible]] = True
            report['terminal_leads'].append({'pin_id': pin['source_pin_id'],
                'length_px': end, 'wire_anchor': [round(x0 + nx*end), round(y0 + ny*end)]})
            break
    return recovered, report
