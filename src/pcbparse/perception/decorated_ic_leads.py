"""Recover connected achromatic pin decorations outside a colored IC outline."""
from __future__ import annotations

import numpy as np

from pcbparse.graph.raster_trace import label_pixels
from pcbparse.perception.symbol_body import groups


def decorated_terminal_strokes(rectangle, palette, *, check=None):
    """Require an exterior stem, a body contact and multiple visible shapes.

    A straight scan through an arrow or inversion circle can miss the lead's
    center. Trace connected pixels through that decoration instead. Only a
    side with several independently observed long stems is replaced; an
    incomplete or ambiguous trace leaves ordinary detection in control.
    """
    x1, y1, x2, y2, hue = rectangle
    if hue < 0:
        return []
    bins, ink = palette
    found = []
    for side in 'LRTB':
        if check:
            check()
        horizontal = side in 'LR'
        normal = -1 if side in 'LT' else 1
        fixed = {'L': x1, 'R': x2, 'T': y1, 'B': y2}[side]
        lo, hi = (y1, y2) if horizontal else (x1, x2)
        extent = ink.shape[1 if horizontal else 0]
        depth = min(90, fixed if normal < 0 else extent - 1 - fixed)
        if depth < 35 or hi - lo < 30:
            continue
        coordinates = fixed + normal * np.arange(1, depth + 1)
        if horizontal:
            strip = ink[lo+5:hi-4, coordinates]
            colors = bins[lo+5:hi-4, coordinates]
        else:
            strip = ink[coordinates, lo+5:hi-4].T
            colors = bins[coordinates, lo+5:hi-4].T
        gray = strip & (colors == -1)
        padded = np.pad(gray, ((1, 1), (0, 0)))
        support = padded[:-2] | padded[1:-1] | padded[2:]
        candidates = groups(np.flatnonzero(support[:, 19:32].mean(1) >= .95))
        rows = []
        for group in candidates:
            if len(group) > 5:
                continue
            position = round(float(np.mean(group)))
            lower, upper = max(0, position-6), min(len(gray), position+7)
            mask = gray[lower:upper, :32]
            labels, _ = label_pixels(mask)
            center = position-lower
            near = set(labels[max(0, center-1):center+2, :2].flat) - {0}
            far = set(labels[max(0, center-1):center+2, 27:32].flat) - {0}
            if not near & far:
                continue
            # Digits printed above neighboring straight leads can occupy both
            # sides of this row. They are not a decoration when the center
            # stroke already runs continuously from the body to the stem.
            shaped = (support[position, :15].mean() < .9 and any(
                mask[:max(0, center-2), distance].any() and mask[center+3:, distance].any()
                for distance in range(2, 13)))
            same = support[position]
            # The last achromatic stem pixel is an observable endpoint only
            # when followed by a clear gap or a differently colored wire.
            end = next((i for i in range(31, len(same)-2) if not same[i:i+3].any()), None)
            if end is None:
                continue
            rows.append((side, lo+5+position, fixed+normal*end, shaped))
        if len(rows) >= 4 and sum(row[3] for row in rows) >= 2:
            found.extend(row[:3] for row in rows)
    return found
