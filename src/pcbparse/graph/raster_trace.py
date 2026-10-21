"""Pin attribution and conservative wire-to-net tracing in original pixels.

Inference consumes component/OCR geometry, detected pin points and wire
probabilities. It never consumes annotated pins, edges or edge.connect.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw


class DSU:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def box(value):
    if not isinstance(value, dict):
        return None
    try:
        b = tuple(float(value[k]) for k in ("x1", "y1", "x2", "y2"))
        if not all(math.isfinite(x) for x in b) or b[2] < b[0] or b[3] < b[1]:
            return None
        return b
    except (KeyError, TypeError, ValueError):
        return None


def point_box_distance(x, y, b):
    return math.hypot(max(b[0]-x, 0, x-b[2]), max(b[1]-y, 0, y-b[3]))


def sanitize_geometry(geometry):
    """Whitelist teammate outputs and explicitly exclude annotation truth."""
    component_fields = ("state", "category", "location", "realBBox", "designator", "value", "model", "net_name", "global_net", "associatedTexts", "searchNameByOCR", "spice", "terminals")
    text_fields = ("state", "content", "location", "category", "details", "component_id", "global_net", "score")
    result = {
        "components": {str(k): {f: v[f] for f in component_fields if f in v} for k, v in (geometry.get("components") or {}).items() if isinstance(v, dict)},
        "texts": {str(k): {f: v[f] for f in text_fields if f in v} for k, v in (geometry.get("texts") or {}).items() if isinstance(v, dict)},
    }
    # Optional explicit label anchors from OCR or a symbol detector. These are
    # teammate predictions, not edge-derived annotations.
    result["net_labels"] = [{k: item[k] for k in ("name", "x", "y", "global", "scope", "radius") if k in item} for item in (geometry.get("net_labels") or []) if isinstance(item, dict)]
    return result


def _thinning_tables():
    tables = np.zeros((2, 256), dtype=bool)
    for code in range(256):
        n = [bool(code & (1 << i)) for i in range(8)]
        count = sum(n)
        transitions = sum(not n[i] and n[(i+1) % 8] for i in range(8))
        constraints = (not (n[0] and n[2] and n[4]) and not (n[2] and n[4] and n[6]),
                       not (n[0] and n[2] and n[6]) and not (n[0] and n[4] and n[6]))
        for phase in (0, 1):
            tables[phase, code] = 2 <= count <= 6 and transitions == 1 and constraints[phase]
    tables.setflags(write=False)
    return tables


_THINNING_TABLES = _thinning_tables()


def thin(mask,check=None):
    """Zhang-Suen thinning; preserve one-pixel skeleton and diagonal paths."""
    x = np.pad(mask.astype(bool), 1)
    for _ in range(100):
        if check:check()
        changed = False
        for phase in (0, 1):
            p = x[1:-1, 1:-1]
            n = [x[:-2, 1:-1], x[:-2, 2:], x[1:-1, 2:], x[2:, 2:], x[2:, 1:-1], x[2:, :-2], x[1:-1, :-2], x[:-2, :-2]]
            # Eight neighboring bits select the same synchronous deletion
            # rule without allocating count/transition arrays per phase.
            code = n[0].astype(np.uint8)
            for i in range(1, 8):
                np.bitwise_or(code, n[i]*np.uint8(1 << i), out=code)
            remove = p & _THINNING_TABLES[phase, code]
            if remove.any():
                p[remove] = False
                changed = True
        if not changed:
            break
    return x[1:-1, 1:-1]


def label_pixels(mask,check=None):
    """Eight-connected regions with stable row-major IDs and coordinates.

    SAUF uses the same first-pixel ordering as the run-based fallback. Keeping
    IDs and coordinate order preserves crossing unions and deterministic nets.
    """
    if check:check()
    if not mask.size:return np.zeros(mask.shape,np.int32),[None]
    try:
        import cv2
    except ImportError:
        return _label_pixels_runs(mask,check=check)
    count,labels=cv2.connectedComponentsWithAlgorithm(
        np.ascontiguousarray(mask,dtype=bool).view(np.uint8),8,cv2.CV_32S,cv2.CCL_WU)
    if check:check()
    ys,xs=np.nonzero(labels)
    if not len(ys):return labels,[None]
    values=labels[ys,xs];order=np.argsort(values,kind='stable')
    coords=np.column_stack((ys[order],xs[order])).astype(np.int32)
    cuts=np.cumsum(np.bincount(values,minlength=count)[1:])[:-1]
    if check:check()
    return labels,[None,*np.split(coords,cuts)]


def _label_pixels_runs(mask,check=None):
    """Label horizontal runs; preserve the original row-major component IDs.

    A forbidden corner diagonal already has an orthogonal path through the
    occupied corner, so the component partition equals eight-connectivity.
    Run unions avoid visiting all eight neighbors of each wire pixel in Python.
    """
    h,w=mask.shape
    labels=np.zeros(mask.shape,np.int32)
    runs=[];parents=[];previous=[]
    def find(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    for y in range(h):
        if check and y%128==0:check()
        delta=np.diff(np.pad(mask[y].astype(bool).astype(np.int8),1))
        starts=np.flatnonzero(delta==1);ends=np.flatnonzero(delta==-1)
        current=[];first=0
        for a,b in zip(starts,ends):
            a,b=int(a),int(b);i=len(runs)
            runs.append((y,a,b));parents.append(i);current.append(i)
            while first<len(previous) and runs[previous[first]][2]<a:first+=1
            j=first
            while j<len(previous) and runs[previous[j]][1]<=b:
                r,s=find(i),find(previous[j])
                if r!=s:parents[max(r,s)]=min(r,s)
                j+=1
        previous=current
    ids={}
    for i,(y,a,b) in enumerate(runs):
        if check and i%4096==0:check()
        root=find(i)
        number=ids.setdefault(root,len(ids)+1)
        labels[y,a:b]=number
    ys,xs=np.nonzero(labels)
    if not len(ys):return labels,[None]
    values=labels[ys,xs];order=np.argsort(values,kind='stable')
    coords=np.column_stack((ys[order],xs[order])).astype(np.int32)
    cuts=np.cumsum(np.bincount(values,minlength=len(ids)+1)[1:])[:-1]
    return labels,[None,*np.split(coords,cuts)]


def repair_axial_gaps(mask, probability, blocked, max_gap=6, gray=None,check=None):
    out = mask.copy()
    # Pixel support does not depend on the evolving wire mask or gap length.
    # Compute it once while retaining the sequential endpoint/fill ordering.
    darkness = (gray < 180) if gray is not None else np.zeros(mask.shape, bool)
    supported = ((probability >= 0.10) | ((probability >= 0.02) & darkness)) & ~blocked
    for axis in (0, 1):
        current = out if axis == 0 else out.T
        allowed = supported if axis == 0 else supported.T
        for gap in range(1, max_gap+1):
            if check:check()
            length = gap+2
            left = current[:, :-length+1]
            right = current[:, length-1:]
            ok = left & right
            # Require model support throughout the gap; never join across an
            # empty region or component interior just because endpoints align.
            for offset in range(1, length-1):
                ok &= allowed[:, offset:offset+left.shape[1]]
            for offset in range(1, length-1):
                current[:, offset:offset+left.shape[1]] |= ok
    return out & ~blocked


def recover_visible_strokes(image, probability, blocked, geometry, min_run=18,check=None):
    """Recover straight raster strokes supported by model seeds, outside symbols/text.

    Whole visible runs can bridge extended model dropouts, but blank pixels,
    physical component interiors and OCR glyph boxes remain excluded.
    """
    gray = np.asarray(Image.fromarray(image).convert("L"))
    excluded = blocked.copy(); h, w = gray.shape
    for token in geometry["texts"].values():
        bounds = box(token.get("location"))
        if token.get("state") != "deleted" and bounds:
            x1, y1, x2, y2 = map(lambda v: int(round(v)), bounds)
            if x2 >= 0 and y2 >= 0 and x1 < w and y1 < h:
                excluded[max(0, y1):min(h, y2+1), max(0, x1):min(w, x2+1)] = True
    visible = (gray < 180) & ~excluded
    recovered = np.zeros((h, w), bool)
    for axis in (0, 1):
        scan = visible if axis == 0 else visible.T
        prob = probability if axis == 0 else probability.T
        target = recovered if axis == 0 else recovered.T
        for row_index in range(scan.shape[0]):
            if check and row_index%128==0:check()
            row = scan[row_index]
            changes = np.diff(np.pad(row.astype(np.int8), 1))
            starts = np.flatnonzero(changes == 1); ends = np.flatnonzero(changes == -1)
            for start, end in zip(starts, ends):
                if end-start < min_run:
                    continue
                support = prob[row_index, start:end]
                if (support >= .3).mean() >= .1 and support.max() >= .4:
                    target[row_index, start:end] = True
    return recovered


def _component_metadata(cid, c, texts):
    assoc = c.get("associatedTexts") or {}
    def find_detail(detail):
        for item in assoc.values():
            if isinstance(item, dict) and item.get("details") == detail:
                return str(item.get("content") or "").strip() or None
        return None
    search = c.get("searchNameByOCR")
    model = c.get("model") or ((texts.get(search) or {}).get("content") if search else None)
    return {"component_id": cid, "category": c.get("category"), "designator": c.get("designator") or find_detail("designator"), "value": c.get("value") or find_detail("value"), "model": model}


def power_name(component):
    name = str(component.get("net_name") or "").strip()
    if name:
        return name
    category = str(component.get("category") or "").lower()
    tokens = [str(t.get("content") or "").strip() for t in (component.get("associatedTexts") or {}).values() if isinstance(t, dict)]
    if category in ("gnd", "ground"):
        for text in tokens:
            if re.fullmatch(r"[A-Za-z_]*GND[A-Za-z_0-9]*", text, re.IGNORECASE):
                return text.upper()
        # A ground class alone cannot distinguish GND, AGND and DGND. Merge
        # only an explicitly observed name; anonymous symbols remain local.
        return None
    if category in ("v", "power", "powerport", "net_input", "net_output", "net_label"):
        # V is a voltage source in some schemas; only single-terminal symbol
        # use is resolved by the caller. Do not infer a supply from 16V ratings.
        for text in tokens:
            if re.fullmatch(r"[+-]\d+(?:\.\d+)?[Vv]\d*|V[A-Za-z0-9_]+|[A-Za-z_]+_\d+[Vv]\d*|\d+(?:\.\d+)?[Vv]_[A-Za-z0-9_]+", text):
                return text.upper()
    return None


def assign_pins(points, geometry, radius=8, trusted_source_owners=False):
    components = [(cid, c, box(c.get("location")), box(c.get("realBBox"))) for cid, c in geometry["components"].items() if c.get("state") != "deleted" and box(c.get("location")) and box(c.get("realBBox"))]
    pins, issues = [], []
    normals = {"left": (-1, 0), "right": (1, 0), "top": (0, -1), "bottom": (0, 1)}
    for i, point in enumerate(sorted(points, key=lambda p: (p["y"], p["x"])), 1):
        x, y = float(point["x"]), float(point["y"])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("non-finite pin coordinate")
        candidates = []
        for cid, c, outer, body in components:
            if point_box_distance(x, y, outer) > radius:
                continue
            sides = {
                "left": math.hypot(x-outer[0], max(outer[1]-y, 0, y-outer[3])),
                "right": math.hypot(x-outer[2], max(outer[1]-y, 0, y-outer[3])),
                "top": math.hypot(y-outer[1], max(outer[0]-x, 0, x-outer[2])),
                "bottom": math.hypot(y-outer[3], max(outer[0]-x, 0, x-outer[2])),
            }
            side = min(sides, key=sides.get)
            # A wire terminal lies outside the physical symbol. Allow two
            # pixels of raster/box error but reject points deep in its body.
            inside = body[0]+2 < x < body[2]-2 and body[1]+2 < y < body[3]-2
            if sides[side] <= radius and not inside:
                candidates.append((sides[side], cid, side))
        candidates.sort()
        item = {**point, "pin_id": f"PIN_{i:04d}", "component_id": None, "side": None, "pin_number": None, "net_id": None, "status": "unassigned", "owner_candidates": [q[1] for q in candidates[:3]], "tip": [x,y], "base": None}
        # Source pins already have a verified owner. Preserve that owner when
        # its geometry remains plausible instead of assigning a nearby part.
        hint=point.get('component_hint')
        if trusted_source_owners and point.get('source_pin_id') and hint in geometry['components']:
            c = geometry['components'][hint]
            outer = box(c.get('location'))
            supported = [q for q in candidates if q[1] == hint]
            side = point.get('source_side')
            if outer is not None and point_box_distance(x,y,outer) <= max(radius,16):
                if side in normals:
                    candidates = [(0.,hint,side)]
                else:
                    candidates = supported
                item['owner_source'] = 'retained_source_pin'
            else:
                candidates = []
        supported=[q for q in candidates if q[1]==hint] if point.get('source_pin_id') and point.get('owner_verified') is True else []
        if supported:
            candidates=supported
            item['owner_source']='supported_source_pin'
        if candidates and (len(candidates) == 1 or candidates[1][0]-candidates[0][0] > 1.0):
            _, cid, side = candidates[0]
            item.update(component_id=cid, side=side, outward_normal=list(normals[side]), status="assigned")
            b=box(geometry["components"][cid]["realBBox"])
            item["base"] = [b[0] if side=="left" else b[2], min(max(y,b[1]),b[3])] if side in ("left","right") else [min(max(x,b[0]),b[2]), b[1] if side=="top" else b[3]]
        else:
            code = "ambiguous_pin_owner" if candidates else "unassigned_pin_owner"
            issues.append({"code": code, "pin_id": item["pin_id"], "candidates": item["owner_candidates"]})
        pins.append(item)
    # Resolve explicit OCR pin-number tokens without consulting component.pins.
    used_texts = set()
    for item in sorted(pins, key=lambda p: -float(p.get("score", 0))):
        cid, side = item["component_id"], item["side"]
        if cid is None:
            continue
        c = geometry["components"][cid]; outer = box(c["location"])
        candidates = []
        for tid, token in geometry["texts"].items():
            if tid in used_texts or token.get("state") == "deleted" or token.get("details") != "pinNumber":
                continue
            if token.get("component_id") not in (None, cid):
                continue
            value = str(token.get("content") or "").strip()
            bounds = box(token.get("location"))
            if not bounds or not re.fullmatch(r"\d+[A-Za-z]?", value):
                continue
            tx, ty = (bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2
            if point_box_distance(tx, ty, outer) > 20:
                continue
            aligned = abs(ty-item["y"]) if side in ("left", "right") else abs(tx-item["x"])
            distance = math.hypot(tx-item["x"], ty-item["y"])
            if aligned <= 5 and distance <= 24:
                candidates.append((distance+2*aligned, tid, value))
        candidates.sort()
        if candidates and (len(candidates) == 1 or candidates[1][0]-candidates[0][0] > 2):
            _, tid, value = candidates[0]; item["pin_number"] = value; item["pin_number_source"] = "ocr"; used_texts.add(tid)
    # Optional externally predicted terminal numbering is an explicit teammate
    # interface, separate from annotation components[*].pins (never read).
    for cid,c in geometry["components"].items():
        candidates=[]
        for index,terminal in enumerate(c.get("terminals") or []):
            if not isinstance(terminal,dict) or terminal.get("pin_number") is None:
                continue
            try: tx,ty=float(terminal["x"]),float(terminal["y"])
            except (KeyError,TypeError,ValueError): continue
            if not math.isfinite(tx) or not math.isfinite(ty): continue
            for item in pins:
                if item["component_id"]==cid:
                    distance=math.hypot(item["x"]-tx,item["y"]-ty)
                    if distance<=5: candidates.append((distance,index,item["pin_id"],terminal))
        used_ids=set();used_indexes=set();by_id={p["pin_id"]:p for p in pins}
        for distance,index,pid,terminal in sorted(candidates,key=lambda q:(q[0],q[1],q[2])):
            if pid in used_ids or index in used_indexes:continue
            used_ids.add(pid);used_indexes.add(index)
            by_id[pid]["pin_number"]=str(terminal["pin_number"])
            by_id[pid]["pin_number_source"]="teammate_terminal_metadata"
            if terminal.get("pin_name") is not None:by_id[pid]["pin_name"]=str(terminal["pin_name"])
    # Stable local terminal identifiers are distinct from physical pin numbers.
    counts = defaultdict(int)
    for item in pins:
        cid = item["component_id"]
        counts[cid] += 1
        item["terminal_id"] = f"{cid or 'UNASSIGNED'}:T{counts[cid]}"
    return pins, issues


def _crossing_candidates(skeleton, radius, check=None):
    """Prefilter the same four-arm windows in stable row-major order."""
    h, w = skeleton.shape
    margin = max(radius+2, 6)
    if h <= 2*margin or w <= 2*margin:
        return []
    candidates = skeleton[margin:h-margin, margin:w-margin].copy()
    near = range(-1, 2)
    arms = ((near, range(-radius-2, -radius)), (near, range(radius+1, radius+3)),
            (range(-radius-2, -radius), near), (range(radius+1, radius+3), near))
    for rows, columns in arms:
        if check:
            check()
        supported = np.zeros_like(candidates)
        for dy in rows:
            for dx in columns:
                supported |= skeleton[margin+dy:h-margin+dy, margin+dx:w-margin+dx]
        candidates &= supported
    yy, xx = np.nonzero(candidates)
    return zip(yy+margin, xx+margin)


def split_crossings(skeleton, image, radius=3, predictor=None, confidence=.9,check=None,*,compact=False):
    """Split four-arm orthogonal crossings unless the image supports a dot.

    T-junctions remain connected. Dot confidence is image-based; uncertain
    crossings are split conservatively and explicitly reported for review.
    """
    if not .5<confidence<=1:
        raise ValueError('crossing confidence must be in (0.5,1]')
    gray = np.asarray(Image.fromarray(image).convert("L"))
    output = skeleton.copy(); crossings = []; visited = []
    if compact:
        # A seven-pixel cut erases neighboring routes on small schematics.
        # Center the cut on the strongest local four-arm evidence before
        # suppressing duplicate pixels of the same intersection.
        radius = 1
        candidates = list(_crossing_candidates(skeleton, radius, check))
        def strength(p):
            y, x = p
            return sum(int(skeleton[y+dy*k, x+dx*k])
                       for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)) for k in (1,2))
        selected = []; excluded = set()
        for index,(y,x) in enumerate(sorted(candidates, key=lambda p:(-strength(p),p[0],p[1]))):
            if check and index%256==0:check()
            if (y,x) in excluded:continue
            selected.append((y,x))
            excluded.update((y+dy,x+dx) for dy in range(-2,3) for dx in range(-2,3))
        candidates = sorted(selected)
    else:
        candidates = _crossing_candidates(skeleton, radius, check)
    for index,(y, x) in enumerate(candidates):
        if check and index%256==0:check()
        y, x = int(y), int(x)
        if not compact and any(abs(y-a) <= radius*2 and abs(x-b) <= radius*2 for a, b in visited):
            continue
        # Estimate line thickness from remote arms; a dot occupies pixels off
        # both strips and has significantly greater central area than a cross.
        remote = gray[y-6:y+7, x-6:x+7] < 150
        strip = remote[:, 5:8] if compact else remote
        vertical = int(np.median(np.concatenate([strip[:2].sum(axis=1), strip[-2:].sum(axis=1)])))
        strip = remote[5:8, :] if compact else remote
        horizontal = int(np.median(np.concatenate([strip[:, :2].sum(axis=0), strip[:, -2:].sum(axis=0)])))
        line_width = max(1, min(vertical, horizontal, 4))
        central = remote[4:9, 4:9]
        baseline = 10*line_width-line_width**2
        excess = int(central.sum())-baseline
        connected = excess >= max(3, line_width*2)
        if compact and connected:
            # A nearby dot or parallel stroke may add ink on one side of
            # this crossing. A local junction must thicken all four corners.
            corner = max(1,(5-line_width)//2)
            connected = all(region.any() for region in
                (central[:corner,:corner],central[:corner,-corner:],
                 central[-corner:,:corner],central[-corner:,-corner:]))
        visited.append((y, x))
        record = {"x": x, "y": y, "connected": connected, "dot_excess_pixels": excess, "line_width": line_width, "decision": "image_dot" if connected else "straight_through_no_dot", "review_required": not connected and excess > 0}
        if compact:record['cut_radius'] = radius
        if not connected:
            output[y-radius:y+radius+1, x-radius:x+radius+1] = False
        crossings.append(record)
    if predictor is not None and crossings:
        if check:check()
        scores=predictor(image,crossings)
        if check:check()
        if len(scores)!=len(crossings) or not all(math.isfinite(float(p)) and 0<=p<=1 for p in scores):
            raise ValueError('invalid crossing predictor output')
        for crossing,score in zip(crossings,scores):
            crossing['connection_probability']=float(score)
            if score>=confidence or score<=1-confidence:
                x,y=crossing['x'],crossing['y'];connected=score>=confidence
                crossing.update(connected=connected,decision='learned_image_connection' if connected else 'learned_image_crossing',review_required=True)
                output[y-radius:y+radius+1,x-radius:x+radius+1]=skeleton[y-radius:y+radius+1,x-radius:x+radius+1] if connected else False
    return output, crossings


def nearest_labels(labels, x, y, radius, normal=None, alignment=None):
    h, w = labels.shape; x0, y0 = int(round(x)), int(round(y))
    left, right = max(0, x0-radius), min(w, x0+radius+1)
    top, bottom = max(0, y0-radius), min(h, y0+radius+1)
    yy, xx = np.nonzero(labels[top:bottom, left:right])
    xx = xx+left; yy = yy+top
    candidates = {}
    for px, py in zip(xx, yy):
        dx, dy = float(px)-x, float(py)-y
        distance = math.hypot(dx, dy)
        if distance > radius:
            continue
        if normal and dx*normal[0]+dy*normal[1] < -2:
            continue
        if normal and alignment is not None and abs(dx*normal[1]-dy*normal[0]) > alignment:
            continue
        number = int(labels[py, px])
        previous = candidates.get(number)
        if previous is None or distance < previous[0]:
            candidates[number] = (distance, int(px), int(py))
    return sorted((distance, number, px, py) for number, (distance, px, py) in candidates.items())


def symbol_port_candidates(image, probability, points, geometry):
    """Recover one-terminal ground/power ports using explicit symbol boxes.

    Require both an image stroke at the boundary and model-supported outward
    continuation. This does not copy terminal coordinates from JSON pins.
    """
    gray = np.asarray(Image.fromarray(image).convert("L")); h, w = gray.shape
    augmented = [dict(p) for p in points]
    nc_boxes = [box(t.get("location")) for t in geometry["texts"].values()
                if t.get("state") != "deleted" and
                str(t.get("content") or "").lower().replace("×", "x").replace(".", "").replace(" ", "")
                in ("x", "nc", "noconnect", "notconnected")]
    for cid, c in geometry["components"].items():
        if c.get("state") == "deleted" or str(c.get("category") or "").lower() not in ("gnd", "ground", "power", "powerport"):
            continue
        outer = box(c.get("location")); body = box(c.get("realBBox"))
        if not outer or not body:
            continue
        x1,y1,x2,y2 = [int(round(v)) for v in outer]
        candidates=[]
        for side, a, b, fixed, normal in (("top",x1,x2,y1,(0,-1)),("bottom",x1,x2,y2,(0,1)),("left",y1,y2,x1,(-1,0)),("right",y1,y2,x2,(1,0))):
            # Port attachment belongs to the middle of a symbol edge. Ignore
            # corners and ground-bar tips at the extreme left/right ends.
            trim=max(2,int((b-a)*0.2))
            for coord in range(a+trim,b-trim+1):
                x,y=(coord,fixed) if side in ("top","bottom") else (fixed,coord)
                if any(b and point_box_distance(x, y, b) <= 6 for b in nc_boxes):
                    continue
                if not 2<=x<w-2 or not 2<=y<h-2:
                    continue
                local=gray[y-2:y+3,x-2:x+3]
                if local.min()>=180:
                    continue
                dark=[]; support=[]
                for step in range(3,17):
                    px,py=x+normal[0]*step,y+normal[1]*step
                    if 1<=px<w-1 and 1<=py<h-1:
                        dark.append(float(gray[py-1:py+2,px-1:px+2].min()<180))
                        support.append(float(probability[py-1:py+2,px-1:px+2].max()))
                if not dark or np.mean(dark)<0.65 or np.mean(support)<0.25:
                    continue
                score=0.5*float(np.mean(dark))+0.5*float(np.mean(support))
                candidates.append((score,x,y,side))
        if not candidates:
            continue
        score,x,y,side=max(candidates)
        if not any(math.hypot(p["x"]-x,p["y"]-y)<=6 for p in augmented):
            augmented.append({"x":x,"y":y,"score":score,"area":1,"source":"symbol_port_supported","symbol_hint":cid,"side_hint":side})
    return augmented


def _axis_morph(mask, axis, radius, *, erode=False):
    padding=[(0,0),(0,0)];padding[axis]=(radius,radius)
    padded=np.pad(mask,padding,constant_values=False)
    result=np.full_like(mask,erode,dtype=bool)
    for start in range(2*radius+1):
        region=[slice(None),slice(None)];region[axis]=slice(start,start+mask.shape[axis])
        if erode:result &= padded[tuple(region)]
        else:result |= padded[tuple(region)]
    return result


def palette_wire_support(visible_wire):
    """Short axial junction gaps and one antialias pixel, using NumPy only."""
    allowed=visible_wire.copy()
    for axis in (0,1):
        allowed |= _axis_morph(_axis_morph(visible_wire,axis,3),axis,3,erode=True)
    return _axis_morph(_axis_morph(allowed,0,1),1,1)


def anchored_achromatic_wires(bins, ink, probability, visible_wire, body_blocked, check=None):
    """Retain desaturated wire runs attached to an observed colored wire.

    Require a nine-pixel axial run, strong wire-model support, and a colored
    wire anchor. Component bodies and unrelated gray glyphs stay excluded.
    """
    candidate=(bins<0)&ink&(probability>=.5)&~body_blocked
    axial=np.zeros_like(candidate)
    for axis in (0,1):
        axial |= _axis_morph(_axis_morph(candidate,axis,4,erode=True),axis,4)
    if not axial.any():return axial
    labels,components=label_pixels(axial,check=check)
    touch=_axis_morph(_axis_morph(visible_wire&~body_blocked,0,1),1,1)
    anchored=np.unique(labels[touch&axial]);anchored=anchored[anchored>0]
    selected=np.zeros(len(components),bool);selected[anchored]=True
    return selected[labels]


def palette_filter_evidence(probability, ink, allowed, body_blocked):
    """Abstain if a proposed palette would erase most strong visible wires.

    Short glyph fragments are insufficient evidence: only axial runs of at
    least nine pixels, outside symbol bodies, enter this independent check.
    """
    strong=(probability>=.5)&ink&~body_blocked
    axial=np.zeros_like(strong)
    for axis in (0,1):
        axial |= _axis_morph(_axis_morph(strong,axis,4,erode=True),axis,4)
    count=int(np.count_nonzero(axial))
    retained=int(np.count_nonzero(axial&allowed))
    applied=count<32 or retained>=.5*count
    return {'applied':applied,'strong_axial_pixels':count,'retained_strong_axial_pixels':retained,
            'reason':'supported' if applied else 'would_remove_majority_of_strong_visible_wires'}


def text_stroke_obstacles(image, geometry):
    """Remove observed glyphs, retaining visibly continuous crossing strokes."""
    dark=image.min(2)<180;h,w=dark.shape
    blocked=np.zeros((h,w),bool)
    for text in geometry['texts'].values():
        bounds=box(text.get('location'))
        if (not bounds or text.get('state')=='deleted' or text.get('score',0)<.9
                or not str(text.get('content','')).strip()):continue
        x1,y1,x2,y2=[round(v) for v in bounds]
        x1,y1,x2,y2=max(1,x1+1),max(1,y1+1),min(w-1,x2),min(h-1,y2)
        if x2<=x1 or y2<=y1:continue
        local=np.ones((y2-y1,x2-x1),bool)
        if x1>=6 and x2+6<=w:
            for y in range(y1,y2):
                row=dark[y,x1-6:x2+6]
                if row[:6].mean()>=.8 and row[-6:].mean()>=.8 and row.mean()>=.85:
                    local[max(0,y-y1-1):y-y1+2,:]=False
        if y1>=6 and y2+6<=h:
            for x in range(x1,x2):
                column=dark[y1-6:y2+6,x]
                if column[:6].mean()>=.8 and column[-6:].mean()>=.8 and column.mean()>=.85:
                    local[:,max(0,x-x1-1):x-x1+2]=False
        blocked[y1:y2,x1:x2] |= local
    return blocked


def trace_nets(image, probabilities, points, geometry, wire_threshold=0.4, snap_radius=10, gap=6, split_cross=True, recover_strokes=False, ocr_box_labels=False, attachment_alignment=None, boundary_ports=False, crossing_predictor=None, crossing_confidence=.9, stroke_min_run=18,ctx=None, trusted_source_owners=False, palette_wire_hue=None, text_obstacles=False, achromatic_wire_recovery=False, palette_stroke_recovery=False,jog_body_obstacles=()):
    """Return a serializable connection graph plus raster debug outputs."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be RGB uint8 [H,W,3]")
    h, w = image.shape[:2]
    check=(lambda:ctx.time_budget.check(stage='raster_trace')) if ctx is not None else None
    if check:check()
    if probabilities.shape not in {(1,h,w),(3,h,w)} or not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite [1|3,H,W] in original coordinates")
    if not 0 <= wire_threshold <= 1 or snap_radius < 0 or gap < 0 or (attachment_alignment is not None and attachment_alignment < 0):
        raise ValueError("invalid topology threshold or radius")
    geometry = sanitize_geometry(geometry or {})
    if boundary_ports:
        from pcbparse.graph.raster_boundary_ports import recover_boundary_ports
        points = recover_boundary_ports(image, probabilities[0], points, geometry, .35, 12)
    points = symbol_port_candidates(image, probabilities[0], points, geometry)
    pins, issues = assign_pins(points, geometry, trusted_source_owners=trusted_source_owners)
    blocked = np.zeros((h, w), bool)
    for component in geometry["components"].values():
        if component.get("state") == "deleted":
            continue
        b = box(component.get("realBBox"))
        if b:
            x1, y1, x2, y2 = (int(round(v)) for v in b)
            # Include the body border: a rectangular symbol outline otherwise
            # creates a false conductive route around the excluded interior.
            if x2 >= 0 and y2 >= 0 and x1 < w and y1 < h:
                blocked[max(0, y1):min(h, y2+1), max(0, x1):min(w, x2+1)] = True
    raster_diagnostics = {'model_wire_pixels_outside_bodies': int(np.count_nonzero(
        (probabilities[0] >= wire_threshold) & ~blocked))}
    body_blocked=blocked.copy() if recover_strokes else None
    if palette_wire_hue is not None:
        from pcbparse.graph.palette_wire import hue_map
        bins,ink = hue_map(image)
        hue = int(palette_wire_hue)
        if not 0 <= hue < 12:
            raise ValueError('invalid palette wire hue')
        visible_wire = ink & (bins>=0) & (np.minimum((bins-hue)%12,(hue-bins)%12)<=1)
        if achromatic_wire_recovery:
            visible_wire |= anchored_achromatic_wires(bins,ink,probabilities[0],visible_wire,blocked,check=check)
        # Junction dots can have the symbol color. Bridge only short gaps
        # between same-color wire strokes; symbol interiors remain blocked.
        allowed = palette_wire_support(visible_wire)
        if palette_stroke_recovery:
            from pcbparse.graph.palette_continuity import supported_palette_strokes
            preserved, palette_recovery = supported_palette_strokes(
                image, probabilities[0], visible_wire, blocked, pins, geometry, check=check)
            allowed |= preserved
        palette_evidence=palette_filter_evidence(probabilities[0],ink,allowed,blocked)
        raster_diagnostics['palette_filter'] = palette_evidence
        if palette_evidence['applied']:
            blocked |= ~allowed
    if text_obstacles:
        blocked |= text_stroke_obstacles(image,geometry)
    wire = (probabilities[0] >= wire_threshold) & ~blocked
    raster_diagnostics['wire_pixels_after_filters'] = int(np.count_nonzero(wire))
    if recover_strokes:
        wire |= recover_visible_strokes(image, probabilities[0], blocked, geometry,min_run=stroke_min_run,check=check)
    gray = np.asarray(Image.fromarray(image).convert("L"))
    wire = repair_axial_gaps(wire, probabilities[0], blocked, gap, gray,check=check)
    if recover_strokes:
        from pcbparse.graph.terminal_wire_gaps import recover_terminal_wire_gaps
        # An unidentified printed chip is still an obstacle for this new
        # repair, as it is for the short-jog pass below.
        for bounds in jog_body_obstacles:
            b=box(bounds)
            if b is not None:
                x1,y1,x2,y2=map(round,b)
                if x2>=0 and y2>=0 and x1<w and y1<h:
                    body_blocked[max(0,y1):min(h,y2+1),max(0,x1):min(w,x2+1)]=True
        wire,terminal_gaps=recover_terminal_wire_gaps(wire,image,probabilities[0],body_blocked,
            pins,geometry,check=check)
        raster_diagnostics['terminal_wire_gaps']=terminal_gaps
    raster_diagnostics['wire_pixels_after_recovery'] = int(np.count_nonzero(wire))
    skeleton = thin(wire,check=check)
    if recover_strokes and gap>0:
        from pcbparse.graph.short_wire_jogs import recover_short_wire_jogs
        jog_blocked=blocked.copy()if jog_body_obstacles else blocked
        for bounds in jog_body_obstacles:
            b=box(bounds)
            if b is None:continue
            x1,y1,x2,y2=[round(v)for v in b]
            if x2>=0 and y2>=0 and x1<w and y1<h:
                jog_blocked[max(0,y1):min(h,y2+1),max(0,x1):min(w,x2+1)]=True
        skeleton,jogs=recover_short_wire_jogs(skeleton,image,jog_blocked,gap,check=check)
        jogs['unexported_body_obstacles']=len(jog_body_obstacles)
        raster_diagnostics['short_wire_jogs']=jogs
        wire |= skeleton
        raster_diagnostics['wire_pixels_after_recovery']=int(np.count_nonzero(wire))
    crossed, crossings = split_crossings(skeleton, image,predictor=crossing_predictor,confidence=crossing_confidence,check=check,compact=True) if split_cross else (skeleton, [])
    labels, segments = label_pixels(crossed,check=check)
    dsu = DSU(len(segments))
    for crossing in crossings:
        if crossing["connected"]:
            continue
        x, y = crossing["x"], crossing["y"]
        radius = crossing.get('cut_radius',3)
        sides = [nearest_labels(labels, x+dx*(radius+1), y+dy*(radius+1), 1, (dx, dy),1) for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))]
        for a, b in ((0, 1), (2, 3)):
            if all(sides[s] and (len(sides[s]) == 1 or sides[s][1][0] > sides[s][0][0]) for s in (a,b)):
                dsu.union(sides[a][0][1], sides[b][0][1])
        if crossing["review_required"]:
            issues.append({"code": "ambiguous_crossing", "x": x, "y": y})
    # Merge pin attachment to the same traced segment; never join merely
    # because pins are nearby or are two terminals of the same component.
    attached = {}; virtual_segments = set()
    for item in pins:
        candidates = nearest_labels(labels, item["x"], item["y"], snap_radius, item.get("outward_normal"), attachment_alignment)
        candidates = [x for x in candidates if len(segments[x[1]]) >= 3]
        roots = {}
        for distance, label, x, y in candidates:
            root = dsu.find(label)
            if root not in roots:
                roots[root] = (distance, label, x, y)
        candidates = sorted(roots.values())
        if candidates and (len(candidates) == 1 or candidates[1][0]-candidates[0][0] > 1.0):
            distance, label, x, y = candidates[0]
            attached[item["pin_id"]] = label
            item["wire_attachment"] = {"x": x, "y": y, "distance_px": round(distance, 3)}
        else:
            code = "ambiguous_wire_attachment" if candidates else "pin_without_detected_wire"
            issues.append({"code": code, "pin_id": item["pin_id"]})
            item["wire_attachment"] = None
            # An untraced terminal is still an electrical vertex. It may be
            # connected globally by an explicit power/ground name, without
            # inventing a physical wire.
            label = len(segments); segments.append(np.asarray([[int(round(item["y"])), int(round(item["x"]))]], np.int32))
            dsu.parent.append(label); virtual_segments.add(label); attached[item["pin_id"]] = label
    # Explicit OCR net-label anchors support semantic/global-net union. A
    # generic nearby text is deliberately not considered an electrical label.
    label_records = []
    for raw in geometry.get("net_labels", []):
        try:
            name = str(raw.get("name") or "").strip(); x, y = float(raw["x"]), float(raw["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if not name or not math.isfinite(x) or not math.isfinite(y):
            continue
        radius=raw.get('radius',snap_radius)
        if type(radius) is not int or not 0<=radius<=snap_radius:
            raise ValueError('net label radius must be an integer within the attachment limit')
        candidates = nearest_labels(labels, x, y, radius)
        if not candidates or (len(candidates)>1 and dsu.find(candidates[0][1]) != dsu.find(candidates[1][1]) and candidates[1][0]-candidates[0][0] <= 1):
            issues.append({"code": "unattached_net_label", "name": name, "x": x, "y": y})
            continue
        label_records.append({"name": name, "segment": candidates[0][1], "global": bool(raw.get("global", True)), "scope": raw.get("scope", "sheet"), "source": "ocr_anchor"})
    if ocr_box_labels:
        # Use only explicitly classified electrical labels from OCR. Never
        # consult edge.netlabel, edge.connect or unclassified nearby text.
        for tid, token in geometry["texts"].items():
            if token.get("state") == "deleted" or str(token.get("details") or "").lower() != "netlabel":
                continue
            name = str(token.get("content") or "").strip()
            bounds = box(token.get("location"))
            if not name or name.isdecimal() or bounds is None:
                continue
            x1, y1, x2, y2 = bounds
            left, right = max(0, int(math.floor(x1))-snap_radius), min(w, int(math.ceil(x2))+snap_radius+1)
            top, bottom = max(0, int(math.floor(y1))-snap_radius), min(h, int(math.ceil(y2))+snap_radius+1)
            if left >= right or top >= bottom:
                continue
            yy, xx = np.nonzero(labels[top:bottom, left:right])
            best = {}
            for px, py in zip(xx+left, yy+top):
                # Pixels inside text boxes can be glyph false positives.
                if x1 <= px <= x2 and y1 <= py <= y2:
                    continue
                distance = point_box_distance(float(px), float(py), bounds)
                label = int(labels[py, px]); root = dsu.find(label)
                if distance <= snap_radius and len(segments[label]) >= 3 and (root not in best or distance < best[root][0]):
                    best[root] = (distance, label)
            choices = sorted(best.values())
            if not choices or (len(choices) > 1 and choices[1][0]-choices[0][0] <= 2):
                issues.append({"code": "unattached_net_label", "name": name, "text_id": tid, "source": "ocr_netlabel_box"})
                continue
            label_records.append({"name": name, "segment": choices[0][1], "global": True, "scope": "sheet", "source": "ocr_netlabel_box"})
    # Ground/power name inference requires teammate symbol metadata. Analog
    # and digital ground remain different when explicit names are supplied.
    for cid, component in geometry["components"].items():
        owned = [p for p in pins if p["component_id"] == cid and p["pin_id"] in attached]
        if trusted_source_owners:
            # Recovered boundary candidates have no output identity. They
            # cannot give a one-terminal supply several conductive ports and
            # thereby join otherwise separate source-observed wires.
            owned = [p for p in owned if p.get('source_pin_id') and p.get('component_hint') == cid]
        name = power_name(component)
        category = str(component.get("category") or "").lower()
        valid_symbol = category in ("gnd", "ground", "power", "powerport") or (category == "v" and len(owned) == 1)
        if name and owned and (component.get("global_net") or valid_symbol):
            for pin in owned:
                label_records.append({"name": str(name).strip(), "segment": attached[pin["pin_id"]], "global": True, "scope": "sheet", "source": "symbol_metadata"})
    conflicted_roots = set()
    if trusted_source_owners:
        names_by_root = defaultdict(set)
        for rec in label_records:
            if rec['global']:
                names_by_root[dsu.find(rec['segment'])].add((str(rec['scope']),rec['name']))
        conflicted_roots = {root for root,names in names_by_root.items() if len(names)>1}
        for root in sorted(conflicted_roots):
            issues.append({'code':'physical_net_name_conflict','segment_root':root,
                           'names':sorted(name for _,name in names_by_root[root]),
                           'global_union_withheld':True})
    named = {}
    for rec in label_records:
        if rec["global"] and dsu.find(rec['segment']) not in conflicted_roots:
            key = (str(rec["scope"]), rec["name"])
            if key in named:
                dsu.union(rec["segment"], named[key])
            else:
                named[key] = rec["segment"]
    grouped = defaultdict(list)
    for segment in range(1, len(segments)):
        grouped[dsu.find(segment)].append(segment)
    nets, root_ids = [], {}
    connected_roots = {dsu.find(segment) for segment in attached.values()}
    for root in sorted(grouped):
        members = grouped[root]
        coords = np.concatenate([segments[i] for i in members])
        physical_members = [i for i in members if i not in virtual_segments]
        if len(coords) < 3 and root not in connected_roots:
            continue
        names = sorted({rec["name"] for rec in label_records if dsu.find(rec["segment"]) == root})
        net_id = f"NET_{len(nets)+1:04d}"; root_ids[root] = net_id
        nets.append({"net_id": net_id, "names": names, "pin_ids": [], "segment_ids": physical_members, "bbox": [int(coords[:,1].min()), int(coords[:,0].min()), int(coords[:,1].max()), int(coords[:,0].max())], "wire_pixels": sum(len(segments[i]) for i in physical_members), "status": "traced" if physical_members else ("semantic_only" if names else "unresolved_or_open")})
        if len(names) > 1:
            issues.append({"code": "conflicting_net_names", "net_id": net_id, "names": names})
    net_by_id = {n["net_id"]: n for n in nets}
    for item in pins:
        if item["pin_id"] in attached:
            item["net_id"] = root_ids[dsu.find(attached[item["pin_id"]])]
        else:
            item["net_id"] = f"NET_{len(nets)+1:04d}"
            isolated = {"net_id": item["net_id"], "names": [], "pin_ids": [], "segment_ids": [], "bbox": None, "wire_pixels": 0, "status": "unresolved_or_open"}
            nets.append(isolated); net_by_id[item["net_id"]] = isolated
        net_by_id[item["net_id"]]["pin_ids"].append(item["pin_id"])
    for item in pins:
        if item["component_id"] is not None and item["pin_number"] is None:
            issues.append({"code": "missing_pin_number", "pin_id": item["pin_id"], "component_id": item["component_id"]})
    graph = {
        "schema_version": "eda.connection_graph.v1", "image_size": [w, h], "coordinate_origin": "top_left_xy_pixels",
        "source": "model_predictions_and_component_ocr_geometry", "uses_annotation_pin_edge_truth": False,
        "components": [_component_metadata(cid, c, geometry["texts"]) for cid, c in geometry["components"].items() if c.get("state") != "deleted"],
        "pins": pins, "nets": nets, "crossings": crossings, "issues": issues,
        "status": "review_required" if issues else "traced", "spice_ready": False,
        "parameters": {"wire_threshold": wire_threshold, "snap_radius": snap_radius, "axial_gap_px": gap, "crossing_split_enabled": split_cross, "recover_visible_strokes": recover_strokes, "ocr_box_labels": ocr_box_labels,
                       "text_obstacles":text_obstacles,
                       "attachment_alignment": attachment_alignment, "boundary_ports": boundary_ports},
    }
    if crossing_predictor is not None:
        graph['parameters'].update(learned_crossings=True,crossing_confidence=crossing_confidence)
    if stroke_min_run!=18:graph['parameters']['stroke_min_run']=stroke_min_run
    if trusted_source_owners:graph['parameters']['trusted_source_owners']=True
    if palette_wire_hue is not None:graph['parameters']['palette_wire_hue']=int(palette_wire_hue)
    if palette_wire_hue is not None and palette_stroke_recovery:
        graph['parameters']['palette_stroke_recovery']=palette_recovery
    raster_diagnostics.update(pins=len(pins),
        pins_with_wire_attachment=sum(p.get('wire_attachment') is not None for p in pins),
        singleton_nets=sum(len(n['pin_ids']) == 1 for n in nets))
    graph['raster_diagnostics'] = raster_diagnostics
    validate_graph(graph)
    return graph, {"skeleton": skeleton, "wire_labels": labels, "wire_mask": wire}


def validate_graph(graph):
    """Check references and membership; this does not certify electrical truth."""
    pins = {p["pin_id"]: p for p in graph["pins"]}
    nets = {n["net_id"]: n for n in graph["nets"]}
    components = {c["component_id"] for c in graph["components"]}
    if len(pins) != len(graph["pins"]) or len(nets) != len(graph["nets"]) or len(components) != len(graph["components"]):
        raise ValueError("duplicate graph IDs")
    members = set()
    for nid, net in nets.items():
        for pid in net["pin_ids"]:
            if pid not in pins or pid in members or pins[pid]["net_id"] != nid:
                raise ValueError("invalid or duplicate net pin membership")
            members.add(pid)
    if members != set(pins):
        raise ValueError("every pin must belong to exactly one net")
    for pin in pins.values():
        if pin["component_id"] is not None and pin["component_id"] not in components:
            raise ValueError("unknown component reference")
        if not all(math.isfinite(float(pin[k])) for k in ("x", "y")):
            raise ValueError("invalid pin coordinates")
    return True


def connection_list(graph):
    """Readable electrical connection candidates, not executable SPICE."""
    by_pin = {p["pin_id"]: p for p in graph["pins"]}
    lines = ["# Connection candidates from image inference", f"# status={graph['status']} spice_ready={graph['spice_ready']}", "# T* denotes local detected terminals, not physical pin numbers"]
    for net in graph["nets"]:
        if not net["pin_ids"]:
            continue
        terminals = []
        for pid in net["pin_ids"]:
            pin = by_pin[pid]
            terminals.append(f"{pin['component_id'] or 'UNASSIGNED'}.{pin['pin_number'] or pin['terminal_id'].split(':')[-1]}({pid})")
        lines.append(f"{net['net_id']} [{','.join(net['names'])}] : {' '.join(terminals)} ; {net['status']}")
    return "\n".join(lines)+"\n"


def draw_graph(image, graph):
    """Compact pin/net overlay with stable net colors."""
    output = Image.fromarray(image).convert("RGB"); draw = ImageDraw.Draw(output)
    for pin in graph["pins"]:
        number = int(pin["net_id"].split("_")[-1]); color = ((number*71)%180+40, (number*101)%180+40, (number*137)%180+40)
        x, y = pin["x"], pin["y"]
        draw.ellipse((x-2, y-2, x+2, y+2), outline=color)
        attach = pin.get("wire_attachment")
        if attach:
            draw.line((x, y, attach["x"], attach["y"]), fill=color)
    return output
