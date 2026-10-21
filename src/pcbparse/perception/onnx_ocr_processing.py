"""Bounded NumPy/Pillow preprocessing for DB detection and CTC recognition.

No Paddle, OpenCV, implicit downloads, annotation access or semantic fields.
Pixel and polygon equivalence with a reference implementation must be measured;
these routines do not assume that an ONNX export alone proves OCR equivalence.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw


class OCRProcessingError(ValueError):
    pass


def _raster(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise OCRProcessingError('expected HWC three-channel uint8 pixels')
    if min(image.shape[:2]) < 1 or image.shape[0] * image.shape[1] > 64_000_000:
        raise OCRProcessingError('raster dimensions exceed bounded OCR input')
    return image


def resize_linear(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Half-pixel bilinear resize without Pillow's downsampling antialiasing."""
    image = _raster(image)
    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        raise OCRProcessingError('resize dimensions must be positive integers')
    if width * height > 16_000_000:
        raise OCRProcessingError('resized raster exceeds pixel bound')
    h, w = image.shape[:2]
    if (height, width) == (h, w): return image.copy()
    xs = np.clip((np.arange(width) + .5) * w / width - .5, 0, w-1)
    ys = np.clip((np.arange(height) + .5) * h / height - .5, 0, h-1)
    x0, y0 = xs.astype(np.int64), ys.astype(np.int64)
    x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
    fx, fy = (xs-x0)[None, :, None], (ys-y0)[:, None, None]
    # Row blocks keep peak temporary memory bounded on large detector inputs.
    result = np.empty((height, width, 3), dtype=np.uint8)
    for start in range(0, height, 64):
        end = min(height, start+64)
        top = image[y0[start:end, None], x0] * (1-fx) + image[y0[start:end, None], x1] * fx
        bottom = image[y1[start:end, None], x0] * (1-fx) + image[y1[start:end, None], x1] * fx
        result[start:end] = np.clip(np.rint(top*(1-fy[start:end])+bottom*fy[start:end]), 0, 255).astype(np.uint8)
    return result


def detection_tensor(bgr: np.ndarray, *, limit_side_len: int = 64,
                     max_side_limit: int = 4000) -> np.ndarray:
    bgr = _raster(bgr)
    if type(limit_side_len) is not int or type(max_side_limit) is not int or not 1 <= limit_side_len <= max_side_limit <= 4000:
        raise OCRProcessingError('invalid detector resize limits')
    h, w = bgr.shape[:2]
    source = bgr
    if h+w < 64:
        source = np.zeros((max(32, h), max(32, w), 3), dtype=np.uint8)
        source[:h, :w] = bgr
    sh, sw = source.shape[:2]
    ratio = max(1., limit_side_len/min(sh, sw))
    rh, rw = int(sh*ratio), int(sw*ratio)
    if max(rh, rw) > max_side_limit:
        ratio = max_side_limit/max(rh, rw)
        rh, rw = int(rh*ratio), int(rw*ratio)
    rh, rw = max(32, round(rh/32)*32), max(32, round(rw/32)*32)
    pixels = resize_linear(source, rw, rh).astype(np.float32) / 255.
    normalized = (pixels-np.array([.485,.456,.406], dtype=np.float32)) / np.array([.229,.224,.225], dtype=np.float32)
    return np.ascontiguousarray(normalized.transpose(2,0,1)[None], dtype=np.float32)


def _regions(mask: np.ndarray, check: Callable[[], None] | None = None):
    """Eight-connected run components; hull vertices need only run endpoints."""
    parents = []
    runs = []
    previous = []
    def find(label):
        while parents[label] != label:
            parents[label] = parents[parents[label]]
            label = parents[label]
        return label
    # Batch transitions within the existing 32-row deadline interval. This
    # avoids padding and scanning each row separately, with bounded temporary
    # memory and identical row order, run endpoints and union operations.
    for first in range(0, len(mask), 32):
        if check is not None: check()
        chunk = mask[first:first+32].astype(np.int8)
        transitions = np.diff(np.pad(chunk, ((0,0),(1,1))), axis=1)
        start_y, start_x = np.nonzero(transitions == 1)
        _, end_x = np.nonzero(transitions == -1)
        counts = np.bincount(start_y, minlength=len(chunk))
        cursor = 0
        for row, count in enumerate(counts):
            y = first+row
            current = []
            offset = 0
            for index in range(cursor, cursor+int(count)):
                start, end = int(start_x[index]), int(end_x[index])-1
                label = len(parents)
                if label >= 200_000: raise OCRProcessingError('detector bitmap exceeds component-run bound')
                parents.append(label)
                while offset < len(previous) and previous[offset][1] < start-1: offset += 1
                k = offset
                while k < len(previous) and previous[k][0] <= end+1:
                    left, right = find(label), find(previous[k][2])
                    if left != right: parents[max(left,right)] = min(left,right)
                    k += 1
                current.append((start,end,label))
                runs.append((start,end,y,label))
            cursor += int(count)
            previous = current
    grouped = defaultdict(list)
    for start,end,y,label in runs:
        grouped[find(label)].extend(((start,y),(end,y)))
    return sorted(grouped.values(), key=lambda points: (min(y for _,y in points), min(x for x,_ in points)), reverse=True)


def convex_hull(points) -> np.ndarray:
    points = sorted(set(tuple(p) for p in points))
    if len(points) < 3: return np.asarray(points, dtype=float).reshape(-1,2)
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2],lower[-1],p) <= 0: lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2],upper[-1],p) <= 0: upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1]+upper[:-1], dtype=float)


def minimum_rectangle(points) -> tuple[np.ndarray, float, float]:
    hull = convex_hull(points)
    if len(hull) < 3: return np.zeros((4,2)), 0., 0.
    deltas = np.roll(hull,-1,axis=0)-hull
    angles = np.unique(np.round(np.mod(np.arctan2(deltas[:,1], deltas[:,0]), math.pi/2), 12))
    best = None
    for angle in angles:
        c,s = math.cos(angle), math.sin(angle)
        axes = np.array([[c,s],[-s,c]])
        rotated = hull@axes.T
        low,high = rotated.min(0), rotated.max(0)
        widths = high-low
        choice = (float(np.prod(widths)), float(angle))
        if best is None or choice < best[0]: best = (choice,low,high,axes,widths)
    _,low,high,axes,widths = best
    quad = np.array([[low[0],low[1]],[high[0],low[1]],[high[0],high[1]],[low[0],high[1]]])@axes
    # Match a stable left pair/right pair order, including rotated rectangles.
    quad = sorted(quad.tolist(), key=lambda p:p[0])
    left = sorted(quad[:2],key=lambda p:p[1]); right = sorted(quad[2:],key=lambda p:p[1])
    return np.asarray([left[0],right[0],right[1],left[1]]), float(widths[0]), float(widths[1])


def _rectangle_score(probability: np.ndarray, quad: np.ndarray) -> float:
    h,w = probability.shape
    low = np.maximum(np.floor(quad.min(0)).astype(int), 0)
    high = np.minimum(np.ceil(quad.max(0)).astype(int), [w-1,h-1])
    if np.any(high < low): return 0.
    mask = Image.new('L', (int(high[0]-low[0]+1), int(high[1]-low[1]+1)), 0)
    local = np.trunc(quad-low).astype(int)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in local.tolist()], fill=1)
    selected = np.asarray(mask, dtype=bool)
    values = probability[low[1]:high[1]+1, low[0]:high[0]+1][selected]
    return float(values.mean(dtype=np.float64)) if len(values) else 0.


def _rounded_rectangle_offset(quad: np.ndarray, distance: float) -> np.ndarray:
    """Offset a convex text rectangle on an integer lattice with round joins.

    Integer vertices and arc samples are retained before fitting the expanded
    rectangle. Scaling the floating supporting lines directly to the source
    image gives different crop pixels when detector dimensions were rounded.
    """
    path = np.trunc(quad).astype(np.int64)
    edges = np.roll(path,-1,axis=0)-path
    lengths = np.linalg.norm(edges,axis=1)
    if np.any(lengths == 0): return np.empty((0,2))
    normals = np.stack((edges[:,1],-edges[:,0]),axis=1)/lengths[:,None]
    signed_area = np.sum(path[:,0]*np.roll(path[:,1],-1)-np.roll(path[:,0],-1)*path[:,1])
    if signed_area < 0: normals = -normals
    tolerance = min(.25,distance*.25)
    total_steps = min(math.pi/math.acos(1-tolerance/distance),distance*math.pi)
    increment = 2*math.pi/total_steps
    c,s = math.cos(increment),math.sin(increment)
    points = []
    def rounded(value):
        return np.where(value < 0,np.ceil(value-.5),np.floor(value+.5)).astype(np.int64)
    for index,vertex in enumerate(path):
        previous,current = normals[index-1].copy(),normals[index]
        angle = math.atan2(previous[0]*current[1]-current[0]*previous[1],previous@current)
        steps = max(1,int(math.floor(total_steps/(2*math.pi)*abs(angle)+.5)))
        rotation = 1 if angle >= 0 else -1
        for _ in range(steps):
            points.append(rounded(vertex+distance*previous))
            previous = np.array([previous[0]*c-previous[1]*s*rotation,
                                 previous[0]*s*rotation+previous[1]*c])
        points.append(rounded(vertex+distance*current))
    return np.asarray(points)


def db_polygons(probability: np.ndarray, *, width: int, height: int,
                threshold: float = .3, box_threshold: float = .6,
                unclip_ratio: float = 1.5, max_candidates: int = 1000,
                check: Callable[[], None] | None = None) -> list[tuple[np.ndarray,float]]:
    probability = np.asarray(probability)
    if probability.ndim != 2 or probability.size > 16_000_000 or probability.size == 0 or not np.isfinite(probability).all():
        raise OCRProcessingError('invalid detector probability bitmap')
    low, high = float(probability.min()), float(probability.max())
    if low < 0 or high > 1:
        # A float32 sigmoid kernel can exceed its mathematical endpoint by
        # one ULP. Normalize only that rounding error; logits and materially
        # out-of-range output still fail closed. In-range bitmaps stay intact.
        endpoint_tolerance = float(np.finfo(np.float32).eps)
        if low < -endpoint_tolerance or high > 1 + endpoint_tolerance:
            raise OCRProcessingError('detector must emit probabilities in [0,1]')
        probability = np.clip(probability, 0, 1)
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (threshold,box_threshold,unclip_ratio)):
        raise OCRProcessingError('invalid finite DB parameters')
    if not 0 < threshold < 1 or not 0 <= box_threshold <= 1 or not 0 < unclip_ratio <= 4:
        raise OCRProcessingError('DB parameter out of range')
    if type(max_candidates) is not int or not 1 <= max_candidates <= 2000 or min(width,height) < 1:
        raise OCRProcessingError('invalid DB candidate/image bound')
    h,w = probability.shape
    result = []
    for index,points in enumerate(_regions(probability > threshold, check)[:max_candidates]):
        if check is not None and index % 16 == 0: check()
        quad,rw,rh = minimum_rectangle(points)
        if min(rw,rh) < 3: continue
        score = _rectangle_score(probability, quad)
        if score < box_threshold: continue
        distance = rw*rh*unclip_ratio/(2*(rw+rh))
        expanded,expanded_w,expanded_h = minimum_rectangle(_rounded_rectangle_offset(quad,distance))
        if min(expanded_w,expanded_h) < 5: continue
        expanded[:,0] = np.clip(np.rint(expanded[:,0]*width/w), 0, width)
        expanded[:,1] = np.clip(np.rint(expanded[:,1]*height/h), 0, height)
        # Upscaled tiny ROIs can collapse a valid detector-space rectangle
        # when rounding/clipping it back to source pixels. Such a polygon
        # contains no crop; reject it here without suppressing crop errors.
        crop_width = int(max(np.linalg.norm(expanded[0]-expanded[1]),
                             np.linalg.norm(expanded[2]-expanded[3])))
        crop_height = int(max(np.linalg.norm(expanded[0]-expanded[3]),
                              np.linalg.norm(expanded[1]-expanded[2])))
        twice_area = abs(np.sum(expanded[:,0]*np.roll(expanded[:,1],-1)
                                - expanded[:,1]*np.roll(expanded[:,0],-1)))
        if min(crop_width,crop_height) < 1 or twice_area == 0:
            continue
        result.append((expanded.astype(np.float32), score))
    result.sort(key=lambda item: (item[0][0,1],item[0][0,0]))
    for i in range(len(result)-1):
        for j in range(i,-1,-1):
            if abs(result[j+1][0][0,1]-result[j][0][0,1]) < 10 and result[j+1][0][0,0] < result[j][0][0,0]:
                result[j],result[j+1] = result[j+1],result[j]
            else: break
    return result


def _cubic(distance):
    a = -.75
    x = np.abs(distance)
    return np.where(x <= 1, (a+2)*x**3-(a+3)*x**2+1,
                    np.where(x < 2, a*x**3-5*a*x**2+8*a*x-4*a, 0.))


def crop_quad(bgr: np.ndarray, quad: np.ndarray, *,
              check: Callable[[], None] | None = None) -> np.ndarray:
    """Perspective crop with cubic sampling and replicated image borders."""
    bgr = _raster(bgr)
    quad = np.asarray(quad,dtype=float)
    if quad.shape != (4,2) or not np.isfinite(quad).all(): raise OCRProcessingError('invalid crop quad')
    width = int(max(np.linalg.norm(quad[0]-quad[1]),np.linalg.norm(quad[2]-quad[3])))
    height = int(max(np.linalg.norm(quad[0]-quad[3]),np.linalg.norm(quad[1]-quad[2])))
    if min(width,height) < 1 or width*height > 16_000_000: raise OCRProcessingError('invalid crop size')
    destination = np.asarray([[0,0],[width,0],[width,height],[0,height]],dtype=float)
    matrix, values = [], []
    for (x,y),(u,v) in zip(destination,quad):
        matrix.extend(([x,y,1,0,0,0,-u*x,-u*y],[0,0,0,x,y,1,-v*x,-v*y]))
        values.extend((u,v))
    try: transform = np.append(np.linalg.solve(matrix,values),1).reshape(3,3)
    except np.linalg.LinAlgError as exc: raise OCRProcessingError('degenerate crop quad') from exc
    # Large false detections must not allocate full-frame float64 sampling
    # grids. Row blocks bound working memory and permit cooperative deadlines.
    output = np.empty((height,width,3),dtype=np.uint8)
    xx = np.arange(width)[None,:]
    for start in range(0,height,64):
        if check is not None: check()
        end = min(height,start+64)
        yy = np.arange(start,end)[:,None]
        denominator = transform[2,0]*xx+transform[2,1]*yy+1
        if np.any(np.abs(denominator) < 1e-8): raise OCRProcessingError('degenerate crop homography')
        sx = np.rint(((transform[0,0]*xx+transform[0,1]*yy+transform[0,2])/denominator)*32)/32
        sy = np.rint(((transform[1,0]*xx+transform[1,1]*yy+transform[1,2])/denominator)*32)/32
        ix,iy = np.floor(sx).astype(int),np.floor(sy).astype(int)
        result = np.zeros((end-start,width,3),dtype=float)
        for dy in (-1,0,1,2):
            wy = _cubic(sy-(iy+dy))
            for dx in (-1,0,1,2):
                weight = wy*_cubic(sx-(ix+dx))
                result += bgr[np.clip(iy+dy,0,bgr.shape[0]-1), np.clip(ix+dx,0,bgr.shape[1]-1)]*weight[:,:,None]
        output[start:end] = np.clip(np.rint(result),0,255).astype(np.uint8)
    return np.ascontiguousarray(np.rot90(output) if height/width >= 1.5 else output)


def recognition_tensor(bgr: np.ndarray, *, height: int = 48,
                       minimum_width: int = 320, maximum_width: int = 3200) -> np.ndarray:
    bgr = _raster(bgr)
    if height != 48 or not 1 <= minimum_width <= maximum_width <= 3200:
        raise OCRProcessingError('unsupported recognizer dimensions')
    h,w = bgr.shape[:2]
    canvas_width = min(maximum_width, max(minimum_width, int(height*w/h)))
    resized_width = min(canvas_width, int(math.ceil(height*w/h)))
    pixels = resize_linear(bgr,resized_width,height).astype(np.float32).transpose(2,0,1)/255.
    output = np.zeros((3,height,canvas_width),dtype=np.float32)
    output[:,:,:resized_width] = (pixels-.5)/.5
    return output


def decode_ctc(probabilities: np.ndarray, characters: tuple[str,...]) -> list[tuple[str,float]]:
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 3 or probabilities.shape[2] != len(characters) or probabilities.shape[1] < 1:
        raise OCRProcessingError('CTC output/charset dimensions differ')
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise OCRProcessingError('CTC output is not finite probabilities')
    indices = probabilities.argmax(-1); scores = probabilities.max(-1)
    outputs = []
    for row,confidence in zip(indices,scores):
        selected = np.ones(len(row),dtype=bool)
        selected[1:] = row[1:] != row[:-1]
        selected &= row != 0
        outputs.append((''.join(characters[i] for i in row[selected]),
                        float(confidence[selected].mean()) if selected.any() else 0.))
    return outputs
