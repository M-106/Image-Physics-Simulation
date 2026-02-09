"""
**Image Source Method (ISM) Ray Propagation and Visibility Maps**

This module implements a fast 2D Image Source Method (ISM) solver for simulating
specular reflections and line-of-sight propagation inside environments described
by images containing wall/obstacle pixels.

The environment is interpreted as a raster map where certain pixel values denote
walls. From this, geometric wall segments are extracted and used to construct
image sources for multiple reflection orders. For each receiver position on a
grid, valid reflection paths from the source are built, validated using a
raster-based visibility test, and accumulated into a *path count map*.

The implementation avoids heavy geometry libraries (e.g., shapely) and instead
relies on:
- analytic segment intersection
- Bresenham raster visibility checks
- OpenCV contour extraction for wall geometry

This makes the method fast, portable, and well suited for large-scale map
evaluation, dataset generation, and simulation experiments.

Core idea:
1. Extract wall boundaries from a segmentation / mask image.
2. Convert walls to geometric segments.
3. Precompute image sources for all reflection sequences up to `max_order`.
4. For a grid of receiver positions:
   - Build candidate reflection paths
   - Check visibility against an occlusion raster
   - Accumulate path count

As ASCII model:
```text
                 ┌──────────────────────────────┐
                 │        Input: Scene          │
                 │  (image, walls, source)      │
                 └──────────────┬───────────────┘
                                │
                                v
                 ┌──────────────────────────────┐
                 │   Raster → Wall Geometry     │
                 │   Extract wall segments      │
                 └──────────────┬───────────────┘
                                │
                                v
                 ┌──────────────────────────────┐
                 │     Build Occlusion Map      │
                 │   (for visibility checks)    │
                 └──────────────┬───────────────┘
                                │
                                v
                 ┌──────────────────────────────┐
                 │  Precompute Image Sources    │
                 │  (all reflection sequences)  │
                 └──────────────┬───────────────┘
                                │
                                v
                 ┌──────────────────────────────┐
                 │     Define Receiver Grid     │
                 └──────────────┬───────────────┘
                                │
                                v
                 ┌──────────────────────────────────────────┐
                 │ For each Receiver position R on the grid │
                 └──────────────┬───────────────────────────┘
                                │
                                v
                 ┌──────────────────────────────────────────┐
                 │ For each Image Source (reflection seq)   │
                 └──────────────┬───────────────────────────┘
                                │
                                v
                 ┌──────────────────────────────┐
                 │  Construct reflection path   │
                 │   (geometry / intersections) │
                 └──────────────┬───────────────┘
                                │
                                v
                 ┌──────────────────────────────┐
                 │  Check path visibility       │
                 │   (raster occlusion test)    │
                 └──────────────┬───────────────┘
                                │
                                v
                 ┌──────────────────────────────┐
                 │  Accumulate contribution     │
                 └──────────────┬───────────────┘
                                │
                                v
                 ┌──────────────────────────────┐
                 │      Write into map          │
                 └──────────────┬───────────────┘
                                │
                                v
                 ┌──────────────────────────────┐
                 │          Output Map          │
                 └──────────────────────────────┘
```

Main features:
- Wall extraction from binary or labeled images
- Exact specular reflection using image source construction
- Raster-based visibility testing (very fast)
- Support for higher reflection orders
- Optional multiprocessing via joblib

Example:
```python
count_map = compute_map_ism_fast(
    source_rel=(0.5, 0.5),
    img=segmentation_img,
    wall_values=[0],
    max_order=2,
    step_px=8,
    mode="count"
)
```

Dependencies:
- numpy
- OpenCV (cv2)
- Optional: joblib (for parallelization)


Functions:
* reflect_point_across_infinite_line(...)
* reflection_map_to_img(...)
* Segment(...)
* _seg_seg_intersection(...)
* _bresenham_points(...)
* is_visible_raster(...)
* build_wall_mask(...)
* get_wall_segments_from_mask(...)
* build_occlusion_from_wallmask(...)
* enumerate_wall_sequences_indices(...)
* precompute_image_sources(...)
* build_path_for_sequence(...)
* check_path_visibility_raster(...)
* compute_reflection_map(...)


Author:<br>
Tobia Ippolito, 2025
"""



# ---------------
# >>> Imports <<<
# ---------------
from __future__ import annotations
from functools import cache

from .math import normalize_point

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable

import numpy as np
import cv2

# optimization
from joblib import Parallel, delayed
import numba



# --------------
# >>> Helper <<<
# --------------

def reflect_point_across_infinite_line(P: Tuple[float, float], A: Tuple[float, float], B: Tuple[float, float]) -> Tuple[float, float]:
    """
    Reflect a 2D point across an infinite line.

    Computes the mirror image of point P with respect to the infinite line
    passing through points A and B.

                  P
                  *
                   \
                    \
    -----------------*----------------   Wand (A-B)
                    proj
                      \
                       \
                        *
                        P'

    Parameters:
    - P (Tuple[float, float]): <br>
        The point to reflect (x, y).
    - A (Tuple[float, float]): <br>
        First point defining the line (x, y).
    - B (Tuple[float, float]): <br>
        Second point defining the line (x, y).

    Returns:
    - Tuple[float, float]: <br>
        The reflected point (x, y) as floats.
    """
    # var setup
    #    -> A/B are both points on the wall
    #    -> P is the point we want to mirror (not receiver, source ,...)
    x1, y1 = A
    x2, y2 = B
    px, py = P

    # get direction of the wall
    #    -> from point to the wall
    ABx = x2 - x1
    ABy = y2 - y1
    # norm this wall direction
    n = math.hypot(ABx, ABy) + 1e-12
    ux, uy = ABx / n, ABy / n

    # direction/vector from the wall to the point
    APx = px - x1
    APy = py - y1

    # projection of point to wall direction to the wall direction
    #   -> finding sweetspot where we are directly under P on wall line/direction
    t = APx * ux + APy * uy
    # point on wall direction/line which is directly under P
    projx = x1 + t * ux
    projy = y1 + t * uy

    # calc reflection/mirror
    rx = projx + (projx - px)
    ry = projy + (projy - py)

    return (float(rx), float(ry))



@numba.njit(cache=True, fastmath=True)
def reflect_point_across_infinite_line_numba(px, py, ax, ay, bx, by):
    """
    Reflect a 2D point across an infinite line defined by two points.

    Computes the mirror image of point P = (px, py) with respect to the
    infinite line passing through A = (ax, ay) and B = (bx, by).
    The function first projects P orthogonally onto the line AB to obtain
    the foot point (projection), and then reflects P across this point
    using the relation:

        P' = 2 * proj - P

    This implementation is written in a Numba-friendly style and avoids
    expensive operations such as `math.hypot`, using only basic arithmetic
    and a square root for normalization.

    Parameters:
    - px (float): <br>
        x-coordinate of the point to reflect.
    - py (float): <br>
        y-coordinate of the point to reflect.
    - ax (float): <br>
        x-coordinate of the first point defining the line.
    - ay (float): <br>
        y-coordinate of the first point defining the line.
    - bx (float): <br>
        x-coordinate of the second point defining the line.
    - by (float): <br>
        y-coordinate of the second point defining the line.

    Returns:
    - Tuple[float, float]: <br>
        The reflected point (rx, ry) across the infinite line AB.
    """
    abx = bx - ax
    aby = by - ay

    # avoid hypot for speed + numba friendliness
    n = math.sqrt(abx * abx + aby * aby) + 1e-12
    ux = abx / n
    uy = aby / n

    apx = px - ax
    apy = py - ay

    # projection parameter on unit direction
    t = apx * ux + apy * uy
    projx = ax + t * ux
    projy = ay + t * uy

    # reflection: P' = 2*proj - P
    rx = 2.0 * projx - px
    ry = 2.0 * projy - py

    return rx, ry



def reflection_map_to_img(reflection_map):
    """
    Convert an reflection map to a uint8 visualization image.

    Normalizes the input reflection_map to [0, 255] by dividing by its maximum
    value (with an epsilon for numerical stability), and converts the result
    to uint8.

    Parameters:
    - reflection_map (np.ndarray): <br>
        A numeric array representing reflection values.

    Returns:
    - np.ndarray: <br>
        A uint8 image array with values in [0, 255].
    """
    vis = reflection_map.copy()
    vis = vis / (vis.max() + 1e-9)
    return (vis * 255).astype(np.float64)  # .astype(np.uint8)



# -----------------------------------------
# >>> Segment representation & geometry <<<
# -----------------------------------------

@numba.njit(cache=True, fastmath=True)
def seg_seg_intersection_xy(x1, y1, x2, y2, x3, y3, x4, y4, eps=1e-9):
    """
    Compute the intersection point of two 2D line segments.

    Determines whether the segment P1→P2 intersects with the segment Q1→Q2.
    The computation is performed using a parametric form and 2D cross products.
    If a unique intersection point exists within both segment bounds
    (including small numerical tolerances), the intersection coordinates are
    returned together with a success flag.

    Parallel or colinear segments are treated as non-intersecting for the
    purposes of the Image Source Model, since such cases are ambiguous for
    specular reflection path construction.

    Parameters:
    - x1 (float): <br>
        x-coordinate of the first endpoint of segment P.
    - y1 (float): <br>
        y-coordinate of the first endpoint of segment P.
    - x2 (float): <br>
        x-coordinate of the second endpoint of segment P.
    - y2 (float): <br>
        y-coordinate of the second endpoint of segment P.
    - x3 (float): <br>
        x-coordinate of the first endpoint of segment Q.
    - y3 (float): <br>
        y-coordinate of the first endpoint of segment Q.
    - x4 (float): <br>
        x-coordinate of the second endpoint of segment Q.
    - y4 (float): <br>
        y-coordinate of the second endpoint of segment Q.
    - eps (float): <br>
        Numerical tolerance used to handle floating point inaccuracies
        when testing for parallelism and segment bounds.

    Returns:
    - Tuple[float, float, bool]: <br>
        (ix, iy, ok) where (ix, iy) is the intersection point if one exists,
        and `ok` indicates whether a valid intersection was found.
    """
    # r = p1->p2, s = q1->q2
    rx = x2 - x1
    ry = y2 - y1
    sx = x4 - x3
    sy = y4 - y3

    rxs = rx * sy - ry * sx
    qpx = x3 - x1
    qpy = y3 - y1
    qpxr = qpx * ry - qpy * rx

    if abs(rxs) <= eps:
        # parallel or colinear -> treat as invalid for your ISM
        return 0.0, 0.0, False

    t = (qpx * sy - qpy * sx) / rxs
    u = (qpx * ry - qpy * rx) / rxs

    if (-eps <= t <= 1.0 + eps) and (-eps <= u <= 1.0 + eps):
        ix = x1 + t * rx
        iy = y1 + t * ry
        return ix, iy, True

    return 0.0, 0.0, False



# -------------------------------
# >>> Raster-based visibility <<<
# -------------------------------

@numba.njit(cache=True, fastmath=True)
def is_visible_raster(p1x:float, p1y:float, p2x:float, p2y:float, occ:np.ndarray, ignore_ends:int=1) -> bool:
    """
    Check line-of-sight visibility between two points using a raster occlusion map.

    This function traces a discrete line between two pixel positions using
    Bresenham's line algorithm and tests whether any raster cell along this
    line is marked as occluded in the provided occlusion map `occ`.
    The algorithm operates entirely in integer pixel space and is written in
    a Numba-friendly style without temporary allocations.

    To allow valid reflection paths that touch wall endpoints, a configurable
    number of pixels at the start and end of the line can be ignored via
    `ignore_ends`. This prevents false occlusion detections when rays start
    or end directly on wall pixels.

    The Bresenham traversal is executed twice:
    1. First pass: counts the number of raster points along the line to
    determine the valid range after removing the ignored endpoints.
    2. Second pass: checks only the relevant interior pixels for occlusion.

    Taken from https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm -> last code shown on the website.

    Parameters:
    - p1x (float): <br>
        x-coordinate of the start point in pixel space.
    - p1y (float): <br>
        y-coordinate of the start point in pixel space.
    - p2x (float): <br>
        x-coordinate of the end point in pixel space.
    - p2y (float): <br>
        y-coordinate of the end point in pixel space.
    - occ (np.ndarray): <br>
        Occlusion map where nonzero values represent blocked pixels.
        Expected indexing is `occ[y, x]`.
    - ignore_ends (int): <br>
        Number of raster pixels to ignore at both ends of the line.

    Returns:
    - bool: <br>
        True if the path between the two points is free of occlusions,
        otherwise False.
    """
    H = occ.shape[0]
    W = occ.shape[1]

    # round always up -> so + 0.5
    x0 = int(p1x + 0.5)
    y0 = int(p1y + 0.5)
    x1 = int(p2x + 0.5)
    y1 = int(p2y + 0.5)

    # clamp endpoints -> should be inside
    if x0 < 0: x0 = 0
    if x0 >= W: x0 = W - 1
    if y0 < 0: y0 = 0
    if y0 >= H: y0 = H - 1
    if x1 < 0: x1 = 0
    if x1 >= W: x1 = W - 1
    if y1 < 0: y1 = 0
    if y1 >= H: y1 = H - 1

    # preperation for breseham algorithm
    # err -> accumulator to decide to go in x or y direction
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy

    # 1. calc length (amount of grid points -> for ignore ends
    # => count grid points between the 2 points
    # => later we can then say ignore the first and last N pixels
    tx, ty = x0, y0
    terr = err
    npts = 1
    while not (tx == x1 and ty == y1):
        # bresenham algorithm to move in pixel-space
        e2 = terr * 2
        if e2 > -dy:
            terr -= dy
            tx += sx
        if e2 < dx:
            terr += dx
            ty += sy
        # increase grid points
        npts += 1

    # not enough grid points -> visible!
    # only start and end point, there cant be something in between
    if npts <= 2:
        return True

    # if the ignore parts is cut away
    # is there even something left to check?
    start_i = ignore_ends
    end_i = npts - ignore_ends
    if start_i >= end_i:
        return True
    
    # 2. run again -> but this time check occlusion
    tx, ty = x0, y0
    terr = err
    i = 0
    while True:
        if i >= start_i and i < end_i:
            # wall between detected?
            if occ[ty, tx] != 0:
                return False

        # end of line reached?
        if tx == x1 and ty == y1:
            break

        # breseham -> continue line
        e2 = terr * 2
        if e2 > -dy:
            terr -= dy
            tx += sx
        if e2 < dx:
            terr += dx
            ty += sy

        i += 1

    return True


# -------------------------------------
# >>> Wall extraction from an image <<<
# -------------------------------------

@dataclass(frozen=True)
class Segment:
    """
    Represent a 2D line segment.

    A lightweight immutable segment representation used for wall geometry
    and intersection tests.

    Attributes:
    - ax (float): x-coordinate of the first endpoint.
    - ay (float): y-coordinate of the first endpoint.
    - bx (float): x-coordinate of the second endpoint.
    - by (float): y-coordinate of the second endpoint.

    Properties:
    - A (Tuple[float, float]): <br>
        First endpoint (ax, ay).
    - B (Tuple[float, float]): <br>
        Second endpoint (bx, by).
    """
    ax: float
    ay: float
    bx: float
    by: float

    @property
    def A(self): return (self.ax, self.ay)

    @property
    def B(self): return (self.bx, self.by)



@numba.njit(cache=True, fastmath=True)
def _max_u8(img_u8):
    m = 0
    H, W = img_u8.shape[0], img_u8.shape[1]
    for y in range(H):
        for x in range(W):
            v = int(img_u8[y, x])
            if v > m:
                m = v
    return m



@numba.njit(cache=True, fastmath=True)
def build_wall_mask_numba_no_values(img_u8):
    """
    Build a 0/255 wall mask from a uint8 image when no explicit wall values are given.

    This function assumes that the input image is already mask-like. If the
    maximum pixel value is small (e.g. < 64), the image is treated as a
    binary or low-range mask and scaled to the range {0, 255}. Otherwise,
    the values are copied directly, assuming the image already represents
    a proper wall mask.

    The implementation is written in a Numba-friendly style and operates
    purely with explicit loops for maximum compatibility and performance.

    Parameters:
    - img_u8 (np.ndarray): <br>
        2D uint8 image interpreted as a mask-like input.

    Returns:
    - np.ndarray: <br>
        A uint8 wall mask with values 0 (free space) and 255 (wall).
    """
    # img_u8: uint8 2D
    H, W = img_u8.shape[0], img_u8.shape[1]
    m = _max_u8(img_u8)
    out = np.empty((H, W), dtype=np.uint8)

    if m < 64:
        # treat as 0/1-ish mask -> scale to 0/255
        for y in range(H):
            for x in range(W):
                out[y, x] = 255 if img_u8[y, x] != 0 else 0
    else:
        # already 0/255 or label-like uint8 -> just copy
        for y in range(H):
            for x in range(W):
                out[y, x] = img_u8[y, x]
    return out



@numba.njit(cache=True, fastmath=True)
def build_wall_mask_numba_values(img_u8, wall_values_i32):
    """
    Build a 0/255 wall mask from a uint8 image using explicit wall label values.

    Each pixel in the input image is compared against a list of wall label
    values. If a pixel matches any of the provided values, it is marked as
    a wall (255), otherwise it is marked as free space (0).

    This replaces the typical `np.isin` operation with explicit loops to
    remain fully compatible with Numba's nopython mode.

    Parameters:
    - img_u8 (np.ndarray): <br>
        2D uint8 image containing label values.
    - wall_values_i32 (np.ndarray): <br>
        1D array of int32 values that should be interpreted as walls.

    Returns:
    - np.ndarray: <br>
        A uint8 wall mask with values 0 (free space) and 255 (wall).
    """
    # img_u8: uint8 2D
    # wall_values_i32: int32 1D, e.g. np.array([3,7,9], np.int32)
    H, W = img_u8.shape[0], img_u8.shape[1]
    out = np.empty((H, W), dtype=np.uint8)
    nvals = wall_values_i32.shape[0]

    for y in range(H):
        for x in range(W):
            v = int(img_u8[y, x])
            is_wall = False
            for k in range(nvals):
                if v == int(wall_values_i32[k]):
                    is_wall = True
                    break
            out[y, x] = 255 if is_wall else 0
    return out



def build_wall_mask(img: np.ndarray, wall_values=None) -> np.ndarray:
    """
    Build a 0/255 wall mask from an input image.

    This function acts as a Python wrapper that prepares the input data and
    delegates the actual computation to Numba-optimized kernels.

    If `wall_values` is None, the image is assumed to be mask-like and
    processed accordingly. Otherwise, the provided label values are used
    to determine which pixels represent walls.

    The input image must be two-dimensional. If it is not of type uint8,
    it is converted before processing.

    Parameters:
    - img (np.ndarray): <br>
        2D input image representing either a mask-like image or a label map.
    - wall_values (optional): <br>
        Iterable of label values that should be interpreted as walls.

    Returns:
    - np.ndarray: <br>
        A uint8 wall mask with values 0 (free space) and 255 (wall).
    """
    if img.ndim != 2:
        raise ValueError("build_wall_mask: erwartet 2D img (H,W). RGB vorher konvertieren.")

    if img.dtype != np.uint8:
        img_u8 = img.astype(np.uint8)
    else:
        img_u8 = img

    if wall_values is None:
        return build_wall_mask_numba_no_values(img_u8)

    wall_vals = np.asarray(wall_values, dtype=np.int32)
    return build_wall_mask_numba_values(img_u8, wall_vals)



def get_wall_segments_from_mask(mask_255: np.ndarray, thickness: int = 1, approx_epsilon: float = 1.5) -> List[Segment]:
    """
    Extract wall boundary segments from a binary wall mask.

    Runs edge detection (Canny) and contour extraction to find wall boundaries,
    then approximates contours to polylines and converts edges into Segment
    instances.

    Parameters:
    - mask_255 (np.ndarray): <br>
        Binary wall mask with values 0 and 255.
    - thickness (int): <br>
        Optional dilation thickness applied to edges before contour extraction.
        Values > 1 thicken edges to improve contour continuity.
    - approx_epsilon (float): <br>
        Epsilon value for contour polygon approximation (cv2.approxPolyDP).
        Higher values simplify contours more aggressively.

    Returns:
    - List[Segment]: <br>
        List of wall boundary segments in pixel coordinates.
    """
    edges = cv2.Canny(mask_255, 100, 200)
    if thickness and thickness > 1:
        k = np.ones((thickness, thickness), np.uint8)
        edges = cv2.dilate(edges, k, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    walls: List[Segment] = []
    for c in contours:
        c2 = cv2.approxPolyDP(c, epsilon=approx_epsilon, closed=True)
        pts = [tuple(p[0].astype(float)) for p in c2]
        if len(pts) < 2:
            continue

        # segments along polyline
        for i in range(len(pts) - 1):
            (x1, y1), (x2, y2) = pts[i], pts[i + 1]
            if x1 == x2 and y1 == y2:
                continue
            walls.append(Segment(x1, y1, x2, y2))

        # close loop
        if len(pts) >= 3:
            (x1, y1), (x2, y2) = pts[-1], pts[0]
            if x1 != x2 or y1 != y2:
                walls.append(Segment(x1, y1, x2, y2))

    return walls



def segments_to_walls4(walls):
    """
    Convert a list of wall segments into a compact float32 array representation.

    Transforms a Python list of `Segment` objects (with endpoints ax/ay and bx/by)
    into a contiguous NumPy array of shape (N, 4), where each row encodes one wall
    segment as:

        [ax, ay, bx, by]

    This representation is convenient for passing wall geometry into Numba-compiled
    kernels, which cannot efficiently work with Python objects or dataclasses.

    Parameters:
    - walls (List[Segment]): <br>
        A list of wall segments represented as `Segment` objects.

    Returns:
    - np.ndarray: <br>
        A float32 array of shape (N, 4) containing wall endpoints.
    """
    walls4 = np.empty((len(walls), 4), dtype=np.float32)

    for i, w in enumerate(walls):
        walls4[i,0] = w.ax 
        walls4[i,1] = w.ay
        walls4[i,2] = w.bx
        walls4[i,3] = w.by

    return walls4



@numba.njit(cache=True, fastmath=True)
def dilate_binary_square(src, k):
    """
    Perform binary dilation using a square structuring element.

    Dilates a binary image `src` (values 0/1) using a kxk square kernel for a
    single iteration. A pixel in the output is set to 1 if any pixel in its
    kxk neighborhood in the input is nonzero.

    This implementation is written in a Numba-friendly style (explicit loops,
    no OpenCV dependency) and is useful for thickening walls in an occlusion map.

    Parameters:
    - src (np.ndarray): <br>
        2D uint8 array containing binary values {0, 1}.
    - k (int): <br>
        Kernel size (wall thickness). Values <= 1 return a copy of `src`.

    Returns:
    - np.ndarray: <br>
        A uint8 binary array (0/1) of the same shape as `src` after dilation.
    """
    # src: uint8 0/1
    # k: wall_thickness (>=1)
    if k <= 1:
        return src.copy()

    H, W = src.shape[0], src.shape[1]
    r = k // 2
    out = np.zeros((H, W), dtype=np.uint8)

    for y in range(H):
        y0 = y - r
        y1 = y + r
        if y0 < 0: y0 = 0
        if y1 >= H: y1 = H - 1

        for x in range(W):
            x0 = x - r
            x1 = x + r
            if x0 < 0: x0 = 0
            if x1 >= W: x1 = W - 1

            # if any neighbor is 1 -> out = 1
            found = 0
            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    if src[yy, xx] != 0:
                        found = 1
                        break
                if found == 1:
                    break
            out[y, x] = found
    return out



@numba.njit(cache=True, fastmath=True)
def apply_mask_to_binary(mask_255):
    """
    Convert a 0/255 wall mask into a binary occlusion map.

    Creates a binary occlusion map from a wall mask where nonzero pixels indicate
    walls. The output contains 0 for free space and 1 for occluded pixels.

    This function is implemented using explicit loops for Numba compatibility.

    Parameters:
    - mask_255 (np.ndarray): <br>
        2D uint8 wall mask, typically with values {0, 255}.

    Returns:
    - np.ndarray: <br>
        A uint8 binary array of shape (H, W) with values:
        - 0 for free space
        - 1 for occluded (wall) pixels
    """
    H, W = mask_255.shape[0], mask_255.shape[1]
    occ = np.empty((H, W), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            occ[y, x] = 1 if mask_255[y, x] != 0 else 0
    return occ



@numba.njit(cache=True, fastmath=True)
def build_occlusion_from_wallmask(mask_255, wall_thickness=1):
    """
    Build a binary occlusion map from a 0/255 wall mask, with optional wall dilation.

    First converts the input wall mask (0/255) into a binary occlusion map (0/1).
    If `wall_thickness` is greater than 1, the binary map is dilated using a
    square kernel to increase the effective wall thickness for subsequent
    visibility checks.

    All computation is performed in Numba-compatible kernels without relying
    on OpenCV operations.

    Parameters:
    - mask_255 (np.ndarray): <br>
        2D uint8 wall mask, typically with values {0, 255}.
    - wall_thickness (int): <br>
        If > 1, applies a single binary dilation step with a square kernel of size
        (wall_thickness x wall_thickness).

    Returns:
    - np.ndarray: <br>
        A uint8 binary occlusion map with values:
        - 0 for free space
        - 1 for occluded (wall) pixels
    """
    occ = apply_mask_to_binary(mask_255)
    if wall_thickness > 1:
        occ = dilate_binary_square(occ, wall_thickness)
    return occ



# -----------------------------------------
# >>> ISM sequence enumeration & precompute
# -----------------------------------------

def enumerate_wall_sequences_indices(n_walls: int, max_order: int, forbid_immediate_repeat: bool = True) -> List[Tuple[int, ...]]:
    """
    Enumerate reflection sequences over wall indices up to a given order.

    Generates all sequences of wall indices representing reflection orders
    from 0 up to `max_order`. The empty sequence () represents the direct path.

    Parameters:
    - n_walls (int): <br>
        Number of available wall segments.
    - max_order (int): <br>
        Maximum reflection order (length of sequence).
    - forbid_immediate_repeat (bool): <br>
        If True, prevents sequences with the same wall repeated consecutively,
        e.g., (..., 3, 3) is disallowed.

    Returns:
    - List[Tuple[int, ...]]: <br>
        List of sequences, each a tuple of wall indices.
    """
    seqs = [()]
    for _ in range(max_order):
        new = []
        for seq in seqs:
            for wi in range(n_walls):
                if forbid_immediate_repeat and len(seq) > 0 and seq[-1] == wi:
                    continue
                new.append(seq + (wi,))
        seqs += new
    return seqs


def precompute_image_sources(
    source_xy: Tuple[float, float],
    walls: List[Segment],
    max_order: int,
    forbid_immediate_repeat: bool = True,
    max_candidates: Optional[int] = None,
) -> List[Tuple[Tuple[int, ...], Tuple[float, float]]]:
    """
    Precompute image source positions for reflection sequences.

    For every reflection sequence up to `max_order`, the source point is
    reflected across the corresponding wall lines to produce an image source
    position S_img.

    Parameters:
    - source_xy (Tuple[float, float]): <br>
        Source position in pixel coordinates (x, y).
    - walls (List[Segment]): <br>
        Wall segments used for reflection.
    - max_order (int): <br>
        Maximum reflection order to consider.
    - forbid_immediate_repeat (bool): <br>
        If True, prevents immediate repetition of the same wall in sequences.
    - max_candidates (Optional[int]): <br>
        If provided, truncates the generated sequence list to at most this many
        candidates (useful as a speed cap).

    Returns:
    - List[Tuple[Tuple[int, ...], Tuple[float, float]]]: <br>
        A list of tuples (seq, S_img) where: 
        - seq is the wall-index sequence
        - S_img is the resulting image source position
    """
    seqs = enumerate_wall_sequences_indices(len(walls), max_order, forbid_immediate_repeat)
    if max_candidates is not None and len(seqs) > max_candidates:
        seqs = seqs[:max_candidates]

    pre: List[Tuple[Tuple[int, ...], Tuple[float, float]]] = []
    for seq in seqs:
        S_img = source_xy
        for wi in seq:
            w = walls[wi]
            S_img = reflect_point_across_infinite_line(S_img, w.A, w.B)
        pre.append((seq, S_img))
    return pre



# ----------------------------------
# >>> ISM path building (no shapely)
# ----------------------------------

@numba.njit(cache=True, fastmath=True)
def build_path_for_sequence(
        sx, sy,
        rx, ry,
        seq, seq_len,
        s_imgx, s_imgy,
        walls4,
        path_out
    ):
    """
    Construct a valid specular reflection path for a given wall sequence.

    Builds the geometric reflection path between a real source position and
    a receiver position using a precomputed image source and a sequence of
    wall indices describing the reflection order.

    The algorithm works backwards from the receiver by repeatedly:
    1. Intersecting the line from the image source to the current virtual
    receiver with the corresponding wall segment.
    2. Reflecting the virtual receiver across that wall line.

    This produces the reflection points in reverse order, which are then
    reversed in-place to form the final path:

        [source, r1, r2, ..., receiver]

    The function is designed for Numba and uses preallocated memory
    (`path_out`) instead of dynamic Python lists.

    Parameters:
    - sx (float): <br>
        x-coordinate of the real source position.
    - sy (float): <br>
        y-coordinate of the real source position.
    - rx (float): <br>
        x-coordinate of the receiver position.
    - ry (float): <br>
        y-coordinate of the receiver position.
    - seq (np.ndarray): <br>
        1D int array of wall indices describing the reflection order.
    - seq_len (int): <br>
        Length of the wall index sequence.
    - s_imgx (float): <br>
        x-coordinate of the precomputed image source for this sequence.
    - s_imgy (float): <br>
        y-coordinate of the precomputed image source for this sequence.
    - walls4 (np.ndarray): <br>
        Array of shape (N,4) containing wall segments as
        [ax, ay, bx, by].
    - path_out (np.ndarray): <br>
        Preallocated float array of shape (>= seq_len+2, 2) used to store
        the resulting path points.

    Returns:
    - Tuple[int, bool]: <br>
        (path_length, ok) where `path_length` is the number of valid points
        written into `path_out`, and `ok` indicates whether a valid
        reflection path could be constructed.
    """
    # seq_len == 0 => direct path
    if seq_len == 0:
        path_out[0, 0] = sx
        path_out[0, 1] = sy
        path_out[1, 0] = rx
        path_out[1, 1] = ry
        return 2, True

    # We'll compute reflection points in reverse order into path_out[1:1+seq_len]
    # and then reverse them in-place.
    rvirt_x = rx
    rvirt_y = ry

    # compute reflection points in reverse order
    for k in range(seq_len - 1, -1, -1):
        wi = seq[k]
        ax = walls4[wi, 0]
        ay = walls4[wi, 1]
        bx = walls4[wi, 2]
        by = walls4[wi, 3]

        # intersect segment (S_img -> R_virtual) with wall segment
        ix, iy, ok = seg_seg_intersection_xy(
            s_imgx, s_imgy, rvirt_x, rvirt_y,
            ax, ay, bx, by
        )
        if not ok:
            return 0, False

        # store in reverse order (later we reverse)
        path_out[1 + (seq_len - 1 - k), 0] = ix
        path_out[1 + (seq_len - 1 - k), 1] = iy

        # update virtual receiver by reflecting across wall line
        rvirt_x, rvirt_y = reflect_point_across_infinite_line_numba(
            rvirt_x, rvirt_y, ax, ay, bx, by
        )

    # reverse the reflection points to get forward order
    i = 1
    j = seq_len
    while i < j:
        tmpx = path_out[i, 0]
        tmpy = path_out[i, 1]
        path_out[i, 0] = path_out[j, 0]
        path_out[i, 1] = path_out[j, 1]
        path_out[j, 0] = tmpx
        path_out[j, 1] = tmpy
        i += 1
        j -= 1

    # write endpoints
    path_out[0, 0] = sx
    path_out[0, 1] = sy
    path_out[seq_len + 1, 0] = rx
    path_out[seq_len + 1, 1] = ry

    return seq_len + 2, True



@numba.njit(cache=True, fastmath=True)
def check_path_visibility_raster(points_xy: List[Tuple[float, float]], occ: np.ndarray, ignore_ends: int = 1) -> bool:
    """
    Check visibility of a multi-segment path using a raster occlusion map.

    Iterates over consecutive point pairs in a path and verifies that each
    segment is visible using a raster-based line-of-sight test
    (`is_visible_raster`). If any segment is occluded, the entire path is
    considered invalid.

    This function is intended to be used with paths produced by
    `build_path_for_sequence`, where the path is represented as an ordered
    list of 2D points.

    Parameters:
    - points_xy (List[Tuple[float, float]]): <br>
        Ordered list of path points in pixel coordinates
        [p0, p1, ..., pn].
    - occ (np.ndarray): <br>
        Binary occlusion map where nonzero values indicate blocked pixels.
        Accessed via `occ[y, x]`.
    - ignore_ends (int): <br>
        Number of pixels to ignore at both ends of each segment during
        the visibility check, allowing rays to touch wall endpoints.

    Returns:
    - bool: <br>
        True if all path segments are visible, otherwise False.
    """
    n = len(points_xy)

    # early stop = not enough points
    if n <= 1:
        return True
    
    # go through every point +
    # check if the point and the next point
    # are visible or if there is a wall between
    for i in range(n-1):
        p1x = points_xy[i][0]
        p1y = points_xy[i][1]
        p2x = points_xy[i+1][0]
        p2y = points_xy[i+1][1]

        if not is_visible_raster(p1x, p1y, p2x, p2y, occ, ignore_ends):
            return False
    return True



# -------------------------------
# >>> Main: noise / hit maps
# -------------------------------

def compute_reflection_map(
    source_rel: Tuple[float, float],
    img: np.ndarray,
    wall_values,
    wall_thickness: int = 1,
    approx_epsilon: float = 1.5,
    max_order: int = 1,
    ignore_zero_order=False,
    step_px: int = 8,
    forbid_immediate_repeat: bool = True,
    max_candidates: Optional[int] = None,
    ignore_ends: int = 1,
    iterative_tracking=False,
    iterative_steps=None,
    parallelization: int = -1
):
    """
    Compute an ISM-based propagation map using fast raster visibility checks.

    Builds wall geometry from an image, precomputes image sources for reflection
    sequences up to `max_order`, and evaluates valid paths from a source position
    to a grid of receiver points.

    Parameters:
    - source_rel (Tuple[float, float]): <br>
        Source position in relative coordinates (sx, sy) in [0, 1], scaled by (W, H).
    - img (np.ndarray): <br>
        Input image describing the environment. Can be (H, W) or (H, W, C).
        Typically a label map or a wall mask source.
    - wall_values: <br>
        Label values that indicate walls. If None, `img` is treated as mask-like.
    - wall_thickness (int): <br>
        Wall thickening used both for edge extraction and occlusion dilation.
    - approx_epsilon (float): <br>
        Polygon approximation epsilon for contour simplification.
    - max_order (int): <br>
        Maximum number of reflections considered (reflection order).
    - ignore_zero_order (bool):  <br>
        Whether to ignore the zero order of reflections.
    - step_px (int): <br>
        Receiver grid stride in pixels. Larger values are faster but coarser.
    - iterative_tracking (bool): <br>
        Whether to calculate multiple timesteps.
    - iterative_steps (int): <br>
        How many timesteps are wanted. -1 and None means all timesteps.
    - forbid_immediate_repeat (bool): <br>
        If True, disallows consecutive reflection on the same wall index.
    - max_candidates (Optional[int]): <br>
        Optional cap on the number of reflection sequences evaluated.
    - ignore_ends (int): <br>
        Ignore N pixels at each end of visibility checks to allow endpoint contact.
    - parallelization (int): <br>
        If nonzero, uses joblib Parallel with n_jobs=parallelization.

    Returns:
    - Tuple[np.ndarray, Optional[np.ndarray]]: <br>
        (count_map, None) <br>
        The map is a float32 array of shape (H, W).
    """
    # input handling
    if img.ndim == 3:
        # if its RGB, convert to grayscale
        img = img[..., 0]

    H, W = img.shape[:2]
    sx = source_rel[0] * W
    sy = source_rel[1] * H

    # calc walls + occlusion map
    wall_mask_255 = build_wall_mask(img, wall_values=wall_values)
    walls = get_wall_segments_from_mask(wall_mask_255, thickness=wall_thickness, approx_epsilon=approx_epsilon)
    occ = build_occlusion_from_wallmask(wall_mask_255, wall_thickness=wall_thickness)

    # transform walls into float32 -> n_walls, 4
    walls4 = segments_to_walls4(walls)

    # precompute sequences + image sources once
    # in python not numba
    pre_py = precompute_image_sources(
        source_xy=(sx, sy),
        walls=walls,
        max_order=max_order,
        forbid_immediate_repeat=forbid_immediate_repeat,
        max_candidates=max_candidates,
    )

    # transform seq tuple to np.int32 (for numba)
    pre = []
    for seq_tuple, S_img in pre_py:
        seq_arr = np.asarray(seq_tuple, dtype=np.int32)
        pre.append((seq_arr, int(seq_arr.shape[0]), float(S_img[0]), float(S_img[1])))

    # get receiver -> in numba style -> as numpy array
    rx_list = []
    ry_list = []
    for y in range(0, H, step_px):
        for x in range(0, W, step_px):
            rx_list.append(x + 0.5)
            ry_list.append(y + 0.5)
    # get receiver grid in numba style
    receivers = np.empty((len(rx_list), 2), dtype=np.float32)
    receivers[:, 0] = np.asarray(rx_list, dtype=np.float32)
    receivers[:, 1] = np.asarray(ry_list, dtype=np.float32)

    # setup for iterative results
    # output (set iterative)
    output_map = np.zeros((H, W), dtype=np.float32)
    snapshots = []  # list of np.ndarray

    # always convert None to -1, means the same for thsi variable
    if iterative_steps is None:
        iterative_steps = -1

    if (not iterative_tracking) or (iterative_tracking and iterative_steps == 1):
        snapshot_every_x_steps = len(pre)
    elif iterative_tracking and iterative_steps == -1:
        snapshot_every_x_steps = 1
    else:
        snapshot_every_x_steps = max(1, int(len(pre) // iterative_steps))

    # worker setup
    def eval_receiver(R, pre_chunk):
        rx = float(R[0])
        ry = float(R[1])

        val = 0.0

        # path_out max len = max_order + 2 (oder seq_len + 2)
        # because seq_len is variable, we allocate once with max_order+2
        path_out = np.empty((max_order + 2, 2), dtype=np.float32)

        for (seq_arr, seq_len, s_imgx, s_imgy) in pre_chunk:
            if ignore_zero_order and seq_len == 0:
                continue

            path_len, ok = build_path_for_sequence(
                sx, sy,
                rx, ry,
                seq_arr, seq_len,
                s_imgx, s_imgy,
                walls4,
                path_out
            )
            if not ok:
                continue

            # Occlusion check (Numba), warning: slice creates view -> ok here
            if not check_path_visibility_raster(path_out[:path_len], occ, ignore_ends):
                continue

            val += 1.0

        return rx, ry, val

    # main loop -> start worker/compuation
    # compute -> optionally iterativly
    for start_idx in range(0, len(pre), snapshot_every_x_steps):
        pre_chunk = pre[start_idx:start_idx + snapshot_every_x_steps]

        if parallelization and parallelization != 0:
            results = Parallel(n_jobs=parallelization, prefer="threads", batch_size=256)(
                delayed(eval_receiver)(cur_receiver, pre_chunk) for cur_receiver in receivers
            )
        else:
            results = [eval_receiver(cur_receiver, pre_chunk) for cur_receiver in receivers]

        # summarize current chunk to final output map

        # process results
        for x, y, v in results:
            ix, iy = int(x), int(y)
            if 0 <= ix < W and 0 <= iy < H:
                output_map[iy, ix] += float(v)

        snapshots.append(output_map.copy())

    # return output
    if not iterative_tracking:  # len(snapshots) <= 1
        return output_map
    else:
        return snapshots  # [::-1]






