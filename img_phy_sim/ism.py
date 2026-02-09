"""
**Image Source Method (ISM) Ray Propagation and Visibility Maps**

This module implements a fast 2D Image Source Method (ISM) solver to simulate
specular reflections and line-of-sight propagation inside environments described
by images (raster maps) that contain wall/obstacle pixels.

The scene is interpreted as a 2D grid where specific pixel values represent
walls. From this raster representation, geometric wall boundary segments are
extracted and used to construct image sources for reflection sequences up to a
given order. For each receiver position on a grid, candidate reflection paths
are constructed, validated with a raster-based visibility test, and accumulated
into a path-count map.

Core pipeline:
1. Build a wall mask from an input image
2. Extract wall boundary segments (OpenCV contours -> polyline edges)
3. Build a binary occlusion map for raster visibility tests
4. Enumerate reflection sequences and precompute image source positions
5. Evaluate receiver points on a grid:
   - Construct candidate reflection paths (segment intersection + reflection)
   - Validate segment visibility against the occlusion raster
   - Accumulate valid paths into an output map

As ASCII model:
```text
                 ┌──────────────────────────────┐
                 │        Input: Scene          │
                 │    (image, walls, source)    │
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
                 │     Check path visibility    │
                 │    (raster occlusion test)   │
                 └──────────────┬───────────────┘
                                │
                                v
                 ┌──────────────────────────────┐
                 │  Accumulate contribution     │
                 └──────────────┬───────────────┘
                                │
                                v
                 ┌──────────────────────────────┐
                 │          Output Map          │
                 └──────────────────────────────┘
```

<br><br>

Numba/parallelization notes:<br>
The hot loop is executed inside Numba kernels.

- `eval_receivers_to_map_kernel`:<br>
  Computes and writes directly into `output_map` inside Numba. This is safe when
  receiver coordinates are unique (e.g., a regular grid where each receiver maps to
  a distinct pixel).

Reflection sequences are packed into flat arrays (`seq_data`, `seq_off`,
``seq_len``) to avoid Python containers and to keep Numba in nopython mode.

Example:
```python
reflection_map = ips.ism.compute_reflection_map(
    source_rel=(0.5, 0.5),
    img=input_,
    wall_values=[0],   
    wall_thickness=1,
    max_order=1,
    step_px=1
)
```

Dependencies:
- numpy
- OpenCV (cv2)
- numba


Public API:
- compute_reflection_map(...)
- build_wall_mask(...)
- get_wall_segments_from_mask(...)
- build_occlusion_from_wallmask(...)
- precompute_image_sources(...)
- enumerate_wall_sequences_indices(...)

Main internal kernels/utilities:
- reflect_point_across_infinite_line(...)
- reflect_point_across_infinite_line_numba(...)
- seg_seg_intersection_xy(...)
- is_visible_raster(...)
- build_path_for_sequence_packed(...)
- check_path_visibility_raster_arr(...)
- pack_precomputed(...)
- build_receivers_grid(...)
- eval_receivers_to_map_kernel(...)
"""



# ---------------
# >>> Imports <<<
# ---------------
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2

# optimization
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
    # direction vector of the wall segment A -> B
    abx = bx - ax
    aby = by - ay

    # normalize to unit direction vector
    # small epsilon avoids division by zero for degenerate segments
    n = math.sqrt(abx * abx + aby * aby) + 1e-12
    ux = abx / n
    uy = aby / n

    # vector from A to the point P
    apx = px - ax
    apy = py - ay

    # projection of AP onto the wall direction
    t = apx * ux + apy * uy
    projx = ax + t * ux
    projy = ay + t * uy

    # reflect point across the infinite line defined by A-B
    # P' = 2 * projection - P
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
    # r = segment p1 -> p2
    rx = x2 - x1
    ry = y2 - y1

    # s = segment q1 -> q2
    sx = x4 - x3
    sy = y4 - y3

    # cross product r x s used to detect parallelism
    rxs = rx * sy - ry * sx

    # vector from p1 to q1
    qpx = x3 - x1
    qpy = y3 - y1

    # cross product (q1 - p1) x r
    qpxr = qpx * ry - qpy * rx

    # if r x s is near zero the segments are parallel or colinear
    # for the reflection model this is treated as an invalid intersection
    if abs(rxs) <= eps:
        # parallel or colinear -> treat as invalid for your ISM
        return 0.0, 0.0, False

    # solve intersection parameters for the parametric segment equations
    t = (qpx * sy - qpy * sx) / rxs
    u = (qpx * ry - qpy * rx) / rxs

    # check whether intersection lies within both segments (with epsilon tolerance)
    if (-eps <= t <= 1.0 + eps) and (-eps <= u <= 1.0 + eps):
        ix = x1 + t * rx
        iy = y1 + t * ry
        return ix, iy, True

    # no valid intersection on the finite segments
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
    """
    Compute the maximum uint8 value in a 2D image using explicit loops

    This helper avoids NumPy reductions so it can be used inside
    Numba-compatible code paths

    Parameters:
    - img_u8 (np.ndarray): <br>
        2D uint8 image

    Returns:
    - int: <br>
        Maximum pixel value found in the image
    """
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

    # determine maximum value in the image to infer its encoding
    m = _max_u8(img_u8)

    out = np.empty((H, W), dtype=np.uint8)

    if m < 64:
        # image likely represents a 0/1 style mask
        # scale any non-zero value up to 255 for consistency
        for y in range(H):
            for x in range(W):
                out[y, x] = 255 if img_u8[y, x] != 0 else 0
    else:
        # image already looks like a proper uint8 mask or label image
        # simply copy values over
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
    # number of wall values to compare against
    nvals = wall_values_i32.shape[0]

    for y in range(H):
        for x in range(W):
            # current pixel value
            v = int(img_u8[y, x])

            # check if this pixel matches any of the wall values
            is_wall = False
            for k in range(nvals):
                if v == int(wall_values_i32[k]):
                    is_wall = True
                    break

            # write wall mask pixel
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

    # if no explicit wall values are given, fall back to a default wall detection routine
    if wall_values is None:
        return build_wall_mask_numba_no_values(img_u8)

    # convert wall values to a compact int array for numba-friendly lookup
    wall_vals = np.asarray(wall_values, dtype=np.int32)

    # build wall mask by checking pixels against the provided wall values
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
    # detect wall boundaries from the binary mask using Canny edge detection
    edges = cv2.Canny(mask_255, 100, 200)

    # optionally thicken edges to merge small gaps and stabilize contour extraction
    if thickness and thickness > 1:
        k = np.ones((thickness, thickness), np.uint8)
        edges = cv2.dilate(edges, k, iterations=1)

    # extract external contours of the edge map
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    walls: List[Segment] = []
    for c in contours:
        # simplify contour to a polygon with fewer points
        c2 = cv2.approxPolyDP(c, epsilon=approx_epsilon, closed=True)

        # convert polygon points to float tuples (x, y)
        pts = [tuple(p[0].astype(float)) for p in c2]
        if len(pts) < 2:
            continue

        # create segments along the polygon polyline
        for i in range(len(pts) - 1):
            (x1, y1), (x2, y2) = pts[i], pts[i + 1]

            # skip degenerate segments
            if x1 == x2 and y1 == y2:
                continue
            walls.append(Segment(x1, y1, x2, y2))

        # close the loop by connecting the last point back to the first
        # only do this if we have a valid polygon-like shape
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

    # for thickness 1 nothing to do, just return a copy
    if k <= 1:
        return src.copy()

    H, W = src.shape[0], src.shape[1]

    # radius of the square neighborhood around each pixel
    r = k // 2

    # output mask after dilation
    out = np.zeros((H, W), dtype=np.uint8)

    for y in range(H):
        # clamp vertical neighborhood to image bounds
        y0 = y - r
        y1 = y + r
        if y0 < 0: y0 = 0
        if y1 >= H: y1 = H - 1

        for x in range(W):
            # clamp horizontal neighborhood to image bounds
            x0 = x - r
            x1 = x + r
            if x0 < 0: x0 = 0
            if x1 >= W: x1 = W - 1

            # check if any pixel in the k x k neighborhood is set
            # this effectively performs a square dilation of the wall mask
            found = 0
            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    if src[yy, xx] != 0:
                        found = 1
                        break
                if found == 1:
                    break

            # mark output pixel as wall if any neighbor was a wall
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
    # start with the empty sequence which represents the direct path (no reflections)
    seqs = [()]

    # iteratively build sequences up to the desired reflection order
    for _ in range(max_order):
        new = []

        # extend all previously known sequences by one additional wall index
        for seq in seqs:
            for wi in range(n_walls):

                # optionally forbid using the same wall twice in a row
                # this avoids degenerate mirror-back reflections
                if forbid_immediate_repeat and len(seq) > 0 and seq[-1] == wi:
                    continue

                # create a new sequence with one more reflection
                new.append(seq + (wi,))

        # append the newly created sequences to the master list
        seqs += new

    # contains sequences of length 0..max_order
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
    # enumerate all candidate wall index sequences up to the desired reflection order
    seqs = enumerate_wall_sequences_indices(len(walls), max_order, forbid_immediate_repeat)

    # optionally cap how many sequences we keep to limit runtime and memory
    if max_candidates is not None and len(seqs) > max_candidates:
        seqs = seqs[:max_candidates]

    # build a list of (sequence, image_source_xy) pairs
    # the image source is obtained by reflecting the real source across each wall
    # in the sequence in forward order
    pre: List[Tuple[Tuple[int, ...], Tuple[float, float]]] = []
    for seq in seqs:
        # start from the real source position
        S_img = source_xy

        # apply successive reflections across each wall line
        for wi in seq:
            w = walls[wi]
            S_img = reflect_point_across_infinite_line(S_img, w.A, w.B)

        # store the sequence and its final image source position
        pre.append((seq, S_img))

    return pre



def pack_precomputed(pre_py):
    """
    Pack Python reflection sequences into flat arrays for Numba kernels.

    The input is a list of tuples where each entry contains:<br>
        (sequence_of_wall_indices, (image_source_x, image_source_y))

    Since Numba cannot efficiently work with Python lists of variable-length
    tuples, this function converts the structure into contiguous flat arrays
    plus offset and length metadata so each sequence can be accessed quickly.

    The result allows any sequence `i` to be reconstructed as:<br>
        seq_data[ seq_off[i] : seq_off[i] + seq_len[i] ]

    Parameters:
    - pre_py (List[Tuple[Tuple[int], Tuple[float, float]]]): <br>
        Python list where each element contains a wall index sequence and
        the corresponding precomputed image source position

    Returns:
    - seq_data (np.ndarray): <br>
        Flat int32 array containing all wall indices of all sequences
        concatenated back-to-back
    - seq_off (np.ndarray): <br>
        int32 array giving the start offset of each sequence in `seq_data`
    - seq_len (np.ndarray): <br>
        int32 array giving the length of each sequence
    - simg_x (np.ndarray): <br>
        float32 array of image source x-coordinates per sequence
    - simg_y (np.ndarray): <br>
        float32 array of image source y-coordinates per sequence
    """
    n = len(pre_py)

    # allocate metadata arrays per sequence
    seq_len = np.empty(n, dtype=np.int32)
    seq_off = np.empty(n, dtype=np.int32)
    simg_x  = np.empty(n, dtype=np.float32)
    simg_y  = np.empty(n, dtype=np.float32)

    # first pass computes lengths, offsets and image source positions
    total = 0
    for i, (seq_tuple, S_img) in enumerate(pre_py):
        L = len(seq_tuple)
        seq_len[i] = L
        seq_off[i] = total
        total += L
        simg_x[i] = float(S_img[0])
        simg_y[i] = float(S_img[1])

    # allocate the flat wall-index array
    seq_data = np.empty(total, dtype=np.int32)

    # second pass writes all sequences consecutively into seq_data
    k = 0
    for (seq_tuple, _S_img) in pre_py:
        for wi in seq_tuple:
            seq_data[k] = int(wi)
            k += 1

    return seq_data, seq_off, seq_len, simg_x, simg_y



# ----------------------------------
# >>> ISM path building (no shapely)
# ----------------------------------

@numba.njit(cache=True, fastmath=True)
def build_path_for_sequence_packed(
        sx, sy, rx, ry,
        seq_data, off0, seq_len,
        s_imgx, s_imgy,
        walls4,
        path_out
    ):
    """
    Construct a specular reflection path for a packed wall-index sequence.

    Builds the geometric reflection path between a real source position and
    a receiver position using a precomputed image source and a sequence of
    wall indices describing the reflection order.

    The algorithm works backwards from the receiver by repeatedly
    1. Intersecting the line from the image source to the current virtual
      receiver with the corresponding wall segment
    2. Reflecting the virtual receiver across that wall line

    This produces reflection points in reverse order which are then reversed
    in-place to form the final path

        [source, r1, r2, ..., receiver]

    The function is designed for Numba and writes into a preallocated array
    `path_out` instead of creating Python lists.

    Parameters:
    - sx (float): <br>
        x-coordinate of the real source position
    - sy (float): <br>
        y-coordinate of the real source position
    - rx (float): <br>
        x-coordinate of the receiver position
    - ry (float): <br>
        y-coordinate of the receiver position
    - seq_data (np.ndarray): <br>
        Flat int array containing all sequences concatenated
    - off0 (int): <br>
        Start offset of this sequence inside `seq_data`
    - seq_len (int): <br>
        Number of wall indices in this sequence
    - s_imgx (float): <br>
        x-coordinate of the precomputed image source for this sequence
    - s_imgy (float): <br>
        y-coordinate of the precomputed image source for this sequence
    - walls4 (np.ndarray): <br>
        Array of shape (N,4) containing wall segments as [ax, ay, bx, by]
    - path_out (np.ndarray): <br>
        Preallocated float array of shape (>= seq_len+2, 2) used to store
        the resulting path points

    Returns:
    - Tuple[int, bool]: <br>
        (path_length, ok) where `path_length` is the number of valid points
        written into `path_out`, and `ok` indicates whether a valid path
        could be constructed
    """
    # direct path with no reflections
    if seq_len == 0:
        path_out[0, 0] = sx; path_out[0, 1] = sy
        path_out[1, 0] = rx; path_out[1, 1] = ry
        return 2, True

    # start from the real receiver as the current virtual receiver
    rvirt_x = rx
    rvirt_y = ry

    for kk in range(seq_len - 1, -1, -1):
        # wall index for this bounce
        wi = seq_data[off0 + kk]

        # wall segment endpoints
        ax = walls4[wi, 0]; ay = walls4[wi, 1]
        bx = walls4[wi, 2]; by = walls4[wi, 3]

        # intersect ray from image source to virtual receiver with the wall segment
        ix, iy, ok = seg_seg_intersection_xy(
            s_imgx, s_imgy, rvirt_x, rvirt_y,
            ax, ay, bx, by
        )
        if not ok:
            return 0, False

        # write intersection as a reflection point in reverse order
        path_out[1 + (seq_len - 1 - kk), 0] = ix
        path_out[1 + (seq_len - 1 - kk), 1] = iy

        # update the virtual receiver by reflecting it across the infinite wall line
        rvirt_x, rvirt_y = reflect_point_across_infinite_line_numba(
            rvirt_x, rvirt_y, ax, ay, bx, by
        )

    # reverse reflection points so they go from first bounce to last bounce
    i = 1
    j = seq_len
    while i < j:
        tmpx = path_out[i, 0]; tmpy = path_out[i, 1]
        path_out[i, 0] = path_out[j, 0]; path_out[i, 1] = path_out[j, 1]
        path_out[j, 0] = tmpx;          path_out[j, 1] = tmpy
        i += 1
        j -= 1

    # insert real endpoints to finalize the path
    path_out[0, 0] = sx; path_out[0, 1] = sy
    path_out[seq_len + 1, 0] = rx
    path_out[seq_len + 1, 1] = ry
    return seq_len + 2, True



@numba.njit(cache=True, fastmath=True)
def check_path_visibility_raster_arr(path_xy: np.ndarray, path_len: int, occ: np.ndarray, ignore_ends: int) -> bool:
    """
    Check whether a reconstructed reflection path is fully visible
    in a rasterized occlusion map.

    The path consists of consecutive line segments:
        [source -> r1 -> r2 -> ... -> receiver]

    Each segment is tested against the binary occlusion grid `occ`
    using a raster visibility test. If any segment intersects an
    occupied pixel (wall), the whole path is considered invalid.

    This function is designed for Numba and operates on a preallocated
    path buffer without creating Python objects.

    Parameters:
    - path_xy (np.ndarray): <br>
        Array of shape (max_order+2, 2) containing path points as (x, y).
        Only the first `path_len` entries are valid.
    - path_len (int): <br>
        Number of valid points in `path_xy` describing the path.
    - occ (np.ndarray): <br>
        2D uint8 occlusion map where non-zero values represent walls.
    - ignore_ends (int): <br>
        Number of pixels near segment endpoints to ignore during
        raster testing. Helps avoid false occlusion from grazing
        or endpoint-touching cases.

    Returns:
    - bool: <br>
        True if all path segments are visible, False otherwise.
    """
    # paths with 0 or 1 point cannot be occluded
    if path_len <= 1:
        return True
    
    # check each consecutive segment of the path
    for i in range(path_len - 1):
        p1x = path_xy[i, 0]
        p1y = path_xy[i, 1]
        p2x = path_xy[i + 1, 0]
        p2y = path_xy[i + 1, 1]

        # raster visibility test for this segment
        if not is_visible_raster(p1x, p1y, p2x, p2y, occ, ignore_ends):
            return False
        
    # all segments are visible
    return True



def build_receivers_grid(H: int, W: int, step_px: int) -> np.ndarray:
    """
    Build a sparse grid of receiver positions over the image domain.

    The receivers are placed on a regular grid with spacing `step_px`
    in both x and y directions. Each receiver is positioned at the
    pixel center using the convention (x + 0.5, y + 0.5), which matches
    how raster pixels are typically interpreted in the reflection kernel.

    The result is a flat list of 2D receiver coordinates that can be
    iterated efficiently inside Numba kernels without additional
    meshgrid logic.

    Parameters:
    - H (int): <br>
        Image height in pixels.
    - W (int): <br>
        Image width in pixels.
    - step_px (int): <br>
        Spacing between receivers in pixels. Larger values reduce the
        number of receivers and speed up computation at the cost of
        spatial resolution.

    Returns:
    - np.ndarray: <br>
        Array of shape (N, 2) with dtype float32 containing receiver
        coordinates as (x+0.5, y+0.5).
    """
    # generate x and y sample positions at pixel centers
    xs = (np.arange(0, W, step_px, dtype=np.float32) + 0.5)
    ys = (np.arange(0, H, step_px, dtype=np.float32) + 0.5)
    
    # create a full grid of receiver coordinates
    xx, yy = np.meshgrid(xs, ys)

    # pack into a flat (N,2) array for efficient iteration
    receivers = np.empty((xx.size, 2), dtype=np.float32)
    receivers[:, 0] = xx.reshape(-1)
    receivers[:, 1] = yy.reshape(-1)
    return receivers



# -------------------------------
# >>> Main: noise / hit maps
# -------------------------------

@numba.njit(cache=True, fastmath=True, parallel=True)
def eval_receivers_to_map_kernel(
    receivers: np.ndarray,      # (N,2) float32; contains x+0.5, y+0.5
    sx: float, sy: float,
    walls4: np.ndarray,         # (nWalls,4) float32
    occ: np.ndarray,            # (H,W) uint8
    max_order: int,
    ignore_zero_order: int,     # 0/1
    ignore_ends: int,
    seq_data: np.ndarray,       # (total,) int32
    seq_off: np.ndarray,        # (nSeq,)  int32  (start offset per seq)
    seq_len: np.ndarray,        # (nSeq,)  int32
    simg_x: np.ndarray,         # (nSeq,)  float32
    simg_y: np.ndarray,         # (nSeq,)  float32
    start_seq: int,
    end_seq: int,
    output_map: np.ndarray      # (H,W) float32
):
    H = output_map.shape[0]
    W = output_map.shape[1]
    nR = receivers.shape[0]

    # iterate over all receiver positions in parallel
    for ri in numba.prange(nR):
        rx = float(receivers[ri, 0])
        ry = float(receivers[ri, 1])

        # convert receiver center coordinate back to integer pixel index
        ix = int(rx)
        iy = int(ry)

        # skip if receiver lies outside the image bounds
        if ix < 0 or ix >= W or iy < 0 or iy >= H:
            continue

        # accumulate contribution for this receiver pixel
        val = 0.0

        # local buffer for reconstructed reflection path
        # size is max_order reflections + source + receiver
        path_out = np.empty((max_order + 2, 2), dtype=np.float32)

        # iterate over the subset of precomputed reflection sequences
        for si in range(start_seq, end_seq):
            L = int(seq_len[si])
            # optionally skip direct line-of-sight paths
            if ignore_zero_order == 1 and L == 0:
                continue

            # offset into the packed wall-index array for this sequence
            off0 = int(seq_off[si])

            # precomputed image source position for this wall sequence
            s_imgx = float(simg_x[si])
            s_imgy = float(simg_y[si])

            # reconstruct the geometric reflection path for this receiver
            path_len, ok = build_path_for_sequence_packed(
                sx, sy,
                rx, ry,
                seq_data, off0, L,
                s_imgx, s_imgy,
                walls4,
                path_out
            )
            if not ok:
                continue

            # check whether the path is occluded by any wall pixels
            if not check_path_visibility_raster_arr(path_out, path_len, occ, ignore_ends):
                continue

            # valid visible reflection path contributes to this pixel
            val += 1.0

        # safe to write because each receiver maps to a unique pixel
        output_map[iy, ix] += val



def compute_reflection_map(
    source_rel,
    img,
    wall_values,
    wall_thickness:int=1,
    approx_epsilon:float=1.5,
    max_order:int=1,
    ignore_zero_order=False,
    step_px:int=1,
    forbid_immediate_repeat: bool = True,
    max_candidates=None,
    ignore_ends:int=1,
    iterative_tracking=False,
    iterative_steps=None
):
    """
    Compute a 2D "reflection contribution" map over an image grid using
    precomputed specular image sources and wall reflections.

    The function:
    1. Converts the input image to grayscale (if needed)
    2. Builds a wall mask from the image based on provided wall pixel values
    3. Extracts wall segments from that mask and builds an occlusion structure
    4. Precomputes valid wall-reflection sequences (up to `max_order`) and their
       corresponding image-source positions
    5. Evaluates reflection paths from a real source to a grid of receiver points,
       accumulating per-receiver contributions into `output_map`

    Optionally, the computation can be performed in chunks and intermediate
    snapshots of the partially accumulated `output_map` can be returned
    (`iterative_tracking=True`).

    Parameters:
    - source_rel (Tuple[float, float] | np.ndarray): <br>
        Source position in *relative* image coordinates (x_rel, y_rel) in [0, 1].
        Converted internally to pixel coordinates via (x_rel*W, y_rel*H).
    - img (np.ndarray): <br>
        Input image. If 3D (H, W, C), only the first channel is used.
        The wall mask is derived from this image.
    - wall_values (Iterable[int] | np.ndarray): <br>
        Pixel values in `img` that should be treated as "wall" when building the
        wall mask (exact matching logic depends on `build_wall_mask`).
    - wall_thickness (int): <br>
        Thickness (in pixels) used when extracting wall segments and building
        the occlusion representation.
    - approx_epsilon (float): <br>
        Polygon/segment simplification tolerance used during wall extraction
        (`get_wall_segments_from_mask`).
    - max_order (int): <br>
        Maximum reflection order (number of wall bounces) to consider when
        precomputing image sources.
    - ignore_zero_order (bool): <br>
        If True, exclude the direct (0-bounce / line-of-sight) contribution.
    - step_px (int): <br>
        Receiver grid spacing in pixels. Larger values reduce runtime by
        evaluating fewer receiver positions.
    - forbid_immediate_repeat (bool): <br>
        If True, disallow reflection sequences that repeat the same wall in
        consecutive steps (helps avoid degenerate/near-degenerate paths).
    - max_candidates (int | None): <br>
        Optional cap on the number of candidate sequences/image sources kept
        during precomputation.
    - ignore_ends (int): <br>
        Number of wall endpoints to ignore during intersection tests (used to
        stabilize grazing / endpoint-hitting cases).
    - iterative_tracking (bool): <br>
        If True, compute in chunks and return intermediate `output_map` snapshots
        as a list of arrays. Useful for debugging/profiling progress.
    - iterative_steps (int | None): <br>
        Controls how many snapshots to collect when `iterative_tracking=True`:
        - None  -> treated as -1 (snapshot every chunk of size 1 sequence)
        - -1    -> snapshot after every sequence (finest granularity)
        - 1     -> snapshot only once at the end (same as full run)
        - k>1   -> snapshot roughly `k` times over the sequence list

    Returns:
    - np.ndarray | List[np.ndarray]: <br>
        If `iterative_tracking` is False: returns `output_map` (H, W) float32.<br>
        If `iterative_tracking` is True: returns a list of snapshot maps, each
        shaped (H, W) float32, showing accumulation progress.
    """
    # If we received a color image, use only the first channel (expected: walls encoded per-pixel)
    if img.ndim == 3:
        img = img[..., 0]

    # Image dimensions and source location in absolute pixel coordinates
    H, W = img.shape[:2]
    sx = float(source_rel[0] * W)
    sy = float(source_rel[1] * H)

    # Build a binary wall mask (0/255), then extract simplified wall segments
    wall_mask_255 = build_wall_mask(img, wall_values=wall_values)
    walls = get_wall_segments_from_mask(
        wall_mask_255,
        thickness=wall_thickness,
        approx_epsilon=approx_epsilon
    )

    # Precompute an occlusion structure derived from the wall mask
    occ = build_occlusion_from_wallmask(wall_mask_255, wall_thickness=wall_thickness)

    # Convert wall segments into a packed (N,4) representation [ax, ay, bx, by]
    #    -> for numba usage
    walls4 = segments_to_walls4(walls)

    # Precompute image sources + wall reflection sequences up to `max_order`
    # Each sequence describes the ordered wall indices to reflect against
    pre_py = precompute_image_sources(
        source_xy=(sx, sy),
        walls=walls,
        max_order=max_order,
        forbid_immediate_repeat=forbid_immediate_repeat,
        max_candidates=max_candidates,
    )

    # Pack variable-length sequences into flat arrays for fast/Numba-friendly access
    seq_data, seq_off, seq_len, simg_x, simg_y = pack_precomputed(pre_py)

    # Create a grid of receiver points
    receivers = build_receivers_grid(H, W, step_px)

    # Accumulated output map (same resolution as the input image)
    output_map = np.zeros((H, W), dtype=np.float32)

    # Optional progressive snapshots of accumulation -> iterative output
    if iterative_tracking:
        snapshots = []

    # Default: snapshot every sequence (finest granularity) if tracking is enabled.
    if iterative_steps is None:
        iterative_steps = -1

    nseq = len(pre_py)

    # Decide chunk size (how many sequences per kernel call) based on snapshot settings
    if (not iterative_tracking) or (iterative_tracking and iterative_steps == 1):
        # Single call covering all sequences (or one snapshot at the end)
        snapshot_every_x_steps = nseq
    elif iterative_tracking and iterative_steps == -1:
        # Snapshot after every sequence
        snapshot_every_x_steps = 1
    else:
        # Snapshot about `iterative_steps` times, compute chunk size accordingly
        snapshot_every_x_steps = max(1, int(nseq // iterative_steps))

    # Convert boolean to int for Numba kernel
    ignore_zero_order_i = 1 if ignore_zero_order else 0

    # Evaluate reflection contributions in chunks of sequences for better control over snapshots
    for start_idx in range(0, nseq, snapshot_every_x_steps):
        end_idx = min(nseq, start_idx + snapshot_every_x_steps)

        # Core kernel: for each receiver and each sequence in [start_idx, end_idx),
        # validate/specular-trace the path (including occlusion) and accumulate into output_map
        eval_receivers_to_map_kernel(
            receivers,
            sx, sy,
            walls4,
            occ,
            max_order,
            ignore_zero_order_i,
            ignore_ends,
            seq_data,
            seq_off,
            seq_len,
            simg_x,
            simg_y,
            start_idx,
            end_idx,
            output_map
        )

        # Store a copy so later updates don't mutate previous snapshots
        if iterative_tracking:
            snapshots.append(output_map.copy())

    # Return either the final map or the progressive snapshots
    if not iterative_tracking:
        return output_map
    else:
        return snapshots









