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

from .math import normalize_point

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable

import numpy as np
import cv2

try:
    from joblib import Parallel, delayed
except Exception:
    Parallel = None
    delayed = None



# --------------
# >>> Helper <<<
# --------------

def reflect_point_across_infinite_line(P: Tuple[float, float], A: Tuple[float, float], B: Tuple[float, float]) -> Tuple[float, float]:
    """
    Reflect a 2D point across an infinite line.

    Computes the mirror image of point P with respect to the infinite line
    passing through points A and B.

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
    x1, y1 = A
    x2, y2 = B
    px, py = P

    ABx = x2 - x1
    ABy = y2 - y1
    n = math.hypot(ABx, ABy) + 1e-12
    ux, uy = ABx / n, ABy / n

    APx = px - x1
    APy = py - y1

    # projection of AP onto AB unit direction
    t = APx * ux + APy * uy
    projx = x1 + t * ux
    projy = y1 + t * uy

    # reflection
    rx = projx + (projx - px)
    ry = projy + (projy - py)
    return (float(rx), float(ry))



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


def _seg_seg_intersection(p0, p1, q0, q1, eps=1e-9) -> Optional[Tuple[float, float]]:
    """
    Compute the intersection point of two 2D line segments.

    Computes the intersection point of segment p0->p1 with segment q0->q1.
    If there is exactly one intersection point (including endpoint touches),
    it returns that point. If segments do not intersect, are parallel, or
    are colinear/overlapping (ambiguous), it returns None.

    Parameters:
    - p0 (Tuple[float, float]): <br>
        Start point of the first segment.
    - p1 (Tuple[float, float]): <br>
        End point of the first segment.
    - q0 (Tuple[float, float]): <br>
        Start point of the second segment.
    - q1 (Tuple[float, float]): <br>
        End point of the second segment.
    - eps (float): <br>
        Numerical tolerance used for parallel/colinear checks and bounds.

    Returns:
    - Optional[Tuple[float, float]]: <br>
        The intersection point (x, y) if a unique intersection exists,
        otherwise None.
    """
    x1, y1 = p0
    x2, y2 = p1
    x3, y3 = q0
    x4, y4 = q1

    # Solve via cross products (parametric)
    r = (x2 - x1, y2 - y1)
    s = (x4 - x3, y4 - y3)

    rxs = r[0] * s[1] - r[1] * s[0]
    q_p = (x3 - x1, y3 - y1)
    qpxr = q_p[0] * r[1] - q_p[1] * r[0]

    if abs(rxs) <= eps:
        # parallel
        if abs(qpxr) <= eps:
            # colinear -> ambiguous for specular ISM here
            return None
        return None

    t = (q_p[0] * s[1] - q_p[1] * s[0]) / rxs
    u = (q_p[0] * r[1] - q_p[1] * r[0]) / rxs

    if -eps <= t <= 1.0 + eps and -eps <= u <= 1.0 + eps:
        ix = x1 + t * r[0]
        iy = y1 + t * r[1]
        return (float(ix), float(iy))

    return None



# -------------------------------
# >>> Raster-based visibility <<<
# -------------------------------

def _bresenham_points(x0: int, y0: int, x1: int, y1: int) -> Iterable[Tuple[int, int]]:
    """
    Generate integer pixel coordinates along a line using Bresenham's algorithm.

    Produces all grid points (x, y) visited by the Bresenham line rasterization
    algorithm between (x0, y0) and (x1, y1), inclusive.

    Parameters:
    - x0 (int): <br>
        Start x coordinate.
    - y0 (int): <br>
        Start y coordinate.
    - x1 (int): <br>
        End x coordinate.
    - y1 (int): <br>
        End y coordinate.

    Returns:
    - Iterable[Tuple[int, int]]: <br>
        An iterator over (x, y) integer points along the rasterized line.
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        yield (x, y)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def is_visible_raster(p1: Tuple[float, float], p2: Tuple[float, float], occ: np.ndarray, ignore_ends: int = 1) -> bool:
    """
    Test line-of-sight visibility between two points using a raster occlusion map.

    Uses Bresenham line traversal between p1 and p2 and checks whether any sampled
    pixel is marked as occluded in `occ`. Optionally ignores a number of pixels
    at both ends of the ray, which is useful to allow rays to touch wall endpoints.

    Parameters:
    - p1 (Tuple[float, float]): <br>
        Start point (x, y) in pixel coordinates.
    - p2 (Tuple[float, float]): <br>
        End point (x, y) in pixel coordinates.
    - occ (np.ndarray): <br>
        Occlusion map where nonzero values indicate blocked pixels.
        Expected shape is (H, W) or compatible indexing occ[y, x].
    - ignore_ends (int): <br>
        Number of pixels to ignore at both ends of the sampled line.

    Returns:
    - bool: <br>
        True if the line between p1 and p2 is not occluded, otherwise False.
    """
    H, W = occ.shape[:2]
    x0, y0 = int(round(p1[0])), int(round(p1[1]))
    x1, y1 = int(round(p2[0])), int(round(p2[1]))

    # clamp endpoints
    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))

    pts = list(_bresenham_points(x0, y0, x1, y1))
    if len(pts) <= 2:
        return True

    a = ignore_ends
    b = len(pts) - ignore_ends
    if a >= b:
        return True

    for (x, y) in pts[a:b]:
        if occ[y, x] != 0:
            return False
    return True


# -------------------------------------
# >>> Wall extraction from an image <<<
# -------------------------------------

def build_wall_mask(img: np.ndarray, wall_values=None) -> np.ndarray:
    """
    Build a 0/255 wall mask from an input image.

    If `wall_values` is provided, pixels in `img` that match any of those values
    are marked as walls (255) and the rest as free space (0). If `wall_values`
    is None, the function assumes `img` is already mask-like and converts it
    to uint8 and optionally scales low-range masks to 0/255.

    Parameters:
    - img (np.ndarray): <br>
        Input image. Can be a label image, grayscale mask, or wall mask source.
    - wall_values (optional): <br>
        Values in `img` that should be interpreted as walls. Typically a list
        or tuple of labels. If None, `img` is treated as a mask-like image.

    Returns:
    - np.ndarray: <br>
        A uint8 wall mask with values 0 (free) and 255 (wall).
    """
    if wall_values is not None:
        mask = (np.isin(img, wall_values).astype(np.uint8) * 255)
    else:
        mask = img
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if np.max(mask) < 64:
            mask = (mask.astype(np.uint8) * 255)
    return mask


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


def build_occlusion_from_wallmask(mask_255: np.ndarray, wall_thickness: int = 1) -> np.ndarray:
    """
    Build a binary occlusion map from a wall mask.

    Converts a 0/255 wall mask into a binary map (0/1). Optionally dilates
    the walls to increase effective thickness for visibility checks.

    Parameters:
    - mask_255 (np.ndarray): <br>
        Wall mask with values 0 and 255.
    - wall_thickness (int): <br>
        If > 1, dilates the occlusion map using a square kernel to thicken walls.

    Returns:
    - np.ndarray: <br>
        Binary occlusion map of dtype uint8 with values:
        - 0 for free space
        - 1 for occluded (wall) pixels
    """
    occ = (mask_255 > 0).astype(np.uint8)
    if wall_thickness and wall_thickness > 1:
        k = np.ones((wall_thickness, wall_thickness), np.uint8)
        occ = cv2.dilate(occ, k, iterations=1)
        occ = (occ > 0).astype(np.uint8)
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

def build_path_for_sequence(
    source_xy: Tuple[float, float],
    receiver_xy: Tuple[float, float],
    seq: Tuple[int, ...],
    S_img: Tuple[float, float],
    walls: List[Segment],
) -> Optional[List[Tuple[float, float]]]:
    """
    Build a valid specular reflection path for a given wall sequence.

    Constructs the reflection points for a candidate wall sequence using a
    virtual-receiver backtracking approach. The method intersects the line from
    the image source S_img to the current virtual receiver with the active wall,
    accumulating reflection points in reverse order.

    Parameters:
    - source_xy (Tuple[float, float]): <br>
        Real source position in pixel coordinates (x, y).
    - receiver_xy (Tuple[float, float]): <br>
        Receiver position in pixel coordinates (x, y).
    - seq (Tuple[int, ...]): <br>
        Wall index sequence describing the reflection order.
    - S_img (Tuple[float, float]): <br>
        Precomputed image source position for `seq`.
    - walls (List[Segment]): <br>
        Wall segments.

    Returns:
    - Optional[List[Tuple[float, float]]]: <br>
        If valid, returns the full path as: <br>
        [source_xy, r1, r2, ..., receiver_xy]. <br>
        Returns None if any required intersection fails.
    """
    if len(seq) == 0:
        return [source_xy, receiver_xy]

    refl_points_rev: List[Tuple[float, float]] = []
    R_virtual = receiver_xy

    # process last reflection first
    for wi in reversed(seq):
        w = walls[wi]
        # intersect segment (S_img -> R_virtual) with wall segment
        hit = _seg_seg_intersection(S_img, R_virtual, w.A, w.B)
        if hit is None:
            return None
        refl_points_rev.append(hit)
        # update virtual receiver by reflecting across this wall line
        R_virtual = reflect_point_across_infinite_line(R_virtual, w.A, w.B)

    refl_points = list(reversed(refl_points_rev))
    return [source_xy] + refl_points + [receiver_xy]



def check_path_visibility_raster(points_xy: List[Tuple[float, float]], occ: np.ndarray, ignore_ends: int = 1) -> bool:
    """
    Check visibility for all segments of a path using a raster occlusion map.

    Runs `is_visible_raster` on every consecutive point pair in the path.

    Parameters:
    - points_xy (List[Tuple[float, float]]): <br>
        Path points [p0, p1, ..., pn].
    - occ (np.ndarray): <br>
        Occlusion map where nonzero values indicate blocked pixels.
    - ignore_ends (int): <br>
        Number of pixels to ignore at the ends of each segment.

    Returns:
    - bool: <br>
        True if every segment is visible, otherwise False.
    """
    for a, b in zip(points_xy[:-1], points_xy[1:]):
        if not is_visible_raster(a, b, occ, ignore_ends=ignore_ends):
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
    parallelization: int = 0
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
    if img.ndim == 3:
        # if it's RGB, convert to grayscale for wall picking if you do wall_values=None usage
        # but if you use wall_values with labels, keep it as is. We'll handle wall_values first.
        pass

    H, W = img.shape[:2]
    S = (source_rel[0] * W, source_rel[1] * H)

    wall_mask_255 = build_wall_mask(img, wall_values=wall_values)
    walls = get_wall_segments_from_mask(wall_mask_255, thickness=wall_thickness, approx_epsilon=approx_epsilon)
    occ = build_occlusion_from_wallmask(wall_mask_255, wall_thickness=wall_thickness)

    # precompute sequences + image sources once
    pre = precompute_image_sources(
        source_xy=S,
        walls=walls,
        max_order=max_order,
        forbid_immediate_repeat=forbid_immediate_repeat,
        max_candidates=max_candidates,
    )

    # receiver grid
    receivers = [(x + 0.5, y + 0.5) for y in range(0, H, step_px) for x in range(0, W, step_px)]

    # output (set iterative)
    output_map = np.zeros((H, W), dtype=np.float32)
    snapshots = []  # list of np.ndarray

    # always convert None to -1, means the same for thsi variable
    if iterative_steps is None:
        iterative_steps = -1

    if not iterative_tracking or (iterative_tracking and iterative_steps == 1):
        snapshot_every_x_steps = len(pre)
    elif iterative_tracking and iterative_steps == -1:
        snapshot_every_x_steps = 1
    else:
        snapshot_every_x_steps = int( len(pre) // iterative_steps )

    def eval_receiver(R, pre):
        val = 0.0
        for (seq, S_img) in pre:
            if ignore_zero_order and len(seq) == 0:
                continue

            pts = build_path_for_sequence(S, R, seq, S_img, walls)
            if pts is None:
                continue
            if not check_path_visibility_raster(pts, occ, ignore_ends=ignore_ends):
                continue

            val += 1.0

        return (R[0], R[1], val)

    # compute -> optional iterativly
    for start_idx in range(0, len(pre), snapshot_every_x_steps):
        pre_chunk = pre[start_idx:start_idx+snapshot_every_x_steps]

        if parallelization and parallelization != 0:
            if Parallel is None:
                raise RuntimeError("joblib not available but parallelization requested.")
            results = Parallel(n_jobs=parallelization, backend="loky", prefer="processes", batch_size=16)(
                delayed(eval_receiver)(R, pre_chunk) for R in receivers
            )
        else:
            results = [eval_receiver(R, pre_chunk) for R in receivers]

        # summarize current chunk to final output map

        # process results
        for x, y, v in results:
            ix, iy = int(x), int(y)
            if 0 <= ix < W and 0 <= iy < H:
                output_map[iy, ix] += float(v)

        snapshots += [output_map.copy()]

    # return output
    if not iterative_tracking:  # len(snapshots) <= 1
        return output_map
    else:
        return snapshots  # [::-1]






