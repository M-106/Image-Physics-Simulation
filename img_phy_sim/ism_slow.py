"""
FIXME

Also known as ISM

https://en.wikipedia.org/wiki/Method_of_images
"""



# ---------------
# >>> Imports <<<
# ---------------
from __future__ import annotations

from .img import open as img_open, get_width_height
from .math import normalize_point

import math
from typing import List, Tuple, Optional, Iterable

import numpy as np
import cv2

from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from shapely.affinity import scale

from joblib import Parallel, delayed



# ----------------------
# >>> Core Functions <<<
# ----------------------

def get_wall_segments(img, wall_values=None, thickness=1, approx_epsilon=1.5):
    """
    Extract wall boundary segments from an image and return shapely LineStrings + STRtree.

    Returns:
      walls: list[LineString]
      tree: STRtree over walls
    """
    IMG_WIDTH, IMG_HEIGHT = get_width_height(img)

    # Build mask
    if wall_values is not None:
        mask = (np.isin(img, wall_values).astype(np.uint8) * 255)
    else:
        mask = img
        if np.max(mask) < 64:
            mask = (mask.astype(np.uint8) * 255)

    # edges -> contours
    edges = cv2.Canny(mask, 100, 200)
    if thickness and thickness > 1:
        k = np.ones((thickness, thickness), np.uint8)
        edges = cv2.dilate(edges, k, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    walls = []
    for c in contours:
        # simplify contour to reduce segment count -> optional
        c2 = cv2.approxPolyDP(c, epsilon=approx_epsilon, closed=True)

        pts = [tuple(p[0].astype(float)) for p in c2]
        if len(pts) < 2:
            continue

        # convert polyline into segments
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if a == b:
                continue
            walls.append(LineString([a, b]))

        # if closed, add last->first
        if len(pts) >= 3:
            a, b = pts[-1], pts[0]
            if a != b:
                walls.append(LineString([a, b]))

    tree = STRtree(walls) if len(walls) > 0 else None
    return walls, tree



def reflect_point_across_segment(P, seg: LineString):
    """
    Reflect point P across the infinite line defined by seg (two points).
    """
    (x1, y1), (x2, y2) = list(seg.coords)
    A = np.array([x1, y1], dtype=float)
    B = np.array([x2, y2], dtype=float)
    P = np.array([P[0], P[1]], dtype=float)

    AB = B - A
    n = np.linalg.norm(AB) + 1e-12
    ABn = AB / n

    AP = P - A
    proj = A + ABn * np.dot(AP, ABn)
    Pref = proj + (proj - P)
    return (float(Pref[0]), float(Pref[1]))



def segment_intersection_point(line: LineString, wall: LineString):
    """
    Return a single intersection point between two LineStrings if it exists and is a Point.
    """
    inter = line.intersection(wall)
    if inter.is_empty:
        return None
    if inter.geom_type == "Point":
        return (inter.x, inter.y)
    # could be MultiPoint / LineString (colinear). For specular ISM we skip ambiguous cases.
    return None



def enumerate_wall_sequences_indices(n_walls, max_order, forbid_immediate_repeat=True):
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



def find_ism_paths_shapely(
    source_xy,
    receiver_xy,
    walls,
    tree,
    max_order=2,
    forbid_immediate_repeat=True,
    max_candidates=None,
    visibility_tol=1e-6,
):
    """
    Compute specular reflection paths using Image Source Method (2D) on shapely wall segments.

    Returns:
      paths: list of paths, where each path is list of points [S, x1, x2, ..., R]
             (x1.. are reflection points in real space order)
    """
    if len(walls) == 0:
        # Only direct path
        return [[source_xy, receiver_xy]]

    # enumerate sequences
    seqs = enumerate_wall_sequences_indices(
        n_walls=len(walls),
        max_order=max_order,
        forbid_immediate_repeat=forbid_immediate_repeat
    )

    # optional: cap candidates (useful if you have many walls)
    if max_candidates is not None and len(seqs) > max_candidates:
        seqs = seqs[:max_candidates]

    paths = []

    # order 0 included as empty seq
    for seq in seqs:
        # Build image source
        S_img = source_xy
        for wi in seq:
            S_img = reflect_point_across_segment(S_img, walls[wi])

        # Candidate line from image source to receiver
        L = LineString([S_img, receiver_xy])

        # Backtrack reflection points
        refl_points = []
        valid = True

        # Process walls in reverse order (last reflection first)
        current_line = L
        # We need a moving "virtual receiver" technique; simplest workable way:
        # iteratively intersect current_line with wall, then reflect the *receiver* across wall
        # This keeps line construction correct.
        R_virtual = receiver_xy

        for wi in reversed(seq):
            wall = walls[wi]

            # line from current image source to current virtual receiver
            current_line = LineString([S_img, R_virtual])

            hit = segment_intersection_point(current_line, wall)
            if hit is None:
                valid = False
                break

            refl_points.append(hit)

            # update virtual receiver by reflecting across the same wall
            R_virtual = reflect_point_across_segment(R_virtual, wall)

        if not valid:
            continue

        # refl_points are from last->first; reverse to get real order
        refl_points = list(reversed(refl_points))

        # Build full path points
        pts = [source_xy] + refl_points + [receiver_xy]

        # Visibility check per segment
        # Allow touching the reflecting wall on the segment endpoints:
        allow = set(walls[wi] for wi in seq)

        ok = True
        for a, b in zip(pts[:-1], pts[1:]):
            # if not is_segment_visible(a, b, walls, tree, allow_touch_walls=allow, tol=visibility_tol):
            if not is_visible_raster(a, b, ):
                ok = False
                break

        if not ok:
            continue

        paths.append(pts)

    return paths



def ism_paths_to_rays(paths, should_scale=True, img_shape=None):
    """
    Convert ISM paths (list of [p0,p1,..,pn]) into your rays format:
      ray[beam][point] where each beam has two points (start,end)
    """
    rays = []
    if should_scale:
        if img_shape is None:
            raise ValueError("img_shape required for scaling")
        H, W = img_shape[:2]

    for pts in paths:
        ray = []
        for a, b in zip(pts[:-1], pts[1:]):
            if should_scale:
                a2 = normalize_point(a, width=W, height=H)
                b2 = normalize_point(b, width=W, height=H)
            else:
                a2, b2 = a, b
            ray.append([a2, b2])
        rays.append(ray)
    return rays



def _tree_query_geoms(tree, geom, fallback_walls):
    """
    Shapely 2.x STRtree.query returns indices (np.ndarray of ints).
    Shapely 1.8 may return geometries.
    This helper always returns geometry objects.
    """
    if tree is None:
        return fallback_walls

    res = tree.query(geom)

    # Shapely 2.x: indices
    try:
        if len(res) > 0 and isinstance(res[0], (int, np.integer)):
            return [fallback_walls[i] for i in res]
    except Exception:
        pass

    # Shapely 1.8: geometries (or empty)
    return list(res)



def is_segment_visible(p1, p2, walls, tree: STRtree, allow_touch_walls=None, tol=1e-6):
    seg = LineString([p1, p2])

    candidates = _tree_query_geoms(tree, seg, walls)

    for w in candidates:
        # w ist jetzt garantiert Geometry
        if allow_touch_walls is not None and w in allow_touch_walls:
            inter = seg.intersection(w)
            if inter.is_empty:
                continue
            if inter.geom_type == "Point":
                if (Point(p1).distance(inter) <= tol) or (Point(p2).distance(inter) <= tol):
                    continue
            return False
        else:
            # Blockiere "echte" Schnitte im Inneren
            inter = seg.intersection(w)
            if inter.is_empty:
                continue

            # Erlaube nur endpoint-touch
            if inter.geom_type == "Point":
                if (Point(p1).distance(inter) <= tol) or (Point(p2).distance(inter) <= tol):
                    continue
                return False

            # MultiPoint / LineString / GeometryCollection => konservativ blockieren
            return False

    return True


def is_visible_raster(p1, p2, occ, tol_endpoints=1):
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])

    # draw line into temp mask (or sample points along the line)
    rr, cc = cv2.line(np.zeros_like(occ, dtype=np.uint8), (x1,y1), (x2,y2), 255, 1).nonzero()
    if len(rr) == 0:
        return True

    # ignore a few pixels at both ends
    if len(rr) > 2*tol_endpoints:
        rr = rr[tol_endpoints:-tol_endpoints]
        cc = cc[tol_endpoints:-tol_endpoints]

    return not np.any(occ[rr, cc])



def trace_ism_paths(
    source_rel,
    receiver_rel,
    img_src,
    wall_values,
    wall_thickness=1,
    max_order=2,
    should_scale_rays=True,
    should_scale_img=True,
    # use_pre_filter=True,
    # prune=True
):
    if isinstance(img_src, np.ndarray):
        img = img_src
    else:
        img = img_open(src=img_src, should_scale=should_scale_img, should_print=False)

    H, W = img.shape[:2]
    S = (source_rel[0] * W, source_rel[1] * H)
    R = (receiver_rel[0] * W, receiver_rel[1] * H)

    walls, tree = get_wall_segments(img, wall_values=wall_values, thickness=wall_thickness)

    paths = find_ism_paths_shapely(
        source_xy=S,
        receiver_xy=R,
        walls=walls,
        tree=tree,
        max_order=max_order,
    )

    rays = ism_paths_to_rays(paths, should_scale=should_scale_rays, img_shape=img.shape)

    return rays, paths



def path_energy(points_xy, *, model="3d", reflection_alpha=0.2, eps=1e-9):
    """
    points_xy: list of (x,y) in pixel coords, e.g. [S, r1, r2, ..., R]
    model: "3d" -> 1/D^2, "2d" -> 1/D
    reflection_alpha: constant absorption per reflection (0..1)
    """
    pts = np.array(points_xy, dtype=float)
    seg = pts[1:] - pts[:-1]
    dists = np.sqrt((seg[:,0]**2 + seg[:,1]**2))

    D = float(dists.sum())
    if D < eps:
        return 0.0

    # geometric spreading
    if model == "2d":
        E = 1.0 / (D + eps)
    else:  # "3d"
        E = 1.0 / (D*D + eps)

    # reflection losses (number of reflections = num_points - 2)
    n_reflections = max(0, len(points_xy) - 2)
    if n_reflections > 0:
        refl_factor = (1.0 - reflection_alpha) ** n_reflections
        E *= refl_factor

    return E



def precompute_image_sources(source_xy, walls, max_order, forbid_immediate_repeat=True):
    seqs = enumerate_wall_sequences_indices(len(walls), max_order, forbid_immediate_repeat)
    pre = []
    for seq in seqs:
        S_img = source_xy
        for wi in seq:
            S_img = reflect_point_across_segment(S_img, walls[wi])
        pre.append((seq, S_img))
    return pre



def compute_noise_map_ism(
    source_rel,
    img_src,
    wall_values,
    wall_thickness=1,
    max_order=2,
    step_px=8,
    energy_model="3d",
    reflection_alpha=0.2,
    parallelization=0,
):
    """
    Returns:
      energy_map: float32 image (H,W) with energy sums at receiver grid points (others = 0)
      db_map: float32 image (H,W) with dB values (others = -inf)
    """
    # load image
    if isinstance(img_src, np.ndarray):
        img = img_src
    else:
        img = img_open(src=img_src, should_scale=True, should_print=False)

    H, W = img.shape[:2]
    S = (source_rel[0] * W, source_rel[1] * H)

    # extract walls once
    walls, tree = get_wall_segments(img, wall_values=wall_values, thickness=wall_thickness)

    # receiver grid (pixel coords)
    receivers = [(x + 0.5, y + 0.5) for y in range(0, H, step_px) for x in range(0, W, step_px)]

    pre = precompute_image_sources(S, walls, max_order)

    def eval_receiver(R):
        # paths = find_ism_paths_shapely(
        #     source_xy=S,
        #     receiver_xy=R,
        #     walls=walls,
        #     tree=tree,
        #     max_order=max_order,
        # )

        # energy sum
        E = 0.0
        # for pts in paths:
        for seq, S_img in pre:
            E += path_energy(
                points_xy=seq,  # pts,
                model=energy_model,
                reflection_alpha=reflection_alpha
            )
        return (R[0], R[1], E)

    # compute
    if parallelization and parallelization != 0:
        results = Parallel(n_jobs=parallelization, backend="loky", prefer="processes", batch_size=1)(
            delayed(eval_receiver)(R) for R in receivers
        )
    else:
        results = [eval_receiver(R) for R in receivers]

    # fill maps
    energy_map = np.zeros((H, W), dtype=np.float32)
    for x, y, E in results:
        ix, iy = int(x), int(y)
        if 0 <= ix < W and 0 <= iy < H:
            energy_map[iy, ix] = E

    # convert to dB (relative)
    db_map = np.full((H, W), -np.inf, dtype=np.float32)
    nonzero = energy_map > 0
    db_map[nonzero] = 10.0 * np.log10(energy_map[nonzero])

    return energy_map, db_map









