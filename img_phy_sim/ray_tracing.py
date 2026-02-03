"""
**Beam Tracing and Visualization Utilities**

This module provides a set of tools for simulating and visualizing the propagation
of rays (beams) through 2D images that may contain reflective or obstructive walls.
It includes methods for tracing ray paths, handling reflections, scaling and
normalizing ray coordinates, and drawing the resulting beam paths onto images.

The core idea is to represent a 2D environment as an image where certain pixel
values correspond to walls or obstacles. Rays are emitted from a relative position
and traced in specified directions, optionally reflecting off walls multiple times.
The results can then be visualized or further processed.

Main features:
- Ray tracing with customizable reflection order and wall detection
- Support for relative and absolute coordinate systems
- Ray scaling utilities for normalization or resizing
- Image rendering functions to visualize rays, walls, and reflection paths
- Flexible output modes: single image, multi-channel, or multiple separate images

Typical workflow:
1. Use `trace_beams()` to simulate multiple beams across an image.
2. Render the rays on an image using `draw_rays()`.

Dependencies:
- numpy
- cv2 (OpenCV)
- internal math + img modules

Example:
```python
img = ips.img.open("scene.png")
rays = ips.ray_tracing.trace_beams(
    rel_position=(0.5, 0.5),
    img_src=img,
    directions_in_degree=[0, 45, 90, 135],
    wall_values=[0],
    wall_thickness=2,
    reflexion_order=2
)
output = ips.ray_tracing.draw_rays(rays, img_shape=img.shape, ray_value=255, ray_thickness=1)
ips.img.imshow(output, size=5)
```

Author:<br>
Tobia Ippolito, 2025

Functions:
- print_rays_info(...)  - Pritn informations about created rays.
- save(...)  - Save rays into a file.
- open(...)  - Load saved rays.
- merge(...)  - Merge 2 or more rays together.
- get_all_pixel_coordinates_in_between(...)  - Use brahams algorithm for getting any line in a quantizied space.
- get_wall_map(...)  - Extract edges and get the wall-map with direction angles of walls. 
- update_pixel_position(...)  - Get the next pixel to come from one point to another in a quantizied system.
- calc_reflection(...)  - Calculate the reflexion of 2 vectors.
- get_img_border_vector(...)  - Get the vector of a border of the image.
- trace_beam(...)  - Trace a single beam with reflexions.
- trace_beams(...)  - Trace multiple beams with reflections through an image.
- scale_rays(...)   - Normalize or rescale ray coordinates.
- draw_rectangle_with_thickness(...) - Draw filled or thick rectangles.
- draw_line_or_point(...) - Draw a single point or a line segment.
- draw_rays(...)    - Visualize traced rays as images or channels.
- trace_and_draw_rays(...)    - Compute and draw rays.
"""



# ---------------
# >>> Imports <<<
# ---------------
from .img import open as img_open, get_width_height
from .math import degree_to_vector, vector_to_degree, normalize_point, \
                  degree_to_vector_numba, vector_to_degree_numba, normalize_point_numba

import builtins
import pickle
import math
import copy

import numpy as np
import cv2

# performance optimization
from joblib import Parallel, delayed
import numba
from numba.typed import List
from numba import types



# --------------
# >>> Helper <<<
# --------------

class RayIterator:
    """
    A container class to save every step of a ray tracing process.
    """
    def __init__(self, other_ray_iterator=None):
        """
        Initialize a RayIterator instance.
        
        Parameters:
        - other_ray_iterator (RayIterator, optional): <br>
            An existing RayIterator to copy. If provided, creates a deep copy 
            of the other iterator's rays_collection. If None, creates an empty iterator.
            
        Returns: <br>
        None
        """
        if other_ray_iterator is None:
            self.rays_collection = []
        else:
            self.rays_collection = copy.deepcopy(other_ray_iterator.rays_collection)

    def __repr__(self):
        """
        Return a string representation of the RayIterator.
        
        Returns:<br>
        str: String representation showing the number of iterations/time-steps.
        """
        return f"RayIterator with {self.len_iterations()} iterations (time-steps)."

    def __iter__(self):
        """
        Make the RayIterator iterable over its rays collections.
        
        Yields:<br>
        list: Each iteration's collection of rays.
        """
        for rays in self.rays_collection:
            yield rays

    def __getitem__(self, key):
        """
        Get rays from the latest iteration using key/index.
        
        Parameters:
        - key (int or slice): Index or slice to access rays in the latest iteration.
        
        Returns:<br>
        list: Rays from the latest iteration corresponding to the key.
        """
        return self.rays_collection[-1][key]
    
    def __len__(self):
        """
        Get the number of rays in the latest iteration.
        
        Returns:<br>
        int: Number of rays in the latest iteration.
        """
        return len(self.rays_collection[-1])
    
    def __add__(self, other):
        """
        Add another RayIterator or value to this RayIterator element-wise.
        
        Parameters:
        - other (RayIterator or any): <br>
            If RayIterator: combines ray collections from both iterators. 
            If other type: adds the value to each ray in all iterations.
        
        Returns:<br>
        RayIterator: New RayIterator containing the combined/adjusted results.
        
        Raises:<br>
        TypeError: If other is not a RayIterator and addition operation fails.
        """
        if isinstance(other, RayIterator):
            new_iterator = RayIterator()
            if self.len_iterations() > other.len_iterations():
                iter_1 = self
                iter_2 = other
            else:
                iter_1 = other
                iter_2 = self

            iter_1 = copy.deepcopy(iter_1)
            iter_2 = copy.deepcopy(iter_2)

            for idx in range(iter_1.len_iterations()):
                cur_addition = iter_1.get_iteration(idx)
                if iter_2.len_iterations() > idx:
                    cur_addition += iter_2.get_iteration(idx)
                elif iter_2.len_iterations() == 0:
                    pass
                else:
                    cur_addition += iter_2.get_iteration(-1)

                new_iterator.add_iteration(cur_addition)

            return new_iterator
        else:
            new_iterator = RayIterator()

            for idx in range(self.len_iterations()):
                cur_addition = self.get_iteration(idx) + other
                new_iterator.add_iteration(cur_addition)

            return new_iterator
        
    def __iadd__(self, other):
        """
        In-place addition of another RayIterator or value.
        
        Parameters:
        - other (RayIterator or any): 
            Object to add to this RayIterator.
        
        Returns:<br>
        RayIterator: self, after performing the in-place addition.
        """
        new_iterator = self.__add__(other) 
        self.rays_collection = new_iterator.rays_collection
        return self
    
    def len_iterations(self):
        """
        Get the total number of iterations/time-steps stored.
        
        Returns:<br>
        int: <br>
            Number of iterations in the rays_collection.
        """
        return len(self.rays_collection)

    def add_iteration(self, rays):
        """
        Add a new iteration (collection of rays) to the iterator.
        
        Parameters:
        - rays (list): Collection of rays to add as a new iteration.
            Format: rays[ray][beam][point] = (x, y)
        
        Returns:<br>
        RayIterator: self, for method chaining.
        """
        self.rays_collection += [copy.deepcopy(rays)]
        return self
    
    def add_rays(self, rays):
        """
        Add rays to the every iteration (in-place modification).
        If one iterator have less steps, the last step will be used for all other iterations.
        Which equals no change for those iterations.
        
        Parameters:
        - rays (list): Rays to add to the latest iteration.
            Format: rays[ray][beam][point] = (x, y)
        
        Returns:<br>
        list: <br>
        The updated rays_collection.
        """
        self.rays_collection = self.__add__(copy.deepcopy(rays)).rays_collection
        return self.rays_collection

    def print_info(self):
        """
        Print statistical information about the ray collections.
        
        Displays:
        - Number of iterations
        - Information about the latest iteration's rays including:
          * Number of rays, beams, reflexions, and points
          * Mean, median, max, min, and variance for beams per ray, 
            reflexions per ray, and points per beam
          * Value range for x and y coordinates
        
        Returns:<br>
        None
        """
        print(f"Ray Iterator with {self.len_iterations()} iterations (time-steps).")
        print("Latest Rays Info:\n")
        print_rays_info(self.rays_collection[-1])
        # for idx, rays in enumerate(self.rays_collection):
        #     print(f"--- Rays Set {idx} ---")
        #     print_rays_info(rays)

    def reduce_to_x_steps(self, x_steps):
        """
        Reduce the number of stored iterations to approximately x_steps.
        
        Uses linear sampling to keep representative iterations while reducing
        memory usage. If x_steps is greater than current iterations, does nothing.
        
        Parameters:
        - x_steps (int): <br>
            Desired number of iterations to keep.
        
        Returns:<br>
        None
        """
        if x_steps >= self.len_iterations():
            return  # nothing to do

        step_size = self.len_iterations() / x_steps
        new_rays_collection = []
        for i in range(x_steps):
            index = int(i * step_size)
            new_rays_collection += [self.get_iteration(index)]

        self.rays_collection = new_rays_collection

    def apply_and_update(self, func):
        """
        Apply a function to each iteration's rays and update in-place.
        
        Parameters:
        - func (callable): <br>
            Function that takes a rays collection and returns a modified rays collection. 
            Will be applied to each iteration.
        
        Returns:<br>
        None
        """
        for i in range(self.len_iterations()):
            self.rays_collection[i] = func(self.rays_collection[i])
    
    def apply_and_return(self, func):
        """
        Apply a function to each iteration's rays and return results.
        
        Parameters:
        - func (callable): <br>
            Function that takes a rays collection and returns some result. 
            Will be applied to each iteration.
        
        Returns:
        list: <br>
            Results of applying func to each iteration's rays.
        """
        results = []
        for i in range(self.len_iterations()):
            results += [func(self.rays_collection[i])]

        return results
    
    def get_iteration(self, index):
        """
        Get a specific iteration's rays collection.
        
        Parameters:
        - index (int): <br>
            Index of the iteration to retrieve. Supports negative indexing (e.g., -1 for last iteration).
        
        Returns:
        list: <br>
            - Rays collection at the specified iteration.
        
        Raises: <br>
        IndexError: If index is out of range.
        """
        if index < -1 * self.len_iterations() or index >= self.len_iterations():
            raise IndexError("RayIterator index out of range.")
        return self.rays_collection[index]

# def convert_rays_to_ray_iterator(rays):
#     """
#     Convert a list of rays into a RayIterator with a single iteration.

#     Parameters:
#     - rays (list): 
#         Nested list structure representing rays. Format: rays[ray][beam][point] = (x, y)
#     Returns:
#     - RayIterator:
#         RayIterator containing the provided rays as its only iteration.
#     """
#     ray_iterator = RayIterator()
#     cur_rays = []

#     max_iterations

#     for ray_idx in range(len(rays)):
#         max_beams = max([len(beams) for beams in rays[ray_idx]])
#         for cur_beam_idx in range(max_beams):
#             ray = []
#             for point in rays[ray_idx][cur_beam_idx]:
#                 cur_rays += [ray]
#         ray_iterator.add_iteration(cur_rays)
#     return ray_iterator

def print_rays_info(rays):
    """
    Print statistical information about a collection of rays.

    Each ray consists of multiple beams, and each beam consists of multiple points.
    The function computes and displays statistics such as:
    - Number of rays, beams, reflexions, and points
    - Mean, median, max, min, and variance for beams per ray, reflexions per ray, and points per beam
    - Value range for x and y coordinates

    Parameters:
    - rays (list): <br>
        Nested list structure representing rays. Format: rays[ray][beam][point] = (x, y)
    """
    if isinstance(rays, RayIterator):
        rays.print_info()
    else:
        nrays = 0
        nbeams = 0
        nbeams_per_ray = []
        nreflexions = 0
        nreflexions_per_ray = []
        npoints = 0
        npoints_per_beam_point = []
        values_per_point = []
        min_x_value = None
        max_x_value = None
        min_y_value = None
        max_y_value = None
        for ray in rays:
            nrays += 1
            cur_beams = 0
            cur_reflexions = 0
            for beam_points in ray:
                nbeams += 1
                nreflexions += 1
                cur_beams += 1
                cur_reflexions += 1
                cur_points = 0
                for x in beam_points:
                    npoints += 1
                    cur_points += 1
                    values_per_point += [len(x)]
                    min_x_value = x[0] if min_x_value is None else min(min_x_value, x[0])
                    max_x_value = x[0] if max_x_value is None else max(max_x_value, x[0])
                    min_y_value = x[1] if min_y_value is None else min(min_y_value, x[1])
                    max_y_value = x[1] if max_y_value is None else max(max_y_value, x[1])
                npoints_per_beam_point += [cur_points]
            nreflexions -= 1
            cur_reflexions -= 1
            nreflexions_per_ray += [cur_reflexions]
            nbeams_per_ray += [cur_beams]

        print(f"Rays: {nrays}")
        print(f"Beams: {nbeams}")
        print(f"    - Mean Beams per Ray: {round(np.mean(nbeams_per_ray), 1)}")
        print(f"        - Median: {round(np.median(nbeams_per_ray), 1)}")
        print(f"        - Max: {round(np.max(nbeams_per_ray), 1)}")
        print(f"        - Min: {round(np.min(nbeams_per_ray), 1)}")
        print(f"        - Variance: {round(np.std(nbeams_per_ray), 1)}")
        print(f"Reflexions: {nreflexions}")
        print(f"    - Mean Reflexions per Ray: {round(np.mean(nreflexions_per_ray), 1)}")
        print(f"        - Median: {round(np.median(nreflexions_per_ray), 1)}")
        print(f"        - Max: {round(np.max(nreflexions_per_ray), 1)}")
        print(f"        - Min: {round(np.min(nreflexions_per_ray), 1)}")
        print(f"        - Variance: {round(np.std(nreflexions_per_ray), 1)}")
        print(f"Points: {npoints}")
        print(f"    - Mean Points per Beam: {round(np.mean(npoints_per_beam_point), 1)}")
        print(f"        - Median: {round(np.median(npoints_per_beam_point), 1)}")
        print(f"        - Max: {round(np.max(npoints_per_beam_point), 1)}")
        print(f"        - Min: {round(np.min(npoints_per_beam_point), 1)}")
        print(f"        - Variance: {round(np.std(npoints_per_beam_point), 1)}")
        print(f"    - Mean Point Values: {round(np.mean(values_per_point), 1)}")
        print(f"        - Median: {round(np.median(values_per_point), 1)}")
        print(f"        - Variance: {round(np.std(values_per_point), 1)}")
        print(f"\nValue-Range:\n  x ∈ [{min_x_value:.2f}, {max_x_value:.2f}]\n  y ∈ [{min_y_value:.2f}, {max_y_value:.2f}]")
        # [ inclusive, ( number is not included

        if nrays > 0:
            print(f"\nExample:\nRay 1, beams: {len(rays[0])}")
            if nbeams > 0:
                print(f"Ray 1, beam 1, points: {len(rays[0][0])}")
                if npoints > 0:
                    print(f"Ray 1, beam 1, point 1: {len(rays[0][0][0])}")
        print("\n")



def save(path, rays):
    """
    Save a list of rays to a text file.

    The rays are serialized using a simple text-based format. 
    Each ray is delimited by '>' and '<', and each point is represented as "x | y".

    Parameters:
    - path (str): <br>
        Path to the file where data should be saved. If no '.txt' extension is present, it will be appended automatically.
    - rays (list): <br>
        Nested list structure representing rays. Format: rays[ray][beam][point] = (x, y)

    Returns:<br>
        None
    """
    if isinstance(rays, RayIterator):
        if not path.endswith(".pkl"):
            path += ".pkl"
        pickle.dump(rays, builtins.open(path, "wb"))
        return
    
    # transform rays into an string
    rays_str = ""
    for ray in rays:
        rays_str += ">\n"
        for beam in ray:
           rays_str += "\n"
           for cur_point in beam:
               rays_str += f"{cur_point[0]} | {cur_point[1]}, " 
        rays_str += "<\n"

    rays_str = rays_str.replace("\n\n", "\n")

    if not path.endswith(".txt"):
        path += ".txt"
    
    with builtins.open(path, "w") as file_:
        file_.write(rays_str)



def open(path, is_iterator=False) -> list:
    """
    Open and parse a ray text file into a structured list.

    The file is expected to follow the same format as produced by `save()`.

    Parameters:
    - path (str): <br>
        Path to the .txt file containing ray data.

    Returns:
    - list: <br>
        Nested list structure representing rays. Format: rays[ray][beam][point] = (x, y)
    """
    if is_iterator or path.endswith(".pkl"):
        if not path.endswith(".pkl"):
            path += ".pkl"
        rays = pickle.load(builtins.open(path, "rb"))
        return rays

    if not path.endswith(".txt"):
        path += ".txt"

    with builtins.open(path, "r") as file_:
        content = file_.read().strip()

    rays = []
    for ray in content.split(">"):
        extracted_ray = []
        for beam in ray.split("\n"):
            extracted_beam = []
            beam = beam.strip()
            if not beam or beam == "<":
                continue

            for point in beam.split(","):
                point = point.strip()
                if not point or point == "<":
                    continue

                try:
                    point_x, point_y = point.strip().split("|")
                except Exception as e:
                    print("Point of error:", point)
                    raise e

                extracted_beam += [(float(point_x), float(point_y))]
            if len(extracted_beam) > 0:
                extracted_ray += [extracted_beam]
        if len(extracted_ray) > 0:
            rays += [extracted_ray]
    return rays



def merge(rays_1, rays_2, *other_rays_):
    """
    Merge multiple ray datasets into a single list.

    Parameters:
    - rays_1 (list): <br>
        First set of rays.
    - rays_2 (list): <br>
        Second set of rays. 
    - *other_rays_ (list): <br>
        Additional ray lists to merge.

    Returns:
    - list: 
        Combined list of all rays.
    """
    merged = rays_1 + rays_2

    for rays in other_rays_:
        merged += rays

    return merged




# ----------------------
# >>> Core Functions <<<
# ----------------------

def get_all_pixel_coordinates_in_between(x1, y1, x2, y2):
    """
    Get all pixel coordinates along a line between two points using Bresenham's algorithm.

    https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

    Parameters:
    - x1 (int): <br>
        Starting x-coordinate.
    - y1 (int): <br>
        Starting y-coordinate.
    - x2 (int): <br>
        Ending x-coordinate.
    - y2 (int): <br>
        Ending y-coordinate.

    Returns:
    - list: 
        List of (x, y) tuples representing all pixels between the start and end points
    """
    coordinates = []

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1

    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    if dx > dy:
        err = dx / 2.0
        while x != x2:
            coordinates += [(x, y)]
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            coordinates += [(x, y)]
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    coordinates += [(x2, y2)]  # include the last point
    return coordinates



def get_wall_map(img, wall_values=None, thickness=1,
                 use_numba_compilation=False):
    """
    Generate a wall map where each pixel encodes the wall orientation (in degrees).

    Parameters:
    - img (numpy.ndarray): <br>
        Input image representing scene or segmentation mask.
    - wall_values (list, optional): <br>
        Specific pixel values considered as walls. If None, all non-zero pixels are treated as walls.
    - thickness (int, optional): <br>
        Thickness of wall lines (default is 1).
    - use_numba_compilation (bool, optional):<br>
        Whether to use the compiled (to machine code) version of compute heavy functions.

    Returns:
    - numpy.ndarray: 
        2D array (same width and height as input) 
        where each wall pixel contains the wall angle in degrees (0-360), 
        and non-wall pixels are set to infinity (np.inf).
    """
    # numba optimization -> change function locally
    if use_numba_compilation:
        get_all_pixel_coordinates_in_between_ = get_all_pixel_coordinates_in_between_numba      
    else:
        get_all_pixel_coordinates_in_between_ = get_all_pixel_coordinates_in_between

    IMG_WIDTH, IMG_HEIGHT =  get_width_height(img)
    wall_map = np.full((IMG_HEIGHT, IMG_WIDTH), np.inf, dtype=np.uint16)  # uint16 to get at least 360 degree/value range

    # only detect edges from objects with specific pixel values
    if wall_values is not None:
        mask = np.isin(img, wall_values).astype(np.uint8) * 255
    else:
        mask = img
        if np.max(mask) < 64:
            mask = mask.astype(np.uint8) * 255

    # detect edges and contours
    edges = cv2.Canny(mask, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # convert contours to line segments
    for c in contours:
        for i in range(len(c)-1):
            x1, y1 = c[i][0]
            x2, y2 = c[i+1][0]
            dy = y2 - y1
            dx = x2 - x1
            angle = math.atan2(dy, dx)
            angle_deg = math.degrees(angle)
            for x, y in get_all_pixel_coordinates_in_between_(x1, y1, x2, y2):
                # wall_map[y, x] = int(angle_deg) % 360

                for tx in range(-thickness, thickness+1):
                    for ty in range(-thickness, thickness+1):
                        nx, ny = x+tx, y+ty
                        if 0 <= nx < IMG_WIDTH and 0 <= ny < IMG_HEIGHT:
                            wall_map[ny, nx] = int(angle_deg) % 360
    return wall_map



def update_pixel_position(direction_in_degree, cur_position, target_line):
    """
    Update the pixel position of a moving point toward a target line based on direction and proximity.

    Combines the direction vector with a vector pointing toward the closest point
    on the target line, ensuring pixel-wise movement (discrete steps).

    Parameters:
    - direction_in_degree (float): <br>
        Movement direction in degrees.
    - cur_position (tuple): <br>
        Current pixel position (x, y).
    - target_line (list): <br>
        Target line defined as [x1, y1, x2, y2].

    Returns:
    - tuple: 
        Updated pixel position (x, y).
    """
    # 1. Calc distance from point to target line

    # perpendicular vector to line (points toward line)
    point = np.array(cur_position)
    line_start_point = np.array(target_line[0:2])
    line_end_point = np.array(target_line[2:4])

    # projection along the line -> throw the point vector vertical/perpendicular on the line and see where it cuts with normed AP to AB
    # t is the length from point to the line, therefore it gets normed
    t = np.dot(point - line_start_point, line_end_point - line_start_point) / (np.dot(line_end_point - line_start_point, line_end_point - line_start_point) + 1e-8)
    
    # limit it to the line id needed -> because we don't want smaller or bigger values than that
    #   -> 0 would be point A
    #   -> 1 would be point B 
    # t = np.clip(t, 0, 1)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    # get closest point by applying the found t as lentgh from startpoint in the line vector direction
    closest = line_start_point + t * (line_end_point - line_start_point)

    # get the final vector to the line
    to_line = closest - point  # vector from current pos to closest point on line
    
    # 2. Calc vector to the degree
    # movement vector based on angle
    rad = math.radians(direction_in_degree)
    move_dir = np.array([math.cos(rad), math.sin(rad)])
    
    # 3. Combine vector to the line and degree vector
    # combine movement towards direction and towards line
    combined = move_dir + to_line * 0.5  # weighting factor
    
    # pick pixel step (continuous to discrete) -> [-1, 0, 1]
    step_x = np.sign(combined[0])
    step_y = np.sign(combined[1])
    
    # clamp to [-1, 1], if bigger/smaller
    # step_x = int(np.clip(step_x, -1, 1))
    if step_x < 0.0:
        step_x = 0.0
    elif step_x > 1.0:
        step_x = 1.0
    # step_y = int(np.clip(step_y, -1, 1))
    if step_y < 0.0:
        step_y = 0.0
    elif step_y > 1.0:
        step_y = 1.0
    
    return (int(cur_position[0] + step_x), int(cur_position[1] + step_y))



def calc_reflection(collide_vector, wall_vector):
    """
    Calculate the reflection of a collision vector against a wall vector.

    The reflection is computed using the wall's normal vector and the formula:
        r = v - 2 * (v · n) * n

    Parameters:
    - collide_vector (array-like): <br>
        Incoming vector (2D).
    - wall_vector (array-like): <br>
        Wall direction vector (2D).

    Returns:
    - numpy.ndarray: 
        Reflected 2D vector.
    """
    # normalize both
    collide_vector = np.array(collide_vector, dtype=np.float64)
    collide_vector /= np.linalg.norm(collide_vector)
    wall_vector = np.array(wall_vector, dtype=np.float64)
    wall_vector /= np.linalg.norm(wall_vector)

    # calculate the normal of the wall
    normal_wall_vector_1 = np.array([-wall_vector[1], wall_vector[0]])  # rotated +90°
    normal_wall_vector_2 = np.array([wall_vector[1], -wall_vector[0]])  # rotated -90°

    # decide which vector is the right one
    #   -> dot product tells which normal faces the incoming vector
    #   -> dor product shows how similiar 2 vectors are => smaller 0 means they show against each other => right vector
    if np.dot(collide_vector, normal_wall_vector_1) < 0:
        normal_wall_vector = normal_wall_vector_1
    else:
        normal_wall_vector = normal_wall_vector_2
    
    # calc the reflection
    return collide_vector - 2 * np.dot(collide_vector, normal_wall_vector) * normal_wall_vector


def get_img_border_vector(position, max_width, max_height):
    """
    Determine the wall normal vector for an image border collision.

    Parameters:
    - position (tuple): <br>
        Current position (x, y).
    - max_width (int): <br>
        Image width.
    - max_height (int): <br>
        Image height.

    Returns:
    - tuple: 
        Border wall vector corresponding to the collision side.
    """
    # print(f"got {position=}")
    if position[0] < 0:
        return (0, 1)
    elif position[0] >= max_width:
        return (0, 1)
    elif position[1] < 0:
        return (1, 0)
    elif position[1] >= max_height:
        return (1, 0)
    else:
        # should never reach that!
        return (0, 0)


def trace_beam(abs_position, 
               img, 
               direction_in_degree, 
               wall_map,
               wall_values, 
               img_border_also_collide=False,
               reflexion_order=3,
               should_scale=True,
               should_return_iterative=False,
               remove_iterative=True):
    """
    Trace a ray (beam) through an image with walls and reflections.

    The beam starts from a given position and follows a direction until it hits
    a wall or border. On collisions, reflections are computed using wall normals.

    Parameters:
    - abs_position (tuple): <br>
        Starting position (x, y) of the beam.
    - img (numpy.ndarray): <br>
        Input image or segmentation map.
    - direction_in_degree (float): <br>
        Initial direction angle of the beam.
    - wall_map (numpy.ndarray): <br>
        Map containing wall orientations in degrees.
    - wall_values (list): <br>
        List of pixel values representing walls.
    - img_border_also_collide (bool, optional): <br>
        Whether the image border acts as a collider (default: False).
    - reflexion_order (int, optional): <br>
        Number of allowed reflections (default: 3).
    - should_scale (bool, optional): <br>
        Whether to normalize positions to [0, 1] (default: True).
    - should_return_iterative (bool, optional): <br>
        Whether to return a RayIterator for step-by-step analysis (default: False).
    - remove_iterative (bool, optional): <br>
        Whether to optimize (ignore) ray-iterator -> only in use if should_return_iterative is False.

    Returns:
    - list: 
        Nested list structure representing the traced ray and its reflections. 
        Format: ray[beam][point] = (x, y)
    """
    reflexion_order += 1  # Reflexion Order == 0 means, no reflections, therefore only 1 loop
    IMG_WIDTH, IMG_HEIGHT =  get_width_height(img)

    ray = []
    if should_return_iterative or not remove_iterative:
        ray_iterator = RayIterator()

    cur_target_abs_position = abs_position
    cur_direction_in_degree = direction_in_degree % 360

    for cur_depth in range(reflexion_order):
        # print(f"(Reflexion Order '{cur_depth}') {ray=}")
        if should_scale:
            current_ray_line = [normalize_point(x=cur_target_abs_position[0], y=cur_target_abs_position[1], width=IMG_WIDTH, height=IMG_HEIGHT)]
        else:
            current_ray_line = [(cur_target_abs_position[0], cur_target_abs_position[1])]
        if should_return_iterative or not remove_iterative:
            ray_iterator.add_iteration([copy.deepcopy(ray)+[current_ray_line]])

        last_abs_position = cur_target_abs_position

        # calculate a target line to update the pixels
        #   target vector
        dx = math.cos(math.radians(cur_direction_in_degree))
        dy = math.sin(math.radians(cur_direction_in_degree))
        target_line = [cur_target_abs_position[0], cur_target_abs_position[1], cur_target_abs_position[0], cur_target_abs_position[1]]
        while (0 <= target_line[2] <= IMG_WIDTH) and (0 <= target_line[3] <= IMG_HEIGHT):
            target_line[2] += 0.01 * dx
            target_line[3] += 0.01 * dy

        # update current ray
        current_position = cur_target_abs_position
        while True:
            # update position
            current_position = update_pixel_position(direction_in_degree=cur_direction_in_degree, cur_position=current_position, target_line=target_line)
        # for current_position in get_all_pixel_coordinates_in_between(current_position[0], current_position[1], target_line[2], target_line[3]):
        #     last_position_saved = False

            # check if ray is at end
            if not (0 <= current_position[0] < IMG_WIDTH and 0 <= current_position[1] < IMG_HEIGHT):
                ray += [current_ray_line]

                if img_border_also_collide:
                     # get reflection angle
                    wall_vector = get_img_border_vector(position=current_position, 
                                                           max_width=IMG_WIDTH, 
                                                           max_height=IMG_HEIGHT)

                    # calc new direct vector
                    new_direction = calc_reflection(collide_vector=degree_to_vector(cur_direction_in_degree), wall_vector=wall_vector)
                    new_direction_in_degree = vector_to_degree(new_direction[0], new_direction[1])

                    # start new beam calculation
                    cur_target_abs_position = last_abs_position
                    cur_direction_in_degree = new_direction_in_degree
                    break
                else:
                    if should_return_iterative:
                        return ray_iterator
                    return ray

            next_pixel = img[int(current_position[1]), int(current_position[0])]

            # check if hit building
            if float(next_pixel) in wall_values:
                last_abs_position = (current_position[0], current_position[1])
                ray += [current_ray_line]

                # get building wall reflection angle
                building_angle = wall_map[int(current_position[1]), int(current_position[0])]
                if building_angle == np.inf:
                    raise Exception("Got inf value from Wall-Map.")
                wall_vector = degree_to_vector(building_angle)

                # calc new direct vector
                new_direction = calc_reflection(collide_vector=degree_to_vector(cur_direction_in_degree), wall_vector=wall_vector)
                new_direction_in_degree = vector_to_degree(new_direction[0], new_direction[1])

                # start new beam calculation
                cur_target_abs_position = last_abs_position
                cur_direction_in_degree = new_direction_in_degree
                break
            else:
                # update current ray
                if should_scale:
                    current_ray_line += [normalize_point(x=current_position[0], y=current_position[1], width=IMG_WIDTH, height=IMG_HEIGHT)]
                else:
                    current_ray_line += [(current_position[0], current_position[1])]
                if should_return_iterative or not remove_iterative:
                    ray_iterator.add_iteration([copy.deepcopy(ray)+[current_ray_line]])
                last_abs_position = (current_position[0], current_position[1])
    
    if should_return_iterative:
        return ray_iterator
    return ray
    


def trace_beam_with_DDA(abs_position, 
               img, 
               direction_in_degree, 
               wall_map,
               wall_values, 
               img_border_also_collide=False,
               reflexion_order=3,
               should_scale=True,
               should_return_iterative=False,
               remove_iterative=True):
    """
    Trace a ray (beam) through a 2D image using a DDA (Digital Differential Analyzer)
    algorithm with precise collision points and physically accurate reflections.

    The beam starts at a given floating-point position and marches through the grid
    until it intersects a wall or exits the image. For each collision, the exact
    hit position is computed using the ray Parameters t_hit, ensuring that reflected
    segments contain meaningful geometry rather than single-point artifacts.
    Reflections are computed using wall normals derived from the `wall_map`.

    Parameters:
    - abs_position (tuple of float): <br>
        Starting position (x, y) of the beam in absolute pixel space.
    - img (numpy.ndarray): <br>
        2D array representing the scene. Pixel values listed in `wall_values`
        are treated as solid walls.
    - direction_in_degree (float): <br>
        Initial direction of the beam in degrees (0° = right, 90° = down).
    - wall_map (numpy.ndarray): <br>
        A map storing wall orientations in degrees for each pixel marked as a wall.
        These angles define the wall normals used for reflection.
    - wall_values (list, tuple, set, float, optional): <br>
        Pixel values identifying walls. Any pixel in this list causes a collision.
        If None, pixel value 0.0 is treated as a wall.
    - img_border_also_collide (bool, optional): <br>
        If True, the image borders behave like reflective walls. If False,
        the ray terminates when leaving the image. Default: False.
    - reflexion_order (int, optional): <br>
        Maximum number of reflections. The ray can rebound this many times before
        the function terminates. Default: 3.
    - should_scale (bool, optional): <br>
        If True, all emitted points (x, y) are normalized to [0, 1] range.
        Otherwise absolute pixel positions are returned. Default: True.
    - should_return_iterative (bool, optional): <br>
        Whether to return a RayIterator for step-by-step analysis (default: False).
    - remove_iterative (bool, optional): <br>
        Whether to optimize (ignore) ray-iterator -> only in use if should_return_iterative is False.

    Returns:
    - list: 
        Nested list structure representing the traced ray and its reflections. 
        Format: ray[beam][point] = (x, y)
    """
    reflexion_order += 1
    IMG_WIDTH, IMG_HEIGHT = get_width_height(img)

    ray = []
    if should_return_iterative or not remove_iterative:
        ray_iterator = RayIterator()

    cur_target_abs_position = (float(abs_position[0]), float(abs_position[1]))
    cur_direction_in_degree = direction_in_degree % 360

    # Normalize wall_values to set of floats
    if wall_values is None:
        wall_values_set = {0.0}
    elif isinstance(wall_values, (list, tuple, set)):
        for idx, v in enumerate(wall_values):
            if idx == 0:
                wall_values_set = {float(v)}
            else:
                wall_values_set.add(float(v))
    else:
        wall_values_set = {float(wall_values)}

    # go through every reflection -> will early stop if hitting a wall (if wall-bouncing is deactivated)
    for cur_depth in range(reflexion_order):
        if should_scale:
            current_ray_line = [normalize_point(x=cur_target_abs_position[0], y=cur_target_abs_position[1], width=IMG_WIDTH, height=IMG_HEIGHT)]
        else:
            current_ray_line = [(cur_target_abs_position[0], cur_target_abs_position[1])]
        if should_return_iterative or not remove_iterative:
            ray_iterator.add_iteration([copy.deepcopy(ray)+[current_ray_line]])

        last_abs_position = cur_target_abs_position

        # direction
        rad = math.radians(cur_direction_in_degree)
        dx = math.cos(rad)
        dy = math.sin(rad)

        eps = 1e-12
        if abs(dx) < eps: dx = 0.0
        if abs(dy) < eps: dy = 0.0

        # start float pos and starting cell
        x0 = float(cur_target_abs_position[0])
        y0 = float(cur_target_abs_position[1])
        cell_x = int(math.floor(x0))
        cell_y = int(math.floor(y0))

        # outside start -> handle border/reflection/exit
        if not (0 <= x0 < IMG_WIDTH and 0 <= y0 < IMG_HEIGHT):
            ray += [current_ray_line]

            if img_border_also_collide:
                wall_vector = get_img_border_vector(position=(x0, y0), max_width=IMG_WIDTH, max_height=IMG_HEIGHT)
                new_direction = calc_reflection(collide_vector=degree_to_vector(cur_direction_in_degree), wall_vector=wall_vector)
                cur_direction_in_degree = vector_to_degree(new_direction[0], new_direction[1])
                cur_target_abs_position = last_abs_position
                continue
            else:
                if should_return_iterative:
                    return ray_iterator
                return ray

        # DDA parameters
        tDeltaX = math.inf if dx == 0.0 else abs(1.0 / dx)
        tDeltaY = math.inf if dy == 0.0 else abs(1.0 / dy)

        if dx > 0:
            stepX = 1
            nextBoundaryX = cell_x + 1.0
            tMaxX = (nextBoundaryX - x0) / dx if dx != 0 else math.inf
        elif dx < 0:
            stepX = -1
            nextBoundaryX = cell_x * 1.0  # left boundary of cell
            tMaxX = (nextBoundaryX - x0) / dx if dx != 0 else math.inf
        else:
            stepX = 0
            tMaxX = math.inf

        if dy > 0:
            stepY = 1
            nextBoundaryY = cell_y + 1.0
            tMaxY = (nextBoundaryY - y0) / dy if dy != 0 else math.inf
        elif dy < 0:
            stepY = -1
            nextBoundaryY = cell_y * 1.0
            tMaxY = (nextBoundaryY - y0) / dy if dy != 0 else math.inf
        else:
            stepY = 0
            tMaxY = math.inf

        max_steps = (IMG_WIDTH + IMG_HEIGHT) * 6
        steps = 0
        last_position_saved = False

        # immediate-start-in-wall handling
        if 0 <= cell_x < IMG_WIDTH and 0 <= cell_y < IMG_HEIGHT:
            start_pixel = float(img[cell_y, cell_x])
            if start_pixel in wall_values_set:
                # compute a collision point precisely at start (we'll use origin)
                # add collision point (start) and reflect
                hit_x = x0
                hit_y = y0
                if should_scale:
                    current_ray_line += [normalize_point(x=hit_x, y=hit_y, width=IMG_WIDTH, height=IMG_HEIGHT)]
                else:
                    current_ray_line += [(hit_x, hit_y)]
                if should_return_iterative or not remove_iterative:
                    ray_iterator.add_iteration([copy.deepcopy(ray)+[current_ray_line]])
                ray += [current_ray_line]

                building_angle = float(wall_map[cell_y, cell_x])
                if not np.isfinite(building_angle):
                    raise Exception("Got non-finite value from Wall-Map.")
                wall_vector = degree_to_vector(building_angle)
                new_direction = calc_reflection(collide_vector=degree_to_vector(cur_direction_in_degree), wall_vector=wall_vector)
                cur_direction_in_degree = vector_to_degree(new_direction[0], new_direction[1])
                ndx, ndy = new_direction[0], new_direction[1]
                cur_target_abs_position = (hit_x + ndx * 1e-3, hit_y + ndy * 1e-3)
                continue

        # DDA main loop
        while steps < max_steps:
            steps += 1

            # choose axis to step and capture t_hit (distance along ray to boundary)
            if tMaxX < tMaxY:
                t_hit = tMaxX
                # step in x
                cell_x += stepX
                tMaxX += tDeltaX
            else:
                t_hit = tMaxY
                # step in y
                cell_y += stepY
                tMaxY += tDeltaY

            # compute exact collision position along ray from origin (x0,y0)
            hit_x = x0 + dx * t_hit
            hit_y = y0 + dy * t_hit

            # For recording the traversal we can append intermediate cell centers encountered so far.
            # But more importantly, append the collision point to the current segment BEFORE storing it.
            if should_scale:
                current_ray_line += [normalize_point(x=hit_x, y=hit_y, width=IMG_WIDTH, height=IMG_HEIGHT)]
            else:
                current_ray_line += [(hit_x, hit_y)]
            if should_return_iterative or not remove_iterative:
                ray_iterator.add_iteration([copy.deepcopy(ray)+[current_ray_line]])

            # Now check if we've left the image bounds (cell_x, cell_y refer to the new cell we stepped into)
            if not (0 <= cell_x < IMG_WIDTH and 0 <= cell_y < IMG_HEIGHT):
                ray += [current_ray_line]
                last_position_saved = True

                if img_border_also_collide:
                    wall_vector = get_img_border_vector(position=(cell_x, cell_y), max_width=IMG_WIDTH, max_height=IMG_HEIGHT)
                    new_direction = calc_reflection(collide_vector=degree_to_vector(cur_direction_in_degree), wall_vector=wall_vector)
                    new_direction_in_degree = vector_to_degree(new_direction[0], new_direction[1])
                    # start next ray from last in-image position (hit_x, hit_y) nudged slightly
                    ndx, ndy = new_direction[0], new_direction[1]
                    cur_target_abs_position = (hit_x + ndx * 1e-3, hit_y + ndy * 1e-3)
                    cur_direction_in_degree = new_direction_in_degree
                    break
                else:
                    if should_return_iterative:
                        return ray_iterator
                    return ray

            # sample the pixel in the cell we stepped into
            next_pixel = float(img[cell_y, cell_x])
            if next_pixel in wall_values_set:
                # we hit a wall — collision point already appended
                last_abs_position = (hit_x, hit_y)
                ray += [current_ray_line]
                last_position_saved = True

                building_angle = float(wall_map[cell_y, cell_x])
                if not np.isfinite(building_angle):
                    raise Exception("Got non-finite value from Wall-Map.")
                wall_vector = degree_to_vector(building_angle)

                new_direction = calc_reflection(collide_vector=degree_to_vector(cur_direction_in_degree), wall_vector=wall_vector)
                new_direction_in_degree = vector_to_degree(new_direction[0], new_direction[1])

                # start next beam from collision point nudged outwards
                ndx, ndy = new_direction[0], new_direction[1]
                cur_target_abs_position = (hit_x + ndx * 1e-3, hit_y + ndy * 1e-3)
                cur_direction_in_degree = new_direction_in_degree
                break
            else:
                # no hit -> continue marching; also add a representative point in the traversed cell (optional)
                # we already appended the exact hit point for this step; for smoother lines you may add cell center too
                last_abs_position = (hit_x, hit_y)
                # continue

        # end DDA loop
        if not last_position_saved:
            ray.append(current_ray_line)

    if should_return_iterative:
        return ray_iterator
    return ray



def trace_beams(rel_position, 
                img_src,
                directions_in_degree, 
                wall_values, 
                wall_thickness=0,
                img_border_also_collide=False,
                reflexion_order=3,
                should_scale_rays=True,
                should_scale_img=True,
                use_dda=True,
                iterative_tracking=False,
                iterative_steps=None,
                parallelization=0,
                parallelization_method="processes",  # "threads", "processes"
                use_numba_compilation=False,
                ignore_iterative_optimization=True):
    """
    Trace multiple rays (beams) from a single position through an image with walls and reflections.

    Each beam starts from a given relative position and follows its assigned direction
    until it collides with a wall or image border. On collisions, reflections are
    computed based on local wall normals extracted from the image.

    Parameters:
    - rel_position (tuple): <br>
        Relative starting position (x, y) in normalized coordinates [0-1].
    - img_src (str or numpy.ndarray): <br>
        Input image (array or file path) used for wall detection.
    - directions_in_degree (list): <br>
        List of beam direction angles (in degrees).
    - wall_values (list or float or None): <br>
        Pixel values representing walls or obstacles.
    - wall_thickness (int, optional): <br>
        Thickness (in pixels) of detected walls (default: 0).
    - img_border_also_collide (bool, optional): <br>
        Whether image borders act as colliders (default: False).
    - reflexion_order (int, optional): <br>
        Number of allowed reflections per beam (default: 3).
    - should_scale_rays (bool, optional): <br>
        Whether to normalize ray coordinates to [0, 1] (default: True).
    - should_scale_img (bool, optional): <br>
        Whether to scale the input image before wall detection (default: True).
    - use_dda (bool, optional): <br>
        Whether to use the DDA-based ray tracing method (default: True).
    - iterative_tracking (bool, optional): 
        Whether to return a RayIterator for step-by-step analysis (default: False).
    - iterative_steps (int, optional):<br>
        Number of steps for iterative reduction if using iterative tracking. `None` for all steps.
    - parallelization (int, optional):<br>
        The amount of workers for parallelization. 0 for no parallelization, -1 for max amount of workers.
    - parallelization_method (str, optional):<br>
        Method to use for parallelization (as soft condition) -> "threads" or "processes"
    - use_numba_compilation (bool, optional):<br>
        Whether to use the compiled (to machine code) version of compute heavy functions.
    - ignore_iterative_optimization (bool, optional): <br>
        Whether to used optimized ignore iteration data if iterative_tracking is False. 

    Returns:
    - list: <br>
        Nested list of traced beams and their reflection segments. 
        Format: rays[beam][segment][point] = (x, y)
    """ 
    if isinstance(img_src, np.ndarray):
        img = img_src
    else:
        img = img_open(src=img_src, should_scale=should_scale_img, should_print=False)
    IMG_WIDTH, IMG_HEIGHT =  get_width_height(img)
    abs_position = (rel_position[0] * IMG_WIDTH, rel_position[1] * IMG_HEIGHT)

    if wall_values is not None and type(wall_values) not in [list, tuple]:
        wall_values = [wall_values]
    wall_map = get_wall_map(img=img, 
                            wall_values=wall_values, 
                            thickness=wall_thickness,
                            use_numba_compilation=use_numba_compilation)
    
    if wall_values is None:
        wall_values = [0.0]

    # create function plan
    ray_planning = []

    for direction_in_degree in directions_in_degree:
        if use_dda:
            if use_numba_compilation:
                ray_tracing_func = trace_beam_with_DDA_numba
            else:
                ray_tracing_func = trace_beam_with_DDA
        else:
            if use_numba_compilation:
                ray_tracing_func = trace_beam_numba
            else:
                ray_tracing_func = trace_beam

        ray_planning.append(lambda dir=direction_in_degree: ray_tracing_func(
                                        abs_position=abs_position, 
                                        img=img,  
                                        direction_in_degree=dir,
                                        wall_map=wall_map,
                                        wall_values=np.array(wall_values, dtype=np.float64) if (use_numba_compilation and use_dda) else wall_values,
                                        img_border_also_collide=img_border_also_collide, 
                                        reflexion_order=reflexion_order,
                                        should_scale=should_scale_rays,
                                        should_return_iterative=iterative_tracking,
                                        remove_iterative=ignore_iterative_optimization
                                    )
        )
        
    if iterative_tracking:
        rays = RayIterator()
    else:
        rays = []

    # compute
    if parallelization == 0:

        for cur_ray_planning in ray_planning:
            cur_ray_result = cur_ray_planning()
            if use_numba_compilation:
                cur_ray_result = numba_to_py(cur_ray_result)
        
            if iterative_tracking:
                rays.add_rays(cur_ray_result)
            else:
                rays.append(cur_ray_result)
    else:
        result_rays = Parallel(n_jobs=parallelization, 
                                # backend="threading" if use_numba_compilation else "loky",     # process-based
                                prefer="threads" if use_numba_compilation else parallelization_method,
                                return_as="generator",
                                batch_size=1        # because of unequal ray lengths
                                )(
                                    delayed(ray_func)() for ray_func in ray_planning
                                )
        
        for cur_ray_result in result_rays:
            if use_numba_compilation:
                cur_ray_result = numba_to_py(cur_ray_result)

            if iterative_tracking:
                rays.add_rays(cur_ray_result)
            else:
                rays.append(cur_ray_result)

    if iterative_tracking and iterative_steps is not None:
        rays.reduce_rays_iteratively(steps=iterative_steps)

    return rays



def scale_rays(rays, 
               max_x=None, max_y=None, 
               new_max_x=None, new_max_y=None,
               detailed_scaling=True):
    """
    Scale ray coordinates between coordinate systems or image resolutions.

    Optionally normalizes rays by old dimensions and rescales them to new ones.
    Can scale all points or just the start/end points of each beam.

    Parameters:
    - rays (list): <br>
        Nested list of rays in the format rays[ray][beam][point] = (x, y).
    - max_x (float, optional): <br>
        Original maximum x-value for normalization.
    - max_y (float, optional): <br>
        Original maximum y-value for normalization.
    - new_max_x (float, optional): <br>
        New maximum x-value after rescaling.
    - new_max_y (float, optional): <br>
        New maximum y-value after rescaling.
    - detailed_scaling (bool, optional): <br>
        If True, scale every point in a beam; otherwise, only endpoints (default: True).

    Returns:
    - list: <br>
        Scaled rays in the same nested format.
    """
    if isinstance(rays, RayIterator):
        rays.apply_and_update(lambda r: scale_rays(r, max_x=max_x, max_y=max_y, new_max_x=new_max_x, new_max_y=new_max_y, detailed_scaling=detailed_scaling))
        return rays

    scaled_rays = []
    for ray in rays:
        scaled_ray = []
        for beams in ray:
            new_beams = copy.deepcopy(beams)
            if detailed_scaling:
                idx_to_process = list(range(len(beams)))
            else:
                idx_to_process = [0, len(beams)-1]

            for idx in idx_to_process:
                x1, y1 = beams[idx] 

                if max_x is not None and max_y is not None:
                    x1 /= max_x
                    y1 /= max_y

                from_cache = (x1, y1)
                if new_max_x is not None and new_max_y is not None:
                    if x1 >= new_max_x/2:
                        print(f"[WARNING] Detected high values scaling. Are you sure you want to scale for example a ray with {x1} to a value like {x1*new_max_x}?")
                    if y1 >= new_max_y/2:
                        print(f"[WARNING] Detected high values scaling. Are you sure you want to scale for example a ray with {y1} to a value like {y1*new_max_y}?")
                    
                    x1 *= new_max_x
                    y1 *= new_max_y

                new_beams[idx] = (x1, y1)

            scaled_ray.append(new_beams)
        scaled_rays.append(scaled_ray)
    return scaled_rays



def draw_rectangle_with_thickness(img, start_point, end_point, value, thickness=1):
    """
    Draw a filled or thick rectangle on an image.

    Expands the given start and end points based on the desired thickness and
    clips coordinates to image bounds to avoid overflow.

    Parameters:
    - img (numpy.ndarray): <br>
        Image array to draw on.
    - start_point (tuple): <br>
        Top-left corner of the rectangle (x, y).
    - end_point (tuple): <br>
        Bottom-right corner of the rectangle (x, y).
    - value (int or float): <br>
        Fill value or color intensity.
    - thickness (int, optional): <br>
        Rectangle border thickness; 
        if <= 0, the rectangle is filled (default: 1).
    """
    # Calculate the expansion -> "thickness"
    if thickness > 0:
        expand = thickness // 2
        x1, y1 = start_point[0] - expand, start_point[1] - expand
        x2, y2 = end_point[0] + expand, end_point[1] + expand
    else:
        # thickness <= 0 → filled rectangle, no expansion
        x1, y1 = start_point
        x2, y2 = end_point

    # Clip coordinates to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1]-1, x2)
    y2 = min(img.shape[0]-1, y2)

    cv2.rectangle(img, (x1, y1), (x2, y2), value, thickness=-1)



def draw_line_or_point(img, start_point, end_point, fill_value, thickness):
    """
    Draw either a line or a single point on an image.

    Determines whether to draw a point or a line based on whether the start and
    end coordinates are identical.

    Parameters:
    - img (numpy.ndarray): <br>
        Image array to draw on.
    - start_point (tuple): <br>
        Starting pixel coordinate (x, y).
    - end_point (tuple): <br>
        Ending pixel coordinate (x, y).
    - fill_value (int or float): <br>
        Value or color used for drawing.
    - thickness (int): <br>
        Thickness of the line or point.
    """
    draw_point = (start_point == end_point)

    if draw_point:
        draw_rectangle_with_thickness(img=img, start_point=start_point, end_point=end_point, value=fill_value, thickness=thickness)
    else:
        cv2.line(img, start_point, end_point, fill_value, thickness)



def draw_rays(rays, detail_draw=True,
              output_format="single_image", # single_image, multiple_images, channels 
              img_background=None, ray_value=255, ray_thickness=1, 
              img_shape=(256, 256), dtype=float, standard_value=0,
              should_scale_rays_to_image=True, original_max_width=None, original_max_height=None,
              show_only_reflections=False):
    """
    Render rays onto an image or a set of images.

    Each ray can be drawn in full detail (every point) or as straight lines between
    beam endpoints. Rays can be scaled to match image dimensions and drawn on a
    single image, multiple images, or separate channels.

    Parameters:
    - rays (list): <br>
        Nested list of rays in the format rays[ray][beam][point] = (x, y).
    - detail_draw (bool, optional): <br>
        Whether to draw every point or just beam endpoints (default: True).
    - output_format (str, optional): <br>
        Output mode: "single_image", "multiple_images", or "channels" (default: "single_image").
    - img_background (numpy.ndarray, optional): <br>
        Background image; if None, a blank image is created.
    - ray_value (int, float, list, or numpy.ndarray, optional): <br>
        Pixel intensity or color per reflection (default: 255).
    - ray_thickness (int, optional): <br>
        Thickness of the drawn lines or points (default: 1).
    - img_shape (tuple, optional): <br>
        Shape of the generated image if no background is given (default: (256, 256)).
    - dtype (type, optional): <br>
        Data type for the output image (default: float).
    - standard_value (int or float, optional): <br>
        Background fill value (default: 0).
    - should_scale_rays_to_image (bool, optional): <br>
        Whether to scale ray coordinates to match the image (default: True).
    - original_max_width (float, optional): <br>
        Original image width before scaling.
    - original_max_height (float, optional): <br>
        Original image height before scaling.
    - show_only_reflections (bool, optional): <br>
        If True, draws only reflected beams (default: False).

    Returns:
        numpy.ndarray or list: <br>
            - Single image if output_format == "single_image" or "channels".
            - List of images if output_format == "multiple_images".
    """
    if isinstance(rays, RayIterator):
        imgs = rays.apply_and_return(lambda r: draw_rays(r, detail_draw=detail_draw,
                                                            output_format=output_format,
                                                            img_background=img_background,
                                                            ray_value=ray_value,
                                                            ray_thickness=ray_thickness,
                                                            img_shape=img_shape,
                                                            dtype=dtype,
                                                            standard_value=standard_value,
                                                            should_scale_rays_to_image=should_scale_rays_to_image,
                                                            original_max_width=original_max_width,
                                                            original_max_height=original_max_height,
                                                            show_only_reflections=show_only_reflections))
        return imgs

    # prepare background image
    if img_background is None:
        img = np.full(shape=img_shape, fill_value=dtype(standard_value), dtype=dtype)
    else:
        img = img_background.copy()

    # rescale rays to fit inside image bounds if desired
    height, width = img.shape[:2]
    # print(f"{height, width=}")
    # print(f"{(original_max_width, original_max_height)=}")
    if should_scale_rays_to_image:
        rays = scale_rays(rays, max_x=original_max_width, max_y=original_max_height, new_max_x=width-1, new_max_y=height-1, detailed_scaling=detail_draw)

    nrays = len(rays)
    if output_format == "channels":
        img = np.repeat(img[..., None], nrays, axis=-1)
        # img_shape += (nrays, )
        # img = np.full(shape=img_shape, fill_value=dtype(standard_value), dtype=dtype)
    elif output_format == "multiple_images":
        imgs = [np.copy(img) for _ in range(nrays)]

    # draw on image
    for idx, ray in enumerate(rays):
        for reflexion_order, beam_points in enumerate(ray):
            
            if detail_draw:
                lines = []
                for cur_point in range(0, len(beam_points)):
                    start_point = tuple(map(lambda x:int(x), beam_points[cur_point]))
                    end_point = tuple(map(lambda x:int(x), beam_points[cur_point]))
                    # end_point = tuple(map(lambda x:int(x), beam_points[cur_point+1]))
                    # -> if as small lines then in range: range(0, len(beam_points)-1)
                    lines.append((start_point, end_point, reflexion_order))
            else:
                start_point = tuple(map(lambda x:int(x), beam_points[0]))
                end_point = tuple(map(lambda x:int(x), beam_points[-1]))
                lines = [(start_point, end_point, reflexion_order)]
            
            for start_point, end_point, reflexion_order in lines:

                if show_only_reflections and reflexion_order == 0:
                    continue

                # get cur ray value
                if type(ray_value) in [list, tuple, np.ndarray]:
                    # if we print without the first line: first value (index 0) belkongs to the reflexion order 1
                    cur_reflexion_index = reflexion_order-1 if show_only_reflections else reflexion_order
                    cur_ray_value = ray_value[min(len(ray_value)-1, cur_reflexion_index)]
                else:
                    cur_ray_value = ray_value

                if output_format == "channels":
                    layer = np.ascontiguousarray(img[..., idx])
                    draw_line_or_point(img=layer, start_point=start_point, end_point=end_point, fill_value=cur_ray_value, thickness=ray_thickness)
                    img[..., idx] = layer
                elif output_format == "multiple_images":
                    draw_line_or_point(img=imgs[idx], start_point=start_point, end_point=end_point, fill_value=cur_ray_value, thickness=ray_thickness)
                else:
                    draw_line_or_point(img=img, start_point=start_point, end_point=end_point, fill_value=cur_ray_value, thickness=ray_thickness)


    if output_format == "multiple_images":
        return imgs
    else:
        return img



def trace_and_draw_rays(rel_position, 
                        img_src,
                        directions_in_degree, 
                        wall_values, 
                        wall_thickness=0,
                        img_border_also_collide=False,
                        reflexion_order=3,
                        should_scale_rays=True,
                        should_scale_img=True,
                        use_dda=True,
                        iterative_tracking=False,
                        iterative_steps=None,
                        parallelization=0,
                        parallelization_method="processes",  # "threads", "processes"
                        use_numba_compilation=False,
                        ignore_iterative_optimization=True,
                        detail_draw=True,
                        output_format="single_image", # single_image, multiple_images, channels 
                        img_background=None, 
                        ray_value=255, 
                        ray_thickness=1, 
                        img_shape=(256, 256), 
                        dtype=float, 
                        standard_value=0,
                        should_scale_rays_to_image=True, 
                        original_max_width=None, 
                        original_max_height=None,
                        show_only_reflections=False):
    """
    Trace multiple rays (beams) from a single position through an image with walls and reflections AND 
    render rays onto an image or a set of images.

    Calls internally `trace_beams` and then `draw_rays`.

    Parameters:
    - rel_position (tuple): <br>
        Relative starting position (x, y) in normalized coordinates [0-1].
    - img_src (str or numpy.ndarray): <br>
        Input image (array or file path) used for wall detection.
    - directions_in_degree (list): <br>
        List of beam direction angles (in degrees).
    - wall_values (list or float or None): <br>
        Pixel values representing walls or obstacles.
    - wall_thickness (int, optional): <br>
        Thickness (in pixels) of detected walls (default: 0).
    - img_border_also_collide (bool, optional): <br>
        Whether image borders act as colliders (default: False).
    - reflexion_order (int, optional): <br>
        Number of allowed reflections per beam (default: 3).
    - should_scale_rays (bool, optional): <br>
        Whether to normalize ray coordinates to [0, 1] (default: True).
    - should_scale_img (bool, optional): <br>
        Whether to scale the input image before wall detection (default: True).
    - use_dda (bool, optional): <br>
        Whether to use the DDA-based ray tracing method (default: True).
    - iterative_tracking (bool, optional): 
        Whether to return a RayIterator for step-by-step analysis (default: False).
    - iterative_steps (int, optional):<br>
        Number of steps for iterative reduction if using iterative tracking. `None` for all steps.
    - parallelization (int, optional):<br>
        The amount of workers for parallelization. 0 for no parallelization, -1 for max amount of workers.
    - parallelization_method (str, optional):<br>
        Method to use for parallelization (as soft condition) -> "threads" or "processes"
    - use_numba_compilation (bool, optional):<br>
        Whether to use the compiled (to machine code) version of compute heavy functions.
    - ignore_iterative_optimization (bool, optional): <br>
        Whether to used optimized ignore iteration data if iterative_tracking is False. 
    - detail_draw (bool, optional): <br>
        Whether to draw every point or just beam endpoints (default: True).
    - output_format (str, optional): <br>
        Output mode: "single_image", "multiple_images", or "channels" (default: "single_image").
    - img_background (numpy.ndarray, optional): <br>
        Background image; if None, a blank image is created.
    - ray_value (int, float, list, or numpy.ndarray, optional): <br>
        Pixel intensity or color per reflection (default: 255).
    - ray_thickness (int, optional): <br>
        Thickness of the drawn lines or points (default: 1).
    - img_shape (tuple, optional): <br>
        Shape of the generated image if no background is given (default: (256, 256)).
    - dtype (type, optional): <br>
        Data type for the output image (default: float).
    - standard_value (int or float, optional): <br>
        Background fill value (default: 0).
    - should_scale_rays_to_image (bool, optional): <br>
        Whether to scale ray coordinates to match the image (default: True).
    - original_max_width (float, optional): <br>
        Original image width before scaling.
    - original_max_height (float, optional): <br>
        Original image height before scaling.
    - show_only_reflections (bool, optional): <br>
        If True, draws only reflected beams (default: False).

    Returns:
        numpy.ndarray or list:
            - Single image if output_format == "single_image" or "channels".
            - List of images if output_format == "multiple_images".
    """
    rays = trace_beams(rel_position, 
                        img_src,
                        directions_in_degree, 
                        wall_values, 
                        wall_thickness=wall_thickness,
                        img_border_also_collide=img_border_also_collide,
                        reflexion_order=reflexion_order,
                        should_scale_rays=should_scale_rays,
                        should_scale_img=should_scale_img,
                        use_dda=use_dda,
                        iterative_tracking=iterative_tracking,
                        iterative_steps=iterative_steps,
                        parallelization=parallelization,
                        parallelization_method=parallelization_method,
                        use_numba_compilation=use_numba_compilation,
                        ignore_iterative_optimization=ignore_iterative_optimization)
    
    return draw_rays(rays, 
                     detail_draw=detail_draw,
                     output_format=output_format,
                     img_background=img_background, 
                     ray_value=ray_value, 
                     ray_thickness=ray_thickness, 
                     img_shape=img_shape, 
                     dtype=dtype, 
                     standard_value=standard_value,
                     should_scale_rays_to_image=should_scale_rays_to_image, 
                     original_max_width=original_max_width, 
                     original_max_height=original_max_height,
                     show_only_reflections=show_only_reflections)



# --------------------------------
# >>> Numba Optimized Versions <<<
# --------------------------------
def numba_to_py(obj):
    """
    Converts numba.typed.List (possibly nested -> rays) to plain Python lists/tuples.
    """
    if isinstance(obj, List):
        return [numba_to_py(x) for x in obj]

    
    if isinstance(obj, tuple):
        return tuple(numba_to_py(x) for x in obj)

    return obj


TYPE_POINT_INT = types.UniTuple(types.int64, 2)
TYPE_POINT_FLOAT = types.UniTuple(types.float64, 2)
TYPE_LINE = types.ListType(TYPE_POINT_FLOAT)

# trace_beam_numba = numba.njit(cache=True, fastmath=True)(trace_beam)
# trace_beam_with_DDA_numba = numba.njit(cache=True, fastmath=True)(trace_beam_with_DDA)
calc_reflection_numba = numba.njit(cache=True, fastmath=True)(calc_reflection)
# get_all_pixel_coordinates_in_between_numba = numba.njit(cache=True, fastmath=True)(get_all_pixel_coordinates_in_between)
get_img_border_vector_numba = numba.njit(cache=True, fastmath=True)(get_img_border_vector)
# update_pixel_position_numba = numba.njit(cache=True, fastmath=True)(update_pixel_position)



@numba.njit(cache=True, fastmath=True)
def update_pixel_position_numba(direction_in_degree, cur_position, target_line):
    # cur_position: (x,y) tuple
    # target_line: array[4] => [x1,y1,x2,y2]
    px = cur_position[0]
    py = cur_position[1]

    ax = target_line[0]
    ay = target_line[1]
    bx = target_line[2]
    by = target_line[3]

    # Directionvector of the line
    vx = bx - ax
    vy = by - ay

    # Projection: t = dot(P-A, V) / dot(V,V)
    wx = px - ax
    wy = py - ay

    denom = vx*vx + vy*vy
    if denom == 0.0:
        # Degenerated Linie: just no movement
        return (px, py)

    t = (wx*vx + wy*vy) / denom

    # clip scalar (Numba-safe)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    # Point on the line
    lx = ax + t * vx
    ly = ay + t * vy

    # Now make a step in direction_in_degree
    rad = math.radians(direction_in_degree)
    dx = math.cos(rad)
    dy = math.sin(rad)

    nx = lx + dx
    ny = ly + dy

    return (nx, ny)



@numba.njit(cache=True, fastmath=True)
def _is_in_wall_values(pixel_value, wall_values):
    # Numba-save Membership-Check -> works with np.ndarray or numba List
    for i in range(len(wall_values)):
        if pixel_value == wall_values[i]:
            return True
    return False



@numba.njit(cache=True, fastmath=True)
def _compute_target_line_outside(x0, y0, dx, dy, w, h):
    # w, h als float
    # Candidats: Intersection with vertical/horizontal borders
    # Taking the smallest t and continue going 0.01
    INF = 1e30

    # X-Border
    if dx > 0.0:
        tx = (w - x0) / dx
    elif dx < 0.0:
        tx = (0.0 - x0) / dx
    else:
        tx = INF

    # Y-Border
    if dy > 0.0:
        ty = (h - y0) / dy
    elif dy < 0.0:
        ty = (0.0 - y0) / dy
    else:
        ty = INF

    # t have to be positive
    if tx <= 0.0:
        tx = INF
    if ty <= 0.0:
        ty = INF

    t_hit = tx if tx < ty else ty
    if t_hit == INF:
        # Direction is 0-vector (should not happen)
        t_hit = 0.0

    # the end point should be stay outside of the building/wall
    t_out = t_hit + 0.01

    x2 = x0 + t_out * dx
    y2 = y0 + t_out * dy

    return x0, y0, x2, y2



@numba.njit(cache=True, fastmath=True)
def trace_beam_numba(abs_position,
                     img,
                     direction_in_degree,
                     wall_map,
                     wall_values,
                     img_border_also_collide=False,
                     reflexion_order=3,
                     should_scale=True,
                     should_return_iterative=False,
                     remove_iterative=True):
    """
    Trace a ray (beam) through an image with walls and reflections.

    The beam starts from a given position and follows a direction until it hits
    a wall or border. On collisions, reflections are computed using wall normals.

    Parameters:
    - abs_position (tuple): <br>
        Starting position (x, y) of the beam.
    - img (numpy.ndarray): <br>
        Input image or segmentation map.
    - direction_in_degree (float): <br>
        Initial direction angle of the beam.
    - wall_map (numpy.ndarray): <br>
        Map containing wall orientations in degrees.
    - wall_values (list): <br>
        List of pixel values representing walls.
    - img_border_also_collide (bool, optional): <br>
        Whether the image border acts as a collider (default: False).
    - reflexion_order (int, optional): <br>
        Number of allowed reflections (default: 3).
    - should_scale (bool, optional): <br>
        Whether to normalize positions to [0, 1] (default: True).
    - should_return_iterative (bool, optional): <br>
        Whether to return a RayIterator for step-by-step analysis (default: False).
    - ignore_iterative_optimization (bool, optional): <br>
        Whether to used optimized ignore iteration data if iterative_tracking is False. 
    - remove_iterative (bool, optional): <br>
        Ignored in this numba version.

    Returns:
    - list: 
        Nested list structure representing the traced ray and its reflections. 
        Format: ray[beam][point] = (x, y)
    """
    if should_return_iterative:
        print("[WARNING] Numba Version can't return a iterative version.")

    reflexion_order += 1
    IMG_HEIGHT = img.shape[0]
    IMG_WIDTH  = img.shape[1]

    ray = List.empty_list(TYPE_LINE)

    cur_target_abs_position = (float(abs_position[0]), float(abs_position[1]))
    cur_direction_in_degree = float(direction_in_degree % 360.0)

    w_f = float(IMG_WIDTH)
    h_f = float(IMG_HEIGHT)

    inf_val = np.inf

    for _ in range(reflexion_order):
        # current_ray_line als typed list
        current_ray_line = List.empty_list(TYPE_POINT_FLOAT)

        if should_scale:
            p = normalize_point_numba(x=cur_target_abs_position[0],
                                 y=cur_target_abs_position[1],
                                 width=IMG_WIDTH,
                                 height=IMG_HEIGHT)
            current_ray_line.append((float(p[0]), float(p[1])))
        else:
            current_ray_line.append((cur_target_abs_position[0], cur_target_abs_position[1]))

        last_abs_position = cur_target_abs_position

        # Direction -> dx,dy
        rad = math.radians(cur_direction_in_degree)
        dx = math.cos(rad)
        dy = math.sin(rad)

        # target_line
        x1, y1, x2, y2 = _compute_target_line_outside(cur_target_abs_position[0],
                                                      cur_target_abs_position[1],
                                                      dx, dy, w_f, h_f)
        # build target line -> start and end point
        target_line = np.empty(4, dtype=np.float64)
        target_line[0] = x1
        target_line[1] = y1
        target_line[2] = x2
        target_line[3] = y2

        current_position = cur_target_abs_position

        while True:
            current_position = update_pixel_position_numba(
                direction_in_degree=cur_direction_in_degree,
                cur_position=current_position,
                target_line=target_line
            )

            x = current_position[0]
            y = current_position[1]

            # Border check
            if not (0.0 <= x < w_f and 0.0 <= y < h_f):
                ray.append(current_ray_line)

                if img_border_also_collide:
                    wall_vector = get_img_border_vector_numba(
                        position=current_position,
                        max_width=IMG_WIDTH,
                        max_height=IMG_HEIGHT
                    )
                    new_direction = calc_reflection_numba(
                        collide_vector=degree_to_vector_numba(cur_direction_in_degree),
                        wall_vector=wall_vector
                    )
                    new_direction_in_degree = vector_to_degree_numba(new_direction[0], new_direction[1])

                    cur_target_abs_position = last_abs_position
                    cur_direction_in_degree = float(new_direction_in_degree)
                    break
                else:
                    return ray

            ix = int(x)
            iy = int(y)

            next_pixel = float(img[iy, ix])

            # Wall check (Numba-save)
            if _is_in_wall_values(next_pixel, wall_values):
                last_abs_position = (x, y)
                ray.append(current_ray_line)

                building_angle = wall_map[iy, ix]
                if building_angle == inf_val:
                    raise Exception("Got inf value from Wall-Map.")

                wall_vector = degree_to_vector_numba(building_angle)

                new_direction = calc_reflection_numba(
                    collide_vector=degree_to_vector_numba(cur_direction_in_degree),
                    wall_vector=wall_vector
                )
                new_direction_in_degree = vector_to_degree_numba(new_direction[0], new_direction[1])

                cur_target_abs_position = last_abs_position
                cur_direction_in_degree = float(new_direction_in_degree)
                break
            else:
                if should_scale:
                    p = normalize_point_numba(x=x, y=y, width=IMG_WIDTH, height=IMG_HEIGHT)
                    current_ray_line.append((float(p[0]), float(p[1])))
                else:
                    current_ray_line.append((x, y))
                last_abs_position = (x, y)

    return ray



@numba.njit(cache=True, fastmath=True)
def trace_beam_with_DDA_numba(abs_position, 
                                img, 
                                direction_in_degree, 
                                wall_map,
                                wall_values, 
                                img_border_also_collide=False,
                                reflexion_order=3,
                                should_scale=True,
                                should_return_iterative=False,
                                remove_iterative=True):
    """
    Trace a ray (beam) through a 2D image using a DDA (Digital Differential Analyzer)
    algorithm with precise collision points and physically accurate reflections.

    The beam starts at a given floating-point position and marches through the grid
    until it intersects a wall or exits the image. For each collision, the exact
    hit position is computed using the ray Parameters t_hit, ensuring that reflected
    segments contain meaningful geometry rather than single-point artifacts.
    Reflections are computed using wall normals derived from the `wall_map`.

    Parameters:
    - abs_position (tuple of float): <br>
        Starting position (x, y) of the beam in absolute pixel space.
    - img (numpy.ndarray): <br>
        2D array representing the scene. Pixel values listed in `wall_values`
        are treated as solid walls.
    - direction_in_degree (float): <br>
        Initial direction of the beam in degrees (0° = right, 90° = down).
    - wall_map (numpy.ndarray): <br>
        A map storing wall orientations in degrees for each pixel marked as a wall.
        These angles define the wall normals used for reflection.
    - wall_values (list, tuple, set, float, optional): <br>
        Pixel values identifying walls. Any pixel in this list causes a collision.
        If None, pixel value 0.0 is treated as a wall.
    - img_border_also_collide (bool, optional): <br>
        If True, the image borders behave like reflective walls. If False,
        the ray terminates when leaving the image. Default: False.
    - reflexion_order (int, optional): <br>
        Maximum number of reflections. The ray can rebound this many times before
        the function terminates. Default: 3.
    - should_scale (bool, optional): <br>
        If True, all emitted points (x, y) are normalized to [0, 1] range.
        Otherwise absolute pixel positions are returned. Default: True.
    - should_return_iterative (bool, optional): <br>
        Whether to return a RayIterator for step-by-step analysis (default: False).
    - remove_iterative (bool, optional): <br>
        Ignored in this numba version.

    Returns:
    - list: 
        Nested list structure representing the traced ray and its reflections. 
        Format: ray[beam][point] = (x, y)
    """
    if should_return_iterative:
        print("[WARNING] Numba Version can't return a iterative version.")

    reflexion_order += 1
    # IMG_WIDTH, IMG_HEIGHT = get_width_height(img)
    IMG_HEIGHT = img.shape[0]
    IMG_WIDTH  = img.shape[1]

    ray = []

    cur_target_abs_position = (float(abs_position[0]), float(abs_position[1]))
    cur_direction_in_degree = direction_in_degree % 360

    # go through every reflection -> will early stop if hitting a wall (if wall-bouncing is deactivated)
    for cur_depth in range(reflexion_order):
        if should_scale:
            current_ray_line = [normalize_point_numba(x=cur_target_abs_position[0], y=cur_target_abs_position[1], width=IMG_WIDTH, height=IMG_HEIGHT)]
        else:
            current_ray_line = [(cur_target_abs_position[0], cur_target_abs_position[1])]

        last_abs_position = cur_target_abs_position

        # direction
        rad = math.radians(cur_direction_in_degree)
        dx = math.cos(rad)
        dy = math.sin(rad)

        eps = 1e-12
        if abs(dx) < eps: dx = 0.0
        if abs(dy) < eps: dy = 0.0

        # start float pos and starting cell
        x0 = float(cur_target_abs_position[0])
        y0 = float(cur_target_abs_position[1])
        cell_x = int(math.floor(x0))
        cell_y = int(math.floor(y0))

        # outside start -> handle border/reflection/exit
        if not (0 <= x0 < IMG_WIDTH and 0 <= y0 < IMG_HEIGHT):
            ray.append(current_ray_line)
            if img_border_also_collide:
                wall_vector = get_img_border_vector_numba(position=(x0, y0), max_width=IMG_WIDTH, max_height=IMG_HEIGHT)
                new_direction = calc_reflection_numba(collide_vector=degree_to_vector_numba(cur_direction_in_degree), wall_vector=wall_vector)
                cur_direction_in_degree = vector_to_degree_numba(new_direction[0], new_direction[1])
                cur_target_abs_position = last_abs_position
                continue
            else:
                return ray

        # DDA parameters
        tDeltaX = math.inf if dx == 0.0 else abs(1.0 / dx)
        tDeltaY = math.inf if dy == 0.0 else abs(1.0 / dy)

        if dx > 0:
            stepX = 1
            nextBoundaryX = cell_x + 1.0
            tMaxX = (nextBoundaryX - x0) / dx if dx != 0 else math.inf
        elif dx < 0:
            stepX = -1
            nextBoundaryX = cell_x * 1.0  # left boundary of cell
            tMaxX = (nextBoundaryX - x0) / dx if dx != 0 else math.inf
        else:
            stepX = 0
            tMaxX = math.inf

        if dy > 0:
            stepY = 1
            nextBoundaryY = cell_y + 1.0
            tMaxY = (nextBoundaryY - y0) / dy if dy != 0 else math.inf
        elif dy < 0:
            stepY = -1
            nextBoundaryY = cell_y * 1.0
            tMaxY = (nextBoundaryY - y0) / dy if dy != 0 else math.inf
        else:
            stepY = 0
            tMaxY = math.inf

        max_steps = (IMG_WIDTH + IMG_HEIGHT) * 6
        steps = 0
        last_position_saved = False

        # immediate-start-in-wall handling
        if 0 <= cell_x < IMG_WIDTH and 0 <= cell_y < IMG_HEIGHT:
            start_pixel = float(img[cell_y, cell_x])
            # if start_pixel in wall_values_set:
            if _is_in_wall_values(start_pixel, wall_values):
                # compute a collision point precisely at start (we'll use origin)
                # add collision point (start) and reflect
                hit_x = x0
                hit_y = y0
                if should_scale:
                    current_ray_line.append(normalize_point_numba(x=hit_x, y=hit_y, width=IMG_WIDTH, height=IMG_HEIGHT))
                else:
                    current_ray_line.append((hit_x, hit_y))
                ray.append(current_ray_line)

                building_angle = float(wall_map[cell_y, cell_x])
                if not np.isfinite(building_angle):
                    raise Exception("Got non-finite value from Wall-Map.")
                wall_vector = degree_to_vector_numba(building_angle)
                new_direction = calc_reflection_numba(collide_vector=degree_to_vector_numba(cur_direction_in_degree), wall_vector=wall_vector)
                cur_direction_in_degree = vector_to_degree_numba(new_direction[0], new_direction[1])
                ndx, ndy = new_direction[0], new_direction[1]
                cur_target_abs_position = (hit_x + ndx * 1e-3, hit_y + ndy * 1e-3)
                continue

        # DDA main loop
        while steps < max_steps:
            steps += 1

            # choose axis to step and capture t_hit (distance along ray to boundary)
            if tMaxX < tMaxY:
                t_hit = tMaxX
                # step in x
                cell_x += stepX
                tMaxX += tDeltaX
                stepped_axis = 'x'
            else:
                t_hit = tMaxY
                # step in y
                cell_y += stepY
                tMaxY += tDeltaY
                stepped_axis = 'y'

            # compute exact collision position along ray from origin (x0,y0)
            hit_x = x0 + dx * t_hit
            hit_y = y0 + dy * t_hit

            # For recording the traversal we can append intermediate cell centers encountered so far.
            # But more importantly, append the collision point to the current segment BEFORE storing it.
            if should_scale:
                current_ray_line.append(normalize_point_numba(x=hit_x, y=hit_y, width=IMG_WIDTH, height=IMG_HEIGHT))
            else:
                current_ray_line.append((hit_x, hit_y))

            # Now check if we've left the image bounds (cell_x, cell_y refer to the new cell we stepped into)
            if not (0 <= cell_x < IMG_WIDTH and 0 <= cell_y < IMG_HEIGHT):
                ray.append(current_ray_line)
                last_position_saved = True

                if img_border_also_collide:
                    wall_vector = get_img_border_vector_numba(position=(cell_x, cell_y), max_width=IMG_WIDTH, max_height=IMG_HEIGHT)
                    new_direction = calc_reflection_numba(collide_vector=degree_to_vector_numba(cur_direction_in_degree), wall_vector=wall_vector)
                    new_direction_in_degree = vector_to_degree_numba(new_direction[0], new_direction[1])
                    # start next ray from last in-image position (hit_x, hit_y) nudged slightly
                    ndx, ndy = new_direction[0], new_direction[1]
                    cur_target_abs_position = (hit_x + ndx * 1e-3, hit_y + ndy * 1e-3)
                    cur_direction_in_degree = new_direction_in_degree
                    break
                else:
                    return ray

            # sample the pixel in the cell we stepped into
            next_pixel = float(img[cell_y, cell_x])
            if _is_in_wall_values(next_pixel, wall_values):
                # we hit a wall — collision point already appended
                last_abs_position = (hit_x, hit_y)
                ray.append(current_ray_line)
                last_position_saved = True

                building_angle = float(wall_map[cell_y, cell_x])
                if not np.isfinite(building_angle):
                    raise Exception("Got non-finite value from Wall-Map.")
                wall_vector = degree_to_vector_numba(building_angle)

                new_direction = calc_reflection_numba(collide_vector=degree_to_vector_numba(cur_direction_in_degree), wall_vector=wall_vector)
                new_direction_in_degree = vector_to_degree_numba(new_direction[0], new_direction[1])

                # start next beam from collision point nudged outwards
                ndx, ndy = new_direction[0], new_direction[1]
                cur_target_abs_position = (hit_x + ndx * 1e-3, hit_y + ndy * 1e-3)
                cur_direction_in_degree = new_direction_in_degree
                break
            else:
                # no hit -> continue marching; also add a representative point in the traversed cell (optional)
                # we already appended the exact hit point for this step; for smoother lines you may add cell center too
                last_abs_position = (hit_x, hit_y)
                # continue

        # end DDA loop
        if not last_position_saved:
            ray.append(current_ray_line)

    return ray



@numba.njit(cache=True, fastmath=True)
def get_all_pixel_coordinates_in_between_numba(x1, y1, x2, y2):
    """
    Get all pixel coordinates along a line between two points using Bresenham's algorithm.

    https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

    Parameters:
    - x1 (int): <br>
        Starting x-coordinate.
    - y1 (int): <br>
        Starting y-coordinate.
    - x2 (int): <br>
        Ending x-coordinate.
    - y2 (int): <br>
        Ending y-coordinate.

    Returns:
    - list: 
        List of (x, y) tuples representing all pixels between the start and end points
    """
    coordinates = List.empty_list(TYPE_POINT_INT)

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1

    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    if dx > dy:
        err = dx / 2.0
        while x != x2:
            coordinates.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            coordinates.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    coordinates.append((x2, y2))  # include the last point
    return list(coordinates)



