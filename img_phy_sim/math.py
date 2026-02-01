"""
**Mathematical and Geometric Utility Functions**

This module provides lightweight mathematical helper functions for 2D geometry
and angle-based computations. It is designed to support higher-level operations
such as beam tracing, image-based simulations, and coordinate normalization.

The functions focus on conversions between angular and Cartesian representations,
as well as normalization and denormalization of image coordinates.

Main features:
- Generate evenly spaced degree ranges for directional sampling
- Convert between degrees and 2D unit vectors
- Convert 2D vectors back to degree angles
- Normalize and denormalize 2D points for image-based coordinate systems

Typical use cases:
- Generating input directions for beam tracing or ray simulation
- Converting between angular and vector representations in geometric algorithms
- Preparing coordinates for normalized image processing workflows

Dependencies:
- math
- numpy

Example:
```python
import img_phy_sim as ips

# Generate sample directions
directions = ips.math.get_linear_degree_range(step_size=45)

# Convert each to a 2D vector
vectors = [ips.math.degree_to_vector(d) for d in directions]

# Convert back to degrees
recovered = [ips.math.vector_to_degree(v) for v in vectors]

# Normalize and denormalize a point
p_norm = ips.math.normalize_point((128, 64), width=256, height=128)
p_pixel = ips.math.denormalize_point(p_norm, width=256, height=128)
```

Author:<br>
Tobia Ippolito

Functions:
- get_linear_degree_range(...) - Generate evenly spaced degrees within a range.
- degree_to_vector(...)        - Convert a degree angle to a 2D unit vector.
- vector_to_degree(...)        - Convert a 2D vector into its corresponding degree.
- normalize_point(...)         - Normalize a 2D point to [0, 1] range.
- denormalize_point(...)       - Denormalize a 2D point to pixel coordinates.
- numpy_info(...)              - Get statistics about an numpy array.
"""



# ---------------
# >>> Imports <<<
# ---------------
import math
import numpy as np

import numba



# -----------------
# >>> Constants <<<
# -----------------
_DEG2RAD = math.pi / 180.0
_RAD2DEG = 180.0 / math.pi



# -----------------
# >>> Functions <<<
# -----------------
def get_linear_degree_range(start=0, stop=360, step_size=10, offset=0):
    """
    Generate a list of degrees within a linear range.

    Parameters:
    - start (int, optional): Starting degree (default is 0).
    - stop (int, optional): Ending degree (default is 360).
    - step_size (int, optional): Step size between degrees (default is 10).
    - offset (int, optional): Offset to add to each degree value (default is 0).

    Returns:
    - list: List of degree values adjusted by offset and modulo 360.
    """
    degree_range = np.arange(start=start, stop=stop, step=step_size).tolist() # list(range(0, 360, step_size))
    return list(map(lambda x: (x+offset) % 360, degree_range))



def degree_to_vector(degree):
    """
    Convert a degree angle to a 2D unit vector.

    Parameters:
    - degree (float): Angle in degrees.

    Returns:
    - list: 2D vector [cos(degree), sin(degree)].
    """
    # rad = math.radians(degree)
    # return [math.cos(rad), math.sin(rad)]
    rad = degree * _DEG2RAD
    return (math.cos(rad), math.sin(rad))




def vector_to_degree(x, y):
    """
    Convert a 2D vector into its corresponding degree angle.

    Parameters:
    - vector (tuple): 2D vector (x, y).

    Returns:
    - int: Angle in degrees within the range [0, 360).
    """
    # x, y = vector
    # degree = math.degrees(math.atan2(y, x))  # atan2 returns angles between -180° and 180°
    # return int( degree % 360 ) 
    deg = math.atan2(y, x) * _RAD2DEG
    return int(deg % 360.0)



def normalize_point(x, y, width, height):
    """
    Normalize a 2D point to the range [0, 1].

    Parameters:
    - point (tuple): (x, y) coordinates of the point.
    - width (int): Image or grid width.
    - height (int): Image or grid height.

    Returns:
    - tuple: Normalized point (x / (width - 1), y / (height - 1)).
    """
    # return (x / (width - 1), y / (height - 1))
    return ( (1.0 / (width - 1.0))*x, 
             (1.0 / (height - 1.0))*y)




def denormalize_point(x, y, width, height):
    """
    Denormalize a 2D point from normalized coordinates back to pixel coordinates.

    Parameters:
    - point (tuple): Normalized (x, y) coordinates.
    - width (int): Image or grid width.
    - height (int): Image or grid height.

    Returns:
    - tuple: Denormalized point (x * (width - 1), y * (height - 1)).
    """
    return (x * (width - 1.0), y * (height - 1.0))



def numpy_info(numpy_array, should_print=True):
    """
    Print and return basic statistical information about a NumPy array.

    Computes common descriptive statistics such as mean, median, standard
    deviation, variance, minimum, maximum, and array shape. This function is
    primarily intended for quick debugging and inspection of intermediate
    results in numerical or image-processing pipelines.

    Parameters:
    - numpy_array (np.ndarray): Input NumPy array to analyze.
    - should_print (bool): If True, print the statistics to stdout.

    Returns:
    - str: A formatted string containing the computed statistics.
    """
    result = "Array Statistics:"
    result += f"\n    - mean = {numpy_array.mean():.2f}"
    result += f"\n    - median = {np.median(numpy_array):.2f}"
    # result += f"\n    - percentile = {np.percentile(numpy_array, [25, 50, 75])}"
    result += f"\n    - std = {numpy_array.std():.2f}"
    result += f"\n    - var = {numpy_array.var():.2f}"
    result += f"\n    - min = {numpy_array.min():.2f}"
    result += f"\n    - max = {numpy_array.max():.2f}"
    result += f"\n    - shape = {numpy_array.shape}"
    
    if should_print:
        print(result)
    return result



# -----------------------
# >>> Numba Functions <<<
# -----------------------
degree_to_vector_numba = numba.njit(cache=True, fastmath=True)(degree_to_vector)
vector_to_degree_numba = numba.njit(cache=True, fastmath=True)(vector_to_degree)
normalize_point_numba = numba.njit(cache=True, fastmath=True)(normalize_point)
denormalize_point_numba = numba.njit(cache=True, fastmath=True)(denormalize_point)


