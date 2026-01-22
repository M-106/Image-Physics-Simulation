"""
IPS: Image Physical Simulation Package

This package provides a suite of tools for simulating, analyzing, and visualizing
rays (beams) in 2D images, along with general-purpose image and mathematical utilities.
It is designed for research workflows involving instance segmentation, ray tracing,
and image-based simulations, but can also be used in general image processing tasks.

Submodules:
- `ray_tracing`<br>
    Tools for tracing rays through images with walls or obstacles,
    handling reflections, scaling, and visualization.<br>
    Key functionalities:
    - trace_beam / trace_beams
    - calc_reflection
    - get_wall_map
    - draw_rays
    - scale_rays
    - Utilities for merging rays, printing info, or getting pixel coordinates

- `math`<br>
    Utilities for 2D geometry, coordinate transformations, and vector math.
    Provides functions for angle-to-vector conversions, normalization, and
    working with linear degree ranges.<br>
    Key functionalities:
    - degree_to_vector / vector_to_degree
    - get_linear_degree_range
    - normalize_point / denormalize_point

- `img`<br>
    Image I/O, visualization, and analysis utilities.
    Includes functions for loading, saving, displaying, and annotating images,
    comparing predictions with ground truth, and plotting block-wise or line-wise
    statistics.<br>
    Key functionalities:
    - open / save
    - imshow / advanced_imshow / show_images / show_samples
    - plot_image_with_values / show_image_with_line_and_profile
    - get_width_height / get_bit_depth

- `eval`<br>
    Accuracy measurement utilities.<br>
    Key functionalities:
    - calc_metrices

- `data`<br>
    Loading utilities for PhysGen Dataset.<br>
    Key functionalities:
    - PhysGenDataset()
    - get_dataloader
    - save_dataset

    
Typical workflow:
1. Prepare an environment image using `img.open()` or generate it programmatically.
2. Trace beams using `ray_tracing.trace_beams()` with specified start positions,
    directions, and wall values.
3. Visualize the rays on images with `ray_tracing.draw_rays()`.
4. Use `img` utilities for inspecting, annotating, or comparing images.
5. Use `math` utilities for vector and angle calculations or normalization.

Dependencies:
- numpy
- OpenCV (cv2)
- matplotlib

Example:
```python
import img_phy_sim as ips

# Load image
img = ips.img.open("scene.png")

# Trace beams
rays = ips.ray_tracing.trace_beams(
    rel_position=(0.5, 0.5),
    img_src=img,
    directions_in_degree=[0, 45, 90, 135],
    wall_values=[0],
    wall_thickness=2,
    reflexion_order=2
)

# Draw rays
output = ips.ray_tracing.draw_rays(rays, img_shape=img.shape, ray_value=255, ray_thickness=1)

# Display result
ips.img.imshow(output, size=5)
```

Author:<br>
Tobia Ippolito, 2025
"""

from . import img
from . import ray_tracing
from . import ism
from . import math
from . import eval
from . import data
# from . import data

