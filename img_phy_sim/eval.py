"""
**Evaluation Utility Functions**

This module provides some Functionalities for Evaluation.

Main features:
- Calculate Metrice between rays and a target ray-image

Typical use cases:
- Evaluation / accuracy measurement

Dependencies:
- math
- numpy
- ips -> draw rays

Example:
```python
import img_phy_sim as ips

dataset = ips.data.PhysGenDataset(mode='test', variation="sound_reflection", input_type="osm", output_type="complex_only")

f1_mean = 0
recall_mean = 0
precision_mean = 0
counter = 0
for (input_img, target_img, idx) in dataset:
    rays = ips.ray_tracing.trace_beams(rel_position=[0.5, 0.5], 
                                        img_src=input_img.squeeze(0).numpy(), 
                                        directions_in_degree=ips.math.get_linear_degree_range(start=0, stop=360, step_size=36),
                                        wall_values=None, 
                                        wall_thickness=0,
                                        img_border_also_collide=False,
                                        reflexion_order=3,
                                        should_scale_rays=True,
                                        should_scale_img=False)
    f1, recall, precision = ips.math.calc_metrices(rays, nm_gt, eval_name=f"{len(ips.math.get_linear_degree_range(start=0, stop=360, step_size=i))} Rays", should_print=False)
    f1_mean += f1
    recall_mean += recall
    precision_mean += precision
    counter += 1

f1_mean /= counter
recall_mean /= counter
precision_mean /= counter

print(f"Baseline Accuracy: F1={f1_mean:.2f}, Recall={recall_mean:.02f}, Precision={precision_mean:.02f}")
```

Functions:
- calc_metrices(...) - Calculate F1, Recall and Precision between rays (or optinal an image) and an image.
"""



# ---------------
# >>> Imports <<<
# ---------------
import math
import numpy as np

from .ray_tracing import draw_rays



# -----------------
# >>> Functions <<<
# -----------------

def calc_metrices(rays, noise_modelling_gt, rays_format_is_image=False, eval_name="", should_print=True):
    """
    Compute Precision, Recall, and F1 score by comparing ray-based predictions
    against a ground truth image.

    The function converts a set of rays into a binary image representation
    (if not already provided as an image), normalizes both prediction and
    ground truth if necessary, and evaluates their pixel-wise overlap.
    All non-zero pixels are treated as positives.

    Parameters:
    - rays (array-like or np.ndarray):<br>
        Ray representation or pre-rendered ray image, depending on
        `rays_format_is_image`.
    - noise_modelling_gt (np.ndarray):<br>
        Ground truth image used for evaluation.
    - rays_format_is_image (bool):<br>
        If True, `rays` is assumed to already be an image.
        If False, rays are rendered into an image using `draw_rays`.
    - eval_name (str):<br>
        Optional identifier printed alongside the evaluation results.
    - should_print (bool):<br>
        If True, print the computed metrics to stdout.

    Returns:
    - tuple:<br>
        (f1, recall, precision) computed from binary pixel overlap.
    """
    # Create image from rays
    if rays_format_is_image:
        ray_img = rays
    else:
        ray_img = draw_rays(rays, detail_draw=True, 
                            output_format="single_image", 
                            img_background=None, ray_value=1.0, ray_thickness=1, 
                            img_shape=(256, 256), dtype=float, standard_value=0,
                            should_scale_rays_to_image=True, original_max_width=None, original_max_height=None)
    
    # Normalize both (if needed)
    if (noise_modelling_gt > 1.0).any():
        # raise ValueError("Noise Modelling Ground Truth Image is not normalized.")
        noise_modelling_gt /= 255

    if (ray_img > 1.0).any():
        # raise ValueError("Ray Image is not normalized.")
        ray_img /= 255

    # Thresholding to binary images
    noise_modelling_gt_binary = noise_modelling_gt != 0.0
    # numpy_info(noise_modelling_gt_binary)
    rays_binary = ray_img != 0.0

    # Recall, Precision, F1 Score
    overlap = noise_modelling_gt_binary * rays_binary

    #     recall - how is the coverage towards the gt?
    recall = np.sum(overlap) / np.sum(noise_modelling_gt_binary)

    #     precision - how many rays hit the right place?
    precision = np.sum(overlap) / np.sum(rays_binary)

    #     f1
    f1 = 2*(precision*recall) / (precision+recall)

    if should_print:
        print(f"Eval {eval_name}: F1={f1:.02f}, Recall={recall:.02f}, Precision={precision:.02f}")
    return f1, recall, precision







