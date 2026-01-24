"""
**Image Input, Output, and Visualization Utilities**

This module provides a comprehensive set of tools for loading, saving, displaying,
and analyzing images, particularly for scientific and machine learning workflows.
It supports grayscale and color images, normalization, inversion, block-wise
statistics, and flexible visualization options for single or multiple images.

The core idea is to provide an easy interface for inspecting and comparing images,
generating informative visualizations, and preparing image data for further processing.

Main features:
- Load and save images with optional normalization
- Display images with flexible size, colormap, and axis options
- Compare multiple sets of images (input, prediction, ground truth, difference)
- Advanced multi-image visualization with custom layouts and titles
- Annotate images with block-wise mean values for quick inspection
- Highlight specific rows or columns and plot their pixel profiles
- Utility functions to get image properties (bit depth, width, height)

Typical workflow:
1. Load images using `open()` or read multiple paths via `show_images()`.
2. Visualize single or multiple images using `imshow()` or `advanced_imshow()`.
3. Compare predictions with ground truth using `show_samples()`.
4. Annotate blocks or highlight pixel profiles using
    `plot_image_with_values()` and `show_image_with_line_and_profile()`.

Dependencies:
- numpy
- cv2 (OpenCV)
- matplotlib
- scikit-image

Example:
```python
img = img.open("example.png", should_scale=True)
img = img * 255  # optional scaling
img.show()
show_samples([img], [pred_img], [ground_truth], model_name="MyModel")
plot_image_with_values(img, block_size=16, cmap="gray")
line_values = show_image_with_line_and_profile([img], axis="row", index=50)
```

Author:<br>
Tobia Ippolito

Functions:
- get_bit_depth(img)                   - Return bit depth of image dtype.
- get_width_height(img, channels_before=0) - Return (width, height) of an image.
- open(src, should_scale=False, should_print=True) - Load an image from disk.
- save(img, src, should_scale=False)  - Save an image to disk.
- imshow(img, size=8, axis_off=True, cmap="gray") - Display an image.
- show_samples(input_samples, pred_samples, real_samples, ...) - Compare multiple images.
- advanced_imshow(img, title=None, image_width=10, ...) - Display single or batch images with customization.
- show_images(image_paths, title=None, image_width=5, ...) - Load and display images from paths.
- plot_image_with_values(img, block_size=8, ...) - Annotate image with block-wise mean values.
- show_image_with_line_and_profile(imgs, axis='row', ...) - Highlight a row/column and plot pixel values.
"""



# ---------------
# >>> Imports <<<
# ---------------
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import block_reduce  # pip install scikit-image



# -------------------------
# >>> Basic Image tools <<<
# -------------------------

def get_bit_depth(img):
    """
    Retrieve the bit depth of an image based on its NumPy data type.

    Parameters:
        img (numpy.ndarray): Input image array.

    Returns:
        int or str: Bit depth of the image (8, 16, 32, or 64).
                    Returns "unknown" if the data type is not recognized.

    Notes:
        The mapping is defined for common image dtypes:
            - np.uint8   →  8-bit
            - np.uint16  → 16-bit
            - np.int16   → 16-bit
            - np.float32 → 32-bit
            - np.float64 → 64-bit
    """
    dtype_to_bits = {
        np.uint8: 8,
        np.uint16: 16,
        np.int16: 16,
        np.float32: 32,
        np.float64: 64
    }
    return dtype_to_bits.get(img.dtype.type, "unknown")


def get_width_height(img, channels_before=0):
    """
    Extract the width and height of an image, optionally offset by leading channels.

    Parameters:
        img (numpy.ndarray): Input image array.
        channels_before (int, optional): Offset in the shape dimension if
                                         channels precede height and width
                                         (default: 0).

    Returns:
        tuple: (width, height) of the image.

    Example:
        >>> img.shape = (256, 512)
        >>> get_width_height(img)
        (512, 256)
    """
    height, width = img.shape[0+channels_before:2+channels_before]
    return width, height



def open(src, should_scale=False, auto_scale_method=True, should_print=True, unchanged=True):
    """
    Load a grayscale image from a file path.

    Parameters:
        src (str): Path to the image file.
        should_scale (bool, optional): If True, scale pixel values to [0, 1]
                                       according to bit depth (default: False).
        auto_scale_method (bool, optionl): If True, the scaling will auto decide to up or down scale 
                                            if should_sclae is also True else always down scale (default: True).
        should_print (bool, optional): If True, print image info to console
                                       (default: True).

    Returns:
        numpy.ndarray: Loaded grayscale image.

    Example:
        >>> img = open("example.png", should_scale=True)
        Loaded Image:
            - Image size: 512x256
            - Bit depth: 8-bit
            - Dtype: float64
    """
    img = cv2.imread(src,  cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_GRAYSCALE)
    height, width = img.shape[:2]

    if should_scale:
        img_max = ((2**get_bit_depth(img)) -1)
        if auto_scale_method:
            if img.max() < img_max*0.2:  # or just img.max() <= 1.1
                img = img * img_max
            else:
                img = img / img_max
        else:
            img = img / img_max

    if should_print:
        print(f"Loaded Image:\n    - Image size: {width}x{height}\
              \n    - Bit depth: {get_bit_depth(img)}-bit\n    - Dtype: {img.dtype}\
              \n    - Mean: {img.mean()}\n    - Max: {img.max()}\n    - Min: {img.min()}")

    return img



def save(img, src, should_scale=False, auto_scale_method=True):
    """
    Save an image to disk.

    Parameters:
        img (numpy.ndarray): Image to save.
        src (str): Destination file path.
        should_scale (bool, optional): If True, scale pixel values to [0, 1]
                                       before saving (default: False).
        auto_scale_method (bool, optionl): If True, the scaling will auto decide to up or down scale 
                                            if should_sclae is also True else always down scale (default: True).

    Notes:
        - The function uses OpenCV's `cv2.imwrite` for saving.
        - The scaling logic divides by the maximum value representable
          by the bit depth, similar to the `open()` function.
    """
    if should_scale:
        img_max = ((2**get_bit_depth(img)) -1)
        if auto_scale_method:
            if img.max() < img_max*0.2:  # or just img.max() <= 1.1
                img = img * img_max
            else:
                img = img / img_max
        else:
            img = img / img_max

    cv2.imwrite(src, img)



def imshow(img, size=8, axis_off=True, cmap="gray"):
    """
    Display an image using Matplotlib.

    Parameters:
        img (numpy.ndarray): Image to display.
        size (int, optional): Display size in inches (default: 8).
        axis_off (bool, optional): If True, hides the axes (default: True).
        cmap (str, optional): Colormap name.
                              Use 'random' for a random Matplotlib colormap (default: 'gray').

    Behavior:
        - If `img` has 3 channels, it is converted from BGR to RGB.
        - If `cmap='random'`, a random colormap is chosen and possibly reversed.
        - Maintains the aspect ratio based on image dimensions.

    Example:
        >>> imshow(img, cmap='random')
        # Displays the image with a randomly selected colormap.
    """
    if cmap == "random":
        cmap = np.random.choice(["viridis",
                                 "magma",
                                 "inferno",
                                 "plasma",
                                 "cividis",
                                 "spring",
                                 "hot",
                                 "hsv",
                                 "CMRmap",
                                 "gnuplot",
                                 "gnuplot2",
                                 "jet",
                                 "turbo"])
        cmap = cmap if np.random.random() > 0.5 else cmap+"_r"

    height, width = img.shape[:2]
    ratio = height / width
    plt.figure(figsize=(size, round(size * ratio)))
    if img.ndim > 2:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap=cmap)
    plt.axis('off' if axis_off else 'on')
    plt.show()



# ----------------------------
# >>> Advanced Image Tools <<<
# ----------------------------
#  from https://github.com/xXAI-botXx/AI/blob/main/src/helper/imshow.py

def show_samples(input_samples, pred_samples, real_samples, model_name="Model",
                 n_samples=3, n_cols=4, image_width=4, cmap="gray",
                 normalize=True, invert=False, axis=False,
                 save_to=None, hspace=0.3, wspace=0.2, use_original_style=False):
    """
    Display multiple sets of sample images (input, prediction, ground truth, difference)
    side by side for visual comparison.

    The function can load images from file paths or accept NumPy arrays directly.
    It arranges them in a grid and can optionally normalize, invert, or save the output.

    Parameters:
        input_samples (list[str] or list[np.ndarray]): Input sample images.
        pred_samples (list[str] or list[np.ndarray]): Model prediction images.
        real_samples (list[str] or list[np.ndarray]): Ground truth images.
        model_name (str, optional): Name of the model to display in titles (default: "Model").
        n_samples (int, optional): Number of sample groups to display (default: 3).
        n_cols (int, optional): Number of columns per sample group (default: 4).
                                Typically: Input | Prediction | Ground Truth | Difference.
        image_width (int, optional): Width of one image in inches (default: 4).
        cmap (str, optional): Colormap for displaying grayscale images (default: "gray").
        normalize (bool, optional): Whether to normalize pixel values to [0, 1] (default: True).
        invert (bool, optional): Whether to invert pixel values (255 - img) (default: False).
        axis (bool, optional): Whether to show image axes (default: False).
        save_to (str, optional): Path to save the figure (default: None).
        hspace (float, optional): Vertical spacing between subplots (default: 0.3).
        wspace (float, optional): Horizontal spacing between subplots (default: 0.2).
        use_original_style (bool, optional): If True, preserves the current matplotlib style (default: False).

    Returns:
        None

    Example:
        >>> show_samples(inputs, preds, reals, model_name="UNet", n_samples=5, cmap="gray")
    """
    
    def load_image(img):
        if isinstance(img, str):
            arr = cv2.imread(img, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        else:
            arr = img.astype(np.float32)
        if invert:
            arr = 255 - arr
        if normalize:
            arr /= 255.0
        return arr

    # Prepare images and titles
    all_images = []
    titles = []
    sub_images = []

    for idx in range(n_samples):
        all_images.extend([input_samples[idx], pred_samples[idx], real_samples[idx], pred_samples[idx]])
        titles.extend(["Input", model_name, "Ground Truth", "Difference"])
        sub_images.extend([None, None, None, real_samples[idx]])

    # Load all images
    img_arrays = [load_image(im) for im in all_images]
    sub_arrays = [load_image(im) if im is not None else None for im in sub_images]

    # Compute differences where applicable
    for i, sub in enumerate(sub_arrays):
        if sub is not None:
            img_arrays[i] = np.abs(img_arrays[i] - sub)

    n_images = len(img_arrays)
    n_rows = n_images // n_cols + int(n_images % n_cols > 0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*image_width, n_rows*image_width))

    if not use_original_style:
        plt_style = 'seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'classic'
        plt.style.use(plt_style)

    axes = axes.ravel() if n_images > 1 else [axes]

    for idx, ax in enumerate(axes):
        if idx >= n_images:
            ax.axis('off')
            continue
        ax.imshow(img_arrays[idx], cmap=cmap)
        ax.set_title(titles[idx], fontsize=10)
        if not axis:
            ax.axis('off')

    plt.subplots_adjust(hspace=hspace, wspace=wspace)

    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        plt.savefig(save_to, dpi=300)

    plt.show()
    if not use_original_style:
        plt.style.use('default')



def advanced_imshow(img, title=None, image_width=10, axis=False,
           color_space="RGB", cmap=None, cols=1, save_to=None,
           hspace=0.2, wspace=0.2,
           use_original_style=False, invert=False):
    """
    Display one or multiple images in a flexible and configurable grid.

    This function supports multiple color spaces, automatic reshaping of 
    input tensors, batch display, color inversion, and saving to disk.

    Parameters:
        img (np.ndarray): Input image or batch of images.
                          Accepted shapes:
                              [H, W], [H, W, C], [N, H, W], or [N, H, W, C].
        title (str or list[str], optional): Overall or per-image titles.
        image_width (int, optional): Width of each image in inches (default: 10).
        axis (bool, optional): Whether to show axes (default: False).
        color_space (str, optional): Color space of the image: "RGB", "BGR", "gray", or "HSV" (default: "RGB").
        cmap (str, optional): Matplotlib colormap for grayscale images (default: None).
        cols (int, optional): Number of columns in the subplot grid (default: 1).
        save_to (str, optional): File path to save the figure (default: None).
        hspace (float, optional): Vertical spacing between subplots (default: 0.2).
        wspace (float, optional): Horizontal spacing between subplots (default: 0.2).
        use_original_style (bool, optional): Keep current Matplotlib style if True (default: False).
        invert (bool, optional): Invert color values (default: False).

    Returns:
        None

    Example:
        >>> advanced_imshow(batch_images, cols=3, color_space="BGR", title="Predictions")
    """
    original_style = plt.rcParams.copy()
    try:
        img_shape = img.shape
    except Exception:
        img = np.array(img)
        img_shape = img.shape
    # Transform to 4D array [N, H, W, C]
    if len(img_shape) == 2:
        img = img.reshape(1, img_shape[0], img_shape[1], 1)
    elif len(img_shape) == 3:
        if img_shape[2] in [1, 3]:  # single image with channels
            img = img.reshape(1, img_shape[0], img_shape[1], img_shape[2])
        else:  # multiple gray images [N,H,W]
            img = img[..., np.newaxis]
    elif len(img_shape) != 4:
        raise ValueError(f"Image(s) have wrong shape: {img_shape}")

    n_images = img.shape[0]
    aspect_ratio = img.shape[2] / img.shape[1]
    rows = n_images // cols + int(n_images % cols > 0)
    width = int(image_width * cols)
    height = int(image_width * rows * aspect_ratio)

    if not use_original_style:
        plt_style = 'seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'classic'
        plt.style.use(plt_style)

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(width, height))
    axes = np.array(axes).ravel()
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    if isinstance(title, str):
        fig.suptitle(title, fontsize=20, y=0.95)

    # Invert images if needed
    if invert:
        max_val = 2**(img.dtype.itemsize*8) - 1
        img = max_val - img

    for idx, ax in enumerate(axes):
        if idx >= n_images:
            ax.axis("off")
            continue
        cur_img = img[idx]

        # Handle color spaces
        used_cmap = cmap
        if color_space.lower() == "bgr" and cur_img.shape[2] == 3:
            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
            used_cmap = None
        elif color_space.lower() == "hsv" and cur_img.shape[2] == 3:
            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_HSV2RGB)
            used_cmap = None
        elif color_space.lower() in ["gray", "grey", "g"]:
            if cur_img.shape[2] == 3:
                cur_img = cv2.cvtColor(cur_img, cv2.COLOR_RGB2GRAY)
            used_cmap = "gray"

        if isinstance(title, (list, tuple)):
            ax.set_title(title[idx], fontsize=12)
        if not axis:
            ax.axis("off")

        ax.imshow(cur_img.squeeze(), cmap=used_cmap)

    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        fig.savefig(save_to, dpi=300)

    plt.show()
    if not use_original_style:
        plt.rcParams.update(original_style)



def show_images(image_paths: list, title=None, image_width=5, axis=False,
                color_space="gray", cmap=None, 
                cols=2, save_to=None, hspace=0.01, wspace=0.01,
                use_original_style=False, invert=False):
    """
    Load and display multiple images from disk using `advanced_imshow`.

    Parameters:
        image_paths (list[str]): List of file paths to load.
        title (str or list[str], optional): Plot title(s).
        image_width (int, optional): Width of each image (default: 5).
        axis (bool, optional): Whether to display axes (default: False).
        color_space (str, optional): Color space to convert images to.
                                     One of: "gray", "rgb", "hsv", "bgr" (default: "gray").
        cmap (str, optional): Colormap for grayscale images (default: None).
        cols (int, optional): Number of columns in the grid (default: 2).
        save_to (str, optional): Path to save the figure (default: None).
        hspace (float, optional): Vertical spacing between subplots (default: 0.01).
        wspace (float, optional): Horizontal spacing between subplots (default: 0.01).
        use_original_style (bool, optional): Keep current Matplotlib style (default: False).
        invert (bool, optional): Whether to invert images (default: False).

    Returns:
        np.ndarray: Loaded images stacked as an array.

    Example:
        >>> show_images(["img1.png", "img2.png"], color_space="rgb", cols=2)
    """
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if color_space.lower() == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif color_space.lower() == "hsv":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space.lower() in ["gray", "grey", "g"]:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img)

    images = np.array(images)
    advanced_imshow(images, title=title, image_width=image_width, axis=axis,
           color_space=color_space, cmap=cmap, cols=cols, save_to=save_to,
           hspace=hspace, wspace=wspace,
           use_original_style=use_original_style, invert=invert)
    return images



def plot_image_with_values(img, block_size=8, cmap='gray', title=None, 
                           font_size=6, save_to=None):
    """
    Plot an image with annotated mean values over non-overlapping blocks.

    Each block represents the mean pixel intensity of its region. The mean
    values are displayed as text annotations directly on the image.

    Parameters:
        img (np.ndarray): 2D grayscale image (H, W) or 3D single-channel image (H, W, 1).
        block_size (int or tuple, optional): Size of each block (default: 8).
        cmap (str, optional): Matplotlib colormap (default: "gray").
        title (str, optional): Plot title (default: None).
        font_size (int, optional): Font size of value annotations (default: 6).
        save_to (str, optional): Path to save the figure (default: None).

    Returns:
        None

    Example:
        ```python
        from imshow import plot_image_with_values
        import cv2

        img = cv2.imread('example.png', cv2.IMREAD_GRAYSCALE)
        plot_image_with_values(img, block_size=16, cmap='gray', title='Mean Block Values', font_size=8)
        ```

        Or:
        ```python
        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(4*4, 6))
        idx = 0

        img_1 = plot(ax[0][0], path=input_samples[idx], title=f"Input", cmap="gray")
        plot_image_with_values(img_1, block_size=16, ax=ax[1][0])

        img_2 = plot(ax[0][1], path=pred_model[idx], title=f"{model_name}")
        plot_image_with_values(img_2, block_size=16, ax=ax[1][1])

        img_3 = plot(ax[0][2], path=real[idx], title=f"ground truth")
        plot_image_with_values(img_3, block_size=16, ax=ax[1][2])

        img_4 = plot(ax[0][3], path=pred_model[idx], title=f"Difference", sub_image=real[idx])
        plot_image_with_values(img_4, block_size=16, ax=ax[1][3])

        plt.show()
        ```
    """
    # Ensure 2D image
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    elif img.ndim == 3 and img.shape[2] > 1:
        raise ValueError("Only grayscale images are supported for block annotation.")

    # Compute mean over blocks
    mean_img = block_reduce(img, block_size=(block_size, block_size), func=np.mean)
    max_value = mean_img.max()

    # Plot
    plt.figure(figsize=(mean_img.shape[1]/2, mean_img.shape[0]/2))
    plt.imshow(mean_img, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Mean Value')

    # Annotate each block
    for i in range(mean_img.shape[0]):
        for j in range(mean_img.shape[1]):
            val = mean_img[i, j]
            color = 'white' if val < max_value/1.5 else 'black'
            plt.text(j, i, f'{val:.1f}', ha='center', va='center', color=color, fontsize=font_size)

    plt.title(title or f'Mean Values over {block_size}x{block_size} Blocks')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        plt.savefig(save_to, dpi=300)
    
    plt.show()

    
def show_image_with_line_and_profile(imgs, axis='row', index=None, titles=None, figsize=(10, 4)):
    """
    Display one or multiple grayscale images with a highlighted line (row or column)
    and plot the corresponding pixel intensity profile below or beside each image.

    Parameters:
        imgs (list[np.ndarray]): List of grayscale images to analyze.
        axis (str, optional): Direction of the line ("row" or "column") (default: "row").
        index (int, optional): Index of the selected line. If None, the central line is used (default: None).
        titles (list[str], optional): Titles for each image (default: ["Image 1", "Image 2", ...]).
        figsize (tuple, optional): Figure size per image pair (default: (10, 4)).

    Returns:
        list[np.ndarray]: List of pixel intensity profiles corresponding to the selected line in each image.

    Example:
        >>> show_image_with_line_and_profile(
        ...     imgs=[img_input, img_pred, img_gt],
        ...     axis="row",
        ...     titles=["Input", "Prediction", "Ground Truth"]
        ... )
    """
    assert axis in ['row', 'column'], "axis must be 'row' or 'column'"

    n_images = len(imgs)
    if titles is None:
        titles = [f"Image {i+1}" for i in range(n_images)]
    line_values_list = []

    fig, axes = plt.subplots(n_images, 2, figsize=(figsize[0]*2, figsize[1]*n_images))

    # Ensure axes is 2D
    if n_images == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (img, ax_pair, title) in enumerate(zip(imgs, axes, titles)):
        h, w = img.shape
        if index is None:
            idx = h // 2 if axis == 'row' else w // 2
        else:
            idx = index

        ax_img, ax_profile = ax_pair

        # Show image with line
        ax_img.imshow(img, cmap='gray', interpolation='nearest')
        if axis == 'row':
            ax_img.axhline(idx, color='red', linewidth=1)
            line_values = img[idx, :]
        else:
            ax_img.axvline(idx, color='red', linewidth=1)
            line_values = img[:, idx]
        line_values_list.append(line_values)

        ax_img.set_title(title)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        # Plot pixel values along the line
        x = np.arange(len(line_values))
        ax_profile.plot(x, line_values, color='black', linewidth=1)
        ax_profile.set_title(f'Pixel values along {axis} {idx}')
        ax_profile.set_xlabel('Pixel index')
        ax_profile.set_ylabel('Intensity')
        ax_profile.grid(True)

    plt.tight_layout()
    plt.show()

    return line_values_list

