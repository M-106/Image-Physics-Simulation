"""
**PhysGen Dataset Loader and Export Utilities**

This module provides a PyTorch-compatible interface for loading, processing,
visualizing, and exporting samples from the PhysGen dataset, a large-scale
physics-based sound propagation dataset. It is designed for machine learning
workflows that operate on image-like representations of physical simulations,
such as sound maps derived from urban environments.

The module wraps the official PhysGen Hugging Face dataset and exposes flexible
options for selecting dataset variations, input/output modalities, and training
splits. In addition, it includes utilities for resizing tensors, retrieving
individual samples, constructing PyTorch DataLoaders, and exporting datasets
to disk as image files.

The core idea is to make PhysGen easy to integrate into deep learning pipelines
while supporting common preprocessing steps such as resizing, modality switching,
and derived target generation (e.g. complex-only outputs).

Main features:
- PyTorch `Dataset` implementation for the PhysGen dataset
- Support for multiple PhysGen variations (baseline, reflection, diffraction, combined)
- Configurable input types (OSM or base simulation)
- Configurable output types (standard sound maps or complex-only targets)
- Automatic image-to-tensor conversion and normalization
- Utilities for resizing tensors to model-friendly dimensions
- Easy access to single samples as NumPy arrays for debugging and visualization
- Dataset export functionality for saving inputs and targets as PNG images
- Command-line interface for batch dataset extraction

Typical workflow:
1. Initialize the dataset via `PhysGenDataset(...)`.
2. Wrap it in a PyTorch `DataLoader` using `get_dataloader()`.
3. Retrieve individual samples for inspection using `get_image()`.
4. Train or evaluate a model using the provided tensors.
5. Optionally export the dataset to disk using `save_dataset()`.

Dependencies:
- torch
- torchvision
- numpy
- cv2 (OpenCV)
- PIL
- datasets (Hugging Face)
- prime_printer (for progress visualization)

PhysGen references:
- Dataset: https://huggingface.co/datasets/mspitzna/physicsgen
- Paper: https://arxiv.org/abs/2503.05333
- GitHub: https://github.com/physicsgen/physicsgen

Example:
```python
from img_phy_sim.data import PhysGenDataset, get_dataloader

dataset = PhysGenDataset(
    variation="sound_reflection",
    mode="train",
    input_type="osm",
    output_type="complex_only"
)

loader = get_dataloader(
    mode="train",
    variation="sound_reflection",
    input_type="osm",
    output_type="complex_only"
)

input_img, target_img, idx = dataset[0]
```

Author:<br>
Tobia Ippolito

Classes:
- PhysGenDataset(Dataset)<br>
    PyTorch dataset wrapper for PhysGen with flexible input/output configuration.

Functions:
- resize_tensor_to_divisible_by_14(tensor)<br>
    Resize tensors so height and width are divisible by 14.
- get_dataloader(...)<br>
    Create a PyTorch DataLoader for the PhysGen dataset.
- get_image(...)<br>
    Retrieve a single dataset sample (optionally as NumPy arrays).
- save_dataset(...)<br>
    Export PhysGen inputs and targets as PNG images to disk.

Also see:
- https://huggingface.co/datasets/mspitzna/physicsgen
- https://arxiv.org/abs/2503.05333
- https://github.com/physicsgen/physicsgen
"""



# ---------------
# >>> Imports <<<
# ---------------
import os
import shutil

import numpy as np
import cv2

DATA_DEPENDENCIES_AVAILABLE = True
try:
    from datasets import load_dataset
except Exception:
    DATA_DEPENDENCIES_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    # import torchvision.transforms as transforms
    from torchvision import transforms
except Exception:
    DATA_DEPENDENCIES_AVAILABLE = False

try:
    import prime_printer as prime
    PRIME_AVAILABLE = True
except Exception:
    PRIME_AVAILABLE = False


if DATA_DEPENDENCIES_AVAILABLE:
    # --------------
    # >>> Helper <<<
    # --------------
    def resize_tensor_to_divisible_by_14(tensor: torch.Tensor) -> torch.Tensor:
        """
        Resize a tensor to the next smaller (H, W) divisible by 14.
        
        Args:
            tensor (torch.Tensor): Input tensor of shape (C, H, W) or (B, C, H, W)
        
        Returns:
            torch.Tensor: Resized tensor
        """
        if tensor.dim() == 3:
            c, h, w = tensor.shape
            new_h = h - (h % 14)
            new_w = w - (w % 14)
            return F.interpolate(tensor.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
        
        elif tensor.dim() == 4:
            b, c, h, w = tensor.shape
            new_h = h - (h % 14)
            new_w = w - (w % 14)
            return F.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        else:
            raise ValueError("Tensor must be 3D (C, H, W) or 4D (B, C, H, W)")



    # ------------------
    # >>> Main Class <<<
    # ------------------
    class PhysGenDataset(Dataset):

        def __init__(self, variation="sound_baseline", mode="train", input_type="osm", output_type="standard"):
            """
            Loads PhysGen Dataset.

            Parameters:
            - variation : str
                Chooses the used dataset variant: sound_baseline, sound_reflection, sound_diffraction, sound_combined.
            - mode : str
                Can be "train", "test", "eval".
            - input_type : str
                Defines the used Input -> "osm", "base_simulation"
            - output_type : str
                Defines the Output -> "standard", "complex_only"
            """
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            # get data
            self.dataset = load_dataset("mspitzna/physicsgen", name=variation, trust_remote_code=True)
            # print("Keys:", self.dataset.keys())
            self.dataset = self.dataset[mode]
            
            self.input_type = input_type
            self.output_type = output_type
            if self.input_type == "base_simulation" or self.output_type == "complex_only":
                self.basesimulation_dataset = load_dataset("mspitzna/physicsgen", name="sound_baseline", trust_remote_code=True)
                self.basesimulation_dataset = self.basesimulation_dataset[mode]

            self.transform = transforms.Compose([
                transforms.ToTensor(),  # Converts [0,255] PIL image to [0,1] FloatTensor
            ])
            print(f"PhysGen ({variation}) Dataset for {mode} got created")

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            sample = self.dataset[idx]
            # print(sample)
            # print(sample.keys())
            if self.input_type == "base_simulation":
                input_img = self.basesimulation_dataset[idx]["soundmap"]
            else:
                input_img = sample["osm"]  # PIL Image
            target_img = sample["soundmap"]  # PIL Image

            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

            # Fix real image size 512x512 > 256x256
            input_img = F.interpolate(input_img.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
            input_img = input_img.squeeze(0)
            # target_img = target_img.unsqueeze(0)

            # # change size
            # input_img = resize_tensor_to_divisible_by_14(input_img)
            # target_img = resize_tensor_to_divisible_by_14(target_img)

            # add fake rgb
            # if input_img.shape[0] == 1:  # shape (B, 1, H, W)
            #     input_img = input_img.repeat(3, 1, 1)  # make it (B, 3, H, W)

            if self.output_type == "complex_only":
                base_simulation_img = self.transform(self.basesimulation_dataset[idx]["soundmap"])
                # base_simulation_img = resize_tensor_to_divisible_by_14(self.transform(self.basesimulation_dataset[idx]["soundmap"]))
                # target_img = torch.abs(target_img[0] - base_simulation_img[0])
                target_img = target_img[0] - base_simulation_img[0]
                target_img = target_img.unsqueeze(0)
                target_img *= -1

            return input_img, target_img, idx



    # ----------------------
    # >>> Loading Helper <<<
    # ----------------------
    def get_dataloader(mode='train', variation="sound_reflection", input_type="osm", output_type="complex_only", shuffle=True):
        dataset = PhysGenDataset(mode=mode, variation=variation, input_type=input_type, output_type=output_type)
        return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=1)



    def get_image(mode='train', variation="sound_reflection", input_type="osm", output_type="complex_only", shuffle=True, 
                return_output=False, as_numpy_array=True):
        dataset = PhysGenDataset(mode=mode, variation=variation, input_type=input_type, output_type=output_type)
        loader = DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=1)
        cur_data = next(iter(loader))
        input_ = cur_data[0]
        output_ = cur_data[1]

        if as_numpy_array:
            input_ = input_.detach().cpu().numpy()
            output_ = output_.detach().cpu().numpy()

            # remove batch channel
            input_ = np.squeeze(input_, axis=0)
            output_ = np.squeeze(output_, axis=0)

            if len(input_.shape) == 3:
                input_ = np.squeeze(input_, axis=0)
                output_ = np.squeeze(output_, axis=0)

            # opencv format
            # if np.issubdtype(img.dtype, np.floating):
            #     img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            #     img = (img * 255).astype(np.uint8)
            input_ = np.transpose(input_, (1, 0))
            output_ = np.transpose(output_, (1, 0))


        result = input_
        if return_output:
            result = [input_, output_]

        return result



    def save_dataset(output_real_path, output_osm_path, 
                    variation, input_type, output_type,
                    data_mode, 
                    info_print=False, progress_print=True):
        # Clearing
        if os.path.exists(output_osm_path) and os.path.isdir(output_osm_path):
            shutil.rmtree(output_osm_path)
            os.makedirs(output_osm_path)
            print(f"Cleared {output_osm_path}.")
        else:
            os.makedirs(output_osm_path)
            print(f"Created {output_osm_path}.")

        if os.path.exists(output_real_path) and os.path.isdir(output_real_path):
            shutil.rmtree(output_real_path)
            os.makedirs(output_real_path)
            print(f"Cleared {output_real_path}.")
        else:
            os.makedirs(output_real_path)
            print(f"Created {output_real_path}.")
        
        # Load Dataset
        dataset = PhysGenDataset(mode=data_mode, variation=variation, input_type=input_type, output_type=output_type)
        data_len = len(dataset)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

        # Save Dataset
        for i, data in enumerate(dataloader):
            if progress_print:
                # print(f'Progress {i+1}/{data_len}')
                if PRIME_AVAILABLE:
                    prime.get_progress_bar(total=data_len, progress=i+1, 
                                        should_clear=True, left_bar_char='|', right_bar_char='|', 
                                        progress_char='#', empty_char=' ', 
                                        front_message='Physgen Data Loading', back_message='', size=15)

            input_img, target_img, idx = data
            idx = idx[0].item() if isinstance(idx, torch.Tensor) else idx

            # forward_img = inference_forward(input_img, model, DEVICE)

            if info_print:
                # print(f"Prediction shape [forward]: {forward_img.shape}")
                print(f"Prediction shape [osm]: {input_img.shape}")
                print(f"Prediction shape [target]: {target_img.shape}")

                print(f"OSM Info:\n    -> shape: {input_img.shape}\n    -> min: {input_img.min()}, max: {input_img.max()}")

            # Transform to Numpy
            # pred_img = forward_img.squeeze(2)
            # if not (0 <= pred_img.min() <= 255 and 0 <= pred_img.max() <=255):
            #     raise ValueError(f"Prediction has values out of 0-256 range => min:{pred_img.min()}, max:{pred_img.max()}")
            # if pred_img.max() <= 1.0:
            #     pred_img *= 255
            # pred_img = pred_img.astype(np.uint8)

            real_img = target_img.squeeze(0).cpu().squeeze(0).detach().numpy()
            if not (0 <= real_img.min() <= 255 and 0 <= real_img.max() <=255):
                raise ValueError(f"Real target has values out of 0-256 range => min:{real_img.min()}, max:{real_img.max()}")
            if info_print:
                print( f"\nReal target has values out of 0-256 range => min:{real_img.min()}, max:{real_img.max()}")
            if real_img.max() <= 1.0:
                real_img *= 255
            if info_print:
                print( f"Real target has values out of 0-256 range => min:{real_img.min()}, max:{real_img.max()}")
            real_img = real_img.astype(np.uint8)
            if info_print:
                print( f"Real target has values out of 0-256 range => min:{real_img.min()}, max:{real_img.max()}")

            if len(input_img.shape) == 4:
                osm_img = input_img[0, 0].cpu().detach().numpy()
            else:
                osm_img = input_img[0].cpu().detach().numpy()
            if not (0 <= osm_img.min() <= 255 and 0 <= osm_img.max() <=255):
                raise ValueError(f"Real target has values out of 0-256 range => min:{osm_img.min()}, max:{osm_img.max()}")
            if osm_img.max() <= 1.0:
                osm_img *= 255
            osm_img = osm_img.astype(np.uint8)

            if info_print:
                print(f"OSM Info:\n    -> shape: {osm_img.shape}\n    -> min: {osm_img.min()}, max: {osm_img.max()}")

            # Save Results
            file_name = f"physgen_{idx}.png"

            # save pred image
            # save_img = os.path.join(output_pred_path, file_name)
            # cv2.imwrite(save_img, pred_img)
            # print(f"    -> saved pred at {save_img}")

            # save real image
            save_img = os.path.join(output_real_path, "target_"+file_name)
            cv2.imwrite(save_img, real_img)
            if info_print:
                print(f"    -> saved real at {save_img}")

            # save osm image
            save_img = os.path.join(output_osm_path, "input_"+file_name)
            cv2.imwrite(save_img, osm_img)
            if info_print:
                print(f"    -> saved osm at {save_img}")
        print(f"\nSuccessfull saved {data_len} datapoints into {os.path.abspath(output_real_path)} & {os.path.abspath(output_osm_path)}")



    # -----------------------
    # >>> Make it runable <<<
    # -----------------------
    if __name__ == "__main__":
        import argparse

        parser = argparse.ArgumentParser(description="Save OSM and real PhysGen dataset images.")

        parser.add_argument("--output_real_path", type=str, required=True, help="Path to save real target images")
        parser.add_argument("--output_osm_path", type=str, required=True, help="Path to save OSM input images")
        parser.add_argument("--variation", type=str, required=True, help="PhysGen variation (e.g. box_texture, box_position, etc.)")
        parser.add_argument("--input_type", type=str, required=True, help="Input type (e.g. osm_depth)")
        parser.add_argument("--output_type", type=str, required=True, help="Output type (e.g. real_depth)")
        parser.add_argument("--data_mode", type=str, required=True, help="Data Mode: train, test, val")
        parser.add_argument("--info_print", action="store_true", help="Print additional info")
        parser.add_argument("--no_progress", action="store_true", help="Disable progress printing")

        args = parser.parse_args()

        save_dataset(
            output_real_path=args.output_real_path,
            output_osm_path=args.output_osm_path,
            variation=args.variation,
            input_type=args.input_type,
            output_type=args.output_type,
            data_mode=args.data_mode,
            info_print=args.info_print,
            progress_print=not args.no_progress
        )

    