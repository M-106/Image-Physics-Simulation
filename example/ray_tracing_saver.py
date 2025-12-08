# ADJUST ME
mode = ["test", "train", "validation"]
reflexion_traces = [360]  # 36
REFLEXION_ORDER = 3

ground_path = "./rays"


# run with: nohup python ray_tracing_saver.py > ray_saver.log 2>&1 &


# Imports
import os
import shutil
from tqdm import tqdm

from joblib import Parallel, delayed

import img_phy_sim as ips



# Dataset
"""
PhysGen Dataset

See:
- https://huggingface.co/datasets/mspitzna/physicsgen
- https://arxiv.org/abs/2503.05333
- https://github.com/physicsgen/physicsgen
"""
import os
import shutil
from PIL import Image

from datasets import load_dataset

import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
# import torchvision.transforms as transforms
from torchvision import transforms

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
        # if progress_print:
            # print(f'Progress {i+1}/{data_len}')
            # prime.get_progress_bar(total=data_len, progress=i+1, 
            #                        should_clear=True, left_bar_char='|', right_bar_char='|', 
            #                        progress_char='#', empty_char=' ', 
            #                        front_message='Physgen Data Loading', back_message='', size=15)

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



def process_single_sample(input_, idx, reflexion_steps, reflexion_order, path):
    idx = int(idx.cpu().numpy().reshape(-1)[0])
    input_ = input_.detach().cpu().numpy()

    # remove batch channel
    input_ = np.squeeze(input_, axis=0)

    if len(input_.shape) == 3:
        input_ = np.squeeze(input_, axis=0)

    input_ = np.transpose(input_, (1, 0))
    
    # calc ray tracing
    rays = ips.ray_tracing.trace_beams(rel_position=[0.5, 0.5], 
                                        img_src=input_, 
                                        directions_in_degree=ips.math.get_linear_degree_range(step_size=reflexion_steps),
                                        wall_values=None, 
                                        wall_thickness=0,
                                        img_border_also_collide=False,
                                        reflexion_order=reflexion_order,
                                        should_scale_rays=True,
                                        should_scale_img=True)

    # save ray tracing
    save_path = os.path.join(path, f"rays_[{idx}].txt")
    ips.ray_tracing.save(path=save_path, rays=rays)



def process_mode_reflexion_combination(cur_mode, cur_reflexion_traces, reflexion_order, ground_path):
    dataset = PhysGenDataset(mode=cur_mode, 
                             variation="sound_reflection", 
                             input_type="osm", 
                             output_type="standart")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    reflexion_steps = 360/cur_reflexion_traces

    path = os.path.join(ground_path, cur_mode, str(cur_reflexion_traces))  # "rays.txt"

    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    batch_list = list(loader)

    Parallel(n_jobs=-1)(
        delayed(process_single_sample)(input_, idx_, reflexion_steps, reflexion_order, path)
        for (input_, _, idx_) in batch_list
    )

    return f"Finished Mode:{cur_mode}, Reflexions:{cur_reflexion_traces}"



def parallel_ray_tracing_computation(mode, reflexion_traces, reflexion_order, ground_path):
    Parallel(n_jobs=-1)(
        delayed(process_mode_reflexion_combination)(cur_mode, cur_reflexion_traces, reflexion_order, ground_path)
        for cur_mode in mode
        for cur_reflexion_traces in reflexion_traces
    )



if __name__ == "__main__":
    print("Starting ray tracing computation...")
    parallel_ray_tracing_computation(mode, reflexion_traces, REFLEXION_ORDER, ground_path)


