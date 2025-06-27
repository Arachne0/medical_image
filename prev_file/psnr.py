import os
import numpy as np
from PIL import Image
import torch
import math
from tqdm import tqdm
import wandb

def calculate_psnr(original, resized, max_value=255.0):
    """
    Calculate PSNR between two images.

    Args:
        original (torch.Tensor): Original high-resolution image.
        resized (torch.Tensor): Resized or restored image.
        max_value (float): Maximum pixel value (default: 255.0 for 8-bit images).

    Returns:
        float: PSNR value.
    """
    # Ensure both images are of the same shape
    assert original.shape == resized.shape, "Images must have the same dimensions."

    # Mean Squared Error (MSE)
    mse = torch.mean((original - resized) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match

    # PSNR calculation
    psnr = 20 * math.log10(max_value / math.sqrt(mse))
    return psnr

def compute_psnr_and_log(original_dir, resized_dirs, max_value=255.0):
    """
    Compute PSNR for all images in the dataset between the original and resized versions,
    log the results to wandb, and print them to the console.

    Args:
        original_dir (str): Path to the directory with original high-resolution images.
        resized_dirs (dict): Dictionary with scale factors as keys and paths to resized images as values.
        max_value (float): Maximum pixel value (default: 255.0 for 8-bit images).

    Returns:
        dict: Average PSNR results for each scale factor.
    """
    # Initialize wandb
    wandb.init(project="NIH_Chest_X-ray",
               entity="hails",
               name=f"BICUBIC_PSNR"
               )

    psnr_results = {scale: [] for scale in resized_dirs.keys()}

    image_files = [f for f in os.listdir(original_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    for image_file in tqdm(image_files, desc="Computing PSNR"):
        original_path = os.path.join(original_dir, image_file)
        original_img = Image.open(original_path).convert('RGB')
        original_tensor = torch.tensor(np.array(original_img), dtype=torch.float32) / 255.0

        for scale, resized_dir in resized_dirs.items():
            resized_path = os.path.join(resized_dir, image_file)
            resized_img = Image.open(resized_path).convert('RGB')
            resized_tensor = torch.tensor(np.array(resized_img), dtype=torch.float32) / 255.0

            psnr = calculate_psnr(original_tensor, resized_tensor, max_value=max_value)
            psnr_results[scale].append(psnr)

            # Log PSNR to wandb
            wandb.log({f"PSNR_{scale}": psnr, "Image": image_file})

            # Print individual PSNR
            print(f"{image_file} - {scale}: {psnr:.2f} dB")

    # Calculate and log average PSNR per scale
    for scale, values in psnr_results.items():
        avg_psnr = sum(values) / len(values)
        wandb.log({f"Average_PSNR_{scale}": avg_psnr})
        print(f"Average PSNR for {scale}: {avg_psnr:.2f} dB")

    return psnr_results

# Example usage
original_dir = '/media/hail/HDD/DataSets/NIH_Chest_Xray/test'  # Original test folder
resized_dirs = {
    '4x': '/media/hail/HDD/DataSets/NIH_Chest_Xray/BICUBIC4x'   # 256x256 upscaled to 1024x1024
}

psnr_results = compute_psnr_and_log(original_dir, resized_dirs)
print("Final PSNR Results:")
for scale, psnrs in psnr_results.items():
    avg_psnr = sum(psnrs) / len(psnrs)
    print(f"{scale}: {avg_psnr:.2f} dB")