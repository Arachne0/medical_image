import os
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from tqdm import tqdm

# Root paths
gt_path = '/home/hail/SH/medical_image/datasets/train/256/Atelectasis'

model_names = [
    'SRGAN',
    'ESRGAN',
    'Real-ESRGAN',
    'swinir',
    'HAT',
    'SRFormer'
]

# Log file
log_file = './sr_evaluation_log.txt'

def main():
    with open(log_file, 'w') as log:
        for model in model_names:
            if model in ['HAT', 'SRFormer']:
                sr_path = f'/home/hail/SH/{model}/save_image/Atelectasis'
            else:
                sr_path = f'/home/hail/SH/medical_image/{model}/save_image/Atelectasis'

            mse_list, rmse_list, psnr_list, ssim_list = [], [], [], []

            gt_files = sorted(os.listdir(gt_path))
            sr_files = sorted(os.listdir(sr_path))

            common_files = set(gt_files).intersection(sr_files)
            if not common_files:
                log.write(f'{model}: No matching files found.\n')
                continue

            for fname in tqdm(common_files, desc=f'Evaluating {model}'):
                gt_img_path = os.path.join(gt_path, fname)
                sr_img_path = os.path.join(sr_path, fname)

                try:
                    gt_img = np.array(Image.open(gt_img_path).convert('RGB'))
                    sr_img = np.array(Image.open(sr_img_path).convert('RGB'))

                    if gt_img.shape != sr_img.shape:
                        sr_img = np.array(Image.fromarray(sr_img).resize(gt_img.shape[:2][::-1], Image.BICUBIC))

                    mse, rmse, psnr, ssim = compute_metrics(gt_img, sr_img)
                    mse_list.append(mse)
                    rmse_list.append(rmse)
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

                except Exception as e:
                    log.write(f'{model}: Error processing {fname}: {e}\n')

            if mse_list:
                avg_mse = np.mean(mse_list)
                avg_rmse = np.mean(rmse_list)
                avg_psnr = np.mean(psnr_list)
                avg_ssim = np.mean(ssim_list)
                log.write(f'{model} - MSE: {avg_mse:.4f}, RMSE: {avg_rmse:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}\n')
            else:
                log.write(f'{model}: No valid comparisons made.\n')


def compute_metrics(img1, img2):
    # Convert to grayscale if needed
    if img1.ndim == 3 and img1.shape[2] == 3:
        img1 = np.dot(img1[..., :3], [0.2989, 0.5870, 0.1140])
        img2 = np.dot(img2[..., :3], [0.2989, 0.5870, 0.1140])

    mse = np.mean((img1 - img2) ** 2)
    rmse = np.sqrt(mse)
    psnr = compute_psnr(img1, img2, data_range=255)
    ssim = compute_ssim(img1, img2, data_range=255)
    return mse, rmse, psnr, ssim


if __name__ == '__main__':
    main()
