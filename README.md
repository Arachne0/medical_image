``` text.
python version 3.10

train image : 56215
test image : 11097
```


```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install timm
pip install matplotlib
pip install scikit-image
pip install opencv-python
pip install pickle
```


## PSNR (Peak Signal-to-Noise Ratio)

Measures the level of distortion or loss between the original and reconstructed images.
Higher values (typically 30~50 dB or more) indicate better quality and greater similarity.
Computed using the mean squared error (MSE) between images.
Commonly used for image quality assessment.



## SSIM (Structural Similarity Index Measure)

Evaluates the structural similarity between two images.
Considers luminance, contrast, and structure to reflect perceptual quality.
Values range from 0 to 1, where 1 indicates perfect similarity.



## PSNRB (Peak Signal-to-Noise Ratio with Blocking Effect)
A variation of PSNR that accounts for blocking artifacts, commonly seen in compressed images (e.g., JPEG).
More suitable for evaluating images with severe blockiness.
Adjusts the PSNR score to reflect block distortion.


## Y-channel Metrics (PSNR_Y, SSIM_Y, PSNRB_Y)

Metrics computed only on the luminance (Y) channel of an image.
Used because the Y channel strongly correlates with human visual perception.
Helps to evaluate perceived quality more accurately than RGB-based metrics.


## 5/21 SH
| Label               | Count |
|---------------------|-------|
| No Finding          | 39302 |
| Infiltration        | 9353  |
| Effusion            | 6589  |
| Atelectasis         | 5728  |
| Nodule              | 4177  |
| Mass                | 3567  |
| Pneumothorax        | 3407  |
| Pleural_Thickening  | 2418  |
| Cardiomegaly        | 1563  |
| Consolidation       | 1521  |
| Emphysema           | 1499  |
| Fibrosis            | 1408  |
| Pneumonia           | 630   |
| Edema               | 276   |
| Hernia              | 192   |




