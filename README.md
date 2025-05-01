``` text.
python version 3.10

train image : 86524
test image : 25596
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


## 4/29 SH
I created a train folder by downscaling 1024x1024 images by a factor of 4 using bicubic interpolation. 
The folder is located at datasets/bicubic_downsample. 
Based on this, I developed a super-resolution code.


