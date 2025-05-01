import os
from PIL import Image
from tqdm import tqdm

def resize_and_save_images(input_dir, output_dir, size=(256, 256)):
    """
    Resize images in the input directory and save them to the output directory.

    Args:
        input_dir (str): Path to the directory containing original images.
        output_dir (str): Path to the directory to save resized images.
        size (tuple): Target size for resizing (width, height).
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get list of image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    for image_file in tqdm(image_files, desc="Resizing images"):
        try:
            # Open image
            img_path = os.path.join(input_dir, image_file)
            img = Image.open(img_path).convert('RGB')

            # Resize image
            img_resized = img.resize(size, Image.BICUBIC)

            # Save resized image to output directory
            output_path = os.path.join(output_dir, image_file)
            img_resized.save(output_path)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

input_dir = '/media/hail/HDD/DataSets/NIH_Chest_Xray/test256'
output_dir = '/media/hail/HDD/DataSets/NIH_Chest_Xray/BICUBIC4x'
resize_and_save_images(input_dir, output_dir, size=(1024, 1024))