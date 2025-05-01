from PIL import Image
import os

# ¿øº» ÀÌ¹ÌÁö Æú´õ
input_dir = '/home/hail/Desktop/medical_image_project/datasets/train'
# ÀúÀåÇÒ Æú´õ
output_dir = '/home/hail/Desktop/medical_image_project/datasets/bicubic_downsample'

# Ãâ·Â Æú´õ°¡ ¾ø´Ù¸é »ý¼º
os.makedirs(output_dir, exist_ok=True)

# Æú´õ ³» ¸ðµç ÆÄÀÏ¿¡ ´ëÇØ Ã³¸®
for filename in os.listdir(input_dir):
    if filename.lower().endswith('.png'):  # PNG ÀÌ¹ÌÁö¸¸ Ã³¸®
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # ÀÌ¹ÌÁö ¿­±â
        img = Image.open(input_path)
        # »õ·Î¿î Å©±â °è»ê (1/4 Å©±â)
        new_size = (img.width // 4, img.height // 4)
        # BicubicÀ¸·Î ¸®»çÀÌÁî
        resized_img = img.resize(new_size, Image.BICUBIC)
        # ÀúÀå
        resized_img.save(output_path)

