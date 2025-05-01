import os

# °æ·Î Á¤ÀÇ
train_dir = '/home/hail/Desktop/medical_image_project/datasets/train'
test_dir = '/home/hail/Desktop/medical_image_project/datasets/test'

# ÇÔ¼ö: ÁöÁ¤µÈ µð·ºÅä¸®¿¡¼­ .png ÆÄÀÏ °³¼ö ¼¼±â
def count_png_images(directory):
    return sum(
        1 for f in os.listdir(directory)
        if f.lower().endswith('.png') and os.path.isfile(os.path.join(directory, f))
    )

# °á°ú Ãâ·Â
train_count = count_png_images(train_dir)
test_count = count_png_images(test_dir)

print(f"Train ÀÌ¹ÌÁö °³¼ö: {train_count}")
print(f"Test ÀÌ¹ÌÁö °³¼ö: {test_count}")
