import os

"""
swinIR
"""

# Source directory containing the original files
src_dir = '/swinir/save_image/Atelectasis/'

# Optional: set a separate output directory, or rename in-place
dst_dir = src_dir  # Rename in-place

os.makedirs(dst_dir, exist_ok=True)

for filename in os.listdir(src_dir):
    if filename.endswith('.png') and '_SwinIR' in filename:
        # Split before the model name suffix
        new_name = filename.replace('_SwinIR', '')
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, new_name)
        os.rename(src_path, dst_path)  # or use shutil.copy(src_path, dst_path) if you want to keep originals
        print(f'Renamed: {filename} to {new_name}')



"""
Real-ESRGAN
"""


# # Source directory containing the original files
# src_dir = '/home/hail/SH/medical_image/Real-ESRGAN/save_image/Atelectasis/'
#
# # Optional: set a separate output directory, or rename in-place
# dst_dir = src_dir  # Rename in-place
#
# os.makedirs(dst_dir, exist_ok=True)
#
# for filename in os.listdir(src_dir):
#     if filename.endswith('.png') and '_out' in filename:
#         # Split before the model name suffix
#         new_name = filename.replace('_out', '')
#         src_path = os.path.join(src_dir, filename)
#         dst_path = os.path.join(dst_dir, new_name)
#         os.rename(src_path, dst_path)  # or use shutil.copy(src_path, dst_path) if you want to keep originals
#         print(f'Renamed: {filename} to {new_name}')


"""
HAT
"""

#
# # Source directory containing the original files
# src_dir = '/home/hail/SH/HAT/save_image/No Finding/'
#
# # Optional: set a separate output directory, or rename in-place
# dst_dir = src_dir  # Rename in-place
#
# os.makedirs(dst_dir, exist_ok=True)
#
# for filename in os.listdir(src_dir):
#     if filename.endswith('.png') and '_HAT_SRx4_ImageNet-pretrain' in filename:
#         # Split before the model name suffix
#         new_name = filename.replace('_HAT_SRx4_ImageNet-pretrain', '')
#         src_path = os.path.join(src_dir, filename)
#         dst_path = os.path.join(dst_dir, new_name)
#         os.rename(src_path, dst_path)  # or use shutil.copy(src_path, dst_path) if you want to keep originals
#         print(f'Renamed: {filename} to {new_name}')
#
#

"""
SRFormer
"""

# # Source and destination directory (same for in-place renaming)
# src_dir = '/home/hail/SH/SRFormer/save_image/Atelectasis/'
# dst_dir = src_dir
#
# os.makedirs(dst_dir, exist_ok=True)
#
# for filename in os.listdir(src_dir):
#     if filename.endswith('.png') and '_SRFormer_DF2k_X4' in filename:
#         # Remove the target substring from filename
#         new_name = filename.replace('_SRFormer_DF2k_X4', '')
#         src_path = os.path.join(src_dir, filename)
#         dst_path = os.path.join(dst_dir, new_name)
#         os.rename(src_path, dst_path)
#         print(f'Renamed: {filename} to {new_name}')