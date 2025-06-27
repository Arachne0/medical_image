import os
import json
import shutil
from tqdm import tqdm  # Import tqdm for progress bar

# Define paths
image_dir = "/home/hail/Desktop/medical_image_project/datasets/train"
label_path = "/home/hail/Desktop/medical_image_project/datasets/train_labels.json"
output_root = "/home/hail/Desktop/medical_image_project/datasets/labeled"

# NIH Chest X-ray class names (14 classes)
class_names = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

# Add one more for "No Finding"
class_names_with_normal = class_names + ["No_Finding"]

# Create output directories
for name in class_names_with_normal:
    os.makedirs(os.path.join(output_root, name), exist_ok=True)

# Load label file
with open(label_path, "r") as f:
    labels = json.load(f)

# Process each image with tqdm
for img_name, label_vector in tqdm(labels.items(), desc="Copying images"):
    src_path = os.path.join(image_dir, img_name)
    if not os.path.exists(src_path):
        continue  # Skip missing files

    # Check if all labels are 0 -> No Finding
    if isinstance(label_vector, list) and all(v == 0 for v in label_vector):
        dst_path = os.path.join(output_root, "No_Finding", img_name)
        shutil.copy2(src_path, dst_path)
    else:
        for idx, v in enumerate(label_vector):
            if v == 1:
                label = class_names[idx]
                dst_path = os.path.join(output_root, label, img_name)
                shutil.copy2(src_path, dst_path)

print("Done: Images copied to 15 labeled folders including 'No_Finding'.")
