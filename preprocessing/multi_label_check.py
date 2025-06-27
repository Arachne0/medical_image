import os
import json

# Define path to the label file
label_path = "/home/hail/Desktop/medical_image_project/datasets/train_labels.json"

# Load label file
with open(label_path, "r") as f:
    labels = json.load(f)

# Count multi-label images
total = 0
multi_label_count = 0
for img_name, label_vector in labels.items():
    if not isinstance(label_vector, list):
        continue  # Skip invalid data
    label_sum = sum(label_vector)
    if label_sum > 1:
        multi_label_count += 1
    total += 1

print(f"Total images with labels: {total}")
print(f"Multi-label images: {multi_label_count}")
print(f"Single-label or No Finding images: {total - multi_label_count}")
