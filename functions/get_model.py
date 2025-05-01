import json
import numpy as np
import os
from collections import defaultdict

# Load original label file
with open('/home/hail/Desktop/medical_image_project/datasets/train_labels.json', 'r') as f:
    full_labels = json.load(f)

# Count frequency for each label
label_count = defaultdict(int)
for labels in full_labels.values():
    for i, val in enumerate(labels):
        if val == 1:
            label_count[i] += 1

# Define proportion to keep per label (e.g., 20% of each label)
label_ratio = {i: 0.2 for i in range(14)}  # Modify this ratio as needed

# Track how many samples we've kept per label
kept_count = defaultdict(int)
selected_samples = dict()

for img_name, labels in full_labels.items():
    keep = False
    for i, val in enumerate(labels):
        if val == 1 and kept_count[i] < int(label_count[i] * label_ratio[i]):
            keep = True
    if keep:
        selected_samples[img_name] = labels
        for i, val in enumerate(labels):
            if val == 1:
                kept_count[i] += 1

print(f"Selected {len(selected_samples)} samples out of {len(full_labels)}")

# Save filtered label file
output_path = '/home/hail/Desktop/medical_image_project/datasets/train_labels_subset.json'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(selected_samples, f, indent=2)
