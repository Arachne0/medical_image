import os
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

# Configuration
SR_DIRS = {
    "Atelectasis": "/home/hail/SH/medical_image/swinir/results/bicubic/Atelectasis",
    "No Finding": "/home/hail/SH/medical_image/swinir/results/bicubic/No Finding"
}
MODEL_PATH = "/home/hail/SH/medical_image/classification/save_folder/resnet50_binary_classifier.pth"
IMG_SIZE = 256
BATCH_SIZE = 32

# Transform (same as training)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Custom Dataset for SR images
class SRXrayDataset(Dataset):
    def __init__(self, root_dirs: dict, transform=None):
        self.img_paths = []
        self.labels = []
        self.transform = transform

        for label_name, dir_path in root_dirs.items():
            paths = glob(os.path.join(dir_path, "*.png")) + glob(os.path.join(dir_path, "*.jpg"))
            label = 1 if label_name == "Atelectasis" else 0
            self.img_paths.extend(paths)
            self.labels.extend([label] * len(paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# Load dataset and DataLoader
sr_dataset = SRXrayDataset(SR_DIRS, transform=transform)
sr_loader = DataLoader(sr_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(device)
model.eval()

# Evaluate
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in tqdm(sr_loader, desc="Evaluating SR classification"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        preds = (torch.sigmoid(outputs) > 0.5).int().squeeze()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Accuracy on SR images: {accuracy:.4f}")
