import os
from glob import glob
from PIL import Image
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

def main(method):
    # Configuration
    SR_DIRS = {
        "Atelectasis": f"/home/hail/SH/medical_image/swinir/results/{method}/Atelectasis",
        "No Finding": f"/home/hail/SH/medical_image/swinir/results/{method}/No Finding"
    }
    MODEL_PATH = "/home/hail/SH/medical_image/classification/save_folder/resnet50_binary_classifier.pth"
    IMG_SIZE = 256
    BATCH_SIZE = 32
    SAMPLE_PER_CLASS = 2500

    # Transform (same as training)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Custom Dataset for SR images (balanced sampling)
    class SRXrayDataset(Dataset):
        def __init__(self, root_dirs: dict, transform=None, sample_per_class=2500):
            self.img_paths = []
            self.labels = []
            self.transform = transform

            for label_name, dir_path in root_dirs.items():
                paths = glob(os.path.join(dir_path, "*.png")) + glob(os.path.join(dir_path, "*.jpg"))
                if len(paths) >= sample_per_class:
                    sampled = random.sample(paths, sample_per_class)
                else:
                    sampled = paths  # use all if not enough
                label = 1 if label_name == "Atelectasis" else 0
                self.img_paths.extend(sampled)
                self.labels.extend([label] * len(sampled))

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            img = Image.open(self.img_paths[idx]).convert("RGB")
            label = self.labels[idx]
            if self.transform:
                img = self.transform(img)
            return img, label

    # Load dataset and DataLoader
    sr_dataset = SRXrayDataset(SR_DIRS, transform=transform, sample_per_class=SAMPLE_PER_CLASS)
    sr_loader = DataLoader(sr_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(device)
    model.eval()

    # Evaluation: TP, FP, FN, TN
    TP = FP = FN = TN = 0

    with torch.no_grad():
        for inputs, labels in tqdm(sr_loader, desc="Evaluating SR classification"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).int().squeeze()

            for pred, label in zip(preds, labels):
                if label == 1 and pred == 1:
                    TP += 1
                elif label == 0 and pred == 1:
                    FP += 1
                elif label == 1 and pred == 0:
                    FN += 1
                elif label == 0 and pred == 0:
                    TN += 1

    # Compute metrics
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")


if __name__ == "__main__":
    main("lanczos")
