import os
import random
import wandb

from PIL import Image
from glob import glob
from tqdm import tqdm
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.optim as optim

# Configuration
TRAIN_DIRS = {
    "Atelectasis": "/home/hail/SH/medical_image/datasets/train/original/Atelectasis",
    "No Finding": "/home/hail/SH/medical_image/datasets/train/original/No Finding"
}
TEST_DIRS = {
    "Atelectasis": "/home/hail/SH/medical_image/datasets/test/original/Atelectasis",
    "No Finding": "/home/hail/SH/medical_image/datasets/test/original/No Finding"
}
IMG_SIZE = 256
TRAIN_SAMPLES_PER_CLASS = 3000
TEST_SAMPLES_PER_CLASS = 1000
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "save_folder/resnet50_binary_classifier.pth"

# Set seed
random.seed(42)
torch.manual_seed(42)

# Transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom Dataset
class ChestXrayFolderDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# Helper to sample data
def sample_data(dir_dict, samples_per_class):
    all_paths = []
    all_labels = []
    for label_name, path in dir_dict.items():
        paths = glob(os.path.join(path, "*.png")) + glob(os.path.join(path, "*.jpg")) + glob(os.path.join(path, "*.jpeg"))
        sampled = random.sample(paths, min(samples_per_class, len(paths)))
        label = 1 if label_name == "Atelectasis" else 0
        all_paths.extend(sampled)
        all_labels.extend([label] * len(sampled))
    return all_paths, all_labels

# Prepare datasets
train_paths, train_labels = sample_data(TRAIN_DIRS, TRAIN_SAMPLES_PER_CLASS)
test_paths, test_labels = sample_data(TEST_DIRS, TEST_SAMPLES_PER_CLASS)

# Create Datasets and Loaders
train_dataset = ChestXrayFolderDataset(train_paths, train_labels, transform)
test_dataset = ChestXrayFolderDataset(test_paths, test_labels, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)  # Binary classification
model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

wandb.init(
    entity="hails",
    project="medical_image_project",
    name="resnet50_binary_classifier",
    config={
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "img_size": IMG_SIZE,
        "model": "resnet50"
    }
)

# Training loop with tqdm
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for inputs, labels in train_bar:
        inputs = inputs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())

        # log loss per batch
        wandb.log({"train/loss_step": loss.item()})

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Avg Train Loss: {total_loss / len(train_loader):.4f}")

    # log avg loss per epoch
    wandb.log({
        "train/loss_epoch": avg_train_loss,
        "epoch": epoch + 1
    })

    # Save model at each epoch
    torch.save(model.state_dict(), MODEL_SAVE_PATH)


# Evaluation with tqdm
model.eval()
correct = 0
total = 0
test_bar = tqdm(test_loader, desc="Evaluating")

with torch.no_grad():
    for inputs, labels in test_bar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        preds = (torch.sigmoid(outputs) > 0.5).int().squeeze()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print(f"Test Accuracy: {test_acc:.4f}")

# log test accuracy
wandb.log({"test/accuracy": test_acc})