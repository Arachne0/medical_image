from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
import pandas as pd

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels , transform = True):
        self.image_paths = image_paths
        self.labels = labels 
        self.transform = transform

        self.default_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        img = self.default_transform(img)
        label = int(self.labels.iloc[idx]["label"])
        return img, label