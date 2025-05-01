import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ChestXrayDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        with open(label_file, 'r') as f:
            self.labels = json.load(f)

        self.image_files = list(self.labels.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.labels[img_name], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


def nih_loader(batch_size=32, num_workers=4, resize=True):

    if resize:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = None

    train_dataset = ChestXrayDataset(
        image_dir='/home/hail/Desktop/medical_image_project/datasets/train',
        label_file='/home/hail/Desktop/medical_image_project/datasets/train_labels.json',
        transform=transform
    )

    test_dataset = ChestXrayDataset(
        image_dir='/home/hail/Desktop/medical_image_project/datasets/test',
        label_file='/home/hail/Desktop/medical_image_project/datasets/test_labels.json',
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader