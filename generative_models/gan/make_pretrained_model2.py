import os
import torch
import torch.nn as nn
import wandb

from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, Grayscale
from network_gan import Discriminator  # Use the same Discriminator used in the GAN training

from tqdm import tqdm


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith('.png')
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)


    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = read_image(img_path).float() / 255.0  # shape: [C, H, W]

        # If RGBA or RGB, convert manually to grayscale
        if image.shape[0] == 4:
            image = image[:3]  # Discard alpha
        if image.shape[0] == 3:
            # Convert to grayscale using luminosity method: Y = 0.299 R + 0.587 G + 0.114 B
            r, g, b = image[0], image[1], image[2]
            image = 0.299 * r + 0.587 * g + 0.114 * b
            image = image.unsqueeze(0)  # shape: [1, H, W]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(1.0)



def main(label_name):
    image_dir = f"/home/hail/Desktop/medical_image_project/datasets/labeled/{label_name}"
    save_dir = "/home/hail/Desktop/medical_image_project/datasets/model_files"
    os.makedirs(save_dir, exist_ok=True)

    # Count the number of PNG images in the directory
    image_files = [
        f for f in os.listdir(image_dir)
        if f.endswith('.png')
    ]
    num_images = len(image_files)
    print(f"[INFO] Found {num_images} PNG images for label '{label_name}' in {image_dir}.")

    # Define image transformations to match GAN training size
    transform = Compose([
        Resize((256, 256)),  # Resize to match GAN input size
        Grayscale(),         # Convert to single channel
    ])

    # Create dataset and dataloader
    dataset = CustomImageDataset(image_dir, transform=transform)
    data_len = len(dataset)
    batch_size = 32

    epochs = max(1, (data_len // (batch_size *2)))
    lr = 0.0002

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    wandb.init(
        project="medical_image_project",
        entity="hails",
        name=f"gan_pretrained_model_{label_name}",
        config={
            "label_name": label_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr
        }
    )

    # Initialize discriminator
    model = Discriminator(img_shape=(1, 256, 256)).to(device)

    wandb.watch(model, log="all", log_freq=10)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

    model.train()
    for epoch in tqdm(range(epochs), desc="Training epochs"):
        epoch_loss = 0.0
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.view(-1, 1).to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})

    # Save trained model
    save_path = os.path.join(save_dir, f"{label_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    wandb.finish()


if __name__ == "__main__":
    main("Cardiomegaly")