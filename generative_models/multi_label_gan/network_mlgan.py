import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_LABELS = 14
IMG_SIZE = 256

class Generator(nn.Module):
    def __init__(self, z_dim=100, label_dim=NUM_LABELS, img_channels=1):
        super(Generator, self).__init__()
        self.fc = nn.Linear(z_dim + label_dim, 1024 * 4 * 4)

        self.deconv = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, img_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = torch.cat([z, labels], dim=1)
        x = self.fc(x)
        x = x.view(-1, 1024, 4, 4)
        return self.deconv(x)


class Discriminator(nn.Module):
    def __init__(self, label_dim=NUM_LABELS, img_channels=1):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Linear(label_dim, IMG_SIZE * IMG_SIZE)

        self.conv = nn.Sequential(
            nn.Conv2d(img_channels + 1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 2, 1)
        )

    def forward(self, img, labels):
        # labels: (B, 14) -> (B, 1, H, W)
        label_map = self.label_embedding(labels).view(-1, 1, IMG_SIZE, IMG_SIZE)
        x = torch.cat([img, label_map], dim=1)
        return self.conv(x).view(-1)
