import numpy as np
import torch
import sys

from network_mlgan import Generator, Discriminator


def main():
    sys.path.append('/home/hail/Desktop/medical_image_project/functions')
    from functions.nih_loader import nih_loader

    z_dim = 100
    batch_size = 1
    NUM_LABELS = 14

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(z_dim=z_dim).to(device)
    D = Discriminator().to(device)

    train_loader, test_loader = nih_loader(batch_size=batch_size, resize=True)

    for real_imgs, labels in train_loader:
        real_imgs = real_imgs[:, 0:1, :, :].to(device)  # use only one channel if RGB
        labels = labels.to(device)
        z = torch.randn(real_imgs.size(0), z_dim).to(device)

        fake_imgs = G(z, labels)
        output = D(fake_imgs, labels)

        print("Real batch shape:", real_imgs.shape)
        print("Generated fake batch shape:", fake_imgs.shape)
        print("Discriminator output:", output.shape)
        break


if __name__ == '__main__':
    main()

