import argparse
import os
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

from functions.nih_loader import nih_loader
from generative_models.gan.gan import dataloader
from network_gan import Generator, Discriminator


def main():
    os.makedirs("images", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--b1", type=float, default=0.5)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--n_cpu", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=28)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--sample_interval", type=int, default=400)
    args = parser.parse_args()

    img_shape = (args.channels, args.img_size, args.img_size)
    cuda = torch.cuda.is_available()

    generator = Generator(args.latent_dim, img_shape)
    discriminator = Discriminator(img_shape)

    if cuda:
        generator.cuda()
        discriminator.cuda()

    adversarial_loss = nn.BCELoss().cuda() if cuda else nn.BCELoss()

    os.makedirs("results", exist_ok=True)

    dataloader, _ = nih_loader(batch_size=opt.batch_size, num_workers=opt.n_cpu, resize=True)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
            real_imgs = Variable(imgs.type(Tensor))

            optimizer_G.zero_grad()
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f]"
                % (epoch, args.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % args.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)


if __name__ == "__main__":
    main()
