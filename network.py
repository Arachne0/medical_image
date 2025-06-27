from re import X
import torch
import torch.nn as nn
import torchvision.models as models


class VGGEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(VGGEncoder, self).__init__()
        base_model = models.vgg16_bn(pretrained=True)
        self.features = base_model.features  # Conv layers
        self.avgpool = base_model.avgpool
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        x = self.features(x)  # [B, 512, 7, 7] for 224 input
        x = self.avgpool(x)  # -> [B, 512, 7, 7]
        x = x.view(x.size(0), -1)  # flatten
        z = self.fc(x)  # -> latent vector
        return z


class Classifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

