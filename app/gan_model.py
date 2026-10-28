"""Assignment 3, Part 1: GAN built to the specified architecture (MNIST, 28x28 grayscale).

Generator
  Input:  noise vector               [batch, 100]
  Linear  100 -> 7*7*128, reshape    [batch, 128, 7, 7]
  ConvTranspose2d 128 -> 64, k=4, s=2, p=1  -> [batch, 64, 14, 14]
  BatchNorm2d(64) + ReLU
  ConvTranspose2d 64 -> 1,  k=4, s=2, p=1   -> [batch, 1, 28, 28]
  Tanh  (pixels in [-1, 1], matching the Normalize((0.5,), (0.5,)) used on real images)

Discriminator
  Input:  image                      [batch, 1, 28, 28]
  Conv2d 1 -> 64,   k=4, s=2, p=1    -> [batch, 64, 14, 14]
  LeakyReLU(0.2)
  Conv2d 64 -> 128, k=4, s=2, p=1    -> [batch, 128, 7, 7]
  BatchNorm2d(128) + LeakyReLU(0.2)
  Flatten -> Linear 7*7*128 -> 1     -> [batch, 1]
  Sigmoid  (probability that the image is real)

Size check for every (transposed) conv above, k=4, s=2, p=1:
  Conv2d:          (n - k + 2p) / s + 1   = (28 - 4 + 2) / 2 + 1 = 14,  (14 - 4 + 2) / 2 + 1 = 7
  ConvTranspose2d: (n - 1) * s - 2p + k   = (7 - 1) * 2 - 2 + 4 = 14,   (14 - 1) * 2 - 2 + 4 = 28
"""

import torch
import torch.nn as nn

Z_DIM = 100
IMAGE_SIZE = 28


class Generator(nn.Module):
    def __init__(self, z_dim: int = Z_DIM):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z.view(z.size(0), -1))  # accepts [B, 100] or [B, 100, 1, 1]
        x = x.view(-1, 128, 7, 7)
        x = self.relu(self.bn1(self.deconv1(x)))  # [B, 64, 14, 14]
        return self.tanh(self.deconv2(x))  # [B, 1, 28, 28]


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.conv1(x))  # [B, 64, 14, 14]
        x = self.act2(self.bn2(self.conv2(x)))  # [B, 128, 7, 7]
        return self.sigmoid(self.fc(self.flatten(x)))  # [B, 1], probability "real"


class GAN(nn.Module):
    """Holds the two networks so the helper library can pass a GAN around as one model."""

    def __init__(self, z_dim: int = Z_DIM):
        super().__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim)
        self.discriminator = Discriminator()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)
