"""Model factory: get_model(name) returns one of FCNN, CNN, EnhancedCNN, VAE, GAN."""

import torch
import torch.nn as nn

from app.cnn_model import AssignmentCNN
from app.gan_model import GAN


class FCNN(nn.Module):
    """Module 4 Practical 1: fully connected network for 28x28 MNIST digits."""

    def __init__(self, input_dim: int = 28 * 28, hidden: int = 128, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class EnhancedCNN(nn.Module):
    """Module 4 Practical 3 style: deeper CNN with batch norm and dropout (3x32x32 CIFAR-10 input)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))  # 32 -> 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class VAE(nn.Module):
    """Module 5: convolutional variational autoencoder for 28x28 MNIST (pixels in [0, 1])."""

    def __init__(self, z_dim: int = 2):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 7x7
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, z_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, z_dim)
        self.fc_decode = nn.Linear(z_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 28x28
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        return self.decoder(self.fc_decode(z).view(-1, 64, 7, 7))

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


MODELS = {
    "FCNN": FCNN,
    "CNN": AssignmentCNN,  # the Assignment 2 spec'd CNN (64x64x3 input)
    "ENHANCEDCNN": EnhancedCNN,
    "VAE": VAE,
    "GAN": GAN,  # the Assignment 3 spec'd GAN (Generator + Discriminator)
}


def get_model(model_name: str, **kwargs) -> nn.Module:
    """Return a new, untrained model: one of FCNN, CNN, EnhancedCNN, VAE, GAN."""
    key = model_name.replace("_", "").replace("-", "").upper()
    if key not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: FCNN, CNN, EnhancedCNN, VAE, GAN")
    model = MODELS[key](**kwargs)
    return model
