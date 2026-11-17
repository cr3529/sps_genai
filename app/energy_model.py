"""CIFAR-10 energy network and Langevin sampling (Module 8)."""

import torch
from torch import nn


class EnergyModel(nn.Module):
    """One unnormalized energy per RGB image; lower means more plausible."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Flatten(), nn.Linear(64 * 2 * 2, 64), nn.SiLU(), nn.Linear(64, 1),
        )

    def forward(self, images):
        return self.net(images).squeeze(1)


def langevin_sample(model, images, steps=60, step_size=10.0, noise_std=0.005,
                    generator=None, return_energies=False):
    """Update pixels, not weights. Return detached images in [-1, 1].

    autograd.grad requests only dE/dx and never populates parameter .grad.
    Explicit enable_grad is essential when called by an inference endpoint.
    Each iteration detaches its input so the sampling graph cannot grow.
    """
    images = images.detach().clone()
    energies = []
    with torch.enable_grad():
        for _ in range(steps):
            with torch.no_grad():
                if noise_std:
                    noise = torch.randn(images.shape, device=images.device,
                                        generator=generator) * noise_std
                    images = (images + noise).clamp(-1, 1)
            images = images.detach().requires_grad_(True)
            energy = model(images)
            grad, = torch.autograd.grad(energy.sum(), images, create_graph=False)
            with torch.no_grad():
                images = (images - step_size * grad.clamp(-0.03, 0.03)).clamp(-1, 1)
                if return_energies:
                    energies.append(float(model(images).mean()))
    result = images.detach()
    return (result, energies) if return_energies else result
