"""Sample images from trained GAN, VAE, diffusion, or energy models."""

import math

import torch
from torchvision.utils import make_grid


def generate_samples(model, device="cpu", num_samples=10, seed=None, show=True, save_path=None,
                     diffusion_steps=100, langevin_steps=256):
    """Draw num_samples points from the latent space N(0, I), decode them, and plot a grid.

    Supports GAN/VAE latent sampling, reverse diffusion, and Langevin dynamics.
    Returns the images as a [num_samples, C, H, W] tensor with pixels in [0, 1].
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")  # no window, e.g. when called from the API or a script
    import matplotlib.pyplot as plt

    from app.diffusion_model import DiffusionModel
    from app.energy_model import EnergyModel, langevin_sample
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    model.to(device)
    model.eval()
    if isinstance(model, DiffusionModel):
        images = model.sample(num_samples, diffusion_steps, generator)
    elif isinstance(model, EnergyModel):
        initial = torch.rand(num_samples, 3, 32, 32, generator=generator, device=device) * 2 - 1
        images = (langevin_sample(model, initial, steps=langevin_steps, generator=generator) + 1) / 2
    else:
        with torch.no_grad():
            z = torch.randn(num_samples, model.z_dim, generator=generator, device=device)
            if hasattr(model, "generator"):  # GAN: Tanh output in [-1, 1]
                images = (model.generator(z) + 1) / 2
            else:  # VAE: Sigmoid output already in [0, 1]
                images = model.decode(z)
    images = images.clamp(0, 1).cpu()

    nrow = math.ceil(math.sqrt(num_samples))
    grid = make_grid(images, nrow=nrow, padding=2, pad_value=1.0)
    plt.figure(figsize=(nrow, nrow))
    plt.imshow(grid.permute(1, 2, 0), cmap="gray")
    plt.axis("off")
    plt.title(f"{num_samples} generated samples")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return images
