"""Sample new images from a trained generative model (GAN or VAE) and plot them on a grid."""

import math

import torch
from torchvision.utils import make_grid


def generate_samples(model, device="cpu", num_samples=10, seed=None, show=True, save_path=None):
    """Draw num_samples points from the latent space N(0, I), decode them, and plot a grid.

    Works for the GAN (model.generator) and the VAE (model.decode).
    Returns the images as a [num_samples, C, H, W] tensor with pixels in [0, 1].
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")  # no window, e.g. when called from the API or a script
    import matplotlib.pyplot as plt

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    model.to(device)
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.z_dim, generator=generator).to(device)
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
