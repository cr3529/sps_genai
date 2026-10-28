"""Assignment 3, Part 2: loads the trained GAN generator and turns random noise into digit images."""

import io
import math
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from app.gan_model import Z_DIM, Generator

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "gan_mnist.pth"
MAX_SAMPLES = 64


class DigitGenerator:
    def __init__(self, weights_path: Path = DEFAULT_WEIGHTS):
        self.generator = Generator(Z_DIM)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        # The checkpoint holds the whole GAN; keep only the generator's weights.
        gen_weights = {k.removeprefix("generator."): v for k, v in state_dict.items() if k.startswith("generator.")}
        self.generator.load_state_dict(gen_weights)
        self.generator.eval()  # batch norm uses its running statistics at inference

    def generate(self, num_samples: int = 16, seed: int | None = None) -> torch.Tensor:
        """Return [num_samples, 1, 28, 28] images with pixels in [0, 1]."""
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        z = torch.randn(num_samples, Z_DIM, generator=g)
        with torch.no_grad():
            return ((self.generator(z) + 1) / 2).clamp(0, 1)  # Tanh [-1, 1] -> [0, 1]

    def generate_png(self, num_samples: int = 16, seed: int | None = None, scale: int = 4) -> bytes:
        """Generate a grid of digits and return it as PNG bytes (enlarged `scale` times for viewing)."""
        images = self.generate(num_samples, seed)
        if scale > 1:
            images = torch.nn.functional.interpolate(images, scale_factor=scale, mode="nearest")
        grid = make_grid(images, nrow=math.ceil(math.sqrt(num_samples)), padding=2 * scale, pad_value=1.0)
        buffer = io.BytesIO()
        save_image(grid, buffer, format="png")
        return buffer.getvalue()
