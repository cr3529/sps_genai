"""Load trained Assignment 4 checkpoints and expose CPU image generation."""

import io
import math
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from app.diffusion_model import DiffusionModel
from app.energy_model import EnergyModel, langevin_sample

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


class CifarGenerator:
    def __init__(self, kind):
        if kind not in ("energy", "diffusion"):
            raise ValueError("Unknown generator")
        path = MODEL_DIR / f"{kind}_cifar10.pth"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path.name}. Run train_generative.py --model {kind} first.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.metadata = checkpoint["metadata"]
        self.kind = kind
        self.model = (DiffusionModel(timesteps=self.metadata["timesteps"],
                                    base_channels=self.metadata["base_channels"])
                      if kind == "diffusion" else EnergyModel())
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval().requires_grad_(False)

    def generate(self, num_samples=4, steps=100, seed=None):
        # Each request gets an independent RNG. Global seeds/weights never change.
        generator = torch.Generator()
        generator.manual_seed(seed) if seed is not None else generator.seed()
        if self.kind == "diffusion":
            return self.model.sample(num_samples, steps, generator)
        initial = torch.rand(num_samples, 3, 32, 32, generator=generator) * 2 - 1
        samples = langevin_sample(self.model, initial, steps=steps, generator=generator)
        return ((samples + 1) / 2).clamp(0, 1)

    def generate_png(self, num_samples=4, steps=100, seed=None):
        samples = self.generate(num_samples, steps, seed)
        samples = torch.nn.functional.interpolate(samples, scale_factor=4, mode="nearest")
        grid = make_grid(samples, nrow=math.ceil(math.sqrt(num_samples)), padding=4, pad_value=1)
        buffer = io.BytesIO()
        save_image(grid, buffer, format="png")
        return buffer.getvalue()
