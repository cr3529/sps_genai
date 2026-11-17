"""Small shared helpers: pick a device, save and load weights."""

from pathlib import Path

import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_model(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path: str | Path, device: str | torch.device = "cpu") -> nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.to(device)
    model.eval()
    return model
