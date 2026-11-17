"""Held-out CIFAR-10 evaluation and final training/sample figures."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from app.cifar_generator import CifarGenerator
from app.energy_model import langevin_sample


def main():
    torch.set_num_threads(3)
    torch.manual_seed(123)
    docs = Path("docs")
    test = CIFAR10("data", train=False, download=True)
    images = torch.from_numpy(test.data.copy()).permute(0, 3, 1, 2).float().div_(127.5).sub_(1)
    loader = DataLoader(TensorDataset(images), batch_size=128, shuffle=False)
    diffusion, energy = CifarGenerator("diffusion"), CifarGenerator("energy")
    sums = dict(diffusion_noise_mse=0.0, zero_noise_predictor_mse=0.0,
                real_energy=0.0, uniform_noise_energy=0.0)
    rng = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for (clean,) in loader:
            t = torch.randint(1000, (len(clean),), generator=rng)
            noise = torch.randn(clean.shape, generator=rng)
            noisy = diffusion.model.noisy_images(clean, t, noise)
            prediction = diffusion.model.ema_network(noisy, t)
            sums["diffusion_noise_mse"] += float((prediction-noise).square().mean()) * len(clean)
            sums["zero_noise_predictor_mse"] += float(noise.square().mean()) * len(clean)
            sums["real_energy"] += float(energy.model(clean).sum())
            uniform = torch.rand(clean.shape, generator=rng) * 2 - 1
            sums["uniform_noise_energy"] += float(energy.model(uniform).sum())
    metrics = {k: v / len(images) for k, v in sums.items()}
    initial = torch.rand((16, 3, 32, 32), generator=rng) * 2 - 1
    initial_energy = float(energy.model(initial).mean())
    refined, energies = langevin_sample(energy.model, initial, steps=256,
                                        noise_std=0, return_energies=True)
    metrics.update(held_out_images=len(images),
                   sampling_initial_energy=initial_energy,
                   sampling_final_energy=float(energy.model(refined).mean()),
                   sampling_mean_pixel_change=float((refined-initial).abs().mean()),
                   checkpoint_epochs={"energy":energy.metadata["epochs"],
                                      "diffusion":diffusion.metadata["epochs"]})
    assert metrics["diffusion_noise_mse"] < metrics["zero_noise_predictor_mse"]
    assert metrics["sampling_final_energy"] < initial_energy
    for kind, generator in (("energy", energy), ("diffusion", diffusion)):
        samples = generator.generate(36, 256 if kind == "energy" else 100, seed=7)
        assert torch.isfinite(samples).all()
        enlarged = torch.nn.functional.interpolate(samples, scale_factor=4, mode="nearest")
        save_image(enlarged, docs / f"assignment4_{kind}_samples.png", nrow=6, padding=4, pad_value=1)
    histories = {kind: json.loads((docs/f"assignment4_{kind}_training.json").read_text())["history"]
                 for kind in ("energy", "diffusion")}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5), layout="constrained")
    h = histories["diffusion"]
    axes[0].plot([r["epoch"] for r in h], [r["noise_mse"] for r in h], marker="o", color="#1b5e83")
    axes[0].set(xlabel="Epoch", ylabel="Training noise MSE", title="Diffusion training")
    h = histories["energy"]
    for key, label, color in (("real_energy", "Real images", "#1b5e83"),
                              ("fake_energy", "Replay samples", "#b5682f")):
        axes[1].plot([r["epoch"] for r in h], [r[key] for r in h], label=label, marker="o", color=color)
    axes[1].set(xlabel="Epoch", ylabel="Mean energy", title="Energy model training")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    fig.savefig(docs/"assignment4_training_curves.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")
    ax.plot([initial_energy]+energies, color="#1b5e83")
    ax.set(xlabel="Pixel update", ylabel="Mean energy",
           title="Energy decrease during deterministic refinement")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(docs/"assignment4_energy_descent.png", dpi=180)
    plt.close(fig)
    (docs/"assignment4_evaluation.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
