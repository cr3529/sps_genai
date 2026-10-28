"""Assignment 3, Part 2: train the spec'd GAN on MNIST using helper_lib, then save it.

Run:  uv run python train_gan.py              (default 30 epochs)
      uv run python train_gan.py --epochs 10
Saves: models/gan_mnist.pth              (used by the /gan/generate endpoint)
       docs/assignment3_gan_samples.png  (64 digits from the trained generator)
       docs/assignment3_gan_progress.png (fixed noise after selected epochs)
       docs/assignment3_gan_losses.png   (discriminator and generator loss per epoch)
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid

from helper_lib import generate_samples, get_data_loader, get_device, get_model, save_model, train_gan

MODEL_PATH = Path("models/gan_mnist.pth")
DOCS = Path("docs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # Pixels scaled to [-1, 1] to match the generator's Tanh output.
    loader = get_data_loader("MNIST", batch_size=args.batch_size, train=True, normalize_to_tanh=True)
    print(f"MNIST training images: {len(loader.dataset)}")

    model = get_model("GAN")
    criterion = nn.BCELoss()
    optimizer = {
        "generator": optim.Adam(model.generator.parameters(), lr=args.lr, betas=(0.5, 0.999)),
        "discriminator": optim.Adam(model.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999)),
    }

    # Train in chunks so we can snapshot the same noise vectors as training progresses.
    fixed_noise = torch.randn(16, model.z_dim)
    checkpoints = sorted({1, 5, 10, args.epochs // 2, args.epochs} & set(range(1, args.epochs + 1)))
    history = {"epoch": [], "d_loss": [], "g_loss": [], "d_real": [], "d_fake": []}
    snapshots, done = [], 0
    for stop in checkpoints:
        train_gan(model, loader, criterion, optimizer, device=device, epochs=stop - done, log_every=0)
        for k in history:
            values = model.history[k]
            history[k] += [v + done for v in values] if k == "epoch" else values
        done = stop
        model.generator.eval()
        with torch.no_grad():
            snapshots.append((stop, (model.generator(fixed_noise.to(device)).cpu() + 1) / 2))
        print(f"== finished epoch {stop}")

    save_model(model, MODEL_PATH)
    print(f"Saved weights to {MODEL_PATH}")

    DOCS.mkdir(exist_ok=True)
    generate_samples(model, device=device, num_samples=64, seed=0, show=False,
                     save_path=DOCS / "assignment3_gan_samples.png")

    fig, axes = plt.subplots(1, len(snapshots), figsize=(2.6 * len(snapshots), 2.9))
    for ax, (ep, imgs) in zip(axes, snapshots):
        ax.imshow(make_grid(imgs, nrow=4, pad_value=1.0).permute(1, 2, 0))
        ax.set_title(f"after epoch {ep}")
        ax.axis("off")
    fig.suptitle("Same 16 noise vectors as training progresses")
    fig.tight_layout()
    fig.savefig(DOCS / "assignment3_gan_progress.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["epoch"], history["d_loss"], label="Discriminator loss")
    ax.plot(history["epoch"], history["g_loss"], label="Generator loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss (epoch average)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS / "assignment3_gan_losses.png", dpi=120)
    plt.close(fig)

    last = -1
    print(
        f"Final epoch: D loss {history['d_loss'][last]:.4f}  G loss {history['g_loss'][last]:.4f}  "
        f"D(real) {history['d_real'][last]:.3f}  D(fake) {history['d_fake'][last]:.3f}"
    )


if __name__ == "__main__":
    main()
