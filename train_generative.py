"""Train the Assignment 4 generators on all 50,000 CIFAR-10 training images.

Examples:
  uv run python train_generative.py --model diffusion --epochs 12
  uv run python train_generative.py --model energy --epochs 10 --langevin-steps 30
  uv run python train_generative.py --model diffusion --epochs 5 --resume
"""

import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from app.diffusion_model import DiffusionModel
from app.energy_model import EnergyModel, langevin_sample
from helper_lib.generative_training import train_diffusion, train_energy
from helper_lib.utils import get_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["energy", "diffusion"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--langevin-steps", type=int, default=30)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    device = torch.device(args.device) if args.device else get_device()
    dataset = CIFAR10("data", train=True, download=True)
    # Keep uint8 images in RAM, normalize once. No resizing or data subset.
    images = torch.from_numpy(dataset.data.copy()).permute(0, 3, 1, 2).float().div_(127.5).sub_(1)
    loader = DataLoader(TensorDataset(images, torch.tensor(dataset.targets)),
                        batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(42), num_workers=0)
    model = DiffusionModel() if args.model == "diffusion" else EnergyModel()
    model.to(device)
    print(f"model={args.model} device={device} images={len(images)} "
          f"parameters={sum(p.numel() for p in model.parameters())}", flush=True)
    path = Path(f"models/{args.model}_cifar10.pth")
    resume_path = Path(f"data/{args.model}_training_resume.pth")
    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    path.parent.mkdir(exist_ok=True)
    params = model.network.parameters() if args.model == "diffusion" else model.parameters()
    optimizer = (torch.optim.AdamW(params, lr=2e-4, weight_decay=1e-4)
                 if args.model == "diffusion" else
                 torch.optim.Adam(params, lr=1e-4, betas=(0.0, 0.999)))
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        model.history = checkpoint["history"]
        start_epoch = checkpoint["epoch"]
        if "replay_buffer" in checkpoint:
            model.replay_buffer = checkpoint["replay_buffer"]
    metadata = dict(model=args.model, dataset="CIFAR-10", training_images=50000,
                    image_shape=[3, 32, 32], batch_size=args.batch_size, seed=42,
                    device=str(device), torch_version=str(torch.__version__),
                    normalization="x / 127.5 - 1", timesteps=1000,
                    base_channels=24, langevin_steps=args.langevin_steps,
                    step_size=10.0, noise_std=0.005, regularization=0.1,
                    replay_buffer_size=1024, restart_probability=0.05,
                    learning_rate=2e-4 if args.model == "diffusion" else 1e-4)
    started = time.time()

    def save_checkpoint(trained, row):
        state = {k: v.detach().cpu() for k, v in trained.state_dict().items()}
        published = dict(state_dict=state, metadata={**metadata, "epochs": row["epoch"]})
        temp = path.with_suffix(".tmp")
        torch.save(published, temp)
        temp.replace(path)
        resume = dict(published, history=trained.history, epoch=row["epoch"],
                      optimizer=optimizer.state_dict())
        if args.model == "energy":
            resume["replay_buffer"] = trained.replay_buffer.detach().cpu()
        torch.save(resume, resume_path)
        (docs / f"assignment4_{args.model}_training.json").write_text(json.dumps(
            {"metadata": published["metadata"], "history": trained.history}, indent=2))

    if args.model == "diffusion":
        train_diffusion(model, loader, optimizer=optimizer, device=device, epochs=args.epochs,
                        callback=save_checkpoint, start_epoch=start_epoch)
    else:
        train_energy(model, loader, optimizer=optimizer, device=device, epochs=args.epochs,
                     steps=args.langevin_steps, callback=save_checkpoint, start_epoch=start_epoch)
    model.eval()
    generator = torch.Generator(device=device).manual_seed(7)
    if args.model == "diffusion":
        samples = model.sample(36, 100, generator)
    else:
        initial = torch.rand(36, 3, 32, 32, generator=generator, device=device) * 2 - 1
        samples = (langevin_sample(model, initial, steps=256, generator=generator) + 1) / 2
    save_image(samples.cpu(), docs / f"assignment4_{args.model}_samples.png", nrow=6, padding=2)
    print(f"Finished in {time.time() - started:.1f} seconds; saved {path}", flush=True)


if __name__ == "__main__":
    main()
