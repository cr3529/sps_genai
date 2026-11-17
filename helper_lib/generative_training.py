"""Explicit parameter and input-gradient training loops for Assignment 4."""

import time

import torch
import torch.nn.functional as F

from app.energy_model import langevin_sample


def _record(model, epoch, values, started, callback):
    row = {"epoch": epoch, **values, "seconds": round(time.time() - started, 2)}
    model.history.append(row)
    print(row, flush=True)
    if callback:
        callback(model, row)


def train_diffusion(model, data_loader, criterion=None, optimizer=None,
                    device="cpu", epochs=10, callback=None, start_epoch=0):
    model.to(device)
    criterion = criterion or F.mse_loss
    optimizer = optimizer or torch.optim.AdamW(model.network.parameters(), lr=2e-4)
    model.history = getattr(model, "history", [])
    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        started = time.time()
        model.network.train()
        total, count = 0.0, 0
        for batch, _ in data_loader:
            images = batch.to(device)
            times = torch.randint(model.timesteps, (len(images),), device=device)
            noise = torch.randn_like(images)
            noisy = model.noisy_images(images, times, noise)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(noisy, times), noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.network.parameters(), 1.0)
            optimizer.step()
            model.update_ema()
            total += loss.item() * len(images)
            count += len(images)
        model.ema_network.eval()
        _record(model, epoch, {"noise_mse": total / count}, started, callback)
    return model


def train_energy(model, data_loader, criterion=None, optimizer=None,
                 device="cpu", epochs=10, steps=30, step_size=10.0,
                 noise_std=0.005, alpha=0.1, buffer_size=1024, callback=None,
                 start_epoch=0):
    """Persistent contrastive divergence with 5% fresh-noise restarts."""
    model.to(device)
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))
    buffer = getattr(model, "replay_buffer", None)
    if buffer is None:
        buffer = torch.rand(buffer_size, 3, 32, 32, device=device) * 2 - 1
    else:
        buffer = buffer.to(device)
    model.history = getattr(model, "history", [])
    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        started = time.time()
        totals = dict(loss=0.0, real_energy=0.0, fake_energy=0.0, contrastive_loss=0.0)
        count = 0
        for batch, _ in data_loader:
            real = batch.to(device)
            real = (real + torch.randn_like(real) * noise_std).clamp(-1, 1)
            indices = torch.randperm(len(buffer), device=device)[:len(real)]
            initial = buffer[indices].clone()
            restart = torch.rand((len(real), 1, 1, 1), device=device) < 0.05
            initial = torch.where(restart, torch.rand_like(initial) * 2 - 1, initial)
            optimizer.zero_grad(set_to_none=True)
            fake = langevin_sample(model, initial, steps, step_size, noise_std)
            buffer[indices] = fake
            # The fake samples are detached: the contrastive update changes weights only.
            model.train()
            real_energy, fake_energy = model(torch.cat((real, fake))).split(len(real))
            contrastive = real_energy.mean() - fake_energy.mean()
            loss = contrastive + alpha * (real_energy.square().mean() + fake_energy.square().mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            values = dict(loss=loss.item(), real_energy=real_energy.mean().item(),
                          fake_energy=fake_energy.mean().item(), contrastive_loss=contrastive.item())
            for k, v in values.items():
                totals[k] += v * len(real)
            count += len(real)
        model.replay_buffer = buffer.detach()
        _record(model, epoch, {k: v / count for k, v in totals.items()}, started, callback)
    return model
