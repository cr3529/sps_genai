"""Training loops: classifiers (train_model), VAE (train_vae) and GAN (train_gan)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train_model(model, data_loader, criterion, optimizer, device="cpu", epochs=10):
    """Standard supervised loop for FCNN / CNN / EnhancedCNN. Returns the trained model."""
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        print(f"Epoch {epoch + 1}/{epochs}  loss {running_loss / total:.4f}  acc {100 * correct / total:.2f}%")
    return model


def vae_loss(recon, x, mu, logvar):
    """Reconstruction (binary cross-entropy) + KL divergence to N(0, I)."""
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl


def train_vae(model, data_loader, optimizer, device="cpu", epochs=10):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for images, _ in data_loader:
            images = images.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(images)
            loss = vae_loss(recon, images, mu, logvar)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}  loss {total / len(data_loader.dataset):.4f}")
    return model


def _gan_optimizers(model, optimizer, lr=2e-4, betas=(0.5, 0.999)):
    """A GAN needs one optimizer per network. Accepts a dict, a (gen, disc) pair, or None (Adam defaults)."""
    if optimizer is None:
        return (
            optim.Adam(model.generator.parameters(), lr=lr, betas=betas),
            optim.Adam(model.discriminator.parameters(), lr=lr, betas=betas),
        )
    if isinstance(optimizer, dict):
        return optimizer["generator"], optimizer["discriminator"]
    opt_g, opt_d = optimizer
    return opt_g, opt_d


def train_gan(model, data_loader, criterion=None, optimizer=None, device="cpu", epochs=10, log_every=200):
    """Train a GAN (model.generator + model.discriminator) and return the model.

    criterion: nn.BCELoss() by default (the discriminator outputs a probability).
    optimizer: {"generator": opt_g, "discriminator": opt_d}, a (opt_g, opt_d) tuple, or None.

    Each batch:
      1. Discriminator step: maximize log D(x) + log(1 - D(G(z)))
         -> BCE(D(real), 1) + BCE(D(fake), 0), with the fake images detached from the generator.
      2. Generator step (non-saturating loss): maximize log D(G(z))  ->  BCE(D(G(z)), 1)

    Per-epoch average losses are stored on model.history.
    """
    criterion = criterion or nn.BCELoss()
    opt_g, opt_d = _gan_optimizers(model, optimizer)
    gen, disc = model.generator.to(device), model.discriminator.to(device)
    model.history = {"epoch": [], "d_loss": [], "g_loss": [], "d_real": [], "d_fake": []}

    for epoch in range(epochs):
        gen.train()
        disc.train()
        sums = {"d_loss": 0.0, "g_loss": 0.0, "d_real": 0.0, "d_fake": 0.0}
        for batch_idx, (real, _) in enumerate(data_loader):
            real = real.to(device)
            b = real.size(0)
            ones = torch.ones(b, 1, device=device)
            zeros = torch.zeros(b, 1, device=device)

            # 1) Discriminator
            z = torch.randn(b, model.z_dim, device=device)
            fake = gen(z)
            d_real = disc(real)
            d_fake = disc(fake.detach())
            d_loss = criterion(d_real, ones) + criterion(d_fake, zeros)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # 2) Generator: try to make the discriminator call the fakes real
            g_loss = criterion(disc(fake), ones)
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            sums["d_loss"] += d_loss.item()
            sums["g_loss"] += g_loss.item()
            sums["d_real"] += d_real.mean().item()
            sums["d_fake"] += d_fake.mean().item()
            if log_every and batch_idx % log_every == 0:
                print(
                    f"  epoch {epoch + 1}/{epochs}  batch {batch_idx}/{len(data_loader)}"
                    f"  D loss {d_loss.item():.4f}  G loss {g_loss.item():.4f}"
                )

        n = len(data_loader)
        model.history["epoch"].append(epoch + 1)
        for k in sums:
            model.history[k].append(sums[k] / n)
        print(
            f"Epoch {epoch + 1}/{epochs}  D loss {sums['d_loss'] / n:.4f}  G loss {sums['g_loss'] / n:.4f}"
            f"  D(real) {sums['d_real'] / n:.3f}  D(fake) {sums['d_fake'] / n:.3f}"
        )
    return model
