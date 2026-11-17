"""Time-conditioned residual UNet, DDPM noise loss, and DDIM generation."""

import copy
import math

import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    """Interleaved sin/cos; emb[2i] = sin(t / max_period**(2i/d))."""

    def __init__(self, dim=64, max_period=10000):
        super().__init__()
        if dim < 2 or dim % 2:
            raise ValueError("Embedding dimension must be positive and even")
        self.register_buffer("frequencies", torch.exp(
            -math.log(max_period) * torch.arange(0, dim, 2).float() / dim))

    def forward(self, timesteps):
        angles = timesteps.float().reshape(-1, 1) * self.frequencies.reshape(1, -1)
        return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (nn.Identity() if in_channels == out_channels else
                     nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x, time):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(time))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """RGB 32x32 -> 16 -> 8 -> 4 bottleneck -> 8 -> 16 -> 32."""

    def __init__(self, base_channels=24, time_dim=96):
        super().__init__()
        widths = [base_channels * n for n in (1, 2, 3, 4)]
        self.time_embedding = nn.Sequential(SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.input = nn.Conv2d(3, widths[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for a, b in zip(widths[:-1], widths[1:]):
            self.down_blocks.append(ResidualBlock(a, a, time_dim))
            self.downsample.append(nn.Conv2d(a, b, 4, stride=2, padding=1))
        self.middle = ResidualBlock(widths[-1], widths[-1], time_dim)
        self.upsample = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for a, b in zip(widths[:0:-1], widths[-2::-1]):
            self.upsample.append(nn.Conv2d(a, b, 3, padding=1))
            self.up_blocks.append(ResidualBlock(b * 2, b, time_dim))
        self.output = nn.Sequential(nn.GroupNorm(8, widths[0]), nn.SiLU(),
                                    nn.Conv2d(widths[0], 3, 3, padding=1))

    def forward(self, noisy_images, timesteps):
        time = self.time_embedding(timesteps)
        x, skips = self.input(noisy_images), []
        for block, down in zip(self.down_blocks, self.downsample):
            x = block(x, time)
            skips.append(x)
            x = down(x)
        x = self.middle(x, time)
        for up, block, skip in zip(self.upsample, self.up_blocks, reversed(skips)):
            x = up(F.interpolate(x, size=skip.shape[-2:], mode="nearest"))
            x = block(torch.cat((x, skip), dim=1), time)
        return self.output(x)


class DiffusionModel(nn.Module):
    def __init__(self, timesteps=1000, base_channels=24):
        super().__init__()
        self.timesteps = timesteps
        self.base_channels = base_channels
        self.network = UNet(base_channels)
        self.ema_network = copy.deepcopy(self.network).requires_grad_(False)
        # Cosine alpha_bar schedule. Work in float64 when constructing ratios.
        s = 0.008
        t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        curve = torch.cos(((t / timesteps + s) / (1 + s)) * math.pi / 2).square()
        curve = curve / curve[0]
        betas = (1 - curve[1:] / curve[:-1]).clamp(0.0001, 0.999).float()
        self.register_buffer("alpha_bars", torch.cumprod(1 - betas, dim=0))

    def forward(self, images, timesteps):
        return self.network(images, timesteps)

    def noisy_images(self, clean, timesteps, noise):
        alpha = self.alpha_bars[timesteps][:, None, None, None]
        return alpha.sqrt() * clean + (1 - alpha).sqrt() * noise

    @torch.no_grad()
    def update_ema(self, decay=0.995):
        for averaged, current in zip(self.ema_network.parameters(), self.network.parameters()):
            averaged.lerp_(current, 1 - decay)

    @torch.no_grad()
    def sample(self, num_samples=16, steps=100, generator=None, initial_noise=None):
        device = self.alpha_bars.device
        x = (torch.randn((num_samples, 3, 32, 32), generator=generator, device=device)
             if initial_noise is None else initial_noise.clone().to(device))
        times = torch.linspace(self.timesteps - 1, 0, steps, device=device).long()
        for i, t in enumerate(times):
            ts = torch.full((x.shape[0],), int(t), device=device, dtype=torch.long)
            predicted_noise = self.ema_network(x, ts)
            alpha = self.alpha_bars[t]
            clean = ((x - (1 - alpha).sqrt() * predicted_noise) / alpha.sqrt()).clamp(-1, 1)
            if i + 1 == len(times):
                x = clean
            else:
                next_alpha = self.alpha_bars[times[i + 1]]
                # Recompute epsilon after clipping x0 for a consistent DDIM step.
                epsilon = (x - alpha.sqrt() * clean) / (1 - alpha).sqrt()
                x = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * epsilon
        return ((x + 1) / 2).clamp(0, 1)
