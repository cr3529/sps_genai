"""Tests of the gradient boundaries and diffusion mathematics."""

import unittest

import torch

from app.diffusion_model import DiffusionModel, SinusoidalEmbedding, UNet
from app.energy_model import EnergyModel, langevin_sample
from helper_lib import get_model


class Assignment4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_embedding(self):
        emb = SinusoidalEmbedding(8)(torch.tensor([1]))[0]
        angles = torch.tensor([1., .1, .01, .001])
        expected = torch.stack((angles.sin(), angles.cos()), -1).flatten()
        torch.testing.assert_close(emb, expected)

    def test_unet_output_shape_and_parameter_gradients(self):
        net = UNet()
        x = torch.randn(2, 3, 32, 32)
        y = net(x, torch.tensor([1, 999]))
        self.assertEqual(y.shape, x.shape)
        y.square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in net.parameters()))

    def test_langevin_lowers_energy_without_weight_gradients(self):
        # E(x)=sum(x^2) has a known minimum. This checks sign and pixel updates.
        class Quadratic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
            def forward(self, x):
                return self.weight * x.flatten(1).square().sum(1)
        model = Quadratic()
        x = torch.full((2, 3, 4, 4), 0.5)
        weight = model.weight.detach().clone()
        with torch.no_grad():
            y = langevin_sample(model, x, steps=5, step_size=1, noise_std=0)
        self.assertLess(model(y).mean(), model(x).mean())
        self.assertIsNone(model.weight.grad)
        self.assertFalse(y.requires_grad)
        torch.testing.assert_close(weight, model.weight)

    def test_energy_training_gradients_after_detached_sampling(self):
        model = EnergyModel()
        fake = langevin_sample(model, torch.rand(2, 3, 32, 32), steps=2)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        real_energy, fake_energy = model(torch.randn_like(fake)), model(fake)
        (real_energy.mean() - fake_energy.mean()).backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_diffusion_forward_and_schedule(self):
        model = DiffusionModel()
        self.assertTrue(torch.all(model.alpha_bars[:-1] > model.alpha_bars[1:]))
        x, noise = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)
        ts = torch.tensor([0, 999])
        a = model.alpha_bars[ts][:, None, None, None]
        torch.testing.assert_close(model.noisy_images(x, ts, noise), a.sqrt()*x+(1-a).sqrt()*noise)

    def test_sampling_range_and_seed(self):
        model = DiffusionModel().eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        a = model.sample(1, steps=3, generator=torch.Generator().manual_seed(5))
        b = model.sample(1, steps=3, generator=torch.Generator().manual_seed(5))
        torch.testing.assert_close(a, b)
        self.assertTrue(torch.isfinite(a).all())
        self.assertGreaterEqual(a.min(), 0)
        self.assertLessEqual(a.max(), 1)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, before[k])

    def test_factory_preserves_previous_models(self):
        for name in ("FCNN", "CNN", "EnhancedCNN", "VAE", "GAN", "Energy", "Diffusion"):
            self.assertIsInstance(get_model(name), torch.nn.Module)


if __name__ == "__main__":
    unittest.main()
