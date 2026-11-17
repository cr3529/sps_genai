"""helper_lib: reusable pieces for the course's models (Module 4 onward).

    from helper_lib import get_data_loader, get_model, train_model, train_gan, generate_samples

Modules
  data_loader.py  get_data_loader()  MNIST / CIFAR-10 DataLoaders
  model.py        get_model()        FCNN, CNN, EnhancedCNN, VAE, GAN, Energy, Diffusion
  trainer.py      classifier, VAE, GAN, energy, and diffusion training
  evaluator.py    evaluate_model()   accuracy + loss for classifiers
  generator.py    generate_samples() sample all four generative model types
  utils.py        save_model(), load_model(), get_device()
"""

from helper_lib.data_loader import get_data_loader
from helper_lib.evaluator import evaluate_model
from helper_lib.generator import generate_samples
from helper_lib.model import get_model
from helper_lib.trainer import train_gan, train_model, train_vae, train_diffusion, train_energy
from helper_lib.utils import get_device, load_model, save_model

__all__ = [
    "get_data_loader",
    "get_model",
    "train_model",
    "train_vae",
    "train_gan",
    "train_diffusion",
    "train_energy",
    "evaluate_model",
    "generate_samples",
    "save_model",
    "load_model",
    "get_device",
]
