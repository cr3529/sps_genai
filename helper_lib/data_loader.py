"""DataLoaders for the datasets used in the course."""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loader(
    dataset: str = "MNIST",
    data_dir: str = "./data",
    batch_size: int = 128,
    train: bool = True,
    normalize_to_tanh: bool = False,
    image_size: int | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """Return a shuffled (train) or ordered (test) DataLoader.

    normalize_to_tanh=True scales pixels to [-1, 1] so real images match a generator
    that ends in Tanh (needed for the GAN). Otherwise pixels stay in [0, 1].
    image_size resizes images (e.g. 64 for the Assignment 2 CNN on CIFAR-10).
    """
    dataset = dataset.upper()
    steps = []
    if image_size is not None:
        steps.append(transforms.Resize((image_size, image_size)))
    steps.append(transforms.ToTensor())

    if dataset == "MNIST":
        if normalize_to_tanh:
            steps.append(transforms.Normalize((0.5,), (0.5,)))
        ds = datasets.MNIST(data_dir, train=train, download=True, transform=transforms.Compose(steps))
    elif dataset in ("CIFAR10", "CIFAR-10"):
        if normalize_to_tanh:
            steps.append(transforms.Normalize((0.5,) * 3, (0.5,) * 3))
        ds = datasets.CIFAR10(data_dir, train=train, download=True, transform=transforms.Compose(steps))
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'MNIST' or 'CIFAR10'.")

    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers)
