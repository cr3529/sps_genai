"""Assignment 2, Part 2: train AssignmentCNN on CIFAR-10 and save the weights.

CIFAR-10 images are 32x32, but the specified architecture takes 64x64 input,
so every image is resized to 64x64 before it reaches the network.

Run:  uv run python train_cnn.py            (default 10 epochs)
      uv run python train_cnn.py --epochs 5
Saves: models/cnn_cifar10.pth  (used by the /classify endpoint)
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from app.cnn_model import IMAGE_SIZE, AssignmentCNN

MODEL_PATH = Path("models/cnn_cifar10.pth")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predicted = model(images).argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),  # light augmentation
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose(
        [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    model = AssignmentCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        test_acc = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch + 1}/{args.epochs}  "
            f"loss={running_loss / len(train_loader):.4f}  test_acc={test_acc:.2f}%"
        )

    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, MODEL_PATH)
    print(f"Final test accuracy: {test_acc:.2f}%  -> saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
