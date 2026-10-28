"""Evaluate a classifier (FCNN / CNN / EnhancedCNN) on a test DataLoader."""

import torch


def evaluate_model(model, data_loader, criterion, device="cpu"):
    """Return (average loss, accuracy %) over the whole loader."""
    model.to(device)
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, 100.0 * correct / total
