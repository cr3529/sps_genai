"""Assignment 2, Part 1: CNN built to the specified architecture.

Input:   RGB image, 64 x 64 x 3  (PyTorch layout: [batch, 3, 64, 64])
Conv2D   16 filters, 3x3, stride 1, padding 1  -> [batch, 16, 64, 64]
ReLU
MaxPool  2x2, stride 2                         -> [batch, 16, 32, 32]
Conv2D   32 filters, 3x3, stride 1, padding 1  -> [batch, 32, 32, 32]
ReLU
MaxPool  2x2, stride 2                         -> [batch, 32, 16, 16]
Flatten                                        -> [batch, 8192]
Linear   100 units                             -> [batch, 100]
ReLU
Linear   10 units (one per class)              -> [batch, 10]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 64
NUM_CLASSES = 10


class AssignmentCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Two 2x2 pools halve 64 twice: 64 -> 32 -> 16, with 32 channels.
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # raw scores (logits); softmax is applied at prediction time
