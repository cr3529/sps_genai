"""Assignment 2, Part 2: loads the trained CNN and classifies a single image."""

import io
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

from app.cnn_model import IMAGE_SIZE, AssignmentCNN

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "cnn_cifar10.pth"

# Same preprocessing used in training: resize to 64x64, convert to a [3, 64, 64] tensor in [0, 1].
preprocess = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)


class ImageClassifier:
    def __init__(self, weights_path: Path = DEFAULT_WEIGHTS):
        self.model = AssignmentCNN()
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()  # inference mode for layers like dropout/batch norm

    def predict(self, image_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        batch = preprocess(image).unsqueeze(0)  # [1, 3, 64, 64]
        with torch.no_grad():
            probs = torch.softmax(self.model(batch), dim=1)[0]
        top = int(torch.argmax(probs))
        return {
            "class": CIFAR10_CLASSES[top],
            "class_index": top,
            "confidence": round(float(probs[top]), 4),
            "probabilities": {
                name: round(float(p), 4) for name, p in zip(CIFAR10_CLASSES, probs)
            },
        }
