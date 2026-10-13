# sps_genai: Applied Generative AI (APANPS5560)

FastAPI app with a bigram text generator (Module 3), a spaCy word embedding endpoint (Assignment 1) and a CNN image classifier trained on CIFAR-10 (Assignment 2).

## Run locally
    uv sync
    uv run fastapi dev app/main.py

## Run with Docker
    docker build -t sps-genai .
    docker run -p 8000:80 sps-genai

Then open http://127.0.0.1:8000/docs

## Endpoints
- POST /generate   body: {"start_word": "bigram", "length": 10}
- POST /embedding  body: {"word": "apple"} returns the 300-dimension spaCy vector
- POST /classify   multipart file upload (any jpg/png); returns the predicted CIFAR-10 class
- GET  /classify/classes  lists the 10 possible classes

## Assignment 2: CNN image classifier
- `app/cnn_model.py`: `AssignmentCNN`, built to the assignment's spec
  (64x64x3 input -> Conv 16 3x3 -> ReLU -> MaxPool 2x2 -> Conv 32 3x3 -> ReLU -> MaxPool 2x2 -> Flatten -> FC 100 -> ReLU -> FC 10)
- `train_cnn.py`: trains it on CIFAR-10 and saves `models/cnn_cifar10.pth`.
  CIFAR-10 images are 32x32, so they are resized to 64x64 to match the specified input size.
  Re-train with `uv run python train_cnn.py` (default 10 epochs).
- `app/image_classifier.py`: loads the saved weights; resizes any uploaded image to 64x64 and returns the class.
- `docs/assignment2_theory.md`: Part 3 answers.

Trained for 10 epochs (Adam, lr 0.001, batch 64): **69.3% accuracy on the 10,000 CIFAR-10 test images**.

Example request:

    curl -X POST http://127.0.0.1:8000/classify -F "file=@cat.png"

Example response (a cat from the CIFAR-10 test set):

    {"filename": "cat.png", "class": "cat", "class_index": 3, "confidence": 0.7511,
     "probabilities": {"airplane": 0.0014, "automobile": 0.0, "bird": 0.0341, "cat": 0.7511, "deer": 0.0498,
                       "dog": 0.0844, "frog": 0.0014, "horse": 0.0667, "ship": 0.0111, "truck": 0.0}}

You can also try it from http://127.0.0.1:8000/docs (open POST /classify, click "Try it out", choose an image).

Code developed with assistance from Claude (Anthropic).
