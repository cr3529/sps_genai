import spacy
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import UnidentifiedImageError
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.digit_generator import MAX_SAMPLES, DigitGenerator
from app.image_classifier import CIFAR10_CLASSES, ImageClassifier

app = FastAPI()

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)
nlp = spacy.load("en_core_web_lg")
image_classifier = ImageClassifier()
digit_generator = DigitGenerator()


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    doc = nlp(request.word)
    if not doc.has_vector:
        raise HTTPException(status_code=404, detail=f"No embedding found for '{request.word}'")
    vector = doc.vector.tolist()
    return {"word": request.word, "dimension": len(vector), "embedding": vector}


@app.get("/classify/classes")
def list_classes():
    return {"classes": CIFAR10_CLASSES}


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """Upload an image (jpg, png, ...); returns its predicted CIFAR-10 class."""
    image_bytes = await file.read()
    try:
        result = image_classifier.predict(image_bytes)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image")
    return {"filename": file.filename, **result}


@app.get(
    "/gan/generate",
    responses={200: {"content": {"image/png": {}}, "description": "PNG grid of generated digits"}},
    response_class=Response,
)
def generate_digits(
    num_samples: int = Query(16, ge=1, le=MAX_SAMPLES, description="How many digits to generate (1-64)"),
    seed: int | None = Query(None, description="Optional random seed, for repeatable output"),
):
    """Generate brand-new handwritten digits with the GAN trained on MNIST; returns a PNG grid."""
    png = digit_generator.generate_png(num_samples=num_samples, seed=seed)
    return Response(content=png, media_type="image/png")


@app.get("/gan/generate/pixels")
def generate_digit_pixels(
    num_samples: int = Query(1, ge=1, le=MAX_SAMPLES),
    seed: int | None = Query(None),
):
    """Same as /gan/generate but returns the raw 28x28 pixel values (0-1) as JSON."""
    images = digit_generator.generate(num_samples=num_samples, seed=seed)
    return {
        "num_samples": num_samples,
        "shape": [1, 28, 28],
        "images": [[[round(float(v), 4) for v in row] for row in img[0]] for img in images],
    }
