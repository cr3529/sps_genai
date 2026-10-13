import spacy
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import UnidentifiedImageError
from pydantic import BaseModel

from app.bigram_model import BigramModel
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
