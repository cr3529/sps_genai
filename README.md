# sps_genai: Applied Generative AI (APANPS5560)

FastAPI app with a bigram text generator (Module 3) and a spaCy word embedding endpoint (Assignment 1).

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

Code developed with assistance from Claude (Anthropic).
