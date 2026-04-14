# Refree FastAPI App

This folder runs the real FIFA Referee AI FastAPI app from `api/main.py`.

## Install

```bash
pip install -r fastapi_app/requirements.txt
```

## Run

```bash
python fastapi_app/run.py
```

Or:

```bash
uvicorn fastapi_app.main:app --reload
```

## Required setup

1. Put `OPENAI_API_KEY` (primary) in `.env`
2. Build the vector index:

```bash
python scripts/build_index.py
```

## Hugging Face Spaces deployment

Use a Docker Space, not Vercel.

Why Hugging Face over Vercel:

- this app needs a full Python server process
- it builds and stores a local ChromaDB index
- it loads local embedding models
- Vercel serverless functions are a poor fit for that workload

The repo includes a `Dockerfile` for Hugging Face Spaces.

Set these Space variables:

- `OPENAI_API_KEY`
- `GROQ_API_KEY` (fallback)
- `OPENROUTER_API_KEY` (fallback)
- `AUTO_BUILD_INDEX=true`

The app listens on port `7860`, which matches Spaces.
