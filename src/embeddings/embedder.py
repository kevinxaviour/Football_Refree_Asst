"""Embedding helpers with OpenAI primary and local fallback."""

import os
from typing import List, Optional

import numpy as np
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
LOCAL_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/e5-small-v2")

_model: Optional[SentenceTransformer] = None
_openai_embedder: Optional[OpenAIEmbeddings] = None


def _normalize(vectors: List[List[float]]) -> List[List[float]]:
    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).tolist()


def _normalize_one(vector: List[float]) -> List[float]:
    arr = np.array(vector, dtype=np.float32)
    denom = np.linalg.norm(arr)
    if denom == 0:
        return arr.tolist()
    return (arr / denom).tolist()


def _use_openai() -> bool:
    return EMBEDDING_PROVIDER == "openai"


def get_openai_embedder() -> OpenAIEmbeddings:
    global _openai_embedder
    if _openai_embedder is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai"
            )
        _openai_embedder = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            api_key=api_key,
        )
        print(f"OpenAI embeddings ready: {OPENAI_EMBEDDING_MODEL}")
    return _openai_embedder


def get_model() -> SentenceTransformer:
    """Load the local embedding model (singleton pattern)."""
    global _model
    if _model is None:
        print(f"Loading local embedding model: {LOCAL_MODEL_NAME}")
        _model = SentenceTransformer(LOCAL_MODEL_NAME)
        print(
            f"Embedding model loaded (dim={_model.get_sentence_embedding_dimension()})"
        )
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate normalized contextual embeddings for document passages."""
    if _use_openai():
        embedder = get_openai_embedder()
        vectors = embedder.embed_documents(texts)
        return _normalize(vectors)

    model = get_model()
    passages = [f"passage: {text.strip()}" for text in texts]
    embeddings = model.encode(
        passages,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """Generate normalized contextual embedding for a search query."""
    if _use_openai():
        embedder = get_openai_embedder()
        return _normalize_one(embedder.embed_query(query))

    model = get_model()
    prefixed = f"query: {query}"
    embedding = model.encode(
        prefixed,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding.tolist()
