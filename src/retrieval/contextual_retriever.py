"""Contextual retrieval using dense search + MMR reranking."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.embeddings.embedder import embed_query
from src.vectorstore.chroma_store import get_collection


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _mmr_select(
    query_embedding: np.ndarray,
    candidates: list[dict[str, Any]],
    top_k: int,
    lambda_param: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = candidates.copy()

    while remaining and len(selected) < top_k:
        best_idx = 0
        best_score = -1e9

        for i, candidate in enumerate(remaining):
            candidate_embedding = candidate["embedding"]
            relevance = _cosine(query_embedding, candidate_embedding)

            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(
                    _cosine(candidate_embedding, item["embedding"]) for item in selected
                )

            score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if score > best_score:
                best_score = score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


def retrieve_context(
    question: str, top_k: int = 8, law_filter: str | None = None
) -> list[dict[str, Any]]:
    collection = get_collection()
    if collection.count() == 0:
        raise RuntimeError(
            "Vector store is empty. Run python scripts/build_index.py first."
        )

    query_embedding = np.array(embed_query(question), dtype=np.float32)
    n_candidates = min(max(top_k * 6, 20), 80)
    where = {"law": {"$eq": law_filter}} if law_filter else None

    raw = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_candidates,
        where=where,
        include=["documents", "metadatas", "embeddings", "distances"],
    )

    candidates: list[dict[str, Any]] = []
    for doc, metadata, emb, distance in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["embeddings"][0],
        raw["distances"][0],
    ):
        candidates.append(
            {
                "text": doc,
                "metadata": metadata,
                "embedding": np.array(emb, dtype=np.float32),
                "score": round(1 - float(distance), 4),
            }
        )

    selected = _mmr_select(
        query_embedding=query_embedding,
        candidates=candidates,
        top_k=top_k,
        lambda_param=0.72,
    )

    for item in selected:
        item.pop("embedding", None)

    return selected
