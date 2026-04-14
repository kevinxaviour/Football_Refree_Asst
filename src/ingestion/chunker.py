"""
src/ingestion/chunker.py
========================
Semantic chunker — groups sentences by meaning, not by character count.

Why semantic chunking over RecursiveCharacterTextSplitter?
- Keeps logically related sentences together (e.g., all conditions of the
  handball rule stay in one chunk, not split mid-rule)
- A chunk boundary only forms where the *meaning* shifts
- Dramatically improves retrieval quality because each chunk is about
  exactly one topic

Strategy:
  1. Split page text into individual sentences
  2. Embed every sentence with the same model used for retrieval
  3. Compute cosine similarity between consecutive sentences
  4. Insert a chunk break wherever similarity drops below SEMANTIC_THRESHOLD
  5. Also break when a chunk exceeds MAX_CHUNK_SENTENCES to avoid huge chunks
"""

import os
import re
from typing import List, Dict, Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── Tuning knobs (override via .env) ──────────────────────────
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", 0.70))
MAX_CHUNK_SENTENCES = int(os.getenv("MAX_CHUNK_SENTENCES", 12))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", 80))


# ── Sentence splitter ─────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex.
    Handles common abbreviations and decimal numbers to avoid false splits.
    """
    # Protect abbreviations like "e.g.", "i.e.", "Law 12.", "Art. 7."
    text = re.sub(r'\b(e\.g|i\.e|vs|etc|Fig|Art|No|approx)\.\s', r'\1<DOT> ', text)
    text = re.sub(r'(\d+)\.(\d+)', r'\1<DECIMAL>\2', text)  

    # Split on sentence-ending punctuation followed by whitespace + capital
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\(\"\'])', text)

    # Restore protected patterns
    sentences = [
        s.replace('<DOT>', '.').replace('<DECIMAL>', '.')
        for s in sentences
    ]

    return [s.strip() for s in sentences if s.strip()]


# ── Cosine similarity helper ──────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ── Core semantic chunker ─────────────────────────────────────

def _semantic_chunk(sentences: List[str], embeddings: np.ndarray) -> List[str]:
    """
    Group sentences into chunks based on semantic similarity.

    A new chunk starts when:
      - Cosine similarity between consecutive sentences < SEMANTIC_THRESHOLD, OR
      - The current chunk has reached MAX_CHUNK_SENTENCES
    """
    if not sentences:
        return []

    chunks = []
    current_group = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = _cosine_sim(embeddings[i - 1], embeddings[i])
        too_long = len(current_group) >= MAX_CHUNK_SENTENCES

        if sim < SEMANTIC_THRESHOLD or too_long:
            # Flush current group
            chunk_text = " ".join(current_group).strip()
            if chunk_text:
                chunks.append(chunk_text)
            current_group = [sentences[i]]
        else:
            current_group.append(sentences[i])

    # Flush last group
    if current_group:
        chunk_text = " ".join(current_group).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


# ── Public API ────────────────────────────────────────────────

def chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Semantically chunk pages into topic-coherent segments.

    Args:
        pages: List of page dicts from pdf_loader.load_pdf()

    Returns:
        List of chunk dicts with text + metadata
    """
    # Import here to avoid circular import; model is already warm from build_index
    from src.embeddings.embedder import get_model

    model = get_model()

    all_chunks = []
    chunk_id = 0

    print(f"   Semantic threshold: {SEMANTIC_THRESHOLD} | Max sentences/chunk: {MAX_CHUNK_SENTENCES}")

    for page in pages:
        text = page.get("text", "").strip()
        if not text:
            continue

        # 1. Split into sentences
        sentences = _split_sentences(text)
        if not sentences:
            continue

        # 2. Embed all sentences at once (batched — fast)
        embeddings = model.encode(
            sentences,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # 3. Group by semantic similarity
        text_chunks = _semantic_chunk(sentences, embeddings)

        # 4. Build chunk dicts
        for chunk_text in text_chunks:
            if len(chunk_text.strip()) < MIN_CHUNK_CHARS:
                continue  # Skip tiny fragments (page numbers, headers, etc.)

            all_chunks.append({
                "id": f"chunk_{chunk_id:05d}",
                "text": chunk_text.strip(),
                "metadata": {
                    "page_num": page["page_num"],
                    "law": page["law"],
                    "source": page["source"],
                    "chunk_id": chunk_id,
                    "sentence_count": len(_split_sentences(chunk_text)),
                }
            })
            chunk_id += 1

    print(f"✅ Created {len(all_chunks)} semantic chunks from {len(pages)} pages")
    if all_chunks:
        avg = sum(len(c["text"]) for c in all_chunks) // len(all_chunks)
        print(f"   Avg chunk size: {avg} chars")

    return all_chunks


def get_law_distribution(chunks: List[Dict[str, Any]]) -> Dict[str, int]:
    """Return count of chunks per Law (for inspection)."""
    distribution: Dict[str, int] = {}
    for chunk in chunks:
        law = chunk["metadata"]["law"]
        distribution[law] = distribution.get(law, 0) + 1
    return dict(sorted(distribution.items()))
