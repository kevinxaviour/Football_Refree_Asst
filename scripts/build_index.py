"""
scripts/build_index.py
=======================
One-time script to:
1. Download the IFAB Laws of the Game PDF
2. Extract text from all pages
3. Chunk into overlapping segments
4. Embed with sentence-transformers
5. Store in ChromaDB

Run once before starting the API:
    python scripts/build_index.py
"""

import os
import time
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from scripts.download_pdf import download_pdf
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_pages, get_law_distribution
from src.vectorstore.chroma_store import index_chunks, get_collection_stats
from src.embeddings.embedder import get_model


def build_index():
    print("=" * 60)
    print("FIFA Laws of the Game — Building RAG Index")
    print("=" * 60)

    start = time.time()

    # Step 1: Ensure PDF exists
    print("\nStep 1/4: Ensuring PDF is available...")
    pdf_path = download_pdf()

    # Step 2: Extract text
    print("\nStep 2/4: Extracting text from PDF...")
    pages = load_pdf(pdf_path)

    # Step 3: Chunk
    print("\nStep 3/4: Chunking into segments...")
    chunks = chunk_pages(pages)

    # Show distribution
    distribution = get_law_distribution(chunks)
    print("\nChunks per Law:")
    for law, count in distribution.items():
        bar = "█" * (count // 3)
        print(f"   {law[:40]:<40} {count:>4} {bar}")

    # Step 4: Embed & Index
    print("\nStep 4/4: Embedding and indexing into ChromaDB...")
    get_model()  # Pre-load embedding model
    index_chunks(chunks)

    # Done
    elapsed = time.time() - start
    stats = get_collection_stats()

    print("\n" + "=" * 60)
    print("Index Built Successfully!")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Time taken:   {elapsed:.1f} seconds")
    print(f"   Saved to:     {stats['persist_dir']}")
    print("\nNow run: uvicorn api.main:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    build_index()
