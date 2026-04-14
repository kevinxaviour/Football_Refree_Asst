"""
api/routes/ingest.py
------------------------
Ingest endpoint — indexes the FIFA Laws PDF into ChromaDB.
"""

import os
from fastapi import APIRouter, HTTPException, BackgroundTasks
from api.schemas import IngestRequest, IngestResponse

router = APIRouter()

_indexing_in_progress = False


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Index the FIFA Laws PDF",
    description="Download and index the IFAB Laws of the Game PDF into the vector database.",
    tags=["Admin"]
)
async def ingest_pdf(request: IngestRequest, background_tasks: BackgroundTasks) -> IngestResponse:
    """Index the Laws of the Game PDF into ChromaDB."""
    global _indexing_in_progress

    if _indexing_in_progress:
        raise HTTPException(status_code=409, detail="Indexing already in progress")

    try:
        from src.ingestion.pdf_loader import load_pdf
        from src.ingestion.chunker import chunk_pages
        from src.vectorstore.chroma_store import index_chunks, delete_collection

        pdf_path = request.pdf_path or os.getenv("PDF_LOCAL_PATH", "./data/laws_of_the_game.pdf")

        if request.force_reindex:
            delete_collection()

        pages = load_pdf(pdf_path)
        chunks = chunk_pages(pages)
        index_chunks(chunks)

        return IngestResponse(
            status="success",
            message=f"Successfully indexed {len(chunks)} chunks from {len(pages)} pages",
            chunks_indexed=len(chunks),
            pages_processed=len(pages)
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
