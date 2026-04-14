"""
api/routes/health.py
api/routes/laws.py

Health check and Laws reference endpoints.
"""

from fastapi import APIRouter
from api.schemas import HealthResponse, LawsListResponse, LawSummary
import os

health_router = APIRouter()
laws_router = APIRouter()


@health_router.get(
    "/health", response_model=HealthResponse, summary="Health check", tags=["System"]
)
async def health_check() -> HealthResponse:
    """Check system health and vector store status."""
    from src.vectorstore.chroma_store import get_collection_stats

    try:
        stats = get_collection_stats()
    except Exception:
        stats = {"status": "error", "total_chunks": 0}

    return HealthResponse(
        status="healthy" if stats.get("total_chunks", 0) > 0 else "not_indexed",
        vector_store=stats,
        model=(
            f"openai={bool(os.getenv('OPENAI_API_KEY'))}, "
            f"groq={bool(os.getenv('GROQ_API_KEY'))}, "
            f"openrouter={bool(os.getenv('OPENROUTER_API_KEY'))}"
        ),
    )


@laws_router.get(
    "/laws",
    response_model=LawsListResponse,
    summary="List all 17 Laws of the Game",
    tags=["Reference"],
)
async def list_laws() -> LawsListResponse:
    """Get a reference list of all 17 Laws of the Game."""
    from src.agents.tools import LAWS_REFERENCE

    laws = [
        LawSummary(
            law_number=num,
            law_name=law["name"],
            full_title=f"Law {num} - {law['name']}",
            summary=law["summary"],
            key_topics=law["key_topics"],
        )
        for num, law in LAWS_REFERENCE.items()
    ]

    return LawsListResponse(laws=laws, total=len(laws))
