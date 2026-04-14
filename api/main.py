"""
api/main.py
===========
FastAPI application entry point.
FIFA Referee AI Assistant — RAG + Agentic AI System
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

load_dotenv()

if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
    os.environ.setdefault("LANGSMITH_PROJECT", "refreeassisant")

# ── Create FastAPI app ─────────────────────────────────────────
# An intelligent assistant for FIFA/IFAB Laws of the Game using:
# - **RAG** (Retrieval-Augmented Generation) with ChromaDB
# - **Contextual retrieval** with reranking over ChromaDB
# - **LLM routing** with OpenAI primary + Groq/OpenRouter fallback
# - **Structured outputs** for production integrations
app = FastAPI(
    title="FIFA Referee AI Assistant",
    description="""
## FIFA Laws of the Game — AI-Powered RAG System
### How to use:
1. Ask any football rules question: `POST /query`
2. Get authoritative answers with Law citations

### Laws Coverage:
All 17 IFAB Laws of the Game (2025/26 edition)
    """,
    version="1.0.0",
    # contact={
    #     "name": "FIFA Referee Assistant",
    #     "url": "https://www.theifab.com",
    # },
    # license_info={
    #     "name": "IFAB Laws of the Game",
    #     "url": "https://www.theifab.com/laws-of-the-game/",
    # },
)

# ── CORS middleware ────────────────────────────────────────────

allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routes ────────────────────────────────────────────

from api.routes.query import router as query_router
from api.routes.ingest import router as ingest_router
from api.routes.health import health_router, laws_router

app.include_router(query_router)
app.include_router(ingest_router)
app.include_router(health_router)
app.include_router(laws_router)


def _should_auto_build_index() -> bool:
    return os.getenv("AUTO_BUILD_INDEX", "false").lower() == "true"


def _ensure_vector_index() -> None:
    from src.vectorstore.chroma_store import get_collection_stats

    stats = get_collection_stats()
    if stats.get("total_chunks", 0) > 0:
        print(f"✅ Vector index ready with {stats['total_chunks']} chunks")
        return

    if not _should_auto_build_index():
        print(
            "⚠️  Vector index is empty. Set AUTO_BUILD_INDEX=true or run python scripts/build_index.py"
        )
        return

    print("🔄 Vector index is empty. Building index for hosted deployment...")
    from scripts.build_index import build_index

    build_index()


# ── Root endpoint with quick-start HTML ───────────────────────


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root() -> str:
    return "HELLO"


# ── Startup event ──────────────────────────────────────────────


@app.on_event("startup")
async def startup_event():
    """Warm up dependencies and ensure the vector index exists."""
    print("FIFA Referee AI Assistant starting up...")
    try:
        from src.embeddings.embedder import get_model

        _ensure_vector_index()
        get_model()  # Load embedding model into memory
        print("System ready!")
    except Exception as e:
        print(f"Startup warning: {e}")


# ── Run directly ───────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("DEBUG", "true").lower() == "true",
        log_level="info",
    )
