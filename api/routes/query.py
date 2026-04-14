"""Primary query endpoint backed by LangChain RAG service."""

from fastapi import APIRouter, HTTPException

from api.schemas import QueryRequest, QueryResponse, SourceChunk, StructuredRulingOut
from src.rag.service import answer_question

router = APIRouter()


@router.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_laws(request: QueryRequest) -> QueryResponse:
    try:
        result = answer_question(
            question=request.question,
            top_k=request.top_k,
            law_filter=request.law_filter,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "Vector store is empty" in message:
            raise HTTPException(
                status_code=503,
                detail="Vector store is not initialized. Run python scripts/build_index.py",
            ) from exc
        raise HTTPException(status_code=500, detail=message) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    sources = [
        SourceChunk(
            text=item["text"],
            law=item["metadata"].get("law", "Unknown law"),
            page_num=item["metadata"].get("page_num", -1),
            score=item["score"],
        )
        for item in result.sources
    ]

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=sources,
        laws_cited=result.laws_cited,
        confidence=result.confidence,
        mode="structured_rag",
        provider_used=result.provider_used,
        trace_id=result.trace_id,
        structured_output=StructuredRulingOut.model_validate(
            result.structured_output.model_dump()
        ),
    )
