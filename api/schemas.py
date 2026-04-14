"""
api/schemas.py
==============
Pydantic models for request/response validation.
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


# ── Request Models ─────────────────────────────────────


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Your football rules question",
        example="Can a goalkeeper pick up a back-pass from a teammate?",
    )
    use_agent: bool = Field(
        default=False, description="Deprecated flag kept for compatibility"
    )
    top_k: int = Field(
        default=8, ge=1, le=25, description="Number of law excerpts to retrieve"
    )
    law_filter: Optional[str] = Field(
        default=None,
        description="Filter search to a specific law, e.g. 'Law 11 - Offside'",
    )


class IngestRequest(BaseModel):
    pdf_path: Optional[str] = Field(
        default=None, description="Path to PDF file. Defaults to configured path."
    )
    force_reindex: bool = Field(
        default=False, description="Delete existing index and rebuild from scratch"
    )


# ── Response Models ────────────────────────────────────


class SourceChunk(BaseModel):
    text: str
    law: str
    page_num: int
    score: float


class AgentStepOut(BaseModel):
    step_num: int
    thought: str
    action: str
    observation: str


class CitationOut(BaseModel):
    law: str
    page_num: int
    quote: str


class StructuredRulingOut(BaseModel):
    situation: str
    applicable_laws: List[str]
    ruling: str
    explanation: str
    key_exceptions: List[str]
    citations: List[CitationOut]
    confidence: Literal["high", "medium", "low"]


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceChunk]
    laws_cited: List[str]
    confidence: str
    mode: str
    provider_used: str
    trace_id: str
    structured_output: StructuredRulingOut
    steps: Optional[List[AgentStepOut]] = None
    clarification_needed: Optional[str] = None


class IngestResponse(BaseModel):
    status: str
    message: str
    chunks_indexed: int
    pages_processed: int


class HealthResponse(BaseModel):
    status: str
    vector_store: Dict[str, Any]
    model: str
    version: str = "1.0.0"


class LawSummary(BaseModel):
    law_number: int
    law_name: str
    full_title: str
    summary: str
    key_topics: List[str]


class LawsListResponse(BaseModel):
    laws: List[LawSummary]
    total: int
