"""Structured LLM output schemas for referee rulings."""

from typing import List, Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    law: str = Field(description="Law identifier, e.g. Law 12 - Fouls and Misconduct")
    page_num: int = Field(description="PDF page number used as evidence")
    quote: str = Field(description="Short supporting excerpt")


class StructuredRuling(BaseModel):
    situation: str
    applicable_laws: List[str]
    ruling: str
    explanation: str
    key_exceptions: List[str]
    citations: List[Citation]
    confidence: Literal["high", "medium", "low"]
