"""Production-oriented RAG orchestration with structured outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.llm_router import invoke_structured
from src.rag.schemas import StructuredRuling
from src.retrieval.contextual_retriever import retrieve_context

SYSTEM_PROMPT = """You are an expert IFAB/FIFA referee assistant.
You must answer strictly from provided context excerpts.
Return accurate and concise rulings with explicit law citations.
If context is insufficient, say so in explanation and lower confidence.
"""


@dataclass
class RAGResult:
    question: str
    answer: str
    structured_output: StructuredRuling
    sources: list[dict[str, Any]]
    laws_cited: list[str]
    confidence: str
    provider_used: str
    trace_id: str


def _format_context(chunks: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        law = chunk["metadata"].get("law", "Unknown law")
        page = chunk["metadata"].get("page_num", "?")
        score = chunk.get("score", 0.0)
        text = chunk["text"].strip()
        sections.append(f"[Source {idx} | {law} | page={page} | score={score}]\n{text}")
    return "\n\n".join(sections)


def _build_user_prompt(question: str, context: str) -> str:
    return (
        "Question:\n"
        f"{question}\n\n"
        "Retrieved context:\n"
        f"{context}\n\n"
        "Instructions:\n"
        "- Ground your answer in the provided sources only.\n"
        "- Fill all structured fields.\n"
        "- Put citations that map to source law/page and include short quotes.\n"
        "- If evidence is weak, set confidence to medium/low."
    )


def _build_readable_answer(data: StructuredRuling) -> str:
    exceptions = "; ".join(data.key_exceptions) if data.key_exceptions else "None"
    return (
        f"Situation: {data.situation}\n"
        f"Applicable Laws: {', '.join(data.applicable_laws)}\n"
        f"Ruling: {data.ruling}\n"
        f"Explanation: {data.explanation}\n"
        f"Key Exceptions: {exceptions}"
    )


def answer_question(
    question: str, top_k: int = 8, law_filter: str | None = None
) -> RAGResult:
    trace_id = str(uuid4())
    chunks = retrieve_context(question=question, top_k=top_k, law_filter=law_filter)
    context = _format_context(chunks)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_build_user_prompt(question, context)),
    ]
    config = {
        "run_name": "fifa_referee_rag",
        "tags": ["rag", "structured-output"],
        "metadata": {"trace_id": trace_id, "top_k": top_k},
    }
    structured, provider = invoke_structured(
        schema=StructuredRuling,
        messages=messages,
        config=config,
    )

    laws_cited = sorted(set(structured.applicable_laws))
    return RAGResult(
        question=question,
        answer=_build_readable_answer(structured),
        structured_output=structured,
        sources=chunks,
        laws_cited=laws_cited,
        confidence=structured.confidence,
        provider_used=provider,
        trace_id=trace_id,
    )
