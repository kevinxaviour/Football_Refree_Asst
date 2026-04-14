"""Tests for current structured RAG API components."""

from api.schemas import QueryRequest
from src.rag.schemas import StructuredRuling


def test_query_request_defaults():
    req = QueryRequest(question="Can a goalkeeper handle a back-pass?")
    assert req.use_agent is False
    assert req.top_k == 8


def test_structured_ruling_schema_validates():
    payload = {
        "situation": "Goalkeeper receives deliberate kick from teammate.",
        "applicable_laws": ["Law 12 - Fouls and Misconduct"],
        "ruling": "Indirect free kick if goalkeeper handles deliberately kicked ball.",
        "explanation": "Law 12 prohibits handling from a deliberate kick by a teammate.",
        "key_exceptions": ["Deflection is not treated as deliberate kick."],
        "citations": [
            {
                "law": "Law 12 - Fouls and Misconduct",
                "page_num": 88,
                "quote": "A goalkeeper cannot handle the ball from a deliberate kick by a teammate.",
            }
        ],
        "confidence": "high",
    }
    model = StructuredRuling.model_validate(payload)
    assert model.confidence == "high"
    assert model.citations[0].page_num == 88
