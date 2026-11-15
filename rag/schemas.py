"""Pydantic models shared by the HTTP API (OpenAPI schema) and the MCP tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from rag.generation import Answer
from rag.retrieval import Hit

Mode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


class Passage(BaseModel):
    chunk_id: int
    doc_id: str
    page_start: int
    page_end: int
    text: str
    score: float = Field(
        description="Score from the mode that ranked it (RRF, BM25, cosine "
        "or cross-encoder logit); only comparable within one response."
    )

    @classmethod
    def of(cls, h: Hit) -> Passage:
        return cls(
            chunk_id=h.chunk_id,
            doc_id=h.doc_id,
            page_start=h.page_start,
            page_end=h.page_end,
            text=h.text,
            score=round(h.score, 6),
        )


class SearchResponse(BaseModel):
    query: str
    mode: Mode
    k: int
    passages: list[Passage]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(5, ge=1, le=20, description="Passages retrieved (hybrid_rerank).")


class CitationModel(BaseModel):
    doc_id: str
    page: int


class AnswerModel(BaseModel):
    answer: str = Field(
        description="Answer text with [doc_id:page] citations, or 'Not found in corpus.'"
    )
    abstained: bool
    provider: str
    top_score: float | None = Field(description="Best cross-encoder score (abstention gate).")
    citations: list[CitationModel]
    invalid_citations: list[CitationModel] = Field(
        description="Citations not backed by a retrieved passage; removed from `answer`."
    )
    unsupported_sentences: list[str] = Field(
        description="Sentences left without any valid citation (flagged, not removed)."
    )

    @classmethod
    def of(cls, a: Answer) -> AnswerModel:
        def cites(cs):
            return [CitationModel(doc_id=c.doc_id, page=c.page) for c in cs]

        return cls(
            answer=a.answer,
            abstained=a.abstained,
            provider=a.provider,
            top_score=None if a.top_score is None else round(a.top_score, 4),
            citations=cites(a.citations),
            invalid_citations=cites(a.invalid_citations),
            unsupported_sentences=a.unsupported_sentences,
        )


class AskResponse(AnswerModel):
    question: str
    passages: list[Passage]


class DocumentMeta(BaseModel):
    doc_id: str
    title: str
    source: str
    doc_type: str
    topic: str | None
    authors: list[str]
    published: str | None
    url: str
    licence: str
    n_pages: int


class DocumentResponse(DocumentMeta):
    pages: list[str] = Field(description="Cleaned page texts; pages[0] is page 1.")


class SummaryResponse(AnswerModel):
    doc_id: str


class CompareResponse(BaseModel):
    query: str
    doc_a: str
    doc_b: str
    passages_a: list[Passage]
    passages_b: list[Passage]


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    db: bool
    chunks: int
    strategy: str
    provider: str
