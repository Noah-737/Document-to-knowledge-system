from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, field_validator

from doc2knowledge.config import Settings
from doc2knowledge.domain import RetrievedChunk
from doc2knowledge.retrieval.service import RetrievalService

router = APIRouter(tags=["retrieval"])


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_ids: list[str] | None = None

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized


class EvidenceResponse(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    text: str
    score: float
    source_page: int | None
    section: str | None


class SearchResponse(BaseModel):
    evidence: list[EvidenceResponse]


def _retrieval(request: Request) -> RetrievalService:
    return cast(RetrievalService, request.app.state.retrieval_service)


def _settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def evidence_response(result: RetrievedChunk) -> EvidenceResponse:
    return EvidenceResponse(
        chunk_id=result.chunk.id,
        document_id=result.document.id,
        filename=result.document.filename,
        text=result.chunk.text,
        score=result.score,
        source_page=result.chunk.source_page,
        section=result.chunk.section,
    )


@router.post("/search", response_model=SearchResponse)
def search_documents(payload: SearchRequest, request: Request) -> SearchResponse:
    results = _retrieval(request).search(
        payload.query,
        top_k=payload.top_k or _settings(request).top_k,
        document_ids=set(payload.document_ids) if payload.document_ids else None,
    )
    return SearchResponse(evidence=[evidence_response(result) for result in results])
