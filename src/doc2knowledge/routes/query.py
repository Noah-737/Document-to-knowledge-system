from __future__ import annotations

from typing import cast

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

from doc2knowledge.config import Settings
from doc2knowledge.domain import Answer, Citation
from doc2knowledge.generation.base import GenerationService
from doc2knowledge.generation.gemma import (
    GenerationNotConfiguredError,
    GenerationProviderError,
)
from doc2knowledge.retrieval.service import RetrievalService
from doc2knowledge.routes.search import EvidenceResponse, evidence_response

router = APIRouter(tags=["generation"])


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_ids: list[str] | None = None

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be blank")
        return normalized


class CitationResponse(BaseModel):
    label: str
    document_id: str
    filename: str
    chunk_id: str
    source_page: int | None
    section: str | None


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    evidence: list[EvidenceResponse]


def _retrieval(request: Request) -> RetrievalService:
    return cast(RetrievalService, request.app.state.retrieval_service)


def _generation(request: Request) -> GenerationService:
    return cast(GenerationService, request.app.state.generation_service)


def _settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def _citation_response(citation: Citation) -> CitationResponse:
    return CitationResponse(
        label=citation.label,
        document_id=citation.document_id,
        filename=citation.filename,
        chunk_id=citation.chunk_id,
        source_page=citation.source_page,
        section=citation.section,
    )


def _query_response(answer: Answer) -> QueryResponse:
    return QueryResponse(
        answer=answer.text,
        citations=[_citation_response(citation) for citation in answer.citations],
        evidence=[evidence_response(result) for result in answer.evidence],
    )


@router.post("/query", response_model=QueryResponse)
def query_documents(payload: QueryRequest, request: Request) -> QueryResponse:
    evidence = _retrieval(request).search(
        payload.question,
        top_k=payload.top_k or _settings(request).top_k,
        document_ids=set(payload.document_ids) if payload.document_ids else None,
    )
    try:
        answer = _generation(request).generate(payload.question, evidence)
    except GenerationNotConfiguredError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "generation_not_configured", "message": str(error)},
        ) from error
    except GenerationProviderError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "generation_provider_error", "message": str(error)},
        ) from error
    return _query_response(answer)
