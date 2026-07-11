from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class DocumentStatus(StrEnum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class Document:
    id: str
    filename: str
    media_type: str
    sha256: str
    status: DocumentStatus
    created_at: datetime
    chunk_count: int
    error_message: str | None


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    ordinal: int
    text: str
    source_page: int | None
    section: str | None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    document: Document
    score: float


@dataclass(frozen=True)
class Citation:
    label: str
    document_id: str
    filename: str
    chunk_id: str
    source_page: int | None
    section: str | None


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation]
    evidence: list[RetrievedChunk]
