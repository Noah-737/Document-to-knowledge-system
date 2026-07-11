from __future__ import annotations

from doc2knowledge.domain import DocumentStatus, RetrievedChunk
from doc2knowledge.embeddings.base import EmbeddingService
from doc2knowledge.storage.metadata import MetadataRepository
from doc2knowledge.storage.vectors import VectorRepository


class RetrievalService:
    def __init__(
        self,
        embeddings: EmbeddingService,
        vectors: VectorRepository,
        metadata: MetadataRepository,
    ) -> None:
        self._embeddings = embeddings
        self._vectors = vectors
        self._metadata = metadata

    def search(
        self,
        query: str,
        *,
        top_k: int,
        document_ids: set[str] | None = None,
    ) -> list[RetrievedChunk]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        vector = self._embeddings.embed_query(normalized_query)
        matches = self._vectors.search(
            vector,
            top_k,
            document_ids=document_ids,
        )
        evidence: list[RetrievedChunk] = []
        for match in matches:
            chunk = self._metadata.get_chunk(match.chunk_id)
            document = self._metadata.get_document(match.document_id)
            if chunk is None or document is None:
                continue
            if document.status is not DocumentStatus.READY:
                continue
            evidence.append(
                RetrievedChunk(
                    chunk=chunk,
                    document=document,
                    score=match.score,
                )
            )
        return evidence
