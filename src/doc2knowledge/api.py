from __future__ import annotations

from fastapi import FastAPI

from doc2knowledge import __version__
from doc2knowledge.config import Settings
from doc2knowledge.embeddings.base import EmbeddingService
from doc2knowledge.embeddings.mixedbread import MixedbreadEmbeddingService
from doc2knowledge.ingestion.service import IngestionService
from doc2knowledge.retrieval.service import RetrievalService
from doc2knowledge.routes.documents import router as documents_router
from doc2knowledge.routes.search import router as search_router
from doc2knowledge.storage.metadata import MetadataRepository
from doc2knowledge.storage.vectors import VectorRepository


def create_app(
    settings: Settings | None = None,
    *,
    embedding_service: EmbeddingService | None = None,
    vector_repository: VectorRepository | None = None,
) -> FastAPI:
    """Create a configured FastAPI application."""

    resolved_settings = settings or Settings()
    metadata = MetadataRepository(resolved_settings.data_dir / "metadata.sqlite3")
    embeddings = embedding_service or MixedbreadEmbeddingService(
        model_name=resolved_settings.embedding_model,
        dimensions=resolved_settings.embedding_dimensions,
    )
    vectors = vector_repository or VectorRepository(
        resolved_settings.data_dir / "vectors",
        model_name=embeddings.model_name,
        dimensions=embeddings.dimensions,
    )
    ingestion_service = IngestionService(
        metadata,
        resolved_settings.data_dir,
        embeddings,
        vectors,
        max_upload_bytes=resolved_settings.max_upload_bytes,
    )
    retrieval_service = RetrievalService(embeddings, vectors, metadata)

    app = FastAPI(
        title="doc2knowledge",
        version=__version__,
        description="Document ingestion, retrieval, and grounded question answering.",
    )
    app.state.settings = resolved_settings
    app.state.metadata_repository = metadata
    app.state.embedding_service = embeddings
    app.state.vector_repository = vectors
    app.state.ingestion_service = ingestion_service
    app.state.retrieval_service = retrieval_service
    app.include_router(documents_router)
    app.include_router(search_router)

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "doc2knowledge",
            "version": __version__,
        }

    return app
