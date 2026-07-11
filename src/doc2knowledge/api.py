from __future__ import annotations

from fastapi import FastAPI

from doc2knowledge import __version__
from doc2knowledge.config import Settings
from doc2knowledge.ingestion.service import IngestionService
from doc2knowledge.routes.documents import router as documents_router
from doc2knowledge.storage.metadata import MetadataRepository


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create a configured FastAPI application."""

    resolved_settings = settings or Settings()
    repository = MetadataRepository(resolved_settings.data_dir / "metadata.sqlite3")
    ingestion_service = IngestionService(
        repository,
        resolved_settings.data_dir,
        max_upload_bytes=resolved_settings.max_upload_bytes,
    )

    app = FastAPI(
        title="doc2knowledge",
        version=__version__,
        description="Document ingestion, retrieval, and grounded question answering.",
    )
    app.state.settings = resolved_settings
    app.state.metadata_repository = repository
    app.state.ingestion_service = ingestion_service
    app.include_router(documents_router)

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "doc2knowledge",
            "version": __version__,
        }

    return app
