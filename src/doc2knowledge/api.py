from __future__ import annotations

from fastapi import FastAPI

from doc2knowledge import __version__
from doc2knowledge.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create a configured FastAPI application."""

    app = FastAPI(
        title="doc2knowledge",
        version=__version__,
        description="Document ingestion, retrieval, and grounded question answering.",
    )
    app.state.settings = settings or Settings()

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "doc2knowledge",
            "version": __version__,
        }

    return app


app = create_app()
