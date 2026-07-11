from collections.abc import Sequence
from pathlib import Path

from fastapi.testclient import TestClient

from doc2knowledge.api import create_app
from doc2knowledge.config import Settings


class FailingEmbeddingService:
    model_name = "test-model"
    dimensions = 3

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("embedding failed")

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0, 0.0]


def test_embedding_failure_marks_document_failed_without_partial_vectors(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            data_dir=tmp_path,
            embedding_model="test-model",
            embedding_dimensions=3,
        ),
        embedding_service=FailingEmbeddingService(),
    )
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/documents",
        files={"file": ("notes.txt", b"alpha evidence", "text/plain")},
    )

    assert response.status_code == 500
    documents = client.get("/documents").json()
    assert len(documents) == 1
    assert documents[0]["status"] == "failed"
    assert documents[0]["error_message"] == "embedding failed"
    assert app.state.vector_repository.size == 0
