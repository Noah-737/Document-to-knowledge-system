from pathlib import Path

from fastapi.testclient import TestClient

from doc2knowledge.api import create_app
from doc2knowledge.config import Settings
from tests.fakes import FakeEmbeddingService


def test_readiness_reports_models_capacity_and_generation_state(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            Settings(
                data_dir=tmp_path,
                embedding_model="test-model",
                embedding_dimensions=3,
                llm_model="gemma-4-31b-it",
                processing_workers=2,
            ),
            embedding_service=FakeEmbeddingService(),
        )
    )

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "embedding_model": "test-model",
        "embedding_dimensions": 3,
        "llm_model": "gemma-4-31b-it",
        "generation_configured": False,
        "processing_workers": 2,
        "indexed_vectors": 0,
    }
    assert "api_key" not in response.text.lower()
    assert "secret" not in response.text.lower()
