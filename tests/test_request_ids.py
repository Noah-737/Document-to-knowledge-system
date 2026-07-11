from pathlib import Path

from fastapi.testclient import TestClient

from doc2knowledge.api import create_app
from doc2knowledge.config import Settings
from tests.fakes import FakeEmbeddingService


def make_client(tmp_path: Path) -> TestClient:
    return TestClient(
        create_app(
            Settings(
                data_dir=tmp_path,
                embedding_model="test-model",
                embedding_dimensions=3,
            ),
            embedding_service=FakeEmbeddingService(),
        )
    )


def test_request_id_is_generated_and_returned(tmp_path: Path) -> None:
    response = make_client(tmp_path).get("/health")

    request_id = response.headers["x-request-id"]
    assert request_id
    assert len(request_id) == 36


def test_valid_caller_request_id_is_preserved(tmp_path: Path) -> None:
    response = make_client(tmp_path).get(
        "/health",
        headers={"X-Request-ID": "client-request-123"},
    )

    assert response.headers["x-request-id"] == "client-request-123"


def test_unsafe_request_id_is_replaced(tmp_path: Path) -> None:
    unsafe_request_id = "bad request id"
    response = make_client(tmp_path).get(
        "/health",
        headers={"X-Request-ID": unsafe_request_id},
    )

    assert response.headers["x-request-id"] != unsafe_request_id
    assert len(response.headers["x-request-id"]) == 36
