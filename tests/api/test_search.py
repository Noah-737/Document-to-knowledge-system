from pathlib import Path

from fastapi.testclient import TestClient

from doc2knowledge.api import create_app
from doc2knowledge.config import Settings
from tests.fakes import FakeEmbeddingService


def test_search_returns_ranked_source_metadata(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            Settings(
                data_dir=tmp_path,
                embedding_model="test-model",
                embedding_dimensions=3,
            ),
            embedding_service=FakeEmbeddingService(),
        )
    )
    alpha = client.post(
        "/documents",
        files={"file": ("alpha.txt", b"alpha evidence", "text/plain")},
    ).json()["document"]
    beta = client.post(
        "/documents",
        files={"file": ("beta.txt", b"beta evidence", "text/plain")},
    ).json()["document"]

    response = client.post("/search", json={"query": "alpha question", "top_k": 2})
    filtered = client.post(
        "/search",
        json={
            "query": "alpha question",
            "top_k": 2,
            "document_ids": [beta["id"]],
        },
    )

    assert response.status_code == 200
    evidence = response.json()["evidence"]
    assert [item["filename"] for item in evidence] == ["alpha.txt", "beta.txt"]
    assert evidence[0]["document_id"] == alpha["id"]
    assert evidence[0]["score"] == 1.0
    assert evidence[0]["text"] == "alpha evidence"

    assert filtered.status_code == 200
    assert [item["document_id"] for item in filtered.json()["evidence"]] == [beta["id"]]


def test_search_rejects_empty_query(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            Settings(
                data_dir=tmp_path,
                embedding_model="test-model",
                embedding_dimensions=3,
            ),
            embedding_service=FakeEmbeddingService(),
        )
    )

    response = client.post("/search", json={"query": "   "})

    assert response.status_code == 422
