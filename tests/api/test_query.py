from pathlib import Path

from fastapi.testclient import TestClient

from doc2knowledge.api import create_app
from doc2knowledge.config import Settings
from doc2knowledge.domain import Answer, Citation, RetrievedChunk
from doc2knowledge.generation.base import GenerationService
from doc2knowledge.generation.gemma import GenerationNotConfiguredError
from tests.fakes import FakeEmbeddingService


class FakeGenerationService(GenerationService):
    def generate(self, question: str, evidence: list[RetrievedChunk]) -> Answer:
        if not evidence:
            return Answer("There is not enough evidence to answer.", [], [])
        first = evidence[0]
        return Answer(
            text=f"Alpha answer [S1] for {question}",
            citations=[
                Citation(
                    label="S1",
                    document_id=first.document.id,
                    filename=first.document.filename,
                    chunk_id=first.chunk.id,
                    source_page=first.chunk.source_page,
                    section=first.chunk.section,
                )
            ],
            evidence=evidence,
        )


class UnconfiguredGenerationService(GenerationService):
    def generate(self, question: str, evidence: list[RetrievedChunk]) -> Answer:
        raise GenerationNotConfiguredError("missing key")


def make_client(
    tmp_path: Path,
    generation_service: GenerationService,
) -> TestClient:
    return TestClient(
        create_app(
            Settings(
                data_dir=tmp_path,
                embedding_model="test-model",
                embedding_dimensions=3,
            ),
            embedding_service=FakeEmbeddingService(),
            generation_service=generation_service,
        )
    )


def test_query_returns_grounded_answer_citations_and_evidence(tmp_path: Path) -> None:
    client = make_client(tmp_path, FakeGenerationService())
    uploaded = client.post(
        "/documents",
        files={"file": ("alpha.txt", b"alpha evidence", "text/plain")},
    ).json()["document"]

    response = client.post("/query", json={"question": "What is Alpha?", "top_k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Alpha answer [S1] for What is Alpha?"
    assert body["citations"][0]["filename"] == "alpha.txt"
    assert body["citations"][0]["document_id"] == uploaded["id"]
    assert body["evidence"][0]["text"] == "alpha evidence"


def test_query_abstains_when_no_documents_are_indexed(tmp_path: Path) -> None:
    client = make_client(tmp_path, FakeGenerationService())

    response = client.post("/query", json={"question": "What is missing?"})

    assert response.status_code == 200
    assert "not enough evidence" in response.json()["answer"].lower()
    assert response.json()["citations"] == []
    assert response.json()["evidence"] == []


def test_query_reports_unconfigured_generation(tmp_path: Path) -> None:
    client = make_client(tmp_path, UnconfiguredGenerationService())

    response = client.post("/query", json={"question": "What is Alpha?"})

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "generation_not_configured"


def test_query_rejects_blank_question(tmp_path: Path) -> None:
    client = make_client(tmp_path, FakeGenerationService())

    response = client.post("/query", json={"question": "   "})

    assert response.status_code == 422
