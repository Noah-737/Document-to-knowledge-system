from datetime import UTC, datetime
from pathlib import Path

from doc2knowledge.domain import Chunk, Document, DocumentStatus
from doc2knowledge.retrieval.service import RetrievalService
from doc2knowledge.storage.metadata import MetadataRepository
from doc2knowledge.storage.vectors import VectorRecord, VectorRepository
from tests.fakes import FakeEmbeddingService


def test_retrieval_ranks_and_filters_evidence(tmp_path: Path) -> None:
    metadata = MetadataRepository(tmp_path / "metadata.sqlite3")
    vectors = VectorRepository(
        tmp_path / "vectors",
        model_name="test-model",
        dimensions=3,
    )
    now = datetime(2026, 7, 11, tzinfo=UTC)
    documents = [
        Document("d1", "alpha.txt", "text/plain", "h1", DocumentStatus.READY, now, 1, None),
        Document("d2", "beta.txt", "text/plain", "h2", DocumentStatus.READY, now, 1, None),
    ]
    chunks = [
        Chunk("c1", "d1", 0, "alpha evidence", 1, "Alpha"),
        Chunk("c2", "d2", 0, "beta evidence", 2, "Beta"),
    ]
    for document in documents:
        metadata.create_document(document)
    metadata.save_chunks(chunks)
    vectors.add(
        [
            VectorRecord("c1", "d1", [1.0, 0.0, 0.0]),
            VectorRecord("c2", "d2", [0.0, 1.0, 0.0]),
        ]
    )
    service = RetrievalService(FakeEmbeddingService(), vectors, metadata)

    results = service.search("alpha question", top_k=2)
    filtered = service.search("alpha question", top_k=2, document_ids={"d2"})

    assert [result.chunk.id for result in results] == ["c1", "c2"]
    assert results[0].document.filename == "alpha.txt"
    assert results[0].score == 1.0
    assert [result.chunk.id for result in filtered] == ["c2"]
