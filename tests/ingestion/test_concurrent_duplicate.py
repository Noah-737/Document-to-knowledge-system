from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock

from doc2knowledge.domain import DocumentStatus
from doc2knowledge.ingestion.service import IngestionService
from doc2knowledge.storage.metadata import MetadataRepository
from doc2knowledge.storage.vectors import VectorRepository
from tests.fakes import FakeEmbeddingService


class RacingMetadataRepository(MetadataRepository):
    def __init__(self, database_path: Path) -> None:
        super().__init__(database_path)
        self._initial_lookup_barrier = Barrier(2)
        self._lookup_lock = Lock()
        self._initial_lookups = 0

    def get_document_by_hash(self, sha256: str):  # type: ignore[no-untyped-def]
        with self._lookup_lock:
            should_wait = self._initial_lookups < 2
            if should_wait:
                self._initial_lookups += 1
        result = super().get_document_by_hash(sha256)
        if should_wait:
            self._initial_lookup_barrier.wait(timeout=5)
        return result


def test_concurrent_identical_uploads_share_one_document(tmp_path: Path) -> None:
    repository = RacingMetadataRepository(tmp_path / "metadata.sqlite3")
    vectors = VectorRepository(
        tmp_path / "vectors",
        model_name="test-model",
        dimensions=3,
    )
    service = IngestionService(
        repository,
        tmp_path,
        FakeEmbeddingService(),
        vectors,
        max_upload_bytes=1024,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(service.ingest, f"notes-{index}.txt", "text/plain", b"alpha")
            for index in range(2)
        ]
        results = [future.result(timeout=10) for future in futures]

    assert sorted(result.duplicate for result in results) == [False, True]
    assert len({result.document.id for result in results}) == 1
    assert vectors.size == 1
    assert len(repository.list_documents()) == 1
    assert repository.list_documents()[0].status is DocumentStatus.READY
