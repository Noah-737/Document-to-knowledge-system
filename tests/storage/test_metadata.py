from datetime import UTC, datetime
from pathlib import Path

from doc2knowledge.domain import Chunk, Document, DocumentStatus
from doc2knowledge.storage.metadata import MetadataRepository


def make_document(document_id: str = "doc-1") -> Document:
    return Document(
        id=document_id,
        filename="notes.txt",
        media_type="text/plain",
        sha256="abc123",
        status=DocumentStatus.PROCESSING,
        created_at=datetime(2026, 7, 11, tzinfo=UTC),
        chunk_count=0,
        error_message=None,
    )


def test_repository_persists_documents_chunks_and_deletion(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = MetadataRepository(database_path)
    document = make_document()
    chunks = [
        Chunk(
            id="chunk-1",
            document_id=document.id,
            ordinal=0,
            text="first chunk",
            source_page=1,
            section="Introduction",
        ),
        Chunk(
            id="chunk-2",
            document_id=document.id,
            ordinal=1,
            text="second chunk",
            source_page=2,
            section=None,
        ),
    ]

    repository.create_document(document)
    repository.save_chunks(chunks)
    repository.update_document(
        Document(
            **{
                **document.__dict__,
                "status": DocumentStatus.READY,
                "chunk_count": len(chunks),
            }
        )
    )

    reopened = MetadataRepository(database_path)
    stored = reopened.get_document(document.id)

    assert stored is not None
    assert stored.status is DocumentStatus.READY
    assert stored.chunk_count == 2
    assert reopened.get_document_by_hash("abc123") == stored
    assert reopened.list_documents() == [stored]
    assert reopened.get_chunks(document.id) == chunks

    assert reopened.delete_document(document.id) is True
    assert reopened.get_document(document.id) is None
    assert reopened.get_chunks(document.id) == []
    assert reopened.delete_document(document.id) is False
