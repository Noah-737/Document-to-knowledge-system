from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4, uuid5

from doc2knowledge.domain import Chunk, Document, DocumentStatus
from doc2knowledge.ingestion.chunking import chunk_document
from doc2knowledge.ingestion.extractors import (
    SUPPORTED_MEDIA_TYPES,
    ExtractionError,
    UnsupportedMediaTypeError,
    extract_document,
)
from doc2knowledge.storage.metadata import MetadataRepository


class UploadTooLargeError(ValueError):
    pass


@dataclass(frozen=True)
class IngestionResult:
    document: Document
    duplicate: bool


class IngestionService:
    def __init__(
        self,
        repository: MetadataRepository,
        data_dir: Path,
        *,
        max_upload_bytes: int,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self._repository = repository
        self._documents_dir = data_dir / "documents"
        self._documents_dir.mkdir(parents=True, exist_ok=True)
        self._max_upload_bytes = max_upload_bytes
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def ingest(self, filename: str, media_type: str, data: bytes) -> IngestionResult:
        if len(data) > self._max_upload_bytes:
            raise UploadTooLargeError(
                f"upload contains {len(data)} bytes; limit is {self._max_upload_bytes}"
            )

        normalized_media_type = media_type.split(";", maxsplit=1)[0].strip().lower()
        if normalized_media_type not in SUPPORTED_MEDIA_TYPES:
            raise UnsupportedMediaTypeError(normalized_media_type)

        content_hash = hashlib.sha256(data).hexdigest()
        duplicate = self._repository.get_document_by_hash(content_hash)
        if duplicate is not None:
            return IngestionResult(document=duplicate, duplicate=True)

        document_id = str(uuid4())
        document = Document(
            id=document_id,
            filename=Path(filename).name or "document",
            media_type=normalized_media_type,
            sha256=content_hash,
            status=DocumentStatus.PROCESSING,
            created_at=datetime.now(UTC),
            chunk_count=0,
            error_message=None,
        )
        self._repository.create_document(document)
        document_dir = self._documents_dir / document.id
        document_dir.mkdir(parents=True, exist_ok=False)
        (document_dir / "source").write_bytes(data)

        try:
            extracted = extract_document(data, normalized_media_type)
            drafts = chunk_document(
                extracted,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
            chunks = [
                Chunk(
                    id=str(uuid5(UUID(document.id), str(draft.ordinal))),
                    document_id=document.id,
                    ordinal=draft.ordinal,
                    text=draft.text,
                    source_page=draft.source_page,
                    section=draft.section,
                )
                for draft in drafts
            ]
            self._repository.save_chunks(chunks)
            ready = replace(
                document,
                status=DocumentStatus.READY,
                chunk_count=len(chunks),
            )
            self._repository.update_document(ready)
            return IngestionResult(document=ready, duplicate=False)
        except ExtractionError as error:
            failed = replace(
                document,
                status=DocumentStatus.FAILED,
                error_message=str(error),
            )
            self._repository.update_document(failed)
            raise

    def get_document(self, document_id: str) -> Document | None:
        return self._repository.get_document(document_id)

    def list_documents(self) -> list[Document]:
        return self._repository.list_documents()

    def delete_document(self, document_id: str) -> bool:
        deleted = self._repository.delete_document(document_id)
        if deleted:
            shutil.rmtree(self._documents_dir / document_id, ignore_errors=True)
        return deleted
