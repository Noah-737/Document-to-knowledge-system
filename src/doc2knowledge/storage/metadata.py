from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import cast

from doc2knowledge.domain import Chunk, Document, DocumentStatus


class MetadataRepository:
    """Persist document and chunk metadata in SQLite."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    source_page INTEGER,
                    section TEXT,
                    UNIQUE(document_id, ordinal)
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                ON chunks(document_id);
                """
            )

    def create_document(self, document: Document) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    id, filename, media_type, sha256, status,
                    created_at, chunk_count, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.id,
                    document.filename,
                    document.media_type,
                    document.sha256,
                    document.status.value,
                    document.created_at.isoformat(),
                    document.chunk_count,
                    document.error_message,
                ),
            )

    def update_document(self, document: Document) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET filename = ?, media_type = ?, sha256 = ?, status = ?,
                    created_at = ?, chunk_count = ?, error_message = ?
                WHERE id = ?
                """,
                (
                    document.filename,
                    document.media_type,
                    document.sha256,
                    document.status.value,
                    document.created_at.isoformat(),
                    document.chunk_count,
                    document.error_message,
                    document.id,
                ),
            )

    def get_document(self, document_id: str) -> Document | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
        return self._document_from_row(row) if row is not None else None

    def get_document_by_hash(self, sha256: str) -> Document | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE sha256 = ?",
                (sha256,),
            ).fetchone()
        return self._document_from_row(row) if row is not None else None

    def list_documents(self) -> list[Document]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at, id"
            ).fetchall()
        return [self._document_from_row(row) for row in rows]

    def save_chunks(self, chunks: Iterable[Chunk]) -> None:
        values = [
            (
                chunk.id,
                chunk.document_id,
                chunk.ordinal,
                chunk.text,
                chunk.source_page,
                chunk.section,
            )
            for chunk in chunks
        ]
        if not values:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO chunks (
                    id, document_id, ordinal, text, source_page, section
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                values,
            )

    def get_chunks(self, document_id: str) -> list[Chunk]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM chunks
                WHERE document_id = ?
                ORDER BY ordinal
                """,
                (document_id,),
            ).fetchall()
        return [self._chunk_from_row(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM chunks WHERE id = ?",
                (chunk_id,),
            ).fetchone()
        return self._chunk_from_row(row) if row is not None else None

    def delete_document(self, document_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM documents WHERE id = ?",
                (document_id,),
            )
        return cursor.rowcount > 0

    @staticmethod
    def _document_from_row(row: sqlite3.Row) -> Document:
        return Document(
            id=cast(str, row["id"]),
            filename=cast(str, row["filename"]),
            media_type=cast(str, row["media_type"]),
            sha256=cast(str, row["sha256"]),
            status=DocumentStatus(cast(str, row["status"])),
            created_at=datetime.fromisoformat(cast(str, row["created_at"])),
            chunk_count=cast(int, row["chunk_count"]),
            error_message=cast(str | None, row["error_message"]),
        )

    @staticmethod
    def _chunk_from_row(row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=cast(str, row["id"]),
            document_id=cast(str, row["document_id"]),
            ordinal=cast(int, row["ordinal"]),
            text=cast(str, row["text"]),
            source_page=cast(int | None, row["source_page"]),
            section=cast(str | None, row["section"]),
        )
