from __future__ import annotations

from dataclasses import dataclass

from doc2knowledge.ingestion.extractors import ExtractedDocument


@dataclass(frozen=True)
class ChunkDraft:
    ordinal: int
    text: str
    source_page: int | None
    section: str | None


def chunk_document(
    document: ExtractedDocument,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[ChunkDraft]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size")

    chunks: list[ChunkDraft] = []
    step = chunk_size - chunk_overlap
    for section in document.sections:
        text = section.text.strip()
        for start in range(0, len(text), step):
            value = text[start : start + chunk_size].strip()
            if value:
                chunks.append(
                    ChunkDraft(
                        ordinal=len(chunks),
                        text=value,
                        source_page=section.source_page,
                        section=section.section,
                    )
                )
            if start + chunk_size >= len(text):
                break
    return chunks
