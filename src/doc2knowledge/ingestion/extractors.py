from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from bs4 import BeautifulSoup
from pypdf import PdfReader


class ExtractionError(ValueError):
    """Base class for extraction failures visible to the caller."""


class UnsupportedMediaTypeError(ExtractionError):
    pass


class EmptyDocumentError(ExtractionError):
    pass


@dataclass(frozen=True)
class ExtractedSection:
    text: str
    source_page: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class ExtractedDocument:
    sections: list[ExtractedSection]


SUPPORTED_MEDIA_TYPES = {
    "application/pdf",
    "text/html",
    "text/plain",
    "text/markdown",
}


def extract_document(data: bytes, media_type: str) -> ExtractedDocument:
    normalized_media_type = media_type.split(";", maxsplit=1)[0].strip().lower()
    if normalized_media_type not in SUPPORTED_MEDIA_TYPES:
        raise UnsupportedMediaTypeError(normalized_media_type)

    if normalized_media_type == "application/pdf":
        sections = _extract_pdf(data)
    elif normalized_media_type == "text/html":
        sections = _extract_html(data)
    else:
        sections = [ExtractedSection(text=_decode_utf8(data))]

    usable_sections = [
        ExtractedSection(
            text=section.text.strip(),
            source_page=section.source_page,
            section=section.section,
        )
        for section in sections
        if section.text.strip()
    ]
    if not usable_sections:
        raise EmptyDocumentError("document contains no extractable text")
    return ExtractedDocument(sections=usable_sections)


def _decode_utf8(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ExtractionError("document is not valid UTF-8") from error


def _extract_html(data: bytes) -> list[ExtractedSection]:
    soup = BeautifulSoup(_decode_utf8(data), "html.parser")
    title_node = soup.find(["h1", "title"])
    title = title_node.get_text(" ", strip=True) if title_node is not None else None
    return [
        ExtractedSection(
            text=soup.get_text("\n", strip=True),
            section=title or None,
        )
    ]


def _extract_pdf(data: bytes) -> list[ExtractedSection]:
    try:
        reader = PdfReader(BytesIO(data))
        return [
            ExtractedSection(
                text=page.extract_text() or "",
                source_page=index,
            )
            for index, page in enumerate(reader.pages, start=1)
        ]
    except Exception as error:
        raise ExtractionError("unable to parse PDF") from error
