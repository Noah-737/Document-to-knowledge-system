import pytest

from doc2knowledge.ingestion.extractors import (
    EmptyDocumentError,
    UnsupportedMediaTypeError,
    extract_document,
)


def test_extracts_text_markdown_and_html() -> None:
    text = extract_document(b"hello\nworld", "text/plain")
    markdown = extract_document(b"# Heading\nbody", "text/markdown")
    html = extract_document(
        b"<html><body><h1>Title</h1><p>Body</p></body></html>",
        "text/html",
    )

    assert [section.text for section in text.sections] == ["hello\nworld"]
    assert [section.text for section in markdown.sections] == ["# Heading\nbody"]
    assert html.sections[0].text == "Title\nBody"
    assert html.sections[0].section == "Title"


def test_rejects_unsupported_and_empty_documents() -> None:
    with pytest.raises(UnsupportedMediaTypeError):
        extract_document(b"data", "application/octet-stream")

    with pytest.raises(EmptyDocumentError):
        extract_document(b"   \n", "text/plain")
