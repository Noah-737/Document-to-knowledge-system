from doc2knowledge.ingestion.chunking import chunk_document
from doc2knowledge.ingestion.extractors import ExtractedDocument, ExtractedSection


def test_chunking_preserves_order_and_source_metadata() -> None:
    extracted = ExtractedDocument(
        sections=[
            ExtractedSection(text="abcdefghij", source_page=3, section="Alpha"),
            ExtractedSection(text="klmnop", source_page=4, section="Beta"),
        ]
    )

    chunks = chunk_document(extracted, chunk_size=6, chunk_overlap=2)

    assert [chunk.ordinal for chunk in chunks] == [0, 1, 2]
    assert [chunk.text for chunk in chunks] == ["abcdef", "efghij", "klmnop"]
    assert chunks[0].source_page == 3
    assert chunks[0].section == "Alpha"
    assert chunks[2].source_page == 4
    assert chunks[2].section == "Beta"
