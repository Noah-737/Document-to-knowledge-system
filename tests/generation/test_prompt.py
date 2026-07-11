from datetime import UTC, datetime

import pytest

from doc2knowledge.domain import Chunk, Document, DocumentStatus, RetrievedChunk
from doc2knowledge.generation.prompt import (
    InvalidCitationError,
    build_grounded_prompt,
    resolve_citations,
)


def evidence() -> list[RetrievedChunk]:
    now = datetime(2026, 7, 11, tzinfo=UTC)
    return [
        RetrievedChunk(
            chunk=Chunk("c1", "d1", 0, "Alpha was founded in 2020.", 3, "History"),
            document=Document(
                "d1",
                "alpha.txt",
                "text/plain",
                "h1",
                DocumentStatus.READY,
                now,
                1,
                None,
            ),
            score=0.93,
        ),
        RetrievedChunk(
            chunk=Chunk("c2", "d2", 0, "Beta launched in 2022.", None, None),
            document=Document(
                "d2",
                "beta.txt",
                "text/plain",
                "h2",
                DocumentStatus.READY,
                now,
                1,
                None,
            ),
            score=0.81,
        ),
    ]


def test_prompt_assigns_stable_labels_and_requires_grounding() -> None:
    prompt, sources = build_grounded_prompt("When was Alpha founded?", evidence())

    assert "Answer only from the supplied sources" in prompt
    assert "[S1] alpha.txt | page 3 | section History" in prompt
    assert "[S2] beta.txt" in prompt
    assert "Question: When was Alpha founded?" in prompt
    assert list(sources) == ["S1", "S2"]


def test_citations_are_resolved_once_in_answer_order() -> None:
    _, sources = build_grounded_prompt("question", evidence())

    citations = resolve_citations("Alpha was founded in 2020 [S1]. See [S1] and [S2].", sources)

    assert [citation.label for citation in citations] == ["S1", "S2"]
    assert citations[0].filename == "alpha.txt"
    assert citations[0].source_page == 3
    assert citations[1].chunk_id == "c2"


def test_unknown_source_label_is_rejected() -> None:
    _, sources = build_grounded_prompt("question", evidence())

    with pytest.raises(InvalidCitationError, match="S9"):
        resolve_citations("Unsupported claim [S9].", sources)
