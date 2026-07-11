from __future__ import annotations

import re

from doc2knowledge.domain import Citation, RetrievedChunk


class InvalidCitationError(ValueError):
    pass


_SOURCE_PATTERN = re.compile(r"\[(S\d+)]")


def build_grounded_prompt(
    question: str,
    evidence: list[RetrievedChunk],
) -> tuple[str, dict[str, RetrievedChunk]]:
    sources = {f"S{index}": result for index, result in enumerate(evidence, start=1)}
    source_blocks: list[str] = []
    for label, result in sources.items():
        location = [result.document.filename]
        if result.chunk.source_page is not None:
            location.append(f"page {result.chunk.source_page}")
        if result.chunk.section:
            location.append(f"section {result.chunk.section}")
        source_blocks.append(f"[{label}] {' | '.join(location)}\n{result.chunk.text.strip()}")

    prompt = "\n\n".join(
        [
            "You are a careful document research assistant.",
            (
                "Answer only from the supplied sources. Cite every factual claim with one or "
                "more source labels such as [S1]. Do not invent sources or use outside "
                "knowledge. If the sources do not support an answer, say exactly: "
                '"There is not enough evidence in the indexed documents to answer."'
            ),
            "Sources:\n" + ("\n\n".join(source_blocks) or "(none)"),
            f"Question: {question.strip()}",
            "Answer:",
        ]
    )
    return prompt, sources


def resolve_citations(
    answer_text: str,
    sources: dict[str, RetrievedChunk],
) -> list[Citation]:
    labels: list[str] = []
    for label in _SOURCE_PATTERN.findall(answer_text):
        if label not in sources:
            raise InvalidCitationError(f"answer referenced unknown source label {label}")
        if label not in labels:
            labels.append(label)

    return [
        Citation(
            label=label,
            document_id=sources[label].document.id,
            filename=sources[label].document.filename,
            chunk_id=sources[label].chunk.id,
            source_page=sources[label].chunk.source_page,
            section=sources[label].chunk.section,
        )
        for label in labels
    ]
