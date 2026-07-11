from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from doc2knowledge.domain import Chunk, Document, DocumentStatus, RetrievedChunk
from doc2knowledge.generation.gemma import (
    GemmaGenerationService,
    GenerationNotConfiguredError,
    GenerationProviderError,
)


class FakeModels:
    def __init__(self, text: str | None = "Alpha was founded in 2020 [S1].") -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None

    def generate_content(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(text=self.text)


class FakeClient:
    def __init__(self, models: FakeModels) -> None:
        self.models = models


def evidence() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=Chunk("c1", "d1", 0, "Alpha was founded in 2020.", 3, "History"),
            document=Document(
                "d1",
                "alpha.txt",
                "text/plain",
                "h1",
                DocumentStatus.READY,
                datetime(2026, 7, 11, tzinfo=UTC),
                1,
                None,
            ),
            score=0.9,
        )
    ]


def test_generation_uses_configured_gemma_model_and_resolves_citations() -> None:
    models = FakeModels()
    service = GemmaGenerationService(
        api_key="secret",
        model_name="gemma-4-31b-it",
        client_factory=lambda api_key: FakeClient(models),
    )

    answer = service.generate("When was Alpha founded?", evidence())

    assert answer.text == "Alpha was founded in 2020 [S1]."
    assert [citation.label for citation in answer.citations] == ["S1"]
    assert answer.evidence == evidence()
    assert models.calls[0]["model"] == "gemma-4-31b-it"
    assert "When was Alpha founded?" in models.calls[0]["contents"]
    assert models.calls[0]["config"]["temperature"] == 0.1


def test_generation_abstains_without_evidence_without_calling_provider() -> None:
    models = FakeModels()
    service = GemmaGenerationService(
        api_key="secret",
        model_name="gemma-4-31b-it",
        client_factory=lambda api_key: FakeClient(models),
    )

    answer = service.generate("Unknown?", [])

    assert "not enough evidence" in answer.text.lower()
    assert answer.citations == []
    assert answer.evidence == []
    assert models.calls == []


def test_generation_rejects_uncited_or_invented_sources() -> None:
    uncited = GemmaGenerationService(
        api_key="secret",
        model_name="gemma-4-31b-it",
        client_factory=lambda api_key: FakeClient(FakeModels("Alpha was founded in 2020.")),
    )
    invented = GemmaGenerationService(
        api_key="secret",
        model_name="gemma-4-31b-it",
        client_factory=lambda api_key: FakeClient(FakeModels("Unsupported claim [S9].")),
    )

    with pytest.raises(GenerationProviderError, match="did not cite"):
        uncited.generate("question", evidence())
    with pytest.raises(GenerationProviderError, match="invalid citation"):
        invented.generate("question", evidence())


def test_missing_key_and_provider_failures_are_explicit() -> None:
    unconfigured = GemmaGenerationService(
        api_key=None,
        model_name="gemma-4-31b-it",
        client_factory=lambda api_key: FakeClient(FakeModels()),
    )
    with pytest.raises(GenerationNotConfiguredError):
        unconfigured.generate("question", evidence())

    models = FakeModels()
    models.error = RuntimeError("quota")
    failing = GemmaGenerationService(
        api_key="secret",
        model_name="gemma-4-31b-it",
        client_factory=lambda api_key: FakeClient(models),
    )
    with pytest.raises(GenerationProviderError, match="provider request failed"):
        failing.generate("question", evidence())

    empty_response = GemmaGenerationService(
        api_key="secret",
        model_name="gemma-4-31b-it",
        client_factory=lambda api_key: FakeClient(FakeModels(text=None)),
    )
    with pytest.raises(GenerationProviderError, match="empty response"):
        empty_response.generate("question", evidence())
