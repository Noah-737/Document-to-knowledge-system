from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any, Protocol, cast

from doc2knowledge.domain import Answer, RetrievedChunk
from doc2knowledge.generation.base import GenerationService
from doc2knowledge.generation.prompt import (
    InvalidCitationError,
    build_grounded_prompt,
    resolve_citations,
)


class GenerationNotConfiguredError(RuntimeError):
    pass


class GenerationProviderError(RuntimeError):
    pass


class ResponseLike(Protocol):
    text: str | None


class ModelsLike(Protocol):
    def generate_content(self, **kwargs: Any) -> ResponseLike: ...


class ClientLike(Protocol):
    models: ModelsLike


ClientFactory = Callable[[str], ClientLike]


def _default_client_factory(api_key: str) -> ClientLike:
    from google import genai

    return cast(ClientLike, genai.Client(api_key=api_key))


class GemmaGenerationService(GenerationService):
    """Generate citation-grounded answers with Gemma through the Gemini API."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model_name: str,
        client_factory: ClientFactory = _default_client_factory,
    ) -> None:
        self._api_key = api_key
        self._model_name = model_name
        self._client_factory = client_factory
        self._client: ClientLike | None = None
        self._client_lock = Lock()

    def generate(self, question: str, evidence: list[RetrievedChunk]) -> Answer:
        if not evidence:
            return Answer(
                text="There is not enough evidence in the indexed documents to answer.",
                citations=[],
                evidence=[],
            )
        if not self._api_key:
            raise GenerationNotConfiguredError("GEMINI_API_KEY is required for generated answers")

        prompt, sources = build_grounded_prompt(question, evidence)
        try:
            response = self._get_client().models.generate_content(
                model=self._model_name,
                contents=prompt,
                config={
                    "temperature": 0.1,
                    "max_output_tokens": 1024,
                },
            )
        except Exception as error:
            raise GenerationProviderError("generation provider request failed") from error

        text = (response.text or "").strip()
        if not text:
            raise GenerationProviderError("generation provider returned an empty response")
        try:
            citations = resolve_citations(text, sources)
        except InvalidCitationError as error:
            raise GenerationProviderError(
                "generated answer contained an invalid citation"
            ) from error

        abstention = "not enough evidence" in text.lower()
        if not citations and not abstention:
            raise GenerationProviderError("generated answer did not cite supplied evidence")
        return Answer(text=text, citations=citations, evidence=evidence)

    def _get_client(self) -> ClientLike:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    if not self._api_key:
                        raise GenerationNotConfiguredError(
                            "GEMINI_API_KEY is required for generated answers"
                        )
                    self._client = self._client_factory(self._api_key)
        return self._client
