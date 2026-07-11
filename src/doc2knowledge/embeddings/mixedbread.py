from __future__ import annotations

from collections.abc import Callable, Sequence
from threading import Lock
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray


class SentenceEncoder(Protocol):
    def encode(
        self,
        sentences: str | Sequence[str],
        *,
        prompt_name: str | None = None,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
    ) -> NDArray[np.float32]: ...


ModelFactory = Callable[[str, int], SentenceEncoder]


def _default_model_factory(model_name: str, dimensions: int) -> SentenceEncoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, truncate_dim=dimensions)
    return cast(SentenceEncoder, model)


class MixedbreadEmbeddingService:
    """Lazy, reusable adapter for mxbai retrieval embeddings."""

    def __init__(
        self,
        *,
        model_name: str,
        dimensions: int,
        model_factory: ModelFactory = _default_model_factory,
    ) -> None:
        self._model_name = model_name
        self._dimensions = dimensions
        self._model_factory = model_factory
        self._model: SentenceEncoder | None = None
        self._model_lock = Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        encoded = self._encoder().encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        array = np.asarray(encoded, dtype=np.float32)
        if array.ndim != 2 or array.shape != (len(texts), self._dimensions):
            raise ValueError(
                f"embedding model returned unexpected document dimensions: {array.shape}"
            )
        return [[float(value) for value in row] for row in array]

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise ValueError("query must not be empty")
        encoded = self._encoder().encode(
            query,
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        array = np.asarray(encoded, dtype=np.float32).reshape(-1)
        if array.shape != (self._dimensions,):
            raise ValueError(f"embedding model returned unexpected query dimensions: {array.shape}")
        return [float(value) for value in array]

    def _encoder(self) -> SentenceEncoder:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = self._model_factory(
                        self._model_name,
                        self._dimensions,
                    )
        return self._model
