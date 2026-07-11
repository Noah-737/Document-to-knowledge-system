from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class EmbeddingService(Protocol):
    @property
    def dimensions(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...
