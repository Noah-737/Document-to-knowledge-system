from collections.abc import Sequence
from typing import Any

import numpy as np

from doc2knowledge.embeddings.mixedbread import MixedbreadEmbeddingService


class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def encode(
        self,
        sentences: str | Sequence[str],
        **kwargs: Any,
    ) -> np.ndarray:
        self.calls.append({"sentences": sentences, **kwargs})
        if isinstance(sentences, str):
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return np.array(
            [[1.0, 0.0, 0.0] for _ in sentences],
            dtype=np.float32,
        )


def test_model_is_lazy_reused_and_query_uses_retrieval_prompt() -> None:
    encoder = FakeEncoder()
    factory_calls: list[tuple[str, int]] = []

    def factory(model_name: str, dimensions: int) -> FakeEncoder:
        factory_calls.append((model_name, dimensions))
        return encoder

    service = MixedbreadEmbeddingService(
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        dimensions=3,
        model_factory=factory,
    )

    assert factory_calls == []
    assert service.embed_documents(["alpha", "beta"]) == [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    assert service.embed_query("question") == [1.0, 0.0, 0.0]
    assert factory_calls == [("mixedbread-ai/mxbai-embed-large-v1", 3)]

    document_call, query_call = encoder.calls
    assert "prompt_name" not in document_call
    assert query_call["prompt_name"] == "query"
    assert document_call["normalize_embeddings"] is True
    assert query_call["normalize_embeddings"] is True
    assert service.dimensions == 3
