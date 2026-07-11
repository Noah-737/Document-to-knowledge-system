from collections.abc import Sequence


class FakeEmbeddingService:
    model_name = "test-model"
    dimensions = 3

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        if "alpha" in lowered:
            return [1.0, 0.0, 0.0]
        if "beta" in lowered:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]
