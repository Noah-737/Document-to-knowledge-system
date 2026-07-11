from pathlib import Path

import pytest

from doc2knowledge.storage.vectors import (
    IncompatibleIndexError,
    VectorRecord,
    VectorRepository,
)


def test_vector_repository_persists_searches_filters_and_deletes(tmp_path: Path) -> None:
    repository = VectorRepository(
        tmp_path / "vectors",
        model_name="test-model",
        dimensions=3,
    )
    repository.add(
        [
            VectorRecord("c1", "d1", [1.0, 0.0, 0.0]),
            VectorRecord("c2", "d2", [0.8, 0.2, 0.0]),
            VectorRecord("c3", "d2", [0.0, 1.0, 0.0]),
        ]
    )

    assert [match.chunk_id for match in repository.search([1.0, 0.0, 0.0], 2)] == [
        "c1",
        "c2",
    ]
    assert [
        match.chunk_id
        for match in repository.search(
            [1.0, 0.0, 0.0],
            2,
            document_ids={"d2"},
        )
    ] == ["c2", "c3"]

    reopened = VectorRepository(
        tmp_path / "vectors",
        model_name="test-model",
        dimensions=3,
    )
    assert [match.chunk_id for match in reopened.search([0.0, 1.0, 0.0], 1)] == ["c3"]

    assert reopened.delete_document("d2") == 2
    assert [match.chunk_id for match in reopened.search([1.0, 0.0, 0.0], 5)] == ["c1"]
    assert reopened.delete_document("missing") == 0


def test_vector_repository_rejects_incompatible_or_invalid_vectors(tmp_path: Path) -> None:
    path = tmp_path / "vectors"
    repository = VectorRepository(path, model_name="test-model", dimensions=3)

    with pytest.raises(ValueError, match="dimensions"):
        repository.add([VectorRecord("c1", "d1", [1.0, 0.0])])

    repository.add([VectorRecord("c1", "d1", [1.0, 0.0, 0.0])])

    with pytest.raises(IncompatibleIndexError):
        VectorRepository(path, model_name="different-model", dimensions=3)

    with pytest.raises(IncompatibleIndexError):
        VectorRepository(path, model_name="test-model", dimensions=2)
