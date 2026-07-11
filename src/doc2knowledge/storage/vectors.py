from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import cast

import faiss
import numpy as np
from numpy.typing import NDArray


class IncompatibleIndexError(RuntimeError):
    pass


@dataclass(frozen=True)
class VectorRecord:
    chunk_id: str
    document_id: str
    vector: list[float]


@dataclass(frozen=True)
class VectorMatch:
    chunk_id: str
    document_id: str
    score: float


@dataclass(frozen=True)
class _Mapping:
    chunk_id: str
    document_id: str


class VectorRepository:
    """One persistent cosine-similarity FAISS collection for all chunks."""

    def __init__(self, path: Path, *, model_name: str, dimensions: int) -> None:
        self._path = path
        self._path.mkdir(parents=True, exist_ok=True)
        self._model_name = model_name
        self._dimensions = dimensions
        self._lock = RLock()
        self._index_path = self._path / "index.faiss"
        self._mapping_path = self._path / "mapping.json"
        self._manifest_path = self._path / "manifest.json"
        self._index: faiss.Index = faiss.IndexFlatIP(dimensions)
        self._mappings: list[_Mapping] = []
        self._load_if_present()

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    def add(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        with self._lock:
            existing_ids = {mapping.chunk_id for mapping in self._mappings}
            incoming_ids = [record.chunk_id for record in records]
            if len(set(incoming_ids)) != len(incoming_ids):
                raise ValueError("chunk IDs must be unique within a batch")
            if existing_ids.intersection(incoming_ids):
                raise ValueError("chunk ID already exists in vector index")

            vectors = self._matrix([record.vector for record in records])
            self._index.add(vectors)
            self._mappings.extend(
                _Mapping(record.chunk_id, record.document_id) for record in records
            )
            self._persist()

    def search(
        self,
        query: list[float],
        top_k: int,
        *,
        document_ids: set[str] | None = None,
    ) -> list[VectorMatch]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        with self._lock:
            if self._index.ntotal == 0:
                return []
            query_matrix = self._matrix([query])
            scores, indices = self._index.search(query_matrix, int(self._index.ntotal))
            matches: list[VectorMatch] = []
            for score, index in zip(scores[0], indices[0], strict=True):
                if index < 0:
                    continue
                mapping = self._mappings[int(index)]
                if document_ids is not None and mapping.document_id not in document_ids:
                    continue
                matches.append(
                    VectorMatch(
                        chunk_id=mapping.chunk_id,
                        document_id=mapping.document_id,
                        score=float(score),
                    )
                )
                if len(matches) == top_k:
                    break
            return matches

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            keep_indices = [
                index
                for index, mapping in enumerate(self._mappings)
                if mapping.document_id != document_id
            ]
            removed = len(self._mappings) - len(keep_indices)
            if removed == 0:
                return 0

            new_index = faiss.IndexFlatIP(self._dimensions)
            if keep_indices:
                vectors = np.vstack(
                    [
                        cast(NDArray[np.float32], self._index.reconstruct(index))
                        for index in keep_indices
                    ]
                ).astype(np.float32)
                new_index.add(vectors)
            self._index = new_index
            self._mappings = [self._mappings[index] for index in keep_indices]
            self._persist()
            return removed

    def _matrix(self, vectors: list[list[float]]) -> NDArray[np.float32]:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self._dimensions:
            raise ValueError(f"vectors must have {self._dimensions} dimensions; got {matrix.shape}")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("zero vectors cannot be indexed")
        return cast(NDArray[np.float32], matrix / norms)

    def _load_if_present(self) -> None:
        present = [
            self._index_path.exists(),
            self._mapping_path.exists(),
            self._manifest_path.exists(),
        ]
        if not any(present):
            return
        if not all(present):
            raise IncompatibleIndexError("vector index files are incomplete")

        manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        if manifest != {
            "model_name": self._model_name,
            "dimensions": self._dimensions,
            "metric": "cosine",
        }:
            raise IncompatibleIndexError("vector index model configuration is incompatible")

        raw_mappings = json.loads(self._mapping_path.read_text(encoding="utf-8"))
        self._mappings = [
            _Mapping(
                chunk_id=str(item["chunk_id"]),
                document_id=str(item["document_id"]),
            )
            for item in raw_mappings
        ]
        self._index = faiss.read_index(str(self._index_path))
        if self._index.d != self._dimensions:
            raise IncompatibleIndexError("stored FAISS dimensions are incompatible")
        if self._index.ntotal != len(self._mappings):
            raise IncompatibleIndexError("vector mapping count does not match FAISS index")

    def _persist(self) -> None:
        temporary_index = self._path / "index.tmp.faiss"
        temporary_mapping = self._path / "mapping.tmp.json"
        temporary_manifest = self._path / "manifest.tmp.json"

        faiss.write_index(self._index, str(temporary_index))
        temporary_mapping.write_text(
            json.dumps([asdict(mapping) for mapping in self._mappings], indent=2),
            encoding="utf-8",
        )
        temporary_manifest.write_text(
            json.dumps(
                {
                    "model_name": self._model_name,
                    "dimensions": self._dimensions,
                    "metric": "cosine",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        os.replace(temporary_index, self._index_path)
        os.replace(temporary_mapping, self._mapping_path)
        os.replace(temporary_manifest, self._manifest_path)
