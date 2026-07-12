# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] - 2026-07-11

### Added

- FastAPI service with health and readiness endpoints.
- PDF, HTML, Markdown, and plain-text ingestion with validation, content-hash deduplication,
  chunking, and document lifecycle APIs.
- Lazy Mixedbread embeddings and a persistent, thread-safe FAISS vector index.
- Semantic search with source metadata and optional document filtering.
- Evidence-grounded Gemma answers with validated source citations.
- Bounded processing, atomic concurrent deduplication, request IDs, and structured logs.
- Retrieval evaluation with recall@k, mean reciprocal rank, and source-diversity metrics.
- Automated linting, formatting, strict type checking, coverage enforcement, package builds,
  and Docker builds.

### Changed

- Replaced the original top-level prototype with the packaged `src/doc2knowledge` application.

### Deferred

- OCR for scanned PDFs, background job orchestration, provider retries and model metrics,
  hybrid retrieval/reranking, and a versioned real-world benchmark corpus remain optional
  follow-up improvements.
