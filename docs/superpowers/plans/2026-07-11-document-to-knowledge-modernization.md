# Document-to-Knowledge Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current upload-to-FAISS prototype into a tested document ingestion, retrieval, and citation-grounded RAG service.

**Architecture:** Build a modular FastAPI monolith under `src/doc2knowledge`. Keep settings, domain models, extractors, embedding adapters, storage, retrieval, and generation behind focused interfaces. Use SQLite for metadata, one persistent FAISS collection for vectors, `mixedbread-ai/mxbai-embed-large-v1` for embeddings, and configurable `gemma-4-31b-it` through the Gemini API for generation.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic Settings, SQLAlchemy/SQLite, sentence-transformers, FAISS CPU, google-genai, pytest, Ruff, mypy, coverage, Docker, GitHub Actions.

## Implementation status

The initial service scope was completed on 2026-07-11 through merged changes #12-#16.
Tasks 1-7 are implemented with deterministic tests. Task 8 delivered bounded processing,
request IDs, structured logging, readiness reporting, and atomic concurrent deduplication;
OCR remains deferred. Task 9 delivered the evaluation library, CLI, tests, example question
schema, and operating documentation; a real benchmark corpus remains user-supplied.

Deferred follow-ups are scanned-PDF OCR, background job orchestration, provider retries and
model/API metrics, hybrid retrieval or reranking, and a versioned real-world benchmark corpus.
They are not part of the documented 0.1.0 feature set.


## Global Constraints

- Default embedding model: `mixedbread-ai/mxbai-embed-large-v1`.
- Default generation model: `gemma-4-31b-it`.
- Normal CI must not download models or call external APIs.
- All expensive adapters must be dependency-injectable and testable with deterministic fakes.
- Runtime files and secrets must never be committed.
- Each PR must leave its branch internally runnable and tested.

---

### Task 1: Foundation, package layout, and CI

**Files:**
- Create: `src/doc2knowledge/__init__.py`
- Create: `src/doc2knowledge/config.py`
- Create: `src/doc2knowledge/api.py`
- Create: `tests/test_health.py`
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`
- Modify: `Dockerfile`
- Modify: `Makefile`
- Modify: `README.md`
- Delete after migration: legacy top-level `doc2knowledge/`

**Interfaces:**
- Produces: `create_app() -> FastAPI`
- Produces: `Settings` with environment-backed configuration

- [ ] Write `tests/test_health.py` asserting `GET /health` returns `{"status": "ok"}`.
- [ ] Run the test and confirm it fails because `doc2knowledge.api` does not exist.
- [ ] Add the package, settings, and minimal app factory.
- [ ] Consolidate all dependencies into `pyproject.toml` and make editable installation discover `src/doc2knowledge`.
- [ ] Make Docker run Uvicorn as a non-root user.
- [ ] Make CI run Ruff, mypy, pytest with coverage, package build, and Docker build.
- [ ] Run the full verification suite and commit.

### Task 2: Domain models and metadata repository

**Files:**
- Create: `src/doc2knowledge/domain.py`
- Create: `src/doc2knowledge/storage/metadata.py`
- Create: `src/doc2knowledge/storage/__init__.py`
- Create: `tests/storage/test_metadata.py`

**Interfaces:**
- Produces: `DocumentStatus`, `Document`, `Chunk`, `RetrievedChunk`, `Citation`, `Answer`
- Produces: `MetadataRepository.create_document`, `get_document`, `list_documents`, `save_chunks`, `get_chunks`, `delete_document`

- [ ] Write repository tests for create/read/list, duplicate hash lookup, chunk persistence, restart persistence, and cascade deletion.
- [ ] Confirm tests fail before implementation.
- [ ] Implement SQLite schema and repository transactions.
- [ ] Run repository tests and commit.

### Task 3: Validated document ingestion

**Files:**
- Create: `src/doc2knowledge/ingestion/extractors.py`
- Create: `src/doc2knowledge/ingestion/chunking.py`
- Create: `src/doc2knowledge/ingestion/service.py`
- Create: `src/doc2knowledge/ingestion/__init__.py`
- Create: `src/doc2knowledge/routes/documents.py`
- Create: `tests/ingestion/test_extractors.py`
- Create: `tests/ingestion/test_service.py`
- Create: `tests/api/test_documents.py`
- Modify: `src/doc2knowledge/api.py`

**Interfaces:**
- Produces: `extract_document(path, media_type) -> ExtractedDocument`
- Produces: `chunk_document(extracted) -> list[ChunkDraft]`
- Produces: `IngestionService.ingest(filename, media_type, data) -> IngestionResult`

- [ ] Write extractor tests for TXT, Markdown, HTML, text PDF, unsupported media, and empty extraction.
- [ ] Write ingestion tests for UUID identity, SHA-256 deduplication, size limit, failure state, and source metadata.
- [ ] Write API tests for upload, list, status, and delete.
- [ ] Confirm tests fail before implementation.
- [ ] Implement extractors, chunking, ingestion orchestration, and routes using fake downstream indexing.
- [ ] Run tests and commit.

### Task 4: Mixedbread embedding adapter

**Files:**
- Create: `src/doc2knowledge/embeddings/base.py`
- Create: `src/doc2knowledge/embeddings/mixedbread.py`
- Create: `src/doc2knowledge/embeddings/__init__.py`
- Create: `tests/embeddings/test_mixedbread.py`

**Interfaces:**
- Produces: `EmbeddingService` protocol
- Produces: `MixedbreadEmbeddingService.embed_documents` and `embed_query`

- [ ] Write tests with a fake SentenceTransformer verifying lazy single initialization, no document prompt, query prompt `query`, normalized output, batch preservation, and 1024 dimensions.
- [ ] Confirm tests fail before implementation.
- [ ] Implement the adapter without loading the model at module import.
- [ ] Add an opt-in integration marker that is excluded from normal CI.
- [ ] Run tests and commit.

### Task 5: Persistent unified FAISS repository

**Files:**
- Create: `src/doc2knowledge/storage/vectors.py`
- Create: `tests/storage/test_vectors.py`
- Modify: `src/doc2knowledge/config.py`

**Interfaces:**
- Produces: `VectorRepository.add`, `search`, `delete_document`, `load`
- Persists: FAISS index, row-to-chunk mapping, model compatibility manifest

- [ ] Write tests for add/search, restart persistence, document deletion, atomic rebuild, and incompatible dimensions/model rejection.
- [ ] Confirm tests fail before implementation.
- [ ] Implement one normalized inner-product FAISS index with a write lock and atomic persistence.
- [ ] Run tests and commit.

### Task 6: Retrieval service and search API

**Files:**
- Create: `src/doc2knowledge/retrieval/service.py`
- Create: `src/doc2knowledge/retrieval/__init__.py`
- Create: `src/doc2knowledge/routes/search.py`
- Create: `tests/retrieval/test_service.py`
- Create: `tests/api/test_search.py`
- Modify: `src/doc2knowledge/api.py`
- Modify: `src/doc2knowledge/ingestion/service.py`

**Interfaces:**
- Produces: `RetrievalService.search(query, top_k, document_ids) -> list[RetrievedChunk]`

- [ ] Write retrieval tests for ranking, top-k, document filtering, missing chunks, and empty queries.
- [ ] Write API tests for `/search` response metadata.
- [ ] Confirm tests fail before implementation.
- [ ] Wire ingestion to embed and index chunks transactionally.
- [ ] Implement retrieval and route.
- [ ] Run tests and commit.

### Task 7: Gemma generation and query API

**Files:**
- Create: `src/doc2knowledge/generation/base.py`
- Create: `src/doc2knowledge/generation/gemma.py`
- Create: `src/doc2knowledge/generation/prompt.py`
- Create: `src/doc2knowledge/generation/__init__.py`
- Create: `src/doc2knowledge/routes/query.py`
- Create: `tests/generation/test_prompt.py`
- Create: `tests/generation/test_gemma.py`
- Create: `tests/api/test_query.py`
- Modify: `src/doc2knowledge/api.py`

**Interfaces:**
- Produces: `GenerationService.generate(question, chunks) -> Answer`
- Uses model setting: `gemma-4-31b-it`

- [ ] Write prompt tests that assign stable `[S1]` labels and require evidence-only answers.
- [ ] Write adapter tests with a fake Google client verifying model selection, timeout/error mapping, and citation-label validation.
- [ ] Write API tests for cited answers, insufficient evidence, missing API key, and provider failure.
- [ ] Confirm tests fail before implementation.
- [ ] Implement Google Gen AI adapter, prompt construction, label resolution, and `/query`.
- [ ] Run tests and commit.

### Task 8: Processing robustness and observability

**Files:**
- Create: `src/doc2knowledge/logging.py`
- Create: `src/doc2knowledge/ingestion/ocr.py`
- Create: `tests/ingestion/test_ocr.py`
- Create: `tests/test_logging.py`
- Modify: ingestion and API modules

**Interfaces:**
- Produces: request/document correlation IDs
- Produces: optional OCR fallback and bounded processing execution

- [ ] Write tests for scanned-PDF fallback, bounded concurrency, structured errors, and secret redaction.
- [ ] Confirm tests fail before implementation.
- [ ] Implement OCR behind an optional dependency and thread-pool processing limits.
- [ ] Add structured logs and request IDs.
- [ ] Run tests and commit.

### Task 9: Retrieval evaluation harness

**Files:**
- Create: `eval/corpus/`
- Create: `eval/questions.json`
- Create: `scripts/evaluate_retrieval.py`
- Create: `tests/eval/test_evaluator.py`
- Modify: `README.md`

**Interfaces:**
- Produces: recall@k and mean reciprocal rank report

- [ ] Write evaluator tests with deterministic ranked results.
- [ ] Confirm tests fail before implementation.
- [ ] Implement corpus loader and metrics.
- [ ] Document how to run fake and real-model evaluation.
- [ ] Run tests and commit.

## Verification before every PR

Run:

```bash
ruff check .
ruff format --check .
mypy src
pytest -q --cov=doc2knowledge --cov-report=term-missing
python -m build
docker build -t doc2knowledge:check .
```

Expected: every command exits successfully. Default tests must not require network access, a downloaded embedding model, or a Gemini API key.
