# Document-to-Knowledge Modernization Design

**Date:** 2026-07-11

## Summary

The repository currently demonstrates a narrow upload-to-embedding flow: FastAPI accepts PDF, HTML, and TXT files, extracts text, chunks it, creates embeddings, and saves one FAISS index per filename. It does not yet provide cross-document retrieval, document metadata, citations, RAG answer generation, reliable packaging, meaningful tests, or a deployable container.

This design evolves the prototype into a small but dependable document research service through a sequence of reviewable pull requests. Each PR must leave the repository runnable and testable. Model integrations are configuration-driven, but the intended defaults are:

- Embeddings: `mixedbread-ai/mxbai-embed-large-v1`
- Generation: `gemma-4-31b-it` through the Gemini API using an API key created in Google AI Studio

## Goals

1. Make installation, packaging, local execution, Docker execution, and CI consistent.
2. Ingest supported documents into stable document and chunk records.
3. Preserve source metadata required for retrieval, deletion, reindexing, and citations.
4. Use one managed vector collection instead of one unnamed index per filename.
5. Provide semantic retrieval across all indexed documents.
6. Generate evidence-grounded answers with Gemma 4 31B and explicit citations.
7. Fail visibly when extraction, embedding, persistence, retrieval, or generation fails.
8. Add tests that validate real application behavior rather than a constant truth assertion.
9. Keep the system suitable for a single-node development deployment while preserving replaceable interfaces for later production infrastructure.

## Non-goals for the initial modernization

- Multi-tenant authorization
- Distributed vector infrastructure
- A production web frontend
- Large-scale job orchestration
- Fine-tuning either model
- Agentic web browsing or external search
- Supporting every office and media format in the first release

## Architectural approach

Use a modular monolith with explicit interfaces. This is the best fit for the current repository size: it avoids premature services while keeping the expensive or replaceable components isolated.

```text
FastAPI application
├── API routes
│   ├── health
│   ├── documents
│   └── query
├── Configuration
├── Domain models
├── Ingestion
│   ├── validation
│   ├── extraction
│   ├── chunking
│   └── metadata creation
├── Embeddings
│   └── mxbai embedding service
├── Storage
│   ├── document metadata repository
│   └── vector repository
├── Retrieval
│   └── semantic search and result assembly
└── Generation
    └── Gemma Gemini API client and citation-grounded prompt
```

The application code will live under `src/doc2knowledge/`. Runtime state will be stored under a configurable data directory and excluded from Git.

## Core component boundaries

### Configuration

A single settings object reads environment variables and supplies typed defaults. Required secrets must be validated only by the components that need them, so ingestion and retrieval can run without a Gemini API key.

Initial settings:

- `DOC2KNOWLEDGE_DATA_DIR=./data`
- `DOC2KNOWLEDGE_EMBEDDING_MODEL=mixedbread-ai/mxbai-embed-large-v1`
- `DOC2KNOWLEDGE_EMBEDDING_DIMENSIONS=1024`
- `DOC2KNOWLEDGE_LLM_MODEL=gemma-4-31b-it`
- `GEMINI_API_KEY` with no default
- `DOC2KNOWLEDGE_TOP_K=6`
- `DOC2KNOWLEDGE_MAX_UPLOAD_BYTES=20971520`

### Domain models

The domain layer defines stable types independent of FastAPI, FAISS, Sentence Transformers, and Google SDK response objects.

Minimum records:

- `Document`: ID, filename, media type, SHA-256 hash, status, created time, chunk count, error message
- `Chunk`: chunk ID, document ID, ordinal, text, source page when available, section when available
- `RetrievedChunk`: chunk metadata plus similarity score
- `Answer`: generated text plus citation records

Document IDs are UUIDs. The content hash detects duplicate uploads. Filenames are display metadata and must never be used as storage identity.

### Ingestion

The ingestion service validates size and media type, writes the original file to a document-specific path, extracts text, creates chunks, embeds them, persists vectors and metadata, then marks the document ready.

Initial supported formats:

- PDF with text extraction
- HTML
- Plain text
- Markdown

Scanned-PDF OCR is deferred to a later PR but must have a distinct `extraction_failed` state rather than silently creating an empty index.

### Chunking

Chunking remains deterministic and configurable. The first implementation may use recursive character splitting, but chunk records must retain ordinal and source metadata. Empty chunks are rejected.

Chunking configuration is stored alongside an index manifest so a future reindex can identify incompatible settings.

### Embeddings

The embedding interface exposes separate document and query methods:

```python
class EmbeddingService(Protocol):
    @property
    def dimensions(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, query: str) -> list[float]: ...
```

The default implementation loads `mixedbread-ai/mxbai-embed-large-v1` once and reuses it. Document text is encoded without a prompt. Queries use the model's retrieval prompt through `prompt_name="query"` or the equivalent exact prefix. Embeddings are normalized before cosine-similarity search.

The model must not be loaded during module import. It is initialized through an application dependency so unit tests can inject a deterministic fake.

### Metadata storage

SQLite stores document and chunk metadata for the first production-capable iteration. It provides transactions, explicit schema, querying, duplicate detection, deletion, and restart persistence without adding infrastructure.

The storage interface keeps SQLite replaceable later.

### Vector storage

The first vector implementation uses one persistent FAISS index for all chunks plus an explicit mapping from FAISS row IDs to chunk IDs. Index updates use a write lock and atomic replacement of the persisted index and mapping files.

The interface supports:

- add chunks
- search
- delete/rebuild by document
- load after restart
- report embedding model and dimensionality compatibility

A model or dimensionality change requires reindexing; the service must refuse to mix incompatible vectors.

### Retrieval

Retrieval accepts a query and optional document IDs, embeds the query, searches the vector repository, loads chunk metadata, and returns ranked `RetrievedChunk` records.

The first implementation is dense-only retrieval. Hybrid BM25 and reranking are later improvements after a retrieval evaluation corpus exists.

### Generation

The generator interface accepts the user question and retrieved chunks and returns an `Answer`.

The default implementation uses the current `google-genai` SDK with model `gemma-4-31b-it`. The prompt requires the model to:

- answer only from supplied context
- cite chunk labels such as `[S1]`
- state that evidence is insufficient when context does not support an answer
- avoid inventing sources

The API returns structured citations resolved from labels to document metadata. Retrieval remains usable when no Gemini API key is configured; only answer generation fails with a clear configuration error.

### API

Initial endpoints:

- `GET /health`: process and storage health
- `POST /documents`: upload one document and return its record
- `GET /documents`: list document records
- `GET /documents/{document_id}`: status and metadata
- `DELETE /documents/{document_id}`: delete metadata, source file, chunks, and vectors
- `POST /search`: return ranked chunks without generation
- `POST /query`: retrieve evidence and generate a cited answer

Large CPU-bound work must not run directly on the event loop. The initial single-node version may use FastAPI thread-pool execution. Background jobs are introduced only when processing-status semantics are implemented.

## Error handling

- Unsupported media type: HTTP 415
- Upload exceeds configured maximum: HTTP 413
- Duplicate content: return the existing document with a duplicate indicator
- Extraction yields no usable text: document status `failed`, HTTP response includes a stable error code
- Embedding model unavailable: document status `failed`, no partial vector commit
- Vector and metadata persistence failure: transaction is rolled back or compensating cleanup runs
- Missing Gemini API key: search still works; `/query` returns HTTP 503 with `generation_not_configured`
- Gemini API timeout or rate limit: HTTP 503 with retryable error metadata
- No sufficiently relevant evidence: return an answer indicating insufficient evidence with no fabricated citations

Errors are logged with document or request IDs and never include secret values.

## Testing strategy

Every code PR follows test-first development.

Required layers:

1. Unit tests for settings, extractors, chunking, embedding adapters, repositories, retrieval, prompts, and citation parsing.
2. API tests using FastAPI's test client and injected fake embedding/generation services.
3. Persistence tests that restart repositories from a temporary directory.
4. Integration tests for a small real embedding model path may be marked separately so normal CI remains deterministic.
5. CI checks formatting, linting, typing, unit tests, coverage threshold, package build, and Docker build.

Network calls and model downloads are forbidden in the default CI test job.

## Deployment shape

The application container runs Uvicorn as a non-root user, stores data in a mounted volume, exposes a health endpoint, and accepts configuration through environment variables. The image does not download the embedding model during build; deployment can pre-warm the model cache or mount a cache volume.

## Pull-request sequence

Each PR is independently reviewable and should be merged before the next begins.

### PR 1: Architecture and implementation plans

- Add this design document.
- Add the detailed TDD implementation plan for the foundation/CI PR.
- No runtime behavior changes.

### PR 2: Foundation, packaging, and trustworthy CI

- Move application package to `src/doc2knowledge/`.
- Consolidate runtime and development dependencies in `pyproject.toml`.
- Add typed settings and an application factory.
- Add a real `/health` test.
- Add Ruff, mypy, pytest, coverage, package-build, and Docker-build checks.
- Make the container run the API.

### PR 3: Document domain and ingestion metadata

- Add document/chunk domain models.
- Add SQLite metadata repository.
- Add stable document IDs and content hashing.
- Add validated upload and extraction behavior.
- Add document list/status/delete endpoints.

### PR 4: Mixedbread embedding service and vector repository

- Add `mixedbread-ai/mxbai-embed-large-v1` adapter.
- Load the model once.
- Correctly distinguish document and query encoding.
- Add a unified persistent FAISS collection and compatibility manifest.
- Add deterministic fake-based tests and an opt-in real-model integration test.

### PR 5: Retrieval API

- Add semantic search service.
- Add `/search` with optional document filters.
- Return ranked evidence with source metadata.
- Add a small retrieval evaluation fixture.

### PR 6: Gemma 4 31B grounded generation

- Add `google-genai` client adapter for `gemma-4-31b-it`.
- Add citation-grounded prompt construction and citation validation.
- Add `/query` and explicit insufficient-evidence behavior.
- Mock all network access in CI.

### PR 7: Processing robustness

- Add background processing state and bounded concurrency.
- Add OCR fallback for scanned PDFs.
- Add structured logging, request IDs, retry policy, and model/API metrics.

### PR 8: Retrieval quality improvements

- Establish evaluation metrics and a repeatable benchmark corpus.
- Add hybrid lexical retrieval or reranking only when measurements show a benefit.

## Acceptance criteria for the modernization

The modernization is successful when a clean checkout can:

1. install from `pyproject.toml`;
2. pass CI without network or model downloads;
3. run locally or in Docker;
4. ingest multiple documents with stable metadata;
5. restart without losing document or vector state;
6. search across all indexed documents;
7. answer with `gemma-4-31b-it` using only retrieved evidence;
8. provide resolvable citations and abstain when evidence is insufficient;
9. delete and reindex documents safely;
10. detect embedding-model incompatibility instead of silently corrupting search.
