# doc2knowledge

A local-first document ingestion, semantic retrieval, and evidence-grounded question-answering service.

## Architecture

- FastAPI application packaged under `src/doc2knowledge`
- SQLite document and chunk metadata
- one persistent, thread-safe FAISS collection across all documents
- `mixedbread-ai/mxbai-embed-large-v1` document/query embeddings
- configurable `gemma-4-31b-it` generation through the Google Gen AI API
- explicit citations resolved to document, chunk, page, and section metadata
- bounded worker-thread processing for extraction, embedding, and indexing
- structured request logs and caller-visible request IDs

## Requirements

- Python 3.11+
- Docker, optional
- enough local memory to load the embedding model for real ingestion/search
- a `GEMINI_API_KEY` created in Google AI Studio for `/query`

## Local development

```bash
make install
make check
make run
```

The API starts on `http://localhost:8000`.

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

`/health` confirms the process is serving requests. `/ready` reports configured models, embedding dimensions, processing capacity, indexed-vector count, and whether generation is configured without exposing secrets or forcing a model download.

## Docker

```bash
docker build -t doc2knowledge:local .
docker run --rm \
  -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -v "$HOME/.cache/huggingface:/home/app/.cache/huggingface" \
  -e GEMINI_API_KEY \
  doc2knowledge:local
```

The embedding model is loaded lazily on the first ingestion or search request. Persist the Hugging Face cache between container runs to avoid repeated downloads.

## API workflow

### Upload a document

```bash
curl -X POST http://localhost:8000/documents \
  -F 'file=@notes.pdf'
```

Supported initial formats are PDF with extractable text, HTML, plain text, and Markdown. Uploads are read up to the configured byte limit, content-hashed, deduplicated, chunked, embedded, and persisted. Concurrent identical uploads atomically resolve to the same document.

### Search evidence

```bash
curl -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"What does the document say about revenue?","top_k":6}'
```

`/search` does not require a Gemini API key. It returns ranked chunks with similarity scores and source metadata. An optional `document_ids` list restricts retrieval to selected documents.

### Generate a cited answer

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What does the document say about revenue?","top_k":6}'
```

The generation prompt contains only retrieved evidence. Non-abstaining answers must cite valid source labels such as `[S1]`; invalid, invented, or uncited answers are rejected.

### List and delete documents

```bash
curl http://localhost:8000/documents
curl http://localhost:8000/documents/<document-id>
curl -X DELETE http://localhost:8000/documents/<document-id>
```

Deletion removes the original file, metadata, chunks, and vectors.

## Request tracing

Every successful response includes an `X-Request-ID`. A safe caller-supplied ID is preserved; malformed values are replaced. Request-completion logs are emitted as JSON with request ID, method, path, status, and duration.

```bash
curl -i http://localhost:8000/health \
  -H 'X-Request-ID: local-debug-123'
```

## Configuration

Application settings use the `DOC2KNOWLEDGE_` prefix unless noted otherwise.

| Variable | Default |
|---|---|
| `DOC2KNOWLEDGE_DATA_DIR` | `./data` |
| `DOC2KNOWLEDGE_EMBEDDING_MODEL` | `mixedbread-ai/mxbai-embed-large-v1` |
| `DOC2KNOWLEDGE_EMBEDDING_DIMENSIONS` | `1024` |
| `DOC2KNOWLEDGE_LLM_MODEL` | `gemma-4-31b-it` |
| `DOC2KNOWLEDGE_TOP_K` | `6` |
| `DOC2KNOWLEDGE_MAX_UPLOAD_BYTES` | `20971520` |
| `DOC2KNOWLEDGE_PROCESSING_WORKERS` | `2` |
| `GEMINI_API_KEY` | unset |

The default test suite injects fake embedding and generation services. CI does not download models or call external APIs.

## Retrieval evaluation

1. Start the service and ingest the benchmark documents.
2. Copy `eval/questions.example.json` to `eval/questions.json`.
3. Replace the example questions and relevant filenames with the benchmark set. Filenames should be stable and unique within the benchmark corpus.
4. Run:

```bash
python scripts/evaluate_retrieval.py eval/questions.json --k 6
```

The command calls `/search` and prints:

- `recall_at_k`: fraction of relevant source documents represented in the first `k` retrieved chunks
- `mean_reciprocal_rank`: reciprocal rank of the first relevant chunk, cut off at `k`
- `mean_unique_sources_at_k`: average number of distinct source documents represented in the first `k` chunks

The diversity metric makes repeated chunks from one source visible instead of silently treating a redundant result list as broad retrieval. Keep the corpus and questions versioned so embedding, chunking, hybrid retrieval, and reranking changes can be compared against the same baseline.

## Development checks

```bash
ruff check .
ruff format --check .
mypy src
pytest -q --cov=doc2knowledge --cov-report=term-missing
python -m build
docker build -t doc2knowledge:check .
```

## Design and implementation plan

- `docs/superpowers/specs/2026-07-11-document-to-knowledge-modernization-design.md`
- `docs/superpowers/plans/2026-07-11-document-to-knowledge-modernization.md`
