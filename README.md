# doc2knowledge

A document ingestion, retrieval, and evidence-grounded question-answering service.

The project is being modernized in staged pull requests. The current foundation provides a packaged FastAPI application, typed configuration, health endpoint, repeatable development commands, Docker runtime, and CI checks.

## Requirements

- Python 3.11+
- Docker, optional

## Local development

```bash
make install
make check
make run
```

The API starts on `http://localhost:8000`.

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok","service":"doc2knowledge","version":"0.1.0"}
```

## Docker

```bash
docker build -t doc2knowledge:local .
docker run --rm -p 8000:8000 -v "$(pwd)/data:/app/data" doc2knowledge:local
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
| `GEMINI_API_KEY` | unset |

No API key or model download is required for the foundation test suite.

## Roadmap

See:

- `docs/superpowers/specs/2026-07-11-document-to-knowledge-modernization-design.md`
- `docs/superpowers/plans/2026-07-11-document-to-knowledge-modernization.md`
