# Development Guide

## Prerequisites

- Python 3.11 or newer
- `make`
- Docker, only for the container build check

The default test suite uses fake embedding and generation services. It does not download a
model, call an external API, or require `GEMINI_API_KEY`.

## Setup

```bash
git clone https://github.com/Noah-737/Document-to-knowledge-system.git
cd Document-to-knowledge-system
make install
```

## Run locally

```bash
make run
```

The API listens on `http://localhost:8000`. Check it with:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

Set `GEMINI_API_KEY` before using `/query`. Ingestion and `/search` work without it.

## Development checks

Run individual checks while iterating:

```bash
make lint
make typecheck
make test
make build
make docker-build
```

Run the complete CI-equivalent suite before committing:

```bash
make check
```

`make check` runs Ruff lint and format checks, strict mypy, pytest with branch coverage and
an 85% minimum, a Python package build, and a Docker image build.

To apply Ruff's safe fixes and formatting:

```bash
make format
```

## Retrieval evaluation

Copy the example question set, replace it with questions for a stable ingested corpus, start
the service, and run the evaluator:

```bash
cp eval/questions.example.json eval/questions.json
python scripts/evaluate_retrieval.py eval/questions.json --k 6
```

See `README.md` for the API workflow, configuration, Docker usage, and metric definitions.
