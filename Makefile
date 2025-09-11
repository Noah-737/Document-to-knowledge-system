.PHONY: help install test lint docker-build

help:
	@echo "Commands:"
	@echo "  install       : Install dependencies."
	@echo "  test          : Run tests."
	@echo "  lint          : Run linter and formatter checks."
	@echo "  docker-build  : Build the Docker image."

install:
	pip install --upgrade pip
	pip install -e ".[dev]"

test:
	pytest -q tests/

lint:
	ruff check .
	ruff format --check .

docker-build:
	docker build -t doc2knowledge .
