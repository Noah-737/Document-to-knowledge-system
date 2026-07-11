.PHONY: install run test lint format typecheck build docker-build check

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

run:
	python -m doc2knowledge

test:
	pytest -q --cov=doc2knowledge --cov-report=term-missing

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

typecheck:
	mypy src

build:
	python -m build

docker-build:
	docker build -t doc2knowledge:local .

check: lint typecheck test build docker-build
