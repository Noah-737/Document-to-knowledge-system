#!/bin/bash
set -e

echo "--- Installing dependencies ---"
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"

echo "--- Running linter ---"
ruff check .
ruff format --check .

echo "--- Running tests ---"
pytest -q
