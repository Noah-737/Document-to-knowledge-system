#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

from doc2knowledge.evaluation import RankingCase, evaluate_rankings


def _parse_k(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 50:
        raise argparse.ArgumentTypeError("k must be between 1 and 50")
    return parsed


def search(base_url: str, query: str, top_k: int) -> tuple[str, ...]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/search",
        data=json.dumps({"query": query, "top_k": top_k}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload: Any = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"search request failed with HTTP {error.code}: {body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"search request failed: {error}") from error

    if not isinstance(payload, dict) or not isinstance(payload.get("evidence"), list):
        raise RuntimeError("search response did not contain an evidence list")
    filenames: list[str] = []
    for item in payload["evidence"]:
        if not isinstance(item, dict) or not isinstance(item.get("filename"), str):
            raise RuntimeError("search evidence item did not contain a filename")
        filenames.append(item["filename"])
    return tuple(filenames)


def load_cases(path: Path, base_url: str, top_k: int) -> list[RankingCase]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("questions"), list):
        raise ValueError("benchmark file must contain a questions list")
    if not payload["questions"]:
        raise ValueError("benchmark questions list must not be empty")

    cases: list[RankingCase] = []
    for item in payload["questions"]:
        if not isinstance(item, dict):
            raise ValueError("every benchmark question must be an object")
        query = item.get("query")
        relevant_values = item.get("relevant_filenames")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("every benchmark question must contain a non-empty query")
        if not isinstance(relevant_values, list) or not relevant_values:
            raise ValueError("every benchmark question must contain relevant_filenames")
        if any(not isinstance(value, str) or not value.strip() for value in relevant_values):
            raise ValueError("relevant_filenames must contain non-empty strings")

        cases.append(
            RankingCase(
                query=query.strip(),
                relevant_ids=frozenset(value.strip() for value in relevant_values),
                ranked_ids=search(base_url, query.strip(), top_k),
            )
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure document recall@k, MRR@k, and source diversity against a running "
            "doc2knowledge service."
        )
    )
    parser.add_argument("questions", type=Path)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--k", type=_parse_k, default=6)
    args = parser.parse_args()

    report = evaluate_rankings(
        load_cases(args.questions, args.base_url, args.k),
        k=args.k,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
