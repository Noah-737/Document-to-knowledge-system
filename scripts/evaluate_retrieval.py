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


def search(base_url: str, query: str, top_k: int) -> tuple[str, ...]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/search",
        data=json.dumps({"query": query, "top_k": top_k}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise RuntimeError(f"search request failed: {error}") from error
    return tuple(str(item["filename"]) for item in payload["evidence"])


def load_cases(path: Path, base_url: str, top_k: int) -> list[RankingCase]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    cases: list[RankingCase] = []
    for item in payload["questions"]:
        query = str(item["query"])
        relevant = frozenset(str(value) for value in item["relevant_filenames"])
        cases.append(
            RankingCase(
                query=query,
                relevant_ids=relevant,
                ranked_ids=search(base_url, query, top_k),
            )
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure recall@k and MRR against a running doc2knowledge service."
    )
    parser.add_argument("questions", type=Path)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()

    report = evaluate_rankings(
        load_cases(args.questions, args.base_url, args.k),
        k=args.k,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
