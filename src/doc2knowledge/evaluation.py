from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RankingCase:
    query: str
    relevant_ids: frozenset[str]
    ranked_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationReport:
    query_count: int
    k: int
    recall_at_k: float
    mean_reciprocal_rank: float


def evaluate_rankings(cases: list[RankingCase], *, k: int) -> EvaluationReport:
    if not cases:
        raise ValueError("at least one ranking case is required")
    if k < 1:
        raise ValueError("k must be positive")
    if any(not case.relevant_ids for case in cases):
        raise ValueError("every ranking case must contain at least one relevant ID")

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    for case in cases:
        top_k = case.ranked_ids[:k]
        retrieved_relevant = len(case.relevant_ids.intersection(top_k))
        recalls.append(retrieved_relevant / len(case.relevant_ids))

        reciprocal_rank = 0.0
        for rank, candidate in enumerate(case.ranked_ids, start=1):
            if candidate in case.relevant_ids:
                reciprocal_rank = 1.0 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)

    return EvaluationReport(
        query_count=len(cases),
        k=k,
        recall_at_k=sum(recalls) / len(recalls),
        mean_reciprocal_rank=sum(reciprocal_ranks) / len(reciprocal_ranks),
    )
