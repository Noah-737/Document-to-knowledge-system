import pytest

from doc2knowledge.evaluation import RankingCase, evaluate_rankings


def test_evaluation_reports_recall_at_k_and_mrr() -> None:
    report = evaluate_rankings(
        [
            RankingCase(
                query="alpha",
                relevant_ids=frozenset({"a", "b"}),
                ranked_ids=("a", "x", "b"),
            ),
            RankingCase(
                query="beta",
                relevant_ids=frozenset({"c"}),
                ranked_ids=("x", "c", "y"),
            ),
        ],
        k=2,
    )

    assert report.query_count == 2
    assert report.k == 2
    assert report.recall_at_k == pytest.approx(0.5)
    assert report.mean_reciprocal_rank == pytest.approx(0.75)


def test_evaluation_handles_no_results_and_validates_inputs() -> None:
    report = evaluate_rankings(
        [RankingCase("missing", frozenset({"a"}), ())],
        k=5,
    )

    assert report.recall_at_k == 0.0
    assert report.mean_reciprocal_rank == 0.0

    with pytest.raises(ValueError, match="at least one"):
        evaluate_rankings([], k=1)
    with pytest.raises(ValueError, match="positive"):
        evaluate_rankings([RankingCase("q", frozenset({"a"}), ())], k=0)
    with pytest.raises(ValueError, match="relevant"):
        evaluate_rankings([RankingCase("q", frozenset(), ())], k=1)
