import pytest

from doc2knowledge.evaluation import RankingCase, evaluate_rankings


def test_evaluation_reports_recall_mrr_and_source_diversity_at_k() -> None:
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
    assert report.recall_at_k == pytest.approx(0.75)
    assert report.mean_reciprocal_rank == pytest.approx(0.75)
    assert report.mean_unique_sources_at_k == pytest.approx(2.0)


def test_mrr_is_cut_off_at_k() -> None:
    report = evaluate_rankings(
        [RankingCase("alpha", frozenset({"a"}), ("x", "y", "a"))],
        k=2,
    )

    assert report.recall_at_k == 0.0
    assert report.mean_reciprocal_rank == 0.0
    assert report.mean_unique_sources_at_k == 2.0


def test_source_diversity_exposes_repeated_chunks_from_one_document() -> None:
    report = evaluate_rankings(
        [RankingCase("alpha", frozenset({"a"}), ("x", "x", "a"))],
        k=3,
    )

    assert report.recall_at_k == 1.0
    assert report.mean_reciprocal_rank == pytest.approx(1 / 3)
    assert report.mean_unique_sources_at_k == 2.0


def test_evaluation_handles_no_results_and_validates_inputs() -> None:
    report = evaluate_rankings(
        [RankingCase("missing", frozenset({"a"}), ())],
        k=5,
    )

    assert report.recall_at_k == 0.0
    assert report.mean_reciprocal_rank == 0.0
    assert report.mean_unique_sources_at_k == 0.0

    with pytest.raises(ValueError, match="at least one"):
        evaluate_rankings([], k=1)
    with pytest.raises(ValueError, match="positive"):
        evaluate_rankings([RankingCase("q", frozenset({"a"}), ())], k=0)
    with pytest.raises(ValueError, match="non-empty query"):
        evaluate_rankings([RankingCase(" ", frozenset({"a"}), ())], k=1)
    with pytest.raises(ValueError, match="at least one relevant"):
        evaluate_rankings([RankingCase("q", frozenset(), ())], k=1)
    with pytest.raises(ValueError, match="Relevant IDs|relevant IDs"):
        evaluate_rankings([RankingCase("q", frozenset({""}), ())], k=1)
    with pytest.raises(ValueError, match="Ranked IDs|ranked IDs"):
        evaluate_rankings([RankingCase("q", frozenset({"a"}), ("",))], k=1)
