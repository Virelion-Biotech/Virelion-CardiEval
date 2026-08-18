import numpy as np
import pytest

from cardieval.ranking import hit_rate_at_k, ndcg_at_k, reciprocal_rank


def test_reciprocal_rank_finds_first_relevant_item():
    assert reciprocal_rank([0, 1, 0], [0.9, 0.8, 0.7]) == pytest.approx(0.5)


def test_hit_rate_at_k():
    assert hit_rate_at_k([0, 0, 1], [0.9, 0.8, 0.7], k=2) == 0.0
    assert hit_rate_at_k([0, 0, 1], [0.9, 0.8, 0.7], k=3) == 1.0


def test_ndcg_is_one_for_ideal_ranking():
    relevance = [3, 2, 0, 1]
    score = [0.99, 0.8, 0.1, 0.7]
    assert ndcg_at_k(relevance, score, k=4) == pytest.approx(1.0)


def test_ranking_rejects_negative_relevance():
    with pytest.raises(ValueError):
        reciprocal_rank([-1, 0], [0.2, 0.1])


def test_ndcg_is_finite():
    result = ndcg_at_k(np.array([0, 1, 2]), np.array([0.2, 0.9, 0.3]))
    assert np.isfinite(result)
