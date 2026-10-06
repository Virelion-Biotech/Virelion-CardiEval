"""Ranking metrics for ordered cardiac challenge retrieval tasks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from math import comb


def _arrays(y_true: Sequence[float | int], score: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    relevance = np.asarray(y_true, dtype=float)
    scores = np.asarray(score, dtype=float)
    if relevance.ndim != 1 or scores.ndim != 1 or len(relevance) != len(scores) or len(relevance) == 0:
        raise ValueError("relevance and score must be non-empty 1D arrays of equal length")
    if not np.all(np.isfinite(relevance)) or not np.all(np.isfinite(scores)):
        raise ValueError("ranking inputs must be finite")
    if np.any(relevance < 0):
        raise ValueError("relevance labels must be non-negative")
    return relevance, scores


def reciprocal_rank(y_true: Sequence[float | int], score: Sequence[float]) -> float:
    """Expected reciprocal rank of the first hit under uniform score-tie order."""
    relevance, scores = _arrays(y_true, score)
    offset = 0
    for score_value in sorted(set(scores), reverse=True):
        block = relevance[scores == score_value]
        size, hits = len(block), int(np.sum(block > 0))
        if hits:
            return float(sum(comb(size-j, hits-1) / comb(size, hits) / (offset+j)
                             for j in range(1, size-hits+2)))
        offset += size
    return 0.0


def hit_rate_at_k(y_true: Sequence[float | int], score: Sequence[float], k: int = 10) -> float:
    """Expected top-k hit indicator under uniform ordering of equal scores."""
    if k < 1:
        raise ValueError("k must be >= 1")
    relevance, scores = _arrays(y_true, score)
    remaining = k
    for score_value in sorted(set(scores), reverse=True):
        block = relevance[scores == score_value]
        size, hits = len(block), int(np.sum(block > 0))
        take = min(remaining, size)
        if hits:
            return float(1-comb(size-hits, take)/comb(size, take))
        remaining -= take
        if remaining <= 0:
            break
    return 0.0


def ndcg_at_k(y_true: Sequence[float | int], score: Sequence[float], k: int = 10) -> float:
    """Linear-gain NDCG at k, averaging gain across equal-score ties."""
    if k < 1:
        raise ValueError("k must be >= 1")
    relevance, scores = _arrays(y_true, score)
    count = min(k, len(scores))
    discounts = 1.0 / np.log2(np.arange(2, count + 2))
    dcg = 0.0
    offset = 0
    for score_value in sorted(set(scores), reverse=True):
        block = relevance[scores == score_value]
        take = min(len(block), count-offset)
        dcg += float(np.mean(block)*np.sum(discounts[offset:offset+take]))
        offset += take
        if offset == count:
            break
    ideal = np.sort(relevance)[::-1][:k]
    idcg = float(np.sum(ideal * discounts[: len(ideal)]))
    return 0.0 if idcg == 0 else dcg / idcg
