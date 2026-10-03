"""Ranking metrics for ordered cardiac challenge retrieval tasks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


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
    """Reciprocal rank of the first relevant item, using relevance > 0."""
    relevance, scores = _arrays(y_true, score)
    order = np.argsort(-scores, kind="stable")
    hits = relevance[order] > 0
    if not np.any(hits):
        return 0.0
    return float(1.0 / (int(np.flatnonzero(hits)[0]) + 1))


def hit_rate_at_k(y_true: Sequence[float | int], score: Sequence[float], k: int = 10) -> float:
    """Whether any relevant item is present in the top-k results."""
    if k < 1:
        raise ValueError("k must be >= 1")
    relevance, scores = _arrays(y_true, score)
    order = np.argsort(-scores, kind="stable")[:k]
    return float(np.any(relevance[order] > 0))


def ndcg_at_k(y_true: Sequence[float | int], score: Sequence[float], k: int = 10) -> float:
    """Normalized discounted cumulative gain at k."""
    if k < 1:
        raise ValueError("k must be >= 1")
    relevance, scores = _arrays(y_true, score)
    pred_order = np.argsort(-scores, kind="stable")[:k]
    gains = relevance[pred_order]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(relevance)[::-1][:k]
    idcg = float(np.sum(ideal * discounts[: len(ideal)]))
    return 0.0 if idcg == 0 else dcg / idcg
