"""Confidence-aware comparison helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .stats import bootstrap_ci


def paired_difference_ci(
    y_true: Sequence,
    pred_a: Sequence,
    pred_b: Sequence,
    metric,
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
    clusters: Sequence | None = None,
) -> tuple[float, float, float]:
    """Return observed paired metric difference and a bootstrap CI."""
    y = np.asarray(y_true)
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    if not (len(y) == len(a) == len(b) and len(y) > 1):
        raise ValueError("all paired inputs must have the same length >= 2")
    if a.shape != b.shape:
        raise ValueError("paired predictions must have identical shapes")
    observed = float(metric(y, a) - metric(y, b))

    def difference_metric(y_boot: np.ndarray, packed: np.ndarray) -> float:
        return float(metric(y_boot, packed[:, 0, ...]) - metric(y_boot, packed[:, 1, ...]))

    packed = np.stack((a, b), axis=1)
    low, high = bootstrap_ci(y, packed, difference_metric, confidence=confidence, n_resamples=n_resamples, seed=seed, clusters=clusters)
    return observed, low, high
