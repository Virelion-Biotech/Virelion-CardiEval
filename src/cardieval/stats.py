"""Statistical utilities for independent model evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import permutation_test, wilcoxon


MetricFn = Callable[[np.ndarray, np.ndarray], float]


def bootstrap_ci(
    y_true: Sequence,
    y_pred: Sequence,
    metric: MetricFn,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI. Seed is explicit for reproducibility."""
    if n_resamples < 100:
        raise ValueError("n_resamples must be >= 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if len(yt) == 0 or len(yt) != len(yp):
        raise ValueError("paired inputs must have equal, non-zero length")
    rng = np.random.default_rng(seed)
    n = len(yt)
    values = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        values[i] = metric(yt[idx], yp[idx])
    alpha = (1 - confidence) / 2
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))


def paired_permutation_pvalue(
    y_true: Sequence,
    pred_a: Sequence,
    pred_b: Sequence,
    metric: MetricFn,
    *,
    n_resamples: int = 5000,
    seed: int = 0,
) -> float:
    """Two-sided paired permutation test for a model-vs-model metric difference."""
    yt = np.asarray(y_true)
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    if not (len(yt) == len(a) == len(b) and len(yt) > 1):
        raise ValueError("all paired inputs must have the same non-zero length")

    def statistic(x: np.ndarray, y: np.ndarray, axis: int = -1) -> np.ndarray:
        if axis != -1:
            x = np.moveaxis(x, axis, -1)
            y = np.moveaxis(y, axis, -1)
        if x.ndim == 1:
            return metric(yt[: x.shape[-1]], x) - metric(yt[: y.shape[-1]], y)
        return np.array([metric(yt, xi) - metric(yt, yi) for xi, yi in zip(x, y)])

    result = permutation_test(
        (a, b),
        statistic,
        permutation_type="samples",
        n_resamples=n_resamples,
        random_state=seed,
        alternative="two-sided",
    )
    return float(result.pvalue)


def wilcoxon_pvalue(sample_a: Sequence[float], sample_b: Sequence[float]) -> float:
    """Paired Wilcoxon signed-rank test for matched per-sample losses/scores."""
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("Wilcoxon inputs must have equal length >= 2")
    return float(wilcoxon(a, b, alternative="two-sided").pvalue)
