"""Statistical utilities for independent model evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import wilcoxon

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def bootstrap_ci(
    y_true: Sequence,
    y_pred: Sequence,
    metric: MetricFn,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    max_attempts: int | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI with rejection of invalid resamples.

    Rejection matters for metrics such as AUROC that are undefined when a
    bootstrap sample contains only one class.
    """
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
    max_attempts = max_attempts or n_resamples * 20
    values: list[float] = []
    attempts = 0
    while len(values) < n_resamples and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        try:
            value = float(metric(yt[idx], yp[idx]))
        except (ValueError, FloatingPointError):
            continue
        if np.isfinite(value):
            values.append(value)
    if len(values) < max(100, int(n_resamples * 0.8)):
        raise ValueError("insufficient valid bootstrap resamples for this metric")
    samples = np.asarray(values)
    alpha = (1 - confidence) / 2
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


def paired_permutation_pvalue(
    y_true: Sequence,
    pred_a: Sequence,
    pred_b: Sequence,
    metric: MetricFn,
    *,
    n_resamples: int = 5000,
    seed: int = 0,
) -> float:
    """Two-sided paired randomization test for model-vs-model performance."""
    if n_resamples < 100:
        raise ValueError("n_resamples must be >= 100")
    yt = np.asarray(y_true)
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    if not (len(yt) == len(a) == len(b) and len(yt) > 1):
        raise ValueError("all paired inputs must have the same non-zero length")

    observed = metric(yt, a) - metric(yt, b)
    rng = np.random.default_rng(seed)
    extreme = 0
    valid = 0
    for _ in range(n_resamples):
        swap = rng.integers(0, 2, size=len(yt), dtype=np.int8).astype(bool)
        x = np.where(swap, b, a)
        y = np.where(swap, a, b)
        try:
            difference = metric(yt, x) - metric(yt, y)
        except (ValueError, FloatingPointError):
            continue
        if np.isfinite(difference):
            valid += 1
            extreme += int(abs(difference) >= abs(observed))
    if valid < max(100, int(n_resamples * 0.8)):
        raise ValueError("insufficient valid permutations for this metric")
    return float((extreme + 1) / (valid + 1))


def wilcoxon_pvalue(sample_a: Sequence[float], sample_b: Sequence[float]) -> float:
    """Paired Wilcoxon signed-rank test for matched per-sample scores/losses."""
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("Wilcoxon inputs must have equal length >= 2")
    return float(wilcoxon(a, b, alternative="two-sided").pvalue)
