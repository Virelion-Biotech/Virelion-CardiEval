"""Probability calibration metrics for classification submissions."""

from __future__ import annotations

import numpy as np


def brier_score(y_true, score) -> float:
    """Binary Brier score; lower is better."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(score, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) == 0 or len(y) != len(p):
        raise ValueError("y_true and score must be non-empty 1-D arrays of equal length")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probability scores must be finite and in [0, 1]")
    labels = np.unique(y)
    if not np.all(np.isin(labels, [0.0, 1.0])):
        raise ValueError("Brier score currently supports binary labels 0/1")
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y_true, score, *, n_bins: int = 10) -> float:
    """Equal-width expected calibration error (ECE); lower is better."""
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(score, dtype=float)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("y_true and score must have equal non-zero length")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probability scores must be in [0, 1]")
    total = len(y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        left, right = edges[i], edges[i + 1]
        mask = (p >= left) & (p <= right) if i == n_bins - 1 else (p >= left) & (p < right)
        if not np.any(mask):
            continue
        ece += np.sum(mask) / total * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
    return float(ece)
