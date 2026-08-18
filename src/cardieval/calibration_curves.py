"""Calibration-curve summaries for probabilistic binary predictions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class CalibrationBin:
    """One reliability-curve bin."""

    def __init__(self, lower: float, upper: float, n: int, mean_predicted: float, observed_rate: float) -> None:
        self.lower = float(lower)
        self.upper = float(upper)
        self.n = int(n)
        self.mean_predicted = float(mean_predicted)
        self.observed_rate = float(observed_rate)

    def model_dump(self) -> dict[str, float | int]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "n": self.n,
            "mean_predicted": self.mean_predicted,
            "observed_rate": self.observed_rate,
        }


def calibration_curve(
    y_true: Sequence[int],
    score: Sequence[float],
    *,
    n_bins: int = 10,
) -> list[CalibrationBin]:
    """Return equal-width reliability bins for binary probabilities."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(score, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("y_true and score must be non-empty 1D arrays of equal length")
    if not np.all(np.isin(np.unique(y), [0.0, 1.0])):
        raise ValueError("calibration_curve currently supports binary labels 0/1")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probability scores must be in [0, 1]")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[CalibrationBin] = []
    for i in range(n_bins):
        lower, upper = edges[i], edges[i + 1]
        mask = (p >= lower) & ((p < upper) if i < n_bins - 1 else (p <= upper))
        n = int(mask.sum())
        if n == 0:
            bins.append(CalibrationBin(lower, upper, 0, float("nan"), float("nan")))
        else:
            bins.append(CalibrationBin(lower, upper, n, float(p[mask].mean()), float(y[mask].mean())))
    return bins
