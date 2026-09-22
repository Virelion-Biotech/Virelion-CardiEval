"""Calibration-curve summaries for probabilistic binary predictions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class CalibrationBin(BaseModel):
    """One reliability-curve bin."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    lower: float = Field(ge=0, le=1)
    upper: float = Field(ge=0, le=1)
    n: int = Field(ge=0)
    mean_predicted: float
    observed_rate: float

    def model_dump(self, *args, **kwargs):
        return super().model_dump(*args, **kwargs)


def calibration_curve(
    y_true: Sequence[int],
    score: Sequence[float],
    *,
    n_bins: int = 10,
) -> list[CalibrationBin]:
    """Return equal-width reliability bins for binary probabilities."""
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(score, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("y_true and score must be non-empty 1D arrays of equal length")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)):
        raise ValueError("y_true and score must contain only finite values")
    if not np.all(np.isin(np.unique(y), [0.0, 1.0])):
        raise ValueError("calibration_curve currently supports binary labels 0/1")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probability scores must be finite and in [0, 1]")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[CalibrationBin] = []
    for i in range(n_bins):
        lower, upper = edges[i], edges[i + 1]
        mask = (p >= lower) & ((p < upper) if i < n_bins - 1 else (p <= upper))
        n = int(mask.sum())
        bins.append(
            CalibrationBin(
                lower=float(lower),
                upper=float(upper),
                n=n,
                mean_predicted=float(p[mask].mean()) if n else 0.0,
                observed_rate=float(y[mask].mean()) if n else 0.0,
            )
        )
    return bins
