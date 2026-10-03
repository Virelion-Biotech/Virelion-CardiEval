"""Multiple-comparison correction utilities for CardiEval."""

from __future__ import annotations

from collections.abc import Sequence
import math


def _validated(pvalues: Sequence[float]) -> list[float]:
    values = [float(p) for p in pvalues]
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in values):
        raise ValueError("p-values must be finite numbers between 0 and 1")
    return values


def bonferroni(pvalues: Sequence[float]) -> list[float]:
    """Bonferroni-adjust p-values while preserving input order."""
    values = _validated(pvalues)
    m = len(values)
    if m == 0:
        return []
    return [min(1.0, p * m) for p in values]


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg false-discovery-rate adjusted p-values."""
    values = _validated(pvalues)
    m = len(values)
    if m == 0:
        return []
    order = sorted(range(m), key=values.__getitem__)
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        index = order[rank - 1]
        adjusted_value = min(running, values[index] * m / rank)
        running = adjusted_value
        adjusted[index] = adjusted_value
    return adjusted
