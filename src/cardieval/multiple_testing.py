"""Multiple-comparison correction utilities for CardiEval."""

from __future__ import annotations

from collections.abc import Sequence


def bonferroni(pvalues: Sequence[float]) -> list[float]:
    """Bonferroni-adjust p-values while preserving input order."""
    values = [float(p) for p in pvalues]
    m = len(values)
    if m == 0:
        return []
    if any(p < 0 or p > 1 for p in values):
        raise ValueError("p-values must be between 0 and 1")
    return [min(1.0, p * m) for p in values]


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg false-discovery-rate adjusted p-values."""
    values = [float(p) for p in pvalues]
    m = len(values)
    if m == 0:
        return []
    if any(p < 0 or p > 1 for p in values):
        raise ValueError("p-values must be between 0 and 1")
    order = sorted(range(m), key=values.__getitem__)
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        index = order[rank - 1]
        adjusted_value = min(running, values[index] * m / rank)
        running = adjusted_value
        adjusted[index] = adjusted_value
    return adjusted
