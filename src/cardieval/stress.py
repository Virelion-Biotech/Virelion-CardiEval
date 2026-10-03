"""Distribution-shift and stress-test summaries."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class StressResult:
    """Performance change between a reference and stress condition."""

    metric: str
    reference: float
    stressed: float
    delta: float
    relative_change: float | None
    direction: str
    degradation: float


def compare_stress(
    metric: str,
    reference: float,
    stressed: float,
    *,
    direction: str,
) -> StressResult:
    """Quantify degradation under a stress condition.

    Degradation is positive when the stressed score is worse than the reference.
    """
    if direction not in {"higher_is_better", "lower_is_better"}:
        raise ValueError("direction must be higher_is_better or lower_is_better")
    if not math.isfinite(reference) or not math.isfinite(stressed):
        raise ValueError("stress scores must be finite")
    if reference == 0:
        relative = None
    else:
        relative = (stressed - reference) / abs(reference)
    if direction == "higher_is_better":
        degradation = reference - stressed
    else:
        degradation = stressed - reference
    return StressResult(
        metric=metric,
        reference=float(reference),
        stressed=float(stressed),
        delta=float(stressed - reference),
        relative_change=relative,
        direction=direction,
        degradation=float(degradation),
    )


def aggregate_stress(results: Sequence[StressResult]) -> float:
    """Return mean degradation across stress conditions for one metric contract."""
    if not results:
        raise ValueError("at least one stress result is required")
    first = results[0]
    if any(item.metric != first.metric for item in results):
        raise ValueError("all stress results must use the same metric")
    if any(item.direction != first.direction for item in results):
        raise ValueError("all stress results must use the same metric direction")
    return float(sum(item.degradation for item in results) / len(results))
