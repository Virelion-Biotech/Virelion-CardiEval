"""Robustness summaries for subgroup and perturbation evaluation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


class RobustnessSummary:
    """Compact numerical summary of subgroup metric spread."""

    def __init__(self, metric: str, values: Mapping[str, float], direction: str) -> None:
        if not values:
            raise ValueError("at least one subgroup value is required")
        if direction not in {"higher_is_better", "lower_is_better"}:
            raise ValueError("invalid metric direction")
        if any(not np.isfinite(value) for value in values.values()):
            raise ValueError("subgroup values must be finite")
        self.metric = metric
        self.values = dict(values)
        self.direction = direction
        ordered = np.asarray(list(self.values.values()), dtype=float)
        self.best = float(np.max(ordered) if direction == "higher_is_better" else np.min(ordered))
        self.worst = float(np.min(ordered) if direction == "higher_is_better" else np.max(ordered))
        self.range = float(self.best - self.worst)
        self.mean = float(np.mean(ordered))
        self.std = float(np.std(ordered, ddof=0))

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "direction": self.direction,
            "values": self.values,
            "best": self.best,
            "worst": self.worst,
            "range": self.range,
            "mean": self.mean,
            "std": self.std,
        }


def subgroup_robustness(values: Mapping[str, float], *, metric: str, direction: str) -> RobustnessSummary:
    """Summarize worst/best performance and spread across declared subgroups."""
    return RobustnessSummary(metric=metric, values=values, direction=direction)


def relative_drop(reference: float, perturbed: float, *, direction: str) -> float:
    """Return fractional degradation of a perturbed score relative to a reference."""
    if not np.isfinite(reference) or not np.isfinite(perturbed):
        raise ValueError("scores must be finite")
    if reference == 0:
        return 0.0 if perturbed == 0 else float("inf")
    if direction == "higher_is_better":
        return float((reference - perturbed) / abs(reference))
    if direction == "lower_is_better":
        return float((perturbed - reference) / abs(reference))
    raise ValueError("direction must be higher_is_better or lower_is_better")
