"""Reproducible paired model comparison utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .metrics import METRIC_DIRECTIONS
from .models import ModelComparison
from .stats import paired_permutation_pvalue, wilcoxon_pvalue


def compare_predictions(
    y_true: Sequence,
    pred_a: Sequence,
    pred_b: Sequence,
    metric,
    *,
    metric_name: str,
    n_resamples: int = 5000,
    seed: int = 0,
) -> ModelComparison:
    """Compare two models on identical samples with paired resampling tests."""
    yt = np.asarray(y_true)
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    if not (len(yt) == len(a) == len(b) and len(yt) > 1):
        raise ValueError("all comparison inputs must have the same length >= 2")
    score_a = float(metric(yt, a))
    score_b = float(metric(yt, b))
    difference = score_a - score_b
    p_perm = paired_permutation_pvalue(
        yt, a, b, metric, n_resamples=n_resamples, seed=seed
    )
    per_sample_a = np.asarray([metric(np.asarray([y]), np.asarray([p])) for y, p in zip(yt, a)])
    per_sample_b = np.asarray([metric(np.asarray([y]), np.asarray([p])) for y, p in zip(yt, b)])
    try:
        p_wilcoxon = wilcoxon_pvalue(per_sample_a, per_sample_b)
    except ValueError:
        p_wilcoxon = None
    direction = METRIC_DIRECTIONS.get(metric_name, "informational")
    if direction == "higher_is_better":
        winner = "model_a" if difference > 0 else "model_b" if difference < 0 else "tie"
    elif direction == "lower_is_better":
        winner = "model_a" if difference < 0 else "model_b" if difference > 0 else "tie"
    else:
        winner = "undetermined"
    return ModelComparison(
        metric=metric_name,
        model_a_score=score_a,
        model_b_score=score_b,
        difference=difference,
        permutation_pvalue=p_perm,
        wilcoxon_pvalue=p_wilcoxon,
        winner=winner,
        n=len(yt),
    )
