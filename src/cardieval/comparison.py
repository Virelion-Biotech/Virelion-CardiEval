"""Reproducible paired model comparison utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .metrics import METRIC_DIRECTIONS
from .models import ModelComparison
from .stats import paired_permutation_pvalue, wilcoxon_pvalue


def compare_predictions(
    y_true: Sequence,
    pred_a: Sequence,
    pred_b: Sequence,
    metric: Callable,
    *,
    metric_name: str,
    samplewise_score: Callable | None = None,
    n_resamples: int = 5000,
    seed: int = 0,
) -> ModelComparison:
    """Compare two models on identical samples with a paired permutation test.

    ``samplewise_score`` is optional. It must return one scalar loss/score per
    sample and is required when a paired Wilcoxon test is desired. This avoids
    incorrectly applying Wilcoxon to metrics such as AUROC or macro-F1 that are
    not additive across individual observations.
    """
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
    p_wilcoxon = None
    if samplewise_score is not None:
        per_a = np.asarray(samplewise_score(yt, a), dtype=float)
        per_b = np.asarray(samplewise_score(yt, b), dtype=float)
        if per_a.shape != per_b.shape or per_a.shape != yt.shape:
            raise ValueError("samplewise_score must return one value per sample")
        p_wilcoxon = wilcoxon_pvalue(per_a, per_b)
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
