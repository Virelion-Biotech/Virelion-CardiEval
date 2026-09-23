"""Metrics with explicit validation and stable return types."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def _arrays(y_true: Sequence, y_pred: Sequence) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y_true)
    b = np.asarray(y_pred)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b) or len(a) == 0:
        raise ValueError("y_true and y_pred must be non-empty 1D arrays of equal length")
    for name, array in (("y_true", a), ("y_pred", b)):
        try:
            numeric = array.astype(float)
        except (TypeError, ValueError):
            continue
        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"{name} must contain only finite values")
    return a, b


def _score_array(score: Sequence[float]) -> np.ndarray:
    s = np.asarray(score, dtype=float)
    if s.ndim != 1 or len(s) == 0:
        raise ValueError("score must be a non-empty 1D array")
    if not np.all(np.isfinite(s)):
        raise ValueError("score must contain only finite values")
    return s


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    a, b = _arrays(y_true, y_pred)
    return float(accuracy_score(a, b))


def balanced_accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    a, b = _arrays(y_true, y_pred)
    return float(balanced_accuracy_score(a, b))


def macro_f1(y_true: Sequence, y_pred: Sequence) -> float:
    a, b = _arrays(y_true, y_pred)
    return float(f1_score(a, b, average="macro", zero_division=0))


def auroc(y_true: Sequence, score: Sequence[float]) -> float:
    a, _ = _arrays(y_true, y_true)
    s = _score_array(score)
    if len(a) != len(s):
        raise ValueError("y_true and score must have equal length")
    if len(np.unique(a)) < 2:
        raise ValueError("AUROC requires at least two observed classes")
    return float(roc_auc_score(a, s))


def auprc(y_true: Sequence, score: Sequence[float]) -> float:
    a, _ = _arrays(y_true, y_true)
    s = _score_array(score)
    if len(a) != len(s):
        raise ValueError("y_true and score must have equal length")
    if len(np.unique(a)) < 2:
        raise ValueError("AUPRC requires at least two observed classes")
    return float(average_precision_score(a, s))


def brier(y_true: Sequence, score: Sequence[float]) -> float:
    a, _ = _arrays(y_true, y_true)
    s = _score_array(score)
    if len(a) != len(s):
        raise ValueError("y_true and score must have equal length")
    if not np.all(np.isin(np.unique(a), [0, 1])):
        raise ValueError("Brier score currently supports binary labels 0/1")
    if np.any((s < 0) | (s > 1)):
        raise ValueError("probability scores must be in [0, 1]")
    return float(brier_score_loss(a.astype(float), s.astype(float)))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    a, b = _arrays(y_true, y_pred)
    return float(mean_absolute_error(a, b))


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    a, b = _arrays(y_true, y_pred)
    return float(np.sqrt(mean_squared_error(a, b)))


METRIC_DIRECTIONS = {
    "accuracy": "higher_is_better",
    "balanced_accuracy": "higher_is_better",
    "macro_f1": "higher_is_better",
    "auroc": "higher_is_better",
    "auprc": "higher_is_better",
    "sensitivity": "higher_is_better",
    "specificity": "higher_is_better",
    "positive_predictive_value": "higher_is_better",
    "negative_predictive_value": "higher_is_better",
    "matthews_correlation": "higher_is_better",
    "cohen_kappa": "higher_is_better",
    "brier": "lower_is_better",
    "ece": "lower_is_better",
    "mae": "lower_is_better",
    "rmse": "lower_is_better",
    "mrr": "higher_is_better",
    "hit_rate@10": "higher_is_better",
    "ndcg@10": "higher_is_better",
}
