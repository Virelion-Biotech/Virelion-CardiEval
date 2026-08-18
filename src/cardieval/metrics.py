"""Metrics with explicit validation and stable return types."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
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
    return a, b


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
    a, s = _arrays(y_true, score)
    if len(np.unique(a)) < 2:
        raise ValueError("AUROC requires at least two observed classes")
    return float(roc_auc_score(a, s))


def auprc(y_true: Sequence, score: Sequence[float]) -> float:
    a, s = _arrays(y_true, score)
    if len(np.unique(a)) < 2:
        raise ValueError("AUPRC requires at least two observed classes")
    return float(average_precision_score(a, s))


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
    "mae": "lower_is_better",
    "rmse": "lower_is_better",
}
