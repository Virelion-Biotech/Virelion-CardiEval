"""Threshold-dependent diagnostic metrics for binary evaluation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

from .metrics import _arrays


def _binary_counts(y_true: Sequence, y_pred: Sequence) -> tuple[int, int, int, int]:
    a, b = _arrays(y_true, y_pred)
    labels = set(np.unique(a).tolist()) | set(np.unique(b).tolist())
    if not labels.issubset({0, 1}):
        raise ValueError("diagnostic binary metrics require labels 0/1")
    tp = int(np.sum((a == 1) & (b == 1)))
    tn = int(np.sum((a == 0) & (b == 0)))
    fp = int(np.sum((a == 0) & (b == 1)))
    fn = int(np.sum((a == 1) & (b == 0)))
    return tp, tn, fp, fn


def sensitivity(y_true: Sequence, y_pred: Sequence) -> float:
    tp, _, _, fn = _binary_counts(y_true, y_pred)
    denominator = tp + fn
    return float(tp / denominator) if denominator else float("nan")


def specificity(y_true: Sequence, y_pred: Sequence) -> float:
    _, tn, fp, _ = _binary_counts(y_true, y_pred)
    denominator = tn + fp
    return float(tn / denominator) if denominator else float("nan")


def positive_predictive_value(y_true: Sequence, y_pred: Sequence) -> float:
    tp, _, fp, _ = _binary_counts(y_true, y_pred)
    denominator = tp + fp
    return float(tp / denominator) if denominator else float("nan")


def negative_predictive_value(y_true: Sequence, y_pred: Sequence) -> float:
    _, tn, _, fn = _binary_counts(y_true, y_pred)
    denominator = tn + fn
    return float(tn / denominator) if denominator else float("nan")


def matthews_correlation(y_true: Sequence, y_pred: Sequence) -> float:
    a, b = _arrays(y_true, y_pred)
    if not set(np.unique(a).tolist()).issubset({0, 1}) or not set(np.unique(b).tolist()).issubset({0, 1}):
        raise ValueError("MCC currently supports binary labels 0/1")
    return float(matthews_corrcoef(a, b))


def cohen_kappa(y_true: Sequence, y_pred: Sequence) -> float:
    a, b = _arrays(y_true, y_pred)
    if not set(np.unique(a).tolist()).issubset({0, 1}) or not set(np.unique(b).tolist()).issubset({0, 1}):
        raise ValueError("Cohen kappa currently supports binary labels 0/1")
    return float(cohen_kappa_score(a, b))


def confusion_matrix_counts(y_true: Sequence, y_pred: Sequence) -> dict[str, int]:
    tp, tn, fp, fn = _binary_counts(y_true, y_pred)
    return {"true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn}
