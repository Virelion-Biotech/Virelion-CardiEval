"""Probability calibration metrics for classification submissions."""

from __future__ import annotations

import numpy as np


def _validate_binary_inputs(y_true, score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(score, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) == 0 or len(y) != len(p):
        raise ValueError("y_true and score must be non-empty 1-D arrays of equal length")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)):
        raise ValueError("y_true and score must contain only finite values")
    if not np.all(np.isin(np.unique(y), [0.0, 1.0])):
        raise ValueError("calibration metrics currently support binary labels 0/1")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probability scores must be finite and in [0, 1]")
    return y, p


def brier_score(y_true, score) -> float:
    """Binary Brier score; lower is better."""
    y, p = _validate_binary_inputs(y_true, score)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y_true, score, *, n_bins: int = 10) -> float:
    """Equal-width expected calibration error (ECE); lower is better."""
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    y, p = _validate_binary_inputs(y_true, score)
    total = len(y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        left, right = edges[i], edges[i + 1]
        mask = (p >= left) & (p <= right) if i == n_bins - 1 else (p >= left) & (p < right)
        if np.any(mask):
            ece += np.sum(mask) / total * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
    return float(ece)


def binary_log_score(y_true, score):
    """Mean negative log likelihood, with explicit machine-precision endpoint clipping."""
    y, p = _validate_binary_inputs(y_true, score)
    epsilon = np.finfo(float).eps
    q = np.clip(p, epsilon, 1 - epsilon)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log1p(-q)))


def calibration_intercept_slope(y_true, score):
    """Unpenalized logistic recalibration diagnostics; reject separation/nonidentifiability.

    These are assessment coefficients on held-out predictions, not instructions
    to refit the prediction model on the validation cohort.
    """
    from scipy.optimize import minimize
    from scipy.special import expit

    y, p = _validate_binary_inputs(y_true, score)
    if len(np.unique(y)) < 2 or np.any((p <= 0) | (p >= 1)):
        raise ValueError(
            "Calibration slope requires two classes and probabilities strictly inside (0,1)"
        )
    logits = np.log(p) - np.log1p(-p)
    if np.ptp(logits) < 1e-10:
        raise ValueError("Calibration slope is not identifiable for constant scores")
    negative, positive = logits[y == 0], logits[y == 1]
    if max(negative) <= min(positive) or max(positive) <= min(negative):
        raise ValueError("Calibration coefficients are separated or quasi-separated")
    design = np.column_stack([np.ones(len(y)), logits])

    def objective(beta):
        eta = design @ beta
        return float(np.sum(np.logaddexp(0, eta) - y * eta))

    def gradient(beta):
        return design.T @ (expit(design @ beta) - y)

    fit = minimize(objective, [0.0, 1.0], jac=gradient, method="BFGS", options={"gtol": 1e-7})
    mu = expit(design @ fit.x)
    hessian = design.T @ ((mu * (1 - mu))[:, None] * design)
    if (
        not fit.success
        or not np.isfinite(fit.x).all()
        or np.linalg.cond(hessian) > 1e10
        or np.max(np.abs(fit.x)) > 30
    ):
        raise ValueError("Calibration coefficients are unstable or separated")
    return {
        "intercept": float(fit.x[0]),
        "slope": float(fit.x[1]),
        "n_records": len(y),
        "interpretation": "held-out diagnostic coefficients; biological-unit uncertainty requires clustered resampling",
    }
