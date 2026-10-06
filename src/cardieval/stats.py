"""Statistical utilities for independent model evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math

import numpy as np
from scipy.stats import beta, norm, wilcoxon

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def _cluster_indices(clusters: Sequence | None, n: int) -> list[np.ndarray]:
    """Validate authoritative independent-unit identifiers in input order."""
    if clusters is None:
        return [np.asarray([i]) for i in range(n)]
    labels = np.asarray(clusters, dtype=object)
    if labels.ndim != 1 or len(labels) != n:
        raise ValueError("clusters must contain one identifier per record")
    groups: dict[object, list[int]] = {}
    for i, label in enumerate(labels):
        if label is None or (isinstance(label, str) and not label.strip()):
            raise ValueError("cluster identifiers must be non-empty")
        if isinstance(label, (float, np.floating)) and not math.isfinite(label):
            raise ValueError("cluster identifiers must be finite")
        try:
            groups.setdefault(label, []).append(i)
        except TypeError as exc:
            raise ValueError("cluster identifiers must be hashable") from exc
    if len(groups) < 2:
        raise ValueError("cluster inference requires at least two independent clusters")
    return [np.asarray(indices) for indices in groups.values()]


def binomial_accuracy_ci(y_true: Sequence, y_pred: Sequence, *,
                         confidence: float = 0.95, method: str = "wilson") -> tuple[float, float]:
    """Two-sided IID correctness interval; never use for repeated cluster rows.

    Wilson is a score interval. ``exact`` is conservative Clopper-Pearson.
    Neither interval supplies population transportability or clinical validation.
    """
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    if yt.ndim != 1 or yp.shape != yt.shape or len(yt) == 0:
        raise ValueError("accuracy inputs must be non-empty equal-length 1D arrays")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if method not in {"wilson", "exact"}:
        raise ValueError("method must be wilson or exact")
    for array in (yt, yp):
        if any(value is None for value in array):
            raise ValueError("accuracy inputs must not contain missing labels")
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise ValueError("accuracy inputs must be finite")
    n, k = len(yt), int(np.count_nonzero(yt == yp))
    alpha = 1-confidence
    if method == "exact":
        low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n-k+1))
        high = 1.0 if k == n else float(beta.ppf(1-alpha / 2, k+1, n-k))
        return low, high
    z = float(norm.ppf(1-alpha / 2))
    proportion = k/n
    denominator = 1+z*z/n
    center = (proportion+z*z/(2*n))/denominator
    radius = z*math.sqrt(proportion*(1-proportion)/n+z*z/(4*n*n))/denominator
    return max(0.0, center-radius), min(1.0, center+radius)


def bootstrap_ci(
    y_true: Sequence,
    y_pred: Sequence,
    metric: MetricFn,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    max_attempts: int | None = None,
    clusters: Sequence | None = None,
    diagnostics: dict | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap; optional clusters resample whole independent units.

    Rows remain paired. With unequal cluster sizes the metric is record-weighted
    within each draw. Coverage, boundary behavior, and rare-class validity must
    be assessed for the estimand; this is not a universal nominal-coverage guarantee.
    """
    if n_resamples < 100:
        raise ValueError("n_resamples must be >= 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if max_attempts is not None and max_attempts < n_resamples:
        raise ValueError("max_attempts must be >= n_resamples")
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if len(yt) == 0 or len(yt) != len(yp):
        raise ValueError("paired inputs must have equal, non-zero length")
    rng = np.random.default_rng(seed)
    n = len(yt)
    groups = _cluster_indices(clusters, n)
    attempts_limit = max_attempts if max_attempts is not None else n_resamples * 20
    values: list[float] = []
    attempts = 0
    while len(values) < n_resamples and attempts < attempts_limit:
        attempts += 1
        if clusters is None:
            idx = rng.integers(0, n, size=n)
        else:
            selected = rng.integers(0, len(groups), size=len(groups))
            idx = np.concatenate([groups[i] for i in selected])
        try:
            value = float(metric(yt[idx], yp[idx]))
        except (ValueError, FloatingPointError):
            continue
        if math.isfinite(value):
            values.append(value)
    if len(values) < max(100, int(n_resamples * 0.8)):
        raise ValueError("insufficient valid bootstrap resamples for this metric")
    samples = np.asarray(values)
    if diagnostics is not None:
        diagnostics.update({"attempted_resamples": attempts, "valid_resamples": len(values),
                            "rejected_resamples": attempts-len(values), "requested_resamples": n_resamples,
                            "seed": seed})
    alpha = (1 - confidence) / 2
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


def paired_permutation_pvalue(
    y_true: Sequence,
    pred_a: Sequence,
    pred_b: Sequence,
    metric: MetricFn,
    *,
    n_resamples: int = 5000,
    seed: int = 0,
    clusters: Sequence | None = None,
) -> float:
    """Two-sided paired exchangeability test, swapping whole clusters if given.

    Up to 12 independent units are enumerated exactly. All assignments must
    define the metric; conditioning on valid assignments is not an exact test.
    Larger designs use Monte Carlo with the plus-one correction.
    """
    if n_resamples < 100:
        raise ValueError("n_resamples must be >= 100")
    yt = np.asarray(y_true)
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    if not (len(yt) == len(a) == len(b) and len(yt) > 1):
        raise ValueError("all paired inputs must have the same length >= 2")
    if a.shape != b.shape:
        raise ValueError("paired predictions must have identical shapes")
    groups = _cluster_indices(clusters, len(yt))
    unit_index = np.empty(len(yt), dtype=int)
    for i, indices in enumerate(groups):
        unit_index[indices] = i

    observed = float(metric(yt, a) - metric(yt, b))
    if not math.isfinite(observed):
        raise ValueError("observed metric difference must be finite")
    rng = np.random.default_rng(seed)
    extreme = 0
    valid = 0
    exact = len(groups) <= 12
    attempts = 2**len(groups) if exact else n_resamples
    for draw in range(attempts):
        if exact:
            unit_swap = np.asarray([(draw >> i) & 1 for i in range(len(groups))], dtype=bool)
        else:
            unit_swap = rng.integers(0, 2, size=len(groups), dtype=np.int8).astype(bool)
        swap = unit_swap[unit_index].reshape((len(yt),) + (1,) * (a.ndim - 1))
        x = np.where(swap, b, a)
        y = np.where(swap, a, b)
        try:
            difference = float(metric(yt, x) - metric(yt, y))
        except (ValueError, FloatingPointError):
            continue
        if math.isfinite(difference):
            valid += 1
            extreme += int(abs(difference) >= abs(observed))
    if valid != attempts:
        raise ValueError("all randomization assignments must produce a finite metric")
    if exact:
        return float(extreme / valid)
    if valid < max(100, int(n_resamples * 0.8)):
        raise ValueError("insufficient valid permutations for this metric")
    return float((extreme + 1) / (valid + 1))


def wilcoxon_pvalue(sample_a: Sequence[float], sample_b: Sequence[float]) -> float:
    """Paired Wilcoxon signed-rank test for matched per-sample scores/losses."""
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b) or len(a) < 2:
        raise ValueError("Wilcoxon inputs must be equal-length 1D arrays with >= 2 values")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Wilcoxon inputs must be finite")
    differences = a - b
    if np.all(differences == 0):
        return 1.0
    result = wilcoxon(a, b, alternative="two-sided")
    if result.pvalue is None or not math.isfinite(float(result.pvalue)):
        raise ValueError("Wilcoxon test did not produce a finite p-value")
    return float(result.pvalue)
