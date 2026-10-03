import math
import warnings

import pytest

from cardieval.diagnostics import (
    cohen_kappa,
    confusion_matrix_counts,
    matthews_correlation,
    negative_predictive_value,
    positive_predictive_value,
    sensitivity,
    specificity,
)

from cardieval.stats import bootstrap_ci


def test_binary_diagnostic_metrics():
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 0, 0, 1]
    assert sensitivity(y_true, y_pred) == 2 / 3
    assert specificity(y_true, y_pred) == 2 / 3
    assert positive_predictive_value(y_true, y_pred) == 2 / 3
    assert negative_predictive_value(y_true, y_pred) == 2 / 3
    assert 0 <= matthews_correlation(y_true, y_pred) <= 1
    assert 0 <= cohen_kappa(y_true, y_pred) <= 1
    assert confusion_matrix_counts(y_true, y_pred) == {
        "true_positive": 2,
        "true_negative": 2,
        "false_positive": 1,
        "false_negative": 1,
    }


def test_undefined_predictive_value_returns_nan():
    assert math.isnan(positive_predictive_value([0, 0], [0, 0]))



def test_cohen_kappa_rejects_single_class_truth():
    with pytest.raises(ValueError, match="at least two observed truth classes"):
        cohen_kappa([0, 0, 0], [0, 0, 0])


def test_cohen_kappa_bootstrap_rejects_invalid_resamples_without_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        low, high = bootstrap_ci(
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            cohen_kappa,
            n_resamples=200,
            seed=42,
        )
    assert -1.0 <= low <= high <= 1.0
    assert caught == []



def test_mcc_rejects_single_class_truth():
    with pytest.raises(ValueError, match="at least two observed truth classes"):
        matthews_correlation([0, 0, 0], [0, 0, 0])


def test_mcc_bootstrap_rejects_invalid_resamples_without_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        low, high = bootstrap_ci(
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            matthews_correlation,
            n_resamples=200,
            seed=42,
        )
    assert -1.0 <= low <= high <= 1.0
    assert caught == []
