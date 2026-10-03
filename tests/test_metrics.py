import warnings

import pytest

from cardieval.metrics import accuracy, balanced_accuracy, macro_f1, mae, rmse
from cardieval.stats import bootstrap_ci


def test_classification_metrics():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]
    assert accuracy(y_true, y_pred) == pytest.approx(0.75)
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(0.75)
    assert macro_f1(y_true, y_pred) == pytest.approx((0.8 + 2 / 3) / 2)


def test_regression_metrics():
    assert mae([1, 2, 4], [1, 4, 1]) == pytest.approx(5 / 3)
    assert rmse([1, 2, 4], [1, 4, 1]) == pytest.approx((13 / 3) ** 0.5)


def test_metrics_reject_non_finite_inputs():
    with pytest.raises(ValueError, match="finite"):
        accuracy([0, 1], [0, float("nan")])
    with pytest.raises(ValueError, match="finite"):
        mae([0, float("inf")], [0, 1])



def test_balanced_accuracy_rejects_single_observed_class():
    with pytest.raises(ValueError, match="at least two observed classes"):
        balanced_accuracy([0, 0, 0], [0, 0, 0])


def test_balanced_accuracy_rejects_prediction_class_absent_from_truth():
    with pytest.raises(ValueError, match="not observed"):
        balanced_accuracy([0, 1, 0, 1], [0, 2, 0, 1])



def test_balanced_accuracy_bootstrap_rejects_invalid_resamples_without_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        low, high = bootstrap_ci(
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            balanced_accuracy,
            n_resamples=200,
            seed=42,
        )
    assert 0.0 <= low <= high <= 1.0
    assert caught == []
