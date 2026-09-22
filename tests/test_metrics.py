import pytest

from cardieval.metrics import accuracy, balanced_accuracy, macro_f1, mae, rmse


def test_classification_metrics():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]
    assert accuracy(y_true, y_pred) == pytest.approx(0.75)
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(0.75)
    assert macro_f1(y_true, y_pred) == pytest.approx((0.8 + 2 / 3) / 2)


def test_regression_metrics():
    assert mae([1, 2, 4], [1, 4, 1]) == pytest.approx(5 / 3)
    assert rmse([1, 2, 4], [1, 4, 1]) == pytest.approx((8 / 3) ** 0.5)


def test_metrics_reject_non_finite_inputs():
    with pytest.raises(ValueError, match="finite"):
        accuracy([0, 1], [0, float("nan")])
    with pytest.raises(ValueError, match="finite"):
        mae([0, float("inf")], [0, 1])
