import pytest

from cardieval.calibration_curves import calibration_curve
from cardieval.stress import aggregate_stress, compare_stress


def test_calibration_curve_has_expected_bins():
    y_true = [0, 0, 1, 1, 0, 1]
    scores = [0.1, 0.2, 0.6, 0.8, 0.3, 0.7]
    bins = calibration_curve(y_true, scores, n_bins=3)
    assert len(bins) == 3
    assert all(hasattr(b, "mean_predicted") and hasattr(b, "observed_rate") for b in bins)


def test_stress_degradation_respects_metric_direction():
    result = compare_stress("auroc", 0.9, 0.75, direction="higher_is_better")
    assert result.degradation == pytest.approx(0.15)
    assert result.delta < 0

    result_low = compare_stress("brier", 0.1, 0.25, direction="lower_is_better")
    assert result_low.degradation == pytest.approx(0.15)
    assert result_low.delta > 0


def test_stress_aggregation_requires_one_metric_contract():
    first = compare_stress("auroc", 0.9, 0.8, direction="higher_is_better")
    second = compare_stress("auroc", 0.9, 0.7, direction="higher_is_better")
    assert aggregate_stress([first, second]) == pytest.approx(0.15)


def test_stress_aggregation_rejects_mixed_metrics():
    first = compare_stress("auroc", 0.9, 0.8, direction="higher_is_better")
    second = compare_stress("brier", 0.1, 0.2, direction="lower_is_better")
    with pytest.raises(ValueError, match="same metric"):
        aggregate_stress([first, second])


def test_stress_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="finite"):
        compare_stress("auroc", float("nan"), 0.8, direction="higher_is_better")
