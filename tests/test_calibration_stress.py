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
