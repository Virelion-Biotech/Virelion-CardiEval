import numpy as np
import pytest

from cardieval.calibration import brier_score, expected_calibration_error


def test_brier_score_known_values() -> None:
    assert brier_score([0, 1], [0.0, 1.0]) == 0.0


def test_ece_perfectly_calibrated_two_bins() -> None:
    assert expected_calibration_error([0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0], n_bins=2) == 0.0


def test_probability_range_is_enforced() -> None:
    with pytest.raises(ValueError):
        brier_score(np.array([0, 1]), np.array([-0.1, 1.1]))


def test_log_score_known_probability_and_unstable_slope():
    from cardieval.calibration import binary_log_score, calibration_intercept_slope

    assert abs(binary_log_score([0, 1], [0.5, 0.5]) - np.log(2)) < 1e-12
    with pytest.raises(ValueError):
        calibration_intercept_slope([0, 1], [0.5, 0.5])
    with pytest.raises(ValueError):
        calibration_intercept_slope([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])


def test_logistic_calibration_known_nonseparated_population():
    from cardieval.calibration import calibration_intercept_slope

    # At probabilities .2 and .8 the empirical frequencies exactly match.
    result = calibration_intercept_slope(
        [0] * 8 + [1] * 2 + [0] * 2 + [1] * 8, [0.2] * 10 + [0.8] * 10
    )
    assert result["intercept"] == pytest.approx(0, abs=1e-6)
    assert result["slope"] == pytest.approx(1, abs=1e-6)
