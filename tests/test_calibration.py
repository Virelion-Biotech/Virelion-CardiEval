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
