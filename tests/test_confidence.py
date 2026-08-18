import numpy as np

from cardieval.confidence import paired_difference_ci
from cardieval.metrics import accuracy


def test_paired_difference_ci_is_reproducible():
    y = np.array([0, 1, 1, 0, 1, 0])
    a = np.array([0, 1, 1, 0, 1, 1])
    b = np.array([0, 1, 0, 0, 1, 0])
    first = paired_difference_ci(y, a, b, accuracy, n_resamples=100, seed=7)
    second = paired_difference_ci(y, a, b, accuracy, n_resamples=100, seed=7)
    assert first == second
    assert first[0] > 0
    assert first[1] <= first[0] <= first[2]
