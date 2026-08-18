from cardiEval.calibration_curves import calibration_curve
from cardieval.stress import aggregate_stress, compare_stress


def test_calibration_curve_has_expected_bins():
    bins = calibration_curve([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], n_bins=4)
    assert len(bins) == 4
    assert sum(item.n for item in bins) == 4
    assert bins[0].observed_rate == 0.0
    assert bins[-1].observed_rate == 1.0


def test_stress_degradation_respects_metric_direction():
    higher = compare_stress("auroc", 0.9, 0.75, direction="higher_is_better")
    lower = compare_stress("mae", 0.2, 0.3, direction="lower_is_better")
    assert higher.degradation == 0.15
    assert lower.degradation == 0.1
    assert aggregate_stress([higher, lower]) == 0.125
