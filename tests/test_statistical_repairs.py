"""Independent scientific counterexamples as regression tests."""
import itertools

import numpy as np
import pytest
from scipy.stats import binomtest

from cardieval.comparison import compare_predictions
from cardieval.confidence import paired_difference_ci
from cardieval.decision import decide_comparison
from cardieval.metrics import accuracy, mae
from cardieval.stats import binomial_accuracy_ci, bootstrap_ci, paired_permutation_pvalue


def test_four_patients_exact_cluster_randomization():
    labels = np.tile([0, 1], 50)
    clusters = np.repeat(np.arange(4), 25)
    assert paired_permutation_pvalue(labels, labels, 1-labels, accuracy, clusters=clusters) == 0.125


def test_packed_prediction_swaps_match_independent_exact_oracle():
    y = np.arange(3)
    a = np.column_stack((y, y+1))
    b = np.column_stack((y+2, y+4))
    def metric(labels, predictions):
        return float(np.mean((predictions-labels[:, None])**2))
    observed = abs(metric(y, a)-metric(y, b))
    extreme = 0
    for signs in itertools.product([False, True], repeat=3):
        x, z = a.copy(), b.copy()
        x[list(signs)] = b[list(signs)]
        z[list(signs)] = a[list(signs)]
        extreme += abs(metric(y, x)-metric(y, z)) >= observed
    assert paired_permutation_pvalue(y, a, b, metric) == extreme/8
    observed_ci, low, high = paired_difference_ci(y, a, b, metric, n_resamples=100)
    assert observed_ci == metric(y, a)-metric(y, b)
    assert low <= observed_ci <= high


def test_cluster_bootstrap_keeps_whole_units_and_variable_size():
    y = np.array([0, 0, 1, 1, 1])
    def metric(labels, predictions):
        # Every draw must include cluster multiplicities, never fractional units.
        assert np.count_nonzero(labels == 0) % 2 == 0
        assert np.count_nonzero(labels == 1) % 3 == 0
        assert np.array_equal(labels, predictions)
        return float(np.mean(labels))
    low, high = bootstrap_ci(y, y, metric, n_resamples=100, clusters=['a','a','b','b','b'])
    assert (low, high) == (0.0, 1.0)


@pytest.mark.parametrize('clusters', [['a']*4, ['a','b'], ['a','b',None,'c'], ['a','b',float('nan'),'c']])
def test_cluster_identifiers_fail_closed(clusters):
    with pytest.raises(ValueError, match='cluster'):
        bootstrap_ci([0,1,0,1], [0,1,0,1], accuracy, n_resamples=100, clusters=clusters)


@pytest.mark.parametrize('successes', [0, 1, 50, 99, 100])
def test_exact_accuracy_interval_matches_scipy_binomial_oracle(successes):
    y = np.zeros(100, dtype=int)
    predictions = np.ones(100, dtype=int)
    predictions[:successes] = 0
    expected = binomtest(successes, 100).proportion_ci(method='exact')
    assert binomial_accuracy_ci(y, predictions, method='exact') == pytest.approx((expected.low, expected.high), abs=1e-10)


def test_wilson_accuracy_boundary_is_non_degenerate():
    low, high = binomial_accuracy_ci([1]*100, [1]*100)
    assert low == pytest.approx(0.9630065017930143)
    assert high == pytest.approx(1.0)
    low, high = binomial_accuracy_ci([1]*100, [0]*100)
    assert low == pytest.approx(0.0, abs=1e-15)
    assert high == pytest.approx(0.0369934982069857)


def test_cluster_comparison_rejects_row_wilcoxon_and_allows_declared_aggregation():
    y = np.zeros(8)
    clusters = np.repeat(np.arange(4), 2)
    arguments = dict(metric_name='mae', clusters=clusters, samplewise_score=lambda labels, p: np.abs(labels-p))
    with pytest.raises(ValueError, match='cluster_aggregate'):
        compare_predictions(y, y, y+1, mae, **arguments)
    result = compare_predictions(y, y, y+1, mae, cluster_aggregate=np.mean, **arguments)
    assert result.permutation_pvalue == 0.125
    assert result.wilcoxon_pvalue == 0.125


def decision(**overrides):
    kwargs = dict(metric='accuracy', direction='higher_is_better', observed_difference=0.03,
                  ci_low=0.001, ci_high=0.06)
    kwargs.update(overrides)
    return decide_comparison(**kwargs)


def test_decision_confidence_is_bound_to_alpha():
    with pytest.raises(ValueError, match='confidence'):
        decision(alpha=1e-6)
    assert decision(alpha=1e-6, ci_confidence=0.999999).decision == 'superior'


def test_equality_pvalue_cannot_gate_noninferiority_margin():
    with pytest.raises(ValueError, match='equality'):
        decision(observed_difference=0, ci_low=-.01, ci_high=.01, margin=.05, adjusted_pvalue=1)
    result = decision(observed_difference=0, ci_low=-.01, ci_high=.01, margin=.05,
                      adjusted_pvalue=.01, pvalue_hypothesis='non_inferiority_margin', pvalue_margin=.05)
    assert result.decision == 'non_inferior'
    with pytest.raises(ValueError, match='matching'):
        decision(margin=.05, adjusted_pvalue=.01, pvalue_hypothesis='superiority_margin', pvalue_margin=.03)


def test_one_sided_bound_cannot_claim_opposite_direction():
    assert decision(ci_sidedness='upper').decision == 'inconclusive'
    assert decision(ci_sidedness='lower').decision == 'superior'
    result = decision(direction='lower_is_better', observed_difference=-.03,
                      ci_low=-.06, ci_high=-.001, ci_sidedness='upper')
    assert result.decision == 'superior'
    assert result.ci_sidedness == 'lower'


def test_boundary_equality_is_not_noninferiority_evidence():
    assert decision(observed_difference=0, ci_low=0, ci_high=0).decision == 'inconclusive'


def test_invalid_randomization_draws_do_not_condition_the_null():
    def metric(y, p):
        if np.array_equal(p, [0,1]):
            raise ValueError('undefined for this assignment')
        return float(np.mean(p))
    with pytest.raises(ValueError, match='all randomization'):
        paired_permutation_pvalue([0,0], [0,0], [1,1], metric)
