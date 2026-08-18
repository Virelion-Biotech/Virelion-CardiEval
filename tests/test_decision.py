from cardieval.decision import decide_comparison, evaluate_release_gates


def test_superiority_with_adjusted_significance():
    result = decide_comparison(
        metric="auroc",
        direction="higher_is_better",
        observed_difference=0.08,
        ci_low=0.03,
        ci_high=0.12,
        adjusted_pvalue=0.01,
    )
    assert result.decision == "superior"


def test_lower_is_better_is_oriented_correctly():
    result = decide_comparison(
        metric="mae",
        direction="lower_is_better",
        observed_difference=-0.08,
        ci_low=-0.12,
        ci_high=-0.03,
        adjusted_pvalue=0.01,
    )
    assert result.decision == "superior"
    assert result.observed_difference == 0.08


def test_failed_release_gate_on_integrity_error():
    result = evaluate_release_gates(
        report_ok=True,
        verification_errors=1,
        subgroup_warnings=0,
        required_primary_metric=True,
    )
    assert not result.passed
    assert any(g.name == "artifact_integrity" and not g.passed for g in result.gates)
