import pytest

from cardieval.clinical_evidence import ClinicalEndpointResult, ClinicalTrialEvidence


def test_primary_endpoint_failure_is_not_overridden_by_positive_subgroup():
    trial = ClinicalTrialEvidence(
        trial_id="trial-1", product_id="cardiac-cell-product", phase="III",
        design="randomised double-blind placebo-controlled", comparator="placebo",
        randomised=True, blinded=True, source_ids=("registry-1",),
        source_type="company_report",
        endpoints=(
            ClinicalEndpointResult(
                endpoint_name="prespecified composite", role="primary",
                population="intention_to_treat", status="not_met", denominator=200,
            ),
            ClinicalEndpointResult(
                endpoint_name="biomarker subgroup", role="exploratory",
                population="subgroup", status="met", denominator=70,
                prespecified=False, multiplicity_adjusted=False,
            ),
        ),
    )
    assert trial.primary_endpoint_status == "primary_endpoint_not_met"
    assert len(trial.subgroup_caveats) == 2


def test_subgroup_requires_prespecification_status():
    with pytest.raises(ValueError, match="prespecified"):
        ClinicalEndpointResult(
            endpoint_name="subgroup", role="secondary", population="subgroup",
            status="met",
        )


def test_invalid_pvalue_rejected():
    with pytest.raises(ValueError, match="between 0 and 1"):
        ClinicalEndpointResult(
            endpoint_name="safety", role="safety", population="safety",
            status="met", p_value=1.5,
        )


def test_trial_requires_sources_and_endpoints():
    with pytest.raises(ValueError, match="at least one endpoint"):
        ClinicalTrialEvidence(
            trial_id="t", product_id="p", phase="II", design="open-label",
            comparator=None, randomised=False, blinded=None, endpoints=(),
            source_ids=(), source_type="other",
        )
