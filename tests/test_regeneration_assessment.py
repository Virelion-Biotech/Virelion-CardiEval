import pytest

from cardieval.regeneration_assessment import (
    DOMAIN_ORDER, RegenerationAssessment, RegenerationDomainEvidence,
)


SOURCE = "https://doi.org/10.1038/s44161-026-00910-7"


def domain(name, status, **kwargs):
    sources = () if status in ("not_assessed", "not_reported") else (SOURCE,)
    return RegenerationDomainEvidence(domain=name, status=status, source_ids=sources, **kwargs)


def assessment(overrides=None):
    values = {
        "cell_cycle_entry": domain("cell_cycle_entry", "direct_support", assay="EdU/Ki67", sample_size=12, denominator=12),
        "mitosis_cytokinesis": domain("mitosis_cytokinesis", "direct_support", assay="pHH3/Aurora B", sample_size=12, denominator=12),
        "cardiomyocyte_number_lineage": domain("cardiomyocyte_number_lineage", "direct_support", assay="total CM count and CycleTrack", sample_size=12, denominator=12),
        "functional_maturation": domain("functional_maturation", "direct_support", assay="sarcomere/calcium/electrophysiology/contractility"),
        "ventricular_function_remodelling": domain("ventricular_function_remodelling", "direct_support", assay="echocardiography", sample_size=12, denominator=12),
        "fibrosis_scar": domain("fibrosis_scar", "no_support", assay="histology", sample_size=12, denominator=12),
        "arrhythmia_safety": domain("arrhythmia_safety", "direct_support", assay="telemetry", follow_up="12 weeks", sample_size=12, denominator=12),
        "off_target_toxicity": domain("off_target_toxicity", "direct_support", assay="organ histology", follow_up="12 weeks", sample_size=12, denominator=12),
    }
    if overrides:
        values.update(overrides)
    return RegenerationAssessment(
        assessment_id="assessment:test", title="Test", model="MI mouse", species="mouse",
        source_ids=(SOURCE,), domains=tuple(values[name] for name in DOMAIN_ORDER),
    )


@pytest.mark.parametrize("name", ["ventricular_function_remodelling", "fibrosis_scar"])
def test_lvef_or_scar_only_never_meets_regeneration(name):
    values = {key: domain(key, "not_assessed") for key in DOMAIN_ORDER}
    values[name] = domain(name, "direct_support", assay="echo or scar histology")
    assert assessment(values).interpretation == "repair_or_preservation_only"


def test_kI67_or_dna_synthesis_only_is_insufficient():
    values = {key: domain(key, "not_assessed") for key in DOMAIN_ORDER}
    values["cell_cycle_entry"] = domain("cell_cycle_entry", "surrogate_support", assay="Ki67 or EdU")
    assert assessment(values).interpretation == "insufficient_evidence"


def test_incomplete_maturation_and_unassessed_safety_yield_mixed_evidence():
    a = assessment({
        "functional_maturation": domain("functional_maturation", "not_assessed"),
        "arrhythmia_safety": domain("arrhythmia_safety", "not_assessed"),
    })
    assert a.interpretation == "mixed_evidence"


def test_complete_core_evidence_can_be_demonstrated_in_model():
    assert assessment().interpretation == "regeneration_demonstrated_in_model"


def test_small_subgroups_and_unadjusted_nominal_pvalues_are_flagged():
    a = assessment({
        "ventricular_function_remodelling": domain(
            "ventricular_function_remodelling", "direct_support",
            assay="echo", sample_size=6, denominator=6, subgroup=True,
            prespecified=False, multiplicity_adjusted=False, p_value=0.02,
        )
    })
    caveats = a.caveats()
    assert any("small sample" in item for item in caveats)
    assert any("not prespecified" in item for item in caveats)
    assert any("multiplicity was not adjusted" in item for item in caveats)
    assert any("nominal p-value" in item for item in caveats)


def test_missing_denominator_and_incomplete_safety_followup_are_reported():
    a = assessment({
        "cell_cycle_entry": domain("cell_cycle_entry", "direct_support", assay="Ki67"),
        "arrhythmia_safety": domain("arrhythmia_safety", "not_assessed"),
    })
    caveats = a.caveats()
    assert any("denominator not reported" in item for item in caveats)
    assert any("safety follow-up duration not reported" in item for item in caveats)


def test_duplicate_domains_rejected():
    a = assessment()
    with pytest.raises(ValueError, match="exactly once"):
        RegenerationAssessment(
            assessment_id="x", title="x", model="x", species="mouse",
            source_ids=(SOURCE,), domains=a.domains[:-1] + (a.domains[0],),
        )
