import json

import pytest

from cardieval.atlas_ingest import (
    build_clinical_evidence_report,
    load_atlas_clinical_trials,
    write_clinical_evidence_report,
)


def _bundle(tmp_path):
    product = {
        "record_type": "therapeutic_product",
        "id": "product-cardiamp",
        "name": "CardiAMP",
        "evidence_ids": ["evidence-outcome"],
    }
    evidence = {
        "record_type": "evidence",
        "id": "evidence-outcome",
        "source_type": "clinical",
        "source_identifier": "Sponsor presentation",
        "citation": "CardiAMP trial results presentation",
        "source_url": "https://example.org/results",
        "context": {
            "clinical_trial_evidence": {
                "trial_id": "NCT00000001",
                "product_id": "product-cardiamp",
                "phase": "Pivotal",
                "design": "Randomized, double-blind, sham-controlled",
                "comparator": "sham",
                "randomised": True,
                "blinded": True,
                "source_type": "company_report",
                "source_verified": True,
                "endpoints": [
                    {
                        "endpoint_name": "Primary composite",
                        "role": "primary",
                        "population": "intention_to_treat",
                        "status": "not_met",
                    },
                    {
                        "endpoint_name": "Biomarker subgroup",
                        "role": "exploratory",
                        "population": "subgroup",
                        "status": "inconclusive",
                        "prespecified": None,
                        "multiplicity_adjusted": None,
                    },
                ],
                "limitations": ["Subgroup interaction not reported"],
            }
        },
    }
    path = tmp_path / "atlas.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in (product, evidence)) + "\n", encoding="utf-8")
    return path


def test_atlas_ingestion_preserves_primary_failure_and_unknown_subgroup(tmp_path):
    trials = load_atlas_clinical_trials(_bundle(tmp_path))
    assert len(trials) == 1
    trial = trials[0]
    assert trial.primary_endpoint_status == "primary_endpoint_not_met"
    assert trial.source_type == "company_report"
    assert trial.source_ids == ("evidence-outcome",)
    assert any("prespecification status unknown" in item for item in trial.subgroup_caveats)
    assert any("interaction test not reported" in item for item in trial.subgroup_caveats)


def test_atlas_ingestion_rejects_evidence_not_linked_from_product(tmp_path):
    path = _bundle(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["evidence_ids"] = []
    path.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not linked from product"):
        load_atlas_clinical_trials(path)


def test_report_does_not_infer_primary_success_from_subgroup(tmp_path):
    report = build_clinical_evidence_report(load_atlas_clinical_trials(_bundle(tmp_path)))
    trial = report["trials"][0]
    assert trial["primary_endpoint_status"] == "primary_endpoint_not_met"
    assert trial["interpretation_policy"] if "interpretation_policy" in trial else True
    assert report["interpretation_policy"]["subgroup_results_do_not_override_primary_endpoint"]


def test_cli_report_writer_is_json_serializable(tmp_path):
    output = tmp_path / "clinical-report.json"
    report = write_clinical_evidence_report(_bundle(tmp_path), output)
    saved = json.loads(output.read_text())
    assert saved["trial_count"] == 1
    assert report["trials"][0]["primary_endpoint_status"] == "primary_endpoint_not_met"
