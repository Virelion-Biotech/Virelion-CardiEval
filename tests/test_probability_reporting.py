import copy
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from cardieval.benchmark_package import BenchmarkPackage
from cardieval.bundle import build_bundle
from cardieval.evaluator import evaluate_submission
from cardieval.models import BenchmarkManifest, PredictionRecord, SubgroupResult
from cardieval.publication import publish_leaderboard
from cardieval.registry import BenchmarkTask


def task(**updates):
    values = dict(
        benchmark_id="probability", version="1", task_id="calibration",
        task_type="binary_classification", splits=["test"],
        allowed_metrics=["auroc", "brier", "log_score", "ece"],
        primary_metric="log_score", primary_direction="lower_is_better",
        requires_authoritative_labels=True, requires_probability_reporting=True,
    )
    values.update(updates)
    return BenchmarkTask(**values)


def inputs(separated=False):
    labels = ([0] * 10 + [1] * 10) if separated else ([0] * 8 + [1] * 2 + [0] * 2 + [1] * 8)
    probabilities = [0.2] * 10 + [0.8] * 10
    ids = [str(i) for i in range(20)]
    manifest = BenchmarkManifest(
        benchmark_id="probability", version="1", task="binary_classification",
        split="test", sample_ids=ids, dataset_sha256="0" * 64,
        authoritative_labels=dict(zip(ids, labels)),
    )
    records = [PredictionRecord(sample_id=i, y_pred=int(p >= 0.5), score=p)
               for i, p in zip(ids, probabilities)]
    return manifest, records


@pytest.fixture(scope="module")
def evaluated():
    manifest, records = inputs()
    return manifest, records, evaluate_submission(manifest, records, model_id="model", task_contract=task())


def test_strict_probability_report_known_values_and_publication(evaluated):
    manifest, _, report = evaluated
    metrics = {m.name: m for m in report.metrics}
    assert metrics["brier"].value == pytest.approx(0.16)
    assert metrics["log_score"].value == pytest.approx(-0.2 * np.log(0.2) - 0.8 * np.log(0.8))
    assert metrics["ece"].value == pytest.approx(0)
    coefficients = metrics["brier"].details["calibration_coefficients"]
    assert coefficients["intercept"] == pytest.approx(0, abs=1e-6)
    assert coefficients["slope"] == pytest.approx(1, abs=1e-6)
    assert sum(b["n"] for b in metrics["ece"].details["bins"]) == 20
    bundle = build_bundle(manifest, report, task_id=task().task_id, submission_sha256="1" * 64)
    assert publish_leaderboard([bundle], task()).n_models == 1


def test_contract_cannot_require_incomplete_or_nonbinary_probability_reporting():
    with pytest.raises(ValueError, match="allowed metrics"):
        task(allowed_metrics=["auroc"], primary_metric="auroc", primary_direction="higher_is_better")
    with pytest.raises(ValueError, match="binary_classification"):
        task(task_type="regression")


def test_missing_scores_fail_before_evaluation():
    manifest, records = inputs()
    records[0] = records[0].model_copy(update={"score": None})
    with pytest.raises(ValueError, match="every record"):
        evaluate_submission(manifest, records, model_id="model", task_contract=task())


@pytest.mark.parametrize("missing", ["brier", "ece", "log_score"])
def test_publication_rejects_incomplete_probability_reports_even_with_valid_hashes(evaluated, missing):
    manifest, _, report = evaluated
    changed = report.model_copy(deep=True)
    changed.metrics = [metric for metric in changed.metrics if metric.name != missing]
    # Keep a valid available primary metric so the probability policy is the rejection.
    contract = task(primary_metric="auroc", primary_direction="higher_is_better")
    changed.primary_metric = "auroc"
    changed.primary_direction = "higher_is_better"
    changed.primary_value = next(m.value for m in changed.metrics if m.name == "auroc")
    bundle = build_bundle(manifest, changed, task_id=contract.task_id, submission_sha256="1" * 64)
    with pytest.raises(ValueError, match="probability reporting requires metrics"):
        publish_leaderboard([bundle], contract)


@pytest.mark.parametrize("name", ["brier", "ece", "log_score"])
def test_probability_policy_rejects_empty_diagnostics(evaluated, name):
    _, _, report = evaluated
    changed = report.model_copy(deep=True)
    next(m for m in changed.metrics if m.name == name).details = {}
    with pytest.raises(ValueError):
        task().validate_report_contract(changed)


def test_reliability_counts_cannot_disagree_with_evaluated_records(evaluated):
    _, _, report = evaluated
    changed = report.model_copy(deep=True)
    metric = next(m for m in changed.metrics if m.name == "ece")
    metric.details["bins"][0]["n"] += 1
    with pytest.raises(ValueError, match="bin counts"):
        task().validate_report_contract(changed)


def test_strict_policy_also_applies_to_declared_subgroup_reports(evaluated):
    _, _, report = evaluated
    changed = report.model_copy(deep=True)
    changed.subgroups = [SubgroupResult(subgroup="site", n=20,
                         metrics=[m for m in changed.metrics if m.name != "ece"])]
    with pytest.raises(ValueError, match="probability reporting requires metrics"):
        task().validate_report_contract(changed)


def test_unidentifiable_calibration_is_explicitly_unavailable_not_passing():
    manifest, records = inputs(separated=True)
    report = evaluate_submission(manifest, records, model_id="model", task_contract=task())
    details = next(m.details for m in report.metrics if m.name == "brier")
    assert "calibration_coefficients" not in details
    assert "separated" in details["calibration_coefficients_status"]
    # Completeness permits an honest unavailable reason; it does not assert calibration.
    task().validate_report_contract(report)


def test_probability_policy_is_bound_to_reference_hash_and_defaults_remain_optional(evaluated):
    manifest, records, strict = evaluated
    optional = task(requires_probability_reporting=False)
    assert BenchmarkTask.model_validate({k: v for k, v in optional.model_dump().items()
                                         if k != "requires_probability_reporting"}).requires_probability_reporting is False
    ordinary = evaluate_submission(manifest, records, model_id="model", task_contract=optional)
    assert ordinary.reference_sha256 != strict.reference_sha256
    assert ordinary.metrics == strict.metrics


def test_strict_policy_survives_package_roundtrip_and_checked_in_schema():
    manifest, _ = inputs()
    package = BenchmarkPackage(benchmark_id=manifest.benchmark_id, version=manifest.version,
                               manifest=manifest, tasks=[task()])
    serialized = package.model_dump(mode="json")
    schema = json.loads((Path(__file__).parents[1] / "schemas/benchmark-package.schema.json").read_text())
    Draft202012Validator(schema).validate(serialized)
    assert BenchmarkPackage.model_validate(copy.deepcopy(serialized)).tasks[0].requires_probability_reporting
    malformed = copy.deepcopy(serialized)
    malformed["tasks"][0]["allowed_metrics"].remove("brier")
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(malformed)
